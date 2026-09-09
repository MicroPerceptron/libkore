//! Bounded parser broker state machine.

use kore_rt::worker::WorkerTerminal;

use crate::{
    registry::{ParserId, ParserPolicy, Registry, Selection, ValidationError},
    schema::{InputIdentity, NormalizedResult},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BrokerJobId {
    slot: u16,
    generation: u16,
}

impl BrokerJobId {
    pub const fn slot(self) -> usize {
        self.slot as usize
    }

    pub const fn generation(self) -> u16 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobState {
    Queued,
    Dispatched,
    ResultReady,
    Cancelled,
    Failed(WorkerTerminal),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dispatch {
    pub job: BrokerJobId,
    pub parser: ParserId,
    pub input: InputIdentity,
    pub output_window_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerRefusal {
    NoParser,
    InputTooLarge,
    QueueFull,
    Concurrency,
    Quarantined,
    StaleJob,
    WrongState,
    InvalidResult(ValidationError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Quarantine {
    strikes: u8,
    until_ns: u64,
}

impl Quarantine {
    const EMPTY: Self = Self {
        strikes: 0,
        until_ns: 0,
    };

    fn record_failure(&mut self, now_ns: u64) {
        self.strikes = self.strikes.saturating_add(1).min(16);
        let shift = self.strikes.saturating_sub(1).min(10);
        let backoff = 1_000_000u64.saturating_mul(1u64 << shift);
        self.until_ns = now_ns.saturating_add(backoff);
    }

    fn record_success(&mut self) {
        self.strikes = 0;
        self.until_ns = 0;
    }
}

struct Job {
    generation: u16,
    parser: ParserId,
    input: InputIdentity,
    output_window_bytes: u64,
    state: JobState,
    result: Option<NormalizedResult>,
}

/// Source-neutral broker queue. Inputs are immutable identities plus a
/// bounded sniff prefix; the actual range capability travels only in the
/// dispatch channel assembled by the owner.
pub struct Broker<const N: usize> {
    jobs: [Option<Job>; N],
    generations: [u16; N],
    next: usize,
    active: u16,
    max_active: u16,
    quarantine: [Quarantine; 7],
}

impl<const N: usize> Broker<N> {
    pub const fn new(max_active: u16) -> Self {
        Self {
            jobs: [const { None }; N],
            generations: [0; N],
            next: 0,
            active: 0,
            max_active,
            quarantine: [Quarantine::EMPTY; 7],
        }
    }

    fn quarantine_mut(&mut self, parser: ParserId) -> Option<&mut Quarantine> {
        let index = usize::from(parser.0.checked_sub(1)?);
        self.quarantine.get_mut(index)
    }

    fn quarantine_of(&self, parser: ParserId) -> Option<Quarantine> {
        let index = usize::from(parser.0.checked_sub(1)?);
        self.quarantine.get(index).copied()
    }

    pub fn submit(
        &mut self,
        input: InputIdentity,
        sniff_prefix: &[u8],
        policy: ParserPolicy,
        output_window_bytes: u64,
        now_ns: u64,
    ) -> Result<(BrokerJobId, Selection), BrokerRefusal> {
        let selection = Registry::select(sniff_prefix, policy).ok_or(BrokerRefusal::NoParser)?;
        if input.len > selection.parser.limits.max_input
            || output_window_bytes > selection.parser.limits.max_output
        {
            return Err(BrokerRefusal::InputTooLarge);
        }
        if self
            .quarantine_of(selection.parser.id)
            .is_some_and(|q| now_ns < q.until_ns)
        {
            return Err(BrokerRefusal::Quarantined);
        }
        let slot = self
            .jobs
            .iter()
            .position(Option::is_none)
            .ok_or(BrokerRefusal::QueueFull)?;
        let mut generation = self.generations[slot].wrapping_add(1);
        if generation == 0 {
            generation = 1;
        }
        self.generations[slot] = generation;
        self.jobs[slot] = Some(Job {
            generation,
            parser: selection.parser.id,
            input,
            output_window_bytes,
            state: JobState::Queued,
            result: None,
        });
        Ok((
            BrokerJobId {
                slot: slot as u16,
                generation,
            },
            selection,
        ))
    }

    fn job_mut(&mut self, id: BrokerJobId) -> Result<&mut Job, BrokerRefusal> {
        let job = self
            .jobs
            .get_mut(id.slot())
            .and_then(Option::as_mut)
            .ok_or(BrokerRefusal::StaleJob)?;
        if job.generation != id.generation {
            return Err(BrokerRefusal::StaleJob);
        }
        Ok(job)
    }

    /// Round-robin dispatch bounded by `max_active`.
    pub fn dispatch_next(&mut self, now_ns: u64) -> Result<Option<Dispatch>, BrokerRefusal> {
        if self.active >= self.max_active {
            return Err(BrokerRefusal::Concurrency);
        }
        for step in 0..N {
            let slot = (self.next + step) % N;
            let Some(job) = self.jobs[slot].as_ref() else {
                continue;
            };
            if job.state != JobState::Queued {
                continue;
            }
            if self
                .quarantine_of(job.parser)
                .is_some_and(|q| now_ns < q.until_ns)
            {
                continue;
            }
            let job = self.jobs[slot].as_mut().expect("same bounded cell");
            job.state = JobState::Dispatched;
            self.active += 1;
            self.next = (slot + 1) % N;
            return Ok(Some(Dispatch {
                job: BrokerJobId {
                    slot: slot as u16,
                    generation: job.generation,
                },
                parser: job.parser,
                input: job.input,
                output_window_bytes: job.output_window_bytes,
            }));
        }
        Ok(None)
    }

    /// Validate and stage a result for its consumer. A cancelled, failed, or
    /// recycled job rejects late publication before any result is retained.
    pub fn complete(
        &mut self,
        id: BrokerJobId,
        result: NormalizedResult,
    ) -> Result<(), BrokerRefusal> {
        let (input, output, parser) = {
            let job = self.job_mut(id)?;
            if job.state != JobState::Dispatched {
                return Err(BrokerRefusal::WrongState);
            }
            (job.input, job.output_window_bytes, job.parser)
        };
        Registry::validate(&result, input, output).map_err(BrokerRefusal::InvalidResult)?;
        let job = self.job_mut(id)?;
        if result.provenance.parser != parser {
            return Err(BrokerRefusal::InvalidResult(ValidationError::Provenance));
        }
        job.result = Some(result);
        job.state = JobState::ResultReady;
        self.active = self.active.saturating_sub(1);
        if let Some(quarantine) = self.quarantine_mut(parser) {
            quarantine.record_success();
        }
        Ok(())
    }

    pub fn cancel(&mut self, id: BrokerJobId) -> Result<(), BrokerRefusal> {
        let job = self.job_mut(id)?;
        match job.state {
            JobState::Queued => job.state = JobState::Cancelled,
            JobState::Dispatched => {
                job.state = JobState::Cancelled;
                self.active = self.active.saturating_sub(1);
            }
            _ => return Err(BrokerRefusal::WrongState),
        }
        Ok(())
    }

    pub fn worker_failed(
        &mut self,
        id: BrokerJobId,
        terminal: WorkerTerminal,
        now_ns: u64,
    ) -> Result<(), BrokerRefusal> {
        let parser = {
            let job = self.job_mut(id)?;
            if job.state != JobState::Dispatched {
                return Err(BrokerRefusal::WrongState);
            }
            job.state = JobState::Failed(terminal);
            job.parser
        };
        self.active = self.active.saturating_sub(1);
        if let Some(quarantine) = self.quarantine_mut(parser) {
            quarantine.record_failure(now_ns);
        }
        Ok(())
    }

    pub fn state(&self, id: BrokerJobId) -> Option<JobState> {
        let job = self.jobs.get(id.slot())?.as_ref()?;
        (job.generation == id.generation).then_some(job.state)
    }

    pub fn take_result(&mut self, id: BrokerJobId) -> Result<NormalizedResult, BrokerRefusal> {
        let job = self.job_mut(id)?;
        if job.state != JobState::ResultReady {
            return Err(BrokerRefusal::WrongState);
        }
        let result = job.result.take().ok_or(BrokerRefusal::WrongState)?;
        self.jobs[id.slot()] = None;
        Ok(result)
    }

    /// Retire a terminal job with no result and reopen its slot.
    pub fn acknowledge_terminal(&mut self, id: BrokerJobId) -> Result<JobState, BrokerRefusal> {
        let state = self.state(id).ok_or(BrokerRefusal::StaleJob)?;
        if !matches!(state, JobState::Cancelled | JobState::Failed(_)) {
            return Err(BrokerRefusal::WrongState);
        }
        self.jobs[id.slot()] = None;
        Ok(state)
    }

    pub fn queued(&self) -> usize {
        self.jobs.iter().flatten().count()
    }

    pub const fn active(&self) -> u16 {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{registry::RUST_JSON, schema::ParserProvenance};

    fn input(version: u64) -> InputIdentity {
        InputIdentity::of(4, version, b"{\"a\":1}")
    }

    #[test]
    fn queue_concurrency_cancellation_and_reuse_are_bounded() {
        let mut broker = Broker::<2>::new(1);
        let (a, _) = broker
            .submit(input(1), b"{\"a\":1}", ParserPolicy::ALL, 1024, 0)
            .unwrap();
        let (b, _) = broker
            .submit(input(2), b"{\"b\":2}", ParserPolicy::ALL, 1024, 0)
            .unwrap();
        assert_eq!(
            broker.submit(input(3), b"{}", ParserPolicy::ALL, 1024, 0),
            Err(BrokerRefusal::QueueFull)
        );
        assert_eq!(broker.dispatch_next(0).unwrap().unwrap().job, a);
        assert_eq!(broker.dispatch_next(0), Err(BrokerRefusal::Concurrency));
        broker.cancel(a).unwrap();
        assert_eq!(broker.state(a), Some(JobState::Cancelled));
        broker.acknowledge_terminal(a).unwrap();
        assert_eq!(broker.dispatch_next(0).unwrap().unwrap().job, b);
        assert_eq!(broker.queued(), 1);
    }

    #[test]
    fn late_results_and_old_generations_cannot_publish() {
        let mut broker = Broker::<1>::new(1);
        let (old, _) = broker
            .submit(input(1), b"{}", ParserPolicy::ALL, 1024, 0)
            .unwrap();
        broker.dispatch_next(0).unwrap();
        broker.cancel(old).unwrap();
        let version = Registry::get(RUST_JSON).unwrap().version;
        let late = NormalizedResult::empty(ParserProvenance::deterministic(
            input(1),
            RUST_JSON,
            version,
        ));
        assert_eq!(broker.complete(old, late), Err(BrokerRefusal::WrongState));
        broker.acknowledge_terminal(old).unwrap();
        let (new, _) = broker
            .submit(input(2), b"{}", ParserPolicy::ALL, 1024, 0)
            .unwrap();
        assert_eq!(new.slot(), old.slot());
        assert_ne!(new.generation(), old.generation());
        assert_eq!(broker.cancel(old), Err(BrokerRefusal::StaleJob));
        assert_eq!(broker.state(new), Some(JobState::Queued));
    }

    #[test]
    fn worker_failure_quarantines_then_success_clears_backoff() {
        let mut broker = Broker::<2>::new(1);
        let (job, _) = broker
            .submit(input(1), b"{}", ParserPolicy::ALL, 1024, 0)
            .unwrap();
        broker.dispatch_next(0).unwrap();
        broker
            .worker_failed(job, WorkerTerminal::UnexpectedDeath, 10)
            .unwrap();
        assert_eq!(
            broker.acknowledge_terminal(job).unwrap(),
            JobState::Failed(WorkerTerminal::UnexpectedDeath)
        );
        assert_eq!(
            broker.submit(input(2), b"{}", ParserPolicy::ALL, 1024, 11),
            Err(BrokerRefusal::Quarantined)
        );
        assert!(
            broker
                .submit(input(2), b"{}", ParserPolicy::ALL, 1024, 2_000_000)
                .is_ok()
        );
    }

    #[test]
    fn validated_deterministic_result_round_trips() {
        let mut broker = Broker::<1>::new(1);
        let identity = input(1);
        let (job, selection) = broker
            .submit(identity, b"{}", ParserPolicy::ALL, 1024, 0)
            .unwrap();
        assert_eq!(selection.parser.id, RUST_JSON);
        broker.dispatch_next(0).unwrap();
        let version = Registry::get(RUST_JSON).unwrap().version;
        let result = NormalizedResult::empty(ParserProvenance::deterministic(
            identity, RUST_JSON, version,
        ));
        broker.complete(job, result.clone()).unwrap();
        assert_eq!(broker.take_result(job).unwrap(), result);
    }
}
