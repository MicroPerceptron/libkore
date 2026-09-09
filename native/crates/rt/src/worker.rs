//! Bounded, cancellable process-worker supervision.
//!
//! This module is policy-neutral runtime infrastructure: a parser broker,
//! codec, tool runner, or inference service supplies an already-founded task
//! plus the exact authority allocation it received. The supervisor neither
//! creates memory nor invents CPU authority. It checks that the allocation is
//! within the declared budget, starts the affine task handle, observes it, and
//! revokes it before publishing exactly one terminal outcome.

use crate::cap::{RetireOutcome, TaskPollOutcome, TaskStartOutcome, TaskToken, TaskWaitError};

/// A recyclable worker-job identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JobId {
    slot: u16,
    generation: u16,
}

impl JobId {
    /// Bounded supervisor slot.
    pub const fn slot(self) -> usize {
        self.slot as usize
    }

    /// Reuse generation of that slot.
    pub const fn generation(self) -> u16 {
        self.generation
    }
}

/// Authority and accounting limits for one worker process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerBudget {
    /// Maximum elapsed wall-clock time after admission.
    pub wall_ns: u64,
    /// Maximum single-CPU execution entitlement. Workers are admitted with
    /// one CPU share, so wall-time revocation is a conservative hard ceiling.
    pub cpu_ns: u64,
    /// Maximum total memory authority conveyed to the child.
    pub memory_bytes: u64,
    /// Maximum scratch subset within `memory_bytes`.
    pub scratch_bytes: u64,
    /// Maximum result-window authority and total publishable output.
    pub output_bytes: u64,
    /// Maximum simultaneously outstanding work records.
    pub outstanding: u16,
}

impl WorkerBudget {
    fn valid(self) -> bool {
        self.wall_ns != 0
            && self.cpu_ns != 0
            && self.memory_bytes != 0
            && self.scratch_bytes <= self.memory_bytes
            && self.outstanding != 0
    }
}

/// Exact resources the founder conveyed to the worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerAllocation {
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub output_window_bytes: u64,
    /// Process worker count. The generic confinement contract is deliberately
    /// single-CPU; admitting more would invalidate `cpu_ns`'s hard wall bound.
    pub cpu_shares: u16,
}

/// Why worker admission refused before the task became live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionRefusal {
    Capacity,
    InvalidBudget,
    AllocationExceedsBudget,
    StartRetryable,
    StartTerminal,
    StartRefused,
    StartUnavailable,
}

/// Refused admission returns the same affine task owner to its caller.
#[derive(Debug)]
pub struct AdmissionError<T> {
    pub reason: AdmissionRefusal,
    pub task: T,
}

/// Typed terminal fact published once teardown has closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerTerminal {
    Success,
    /// The worker's panic handler exited through its own task capability.
    Panic,
    /// The worker exited itself with a non-zero status.
    Failed(i32),
    Cancelled,
    Deadline,
    OutputBudget,
    OutstandingBudget,
    /// The task lineage disappeared without an exit report.
    UnexpectedDeath,
    /// Lifecycle observation itself returned a terminal infrastructure error.
    ObservationFailure,
}

/// Start result needed by the generic supervisor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerStart {
    Active,
    Retryable,
    Terminal,
    Refused,
    Unavailable,
}

/// Non-blocking lifecycle probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerProbe {
    Running,
    Exited(i32),
    Reaped,
    Retryable,
    Unavailable,
}

/// Result of consuming a task's teardown authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerStop {
    Closed,
}

/// Trait seam between the policy-neutral supervisor and a concrete task
/// lifecycle authority. Production uses [`TaskToken`]; host tests use a small
/// affine state machine without simulating kernel capabilities.
pub trait WorkerTask: Sized {
    fn start(&self) -> WorkerStart;
    fn probe(&self) -> WorkerProbe;
    fn stop(self) -> Result<WorkerStop, Self>;
}

impl WorkerTask for TaskToken {
    fn start(&self) -> WorkerStart {
        match TaskToken::start(self) {
            TaskStartOutcome::Active => WorkerStart::Active,
            TaskStartOutcome::Retryable => WorkerStart::Retryable,
            TaskStartOutcome::Terminal => WorkerStart::Terminal,
            TaskStartOutcome::Refused(_) => WorkerStart::Refused,
            TaskStartOutcome::Unavailable => WorkerStart::Unavailable,
        }
    }

    fn probe(&self) -> WorkerProbe {
        match TaskToken::poll(self) {
            Ok(TaskPollOutcome::Running) => WorkerProbe::Running,
            Ok(TaskPollOutcome::Exited(status)) => WorkerProbe::Exited(status),
            Ok(TaskPollOutcome::Reaped) => WorkerProbe::Reaped,
            Err(TaskWaitError::Retryable | TaskWaitError::NoSlots) => WorkerProbe::Retryable,
            Err(
                TaskWaitError::Transport
                | TaskWaitError::Refused(_)
                | TaskWaitError::MalformedCompletion
                | TaskWaitError::MalformedReport,
            ) => WorkerProbe::Unavailable,
        }
    }

    fn stop(self) -> Result<WorkerStop, Self> {
        self.kill().map(|outcome| match outcome {
            RetireOutcome::Retired | RetireOutcome::AlreadyDead => WorkerStop::Closed,
            RetireOutcome::Retryable => unreachable!("TaskToken::kill returns retry as Err"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Running,
    Stopping(WorkerTerminal),
    Terminal(WorkerTerminal),
}

struct Job<T> {
    task: Option<T>,
    generation: u16,
    deadline_ns: u64,
    output_limit: u64,
    output_used: u64,
    outstanding_limit: u16,
    outstanding: u16,
    state: State,
}

/// Fixed-capacity worker supervisor. No global state, allocation, lock, or
/// cleanup queue is involved; every cell owns at most one affine task handle.
pub struct WorkerSupervisor<T, const N: usize> {
    jobs: [Option<Job<T>>; N],
    generations: [u16; N],
}

impl<T: WorkerTask, const N: usize> WorkerSupervisor<T, N> {
    pub const fn new() -> Self {
        Self {
            jobs: [const { None }; N],
            generations: [0; N],
        }
    }

    /// Admit and start a founded worker. Every refusal returns `task`.
    pub fn admit(
        &mut self,
        task: T,
        budget: WorkerBudget,
        allocation: WorkerAllocation,
        now_ns: u64,
    ) -> Result<JobId, AdmissionError<T>> {
        if !budget.valid() {
            return Err(AdmissionError {
                reason: AdmissionRefusal::InvalidBudget,
                task,
            });
        }
        if allocation.cpu_shares != 1
            || allocation.memory_bytes > budget.memory_bytes
            || allocation.scratch_bytes > budget.scratch_bytes
            || allocation.scratch_bytes > allocation.memory_bytes
            || allocation.output_window_bytes > budget.output_bytes
        {
            return Err(AdmissionError {
                reason: AdmissionRefusal::AllocationExceedsBudget,
                task,
            });
        }
        let Some(slot) = self.jobs.iter().position(Option::is_none) else {
            return Err(AdmissionError {
                reason: AdmissionRefusal::Capacity,
                task,
            });
        };
        let start = task.start();
        if start != WorkerStart::Active {
            let reason = match start {
                WorkerStart::Active => unreachable!(),
                WorkerStart::Retryable => AdmissionRefusal::StartRetryable,
                WorkerStart::Terminal => AdmissionRefusal::StartTerminal,
                WorkerStart::Refused => AdmissionRefusal::StartRefused,
                WorkerStart::Unavailable => AdmissionRefusal::StartUnavailable,
            };
            return Err(AdmissionError { reason, task });
        }
        let mut generation = self.generations[slot].wrapping_add(1);
        if generation == 0 {
            generation = 1;
        }
        self.generations[slot] = generation;
        let lifetime = budget.wall_ns.min(budget.cpu_ns);
        self.jobs[slot] = Some(Job {
            task: Some(task),
            generation,
            deadline_ns: now_ns.saturating_add(lifetime),
            output_limit: budget.output_bytes,
            output_used: 0,
            outstanding_limit: budget.outstanding,
            outstanding: 0,
            state: State::Running,
        });
        Ok(JobId {
            slot: slot as u16,
            generation,
        })
    }

    fn job_mut(&mut self, id: JobId) -> Option<&mut Job<T>> {
        let job = self.jobs.get_mut(id.slot())?.as_mut()?;
        (job.generation == id.generation).then_some(job)
    }

    fn request_stop(&mut self, id: JobId, terminal: WorkerTerminal) -> bool {
        let Some(job) = self.job_mut(id) else {
            return false;
        };
        if !matches!(job.state, State::Running) {
            return false;
        }
        job.state = State::Stopping(terminal);
        self.finish_stop(id);
        true
    }

    fn finish_stop(&mut self, id: JobId) {
        let Some(job) = self.job_mut(id) else {
            return;
        };
        let State::Stopping(terminal) = job.state else {
            return;
        };
        let Some(task) = job.task.take() else {
            return;
        };
        match task.stop() {
            Ok(WorkerStop::Closed) => job.state = State::Terminal(terminal),
            Err(task) => job.task = Some(task),
        }
    }

    /// Owner-requested cancellation. The first observed terminal edge wins.
    pub fn cancel(&mut self, id: JobId) -> bool {
        self.request_stop(id, WorkerTerminal::Cancelled)
    }

    /// Reserve output bytes before making them visible to a consumer.
    pub fn charge_output(&mut self, id: JobId, bytes: u64) -> bool {
        let Some(job) = self.job_mut(id) else {
            return false;
        };
        if !matches!(job.state, State::Running) {
            return false;
        }
        let Some(next) = job.output_used.checked_add(bytes) else {
            self.request_stop(id, WorkerTerminal::OutputBudget);
            return false;
        };
        if next > job.output_limit {
            self.request_stop(id, WorkerTerminal::OutputBudget);
            return false;
        }
        job.output_used = next;
        true
    }

    /// Acquire one bounded outstanding-work credit.
    pub fn begin_work(&mut self, id: JobId) -> bool {
        let Some(job) = self.job_mut(id) else {
            return false;
        };
        if !matches!(job.state, State::Running) {
            return false;
        }
        if job.outstanding >= job.outstanding_limit {
            self.request_stop(id, WorkerTerminal::OutstandingBudget);
            return false;
        }
        job.outstanding += 1;
        true
    }

    /// Return one outstanding-work credit. Underflow is a stale/invalid
    /// completion and cannot alter another generation.
    pub fn finish_work(&mut self, id: JobId) -> bool {
        let Some(job) = self.job_mut(id) else {
            return false;
        };
        if !matches!(job.state, State::Running) || job.outstanding == 0 {
            return false;
        }
        job.outstanding -= 1;
        true
    }

    /// Bounded lifecycle drive. At most one probe and one teardown attempt are
    /// performed per live slot.
    pub fn poll(&mut self, now_ns: u64) {
        for slot in 0..N {
            let Some(job) = self.jobs[slot].as_mut() else {
                continue;
            };
            let id = JobId {
                slot: slot as u16,
                generation: job.generation,
            };
            match job.state {
                State::Running if now_ns >= job.deadline_ns => {
                    let _ = self.request_stop(id, WorkerTerminal::Deadline);
                }
                State::Running => {
                    let probe = job.task.as_ref().map(WorkerTask::probe);
                    match probe {
                        Some(WorkerProbe::Running | WorkerProbe::Retryable) => {}
                        Some(WorkerProbe::Exited(0)) => {
                            let _ = self.request_stop(id, WorkerTerminal::Success);
                        }
                        Some(WorkerProbe::Exited(crate::entry::PANIC_EXIT_STATUS)) => {
                            let _ = self.request_stop(id, WorkerTerminal::Panic);
                        }
                        Some(WorkerProbe::Exited(status)) => {
                            let _ = self.request_stop(id, WorkerTerminal::Failed(status));
                        }
                        Some(WorkerProbe::Reaped) => {
                            let _ = self.request_stop(id, WorkerTerminal::UnexpectedDeath);
                        }
                        Some(WorkerProbe::Unavailable) | None => {
                            let _ = self.request_stop(id, WorkerTerminal::ObservationFailure);
                        }
                    }
                }
                State::Stopping(_) => self.finish_stop(id),
                State::Terminal(_) => {}
            }
        }
    }

    /// Take the sole terminal result and reopen the bounded slot.
    pub fn take_terminal(&mut self, id: JobId) -> Option<WorkerTerminal> {
        let job = self.jobs.get(id.slot())?.as_ref()?;
        if job.generation != id.generation {
            return None;
        }
        let State::Terminal(terminal) = job.state else {
            return None;
        };
        self.jobs[id.slot()] = None;
        Some(terminal)
    }

    pub fn live(&self) -> usize {
        self.jobs.iter().flatten().count()
    }
}

impl<T: WorkerTask, const N: usize> Default for WorkerSupervisor<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use std::rc::Rc;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum MockState {
        Running,
        Exited(i32),
        Reaped,
    }

    #[derive(Debug)]
    struct MockTask {
        state: Rc<Cell<MockState>>,
        stop_refusals: Rc<Cell<u8>>,
        stopped: Rc<Cell<u32>>,
    }

    impl MockTask {
        fn new() -> (Self, Rc<Cell<MockState>>, Rc<Cell<u32>>) {
            let state = Rc::new(Cell::new(MockState::Running));
            let stopped = Rc::new(Cell::new(0));
            (
                Self {
                    state: state.clone(),
                    stop_refusals: Rc::new(Cell::new(0)),
                    stopped: stopped.clone(),
                },
                state,
                stopped,
            )
        }
    }

    impl WorkerTask for MockTask {
        fn start(&self) -> WorkerStart {
            WorkerStart::Active
        }

        fn probe(&self) -> WorkerProbe {
            match self.state.get() {
                MockState::Running => WorkerProbe::Running,
                MockState::Exited(status) => WorkerProbe::Exited(status),
                MockState::Reaped => WorkerProbe::Reaped,
            }
        }

        fn stop(self) -> Result<WorkerStop, Self> {
            let refused = self.stop_refusals.get();
            if refused != 0 {
                self.stop_refusals.set(refused - 1);
                return Err(self);
            }
            self.stopped.set(self.stopped.get() + 1);
            Ok(WorkerStop::Closed)
        }
    }

    fn budget() -> WorkerBudget {
        WorkerBudget {
            wall_ns: 100,
            cpu_ns: 80,
            memory_bytes: 4096,
            scratch_bytes: 1024,
            output_bytes: 32,
            outstanding: 2,
        }
    }

    fn allocation() -> WorkerAllocation {
        WorkerAllocation {
            memory_bytes: 4096,
            scratch_bytes: 1024,
            output_window_bytes: 32,
            cpu_shares: 1,
        }
    }

    #[test]
    fn completion_is_observed_once_after_affine_teardown() {
        let (task, state, stopped) = MockTask::new();
        let mut sup = WorkerSupervisor::<MockTask, 1>::new();
        let id = sup.admit(task, budget(), allocation(), 10).unwrap();
        state.set(MockState::Exited(0));
        sup.poll(11);
        assert_eq!(stopped.get(), 1);
        assert_eq!(sup.take_terminal(id), Some(WorkerTerminal::Success));
        assert_eq!(sup.take_terminal(id), None);
        assert_eq!(sup.live(), 0);
    }

    #[test]
    fn noncooperative_worker_is_revoked_at_the_conservative_cpu_deadline() {
        let (task, _state, stopped) = MockTask::new();
        let mut sup = WorkerSupervisor::<MockTask, 1>::new();
        let id = sup.admit(task, budget(), allocation(), 10).unwrap();
        sup.poll(89);
        assert_eq!(stopped.get(), 0);
        sup.poll(90);
        assert_eq!(stopped.get(), 1);
        assert_eq!(sup.take_terminal(id), Some(WorkerTerminal::Deadline));
    }

    #[test]
    fn budgets_cancel_and_generation_reuse_cannot_double_publish() {
        let (task, _state, stopped) = MockTask::new();
        let mut sup = WorkerSupervisor::<MockTask, 1>::new();
        let first = sup.admit(task, budget(), allocation(), 0).unwrap();
        assert!(sup.charge_output(first, 32));
        assert!(!sup.charge_output(first, 1));
        assert!(!sup.cancel(first), "budget exhaustion already won");
        assert_eq!(stopped.get(), 1);
        assert_eq!(sup.take_terminal(first), Some(WorkerTerminal::OutputBudget));

        let (task, _, _) = MockTask::new();
        let second = sup.admit(task, budget(), allocation(), 0).unwrap();
        assert_eq!(second.slot(), first.slot());
        assert_ne!(second.generation(), first.generation());
        assert!(
            !sup.cancel(first),
            "stale generation cannot cancel replacement"
        );
        assert!(sup.cancel(second));
        assert_eq!(sup.take_terminal(second), Some(WorkerTerminal::Cancelled));
    }

    #[test]
    fn teardown_refusal_is_retried_before_any_terminal_result_is_visible() {
        let (task, state, stopped) = MockTask::new();
        task.stop_refusals.set(1);
        let mut sup = WorkerSupervisor::<MockTask, 1>::new();
        let id = sup.admit(task, budget(), allocation(), 0).unwrap();
        state.set(MockState::Exited(7));
        sup.poll(1);
        assert_eq!(sup.take_terminal(id), None);
        assert_eq!(stopped.get(), 0);
        sup.poll(2);
        assert_eq!(stopped.get(), 1);
        assert_eq!(sup.take_terminal(id), Some(WorkerTerminal::Failed(7)));
    }

    #[test]
    fn allocation_limits_and_capacity_return_the_affine_task() {
        let (task, _, _) = MockTask::new();
        let mut sup = WorkerSupervisor::<MockTask, 1>::new();
        let mut too_large = allocation();
        too_large.memory_bytes += 1;
        let err = sup.admit(task, budget(), too_large, 0).unwrap_err();
        assert_eq!(err.reason, AdmissionRefusal::AllocationExceedsBudget);

        let id = sup.admit(err.task, budget(), allocation(), 0).unwrap();
        let (other, _, _) = MockTask::new();
        let err = sup.admit(other, budget(), allocation(), 0).unwrap_err();
        assert_eq!(err.reason, AdmissionRefusal::Capacity);
        assert!(sup.cancel(id));
        assert_eq!(sup.take_terminal(id), Some(WorkerTerminal::Cancelled));
    }

    #[test]
    fn peer_death_and_outstanding_limit_are_typed_terminals() {
        let (task, state, _) = MockTask::new();
        let mut sup = WorkerSupervisor::<MockTask, 1>::new();
        let id = sup.admit(task, budget(), allocation(), 0).unwrap();
        state.set(MockState::Reaped);
        sup.poll(1);
        assert_eq!(sup.take_terminal(id), Some(WorkerTerminal::UnexpectedDeath));

        let (task, _, _) = MockTask::new();
        let id = sup.admit(task, budget(), allocation(), 0).unwrap();
        assert!(sup.begin_work(id));
        assert!(sup.begin_work(id));
        assert!(!sup.begin_work(id));
        assert_eq!(
            sup.take_terminal(id),
            Some(WorkerTerminal::OutstandingBudget)
        );
    }

    #[test]
    fn panic_exit_is_distinct_from_an_application_failure() {
        let (task, state, _) = MockTask::new();
        let mut sup = WorkerSupervisor::<MockTask, 1>::new();
        let id = sup.admit(task, budget(), allocation(), 0).unwrap();
        state.set(MockState::Exited(crate::entry::PANIC_EXIT_STATUS));
        sup.poll(1);
        assert_eq!(sup.take_terminal(id), Some(WorkerTerminal::Panic));
    }
}
