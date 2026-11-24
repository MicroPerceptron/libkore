//! Resumable execution on one possessed accelerator pair.

mod set;
pub use set::ExecutionSet;

use super::RingAccel;
use crate::service::{AccelBindings, AccelEventState, AccelRefusal, AccelRequest, AccelResponse};
use kore_driver::accel::{LaunchShape, Workgroups};
use kore_rt::{svc::CallError, transport::Transport};

/// Artifact-defined work on object IDs belonging to one possessed face.
/// This is request data; possession and object ownership are checked by the
/// service. A sequence retains its input iterator while execution is live.
#[derive(Clone, Copy)]
pub struct ArtifactExecution {
    pub queue: u64,
    pub program: u64,
    pub bindings: AccelBindings,
    pub timeout_ns: u64,
}

impl ArtifactExecution {
    fn request(self) -> AccelRequest {
        #[allow(deprecated)]
        AccelRequest::Submit {
            queue: self.queue,
            program: self.program,
            bindings: self.bindings,
            timeout_ns: self.timeout_ns,
            dispatch: kore_driver::accel::Dispatch::scalar(),
        }
    }
}

/// Portable launch data retained by an asynchronous execution ledger. Explicit
/// geometry is preserved across admission retries and event retirement; the
/// provider still validates the program, bindings and device limits. Object
/// IDs are request data, not authority to access a device or allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchExecution {
    pub queue: u64,
    pub program: u64,
    pub bindings: AccelBindings,
    pub timeout_ns: u64,
    pub launch: LaunchShape,
}

impl LaunchExecution {
    fn request(self) -> AccelRequest {
        AccelRequest::SubmitLaunch {
            queue: self.queue,
            program: self.program,
            bindings: self.bindings,
            timeout_ns: self.timeout_ns,
            launch: self.launch,
        }
    }
}

/// One member of a staged ordered table: the same program, bindings and
/// launch vocabulary as a single [`LaunchExecution`], plus the member's
/// ordering edge to the members staged before it in the same table.
///
/// `barrier == true` makes the member wait until every earlier member's
/// writes are complete and visible; `false` declares it independent of the
/// members since the most recent barrier, so a provider may overlap them
/// (`kore_driver::accel::OrderedExecution`). A table's first member has no
/// predecessor inside it, so its flag is ignored. A `gate` holds the member
/// until a host word reaches a value; a `signal` writes one when it retires
/// (`kore_driver::accel::OrderedExecution::gate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderedLaunch {
    pub queue: u64,
    pub program: u64,
    pub bindings: AccelBindings,
    pub timeout_ns: u64,
    pub launch: LaunchShape,
    pub barrier: bool,
    pub gate: Option<crate::service::AccelSync>,
    pub signal: Option<crate::service::AccelSync>,
}

/// An artifact-defined launch keeps the historical staged-table meaning:
/// every member depends on all members before it.
impl From<ArtifactExecution> for OrderedLaunch {
    fn from(value: ArtifactExecution) -> Self {
        Self {
            queue: value.queue,
            program: value.program,
            bindings: value.bindings,
            timeout_ns: value.timeout_ns,
            launch: LaunchShape::ArtifactDefined,
            barrier: true,
            gate: None,
            signal: None,
        }
    }
}

/// A portable launch staged on its own is fully ordered behind its
/// predecessors; clear [`OrderedLaunch::barrier`] to declare independence.
impl From<LaunchExecution> for OrderedLaunch {
    fn from(value: LaunchExecution) -> Self {
        Self {
            queue: value.queue,
            program: value.program,
            bindings: value.bindings,
            timeout_ns: value.timeout_ns,
            launch: value.launch,
            barrier: true,
            gate: None,
            signal: None,
        }
    }
}

/// Whether one staged table may name the same program more than once.
///
/// A provider that materializes one invocation image per program (the NPU's
/// firmware batch) cannot run two members of one program inside a table, so
/// packing must split there. A provider that encodes every member as its own
/// dispatch over shared, immutable code (Xe) admits repeats, and splitting
/// would only cost a host completion round trip per repeat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramReuse {
    /// Start a new table at a program already named in the current one.
    Split,
    /// Keep repeated programs in one table.
    Share,
}

/// One ordered batch in the possessed face's staging window. The service
/// snapshots and validates its entries before publishing any device work.
/// Staging must not change until the submit reply has been consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderedExecution {
    pub queue: u64,
    pub stage: u64,
    pub len: u64,
    pub count: u32,
    pub timeout_ns: u64,
}

/// Submission forms driven by the same cancellation and retirement engine.
#[derive(Clone, Copy)]
pub enum ExecutionSubmission {
    Artifact(ArtifactExecution),
    Launch(LaunchExecution),
    Ordered(OrderedExecution),
}

impl From<LaunchExecution> for ExecutionSubmission {
    fn from(value: LaunchExecution) -> Self {
        Self::Launch(value)
    }
}

impl From<ArtifactExecution> for ExecutionSubmission {
    fn from(value: ArtifactExecution) -> Self {
        Self::Artifact(value)
    }
}
impl From<OrderedExecution> for ExecutionSubmission {
    fn from(value: OrderedExecution) -> Self {
        Self::Ordered(value)
    }
}
impl ExecutionSubmission {
    fn request(self) -> AccelRequest {
        match self {
            Self::Artifact(work) => work.request(),
            Self::Launch(work) => work.request(),
            Self::Ordered(work) => AccelRequest::SubmitSequence {
                queue: work.queue,
                stage: work.stage,
                len: work.len,
                count: work.count,
                timeout_ns: work.timeout_ns,
            },
        }
    }
}

/// Pack ordered work without assuming that a provider supports simultaneous
/// invocation images for repeated program IDs. Split at a repeated program,
/// a queue/deadline change, or the protocol's entry ceiling. Every group uses
/// each program once; data dependencies stay in order across group fences.
///
/// `staging` is an exclusively reserved command region starting at `offset`
/// in the attached window, not a tensor copy. The caller retains it unchanged
/// while a ledger can still submit or resume these requests. `groups` needs
/// at least one slot per launch (the worst case); unused slots become None.
/// This function allocates nothing and publishes no work.
pub fn stage_ordered_executions(
    launches: &[ArtifactExecution],
    staging: &mut [u8],
    offset: u64,
    groups: &mut [Option<OrderedExecution>],
) -> Result<usize, AccelRefusal> {
    stage_ordered_launches(launches, staging, offset, groups, ProgramReuse::Split)
}

/// Stage `launches` as ordered tables, from `offset` in the client's stage
/// (`staging` its bytes from there), splitting where a table must end: at
/// [`MAX_ACCEL_SEQUENCE_ENTRIES`](kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES)
/// members, a change of queue or timeout, a repeated program under
/// [`ProgramReuse::Split`], or past
/// [`MAX_ACCEL_SEQUENCE_SYNCS`](kore_abi::svc::MAX_ACCEL_SEQUENCE_SYNCS) gate
/// and signal records. Each table is its entries, then its members' sync
/// records in member order (`AccelReqWire::OP_SUBMIT_SEQUENCE`); `groups`
/// receives one submission a table. Returns the table count.
pub fn stage_ordered_launches<L: Copy + Into<OrderedLaunch>>(
    launches: &[L],
    staging: &mut [u8],
    offset: u64,
    groups: &mut [Option<OrderedExecution>],
    reuse: ProgramReuse,
) -> Result<usize, AccelRefusal> {
    use zerocopy::IntoBytes;
    let entry_bytes = core::mem::size_of::<kore_abi::svc::AccelSequenceEntryWire>();
    let sync_bytes = core::mem::size_of::<kore_abi::svc::AccelSequenceSyncWire>();
    if launches.is_empty() || groups.len() < launches.len() {
        return Err(AccelRefusal::InvalidArgument);
    }
    let entry = |launch: OrderedLaunch| crate::service::AccelSequenceEntry {
        program: launch.program,
        bindings: launch.bindings,
        launch: launch.launch,
        barrier: launch.barrier,
        gate: launch.gate,
        signal: launch.signal,
    };
    let records = |launch: OrderedLaunch| usize::from(launch.gate.is_some()) + usize::from(launch.signal.is_some());
    // Where a table ending before `index` must end. The scan is bounded by
    // MAX_ACCEL_SEQUENCE_ENTRIES, independent of model size. Under `Split`,
    // duplicate program instances are never presumed safe.
    let splits = |first: usize, index: usize| {
        let launch: OrderedLaunch = launches[index].into();
        let head: OrderedLaunch = launches[first].into();
        let synced: usize = launches[first..=index].iter().map(|launch| records((*launch).into())).sum();
        index - first == kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES
            || launch.queue != head.queue
            || launch.timeout_ns != head.timeout_ns
            || synced > kore_abi::svc::MAX_ACCEL_SEQUENCE_SYNCS
            || (reuse == ProgramReuse::Split
                && launches[first..index]
                    .iter()
                    .any(|prior| (*prior).into().program == launch.program))
    };
    let mut put = |cursor: &mut usize, bytes: &[u8]| -> Result<(), AccelRefusal> {
        let end = cursor.checked_add(bytes.len()).ok_or(AccelRefusal::OutOfBounds)?;
        if end > staging.len() || offset.checked_add(end as u64).is_none() {
            return Err(AccelRefusal::OutOfBounds);
        }
        staging[*cursor..end].copy_from_slice(bytes);
        *cursor = end;
        Ok(())
    };
    groups.fill(None);
    let (mut first, mut tables, mut cursor) = (0, 0, 0usize);
    for index in 1..=launches.len() {
        if index < launches.len() && !splits(first, index) {
            continue;
        }
        // The table `first..index`: its entries, then its sync records.
        let start = cursor;
        for launch in &launches[first..index] {
            put(&mut cursor, crate::wire::encode_sequence_entry(entry((*launch).into())).as_bytes())?;
        }
        for launch in &launches[first..index] {
            for sync in crate::wire::encode_sequence_syncs(&entry((*launch).into())) {
                put(&mut cursor, sync.as_bytes())?;
            }
        }
        debug_assert_eq!(cursor - start - (index - first) * entry_bytes, {
            launches[first..index].iter().map(|launch| records((*launch).into())).sum::<usize>() * sync_bytes
        });
        let head: OrderedLaunch = launches[first].into();
        groups[tables] = Some(OrderedExecution {
            queue: head.queue,
            stage: offset + start as u64,
            len: (cursor - start) as u64,
            count: (index - first) as u32,
            timeout_ns: head.timeout_ns,
        });
        tables += 1;
        first = index;
    }
    Ok(tables)
}

/// A completed lifecycle, including release of the event's binding references.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionOutcome {
    /// Stopped locally before the next submission was published.
    NotSubmitted,
    /// Admission was refused without creating an event.
    Refused(AccelRefusal),
    /// The event was observed terminal and its destruction was acknowledged.
    Retired(AccelEventState),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionProgress {
    /// More observation or event retirement is required. This is not a fence.
    Pending,
    Finished(ExecutionOutcome),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    Infrastructure(CallError),
    /// A poll or retirement was refused. The runner still retains the event.
    Refused(AccelRefusal),
}

#[derive(Clone, Copy)]
enum Phase {
    Submit(AccelRequest),
    Poll(u64),
    Cancel(u64),
    Retire {
        event: u64,
        terminal: AccelEventState,
    },
    Finished(ExecutionOutcome),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Intent {
    Run,
    Cancel,
    Drain,
}

/// Ordered submissions, authoritative observation, and event retirement.
///
/// Retain this value when cancelling an `advance` future: the next call resumes
/// the exact request, including cancellation after publication but before the
/// submission reply. The exclusive client borrow prevents another operation
/// from consuming that reply. No allocation, provider-specific polling, or
/// executor is introduced here.
///
/// Dropping the runner does not cancel device work or free server objects.
/// An owner abandoning it must retain those resources until device teardown;
/// only `Finished` proves this lifecycle has released its event references.
#[must_use = "retain and advance the execution until event retirement is acknowledged"]
pub struct Execution<'a, 'w, 't, T: Transport, I = core::iter::Empty<ArtifactExecution>> {
    accel: &'a mut RingAccel<'w, 't, T>,
    phase: Phase,
    remaining: I,
    completed: usize,
    intent: Intent,
}

/// Task-owned execution state for a backend that also owns its accelerator
/// face. Each advance borrows the face only for that operation, so cancelling
/// the future leaves the exact request and event ledger in this value. The
/// owner must retain the face and all bound buffers until `Finished`.
#[must_use = "retain the ledger until event retirement is acknowledged"]
pub struct ExecutionLedger<I = core::iter::Empty<ArtifactExecution>> {
    phase: Phase,
    remaining: I,
    completed: usize,
    intent: Intent,
}

impl<'w, 't, T: Transport> RingAccel<'w, 't, T> {
    /// Prepare a detached, ordered execution on an idle face. Construction
    /// publishes nothing. One task may interleave retained ledgers only at
    /// `execution_request_idle()` boundaries. A timed-out/cancelled observation
    /// retains its exact request correlation and must resume that ledger first.
    pub fn execution_ledger_sequence<I: Iterator>(
        &self,
        mut launches: I,
    ) -> Option<ExecutionLedger<I>>
    where
        I::Item: Into<ExecutionSubmission>,
    {
        if !self.client.is_idle() {
            return None;
        }
        let first = launches.next()?;
        Some(ExecutionLedger {
            phase: Phase::Submit(first.into().request()),
            remaining: launches,
            completed: 0,
            intent: Intent::Run,
        })
    }

    /// Whether no request/reply correlation is retained by the face. A queue
    /// owner may switch between execution ledgers only at this boundary;
    /// event ownership remains in each ledger until acknowledged retirement.
    pub fn execution_request_idle(&self) -> bool {
        self.client.is_idle()
    }

    /// Start a resumable artifact-defined execution on an idle pair. The
    /// supplied object IDs must belong to this possessed face. Construction
    /// publishes nothing; the first `advance` performs admission. Returns None
    /// unless the pair is idle, including when it is poisoned.
    pub fn execution(
        &mut self,
        queue: u64,
        program: u64,
        bindings: AccelBindings,
        timeout_ns: u64,
    ) -> Option<Execution<'_, 'w, 't, T>> {
        if !self.client.is_idle() {
            return None;
        }
        Some(Execution {
            accel: self,
            phase: Phase::Submit(
                ArtifactExecution {
                    queue,
                    program,
                    bindings,
                    timeout_ns,
                }
                .request(),
            ),
            remaining: core::iter::empty(),
            completed: 0,
            intent: Intent::Run,
        })
    }

    /// Start one execution with explicit workgroup geometry on the same
    /// resumable submit, completion, and retirement path as [`Self::execution`].
    /// The provider validates the artifact's local shape and device limits.
    pub fn execution_workgroups(
        &mut self,
        queue: u64,
        program: u64,
        bindings: AccelBindings,
        workgroups: Workgroups,
        timeout_ns: u64,
    ) -> Option<Execution<'_, 'w, 't, T>> {
        if !self.client.is_idle() {
            return None;
        }
        Some(Execution {
            accel: self,
            phase: Phase::Submit(
                LaunchExecution {
                    queue,
                    program,
                    bindings,
                    timeout_ns,
                    launch: LaunchShape::Workgroups(workgroups),
                }
                .request(),
            ),
            remaining: core::iter::empty(),
            completed: 0,
            intent: Intent::Run,
        })
    }

    /// Start one provider-ordered staged execution table. The table must stay
    /// immutable in the attached staging window until the submission reply is
    /// consumed; the service copies object identities into the event ledger at
    /// admission, after which the client may reuse staging.
    pub fn ordered_execution(
        &mut self,
        queue: u64,
        stage: u64,
        len: u64,
        count: u32,
        timeout_ns: u64,
    ) -> Option<Execution<'_, 'w, 't, T>> {
        if !self.client.is_idle() || count == 0 {
            return None;
        }
        Some(Execution {
            accel: self,
            phase: Phase::Submit(AccelRequest::SubmitSequence {
                queue,
                stage,
                len,
                count,
                timeout_ns,
            }),
            remaining: core::iter::empty(),
            completed: 0,
            intent: Intent::Run,
        })
    }

    /// Execute ordered work on this same face. Advance to the next submission
    /// only after Complete and acknowledged event destruction. Refusal,
    /// failure, cancellation or uncertainty never admits a successor.
    /// Items may be single launches or staged ordered tables
    /// ([`stage_ordered_launches`]), each admitted as one event.
    /// Returns None for an empty sequence or a pair that is not idle (including
    /// poisoned pairs). The iterator is not advanced when the pair is occupied.
    pub fn execution_sequence<I: Iterator>(
        &mut self,
        mut launches: I,
    ) -> Option<Execution<'_, 'w, 't, T, I>>
    where
        I::Item: Into<ExecutionSubmission>,
    {
        if !self.client.is_idle() {
            return None;
        }
        let first = launches.next()?;
        Some(Execution {
            accel: self,
            phase: Phase::Submit(first.into().request()),
            remaining: launches,
            completed: 0,
            intent: Intent::Run,
        })
    }
}

impl<I: Iterator> ExecutionLedger<I>
where
    I::Item: Into<ExecutionSubmission>,
{
    /// Stop after the current event is retired; an unpublished request is
    /// skipped. Retain the ledger and continue advancing after cancellation.
    pub fn request_cancel(&mut self) {
        if !matches!(self.phase, Phase::Finished(_)) {
            self.intent = Intent::Cancel;
        }
    }

    pub fn completed(&self) -> usize {
        self.completed
    }

    pub fn event(&self) -> Option<u64> {
        match self.phase {
            Phase::Poll(event) | Phase::Cancel(event) | Phase::Retire { event, .. } => Some(event),
            _ => None,
        }
    }

    pub fn awaiting_device_completion(&self) -> bool {
        matches!(self.phase, Phase::Poll(_))
    }

    pub fn progress(&self) -> ExecutionProgress {
        match self.phase {
            Phase::Finished(outcome) => ExecutionProgress::Finished(outcome),
            _ => ExecutionProgress::Pending,
        }
    }

    /// Retry a confirmed Busy admission on a later bounded step. This is for
    /// queue pipelines: the caller must keep the same staging table immutable
    /// until an admission reply succeeds, and impose an overall deadline.
    /// Transport uncertainty always retains its exact outstanding request.
    pub async fn advance_queued_bounded<T: Transport>(
        &mut self,
        accel: &mut RingAccel<'_, '_, T>,
        budget_ns: u64,
    ) -> Result<ExecutionProgress, ExecutionError> {
        let before = self.phase;
        let progress = self.advance_with_budget(accel, Some(budget_ns)).await?;
        if matches!(before, Phase::Submit(_))
            && self.intent == Intent::Run
            && progress
                == ExecutionProgress::Finished(ExecutionOutcome::Refused(AccelRefusal::Busy))
        {
            self.phase = before;
            return Ok(ExecutionProgress::Pending);
        }
        Ok(progress)
    }

    pub async fn advance_bounded<T: Transport>(
        &mut self,
        accel: &mut RingAccel<'_, '_, T>,
        budget_ns: u64,
    ) -> Result<ExecutionProgress, ExecutionError> {
        self.advance_with_budget(accel, Some(budget_ns)).await
    }

    pub async fn advance<T: Transport>(
        &mut self,
        accel: &mut RingAccel<'_, '_, T>,
    ) -> Result<ExecutionProgress, ExecutionError> {
        self.advance_with_budget(accel, None).await
    }

    async fn advance_with_budget<T: Transport>(
        &mut self,
        accel: &mut RingAccel<'_, '_, T>,
        budget_ns: Option<u64>,
    ) -> Result<ExecutionProgress, ExecutionError> {
        if let Phase::Finished(_) = self.phase {
            return Ok(self.progress());
        }
        advance_phase(accel, &mut self.phase, &mut self.intent, budget_ns).await?;
        advance_sequence(
            &mut self.phase,
            &mut self.remaining,
            &mut self.completed,
            self.intent,
        );
        Ok(self.progress())
    }
}

impl<T: Transport, I: Iterator> Execution<'_, '_, '_, T, I>
where
    I::Item: Into<ExecutionSubmission>,
{
    /// Stop this sequence and request cancellation of its current admission.
    /// This publishes nothing. Keep advancing: an outstanding submit or poll
    /// must finish its exact observation before cancellation can be sent.
    /// An unpublished submission is skipped; a completed event keeps its real
    /// outcome. No successor starts, even if completion beats cancellation.
    ///
    /// A confirmed cancellation refusal returns to observation and retirement;
    /// it cannot resume the sequence. Call again to retry cancellation after
    /// such a refusal. Transport uncertainty retains the original request.
    pub fn request_cancel(&mut self) {
        if !matches!(self.phase, Phase::Finished(_)) {
            self.intent = Intent::Cancel;
        }
    }

    /// Successfully executed and retired entries, excluding any live event.
    pub fn completed(&self) -> usize {
        self.completed
    }

    /// The admitted event, including while its terminal observation is waiting
    /// for actual retirement. `None` before the admission reply is not proof
    /// that no device work exists.
    pub fn event(&self) -> Option<u64> {
        match self.phase {
            Phase::Poll(event) | Phase::Cancel(event) | Phase::Retire { event, .. } => Some(event),
            _ => None,
        }
    }

    /// Whether progress is currently waiting on the accelerator rather than
    /// on another control-plane operation.
    ///
    /// A completion notice is useful only in this phase. In particular, a
    /// terminal poll moves the lifecycle to `Retire`; callers should advance
    /// that request immediately instead of sleeping for a notice that has
    /// already been consumed. The event id alone cannot distinguish those two
    /// states because retirement deliberately retains it.
    pub fn awaiting_device_completion(&self) -> bool {
        matches!(self.phase, Phase::Poll(_))
    }

    /// Observe one protocol operation. Cancellation preserves the exact
    /// continuation. Infrastructure errors leave it resumable where the
    /// transport permits; a poisoned pair requires owner teardown.
    ///
    /// `Pending` includes a refused Busy retirement even after a failed or
    /// cancelled event: a terminal failure is not a DMA quiescence proof.
    pub async fn advance(&mut self) -> Result<ExecutionProgress, ExecutionError> {
        self.advance_with_budget(None).await
    }

    /// Observe for at most this reply-wait slice. Expiry preserves the exact
    /// request; it is not cancellation or proof of failed admission. Requires
    /// the client's executor timer, as does `SvcClient::call_bounded`.
    pub async fn advance_bounded(
        &mut self,
        budget_ns: u64,
    ) -> Result<ExecutionProgress, ExecutionError> {
        self.advance_with_budget(Some(budget_ns)).await
    }

    async fn advance_with_budget(
        &mut self,
        budget_ns: Option<u64>,
    ) -> Result<ExecutionProgress, ExecutionError> {
        if let Phase::Finished(outcome) = self.phase {
            return Ok(ExecutionProgress::Finished(outcome));
        }
        advance_phase(self.accel, &mut self.phase, &mut self.intent, budget_ns).await?;
        advance_sequence(
            &mut self.phase,
            &mut self.remaining,
            &mut self.completed,
            self.intent,
        );
        Ok(match self.phase {
            Phase::Finished(outcome) => ExecutionProgress::Finished(outcome),
            _ => ExecutionProgress::Pending,
        })
    }
}

// One protocol engine for serial sequences and independently live admissions.
async fn advance_phase<T: Transport>(
    accel: &mut RingAccel<'_, '_, T>,
    phase: &mut Phase,
    intent: &mut Intent,
    budget_ns: Option<u64>,
) -> Result<Option<AccelEventState>, ExecutionError> {
    prepare_cancellation(phase, *intent, accel.client.is_idle());
    let request = match *phase {
        Phase::Submit(request) => request,
        Phase::Poll(event) => AccelRequest::PollEvent { event },
        Phase::Cancel(event) => AccelRequest::CancelEvent { event },
        Phase::Retire { event, .. } => AccelRequest::DestroyEvent { event },
        Phase::Finished(_) => return Ok(None),
    };
    // NoOutstanding proves that cancellation happened before publication,
    // or the preceding operation was fully consumed. Other errors never
    // authorize a replacement request. There is no await between consuming
    // a reply and recording its next phase below.
    let resumed = match budget_ns {
        Some(budget) => accel.client.resume_exact_bounded(&request, budget).await,
        None => accel.client.resume_exact(&request).await,
    };
    let reply = match resumed {
        Err(CallError::NoOutstanding) => match budget_ns {
            Some(budget) => accel.client.call_bounded(&request, budget).await,
            None => accel.client.call(&request).await,
        },
        result => result,
    }
    .map_err(ExecutionError::Infrastructure)?;
    if reply.grants != 0 {
        accel.client.poison_retained(reply.retained);
        return Err(ExecutionError::Infrastructure(CallError::Wire));
    }
    if let Err(error) = accept_execution_response(phase, intent, reply.response) {
        if error == ExecutionError::Infrastructure(CallError::Wire) {
            accel.client.poison();
        }
        return Err(error);
    }
    Ok(match (request, reply.response) {
        (AccelRequest::PollEvent { .. }, AccelResponse::Event { state, .. }) => Some(state),
        _ => None,
    })
}

// Never replace a published operation. In particular, a lost submit reply
// can hide an admitted event, while a lost poll reply still owns correlation.
fn prepare_cancellation(phase: &mut Phase, intent: Intent, idle: bool) {
    if intent != Intent::Cancel || !idle {
        return;
    }
    match *phase {
        Phase::Submit(_) => *phase = Phase::Finished(ExecutionOutcome::NotSubmitted),
        Phase::Poll(event) => *phase = Phase::Cancel(event),
        _ => {}
    }
}

fn accept_execution_response(
    phase: &mut Phase,
    intent: &mut Intent,
    response: AccelResponse,
) -> Result<(), ExecutionError> {
    let cancelling = matches!(*phase, Phase::Cancel(_));
    let result = accept_response(phase, response);
    if cancelling && !matches!(result, Err(ExecutionError::Infrastructure(_))) {
        *intent = Intent::Drain;
    }
    result
}

// Called exactly once after consuming a reply. A subsequent observation of an
// already-finished sequence returns before this transition, preserving counts.
fn advance_sequence<I: Iterator>(
    phase: &mut Phase,
    remaining: &mut I,
    completed: &mut usize,
    intent: Intent,
) where
    I::Item: Into<ExecutionSubmission>,
{
    if matches!(
        *phase,
        Phase::Finished(ExecutionOutcome::Retired(AccelEventState::Complete))
    ) {
        *completed += 1;
        if intent == Intent::Run
            && let Some(next) = remaining.next()
        {
            *phase = Phase::Submit(next.into().request());
        }
    }
}

// Pure reply reduction is also used by the real ring path above. In particular,
// a failed event does not bypass the separate, potentially Busy retirement.
fn accept_response(phase: &mut Phase, response: AccelResponse) -> Result<(), ExecutionError> {
    match (*phase, response) {
        (
            Phase::Submit(_),
            AccelResponse::Submitted { event } | AccelResponse::SubmitIndeterminate { event, .. },
        ) if event != 0 => {
            *phase = Phase::Poll(event);
        }
        (Phase::Submit(_), AccelResponse::Refused(refusal)) => {
            *phase = Phase::Finished(ExecutionOutcome::Refused(refusal));
        }
        (Phase::Poll(event), AccelResponse::Event { event: got, state }) if got == event => {
            if state != AccelEventState::Pending {
                *phase = Phase::Retire {
                    event,
                    terminal: state,
                };
            }
        }
        (Phase::Cancel(event), AccelResponse::Cancelled { event: got }) if got == event => {
            *phase = Phase::Retire {
                event,
                terminal: AccelEventState::Cancelled,
            };
        }
        (Phase::Cancel(event), AccelResponse::Refused(refusal)) => {
            *phase = Phase::Poll(event);
            // Busy commonly means completion won. Observe that outcome rather
            // than inventing Cancelled or dropping its event references.
            if refusal != AccelRefusal::Busy {
                return Err(ExecutionError::Refused(refusal));
            }
        }
        (Phase::Retire { event, terminal }, AccelResponse::Released { subject })
            if subject == event =>
        {
            *phase = Phase::Finished(ExecutionOutcome::Retired(terminal));
        }
        (Phase::Retire { .. }, AccelResponse::Refused(AccelRefusal::Busy)) => {}
        (_, AccelResponse::Refused(refusal)) => return Err(ExecutionError::Refused(refusal)),
        _ => {
            return Err(ExecutionError::Infrastructure(CallError::Wire));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)]
    use super::*;

    fn bindings() -> AccelBindings {
        AccelBindings::new(&[crate::service::AccelBinding {
            buffer: 3,
            offset: 0,
            len: 4,
            slot: 0,
            access: virtio_accel_core::AccessMode::Read,
        }])
        .unwrap()
    }

    #[test]
    fn ordered_packing_preserves_bindings_and_splits_reused_programs() {
        use zerocopy::FromBytes;
        let launches = [1, 2, 3, 2, 3, 4].map(|program| ArtifactExecution {
            queue: 7,
            program,
            bindings: bindings(),
            timeout_ns: 99,
        });
        const ENTRY: usize = core::mem::size_of::<kore_abi::svc::AccelSequenceEntryWire>();
        let mut bytes = [0xa5; 6 * ENTRY + 1];
        let mut groups = [None; 6];
        assert_eq!(
            stage_ordered_executions(&launches, &mut bytes, 4096, &mut groups),
            Ok(2)
        );
        assert_eq!(groups[0].unwrap().count, 3);
        assert_eq!(groups[1].unwrap().count, 3);
        assert_eq!(groups[1].unwrap().stage, 4096 + 3 * ENTRY as u64);
        assert!(groups[2..].iter().all(Option::is_none));
        assert_eq!(bytes[6 * ENTRY], 0xa5);
        for (index, launch) in launches.iter().enumerate() {
            let wire = kore_abi::svc::AccelSequenceEntryWire::read_from_bytes(
                &bytes[index * ENTRY..(index + 1) * ENTRY],
            )
            .unwrap();
            let entry = crate::wire::decode_sequence_entry(&wire, &mut core::iter::empty()).unwrap();
            assert_eq!(entry.program, launch.program);
            assert_eq!(entry.bindings, launch.bindings);
        }
        let before = bytes;
        assert_eq!(
            stage_ordered_executions(&launches, &mut bytes[..ENTRY], 0, &mut groups),
            Err(AccelRefusal::OutOfBounds)
        );
        assert_eq!(bytes, before);
        assert_eq!(
            stage_ordered_executions(&launches, &mut bytes, u64::MAX, &mut groups),
            Err(AccelRefusal::OutOfBounds)
        );
    }

    /// Each staged table is its entries, then its members' gate and signal
    /// records; a table ends before its records would pass the ceiling.
    #[test]
    fn staged_tables_carry_their_sync_records_and_split_at_their_ceiling() {
        const ENTRY: usize = core::mem::size_of::<kore_abi::svc::AccelSequenceEntryWire>();
        const SYNC: usize = core::mem::size_of::<kore_abi::svc::AccelSequenceSyncWire>();
        let word = |offset| Some(crate::service::AccelSync { buffer: 0x30, offset, value: offset });
        let launch = |index: u64| OrderedLaunch {
            queue: 7,
            program: 1,
            bindings: bindings(),
            timeout_ns: 99,
            launch: LaunchShape::ArtifactDefined,
            barrier: true,
            // Every member signals; every other one is gated too.
            gate: if index % 2 == 1 { word(index * 16) } else { None },
            signal: word(index * 16 + 8),
        };
        let launches: alloc::vec::Vec<OrderedLaunch> = (0..40).map(launch).collect();
        let mut bytes = alloc::vec![0; 40 * ENTRY + 60 * SYNC];
        let mut groups = [None; 40];
        let tables = stage_ordered_launches(&launches, &mut bytes, 4096, &mut groups, ProgramReuse::Share).unwrap();
        // 40 members carry 60 records: the first table stops at 48 (32
        // members), the second holds the other 8 members' 12.
        assert_eq!(tables, 2);
        let (first, second) = (groups[0].unwrap(), groups[1].unwrap());
        assert_eq!((first.count, first.stage, first.len), (32, 4096, (32 * ENTRY + 48 * SYNC) as u64));
        assert_eq!(
            (second.count, second.stage, second.len),
            (8, 4096 + first.len, (8 * ENTRY + 12 * SYNC) as u64)
        );
        // Each table decodes on its own back to its members' words.
        let mut member = 0;
        for group in [first, second] {
            let start = (group.stage - 4096) as usize;
            let table = &bytes[start..start + group.len as usize];
            for entry in crate::wire::decode_sequence(table, group.count as usize).unwrap() {
                assert_eq!((entry.gate, entry.signal), (launches[member].gate, launches[member].signal));
                member += 1;
            }
        }
        assert_eq!(member, 40);
        // Too little stage for the records refuses.
        let mut short = alloc::vec![0; 40 * ENTRY];
        assert_eq!(
            stage_ordered_launches(&launches, &mut short, 4096, &mut groups, ProgramReuse::Share),
            Err(AccelRefusal::OutOfBounds)
        );
    }

    #[test]
    fn ordered_launches_carry_geometry_and_barriers_and_may_share_programs() {
        use zerocopy::FromBytes;
        const ENTRY: usize = core::mem::size_of::<kore_abi::svc::AccelSequenceEntryWire>();
        let explicit = LaunchShape::Workgroups(Workgroups::new([257, 3, 2], [32, 1, 1]).unwrap());
        let launches = [1, 2, 1, 2].map(|program| OrderedLaunch {
            queue: 7,
            program,
            bindings: bindings(),
            timeout_ns: 99,
            launch: if program == 1 {
                explicit
            } else {
                LaunchShape::ArtifactDefined
            },
            barrier: program == 2,
            gate: None,
            signal: None,
        });
        let decode = |bytes: &[u8], index: usize| {
            let wire = kore_abi::svc::AccelSequenceEntryWire::read_from_bytes(
                &bytes[index * ENTRY..(index + 1) * ENTRY],
            )
            .unwrap();
            crate::wire::decode_sequence_entry(&wire, &mut core::iter::empty()).unwrap()
        };
        let mut bytes = [0; 4 * ENTRY];
        let mut groups = [None; 4];
        // A provider that shares immutable code keeps repeats in one table.
        assert_eq!(
            stage_ordered_launches(&launches, &mut bytes, 512, &mut groups, ProgramReuse::Share),
            Ok(1)
        );
        let table = groups[0].unwrap();
        assert_eq!((table.count, table.stage), (4, 512));
        assert_eq!(table.len, 4 * ENTRY as u64);
        for (index, launch) in launches.iter().enumerate() {
            let entry = decode(&bytes, index);
            assert_eq!(entry.program, launch.program);
            assert_eq!(entry.bindings, launch.bindings);
            assert_eq!(entry.launch, launch.launch);
            assert_eq!(entry.barrier, launch.barrier);
        }
        // The same members still fence at every repeat under `Split`.
        assert_eq!(
            stage_ordered_launches(&launches, &mut bytes, 0, &mut groups, ProgramReuse::Split),
            Ok(2)
        );
        assert_eq!(groups[0].unwrap().count, 2);
        assert_eq!(groups[1].unwrap().stage, 2 * ENTRY as u64);

        // Artifact executions keep the historical meaning: artifact-defined
        // members, each dependent on everything staged before it.
        let artifact = [3, 4].map(|program| ArtifactExecution {
            queue: 7,
            program,
            bindings: bindings(),
            timeout_ns: 99,
        });
        assert_eq!(
            stage_ordered_executions(&artifact, &mut bytes, 0, &mut groups),
            Ok(1)
        );
        for index in 0..2 {
            let entry = decode(&bytes, index);
            assert_eq!(entry.launch, LaunchShape::ArtifactDefined);
            assert!(entry.barrier);
        }
    }

    #[test]
    fn ordered_packing_obeys_wire_ceiling_queue_and_deadline_boundaries() {
        const N: usize = kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES + 3;
        let mut launches: [ArtifactExecution; N] = core::array::from_fn(|i| ArtifactExecution {
            queue: 1,
            program: i as u64 + 1,
            bindings: bindings(),
            timeout_ns: 99,
        });
        launches[N - 2].queue = 2;
        launches[N - 1].queue = 2;
        launches[N - 1].timeout_ns = 100;
        let mut bytes = [0; N * core::mem::size_of::<kore_abi::svc::AccelSequenceEntryWire>()];
        let mut groups = [None; N];
        assert_eq!(
            stage_ordered_executions(&launches, &mut bytes, 0, &mut groups),
            Ok(4)
        );
        assert_eq!(
            groups[0].unwrap().count as usize,
            kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES
        );
        assert_eq!(groups[1].unwrap().count, 1);
        assert_eq!(groups[2].unwrap().queue, 2);
        assert_eq!(groups[3].unwrap().timeout_ns, 100);
    }

    #[test]
    fn ordered_failure_and_busy_retirement_never_publish_a_successor() {
        let batch = OrderedExecution {
            queue: 1,
            stage: 4096,
            len: 256,
            count: 2,
            timeout_ns: 99,
        };
        let mut phase = Phase::Submit(ExecutionSubmission::from(batch).request());
        let mut remaining = [batch].into_iter();
        let mut completed = 0;
        accept_response(
            &mut phase,
            AccelResponse::SubmitIndeterminate {
                event: 9,
                refusal: AccelRefusal::DeviceLost,
            },
        )
        .unwrap();
        accept_response(
            &mut phase,
            AccelResponse::Event {
                event: 9,
                state: AccelEventState::Failed(AccelRefusal::DeviceLost),
            },
        )
        .unwrap();
        accept_response(&mut phase, AccelResponse::Refused(AccelRefusal::Busy)).unwrap();
        advance_sequence(&mut phase, &mut remaining, &mut completed, Intent::Run);
        assert_eq!(remaining.len(), 1);
        assert!(matches!(phase, Phase::Retire { event: 9, .. }));
        accept_response(&mut phase, AccelResponse::Released { subject: 9 }).unwrap();
        advance_sequence(&mut phase, &mut remaining, &mut completed, Intent::Run);
        assert_eq!(completed, 0);
        assert_eq!(remaining.len(), 1);
        assert!(matches!(
            phase,
            Phase::Finished(ExecutionOutcome::Retired(_))
        ));
    }

    #[test]
    fn detached_ledger_retains_uncertain_submission_after_cancelled_observation() {
        uncertain_submission(
            OrderedExecution {
                queue: 1,
                stage: 0,
                len: 256,
                count: 1,
                timeout_ns: 100,
            }
            .into(),
        );
    }

    #[test]
    fn detached_workgroups_preserve_geometry_across_uncertain_admission() {
        uncertain_submission(
            LaunchExecution {
                queue: 1,
                program: 2,
                bindings: bindings(),
                timeout_ns: 100,
                launch: LaunchShape::Workgroups(Workgroups::new([257, 3, 2], [32, 1, 1]).unwrap()),
            }
            .into(),
        );
    }

    fn uncertain_submission(launch: ExecutionSubmission) {
        use crate::wire::AccelWire;
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        use kore_abi::session::SlotRef;
        use kore_rt::{
            svc::{ClientEndpoints, REP_CAP, REQ_CAP, SvcClient},
            transport::TrapTransport,
        };

        fn ready<F: Future>(future: F) -> F::Output {
            let mut future = core::pin::pin!(future);
            match future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
            {
                Poll::Ready(value) => value,
                Poll::Pending => panic!("local transport refusal must be ready"),
            }
        }
        const BYTES: usize = kore_rt::svc::window_bytes::<AccelWire>(REQ_CAP, REP_CAP);
        #[repr(align(64))]
        struct Window([u8; BYTES]);
        let mut window = Window([0; BYTES]);
        let (rings, mut server) =
            kore_rt::svc::split_window::<AccelWire>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        let transport = TrapTransport::new();
        let mut accel = RingAccel {
            client: SvcClient::new(
                &transport,
                rings,
                ClientEndpoints {
                    kick: SlotRef::from_wire(1),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            ),
            return_tx: SlotRef::from_wire(2),
            return_land: SlotRef::from_wire(3),
        };
        let mut ledger = accel
            .execution_ledger_sequence([launch].into_iter())
            .unwrap();
        assert_eq!(
            ready(ledger.advance(&mut accel)),
            Err(ExecutionError::Infrastructure(CallError::Transport)),
        );
        let published = server.req.pop().expect("submission was published once");
        assert_eq!(crate::wire::decode_req(&published), Some(launch.request()));
        ledger.request_cancel();
        assert_eq!(
            ready(ledger.advance(&mut accel)),
            Err(ExecutionError::Infrastructure(CallError::Transport)),
        );
        assert!(
            server.req.pop().is_none(),
            "cancel may not replace an uncertain submit"
        );
        assert_eq!(ledger.progress(), ExecutionProgress::Pending);
        assert_eq!(ledger.event(), None);
    }

    fn submit() -> Phase {
        #[allow(deprecated)]
        Phase::Submit(AccelRequest::Submit {
            queue: 1,
            program: 2,
            timeout_ns: 100,
            bindings: bindings(),
            dispatch: kore_driver::accel::Dispatch::scalar(),
        })
    }

    #[test]
    fn real_host_transport_failure_preserves_the_only_published_submission() {
        use crate::wire::AccelWire;
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        use kore_abi::session::SlotRef;
        use kore_rt::{
            svc::{ClientEndpoints, REP_CAP, REQ_CAP, SvcClient},
            transport::TrapTransport,
        };
        const BYTES: usize = kore_rt::svc::window_bytes::<AccelWire>(REQ_CAP, REP_CAP);
        #[repr(align(64))]
        struct Window([u8; BYTES]);
        let mut window = Window([0; BYTES]);
        let (rings, mut server) =
            kore_rt::svc::split_window::<AccelWire>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        // Host TrapTransport actually refuses: no mocked successful kernel
        // commit, fabricated device completion, or test executor is involved.
        let transport = TrapTransport::new();
        let mut accel = RingAccel {
            client: SvcClient::new(
                &transport,
                rings,
                ClientEndpoints {
                    kick: SlotRef::from_wire(1),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            ),
            return_tx: SlotRef::from_wire(2),
            return_land: SlotRef::from_wire(3),
        };
        {
            let workgroups = Workgroups::new([2, 1, 1], [256, 1, 1]).unwrap();
            let mut xve = accel
                .execution_workgroups(1, 2, bindings(), workgroups, 100)
                .unwrap();
            assert!(matches!(
                xve.phase,
                Phase::Submit(AccelRequest::SubmitLaunch {
                    launch: LaunchShape::Workgroups(actual),
                    ..
                }) if actual == workgroups
            ));
            xve.request_cancel();
            let mut future = core::pin::pin!(xve.advance());
            assert_eq!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Ok(ExecutionProgress::Finished(
                    ExecutionOutcome::NotSubmitted
                )))
            );
            assert!(
                server.req.pop().is_none(),
                "local stop must not publish XVE"
            );
        }
        {
            let mut stopped = accel.execution(1, 2, bindings(), 100).unwrap();
            stopped.request_cancel();
            for _ in 0..2 {
                let mut future = core::pin::pin!(stopped.advance());
                assert_eq!(
                    future
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop())),
                    Poll::Ready(Ok(ExecutionProgress::Finished(
                        ExecutionOutcome::NotSubmitted
                    )))
                );
            }
            assert!(server.req.pop().is_none(), "local stop must not publish");
        }
        let mut execution = accel.execution(1, 2, bindings(), 100).unwrap();
        for attempt in 0..2 {
            if attempt == 1 {
                execution.request_cancel();
            }
            let mut future = core::pin::pin!(execution.advance());
            let result = future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()));
            assert_eq!(
                result,
                Poll::Ready(Err(ExecutionError::Infrastructure(CallError::Transport)))
            );
            if attempt == 0 {
                let wire = server
                    .req
                    .pop()
                    .expect("the request preceded the failed kick");
                assert!(matches!(
                    crate::wire::decode_req(&wire),
                    Some(AccelRequest::Submit {
                        queue: 1,
                        program: 2,
                        ..
                    })
                ));
            }
            assert!(
                server.req.pop().is_none(),
                "resume must not duplicate admission"
            );
        }
        assert_eq!(
            execution.event(),
            None,
            "missing reply cannot invent an event"
        );
        drop(execution);
        assert!(!accel.client.is_idle());
        assert!(accel.execution(1, 2, bindings(), 100).is_none());
    }

    #[test]
    fn cancellation_waits_for_exact_observation_and_acknowledged_retirement() {
        let mut phase = submit();
        let mut intent = Intent::Cancel;
        prepare_cancellation(&mut phase, intent, false);
        assert!(
            matches!(phase, Phase::Submit(_)),
            "unknown admission must be resumed"
        );
        accept_execution_response(
            &mut phase,
            &mut intent,
            AccelResponse::SubmitIndeterminate {
                event: 42,
                refusal: AccelRefusal::DeadlineExpired,
            },
        )
        .unwrap();
        prepare_cancellation(&mut phase, intent, false);
        assert!(
            matches!(phase, Phase::Poll(42)),
            "a published poll keeps its correlation"
        );
        prepare_cancellation(&mut phase, intent, true);
        assert!(matches!(phase, Phase::Cancel(42)));
        assert_eq!(
            accept_execution_response(
                &mut phase,
                &mut intent,
                AccelResponse::Cancelled { event: 43 }
            ),
            Err(ExecutionError::Infrastructure(CallError::Wire))
        );
        assert!(matches!(phase, Phase::Cancel(42)));
        assert!(intent == Intent::Cancel);
        accept_execution_response(
            &mut phase,
            &mut intent,
            AccelResponse::Cancelled { event: 42 },
        )
        .unwrap();
        assert!(intent == Intent::Drain);
        for _ in 0..3 {
            accept_execution_response(
                &mut phase,
                &mut intent,
                AccelResponse::Refused(AccelRefusal::Busy),
            )
            .unwrap();
            assert!(matches!(
                phase,
                Phase::Retire {
                    event: 42,
                    terminal: AccelEventState::Cancelled
                }
            ));
        }
        accept_execution_response(
            &mut phase,
            &mut intent,
            AccelResponse::Released { subject: 42 },
        )
        .unwrap();
        assert!(matches!(
            phase,
            Phase::Finished(ExecutionOutcome::Retired(AccelEventState::Cancelled))
        ));
    }

    #[test]
    fn cancellation_refusal_drains_real_outcome_without_launching_successors() {
        for refusal in [AccelRefusal::Busy, AccelRefusal::Unsupported] {
            let mut phase = Phase::Cancel(42);
            let mut intent = Intent::Cancel;
            let result =
                accept_execution_response(&mut phase, &mut intent, AccelResponse::Refused(refusal));
            assert_eq!(
                result,
                if refusal == AccelRefusal::Busy {
                    Ok(())
                } else {
                    Err(ExecutionError::Refused(refusal))
                }
            );
            assert!(intent == Intent::Drain);
            prepare_cancellation(&mut phase, intent, true);
            assert!(
                matches!(phase, Phase::Poll(42)),
                "refusal resumes observation"
            );
            accept_execution_response(
                &mut phase,
                &mut intent,
                AccelResponse::Event {
                    event: 42,
                    state: AccelEventState::Complete,
                },
            )
            .unwrap();
            accept_execution_response(
                &mut phase,
                &mut intent,
                AccelResponse::Released { subject: 42 },
            )
            .unwrap();
            let mut remaining = core::iter::once(ArtifactExecution {
                queue: 1,
                program: 9,
                bindings: bindings(),
                timeout_ns: 100,
            });
            let mut completed = 0;
            advance_sequence(&mut phase, &mut remaining, &mut completed, intent);
            assert_eq!(completed, 1, "completion won and must remain successful");
            assert_eq!(
                remaining.len(),
                1,
                "stop intent survives cancellation refusal"
            );
            assert!(matches!(
                phase,
                Phase::Finished(ExecutionOutcome::Retired(AccelEventState::Complete))
            ));
        }
    }

    #[test]
    fn successors_wait_for_successful_retirement_and_failure_stops_the_sequence() {
        let next = ArtifactExecution {
            queue: 1,
            program: 9,
            bindings: bindings(),
            timeout_ns: 100,
        };
        let mut remaining = core::iter::once(next);
        let mut completed = 0;
        let mut phase = Phase::Poll(42);
        for response in [
            AccelResponse::Event {
                event: 42,
                state: AccelEventState::Pending,
            },
            AccelResponse::Event {
                event: 42,
                state: AccelEventState::Complete,
            },
            AccelResponse::Refused(AccelRefusal::Busy),
        ] {
            accept_response(&mut phase, response).unwrap();
            advance_sequence(&mut phase, &mut remaining, &mut completed, Intent::Run);
            assert_eq!(completed, 0);
            assert_eq!(
                remaining.len(),
                1,
                "completion alone cannot submit a successor"
            );
        }
        accept_response(&mut phase, AccelResponse::Released { subject: 42 }).unwrap();
        advance_sequence(&mut phase, &mut remaining, &mut completed, Intent::Run);
        assert_eq!(completed, 1);
        assert!(matches!(
            phase,
            Phase::Submit(AccelRequest::Submit { program: 9, .. })
        ));
        for outcome in [
            ExecutionOutcome::Refused(AccelRefusal::NotReady),
            ExecutionOutcome::Retired(AccelEventState::Failed(AccelRefusal::Internal)),
            ExecutionOutcome::Retired(AccelEventState::Cancelled),
        ] {
            let mut phase = Phase::Finished(outcome);
            let mut remaining = core::iter::once(next);
            advance_sequence(&mut phase, &mut remaining, &mut completed, Intent::Run);
            assert_eq!(remaining.len(), 1);
            assert_eq!(completed, 1);
        }
    }

    #[test]
    fn indeterminate_failure_retains_event_until_retirement_acknowledged() {
        let mut phase = submit();
        accept_response(
            &mut phase,
            AccelResponse::SubmitIndeterminate {
                event: 42,
                refusal: AccelRefusal::DeadlineExpired,
            },
        )
        .unwrap();
        assert!(matches!(phase, Phase::Poll(42)));
        accept_response(
            &mut phase,
            AccelResponse::Event {
                event: 42,
                state: AccelEventState::Failed(AccelRefusal::DeadlineExpired),
            },
        )
        .unwrap();
        for _ in 0..3 {
            accept_response(&mut phase, AccelResponse::Refused(AccelRefusal::Busy)).unwrap();
            assert!(matches!(phase, Phase::Retire { event: 42, .. }));
        }
        accept_response(&mut phase, AccelResponse::Released { subject: 42 }).unwrap();
        assert!(matches!(
            phase,
            Phase::Finished(ExecutionOutcome::Retired(AccelEventState::Failed(
                AccelRefusal::DeadlineExpired
            )))
        ));
    }

    #[test]
    fn wrong_identity_or_reply_never_releases_retained_event() {
        let mut phase = Phase::Poll(42);
        for reply in [
            AccelResponse::Event {
                event: 43,
                state: AccelEventState::Complete,
            },
            AccelResponse::Released { subject: 42 },
        ] {
            assert_eq!(
                accept_response(&mut phase, reply),
                Err(ExecutionError::Infrastructure(CallError::Wire))
            );
            assert!(matches!(phase, Phase::Poll(42)));
        }
        accept_response(
            &mut phase,
            AccelResponse::Event {
                event: 42,
                state: AccelEventState::Complete,
            },
        )
        .unwrap();
        assert_eq!(
            accept_response(&mut phase, AccelResponse::Released { subject: 43 }),
            Err(ExecutionError::Infrastructure(CallError::Wire))
        );
        assert!(matches!(
            phase,
            Phase::Retire {
                event: 42,
                terminal: AccelEventState::Complete
            }
        ));
    }

    #[test]
    fn refusal_is_terminal_only_before_admission() {
        let mut phase = submit();
        accept_response(&mut phase, AccelResponse::Refused(AccelRefusal::NotReady)).unwrap();
        assert!(matches!(
            phase,
            Phase::Finished(ExecutionOutcome::Refused(AccelRefusal::NotReady))
        ));
        for mut phase in [
            Phase::Poll(42),
            Phase::Retire {
                event: 42,
                terminal: AccelEventState::Cancelled,
            },
        ] {
            assert_eq!(
                accept_response(&mut phase, AccelResponse::Refused(AccelRefusal::NotReady)),
                Err(ExecutionError::Refused(AccelRefusal::NotReady))
            );
            assert!(!matches!(phase, Phase::Finished(_)));
        }
    }
}
