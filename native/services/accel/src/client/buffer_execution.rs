//! Compose retained buffer custody with the existing execution ledger.

use super::{
    ExecutionError, ExecutionLedger, ExecutionOutcome, ExecutionProgress, ExportReply, ImportReply,
    ImportTarget, LaunchExecution, RingAccel,
};
use crate::service::{AccelEventState, AccelRefusal, MAX_ACCEL_BINDINGS};
use kore_rt::{
    cap::StashedMemory,
    svc::{CallError, RetainedReturnCallError},
    transport::Transport,
};

/// One input to a buffer-backed execution. Resident buffers stay with their
/// existing owner; transferred allocations are returned only after export.
/// No local CPU alias may access a transferred allocation until it is returned.
pub enum ExecutionBuffer {
    Unused,
    Resident(u64),
    Transfer {
        target: ImportTarget,
        source: StashedMemory,
    },
}

/// Bind the provider's final import IDs to one explicit launch. This is the
/// existing buffer continuation's binding callback, also implementable by a
/// named plan that a runtime can store without boxing an anonymous closure.
/// Implementations carry request metadata, never allocation authority. Called
/// once after successful imports; refusal unwinds custody without submission.
pub trait LaunchBuilder<const N: usize> {
    fn build(&self, buffers: &[u64; N]) -> Result<LaunchExecution, AccelRefusal>;
}

impl<const N: usize, F> LaunchBuilder<N> for F
where
    F: Fn(&[u64; N]) -> Result<LaunchExecution, AccelRefusal>,
{
    fn build(&self, buffers: &[u64; N]) -> Result<LaunchExecution, AccelRefusal> {
        self(buffers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            Poll::Ready(result) => result,
            Poll::Pending => panic!("local custody transition must not wait"),
        }
    }

    fn with_face(test: impl FnOnce(&mut RingAccel<'_, '_, TrapTransport>)) {
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
        test(&mut accel);
        assert!(
            server.req.pop().is_none(),
            "local refusal/cancellation must not publish or export resident buffers"
        );
    }

    #[test]
    fn cancellation_before_admission_leaves_resident_buffers_with_their_owner() {
        with_face(|accel| {
            let mut execution = accel
                .buffer_execution([ExecutionBuffer::Resident(7)], |_: &[u64; 1]| {
                    panic!("cancelled execution must not construct a launch")
                })
                .ok()
                .unwrap();
            execution.request_cancel();
            assert_eq!(
                ready(execution.advance_bounded(accel, 1)),
                Ok(ExecutionProgress::Finished(ExecutionOutcome::NotSubmitted))
            );
            assert!(execution.take_buffer(0).is_none());
            assert_eq!(execution.buffers[0].device_id(), 7);
        });
    }

    #[test]
    fn binding_refusal_does_not_submit_or_export_resident_weights() {
        with_face(|accel| {
            let mut execution = accel
                .buffer_execution([ExecutionBuffer::Resident(9)], |ids: &[u64; 1]| {
                    assert_eq!(*ids, [9]);
                    Err(AccelRefusal::InvalidArgument)
                })
                .ok()
                .unwrap();
            // Resident admission needs no scheduler round trip before binding.
            assert_eq!(
                ready(execution.advance_bounded(accel, 1)),
                Ok(ExecutionProgress::Pending)
            );
            assert_eq!(
                ready(execution.advance_bounded(accel, 1)),
                Ok(ExecutionProgress::Finished(ExecutionOutcome::Refused(
                    AccelRefusal::InvalidArgument
                )))
            );
            assert!(execution.take_buffer(0).is_none());
            assert_eq!(
                execution.failure_activity(),
                Some(BufferExecutionActivity::Execute)
            );
        });
    }

    #[test]
    fn resident_binding_ceiling_is_admitted_in_one_advance_without_wire_traffic() {
        use core::cell::Cell;
        with_face(|accel| {
            let calls = Cell::new(0);
            let mut execution = accel
                .buffer_execution(
                    core::array::from_fn(|index| {
                        if index % 2 == 0 {
                            ExecutionBuffer::Resident(index as u64 + 1)
                        } else {
                            ExecutionBuffer::Unused
                        }
                    }),
                    |ids: &[u64; MAX_ACCEL_BINDINGS]| {
                        calls.set(calls.get() + 1);
                        for (index, &id) in ids.iter().enumerate() {
                            assert_eq!(id, if index % 2 == 0 { index as u64 + 1 } else { 0 });
                        }
                        Err(AccelRefusal::Unsupported)
                    },
                )
                .ok()
                .unwrap();
            let activity = Cell::new(None);
            assert_eq!(
                ready(execution.advance_bounded_observed(accel, 1, |a| activity.set(Some(a)))),
                Ok(ExecutionProgress::Pending)
            );
            assert_eq!(activity.get(), Some(BufferExecutionActivity::Execute));
            assert_eq!(
                calls.get(),
                1,
                "all local slots precede the same binding attempt"
            );
            assert_eq!(
                ready(execution.advance_bounded(accel, 1)),
                Ok(ExecutionProgress::Finished(ExecutionOutcome::Refused(
                    AccelRefusal::Unsupported
                )))
            );
            assert_eq!(calls.get(), 1, "binding refusal cannot be replayed");
        });
    }

    #[test]
    fn cancellation_cannot_discard_unresolved_transferred_custody() {
        with_face(|accel| {
            let mut execution = accel
                .buffer_execution([ExecutionBuffer::Unused], |_: &[u64; 1]| {
                    panic!("unresolved import must not construct a launch")
                })
                .ok()
                .unwrap();
            // Model the boundary after capability transfer. Loss of correlation
            // must not become a successful cancellation or permission to reuse.
            execution.buffers[0] = BufferState::Import {
                target: ImportTarget {
                    context: 1,
                    bytes: 4096,
                    alignment: 4096,
                    usage: 0,
                },
                source: None,
            };
            execution.request_cancel();
            assert_eq!(
                ready(execution.advance_bounded(accel, 1)),
                Err(BufferExecutionError::Import(
                    RetainedReturnCallError::AfterTransfer(CallError::NoOutstanding)
                ))
            );
            assert_eq!(execution.progress(), ExecutionProgress::Pending);
            assert!(execution.take_buffer(0).is_none());
        });
    }

    #[test]
    fn returned_custody_is_unavailable_before_retirement_and_taken_once() {
        struct Lease;
        let mut state = BufferState::Local(Lease);
        assert!(state.take_local(false).is_none());
        assert!(state.take_local(true).is_some());
        assert!(state.take_local(true).is_none());
        let mut unknown = BufferState::<Lease>::Consumed;
        assert!(unknown.take_local(true).is_none());
    }

    #[test]
    fn uncertain_submission_keeps_owned_buffers_until_event_retirement() {
        use crate::service::{AccelBinding, AccelBindings, AccelRequest};
        use kore_driver::accel::{LaunchShape, Workgroups};
        use kore_rt::executor::Doorbell;

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
        let mut executor = kore_rt::Executor::from_standing(
            &transport,
            Doorbell {
                chan: 4,
                recv: SlotRef::from_wire(5),
                land: SlotRef::from_wire(6),
                recv_tag: 7,
            },
        );
        let mut execution = accel
            .buffer_execution(
                [
                    ExecutionBuffer::Resident(7),
                    ExecutionBuffer::Unused,
                    ExecutionBuffer::Unused,
                    ExecutionBuffer::Unused,
                    ExecutionBuffer::Resident(9),
                ],
                |ids: &[u64; 5]| {
                    assert_eq!(*ids, [7, 0, 17, 0, 9]);
                    Ok(LaunchExecution {
                        queue: 1,
                        program: 2,
                        timeout_ns: 100,
                        bindings: AccelBindings::new(&[AccelBinding {
                            buffer: ids[2],
                            offset: 0,
                            len: 4,
                            slot: 0,
                            access: virtio_accel_core::AccessMode::Write,
                        }])
                        .unwrap(),
                        launch: LaunchShape::Workgroups(Workgroups::new([1; 3], [1; 3]).unwrap()),
                    })
                },
            )
            .ok()
            .unwrap();
        // Admission has returned an owned buffer ID. Exercise the real ring
        // publication and host transport refusal, not a fabricated completion.
        execution.buffers[2] = BufferState::Imported(17);
        execution.phase = Phase::Import(3);
        assert_eq!(
            executor.run(execution.advance_bounded(&mut accel, 100)),
            Err(BufferExecutionError::Execution(
                ExecutionError::Infrastructure(CallError::Transport)
            ))
        );
        let published = server.req.pop().expect("one published launch");
        assert!(matches!(
            crate::wire::decode_req(&published),
            Some(AccelRequest::SubmitLaunch { .. })
        ));
        execution.request_cancel();
        assert_eq!(
            executor.run(execution.advance_bounded(&mut accel, 100)),
            Err(BufferExecutionError::Execution(
                ExecutionError::Infrastructure(CallError::Transport)
            ))
        );
        assert!(
            server.req.pop().is_none(),
            "neither export nor cancel may replace an uncertain submit"
        );
        assert_eq!(execution.progress(), ExecutionProgress::Pending);
        assert_eq!(execution.buffers[2].device_id(), 17);
        assert!(execution.take_buffer(2).is_none());
    }
}

enum BufferState<M> {
    Unused,
    Resident(u64),
    Import {
        target: ImportTarget,
        source: Option<M>,
    },
    Imported(u64),
    Local(M),
    /// A refused import left cleanup custody with the provider.
    Consumed,
}

impl<M> BufferState<M> {
    fn may_abandon_import(&self) -> bool {
        !matches!(self, Self::Import { source: None, .. })
    }
    fn device_id(&self) -> u64 {
        match self {
            Self::Resident(id) | Self::Imported(id) => *id,
            _ => 0,
        }
    }

    fn take_local(&mut self, finished: bool) -> Option<M> {
        if !finished {
            return None;
        }
        match self {
            Self::Import { source, .. } => source.take(),
            Self::Local(_) => match core::mem::replace(self, Self::Unused) {
                Self::Local(source) => Some(source),
                _ => None,
            },
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferExecutionError {
    Import(RetainedReturnCallError),
    Execution(ExecutionError),
    Export(CallError),
    /// Export refused; the same buffer ID remains retained for retry.
    Refused(AccelRefusal),
    /// The executor's own refusal under strict placement: a member with a
    /// loaded package it would not place, named by the reason (see
    /// `kore_axiom::xe_executor::MEMBER_DECLINES`), instead of a silent
    /// CPU fallback.
    Declined(&'static str),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Import(usize),
    Execute,
    Export,
    Finished,
}

/// Activity selected for one bounded advance, after local admission skips.
/// Execution includes publication, event observation and retirement; it is
/// not a device timestamp. Observers receive metadata, never buffer authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferExecutionActivity {
    Import { index: usize, bytes: u64 },
    Execute,
    Export { index: Option<usize> },
    Finished,
}

/// Task-owned import → execution/retirement → export continuation.
///
/// Construction publishes nothing. Each advance borrows the same possessed
/// face for at most one protocol operation. The caller reserves that face,
/// queue, program, and resident buffers until Finished; no other operation
/// may consume its pending replies. Allocation ownership stays here across
/// cancelled futures. Dropping this owner does not cancel submitted work:
/// imported allocations stay with the provider for face teardown/containment.
///
/// `make_launch` receives the final buffer IDs (zero for Unused slots) once,
/// after all imports succeed. It must bind only these IDs or other buffers
/// whose owner is retained for the same lifetime. A local binding refusal
/// unwinds imports without submitting work. Tensor bytes never use staging.
/// The fixed array is bounded by the protocol's binding ceiling, independent
/// of model size; resident weights need neither re-import nor payload copies.
#[must_use = "retain buffer custody until Finished or owning-face containment"]
pub struct BufferExecution<const N: usize, F> {
    buffers: [BufferState<StashedMemory>; N],
    make_launch: F,
    execution: Option<ExecutionLedger<core::iter::Once<LaunchExecution>>>,
    phase: Phase,
    outcome: ExecutionOutcome,
    cancelling: bool,
    failure_activity: Option<BufferExecutionActivity>,
}

impl<T: Transport> RingAccel<'_, '_, T> {
    /// Start on an idle face. Invalid import geometry or IDs return all input
    /// custody and the launch builder unchanged; the provider still enforces
    /// allocation bounds, access rights, contexts, and program bindings.
    pub fn buffer_execution<const N: usize, F>(
        &self,
        buffers: [ExecutionBuffer; N],
        make_launch: F,
    ) -> Result<BufferExecution<N, F>, ([ExecutionBuffer; N], F)>
    where
        F: LaunchBuilder<N>,
    {
        let valid = N != 0
            && N <= MAX_ACCEL_BINDINGS
            && buffers.iter().all(|buffer| match buffer {
                ExecutionBuffer::Unused => true,
                ExecutionBuffer::Resident(id) => *id != 0,
                ExecutionBuffer::Transfer { target, .. } => {
                    target.context != 0 && target.bytes != 0 && target.alignment.is_power_of_two()
                }
            });
        if !self.client.is_idle() || !valid {
            return Err((buffers, make_launch));
        }
        Ok(BufferExecution {
            buffers: buffers.map(|buffer| match buffer {
                ExecutionBuffer::Unused => BufferState::Unused,
                ExecutionBuffer::Resident(id) => BufferState::Resident(id),
                ExecutionBuffer::Transfer { target, source } => BufferState::Import {
                    target,
                    source: Some(source),
                },
            }),
            make_launch,
            execution: None,
            phase: Phase::Import(0),
            outcome: ExecutionOutcome::NotSubmitted,
            cancelling: false,
            failure_activity: None,
        })
    }
}

impl<const N: usize, F> BufferExecution<N, F>
where
    F: LaunchBuilder<N>,
{
    /// First refused activity, retained across export cleanup. This is
    /// diagnostic metadata only; it grants no access to a transferred buffer.
    pub fn failure_activity(&self) -> Option<BufferExecutionActivity> {
        self.failure_activity
    }

    /// Stop admitting further buffers/work. An in-flight import first resolves
    /// its exact reply; late success is exported. Existing execution uses the
    /// common cancellation/retirement engine before any export is attempted.
    pub fn request_cancel(&mut self) {
        self.cancelling = true;
        if let Some(execution) = &mut self.execution {
            execution.request_cancel();
        }
    }

    pub fn awaiting_device_completion(&self) -> bool {
        self.phase == Phase::Execute
            && self
                .execution
                .as_ref()
                .is_some_and(|e| e.awaiting_device_completion())
    }

    /// Recover an exported or never-transferred allocation once. None never
    /// authorizes CPU access: it may denote provider-owned containment custody.
    pub fn take_buffer(&mut self, index: usize) -> Option<StashedMemory> {
        self.buffers
            .get_mut(index)?
            .take_local(self.phase == Phase::Finished)
    }

    pub fn progress(&self) -> ExecutionProgress {
        if self.phase == Phase::Finished {
            ExecutionProgress::Finished(self.outcome)
        } else {
            ExecutionProgress::Pending
        }
    }

    /// Skip only local admission transitions. The index strictly increases up
    /// to the wire binding ceiling; there is no device observation or retry.
    /// An unresolved transfer stays on its exact import even after cancellation.
    fn skip_local_imports(&mut self, idle: bool) {
        while let Phase::Import(index) = self.phase {
            if self.cancelling
                && idle
                && self
                    .buffers
                    .get(index)
                    .is_none_or(BufferState::may_abandon_import)
            {
                self.phase = Phase::Export;
                break;
            }
            if index == N {
                self.phase = Phase::Execute;
                break;
            }
            if matches!(self.buffers[index], BufferState::Import { .. }) {
                break;
            }
            self.phase = Phase::Import(index + 1);
        }
    }

    /// Observe one operation with the caller's reply-wait budget. Timeouts and
    /// cancelled observations preserve the exact continuation, not permission
    /// to access memory. The caller supplies an overall deadline and keeps
    /// this owner for cancellation/draining after that deadline expires.
    /// Resident/unused admission slots are skipped locally before that one
    /// operation, without yielding or spending a separate reply-wait budget.
    pub async fn advance_bounded<T: Transport>(
        &mut self,
        accel: &mut RingAccel<'_, '_, T>,
        budget_ns: u64,
    ) -> Result<ExecutionProgress, BufferExecutionError> {
        self.advance_bounded_observed(accel, budget_ns, |_| {})
            .await
    }

    /// Like `advance_bounded`, with a call-scoped observer invoked once after
    /// local skips and before the selected activity. The callback must not
    /// access transferred buffers or reenter the reserved face. This permits
    /// host instrumentation without changing custody or introducing a clock
    /// into the transport-independent continuation. A cancelled future may
    /// have invoked the callback without completing the observation.
    pub async fn advance_bounded_observed<T: Transport>(
        &mut self,
        accel: &mut RingAccel<'_, '_, T>,
        budget_ns: u64,
        observe: impl FnOnce(BufferExecutionActivity),
    ) -> Result<ExecutionProgress, BufferExecutionError> {
        self.skip_local_imports(accel.client.is_idle());
        let activity = match self.phase {
            Phase::Import(index) => BufferExecutionActivity::Import {
                index,
                bytes: match &self.buffers[index] {
                    BufferState::Import { target, .. } => target.bytes,
                    _ => 0,
                },
            },
            Phase::Execute => BufferExecutionActivity::Execute,
            Phase::Export => BufferExecutionActivity::Export {
                index: self
                    .buffers
                    .iter()
                    .rposition(|b| matches!(b, BufferState::Imported(_))),
            },
            Phase::Finished => BufferExecutionActivity::Finished,
        };
        observe(activity);
        match self.phase {
            Phase::Finished => return Ok(self.progress()),
            Phase::Import(index) => {
                if let BufferState::Import { target, source } = &mut self.buffers[index] {
                    let reply = accel
                        .advance_retained_import(*target, source, budget_ns)
                        .await
                        .map_err(BufferExecutionError::Import)?;
                    match reply {
                        ImportReply::Imported { buffer } => {
                            self.buffers[index] = BufferState::Imported(buffer);
                            self.phase = Phase::Import(index + 1);
                        }
                        ImportReply::Returned { refusal, lease } => {
                            self.buffers[index] = BufferState::Local(lease);
                            self.outcome = ExecutionOutcome::Refused(refusal);
                            self.phase = Phase::Export;
                        }
                        ImportReply::Refused(refusal) => {
                            self.buffers[index] = BufferState::Consumed;
                            self.outcome = ExecutionOutcome::Refused(refusal);
                            self.phase = Phase::Export;
                        }
                    }
                }
            }
            Phase::Execute => {
                if self.execution.is_none() {
                    if self.cancelling {
                        self.phase = Phase::Export;
                        return Ok(self.progress());
                    }
                    let ids = core::array::from_fn(|index| self.buffers[index].device_id());
                    let launch = match self.make_launch.build(&ids) {
                        Ok(launch) => launch,
                        Err(refusal) => {
                            self.failure_activity.get_or_insert(activity);
                            self.outcome = ExecutionOutcome::Refused(refusal);
                            self.phase = Phase::Export;
                            return Ok(self.progress());
                        }
                    };
                    self.execution = Some(
                        accel
                            .execution_ledger_sequence(core::iter::once(launch))
                            .ok_or(BufferExecutionError::Execution(
                                ExecutionError::Infrastructure(CallError::Outstanding),
                            ))?,
                    );
                }
                // The ledger is recorded before its first await. This is the
                // same publication/retirement engine used by resident NPU work.
                if let Some(execution) = &mut self.execution {
                    match execution
                        .advance_bounded(accel, budget_ns)
                        .await
                        .map_err(BufferExecutionError::Execution)?
                    {
                        ExecutionProgress::Pending => {}
                        ExecutionProgress::Finished(outcome) => {
                            self.outcome = outcome;
                            self.phase = Phase::Export;
                        }
                    }
                }
            }
            Phase::Export => {
                // A wire-bounded reverse scan, not a model-sized rescan. Never
                // release/re-import Resident entries owned by the target cache.
                if let Some(index) = self
                    .buffers
                    .iter()
                    .rposition(|b| matches!(b, BufferState::Imported(_)))
                {
                    let buffer = self.buffers[index].device_id();
                    match accel
                        .advance_export_buffer(buffer, budget_ns)
                        .await
                        .map_err(BufferExecutionError::Export)?
                    {
                        ExportReply::Returned(lease) => {
                            self.buffers[index] = BufferState::Local(lease)
                        }
                        ExportReply::Refused(refusal) => {
                            self.failure_activity.get_or_insert(activity);
                            return Err(BufferExecutionError::Refused(refusal));
                        }
                    }
                } else {
                    self.phase = Phase::Finished;
                }
            }
        }
        if matches!(
            self.outcome,
            ExecutionOutcome::Refused(_) | ExecutionOutcome::Retired(AccelEventState::Failed(_))
        ) {
            self.failure_activity.get_or_insert(activity);
        }
        Ok(self.progress())
    }
}
