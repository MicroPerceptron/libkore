//! Bounded independent admissions over one possessed accelerator pair.

use super::*;

/// Caller-scheduled independent executions, using the same protocol lifecycle
/// as [`Execution`]. A set exclusively borrows one face, but holds multiple
/// live events: advance each admission once before polling either to permit
/// backend overlap. It does not promise simultaneous hardware execution.
///
/// The caller supplies independent bindings (shared read-only inputs are fine)
/// and chooses observation order and cancellation per entry. A refusal or
/// cancellation affects only that entry. Dropping the set does not cancel work;
/// retain its objects and face until every entry is Finished or owner teardown.
#[must_use = "retain every admission until its event retirement is acknowledged"]
pub struct ExecutionSet<'a, 'w, 't, T: Transport, const N: usize> {
    accel: &'a mut RingAccel<'w, 't, T>,
    phases: [Phase; N],
    intents: [Intent; N],
    observed: [Option<AccelEventState>; N],
    // Set before awaiting: dropping the observation future cannot forget which
    // entry owns the uncorrelated reply stream. Cleared only on an idle pair.
    active: Option<usize>,
}

impl<'w, 't, T: Transport> RingAccel<'w, 't, T> {
    /// Construct without publication. Reject an empty set or an occupied face.
    /// Array length is the caller's explicit bound; no allocation is performed.
    pub fn execution_set<const N: usize>(
        &mut self,
        launches: [ArtifactExecution; N],
    ) -> Option<ExecutionSet<'_, 'w, 't, T, N>> {
        if N == 0 || !self.client.is_idle() {
            return None;
        }
        Some(ExecutionSet {
            accel: self,
            phases: launches.map(|launch| Phase::Submit(launch.request())),
            intents: [Intent::Run; N],
            observed: [None; N],
            active: None,
        })
    }
}

impl<T: Transport, const N: usize> ExecutionSet<'_, '_, '_, T, N> {
    /// Entry whose exact protocol operation must be resumed before another
    /// entry may publish. None does not mean the admitted events are terminal.
    pub fn outstanding_entry(&self) -> Option<usize> {
        self.active.filter(|_| !self.accel.client.is_idle())
    }

    /// Event ID while owned, including terminal states awaiting destruction.
    /// None before a reply is not proof that admission did not occur.
    pub fn event(&self, index: usize) -> Option<u64> {
        match *self.phases.get(index)? {
            Phase::Poll(event) | Phase::Cancel(event) | Phase::Retire { event, .. } => Some(event),
            _ => None,
        }
    }

    /// Most recent acknowledged PollEvent observation, never inferred from
    /// admission, cancellation or retirement. Pending is historical evidence,
    /// not a claim that the event remains pending now. Poll both only after
    /// both admission replies to establish overlapping pending lifetimes.
    pub fn observed(&self, index: usize) -> Option<AccelEventState> {
        self.observed.get(index).copied().flatten()
    }

    /// None for an invalid index; Finished includes acknowledged retirement.
    pub fn progress(&self, index: usize) -> Option<ExecutionProgress> {
        Some(match *self.phases.get(index)? {
            Phase::Finished(outcome) => ExecutionProgress::Finished(outcome),
            _ => ExecutionProgress::Pending,
        })
    }

    /// Record stop intent for this entry only, without publishing. An in-flight
    /// operation (including one for a peer) must first resume exactly. Returns
    /// false for an invalid index; already-finished entries remain unchanged.
    pub fn request_cancel(&mut self, index: usize) -> bool {
        let Some(phase) = self.phases.get(index) else {
            return false;
        };
        if !matches!(phase, Phase::Finished(_)) {
            self.intents[index] = Intent::Cancel;
        }
        true
    }

    pub async fn advance(&mut self, index: usize) -> Result<ExecutionProgress, ExecutionError> {
        self.advance_with_budget(index, None).await
    }

    /// One bounded protocol operation; timeout retains the entry and exact
    /// request. Advancing a different entry while occupied returns Outstanding
    /// without consuming the reply or changing either lifecycle.
    pub async fn advance_bounded(
        &mut self,
        index: usize,
        budget_ns: u64,
    ) -> Result<ExecutionProgress, ExecutionError> {
        self.advance_with_budget(index, Some(budget_ns)).await
    }

    async fn advance_with_budget(
        &mut self,
        index: usize,
        budget_ns: Option<u64>,
    ) -> Result<ExecutionProgress, ExecutionError> {
        if index >= N {
            return Err(ExecutionError::Refused(AccelRefusal::InvalidArgument));
        }
        if let Some(active) = self.outstanding_entry()
            && active != index
        {
            return Err(ExecutionError::Infrastructure(CallError::Outstanding));
        }
        self.active = Some(index);
        let result = advance_phase(
            self.accel,
            &mut self.phases[index],
            &mut self.intents[index],
            budget_ns,
        )
        .await;
        if self.accel.client.is_idle() {
            self.active = None;
        }
        if let Some(state) = result? {
            self.observed[index] = Some(state);
        }
        Ok(self.progress(index).expect("validated index"))
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

    fn launch(program: u64) -> ArtifactExecution {
        ArtifactExecution {
            queue: 1,
            program,
            bindings: AccelBindings::new(&[crate::service::AccelBinding {
                buffer: 3,
                offset: 0,
                len: 4,
                slot: 0,
                access: virtio_accel_core::AccessMode::Read,
            }])
            .unwrap(),
            timeout_ns: 100,
        }
    }
    fn ready<F: Future>(future: F) -> F::Output {
        let mut future = core::pin::pin!(future);
        match future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("host refusal/local transition must be ready"),
        }
    }

    #[test]
    fn interrupted_entry_cannot_be_replaced_by_peer_admission_or_cancellation() {
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
        assert!(accel.execution_set([]).is_none());
        let mut set = accel.execution_set([launch(7), launch(8)]).unwrap();
        assert_eq!(set.progress(2), None);
        assert!(!set.request_cancel(2));
        assert_eq!(
            ready(set.advance(2)),
            Err(ExecutionError::Refused(AccelRefusal::InvalidArgument))
        );
        assert!(server.req.pop().is_none());
        assert_eq!(
            ready(set.advance(0)),
            Err(ExecutionError::Infrastructure(CallError::Transport))
        );
        let published = server.req.pop().expect("publication precedes failed kick");
        #[allow(deprecated)]
        {
            assert!(matches!(
                crate::wire::decode_req(&published),
                Some(AccelRequest::Submit { program: 7, .. })
            ));
        }
        assert_eq!(set.outstanding_entry(), Some(0));
        assert_eq!(
            set.observed(0),
            None,
            "admission uncertainty is not a pending observation"
        );
        set.request_cancel(1);
        set.request_cancel(0);
        for _ in 0..2 {
            assert_eq!(
                ready(set.advance(1)),
                Err(ExecutionError::Infrastructure(CallError::Outstanding))
            );
            assert_eq!(
                ready(set.advance(0)),
                Err(ExecutionError::Infrastructure(CallError::Transport))
            );
            assert!(server.req.pop().is_none(), "no replacement publication");
        }
        assert_eq!(set.event(0), None, "lost reply cannot invent an event");
        assert_eq!(set.progress(1), Some(ExecutionProgress::Pending));
        drop(set);
        assert!(
            accel.execution_set([launch(9)]).is_none(),
            "abandonment cannot release correlation"
        );
    }

    #[test]
    fn cancellation_of_one_unpublished_entry_leaves_its_peer_unstarted_and_face_reusable() {
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
        let mut set = accel.execution_set([launch(7), launch(8)]).unwrap();
        set.request_cancel(1);
        for _ in 0..2 {
            assert_eq!(
                ready(set.advance(1)),
                Ok(ExecutionProgress::Finished(ExecutionOutcome::NotSubmitted))
            );
            assert_eq!(set.progress(0), Some(ExecutionProgress::Pending));
            assert_eq!(set.outstanding_entry(), None);
        }
        set.request_cancel(0);
        assert_eq!(
            ready(set.advance(0)),
            Ok(ExecutionProgress::Finished(ExecutionOutcome::NotSubmitted))
        );
        assert_eq!(set.observed(0), None);
        assert_eq!(set.observed(1), None);
        assert!(server.req.pop().is_none());
        drop(set);
        assert!(accel.execution_set([launch(9)]).is_some());
    }
}
