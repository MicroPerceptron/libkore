//! Resumable creation and acknowledged release of a bounded object batch.
use super::RingAccel;
use crate::service::{AccelRefusal, AccelRequest, AccelResponse};
use kore_rt::{svc::CallError, transport::Transport};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectBatchProgress {
    Pending,
    Ready,
    /// All known allocations have acknowledged release. A refusal records the
    /// allocation that stopped the batch; None means caller-requested release.
    Released(Option<AccelRefusal>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectBatchError {
    Infrastructure(CallError),
    /// Release was refused; the batch retains the same object for retry.
    Refused(AccelRefusal),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Allocate,
    Ready,
    Release,
    Closed,
}

/// Borrowed, fixed-capacity ownership ledger on one possessed accelerator face.
/// A failed creation automatically unwinds earlier objects in reverse
/// order. Busy release and transport uncertainty never discard an ID.
///
/// Retain this value when cancelling advance. Dropping it does not release
/// server objects; an abandoning owner must retain the face for teardown.
/// IDs are request data, not independent capabilities. Access through client()
/// is available only after every allocation succeeds; provider checks still
/// enforce live references when release is requested.
#[must_use = "retain the batch until release is acknowledged or the owning face is torn down"]
pub struct ObjectBatch<'a, 'w, 't, T: Transport, K: BatchKind> {
    kind: core::marker::PhantomData<K>,
    accel: &'a mut RingAccel<'w, 't, T>,
    requests: &'a [AccelRequest],
    ids: &'a mut [u64],
    live: usize,
    phase: Phase,
    closing: bool,
    refusal: Option<AccelRefusal>,
}

mod sealed {
    pub trait Sealed {}
}

/// Closed creation/release pairs supported by the common batch engine.
#[doc(hidden)]
pub trait BatchKind: sealed::Sealed {
    fn accepts(request: &AccelRequest) -> bool;
    fn created(response: AccelResponse) -> Option<u64>;
    fn release(id: u64) -> AccelRequest;
}
pub struct BufferObjects;
pub struct ProgramObjects;
impl sealed::Sealed for BufferObjects {}
impl sealed::Sealed for ProgramObjects {}
impl BatchKind for BufferObjects {
    fn accepts(r: &AccelRequest) -> bool {
        matches!(r, AccelRequest::CreateBuffer { .. })
    }
    fn created(r: AccelResponse) -> Option<u64> {
        if let AccelResponse::BufferMade { buffer } = r {
            Some(buffer)
        } else {
            None
        }
    }
    fn release(buffer: u64) -> AccelRequest {
        AccelRequest::FreeBuffer { buffer }
    }
}
impl BatchKind for ProgramObjects {
    fn accepts(r: &AccelRequest) -> bool {
        matches!(r, AccelRequest::LoadProgram { .. })
    }
    fn created(r: AccelResponse) -> Option<u64> {
        if let AccelResponse::ProgramMade { program } = r {
            Some(program)
        } else {
            None
        }
    }
    fn release(program: u64) -> AccelRequest {
        AccelRequest::UnloadProgram { program }
    }
}
pub type BufferBatch<'a, 'w, 't, T> = ObjectBatch<'a, 'w, 't, T, BufferObjects>;
pub type ProgramBatch<'a, 'w, 't, T> = ObjectBatch<'a, 'w, 't, T, ProgramObjects>;
pub use ObjectBatchError as BufferBatchError;
pub use ObjectBatchProgress as BufferBatchProgress;

impl<'w, 't, T: Transport> RingAccel<'w, 't, T> {
    /// Start a buffer batch on an idle face, in initially empty caller storage.
    /// Every request must be CreateBuffer and have a corresponding ID slot.
    /// Provider admission owns geometry, context, domain, usage and budgets.
    pub fn allocate_buffers<'a>(
        &'a mut self,
        requests: &'a [AccelRequest],
        ids: &'a mut [u64],
    ) -> Option<BufferBatch<'a, 'w, 't, T>> {
        self.object_batch(requests, ids)
    }

    /// Load a batch of already-staged programs on this idle possessed face.
    /// Every request must be LoadProgram and have a corresponding ID slot.
    /// Keep all referenced staging bytes unchanged until Ready or Released;
    /// cancellation of an observer does not end that obligation. The exclusive
    /// client borrow prevents another protocol operation from changing staging,
    /// but callers still own any local CPU aliases. Admission checks the actual
    /// staging range, artifact, context and resident budget at the provider.
    pub fn load_programs<'a>(
        &'a mut self,
        requests: &'a [AccelRequest],
        ids: &'a mut [u64],
    ) -> Option<ProgramBatch<'a, 'w, 't, T>> {
        self.object_batch(requests, ids)
    }

    /// Retire existing buffers, including successful imports, on this idle face.
    /// Supply each live ID exactly once, without zero placeholders. IDs remain
    /// in caller storage until their matching release is acknowledged; release
    /// runs in reverse order. Validation publishes nothing and changes no IDs.
    /// Provider checks still authenticate the face and reject live references.
    /// Retain the batch across cancelled observations or transport uncertainty.
    pub fn release_buffers<'a>(
        &'a mut self,
        ids: &'a mut [u64],
    ) -> Option<BufferBatch<'a, 'w, 't, T>> {
        self.release_batch(ids)
    }

    /// The same acknowledged retirement for programs loaded outside a batch.
    /// Live events referring to these programs must be retired first.
    pub fn unload_programs<'a>(
        &'a mut self,
        ids: &'a mut [u64],
    ) -> Option<ProgramBatch<'a, 'w, 't, T>> {
        self.release_batch(ids)
    }

    fn release_batch<'a, K: BatchKind>(
        &'a mut self,
        ids: &'a mut [u64],
    ) -> Option<ObjectBatch<'a, 'w, 't, T, K>> {
        if !self.client.is_idle()
            || ids
                .iter()
                .enumerate()
                .any(|(i, id)| *id == 0 || ids[..i].contains(id))
        {
            return None;
        }
        let live = ids.len();
        Some(ObjectBatch {
            kind: core::marker::PhantomData,
            accel: self,
            requests: &[],
            ids,
            live,
            phase: if live == 0 {
                Phase::Closed
            } else {
                Phase::Release
            },
            closing: true,
            refusal: None,
        })
    }

    fn object_batch<'a, K: BatchKind>(
        &'a mut self,
        requests: &'a [AccelRequest],
        ids: &'a mut [u64],
    ) -> Option<ObjectBatch<'a, 'w, 't, T, K>> {
        if !self.client.is_idle()
            || requests.len() != ids.len()
            || ids.iter().any(|id| *id != 0)
            || requests.iter().any(|r| !K::accepts(r))
        {
            return None;
        }
        Some(ObjectBatch {
            kind: core::marker::PhantomData,
            accel: self,
            requests,
            ids,
            live: 0,
            phase: if requests.is_empty() {
                Phase::Ready
            } else {
                Phase::Allocate
            },
            closing: false,
            refusal: None,
        })
    }
}
impl<T: Transport> BufferBatch<'_, '_, '_, T> {
    pub fn buffers(&self) -> Option<&[u64]> {
        self.objects()
    }
}
impl<T: Transport> ProgramBatch<'_, '_, '_, T> {
    pub fn programs(&self) -> Option<&[u64]> {
        self.objects()
    }
}
impl<'w, 't, T: Transport, K: BatchKind> ObjectBatch<'_, 'w, 't, T, K> {
    /// Object IDs become usable only when the entire batch is ready.
    fn objects(&self) -> Option<&[u64]> {
        (self.phase == Phase::Ready && !self.closing).then_some(&self.ids[..self.live])
    }
    /// Known allocations still awaiting acknowledged release, for diagnostics.
    /// An empty slice during an outstanding create is not proof of no allocation.
    pub fn retained(&self) -> &[u64] {
        &self.ids[..self.live]
    }
    pub fn client(&mut self) -> Option<&mut RingAccel<'w, 't, T>> {
        if self.phase != Phase::Ready || self.closing {
            return None;
        }
        Some(self.accel)
    }
    /// Stop allocating and release this batch. An outstanding create must first
    /// receive its exact reply, including when it succeeds after this request.
    /// Returns false if an operation borrowed through client() is outstanding;
    /// resume that operation before trying again. No close intent is recorded.
    pub fn request_release(&mut self) -> bool {
        if self.phase == Phase::Ready && !self.accel.client.is_idle() {
            return false;
        }
        self.closing = true;
        true
    }

    pub async fn advance(&mut self) -> Result<ObjectBatchProgress, ObjectBatchError> {
        self.advance_with_budget(None).await
    }
    pub async fn advance_bounded(
        &mut self,
        budget_ns: u64,
    ) -> Result<ObjectBatchProgress, ObjectBatchError> {
        self.advance_with_budget(Some(budget_ns)).await
    }
    async fn advance_with_budget(
        &mut self,
        budget: Option<u64>,
    ) -> Result<ObjectBatchProgress, ObjectBatchError> {
        if self.closing
            && self.accel.client.is_idle()
            && matches!(self.phase, Phase::Allocate | Phase::Ready)
        {
            self.phase = Phase::Release;
        }
        if self.phase == Phase::Ready {
            return Ok(ObjectBatchProgress::Ready);
        }
        if self.phase == Phase::Release && self.live == 0 {
            self.phase = Phase::Closed;
        }
        let request = match self.phase {
            Phase::Allocate => self.requests[self.live],
            Phase::Release => K::release(self.ids[self.live - 1]),
            Phase::Closed => return Ok(ObjectBatchProgress::Released(self.refusal)),
            Phase::Ready => unreachable!(),
        };
        let resumed = match budget {
            Some(ns) => self.accel.client.resume_exact_bounded(&request, ns).await,
            None => self.accel.client.resume_exact(&request).await,
        };
        let reply = match resumed {
            Err(CallError::NoOutstanding) => match budget {
                Some(ns) => self.accel.client.call_bounded(&request, ns).await,
                None => self.accel.client.call(&request).await,
            },
            result => result,
        }
        .map_err(ObjectBatchError::Infrastructure)?;
        if reply.grants != 0 {
            self.accel.client.poison_retained(reply.retained);
            return Err(ObjectBatchError::Infrastructure(CallError::Wire));
        }
        if let Err(error) = accept_response::<K>(
            &mut self.phase,
            &mut self.live,
            self.ids,
            self.closing,
            &mut self.refusal,
            reply.response,
        ) {
            if error == ObjectBatchError::Infrastructure(CallError::Wire) {
                self.accel.client.poison();
            }
            return Err(error);
        }
        Ok(match self.phase {
            Phase::Ready => ObjectBatchProgress::Ready,
            Phase::Closed => ObjectBatchProgress::Released(self.refusal),
            _ => ObjectBatchProgress::Pending,
        })
    }
}

fn accept_response<K: BatchKind>(
    phase: &mut Phase,
    live: &mut usize,
    ids: &mut [u64],
    closing: bool,
    refusal: &mut Option<AccelRefusal>,
    response: AccelResponse,
) -> Result<(), ObjectBatchError> {
    match (*phase, response) {
        (Phase::Allocate, response)
            if K::created(response).is_some_and(|id| id != 0 && !ids[..*live].contains(&id)) =>
        {
            ids[*live] = K::created(response).unwrap();
            *live += 1;
            if closing {
                *phase = Phase::Release;
            } else if *live == ids.len() {
                *phase = Phase::Ready;
            }
        }
        (Phase::Allocate, AccelResponse::Refused(error)) => {
            *refusal = Some(error);
            *phase = if *live == 0 {
                Phase::Closed
            } else {
                Phase::Release
            };
        }
        (Phase::Release, AccelResponse::Released { subject })
            if *live != 0 && subject == ids[*live - 1] =>
        {
            *live -= 1;
            ids[*live] = 0;
            if *live == 0 {
                *phase = Phase::Closed;
            }
        }
        (Phase::Release, AccelResponse::Refused(AccelRefusal::Busy)) => {}
        (Phase::Release, AccelResponse::Refused(error)) => {
            return Err(ObjectBatchError::Refused(error));
        }
        _ => return Err(ObjectBatchError::Infrastructure(CallError::Wire)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_allocation_refusal_unwinds_only_acknowledged_ids() {
        let mut ids = [0; 3];
        let mut live = 0;
        let mut phase = Phase::Allocate;
        let mut refusal = None;
        for buffer in [41, 42] {
            accept_response::<BufferObjects>(
                &mut phase,
                &mut live,
                &mut ids,
                false,
                &mut refusal,
                AccelResponse::BufferMade { buffer },
            )
            .unwrap();
        }
        accept_response::<BufferObjects>(
            &mut phase,
            &mut live,
            &mut ids,
            false,
            &mut refusal,
            AccelResponse::Refused(AccelRefusal::NotReady),
        )
        .unwrap();
        assert!(phase == Phase::Release);
        assert_eq!(refusal, Some(AccelRefusal::NotReady));
        for response in [
            AccelResponse::Refused(AccelRefusal::Busy),
            AccelResponse::Refused(AccelRefusal::DeviceLost),
            AccelResponse::Released { subject: 41 },
        ] {
            let result = accept_response::<BufferObjects>(
                &mut phase,
                &mut live,
                &mut ids,
                false,
                &mut refusal,
                response,
            );
            assert!(
                result.is_err() || matches!(response, AccelResponse::Refused(AccelRefusal::Busy))
            );
            assert_eq!(ids, [41, 42, 0]);
            assert_eq!(live, 2);
        }
        for subject in [42, 41] {
            accept_response::<BufferObjects>(
                &mut phase,
                &mut live,
                &mut ids,
                false,
                &mut refusal,
                AccelResponse::Released { subject },
            )
            .unwrap();
        }
        assert!(phase == Phase::Closed);
        assert_eq!(live, 0);
        assert_eq!(ids, [0; 3]);
    }

    #[test]
    fn stop_after_publication_records_late_allocation_for_release() {
        let mut ids = [0; 2];
        let mut live = 0;
        let mut phase = Phase::Allocate;
        let mut refusal = None;
        accept_response::<BufferObjects>(
            &mut phase,
            &mut live,
            &mut ids,
            true,
            &mut refusal,
            AccelResponse::BufferMade { buffer: 17 },
        )
        .unwrap();
        assert!(phase == Phase::Release);
        assert_eq!(ids, [17, 0]);
        accept_response::<BufferObjects>(
            &mut phase,
            &mut live,
            &mut ids,
            true,
            &mut refusal,
            AccelResponse::Released { subject: 17 },
        )
        .unwrap();
        assert!(phase == Phase::Closed);
        assert_eq!(refusal, None);
    }

    #[test]
    fn malformed_allocation_does_not_publish_ready_or_overwrite_custody() {
        for buffer in [0, 17] {
            let mut ids = [17, 0];
            let mut live = 1;
            let mut phase = Phase::Allocate;
            let mut refusal = None;
            assert_eq!(
                accept_response::<BufferObjects>(
                    &mut phase,
                    &mut live,
                    &mut ids,
                    false,
                    &mut refusal,
                    AccelResponse::BufferMade { buffer }
                ),
                Err(ObjectBatchError::Infrastructure(CallError::Wire))
            );
            assert_eq!(ids, [17, 0]);
            assert_eq!(live, 1);
            assert!(phase == Phase::Allocate);
        }
    }

    fn transport_failure_preserves_creation<K: BatchKind>(request: AccelRequest) {
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
        let requests = [request];
        let mut ids = [0];
        assert!(
            accel
                .object_batch::<K>(&[AccelRequest::Query], &mut ids)
                .is_none()
        );
        let mut occupied = [17];
        assert!(accel.object_batch::<K>(&requests, &mut occupied).is_none());
        assert_eq!(occupied, [17]);
        assert!(server.req.pop().is_none(), "invalid batch must not publish");
        {
            let mut batch = accel.object_batch::<K>(&requests, &mut ids).unwrap();
            assert!(batch.request_release());
            let mut future = core::pin::pin!(batch.advance());
            assert_eq!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Ok(ObjectBatchProgress::Released(None)))
            );
            assert!(server.req.pop().is_none());
        }
        let mut batch = accel.object_batch::<K>(&requests, &mut ids).unwrap();
        for attempt in 0..2 {
            if attempt == 1 {
                assert!(batch.request_release());
            }
            let mut future = core::pin::pin!(batch.advance());
            assert_eq!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Err(ObjectBatchError::Infrastructure(CallError::Transport)))
            );
            if attempt == 0 {
                assert_eq!(
                    crate::wire::decode_req(&server.req.pop().unwrap()),
                    Some(request)
                );
            }
            assert!(server.req.pop().is_none());
        }
        assert_eq!(batch.retained(), &[]);
        assert!(batch.objects().is_none());
        assert!(batch.client().is_none());
        drop(batch);
        assert!(!accel.client.is_idle());
    }
    #[test]
    fn real_transport_failure_preserves_both_creation_kinds() {
        transport_failure_preserves_creation::<BufferObjects>(AccelRequest::CreateBuffer {
            ctx: 1,
            bytes: 4096,
            align: 4096,
            domain: 3,
            usage: 3,
        });
        transport_failure_preserves_creation::<ProgramObjects>(AccelRequest::LoadProgram {
            ctx: 1,
            stage: 4096,
            len: 128,
            resident_bytes: 8192,
            format: 1,
            target: [0; 12],
        });
    }

    fn existing_objects_retain_custody_on_uncertain_release<K: BatchKind>() {
        use crate::wire::AccelWire;
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
        for mut invalid in [[0, 17], [17, 0], [17, 17]] {
            let original = invalid;
            assert!(accel.release_batch::<K>(&mut invalid).is_none());
            assert_eq!(invalid, original);
        }
        {
            let mut empty = [];
            let mut batch = accel.release_batch::<K>(&mut empty).unwrap();
            for _ in 0..2 {
                assert_eq!(
                    kore_rt::try_block_on(batch.advance(), 1),
                    Some(Ok(ObjectBatchProgress::Released(None)))
                );
            }
        }
        assert!(
            server.req.pop().is_none(),
            "constructors and empty release publish nothing"
        );
        let mut ids = [41, 42];
        let mut batch = accel.release_batch::<K>(&mut ids).unwrap();
        assert!(batch.objects().is_none());
        assert!(
            batch.client().is_none(),
            "retirement cannot lend the face for new work"
        );
        for attempt in 0..2 {
            assert_eq!(
                kore_rt::try_block_on(batch.advance(), 1),
                Some(Err(ObjectBatchError::Infrastructure(CallError::Transport)))
            );
            if attempt == 0 {
                assert_eq!(
                    crate::wire::decode_req(&server.req.pop().unwrap()),
                    Some(K::release(42))
                );
            }
            assert!(
                server.req.pop().is_none(),
                "resume must not publish a second release"
            );
            assert_eq!(batch.retained(), &[41, 42]);
        }
        drop(batch);
        assert_eq!(ids, [41, 42]);
        assert!(
            accel.release_batch::<K>(&mut ids).is_none(),
            "outstanding release must be resumed"
        );
    }

    #[test]
    fn existing_buffers_and_programs_share_exact_retirement() {
        existing_objects_retain_custody_on_uncertain_release::<BufferObjects>();
        existing_objects_retain_custody_on_uncertain_release::<ProgramObjects>();
    }

    #[test]
    fn program_refusal_unloads_prior_programs_and_rejects_buffer_replies() {
        let mut ids = [0; 2];
        let mut live = 0;
        let mut phase = Phase::Allocate;
        let mut refusal = None;
        assert_eq!(
            accept_response::<ProgramObjects>(
                &mut phase,
                &mut live,
                &mut ids,
                false,
                &mut refusal,
                AccelResponse::BufferMade { buffer: 41 }
            ),
            Err(ObjectBatchError::Infrastructure(CallError::Wire))
        );
        assert_eq!(ids, [0; 2]);
        accept_response::<ProgramObjects>(
            &mut phase,
            &mut live,
            &mut ids,
            false,
            &mut refusal,
            AccelResponse::ProgramMade { program: 41 },
        )
        .unwrap();
        accept_response::<ProgramObjects>(
            &mut phase,
            &mut live,
            &mut ids,
            false,
            &mut refusal,
            AccelResponse::Refused(AccelRefusal::NotReady),
        )
        .unwrap();
        assert!(phase == Phase::Release);
        assert_eq!(
            ProgramObjects::release(ids[live - 1]),
            AccelRequest::UnloadProgram { program: 41 }
        );
        accept_response::<ProgramObjects>(
            &mut phase,
            &mut live,
            &mut ids,
            false,
            &mut refusal,
            AccelResponse::Refused(AccelRefusal::Busy),
        )
        .unwrap();
        assert_eq!(ids, [41, 0]);
        assert!(
            accept_response::<ProgramObjects>(
                &mut phase,
                &mut live,
                &mut ids,
                false,
                &mut refusal,
                AccelResponse::Released { subject: 42 }
            )
            .is_err()
        );
        assert_eq!(ids, [41, 0]);
        accept_response::<ProgramObjects>(
            &mut phase,
            &mut live,
            &mut ids,
            false,
            &mut refusal,
            AccelResponse::Released { subject: 41 },
        )
        .unwrap();
        assert!(phase == Phase::Closed);
        assert_eq!(refusal, Some(AccelRefusal::NotReady));
        assert_eq!(ids, [0; 2]);
        phase = Phase::Allocate;
        for (index, program) in [51, 52].into_iter().enumerate() {
            accept_response::<ProgramObjects>(
                &mut phase,
                &mut live,
                &mut ids,
                false,
                &mut refusal,
                AccelResponse::ProgramMade { program },
            )
            .unwrap();
            assert_eq!(
                phase == Phase::Ready,
                index == 1,
                "partial load is not ready"
            );
        }
        assert_eq!(ids, [51, 52]);
        assert!(!ProgramObjects::accepts(&AccelRequest::CreateBuffer {
            ctx: 1,
            bytes: 4096,
            align: 4096,
            domain: 3,
            usage: 3
        }));
        assert!(!BufferObjects::accepts(&AccelRequest::LoadProgram {
            ctx: 1,
            stage: 0,
            len: 128,
            resident_bytes: 4096,
            format: 1,
            target: [0; 12]
        }));
    }
}
