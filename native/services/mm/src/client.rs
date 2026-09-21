//! `client` — the mm client face for **peers** (`res/USERLAND.md` §4).
//!
//! [`RingMm`] is the born-async ring + channel proxy over the pair's
//! shared window; [`alloc_props`] builds the request artifact an
//! [`MmRequest::Alloc`] carries. The synchronous global allocator uses its
//! founder-provisioned `HeapSupplySource` so bootstrap cannot depend on mm.
//! Named service heaps can instead use [`crate::heap::MmSource`]: allocation
//! misses return to the async caller, which awaits a real mm refill and
//! retries. No allocator callback polls the executor or simulates a grant.
//!
//! **The grant channel is retained** (slice D): both grant flavors
//! outlive the call, so the runtime rotates an owned landing per
//! delivery and the client takes possession through a typed claim —
//! [`supply`](MmClient::supply) converts its delivered root to a
//! [`StashedLease`] (the image client's shape), while a raw
//! [`request`](MmClient::request) hands back the affine
//! [`ReturnedCapability`] for the caller to shape (the future heap-grow
//! rung's view-consumption story is deliberately its own to design,
//! against the real loop).

use kore_abi::mem::{AllocProps, DmaClass, MemAttrs, NumaNode};

use kore_rt::cap::{ReturnedCapability, StashedLease};
use kore_rt::heap::EXTENT_ALIGN;
use kore_rt::svc::{CallError, CallReply};

use crate::service::{MmRequest, MmResponse, SUPPLY_RIGHTS};

/// The **request artifact**: the [`AllocProps`] a heap-extent request
/// carries ([`MmRequest::Alloc`]) — `min_bytes` page-rounded, CPU
/// write-back R/W, NUMA-agnostic, non-DMA.  Exposed so a client and a
/// test can pin the shape the wire depends on.
///
/// [`MmRequest::Alloc`]: crate::service::MmRequest::Alloc
pub fn alloc_props(min_bytes: usize) -> AllocProps {
    AllocProps {
        size: min_bytes.next_multiple_of(EXTENT_ALIGN) as u64,
        align: EXTENT_ALIGN as u64,
        numa: NumaNode::ANY,
        dma: DmaClass::None,
        attrs: MemAttrs::WRITE_BACK | MemAttrs::READ | MemAttrs::WRITE,
        contiguous: false,
    }
}

/// Whether a failed supply observation left a request owned by this caller.
/// `Pending` includes interrupted reply collection and a poisoned published
/// request; neither permits a replacement supply to be issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupplyError {
    NotStarted(CallError),
    Pending(CallError),
}

/// The **client** side of the mm protocol for *peers*
/// (`res/USERLAND.md` §4: "MmClient exists for peers; the global
/// allocator does NOT use it").  A `Granted` reply's authority arrives
/// beside the response as a delivered, runtime-owned capability —
/// possession-shaped, never reply geometry.
#[allow(async_fn_in_trait)] // clients are generic over the impl (§4)
pub trait MmClient {
    /// Issue one request; a granted capability rides beside the reply as
    /// its affine landing owner (dropping it revokes the delivery — no
    /// authority leaks on an ignored grant).
    async fn request(&mut self, req: MmRequest) -> (MmResponse, Option<ReturnedCapability>);

    /// Mint one child supply ([`MmRequest::Supply`], page-rounded,
    /// default alignment): the delivered **root lease**, possessed as a
    /// [`StashedLease`] — `slot()` is what stamp views carve off and the
    /// spawn convey moves.  A reply whose grant is not a lease of
    /// [`SUPPLY_RIGHTS`] is refused fail-closed (`None` beside the
    /// data-plane response).
    async fn supply(&mut self, bytes: u64) -> (MmResponse, Option<StashedLease>);

    /// Start one fallible supply request. A deadline is an absolute executor
    /// time; None retains the ordinary unbounded observation contract.
    /// Cancellation must retain the exact request in this client's pair.
    async fn supply_observed(
        &mut self,
        bytes: u64,
        deadline_ns: Option<u64>,
    ) -> Result<(MmResponse, Option<StashedLease>), SupplyError>;

    /// Continue this caller's original supply request on the same held pair.
    /// Compare its complete request before progress; never republish a pending
    /// supply or consume an unrelated operation's reply.
    async fn resume_supply_bounded(
        &mut self,
        bytes: u64,
        budget_ns: u64,
    ) -> Result<(MmResponse, Option<StashedLease>), CallError>;
    /// [`supply`](Self::supply) in a named frame class: `align` is the
    /// lease alignment in bytes (`0`/4 KiB base class, 2 MiB or 1 GiB huge
    /// classes — `res/ACCEL_MEMORY_SCALE.md` §1.2, the alignment *is* the
    /// class).  A huge-class supply is rounded to a whole number of its
    /// units and metered against mm's huge quota, so an accelerator working
    /// set neither competes with nor drains the launcher's base allowance.
    async fn supply_in_class(
        &mut self,
        bytes: u64,
        align: u64,
    ) -> (MmResponse, Option<StashedLease>);

    /// Read the lessor's bounded provider-owned snapshot. A malformed or
    /// refused response returns `None`; the operation never carries a grant.
    async fn stats(&mut self) -> Option<kore_abi::monitor::MemorySnapshot>;
}

/// The **ring + channel proxy** (`res/USERLAND.md` §5.6): the peer mm
/// client over the pair's shared window — request pushed, doorbell
/// kicked, reply awaited; a `Granted` reply announces one transfer,
/// which lands in a runtime-rotated owned slot (the retained-channel
/// contract) and reaches the caller as its affine owner.
///
/// Legacy request/supply conveniences panic on infrastructure failure because
/// `MmRefusal` has no infrastructure variant. Owners that retain partial work
/// use `supply_observed` and `resume_supply_bounded`: transport errors remain
/// distinct from lessor refusals and preserve the original request.
pub struct RingMm<'w, 't, T: kore_rt::transport::Transport>(
    pub kore_rt::svc::SvcClient<'w, 't, T, crate::service::MemoryManager>,
);

impl<T: kore_rt::transport::Transport> RingMm<'_, '_, T> {
    fn finish_supply(
        &mut self,
        mut reply: CallReply<MmResponse>,
    ) -> Result<(MmResponse, Option<StashedLease>), CallError> {
        if reply.delivered.iter().any(Option::is_some)
            || reply.retained.iter().skip(1).any(Option::is_some)
        {
            self.0.poison_retained(reply.retained);
            return Err(CallError::Wire);
        }
        match reply.response {
            MmResponse::Granted if reply.grants == 1 => {
                let Some(owner) = reply.retained[0].take() else {
                    self.0.poison_retained(reply.retained);
                    return Err(CallError::Wire);
                };
                match owner.try_into_stashed_lease(SUPPLY_RIGHTS) {
                    Ok(lease) => Ok((reply.response, Some(lease))),
                    Err(owner) => {
                        reply.retained[0] = Some(owner);
                        self.0.poison_retained(reply.retained);
                        Err(CallError::Wire)
                    }
                }
            }
            MmResponse::Refused(_) if reply.grants == 0 && reply.retained[0].is_none() => {
                Ok((reply.response, None))
            }
            _ => {
                self.0.poison_retained(reply.retained);
                Err(CallError::Wire)
            }
        }
    }

    fn finish_stats(
        &mut self,
        reply: CallReply<MmResponse>,
    ) -> Result<Option<kore_abi::monitor::MemorySnapshot>, CallError> {
        if reply.grants != 0
            || reply.delivered.iter().any(Option::is_some)
            || reply.retained.iter().any(Option::is_some)
        {
            self.0.poison_retained(reply.retained);
            return Err(CallError::Wire);
        }
        match reply.response {
            MmResponse::Stats(stats) if stats.validate() => Ok(stats.available().then_some(stats)),
            MmResponse::Refused(_) => Ok(None),
            _ => {
                self.0.poison_retained(reply.retained);
                Err(CallError::Wire)
            }
        }
    }

    /// Observe a bounded, grant-free provider snapshot. Cancellation and
    /// transport failure leave any published Stats request in this same pair.
    pub async fn stats_observed(
        &mut self,
        deadline_ns: u64,
    ) -> Result<Option<kore_abi::monitor::MemorySnapshot>, CallError> {
        let reply = self.0.call_until(&MmRequest::Stats, deadline_ns).await?;
        self.finish_stats(reply)
    }

    /// Resume only the original Stats request; a supply or other operation
    /// cannot be consumed as telemetry, even when its reply is already queued.
    pub async fn resume_stats_bounded(
        &mut self,
        budget_ns: u64,
    ) -> Result<Option<kore_abi::monitor::MemorySnapshot>, CallError> {
        let reply = self
            .0
            .resume_exact_bounded(&MmRequest::Stats, budget_ns)
            .await?;
        self.finish_stats(reply)
    }

    async fn call(&mut self, req: &MmRequest) -> (MmResponse, Option<ReturnedCapability>) {
        let mut reply = self
            .0
            .call(req)
            .await
            .expect("mm request path failed (torn rings/transport — a broken pair)");
        let delivered = (reply.grants > 0)
            .then(|| reply.retained.get_mut(0).and_then(Option::take))
            .flatten();
        (reply.response, delivered)
    }
}

/// **Administration of per-launch faces** — exec's pair only, over its
/// retained return leg (`return_tx` sends, `return_land` is the landing the
/// request announces). The shapes are the accelerator and power faces'.
impl<T: kore_rt::transport::Transport> RingMm<'_, '_, T> {
    /// Derive and convey one face component atomically while retaining the
    /// source lineage. `AfterTransfer` means custody moved: resume this
    /// operation rather than deriving and conveying a second component.
    pub fn is_idle(&self) -> bool {
        self.0.is_idle()
    }

    /// Observe one already-published admission component without sending again.
    pub async fn resume_admit_component_bounded(
        &mut self,
        part: kore_rt::svc::PeerBundlePart,
        return_land: kore_abi::session::SlotRef,
        budget_ns: u64,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        let request = MmRequest::AdmitPeer {
            part,
            announced: return_land,
        };
        let reply = self.0.resume_exact_bounded(&request, budget_ns).await?;
        if reply.grants != 0
            || reply.delivered.iter().any(Option::is_some)
            || reply.retained.iter().any(Option::is_some)
        {
            self.0.poison_retained(reply.retained);
            return Err(kore_rt::svc::CallError::Wire);
        }
        Ok(reply.response)
    }

    async fn cleanup_bounded(
        &mut self,
        request: MmRequest,
        budget_ns: u64,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        let pair = &mut self.0;
        let reply = if pair.is_idle() {
            pair.call_bounded(&request, budget_ns).await?
        } else {
            pair.resume_exact_bounded(&request, budget_ns).await?
        };
        if reply.grants != 0
            || reply.delivered.iter().any(Option::is_some)
            || reply.retained.iter().any(Option::is_some)
        {
            pair.poison_retained(reply.retained);
            return Err(kore_rt::svc::CallError::Wire);
        }
        Ok(reply.response)
    }

    pub async fn take_admission_bounded(
        &mut self,
        budget_ns: u64,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        self.cleanup_bounded(MmRequest::TakeAdmission, budget_ns)
            .await
    }
    pub async fn retire_peer_bounded(
        &mut self,
        peer: kore_rt::svc::PeerId,
        budget_ns: u64,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        self.cleanup_bounded(MmRequest::RetirePeer { peer }, budget_ns)
            .await
    }
    pub async fn confirm_retired_bounded(
        &mut self,
        peer: kore_rt::svc::PeerId,
        budget_ns: u64,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        self.cleanup_bounded(MmRequest::ConfirmRetired { peer }, budget_ns)
            .await
    }

    pub async fn admit_component_staged<F>(
        &mut self,
        return_tx: kore_abi::session::SlotRef,
        return_land: kore_abi::session::SlotRef,
        part: kore_rt::svc::PeerBundlePart,
        rights: kore_abi::cap::Rights,
        stage: F,
    ) -> Result<MmResponse, kore_rt::svc::StagedReturnError>
    where
        F: FnOnce(
            kore_rt::svc::ReturnDeriver<'_>,
        ) -> Result<kore_rt::svc::StagedReturn, kore_rt::svc::ReturnDeriveError>,
    {
        let request = MmRequest::AdmitPeer {
            part,
            announced: return_land,
        };
        let reply = self
            .0
            .call_with_staged_return(&request, return_tx, rights, stage)
            .await?;
        Ok(reply.response)
    }

    /// Consume the identity of the most recently assembled face.
    pub async fn take_peer_admission(&mut self) -> Result<MmResponse, kore_rt::svc::CallError> {
        self.0
            .call(&MmRequest::TakeAdmission)
            .await
            .map(|reply| reply.response)
    }

    /// Request retirement of one exact face generation.
    pub async fn retire_peer(
        &mut self,
        peer: kore_rt::svc::PeerId,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        self.0
            .call(&MmRequest::RetirePeer { peer })
            .await
            .map(|reply| reply.response)
    }

    /// Fence retirement: `Accepted` only once the face is closed and every
    /// supply it minted has returned, so the caller reclaims the face's
    /// local backing no earlier. `Refused(Pending)` means ask again later.
    pub async fn confirm_retired(
        &mut self,
        peer: kore_rt::svc::PeerId,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        self.0
            .call(&MmRequest::ConfirmRetired { peer })
            .await
            .map(|reply| reply.response)
    }
}

/// Where an engine-profile process finds its per-launch mm face.
///
/// Window applications occupy index 0 of the `SVC_RING`/`SVC_SEND`/
/// `SVC_GRANT` namespaces, the NPU and power faces index 1 and the GPU face
/// index 2, so the mm face takes index 3 and an engine program may hold
/// every one of them (exec pins the disjointness where it sees all the
/// faces). A face mints supplies and reads the books only, so no return leg
/// reaches the process.
pub mod process_face {
    /// mm request doorbell (`role::SVC_SEND`).
    pub const REQUEST_KICK: u16 = 3;
    /// mm request/reply ring backing (`role::SVC_RING`).
    pub const CONTROL_RING: u16 = 3;
    /// Receiver for mm's reply grants (`role::SVC_GRANT`).
    pub const GRANT_RECV: u16 = 3;
}

/// A process's own mm face plus the affine ring lease that keeps its mapped
/// control window alive. Its supplies are metered on the face's own lane.
#[cfg(target_os = "none")]
pub struct ProcessMm {
    pub client: RingMm<'static, 'static, kore_rt::transport::TrapTransport>,
    _ring: kore_rt::cap::LeaseToken,
}

/// The class a heap reservation is minted in: 2 MiB, so every granule the
/// heap window maps off it is a huge leaf and lands end to end, and so it is
/// metered against the face's huge-class quota (the boot policy's), not the
/// base class's.
#[cfg(target_os = "none")]
const HEAP_RESERVATION_CLASS: u64 = 0x20_0000;

#[cfg(target_os = "none")]
impl ProcessMm {
    /// **Reserve heap before a large synchronous job.** Ask mm for `bytes`
    /// more (rounded up to whole 2 MiB units) on this process's own face and
    /// adopt the grant into the heap ([`kore_rt::heap::adopt_supply`]), so
    /// the job's allocations — which cannot wait on a service — find room.
    ///
    /// The response is mm's own on refusal (`Quota` names the face's
    /// budget); `Refused(Denied)` also covers a grant the heap could not
    /// adopt (the lease is then revoked, never leaked).
    pub async fn reserve_heap(&mut self, bytes: u64) -> Result<(), MmResponse> {
        let (response, lease) = self
            .client
            .supply_in_class(bytes, HEAP_RESERVATION_CLASS)
            .await;
        let lease = lease.ok_or(response)?;
        kore_rt::heap::adopt_supply(lease)
            .map_err(|_| MmResponse::Refused(crate::service::MmRefusal::Denied))
    }
}

/// Attach the per-launch mm face an engine-profile launch delivered.
/// `None` for a process launched without one.
#[cfg(target_os = "none")]
pub fn attach_process_face(rt: &kore_rt::entry::Rt) -> Option<ProcessMm> {
    use kore_abi::task::role;
    use kore_rt::cap::verbs;
    use kore_rt::svc::{ClientEndpoints, GrantPin, SvcClient, client_rings};

    let boot = rt.bootstrap();
    let ring = verbs::take_lease(boot.handle(role::SVC_RING, process_face::CONTROL_RING)?)?;
    let window = verbs::map_held_rw(ring.slot())?;
    // SAFETY: this process is the sole client-side attachment over the
    // task-lifetime ring lease delivered by exec.
    let window: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
    let rings =
        client_rings::<crate::MemoryManager>(window, kore_rt::svc::REQ_CAP, kore_rt::svc::REP_CAP)
            .ok()?;
    let kick = boot.handle(role::SVC_SEND, process_face::REQUEST_KICK)?;
    let grant = boot.handle(role::SVC_GRANT, process_face::GRANT_RECV)?;
    let mut grants = [const { None }; kore_rt::service::MAX_GRANTS];
    grants[0] = Some(GrantPin::retained(grant));
    Some(ProcessMm {
        client: RingMm(
            SvcClient::new(
                kore_rt::entry::transport(),
                rings,
                ClientEndpoints { kick, grants },
            )
            .with_reply_wait(rt.doorbell()?),
        ),
        _ring: ring,
    })
}

impl<T: kore_rt::transport::Transport> MmClient for RingMm<'_, '_, T> {
    async fn request(&mut self, req: MmRequest) -> (MmResponse, Option<ReturnedCapability>) {
        self.call(&req).await
    }

    async fn supply(&mut self, bytes: u64) -> (MmResponse, Option<StashedLease>) {
        self.supply_in_class(bytes, 0).await
    }

    async fn supply_in_class(
        &mut self,
        bytes: u64,
        align: u64,
    ) -> (MmResponse, Option<StashedLease>) {
        let (response, delivered) = self.call(&MmRequest::Supply { size: bytes, align }).await;
        let lease = delivered.and_then(|owner| {
            // Kind + rights checked at possession; a mismatched delivery
            // drops through the owner (revoked, slot reopened) — the
            // fail-closed posture, never an optimistic bind.
            owner.try_into_stashed_lease(SUPPLY_RIGHTS).ok()
        });
        (response, lease)
    }

    async fn supply_observed(
        &mut self,
        bytes: u64,
        deadline_ns: Option<u64>,
    ) -> Result<(MmResponse, Option<StashedLease>), SupplyError> {
        if !self.0.is_idle() {
            return Err(SupplyError::NotStarted(CallError::Outstanding));
        }
        let request = MmRequest::Supply {
            size: bytes,
            align: 0,
        };
        let result = match deadline_ns {
            Some(deadline) => self.0.call_until(&request, deadline).await,
            None => self.0.call(&request).await,
        };
        let reply = result.map_err(|error| {
            if self.0.is_idle() {
                SupplyError::NotStarted(error)
            } else {
                SupplyError::Pending(error)
            }
        })?;
        self.finish_supply(reply).map_err(SupplyError::Pending)
    }

    async fn resume_supply_bounded(
        &mut self,
        bytes: u64,
        budget_ns: u64,
    ) -> Result<(MmResponse, Option<StashedLease>), CallError> {
        let reply = self
            .0
            .resume_exact_bounded(
                &MmRequest::Supply {
                    size: bytes,
                    align: 0,
                },
                budget_ns,
            )
            .await?;
        self.finish_supply(reply)
    }

    async fn stats(&mut self) -> Option<kore_abi::monitor::MemorySnapshot> {
        let (response, delivered) = self.call(&MmRequest::Stats).await;
        if delivered.is_some() {
            return None;
        }
        match response {
            MmResponse::Stats(stats) if stats.validate() && stats.available() => Some(stats),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{MemoryManager, MmRefusal};
    use core::cell::Cell;
    use kore_abi::session::{Committed, SlotRef, Status};
    use kore_rt::svc::{ClientEndpoints, REP_CAP, REQ_CAP, SvcClient, SvcProtocol};

    struct ReplyTransport(Cell<bool>, Cell<usize>);
    impl kore_rt::transport::Transport for ReplyTransport {
        type Error = ();
        async fn commit(&self, s: &kore_rt::session::SessionBuilder) -> Result<Committed, ()> {
            self.commit_now(s)
        }
        fn commit_now(&self, _: &kore_rt::session::SessionBuilder) -> Result<Committed, ()> {
            self.1.set(self.1.get() + 1);
            if self.0.get() {
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            } else {
                Err(())
            }
        }
    }
    const WINDOW_BYTES: usize = kore_rt::svc::window_bytes::<MemoryManager>(REQ_CAP, REP_CAP);
    #[repr(align(64))]
    struct Window([u8; WINDOW_BYTES]);
    fn run<F: core::future::Future>(future: F) -> F::Output {
        let park = kore_rt::transport::TrapTransport::new();
        kore_rt::Executor::from_standing(
            &park,
            kore_rt::executor::Doorbell {
                chan: 0,
                recv: SlotRef::from_wire(101),
                land: SlotRef::from_wire(102),
                recv_tag: 7,
            },
        )
        .run(future)
    }

    #[test]
    fn supply_recovery_preserves_an_unrelated_mm_reply_and_does_not_start_a_second_request() {
        let transport = ReplyTransport(Cell::new(false), Cell::new(0));
        let mut window = Window([0; WINDOW_BYTES]);
        let (client, mut serve) =
            kore_rt::svc::split_window::<MemoryManager>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        let mut mm = RingMm(SvcClient::new(
            &transport,
            client,
            ClientEndpoints {
                kick: SlotRef::from_wire(100),
                grants: [None; kore_rt::service::MAX_GRANTS],
            },
        ));
        let request = MmRequest::Release {
            slot: SlotRef::from_wire(0),
        };
        assert!(matches!(
            run(mm.0.call_until(&request, 100)),
            Err(CallError::Transport)
        ));
        let original = serve.req.pop().unwrap();
        serve
            .rep
            .push(MemoryManager::encode_rep(&MmResponse::Released, 0))
            .unwrap();
        let calls = transport.1.get();
        for bytes in [4096, 8192] {
            assert!(matches!(
                run(mm.supply_observed(bytes, Some(100))),
                Err(SupplyError::NotStarted(CallError::Outstanding))
            ));
            assert!(matches!(
                run(mm.resume_supply_bounded(bytes, 1)),
                Err(CallError::Outstanding)
            ));
            assert_eq!(mm.0.outstanding_request(), Some(&original));
            assert_eq!(transport.1.get(), calls);
            assert!(serve.req.pop().is_none());
        }
        assert_eq!(run(mm.stats_observed(100)), Err(CallError::Outstanding));
        assert_eq!(run(mm.resume_stats_bounded(1)), Err(CallError::Outstanding));
        assert_eq!(mm.0.outstanding_request(), Some(&original));
        assert_eq!(transport.1.get(), calls);
        transport.0.set(true);
        let reply = run(mm.0.resume_exact_bounded(&request, 1)).unwrap();
        assert_eq!(reply.response, MmResponse::Released);
        assert!(mm.0.is_idle());
        assert!(serve.req.pop().is_none());
    }

    #[test]
    fn a_supply_refusal_settles_but_a_granted_reply_without_its_owner_poisons() {
        let transport = ReplyTransport(Cell::new(true), Cell::new(0));
        let mut window = Window([0; WINDOW_BYTES]);
        let (client, _) =
            kore_rt::svc::split_window::<MemoryManager>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        let mut mm = RingMm(SvcClient::new(
            &transport,
            client,
            ClientEndpoints {
                kick: SlotRef::from_wire(100),
                grants: [None; kore_rt::service::MAX_GRANTS],
            },
        ));
        let reply = |response, grants| CallReply {
            response,
            grants,
            delivered: [None; kore_rt::service::MAX_GRANTS],
            retained: [const { None }; kore_rt::service::MAX_GRANTS],
        };
        assert!(matches!(
            mm.finish_supply(reply(MmResponse::Refused(MmRefusal::Quota), 0)),
            Ok((MmResponse::Refused(MmRefusal::Quota), None))
        ));
        assert!(mm.0.is_idle());
        assert!(matches!(
            mm.finish_supply(reply(MmResponse::Granted, 1)),
            Err(CallError::Wire)
        ));
        assert!(!mm.0.is_idle());
        assert!(matches!(
            run(mm.resume_supply_bounded(4096, 1)),
            Err(CallError::Poisoned)
        ));
    }

    #[test]
    fn stats_rejects_grants_and_wrong_reply_kinds_without_reopening_the_pair() {
        for (response, grants) in [
            (
                MmResponse::Stats(kore_abi::monitor::MemorySnapshot::EMPTY),
                1,
            ),
            (MmResponse::Released, 0),
        ] {
            let transport = ReplyTransport(Cell::new(true), Cell::new(0));
            let mut window = Window([0; WINDOW_BYTES]);
            let (client, _) =
                kore_rt::svc::split_window::<MemoryManager>(&mut window.0, REQ_CAP, REP_CAP)
                    .unwrap();
            let mut mm = RingMm(SvcClient::new(
                &transport,
                client,
                ClientEndpoints {
                    kick: SlotRef::from_wire(100),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            ));
            let reply = |response, grants| CallReply {
                response,
                grants,
                delivered: [None; kore_rt::service::MAX_GRANTS],
                retained: [const { None }; kore_rt::service::MAX_GRANTS],
            };
            assert_eq!(
                mm.finish_stats(reply(MmResponse::Refused(MmRefusal::Denied), 0)),
                Ok(None)
            );
            assert_eq!(
                mm.finish_stats(reply(
                    MmResponse::Stats(kore_abi::monitor::MemorySnapshot::EMPTY),
                    0
                )),
                Ok(None)
            );
            assert_eq!(
                mm.finish_stats(reply(response, grants)),
                Err(CallError::Wire)
            );
            assert!(!mm.0.is_idle());
            assert_eq!(run(mm.resume_stats_bounded(1)), Err(CallError::Poisoned));
        }
    }

    #[test]
    fn alloc_props_builds_the_real_request_artifact() {
        // The wire artifact `MmRequest::Alloc` carries — pinned here.
        let p = alloc_props(1000);
        assert_eq!(p.align, EXTENT_ALIGN as u64);
        assert_eq!(p.size, EXTENT_ALIGN as u64, "min_bytes rounded to a page");
        assert_eq!(p.dma, DmaClass::None, "a CPU heap extent is non-DMA");
        assert!(!p.contiguous, "a heap extent needs no contiguity");
        assert!(
            p.attrs
                .contains(MemAttrs::WRITE_BACK | MemAttrs::READ | MemAttrs::WRITE)
        );
        assert_eq!(p.numa, NumaNode::ANY);
    }
}

#[cfg(target_os = "none")]
impl RingMm<'static, 'static, kore_rt::transport::TrapTransport> {
    /// Attach this task's declared mm client and its actual executor bell.
    pub fn attach_declared(rt: &kore_rt::entry::Rt, ordinal: u16) -> Option<Self> {
        use kore_abi::task::{SERVICE_CLIENT_INDEX_BASE, role};
        let index = SERVICE_CLIENT_INDEX_BASE.checked_add(ordinal)?;
        let window = kore_rt::cap::verbs::map_view(kore_rt::cap::verbs::take_view(
            rt.bootstrap().handle(role::SVC_RING, index)?,
        )?)?;
        // SAFETY: the boot declaration assigns this task the sole client half.
        let bytes = unsafe { &mut *window.as_ptr() };
        let rings = kore_rt::svc::client_rings::<crate::MemoryManager>(
            bytes,
            kore_rt::svc::REQ_CAP,
            kore_rt::svc::REP_CAP,
        )
        .ok()?;
        let kick = rt.bootstrap().handle(role::SVC_SEND, index)?;
        let grant_index = SERVICE_CLIENT_INDEX_BASE
            .checked_add(ordinal.checked_mul(kore_rt::service::MAX_GRANTS as u16)?)?;
        let grant = rt.bootstrap().handle(role::SVC_GRANT, grant_index)?;
        let mut grants = [const { None }; kore_rt::service::MAX_GRANTS];
        grants[0] = Some(kore_rt::svc::GrantPin::retained(grant));
        Some(Self(
            kore_rt::svc::SvcClient::new(
                kore_rt::entry::transport(),
                rings,
                kore_rt::svc::ClientEndpoints { kick, grants },
            )
            .with_reply_wait(rt.doorbell()?),
        ))
    }
}
