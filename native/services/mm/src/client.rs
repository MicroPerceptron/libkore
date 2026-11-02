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

/// The **client** side of the mm protocol for *peers*
/// (`res/USERLAND.md` §4: "MmClient exists for peers; the global
/// allocator does NOT use it").  A `Granted` reply's authority arrives
/// beside the response as a delivered, runtime-owned capability —
/// possession-shaped, never reply geometry.
#[allow(async_fn_in_trait)] // clients are generic over the impl (§4)
pub trait MmClient {
    /// Whether a fresh request can be published. A bounded administration
    /// wait can leave an exact continuation owning this exclusive pair.
    fn is_idle(&self) -> bool;

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
/// An infrastructure failure of the request path **panics** (the
/// executor's transport-failure posture): the closed `MmRefusal`
/// vocabulary has no honest infra arm, and inventing one would let a
/// torn pair masquerade as a lessor refusal.
pub struct RingMm<'w, 't, T: kore_rt::transport::Transport>(
    pub kore_rt::svc::SvcClient<'w, 't, T, crate::service::MemoryManager>,
);

impl<T: kore_rt::transport::Transport> RingMm<'_, '_, T> {
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

    /// Carry one exact administration request forward under `observation`.
    /// Maintenance observes in slices, so a departing process that still has
    /// DMA supplies outstanding cannot monopolize exec's serve loop; an
    /// expired slice retains the exact request on this pair for the next
    /// call with the same request.
    async fn peer_control(
        &mut self,
        request: MmRequest,
        observation: kore_rt::svc::Observation,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        use kore_rt::svc::CallError;
        let reply = self.0.call_exact(&request, observation).await?;
        if reply.grants != 0 {
            self.0.poison_retained(reply.retained);
            return Err(CallError::Wire);
        }
        Ok(reply.response)
    }

    pub async fn take_peer_admission_observed(
        &mut self,
        observation: kore_rt::svc::Observation,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        self.peer_control(MmRequest::TakeAdmission, observation)
            .await
    }

    pub async fn retire_peer_observed(
        &mut self,
        peer: kore_rt::svc::PeerId,
        observation: kore_rt::svc::Observation,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        self.peer_control(MmRequest::RetirePeer { peer }, observation)
            .await
    }

    pub async fn confirm_retired_observed(
        &mut self,
        peer: kore_rt::svc::PeerId,
        observation: kore_rt::svc::Observation,
    ) -> Result<MmResponse, kore_rt::svc::CallError> {
        self.peer_control(MmRequest::ConfirmRetired { peer }, observation)
            .await
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
#[cfg(any(target_os = "none", test))]
const HEAP_RESERVATION_CLASS: u64 = 0x20_0000;

/// The most one lease of a heap reservation asks for.  A lease is one
/// physically contiguous range of its class, and a reservation is as many
/// leases as it takes: 128 MiB is what the kernel's 2 MiB class has lent a
/// single launch on the 4 GiB acceptance machine (the wasm witness's), where
/// one 1 GiB ask was refused outright.  The heap adopts each lease
/// ([`kore_rt::heap::MAX_ADOPTED`] of them at most), and grows through them
/// in turn.
#[cfg(any(target_os = "none", test))]
pub const HEAP_RESERVATION_LEASE_BYTES: u64 = 128 << 20;

/// The least one lease asks for.  A `Denied` grant is the class without a
/// free run that long, so the ask halves down to this and no further: a
/// class that cannot lend 16 MiB contiguous is spent, not fragmented.  The
/// `wasm` command's 1 GiB ceiling is 64 such leases, the adoption bound.
#[cfg(any(target_os = "none", test))]
pub const HEAP_RESERVATION_LEASE_MIN_BYTES: u64 = 16 << 20;

/// Why a heap reservation stopped short, and how much of it stands.
#[cfg(any(target_os = "none", test))]
#[derive(Debug)]
pub struct HeapReservationRefused {
    /// mm's own answer (`Quota` names the face's budget), or `Denied` for a
    /// grant the heap could not adopt (the lease is then revoked).
    pub response: MmResponse,
    /// Bytes reserved and adopted before the refusal — the heap's to keep.
    pub reserved: u64,
}

/// Why one lease of a reservation did not join the heap.
#[cfg(any(target_os = "none", test))]
#[derive(Debug)]
enum Unlent {
    /// mm did not lend it: its own answer.
    Refused(MmResponse),
    /// mm lent it and the heap could not adopt it (every adoption entry
    /// taken); the lease was revoked.  A shorter one would fare no better.
    NotAdopted,
}

/// [`ProcessMm::reserve_heap`]'s policy, apart from the face: `lend(ask)`
/// asks for one `ask`-byte lease of the class and adopts it into the heap.
#[cfg(any(target_os = "none", test))]
async fn reserve_in_leases(
    bytes: u64,
    lend: &mut impl AsyncFnMut(u64) -> Result<(), Unlent>,
) -> Result<u64, HeapReservationRefused> {
    let mut reserved = 0;
    let mut lease_bytes = HEAP_RESERVATION_LEASE_BYTES;
    while reserved < bytes {
        // Asked in whole units of the class, as mm lends it, so `reserved`
        // counts exactly what the heap adopted.
        let ask = (bytes - reserved)
            .min(lease_bytes)
            .next_multiple_of(HEAP_RESERVATION_CLASS);
        match lend(ask).await {
            Ok(()) => reserved += ask,
            // A run the class could not lend: each retry asks for strictly
            // less than the run just refused, so an ask already at the floor
            // is not repeated.
            Err(Unlent::Refused(MmResponse::Refused(crate::service::MmRefusal::Denied)))
                if ask > HEAP_RESERVATION_LEASE_MIN_BYTES =>
            {
                lease_bytes = (ask / 2)
                    .next_multiple_of(HEAP_RESERVATION_CLASS)
                    .max(HEAP_RESERVATION_LEASE_MIN_BYTES);
            }
            Err(Unlent::Refused(response)) => {
                return Err(HeapReservationRefused { response, reserved });
            }
            Err(Unlent::NotAdopted) => {
                return Err(HeapReservationRefused {
                    response: MmResponse::Refused(crate::service::MmRefusal::Denied),
                    reserved,
                });
            }
        }
    }
    Ok(reserved)
}

/// [`ProcessMm::reserve_heap_ladder`]'s policy, apart from the face.
#[cfg(any(target_os = "none", test))]
async fn reserve_down_ladder(
    targets: &[u64],
    lend: &mut impl AsyncFnMut(u64) -> Result<(), Unlent>,
) -> (u64, Option<HeapReservationRefused>) {
    let mut reserved = 0;
    let mut refusal = None;
    for &target in targets {
        if reserved >= target {
            return (reserved, None);
        }
        match reserve_in_leases(target - reserved, lend).await {
            Ok(gained) => return (reserved + gained, None),
            Err(refused) => {
                reserved += refused.reserved;
                refusal = Some(refused);
            }
        }
    }
    (reserved, refusal)
}

/// [`ProcessMm::reserve_heap_window`]'s policy, apart from the face: one ask
/// for the whole window, in whole units of the class, never split or retried
/// shorter — a window is one carve, so a shorter run cannot hold it.  A
/// window no ask can represent is refused as `BadProps` without asking.
#[cfg(any(target_os = "none", test))]
async fn reserve_window(
    bytes: u64,
    lend: &mut impl AsyncFnMut(u64) -> Result<(), Unlent>,
) -> Result<u64, HeapReservationRefused> {
    let refused = |refusal| HeapReservationRefused {
        response: MmResponse::Refused(refusal),
        reserved: 0,
    };
    let ask = bytes
        .max(1)
        .checked_next_multiple_of(HEAP_RESERVATION_CLASS)
        .ok_or(refused(crate::service::MmRefusal::BadProps))?;
    match lend(ask).await {
        Ok(()) => Ok(ask),
        Err(Unlent::Refused(response)) => Err(HeapReservationRefused {
            response,
            reserved: 0,
        }),
        Err(Unlent::NotAdopted) => Err(refused(crate::service::MmRefusal::Denied)),
    }
}

#[cfg(target_os = "none")]
impl ProcessMm {
    /// **Reserve heap before a large synchronous job.** Ask mm for `bytes`
    /// more on this process's own face — as leases of at most
    /// [`HEAP_RESERVATION_LEASE_BYTES`], each rounded up to whole 2 MiB
    /// units, a `Denied` run retried at half its length down to
    /// [`HEAP_RESERVATION_LEASE_MIN_BYTES`] — and adopt each grant into the
    /// heap ([`kore_rt::heap::adopt_supply`]), so the job's allocations —
    /// which cannot wait on a service — find room.  `Ok` is the bytes the
    /// heap adopted: `bytes` rounded up to the 2 MiB unit.
    ///
    /// A refusal carries mm's own answer (`Quota` names the face's budget
    /// and ends the reservation at once; `Denied` is a run the class could
    /// not lend at the floor, and also a grant the heap could not adopt,
    /// which revokes the lease, never leaks it) and how much stood before
    /// it.  Leases adopted before a refusal stay the heap's: a caller that
    /// goes on anyway has that much more room, not less.
    pub async fn reserve_heap(&mut self, bytes: u64) -> Result<u64, HeapReservationRefused> {
        reserve_in_leases(bytes, &mut async |ask| {
            self.lend(ask, kore_rt::heap::adopt_supply).await
        })
        .await
    }

    /// Make room for a step that maps `bytes` afresh: what the supplies the
    /// heap adopted still hold ([`kore_rt::heap::heap_room`]), topped up on
    /// this face when short, by at least [`HEAP_RESERVATION_LEASE_MIN_BYTES`]
    /// so a long-lived process holds few leases. `false` when the room still
    /// falls short; what was adopted before a refusal stays the heap's.
    pub async fn ensure_heap_room(&mut self, bytes: u64) -> bool {
        let room = kore_rt::heap::heap_room();
        if room >= bytes {
            return true;
        }
        let _ = self
            .reserve_heap((bytes - room).max(HEAP_RESERVATION_LEASE_MIN_BYTES))
            .await;
        kore_rt::heap::heap_room() >= bytes
    }

    /// **Reserve room for one window of `bytes`** that this process will
    /// map itself ([`kore_rt::heap::reserve_from_top`]) — a guest memory's
    /// next commits, a code image — when no supply it holds for windows
    /// has that much room left ([`kore_rt::heap::window_supply_room`]).  A
    /// window is one carve, so this is **one** lease of `bytes` rounded up
    /// to the 2 MiB unit, never split as [`reserve_heap`](Self::reserve_heap) splits:
    /// adopted for windows alone ([`kore_rt::heap::adopt_window_supply`]),
    /// so heap growth cannot shorten it before the window is carved.  `Ok`
    /// is the lease's length; a refusal is mm's answer (`Denied` when the
    /// class has no run that long, or the heap could not adopt the lease),
    /// with nothing reserved.
    pub async fn reserve_heap_window(&mut self, bytes: u64) -> Result<u64, HeapReservationRefused> {
        reserve_window(bytes, &mut async |ask| {
            self.lend(ask, kore_rt::heap::adopt_window_supply).await
        })
        .await
    }

    /// [`reserve_heap`](Self::reserve_heap) down a ladder of `targets`,
    /// largest first: the heap grows to the first target the face can bring
    /// it to.  A refused step keeps what it adopted, so the next step asks
    /// only for what its own target still lacks, and the bytes returned are
    /// exactly what the heap gained — the number a caller takes off a census
    /// it read before reserving.  The refusal is the last step's, returned
    /// only when even the smallest target was not reached.
    pub async fn reserve_heap_ladder(
        &mut self,
        targets: &[u64],
    ) -> (u64, Option<HeapReservationRefused>) {
        reserve_down_ladder(targets, &mut async |ask| {
            self.lend(ask, kore_rt::heap::adopt_supply).await
        })
        .await
    }

    /// One lease of a reservation, adopted by `adopt`.
    async fn lend(
        &mut self,
        ask: u64,
        adopt: fn(kore_rt::cap::StashedLease) -> Result<(), kore_rt::cap::StashedLease>,
    ) -> Result<(), Unlent> {
        let (response, lease) = self
            .client
            .supply_in_class(ask, HEAP_RESERVATION_CLASS)
            .await;
        let lease = lease.ok_or(Unlent::Refused(response))?;
        // A refused adoption hands the lease back; dropping it revokes it.
        adopt(lease).map_err(|_| Unlent::NotAdopted)
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
    fn is_idle(&self) -> bool {
        self.0.is_idle()
    }

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

    const MIB: u64 = 1 << 20;

    /// A face that lends runs no longer than `run` while its `quota` lasts,
    /// into a heap with `slots` adoption entries left — and records every
    /// ask and every byte it lent.
    struct Face {
        quota: u64,
        run: u64,
        slots: usize,
        asks: std::vec::Vec<u64>,
        lent: u64,
    }

    impl Face {
        fn new(quota: u64, run: u64, slots: usize) -> Self {
            Self {
                quota,
                run,
                slots,
                asks: std::vec::Vec::new(),
                lent: 0,
            }
        }

        fn lend(&mut self, ask: u64) -> Result<(), Unlent> {
            use crate::service::MmRefusal::{Denied, Quota};
            self.asks.push(ask);
            assert_eq!(ask % HEAP_RESERVATION_CLASS, 0, "asks are whole units");
            if ask > self.quota {
                return Err(Unlent::Refused(MmResponse::Refused(Quota)));
            }
            if ask > self.run {
                return Err(Unlent::Refused(MmResponse::Refused(Denied)));
            }
            // The lease was lent, so it counts against the quota even when
            // the heap cannot adopt it (mm sees its revocation later).
            self.quota -= ask;
            if self.slots == 0 {
                return Err(Unlent::NotAdopted);
            }
            self.slots -= 1;
            self.lent += ask;
            Ok(())
        }

        fn reserve(&mut self, bytes: u64) -> Result<u64, HeapReservationRefused> {
            kore_rt::block_on(reserve_in_leases(bytes, &mut async |ask| self.lend(ask)))
        }

        fn ladder(&mut self, targets: &[u64]) -> (u64, Option<HeapReservationRefused>) {
            kore_rt::block_on(reserve_down_ladder(targets, &mut async |ask| {
                self.lend(ask)
            }))
        }

        fn window(&mut self, bytes: u64) -> Result<u64, HeapReservationRefused> {
            kore_rt::block_on(reserve_window(bytes, &mut async |ask| self.lend(ask)))
        }
    }

    /// A class of free runs, lent first-fit: a lease is carved from the
    /// first run at least as long, which shrinks by it.
    struct Runs(std::vec::Vec<u64>);

    impl Runs {
        fn lend(&mut self, ask: u64) -> Result<(), Unlent> {
            use crate::service::MmRefusal::Denied;
            let run = self
                .0
                .iter_mut()
                .find(|run| **run >= ask)
                .ok_or(Unlent::Refused(MmResponse::Refused(Denied)))?;
            *run -= ask;
            Ok(())
        }

        fn reserve(&mut self, bytes: u64) -> Result<u64, HeapReservationRefused> {
            kore_rt::block_on(reserve_in_leases(bytes, &mut async |ask| self.lend(ask)))
        }

        fn window(&mut self, bytes: u64) -> Result<u64, HeapReservationRefused> {
            kore_rt::block_on(reserve_window(bytes, &mut async |ask| self.lend(ask)))
        }
    }

    /// Why a program that needs both reserves its window first. The heap
    /// asks for its whole length, then halves, so its first lease takes
    /// the class's longest run; a window asked for afterwards finds no run
    /// of its own length, though one stood free before (the `wasm` command's
    /// CPython window beside a loaded model). Asked first, the window takes
    /// its run and the heap gathers its bytes from what is left.
    #[test]
    fn a_window_reserved_before_the_heap_finds_the_run_it_needs() {
        use crate::service::MmRefusal::Denied;
        let fragmented = || Runs(std::vec![40 * MIB, 16 * MIB, 16 * MIB, 16 * MIB, 16 * MIB]);
        let (window, heap) = (22 * MIB, 70 * MIB);

        let mut heap_first = fragmented();
        assert_eq!(heap_first.reserve(heap).ok(), Some(heap));
        let refused = heap_first.window(window).unwrap_err();
        assert_eq!((refused.response, refused.reserved), (MmResponse::Refused(Denied), 0));

        let mut window_first = fragmented();
        assert_eq!(window_first.window(window).ok(), Some(window));
        assert_eq!(window_first.reserve(heap).ok(), Some(heap));
    }

    /// A window is one lease of its whole length, rounded to the class, and
    /// never split: past the longest run it is refused at once, keeping
    /// nothing, where a reservation would have taken shorter leases.
    #[test]
    fn a_window_is_one_whole_lease() {
        use crate::service::MmRefusal::{BadProps, Denied, Quota};
        let mut face = Face::new(u64::MAX, u64::MAX, 64);
        assert_eq!(face.window(200 * MIB + 1).ok(), Some(202 * MIB));
        assert_eq!(face.window(0).ok(), Some(2 * MIB));
        assert_eq!(face.asks, [202 * MIB, 2 * MIB]);
        // Longer than the class's longest run: one ask, refused.
        let mut face = Face::new(u64::MAX, 64 * MIB, 64);
        let refused = face.window(80 * MIB).unwrap_err();
        assert_eq!(refused.response, MmResponse::Refused(Denied));
        assert_eq!((refused.reserved, face.lent), (0, 0));
        assert_eq!(face.asks, [80 * MIB]);
        // The quota's answer is mm's own; an unadoptable lease is Denied.
        let mut face = Face::new(32 * MIB, u64::MAX, 64);
        assert_eq!(
            face.window(48 * MIB).unwrap_err().response,
            MmResponse::Refused(Quota)
        );
        let mut face = Face::new(u64::MAX, u64::MAX, 0);
        assert_eq!(
            face.window(48 * MIB).unwrap_err().response,
            MmResponse::Refused(Denied)
        );
        // A window no ask can represent is refused without asking.
        let mut face = Face::new(u64::MAX, u64::MAX, 64);
        assert_eq!(
            face.window(u64::MAX).unwrap_err().response,
            MmResponse::Refused(BadProps)
        );
        assert!(face.asks.is_empty());
    }

    #[test]
    fn a_reservation_is_whole_leases_of_the_class() {
        let mut face = Face::new(u64::MAX, u64::MAX, 64);
        assert_eq!(face.reserve(300 * MIB + 1).ok(), Some(302 * MIB));
        assert_eq!(face.asks, [128 * MIB, 128 * MIB, 46 * MIB]);
        assert_eq!(face.lent, 302 * MIB);
    }

    #[test]
    fn a_denied_run_is_retried_strictly_shorter_down_to_the_floor() {
        // A class whose longest free run is 40 MiB lends 32 MiB leases.
        let mut face = Face::new(u64::MAX, 40 * MIB, 64);
        assert_eq!(face.reserve(64 * MIB).ok(), Some(64 * MIB));
        assert_eq!(face.asks, [64 * MIB, 32 * MIB, 32 * MIB]);
        // One without a 16 MiB run is spent: the floor is asked once.
        let mut face = Face::new(u64::MAX, 8 * MIB, 64);
        let refused = face.reserve(64 * MIB).unwrap_err();
        assert_eq!(face.asks, [64 * MIB, 32 * MIB, 16 * MIB]);
        assert_eq!(refused.reserved, 0);
        // An ask already under the floor is not repeated either.
        let mut face = Face::new(u64::MAX, 8 * MIB, 64);
        assert!(face.reserve(10 * MIB).is_err());
        assert_eq!(face.asks, [10 * MIB]);
    }

    #[test]
    fn a_refusal_keeps_what_was_adopted() {
        use crate::service::MmRefusal::{Denied, Quota};
        let mut face = Face::new(200 * MIB, u64::MAX, 64);
        let refused = face.reserve(256 * MIB).unwrap_err();
        assert_eq!(refused.response, MmResponse::Refused(Quota));
        assert_eq!((refused.reserved, face.lent), (128 * MIB, 128 * MIB));
        // A grant the heap cannot adopt ends the reservation: a shorter one
        // would fare no better, so nothing more is asked.
        let mut face = Face::new(u64::MAX, u64::MAX, 1);
        let refused = face.reserve(256 * MIB).unwrap_err();
        assert_eq!(refused.response, MmResponse::Refused(Denied));
        assert_eq!((refused.reserved, face.lent), (128 * MIB, 128 * MIB));
        assert_eq!(face.asks, [128 * MIB, 128 * MIB]);
    }

    /// The census a caller reads before reserving is short by exactly what
    /// the ladder returns, whatever stopped it: a step refused part-way
    /// keeps its leases, and the next step asks only for its own deficit.
    #[test]
    fn the_ladder_returns_exactly_what_the_heap_gained() {
        const LADDER: [u64; 3] = [256 * MIB, 128 * MIB, 64 * MIB];
        // A 256 MiB step that adopts 192 MiB in 64 MiB runs, then meets
        // the quota, already stands past the 128 MiB step.
        let mut face = Face::new(192 * MIB, 64 * MIB, 64);
        let (gained, refusal) = face.ladder(&LADDER);
        assert_eq!((gained, face.lent), (192 * MIB, 192 * MIB));
        assert!(refusal.is_none());
        assert_eq!(
            face.asks,
            [128 * MIB, 64 * MIB, 64 * MIB, 64 * MIB, 64 * MIB]
        );
        // A quota under the first two steps' first lease reaches the third.
        let mut face = Face::new(100 * MIB, u64::MAX, 64);
        assert_eq!(face.ladder(&LADDER).0, 64 * MIB);
        assert_eq!(face.asks, [128 * MIB, 128 * MIB, 64 * MIB]);
        // A 64 MiB step refused after keeping 32 MiB asks the 40 MiB step
        // for the 8 MiB it still lacks, not for 40 MiB more.
        let mut face = Face::new(u64::MAX, 32 * MIB, 1);
        let (gained, refusal) = face.ladder(&[64 * MIB, 40 * MIB]);
        assert_eq!((gained, face.lent), (32 * MIB, 32 * MIB));
        assert_eq!(face.asks, [64 * MIB, 32 * MIB, 32 * MIB, 8 * MIB]);
        assert!(refusal.is_some());
        // Every face: the gain is what was lent, never past the top step,
        // and a refusal is reported exactly when no step was reached.
        for quota in [0, 10, 40, 64, 100, 128, 192, 250, 256, 1024] {
            for run in [8, 16, 24, 40, 64, 128, 1024] {
                for slots in [0, 1, 2, 3, 64] {
                    let mut face = Face::new(quota * MIB, run * MIB, slots);
                    let (gained, refusal) = face.ladder(&LADDER);
                    let case = (quota, run, slots);
                    assert_eq!(gained, face.lent, "{case:?}");
                    assert!(gained <= LADDER[0], "{case:?}");
                    assert_eq!(refusal.is_some(), gained < LADDER[2], "{case:?}");
                }
            }
        }
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
