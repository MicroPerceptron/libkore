//! `svc::serve` — the serve-loop machine (`res/USERLAND.md` §5.5): the
//! executor task every kore-svc-* runs its `SystemService` under.
//!
//! One request = drain the request ring → **fail-closed wire decode** →
//! `serve()` (sync, grants + one reply, #206 — untouched) → **effect
//! the grants as ONE staged session** of capability transfers into the
//! client's pre-pinned reply posts (a `GrantArg::Window` grant stages
//! its derive first and transfers the staged child — SSA in the same
//! session) → push the wire reply (+ the *effected* grant count) →
//! notice-kick the client's doorbell (its own session, droppable).
//!
//! **The backpressure split (§5.5, verbatim)**: ring-borne replies and
//! notices are level-triggered and droppable — a full reply ring is the
//! client's problem (counted, never blocking).  Grant-bearing replies
//! are edge-delivered transfers with no ring state to re-read, so the
//! client pre-pins the posts and a conforming client can never make the
//! transfer `WouldBlock`; a `WouldBlock` (or any refusal) on the grant
//! session therefore indicates a **protocol-violating client** — the
//! service *drops the grant* (bounded revoke of anything the session
//! derived) and *flags* it (the counter + a `grants = 0` reply), and
//! **never blocks**.

#[cfg(feature = "heap")]
use alloc::boxed::Box;

use kore_abi::session::{Committed, Notice, SlotRef, Status};

use crate::cap::{MappedLeaseView, ReturnedCapability, StashedLease};
use crate::chan::ChanTx;
use crate::service::{GrantArg, MAX_GRANTS, PeerClose, ReturnMode, step_with_return_async};
use crate::session::{SessionBuilder, flows};
use crate::transport::{ParkPost, Transport, stage_grant};

use super::{COOKIE_REPLY, COOKIE_REPORT, ServeRings};

use crate::service::SystemService;

/// The affine owner of a bound service.
///
/// Production service images enable `heap`, so their often-large bounded
/// service state is pinned once instead of being copied through async stack
/// frames. Allocator-free `kore-rt` clients retain the original inline owner;
/// they compile the protocol machinery but never enter the service runtime.
#[cfg(feature = "heap")]
pub(crate) type ServiceOwner<S> = Box<S>;

#[cfg(not(feature = "heap"))]
pub(crate) type ServiceOwner<S> = S;

#[cfg(feature = "heap")]
pub(crate) fn own_service<S>(service: S) -> ServiceOwner<S> {
    Box::new(service)
}

#[cfg(not(feature = "heap"))]
pub(crate) fn own_service<S>(service: S) -> ServiceOwner<S> {
    service
}

/// The service's channel wiring for one pair (§5.3): the client-kick
/// send end, the grant-channel send ends, the derive scratch a
/// window-carving grant stages through, and the return channel the
/// service stands its cap-return post on.
#[derive(Clone, Copy, Debug)]
pub struct ServeEndpoints {
    /// The held `ChanSend` to the **client's** doorbell.
    pub kick: SlotRef,
    /// Grant-channel send ends, `[0..REPLY_GRANTS]` populated —
    /// transfer `k` of a reply rides channel `k`.
    pub grant_tx: [Option<SlotRef>; MAX_GRANTS],
    /// Scratch result slots for `GrantArg::Window` derives (one per
    /// possible grant; unused by transfer-only protocols).
    pub scratch: [Option<SlotRef>; MAX_GRANTS],
    /// The return channel's `(recv, land)` pair (`role::SVC_RETURN`). Fixed
    /// protocols donate `land` directly (e.g. mm's `Release { slot }`). A
    /// A retained-return service uses `land` as the published wire
    /// identity and donates a runtime-owned persistent target instead. `None`
    /// means no return leg and no standing post.
    pub return_post: Option<(SlotRef, SlotRef)>,
}

/// The serve loop's visible books (the mm `leaked_bytes` doctrine:
/// every anomaly is counted, never silent).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServeStats {
    /// Requests whose reply **landed in the reply ring**.
    ///
    /// Deliberately not "requests stepped": a delta on this counter is what
    /// [`entry::serve`](crate::entry::serve) turns into a client wake, so it
    /// has to mean "this client has something to collect". A step whose
    /// reply was lost to a full ring counts under
    /// [`replies_dropped`](Self::replies_dropped) instead and nowhere else, so
    /// the two together are still the served-step count. Every host that
    /// publishes this type — [`ServeHost`] and the bespoke display/input
    /// loops — counts it this way, so a consumer never has to know which
    /// produced the record.
    pub served: u64,
    /// Ring records that failed the fail-closed decode — dropped.
    pub wire_refused: u64,
    /// Grant sessions dropped on a protocol-violating client (§5.5).
    pub grants_dropped: u64,
    /// Replies lost to a full reply ring (the client's problem).
    pub replies_dropped: u64,
    /// Returned capabilities whose first teardown attempt refused and remain
    /// owned by the host for bounded retry.
    pub return_cleanup_deferred: u64,
}

/// A task arena has at most this many live records, so a cleanup owner for
/// every rejected returned capability fits without allocation.  The runtime
/// opens only the founder-stamped dynamic interval below the durable linker
/// band, so the number of simultaneously reservable return targets is
/// strictly smaller than this defensive ceiling.
const RETURN_CLEANUP_CAP: usize = crate::entry::ARENA_SLOT_CEILING as usize;

/// Maximum OCC attempts for one already-built grant transaction.  Reusing
/// the descriptor is load-bearing: rebuilding window grants under a new
/// scratch choice would sever the service reply from the exact authority it
/// reserved while handling the request.
const GRANT_COMMIT_RETRIES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrantCommitAction {
    Committed,
    Retry,
    Drop,
}

/// Classify the atomic grant transaction's completion.  I3 proves a
/// `Conflict` at a real invocation applied nothing, so the exact descriptor
/// may be retried.  Every other refusal is terminal for this response and
/// follows the existing bounded drop/repair path.
fn grant_commit_action(committed: Committed, invocations: usize) -> GrantCommitAction {
    match committed.status() {
        Status::Ok if committed.failing_index() == 0 => GrantCommitAction::Committed,
        Status::Conflict if usize::from(committed.failing_index()) < invocations => {
            GrantCommitAction::Retry
        }
        _ => GrantCommitAction::Drop,
    }
}

/// Inline bootstrap serving pairs. Owners with a named arena may grow their
/// peer population; this only bounds the manifest's fixed pair roster.
pub const MAX_SERVE_PAIRS: usize =
    kore_abi::task::DESKTOP_BOOTSTRAP_WINDOW_SLOTS + kore_abi::task::DESKTOP_CONTROLLER_CAPACITY;

/// A recyclable serving-peer identity.
///
/// The slot selects one bounded host cell and the generation distinguishes
/// successive occupants of that cell.  It is deliberately opaque: callers
/// may retain it for cancellation/observation, but cannot manufacture a
/// current peer from a bare pair index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PeerId {
    slot: u8,
    generation: u16,
}

/// Caller-reserved virtual-address band used for recyclable peer rings.
///
/// The service founder owns the band choice; the generic runtime only checks
/// its arithmetic and places peer slot `n` at `base + n * stride`. Reusing a
/// slot therefore reuses the same VA only after the prior affine mapping has
/// been retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerMapPlan {
    pub base: u64,
    pub stride: u64,
}

impl PeerMapPlan {
    fn address(self, slot: usize, bytes: u64) -> Option<u64> {
        if self.base & 0xfff != 0
            || self.stride == 0
            || self.stride & 0xfff != 0
            || bytes == 0
            || bytes > self.stride
        {
            return None;
        }
        self.base
            .checked_add((slot as u64).checked_mul(self.stride)?)
    }
}

/// Affine resources conveyed by an authenticated acceptor client for one new
/// serving pair.
///
/// Every capability came from the retained return leg and owns its dynamic
/// arena reservation. The wire request may select which bundle component is
/// being delivered, but admission independently verifies the kernel-recorded
/// species before installing any endpoint.
#[derive(Debug)]
pub struct RecyclablePeerBundle {
    ring: ReturnedCapability,
    kick: ReturnedCapability,
    grants: [Option<ReturnedCapability>; MAX_GRANTS],
    return_recv: Option<ReturnedCapability>,
}

/// One component of a dynamically conveyed serving pair.
///
/// The tag selects a bounded assembler cell; it never establishes resource
/// species or rights. [`PeerBundleAssembler::accepts`] checks those facts
/// against the kernel-authenticated record before ownership leaves the serve
/// host's ordinary returned-input cleanup path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerBundlePart {
    Ring,
    Kick,
    Grant(u8),
    ReturnRecv,
}

/// Why a returned capability cannot occupy the requested bundle component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerAssemblyRefusal {
    Duplicate,
    GrantOutOfRange,
    UnexpectedReturn,
    WrongAuthority,
}

/// Refused assembly preserves the same affine returned-capability owner.
#[derive(Debug)]
pub struct PeerAssemblyError {
    pub reason: PeerAssemblyRefusal,
    pub capability: ReturnedCapability,
}

/// Bounded, affine assembly of one recyclable peer.
///
/// A single rendezvous post receives the mixed lease/channel bundle one item
/// at a time. The assembler owns no allocator and no raw slot names: accepted
/// items are the runtime's affine returned-capability owners, and an item that
/// fails validation remains with the caller for the host's retryable cleanup.
#[derive(Debug)]
pub struct PeerBundleAssembler {
    ring: Option<ReturnedCapability>,
    kick: Option<ReturnedCapability>,
    grants: [Option<ReturnedCapability>; MAX_GRANTS],
    return_recv: Option<ReturnedCapability>,
    expected_grants: usize,
    requires_return: bool,
}

impl PeerBundleAssembler {
    pub const fn new(expected_grants: usize, requires_return: bool) -> Self {
        Self {
            ring: None,
            kick: None,
            grants: [const { None }; MAX_GRANTS],
            return_recv: None,
            expected_grants,
            requires_return,
        }
    }

    /// Whether no component from an unfinished bundle remains owned.
    pub fn is_empty(&self) -> bool {
        self.ring.is_none()
            && self.kick.is_none()
            && self.grants.iter().all(Option::is_none)
            && self.return_recv.is_none()
    }

    /// Retire every component of an abandoned incomplete bundle.
    ///
    /// A new `Ring` is the protocol boundary between bundles. Clients resume
    /// an ambiguous Ring call rather than republishing it, so observing a new
    /// Ring while this assembler is non-empty proves that the old assembly
    /// was abandoned. Each owner remains in place until revoke or dead-lineage
    /// observation closes it; callers must refuse the replacement meanwhile.
    pub fn retire_incomplete(&mut self) -> bool {
        fn retire(slot: &mut Option<ReturnedCapability>) {
            let closed = slot
                .as_mut()
                .is_some_and(|capability| capability.try_retire_outcome().is_closed());
            if closed {
                *slot = None;
            }
        }

        retire(&mut self.ring);
        retire(&mut self.kick);
        for grant in &mut self.grants {
            retire(grant);
        }
        retire(&mut self.return_recv);
        self.is_empty()
    }

    /// Validate a proposed component without taking its affine owner.
    pub fn accepts(
        &self,
        part: PeerBundlePart,
        capability: &ReturnedCapability,
    ) -> Result<(), PeerAssemblyRefusal> {
        use kore_abi::cap::{ResourceKind, Rights};

        let accepted = match part {
            PeerBundlePart::Ring => {
                if self.ring.is_some() {
                    return Err(PeerAssemblyRefusal::Duplicate);
                }
                capability.is_lease_with(
                    Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
                )
            }
            PeerBundlePart::Kick => {
                if self.kick.is_some() {
                    return Err(PeerAssemblyRefusal::Duplicate);
                }
                capability.is_kind_with(ResourceKind::ChanSend, Rights::ACCESS)
            }
            PeerBundlePart::Grant(index) => {
                let index = usize::from(index);
                if index >= self.expected_grants || index >= MAX_GRANTS {
                    return Err(PeerAssemblyRefusal::GrantOutOfRange);
                }
                if self.grants[index].is_some() {
                    return Err(PeerAssemblyRefusal::Duplicate);
                }
                capability.is_kind_with(ResourceKind::ChanSend, Rights::ACCESS)
            }
            PeerBundlePart::ReturnRecv => {
                if !self.requires_return {
                    return Err(PeerAssemblyRefusal::UnexpectedReturn);
                }
                if self.return_recv.is_some() {
                    return Err(PeerAssemblyRefusal::Duplicate);
                }
                capability.is_kind_with(ResourceKind::ChanRecv, Rights::ACCESS)
            }
        };
        accepted
            .then_some(())
            .ok_or(PeerAssemblyRefusal::WrongAuthority)
    }

    /// Validate and insert one owner, emitting a bundle when complete.
    pub fn push(
        &mut self,
        part: PeerBundlePart,
        capability: ReturnedCapability,
    ) -> Result<Option<RecyclablePeerBundle>, PeerAssemblyError> {
        if let Err(reason) = self.accepts(part, &capability) {
            return Err(PeerAssemblyError { reason, capability });
        }
        match part {
            PeerBundlePart::Ring => self.ring = Some(capability),
            PeerBundlePart::Kick => self.kick = Some(capability),
            PeerBundlePart::Grant(index) => self.grants[usize::from(index)] = Some(capability),
            PeerBundlePart::ReturnRecv => self.return_recv = Some(capability),
        }
        if self.ring.is_none()
            || self.kick.is_none()
            || self.grants[..self.expected_grants]
                .iter()
                .any(Option::is_none)
            || self.grants[self.expected_grants..]
                .iter()
                .any(Option::is_some)
            || self.requires_return != self.return_recv.is_some()
        {
            return Ok(None);
        }
        Ok(Some(RecyclablePeerBundle::new(
            self.ring.take().unwrap(),
            self.kick.take().unwrap(),
            core::array::from_fn(|index| self.grants[index].take()),
            self.return_recv.take(),
        )))
    }
}

impl RecyclablePeerBundle {
    pub fn new(
        ring: ReturnedCapability,
        kick: ReturnedCapability,
        grants: [Option<ReturnedCapability>; MAX_GRANTS],
        return_recv: Option<ReturnedCapability>,
    ) -> Self {
        Self {
            ring,
            kick,
            grants,
            return_recv,
        }
    }
}

/// Why an owned recyclable peer could not be installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerAdmissionRefusal {
    Capacity,
    InvalidMapPlan,
    WrongRingAuthority,
    WrongEndpointAuthority,
    WrongGrantCount,
    WrongReturnShape,
    RingGeometry,
    NoScratchSlots,
    MapFailed,
    RingMalformed,
    /// The generic host installed the pair, but the domain service already
    /// holds the state this arrival would address — the slot is occupied by a
    /// registration that was never retired. The host retires the pair before
    /// it can be polled.
    ///
    /// Distinct from [`Self::PolicyUnavailable`] because a collision is not a
    /// capacity condition: nothing is full, one row is stale. Collapsing the
    /// two let a domain report a stale registration as exhaustion, which is a
    /// refusal a caller cannot act on — it retries a table that has room.
    PolicyCollision,
    /// The generic host installed the pair, but the domain service could not
    /// create the state that arrival would address. The host retires the pair
    /// before it can be polled.
    PolicyUnavailable,
}

/// Admission refusal retaining custody of every not-yet-installed input.
#[derive(Debug)]
pub struct PeerAdmissionError {
    pub reason: PeerAdmissionRefusal,
    pub bundle: Option<RecyclablePeerBundle>,
    cleanup: Option<PeerResources>,
}

impl PeerAdmissionError {
    /// Retry teardown for a refusal that crossed the mapping boundary.
    /// Returns `true` once every local mapping/capability name is closed.
    pub fn retry_cleanup(&mut self) -> bool {
        if let Some(bundle) = self.bundle.as_mut() {
            let ring = bundle.ring.try_retire_outcome().is_closed();
            let kick = bundle.kick.try_retire_outcome().is_closed();
            let grants = bundle
                .grants
                .iter_mut()
                .flatten()
                .all(|owner| owner.try_retire_outcome().is_closed());
            let returned = bundle
                .return_recv
                .as_mut()
                .is_none_or(|owner| owner.try_retire_outcome().is_closed());
            if !(ring && kick && grants && returned) {
                return false;
            }
            self.bundle = None;
        }
        let Some(cleanup) = self.cleanup.as_mut() else {
            return true;
        };
        if cleanup.try_cleanup() {
            self.cleanup = None;
            true
        } else {
            false
        }
    }
}

/// The complete local ownership tree of one dynamically attached pair.
#[derive(Debug)]
struct PeerResources {
    mapping: Option<MappedLeaseView>,
    ring: Option<StashedLease>,
    endpoints: [Option<ReturnedCapability>; MAX_GRANTS + 2],
    scratch: [Option<ReturnedCapability>; MAX_GRANTS],
}

impl PeerResources {
    fn try_cleanup(&mut self) -> bool {
        if let Some(mapping) = self.mapping.as_mut() {
            if !mapping.try_retire_outcome().is_closed() {
                return false;
            }
            self.mapping = None;
        }
        for endpoint in &mut self.endpoints {
            if let Some(owner) = endpoint.as_mut() {
                if !owner.try_retire_outcome().is_closed() {
                    return false;
                }
                *endpoint = None;
            }
        }
        for scratch in &mut self.scratch {
            if let Some(owner) = scratch.as_mut() {
                if !owner.try_retire_outcome().is_closed() {
                    return false;
                }
                *scratch = None;
            }
        }
        if let Some(ring) = self.ring.as_mut() {
            if !ring.try_retire_outcome().is_closed() {
                return false;
            }
            self.ring = None;
        }
        true
    }
}

impl PeerId {
    const SLOT_BITS: u16 = u8::BITS as u16;
    /// The compact arrival representation dedicates one byte to the slot.
    pub const SLOT_CAPACITY: usize = 1 << Self::SLOT_BITS;
    const GENERATION_MASK: u16 = (1 << (16 - Self::SLOT_BITS)) - 1;

    /// Bounded host slot occupied by this peer.
    pub const fn slot(self) -> usize {
        self.slot as usize
    }

    /// Non-zero reuse generation for this peer.
    pub const fn generation(self) -> u16 {
        self.generation
    }

    /// Structural identity stamped into [`Serve::via`](crate::service::Serve::via).
    ///
    /// Static founder pairs retain the historical `0..MAX_SERVE_PAIRS`
    /// values. Recyclable peers always have a non-zero generation, so their
    /// encoded values cannot alias a static pair identity.
    pub const fn arrival(self) -> u16 {
        (self.generation << Self::SLOT_BITS) | self.slot as u16
    }

    /// Reconstitute a host-authored recyclable identity from its compact
    /// control-plane representation.
    ///
    /// This is not authority: users of the returned ticket must still present
    /// it through an authenticated administration pair, and the serving host
    /// revalidates both the exact slot generation and occupancy before doing
    /// lifecycle work. Static pair identities (generation zero) and out-of-
    /// range slots are refused here.
    pub const fn from_arrival(arrival: u16) -> Option<Self> {
        let slot = (arrival & ((1 << Self::SLOT_BITS) - 1)) as u8;
        let generation = (arrival >> Self::SLOT_BITS) & Self::GENERATION_MASK;
        if generation == 0 {
            return None;
        }
        Some(Self { slot, generation })
    }

    const fn via(self) -> u16 {
        self.arrival()
    }
}

/// **One serving pair's state** — the per-client half of the serve loop:
/// its ring halves, channel wiring, return-post machinery, and books.
/// Everything per-*service* (the bound service value, the transport) stays
/// in [`ServeHost`], which drives each pair against them — so a
/// multi-client service is N of these over one service, one §5.5 pipeline
/// implementation, and every request stamped with its arrival pair
/// ([`Serve::via`](crate::service::Serve::via) — the §5.8 structural
/// identity `res/DEVICE_SERVICE.md` §10 requires of a multi-client
/// broker).
pub struct ServePair<'w, S: SystemService> {
    rings: ServeRings<'w, S>,
    ep: ServeEndpoints,
    /// The identity requests off this pair are stamped with — the pair's
    /// index, fixed at attach and never wire data.
    endpoint: u16,
    stats: ServeStats,
    /// Whether the return post currently stands (the client's
    /// `pin_and_kick` discipline, service side): a delivery consumes it,
    /// and it is re-stood on a later cycle.
    return_standing: bool,
    /// Exact currently-stood Walking post, retained both for the executor's
    /// alternate park arm and for capability-delivery request synthesis.
    walking_post: Option<ParkPost>,
    /// Opt-in retained-return target.  Reserved before it is posted, then
    /// moved whole into the service when a capability lands; a fresh target
    /// is armed before the reply becomes visible.
    return_target: Option<ReturnedCapability>,
    /// Rejected deliveries whose revoke transiently refused. One entry owns
    /// both the live record and its allocator reservation; at most one is
    /// retried per request so a hostile registrar cannot turn cleanup into an
    /// unbounded hot-path loop.
    return_cleanup: [Option<ReturnedCapability>; RETURN_CLEANUP_CAP],
    return_cleanup_pending: usize,
    return_cleanup_cursor: usize,
    /// Lifecycle request emitted by the last completed serve step. The outer
    /// host takes it only after the data/grant reply path is complete.
    host_action: Option<crate::service::HostAction>,
}

impl<'w, S: SystemService> ServePair<'w, S> {
    fn request_ready(&mut self, svc: &mut S) -> bool {
        let Some(wire) = self.rings.req.peek() else {
            return false;
        };
        S::decode_req(&wire).is_none_or(|request| svc.request_ready(&request, self.endpoint))
    }
    /// A pair over its attached ring halves and channel wiring, stamped
    /// with `endpoint` as its structural identity.
    pub fn new(rings: ServeRings<'w, S>, ep: ServeEndpoints, endpoint: u16) -> Self {
        Self::with_return_target(rings, ep, endpoint, None)
    }

    /// Entry-path constructor carrying a retained-return target reserved
    /// before [`SystemService::start_on_metal`](crate::service::SystemService::start_on_metal).
    pub(crate) fn with_return_target(
        rings: ServeRings<'w, S>,
        ep: ServeEndpoints,
        endpoint: u16,
        return_target: Option<ReturnedCapability>,
    ) -> Self {
        Self {
            rings,
            ep,
            endpoint,
            stats: ServeStats::default(),
            return_standing: false,
            walking_post: None,
            return_target,
            return_cleanup: [const { None }; RETURN_CLEANUP_CAP],
            return_cleanup_pending: 0,
            return_cleanup_cursor: 0,
            host_action: None,
        }
    }

    /// Retire one rejected return or retain its affine owner for a later
    /// bounded retry. Filling the array while holding one more returned owner
    /// would require more live records than the task arena contains; the
    /// defensive fallback forgets the value (quarantining its reservation)
    /// rather than reopening a possibly-live slot.
    fn retire_or_defer_return(&mut self, mut returned: ReturnedCapability) {
        if returned.try_retire() {
            return;
        }
        if let Some(dst) = self.return_cleanup.iter_mut().find(|slot| slot.is_none()) {
            *dst = Some(returned);
            self.return_cleanup_pending += 1;
            self.stats.return_cleanup_deferred += 1;
            return;
        }
        core::mem::forget(returned);
    }

    /// Give one deferred owner one teardown attempt. This is deliberately one
    /// syscall at most per served request, independent of backlog size.
    fn retry_one_return_cleanup(&mut self) {
        if self.return_cleanup_pending == 0 {
            return;
        }
        for _ in 0..RETURN_CLEANUP_CAP {
            let index = self.return_cleanup_cursor;
            self.return_cleanup_cursor = (self.return_cleanup_cursor + 1) % RETURN_CLEANUP_CAP;
            let Some(returned) = self.return_cleanup[index].as_mut() else {
                continue;
            };
            if returned.try_retire() {
                self.return_cleanup[index] = None;
                self.return_cleanup_pending -= 1;
            }
            return;
        }
    }

    /// Ensure this pair's return receive stands. Retained-return services
    /// donate an allocator-owned target, Walking services select their next
    /// long-lived landing through the service hook, and ordinary services
    /// retain the fixed founder-published landing convention.
    ///
    /// A pair with **no return post** is lawful in two shapes: any pair of a
    /// non-rotating protocol, and a NON-ZERO pair of a rotating one — the
    /// landing rotation lives on pair 0 alone, and an extra pair founded
    /// post-less simply cannot receive transfers (its
    /// `advertised_return` is `None`, so an `Attach`-class request off it
    /// refuses through the protocol's own vocabulary). Only pair 0 of a
    /// Retained, RetainedAny, or Walking protocol founded post-less is a
    /// wiring fault.
    async fn ensure_return_post<T: Transport>(&mut self, svc: &mut S, transport: &T) -> bool {
        let walking = matches!(S::RETURN_MODE, ReturnMode::Walking);
        let retained = S::RETURN_MODE.is_retained();
        let Some((recv, fixed_land)) = self.ep.return_post else {
            return self.endpoint != 0 || !(retained || walking);
        };
        if !walking && self.return_standing {
            return true;
        }
        if retained && self.return_target.is_none() {
            self.return_target = ReturnedCapability::reserve();
            if self.return_target.is_none() {
                return false;
            }
        }
        let land = match S::RETURN_MODE {
            ReturnMode::Fixed => fixed_land,
            ReturnMode::Retained { .. } | ReturnMode::RetainedAny => {
                match self.return_target.as_ref() {
                    Some(target) => target.target(),
                    None => return false,
                }
            }
            // No free landing is a lawful saturated state. The reply that
            // filled the table still goes out; its client cannot transfer
            // another capability until service-owned teardown raises a new
            // pending target.
            ReturnMode::Walking => match svc.pending_return_post() {
                Some(target) => target,
                None => return true,
            },
        };
        // Walking posts are part of the preceding return's completion
        // contract: its success reply must not outrun the successor receive.
        // TCG can keep the channel in its Consuming handoff for longer than a
        // handful of microsecond deadlines, so give that peer-side transition
        // a bounded 6.4 ms convergence window before the outer repair loop.
        // Fixed/retained posts do not take this WouldBlock branch.
        for _ in 0..64 {
            let mut s = SessionBuilder::new();
            let posted = match S::RETURN_MODE {
                ReturnMode::Fixed | ReturnMode::Walking => s.post_recv(recv, land),
                ReturnMode::Retained {
                    kind,
                    minimum_rights,
                } => s.post_recv_typed(recv, land, kind, minimum_rights),
                ReturnMode::RetainedAny => s.post_recv(recv, land),
            };
            if posted.is_err() {
                return false;
            }
            if let Some(target) = self.return_target.as_mut() {
                // Arm before the transport boundary. A transport loss or OCC
                // result can leave the receive standing even when userland
                // cannot observe the commit outcome; Armed Drop therefore
                // quarantines this slot against a late return.
                target.arm_post();
            }
            let Ok(committed) = transport.commit(&s).await else {
                continue;
            };
            match committed.status() {
                Status::Ok if committed.failing_index() == 0 => {
                    if walking {
                        let Some((chan, recv_tag)) = crate::cap::verbs::recv_pin(recv) else {
                            #[cfg(target_os = "none")]
                            return false;
                            #[cfg(not(target_os = "none"))]
                            {
                                svc.return_post_stood();
                                return true;
                            }
                        };
                        self.walking_post = Some(ParkPost {
                            chan,
                            land,
                            recv_tag,
                        });
                        svc.return_post_stood();
                    } else {
                        self.return_standing = true;
                    }
                    return true;
                }
                // A walking post changes its landing slot after every
                // transfer. `WouldBlock` says only that this channel is busy;
                // it cannot prove that the requested successor already
                // stands. Acknowledging it would advance service metadata to
                // a slot the kernel never posted, so retry without calling
                // `return_post_stood`.
                Status::WouldBlock if committed.failing_index() == 0 && walking => {
                    // Delivery may wake the receiver before the sending task
                    // has finished the Board's consuming-to-empty handoff.
                    // Immediate retries all run in this same timeslice and
                    // cannot make that peer-side transition progress. Yield
                    // through the executor before selecting the same fresh
                    // landing again; unlike accepting WouldBlock, this never
                    // claims a post the kernel did not install.
                    crate::sleep_for(100_000).await;
                    continue;
                }
                Status::WouldBlock if committed.failing_index() == 0 => {
                    self.return_standing = true;
                    return true;
                }
                // An OCC race — retry; anything else is not convergent
                // here and the per-cycle repair keeps trying later.
                Status::Conflict if committed.failing_index() == 0 => continue,
                _ => return false,
            }
        }
        false
    }

    /// Serve exactly one pending request off this pair, if any — pop →
    /// the §5.5 pipeline against the shared service. `false` when the pair's
    /// request ring was empty.
    pub async fn serve_next<T: Transport>(&mut self, svc: &mut S, transport: &T) -> bool {
        match self
            .request_ready(svc)
            .then(|| self.rings.req.pop())
            .flatten()
        {
            Some(wire) => {
                self.serve_one(svc, transport, &wire).await;
                true
            }
            None if self.return_delivery_ready() => self.serve_return(svc, transport).await,
            None if matches!(S::RETURN_MODE, ReturnMode::Walking)
                && self.walking_post.is_none()
                && svc.pending_return_post().is_some() =>
            {
                let _ = self.ensure_return_post(svc, transport).await;
                false
            }
            None => false,
        }
    }

    /// Consume one walking-return landing as its implicit serve step —
    /// the arm both drains share: [`serve_next`](Self::serve_next) (the
    /// caller-owned drive; exec's loop) and [`ServeHost::run`] (the
    /// generic executor task, whose absence of this arm was the minted-
    /// face bring-up's silent hang: the landing sat unconsumed forever
    /// while every re-stood post refused over it).
    pub(crate) async fn serve_return<T: Transport>(&mut self, svc: &mut S, transport: &T) -> bool {
        if !self.return_delivery_ready() {
            return false;
        }
        let Some(req) = S::implicit_return_request() else {
            return false;
        };
        self.walking_post = None;
        self.retry_one_return_cleanup();
        // Walking returns raise their successor in the handler and
        // advertise no fixed landing, so there is no founder-published
        // slot for a refusal to strand.
        self.serve_request(svc, transport, req, None, None, false, false)
            .await;
        true
    }

    fn return_delivery_ready(&self) -> bool {
        matches!(S::RETURN_MODE, ReturnMode::Walking)
            && self.walking_post.is_some_and(|post| {
                matches!(
                    crate::cap::verbs::take_delivery(post.land),
                    Some(crate::cap::RawDelivery::Cap)
                )
            })
    }

    fn walking_post(&self) -> Option<ParkPost> {
        self.walking_post
    }

    fn take_host_action(&mut self) -> Option<crate::service::HostAction> {
        self.host_action.take()
    }

    /// One request through the pinned §5.5 pipeline.
    async fn serve_one<T: Transport>(&mut self, svc: &mut S, transport: &T, wire: &S::ReqWire) {
        debug_assert!(self.host_action.is_none());
        self.retry_one_return_cleanup();
        // 0 · The return post (the §5.5 reverse leg). A capability delivery
        // consumes the standing receive. Ordinary services retire the fixed
        // landing in their serve step; retained-return services move the
        // allocator-owned landing into their own lifecycle and arm a fresh
        // target before this request's reply becomes visible.
        let retained_return = S::RETURN_MODE.is_retained();
        let walking_return = matches!(S::RETURN_MODE, ReturnMode::Walking);
        let advertised_return = (!walking_return)
            .then(|| self.ep.return_post.map(|(_, land)| land))
            .flatten();
        let mut returned = None;
        // Whether THIS request arrived with a capability sitting in the
        // **fixed** landing — the founder-published slot every future
        // transfer on this pair reuses.  Carried into `serve_request` so
        // the host can prove the landing was vacated even when the
        // handler refused (see the invariant there).
        let mut delivered_fixed = false;
        if !walking_return && let Some((_, fixed_land)) = self.ep.return_post {
            let target = if retained_return {
                self.return_target.as_ref().map(ReturnedCapability::target)
            } else {
                Some(fixed_land)
            };
            if let Some(target) = target {
                match crate::cap::verbs::take_delivery(target) {
                    Some(crate::cap::RawDelivery::Cap) => {
                        self.return_standing = false;
                        delivered_fixed = !retained_return;
                        if retained_return && let Some(mut owner) = self.return_target.take() {
                            owner.activate();
                            returned = Some(owner);
                        }
                    }
                    Some(crate::cap::RawDelivery::Notice(..)) => {
                        // A return channel is capability-only. The post was
                        // nevertheless consumed; re-arm the still-reserved
                        // target and let the request fail through its own
                        // closed vocabulary.
                        self.return_standing = false;
                    }
                    None => {}
                }
            }
        }

        // Fail-closed decode: garbage on the ring is dropped + counted
        // (there is nothing lawful to reply to a record that does not
        // parse — the conforming client never sees this path).
        let Some(req) = S::decode_req(wire) else {
            self.stats.wire_refused += 1;
            if let Some(returned) = returned {
                self.retire_or_defer_return(returned);
            }
            let _ = self.ensure_return_post(svc, transport).await;
            return;
        };

        self.serve_request(
            svc,
            transport,
            req,
            returned,
            advertised_return,
            retained_return,
            delivered_fixed,
        )
        .await;
    }

    /// One decoded request through the common authority/grant/reply pipeline.
    async fn serve_request<T: Transport>(
        &mut self,
        svc: &mut S,
        transport: &T,
        req: S::Request,
        mut returned: Option<ReturnedCapability>,
        advertised_return: Option<SlotRef>,
        retained_return: bool,
        delivered_fixed: bool,
    ) {
        if retained_return && !self.ensure_return_post(svc, transport).await {
            // Never publish success while the return leg has no successor.
            // Refuse this request by withholding its delivered authority;
            // dropping it attempts revoke, then the freed reservation gets one
            // immediate chance to become the replacement post.
            if let Some(returned) = returned.take() {
                self.retire_or_defer_return(returned);
            }
            let _ = self.ensure_return_post(svc, transport).await;
        }

        // The sync authority join-point (#206) — unchanged, §2.2 — with
        // the arrival pair's structural stamp joined to the scope.
        let (mut out, returned) = step_with_return_async(
            svc,
            req,
            returned,
            advertised_return,
            self.endpoint,
            self.ep.grant_tx,
        )
        .await;
        if let Some(returned) = returned {
            self.retire_or_defer_return(returned);
        }

        // Fixed-return services make their landing vacant in `serve`; Walking
        // services raise their successor there. Re-arm in this same request,
        // before the reply/kick, so a direct handoff cannot resume the client
        // against a missing receive (L4 reply-and-wait ordering).
        //
        // **A refused request must never wedge its pair.**  Vacating the
        // fixed landing used to be a per-protocol obligation discharged
        // inside the handler (the output row's `attach` retires the
        // delivered name).  A handler that refuses BEFORE reaching that
        // code — a request for a terminal this machine does not have, a
        // decode that fails deeper in, an early error return — leaves the
        // client's exact-moved capability live in a slot the FOUNDER
        // published and every future transfer on this pair reuses.  The
        // next `post_recv` then refuses `BadArgs` against the kernel's
        // occupancy guard, forever: one refusal retires the whole pair,
        // and the client blocks on a reply that can never come.  So the
        // host proves the vacancy itself rather than trusting the
        // vocabulary (#578 slice 3 booted straight into this).
        if !retained_return {
            let mut vacated = true;
            if delivered_fixed
                && let Some(land) = advertised_return
                && matches!(
                    crate::cap::verbs::take_delivery(land),
                    Some(crate::cap::RawDelivery::Cap)
                )
            {
                // Still bound after the step ⇒ the handler did not consume
                // it.  `take_delivery` is a pure read, so this re-probe is
                // exact: a handler that DID take the name leaves the slot
                // vacant and this costs one load.
                vacated = crate::cap::verbs::retire_slot_outcome(land).is_closed();
                if !vacated {
                    // Retryable teardown: leave `return_standing` false so
                    // the per-cycle repair re-attempts both the retire and
                    // the post.  The reply still goes out.
                    self.stats.return_cleanup_deferred += 1;
                }
            }
            if vacated {
                let _ = self.ensure_return_post(svc, transport).await;
            }
        }

        // Effect the grants as ONE staged session (§5.5).
        let granted = out.granted();
        let mut effected = 0u8;
        if granted > 0 {
            match self.stage_grants(transport, &out).await {
                Ok(()) => {
                    out.release_transferred_grants();
                    effected = granted as u8;
                    if let Some(reply) = out.reply_mut() {
                        svc.grants_committed(reply, granted, self.endpoint);
                    }
                }
                Err(derived) => {
                    // The §5.5 violation path: drop (bounded revoke of
                    // anything derived) + flag, never block.  Then let
                    // the service undo policy it reserved before emitting
                    // the now-un-effected batch and make its reply honest.
                    self.stats.grants_dropped += 1;
                    drop_derived(transport, &derived).await;
                    if let Some(reply) = out.reply_mut() {
                        svc.grants_dropped(reply, granted);
                    }
                }
            }
        }

        // The reply push (+ the *effected* count — a flagged client
        // reads `grants = 0` and collects nothing) …
        let host_action = out.take_host_action();
        let Some(reply) = out.into_reply() else {
            // Every kore service replies; a reply-less step is a
            // service bug surfaced by its own tests — count and move on.
            self.stats.wire_refused += 1;
            self.host_action = host_action;
            return;
        };
        if self
            .rings
            .rep
            .push(S::encode_rep(&reply, effected))
            .is_err()
        {
            self.stats.replies_dropped += 1; // the client's problem (§5.5)
        } else {
            self.stats.served += 1;
        }

        // … and the notice-kick: its own session, droppable (§3.1).
        let tx = ChanTx::new(transport, self.ep.kick);
        let _ = tx
            .send(Notice {
                cookie: COOKIE_REPLY,
                payload: 0,
            })
            .await;
        self.host_action = host_action;
    }

    /// Stage + commit the grant session; on any non-`Ok` outcome yield
    /// the scratch slots that staged derives (for the bounded revoke).
    async fn stage_grants<T: Transport>(
        &mut self,
        transport: &T,
        out: &crate::service::Outcome<S::Response>,
    ) -> Result<(), [Option<SlotRef>; MAX_GRANTS]> {
        stage_pair_grants(transport, &self.ep.grant_tx, &self.ep.scratch, out).await
    }
}

/// Stage + commit one serve step's grant session over a pair's grant
/// lanes and derive scratch — the §5.5 pipeline's authority leg, shared
/// by the fixed pairs and the minted bank (one engine, instantiated
/// wider).  On any non-`Ok` outcome yields the scratch slots that staged
/// derives (for the bounded revoke).
async fn stage_pair_grants<T: Transport, R>(
    transport: &T,
    grant_tx: &[Option<SlotRef>; MAX_GRANTS],
    scratch: &[Option<SlotRef>; MAX_GRANTS],
    out: &crate::service::Outcome<R>,
) -> Result<(), [Option<SlotRef>; MAX_GRANTS]> {
    let mut s = SessionBuilder::new();
    let mut derived: [Option<SlotRef>; MAX_GRANTS] = [None; MAX_GRANTS];
    let mut staged_any = false;
    for (k, g) in out.grants().enumerate() {
        if matches!(g.arg, GrantArg::Direct) {
            continue;
        }
        let Some(tx) = grant_tx.get(k).copied().flatten() else {
            return Err(derived); // wiring short for this protocol
        };
        let staged = match g.arg {
            GrantArg::Direct => unreachable!("direct grants are filtered above"),
            // A plain attenuation: the transfer *is* the derive —
            // re-derivation into the peer's posted slot (§5.2).
            GrantArg::None => s.send(tx, g.subject, g.rights),
            GrantArg::Exact => s.send_exact(tx, g.subject),
            // A window carve: stage the derive into scratch, then
            // transfer the staged child — SSA in one session.  The
            // scratch carries `g.rights + DERIVE` (`stage_grant`'s
            // doing — the kernel's transfer gate demands it); the
            // **send** requests exactly `g.rights`, which is what the
            // peer ends up holding (the send arm clamps the moved cap
            // to `requested`).  Delivered rights are the grant's, never
            // the scratch's.
            //
            // A **device hand-out** takes the identical route
            // (`res/DEVICE_SERVICE.md` §10): the broker's inventory is
            // the subject, the per-device derive lands in scratch, and
            // the send moves it — a device grant is a carve off a
            // store, so it is the same two-step, not a new one.  A
            // **supply-lease mint** (`GrantArg::Lease`, the lessor's
            // region endowment) is the same two-step again, as are a
            // **sub-lease carve** and a **channel recv-end derive**
            // (`GrantArg::Sub`/`ChanRecv` — a minted loader face's
            // pieces, slice C): the fresh child stages in scratch,
            // the send moves it whole.
            GrantArg::Attenuate { .. }
            | GrantArg::Window { .. }
            | GrantArg::Sublease { .. }
            | GrantArg::Device { .. }
            | GrantArg::Lease { .. }
            | GrantArg::Sub { .. }
            | GrantArg::ChanRecv
            | GrantArg::ChanSend => {
                let Some(sc) = scratch.get(k).copied().flatten() else {
                    return Err(derived);
                };
                match stage_grant(&mut s, &g, sc) {
                    Ok(st) => {
                        derived[k] = Some(sc);
                        s.send(tx, st, g.rights)
                    }
                    Err(_) => return Err(derived),
                }
            }
        };
        if staged.is_err() {
            return Err(derived);
        }
        staged_any = true;
    }
    if !staged_any {
        return Ok(());
    }
    for _ in 0..GRANT_COMMIT_RETRIES {
        let Ok(committed) = transport.commit(&s).await else {
            return Err(derived);
        };
        match grant_commit_action(committed, s.staged()) {
            GrantCommitAction::Committed => return Ok(()),
            GrantCommitAction::Retry => continue,
            GrantCommitAction::Drop => return Err(derived),
        }
    }
    Err(derived)
}

/// The bounded revoke of a dropped grant session's derived children
/// (§5.5's "drop the grant").  Under kernel I3 the refused session
/// unwound and these slots are vacant — the revoke then refuses
/// harmlessly; under a partial apply it reaps what
/// landed.  Best-effort by design: the authority is either unwound,
/// reaped, or tag-dead — never silently live.
async fn drop_derived<T: Transport>(transport: &T, derived: &[Option<SlotRef>; MAX_GRANTS]) {
    let mut s = SessionBuilder::new();
    let mut any = false;
    for slot in derived.iter().flatten() {
        any = true;
        let _ = flows::revoke_batch(&mut s, &[*slot]);
    }
    if any {
        let _ = transport.commit(&s).await;
    }
}

/// Dynamically minted serving pairs one host can carry beside its fixed bank
/// (slice C — the multi-client seam). The ABI owns the geometry because every
/// service founder's supply and the generic runtime table must agree exactly.
/// Individual services impose their own smaller product bounds; this is the
/// shared host ceiling, not an executable-generation or task quota.
pub const MAX_MINTED_PAIRS: usize = kore_abi::task::MAX_MINTED_FACE_PAIRS;

/// A **dynamically minted serving pair** — [`ServePair`]'s lean sibling
/// for the runtime-attached bank ([`SystemService::pending_pair_attach`]).
///
/// Deliberately NOT a [`ServePair`]: that type's return-cleanup array is
/// priced for pairs that can receive transfers, and a minted pair never
/// can — it is post-less *by type* (no return machinery at all) and
/// optionally kickless (a
/// loader face's client spin-polls).  It drains through the same loop
/// and the same grant pipeline ([`stage_pair_grants`]) as the fixed
/// bank — one engine, instantiated wider, never a parallel path.
pub struct MintedPair<'w, S: SystemService> {
    rings: ServeRings<'w, S>,
    /// Grant-channel send ends, `[0..REPLY_GRANTS]` populated.
    grant_tx: [Option<SlotRef>; MAX_GRANTS],
    /// The shared derive scratch (cloned from pair 0 at attach — the
    /// serve loop is sequential, so one scratch band serves every pair).
    scratch: [Option<SlotRef>; MAX_GRANTS],
    /// Reply kick toward the pair's client, if it has one.
    kick: Option<SlotRef>,
    /// The identity requests off this pair are stamped with
    /// ([`Serve::via`](crate::service::Serve::via)) — service-chosen at
    /// mint, fixed at attach, never wire data.
    endpoint: u16,
    /// Receiver-authored authority proof required before each dequeue.
    admission: Option<crate::service::MintedAdmission>,
    /// One landed proof admits exactly one request.
    admitted: bool,
    stats: ServeStats,
}

impl<'w, S: SystemService> MintedPair<'w, S> {
    /// Ask the domain service whether the authority-bearing transaction for
    /// the request currently at this pair's head has landed.
    ///
    /// Fixed recyclable pairs already pass every peeked request through
    /// [`SystemService::request_ready`]. Runtime-minted pairs are the same
    /// protocol reached through a different bounded fan-in cell, so skipping
    /// the hook here would make an otherwise generic client face unusable by
    /// protocols whose ring publication precedes an atomic capability
    /// transaction (notably exec's invocation bundle).
    fn request_ready(&mut self, svc: &mut S) -> bool {
        let Some(wire) = self.rings.req.peek() else {
            return false;
        };
        S::decode_req(&wire).is_none_or(|request| svc.request_ready(&request, self.endpoint))
    }

    fn admission_ready(&mut self) -> bool {
        let Some(gate) = self.admission else {
            return true;
        };
        if self.admitted {
            return true;
        }
        let matches = match (gate.kind, crate::cap::verbs::take_delivery(gate.land)) {
            (
                crate::service::MintedAdmissionKind::Notice(expected),
                Some(crate::cap::RawDelivery::Notice(cookie, _)),
            ) => cookie == expected,
            (
                crate::service::MintedAdmissionKind::Capability,
                Some(crate::cap::RawDelivery::Cap),
            ) => true,
            _ => false,
        };
        if matches {
            self.admitted = true;
        }
        matches
    }

    /// One request through the post-less §5.5 pipeline: decode →
    /// serve step (no return leg to join — the pair cannot receive
    /// transfers) → grants → reply → optional kick.
    async fn serve_one<T: Transport>(&mut self, svc: &mut S, transport: &T, wire: &S::ReqWire) {
        let Some(req) = S::decode_req(wire) else {
            self.stats.wire_refused += 1;
            return;
        };
        let (mut out, returned) =
            step_with_return_async(svc, req, None, None, self.endpoint, self.grant_tx).await;
        debug_assert!(returned.is_none(), "a post-less pair delivered a return");
        let granted = out.granted();
        let mut effected = 0u8;
        if granted > 0 {
            match stage_pair_grants(transport, &self.grant_tx, &self.scratch, &out).await {
                Ok(()) => {
                    out.release_transferred_grants();
                    effected = granted as u8;
                    if let Some(reply) = out.reply_mut() {
                        svc.grants_committed(reply, granted, self.endpoint);
                    }
                }
                Err(derived) => {
                    self.stats.grants_dropped += 1;
                    drop_derived(transport, &derived).await;
                    if let Some(reply) = out.reply_mut() {
                        svc.grants_dropped(reply, granted);
                    }
                }
            }
        }
        let Some(reply) = out.into_reply() else {
            self.stats.wire_refused += 1;
            return;
        };
        if self
            .rings
            .rep
            .push(S::encode_rep(&reply, effected))
            .is_err()
        {
            self.stats.replies_dropped += 1;
        }
        self.stats.served += 1;
        if let Some(kick) = self.kick {
            let tx = ChanTx::new(transport, kick);
            let _ = tx
                .send(Notice {
                    cookie: COOKIE_REPLY,
                    payload: 0,
                })
                .await;
        }
    }
}

/// The generic serve host: owns the bound service, a bank of serving
/// pairs ([`ServePair`] — one for the ordinary single-client shape, up to
/// [`MAX_SERVE_PAIRS`] for a multi-client service such as the device
/// broker), and a runtime-attached bank of minted pairs
/// ([`MintedPair`] — slice C's per-child faces).  [`run`](Self::run) is
/// the executor task (§5.5: never a bespoke loop).
struct PeerTable<T, const N: usize> {
    inline: [T; N],
    #[cfg(feature = "heap")]
    dynamic: Option<alloc::vec::Vec<T, &'static (dyn core::alloc::Allocator + Sync)>>,
}
impl<T: Default, const N: usize> PeerTable<T, N> {
    fn new(inline: [T; N]) -> Self {
        Self {
            inline,
            #[cfg(feature = "heap")]
            dynamic: None,
        }
    }
    #[cfg(feature = "heap")]
    fn enable(&mut self, arena: &'static (dyn core::alloc::Allocator + Sync)) -> bool {
        if self.dynamic.is_some() {
            return true;
        }
        let mut values = alloc::vec::Vec::new_in(arena);
        if values.try_reserve_exact(N).is_err() {
            return false;
        }
        for value in &mut self.inline {
            values.push(core::mem::take(value));
        }
        self.dynamic = Some(values);
        true
    }
    fn reserve_one(&mut self) -> Option<()> {
        #[cfg(feature = "heap")]
        if let Some(values) = self.dynamic.as_mut() {
            return values.try_reserve(1).ok();
        }
        None
    }
    fn push_default(&mut self) {
        #[cfg(feature = "heap")]
        if let Some(values) = self.dynamic.as_mut() {
            values.push(T::default());
            return;
        }
        unreachable!("peer growth must reserve every table first")
    }
}
impl<T, const N: usize> core::ops::Deref for PeerTable<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        #[cfg(feature = "heap")]
        if let Some(values) = &self.dynamic {
            return values;
        }
        &self.inline
    }
}
impl<T, const N: usize> core::ops::DerefMut for PeerTable<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        #[cfg(feature = "heap")]
        if let Some(values) = &mut self.dynamic {
            return values;
        }
        &mut self.inline
    }
}
impl<'a, T, const N: usize> IntoIterator for &'a mut PeerTable<T, N> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
impl<'a, T, const N: usize> IntoIterator for &'a PeerTable<T, N> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct ServeHost<'w, 't, T: Transport, S: SystemService> {
    /// The one affine service owner. Services commonly carry large bounded
    /// arenas; keeping that state inline made async host transitions copy its
    /// complete footprint through the fixed user stack. This box is created
    /// once, before serving, and the request path remains allocation-free.
    svc: ServiceOwner<S>,
    transport: &'t T,
    pairs: PeerTable<Option<ServePair<'w, S>>, MAX_SERVE_PAIRS>,
    /// The runtime-attached bank — filled and drained through the
    /// [`SystemService::pending_pair_attach`] /
    /// [`pending_pair_detach`](SystemService::pending_pair_detach)
    /// handshake, served by the same loop.
    minted: PeerTable<Option<MintedPair<'w, S>>, MAX_MINTED_PAIRS>,
    /// Reuse generation for each static/dynamic serving cell. Zero denotes a
    /// founder-installed pair; recyclable admission advances to non-zero.
    generations: PeerTable<u16, MAX_SERVE_PAIRS>,
    /// Affine owners for dynamically installed pair mappings and endpoints.
    peer_resources: PeerTable<Option<PeerResources>, MAX_SERVE_PAIRS>,
    /// Terminal notification held until the corresponding affine mapping and
    /// endpoints are physically closed.  Keeping this separate from
    /// `generations` is what makes `SystemService::peer_closed` a teardown
    /// witness rather than merely a removal-from-polling notification.
    peer_closing: PeerTable<Option<(u16, crate::service::PeerClose)>, MAX_SERVE_PAIRS>,
    /// Refused admission whose resources still require bounded teardown.
    pending_admission_cleanup: Option<PeerAdmissionError>,
    /// Refusal withheld until [`pending_admission_cleanup`] is physically
    /// closed. Publishing it earlier would let the launcher reclaim the
    /// common backing while the host still owned a mapped view.
    pending_admission_refusal: Option<PeerAdmissionRefusal>,
    /// Round-robin cursor over the fixed protocol pairs. Without it, a
    /// continuously-ready pair zero can starve every later filesystem face.
    next_pair: usize,
}

impl<'w, 't, T: Transport, S: SystemService> ServeHost<'w, 't, T, S> {
    /// A host over a bound service, its attached ring halves, and the
    /// pair's wiring — the single-pair shape every ordinary service uses
    /// (pair 0, identity 0).
    pub fn new(svc: S, transport: &'t T, rings: ServeRings<'w, S>, ep: ServeEndpoints) -> Self {
        Self::new_owned_with_return_target(own_service(svc), transport, rings, ep, None)
    }

    /// Entry-path constructor carrying a retained-return target reserved
    /// before [`SystemService::start_on_metal`](crate::service::SystemService::start_on_metal).
    pub(crate) fn new_owned_with_return_target(
        svc: ServiceOwner<S>,
        transport: &'t T,
        rings: ServeRings<'w, S>,
        ep: ServeEndpoints,
        return_target: Option<ReturnedCapability>,
    ) -> Self {
        let mut pairs = [const { None }; MAX_SERVE_PAIRS];
        pairs[0] = Some(ServePair::with_return_target(rings, ep, 0, return_target));
        Self {
            svc,
            transport,
            pairs: PeerTable::new(pairs),
            minted: PeerTable::new([const { None }; MAX_MINTED_PAIRS]),
            generations: PeerTable::new([0; MAX_SERVE_PAIRS]),
            peer_resources: PeerTable::new([const { None }; MAX_SERVE_PAIRS]),
            peer_closing: PeerTable::new([None; MAX_SERVE_PAIRS]),
            pending_admission_cleanup: None,
            pending_admission_refusal: None,
            next_pair: 0,
        }
    }

    /// Opt into arena-backed peer population. Fixed-only services keep their
    /// allocation-free bootstrap table; only this owner can authorize growth.
    #[cfg(feature = "heap")]
    pub fn enable_peer_growth(
        &mut self,
        arena: &'static (dyn core::alloc::Allocator + Sync),
    ) -> bool {
        self.pairs.enable(arena)
            && self.generations.enable(arena)
            && self.peer_resources.enable(arena)
            && self.peer_closing.enable(arena)
    }

    #[cfg(feature = "heap")]
    pub fn enable_minted_growth(
        &mut self,
        arena: &'static (dyn core::alloc::Allocator + Sync),
    ) -> bool {
        self.minted.enable(arena)
    }

    fn free_peer_slot(&mut self) -> Option<usize> {
        if let Some(slot) = (0..self.pairs.len()).find(|&slot| {
            self.pairs[slot].is_none()
                && self.peer_resources[slot].is_none()
                && self.peer_closing[slot].is_none()
                && self.generations[slot] < PeerId::GENERATION_MASK
        }) {
            return Some(slot);
        }
        let slot = self.pairs.len();
        if slot == PeerId::SLOT_CAPACITY {
            return None;
        }
        self.pairs.reserve_one()?;
        self.generations.reserve_one()?;
        self.peer_resources.reserve_one()?;
        self.peer_closing.reserve_one()?;
        self.pairs.push_default();
        self.generations.push_default();
        self.peer_resources.push_default();
        self.peer_closing.push_default();
        Some(slot)
    }

    /// Drain the service's minted-pair handshakes: attach every pending
    /// spec (the bank permitting, identities unique), detach every
    /// retired identity.  Called after each served request, so a pair a
    /// serve step minted drains from its very next wake and a retired
    /// one stops immediately.
    fn settle_minted(&mut self) {
        while let Some(via) = self.svc.pending_pair_detach() {
            for slot in self.minted.iter_mut() {
                if slot.as_ref().is_some_and(|p| p.endpoint == via) {
                    *slot = None;
                }
            }
        }
        while let Some(spec) = self.svc.pending_pair_attach() {
            let via = spec.via;
            let collides = (via as usize) < MAX_SERVE_PAIRS
                || self.minted.iter().flatten().any(|p| p.endpoint == via);
            if !collides
                && self.minted.iter().all(Option::is_some)
                && self.minted.len() < u16::MAX as usize - MAX_SERVE_PAIRS
                && self.minted.reserve_one().is_some()
            {
                self.minted.push_default();
            }
            let free = self.minted.iter_mut().find(|s| s.is_none());
            let attached = if collides {
                false
            } else if let Some(slot) = free {
                let (base, len) = spec.window_raw();
                // SAFETY: `MintedPairSpec::new`'s contract — the minting
                // service warranted a live, exclusive, host-outliving
                // mapping of the pair window; this is the one place the
                // warrant is spent.
                let window: &'w mut [u8] =
                    unsafe { core::slice::from_raw_parts_mut(base.as_ptr(), len) };
                let control = super::window_bytes::<S>(S::REQUEST_CAPACITY, S::RESPONSE_CAPACITY);
                if window.len() < control {
                    self.svc.pair_attached(via, false);
                    continue;
                }
                let (window, _) = window.split_at_mut(control);
                match super::serve_rings::<S>(window, S::REQUEST_CAPACITY, S::RESPONSE_CAPACITY) {
                    Ok(rings) => {
                        let scratch = self.pairs[0]
                            .as_ref()
                            .map(|p| p.ep.scratch)
                            .unwrap_or([None; MAX_GRANTS]);
                        *slot = Some(MintedPair {
                            rings,
                            grant_tx: spec.grant_tx,
                            scratch,
                            kick: spec.kick,
                            endpoint: via,
                            admission: spec.admission,
                            admitted: false,
                            stats: ServeStats::default(),
                        });
                        true
                    }
                    // A window the service did not format refuses here
                    // rather than serving garbage.
                    Err(_) => false,
                }
            } else {
                false
            };
            self.svc.pair_attached(via, attached);
        }
    }

    /// Wire serving pair `index` (identity = the index — structural,
    /// fixed at attach).  `false` if `index` is 0 (the constructor's),
    /// out of range, or already occupied.
    ///
    /// Every pair that has a return leg owns an independent retained-return
    /// rotation. A pair may deliberately be post-less (the image service's
    /// static loader faces); returned-input requests on that face receive no
    /// authority and refuse through the service's typed vocabulary. Walking
    /// rotation remains pair 0-only because its landing is selected by the
    /// service's reply-and-wait state.
    pub fn attach_pair(
        &mut self,
        index: usize,
        rings: ServeRings<'w, S>,
        ep: ServeEndpoints,
    ) -> bool {
        self.attach_pair_with_return_target(index, rings, ep, None)
    }

    /// Founder-installed pair attachment carrying a retained-return target
    /// reserved before service startup. Public callers attach ordinary pairs
    /// through [`attach_pair`](Self::attach_pair); the entry ladder uses this
    /// form so a multi-pair service's return capacity cannot be consumed by
    /// its own startup allocations.
    pub(crate) fn attach_pair_with_return_target(
        &mut self,
        index: usize,
        rings: ServeRings<'w, S>,
        ep: ServeEndpoints,
        return_target: Option<ReturnedCapability>,
    ) -> bool {
        if index == 0 {
            return false;
        }
        match self.pairs.get_mut(index) {
            Some(slot @ None) => {
                *slot = Some(ServePair::with_return_target(
                    rings,
                    ep,
                    index as u16,
                    return_target,
                ));
                true
            }
            _ => false,
        }
    }

    /// Admit one peer into the first vacant bounded cell and return its
    /// generational identity.
    ///
    /// `None` is typed capacity backpressure: no partial pair is installed,
    /// and retiring any live recyclable peer makes admission possible again.
    /// The caller remains responsible for creating and authenticating the
    /// ring/endpoints before this final host attachment; the host derives
    /// request identity only from the returned cell/generation.
    pub async fn attach_recyclable(
        &mut self,
        rings: ServeRings<'w, S>,
        ep: ServeEndpoints,
    ) -> Option<PeerId> {
        let slot = self.free_peer_slot()?;
        let generation = self.generations[slot] + 1;
        let id = PeerId {
            slot: slot as u8,
            generation,
        };
        let mut pair = ServePair::new(rings, ep, id.via());
        // Dynamic peers are admitted after startup's fixed-pair return posts.
        // Publish the generation only after its own retained-return receive is
        // standing, otherwise the first capability reply can race a missing
        // post and be lost.
        if !pair.ensure_return_post(&mut self.svc, self.transport).await {
            return None;
        }
        self.pairs[slot] = Some(pair);
        self.generations[slot] = generation;
        Some(id)
    }

    /// Authenticate, map, attach, and own one complete recyclable pair.
    ///
    /// The ring is a writable held lease, the kick/grant endpoints are send
    /// endpoints, and the optional return endpoint is a receive endpoint.
    /// Their public wire fields never establish those facts: admission checks
    /// the kernel-authenticated records retained by each affine owner.
    pub async fn attach_owned_recyclable(
        &mut self,
        mut bundle: RecyclablePeerBundle,
        plan: PeerMapPlan,
    ) -> Result<PeerId, PeerAdmissionError> {
        use kore_abi::cap::{ResourceKind, Rights};

        let Some(slot) = self.free_peer_slot() else {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::Capacity,
                bundle: Some(bundle),
                cleanup: None,
            });
        };
        let required_bytes =
            super::window_bytes::<S>(S::REQUEST_CAPACITY, S::RESPONSE_CAPACITY) as u64;
        let Some(va) = plan.address(slot, required_bytes) else {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::InvalidMapPlan,
                bundle: Some(bundle),
                cleanup: None,
            });
        };
        let ring_rights = Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE;
        if !bundle.ring.is_lease_with(ring_rights) {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::WrongRingAuthority,
                bundle: Some(bundle),
                cleanup: None,
            });
        }
        if !bundle
            .kick
            .is_kind_with(ResourceKind::ChanSend, Rights::ACCESS)
        {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::WrongEndpointAuthority,
                bundle: Some(bundle),
                cleanup: None,
            });
        }
        let grant_count = bundle.grants.iter().flatten().count();
        if grant_count != S::REPLY_GRANTS
            || bundle.grants[..S::REPLY_GRANTS].iter().any(Option::is_none)
            || bundle.grants[S::REPLY_GRANTS..].iter().any(Option::is_some)
        {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::WrongGrantCount,
                bundle: Some(bundle),
                cleanup: None,
            });
        }
        if bundle
            .grants
            .iter()
            .flatten()
            .any(|endpoint| !endpoint.is_kind_with(ResourceKind::ChanSend, Rights::ACCESS))
        {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::WrongEndpointAuthority,
                bundle: Some(bundle),
                cleanup: None,
            });
        }
        let expects_return = S::RETURN_MODE.is_retained();
        if expects_return != bundle.return_recv.is_some() {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::WrongReturnShape,
                bundle: Some(bundle),
                cleanup: None,
            });
        }
        if bundle
            .return_recv
            .as_ref()
            .is_some_and(|endpoint| !endpoint.is_kind_with(ResourceKind::ChanRecv, Rights::ACCESS))
        {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::WrongEndpointAuthority,
                bundle: Some(bundle),
                cleanup: None,
            });
        }

        let ring = match bundle.ring.try_into_stashed_lease(ring_rights) {
            Ok(ring) => ring,
            Err(ring) => {
                bundle.ring = ring;
                return Err(PeerAdmissionError {
                    reason: PeerAdmissionRefusal::WrongRingAuthority,
                    bundle: Some(bundle),
                    cleanup: None,
                });
            }
        };
        let Some(ring_slot) = ring.slot() else {
            unreachable!("a freshly checked stashed lease is live")
        };
        let mut endpoints = [const { None }; MAX_GRANTS + 2];
        endpoints[0] = Some(bundle.kick);
        for (index, grant) in bundle.grants.into_iter().enumerate() {
            endpoints[index + 1] = grant;
        }
        endpoints[MAX_GRANTS + 1] = bundle.return_recv;
        let mut resources = PeerResources {
            mapping: None,
            ring: Some(ring),
            endpoints,
            scratch: [const { None }; MAX_GRANTS],
        };
        for index in 0..S::REPLY_GRANTS {
            let Some(scratch) = ReturnedCapability::reserve() else {
                return Err(PeerAdmissionError {
                    reason: PeerAdmissionRefusal::NoScratchSlots,
                    bundle: None,
                    cleanup: Some(resources),
                });
            };
            resources.scratch[index] = Some(scratch);
        }
        let Some((_, ring_bytes)) = crate::cap::verbs::lease_geometry(ring_slot) else {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::RingGeometry,
                bundle: None,
                cleanup: Some(resources),
            });
        };
        if ring_bytes < required_bytes || ring_bytes > plan.stride {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::RingGeometry,
                bundle: None,
                cleanup: Some(resources),
            });
        }
        let Some(mapping) = crate::cap::verbs::map_owned_lease_at_rw(ring_slot, va, ring_bytes)
        else {
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::MapFailed,
                bundle: None,
                cleanup: Some(resources),
            });
        };
        resources.mapping = Some(mapping);

        let window = unsafe {
            resources
                .mapping
                .as_mut()
                .expect("mapping was just installed")
                .as_mut()
        };
        // SAFETY: `mapping` owns this exact local VA until the peer is removed
        // from `pairs`; retirement always removes the pair before unmapping.
        let window: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(window.as_mut_ptr(), window.len()) };
        let (pair_window, aux) = window.split_at_mut(required_bytes as usize);
        let rings =
            match super::serve_rings::<S>(pair_window, S::REQUEST_CAPACITY, S::RESPONSE_CAPACITY) {
                Ok(rings) => rings,
                Err(_) => {
                    return Err(PeerAdmissionError {
                        reason: PeerAdmissionRefusal::RingMalformed,
                        bundle: None,
                        cleanup: Some(resources),
                    });
                }
            };

        let kick = resources.endpoints[0]
            .as_ref()
            .and_then(ReturnedCapability::slot)
            .unwrap();
        let mut grant_tx = [None; MAX_GRANTS];
        let mut scratch = [None; MAX_GRANTS];
        for index in 0..S::REPLY_GRANTS {
            grant_tx[index] = resources.endpoints[index + 1]
                .as_ref()
                .and_then(ReturnedCapability::slot);
            scratch[index] = resources.scratch[index]
                .as_ref()
                .map(ReturnedCapability::target);
        }
        let return_post = resources.endpoints[MAX_GRANTS + 1]
            .as_ref()
            .and_then(ReturnedCapability::slot)
            .map(|recv| (recv, SlotRef::from_wire(0xffff_0000 | slot as u32)));
        let ep = ServeEndpoints {
            kick,
            grant_tx,
            scratch,
            return_post,
        };
        let id = match self.attach_recyclable(rings, ep).await {
            Some(id) => id,
            None => {
                return Err(PeerAdmissionError {
                    reason: PeerAdmissionRefusal::WrongReturnShape,
                    bundle: None,
                    cleanup: Some(resources),
                });
            }
        };
        debug_assert_eq!(id.slot(), slot);
        self.peer_resources[slot] = Some(resources);
        if !self.svc.attach_peer_aux(id, aux) {
            // The host keeps any refused cleanup in this closing cell. The
            // auxiliary detach hook runs before the map can be retired.
            self.retire_peer(id, PeerClose::Protocol);
            return Err(PeerAdmissionError {
                reason: PeerAdmissionRefusal::RingMalformed,
                bundle: None,
                cleanup: None,
            });
        }
        Ok(id)
    }

    /// Retire exactly the peer named by `id`.
    ///
    /// The generation check occurs before any mutation, so a late close,
    /// death notice, or cancellation from an earlier occupant cannot retire
    /// its replacement. All terminal causes remove the pair first and then
    /// call the service's single [`SystemService::peer_closed`] cleanup hook.
    pub fn retire_peer(&mut self, id: PeerId, reason: PeerClose) -> bool {
        let slot = id.slot();
        if self.generations.get(slot).copied() != Some(id.generation)
            || self.pairs.get(slot).is_none_or(Option::is_none)
        {
            return false;
        }
        drop(self.pairs[slot].take());
        self.svc.peer_detaching(id);
        debug_assert!(self.peer_closing[slot].is_none());
        self.peer_closing[slot] = Some((id.via(), reason));
        if self.peer_resources[slot]
            .as_mut()
            .is_some_and(PeerResources::try_cleanup)
        {
            self.peer_resources[slot] = None;
        }
        self.finish_peer_close(slot);
        true
    }

    /// Publish the service cleanup edge only after every host-owned resource
    /// is gone.  The domain hook may detach session state and thereby let an
    /// authenticated admin client reclaim its local channel roots/backing, so
    /// calling it earlier would turn retryable teardown into a use-after-free.
    fn finish_peer_close(&mut self, slot: usize) -> bool {
        if self.peer_resources[slot].is_some() {
            return false;
        }
        let Some((via, reason)) = self.peer_closing[slot] else {
            return false;
        };
        if self.svc.peer_closed(via, reason) {
            self.peer_closing[slot] = None;
            true
        } else {
            false
        }
    }

    /// Give every deferred peer teardown one bounded retry. Returns the number
    /// of cells that reached complete closure in this pass.
    pub fn retry_peer_cleanup(&mut self) -> usize {
        let mut closed = 0;
        for slot in 0..self.pairs.len() {
            // A populated pair still owns and actively dereferences this
            // mapping. Only a cell whose pair has already been removed by
            // `retire_peer` is deferred teardown work. Retrying every resource
            // here would revoke a live peer immediately before polling it.
            if self.pairs[slot].is_none()
                && self.peer_resources[slot]
                    .as_mut()
                    .is_some_and(PeerResources::try_cleanup)
            {
                self.peer_resources[slot] = None;
            }
            closed += usize::from(
                self.pairs[slot].is_none()
                    && self.peer_resources[slot].is_none()
                    && self.finish_peer_close(slot),
            );
        }
        closed
    }

    /// Whether a caller-owned serving loop owes a teardown retry. An absent
    /// client cannot be relied on to send another doorbell after a refusal.
    /// This reports work only; custody and cleanup remain owned by this host.
    pub fn has_pending_cleanup(&self) -> bool {
        self.pending_admission_cleanup.is_some()
            || self.peer_closing.iter().any(Option::is_some)
            || self
                .pairs
                .iter()
                .flatten()
                .any(|pair| pair.return_cleanup_pending != 0)
    }

    fn retry_admission_cleanup(&mut self) -> bool {
        let Some(cleanup) = self.pending_admission_cleanup.as_mut() else {
            return true;
        };
        if cleanup.retry_cleanup() {
            self.pending_admission_cleanup = None;
            if let Some(reason) = self.pending_admission_refusal.take() {
                let _ = self.svc.peer_admitted(Err(reason));
            }
            true
        } else {
            false
        }
    }

    async fn handle_host_action(&mut self, source: usize, action: crate::service::HostAction) {
        match action {
            crate::service::HostAction::AdmitPeer { bundle, map } => {
                match self.attach_owned_recyclable(bundle, map).await {
                    Ok(peer) => {
                        if !self.svc.peer_admitted(Ok(peer)) {
                            // Admission crossed the mapping boundary but the
                            // domain service could not create the state this
                            // peer would address.  Retire it before it can
                            // issue a request; capacity is not leaked and no
                            // structurally live pair is left policy-less.
                            let _ = self.retire_peer(peer, crate::service::PeerClose::Protocol);
                        }
                    }
                    Err(mut error) => {
                        let reason = error.reason;
                        if error.retry_cleanup() {
                            let _ = self.svc.peer_admitted(Err(reason));
                        } else {
                            debug_assert!(self.pending_admission_cleanup.is_none());
                            debug_assert!(self.pending_admission_refusal.is_none());
                            self.pending_admission_cleanup = Some(error);
                            self.pending_admission_refusal = Some(reason);
                        }
                    }
                }
            }
            crate::service::HostAction::ClosePeer(reason) => {
                let generation = self.generations.get(source).copied().unwrap_or(0);
                if generation != 0 {
                    let _ = self.retire_peer(
                        PeerId {
                            slot: source as u8,
                            generation,
                        },
                        reason,
                    );
                }
            }
            crate::service::HostAction::RetirePeer { peer, reason } => {
                let _ = self.retire_peer(peer, reason);
            }
        }
    }

    /// Whether `id` still names the live occupant of its cell.
    pub fn peer_is_live(&self, id: PeerId) -> bool {
        self.generations.get(id.slot()).copied() == Some(id.generation)
            && self.pairs.get(id.slot()).is_some_and(Option::is_some)
    }

    /// Notify one exact recyclable generation through the endpoint the host
    /// owns. A service never copies an affine endpoint out of its peer bundle.
    pub async fn notify_peer(&self, id: PeerId, notice: Notice) -> bool {
        if !self.peer_is_live(id) {
            return false;
        }
        let pair = self.pairs[id.slot()].as_ref().expect("live generation");
        ChanTx::new(self.transport, pair.ep.kick)
            .send(notice)
            .await
            .is_ok()
    }

    /// Notify one founder-staged pair's client through the endpoint the
    /// host owns — the static-pair sibling of [`notify_peer`](Self::notify_peer),
    /// for services whose fixed control faces carry level-triggered state
    /// (a compositor's controller, say) that the client drains on a wake.
    pub async fn notify_pair(&self, pair: usize, notice: Notice) -> bool {
        let Some(kick) = self
            .pairs
            .get(pair)
            .and_then(Option::as_ref)
            .map(|pair| pair.ep.kick)
        else {
            return false;
        };
        ChanTx::new(self.transport, kick).send(notice).await.is_ok()
    }

    /// The books so far — summed across pairs (each counter is a
    /// monotone event count, so the sum is the service's total).
    pub fn stats(&self) -> ServeStats {
        let mut total = ServeStats::default();
        for pair in self.pairs.iter().flatten() {
            total.served += pair.stats.served;
            total.wire_refused += pair.stats.wire_refused;
            total.grants_dropped += pair.stats.grants_dropped;
            total.replies_dropped += pair.stats.replies_dropped;
            total.return_cleanup_deferred += pair.stats.return_cleanup_deferred;
        }
        total
    }

    /// Stand every pair's return post for the first time — called
    /// **before** the readiness beacon (`serve_forever`), so a client that
    /// sees `COOKIE_SERVING` can never win a return send against an
    /// un-stood post (the board law: a send with no visible post refuses
    /// `WouldBlock`, and under cooperative scheduling a tight client
    /// retry loop can starve the service out of ever posting).  The
    /// per-cycle repair in `serve_one` re-stands each after a consumed
    /// delivery.
    pub async fn stand_return_post(&mut self) -> bool {
        let Self {
            svc,
            transport,
            pairs,
            ..
        } = self;
        for pair in pairs.iter_mut().flatten() {
            if !pair.ensure_return_post(svc, *transport).await {
                return false;
            }
        }
        true
    }

    /// The bound service (read access — `summarize` and friends).
    pub fn service(&self) -> &S {
        &self.svc
    }

    /// The bound service (mutable) — a caller-owned drive loop's device
    /// half (a deadline-driven backend drain folded into the same
    /// executor root as [`serve_next`](Self::serve_next)) reaches the
    /// service through this.
    pub fn service_mut(&mut self) -> &mut S {
        &mut self.svc
    }

    /// The currently-stood walking return post, if pair zero owns one. A
    /// caller-owned serve loop may fold it into its next executor park.
    pub fn walking_post(&self) -> Option<ParkPost> {
        self.pairs[0].as_ref().and_then(ServePair::walking_post)
    }

    /// Whether pair zero still owes its next Walking return post.
    ///
    /// A completed rendezvous can wake this task before the sender has
    /// published the Board's final `Consuming -> Empty` edge. In that short
    /// interval re-posting lawfully returns `WouldBlock`. If the bounded
    /// in-call retries all observe that interval, the serve loop must use a
    /// deadline rather than park forever: no post stands, so a prospective
    /// sender cannot provide the next wake itself.
    pub fn walking_post_pending(&self) -> bool {
        matches!(S::RETURN_MODE, ReturnMode::Walking)
            && self.pairs[0]
                .as_ref()
                .is_some_and(|pair| pair.walking_post().is_none())
            && self.svc.pending_return_post().is_some()
    }

    /// Record pair zero's completed walking return after
    /// [`crate::executor::WakeReason::Delivery`].
    ///
    /// This is deliberately separate from [`serve_next`](Self::serve_next):
    /// request-ring readiness is level-triggered data, while capability
    /// delivery is ready only after the executor observes the kernel's
    /// rendezvous-completion edge.
    pub async fn serve_completed_return(&mut self) -> bool {
        let Some(pair) = self.pairs[0].as_mut() else {
            return false;
        };
        pair.serve_return(&mut self.svc, self.transport).await
    }

    /// Wake each fixed client whose auxiliary event ring received a batch.
    /// One notice per pair, irrespective of the number of records: rings are
    /// level-triggered and the client drains them whole.
    pub async fn notify_pairs(&self, pairs: u16) {
        for (k, pair) in self.pairs.iter().take(u16::BITS as usize).enumerate() {
            if pairs & (1u16 << k) == 0 {
                continue;
            }
            let Some(kick) = pair.as_ref().map(|p| p.ep.kick) else {
                continue;
            };
            let _ = ChanTx::new(self.transport, kick)
                .send(Notice {
                    cookie: COOKIE_REPORT,
                    payload: 0,
                })
                .await;
        }
    }

    /// Deadline-aware sibling of the ordinary ring-only drive.  It performs
    /// the same total request drain, but folds the service's current minimum
    /// into the executor's existing `WAIT_DELIVERY | WAIT_SLEEP` park.  There
    /// is still one kernel timer per task, never one comparator per alarm.
    pub async fn run_deadline_driven(mut self) -> ! {
        loop {
            // Establish the service's monotonic anchor before serving a
            // relative arm request in this pass.
            let now = crate::executor::clock_ns().await;
            let notify = self.svc.drive_deadlines(now);
            self.notify_pairs(notify).await;

            while self.serve_next().await {}

            // A just-served arm may already be due. Settle it before choosing
            // the next minimum, so an expired request never becomes a hot
            // already-past park loop.
            let now = crate::executor::clock_ns().await;
            let notify = self.svc.drive_deadlines(now);
            self.notify_pairs(notify).await;

            match self.svc.next_deadline() {
                Some(deadline) => {
                    let _ = crate::executor::next_wake_until(deadline).await;
                }
                None => {
                    let _ = crate::executor::next_wake().await;
                }
            }
        }
    }

    /// Serve forever — the §5.5 executor task: level-triggered over every
    /// pair's request ring (a client's doorbell kick wakes the executor;
    /// the §3.1 total drain re-polls the fan-in — the
    /// [`Consumer::next`] idiom, fanned in).
    ///
    /// [`Consumer::next`]: kore_ring::spsc::Consumer::next
    pub async fn run(mut self) -> ! {
        /// One wake's unit of work: a protocol request off pair `k` or a
        /// minted pair's request.
        enum Work<W> {
            Pair(usize, W),
            Minted(usize, W),
            /// A walking-return landing on pair `k` awaits its implicit
            /// step (`SvcProtocol::implicit_return_request`).
            Return(usize),
            /// The kernel asked this row to quiesce its devices (#809 §L);
            /// the payload is the remaining budget in nanoseconds.
            Quiesce(u64),
        }
        loop {
            self.retry_peer_cleanup();
            while !self.retry_admission_cleanup() {
                crate::sleep_for(1_000).await;
            }
            let host_action = {
                let Self {
                    svc,
                    transport,
                    pairs,
                    minted,
                    ..
                } = &mut self;
                let work = core::future::poll_fn(|cx| {
                    // The terminal ask first: it arrives as a doorbell wake
                    // with no ring work, so it must be observed inside the
                    // poll or the park would swallow it.
                    if let Some(budget_ns) = crate::executor::quiesce_asked(cx) {
                        return core::task::Poll::Ready(Work::Quiesce(budget_ns));
                    }
                    for (k, pair) in pairs.iter_mut().enumerate() {
                        if let Some(w) = pair
                            .as_mut()
                            .and_then(|p| p.request_ready(svc).then(|| p.rings.req.pop()).flatten())
                        {
                            return core::task::Poll::Ready(Work::Pair(k, w));
                        }
                    }
                    for (k, pair) in minted.iter_mut().enumerate() {
                        if let Some(pair) = pair.as_mut()
                            && pair.admission_ready()
                            && pair.request_ready(svc)
                            && let Some(w) = pair.rings.req.pop()
                        {
                            return core::task::Poll::Ready(Work::Minted(k, w));
                        }
                    }
                    // The walking return plane, request-less by design: a
                    // conveyed capability landed on a stood post and its
                    // implicit step is the work.
                    for (k, pair) in pairs.iter_mut().enumerate() {
                        if pair.as_mut().is_some_and(|p| p.return_delivery_ready()) {
                            return core::task::Poll::Ready(Work::Return(k));
                        }
                    }
                    core::task::Poll::Pending
                })
                .await;
                match work {
                    Work::Pair(k, wire) => {
                        if let Some(pair) = pairs[k].as_mut() {
                            pair.serve_one(svc, *transport, &wire).await;
                            pair.take_host_action().map(|action| (k, action))
                        } else {
                            None
                        }
                    }
                    Work::Minted(k, wire) => {
                        if let Some(pair) = minted[k].as_mut() {
                            pair.serve_one(svc, *transport, &wire).await;
                            pair.admitted = false;
                        }
                        None
                    }
                    Work::Return(k) => {
                        if let Some(pair) = pairs[k].as_mut() {
                            let _ = pair.serve_return(svc, *transport).await;
                        }
                        None
                    }
                    // The row's terminal sequence: the kernel observes
                    // completion at the device epoch, so the sweep's record
                    // stays local (the row keeps serving its terminal face).
                    Work::Quiesce(budget_ns) => {
                        let _ = svc
                            .quiesce_devices(crate::executor::quiesce_deadline(budget_ns))
                            .await;
                        None
                    }
                }
            };
            if let Some((source, action)) = host_action {
                self.handle_host_action(source, action).await;
            }
            // The minted-pair handshake, drained after every unit of
            // work: a pair a serve step just minted is attached before
            // the next wake, and a retired one stops now.
            self.settle_minted();
        }
    }

    /// Serve exactly one pending request from the first ready pair, if
    /// any — the **bounded drive** (host tests; a caller-owned drain):
    /// pop → the §5.5 pipeline.  `false` when every request ring was
    /// empty.  The production loop is [`run`](Self::run).
    pub async fn serve_next(&mut self) -> bool {
        self.retry_peer_cleanup();
        if !self.retry_admission_cleanup() {
            return false;
        }
        let mut served = false;
        for step in 0..self.pairs.len() {
            let source = (self.next_pair + step) % self.pairs.len();
            let did_serve = if let Some(pair) = self.pairs[source].as_mut() {
                pair.serve_next(&mut self.svc, self.transport).await
            } else {
                false
            };
            if did_serve {
                let action = self.pairs[source]
                    .as_mut()
                    .and_then(ServePair::take_host_action);
                if let Some(action) = action {
                    self.handle_host_action(source, action).await;
                }
                self.next_pair = (source + 1) % self.pairs.len();
                served = true;
                break;
            }
        }
        if !served {
            'minted: for pair in self.minted.iter_mut().flatten() {
                if pair.admission_ready()
                    && pair.request_ready(&mut self.svc)
                    && let Some(wire) = pair.rings.req.pop()
                {
                    pair.serve_one(&mut self.svc, self.transport, &wire).await;
                    pair.admitted = false;
                    served = true;
                    break 'minted;
                }
            }
        }
        // The handshake drains even on an idle pass, so a host test can
        // attach a freshly minted pair without a request in flight.
        self.settle_minted();
        served
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use core::convert::Infallible;

    use kore_abi::cap::Rights;
    use kore_abi::task::{Bootstrap, Profile};

    use crate::service::{BindError, Bundle, GrantArg, Serve, SystemService};
    use crate::svc::{REP_CAP, REQ_CAP, SvcProtocol, split_window, window_bytes};
    use crate::transport::TrapTransport;

    /// `retire_incomplete` is the protocol boundary between bundles.  An
    /// asker that spends its bounded admission re-ask and gives up leaves
    /// its accepted components behind; without retirement the next launch's
    /// matching part answers `Duplicate` forever, the bundle never
    /// completes, and no admission verdict is ever published.  A new `Ring`
    /// proves the abandonment — so the leftovers are retired, and while any
    /// of them is still closing the replacement must be refused rather than
    /// assembled beside a live owner.
    ///
    /// Host coverage stops at the assembler's own state.  Seating the
    /// replacement cannot be exercised off metal: `is_lease_with` and
    /// `is_kind_with` are `false` for `not(target_os = "none")`, so
    /// `accepts` always answers `WrongAuthority` and `push` can never seat a
    /// component here.  What this proves is the decision the three services
    /// key on — whether the cell is clear for the replacement, or still owes
    /// a close.
    #[test]
    fn a_new_ring_retires_an_abandoned_partial_bundle() {
        let mut assembler = PeerBundleAssembler::new(0, true);

        // Nothing was abandoned: the services' `!is_empty()` guard never
        // fires, and retirement of an empty cell reports it clear.
        assert!(assembler.is_empty());
        assert!(assembler.retire_incomplete());

        // An asker accepted a kick and a return endpoint, then gave up.
        assembler.kick = Some(ReturnedCapability::test_live(SlotRef::from_wire(11)));
        assembler.return_recv = Some(ReturnedCapability::test_live(SlotRef::from_wire(12)));
        assert!(!assembler.is_empty());

        // Those owners have not closed, so the replacement is refused and
        // every leftover stays owned for a later attempt.  This is the
        // `Pending`/`Conflict`/`Busy` arm in mm, dagfs, power and accel.
        assert!(!assembler.retire_incomplete());
        assert!(assembler.kick.is_some());
        assert!(assembler.return_recv.is_some());

        // A partial close releases what settled and keeps the rest, so the
        // retry never re-retires an owner that is already gone.
        assembler.kick = Some(ReturnedCapability::test_dead_lineage(SlotRef::from_wire(11)));
        assert!(!assembler.retire_incomplete());
        assert!(assembler.kick.is_none());
        assert!(assembler.return_recv.is_some());

        // Once the last lineage settles the cell is clear again — the state
        // a replacement bundle needs, and the one that lets the service
        // admit this `Ring` instead of refusing it.
        assembler.return_recv = Some(ReturnedCapability::test_dead_lineage(SlotRef::from_wire(12)));
        assert!(assembler.retire_incomplete());
        assert!(assembler.is_empty());
        assert!(assembler.return_recv.is_none());
    }

    struct NoCaps;

    impl Bundle for NoCaps {
        const PROFILE: Profile = Profile::new("grant-drop-test", &[]);

        fn resolve(_: &Bootstrap<'_>) -> Result<Self, BindError> {
            Ok(Self)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Reply {
        Granted,
        Denied,
    }

    /// A service that reserves one local fact before emitting authority —
    /// the same lifecycle shape as the device broker's claim ledger.
    struct DropAware {
        reserved: bool,
        repaired: bool,
        committed: Option<u16>,
    }

    impl SystemService for DropAware {
        const NAME: &'static str = "grant-drop-test";
        type Endowment = NoCaps;

        fn bind(_: NoCaps) -> Self {
            Self {
                reserved: false,
                repaired: false,
                committed: None,
            }
        }

        fn serve(&mut self, _: u64, sv: &mut Serve<'_, Reply>) {
            self.reserved = true;
            sv.grant(SlotRef::from_wire(7), Rights::ACCESS, GrantArg::None);
            sv.reply(Reply::Granted);
        }

        fn grants_committed(&mut self, response: &Reply, emitted: usize, via: u16) {
            assert_eq!(*response, Reply::Granted);
            assert_eq!(emitted, 1);
            assert!(self.reserved && !self.repaired);
            assert_eq!(
                self.committed.replace(via),
                None,
                "one success notification"
            );
        }

        fn grants_dropped(&mut self, response: &mut Reply, emitted: usize) {
            assert_eq!(emitted, 1);
            assert!(self.reserved);
            self.reserved = false;
            self.repaired = true;
            *response = Reply::Denied;
        }
    }

    impl SvcProtocol for DropAware {
        type Request = u64;
        type Response = Reply;
        type ReqWire = u64;
        type RepWire = u64;

        const REPLY_GRANTS: usize = 1;

        fn encode_req(req: &u64) -> u64 {
            *req
        }

        fn decode_req(wire: &u64) -> Option<u64> {
            Some(*wire)
        }

        fn encode_rep(rep: &Reply, grants: u8) -> u64 {
            let tag = match rep {
                Reply::Granted => 1,
                Reply::Denied => 2,
            };
            (tag << 8) | u64::from(grants)
        }

        fn decode_rep(wire: &u64) -> Option<(Reply, u8)> {
            let rep = match wire >> 8 {
                1 => Reply::Granted,
                2 => Reply::Denied,
                _ => return None,
            };
            Some((rep, *wire as u8))
        }
    }

    /// A composed service whose synchronous seam is deliberately a refusal.
    /// The host must await the override before publishing the reply.
    struct AsyncAware {
        sync_calls: u8,
        async_calls: u8,
    }

    impl SystemService for AsyncAware {
        const NAME: &'static str = "async-aware-test";
        type Endowment = NoCaps;

        fn bind(_: NoCaps) -> Self {
            Self {
                sync_calls: 0,
                async_calls: 0,
            }
        }

        fn serve(&mut self, _: u64, sv: &mut Serve<'_, Reply>) {
            self.sync_calls += 1;
            sv.reply(Reply::Denied);
        }

        async fn serve_async(&mut self, _: u64, sv: &mut Serve<'_, Reply>) {
            // Force one suspension so the test cannot pass through a
            // synchronous-before-return implementation by accident.
            let mut yielded = false;
            core::future::poll_fn(|cx| {
                if core::mem::replace(&mut yielded, true) {
                    core::task::Poll::Ready(())
                } else {
                    cx.waker().wake_by_ref();
                    core::task::Poll::Pending
                }
            })
            .await;
            self.async_calls += 1;
            sv.reply(Reply::Granted);
        }
    }

    impl SvcProtocol for AsyncAware {
        type Request = u64;
        type Response = Reply;
        type ReqWire = u64;
        type RepWire = u64;

        const REPLY_GRANTS: usize = 0;

        fn encode_req(req: &u64) -> u64 {
            *req
        }

        fn decode_req(wire: &u64) -> Option<u64> {
            Some(*wire)
        }

        fn encode_rep(rep: &Reply, grants: u8) -> u64 {
            DropAware::encode_rep(rep, grants)
        }

        fn decode_rep(wire: &u64) -> Option<(Reply, u8)> {
            DropAware::decode_rep(wire)
        }
    }

    /// A service-selected return landing whose successor must stand before
    /// each reply becomes visible.
    struct Walking {
        pending: Option<SlotRef>,
        stood: u8,
        served: u8,
    }

    impl SystemService for Walking {
        const NAME: &'static str = "walking-return-test";
        type Endowment = NoCaps;
        const RETURN_MODE: ReturnMode = ReturnMode::Walking;

        fn bind(_: NoCaps) -> Self {
            Self {
                pending: Some(SlotRef::from_wire(71)),
                stood: 0,
                served: 0,
            }
        }

        fn serve(&mut self, _: u64, sv: &mut Serve<'_, Reply>) {
            assert!(
                self.pending.is_none(),
                "the request arrived on a stood post"
            );
            self.served += 1;
            self.pending = Some(SlotRef::from_wire(71 + u32::from(self.served)));
            sv.reply(Reply::Granted);
        }

        fn pending_return_post(&self) -> Option<SlotRef> {
            self.pending
        }

        fn return_post_stood(&mut self) {
            assert!(
                self.pending.take().is_some(),
                "an acknowledgement names a pending post"
            );
            self.stood += 1;
        }
    }

    impl SvcProtocol for Walking {
        type Request = u64;
        type Response = Reply;
        type ReqWire = u64;
        type RepWire = u64;

        const REPLY_GRANTS: usize = 0;

        fn encode_req(req: &u64) -> u64 {
            *req
        }

        fn decode_req(wire: &u64) -> Option<u64> {
            Some(*wire)
        }

        fn encode_rep(rep: &Reply, grants: u8) -> u64 {
            DropAware::encode_rep(rep, grants)
        }

        fn decode_rep(wire: &u64) -> Option<(Reply, u8)> {
            DropAware::decode_rep(wire)
        }

        fn implicit_return_request() -> Option<u64> {
            Some(0)
        }

        fn is_implicit_return_request(request: &u64) -> bool {
            *request == 0
        }
    }

    /// A truthful synchronous transport for host-testing serve-pipeline
    /// ordering. It models only successful commits, not kernel behavior.
    struct OkTransport {
        commits: Cell<u8>,
    }

    impl OkTransport {
        const fn new() -> Self {
            Self {
                commits: Cell::new(0),
            }
        }
    }

    impl Transport for OkTransport {
        type Error = Infallible;

        async fn commit(&self, session: &SessionBuilder) -> Result<Committed, Self::Error> {
            assert!(
                session.staged() > 0,
                "the host never commits an empty session"
            );
            self.commits.set(self.commits.get() + 1);
            Ok(Committed::from_raw(Status::Ok.encode(0)))
        }

        fn commit_now(&self, session: &SessionBuilder) -> Result<Committed, Self::Error> {
            assert!(
                session.staged() > 0,
                "the host never commits an empty session"
            );
            self.commits.set(self.commits.get() + 1);
            Ok(Committed::from_raw(Status::Ok.encode(0)))
        }
    }

    /// Drive a future whose transport calls all resolve immediately.  The
    /// host-side `TrapTransport` is the runtime's honest typed refusal, not a
    /// simulated kernel.
    fn drive<F: core::future::Future>(fut: F) -> F::Output {
        crate::block::try_block_on(fut, 8).expect("future did not resolve without an external wake")
    }

    fn window() -> &'static mut [u8] {
        let bytes = window_bytes::<DropAware>(REQ_CAP, REP_CAP);
        let layout = std::alloc::Layout::from_size_align(bytes, 64).expect("window layout");
        // SAFETY: the test intentionally leaks one correctly aligned allocation
        // for the ring halves' shared lifetime.
        unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            assert!(!ptr.is_null());
            core::slice::from_raw_parts_mut(ptr, bytes)
        }
    }

    fn walking_window() -> &'static mut [u8] {
        let bytes = window_bytes::<Walking>(REQ_CAP, REP_CAP);
        let layout = std::alloc::Layout::from_size_align(bytes, 64).expect("window layout");
        // SAFETY: the test leaks one aligned allocation for the split ring's
        // shared lifetime, exactly as `window()` above.
        unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            assert!(!ptr.is_null());
            core::slice::from_raw_parts_mut(ptr, bytes)
        }
    }

    fn async_window() -> &'static mut [u8] {
        let bytes = window_bytes::<AsyncAware>(REQ_CAP, REP_CAP);
        let layout = std::alloc::Layout::from_size_align(bytes, 64).expect("window layout");
        // SAFETY: the test leaks one aligned allocation for the split ring's
        // shared lifetime, exactly as `window()` above.
        unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            assert!(!ptr.is_null());
            core::slice::from_raw_parts_mut(ptr, bytes)
        }
    }

    /// The host publishes no reply before the service's async override
    /// settles: the forced suspension must be driven, the sync seam must
    /// never run, and the awaited reply is the one that lands on the ring.
    #[test]
    fn serve_host_awaits_the_async_service_override() {
        let (mut client, rings) =
            split_window::<AsyncAware>(async_window(), REQ_CAP, REP_CAP).expect("async rings");
        let endpoints = ServeEndpoints {
            kick: SlotRef::from_wire(60),
            grant_tx: [None; MAX_GRANTS],
            scratch: [None; MAX_GRANTS],
            return_post: None,
        };
        let transport = OkTransport::new();
        let mut host = ServeHost::new(AsyncAware::bind(NoCaps), &transport, rings, endpoints);

        client.req.push(7).expect("request fits");
        assert!(drive(host.serve_next()));
        assert_eq!(host.service().sync_calls, 0);
        assert_eq!(host.service().async_calls, 1);
        assert_eq!(
            AsyncAware::decode_rep(&client.rep.pop().expect("reply")),
            Some((Reply::Granted, 0))
        );
    }

    /// Walking return posts are host-owned reply-and-wait state: the first
    /// stands before readiness, and every successor is acknowledged before
    /// `serve_next` returns with its reply visible.
    #[test]
    fn walking_return_rearms_each_successor_before_reply_completion() {
        let (mut client, rings) =
            split_window::<Walking>(walking_window(), REQ_CAP, REP_CAP).expect("walking rings");
        let endpoints = ServeEndpoints {
            kick: SlotRef::from_wire(60),
            grant_tx: [None; MAX_GRANTS],
            scratch: [None; MAX_GRANTS],
            return_post: Some((SlotRef::from_wire(70), SlotRef::from_wire(71))),
        };
        let transport = OkTransport::new();
        let mut host = ServeHost::new(Walking::bind(NoCaps), &transport, rings, endpoints);

        assert!(drive(host.stand_return_post()));
        assert_eq!(host.service().stood, 1, "initial post precedes readiness");
        assert!(host.service().pending.is_none());

        for request in [1, 2] {
            client.req.push(request).expect("request fits");
            assert!(drive(host.serve_next()));
            assert_eq!(host.service().stood, request as u8 + 1);
            assert!(host.service().pending.is_none(), "successor already stands");
            assert_eq!(
                Walking::decode_rep(&client.rep.pop().expect("reply")),
                Some((Reply::Granted, 0))
            );
        }
        assert_eq!(
            transport.commits.get(),
            5,
            "one initial post + post/kick per reply"
        );
    }

    /// If a Walking successor could not stand before the in-call retry bound,
    /// the host exposes that owed post to the caller-owned park policy. This
    /// is the no-lost-wake half of the protocol: with no post standing, a new
    /// sender cannot be relied on to wake the service.
    #[test]
    fn walking_return_exposes_an_owed_post_before_unbounded_park() {
        let (_client, rings) =
            split_window::<Walking>(walking_window(), REQ_CAP, REP_CAP).expect("walking rings");
        let endpoints = ServeEndpoints {
            kick: SlotRef::from_wire(60),
            grant_tx: [None; MAX_GRANTS],
            scratch: [None; MAX_GRANTS],
            return_post: Some((SlotRef::from_wire(70), SlotRef::from_wire(71))),
        };
        let transport = OkTransport::new();
        let mut host = ServeHost::new(Walking::bind(NoCaps), &transport, rings, endpoints);

        assert!(
            host.walking_post_pending(),
            "the initial selected landing is owed before readiness"
        );
        assert!(drive(host.stand_return_post()));
        assert!(
            !host.walking_post_pending(),
            "the stood post closes the debt"
        );

        // Model the state left by a completed transfer whose immediate
        // successor-post attempts all lost the sender-completion race.
        host.pairs[0].as_mut().expect("pair zero").walking_post = None;
        host.service_mut().pending = Some(SlotRef::from_wire(72));
        assert!(
            host.walking_post_pending(),
            "the caller must select a retry deadline instead of parking"
        );
    }

    /// A refused grant commit invokes the service repair hook before the
    /// reply is encoded, and the wire carries the actual effected count (0).
    #[test]
    fn a_dropped_grant_repairs_service_state_and_reply() {
        let (mut client, rings) =
            split_window::<DropAware>(window(), REQ_CAP, REP_CAP).expect("test service rings");
        let mut grant_tx = [None; MAX_GRANTS];
        grant_tx[0] = Some(SlotRef::from_wire(61));
        let endpoints = ServeEndpoints {
            kick: SlotRef::from_wire(60),
            grant_tx,
            scratch: [None; MAX_GRANTS],
            return_post: None,
        };
        let transport = TrapTransport::new();
        let service = DropAware::bind(NoCaps);
        let mut host = ServeHost::new(service, &transport, rings, endpoints);

        client.req.push(1).expect("request ring has room");
        assert!(drive(host.serve_next()));

        let wire = client.rep.pop().expect("grant-drop reply");
        assert_eq!(DropAware::decode_rep(&wire), Some((Reply::Denied, 0)));
        assert!(!host.service().reserved);
        assert!(host.service().repaired);
        assert_eq!(host.service().committed, None);
        assert_eq!(host.stats().grants_dropped, 1);
    }

    #[test]
    fn committed_grants_notify_both_serving_paths_with_the_structural_owner() {
        for minted in [false, true] {
            let (mut client, rings) =
                split_window::<DropAware>(window(), REQ_CAP, REP_CAP).unwrap();
            let mut grant_tx = [None; MAX_GRANTS];
            grant_tx[0] = Some(SlotRef::from_wire(61));
            let transport = OkTransport::new();
            let mut service = DropAware::bind(NoCaps);
            if minted {
                let mut pair = MintedPair {
                    rings,
                    grant_tx,
                    scratch: [None; MAX_GRANTS],
                    kick: None,
                    endpoint: 19,
                    admission: None,
                    admitted: false,
                    stats: ServeStats::default(),
                };
                drive(pair.serve_one(&mut service, &transport, &1));
                assert_eq!(service.committed, Some(19));
                assert!(service.reserved && !service.repaired);
            } else {
                let endpoints = ServeEndpoints {
                    kick: SlotRef::from_wire(60),
                    grant_tx,
                    scratch: [None; MAX_GRANTS],
                    return_post: None,
                };
                let mut host = ServeHost::new(service, &transport, rings, endpoints);
                client.req.push(1).unwrap();
                assert!(drive(host.serve_next()));
                assert_eq!(host.service().committed, Some(0));
                assert!(host.service().reserved && !host.service().repaired);
            }
            assert_eq!(
                DropAware::decode_rep(&client.rep.pop().unwrap()),
                Some((Reply::Granted, 1))
            );
        }
    }

    #[test]
    fn only_a_well_formed_conflict_retries_the_exact_grant_descriptor() {
        assert_eq!(
            grant_commit_action(Committed::from_raw(Status::Ok.encode(0)), 3),
            GrantCommitAction::Committed
        );
        assert_eq!(
            grant_commit_action(Committed::from_raw(Status::Conflict.encode(2)), 3),
            GrantCommitAction::Retry
        );
        for committed in [
            Committed::from_raw(Status::Denied.encode(1)),
            Committed::from_raw(Status::WouldBlock.encode(0)),
            Committed::from_raw(Status::Conflict.encode(3)),
            Committed::from_raw(Status::Ok.encode(1)),
        ] {
            assert_eq!(grant_commit_action(committed, 3), GrantCommitAction::Drop);
        }
    }
    /// A service that answers with the arrival pair's structural identity —
    /// the multi-client broker's whole reason for [`Serve::via`].
    struct WhoAsked;

    impl SystemService for WhoAsked {
        const NAME: &'static str = "who-asked";
        type Endowment = NoCaps;

        fn bind(_: NoCaps) -> Self {
            Self
        }

        fn serve(&mut self, req: u64, sv: &mut crate::service::Serve<'_, Reply>) {
            assert_eq!(
                u64::from(sv.via()),
                req,
                "the stamp is the pair the request arrived on"
            );
            sv.reply(Reply::Granted);
        }
    }

    impl SvcProtocol for WhoAsked {
        type Request = u64;
        type Response = Reply;
        type ReqWire = u64;
        type RepWire = u64;
        const REPLY_GRANTS: usize = 0;
        const SERVE_PAIRS: usize = 2;

        fn encode_req(req: &u64) -> u64 {
            *req
        }
        fn decode_req(wire: &u64) -> Option<u64> {
            Some(*wire)
        }
        fn encode_rep(rep: &Reply, grants: u8) -> u64 {
            DropAware::encode_rep(rep, grants)
        }
        fn decode_rep(wire: &u64) -> Option<(Reply, u8)> {
            DropAware::decode_rep(wire)
        }
    }

    fn who_window() -> &'static mut [u8] {
        let bytes = window_bytes::<WhoAsked>(REQ_CAP, REP_CAP);
        let layout = std::alloc::Layout::from_size_align(bytes, 64).expect("window layout");
        // SAFETY: as `window()` — one leaked, aligned allocation per pair.
        unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            assert!(!ptr.is_null());
            core::slice::from_raw_parts_mut(ptr, bytes)
        }
    }

    fn who_endpoints(kick: u32) -> ServeEndpoints {
        ServeEndpoints {
            kick: SlotRef::from_wire(kick),
            grant_tx: [None; MAX_GRANTS],
            scratch: [None; MAX_GRANTS],
            return_post: None,
        }
    }

    /// Two pairs over ONE service: each request is served against the shared
    /// state, stamped with its own arrival pair, and replied on its own
    /// rings — the `res/DEVICE_SERVICE.md` §10 multi-client shape.
    #[test]
    fn two_pairs_share_one_service_and_stamp_arrival_identity() {
        let (mut client0, rings0) =
            split_window::<WhoAsked>(who_window(), REQ_CAP, REP_CAP).expect("pair 0 rings");
        let (mut client1, rings1) =
            split_window::<WhoAsked>(who_window(), REQ_CAP, REP_CAP).expect("pair 1 rings");
        let transport = TrapTransport::new();
        let mut host = ServeHost::new(WhoAsked, &transport, rings0, who_endpoints(60));
        assert!(
            host.attach_pair(1, rings1, who_endpoints(61)),
            "the second pair attaches"
        );
        assert!(
            !host.attach_pair(0, rings0_placeholder(), who_endpoints(62)),
            "pair 0 is the constructor's alone"
        );

        // Each request carries the pair index it is pushed on; the service
        // asserts the stamp matches.
        client1.req.push(1).expect("pair 1 request");
        client0.req.push(0).expect("pair 0 request");
        client0
            .req
            .push(0)
            .expect("pair 0 remains continuously ready");
        assert!(drive(host.serve_next()));
        assert!(drive(host.serve_next()));

        // The second drive must rotate to pair 1 even though pair 0 still has
        // work. A fixed scan from zero made a hot first client starve every
        // later pair forever on metal.
        assert_eq!(
            WhoAsked::decode_rep(&client1.rep.pop().expect("pair 1 fair reply")),
            Some((Reply::Granted, 0))
        );
        assert!(drive(host.serve_next()));
        assert!(!drive(host.serve_next()), "both rings drained");

        assert_eq!(
            WhoAsked::decode_rep(&client0.rep.pop().expect("pair 0 reply")),
            Some((Reply::Granted, 0))
        );
        assert_eq!(
            WhoAsked::decode_rep(&client0.rep.pop().expect("pair 0 second reply")),
            Some((Reply::Granted, 0))
        );
        assert_eq!(host.stats().served, 3, "the books sum across pairs");
    }

    /// A rings value for the double-attach refusal check — never served.
    fn rings0_placeholder() -> ServeRings<'static, WhoAsked> {
        let (_, rings) =
            split_window::<WhoAsked>(who_window(), REQ_CAP, REP_CAP).expect("placeholder rings");
        rings
    }

    struct Lifecycle {
        closed: [Option<(u16, PeerClose)>; 8],
        count: usize,
        close_attempts: usize,
        defer_close_once: bool,
        detached: [Option<u16>; MAX_SERVE_PAIRS],
    }

    impl SystemService for Lifecycle {
        const NAME: &'static str = "lifecycle";
        type Endowment = NoCaps;

        fn bind(_: NoCaps) -> Self {
            Self {
                closed: [None; 8],
                count: 0,
                close_attempts: 0,
                defer_close_once: false,
                detached: [None; MAX_SERVE_PAIRS],
            }
        }

        fn serve(&mut self, req: u64, sv: &mut Serve<'_, Reply>) {
            let close = req & (1 << 63) != 0;
            assert_eq!(u64::from(sv.via()), req & !(1 << 63));
            sv.reply(Reply::Granted);
            if close {
                sv.close_peer(PeerClose::Orderly);
            }
        }

        fn peer_detaching(&mut self, peer: PeerId) {
            self.detached[peer.slot()] = Some(peer.arrival());
        }

        fn peer_closed(&mut self, via: u16, reason: PeerClose) -> bool {
            let peer = PeerId::from_arrival(via).expect("recyclable close");
            assert_eq!(
                self.detached[peer.slot()],
                Some(via),
                "auxiliary references close before host/domain teardown"
            );
            self.close_attempts += 1;
            if core::mem::take(&mut self.defer_close_once) {
                return false;
            }
            self.closed[self.count] = Some((via, reason));
            self.count += 1;
            true
        }
    }

    impl SvcProtocol for Lifecycle {
        type Request = u64;
        type Response = Reply;
        type ReqWire = u64;
        type RepWire = u64;
        const REPLY_GRANTS: usize = 0;

        fn encode_req(req: &u64) -> u64 {
            *req
        }
        fn decode_req(wire: &u64) -> Option<u64> {
            Some(*wire)
        }
        fn encode_rep(rep: &Reply, grants: u8) -> u64 {
            DropAware::encode_rep(rep, grants)
        }
        fn decode_rep(wire: &u64) -> Option<(Reply, u8)> {
            DropAware::decode_rep(wire)
        }
    }

    fn lifecycle_window() -> (
        crate::svc::ClientRings<'static, Lifecycle>,
        ServeRings<'static, Lifecycle>,
    ) {
        let bytes = window_bytes::<Lifecycle>(REQ_CAP, REP_CAP);
        let layout = std::alloc::Layout::from_size_align(bytes, 64).expect("window layout");
        // SAFETY: one leaked aligned allocation, split only through SPSC.
        let window = unsafe {
            let ptr = std::alloc::alloc_zeroed(layout);
            assert!(!ptr.is_null());
            core::slice::from_raw_parts_mut(ptr, bytes)
        };
        split_window::<Lifecycle>(window, REQ_CAP, REP_CAP).expect("lifecycle rings")
    }

    #[test]
    fn recyclable_peer_stands_its_return_post_before_publication() {
        let (_static_client, static_rings) = lifecycle_window();
        let transport = OkTransport::new();
        let mut host = ServeHost::new(
            Lifecycle::bind(NoCaps),
            &transport,
            static_rings,
            who_endpoints(70),
        );
        let (_client, rings) = lifecycle_window();
        let endpoints = ServeEndpoints {
            kick: SlotRef::from_wire(71),
            grant_tx: [None; MAX_GRANTS],
            scratch: [None; MAX_GRANTS],
            return_post: Some((SlotRef::from_wire(72), SlotRef::from_wire(73))),
        };
        let peer = drive(host.attach_recyclable(rings, endpoints)).expect("dynamic peer");
        assert_eq!(transport.commits.get(), 1, "the receive post is committed");
        assert!(
            host.pairs[peer.slot()]
                .as_ref()
                .is_some_and(|pair| pair.return_standing),
            "a published peer is immediately ready for its first return"
        );
    }

    #[test]
    #[cfg(feature = "heap")]
    fn dynamic_peer_table_crosses_bootstrap_population() {
        let bytes = 1024 * 1024;
        let layout = std::alloc::Layout::from_size_align(bytes, 4096).unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null());
        let source = unsafe { crate::heap::StaticMemSource::over(base, bytes) };
        let arena = std::boxed::Box::leak(std::boxed::Box::new(crate::Heap::new_in(source)));
        let (_client, rings) = lifecycle_window();
        let transport = TrapTransport::new();
        let mut host = ServeHost::new(
            Lifecycle::bind(NoCaps),
            &transport,
            rings,
            who_endpoints(70),
        );
        assert!(host.enable_peer_growth(arena));
        for slot in 1..32 {
            let (_client, rings) = lifecycle_window();
            let peer =
                drive(host.attach_recyclable(rings, who_endpoints(70 + slot as u32))).unwrap();
            assert_eq!(peer.slot(), slot);
            assert_eq!(PeerId::from_arrival(peer.arrival()), Some(peer));
        }
        assert_eq!(host.pairs.len(), 32);
    }

    #[test]
    fn recyclable_capacity_tracks_the_fixed_pairs_actually_attached() {
        let (_static_client, static_rings) = lifecycle_window();
        let transport = TrapTransport::new();
        let mut one_fixed = ServeHost::new(
            Lifecycle::bind(NoCaps),
            &transport,
            static_rings,
            who_endpoints(70),
        );
        for endpoint in 71..70 + MAX_SERVE_PAIRS as u32 {
            let (_client, rings) = lifecycle_window();
            drive(one_fixed.attach_recyclable(rings, who_endpoints(endpoint)))
                .expect("one fixed pair leaves the remaining cells recyclable");
        }
        let (_overflow, rings) = lifecycle_window();
        assert!(
            drive(one_fixed.attach_recyclable(rings, who_endpoints(74))).is_none(),
            "all host cells are occupied"
        );

        let (_static_client, static_rings) = lifecycle_window();
        let mut two_fixed = ServeHost::new(
            Lifecycle::bind(NoCaps),
            &transport,
            static_rings,
            who_endpoints(80),
        );
        let (_fixed_client, fixed_rings) = lifecycle_window();
        assert!(two_fixed.attach_pair(1, fixed_rings, who_endpoints(81)));
        for endpoint in 82..80 + MAX_SERVE_PAIRS as u32 {
            let (_client, rings) = lifecycle_window();
            drive(two_fixed.attach_recyclable(rings, who_endpoints(endpoint)))
                .expect("two fixed pairs leave the remaining cells recyclable");
        }
        let (_overflow, rings) = lifecycle_window();
        assert!(
            drive(two_fixed.attach_recyclable(rings, who_endpoints(84))).is_none(),
            "the second fixed pair consumes one recyclable opportunity"
        );
    }

    #[test]
    fn recyclable_peers_refuse_capacity_reuse_slots_and_reject_stale_close() {
        let (_static_client, static_rings) = lifecycle_window();
        let transport = TrapTransport::new();
        let mut host = ServeHost::new(
            Lifecycle::bind(NoCaps),
            &transport,
            static_rings,
            who_endpoints(70),
        );

        let (mut c1, r1) = lifecycle_window();
        let (mut c2, r2) = lifecycle_window();
        let (_c3, r3) = lifecycle_window();
        let (_overflow, r4) = lifecycle_window();
        let p1 = drive(host.attach_recyclable(r1, who_endpoints(71))).expect("first dynamic peer");
        let p2 = drive(host.attach_recyclable(r2, who_endpoints(72))).expect("second dynamic peer");
        let _p3 = drive(host.attach_recyclable(r3, who_endpoints(73))).expect("third dynamic peer");
        for slot in 4..MAX_SERVE_PAIRS {
            let (_client, rings) = lifecycle_window();
            let peer = drive(host.attach_recyclable(rings, who_endpoints(80 + slot as u32)))
                .expect("remaining recyclable cell");
            assert_eq!(PeerId::from_arrival(peer.arrival()), Some(peer));
        }
        assert!(
            drive(host.attach_recyclable(r4, who_endpoints(74))).is_none(),
            "the bounded host reports capacity instead of partial admission"
        );
        assert_ne!(p1, p2);
        assert!(host.peer_is_live(p1));
        assert!(host.peer_is_live(p2));

        c1.req.push(u64::from(p1.via())).expect("peer 1 request");
        c2.req.push(u64::from(p2.via())).expect("peer 2 request");
        assert!(drive(host.serve_next()));
        assert!(drive(host.serve_next()));
        assert!(c1.rep.pop().is_some());
        assert!(c2.rep.pop().is_some());

        assert!(host.retire_peer(p1, PeerClose::Died));
        assert!(!host.peer_is_live(p1));
        assert_eq!(
            host.service().closed[0],
            Some((p1.via(), PeerClose::Died)),
            "death uses the one service cleanup hook"
        );

        let (mut replacement_client, replacement_rings) = lifecycle_window();
        let replacement = drive(host.attach_recyclable(replacement_rings, who_endpoints(75)))
            .expect("retirement reopens capacity");
        assert_eq!(replacement.slot(), p1.slot(), "the cell is recycled");
        assert_ne!(replacement.generation(), p1.generation());
        assert!(host.peer_is_live(replacement));
        assert!(
            !host.retire_peer(p1, PeerClose::Orderly),
            "a stale close cannot retire the replacement generation"
        );
        assert!(host.peer_is_live(replacement));

        replacement_client
            .req
            .push(u64::from(replacement.via()))
            .expect("replacement request");
        assert!(drive(host.serve_next()));
        assert!(replacement_client.rep.pop().is_some());
        assert!(host.retire_peer(replacement, PeerClose::Revoked));
        assert_eq!(
            host.service().closed[1],
            Some((replacement.via(), PeerClose::Revoked))
        );

        c2.req
            .push(u64::from(p2.via()) | (1 << 63))
            .expect("peer 2 orderly close");
        assert!(drive(host.serve_next()));
        assert!(c2.rep.pop().is_some(), "the close reply is published first");
        assert!(!host.peer_is_live(p2));
        assert_eq!(
            host.service().closed[2],
            Some((p2.via(), PeerClose::Orderly)),
            "the founded run path can reach host-owned retirement"
        );
    }

    #[test]
    fn recyclable_peer_slot_stays_quarantined_until_domain_cleanup_commits() {
        let (_static_client, static_rings) = lifecycle_window();
        let transport = TrapTransport::new();
        let mut host = ServeHost::new(
            Lifecycle::bind(NoCaps),
            &transport,
            static_rings,
            who_endpoints(80),
        );
        let (_client, rings) = lifecycle_window();
        let peer = drive(host.attach_recyclable(rings, who_endpoints(81))).expect("dynamic peer");
        for index in 2..MAX_SERVE_PAIRS {
            let (_client, rings) = lifecycle_window();
            drive(host.attach_recyclable(rings, who_endpoints(81 + index as u32)))
                .expect("fill every other dynamic cell");
        }
        host.service_mut().defer_close_once = true;
        assert!(!host.has_pending_cleanup());

        assert!(host.retire_peer(peer, PeerClose::Died));
        assert!(host.has_pending_cleanup());
        assert_eq!(host.service().close_attempts, 1);
        assert_eq!(host.service().count, 0, "domain teardown deferred");
        let (_replacement, replacement_rings) = lifecycle_window();
        assert!(
            drive(host.attach_recyclable(replacement_rings, who_endpoints(90))).is_none(),
            "the closing cell is not reusable"
        );

        assert_eq!(host.retry_peer_cleanup(), 1);
        assert!(!host.has_pending_cleanup());
        assert_eq!(host.service().close_attempts, 2);
        assert_eq!(
            host.service().closed[0],
            Some((peer.via(), PeerClose::Died))
        );
        let (_replacement, replacement_rings) = lifecycle_window();
        let replacement = drive(host.attach_recyclable(replacement_rings, who_endpoints(91)))
            .expect("committed cleanup reopens the cell");
        assert_eq!(replacement.slot(), peer.slot());
        assert_ne!(replacement.generation(), peer.generation());
    }

    #[test]
    fn recyclable_map_plan_is_slot_bounded_and_overflow_safe() {
        let plan = PeerMapPlan {
            base: 0x4000_0000,
            stride: 0x2000,
        };
        assert_eq!(plan.address(0, 0x1000), Some(0x4000_0000));
        assert_eq!(plan.address(3, 0x2000), Some(0x4000_6000));
        assert_eq!(plan.address(0, 0x2001), None);
        assert_eq!(
            PeerMapPlan {
                base: u64::MAX & !0xfff,
                stride: 0x1000,
            }
            .address(2, 0x1000),
            None
        );
    }
}
