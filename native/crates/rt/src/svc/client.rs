//! `svc::client` — the born-async client machine (`res/USERLAND.md`
//! §5.6): the ring + channel proxy every per-service client wraps.
//!
//! One call = **stand every reply post while still idle** → **push the
//! request** → **kick** → **await the reply ring** → **collect the
//! announced grant deliveries**.  The pre-publication order is load-bearing:
//! after replying to a prior request the service may still be running on
//! another core and poll the ring without consuming a new kick. Publishing a
//! request before a freshly rotated grant post stands would let it attempt the
//! grant against no receiver. Post convergence is bounded: an already-standing
//! channel reports `WouldBlock` at its own invocation and drops out of the next
//! attempt; an ambiguous transport result keeps a retained target armed so it
//! can never be reused beneath a late delivery.
//!
//! Grant posts are **pre-pinned before the kick** and stay pinned on
//! the board until consumed (§5.5): that is what makes the service's
//! grant transfer unable to `WouldBlock` against a conforming client.

use core::marker::PhantomData;
use core::task::Poll;

use kore_abi::cap::Rights;
use kore_abi::session::{Committed, SlotRef, Status};

use crate::cap::{
    ChanRecvToken, ChanSendToken, HeldChanSendToken, LeaseToken, LeasedToken, RawDelivery,
    ReturnedCapability, StagedCapabilityReservation, StashedLease, verbs,
};
use crate::chan::SendError;
use crate::executor::{Doorbell, settle_inline_doorbell, stand_doorbell};
use crate::service::MAX_GRANTS;
use crate::session::{BuildError, SessionBuilder, Staged};
use crate::transport::{ParkSpec, ParkTransport, Transport};

use super::{ClientRings, SvcProtocol};

#[cfg(target_os = "none")]
fn live_send_destination(slot: SlotRef) -> bool {
    verbs::capability_rights(slot, kore_abi::cap::ResourceKind::ChanSend)
        .is_some_and(|rights| rights.contains(Rights::ACCESS))
}

#[cfg(not(target_os = "none"))]
const fn live_send_destination(_slot: SlotRef) -> bool {
    true
}

/// One pre-pinnable grant channel on the client side.
///
/// Fixed protocols donate the same caller-owned landing on every round.
/// Retained protocols instead rotate a runtime-owned landing after every
/// delivery, so representing them with a fabricated fixed slot would create
/// an authority name that the protocol must never use.
#[derive(Clone, Copy, Debug)]
pub struct GrantPin {
    /// The held `ChanRecv` record's slot.
    recv: SlotRef,
    landing: GrantLanding,
}

#[derive(Clone, Copy, Debug)]
enum GrantLanding {
    /// One caller-owned landing reused after each protocol-specific consume.
    Fixed(SlotRef),
    /// A fresh runtime-owned landing for every authority-bearing reply.
    Retained,
}

impl GrantPin {
    /// Pin a grant channel whose delivery is consumed from a fixed slot before
    /// the next request may reuse it.
    pub const fn fixed(recv: SlotRef, land: SlotRef) -> Self {
        Self {
            recv,
            landing: GrantLanding::Fixed(land),
        }
    }

    /// Pin a grant channel whose delivered capability outlives the call.
    ///
    /// The runtime reserves and rotates the actual landing; callers therefore
    /// supply no dummy arena name here.
    pub const fn retained(recv: SlotRef) -> Self {
        Self {
            recv,
            landing: GrantLanding::Retained,
        }
    }

    fn fixed_landing(self) -> Option<SlotRef> {
        match self.landing {
            GrantLanding::Fixed(slot) => Some(slot),
            GrantLanding::Retained => None,
        }
    }

    fn matches_retention(self, retain: bool) -> bool {
        matches!(
            (self.landing, retain),
            (GrantLanding::Retained, true) | (GrantLanding::Fixed(_), false)
        )
    }
}

/// The client's channel wiring for one pair (§5.3): the kick send end
/// plus the grant-channel bank ([`SvcProtocol::REPLY_GRANTS`] channels;
/// one standing post per channel is the board law).
#[derive(Clone, Copy, Debug)]
pub struct ClientEndpoints {
    /// The held `ChanSend` to the **service's** doorbell.
    pub kick: SlotRef,
    /// The grant-channel recv pins, `[0..REPLY_GRANTS]` populated.
    pub grants: [Option<GrantPin>; MAX_GRANTS],
}

/// The re-kick interval inside a reply wait.
///
/// Also the granularity of a bounded reply observation when the clock cannot
/// answer: the budget is spent in units of this interval, so it stays a
/// duration either way.
const REPLY_REKICK_NS: u64 = 1_000_000;

/// One fixed executor-clock cadence for level-triggered request re-kicks.
///
/// The deadline moves only after it is reached. Rebuilding a relative
/// `now + REPLY_REKICK_NS` wait after every unrelated wake would let a busy
/// doorbell postpone the re-kick forever.
#[derive(Clone, Copy, Debug)]
struct ReplyRekickCadence {
    deadline_ns: u64,
}

impl ReplyRekickCadence {
    const fn after(now_ns: u64) -> Self {
        Self {
            deadline_ns: now_ns.saturating_add(REPLY_REKICK_NS),
        }
    }

    const fn deadline(self) -> u64 {
        self.deadline_ns
    }

    const fn due(self, now_ns: u64) -> bool {
        now_ns >= self.deadline_ns
    }

    fn rearm(&mut self, now_ns: u64) {
        *self = Self::after(now_ns);
    }
}

/// One bounded observation of an already-correlated request.
///
/// The executor deadline is sampled before publication and then carried
/// through both the compact initial kick/park and the ordinary async reply
/// loop. Starting a fresh duration after the compact park would leave that
/// first (potentially silent) wait outside the caller's bound.
#[derive(Clone, Copy, Debug)]
enum ReplyLimit {
    Duration(u64),
    Deadline(u64),
}

impl ReplyLimit {
    const fn deadline(self, now_ns: u64) -> u64 {
        match self {
            Self::Duration(budget) => now_ns.saturating_add(budget),
            Self::Deadline(deadline) => deadline,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ReplyObservation {
    budget_ns: u64,
    started_ns: Option<u64>,
    deadline_ns: u64,
    elapsed: bool,
}

impl ReplyObservation {
    async fn begin(limit: Option<ReplyLimit>) -> Option<Self> {
        let limit = limit?;
        let started_ns = crate::clock::uptime().ok().map(|now| now.as_nanos());
        let now_ns = crate::executor::clock_ns().await;
        let deadline_ns = limit.deadline(now_ns);
        let budget_ns = deadline_ns.saturating_sub(now_ns);
        Some(Self {
            budget_ns,
            started_ns,
            deadline_ns,
            elapsed: false,
        })
    }

    fn note_deadline(&mut self) {
        self.elapsed = true;
    }

    const fn expired(self, executor_now_ns: u64) -> bool {
        self.elapsed || executor_now_ns >= self.deadline_ns
    }

    const fn next_deadline(self, cadence_deadline_ns: u64) -> u64 {
        if self.deadline_ns < cadence_deadline_ns {
            self.deadline_ns
        } else {
            cadence_deadline_ns
        }
    }
}

/// Whether this wake should re-kick the still-published request.
///
/// A readable monotonic clock lets [`ReplyRekickCadence`] pace traffic. Without
/// one, an early non-deadline wake carries no elapsed-time proof, but re-kicking
/// is safe and prevents a stream of such wakes from starving the level edge.
const fn reply_rekick_needed(elapsed_tick: bool, absolute_clock: bool) -> bool {
    elapsed_tick || !absolute_clock
}

/// How many [`REPLY_REKICK_NS`] ticks a budget is worth, for the clock-less
/// half of the reply bound.
///
/// Floors at one tick. A protocol whose budget is shorter than a single
/// re-kick interval would otherwise get `tick_budget == 0` and time out on the
/// first deadline wake without ever having waited — which reads as "the
/// service is dead" when nothing was asked of it yet.
const fn reply_budget_ticks(budget_ns: u64) -> u64 {
    let whole = budget_ns / REPLY_REKICK_NS;
    let ticks = whole + (budget_ns % REPLY_REKICK_NS != 0) as u64;
    match ticks {
        0 => 1,
        ticks => ticks,
    }
}

/// Observe one executor wake against an optional reply-observation budget.
///
/// `None` is the ordinary call policy and can never expire. A bounded caller
/// prefers a stable clock pair, which is checked after every wake so unrelated
/// traffic cannot hide an elapsed absolute deadline. If either sample is
/// unavailable, only a proven elapsed re-kick cadence spends the fallback.
fn reply_wait_expired(
    budget_ns: Option<u64>,
    started_ns: Option<u64>,
    now_ns: Option<u64>,
    elapsed_tick: bool,
    ticks: &mut u64,
) -> bool {
    let Some(budget_ns) = budget_ns else {
        return false;
    };
    match (started_ns, now_ns) {
        (Some(started), Some(now)) => now.saturating_sub(started) >= budget_ns,
        _ if elapsed_tick => {
            *ticks = ticks.saturating_add(1);
            *ticks >= reply_budget_ticks(budget_ns)
        }
        _ => false,
    }
}

/// Why a call failed — **infrastructure** failures only (a service's
/// refusal is a normal response, never an error here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallError {
    /// This client already owns a preparing or published request. Call
    /// [`SvcClient::resume`] instead of starting another
    /// request into the uncorrelated SPSC reply stream.
    Outstanding,
    /// No published request exists for [`SvcClient::resume`] to continue.
    NoOutstanding,
    /// A reply or grant stream violated its protocol after the request was
    /// published. Without request IDs the pair cannot be resynchronized and
    /// must not be reused.
    Poisoned,
    /// The request ring is full — the service is behind and the bounded
    /// client backs off (its own capacity choice made visible).
    RingFull,
    /// The peer wrote an out-of-vocabulary reply (fail-closed decode),
    /// or a grant channel delivered something that is not a capability.
    Wire,
    /// The endpoint wiring is short for this protocol (fewer grant channels
    /// than [`SvcProtocol::REPLY_GRANTS`]), or a bounded reply observation was
    /// requested from a direct-polling client that owns no executor timer.
    Wiring,
    /// The gate refused the request session with `status`.
    Refused(Status),
    /// The transport could not commit at all (the trap is not open).
    Transport,
    /// The bounded [posts + notice] retry never converged, or a retained grant
    /// landing could not yet reserve runtime-owned slot capacity.
    Backpressure,
    /// The service did not reply within the interval explicitly supplied to
    /// [`SvcClient::call_bounded`] or [`SvcClient::resume_bounded`].
    ///
    /// **Resumable, and deliberately not `Poisoned`.** The request is still
    /// published and this client is still `AwaitingReply`, so a late reply
    /// remains correlated: a caller that believes the service is merely slow
    /// calls a matching bounded resume and waits another observation slice.
    /// Ordinary [`SvcClient::call`] remains unbounded: a timeout is safe only
    /// for an operation whose caller retains the request identity and owns a
    /// terminal policy for the still-live pair.
    Timeout,
}

/// Failure of an authority-bearing call, split by proven source custody.
///
/// A non-`Ok` commit is not generally proof that the source survived: an OCC
/// conflict may abort after the held lease was moved, in which case unwind
/// retires the fresh delivered lineage rather than restoring the sender's
/// name. The source token is therefore returned only for outcomes that prove
/// the held twin was not moved. The returned record can still bind `Revoked`
/// (an ancestor may have been reaped concurrently), so this is a custody
/// witness, not a liveness promise.
#[derive(Debug)]
pub enum ReturnCallError<C = LeaseToken> {
    /// The transfer did not move the source out of the caller's holdings.
    NotMoved {
        /// Infrastructure failure.
        error: CallError,
        /// Caller-custodied source (which may since have been revoked).
        capability: C,
    },
    /// Source custody is gone or cannot be proven, or the later data leg
    /// failed. No caller-side token may be reconstructed.
    AfterTransfer(CallError),
}

/// Failure of a cancellation-safe exact return whose affine owner remains in
/// the caller's `Option` until the synchronous transfer boundary.
///
/// `NotMoved` leaves that option populated. `AfterTransfer` leaves it empty
/// and the [`SvcClient`] either owns the correlated request or has poisoned
/// the pair. This split lets a higher-level state machine survive cancellation
/// during async grant-post preparation without moving its only cleanup owner
/// onto the cancelled future's stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetainedReturnCallError {
    /// The source is still held in the caller's option.
    NotMoved(CallError),
    /// The source moved or local custody can no longer be proved.
    AfterTransfer(CallError),
}

/// Maximum explicitly delegated auxiliary capabilities in one generic
/// invocation. The fixed bound is part of the atomic session budget, not a
/// command catalog.
pub const MAX_INVOCATION_AUXILIARIES: usize = 4;

/// Species-erased affine capability allowed in an invocation auxiliary lane.
/// Bare slots cannot be constructed as this type, so type erasure does not
/// weaken custody proof.
#[derive(Debug)]
pub enum InvocationAuxiliaryCapability {
    Lease(LeaseToken),
    Leased(LeasedToken),
    Stashed(StashedLease),
    ChanSend(ChanSendToken),
    ChanRecv(ChanRecvToken),
}

macro_rules! auxiliary_from {
    ($token:ty, $variant:ident) => {
        impl From<$token> for InvocationAuxiliaryCapability {
            fn from(value: $token) -> Self {
                Self::$variant(value)
            }
        }
    };
}

auxiliary_from!(LeaseToken, Lease);
auxiliary_from!(LeasedToken, Leased);
auxiliary_from!(StashedLease, Stashed);
auxiliary_from!(ChanSendToken, ChanSend);
auxiliary_from!(ChanRecvToken, ChanRecv);

/// How one affine auxiliary crosses into exec's atomic invocation bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuxiliaryTransfer {
    /// Re-derive beneath exec's posted landing and attenuate to `rights`.
    Derived,
    /// Move the existing record exactly. The source need not carry DERIVE.
    Exact,
}

/// One affine auxiliary plus its explicit transfer policy.
#[derive(Debug)]
pub struct InvocationAuxiliary {
    capability: InvocationAuxiliaryCapability,
    rights: Rights,
    transfer: AuxiliaryTransfer,
}

impl InvocationAuxiliary {
    /// Carry an attenuated descendant. The source must hold DERIVE in
    /// addition to the requested delivered rights.
    pub fn new(capability: impl Into<InvocationAuxiliaryCapability>, rights: Rights) -> Self {
        Self {
            capability: capability.into(),
            rights,
            transfer: AuxiliaryTransfer::Derived,
        }
    }

    /// Move an already-attenuated affine record exactly into exec custody.
    /// Its authenticated kind, rights, tag, and lineage are preserved.
    pub fn exact(capability: impl Into<InvocationAuxiliaryCapability>) -> Self {
        Self {
            capability: capability.into(),
            rights: Rights::empty(),
            transfer: AuxiliaryTransfer::Exact,
        }
    }

    pub fn into_parts(self) -> (InvocationAuxiliaryCapability, Rights) {
        (self.capability, self.rights)
    }
}

/// Failure of an admission-gated call carrying the fixed generic context and
/// a bounded auxiliary set.
///
/// The inputs are reconstructed only when the kernel completion proves that
/// the atomic session moved neither of them. The admission endpoint is
/// borrowed: it is reusable launch authority, not invocation custody.
#[derive(Debug)]
pub enum AffineBundleCallError<A, B, C> {
    NotMoved {
        error: CallError,
        first: A,
        second: B,
        third: C,
        auxiliaries: [Option<InvocationAuxiliary>; MAX_INVOCATION_AUXILIARIES],
    },
    AfterTransfer(CallError),
}

/// Failure of a return call whose transferred capability is freshly staged
/// inside the transfer session.
///
/// There is no pre-existing token to return: the caller retains its held
/// derivation roots when the session does not transfer, while an
/// [`AfterTransfer`](Self::AfterTransfer) result means the service may already
/// own the fresh child and the uncorrelated pair cannot accept another call
/// unless one was published and is resumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagedReturnError {
    /// The transfer session did not move a capability to the service. The
    /// pair remains idle and may be retried.
    NotTransferred(CallError),
    /// The capability moved, or local custody can no longer be proved. If no
    /// request was published, the pair is poisoned before this is returned.
    AfterTransfer(CallError),
}

/// Why an approved fresh-child derive could not be staged for a return call.
///
/// This is deliberately narrower than [`BuildError`]: callers never choose a
/// result slot or gain access to the underlying session builder. Slot
/// exhaustion is therefore a runtime backpressure fact, while `Build` means
/// the bounded descriptor rejected the single approved derive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnDeriveError {
    /// The runtime-owned persistent-slot allocator has no vacant result.
    NoSlots,
    /// The borrowed source no longer exposes a live capability name.
    SourceUnavailable,
    /// The bounded session builder refused the approved derive.
    Build(BuildError),
}

impl From<BuildError> for ReturnDeriveError {
    fn from(error: BuildError) -> Self {
        Self::Build(error)
    }
}

impl ReturnDeriveError {
    fn call_error(self) -> CallError {
        match self {
            Self::NoSlots => CallError::Backpressure,
            Self::SourceUnavailable => CallError::Refused(Status::Revoked),
            Self::Build(_) => CallError::Refused(Status::Malformed),
        }
    }
}

mod return_capability_sealed {
    use super::{Rights, SlotRef};

    pub trait Sealed: Sized {
        fn return_slot(&self) -> Option<SlotRef>;

        /// Holder-readable rights used only to refuse an impossible
        /// derive-shaped rendezvous before its request record is published.
        /// The kernel remains the authoritative gate at commit.
        fn return_rights(&self) -> Option<Rights>;

        /// Settle local Rust ownership after a kernel-proven move. Plain
        /// possession witnesses carry no allocator reservation; dynamic
        /// owners override this to release the now-vacant local slot without
        /// attempting an invalid second revoke.
        fn on_proven_moved(self) {}
    }

    impl Sealed for super::LeaseToken {
        fn return_slot(&self) -> Option<SlotRef> {
            Some(self.slot())
        }

        fn return_rights(&self) -> Option<Rights> {
            Some(self.rights())
        }
    }

    impl Sealed for super::LeasedToken {
        fn return_slot(&self) -> Option<SlotRef> {
            Some(self.slot())
        }

        fn return_rights(&self) -> Option<Rights> {
            Some(self.rights())
        }
    }

    impl Sealed for super::StashedLease {
        fn return_slot(&self) -> Option<SlotRef> {
            self.slot()
        }

        fn return_rights(&self) -> Option<Rights> {
            self.slot().and_then(|slot| {
                super::verbs::capability_rights(slot, kore_abi::cap::ResourceKind::MemLease)
            })
        }

        fn on_proven_moved(self) {
            self.release_after_proven_move();
        }
    }

    impl Sealed for crate::cap::FabricRouteToken {
        fn return_slot(&self) -> Option<SlotRef> {
            Some(self.slot())
        }
        fn return_rights(&self) -> Option<Rights> {
            super::verbs::capability_rights(
                self.slot(),
                kore_abi::cap::ResourceKind::MemoryFabricRoute,
            )
        }
        fn on_proven_moved(self) {
            self.release_after_proven_move();
        }
    }

    impl Sealed for crate::cap::StashedMemory {
        fn return_slot(&self) -> Option<SlotRef> {
            self.slot()
        }
        fn return_rights(&self) -> Option<Rights> {
            match self {
                Self::Lease(lease) => Sealed::return_rights(lease),
                Self::Route(route) => Sealed::return_rights(route),
            }
        }
        fn on_proven_moved(self) {
            match self {
                Self::Lease(lease) => Sealed::on_proven_moved(lease),
                Self::Route(route) => Sealed::on_proven_moved(route),
            }
        }
    }

    impl Sealed for super::ChanSendToken {
        fn return_slot(&self) -> Option<SlotRef> {
            self.live_slot()
        }

        fn return_rights(&self) -> Option<Rights> {
            Some(self.rights())
        }

        fn on_proven_moved(self) {
            self.release_after_proven_move();
        }
    }

    impl Sealed for super::HeldChanSendToken {
        fn return_slot(&self) -> Option<SlotRef> {
            Some(self.slot())
        }

        fn return_rights(&self) -> Option<Rights> {
            Some(self.rights())
        }
    }

    impl Sealed for super::ChanRecvToken {
        fn return_slot(&self) -> Option<SlotRef> {
            self.live_slot()
        }

        fn return_rights(&self) -> Option<Rights> {
            Some(self.rights())
        }

        fn on_proven_moved(self) {
            self.release_after_proven_move();
        }
    }

    impl Sealed for super::InvocationAuxiliaryCapability {
        fn return_slot(&self) -> Option<SlotRef> {
            match self {
                Self::Lease(value) => Sealed::return_slot(value),
                Self::Leased(value) => Sealed::return_slot(value),
                Self::Stashed(value) => Sealed::return_slot(value),
                Self::ChanSend(value) => Sealed::return_slot(value),
                Self::ChanRecv(value) => Sealed::return_slot(value),
            }
        }

        fn return_rights(&self) -> Option<Rights> {
            match self {
                Self::Lease(value) => Sealed::return_rights(value),
                Self::Leased(value) => Sealed::return_rights(value),
                Self::Stashed(value) => Sealed::return_rights(value),
                Self::ChanSend(value) => Sealed::return_rights(value),
                Self::ChanRecv(value) => Sealed::return_rights(value),
            }
        }

        fn on_proven_moved(self) {
            match self {
                Self::Lease(value) => Sealed::on_proven_moved(value),
                Self::Leased(value) => Sealed::on_proven_moved(value),
                Self::Stashed(value) => Sealed::on_proven_moved(value),
                Self::ChanSend(value) => Sealed::on_proven_moved(value),
                Self::ChanRecv(value) => Sealed::on_proven_moved(value),
            }
        }
    }
}

/// An affine capability witness that may ride a service pair's retained
/// return leg.
///
/// This trait is sealed: adding a witness opts that capability species into
/// the generic move protocol, so callers cannot substitute a bare
/// [`SlotRef`] or a copyable wrapper and thereby manufacture a false custody
/// proof. The private supertrait exposes only the arena name already carried
/// by the witness and settles type-specific local ownership after a proven
/// move; callers cannot invoke that settlement hook themselves.
/// [`SvcClient::call_with_return`] consumes the witness exactly when the
/// kernel reports that custody moved (or can no longer be proven local).
pub trait ReturnCapability: return_capability_sealed::Sealed {}

impl ReturnCapability for LeaseToken {}
impl ReturnCapability for LeasedToken {}
impl ReturnCapability for StashedLease {}
impl ReturnCapability for crate::cap::FabricRouteToken {}
impl ReturnCapability for crate::cap::StashedMemory {}
impl ReturnCapability for ChanSendToken {}
impl ReturnCapability for HeldChanSendToken {}
impl ReturnCapability for ChanRecvToken {}
impl ReturnCapability for InvocationAuxiliaryCapability {}

/// Prove the local half of a derive-shaped rendezvous before publishing the
/// correlated request record. A send needs `DERIVE` on the source even when
/// the receiver is intentionally attenuated to non-delegable rights.
fn preflight_return_send<C: ReturnCapability>(
    capability: &C,
    requested: Rights,
) -> Result<SlotRef, CallError> {
    let slot = return_capability_sealed::Sealed::return_slot(capability)
        .ok_or(CallError::Refused(Status::Revoked))?;
    let available = return_capability_sealed::Sealed::return_rights(capability)
        .ok_or(CallError::Refused(Status::Revoked))?;
    if !available.contains(Rights::DERIVE | requested) {
        return Err(CallError::Refused(Status::Denied));
    }
    Ok(slot)
}

/// Prove only that an exact-move source still has a live local name. Exact
/// transfer preserves the current record and therefore requires no DERIVE.
fn preflight_exact_return<C: ReturnCapability>(capability: &C) -> Result<SlotRef, CallError> {
    return_capability_sealed::Sealed::return_slot(capability)
        .ok_or(CallError::Refused(Status::Revoked))
}

fn invocation_sources_are_distinct(slots: &[SlotRef]) -> bool {
    slots
        .iter()
        .enumerate()
        .all(|(index, slot)| !slots[..index].contains(slot))
}

mod return_lease_source_sealed {
    use super::SlotRef;

    pub trait Sealed {
        fn source_slot(&self) -> Option<SlotRef>;
    }

    impl Sealed for super::LeaseToken {
        fn source_slot(&self) -> Option<SlotRef> {
            Some(self.slot())
        }
    }

    impl Sealed for super::StashedLease {
        fn source_slot(&self) -> Option<SlotRef> {
            self.slot()
        }
    }
}

/// A held lease root from which a single fresh child may be staged for a
/// service return leg.
///
/// The trait is sealed so an external caller cannot substitute a raw arena
/// name. The source is borrowed: only the newly derived child crosses the
/// rendezvous boundary, while the lessor position remains with the caller.
pub trait ReturnLeaseSource: return_lease_source_sealed::Sealed {}

impl ReturnLeaseSource for LeaseToken {}
impl ReturnLeaseSource for StashedLease {}

/// One freshly staged result paired with the runtime allocator reservation
/// for its arena slot.
///
/// Opaque and affine by construction. A client may hand it only back to the
/// call that created its [`ReturnDeriver`]; there is no result-slot accessor
/// and therefore no way to alias the runtime's retained grant landings.
#[derive(Debug)]
pub struct StagedReturn {
    staged: Staged,
    reservation: StagedCapabilityReservation,
}

impl StagedReturn {
    fn begin_transfer(&mut self) {
        self.reservation.begin_transfer();
    }

    fn settle_proven(&mut self) {
        self.reservation.settle_proven();
    }

    fn quarantine(&mut self) {
        self.reservation.quarantine();
    }
}

/// A constrained builder for a fresh capability sent over a service return
/// leg.
///
/// The underlying [`SessionBuilder`] is deliberately inaccessible, and every
/// method consumes this facade. One call can therefore stage exactly one
/// approved derive into one runtime-reserved slot; it cannot hide task exit,
/// `Found`, mapping, rendezvous, revoke, parking, or an unowned intermediate
/// capability ahead of the transfer. The client appends the sole send itself.
pub struct ReturnDeriver<'b> {
    builder: &'b mut SessionBuilder,
}

impl ReturnDeriver<'_> {
    /// Derive one revocable launcher-local `ChannelSpace` child from a held
    /// mint grant. This is used when founding an independently operating
    /// child shell: the child mints its own invocation-completion channels,
    /// so the parent retains no channel root or endpoint for those paths.
    pub fn channel_space_from(self, source: SlotRef) -> Result<StagedReturn, ReturnDeriveError> {
        let reservation =
            StagedCapabilityReservation::reserve().ok_or(ReturnDeriveError::NoSlots)?;
        let staged = self.builder.derive_channel_space_grant(
            source,
            crate::cap::CHANNEL_SPACE_RIGHTS,
            reservation.target(),
        )?;
        Ok(StagedReturn {
            staged,
            reservation,
        })
    }

    /// Carve an affine sub-lease window from a held supply lease.
    pub fn sublease<S: ReturnLeaseSource>(
        self,
        supply: &S,
        rights: Rights,
        offset: u64,
        len: u64,
    ) -> Result<StagedReturn, ReturnDeriveError> {
        let source = return_lease_source_sealed::Sealed::source_slot(supply)
            .ok_or(ReturnDeriveError::SourceUnavailable)?;
        let reservation =
            StagedCapabilityReservation::reserve().ok_or(ReturnDeriveError::NoSlots)?;
        let staged =
            self.builder
                .carve_sublease(source, rights, offset, len, reservation.target())?;
        Ok(StagedReturn {
            staged,
            reservation,
        })
    }

    /// Carve a detached geometry-only window from a held supply lease.
    pub fn lease_window<S: ReturnLeaseSource>(
        self,
        supply: &S,
        rights: Rights,
        offset: u64,
        len: u64,
    ) -> Result<StagedReturn, ReturnDeriveError> {
        let source = return_lease_source_sealed::Sealed::source_slot(supply)
            .ok_or(ReturnDeriveError::SourceUnavailable)?;
        let reservation =
            StagedCapabilityReservation::reserve().ok_or(ReturnDeriveError::NoSlots)?;
        let staged =
            self.builder
                .carve_window(source, rights, offset, len, reservation.target())?;
        Ok(StagedReturn {
            staged,
            reservation,
        })
    }

    /// Derive a fresh **send end** off a held `ChanSend` — send-from-send,
    /// the per-child kick a launcher conveys at a spawn (slice C: the
    /// child's `role::LOADER@0` is a leaf off the parent's own image
    /// kick, exactly as the founder derives every kick leaf off a
    /// retained send).
    ///
    /// The source is a boot-role handle (`role::SVC_SEND@…`), which has
    /// no affine token wrapper — a founder-delivered send is named by its
    /// slot alone.  The facade's constraint is unchanged where it
    /// matters: the RESULT slot is runtime-reserved (never caller-named),
    /// exactly one approved derive is staged, and a source slot that does
    /// not bind as a live `ChanSend` refuses at the kernel's derive gate,
    /// fail-closed.
    pub fn send_from(
        self,
        kick: SlotRef,
        rights: Rights,
    ) -> Result<StagedReturn, ReturnDeriveError> {
        let reservation =
            StagedCapabilityReservation::reserve().ok_or(ReturnDeriveError::NoSlots)?;
        let staged = self
            .builder
            .derive_chan_send(kick, rights, reservation.target())?;
        Ok(StagedReturn {
            staged,
            reservation,
        })
    }

    /// Derive a fresh receive end from a held boot-role receive source.
    /// Launchers use this for a child's doorbell and authenticated control
    /// lane: the parent retains the attenuated source it was endowed with,
    /// while the spawned child receives one independently revocable leaf.
    pub fn recv_from(
        self,
        recv: SlotRef,
        rights: Rights,
    ) -> Result<StagedReturn, ReturnDeriveError> {
        let reservation =
            StagedCapabilityReservation::reserve().ok_or(ReturnDeriveError::NoSlots)?;
        let staged = self
            .builder
            .derive_chan_recv(recv, rights, reservation.target())?;
        Ok(StagedReturn {
            staged,
            reservation,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReturnCommit {
    Moved,
    NotMoved(CallError),
    AfterTransfer(CallError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestParkCommit {
    /// The compact kick committed and the standing reply post delivered.
    Woke,
    /// The kick could not consume a post (the peer is already awake), or the
    /// park could not arm. The published ring record remains level-triggered;
    /// fall back to the executor's ordinary async drain.
    AsyncFallback,
    /// OCC changed under prepare; replay the same compact descriptor.
    Retry,
    /// The bounded observation deadline won. The request and standing reply
    /// post both remain live; the caller must return `Timeout` without
    /// poisoning or replacing either one.
    Deadline,
    /// The pair or its reply wait is no longer usable.
    Terminal(CallError),
}

/// Classify one compact request-kick + standing-reply-park completion.
///
/// There is one invocation. `Ok` is therefore the folded wait result and
/// proves the doorbell delivered; `WouldBlock` is deliberately ambiguous
/// between the droppable request kick and a refused park, but both require the
/// same safe action: return to the level-triggered async reply future.
fn classify_request_park_commit(committed: Committed, deadline_armed: bool) -> RequestParkCommit {
    let raw_status = committed.raw() as u16;
    let status = committed.status();
    let failing = committed.failing_index();
    if raw_status > Status::Unfounded as u16 || failing != 0 {
        return RequestParkCommit::Terminal(CallError::Refused(Status::Malformed));
    }
    match status {
        Status::Ok => RequestParkCommit::Woke,
        Status::WouldBlock => RequestParkCommit::AsyncFallback,
        Status::Conflict => RequestParkCommit::Retry,
        Status::Deadline if deadline_armed => RequestParkCommit::Deadline,
        status @ (Status::Denied
        | Status::Forged
        | Status::Revoked
        | Status::Malformed
        | Status::Deadline
        | Status::Irq
        | Status::Exited
        | Status::AltDelivery
        | Status::Unfounded) => RequestParkCommit::Terminal(CallError::Refused(status)),
    }
}

/// What authority existed before the transfer session began. A held source
/// can lose its sole local name on a late conflict; a staged source is a fresh
/// child of roots the caller still holds, and I3 discards that child together
/// with every earlier derive when the terminal send conflicts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReturnSource {
    Held,
    Staged,
}

/// Classify an authority-transfer session by what its completion proves about
/// source custody. A held-token call contains only the send; a staged call may
/// contain a bounded derive chain before its terminal send. Kept pure so every
/// status/index arm remains pinned by host tests; transport behavior itself is
/// metal-gated.
fn classify_return_commit(
    committed: Committed,
    invocations: usize,
    source: ReturnSource,
) -> ReturnCommit {
    let raw_status = committed.raw() as u16;
    let status = committed.status();
    let failing = usize::from(committed.failing_index());
    let impossible_index = if status == Status::Ok {
        failing != 0
    } else {
        failing >= invocations
    };
    if invocations == 0 || raw_status > Status::Unfounded as u16 || impossible_index {
        // Unknown status vocabulary, an empty transfer, or an impossible
        // invocation index cannot prove that the affine move did not run.
        return ReturnCommit::AfterTransfer(CallError::Refused(Status::Malformed));
    }
    match status {
        Status::Ok => ReturnCommit::Moved,
        // Prepare refusals are pure; a dispatch refusal unwinds every earlier
        // staged derive; and the send's WouldBlock/Denied paths restore or
        // never release a held twin. No peer landing becomes visible.
        status @ (Status::Denied | Status::Forged | Status::Revoked | Status::WouldBlock) => {
            ReturnCommit::NotMoved(CallError::Refused(status))
        }
        // A staged return transfers only a child created in this very
        // transaction. I3 unwinds that derive chain and restores the post on
        // conflict; the caller's held roots never crossed the custody
        // boundary, so the operation is retryable. A held source is different:
        // conflict can be observed after the peer landing was written, and its
        // sole affine name cannot be reconstructed.
        Status::Conflict if source == ReturnSource::Staged => {
            ReturnCommit::NotMoved(CallError::Refused(Status::Conflict))
        }
        // A malformed completion cannot prove which descriptor ran, and
        // WAIT-only statuses are impossible for this unflagged session. All
        // fail closed on the far side of the custody boundary.
        status @ (Status::Conflict
        | Status::Malformed
        | Status::Deadline
        | Status::Irq
        | Status::Exited
        | Status::AltDelivery
        | Status::Unfounded) => ReturnCommit::AfterTransfer(CallError::Refused(status)),
    }
}

/// A send-and-park completion folds the wait result over the invocation
/// result. `Ok` therefore proves both the move and a wake; `Revoked` may be a
/// post-commit doorbell reap and must conservatively cross the custody
/// boundary. `WouldBlock` is the rendezvous send's ordinary no-post refusal:
/// a running ring-3 task has an active scheduler park token, so the subsequent
/// standing-doorbell arm cannot produce that status on its own.
fn classify_parked_return_commit(committed: Committed, invocations: usize) -> ReturnCommit {
    let status = committed.status();
    let failing = usize::from(committed.failing_index());
    if invocations == 0
        || (status == Status::Ok && failing != 0)
        || (status != Status::Ok && failing >= invocations)
    {
        return ReturnCommit::AfterTransfer(CallError::Refused(Status::Malformed));
    }
    match status {
        Status::Ok => ReturnCommit::Moved,
        status @ (Status::Denied | Status::Forged | Status::WouldBlock) => {
            ReturnCommit::NotMoved(CallError::Refused(status))
        }
        // Unfounded is the committed founding whose assembly was discarded:
        // the bundle's affine inputs crossed the custody boundary and no
        // task came back, so it must never read as a retryable NotMoved.
        status @ (Status::Revoked
        | Status::Conflict
        | Status::Malformed
        | Status::Deadline
        | Status::Irq
        | Status::Exited
        | Status::AltDelivery
        | Status::Unfounded) => ReturnCommit::AfterTransfer(CallError::Refused(status)),
    }
}

/// Structural half of the one-shot derivation facade: exactly one approved
/// derive must have been appended, and the returned SSA value must be that
/// invocation's result. The facade supplies the semantic half by exposing
/// only lease-child methods and no builder access.
fn is_single_return_derive(builder: &SessionBuilder, checkpoint: usize, staged: Staged) -> bool {
    builder.staged() == checkpoint + 1 && builder.is_derivation_tail_since(checkpoint, staged)
}

/// What a call resolved to: the decoded response plus the delivered grant
/// authority, in grant order.
///
/// Fixed protocols receive a landing name in [`delivered`](Self::delivered)
/// and possess it through their protocol verb. Protocols whose grants outlive
/// a reusable landing receive an affine runtime owner in
/// [`retained`](Self::retained) instead.
#[derive(Debug)]
pub struct CallReply<R> {
    /// The decoded wire response.
    pub response: R,
    /// Landing slots of the delivered grants (`[0..grants]` populated).
    pub delivered: [Option<SlotRef>; MAX_GRANTS],
    /// Affine owners of retained grants (`[0..grants]` populated for channels
    /// whose [`SvcProtocol::retain_grant`] policy is true). Each owner carries
    /// the allocator reservation for its dynamically rotated landing.
    pub retained: [Option<ReturnedCapability>; MAX_GRANTS],
    /// How many grants the reply announced (and this call collected).
    pub grants: u8,
}

/// The [posts + notice] retry ceiling: `MAX_GRANTS` posts can each
/// drop out once (already standing) + one notice-only attempt + slack
/// for kernel-side races.
const RETRY_BOUND: usize = MAX_GRANTS + 4;

type ParkCommit<T> =
    fn(&T, &SessionBuilder, &ParkSpec) -> Result<Committed, <T as Transport>::Error>;

/// Opt-in ownership of an executor doorbell park. The function pointer is a
/// monomorphized trait-seam witness: ordinary protocol wrappers remain
/// generic over `T: Transport`, while only a `T: ParkTransport` constructor
/// can install the sealed blocking crossing.
struct ReplyWait<T: Transport> {
    doorbell: Doorbell,
    commit: ParkCommit<T>,
    owner: ReplyWaitOwner,
    /// A direct owner has no executor to establish/repair the post. The first
    /// call stands it; an ambiguous `WouldBlock` makes the next call verify it
    /// again before publishing another request.
    needs_stand: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplyWaitOwner {
    Executor,
    Direct,
}

fn commit_reply_park<T: ParkTransport>(
    transport: &T,
    session: &SessionBuilder,
    spec: &ParkSpec,
) -> Result<Committed, T::Error> {
    transport.commit_and_park(session, spec)
}

/// The generic ring + channel client one pair's per-service wrapper
/// owns (`&mut` calls: one outstanding request per client — the SPSC
/// reality of its rings).
pub struct SvcClient<'w, 't, T: Transport, S: SvcProtocol> {
    rings: ClientRings<'w, S>,
    ep: ClientEndpoints,
    /// Which grant posts this side believes stand (the lazily-tracked
    /// `ChanRx::standing` discipline, per channel).
    standing: [bool; MAX_GRANTS],
    /// Runtime-owned targets for retained grant channels. A consumed landing
    /// moves into the reply and leaves `None`, forcing the next post to reserve
    /// a fresh name instead of overwriting still-live authority.
    retained_landings: [Option<ReturnedCapability>; MAX_GRANTS],
    state: ClientState<S::ReqWire, S::Response>,
    transport: &'t T,
    /// Present only when the caller owns this executor's blocking park.
    reply_wait: Option<ReplyWait<T>>,
    /// The client is polled to completion by a caller-owned noop-waker loop.
    /// Such a caller has no executor wake registry and must observe the
    /// level-triggered reply/grant rings directly.
    direct_polling: bool,
    _p: PhantomData<S>,
}

#[derive(Debug)]
enum ClientState<W, R> {
    Idle,
    /// Plain-call preflight owns the pair before its first cancellable post.
    /// No request is on the wire yet; only this exact continuation may publish it.
    Preparing {
        request: W,
    },
    AwaitingReply {
        /// Exact request record already published into the uncorrelated ring.
        request: W,
    },
    /// The reply ring record has been consumed, but some announced capability
    /// deliveries are still pending. This state lives in `SvcClient`, not the
    /// future's stack, so cancellation cannot lose the decoded response or an
    /// already-collected prefix.
    Collecting {
        /// Exact encoded request whose reply was consumed. Retained through
        /// grant collection so a cancelled operation can resume with its
        /// protocol correlation intact.
        request: W,
        response: R,
        grants: u8,
        delivered: [Option<SlotRef>; MAX_GRANTS],
        retained: [Option<ReturnedCapability>; MAX_GRANTS],
        next: usize,
    },
    Poisoned {
        /// Fixed grant landings whose revoke has not yet proved closure.
        cleanup: [Option<SlotRef>; MAX_GRANTS],
        /// Dynamically named returned grants whose revoke has not yet proved
        /// closure. Keeping the affine owners here preserves both the retry
        /// handle and the allocator reservation across a poisoned pair.
        retained: [Option<ReturnedCapability>; MAX_GRANTS],
    },
}

/// Prove that a typed continuation names the exact request already owning
/// this uncorrelated pair.
///
/// This check is deliberately pure and runs before any kick, reply-ring pop,
/// or grant collection.  A caller that retained lifecycle row B therefore
/// cannot consume row A's late reply merely by invoking B's resume path.  The
/// outstanding state remains byte-for-byte intact on a mismatch.
fn exact_outstanding_request<W: PartialEq, R>(
    state: &ClientState<W, R>,
    expected: &W,
) -> Result<(), CallError> {
    match state {
        ClientState::Preparing { request }
        | ClientState::AwaitingReply { request }
        | ClientState::Collecting { request, .. }
            if request == expected =>
        {
            Ok(())
        }
        ClientState::Preparing { .. }
        | ClientState::AwaitingReply { .. }
        | ClientState::Collecting { .. } => Err(CallError::Outstanding),
        ClientState::Idle => Err(CallError::NoOutstanding),
        ClientState::Poisoned { .. } => Err(CallError::Poisoned),
    }
}

impl<'w, 't, T: Transport, S: SvcProtocol> SvcClient<'w, 't, T, S> {
    /// A client over its attached ring halves and channel wiring.
    pub fn new(transport: &'t T, rings: ClientRings<'w, S>, ep: ClientEndpoints) -> Self {
        Self {
            rings,
            ep,
            standing: [false; MAX_GRANTS],
            retained_landings: core::array::from_fn(|_| None),
            state: ClientState::Idle,
            transport,
            reply_wait: None,
            direct_polling: false,
            _p: PhantomData,
        }
    }

    /// The transport already borrowed by this client. Dynamically minted
    /// sub-clients share it without borrowing the parent client itself.
    pub fn transport(&self) -> &'t T {
        self.transport
    }

    /// The service doorbell endpoint shared by fixed and dynamically minted
    /// pairs. Pair identity remains structural in the ring topology.
    pub fn request_kick(&self) -> SlotRef {
        self.ep.kick
    }

    /// Select direct level-triggered polling for a synchronously driven
    /// client that owns no executor park.
    ///
    /// This is the explicit counterpart of a caller polling the returned
    /// future with a noop waker. It neither invents a doorbell nor touches an
    /// executor's standing post: request kicks remain ordinary droppable
    /// notices, while reply records and announced grant landings are polled
    /// until visible. Executor-owned clients should use
    /// [`with_reply_wait`](Self::with_reply_wait) instead.
    pub fn with_direct_polling(mut self) -> Self {
        debug_assert!(
            self.reply_wait.is_none(),
            "direct polling and a blocking reply wait are distinct owners"
        );
        self.direct_polling = true;
        self
    }

    /// Fuse ordinary request notification with a park on the owning
    /// executor's already-standing reply doorbell.
    ///
    /// Opt-in is structural: only the sealed [`ParkTransport`] can install
    /// the monomorphized commit function. A hand-driven bootstrap client
    /// either keeps the ordinary async kick path or explicitly chooses the
    /// direct owner below.
    pub fn with_reply_wait(mut self, reply_wait: Doorbell) -> Self
    where
        T: ParkTransport,
    {
        self.reply_wait = Some(ReplyWait {
            doorbell: reply_wait,
            commit: commit_reply_park::<T>,
            owner: ReplyWaitOwner::Executor,
            needs_stand: false,
        });
        self
    }

    /// Fuse request notification and reply waiting for a pre-executor caller
    /// that exclusively owns `reply_wait`.
    ///
    /// Unlike [`with_reply_wait`](Self::with_reply_wait), this mode establishes
    /// and repairs the standing post itself and performs no executor total-
    /// drain bookkeeping. It is therefore appropriate for startup/linker
    /// code driven by a noop waker, and must not be installed on a client that
    /// runs concurrently with an executor over the same doorbell.
    pub fn with_direct_reply_wait(mut self, reply_wait: Doorbell) -> Self
    where
        T: ParkTransport,
    {
        self.reply_wait = Some(ReplyWait {
            doorbell: reply_wait,
            commit: commit_reply_park::<T>,
            owner: ReplyWaitOwner::Direct,
            needs_stand: true,
        });
        self
    }

    /// Whether this exclusive pair can publish a fresh request.
    ///
    /// This is an observation only: callers must still use a mutable call
    /// method, whose state transition is the authority that prevents two
    /// publications on the uncorrelated SPSC reply stream.
    pub fn is_idle(&self) -> bool {
        matches!(self.state, ClientState::Idle)
    }

    /// Inert request identity retained by this SPSC owner. Protocol wrappers
    /// use it to route recovery to the operation actually awaiting completion;
    /// this borrow neither consumes a reply nor makes a poisoned pair reusable.
    pub fn outstanding_request(&self) -> Option<&S::ReqWire> {
        match &self.state {
            ClientState::Preparing { request }
            | ClientState::AwaitingReply { request }
            | ClientState::Collecting { request, .. } => Some(request),
            ClientState::Idle | ClientState::Poisoned { .. } => None,
        }
    }

    /// Whether reply progress is driven by a pre-executor caller polling this
    /// future directly rather than by the kore executor's wake registry.
    fn is_direct_waiter(&self) -> bool {
        self.reply_wait
            .as_ref()
            .is_some_and(|wait| wait.owner == ReplyWaitOwner::Direct)
    }

    /// Whether reply visibility is driven without the executor wake table.
    fn is_direct_progress(&self) -> bool {
        self.direct_polling || self.is_direct_waiter()
    }

    /// Issue one request and await its reply (+ announced grants) — the §5.6
    /// shape. See the module docs for the exact protocol.
    ///
    /// This ordinary form is deliberately unbounded. A generic timeout leaves
    /// the SPSC pair occupied, and an unaudited caller that treats the error as
    /// terminal will make every later call fail with [`CallError::Outstanding`].
    /// Operations that retain an exact continuation and define what expiry
    /// means opt in through [`call_bounded`](Self::call_bounded).
    pub async fn call(&mut self, req: &S::Request) -> Result<CallReply<S::Response>, CallError> {
        self.call_with_budget(req, None).await
    }

    /// Issue one request and expose a bounded **observation slice** of its
    /// reply wait.
    ///
    /// Expiry returns [`CallError::Timeout`] without cancelling or replacing
    /// the request. The pair remains occupied by the exact encoded request;
    /// the caller must continue it with [`resume_bounded`](Self::resume_bounded)
    /// (or deliberately park/tear down the pair). This method is explicit so
    /// only an operation with that lifecycle policy can opt in.
    pub async fn call_bounded(
        &mut self,
        req: &S::Request,
        budget_ns: u64,
    ) -> Result<CallReply<S::Response>, CallError> {
        self.call_with_budget(req, Some(budget_ns)).await
    }

    /// Start an ordinary request under an existing absolute executor deadline.
    pub async fn call_until(
        &mut self,
        req: &S::Request,
        deadline_ns: u64,
    ) -> Result<CallReply<S::Response>, CallError> {
        self.call_with_limit(req, Some(ReplyLimit::Deadline(deadline_ns)))
            .await
    }

    async fn call_with_budget(
        &mut self,
        req: &S::Request,
        budget_ns: Option<u64>,
    ) -> Result<CallReply<S::Response>, CallError> {
        self.call_with_limit(req, budget_ns.map(ReplyLimit::Duration))
            .await
    }

    async fn call_with_limit(
        &mut self,
        req: &S::Request,
        limit: Option<ReplyLimit>,
    ) -> Result<CallReply<S::Response>, CallError> {
        self.require_idle()?;
        // Direct polling intentionally owns no executor timer. Refuse the
        // bounded shape before publishing anything rather than claim a bound
        // this progress mode cannot enforce and strand an unseen live request.
        if limit.is_some() && self.is_direct_progress() {
            return Err(CallError::Wiring);
        }
        let request = S::encode_req(req);
        self.prepare_call(&request)?;
        self.state = ClientState::Preparing { request };
        self.publish_prepared(limit).await
    }

    async fn publish_prepared(
        &mut self,
        limit: Option<ReplyLimit>,
    ) -> Result<CallReply<S::Response>, CallError> {
        let ClientState::Preparing { request } = self.state else {
            return Err(CallError::NoOutstanding);
        };
        // The fixed wiring and retained landings were validated/reserved
        // before entering Preparing. Cancellation retains their actual owners.
        let grants = S::request_grants(&request);
        if let Err(error) = self.ensure_reply_wait_post().await {
            self.state = ClientState::Idle;
            return Err(error);
        }
        if let Err(error) = self.ensure_grant_posts(grants).await {
            self.state = ClientState::Idle;
            return Err(error);
        }
        // Start the observation before the request becomes visible, so the
        // compact initial kick/park is inside the same bound as every later
        // cadence wait. `None` performs no clock lookup and preserves the
        // ordinary unbounded path.
        let mut observation = ReplyObservation::begin(limit).await;
        // 2 · The ring write — happens-before the kick (§5.5).
        if self.rings.req.push(request).is_err() {
            self.state = ClientState::Idle;
            return Err(CallError::RingFull);
        }
        self.state = ClientState::AwaitingReply { request };

        // 3 · Notify + (for a configured blocking client) wait in one compact
        //     all-GPR session. An unconfigured hand-driven client keeps the
        //     notice-only kick. WouldBlock remains droppable in either form.
        if self.reply_wait.is_some() {
            if self
                .kick_and_park_published_request(observation.map(|slice| slice.deadline_ns))
                .await?
                == RequestParkCommit::Deadline
            {
                // The deadline result leaves the standing post and request
                // untouched. Let `collect_reply` perform one final ring check
                // before exposing the resumable timeout.
                if let Some(slice) = observation.as_mut() {
                    slice.note_deadline();
                }
            }
        } else {
            self.kick_published_request().await?;
        }

        self.collect_reply(observation).await
    }

    /// Continue the sole preparing or already-published request after a transport,
    /// backpressure, or cancellation boundary. The SPSC pair carries no
    /// request identifier, so publishing a replacement while this state is
    /// outstanding would let its late reply satisfy the wrong call.
    pub async fn resume(&mut self) -> Result<CallReply<S::Response>, CallError> {
        self.resume_with_limit(None).await
    }

    /// Continue the sole outstanding request for one bounded observation
    /// slice. Like [`call_bounded`](Self::call_bounded), expiry preserves the
    /// exact request and never republishes it.
    pub async fn resume_bounded(
        &mut self,
        budget_ns: u64,
    ) -> Result<CallReply<S::Response>, CallError> {
        self.resume_with_limit(Some(ReplyLimit::Duration(budget_ns)))
            .await
    }

    /// Observe the outstanding request against an existing absolute executor
    /// deadline. Preparation or rescheduling cannot extend that deadline.
    pub async fn resume_until(
        &mut self,
        deadline_ns: u64,
    ) -> Result<CallReply<S::Response>, CallError> {
        self.resume_with_limit(Some(ReplyLimit::Deadline(deadline_ns)))
            .await
    }

    async fn resume_with_limit(
        &mut self,
        limit: Option<ReplyLimit>,
    ) -> Result<CallReply<S::Response>, CallError> {
        if matches!(&self.state, ClientState::Poisoned { .. }) {
            self.retry_poison_cleanup();
            return Err(CallError::Poisoned);
        }
        // As for `call_with_budget`: preserve the already-live request and
        // tell a direct owner to use the ordinary unbounded resume.
        if limit.is_some()
            && self.is_direct_progress()
            && matches!(
                &self.state,
                ClientState::Preparing { .. }
                    | ClientState::AwaitingReply { .. }
                    | ClientState::Collecting { .. }
            )
        {
            return Err(CallError::Wiring);
        }
        if matches!(self.state, ClientState::Preparing { .. }) {
            return self.publish_prepared(limit).await;
        }
        match &self.state {
            ClientState::Preparing { .. }
            | ClientState::AwaitingReply { .. }
            | ClientState::Collecting { .. } => {}
            ClientState::Idle => return Err(CallError::NoOutstanding),
            ClientState::Poisoned { .. } => return Err(CallError::Poisoned),
        }
        // A resumed slice likewise starts before its first progress action;
        // it never republishes the correlated request.
        let observation = ReplyObservation::begin(limit).await;
        match &self.state {
            ClientState::AwaitingReply { .. } => {
                self.ensure_reply_wait_post().await?;
                self.kick_published_request().await?
            }
            ClientState::Collecting { .. } => {}
            ClientState::Preparing { .. } | ClientState::Idle | ClientState::Poisoned { .. } => {
                unreachable!()
            }
        }
        self.collect_reply(observation).await
    }

    /// Continue only when `req` is the exact preparing or published request
    /// already owned by this pair.
    ///
    /// Unlike [`resume`](Self::resume), this is safe for a shared lifecycle
    /// cleanup engine whose rows may be visited in a different order from
    /// their requests. The encoded request is compared before any observable
    /// transport action or reply consumption. A different outstanding
    /// request returns [`CallError::Outstanding`] and remains outstanding;
    /// callers must route progress back to its affine owner rather than retry
    /// blindly under the new verb.
    pub async fn resume_exact(
        &mut self,
        req: &S::Request,
    ) -> Result<CallReply<S::Response>, CallError>
    where
        S::ReqWire: PartialEq,
    {
        let expected = S::encode_req(req);
        if let Err(error) = exact_outstanding_request(&self.state, &expected) {
            if error == CallError::Poisoned {
                self.retry_poison_cleanup();
            }
            return Err(error);
        }
        self.resume().await
    }

    /// Bounded counterpart of [`resume_exact`](Self::resume_exact). The
    /// expected request is checked before any kick or reply consumption, then
    /// the same live request receives one more observation slice.
    pub async fn resume_exact_bounded(
        &mut self,
        req: &S::Request,
        budget_ns: u64,
    ) -> Result<CallReply<S::Response>, CallError>
    where
        S::ReqWire: PartialEq,
    {
        let expected = S::encode_req(req);
        if let Err(error) = exact_outstanding_request(&self.state, &expected) {
            if error == CallError::Poisoned {
                self.retry_poison_cleanup();
            }
            return Err(error);
        }
        self.resume_bounded(budget_ns).await
    }

    /// Continue the outstanding call and return the exact encoded request
    /// beside its completed reply.
    ///
    /// Protocol wrappers with several operations sharing one response variant
    /// must select their typed completion from this stored request, never from
    /// whichever resume method the caller happened to choose after
    /// cancellation. The request remains available in both `AwaitingReply`
    /// and `Collecting`, including the cancellation window after the reply
    /// record was consumed but before all grants arrived.
    pub async fn resume_correlated(
        &mut self,
    ) -> Result<(S::ReqWire, CallReply<S::Response>), CallError> {
        let request = match &self.state {
            ClientState::Preparing { request }
            | ClientState::AwaitingReply { request }
            | ClientState::Collecting { request, .. } => *request,
            ClientState::Idle => return Err(CallError::NoOutstanding),
            ClientState::Poisoned { .. } => {
                self.retry_poison_cleanup();
                return Err(CallError::Poisoned);
            }
        };
        let reply = self.resume().await?;
        Ok((request, reply))
    }

    /// Issue a request whose authority input rides the pair's return channel.
    ///
    /// Ordering is deliberately different from [`call`](Self::call): prove
    /// the single-producer request ring has space, transfer into the service's
    /// already-standing return post, then publish the ring record and kick.
    /// A protocol-declared implicit return request stops after the transfer:
    /// the capability delivery itself is its request and wakes the service's
    /// receive-any park.
    /// Only the consumer can increase free capacity between the preflight and
    /// push, so the post-transfer push cannot become full. Publishing first
    /// would let the service drain a `Register` before its capability exists.
    pub async fn call_with_return<C: ReturnCapability>(
        &mut self,
        req: &S::Request,
        return_tx: SlotRef,
        capability: C,
        rights: kore_abi::cap::Rights,
    ) -> Result<CallReply<S::Response>, ReturnCallError<C>> {
        self.begin_with_return(req, return_tx, capability, rights)
            .await?;
        self.collect_reply(None)
            .await
            .map_err(ReturnCallError::AfterTransfer)
    }

    /// Issue a request by moving one exact held affine capability through the
    /// pair's return channel. The source need not carry `DERIVE`; its current
    /// kind, rights, lineage, and tag become the service's custody unchanged.
    pub async fn call_with_exact_return<C: ReturnCapability>(
        &mut self,
        req: &S::Request,
        return_tx: SlotRef,
        capability: C,
    ) -> Result<CallReply<S::Response>, ReturnCallError<C>> {
        self.begin_with_return_mode(req, return_tx, capability, Rights::empty(), true)
            .await?;
        self.collect_reply(None)
            .await
            .map_err(ReturnCallError::AfterTransfer)
    }

    /// Issue an exact return while retaining pre-transfer ownership in the
    /// caller's state across every cancellable preflight await.
    ///
    /// The caller installs `Some(capability)` before polling this operation.
    /// Grant-post preparation may await, but borrows that option rather than
    /// moving its affine owner onto this future's stack. The actual transfer
    /// uses the transport's non-parking commit, so the option changes from
    /// `Some` to `None` without an intervening cancellation point. Once empty,
    /// the encoded request is installed in this client's correlated state
    /// before the next await. A cancelled caller can therefore decide solely
    /// from its retained option and this client's state whether to retry the
    /// source or resume the exact request; it never has to guess custody.
    pub async fn call_with_exact_retained_return<C: ReturnCapability>(
        &mut self,
        req: &S::Request,
        return_tx: SlotRef,
        capability: &mut Option<C>,
    ) -> Result<CallReply<S::Response>, RetainedReturnCallError> {
        self.begin_with_exact_retained_return(req, return_tx, capability)
            .await?;
        self.collect_reply(None)
            .await
            .map_err(RetainedReturnCallError::AfterTransfer)
    }

    /// Publish an exact retained return without collecting its reply. After
    /// success, the pair owns the encoded request until exact resumption or
    /// terminal cleanup. This split permits a caller to hand observation to
    /// another phase without moving or reconstructing the source capability.
    /// Pre-transfer cancellation still leaves the caller's option populated.
    pub async fn begin_with_exact_retained_return<C: ReturnCapability>(
        &mut self,
        req: &S::Request,
        return_tx: SlotRef,
        capability: &mut Option<C>,
    ) -> Result<(), RetainedReturnCallError> {
        self.require_idle()
            .map_err(RetainedReturnCallError::NotMoved)?;
        let request = S::encode_req(req);
        let grants = self
            .prepare_call(&request)
            .map_err(RetainedReturnCallError::NotMoved)?;
        let implicit = S::is_implicit_return_request(req);
        if !implicit && self.rings.req.capacity_left() == 0 {
            return Err(RetainedReturnCallError::NotMoved(CallError::RingFull));
        }

        // Preflight and transfer are both non-parking. The caller's option
        // changes custody before the first possible suspension.
        self.ensure_grant_posts_affine(grants)
            .map_err(RetainedReturnCallError::NotMoved)?;

        let capability_slot = capability
            .as_ref()
            .and_then(return_capability_sealed::Sealed::return_slot)
            .ok_or(RetainedReturnCallError::NotMoved(CallError::Refused(
                Status::Revoked,
            )))?;
        let mut transfer = SessionBuilder::new();
        transfer
            .send_exact(return_tx, capability_slot)
            .map_err(|_| {
                RetainedReturnCallError::NotMoved(CallError::Refused(Status::Malformed))
            })?;

        // A plain transport commit never parks. On the production trap path
        // it is one synchronous syscall, so custody and the retained option
        // transition together with no cancellable Rust future in between.
        let committed = self
            .transport
            .commit_affine(&transfer)
            .map_err(|_| RetainedReturnCallError::NotMoved(CallError::Transport))?;
        match classify_return_commit(committed, transfer.staged(), ReturnSource::Held) {
            ReturnCommit::Moved => {
                let moved = capability
                    .take()
                    .expect("retained exact-return source disappeared before commit");
                return_capability_sealed::Sealed::on_proven_moved(moved);
            }
            ReturnCommit::NotMoved(error) => {
                return Err(RetainedReturnCallError::NotMoved(error));
            }
            ReturnCommit::AfterTransfer(error) => {
                // The source may have moved. Consume the sole local witness
                // and quarantine this uncorrelated pair exactly as the legacy
                // consuming API does for the same completion class.
                drop(capability.take());
                self.poison();
                return Err(RetainedReturnCallError::AfterTransfer(error));
            }
        }

        if !implicit && self.rings.req.push(request).is_err() {
            self.poison();
            return Err(RetainedReturnCallError::AfterTransfer(CallError::RingFull));
        }
        self.state = ClientState::AwaitingReply { request };
        if !implicit {
            self.kick_published_request()
                .await
                .map_err(RetainedReturnCallError::AfterTransfer)?;
        }
        Ok(())
    }

    /// Issue a request by moving one exact capability already held in this
    /// task's arena through the pair's return channel.
    ///
    /// This is the held-slot twin of [`call_with_exact_return`](Self::call_with_exact_return).
    /// It exists for an input that landed as one member of a larger atomic
    /// transaction: the caller owns the arena name but has no second affine
    /// Rust wrapper to manufacture. The exact move preserves the input's
    /// attenuated rights, and the request is not published until the kernel
    /// proves that move committed.
    pub async fn call_with_exact_held_return(
        &mut self,
        req: &S::Request,
        return_tx: SlotRef,
        capability: SlotRef,
    ) -> Result<CallReply<S::Response>, StagedReturnError> {
        self.require_idle()
            .map_err(StagedReturnError::NotTransferred)?;
        let request = S::encode_req(req);
        let grants = self
            .prepare_call(&request)
            .map_err(StagedReturnError::NotTransferred)?;
        let implicit = S::is_implicit_return_request(req);
        if !implicit && self.rings.req.capacity_left() == 0 {
            return Err(StagedReturnError::NotTransferred(CallError::RingFull));
        }
        self.ensure_grant_posts_affine(grants)
            .map_err(StagedReturnError::NotTransferred)?;

        let mut transfer = SessionBuilder::new();
        transfer.send_exact(return_tx, capability).map_err(|_| {
            StagedReturnError::NotTransferred(CallError::Refused(Status::Malformed))
        })?;
        let committed = self
            .transport
            .commit_affine(&transfer)
            .map_err(|_| StagedReturnError::NotTransferred(CallError::Transport))?;
        match classify_return_commit(committed, transfer.staged(), ReturnSource::Held) {
            ReturnCommit::Moved => {}
            ReturnCommit::NotMoved(error) => {
                return Err(StagedReturnError::NotTransferred(error));
            }
            ReturnCommit::AfterTransfer(error) => {
                self.poison();
                return Err(StagedReturnError::AfterTransfer(error));
            }
        }

        if !implicit && self.rings.req.push(request).is_err() {
            self.poison();
            return Err(StagedReturnError::AfterTransfer(CallError::RingFull));
        }
        self.state = ClientState::AwaitingReply { request };
        if !implicit {
            self.kick_published_request()
                .await
                .map_err(StagedReturnError::AfterTransfer)?;
        }
        self.collect_reply(None)
            .await
            .map_err(StagedReturnError::AfterTransfer)
    }

    /// Publish a return-bearing request without awaiting its reply.
    ///
    /// This is the split-phase form used when the caller itself drives an
    /// embedded [`ServeHost`](crate::svc::ServeHost): the capability move,
    /// request publication, and kick retain the same ordering and affine
    /// settlement as [`call_with_return`](Self::call_with_return), while
    /// [`resume`](Self::resume) collects the reply after the host has drained
    /// the ring. The client remains non-idle between the two calls, so no
    /// second uncorrelated request can overtake the first.
    pub async fn begin_with_return<C: ReturnCapability>(
        &mut self,
        req: &S::Request,
        return_tx: SlotRef,
        capability: C,
        rights: kore_abi::cap::Rights,
    ) -> Result<(), ReturnCallError<C>> {
        self.begin_with_return_mode(req, return_tx, capability, rights, false)
            .await
    }

    async fn begin_with_return_mode<C: ReturnCapability>(
        &mut self,
        req: &S::Request,
        return_tx: SlotRef,
        capability: C,
        rights: kore_abi::cap::Rights,
        exact: bool,
    ) -> Result<(), ReturnCallError<C>> {
        if let Err(error) = self.require_idle() {
            return Err(ReturnCallError::NotMoved { error, capability });
        }
        let request = S::encode_req(req);
        let grants = match self.prepare_call(&request) {
            Ok(grants) => grants,
            Err(error) => return Err(ReturnCallError::NotMoved { error, capability }),
        };
        let implicit = S::is_implicit_return_request(req);
        if !implicit && self.rings.req.capacity_left() == 0 {
            return Err(ReturnCallError::NotMoved {
                error: CallError::RingFull,
                capability,
            });
        }
        if let Err(error) = self.ensure_grant_posts_affine(grants) {
            return Err(ReturnCallError::NotMoved { error, capability });
        }

        let Some(capability_slot) = return_capability_sealed::Sealed::return_slot(&capability)
        else {
            return Err(ReturnCallError::NotMoved {
                error: CallError::Refused(Status::Revoked),
                capability,
            });
        };
        let mut transfer = SessionBuilder::new();
        let built = if exact {
            transfer.send_exact(return_tx, capability_slot)
        } else {
            transfer.send(return_tx, capability_slot, rights)
        };
        if built.is_err() {
            return Err(ReturnCallError::NotMoved {
                error: CallError::Refused(Status::Malformed),
                capability,
            });
        }
        let committed = match self.transport.commit_affine(&transfer) {
            Ok(committed) => committed,
            Err(_) => {
                return Err(ReturnCallError::NotMoved {
                    error: CallError::Transport,
                    capability,
                });
            }
        };
        match classify_return_commit(committed, transfer.staged(), ReturnSource::Held) {
            ReturnCommit::Moved => {}
            ReturnCommit::NotMoved(error) => {
                return Err(ReturnCallError::NotMoved { error, capability });
            }
            ReturnCommit::AfterTransfer(error) => {
                drop(capability);
                // No request was published, but the standing return post may
                // have been consumed. This pair has no correlation id with
                // which to distinguish that landing from a later convey.
                self.poison();
                return Err(ReturnCallError::AfterTransfer(error));
            }
        }
        // The successful rendezvous moved and scrubbed the source record.
        // Settle the affine witness now; a dynamically reserved owner releases
        // only its allocator reservation, while plain possession witnesses
        // simply disappear.
        return_capability_sealed::Sealed::on_proven_moved(capability);

        if !implicit {
            // Guaranteed by the capacity preflight and exclusive Producer
            // borrow: this task is the sole tail writer, while the peer can
            // only advance head and create more space.
            if self.rings.req.push(request).is_err() {
                self.poison();
                return Err(ReturnCallError::AfterTransfer(CallError::RingFull));
            }
        }
        self.state = ClientState::AwaitingReply { request };
        if !implicit {
            self.kick_published_request()
                .await
                .map_err(ReturnCallError::AfterTransfer)?;
        }
        Ok(())
    }

    /// Publish one request whose executable authority, exact supply, process
    /// context, and one-shot witness commit in a single session.
    ///
    /// The request record is written first, but an admission-gated serving
    /// pair cannot dequeue it until the notice sent through `admission`
    /// lands. The two transfers, proof, and request kick therefore become
    /// observable together. The executable endpoint itself remains with the
    /// caller and can authorize later invocations.
    pub async fn call_with_invocation_bundle<
        A: ReturnCapability,
        B: ReturnCapability,
        C: ReturnCapability,
    >(
        &mut self,
        req: &S::Request,
        first: A,
        first_rights: Rights,
        executable: SlotRef,
        second: B,
        second_rights: Rights,
        process_tx: SlotRef,
        proof_tx: SlotRef,
        third: C,
        third_rights: Rights,
        third_tx: SlotRef,
        auxiliaries: [Option<InvocationAuxiliary>; MAX_INVOCATION_AUXILIARIES],
        auxiliary_tx: [SlotRef; MAX_INVOCATION_AUXILIARIES],
    ) -> Result<CallReply<S::Response>, AffineBundleCallError<A, B, C>> {
        self.begin_invocation_bundle(
            req,
            first,
            first_rights,
            executable,
            second,
            second_rights,
            process_tx,
            proof_tx,
            third,
            third_rights,
            third_tx,
            auxiliaries,
            auxiliary_tx,
        )?;
        self.resume()
            .await
            .map_err(AffineBundleCallError::AfterTransfer)
    }

    /// Commit all invocation inputs and install reply correlation without any
    /// suspension. The caller then observes this same request with `resume` or
    /// `resume_bounded`; cancellation never requires republishing its inputs.
    pub fn begin_invocation_bundle<
        A: ReturnCapability,
        B: ReturnCapability,
        C: ReturnCapability,
    >(
        &mut self,
        req: &S::Request,
        first: A,
        first_rights: Rights,
        executable: SlotRef,
        second: B,
        second_rights: Rights,
        process_tx: SlotRef,
        proof_tx: SlotRef,
        third: C,
        third_rights: Rights,
        third_tx: SlotRef,
        auxiliaries: [Option<InvocationAuxiliary>; MAX_INVOCATION_AUXILIARIES],
        auxiliary_tx: [SlotRef; MAX_INVOCATION_AUXILIARIES],
    ) -> Result<(), AffineBundleCallError<A, B, C>> {
        macro_rules! not_moved {
            ($error:expr) => {
                AffineBundleCallError::NotMoved {
                    error: $error,
                    first,
                    second,
                    third,
                    auxiliaries,
                }
            };
        }

        if let Err(error) = self.require_idle() {
            return Err(not_moved!(error));
        }
        // An absent or locally retired rendezvous destination is a
        // deterministic local refusal. Check every endpoint that this bundle
        // will address before publishing its request record; otherwise the
        // kernel would correctly roll custody back after publication and the
        // unrewindable SPSC record would force fail-closed pair poisoning.
        if ![executable, process_tx, proof_tx, third_tx]
            .into_iter()
            .all(live_send_destination)
            || auxiliaries
                .iter()
                .zip(auxiliary_tx)
                .any(|(auxiliary, destination)| {
                    auxiliary.is_some() && !live_send_destination(destination)
                })
        {
            return Err(not_moved!(CallError::Refused(Status::Revoked)));
        }
        let request = S::encode_req(req);
        let grants = match self.prepare_call(&request) {
            Ok(grants) => grants,
            Err(error) => return Err(not_moved!(error)),
        };
        if self.rings.req.capacity_left() == 0 {
            return Err(not_moved!(CallError::RingFull));
        }
        if let Err(error) = self.ensure_grant_posts_affine(grants) {
            return Err(not_moved!(error));
        }
        let first_slot = match preflight_return_send(&first, first_rights) {
            Ok(slot) => slot,
            Err(error) => return Err(not_moved!(error)),
        };
        let second_slot = match preflight_return_send(&second, second_rights) {
            Ok(slot) => slot,
            Err(error) => return Err(not_moved!(error)),
        };
        let third_slot = match preflight_return_send(&third, third_rights) {
            Ok(slot) => slot,
            Err(error) => return Err(not_moved!(error)),
        };
        let mut source_slots = [SlotRef::from_wire(0); 3 + MAX_INVOCATION_AUXILIARIES];
        source_slots[0] = first_slot;
        source_slots[1] = second_slot;
        source_slots[2] = third_slot;
        let mut source_count = 3;
        for auxiliary in auxiliaries.iter().flatten() {
            let checked = match auxiliary.transfer {
                AuxiliaryTransfer::Derived => {
                    preflight_return_send(&auxiliary.capability, auxiliary.rights)
                }
                AuxiliaryTransfer::Exact => preflight_exact_return(&auxiliary.capability),
            };
            let slot = match checked {
                Ok(slot) => slot,
                Err(error) => return Err(not_moved!(error)),
            };
            source_slots[source_count] = slot;
            source_count += 1;
        }
        if !invocation_sources_are_distinct(&source_slots[..source_count]) {
            return Err(not_moved!(CallError::Refused(Status::Malformed)));
        }
        let mut witness = match StagedCapabilityReservation::reserve() {
            Some(reservation) => reservation,
            None => return Err(not_moved!(CallError::Backpressure)),
        };

        let mut transfer = SessionBuilder::new();
        let mut malformed = transfer
            .invocation_witness(executable, first_slot, witness.target())
            .and_then(|proof| {
                transfer.send(
                    proof_tx,
                    proof,
                    Rights::from_bits_truncate(kore_abi::session::INVOCATION_WITNESS_RIGHTS.bits()),
                )
            })
            .and_then(|_| transfer.send(process_tx, second_slot, second_rights))
            .and_then(|_| transfer.send(third_tx, third_slot, third_rights))
            .and_then(|_| transfer.send(executable, first_slot, first_rights))
            .is_err();
        for (auxiliary, destination) in auxiliaries.iter().zip(auxiliary_tx) {
            if malformed {
                break;
            }
            let Some(auxiliary) = auxiliary.as_ref() else {
                continue;
            };
            // Preflight above proved this exact affine owner's live local
            // name and transfer allowance before any request publication.
            let Some(slot) = return_capability_sealed::Sealed::return_slot(&auxiliary.capability)
            else {
                return Err(not_moved!(CallError::Refused(Status::Revoked)));
            };
            malformed = match auxiliary.transfer {
                AuxiliaryTransfer::Derived => {
                    transfer.send(destination, slot, auxiliary.rights).is_err()
                }
                AuxiliaryTransfer::Exact => transfer.send_exact(destination, slot).is_err(),
            };
        }
        if malformed {
            return Err(not_moved!(CallError::Refused(Status::Malformed)));
        }

        // Exclusive producer ownership plus the capacity preflight makes
        // this infallible. Crucially, every deterministic rights, liveness,
        // duplicate-source, and descriptor-shape check completed before this
        // publish. The receiver-authored admission gate still keeps the
        // record unobservable until the complete session below commits.
        if self.rings.req.push(request).is_err() {
            return Err(not_moved!(CallError::RingFull));
        }
        witness.begin_transfer();
        // Publication, the non-parking affine commit and installation of
        // reply custody below have no intervening cancellable await. The
        // admission proof keeps the prewritten request unservable until the
        // complete transfer commits.
        let committed = match self.transport.commit_affine(&transfer) {
            Ok(committed) => committed,
            Err(_) => {
                witness.quarantine();
                drop(first);
                drop(second);
                drop(third);
                drop(auxiliaries);
                self.poison();
                return Err(AffineBundleCallError::AfterTransfer(CallError::Transport));
            }
        };
        match classify_return_commit(committed, transfer.staged(), ReturnSource::Held) {
            ReturnCommit::Moved => {
                witness.settle_proven();
                return_capability_sealed::Sealed::on_proven_moved(first);
                return_capability_sealed::Sealed::on_proven_moved(second);
                return_capability_sealed::Sealed::on_proven_moved(third);
                for auxiliary in auxiliaries.into_iter().flatten() {
                    return_capability_sealed::Sealed::on_proven_moved(auxiliary.capability);
                }
                self.state = ClientState::AwaitingReply { request };
                // A request kick is a droppable wake hint, not invocation
                // custody. Fusing it into the affine transaction lets a
                // lawfully busy notice channel roll back every bundle member.
                // The receiver's `request_ready` gate keeps this prewritten
                // record invisible until the complete transfer above lands.
                Ok(())
            }
            ReturnCommit::NotMoved(error) => {
                witness.settle_proven();
                // The kernel proved custody stayed local, but the SPSC ring
                // has already published the correlated record and exposes no
                // producer rollback once its tail is visible. Leaving that
                // record paired with a future admission would cross-wire two
                // launches, so this genuinely post-publication case remains
                // fail-closed until the protocol grows an authenticated
                // request-cancellation acknowledgement.
                self.poison();
                Err(not_moved!(error))
            }
            ReturnCommit::AfterTransfer(error) => {
                witness.quarantine();
                drop(first);
                drop(second);
                drop(third);
                drop(auxiliaries);
                self.poison();
                Err(AffineBundleCallError::AfterTransfer(error))
            }
        }
    }

    /// Issue a protocol-declared implicit return request as one L4-style
    /// **send-and-wait** crossing.
    ///
    /// The capability send and the caller's already-standing reply doorbell
    /// park share one session. A normal round therefore never returns to
    /// userland between request delivery and reply wake. The descriptor is
    /// only 40 bytes (header + one send + one `ParkReq`), so x86-64 carries it
    /// wholly in its six caller-saved payload registers. After wake, this
    /// method consumes and re-posts the shared executor bell before returning
    /// to the caller and schedules the executor's required total drain.
    pub async fn call_with_return_parked<C: ReturnCapability>(
        &mut self,
        req: &S::Request,
        return_tx: SlotRef,
        capability: C,
        rights: kore_abi::cap::Rights,
        reply_wait: Doorbell,
    ) -> Result<CallReply<S::Response>, ReturnCallError<C>>
    where
        T: ParkTransport,
    {
        if !S::is_implicit_return_request(req) {
            return Err(ReturnCallError::NotMoved {
                error: CallError::Refused(Status::Malformed),
                capability,
            });
        }
        if let Err(error) = self.require_idle() {
            return Err(ReturnCallError::NotMoved { error, capability });
        }
        let request = S::encode_req(req);
        let grants = match self.prepare_call(&request) {
            Ok(grants) => grants,
            Err(error) => return Err(ReturnCallError::NotMoved { error, capability }),
        };
        if let Err(error) = self.ensure_grant_posts_affine(grants) {
            return Err(ReturnCallError::NotMoved { error, capability });
        }

        let Some(capability_slot) = return_capability_sealed::Sealed::return_slot(&capability)
        else {
            return Err(ReturnCallError::NotMoved {
                error: CallError::Refused(Status::Revoked),
                capability,
            });
        };
        let mut transfer = SessionBuilder::new();
        if transfer.send(return_tx, capability_slot, rights).is_err() {
            return Err(ReturnCallError::NotMoved {
                error: CallError::Refused(Status::Malformed),
                capability,
            });
        }
        transfer.park_standing(
            reply_wait.chan,
            reply_wait.land.index() as u16,
            reply_wait.recv_tag,
        );
        let committed = match self.transport.commit_and_park(
            &transfer,
            &ParkSpec {
                chan: reply_wait.chan,
                land: reply_wait.land,
                alternate: None,
                deadline_ns: None,
                irq: false,
            },
        ) {
            Ok(committed) => committed,
            Err(_) => {
                return Err(ReturnCallError::NotMoved {
                    error: CallError::Transport,
                    capability,
                });
            }
        };
        match classify_parked_return_commit(committed, transfer.staged()) {
            ReturnCommit::Moved => {}
            ReturnCommit::NotMoved(error) => {
                return Err(ReturnCallError::NotMoved { error, capability });
            }
            ReturnCommit::AfterTransfer(error) => {
                drop(capability);
                self.poison();
                return Err(ReturnCallError::AfterTransfer(error));
            }
        }
        return_capability_sealed::Sealed::on_proven_moved(capability);

        self.state = ClientState::AwaitingReply { request };
        if !settle_inline_doorbell(self.transport, reply_wait, committed.clock_ms()).await {
            self.poison();
            return Err(ReturnCallError::AfterTransfer(CallError::Wire));
        }
        self.collect_reply(None)
            .await
            .map_err(ReturnCallError::AfterTransfer)
    }

    /// Issue a request whose authority input is freshly derived and sent in
    /// one bounded session.
    ///
    /// `stage` receives one consumed [`ReturnDeriver`], not the underlying
    /// session builder. It can append exactly one approved derive into a
    /// runtime-reserved result and returns the opaque [`StagedReturn`] owner.
    /// This method verifies that single-invocation provenance, appends the
    /// rendezvous send, and commits the two-op transaction before publishing
    /// the ring record. Therefore the service cannot observe the request
    /// before its retained return post owns the capability, and neither a raw
    /// result name nor an unowned intermediate can escape the facade.
    pub async fn call_with_staged_return<F>(
        &mut self,
        req: &S::Request,
        return_tx: SlotRef,
        rights: kore_abi::cap::Rights,
        stage: F,
    ) -> Result<CallReply<S::Response>, StagedReturnError>
    where
        F: FnOnce(ReturnDeriver<'_>) -> Result<StagedReturn, ReturnDeriveError>,
    {
        self.require_idle()
            .map_err(StagedReturnError::NotTransferred)?;
        let request = S::encode_req(req);
        let grants = self
            .prepare_call(&request)
            .map_err(StagedReturnError::NotTransferred)?;
        let implicit = S::is_implicit_return_request(req);
        if !implicit && self.rings.req.capacity_left() == 0 {
            return Err(StagedReturnError::NotTransferred(CallError::RingFull));
        }
        self.ensure_grant_posts_affine(grants)
            .map_err(StagedReturnError::NotTransferred)?;

        let mut transfer = SessionBuilder::new();
        let checkpoint = transfer.staged();
        let mut capability = {
            let deriver = ReturnDeriver {
                builder: &mut transfer,
            };
            stage(deriver).map_err(|error| StagedReturnError::NotTransferred(error.call_error()))?
        };
        if !is_single_return_derive(&transfer, checkpoint, capability.staged) {
            return Err(StagedReturnError::NotTransferred(CallError::Refused(
                Status::Malformed,
            )));
        }
        transfer
            .send(return_tx, capability.staged, rights)
            .map_err(|_| {
                StagedReturnError::NotTransferred(CallError::Refused(Status::Malformed))
            })?;

        // The staged reservation crosses one non-parking commit and is
        // settled before the next await. An impossible completion still
        // quarantines it because local occupancy cannot then be proved.
        capability.begin_transfer();
        let committed = match self.transport.commit_affine(&transfer) {
            Ok(committed) => committed,
            Err(_) => {
                // Transport::Error means "could not commit", so the reserved
                // result stayed vacant and can be reopened.
                capability.settle_proven();
                return Err(StagedReturnError::NotTransferred(CallError::Transport));
            }
        };
        match classify_return_commit(committed, transfer.staged(), ReturnSource::Staged) {
            ReturnCommit::Moved => capability.settle_proven(),
            ReturnCommit::NotMoved(error) => {
                capability.settle_proven();
                return Err(StagedReturnError::NotTransferred(error));
            }
            ReturnCommit::AfterTransfer(error) => {
                // The reply-less transfer may have consumed exec's standing
                // post. Without a request id, later calls cannot distinguish
                // that landing. Keep the allocator reservation quarantined as
                // well: an impossible completion cannot prove the local result
                // slot vacant.
                capability.quarantine();
                self.poison();
                return Err(StagedReturnError::AfterTransfer(error));
            }
        }

        if !implicit {
            if self.rings.req.push(request).is_err() {
                self.poison();
                return Err(StagedReturnError::AfterTransfer(CallError::RingFull));
            }
        }
        self.state = ClientState::AwaitingReply { request };
        if !implicit {
            self.kick_published_request()
                .await
                .map_err(StagedReturnError::AfterTransfer)?;
        }
        self.collect_reply(None)
            .await
            .map_err(StagedReturnError::AfterTransfer)
    }

    /// Await, decode, and possess one already-issued call's reply.
    async fn collect_reply(
        &mut self,
        observation: Option<ReplyObservation>,
    ) -> Result<CallReply<S::Response>, CallError> {
        let awaiting_request = match &self.state {
            ClientState::AwaitingReply { request } => Some(*request),
            _ => None,
        };
        if let Some(request) = awaiting_request {
            // 3 · The reply record. The service's doorbell normally drives
            //     the re-poll; the folded deadline closes the narrow
            //     drain→repost race in which a lawful droppable kick can
            //     coincide with publication. Re-kicking is safe because the
            //     ring record is level-triggered and this pair has exactly
            //     one correlated outstanding request.
            let wire = if self.is_direct_progress() {
                // A direct owner is deliberately driven with a noop waker by
                // the startup linker. `next_wake_for` requires an installed
                // kore executor and would panic (then park in the stub panic
                // handler). The ring is level-triggered, so repeated direct
                // polls are the complete progress mechanism here.
                self.rings.rep.next().await
            } else if let Some(wire) = self.rings.rep.pop() {
                // An already-visible reply needs neither a timer nor a wake.
                // Preserve that immediately-ready shape for executor-free
                // blocking adapters; only an empty ring may enter the
                // executor-clocked re-kick loop below.
                wire
            } else {
                // An explicit bounded observation prefers an absolute
                // deadline from the clock; where the clock cannot answer, it
                // counts fixed re-kick cadences instead. Ordinary calls carry
                // `None` and retain the historical unbounded wait. The
                // cadence's absolute executor deadline stays put across
                // unrelated wakes, so neither the re-kick nor a bounded
                // fallback tick can be starved by traffic. A target without
                // the zero-trap clock conservatively re-kicks on every wake;
                // only a reached cadence spends its fallback budget.
                // Re-kick pacing is independent of whether this call chose a
                // bounded observation. Ordinary unbounded calls still use the
                // zero-trap clock when it exists instead of treating every
                // unrelated wake as a clockless reason to kick again.
                let absolute_clock = crate::clock::uptime().is_ok();
                let mut ticks: u64 = 0;
                let mut rekick = ReplyRekickCadence::after(crate::executor::clock_ns().await);
                loop {
                    if let Some(wire) = self.rings.rep.pop() {
                        break wire;
                    }
                    let executor_now_ns = crate::executor::clock_ns().await;
                    if observation.is_some_and(|slice| slice.expired(executor_now_ns)) {
                        // The request stays published and the state stays
                        // `AwaitingReply`, so `resume` can wait again.
                        return Err(CallError::Timeout);
                    }
                    let wait_deadline = observation.map_or(rekick.deadline(), |slice| {
                        slice.next_deadline(rekick.deadline())
                    });
                    crate::executor::next_wake_until(wait_deadline).await;
                    let executor_now_ns = crate::executor::clock_ns().await;
                    let elapsed_tick = rekick.due(executor_now_ns);
                    let expired = observation.is_some_and(|slice| slice.expired(executor_now_ns))
                        || reply_wait_expired(
                            observation.map(|slice| slice.budget_ns),
                            observation.and_then(|slice| slice.started_ns),
                            crate::clock::uptime().ok().map(|now| now.as_nanos()),
                            elapsed_tick,
                            &mut ticks,
                        );
                    if expired {
                        // The request stays published and the state stays
                        // `AwaitingReply`, so `resume` can wait again.
                        return Err(CallError::Timeout);
                    }
                    if reply_rekick_needed(elapsed_tick, absolute_clock) {
                        self.kick_published_request().await?;
                    }
                    if elapsed_tick {
                        rekick.rearm(crate::executor::clock_ns().await);
                    }
                }
            };
            let Some((response, grants)) = S::decode_rep_for(&request, &wire) else {
                self.poison_after_reply();
                return Err(CallError::Wire);
            };
            if usize::from(grants) > S::request_grants(&request)
                || usize::from(grants) > S::REPLY_GRANTS
                || usize::from(grants) > MAX_GRANTS
            {
                self.poison_after_reply();
                return Err(CallError::Wire);
            }
            if !S::reply_matches(&request, &response, grants) {
                self.poison_after_reply();
                return Err(CallError::Wire);
            }
            self.state = ClientState::Collecting {
                request,
                response,
                grants,
                delivered: [None; MAX_GRANTS],
                retained: core::array::from_fn(|_| None),
                next: 0,
            };
        }

        // 4 · Collect exactly the announced deliveries from the pre-pinned
        //     posts. Every completed prefix update is written into `state`
        //     before the next await, making cancellation/resume lossless.
        loop {
            let (next, grants) = match &self.state {
                ClientState::Collecting { next, grants, .. } => (*next, usize::from(*grants)),
                ClientState::Poisoned { .. } => return Err(CallError::Poisoned),
                _ => return Err(CallError::NoOutstanding),
            };
            if next == grants {
                let completed = core::mem::replace(&mut self.state, ClientState::Idle);
                let ClientState::Collecting {
                    response,
                    grants,
                    delivered,
                    retained,
                    ..
                } = completed
                else {
                    self.poison_after_reply();
                    return Err(CallError::Poisoned);
                };
                return Ok(CallReply {
                    response,
                    delivered,
                    retained,
                    grants,
                });
            }

            let Some(pin) = self.ep.grants[next] else {
                self.poison_collecting();
                return Err(CallError::Wiring);
            };
            let retain = S::retain_grant(next);
            if !pin.matches_retention(retain) {
                self.poison_collecting();
                return Err(CallError::Wiring);
            }
            let landing = if retain {
                let Some(owner) = self.retained_landings[next].as_ref() else {
                    self.poison_collecting();
                    return Err(CallError::Wiring);
                };
                owner.target()
            } else {
                let Some(landing) = pin.fixed_landing() else {
                    self.poison_collecting();
                    return Err(CallError::Wiring);
                };
                landing
            };
            let delivery = if self.is_direct_progress() {
                core::future::poll_fn(|_cx| match verbs::take_delivery(landing) {
                    Some(delivery) => Poll::Ready(delivery),
                    None => Poll::Pending,
                })
                .await
            } else {
                loop {
                    if let Some(delivery) = verbs::take_delivery(landing) {
                        break delivery;
                    }
                    // Grant transfers precede the reply, but a delivery may
                    // become visible just after the reply kick was consumed.
                    // A bounded folded wake makes that visibility race
                    // self-healing without a polling thread.
                    crate::executor::next_wake_for(1_000_000).await;
                }
            };
            if matches!(delivery, RawDelivery::Notice(..)) {
                // The post was consumed, so a retained target that never
                // became authority may safely release its reservation.
                self.standing[next] = false;
                if retain {
                    if let Some(mut owner) = self.retained_landings[next].take() {
                        owner.disarm_post();
                        drop(owner);
                    }
                }
                self.poison_collecting();
                return Err(CallError::Wire);
            }

            self.standing[next] = false;
            let retained_owner = if retain {
                let Some(mut owner) = self.retained_landings[next].take() else {
                    self.poison_collecting();
                    return Err(CallError::Wiring);
                };
                owner.activate();
                Some(owner)
            } else {
                None
            };
            let ClientState::Collecting {
                delivered,
                retained,
                next: progress,
                ..
            } = &mut self.state
            else {
                self.poison_after_reply();
                return Err(CallError::Poisoned);
            };
            if let Some(owner) = retained_owner {
                retained[next] = Some(owner);
            } else {
                delivered[next] = Some(landing);
            }
            *progress = next + 1;
        }
    }

    fn require_idle(&mut self) -> Result<(), CallError> {
        match &self.state {
            ClientState::Idle => Ok(()),
            ClientState::Preparing { .. }
            | ClientState::AwaitingReply { .. }
            | ClientState::Collecting { .. } => Err(CallError::Outstanding),
            ClientState::Poisoned { .. } => {
                self.retry_poison_cleanup();
                Err(CallError::Poisoned)
            }
        }
    }

    /// Validate immutable wiring and reserve every dynamic landing before a
    /// request (or its affine return input) becomes externally visible.
    fn prepare_call(&mut self, request: &S::ReqWire) -> Result<usize, CallError> {
        let grants = S::request_grants(request);
        if S::REPLY_GRANTS > MAX_GRANTS || grants > S::REPLY_GRANTS {
            return Err(CallError::Wiring);
        }
        for k in 0..grants {
            let pin = self.ep.grants[k].ok_or(CallError::Wiring)?;
            let retain = S::retain_grant(k);
            if !pin.matches_retention(retain) {
                return Err(CallError::Wiring);
            }
            if !retain {
                continue;
            }
            if self.retained_landings[k].is_none() {
                if self.standing[k] {
                    return Err(CallError::Wiring);
                }
                self.retained_landings[k] = ReturnedCapability::reserve();
                if self.retained_landings[k].is_none() {
                    return Err(CallError::Backpressure);
                }
            }
        }
        Ok(grants)
    }

    fn poison_collecting(&mut self) {
        self.poison_after_reply();
    }

    fn poison_after_reply(&mut self) {
        let state = core::mem::replace(
            &mut self.state,
            ClientState::Poisoned {
                cleanup: [None; MAX_GRANTS],
                retained: core::array::from_fn(|_| None),
            },
        );
        let mut cleanup = [None; MAX_GRANTS];
        let mut retained = core::array::from_fn(|_| None);
        match state {
            ClientState::Collecting {
                delivered,
                retained: prior,
                ..
            } => {
                for slot in delivered.into_iter().flatten() {
                    Self::remember_cleanup(&mut cleanup, slot);
                }
                retained = prior;
            }
            ClientState::Poisoned {
                cleanup: prior,
                retained: prior_retained,
            } => {
                for slot in prior.into_iter().flatten() {
                    Self::remember_cleanup(&mut cleanup, slot);
                }
                retained = prior_retained;
            }
            _ => {}
        }
        // The service commits its grant session before publishing the reply.
        // If reply metadata itself is malformed, scan every protocol landing:
        // the wire count cannot be trusted to enumerate the live records.
        for k in 0..S::REPLY_GRANTS.min(MAX_GRANTS) {
            if S::retain_grant(k) {
                if let Some(owner) = self.take_retained_delivery(k) {
                    Self::remember_retained(&mut retained, owner);
                }
                continue;
            }
            let Some(pin) = self.ep.grants[k] else {
                continue;
            };
            let Some(landing) = pin.fixed_landing() else {
                continue;
            };
            match verbs::take_delivery(landing) {
                Some(RawDelivery::Cap) => {
                    self.standing[k] = false;
                    Self::remember_cleanup(&mut cleanup, landing);
                }
                Some(RawDelivery::Notice(..)) => self.standing[k] = false,
                None => {}
            }
        }
        Self::settle_cleanup(&mut cleanup);
        Self::settle_retained(&mut retained);
        self.state = ClientState::Poisoned { cleanup, retained };
    }

    fn remember_cleanup(cleanup: &mut [Option<SlotRef>; MAX_GRANTS], slot: SlotRef) {
        if cleanup.iter().flatten().any(|known| *known == slot) {
            return;
        }
        if let Some(vacant) = cleanup.iter_mut().find(|entry| entry.is_none()) {
            *vacant = Some(slot);
        }
    }

    fn settle_cleanup(cleanup: &mut [Option<SlotRef>; MAX_GRANTS]) {
        for entry in cleanup.iter_mut() {
            let Some(slot) = *entry else { continue };
            if verbs::retire_slot_outcome(slot).is_closed() {
                *entry = None;
            }
        }
    }

    fn settle_retained(retained: &mut [Option<ReturnedCapability>; MAX_GRANTS]) {
        for entry in retained.iter_mut() {
            let Some(owner) = entry.as_mut() else {
                continue;
            };
            if owner.try_retire_outcome().is_closed() {
                *entry = None;
            }
        }
    }

    fn remember_retained(
        retained: &mut [Option<ReturnedCapability>; MAX_GRANTS],
        owner: ReturnedCapability,
    ) {
        if let Some(dst) = retained.iter_mut().find(|entry| entry.is_none()) {
            *dst = Some(owner);
        } else {
            // The array is protocol-wide MAX_GRANTS, so this requires corrupt
            // duplicate accounting. Leak the affine owner deliberately: its
            // allocator bit must remain quarantined if teardown cannot be
            // retried, never reopened beneath live authority.
            core::mem::forget(owner);
        }
    }

    fn retry_poison_cleanup(&mut self) {
        let mut arrived = core::array::from_fn(|_| None);
        for k in 0..S::REPLY_GRANTS.min(MAX_GRANTS) {
            if S::retain_grant(k) {
                if let Some(owner) = self.take_retained_delivery(k) {
                    Self::remember_retained(&mut arrived, owner);
                }
            }
        }
        if let ClientState::Poisoned { cleanup, retained } = &mut self.state {
            for owner in arrived.into_iter().flatten() {
                Self::remember_retained(retained, owner);
            }
            Self::settle_cleanup(cleanup);
            Self::settle_retained(retained);
        }
    }

    /// Best-effort cleanup for one dynamically owned landing after the pair is
    /// poisoned. A delivery consumes the post and lets the owner become live
    /// (capability) or disarm and release (notice). If no delivery is visible,
    /// the Armed owner remains in place: a late sender may still target it, so
    /// task-lifetime quarantine is safer than allocator reuse.
    fn take_retained_delivery(&mut self, k: usize) -> Option<ReturnedCapability> {
        let Some(owner) = self.retained_landings[k].as_ref() else {
            return None;
        };
        match verbs::take_delivery(owner.target()) {
            Some(RawDelivery::Cap) => {
                self.standing[k] = false;
                if let Some(mut owner) = self.retained_landings[k].take() {
                    owner.activate();
                    return Some(owner);
                }
            }
            Some(RawDelivery::Notice(..)) => {
                self.standing[k] = false;
                if let Some(mut owner) = self.retained_landings[k].take() {
                    owner.disarm_post();
                    drop(owner);
                }
            }
            None => {}
        }
        None
    }

    /// Kick without a request — re-announce ring state after a dropped
    /// notice, or nudge a freshly-founded service. Its own compact session,
    /// droppable under the same total-drain contract as a generic notice.
    pub async fn kick(&self) -> Result<(), SendError> {
        let mut s = SessionBuilder::new();
        s.request_kick(self.ep.kick)
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        let committed = self
            .transport
            .commit(&s)
            .await
            .map_err(|_| SendError::Transport)?;
        if committed.failing_index() != 0 {
            return Err(SendError::Refused(Status::Malformed));
        }
        match committed.status() {
            Status::Ok | Status::WouldBlock => Ok(()), // droppable
            status => Err(SendError::Refused(status)),
        }
    }

    /// Permanently prevent reuse after a protocol wrapper discovers a defect
    /// the generic wire codec could not express (for example, a delivered
    /// record that cannot be possessed as the protocol's promised type).
    pub fn poison(&mut self) {
        self.poison_after_reply();
    }

    /// Poison this uncorrelated pair and return already-collected dynamic
    /// grant owners to its bounded teardown state. Protocol wrappers call
    /// this when a typed possession check fails after [`CallReply`] has moved
    /// authority out of the generic machine.
    pub fn poison_retained(&mut self, returned: [Option<ReturnedCapability>; MAX_GRANTS]) {
        self.poison_after_reply();
        let ClientState::Poisoned { retained, .. } = &mut self.state else {
            return;
        };
        for owner in returned.into_iter().flatten() {
            Self::remember_retained(retained, owner);
        }
        Self::settle_retained(retained);
    }

    /// Stand every non-standing grant receive before request publication.
    ///
    /// The session contains posts only. Therefore every valid `WouldBlock`
    /// index names an already-standing post; there is no notice index to
    /// reinterpret. Retained targets arm before crossing the transport so an
    /// ambiguous error quarantines the allocator name rather than releasing a
    /// slot a late commit may still target.
    async fn ensure_grant_posts(&mut self, expect: usize) -> Result<(), CallError> {
        for _ in 0..RETRY_BOUND {
            let (session, posted, count) = self.grant_post_batch(expect)?;
            if count == 0 {
                return Ok(());
            }
            let committed = self
                .transport
                .commit(&session)
                .await
                .map_err(|_| CallError::Transport)?;
            if self.settle_grant_posts(&posted[..count], committed)? {
                return Ok(());
            }
        }
        Err(CallError::Backpressure)
    }

    /// Native affine calls cannot suspend while their input owners reside in
    /// the calling future. Post preparation is a bounded sequence of sessions
    /// with no WAIT bits, so the same batch/settlement rules can run on the
    /// non-parking transport before the transfer and correlation are installed.
    fn ensure_grant_posts_affine(&mut self, expect: usize) -> Result<(), CallError> {
        for _ in 0..RETRY_BOUND {
            let (session, posted, count) = self.grant_post_batch(expect)?;
            if count == 0 {
                return Ok(());
            }
            let committed = self
                .transport
                .commit_affine(&session)
                .map_err(|_| CallError::Transport)?;
            if self.settle_grant_posts(&posted[..count], committed)? {
                return Ok(());
            }
        }
        Err(CallError::Backpressure)
    }

    fn grant_post_batch(
        &mut self,
        expect: usize,
    ) -> Result<(SessionBuilder, [usize; MAX_GRANTS], usize), CallError> {
        debug_assert!(expect <= S::REPLY_GRANTS && expect <= MAX_GRANTS);
        let mut s = SessionBuilder::new();
        // The non-standing posts, in grant order; remember which
        // invocation index maps to which channel.
        let mut posted: [usize; MAX_GRANTS] = [0; MAX_GRANTS];
        let mut n = 0;
        for k in 0..expect {
            if !self.standing[k] {
                let pin = self.ep.grants[k].ok_or(CallError::Wiring)?;
                let retain = S::retain_grant(k);
                if !pin.matches_retention(retain) {
                    return Err(CallError::Wiring);
                }
                let landing = if retain {
                    if self.retained_landings[k].is_none() {
                        // The bitmap is a bounded fast index, while these
                        // affine owners are the authoritative local set.
                        // If a stale clear bit yields a name an armed lane
                        // still owns, the CAS has now re-closed that bit;
                        // quarantine the duplicate witness and choose a
                        // genuinely fresh name rather than building a
                        // DoubleAssign receive batch.
                        self.retained_landings[k] = ReturnedCapability::reserve();
                    }
                    let Some(owner) = self.retained_landings[k].as_ref() else {
                        return Err(CallError::Backpressure);
                    };
                    owner.target()
                } else {
                    pin.fixed_landing().ok_or(CallError::Wiring)?
                };
                let staged = match S::grant_policy(k) {
                    Some((kind, minimum_rights)) => {
                        s.post_recv_typed(pin.recv, landing, kind, minimum_rights)
                    }
                    None => s.post_recv(pin.recv, landing),
                };
                staged.map_err(|_| CallError::Refused(Status::Malformed))?;
                if retain {
                    if let Some(owner) = self.retained_landings[k].as_mut() {
                        // Conservatively arm before the transport boundary.
                        // Any later refusal reuses this same target. If the
                        // transport outcome is ambiguous, Armed Drop keeps the
                        // allocator bit quarantined until task teardown.
                        owner.arm_post();
                    }
                }
                posted[n] = k;
                n += 1;
            }
        }
        Ok((s, posted, n))
    }

    fn settle_grant_posts(
        &mut self,
        posted: &[usize],
        committed: Committed,
    ) -> Result<bool, CallError> {
        match committed.status() {
            Status::Ok => {
                if committed.failing_index() != 0 {
                    return Err(CallError::Refused(Status::Malformed));
                }
                for &lane in posted {
                    self.standing[lane] = true;
                }
                Ok(true)
            }
            Status::WouldBlock => {
                let Some(&lane) = posted.get(committed.failing_index() as usize) else {
                    return Err(CallError::Refused(Status::Malformed));
                };
                // This post already stands; the atomic session unwound all
                // other staged posts. The next batch excludes this lane.
                self.standing[lane] = true;
                Ok(false)
            }
            Status::Conflict if (committed.failing_index() as usize) < posted.len() => Ok(false),
            Status::Conflict => Err(CallError::Refused(Status::Malformed)),
            status => Err(CallError::Refused(status)),
        }
    }

    /// Establish or repair the reply post owned by a direct (pre-executor)
    /// waiter. Executor-owned waits are already stood by the executor's pinned
    /// idle discipline and deliberately do no work here.
    async fn ensure_reply_wait_post(&mut self) -> Result<(), CallError> {
        let Some(wait) = self.reply_wait.as_mut() else {
            return Ok(());
        };
        if wait.owner != ReplyWaitOwner::Direct || !wait.needs_stand {
            return Ok(());
        }
        stand_doorbell(self.transport, wait.doorbell)
            .await
            .ok_or(CallError::Transport)?;
        wait.needs_stand = false;
        Ok(())
    }

    /// Consume and re-stand a direct wait's delivered notice. Repair becomes
    /// necessary before the first cancellation point, so dropping this future
    /// cannot make the next call trust an empty Board entry.
    async fn settle_direct_reply_wait(&mut self, doorbell: Doorbell) -> bool {
        let Some(wait) = self.reply_wait.as_mut() else {
            return false;
        };
        wait.needs_stand = true;
        if verbs::take_notice(doorbell.land).is_none() {
            return false;
        }
        if stand_doorbell(self.transport, doorbell).await.is_none() {
            return false;
        }
        if let Some(wait) = self.reply_wait.as_mut() {
            wait.needs_stand = false;
        }
        true
    }

    /// Finish a direct wait after the fused kick could not also park.
    ///
    /// `WouldBlock` from the fused descriptor is deliberately ambiguous: the
    /// service may already have consumed the request kick while the reply post
    /// still stands, or the reply may have landed before the park arm. A
    /// pre-executor caller has no executor to settle either case later, so it
    /// first consumes an already-landed notice and otherwise parks an empty
    /// descriptor on the same post. A refused empty park repairs the post and
    /// retries without publishing a duplicate request.
    async fn finish_direct_reply_wait(&mut self, doorbell: Doorbell) -> Result<(), CallError> {
        let Some(wait) = self.reply_wait.as_ref() else {
            return Err(CallError::Wire);
        };
        let commit = wait.commit;
        for _ in 0..RETRY_BOUND {
            if verbs::take_notice(doorbell.land).is_some() {
                if let Some(wait) = self.reply_wait.as_mut() {
                    wait.needs_stand = true;
                }
                self.ensure_reply_wait_post().await?;
                return Ok(());
            }

            let mut session = SessionBuilder::new();
            session.park_standing(
                doorbell.chan,
                doorbell.land.index() as u16,
                doorbell.recv_tag,
            );
            let spec = ParkSpec {
                chan: doorbell.chan,
                land: doorbell.land,
                alternate: None,
                deadline_ns: None,
                irq: false,
            };
            let committed =
                commit(self.transport, &session, &spec).map_err(|_| CallError::Transport)?;
            if committed.raw() as u16 > Status::Unfounded as u16 || committed.failing_index() != 0 {
                return Err(CallError::Refused(Status::Malformed));
            }
            match committed.status() {
                Status::Ok => {
                    return self
                        .settle_direct_reply_wait(doorbell)
                        .await
                        .then_some(())
                        .ok_or(CallError::Wire);
                }
                Status::WouldBlock | Status::Conflict => {
                    if let Some(wait) = self.reply_wait.as_mut() {
                        wait.needs_stand = true;
                    }
                    if let Err(error) = self.ensure_reply_wait_post().await {
                        // A reply can win between the empty park's refusal
                        // and the repair attempt. In that case the occupied
                        // landing is the positive delivery witness: consume
                        // it, re-stand once, and finish instead of turning a
                        // successful response into a transport error.
                        if verbs::take_notice(doorbell.land).is_none() {
                            return Err(error);
                        }
                        if let Some(wait) = self.reply_wait.as_mut() {
                            wait.needs_stand = true;
                        }
                        self.ensure_reply_wait_post().await?;
                        return Ok(());
                    }
                }
                status @ (Status::Denied
                | Status::Forged
                | Status::Revoked
                | Status::Malformed
                | Status::Deadline
                | Status::Irq
                | Status::Exited
                | Status::AltDelivery
                | Status::Unfounded) => return Err(CallError::Refused(status)),
            }
        }
        Err(CallError::Backpressure)
    }

    /// Deliver the already-published request's compact kick and block on the
    /// executor's standing reply post in the same session. A `WouldBlock`
    /// completion falls back to the ordinary level-triggered future: whether
    /// the request post was already consumed or the park could not arm, the
    /// ring record is visible and an awake executor will drain it.
    async fn kick_and_park_published_request(
        &mut self,
        deadline_ns: Option<u64>,
    ) -> Result<RequestParkCommit, CallError> {
        let Some(wait) = self.reply_wait.as_ref() else {
            self.kick_published_request().await?;
            return Ok(RequestParkCommit::AsyncFallback);
        };
        let doorbell = wait.doorbell;
        let commit = wait.commit;
        let owner = wait.owner;
        for _ in 0..RETRY_BOUND {
            let mut session = SessionBuilder::new();
            session
                .request_kick(self.ep.kick)
                .map_err(|_| CallError::Refused(Status::Malformed))?;
            session.park_standing(
                doorbell.chan,
                doorbell.land.index() as u16,
                doorbell.recv_tag,
            );
            if let Some(deadline_ns) = deadline_ns {
                session.wait_sleep(deadline_ns);
            }
            let spec = ParkSpec {
                chan: doorbell.chan,
                land: doorbell.land,
                alternate: None,
                deadline_ns,
                irq: false,
            };
            let committed =
                commit(self.transport, &session, &spec).map_err(|_| CallError::Transport)?;
            match classify_request_park_commit(committed, deadline_ns.is_some()) {
                RequestParkCommit::Woke => {
                    let settled = match owner {
                        ReplyWaitOwner::Executor => {
                            settle_inline_doorbell(self.transport, doorbell, committed.clock_ms())
                                .await
                        }
                        ReplyWaitOwner::Direct => self.settle_direct_reply_wait(doorbell).await,
                    };
                    if settled {
                        return Ok(RequestParkCommit::Woke);
                    }
                    self.poison_after_reply();
                    return Err(CallError::Wire);
                }
                RequestParkCommit::AsyncFallback => {
                    if owner == ReplyWaitOwner::Direct {
                        self.finish_direct_reply_wait(doorbell).await?;
                    }
                    return Ok(RequestParkCommit::AsyncFallback);
                }
                RequestParkCommit::Retry => continue,
                RequestParkCommit::Deadline => return Ok(RequestParkCommit::Deadline),
                RequestParkCommit::Terminal(error) => {
                    self.poison_after_reply();
                    return Err(error);
                }
            }
        }
        Err(CallError::Backpressure)
    }

    /// Kick one request that is already published. Only transport loss,
    /// bounded conflicts, and cancellation remain resumable; a terminal
    /// endpoint refusal poisons the uncorrelated pair. Resume deliberately
    /// uses this non-parking form: cancellation may have happened after a
    /// prior inline wake, and re-parking after the reply already landed would
    /// wait for an edge that correctly will not be sent twice.
    async fn kick_published_request(&mut self) -> Result<(), CallError> {
        for _ in 0..RETRY_BOUND {
            match self.kick().await {
                Ok(()) => return Ok(()),
                Err(SendError::Transport) => return Err(CallError::Transport),
                Err(SendError::Refused(Status::Conflict)) => continue,
                Err(SendError::WouldBlock) => return Ok(()),
                Err(SendError::Refused(status)) => {
                    let error = CallError::Refused(status);
                    self.poison_after_reply();
                    return Err(error);
                }
            }
        }
        Err(CallError::Backpressure)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_ordinary_reply_wait_has_no_implicit_timeout() {
        let mut ticks = u64::MAX;
        assert!(!super::reply_wait_expired(
            None,
            Some(1),
            Some(u64::MAX),
            false,
            &mut ticks,
        ));
        assert_eq!(ticks, u64::MAX, "unbounded waits spend no hidden budget");
    }

    #[test]
    fn only_an_explicit_reply_budget_expires() {
        let mut ticks = 0;
        assert!(!super::reply_wait_expired(
            Some(2_000_000),
            Some(10),
            Some(2_000_009),
            true,
            &mut ticks,
        ));
        assert!(super::reply_wait_expired(
            Some(2_000_000),
            Some(10),
            Some(2_000_010),
            true,
            &mut ticks,
        ));

        let mut fallback_ticks = 0;
        assert!(!super::reply_wait_expired(
            Some(2_000_000),
            None,
            None,
            true,
            &mut fallback_ticks,
        ));
        assert!(super::reply_wait_expired(
            Some(2_000_000),
            None,
            None,
            true,
            &mut fallback_ticks,
        ));
    }

    #[test]
    fn an_absolute_reply_budget_expires_on_a_non_deadline_wake() {
        let mut ticks = 0;
        assert!(super::reply_wait_expired(
            Some(2_000_000),
            Some(10),
            Some(2_000_010),
            false,
            &mut ticks,
        ));
        assert_eq!(
            ticks, 0,
            "an absolute-clock decision spends no fallback tick"
        );
    }

    #[test]
    fn early_wakes_do_not_spend_clockless_fallback_ticks() {
        let mut ticks = 0;
        for _ in 0..16 {
            assert!(!super::reply_wait_expired(
                Some(2_000_000),
                None,
                None,
                false,
                &mut ticks,
            ));
        }
        assert_eq!(ticks, 0);

        assert!(!super::reply_wait_expired(
            Some(2_000_000),
            None,
            None,
            true,
            &mut ticks,
        ));
        assert!(super::reply_wait_expired(
            Some(2_000_000),
            None,
            None,
            true,
            &mut ticks,
        ));
    }

    #[test]
    fn unrelated_wakes_do_not_slide_the_reply_rekick_deadline() {
        let mut cadence = super::ReplyRekickCadence::after(10);
        let deadline = 10 + super::REPLY_REKICK_NS;
        assert_eq!(cadence.deadline(), deadline);

        for now in [11, 100, 10_000, deadline - 1] {
            assert!(!cadence.due(now));
            assert_eq!(
                cadence.deadline(),
                deadline,
                "an early wake must retain the original absolute cadence"
            );
        }

        assert!(cadence.due(deadline));
        cadence.rearm(deadline + 7);
        assert_eq!(cadence.deadline(), deadline + 7 + super::REPLY_REKICK_NS);
    }

    #[test]
    fn a_clockless_early_wake_rekicks_without_spending_a_fallback_tick() {
        let cadence = super::ReplyRekickCadence::after(10);
        let elapsed_tick = cadence.due(11);
        assert!(!elapsed_tick);
        assert!(super::reply_rekick_needed(elapsed_tick, false));

        let mut ticks = 0;
        assert!(!super::reply_wait_expired(
            Some(super::REPLY_REKICK_NS),
            None,
            None,
            elapsed_tick,
            &mut ticks,
        ));
        assert_eq!(ticks, 0);
    }

    #[test]
    fn an_unbounded_clocked_early_wake_does_not_rekick() {
        let observation: Option<super::ReplyObservation> = None;
        assert!(observation.is_none(), "ordinary calls carry no budget");
        assert!(!super::reply_rekick_needed(false, true));
        assert!(super::reply_rekick_needed(true, true));
    }

    #[test]
    fn bounded_call_refuses_direct_progress_before_publication() {
        use core::future::Future;
        use core::task::{Context, Poll, Waker};

        let transport = PendingKickTransport {
            transfer: Committed::from_raw(Status::Ok.encode(0)),
        };
        let mut window = RetainedTestWindow([0; RETAINED_TEST_WINDOW_BYTES]);
        let (rings, _serve) =
            crate::svc::split_window::<RetainedTestProtocol>(&mut window.0, REQ_CAP, REP_CAP)
                .unwrap();
        let mut client = SvcClient::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(90),
                grants: [None; MAX_GRANTS],
            },
        )
        .with_direct_polling();

        {
            let mut future = core::pin::pin!(client.call_bounded(&17, REPLY_REKICK_NS));
            let mut context = Context::from_waker(Waker::noop());
            assert!(matches!(
                future.as_mut().poll(&mut context),
                Poll::Ready(Err(CallError::Wiring))
            ));
        }
        assert!(
            client.is_idle(),
            "the refused bounded call published nothing"
        );
    }

    /// The clock-less half of the reply bound must still be a duration.
    #[test]
    fn a_reply_budget_is_at_least_one_rekick_interval() {
        // The default: 5 s of 1 ms ticks.
        assert_eq!(super::reply_budget_ticks(5_000_000_000), 5_000);
        assert_eq!(super::reply_budget_ticks(super::REPLY_REKICK_NS), 1);
        // Shorter than one interval, and zero, still buy one wait rather than
        // reporting a timeout before anything was awaited.
        assert_eq!(super::reply_budget_ticks(super::REPLY_REKICK_NS - 1), 1);
        assert_eq!(super::reply_budget_ticks(0), 1);
        // A partial interval rounds up: the clock-less fallback must not
        // report expiry before the caller's requested duration elapsed.
        assert_eq!(super::reply_budget_ticks(super::REPLY_REKICK_NS + 1), 2);
    }

    /// A protocol may raise its ceiling; the tick count must track it and not
    /// silently saturate.
    #[test]
    fn a_raised_reply_budget_scales_its_tick_count() {
        assert_eq!(super::reply_budget_ticks(30_000_000_000), 30_000);
        assert!(super::reply_budget_ticks(u64::MAX) > 0);
    }

    use super::*;
    use crate::svc::{REP_CAP, REQ_CAP};

    struct RetainedTestProtocol;

    const RETAINED_TEST_WINDOW_BYTES: usize =
        crate::svc::window_bytes::<RetainedTestProtocol>(REQ_CAP, REP_CAP);

    #[repr(align(64))]
    struct RetainedTestWindow([u8; RETAINED_TEST_WINDOW_BYTES]);

    impl SvcProtocol for RetainedTestProtocol {
        type Request = u64;
        type Response = u64;
        type ReqWire = u64;
        type RepWire = u64;

        const REPLY_GRANTS: usize = 0;

        fn encode_req(req: &Self::Request) -> Self::ReqWire {
            *req
        }

        fn decode_req(wire: &Self::ReqWire) -> Option<Self::Request> {
            Some(*wire)
        }

        fn encode_rep(rep: &Self::Response, _: u8) -> Self::RepWire {
            *rep
        }

        fn decode_rep(wire: &Self::RepWire) -> Option<(Self::Response, u8)> {
            Some((*wire, 0))
        }
    }

    struct PendingKickTransport {
        transfer: Committed,
    }

    impl Transport for PendingKickTransport {
        type Error = core::convert::Infallible;

        async fn commit(&self, _: &SessionBuilder) -> Result<Committed, Self::Error> {
            core::future::pending().await
        }

        fn commit_now(&self, _: &SessionBuilder) -> Result<Committed, Self::Error> {
            Ok(self.transfer)
        }
    }

    fn silent_reply_deadline(
        _: &crate::transport::TrapTransport,
        session: &SessionBuilder,
        spec: &ParkSpec,
    ) -> Result<Committed, crate::transport::TrapUnavailable> {
        assert_eq!(spec.deadline_ns, Some(REPLY_REKICK_NS));
        assert!(session.flags().contains(
            kore_abi::session::SessionFlags::WAIT_DELIVERY
                | kore_abi::session::SessionFlags::WAIT_SLEEP
        ));
        assert_eq!(
            session.encoded_len(),
            Ok(48),
            "kick + standing park + deadline stays in x86's GPR carrier"
        );
        Ok(Committed::from_raw(Status::Deadline.encode(0)))
    }

    #[test]
    fn cancelled_plain_preflight_reserves_the_pair_before_request_publication() {
        use core::{
            cell::Cell,
            future::Future,
            task::{Context, Poll, Waker},
        };
        struct Protocol;
        impl SvcProtocol for Protocol {
            type Request = u64;
            type Response = u64;
            type ReqWire = u64;
            type RepWire = u64;
            const REPLY_GRANTS: usize = 1;
            fn encode_req(r: &u64) -> u64 {
                *r
            }
            fn decode_req(r: &u64) -> Option<u64> {
                Some(*r)
            }
            fn encode_rep(r: &u64, _: u8) -> u64 {
                *r
            }
            fn decode_rep(r: &u64) -> Option<(u64, u8)> {
                Some((*r, 0))
            }
        }
        struct Posts(Cell<bool>);
        impl Transport for Posts {
            type Error = core::convert::Infallible;
            fn commit_now(&self, _: &SessionBuilder) -> Result<Committed, Self::Error> {
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            }
            async fn commit(&self, _: &SessionBuilder) -> Result<Committed, Self::Error> {
                if !self.0.get() {
                    core::future::pending::<()>().await;
                }
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            }
        }
        const BYTES: usize = crate::svc::window_bytes::<Protocol>(REQ_CAP, REP_CAP);
        #[repr(align(64))]
        struct Window([u8; BYTES]);
        let mut window = Window([0; BYTES]);
        let (rings, mut serve) =
            crate::svc::split_window::<Protocol>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        let transport = Posts(Cell::new(false));
        let mut grants = [None; MAX_GRANTS];
        // Fixed-slot protocol fixture; no native retained grant is fabricated.
        grants[0] = Some(GrantPin::fixed(
            SlotRef::from_wire(7),
            SlotRef::from_wire(8),
        ));
        let mut client = SvcClient::<_, Protocol>::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(9),
                grants,
            },
        )
        .with_direct_polling();
        let mut cx = Context::from_waker(Waker::noop());
        {
            let mut first = core::pin::pin!(client.call(&17));
            assert!(first.as_mut().poll(&mut cx).is_pending());
        }
        assert!(serve.req.pop().is_none());
        assert!(!client.is_idle());
        assert_eq!(client.outstanding_request(), Some(&17));
        for request in [17, 23] {
            let mut replacement = core::pin::pin!(client.call(&request));
            assert!(matches!(
                replacement.as_mut().poll(&mut cx),
                Poll::Ready(Err(CallError::Outstanding))
            ));
        }
        {
            let mut wrong = core::pin::pin!(client.resume_exact(&23));
            assert!(matches!(
                wrong.as_mut().poll(&mut cx),
                Poll::Ready(Err(CallError::Outstanding))
            ));
        }
        transport.0.set(true);
        {
            let mut resumed = core::pin::pin!(client.resume_exact(&17));
            assert!(resumed.as_mut().poll(&mut cx).is_pending());
        }
        assert_eq!(serve.req.pop(), Some(17));
        assert!(serve.req.pop().is_none());
        serve.rep.push(91).unwrap();
        {
            let mut resumed = core::pin::pin!(client.resume_exact(&17));
            let Poll::Ready(Ok(reply)) = resumed.as_mut().poll(&mut cx) else {
                panic!("original reply")
            };
            assert_eq!(reply.response, 91);
        }
        assert!(client.is_idle());
        assert!(serve.req.pop().is_none());
    }

    #[test]
    fn bounded_executor_reply_park_times_out_without_canceling_the_request() {
        let transport = crate::transport::TrapTransport::new();
        let mut window = RetainedTestWindow([0; RETAINED_TEST_WINDOW_BYTES]);
        let (rings, mut serve) =
            crate::svc::split_window::<RetainedTestProtocol>(&mut window.0, REQ_CAP, REP_CAP)
                .unwrap();
        let reply_wait = Doorbell {
            chan: 7,
            recv: SlotRef::from_wire(101),
            land: SlotRef::from_wire(102),
            recv_tag: 0x1234,
        };
        let mut client = SvcClient::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(100),
                grants: [None; MAX_GRANTS],
            },
        )
        .with_reply_wait(reply_wait);
        client.reply_wait.as_mut().unwrap().commit = silent_reply_deadline;

        // `from_standing` models the reply post the owning executor already
        // maintains. The silent peer consumes no request and produces no
        // reply; the fused park's real Deadline outcome must therefore be the
        // completion, without falling through to an unbounded executor idle.
        let mut executor = crate::Executor::from_standing(&transport, reply_wait);
        let result = executor.run(client.call_bounded(&17, REPLY_REKICK_NS));

        assert!(matches!(result, Err(CallError::Timeout)));
        assert_eq!(client.outstanding_request(), Some(&17));
        assert!(matches!(
            &client.state,
            ClientState::AwaitingReply { request } if *request == 17
        ));
        assert_eq!(
            serve.req.pop(),
            Some(17),
            "the sole request stays published"
        );
    }

    #[test]
    fn absolute_invocation_observation_keeps_the_deadline_and_late_reply() {
        use core::cell::Cell;
        struct Kicks(Cell<usize>);
        impl Transport for Kicks {
            type Error = core::convert::Infallible;
            async fn commit(&self, session: &SessionBuilder) -> Result<Committed, Self::Error> {
                self.commit_now(session)
            }
            fn commit_now(&self, _: &SessionBuilder) -> Result<Committed, Self::Error> {
                self.0.set(self.0.get() + 1);
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            }
        }
        // Delayed observation never refreshes a lifetime into a new duration.
        for now in [0, 12, 100, u64::MAX] {
            assert_eq!(ReplyLimit::Deadline(12).deadline(now), 12);
        }
        let transport = Kicks(Cell::new(0));
        let mut window = RetainedTestWindow([0; RETAINED_TEST_WINDOW_BYTES]);
        let (rings, mut serve) =
            crate::svc::split_window::<RetainedTestProtocol>(&mut window.0, REQ_CAP, REP_CAP)
                .unwrap();
        let mut client = SvcClient::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(100),
                grants: [None; MAX_GRANTS],
            },
        );
        let park = crate::transport::TrapTransport::new();
        let doorbell = Doorbell {
            chan: 7,
            recv: SlotRef::from_wire(101),
            land: SlotRef::from_wire(102),
            recv_tag: 0x1234,
        };
        let mut executor = crate::Executor::from_standing(&park, doorbell);
        assert!(matches!(
            executor.run(client.call_bounded(&17, 0)),
            Err(CallError::Timeout)
        ));
        assert_eq!(serve.req.pop(), Some(17));
        for _ in 0..2 {
            assert!(matches!(
                executor.run(client.resume_until(0)),
                Err(CallError::Timeout)
            ));
            assert_eq!(client.outstanding_request(), Some(&17));
            assert!(serve.req.pop().is_none());
        }
        serve.rep.push(91).unwrap();
        let reply = executor.run(client.resume_until(0)).unwrap();
        assert_eq!(reply.response, 91);
        assert!(client.is_idle());
        assert!(serve.req.pop().is_none());
        assert_eq!(transport.0.get(), 4);
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum LifecycleRequest {
        Retire(u32),
        Confirm(u32),
    }

    #[test]
    fn interrupted_confirm_for_row_a_cannot_be_resumed_as_row_b() {
        let row_a = LifecycleRequest::Confirm(0xa);
        let awaiting: ClientState<LifecycleRequest, u64> =
            ClientState::AwaitingReply { request: row_a };

        for row_b in [
            LifecycleRequest::Retire(0xb),
            LifecycleRequest::Confirm(0xb),
        ] {
            assert_eq!(
                exact_outstanding_request(&awaiting, &row_b),
                Err(CallError::Outstanding),
                "row B must not consume row A's outstanding confirmation"
            );
        }
        assert!(matches!(
            awaiting,
            ClientState::AwaitingReply { request } if request == row_a
        ));
        assert_eq!(exact_outstanding_request(&awaiting, &row_a), Ok(()));
    }

    #[test]
    fn exact_resume_mismatch_preserves_a_partially_collected_reply() {
        let row_a = LifecycleRequest::Confirm(0xa);
        let collecting = ClientState::Collecting {
            request: row_a,
            response: 7u64,
            grants: 1,
            delivered: [const { None }; MAX_GRANTS],
            retained: [const { None }; MAX_GRANTS],
            next: 0,
        };

        assert_eq!(
            exact_outstanding_request(&collecting, &LifecycleRequest::Confirm(0xb)),
            Err(CallError::Outstanding)
        );
        assert!(matches!(
            collecting,
            ClientState::Collecting {
                request,
                response: 7,
                grants: 1,
                next: 0,
                ..
            } if request == row_a
        ));
    }

    #[derive(Debug, PartialEq, Eq)]
    struct TestCapability {
        slot: Option<SlotRef>,
        rights: Rights,
        marker: u32,
    }

    impl return_capability_sealed::Sealed for TestCapability {
        fn return_slot(&self) -> Option<SlotRef> {
            self.slot
        }

        fn return_rights(&self) -> Option<Rights> {
            Some(self.rights)
        }
    }

    impl ReturnCapability for TestCapability {}

    #[test]
    fn minted_endpoint_owners_are_approved_return_capabilities() {
        fn assert_return_capability<T: ReturnCapability>() {}

        assert_return_capability::<ChanSendToken>();
        assert_return_capability::<ChanRecvToken>();
        assert_return_capability::<HeldChanSendToken>();
    }

    #[test]
    fn typed_not_moved_error_returns_the_exact_source() {
        let source = TestCapability {
            slot: Some(SlotRef::from_wire(19)),
            rights: Rights::ACCESS | Rights::DERIVE,
            marker: 0xfeed,
        };
        let error: ReturnCallError<TestCapability> = ReturnCallError::NotMoved {
            error: CallError::RingFull,
            capability: source,
        };
        let ReturnCallError::NotMoved { error, capability } = error else {
            panic!("source-preserving error changed variant");
        };
        assert_eq!(error, CallError::RingFull);
        assert_eq!(capability.marker, 0xfeed);
        assert_eq!(capability.slot, Some(SlotRef::from_wire(19)));
    }

    #[test]
    fn affine_post_preparation_and_transfer_complete_before_the_first_await() {
        use core::{
            cell::Cell,
            future::Future,
            task::{Context, Poll, Waker},
        };
        struct Protocol;
        impl SvcProtocol for Protocol {
            type Request = u64;
            type Response = u64;
            type ReqWire = u64;
            type RepWire = u64;
            const REPLY_GRANTS: usize = 1;
            fn encode_req(r: &u64) -> u64 {
                *r
            }
            fn decode_req(r: &u64) -> Option<u64> {
                Some(*r)
            }
            fn encode_rep(r: &u64, _: u8) -> u64 {
                *r
            }
            fn decode_rep(r: &u64) -> Option<(u64, u8)> {
                Some((*r, 0))
            }
        }
        struct Script {
            first: Committed,
            calls: Cell<usize>,
            repeat: bool,
        }
        impl Transport for Script {
            type Error = core::convert::Infallible;
            async fn commit(&self, _: &SessionBuilder) -> Result<Committed, Self::Error> {
                core::future::pending().await
            }
            fn commit_now(&self, session: &SessionBuilder) -> Result<Committed, Self::Error> {
                assert!(!session.flags().intersects(
                    kore_abi::session::SessionFlags::WAIT_DELIVERY
                        | kore_abi::session::SessionFlags::WAIT_IRQ
                        | kore_abi::session::SessionFlags::WAIT_SLEEP
                ));
                let at = self.calls.get();
                self.calls.set(at + 1);
                Ok(if at == 0 || self.repeat {
                    self.first
                } else {
                    Committed::from_raw(Status::Ok.encode(0))
                })
            }
        }
        const BYTES: usize = crate::svc::window_bytes::<Protocol>(REQ_CAP, REP_CAP);
        #[repr(align(64))]
        struct Window([u8; BYTES]);
        for (status, index, repeat, error, calls) in [
            (Status::Ok, 0, false, None, 2),
            (Status::WouldBlock, 0, false, None, 2),
            (
                Status::Conflict,
                0,
                true,
                Some(CallError::Backpressure),
                RETRY_BOUND,
            ),
            (
                Status::WouldBlock,
                1,
                false,
                Some(CallError::Refused(Status::Malformed)),
                1,
            ),
            (
                Status::Denied,
                0,
                false,
                Some(CallError::Refused(Status::Denied)),
                1,
            ),
        ] {
            let transport = Script {
                first: Committed::from_raw(status.encode(index)),
                calls: Cell::new(0),
                repeat,
            };
            let mut window = Window([0; BYTES]);
            let (rings, mut serve) =
                crate::svc::split_window::<Protocol>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
            let mut grants = [None; MAX_GRANTS];
            grants[0] = Some(GrantPin::fixed(
                SlotRef::from_wire(7),
                SlotRef::from_wire(8),
            ));
            let mut client = SvcClient::new(
                &transport,
                rings,
                ClientEndpoints {
                    kick: SlotRef::from_wire(9),
                    grants,
                },
            );
            let source = TestCapability {
                slot: Some(SlotRef::from_wire(92)),
                rights: Rights::ACCESS | Rights::DERIVE,
                marker: 0xace,
            };
            let mut cx = Context::from_waker(Waker::noop());
            {
                let mut call = core::pin::pin!(client.call_with_return(
                    &17,
                    SlotRef::from_wire(93),
                    source,
                    Rights::ACCESS
                ));
                match (error, call.as_mut().poll(&mut cx)) {
                    (None, Poll::Pending) => {} // post-transfer kick only
                    (
                        Some(expected),
                        Poll::Ready(Err(ReturnCallError::NotMoved { error, capability })),
                    ) => {
                        assert_eq!(error, expected);
                        assert_eq!(capability.marker, 0xace);
                        assert_eq!(capability.slot, Some(SlotRef::from_wire(92)));
                    }
                    _ => panic!("unexpected affine preflight outcome"),
                }
            }
            assert_eq!(transport.calls.get(), calls);
            if error.is_none() {
                assert_eq!(serve.req.pop(), Some(17));
                assert_eq!(client.outstanding_request(), Some(&17));
                assert!(!client.is_idle());
            } else {
                assert!(serve.req.pop().is_none());
                assert!(client.is_idle());
            }
        }
    }

    #[test]
    fn owned_return_installs_reply_custody_before_the_next_cancellable_await() {
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        // The async transport never completes. A lawful transfer must instead
        // use the non-parking affine path, then suspend only in the kick.
        let transport = PendingKickTransport {
            transfer: Committed::from_raw(Status::Ok.encode(0)),
        };
        let mut window = RetainedTestWindow([0; RETAINED_TEST_WINDOW_BYTES]);
        let (rings, mut serve) =
            crate::svc::split_window::<RetainedTestProtocol>(&mut window.0, REQ_CAP, REP_CAP)
                .unwrap();
        let mut client = SvcClient::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(91),
                grants: [None; MAX_GRANTS],
            },
        );
        let mut cx = Context::from_waker(Waker::noop());
        {
            // Generic affine witness fixture; no native capability is minted.
            let source = TestCapability {
                slot: Some(SlotRef::from_wire(92)),
                rights: Rights::ACCESS | Rights::DERIVE,
                marker: 0xace,
            };
            let mut transfer = core::pin::pin!(client.call_with_return(
                &17,
                SlotRef::from_wire(93),
                source,
                Rights::ACCESS
            ));
            assert!(transfer.as_mut().poll(&mut cx).is_pending());
        }
        assert_eq!(serve.req.pop(), Some(17));
        assert_eq!(client.outstanding_request(), Some(&17));
        assert!(!client.is_idle());
        {
            let mut replacement = core::pin::pin!(client.call(&18));
            assert!(matches!(
                replacement.as_mut().poll(&mut cx),
                Poll::Ready(Err(CallError::Outstanding))
            ));
        }
        assert!(serve.req.pop().is_none());
    }

    #[test]
    fn exact_held_return_installs_reply_custody_before_the_next_cancellable_await() {
        use core::{
            future::Future,
            task::{Context, Waker},
        };
        let transport = PendingKickTransport {
            transfer: Committed::from_raw(Status::Ok.encode(0)),
        };
        let mut window = RetainedTestWindow([0; RETAINED_TEST_WINDOW_BYTES]);
        let (rings, mut serve) =
            crate::svc::split_window::<RetainedTestProtocol>(&mut window.0, REQ_CAP, REP_CAP)
                .unwrap();
        let mut client = SvcClient::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(91),
                grants: [None; MAX_GRANTS],
            },
        );
        let mut cx = Context::from_waker(Waker::noop());
        {
            let mut transfer = core::pin::pin!(client.call_with_exact_held_return(
                &17,
                SlotRef::from_wire(93),
                SlotRef::from_wire(92)
            ));
            assert!(transfer.as_mut().poll(&mut cx).is_pending());
        }
        assert_eq!(serve.req.pop(), Some(17));
        assert_eq!(client.outstanding_request(), Some(&17));
        assert!(!client.is_idle());
    }

    #[test]
    fn retained_exact_return_installs_correlation_before_post_move_cancellation() {
        use core::future::Future;
        use core::task::{Context, Poll, Waker};

        let transport = PendingKickTransport {
            transfer: Committed::from_raw(Status::Ok.encode(0)),
        };
        let mut window = RetainedTestWindow([0; RETAINED_TEST_WINDOW_BYTES]);
        let (rings, _serve) =
            crate::svc::split_window::<RetainedTestProtocol>(&mut window.0, REQ_CAP, REP_CAP)
                .unwrap();
        let mut client = SvcClient::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(91),
                grants: [None; MAX_GRANTS],
            },
        );
        let mut source = Some(TestCapability {
            slot: Some(SlotRef::from_wire(92)),
            rights: Rights::ACCESS,
            marker: 0xace,
        });
        let request = 17;

        {
            let mut future = core::pin::pin!(client.call_with_exact_retained_return(
                &request,
                SlotRef::from_wire(93),
                &mut source,
            ));
            let mut context = Context::from_waker(Waker::noop());
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
            // Dropping here models cancellation in the request-kick await.
        }

        assert!(
            source.is_none(),
            "the committed move consumed the sole owner"
        );
        assert!(matches!(
            client.state,
            ClientState::AwaitingReply { request: 17 }
        ));
        assert!(!client.is_idle());
    }

    #[test]
    fn retained_exact_return_keeps_owner_on_proven_not_moved() {
        use core::future::Future;
        use core::task::{Context, Poll, Waker};

        let transport = PendingKickTransport {
            transfer: Committed::from_raw(Status::WouldBlock.encode(0)),
        };
        let mut window = RetainedTestWindow([0; RETAINED_TEST_WINDOW_BYTES]);
        let (rings, _serve) =
            crate::svc::split_window::<RetainedTestProtocol>(&mut window.0, REQ_CAP, REP_CAP)
                .unwrap();
        let mut client = SvcClient::new(
            &transport,
            rings,
            ClientEndpoints {
                kick: SlotRef::from_wire(94),
                grants: [None; MAX_GRANTS],
            },
        );
        let mut source = Some(TestCapability {
            slot: Some(SlotRef::from_wire(95)),
            rights: Rights::ACCESS,
            marker: 0xbeef,
        });
        {
            let mut future = core::pin::pin!(client.call_with_exact_retained_return(
                &18,
                SlotRef::from_wire(96),
                &mut source,
            ));
            let mut context = Context::from_waker(Waker::noop());

            assert!(matches!(
                future.as_mut().poll(&mut context),
                Poll::Ready(Err(RetainedReturnCallError::NotMoved(CallError::Refused(
                    Status::WouldBlock
                ))))
            ));
        }
        assert_eq!(source.as_ref().unwrap().marker, 0xbeef);
        assert!(client.is_idle());
    }

    #[test]
    fn invocation_preflight_refuses_non_delegable_source_without_moving_it() {
        let source = TestCapability {
            slot: Some(SlotRef::from_wire(23)),
            rights: Rights::ACCESS | Rights::REVOKE,
            marker: 0xc0de,
        };
        assert_eq!(
            preflight_return_send(&source, Rights::ACCESS | Rights::REVOKE),
            Err(CallError::Refused(Status::Denied))
        );
        assert_eq!(source.slot, Some(SlotRef::from_wire(23)));
        assert_eq!(source.marker, 0xc0de);
    }

    #[test]
    fn invocation_preflight_accepts_delegable_intermediate_and_attenuation() {
        let source = TestCapability {
            slot: Some(SlotRef::from_wire(24)),
            rights: Rights::ACCESS | Rights::DERIVE | Rights::REVOKE,
            marker: 0xcafe,
        };
        assert_eq!(
            preflight_return_send(&source, Rights::ACCESS | Rights::REVOKE),
            Ok(SlotRef::from_wire(24))
        );
        assert!(!((Rights::ACCESS | Rights::REVOKE).contains(Rights::DERIVE)));
    }

    #[test]
    fn invocation_exact_preflight_accepts_an_already_attenuated_source() {
        let source = TestCapability {
            slot: Some(SlotRef::from_wire(25)),
            rights: Rights::ACCESS | Rights::REVOKE,
            marker: 0xbeef,
        };
        assert_eq!(preflight_exact_return(&source), Ok(SlotRef::from_wire(25)));
        assert_eq!(source.marker, 0xbeef);
    }

    #[test]
    fn return_commit_classification_preserves_the_custody_boundary() {
        assert_eq!(
            classify_return_commit(
                Committed::from_raw(Status::Ok.encode(0)),
                1,
                ReturnSource::Held,
            ),
            ReturnCommit::Moved
        );
        for status in [
            Status::Denied,
            Status::Forged,
            Status::Revoked,
            Status::WouldBlock,
        ] {
            assert_eq!(
                classify_return_commit(
                    Committed::from_raw(status.encode(0)),
                    1,
                    ReturnSource::Held,
                ),
                ReturnCommit::NotMoved(CallError::Refused(status))
            );
        }
        for status in [
            Status::Conflict,
            Status::Malformed,
            Status::Deadline,
            Status::Irq,
            Status::Exited,
        ] {
            assert_eq!(
                classify_return_commit(
                    Committed::from_raw(status.encode(0)),
                    1,
                    ReturnSource::Held,
                ),
                ReturnCommit::AfterTransfer(CallError::Refused(status))
            );
        }

        // A staged return contains one derive followed by its terminal send.
        // A known pure refusal at either real invocation proves that no
        // capability landed.
        assert_eq!(
            classify_return_commit(
                Committed::from_raw(Status::Denied.encode(1)),
                2,
                ReturnSource::Staged,
            ),
            ReturnCommit::NotMoved(CallError::Refused(Status::Denied))
        );
        assert_eq!(
            classify_return_commit(
                Committed::from_raw(Status::Conflict.encode(1)),
                2,
                ReturnSource::Staged,
            ),
            ReturnCommit::NotMoved(CallError::Refused(Status::Conflict))
        );
    }

    #[test]
    fn malformed_transfer_completion_never_resurrects_the_source() {
        assert_eq!(
            classify_return_commit(
                Committed::from_raw(Status::Denied.encode(1)),
                1,
                ReturnSource::Held,
            ),
            ReturnCommit::AfterTransfer(CallError::Refused(Status::Malformed))
        );
        assert_eq!(
            classify_return_commit(Committed::from_raw(0xffff), 1, ReturnSource::Held),
            ReturnCommit::AfterTransfer(CallError::Refused(Status::Malformed))
        );
        assert_eq!(
            classify_return_commit(
                Committed::from_raw(Status::Ok.encode(0)),
                0,
                ReturnSource::Held,
            ),
            ReturnCommit::AfterTransfer(CallError::Refused(Status::Malformed))
        );
    }

    #[test]
    fn atomic_invocation_refusal_never_claims_custody_moved() {
        let retryable = classify_return_commit(
            Committed::from_raw(Status::WouldBlock.encode(1)),
            5,
            ReturnSource::Held,
        );
        assert_eq!(
            retryable,
            ReturnCommit::NotMoved(CallError::Refused(Status::WouldBlock))
        );
        for status in [
            Status::Denied,
            Status::Forged,
            Status::Revoked,
            Status::Conflict,
            Status::Malformed,
        ] {
            assert_ne!(
                classify_return_commit(
                    Committed::from_raw(status.encode(1)),
                    5,
                    ReturnSource::Held,
                ),
                retryable,
                "distinct refusals retain their exact custody verdict"
            );
        }
    }

    #[test]
    fn parked_return_classification_preserves_the_custody_boundary() {
        assert_eq!(
            classify_parked_return_commit(Committed::from_raw(Status::Ok.encode(0)), 1),
            ReturnCommit::Moved
        );
        for status in [Status::Denied, Status::Forged, Status::WouldBlock] {
            assert_eq!(
                classify_parked_return_commit(Committed::from_raw(status.encode(0)), 1),
                ReturnCommit::NotMoved(CallError::Refused(status))
            );
        }
        for status in [
            Status::Revoked,
            Status::Conflict,
            Status::Malformed,
            Status::Deadline,
            Status::Irq,
            Status::Exited,
        ] {
            assert_eq!(
                classify_parked_return_commit(Committed::from_raw(status.encode(0)), 1),
                ReturnCommit::AfterTransfer(CallError::Refused(status))
            );
        }
        assert_eq!(
            classify_parked_return_commit(Committed::from_raw(Status::Denied.encode(1)), 1,),
            ReturnCommit::AfterTransfer(CallError::Refused(Status::Malformed))
        );
        assert_eq!(
            classify_parked_return_commit(Committed::from_raw(Status::Ok.encode(0)), 0),
            ReturnCommit::AfterTransfer(CallError::Refused(Status::Malformed))
        );
    }

    #[test]
    fn request_park_classification_preserves_level_triggered_fallback() {
        assert_eq!(
            classify_request_park_commit(Committed::from_raw(Status::Ok.encode(0)), false),
            RequestParkCommit::Woke
        );
        assert_eq!(
            classify_request_park_commit(Committed::from_raw(Status::WouldBlock.encode(0)), false,),
            RequestParkCommit::AsyncFallback
        );
        assert_eq!(
            classify_request_park_commit(Committed::from_raw(Status::Conflict.encode(0)), false),
            RequestParkCommit::Retry
        );
        for status in [
            Status::Denied,
            Status::Forged,
            Status::Revoked,
            Status::Malformed,
            Status::Deadline,
            Status::Irq,
            Status::Exited,
        ] {
            assert_eq!(
                classify_request_park_commit(Committed::from_raw(status.encode(0)), false),
                RequestParkCommit::Terminal(CallError::Refused(status))
            );
        }
        assert_eq!(
            classify_request_park_commit(Committed::from_raw(Status::Deadline.encode(0)), true,),
            RequestParkCommit::Deadline
        );
        assert_eq!(
            classify_request_park_commit(Committed::from_raw(Status::Denied.encode(1)), true),
            RequestParkCommit::Terminal(CallError::Refused(Status::Malformed))
        );
        assert_eq!(
            classify_request_park_commit(Committed::from_raw(0xffff), true),
            RequestParkCommit::Terminal(CallError::Refused(Status::Malformed))
        );
    }

    #[test]
    fn staged_return_is_one_derive_before_one_send() {
        let source = SlotRef::from_wire(7);
        let result = SlotRef::from_wire(8);
        let return_tx = SlotRef::from_wire(9);
        let mut transfer = SessionBuilder::new();
        let checkpoint = transfer.staged();
        let staged = transfer
            .carve_window(
                source,
                kore_abi::cap::Rights::ACCESS | kore_abi::cap::Rights::DERIVE,
                0,
                0x1000,
                result,
            )
            .expect("one lease-child derive fits");
        assert!(is_single_return_derive(&transfer, checkpoint, staged));
        transfer
            .send(return_tx, staged, kore_abi::cap::Rights::ACCESS)
            .expect("terminal send fits");

        let mut wire = [0u8; crate::session::DESCRIPTOR_MAX];
        transfer.finish(&mut wire).expect("descriptor fits");
        assert_eq!(u16::from_le_bytes([wire[0], wire[1]]), 2);
        assert_eq!(
            u16::from_le_bytes([wire[8], wire[9]]),
            kore_abi::session::OpCode::Derive as u16
        );
        let send = 8 + 16;
        assert_eq!(
            u16::from_le_bytes([wire[send], wire[send + 1]]),
            kore_abi::session::OpCode::Rendezvous as u16
        );
        assert_eq!(
            u32::from_le_bytes([
                wire[send + 4],
                wire[send + 5],
                wire[send + 6],
                wire[send + 7],
            ]),
            return_tx.index()
        );
        assert_eq!(
            u32::from_le_bytes([
                wire[send + 8],
                wire[send + 9],
                wire[send + 10],
                wire[send + 11],
            ]),
            result.index()
        );
    }

    #[test]
    fn staged_return_rejects_a_result_not_derived_after_its_checkpoint() {
        let mut transfer = SessionBuilder::new();
        let old = transfer
            .derive_raw(
                SlotRef::from_wire(20),
                kore_abi::cap::Rights::DERIVE,
                0,
                SlotRef::from_wire(21),
                None,
            )
            .expect("one derive fits");
        let checkpoint = transfer.staged();
        assert!(!is_single_return_derive(&transfer, checkpoint, old));

        let mut another = SessionBuilder::new();
        let foreign = another
            .derive_raw(
                SlotRef::from_wire(22),
                kore_abi::cap::Rights::DERIVE,
                0,
                SlotRef::from_wire(23),
                None,
            )
            .expect("one derive fits");
        assert!(!is_single_return_derive(&transfer, checkpoint, foreign));
    }

    #[test]
    fn staged_return_rejects_intermediate_derives_and_side_effects() {
        let mut transfer = SessionBuilder::new();
        let checkpoint = transfer.staged();
        let parent = transfer
            .carve_sublease(
                SlotRef::from_wire(30),
                kore_abi::cap::Rights::ACCESS | kore_abi::cap::Rights::DERIVE,
                0,
                0x2000,
                SlotRef::from_wire(31),
            )
            .expect("parent derive fits");
        let leaf = transfer
            .carve_window(
                parent.into(),
                kore_abi::cap::Rights::ACCESS,
                0,
                0x1000,
                SlotRef::from_wire(32),
            )
            .expect("leaf derive fits");
        assert!(
            !is_single_return_derive(&transfer, checkpoint, leaf),
            "a hidden intermediate makes the fresh-child owner incomplete"
        );

        transfer
            .revoke(leaf.slot())
            .expect("the adversarial side effect fits");
        assert!(
            !is_single_return_derive(&transfer, checkpoint, leaf),
            "non-derive side effects are rejected before commit"
        );
    }
}
