//! # `chan` — typed channel endpoints over rendezvous
//! (`res/USERLAND.md` §4/§5.2).
//!
//! What kore-rt never had until Phase A: `ChanTx`/`ChanRx` capability
//! witnesses over `ChanSend`/`ChanRecv` records, carrying the §4 pinned
//! verbs — the notice-mode kick, the landed capability transfer, and the
//! never-parking recv.  Both are generic over the transport seam (§2:
//! clients are generic over the seam).
//!
//! The division of labor (§5.3): **payloads ride rings; authority rides
//! rendezvous; nothing else exists.**  A channel carries only notices
//! (bind-inert data — "look at your ring") and capability transfers
//! (grant-bearing replies); a `WouldBlock`ed notice is *droppable*
//! exactly when the receiver runs the §3.1 order (its ring write is
//! already visible to the drain that follows the receiver's re-post).

use kore_abi::cap::Rights;
use kore_abi::session::{Notice, SlotRef, Status};

use crate::cap::{
    ChanRecvToken, ChanSendToken, RawDelivery, ReturnedCapability, StagedCapabilityReservation,
    StashedLease, verbs,
};
use crate::session::{SessionBuilder, Staged};
use crate::transport::{ParkPost, Transport};

/// What a [`ChanRx::recv`] resolved to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// A doorbell notice — consumed out of the landing slot; the cookie
    /// names the source to look at (§3.1: the drain is total, the cookie
    /// a hint).
    Notice(Notice),
    /// A capability record landed in the named slot (a grant-bearing
    /// transfer, §5.5) — take it through the possession verbs; the next
    /// `recv` re-posts.
    Cap(SlotRef),
}

/// Why a send refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SendError {
    /// No live posted recv — the whole-session non-blocking refusal
    /// (SYSCALL.md D2).  For a §3.1-ordered doorbell kick this is
    /// **droppable**: the peer is awake and will drain every source
    /// after it re-posts.
    WouldBlock,
    /// The gate refused with `status` (revoked peer, malformed staging…).
    Refused(Status),
    /// The transport could not commit at all (the trap is not yet open).
    Transport,
}

/// Why a recv refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecvError {
    /// The gate refused the post with `status`.
    Refused(Status),
    /// The transport could not commit at all.
    Transport,
}

/// An exact affine rendezvous could either prove that the caller still owns
/// the source or cross a boundary after which recreating that owner would be
/// unsound.
#[derive(Debug)]
pub enum ExactTransferError<C> {
    /// I3 proved the complete session rolled back. The same affine owner is
    /// returned and may be retried or retired by the caller.
    NotMoved { error: SendError, capability: C },
    /// The transport or completion could not prove where custody settled.
    /// The local arena name remains quarantined and no owner is fabricated.
    AfterTransfer(SendError),
}

mod exact_transfer_sealed {
    use super::*;

    pub trait Sealed: Sized {
        fn source_slot(&self) -> SlotRef;
        fn release_after_move(self);
    }

    impl Sealed for StashedLease {
        fn source_slot(&self) -> SlotRef {
            self.slot()
                .expect("a consumed exact-transfer lease must still be live")
        }

        fn release_after_move(self) {
            self.release_after_proven_move();
        }
    }

    impl Sealed for ChanSendToken {
        fn source_slot(&self) -> SlotRef {
            self.slot()
        }

        fn release_after_move(self) {
            self.release_after_proven_move();
        }
    }

    impl Sealed for ChanRecvToken {
        fn source_slot(&self) -> SlotRef {
            self.slot()
        }

        fn release_after_move(self) {
            self.release_after_proven_move();
        }
    }
}

/// The closed set of runtime-owned capability species the kernel's exact
/// rendezvous move accepts. This is deliberately sealed: callers cannot teach
/// the runtime how to discard an arbitrary affine owner after a move.
pub trait ExactTransferCapability: exact_transfer_sealed::Sealed {}

impl ExactTransferCapability for StashedLease {}
impl ExactTransferCapability for ChanSendToken {}
impl ExactTransferCapability for ChanRecvToken {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DerivedTransfer {
    Moved,
    RolledBack(SendError),
    Indeterminate,
}

/// Completion of a post into a runtime-rotated landing.
///
/// `WouldBlock` is not evidence that this fresh landing stands. It means the
/// posting session rolled back because the channel was still busy with the
/// preceding handoff. Fixed receivers may infer an older post still stands in
/// that case because they reuse one landing; [`CapRx`] and [`NoticeRx`] rotate
/// to a new reservation and must retry before advertising capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RotatingPostCommit {
    Standing,
    Retry,
    Refused(Status),
}

fn classify_rotating_post(committed: kore_abi::session::Committed) -> RotatingPostCommit {
    match committed.status() {
        Status::Ok => RotatingPostCommit::Standing,
        Status::WouldBlock => RotatingPostCommit::Retry,
        status => RotatingPostCommit::Refused(status),
    }
}

fn classify_derived_transfer(
    committed: kore_abi::session::Committed,
    invocations: usize,
) -> DerivedTransfer {
    let status = committed.status();
    let failing = usize::from(committed.failing_index());
    if status == Status::Ok && failing == 0 && invocations != 0 {
        return DerivedTransfer::Moved;
    }
    if failing < invocations
        && matches!(
            status,
            Status::Denied
                | Status::Forged
                | Status::Revoked
                | Status::WouldBlock
                | Status::Conflict
        )
    {
        return DerivedTransfer::RolledBack(if status == Status::WouldBlock {
            SendError::WouldBlock
        } else {
            SendError::Refused(status)
        });
    }
    DerivedTransfer::Indeterminate
}

/// The **send end** witness — a peer's kick-and-transfer authority over
/// one channel (`res/USERLAND.md` §4, pinned).
pub struct ChanTx<'t, T: Transport> {
    transport: &'t T,
    /// The held `ChanSend` record's slot.
    send: SlotRef,
}

impl<'t, T: Transport> ChanTx<'t, T> {
    /// A send-end witness over the `ChanSend` record in `send`.
    pub fn new(transport: &'t T, send: SlotRef) -> Self {
        Self { transport, send }
    }

    /// One session-committed **notice-mode send** (§3.4.1) — its own
    /// session, never batched (`WouldBlock` is a whole-session refusal,
    /// §3.1).  `Err(WouldBlock)` is droppable exactly when the §3.1
    /// order holds for the receiver: the caller's ring write
    /// happens-before this kick, and a consumed post means the receiver
    /// is awake and will drain every source after it re-posts.
    pub async fn send(&self, notice: Notice) -> Result<(), SendError> {
        let mut s = SessionBuilder::new();
        s.notice(self.send, notice.cookie, notice.payload)
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
            Status::Ok => Ok(()),
            Status::WouldBlock => Err(SendError::WouldBlock),
            s => Err(SendError::Refused(s)),
        }
    }

    /// Capability transfer (the landed send): `DERIVE`-gated
    /// re-derivation of `staged`'s capability into the peer's posted
    /// slot, attenuated to `rights`.  Grant-bearing replies use this
    /// (§5.5); the peer's posted recv pins where it lands.
    pub async fn transfer(&self, staged: Staged, rights: Rights) -> Result<(), SendError> {
        let mut s = SessionBuilder::new();
        s.send(self.send, staged, rights)
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
            Status::Ok => Ok(()),
            Status::WouldBlock => Err(SendError::WouldBlock),
            s => Err(SendError::Refused(s)),
        }
    }

    /// Derive an attenuated child from a held send endpoint and transfer that
    /// child in the same atomic session.  The held source remains local; only
    /// the fresh lineage leaf crosses the rendezvous.
    pub async fn transfer_derived_send(
        &self,
        source: SlotRef,
        rights: Rights,
    ) -> Result<(), SendError> {
        let mut reservation =
            StagedCapabilityReservation::reserve().ok_or(SendError::Refused(Status::Denied))?;
        let mut session = SessionBuilder::new();
        // A rendezvous transfer is itself a derivation gate.  The staged
        // scratch therefore needs DERIVE even when the *delivered* leaf is
        // execute-only; the send below is the final attenuation and strips
        // that scratch-only pivot before the capability crosses tasks.
        let staged = session
            .derive_chan_send(source, rights | Rights::DERIVE, reservation.target())
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        session
            .send(self.send, staged, rights)
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        reservation.begin_transfer();
        let committed = match self.transport.commit(&session).await {
            Ok(committed) => committed,
            Err(_) => {
                reservation.settle_proven();
                return Err(SendError::Transport);
            }
        };
        match classify_derived_transfer(committed, session.staged()) {
            DerivedTransfer::Moved => {
                reservation.settle_proven();
                Ok(())
            }
            DerivedTransfer::RolledBack(error) => {
                reservation.settle_proven();
                Err(error)
            }
            DerivedTransfer::Indeterminate => {
                reservation.quarantine();
                Err(SendError::Refused(Status::Malformed))
            }
        }
    }

    /// Derive an attenuated child from a held receive endpoint and transfer
    /// that child atomically to this channel.
    ///
    /// This is the task-owned executor-endpoint counterpart of
    /// [`transfer_derived_send`](Self::transfer_derived_send): a launcher
    /// keeps the revocation pivot while the founded command receives only
    /// its private wait endpoint. It conveys no command bytes and grants no
    /// authority over any other task's lifecycle.
    pub async fn transfer_derived_recv(
        &self,
        source: SlotRef,
        rights: Rights,
    ) -> Result<(), SendError> {
        let mut reservation =
            StagedCapabilityReservation::reserve().ok_or(SendError::Refused(Status::Denied))?;
        let mut session = SessionBuilder::new();
        let staged = session
            .derive_chan_recv(source, rights | Rights::DERIVE, reservation.target())
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        session
            .send(self.send, staged, rights)
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        reservation.begin_transfer();
        let committed = match self.transport.commit(&session).await {
            Ok(committed) => committed,
            Err(_) => {
                reservation.settle_proven();
                return Err(SendError::Transport);
            }
        };
        match classify_derived_transfer(committed, session.staged()) {
            DerivedTransfer::Moved => {
                reservation.settle_proven();
                Ok(())
            }
            DerivedTransfer::RolledBack(error) => {
                reservation.settle_proven();
                Err(error)
            }
            DerivedTransfer::Indeterminate => {
                reservation.quarantine();
                Err(SendError::Refused(Status::Malformed))
            }
        }
    }

    /// Derive and transfer a bounded child of the caller's rendezvous mint.
    /// The machine root never crosses tasks; the receiver can mint only below
    /// this revocable lineage.
    pub async fn transfer_derived_channel_space(&self, source: SlotRef) -> Result<(), SendError> {
        let mut reservation =
            StagedCapabilityReservation::reserve().ok_or(SendError::Refused(Status::Denied))?;
        let mut session = SessionBuilder::new();
        let staged = session
            .derive_channel_space_grant(
                source,
                crate::cap::CHANNEL_SPACE_RIGHTS,
                reservation.target(),
            )
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        session
            .send(self.send, staged, crate::cap::CHANNEL_SPACE_RIGHTS)
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        reservation.begin_transfer();
        let committed = match self.transport.commit(&session).await {
            Ok(committed) => committed,
            Err(_) => {
                reservation.settle_proven();
                return Err(SendError::Transport);
            }
        };
        match classify_derived_transfer(committed, session.staged()) {
            DerivedTransfer::Moved => {
                reservation.settle_proven();
                Ok(())
            }
            DerivedTransfer::RolledBack(error) => {
                reservation.settle_proven();
                Err(error)
            }
            DerivedTransfer::Indeterminate => {
                reservation.quarantine();
                Err(SendError::Refused(Status::Malformed))
            }
        }
    }

    /// Carve one bounded child lease from a held lessor and transfer that
    /// affine child directly to this endpoint. The source lessor remains with
    /// the caller; the receiver gets only the requested window and rights.
    pub async fn transfer_sublease(
        &self,
        source: SlotRef,
        offset: u64,
        len: u64,
        rights: Rights,
    ) -> Result<(), SendError> {
        let mut reservation =
            StagedCapabilityReservation::reserve().ok_or(SendError::Refused(Status::Denied))?;
        let mut session = SessionBuilder::new();
        // `SessionBuilder::send` is itself derive-shaped: the staged child is
        // re-derived beneath the receiver's standing post. Keep DERIVE only
        // on this sender-owned intermediate, then attenuate it away in the
        // rendezvous operation below. Carving the intermediate directly with
        // `rights` makes every execute-only/final A|M|R sublease unreachable:
        // the kernel correctly refuses the second derivation with Denied.
        let staging_rights = sublease_transfer_staging_rights(rights);
        let staged = session
            .carve_sublease(source, staging_rights, offset, len, reservation.target())
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        session
            .send(self.send, staged, rights)
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        reservation.begin_transfer();
        let committed = match self.transport.commit(&session).await {
            Ok(committed) => committed,
            Err(_) => {
                reservation.settle_proven();
                return Err(SendError::Transport);
            }
        };
        match classify_derived_transfer(committed, session.staged()) {
            DerivedTransfer::Moved => {
                reservation.settle_proven();
                Ok(())
            }
            DerivedTransfer::RolledBack(error) => {
                reservation.settle_proven();
                Err(error)
            }
            DerivedTransfer::Indeterminate => {
                reservation.quarantine();
                Err(SendError::Refused(Status::Malformed))
            }
        }
    }

    /// Move one exact runtime-owned affine capability through this channel.
    ///
    /// Unlike [`transfer`](Self::transfer), this operation cannot attenuate
    /// and does not require `DERIVE` on the source. It is the delegation shape
    /// for already-attenuated receive ends and other single-owner session
    /// components. Only an I3-proven rollback returns the original owner;
    /// ambiguous completion quarantines the local name and remains fail-closed.
    pub async fn transfer_exact<C: ExactTransferCapability>(
        &self,
        capability: C,
    ) -> Result<(), ExactTransferError<C>> {
        let source = exact_transfer_sealed::Sealed::source_slot(&capability);
        let mut session = SessionBuilder::new();
        if session.send_exact(self.send, source).is_err() {
            return Err(ExactTransferError::NotMoved {
                error: SendError::Refused(Status::Malformed),
                capability,
            });
        }
        let committed = match self.transport.commit(&session).await {
            Ok(committed) => committed,
            Err(_) => {
                core::mem::forget(capability);
                return Err(ExactTransferError::AfterTransfer(SendError::Transport));
            }
        };
        match classify_derived_transfer(committed, session.staged()) {
            DerivedTransfer::Moved => {
                exact_transfer_sealed::Sealed::release_after_move(capability);
                Ok(())
            }
            DerivedTransfer::RolledBack(error) => {
                Err(ExactTransferError::NotMoved { error, capability })
            }
            DerivedTransfer::Indeterminate => {
                core::mem::forget(capability);
                Err(ExactTransferError::AfterTransfer(SendError::Refused(
                    Status::Malformed,
                )))
            }
        }
    }

    /// Carve one bounded detached view from a held lessor and transfer it to
    /// this endpoint. The sender-owned intermediate is delegable only long
    /// enough for the derive-shaped rendezvous; the receiver gets exactly
    /// `rights` and therefore cannot delegate the memory window further.
    pub async fn transfer_view(
        &self,
        source: SlotRef,
        offset: u64,
        len: u64,
        rights: Rights,
    ) -> Result<(), SendError> {
        let mut reservation =
            StagedCapabilityReservation::reserve().ok_or(SendError::Refused(Status::Denied))?;
        let mut session = SessionBuilder::new();
        let staged = session
            .carve_window(
                source,
                sublease_transfer_staging_rights(rights),
                offset,
                len,
                reservation.target(),
            )
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        session
            .send(self.send, staged, rights)
            .map_err(|_| SendError::Refused(Status::Malformed))?;
        reservation.begin_transfer();
        let committed = match self.transport.commit(&session).await {
            Ok(committed) => committed,
            Err(_) => {
                reservation.settle_proven();
                return Err(SendError::Transport);
            }
        };
        match classify_derived_transfer(committed, session.staged()) {
            DerivedTransfer::Moved => {
                reservation.settle_proven();
                Ok(())
            }
            DerivedTransfer::RolledBack(error) => {
                reservation.settle_proven();
                Err(error)
            }
            DerivedTransfer::Indeterminate => {
                reservation.quarantine();
                Err(SendError::Refused(Status::Malformed))
            }
        }
    }
}

/// Rights for the sender-owned intermediate used by a derived rendezvous.
/// The receiver still receives exactly `final_rights` through `send`.
const fn sublease_transfer_staging_rights(final_rights: Rights) -> Rights {
    final_rights.union(Rights::DERIVE)
}

/// The **recv end** witness — one channel's standing-post owner
/// (`res/USERLAND.md` §4, pinned).  `&mut` methods: one future awaits a
/// given `ChanRx` at a time.
pub struct ChanRx<'t, T: Transport> {
    transport: &'t T,
    /// The held `ChanRecv` record's slot.
    recv: SlotRef,
    /// The landing slot every post of this end donates.
    land: SlotRef,
    /// Whether this end believes its post stands (re-posted lazily).
    standing: bool,
}

/// A receive end whose landing is runtime-owned and therefore may retain an
/// unbounded sequence of affine capability deliveries over time.  Unlike
/// [`ChanRx`], it accepts capabilities only: a notice on this lane is a
/// protocol violation and cannot be confused with authority.
pub struct CapRx<'t, T: Transport> {
    transport: &'t T,
    recv: SlotRef,
    landing: Option<ReturnedCapability>,
    standing: bool,
}

/// A receive endpoint whose landing is runtime-owned and which accepts only
/// notices. This is the affine counterpart of [`CapRx`] for transient typed
/// completion channels: callers do not reserve or guess a fixed arena slot.
pub struct NoticeRx<'t, T: Transport> {
    transport: &'t T,
    recv: SlotRef,
    landing: Option<ReturnedCapability>,
    standing: bool,
}

impl<'t, T: Transport> NoticeRx<'t, T> {
    pub fn new(transport: &'t T, recv: SlotRef) -> Self {
        Self {
            transport,
            recv,
            landing: None,
            standing: false,
        }
    }

    /// Consume one notice or ensure a fresh post stands. A delivered
    /// capability is a protocol violation and is retired rather than exposed.
    pub fn try_recv(&mut self) -> Result<Option<Notice>, RecvError> {
        if self.standing {
            let Some(landing) = self.landing.as_mut() else {
                return Err(RecvError::Refused(Status::Malformed));
            };
            match verbs::take_delivery(landing.target()) {
                Some(RawDelivery::Notice(cookie, payload)) => {
                    self.standing = false;
                    let mut consumed = self
                        .landing
                        .take()
                        .ok_or(RecvError::Refused(Status::Malformed))?;
                    consumed.disarm_post();
                    return Ok(Some(Notice { cookie, payload }));
                }
                Some(RawDelivery::Cap) => {
                    self.standing = false;
                    let mut landed = self
                        .landing
                        .take()
                        .ok_or(RecvError::Refused(Status::Malformed))?;
                    landed.activate();
                    let _ = landed.try_retire();
                    return Err(RecvError::Refused(Status::Malformed));
                }
                None => return Ok(None),
            }
        }

        // A landing retained by `rearm` is reused: the same slot, scrubbed
        // by the repost, so a post the kernel already resolved cannot leave
        // its reservation stranded.
        let mut landing = match self.landing.take() {
            Some(landing) => landing,
            None => ReturnedCapability::reserve().ok_or(RecvError::Refused(Status::Denied))?,
        };
        let mut session = SessionBuilder::new();
        session
            .post_recv(self.recv, landing.target())
            .map_err(|_| RecvError::Refused(Status::Malformed))?;
        match self.transport.commit_now(&session) {
            Ok(c) if classify_rotating_post(c) == RotatingPostCommit::Standing => {
                landing.arm_post();
                self.landing = Some(landing);
                self.standing = true;
                Ok(None)
            }
            // A refused repost retains the landing (Reserved, never released):
            // after `rearm` it may be the target of a post the kernel still
            // holds, and a released slot re-reserved elsewhere would put two
            // owners under one standing post. Retained, it is simply reused
            // by the next attempt; a fresh reservation costs nothing to keep.
            Ok(c) => {
                self.landing = Some(landing);
                match classify_rotating_post(c) {
                    RotatingPostCommit::Retry => Err(RecvError::Refused(Status::WouldBlock)),
                    RotatingPostCommit::Refused(status) => Err(RecvError::Refused(status)),
                    RotatingPostCommit::Standing => unreachable!("handled above"),
                }
            }
            Err(_) => {
                self.landing = Some(landing);
                Err(RecvError::Transport)
            }
        }
    }

    /// The affine post currently standing for this receiver, for composition
    /// as the executor's one alternate delivery arm.
    ///
    /// Call [`try_recv`](Self::try_recv) first: it either consumes a delivered
    /// notice or establishes this post. The returned description carries no
    /// authority by itself; `self` retains both the receive endpoint and its
    /// runtime-reserved landing and must consume/re-post after a delivery.
    /// The standing post, for `executor::next_wake_on` — `None` until a
    /// [`try_recv`](Self::try_recv) has stood one.
    pub fn standing_post(&self) -> Option<ParkPost> {
        if !self.standing {
            return None;
        }
        let landing = self.landing.as_ref()?;
        let (chan, recv_tag) = verbs::recv_pin(self.recv)?;
        Some(ParkPost {
            chan,
            land: landing.target(),
            recv_tag,
        })
    }

    /// Await one notice on a capability-owned configuration channel. These
    /// channels intentionally have no separate wake endpoint, so a bounded
    /// executor timer supplies the level-triggered re-poll.
    pub async fn recv(&mut self) -> Result<Notice, RecvError> {
        loop {
            match self.try_recv() {
                Ok(Some(notice)) => return Ok(notice),
                Ok(None) | Err(RecvError::Refused(Status::WouldBlock)) => {
                    crate::executor::next_wake_for(100_000).await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Forget a standing post the kernel has already resolved. The executor's
    /// alternate arm reports `Delivery` for a post it found gone; when the
    /// landing then reads vacant (a spurious wake — no record reached it),
    /// parking again on the same belief would resolve gone again at once, a
    /// livelock at trap speed. Disarm the belief and keep the reserved
    /// landing: the next [`try_recv`](Self::try_recv) re-posts onto it.
    pub fn rearm(&mut self) {
        if !self.standing {
            return;
        }
        if let Some(landing) = self.landing.as_mut() {
            landing.disarm_post();
        }
        self.standing = false;
    }

    /// Settle an alternate-delivery wake on this post: the notice, if one
    /// landed, or `None` after re-arming a spuriously resolved post. Every
    /// alternate-post consumer treats the notice as a wake and re-reads its
    /// own state for truth, so `None` is ordinary, never an error.
    pub fn settle_delivery(&mut self) -> Result<Option<Notice>, RecvError> {
        if !self.standing {
            return self.try_recv();
        }
        let delivered = self.try_recv()?;
        if delivered.is_none() {
            self.rearm();
        }
        Ok(delivered)
    }

    /// Release a possibly standing runtime-owned landing after the caller has
    /// revoked this channel's root. Root retirement is the proof that no late
    /// delivery can race reuse of the donated arena slot.
    pub fn close_after_root_revoked(mut self) {
        if let Some(landing) = self.landing.as_mut() {
            landing.disarm_post();
        }
        self.landing = None;
        self.standing = false;
    }
}

impl<'t, T: Transport> CapRx<'t, T> {
    pub fn new(transport: &'t T, recv: SlotRef) -> Self {
        Self {
            transport,
            recv,
            landing: None,
            standing: false,
        }
    }

    /// Consume one capability if present, otherwise ensure a fresh standing
    /// post.  The returned owner carries both the landed record and its arena
    /// reservation, so a later post can safely use a different slot.
    pub fn try_recv(&mut self) -> Result<Option<ReturnedCapability>, RecvError> {
        if self.standing {
            let Some(landing) = self.landing.as_mut() else {
                return Err(RecvError::Refused(Status::Malformed));
            };
            match verbs::take_delivery(landing.target()) {
                Some(RawDelivery::Cap) => {
                    self.standing = false;
                    let mut landed = self
                        .landing
                        .take()
                        .ok_or(RecvError::Refused(Status::Malformed))?;
                    landed.activate();
                    return Ok(Some(landed));
                }
                Some(RawDelivery::Notice(_, _)) => {
                    self.standing = false;
                    let mut landed = self
                        .landing
                        .take()
                        .ok_or(RecvError::Refused(Status::Malformed))?;
                    landed.disarm_post();
                    return Err(RecvError::Refused(Status::Malformed));
                }
                None => return Ok(None),
            }
        }

        let mut landing =
            ReturnedCapability::reserve().ok_or(RecvError::Refused(Status::Denied))?;
        let mut session = SessionBuilder::new();
        session
            .post_recv(self.recv, landing.target())
            .map_err(|_| RecvError::Refused(Status::Malformed))?;
        match self.transport.commit_now(&session) {
            Ok(c) if classify_rotating_post(c) == RotatingPostCommit::Standing => {
                landing.arm_post();
                self.landing = Some(landing);
                self.standing = true;
                Ok(None)
            }
            Ok(c) => match classify_rotating_post(c) {
                RotatingPostCommit::Retry => Err(RecvError::Refused(Status::WouldBlock)),
                RotatingPostCommit::Refused(status) => Err(RecvError::Refused(status)),
                RotatingPostCommit::Standing => unreachable!("handled above"),
            },
            Err(_) => Err(RecvError::Transport),
        }
    }

    /// Await one capability using the executor's level-triggered wake path.
    /// The post itself remains synchronous runtime bookkeeping; only the wait
    /// yields, so there is still exactly one standing donation per channel.
    pub async fn recv(&mut self) -> Result<ReturnedCapability, RecvError> {
        loop {
            match self.try_recv() {
                Ok(Some(capability)) => return Ok(capability),
                Ok(None) | Err(RecvError::Refused(Status::WouldBlock)) => {
                    // Capability-only configuration channels have no separate
                    // notice endpoint. Keep their wait bounded and
                    // level-triggered without relying on an unregistered
                    // waker to poll this future a second time.
                    crate::executor::next_wake_for(100_000).await;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

impl<'t, T: Transport> ChanRx<'t, T> {
    /// A recv-end witness over the `ChanRecv` record in `recv`,
    /// delivering into `land`.
    pub fn new(transport: &'t T, recv: SlotRef, land: SlotRef) -> Self {
        Self {
            transport,
            recv,
            land,
            standing: false,
        }
    }

    /// Consume one delivery if present, otherwise ensure the standing post
    /// and return immediately. Serve loops with several independent sources
    /// use this during their total drain before awaiting the one executor
    /// doorbell.
    pub fn try_recv(&mut self) -> Result<Option<Delivery>, RecvError> {
        // The landing alias is read-only. Consuming a delivery therefore
        // marks the post non-standing; the *next repost* is what clears the
        // old landing record in the kernel. Never inspect that record again
        // while non-standing, or the same notice is reported twice.
        if self.standing {
            if let Some(raw) = verbs::take_delivery(self.land) {
                self.standing = false;
                let delivery = match raw {
                    RawDelivery::Notice(cookie, payload) => {
                        Delivery::Notice(Notice { cookie, payload })
                    }
                    RawDelivery::Cap => Delivery::Cap(self.land),
                };
                return Ok(Some(delivery));
            }
            return Ok(None);
        }

        let mut s = SessionBuilder::new();
        if s.post_recv(self.recv, self.land).is_err() {
            return Err(RecvError::Refused(Status::Malformed));
        }
        match self.transport.commit_now(&s) {
            Ok(c) if matches!(c.status(), Status::Ok | Status::WouldBlock) => {
                self.standing = true;
            }
            Ok(c) => return Err(RecvError::Refused(c.status())),
            Err(_) => return Err(RecvError::Transport),
        }
        Ok(None)
    }

    /// Receive the next delivery: **post (if not standing) as its own
    /// commit; Ready if a delivery is consumable, else Pending** —
    /// level-triggered, re-polled by the executor's §3.1 drain.
    /// **NEVER parks** — the executor's doorbell idle-park is the
    /// runtime's only `WAIT_DELIVERY` (§4).
    ///
    /// The post-if-not-standing rides the sync spine (`commit_now`): a
    /// `poll` cannot await a nested commit, and the post is exactly the
    /// kind of runtime-internal bookkeeping commit the spine exists for
    /// — one trap on target.
    pub async fn recv(&mut self) -> Result<Delivery, RecvError> {
        core::future::poll_fn(|_cx| match self.try_recv() {
            Ok(Some(delivery)) => core::task::Poll::Ready(Ok(delivery)),
            Ok(None) => core::task::Poll::Pending,
            Err(error) => core::task::Poll::Ready(Err(error)),
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::session::Committed;

    #[test]
    fn derived_transfer_completion_is_fail_closed() {
        assert_eq!(
            classify_derived_transfer(Committed::from_raw(0), 2),
            DerivedTransfer::Moved
        );
        assert_eq!(
            classify_derived_transfer(
                Committed::from_raw(Status::WouldBlock as u64 | (1u64 << 16)),
                2,
            ),
            DerivedTransfer::RolledBack(SendError::WouldBlock)
        );
        assert_eq!(
            classify_derived_transfer(
                Committed::from_raw(Status::Conflict as u64 | (1u64 << 16)),
                2,
            ),
            DerivedTransfer::RolledBack(SendError::Refused(Status::Conflict))
        );
        assert_eq!(
            classify_derived_transfer(
                Committed::from_raw(Status::Malformed as u64 | (2u64 << 16)),
                2,
            ),
            DerivedTransfer::Indeterminate
        );
    }

    #[test]
    fn sublease_transfer_keeps_derive_only_on_the_intermediate() {
        let final_rights = Rights::ACCESS | Rights::MODIFY | Rights::REVOKE;
        let staging = sublease_transfer_staging_rights(final_rights);
        assert_eq!(staging, final_rights | Rights::DERIVE);
        assert!(!final_rights.contains(Rights::DERIVE));
    }

    #[test]
    fn rotating_post_never_claims_rolled_back_capacity() {
        assert_eq!(
            classify_rotating_post(Committed::from_raw(Status::Ok as u64)),
            RotatingPostCommit::Standing
        );
        assert_eq!(
            classify_rotating_post(Committed::from_raw(Status::WouldBlock as u64)),
            RotatingPostCommit::Retry
        );
        assert_eq!(
            classify_rotating_post(Committed::from_raw(Status::Revoked as u64)),
            RotatingPostCommit::Refused(Status::Revoked)
        );
    }
}
