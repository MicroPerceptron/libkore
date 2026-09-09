//! One controller driving several designated regions — the completion demux.
//!
//! A disk may carry more than one region this system owns under distinct
//! lifecycle policies (a self-healing build-owned store beside conservative
//! user data).  Each region wants its own [`BlockChannel`] — its own
//! registered buffers, its own request table, its own [`RegionDevice`] clamp
//! — but there is exactly **one** controller beneath them, and a channel
//! owns its backend by value.
//!
//! [`SharedBackend`] owns that one backend and hands out one [`Lane`] per
//! region.  A lane *is* a [`DirectBlockDevice`], so the ordinary composition
//! stacks on top of it unchanged:
//!
//! ```text
//! SharedBackend<D>            one affine backend, one submission/completion queue
//!   ├ Lane 0 → RegionDevice → BlockChannel → provider   (region A)
//!   └ Lane 1 → RegionDevice → BlockChannel → provider   (region B)
//! ```
//!
//! ## Why the naive split is wrong
//!
//! [`BlockChannel`] draws its cookies from a private counter that starts at
//! `1`.  Two channels over one backend therefore mint the **same** cookie
//! sequence, and a completion carries nothing else to tell them apart.  Both
//! failure modes are silent:
//!
//! - **Cross-completion (corruption).**  Lane A's cookie `5` completes; lane
//!   B polls first, matches `5` against *its own* pending table, and settles
//!   B's request — releasing B's registered buffer while B's transfer is
//!   still in flight.
//! - **Dropped completion (a wedge).**  Lane A polls and receives B's
//!   completion.  [`BlockChannel::poll`] cannot place it, reports
//!   `StaleCompletion`, and it is gone.  B waits for a completion that will
//!   never come again.
//!
//! Both are answered by structure, not by care at the call sites.
//!
//! ## The demux
//!
//! **Tagged cookies.**  A lane stamps its index into the cookie's high
//! [`LANE_BITS`] on the way down and strips it on the way up, so the channel
//! above never sees the tag and the two cookie spaces cannot alias.  A
//! payload too wide for the remaining bits is **refused**
//! ([`BlockError::Range`]) rather than truncated: truncation would alias two
//! distinct requests, which is the corruption this exists to prevent.
//!
//! **Per-lane mailboxes.**  Whichever lane drains the backend routes *every*
//! completion it sees by tag — its own included — into the owning lane's
//! mailbox, and then serves itself from its own mailbox.  A completion is
//! therefore never observed by a lane that cannot place it, so it is never
//! dropped.
//!
//! **The mailbox cannot overflow.**  Let `outstanding[L]` be the commands
//! lane `L` has submitted but not yet been *delivered*.  A lane's submission
//! is refused once `outstanding[L] == INFLIGHT`; routing moves a completion
//! from the backend into a mailbox without changing the count; delivery pops
//! the mailbox and decrements it.  So `mail[L].len() <= outstanding[L] <=
//! INFLIGHT` holds at every step, and a mailbox sized `INFLIGHT` has room by
//! construction.  The cap lives here — at the same seam that mints the tag —
//! rather than being inherited from the channel's `OUTSTANDING`, because the
//! bound is what makes the no-drop argument true.
//!
//! A completion that survives all of that (a tag naming a retired lane, a
//! duplicate, a cookie for nothing outstanding) is a backend protocol
//! violation.  It is counted as [`SharedStats::stray`] and discarded, because
//! by the invariant above it corresponds to no live request — but it is
//! counted, since on a serial-less machine a number is the only way that
//! fact ever surfaces.
//!
//! ## Ownership and lifetime
//!
//! Lanes borrow the backend (`&'a SharedBackend`), so the borrow checker
//! proves the controller outlives every lane — no reference count, no
//! allocation, and this crate keeps its empty dependency list.  A composition
//! that must outlive its own stack frame (a service's mounted filesystem)
//! gives the backend a `'static` home; the lifetime is then a concrete
//! `'static` in the provider's type rather than a self-reference.
//!
//! One worker owns and drives all of a backend's lanes, exactly as it owns
//! one channel today.  Interior mutability is [`RefCell`], and no borrow is
//! held across a suspension point — which is why a backend may only be shared
//! when its queue advances on its own
//! ([`DirectBlockDevice::INDEPENDENT_PROGRESS`]); a backend whose `progress`
//! genuinely parks would have to hold the shared borrow across an `await`.
//! [`SharedBackend::new`] refuses such a backend rather than leaving the
//! conflict latent.
//!
//! [`BlockChannel`]: crate::block::BlockChannel
//! [`BlockChannel::poll`]: crate::block::BlockChannel::poll
//! [`RegionDevice`]: crate::block::RegionDevice

use core::cell::{Cell, RefCell};

use super::{
    BlockError, BlockGeometry, BlockHealth, DeviceCommand, DeviceCompletion, DirectBlockDevice,
};

/// Width of the lane tag carried in a shared cookie's high bits.
pub const LANE_BITS: u32 = 8;
/// Bit position of the lane tag.
const LANE_SHIFT: u32 = u64::BITS - LANE_BITS;
/// Lanes one shared backend can address with [`LANE_BITS`].
pub const MAX_LANES: usize = 1usize << LANE_BITS;
/// Largest request cookie a lane may carry beneath its tag.  A wider cookie
/// is refused, never truncated — truncation aliases distinct requests.
pub const MAX_LANE_COOKIE: u64 = (1u64 << LANE_SHIFT) - 1;

/// Counters describing how the demux behaved.  Read without borrowing the
/// backend, so a status answer can never contend with the data plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedStats {
    /// Completions drained from the backend and placed in a mailbox.
    pub routed: u64,
    /// Completions handed up to their owning lane.
    pub delivered: u64,
    /// Completions belonging to no live outstanding request — a backend
    /// protocol violation, counted rather than merely discarded.
    pub stray: u64,
    /// Submissions refused because the lane was already at `INFLIGHT`.
    pub backpressure: u64,
    /// Times the shared borrow was already held.  Structurally unreachable
    /// (one worker, no borrow across a suspension); non-zero means that
    /// invariant broke and the demux degraded instead of panicking.
    pub conflicts: u64,
}

/// A lane's bounded holding queue for completions drained by another lane.
struct Mailbox<const N: usize> {
    slots: [Option<DeviceCompletion>; N],
    head: usize,
    count: usize,
}

impl<const N: usize> Mailbox<N> {
    fn new() -> Self {
        Self {
            slots: [None; N],
            head: 0,
            count: 0,
        }
    }

    fn len(&self) -> usize {
        self.count
    }

    /// Place one completion.  `false` means the invariant documented on this
    /// module broke — the caller counts it as stray rather than overwriting a
    /// completion that a lane is still owed.
    fn push(&mut self, completion: DeviceCompletion) -> bool {
        if self.count == N {
            return false;
        }
        let tail = (self.head + self.count) % N;
        self.slots[tail] = Some(completion);
        self.count += 1;
        true
    }

    fn pop(&mut self) -> Option<DeviceCompletion> {
        if self.count == 0 {
            return None;
        }
        let completion = self.slots[self.head].take();
        self.head = (self.head + 1) % N;
        self.count -= 1;
        completion
    }
}

/// The shared mutable half: the backend and the demux books.
struct Inner<D: DirectBlockDevice, const LANES: usize, const INFLIGHT: usize> {
    device: D,
    mail: [Mailbox<INFLIGHT>; LANES],
    /// Per lane: submitted and not yet delivered.  Bounds the mailbox.
    outstanding: [usize; LANES],
    /// Per lane: handed out once (a lane is affine — never claimed twice).
    claimed: [bool; LANES],
    /// Per lane: claimed and not yet retired.
    live: [bool; LANES],
}

/// One backend shared by `LANES` cookie-disjoint lanes, each admitting at
/// most `INFLIGHT` commands.
pub struct SharedBackend<D: DirectBlockDevice, const LANES: usize, const INFLIGHT: usize> {
    inner: RefCell<Inner<D, LANES, INFLIGHT>>,
    /// Cached at construction: geometry is immutable for a provisioned
    /// backend, so the hot read path never takes the shared borrow.
    geometry: BlockGeometry,
    max_buffer_bytes: usize,
    routed: Cell<u64>,
    delivered: Cell<u64>,
    stray: Cell<u64>,
    backpressure: Cell<u64>,
    conflicts: Cell<u64>,
}

impl<D: DirectBlockDevice, const LANES: usize, const INFLIGHT: usize>
    SharedBackend<D, LANES, INFLIGHT>
{
    /// Take ownership of one backend to share.
    ///
    /// Refuses a backend whose queue needs an executor-aware
    /// [`progress`](DirectBlockDevice::progress) step
    /// ([`BlockError::Unsupported`]): a parking `progress` future would have
    /// to hold the shared borrow across an `await`, and refusing at
    /// construction is the only place that conflict can be named before it
    /// becomes a runtime fault.
    pub fn new(device: D) -> Result<Self, BlockError> {
        if LANES == 0 || LANES > MAX_LANES || INFLIGHT == 0 {
            return Err(BlockError::Geometry);
        }
        if !D::INDEPENDENT_PROGRESS {
            return Err(BlockError::Unsupported);
        }
        let geometry = device.geometry().validate()?;
        let max_buffer_bytes = device.max_buffer_bytes();
        Ok(Self {
            inner: RefCell::new(Inner {
                device,
                mail: core::array::from_fn(|_| Mailbox::new()),
                outstanding: [0; LANES],
                claimed: [false; LANES],
                live: [false; LANES],
            }),
            geometry,
            max_buffer_bytes,
            routed: Cell::new(0),
            delivered: Cell::new(0),
            stray: Cell::new(0),
            backpressure: Cell::new(0),
            conflicts: Cell::new(0),
        })
    }

    /// The whole backend's geometry.  A lane reports this unchanged; the
    /// designated extent is applied by the [`RegionDevice`](crate::block::RegionDevice)
    /// clamp above the lane, which is what keeps the clamp beneath the
    /// channel where code above it cannot name a block outside the region.
    pub fn geometry(&self) -> BlockGeometry {
        self.geometry
    }

    /// Claim lane `lane`.  Refused if it is out of range or already claimed —
    /// a lane is affine, so two compositions can never share one cookie tag.
    pub fn claim(&self, lane: usize) -> Result<Lane<'_, D, LANES, INFLIGHT>, BlockError> {
        if lane >= LANES {
            return Err(BlockError::Range);
        }
        let mut inner = self.inner.try_borrow_mut().map_err(|_| self.conflict())?;
        if inner.claimed[lane] {
            return Err(BlockError::Backpressure);
        }
        inner.claimed[lane] = true;
        inner.live[lane] = true;
        Ok(Lane { core: self, lane })
    }

    /// How many commands lane `lane` has submitted and not yet been
    /// delivered.  `None` when the books are momentarily borrowed.
    pub fn outstanding(&self, lane: usize) -> Option<usize> {
        let inner = self.inner.try_borrow().ok()?;
        inner.outstanding.get(lane).copied()
    }

    /// Whether lane `lane` is claimed and not yet retired.
    pub fn is_live(&self, lane: usize) -> bool {
        self.inner
            .try_borrow()
            .ok()
            .and_then(|inner| inner.live.get(lane).copied())
            .unwrap_or(false)
    }

    pub fn stats(&self) -> SharedStats {
        SharedStats {
            routed: self.routed.get(),
            delivered: self.delivered.get(),
            stray: self.stray.get(),
            backpressure: self.backpressure.get(),
            conflicts: self.conflicts.get(),
        }
    }

    /// Run one infrequent control-plane operation against the shared physical
    /// backend. No lane may retain the borrow and no request may be in flight;
    /// either conflict is a typed transport failure rather than a `RefCell`
    /// panic or control/data-plane overlap.
    pub fn control<T>(&self, f: impl FnOnce(&mut D) -> T) -> Result<T, BlockError> {
        let mut inner = self.inner.try_borrow_mut().map_err(|_| self.conflict())?;
        if inner
            .outstanding
            .iter()
            .any(|outstanding| *outstanding != 0)
        {
            return Err(BlockError::Draining);
        }
        Ok(f(&mut inner.device))
    }

    /// Recover the backend.  Reachable only once every lane is gone, which
    /// the borrow checker enforces: a lane holds `&self`.
    pub fn into_device(self) -> D {
        self.inner.into_inner().device
    }

    fn conflict(&self) -> BlockError {
        self.conflicts.set(self.conflicts.get().saturating_add(1));
        BlockError::Transport
    }
}

/// One lane of a shared backend: a [`DirectBlockDevice`] whose cookies carry
/// this lane's tag and whose completions are demuxed to it alone.
pub struct Lane<'a, D: DirectBlockDevice, const LANES: usize, const INFLIGHT: usize> {
    core: &'a SharedBackend<D, LANES, INFLIGHT>,
    lane: usize,
}

impl<'a, D: DirectBlockDevice, const LANES: usize, const INFLIGHT: usize>
    Lane<'a, D, LANES, INFLIGHT>
{
    /// This lane's index — its cookie tag.
    pub fn index(&self) -> usize {
        self.lane
    }

    /// The backend this lane shares, for stats and sibling inspection.
    pub fn backend(&self) -> &'a SharedBackend<D, LANES, INFLIGHT> {
        self.core
    }

    fn tag(&self, cookie: u64) -> Result<u64, BlockError> {
        if cookie > MAX_LANE_COOKIE {
            return Err(BlockError::Range);
        }
        Ok(((self.lane as u64) << LANE_SHIFT) | cookie)
    }

    fn submit_inner(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut D::Buffer>,
        cookie: u64,
        deferred: bool,
    ) -> Result<(), BlockError> {
        let tagged = self.tag(cookie)?;
        let core = self.core;
        let mut inner = core.inner.try_borrow_mut().map_err(|_| core.conflict())?;
        if !inner.live[self.lane] {
            return Err(BlockError::Revoked);
        }
        // The cap that makes the mailbox large enough by construction.
        if inner.outstanding[self.lane] >= INFLIGHT {
            core.backpressure
                .set(core.backpressure.get().saturating_add(1));
            return Err(BlockError::Backpressure);
        }
        let outcome = if deferred {
            inner.device.stage(command, buffer, tagged)
        } else {
            inner.device.submit(command, buffer, tagged)
        };
        outcome?;
        inner.outstanding[self.lane] += 1;
        Ok(())
    }

    /// File one completion under the lane its tag names.
    ///
    /// A tag outside the lane set, a retired lane, nothing outstanding, or a
    /// full mailbox all mean this completion answers no live request.  The
    /// invariant documented on this module says that cannot happen; count it
    /// so a backend that does it anyway is visible rather than merely
    /// swallowed.
    fn file(
        core: &SharedBackend<D, LANES, INFLIGHT>,
        inner: &mut Inner<D, LANES, INFLIGHT>,
        completion: DeviceCompletion,
    ) {
        let lane = (completion.cookie >> LANE_SHIFT) as usize;
        let placeable = lane < LANES
            && inner.live[lane]
            && inner.outstanding[lane] > inner.mail[lane].len()
            && inner.mail[lane].push(completion);
        if placeable {
            core.routed.set(core.routed.get().saturating_add(1));
        } else {
            core.stray.set(core.stray.get().saturating_add(1));
        }
    }

    /// Drain a bounded batch from the backend into the lane mailboxes — one
    /// completion-head doorbell for the whole prefix — and report how many
    /// moved, so a caller can drain again rather than answer short.
    fn route(
        core: &SharedBackend<D, LANES, INFLIGHT>,
        inner: &mut Inner<D, LANES, INFLIGHT>,
    ) -> usize {
        let mut batch = [None; INFLIGHT];
        let count = inner.device.poll_batch(&mut batch);
        for completion in batch[..count].iter().flatten().copied() {
            Self::file(core, inner, completion);
        }
        count
    }

    /// Drain until this lane holds `want` completions or the backend runs
    /// dry.  A single batch is capped at `INFLIGHT`, so an interleaved queue
    /// can hand back a short prefix while this lane's completions are still
    /// sitting behind a sibling's — and a short answer costs the caller a
    /// wait budget it did not need to spend.  Terminates because every
    /// further pass must move at least one completion off a finite queue.
    fn route_until(
        core: &SharedBackend<D, LANES, INFLIGHT>,
        inner: &mut Inner<D, LANES, INFLIGHT>,
        lane: usize,
        want: usize,
    ) {
        while inner.mail[lane].len() < want && Self::route(core, inner) > 0 {}
    }

    /// Pop this lane's next delivered completion, stripping the tag so the
    /// channel above sees the cookie it minted.
    fn deliver(
        core: &SharedBackend<D, LANES, INFLIGHT>,
        inner: &mut Inner<D, LANES, INFLIGHT>,
        lane: usize,
    ) -> Option<DeviceCompletion> {
        let completion = inner.mail[lane].pop()?;
        inner.outstanding[lane] = inner.outstanding[lane].saturating_sub(1);
        core.delivered.set(core.delivered.get().saturating_add(1));
        Some(DeviceCompletion {
            cookie: completion.cookie & MAX_LANE_COOKIE,
            ..completion
        })
    }
}

impl<'a, D: DirectBlockDevice, const LANES: usize, const INFLIGHT: usize> DirectBlockDevice
    for Lane<'a, D, LANES, INFLIGHT>
{
    type Buffer = D::Buffer;

    /// A shared backend is only ever built over an independently-progressing
    /// queue ([`SharedBackend::new`]), so a lane advances on its own too.
    const INDEPENDENT_PROGRESS: bool = true;

    fn geometry(&self) -> BlockGeometry {
        self.core.geometry
    }

    fn max_buffer_bytes(&self) -> usize {
        self.core.max_buffer_bytes
    }

    fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError> {
        let core = self.core;
        core.inner
            .try_borrow_mut()
            .map_err(|_| core.conflict())?
            .device
            .allocate_buffer(bytes)
    }

    fn health(&mut self) -> Result<BlockHealth, BlockError> {
        let core = self.core;
        core.inner
            .try_borrow_mut()
            .map_err(|_| core.conflict())?
            .device
            .health()
    }

    /// The lanes share one controller, so they share its liveness. A borrow
    /// conflict answers `Unknown` rather than vouching for a device it could
    /// not read.
    fn liveness(&mut self) -> super::Liveness {
        let core = self.core;
        match core.inner.try_borrow_mut() {
            Ok(mut inner) => inner.device.liveness(),
            Err(_) => {
                core.conflict();
                super::Liveness::Unknown
            }
        }
    }

    fn submit(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        self.submit_inner(command, buffer, cookie, false)
    }

    fn stage(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        self.submit_inner(command, buffer, cookie, true)
    }

    fn publish(&mut self) {
        // Publishing a sibling's staged prefix early is lawful — the trait
        // forbids relying on staged commands staying invisible — and one
        // doorbell for both lanes is the point of sharing a queue.
        if let Ok(mut inner) = self.core.inner.try_borrow_mut() {
            inner.device.publish();
        } else {
            self.core.conflict();
        }
    }

    fn finish(&mut self, buffer: Option<&mut Self::Buffer>) -> Result<(), BlockError> {
        let core = self.core;
        core.inner
            .try_borrow_mut()
            .map_err(|_| core.conflict())?
            .device
            .finish(buffer)
    }

    fn poll(&mut self) -> Option<DeviceCompletion> {
        let core = self.core;
        let Ok(mut inner) = core.inner.try_borrow_mut() else {
            core.conflict();
            return None;
        };
        Self::route_until(core, &mut inner, self.lane, 1);
        Self::deliver(core, &mut inner, self.lane)
    }

    fn poll_batch(&mut self, out: &mut [Option<DeviceCompletion>]) -> usize {
        let core = self.core;
        let Ok(mut inner) = core.inner.try_borrow_mut() else {
            core.conflict();
            return 0;
        };
        Self::route_until(core, &mut inner, self.lane, out.len());
        let mut count = 0;
        for slot in out.iter_mut() {
            let Some(completion) = Self::deliver(core, &mut inner, self.lane) else {
                break;
            };
            *slot = Some(completion);
            count += 1;
        }
        count
    }

    /// One bounded wait for *this lane's* next completion.
    ///
    /// A sibling's completion arriving instead is absorbed into its mailbox
    /// and the wait continues — surfacing `None` there would make the caller
    /// declare a live request lost.  The loop is bounded by the completions
    /// that can exist at all: every sibling's `outstanding`, each at most
    /// `INFLIGHT`, plus one wait for this lane's own.
    fn wait(&mut self) -> Option<DeviceCompletion> {
        let core = self.core;
        let Ok(mut inner) = core.inner.try_borrow_mut() else {
            core.conflict();
            return None;
        };
        let mut budget = LANES.saturating_mul(INFLIGHT).saturating_add(1);
        loop {
            Self::route_until(core, &mut inner, self.lane, 1);
            if let Some(completion) = Self::deliver(core, &mut inner, self.lane) {
                return Some(completion);
            }
            if inner.outstanding[self.lane] == 0 {
                // Nothing of ours can complete; never spend a wait budget
                // (or park on a route) on a sibling's behalf.
                return None;
            }
            budget = budget.checked_sub(1)?;
            // `wait` is synchronous — a backend parks the thread on its own
            // delivery route rather than yielding to the executor — so
            // holding the shared borrow across it cannot conflict.
            let completion = inner.device.wait()?;
            Self::file(core, &mut inner, completion);
        }
    }

    /// Route whatever the backend has already delivered.
    ///
    /// A shared backend's queue advances on its own, so there is no
    /// correlated step to await here — and a lane must not hold the shared
    /// borrow across a suspension point.  Draining into the mailboxes is the
    /// real progress this seam can make.
    async fn progress(&mut self) -> Result<(), BlockError> {
        let core = self.core;
        let mut inner = core.inner.try_borrow_mut().map_err(|_| core.conflict())?;
        while Self::route(core, &mut inner) > 0 {}
        Ok(())
    }

    /// Retire this lane.  The backend itself is quiesced by the **last** lane
    /// to retire: tearing the controller down under a sibling's in-flight
    /// transfer would strand its DMA.
    fn quiesce(&mut self) -> Result<(), BlockError> {
        let core = self.core;
        let mut inner = core.inner.try_borrow_mut().map_err(|_| core.conflict())?;
        if inner.outstanding[self.lane] != 0 {
            return Err(BlockError::Draining);
        }
        inner.live[self.lane] = false;
        if inner.live.iter().any(|live| *live) {
            return Ok(());
        }
        inner.device.quiesce()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::block::memory::MemoryBlockDevice;
    use crate::block::{BlockChannel, CompletionMode, RegionDevice};

    // 4096 disk bytes over 512-byte blocks = an 8-block backend.
    type Device = MemoryBlockDevice<4096, 512>;
    type Shared = SharedBackend<Device, 2, 4>;

    fn shared() -> Shared {
        SharedBackend::new(Device::new(512, 8).unwrap()).unwrap()
    }

    /// The whole point: the two lanes' cookie spaces cannot alias, so a
    /// completion can only ever settle the request that produced it.
    #[test]
    fn lanes_tag_their_cookies_disjointly() {
        let shared = shared();
        let a = shared.claim(0).unwrap();
        let b = shared.claim(1).unwrap();
        assert_eq!(a.tag(1).unwrap(), 1);
        assert_eq!(b.tag(1).unwrap(), 1u64 << LANE_SHIFT | 1);
        assert_ne!(a.tag(1).unwrap(), b.tag(1).unwrap());
        // A payload that would collide with the tag is refused, never
        // truncated into another request's cookie.
        assert_eq!(a.tag(MAX_LANE_COOKIE), Ok(MAX_LANE_COOKIE));
        assert_eq!(b.tag(MAX_LANE_COOKIE + 1), Err(BlockError::Range));
        assert_eq!(a.tag(u64::MAX), Err(BlockError::Range));
    }

    #[test]
    fn a_lane_is_affine_and_bounded() {
        let shared = shared();
        let _a = shared.claim(0).unwrap();
        assert_eq!(shared.claim(0).err(), Some(BlockError::Backpressure));
        assert_eq!(shared.claim(2).err(), Some(BlockError::Range));
        assert!(shared.is_live(0));
        assert!(!shared.is_live(1));
    }

    /// The demux's first duty: a completion drained by the wrong lane still
    /// reaches its owner.  Lane 1 drains both, keeps its own, and lane 0
    /// finds its completion waiting rather than lost.
    #[test]
    fn a_completion_drained_by_a_sibling_is_never_dropped() {
        let shared = shared();
        let mut a = shared.claim(0).unwrap();
        let mut b = shared.claim(1).unwrap();

        a.submit(DeviceCommand::Flush, None, 1).unwrap();
        b.submit(DeviceCommand::Flush, None, 1).unwrap();
        assert_eq!(shared.outstanding(0), Some(1));
        assert_eq!(shared.outstanding(1), Some(1));

        // Lane 1 polls first and drains the queue — including lane 0's.
        let completion = b.poll().unwrap();
        assert_eq!(completion.cookie, 1);
        assert_eq!(shared.outstanding(1), Some(0));
        assert_eq!(shared.stats().stray, 0);

        // Lane 0's completion survived the sibling's drain.
        let completion = a.poll().unwrap();
        assert_eq!(completion.cookie, 1);
        assert_eq!(shared.outstanding(0), Some(0));
        assert_eq!(shared.stats().stray, 0);
        assert_eq!(shared.stats().delivered, 2);
        // Nothing left for either lane.
        assert!(a.poll().is_none());
        assert!(b.poll().is_none());
    }

    /// `wait` must not report loss because a *sibling's* completion arrived
    /// first: it absorbs the sibling's and keeps waiting for its own.
    #[test]
    fn wait_absorbs_a_siblings_completion_and_still_returns_its_own() {
        let shared = shared();
        let mut a = shared.claim(0).unwrap();
        let mut b = shared.claim(1).unwrap();
        b.submit(DeviceCommand::Flush, None, 9).unwrap();
        a.submit(DeviceCommand::Flush, None, 7).unwrap();

        let completion = a.wait().expect("its own completion, not a sibling's");
        assert_eq!(completion.cookie, 7);
        // The sibling's is still owed, not consumed by the waiter.
        assert_eq!(shared.outstanding(1), Some(1));
        assert_eq!(b.poll().unwrap().cookie, 9);
        assert_eq!(shared.stats().stray, 0);
    }

    /// An idle lane never spends a wait budget on a sibling's behalf.
    #[test]
    fn wait_on_an_idle_lane_is_not_loss_hunting() {
        let shared = shared();
        let mut a = shared.claim(0).unwrap();
        let mut b = shared.claim(1).unwrap();
        b.submit(DeviceCommand::Flush, None, 3).unwrap();
        assert!(a.wait().is_none());
        // The sibling's completion is still there for the sibling.
        assert_eq!(b.wait().unwrap().cookie, 3);
    }

    /// The mailbox is sized by the same cap that gates submission, so the
    /// no-overflow argument is enforced where it is made.
    #[test]
    fn a_lane_cannot_outrun_its_mailbox() {
        let shared = shared();
        let mut a = shared.claim(0).unwrap();
        // INFLIGHT = 4: the fifth submission is refused, not queued into a
        // mailbox that has no room for its completion.
        for cookie in 1..=4 {
            a.submit(DeviceCommand::Flush, None, cookie).unwrap();
        }
        assert_eq!(
            a.submit(DeviceCommand::Flush, None, 5),
            Err(BlockError::Backpressure)
        );
        assert_eq!(shared.stats().backpressure, 1);
        // Drain one and the lane admits exactly one more.
        assert_eq!(a.poll().unwrap().cookie, 1);
        a.submit(DeviceCommand::Flush, None, 5).unwrap();
        assert_eq!(shared.outstanding(0), Some(4));
        // Every completion is accounted for; none strayed.
        let mut drained = std::vec::Vec::new();
        while let Some(completion) = a.poll() {
            drained.push(completion.cookie);
        }
        assert_eq!(drained, std::vec![2, 3, 4, 5]);
        assert_eq!(shared.stats().stray, 0);
    }

    #[test]
    fn poll_batch_serves_only_its_own_lane() {
        let shared = shared();
        let mut a = shared.claim(0).unwrap();
        let mut b = shared.claim(1).unwrap();
        for cookie in 1..=3 {
            a.submit(DeviceCommand::Flush, None, cookie).unwrap();
            b.submit(DeviceCommand::Flush, None, cookie + 100).unwrap();
        }
        let mut out = [None; 4];
        assert_eq!(a.poll_batch(&mut out), 3);
        assert_eq!(
            out.iter()
                .flatten()
                .map(|c| c.cookie)
                .collect::<std::vec::Vec<_>>(),
            std::vec![1, 2, 3]
        );
        let mut out = [None; 4];
        assert_eq!(b.poll_batch(&mut out), 3);
        assert_eq!(
            out.iter()
                .flatten()
                .map(|c| c.cookie)
                .collect::<std::vec::Vec<_>>(),
            std::vec![101, 102, 103]
        );
        assert_eq!(shared.stats().stray, 0);
    }

    /// The controller stays up until the last lane retires — quiescing it
    /// under a sibling's mount would strand that sibling's transfers.
    #[test]
    fn the_last_lane_out_quiesces_the_backend() {
        let shared = shared();
        let mut a = shared.claim(0).unwrap();
        let mut b = shared.claim(1).unwrap();
        a.submit(DeviceCommand::Flush, None, 1).unwrap();
        // A lane with work in flight refuses to retire.
        assert_eq!(a.quiesce(), Err(BlockError::Draining));
        assert_eq!(a.poll().unwrap().cookie, 1);
        a.quiesce().unwrap();
        assert!(!shared.is_live(0));
        assert!(shared.is_live(1));
        // A retired lane conveys no submission authority.
        assert_eq!(
            a.submit(DeviceCommand::Flush, None, 2),
            Err(BlockError::Revoked)
        );
        b.quiesce().unwrap();
        assert!(!shared.is_live(1));
    }

    #[test]
    fn control_plane_access_requires_every_lane_to_be_idle() {
        let shared = shared();
        let mut lane = shared.claim(0).unwrap();
        lane.submit(DeviceCommand::Flush, None, 1).unwrap();
        assert_eq!(
            shared.control(|device| device.geometry()).unwrap_err(),
            BlockError::Draining
        );
        assert_eq!(lane.poll().unwrap().cookie, 1);
        assert_eq!(
            shared.control(|device| device.geometry()).unwrap(),
            lane.geometry()
        );
    }

    /// A backend that needs an executor-aware progress step cannot be
    /// shared: its future would hold the shared borrow across an `await`.
    #[test]
    fn a_parking_backend_is_refused_at_construction() {
        struct Parking(Device);
        impl DirectBlockDevice for Parking {
            type Buffer = <Device as DirectBlockDevice>::Buffer;
            fn geometry(&self) -> BlockGeometry {
                self.0.geometry()
            }
            fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError> {
                self.0.allocate_buffer(bytes)
            }
            fn submit(
                &mut self,
                command: DeviceCommand,
                buffer: Option<&mut Self::Buffer>,
                cookie: u64,
            ) -> Result<(), BlockError> {
                self.0.submit(command, buffer, cookie)
            }
            fn poll(&mut self) -> Option<DeviceCompletion> {
                self.0.poll()
            }
            fn quiesce(&mut self) -> Result<(), BlockError> {
                self.0.quiesce()
            }
        }
        let device = Parking(Device::new(512, 8).unwrap());
        assert!(!Parking::INDEPENDENT_PROGRESS);
        assert_eq!(
            SharedBackend::<Parking, 2, 4>::new(device).err(),
            Some(BlockError::Unsupported)
        );
    }

    /// The composition slice 4 actually mounts: two disjoint regions, two
    /// channels, one controller.  Both channels mint cookies from `1`, and
    /// each region's bytes must land only in that region.
    #[test]
    fn two_region_channels_over_one_backend_stay_disjoint() {
        let shared = shared();
        // Blocks [0, 4) and [4, 8) of the same 8-block backend.
        let system = RegionDevice::whole(shared.claim(0).unwrap())
            .unwrap()
            .narrow(0, 4)
            .map_err(|(_, error)| error)
            .unwrap();
        let user = RegionDevice::whole(shared.claim(1).unwrap())
            .unwrap()
            .narrow(4, 4)
            .map_err(|(_, error)| error)
            .unwrap();
        let mut system: BlockChannel<_, 2, 4> =
            BlockChannel::new(system, CompletionMode::Poll).unwrap();
        let mut user: BlockChannel<_, 2, 4> =
            BlockChannel::new(user, CompletionMode::Poll).unwrap();
        assert_eq!(system.geometry().block_count, 4);
        assert_eq!(user.geometry().block_count, 4);

        let system_buffer = system.register_buffer(512).unwrap();
        let user_buffer = user.register_buffer(512).unwrap();
        system.write_buffer(system_buffer, 0, &[0xa5; 512]).unwrap();
        user.write_buffer(user_buffer, 0, &[0x5a; 512]).unwrap();

        // Both channels are at cookie 1 — the collision the demux answers.
        let system_request = system.submit_write(0, 1, system_buffer, 0).unwrap();
        let user_request = user.submit_write(0, 1, user_buffer, 0).unwrap();
        assert_eq!(system_request.0, user_request.0);

        // Each channel settles its own request and nothing else.
        let completion = system.wait().unwrap();
        assert_eq!(completion.request, system_request);
        completion.result.unwrap();
        let completion = user.wait().unwrap();
        assert_eq!(completion.request, user_request);
        completion.result.unwrap();
        assert_eq!(system.late_completions(), 0);
        assert_eq!(user.late_completions(), 0);
        assert_eq!(shared.stats().stray, 0);

        // Region 0's write landed at backend block 0, region 1's at block 4:
        // read each back through its own clamp.
        system.fill_buffer(system_buffer, 0).unwrap();
        let request = system.submit_read(0, 1, system_buffer, 0).unwrap();
        assert_eq!(system.wait().unwrap().request, request);
        let mut back = [0u8; 512];
        system.read_buffer(system_buffer, 0, &mut back).unwrap();
        assert_eq!(back, [0xa5; 512]);

        user.fill_buffer(user_buffer, 0).unwrap();
        let request = user.submit_read(0, 1, user_buffer, 0).unwrap();
        assert_eq!(user.wait().unwrap().request, request);
        let mut back = [0u8; 512];
        user.read_buffer(user_buffer, 0, &mut back).unwrap();
        assert_eq!(back, [0x5a; 512]);

        // Neither channel can name a block in the other's region.
        assert!(system.submit_write(4, 1, system_buffer, 0).is_err());
        assert!(user.submit_write(4, 1, user_buffer, 0).is_err());
    }

    /// A completion that answers no live request is counted, never delivered
    /// to a lane that would mis-settle it.  By the module's invariant a
    /// healthy backend cannot produce one — which is exactly why it has to be
    /// visible when a backend does.
    #[test]
    fn a_completion_answering_no_live_request_is_counted_not_delivered() {
        let shared = shared();
        let mut a = shared.claim(0).unwrap();
        // Straight onto the backend, beneath the tagging seam: a cookie for a
        // lane that exists but has nothing outstanding, and one whose tag
        // names no lane at all.
        {
            let mut inner = shared.inner.borrow_mut();
            inner.device.submit(DeviceCommand::Flush, None, 1).unwrap();
            inner
                .device
                .submit(DeviceCommand::Flush, None, 7u64 << LANE_SHIFT | 1)
                .unwrap();
        }
        assert!(a.poll().is_none());
        let stats = shared.stats();
        assert_eq!(stats.stray, 2);
        assert_eq!(stats.delivered, 0);
        assert_eq!(stats.routed, 0);
        // The lane is still healthy afterwards: its own work still settles.
        a.submit(DeviceCommand::Flush, None, 1).unwrap();
        assert_eq!(a.poll().unwrap().cookie, 1);
        assert_eq!(shared.stats().stray, 2);
    }

    /// With completions held, a sibling's drain cannot manufacture one:
    /// nothing is delivered, nothing strays, and both requests stay owed.
    #[test]
    fn a_held_queue_delivers_to_neither_lane() {
        let shared = shared();
        let mut a = shared.claim(0).unwrap();
        let mut b = shared.claim(1).unwrap();
        a.submit(DeviceCommand::Flush, None, 1).unwrap();
        b.submit(DeviceCommand::Flush, None, 1).unwrap();
        {
            let mut inner = shared.inner.borrow_mut();
            inner.device.hold_completions(true);
        }
        assert!(a.poll().is_none());
        assert!(b.poll().is_none());
        assert_eq!(shared.stats().delivered, 0);
        assert_eq!(shared.stats().stray, 0);
        {
            let mut inner = shared.inner.borrow_mut();
            inner.device.hold_completions(false);
        }
        assert_eq!(a.poll().unwrap().cookie, 1);
        assert_eq!(b.poll().unwrap().cookie, 1);
    }
}
