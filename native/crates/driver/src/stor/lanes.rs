//! **Lanes over the engine** — the transport underneath today's filesystem
//! composition (`res/STORAGE_TRANSPORT.md` §9, slice S1b).
//!
//! [`EngineBackend`] owns one [`StorEngine`] and gives each region a lane:
//! a ring on the engine carrying a whole-namespace grant, driven through an
//! in-process [`StorClient`]. A lane is a [`DirectBlockDevice`], so the
//! ordinary composition — region clamp, block channel, provider — stacks on
//! it unchanged, as it stacked on the shared-backend lanes the engine
//! replaced:
//!
//! ```text
//! EngineBackend<Q>              one engine, one queue set
//!   ├ lane 0 → RegionDevice → BlockChannel → provider   (reads Demand, writes Commit)
//!   ├ lane 1 → RegionDevice → BlockChannel → provider
//!   └ client rings (after the lanes): windows other tasks lent, grants
//!     installed by the filesystem (`attach_client`, `install`)
//! ```
//!
//! What changes underneath: a lane's commands are ring submissions in a
//! class, so the engine keeps clients' bulk reads inside the Stream budget
//! while filesystem reads fill first; a completion names its own ring, so there is
//! no cookie tag and no demux; and a ring's completion channel can never
//! drop a completion. The lane's region clamp above it is unchanged — the
//! whole-namespace grant is only the transport's ceiling.
//!
//! One worker owns the backend and every lane, as it owned the shared
//! backend: interior mutability is `RefCell`, and no borrow is held across
//! a suspension point.

use core::cell::{Cell, RefCell};
use core::ptr::NonNull;

use kore_abi::stor::{Class, GrantId, StorStatus};
use kore_ring::stor::{Op, Request, StorClient, split};

use super::QueueSet;
use crate::seam::Quiet;
use super::engine::{
    ClassPolicy, EngineStats, Expiry, RingId, Settle, StorEngine, StorHost, StorRefusal, TargetAccess,
};
use super::grant::{ClassSet, ExtentRun, Grant, MAX_RUNS, Rights};
use crate::block::{
    BlockBuffer, BlockError, BlockGeometry, BlockHealth, DeviceCommand, DeviceCompletion,
    DirectBlockDevice, Liveness,
};

/// The class each command kind of a lane rides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneClasses {
    /// Reads.
    pub read: Class,
    /// Writes.
    pub write: Class,
    /// Flushes.
    pub flush: Class,
    /// Discards.
    pub discard: Class,
}

impl LaneClasses {
    /// A filesystem region: reads on Demand, the durability path on Commit.
    pub const FILESYSTEM: Self = Self {
        read: Class::Demand,
        write: Class::Commit,
        flush: Class::Commit,
        discard: Class::Commit,
    };
    /// A bulk streaming region: everything on Stream.
    pub const STREAM: Self = Self {
        read: Class::Stream,
        write: Class::Stream,
        flush: Class::Stream,
        discard: Class::Stream,
    };
}

/// A lane's registered buffer: an engine target the lane adopted, with the
/// CPU view its owner reads and writes between commands.
pub struct LaneBuffer {
    slot: u16,
    ptr: NonNull<u8>,
    len: usize,
}

impl LaneBuffer {
    /// The target slot on the lane's ring.
    pub const fn slot(&self) -> u16 {
        self.slot
    }
}

impl BlockBuffer for LaneBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view was taken from the adopted target, which the
        // engine keeps at a fixed address for as long as it is adopted, and
        // a block channel reaches the bytes only with no command in flight
        // against the buffer.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `bytes`.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

/// The engine's side effects in-process: nothing to kick (the lanes poll),
/// no authority to tell (the lanes never revoke); expiries are counted.
#[derive(Default)]
struct LaneHost {
    expired: u64,
}

impl StorHost for LaneHost {
    fn kick(&mut self, _ring: RingId) {}
    fn drained(&mut self, _ring: RingId, _grant: GrantId) {}
    fn departed(&mut self, _ring: RingId) {}
    fn expired(&mut self, _expiry: Expiry) {
        self.expired += 1;
    }
}

struct LaneState<'w> {
    client: RefCell<StorClient<'w>>,
    ring: RingId,
    grant: GrantId,
    classes: LaneClasses,
    claimed: Cell<bool>,
    retired: Cell<bool>,
    /// Submissions accepted and not yet reaped.
    open: Cell<u32>,
}

/// The whole-namespace grant a lane carries: every block, in as many runs
/// as the run length's width needs.
fn namespace_grant(geometry: BlockGeometry) -> Result<Grant, BlockError> {
    let mut runs = [ExtentRun::default(); MAX_RUNS];
    let mut count = 0;
    let mut lba = 0u64;
    while lba < geometry.block_count {
        if count == MAX_RUNS {
            return Err(BlockError::Geometry);
        }
        let blocks = (geometry.block_count - lba).min(u64::from(u32::MAX)) as u32;
        runs[count] = ExtentRun { lba, blocks };
        count += 1;
        lba += u64::from(blocks);
    }
    Grant::new(
        &runs[..count],
        Rights::READ | Rights::WRITE | Rights::FLUSH | Rights::DISCARD,
        ClassSet::ALL,
    )
    .map_err(|_| BlockError::Geometry)
}

const fn block_error(status: StorStatus) -> BlockError {
    match status {
        StorStatus::Range => BlockError::Range,
        StorStatus::Alignment => BlockError::Alignment,
        StorStatus::Rights => BlockError::ReadOnly,
        StorStatus::Revoked => BlockError::Revoked,
        StorStatus::Stale => BlockError::UnknownBuffer,
        StorStatus::Deadline | StorStatus::Lost => BlockError::Lost,
        StorStatus::Cancelled => BlockError::Cancelled,
        StorStatus::Media => BlockError::Media,
        StorStatus::Busy => BlockError::Backpressure,
        StorStatus::Unsupported | StorStatus::Class | StorStatus::Invalid => {
            BlockError::Unsupported
        }
    }
}

/// What one [`EngineBackend::pump`] left behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pump {
    /// Device commands are in flight: their completion wakes the host, and
    /// that wake pumps again, so a record published meanwhile is seen then.
    Busy,
    /// Nothing is in flight and every ring is parked: the host may sleep. A
    /// client that publishes now sees the park and owes a notice.
    Parked,
    /// Work stayed pending through every pass (or the engine was borrowed):
    /// pump again before sleeping.
    Again,
}

/// Passes one pump makes before giving up on reaching `Busy` or `Parked`.
const PUMP_PASSES: usize = 4;

const fn refusal_error(refusal: StorRefusal) -> BlockError {
    match refusal {
        StorRefusal::Stale => BlockError::UnknownBuffer,
        StorRefusal::Full => BlockError::Backpressure,
        StorRefusal::Busy => BlockError::BufferBusy,
        StorRefusal::Departed => BlockError::Revoked,
        StorRefusal::Invalid => BlockError::Unsupported,
    }
}

/// One engine shared by `LANES` lanes and `RINGS - LANES` client rings.
pub struct EngineBackend<
    'w,
    Q: QueueSet,
    P: ClassPolicy,
    const LANES: usize,
    const RINGS: usize,
    const GRANTS: usize,
    const TARGETS: usize,
    const OPS: usize,
    const COMMANDS: usize,
> {
    /// Placed by the caller ([`StorEngine::init_in`]), so the tables never
    /// pass through a stack frame on their way to the heap.
    engine: RefCell<&'w mut StorEngine<'w, Q, P, RINGS, GRANTS, TARGETS, OPS, COMMANDS>>,
    host: RefCell<LaneHost>,
    lanes: [LaneState<'w>; LANES],
    geometry: BlockGeometry,
    max_buffer_bytes: usize,
    conflicts: Cell<u64>,
}

impl<
    'w,
    Q: QueueSet,
    P: ClassPolicy,
    const LANES: usize,
    const RINGS: usize,
    const GRANTS: usize,
    const TARGETS: usize,
    const OPS: usize,
    const COMMANDS: usize,
> EngineBackend<'w, Q, P, LANES, RINGS, GRANTS, TARGETS, OPS, COMMANDS>
where
    Q::Target: BlockBuffer,
{
    /// Lanes over a freshly placed `engine` (no ring attached yet), one lane
    /// per window (each exactly [`kore_ring::stor::WINDOW_BYTES`],
    /// page-aligned, owned for `'w`), lane `i` riding `classes[i]` and
    /// taking ring slot `i`.
    pub fn new(
        engine: &'w mut StorEngine<'w, Q, P, RINGS, GRANTS, TARGETS, OPS, COMMANDS>,
        windows: [&'w mut [u8]; LANES],
        classes: [LaneClasses; LANES],
    ) -> Result<Self, BlockError> {
        // Lanes take the first ring slots (attached here, before any
        // client) and the first grant slot of their ring.
        if RINGS < LANES || GRANTS == 0 {
            return Err(BlockError::Geometry);
        }
        let geometry = engine.queues().geometry().validate()?;
        let max_buffer_bytes = engine.queues().max_target_bytes();
        let grant = namespace_grant(geometry)?;
        let mut built: [Option<LaneState<'w>>; LANES] = core::array::from_fn(|_| None);
        for (index, window) in windows.into_iter().enumerate() {
            let (client, service) = split(window).map_err(|_| BlockError::Geometry)?;
            let ring = engine
                .attach(service, ClassSet::ALL)
                .map_err(|_| BlockError::Backpressure)?;
            if usize::from(ring.slot()) != index {
                // The engine was not fresh: a lane would share the client
                // slot range, and the client calls could reach it.
                return Err(BlockError::Geometry);
            }
            let grant = engine.install(ring, grant).map_err(|_| BlockError::Backpressure)?;
            built[index] = Some(LaneState {
                client: RefCell::new(StorClient::new(client)),
                ring,
                grant,
                classes: classes[index],
                claimed: Cell::new(false),
                retired: Cell::new(false),
                open: Cell::new(0),
            });
        }
        Ok(Self {
            engine: RefCell::new(engine),
            host: RefCell::new(LaneHost::default()),
            lanes: built.map(|lane| lane.expect("every window built a lane")),
            geometry,
            max_buffer_bytes,
            conflicts: Cell::new(0),
        })
    }

    /// Take the engine back: every lane's ring detached and retired, its
    /// targets dropped. For a backend whose device outlives it — a test's
    /// reboot of one disk, an orderly teardown. Refused, the backend handed
    /// back, while any lane has a submission open. A client ring still
    /// attached stays attached; `StorEngine::into_queues` then refuses.
    pub fn into_engine(
        self,
    ) -> Result<&'w mut StorEngine<'w, Q, P, RINGS, GRANTS, TARGETS, OPS, COMMANDS>, Self> {
        if self.lanes.iter().any(|lane| lane.open.get() != 0) {
            return Err(self);
        }
        let engine = self.engine.into_inner();
        for lane in &self.lanes {
            let _ = engine.detach(lane.ring);
            for slot in 0..TARGETS {
                drop(engine.release_target(lane.ring, slot as u16));
            }
            let _ = engine.retire(lane.ring);
        }
        Ok(engine)
    }

    /// The namespace's geometry.
    pub fn geometry(&self) -> BlockGeometry {
        self.geometry
    }

    /// Claim lane `lane` — once.
    pub fn claim(&self, lane: usize) -> Result<EngineLane<'_, 'w, Q, P, LANES, RINGS, GRANTS, TARGETS, OPS, COMMANDS>, BlockError> {
        let state = self.lanes.get(lane).ok_or(BlockError::Range)?;
        if state.claimed.replace(true) {
            return Err(BlockError::Backpressure);
        }
        Ok(EngineLane { core: self, lane })
    }

    /// Submissions lane `lane` has open.
    pub fn outstanding(&self, lane: usize) -> Option<usize> {
        self.lanes.get(lane).map(|state| state.open.get() as usize)
    }

    /// Whether lane `lane` is claimed and not retired.
    pub fn is_live(&self, lane: usize) -> bool {
        self.lanes
            .get(lane)
            .is_some_and(|state| state.claimed.get() && !state.retired.get())
    }

    /// Engine counters, with the expiries the host counted.
    pub fn stats(&self) -> Option<(EngineStats, u64)> {
        let engine = self.engine.try_borrow().ok()?;
        let host = self.host.try_borrow().ok()?;
        Some((*engine.stats(), host.expired))
    }

    /// Device commands in flight, every ring together.
    pub fn commands_in_flight(&self) -> Option<usize> {
        Some(self.engine.try_borrow().ok()?.commands_in_flight())
    }

    /// Borrow conflicts refused — a re-entrancy bug surfaced as a count.
    pub fn conflicts(&self) -> u64 {
        self.conflicts.get()
    }

    fn engine(&self) -> Result<core::cell::RefMut<'_, &'w mut StorEngine<'w, Q, P, RINGS, GRANTS, TARGETS, OPS, COMMANDS>>, BlockError> {
        self.engine.try_borrow_mut().map_err(|_| {
            self.conflicts.set(self.conflicts.get() + 1);
            BlockError::Transport
        })
    }

    /// A control step against the queue set, with commands in flight or not.
    pub fn control<T>(&self, f: impl FnOnce(&mut Q) -> T) -> Result<T, BlockError> {
        Ok(f(self.engine()?.queues_mut()))
    }

    /// A control step that needs the device idle — an IOMMU retirement,
    /// teardown. Refused with commands in flight.
    pub fn control_idle<T>(&self, f: impl FnOnce(&mut Q) -> T) -> Result<T, BlockError> {
        let mut engine = self.engine()?;
        if engine.commands_in_flight() != 0 {
            return Err(BlockError::Draining);
        }
        Ok(f(engine.queues_mut()))
    }

    /// Read the queue set without driving it — testimony only.
    pub fn inspect<T>(&self, f: impl FnOnce(&Q) -> T) -> Option<T> {
        self.engine.try_borrow().ok().map(|engine| f(engine.queues()))
    }

    /// Adopt caller memory into `lane` as a streaming target.
    pub fn adopt(
        &self,
        lane: usize,
        target: Q::Target,
        access: TargetAccess,
    ) -> Result<LaneBuffer, (Q::Target, BlockError)> {
        let Some(state) = self.lanes.get(lane) else {
            return Err((target, BlockError::Range));
        };
        let mut engine = match self.engine() {
            Ok(engine) => engine,
            Err(error) => return Err((target, error)),
        };
        let slot = engine
            .adopt_target(state.ring, target, access)
            .map_err(|(target, _)| (target, BlockError::Backpressure))?;
        let view = engine
            .target_mut(state.ring, slot)
            .expect("a just-adopted target is idle");
        let len = view.len();
        let ptr = NonNull::new(view.bytes_mut().as_mut_ptr()).unwrap_or(NonNull::dangling());
        Ok(LaneBuffer { slot, ptr, len })
    }

    /// Hand `buffer` back from `lane`, once no command uses it.
    pub fn release(&self, lane: usize, buffer: LaneBuffer) -> Result<Quiet<Q::Target>, (LaneBuffer, BlockError)> {
        let Some(state) = self.lanes.get(lane) else {
            return Err((buffer, BlockError::Range));
        };
        let mut engine = match self.engine() {
            Ok(engine) => engine,
            Err(error) => return Err((buffer, error)),
        };
        engine
            .release_target(state.ring, buffer.slot)
            .map_err(|_| (buffer, BlockError::BufferBusy))
    }

    // ── client rings ────────────────────────────────────────────────────
    //
    // A client ring is a window another task funded and lent: its channels,
    // indices and records are shared with a party this worker does not
    // trust. The engine touches it only through the channel protocols, which
    // mask every index and decode any bytes, so a hostile client can wreck
    // its own ring and nothing else. Client rings take the slots after the
    // lanes, and a lane's ring is never reachable through these calls.

    fn client_ring(ring: RingId) -> Result<RingId, BlockError> {
        if usize::from(ring.slot()) < LANES {
            return Err(BlockError::Range);
        }
        Ok(ring)
    }

    /// Attach a client's formatted window as a ring admitting `classes`.
    pub fn attach_client(&self, window: &'w mut [u8], classes: ClassSet) -> Result<RingId, BlockError> {
        let service = kore_ring::stor::service(window).map_err(|_| BlockError::Geometry)?;
        self.engine()?
            .attach(service, classes)
            .map_err(|(_, _)| BlockError::Backpressure)
    }

    /// Install `grant` into client `ring`.
    pub fn install(&self, ring: RingId, grant: Grant) -> Result<GrantId, BlockError> {
        let ring = Self::client_ring(ring)?;
        self.engine()?
            .install(ring, grant)
            .map_err(|_| BlockError::Backpressure)
    }

    /// Revoke `grant` from client `ring`; [`Settle::Later`] until nothing
    /// in flight names it.
    pub fn revoke(&self, ring: RingId, grant: GrantId) -> Result<Settle, BlockError> {
        let ring = Self::client_ring(ring)?;
        self.engine()?
            .revoke(ring, grant)
            .map_err(|_| BlockError::UnknownBuffer)
    }

    /// Adopt caller memory into client `ring` as target slot `n`. A refused
    /// target comes back [`Quiet`]: the engine never took it, so no command
    /// can name it.
    pub fn adopt_client(
        &self,
        ring: RingId,
        target: Q::Target,
        access: TargetAccess,
    ) -> Result<u16, (Quiet<Q::Target>, BlockError)> {
        if let Err(error) = Self::client_ring(ring) {
            return Err((Quiet::witness(target), error));
        }
        let mut engine = match self.engine() {
            Ok(engine) => engine,
            Err(error) => return Err((Quiet::witness(target), error)),
        };
        engine
            .adopt_target(ring, target, access)
            .map_err(|(target, refusal)| (Quiet::witness(target), refusal_error(refusal)))
    }

    /// Hand target `slot` of client `ring` back, once no command uses it.
    pub fn release_client(&self, ring: RingId, slot: u16) -> Result<Quiet<Q::Target>, BlockError> {
        let ring = Self::client_ring(ring)?;
        self.engine()?
            .release_target(ring, slot)
            .map_err(refusal_error)
    }

    /// Detach client `ring`: nothing more is popped; what the device holds
    /// finishes. [`Settle::Now`] when nothing was open.
    pub fn detach_client(&self, ring: RingId) -> Result<Settle, BlockError> {
        let ring = Self::client_ring(ring)?;
        self.engine()?.detach(ring).map_err(refusal_error)
    }

    /// Submissions client `ring` has popped and not completed.
    pub fn client_open(&self, ring: RingId) -> Option<u32> {
        let ring = Self::client_ring(ring).ok()?;
        self.engine.try_borrow().ok()?.open(ring)
    }

    /// Retire a detached, drained client ring whose targets are all
    /// released. Its window is no longer referenced once this returns
    /// `Ok`, and not before: only then may the lent mapping go.
    pub fn retire_client(&self, ring: RingId) -> Result<(), BlockError> {
        let ring = Self::client_ring(ring)?;
        self.engine()?.retire(ring).map(drop).map_err(refusal_error)
    }

    /// One pass on the host's wake: advance, and when the engine is idle
    /// commit every ring to parking. See [`Pump`].
    pub fn pump(&self) -> Pump {
        let Ok(mut engine) = self.engine() else {
            return Pump::Again;
        };
        for _ in 0..PUMP_PASSES {
            self.advance(&mut engine);
            if engine.commands_in_flight() != 0 {
                return Pump::Busy;
            }
            if engine.park_idle() {
                return Pump::Parked;
            }
        }
        Pump::Again
    }

    fn advance(&self, engine: &mut StorEngine<'w, Q, P, RINGS, GRANTS, TARGETS, OPS, COMMANDS>) {
        if let Ok(mut host) = self.host.try_borrow_mut() {
            engine.advance(&mut *host);
        }
    }

    fn stage(&self, lane: usize, command: DeviceCommand, buffer: Option<&LaneBuffer>, cookie: u64) -> Result<(), BlockError> {
        let state = &self.lanes[lane];
        if state.retired.get() {
            return Err(BlockError::Revoked);
        }
        let block = u64::from(self.geometry.block_size);
        let request = |op, class_lba: Option<(u64, u32)>, target: Option<(u16, usize)>| {
            let (offset, len) = class_lba.map_or((0, 0), |(lba, blocks)| {
                (lba * block, (u64::from(blocks) * block) as u32)
            });
            let (target, target_offset) = target.map_or((0, 0), |(slot, at)| (slot, at as u64));
            Request {
                op,
                grant: state.grant,
                offset,
                len,
                target,
                target_offset,
                user_data: cookie,
                timeline: 0,
                deadline_ns: 0,
            }
        };
        let (class, request) = match command {
            DeviceCommand::Read {
                lba,
                blocks,
                buffer_offset,
            } => {
                let buffer = buffer.ok_or(BlockError::UnknownBuffer)?;
                (
                    state.classes.read,
                    request(Op::Read, Some((lba, blocks)), Some((buffer.slot, buffer_offset))),
                )
            }
            DeviceCommand::Write {
                lba,
                blocks,
                buffer_offset,
            } => {
                let buffer = buffer.ok_or(BlockError::UnknownBuffer)?;
                (
                    state.classes.write,
                    request(Op::Write, Some((lba, blocks)), Some((buffer.slot, buffer_offset))),
                )
            }
            DeviceCommand::Flush => (state.classes.flush, request(Op::Flush, None, None)),
            DeviceCommand::Discard { lba, blocks } => (
                state.classes.discard,
                request(Op::Discard, Some((lba, blocks)), None),
            ),
        };
        if u64::from(request.len) != request.len as u64 {
            return Err(BlockError::TooLarge);
        }
        state
            .client
            .try_borrow_mut()
            .map_err(|_| BlockError::Transport)?
            .submit(class, &request)
            .map_err(|_| BlockError::Backpressure)?;
        state.open.set(state.open.get() + 1);
        Ok(())
    }

    fn publish(&self, lane: usize) {
        if let Ok(mut client) = self.lanes[lane].client.try_borrow_mut() {
            for class in Class::ALL {
                let _ = client.publish(class);
            }
        }
        if let Ok(mut engine) = self.engine() {
            self.advance(&mut engine);
        }
    }

    fn poll(&self, lane: usize) -> Option<DeviceCompletion> {
        let state = &self.lanes[lane];
        if let Ok(mut engine) = self.engine() {
            self.advance(&mut engine);
        }
        let completion = state.client.try_borrow_mut().ok()?.reap()?;
        state.open.set(state.open.get().saturating_sub(1));
        Some(DeviceCompletion {
            cookie: completion.user_data,
            result: completion.result.map(|_| ()).map_err(block_error),
        })
    }

    fn wait(&self, lane: usize) -> Option<DeviceCompletion> {
        // Bounded: each empty park is the queue set's own wait budget, and
        // two in a row with nothing moving is the caller's expiry.
        let mut empty = 0;
        loop {
            if let Some(completion) = self.poll(lane) {
                return Some(completion);
            }
            if self.lanes[lane].open.get() == 0 {
                return None;
            }
            let parked = self.engine().map(|mut engine| engine.queues_mut().park()).unwrap_or(false);
            if parked {
                empty = 0;
            } else {
                empty += 1;
                if empty >= 2 {
                    return self.poll(lane);
                }
            }
        }
    }

    fn quiesce(&self, lane: usize) -> Result<(), BlockError> {
        let state = &self.lanes[lane];
        if state.open.get() != 0 {
            return Err(BlockError::Draining);
        }
        state.retired.set(true);
        if self.lanes.iter().all(|lane| lane.retired.get() || !lane.claimed.get()) {
            self.engine()?.queues_mut().quiesce()
        } else {
            Ok(())
        }
    }
}

/// One lane: a [`DirectBlockDevice`] whose commands ride its own ring.
pub struct EngineLane<
    'a,
    'w,
    Q: QueueSet,
    P: ClassPolicy,
    const LANES: usize,
    const RINGS: usize,
    const GRANTS: usize,
    const TARGETS: usize,
    const OPS: usize,
    const COMMANDS: usize,
> {
    core: &'a EngineBackend<'w, Q, P, LANES, RINGS, GRANTS, TARGETS, OPS, COMMANDS>,
    lane: usize,
}

impl<
    'a,
    'w,
    Q: QueueSet,
    P: ClassPolicy,
    const LANES: usize,
    const RINGS: usize,
    const GRANTS: usize,
    const TARGETS: usize,
    const OPS: usize,
    const COMMANDS: usize,
> EngineLane<'a, 'w, Q, P, LANES, RINGS, GRANTS, TARGETS, OPS, COMMANDS>
where
    Q::Target: BlockBuffer,
{
    /// This lane's index.
    pub fn index(&self) -> usize {
        self.lane
    }

    /// The backend it rides.
    pub fn backend(&self) -> &'a EngineBackend<'w, Q, P, LANES, RINGS, GRANTS, TARGETS, OPS, COMMANDS> {
        self.core
    }
}

impl<
    'a,
    'w,
    Q: QueueSet,
    P: ClassPolicy,
    const LANES: usize,
    const RINGS: usize,
    const GRANTS: usize,
    const TARGETS: usize,
    const OPS: usize,
    const COMMANDS: usize,
> DirectBlockDevice for EngineLane<'a, 'w, Q, P, LANES, RINGS, GRANTS, TARGETS, OPS, COMMANDS>
where
    Q::Target: BlockBuffer,
{
    type Buffer = LaneBuffer;

    /// The engine advances inside every poll; nothing awaits a reply.
    const INDEPENDENT_PROGRESS: bool = true;

    fn geometry(&self) -> BlockGeometry {
        self.core.geometry
    }

    fn max_buffer_bytes(&self) -> usize {
        self.core.max_buffer_bytes
    }

    fn liveness(&mut self) -> Liveness {
        self.core
            .engine()
            .map(|mut engine| engine.liveness())
            .unwrap_or(Liveness::Unknown)
    }

    fn clock_ns(&self) -> Option<u64> {
        self.core.inspect(|queues| queues.clock_ns()).flatten()
    }

    fn health(&mut self) -> Result<BlockHealth, BlockError> {
        self.core.control(|queues| queues.health())?
    }

    fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError> {
        let target = self.core.control(|queues| queues.allocate(bytes))??;
        self.core
            .adopt(self.lane, target, TargetAccess::Both)
            .map_err(|(_, error)| error)
    }

    fn submit(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        self.core.stage(self.lane, command, buffer.as_deref(), cookie)?;
        self.core.publish(self.lane);
        Ok(())
    }

    fn stage(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        self.core.stage(self.lane, command, buffer.as_deref(), cookie)
    }

    fn publish(&mut self) {
        self.core.publish(self.lane);
    }

    fn poll(&mut self) -> Option<DeviceCompletion> {
        self.core.poll(self.lane)
    }

    fn quiesce(&mut self) -> Result<(), BlockError> {
        self.core.quiesce(self.lane)
    }

    fn wait(&mut self) -> Option<DeviceCompletion> {
        self.core.wait(self.lane)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::boxed::Box;

    use super::*;
    use crate::block::memory::MemoryBlockDevice;
    use crate::block::{BlockChannel, CompletionMode, RegionDevice};
    use crate::stor::Single;
    use crate::stor::engine::OneQueue;
    use kore_ring::stor::WINDOW_BYTES;

    // Small inline fixtures: the memory double carries its disk and every
    // adopted target inline, and a debug build copies the backend on the
    // test thread's stack while it is built.
    type Dev = MemoryBlockDevice<{ 32 * 1024 }, 4096>;
    type Backend = EngineBackend<'static, Single<Dev, 32>, OneQueue, 2, 3, 2, 4, 32, 32>;

    fn window() -> &'static mut [u8] {
        let layout = std::alloc::Layout::from_size_align(WINDOW_BYTES, 4096).unwrap();
        // SAFETY: non-zero size; leaked for a 'static window.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        // SAFETY: freshly allocated, exclusively ours.
        unsafe { core::slice::from_raw_parts_mut(ptr, WINDOW_BYTES) }
    }

    type Engine = StorEngine<'static, Single<Dev, 32>, OneQueue, 3, 2, 4, 32, 32>;

    fn backend() -> &'static Backend {
        let dev = Dev::new(512, 8).unwrap();
        let place = Box::leak(Box::new(core::mem::MaybeUninit::<Engine>::uninit()));
        let engine = Engine::init_in(place, Single::new(dev)).ok().expect("policy fits");
        Box::leak(Box::new(
            Backend::new(engine, [window(), window()], [LaneClasses::FILESYSTEM, LaneClasses::STREAM])
                .unwrap(),
        ))
    }

    /// The ordinary composition runs unchanged on an engine lane: a block
    /// channel over a region clamp writes and reads back, and the two lanes'
    /// regions never see each other's bytes.
    #[test]
    fn a_block_channel_round_trips_over_an_engine_lane() {
        let backend = backend();
        let lane0 = backend.claim(0).unwrap();
        let lane1 = backend.claim(1).unwrap();
        assert_eq!(backend.claim(0).err(), Some(BlockError::Backpressure));
        let region0 = RegionDevice::whole(lane0)
            .unwrap()
            .narrow(0, 32)
            .ok()
            .expect("region 0");
        let region1 = RegionDevice::whole(lane1)
            .unwrap()
            .narrow(32, 32)
            .ok()
            .expect("region 1");
        let mut a: BlockChannel<_, 2, 4> = BlockChannel::new(region0, CompletionMode::Poll).unwrap();
        let mut b: BlockChannel<_, 2, 4> = BlockChannel::new(region1, CompletionMode::Poll).unwrap();

        let buf_a = a.register_buffer(4096).unwrap();
        let buf_b = b.register_buffer(4096).unwrap();
        a.fill_buffer(buf_a, 0xaa).unwrap();
        b.fill_buffer(buf_b, 0xbb).unwrap();
        let wa = a.submit_write(0, 8, buf_a, 0).unwrap();
        let wb = b.submit_write(0, 8, buf_b, 0).unwrap();
        assert_eq!(a.wait().unwrap().request, wa);
        assert_eq!(b.wait().unwrap().request, wb);

        a.fill_buffer(buf_a, 0).unwrap();
        let ra = a.submit_read(0, 8, buf_a, 0).unwrap();
        let done = a.wait().unwrap();
        assert_eq!((done.request, done.result), (ra, Ok(())));
        let mut back = [0u8; 4096];
        a.read_buffer(buf_a, 0, &mut back).unwrap();
        assert!(back.iter().all(|&byte| byte == 0xaa), "lane 0 reads its own bytes");
        // Region 1 (LBA 32..) holds B's bytes, not A's.
        b.fill_buffer(buf_b, 0).unwrap();
        let rb = b.submit_read(0, 8, buf_b, 0).unwrap();
        assert_eq!(b.wait().unwrap().request, rb);
        b.read_buffer(buf_b, 0, &mut back).unwrap();
        assert!(back.iter().all(|&byte| byte == 0xbb));
        assert_eq!(backend.outstanding(0), Some(0));
        let (stats, expired) = backend.stats().unwrap();
        assert_eq!(stats.stale_completions, 0);
        assert_eq!(expired, 0);
    }

    /// Quiesce is refused while a lane has anything open and stops the
    /// device only when the last claimed lane retires.
    #[test]
    fn the_last_lane_to_quiesce_stops_the_device() {
        let backend = backend();
        let mut lane0 = backend.claim(0).unwrap();
        let mut lane1 = backend.claim(1).unwrap();
        backend.control(|q| q.device_mut().hold_completions(true)).unwrap();
        lane0.submit(DeviceCommand::Flush, None, 1).unwrap();
        assert_eq!(lane0.quiesce(), Err(BlockError::Draining));
        backend.control(|q| q.device_mut().hold_completions(false)).unwrap();
        assert_eq!(lane0.wait().map(|c| c.cookie), Some(1));
        assert_eq!(lane0.quiesce(), Ok(()));
        assert!(!backend.is_live(0));
        assert!(backend.is_live(1));
        assert_eq!(lane1.quiesce(), Ok(()));
        assert_eq!(lane0.submit(DeviceCommand::Flush, None, 2), Err(BlockError::Revoked));
    }

    /// An adopted caller target serves reads on the stream lane, and comes
    /// back only once idle.
    #[test]
    fn an_adopted_target_serves_the_stream_lane_and_releases_when_idle() {
        let backend = backend();
        let mut lane1 = backend.claim(1).unwrap();
        let target = backend.control(|q| q.allocate(4096)).unwrap().unwrap();
        let mut buffer = backend.adopt(1, target, TargetAccess::ReadInto).ok().unwrap();
        backend.control(|q| q.device_mut().hold_completions(true)).unwrap();
        lane1
            .submit(
                DeviceCommand::Read {
                    lba: 0,
                    blocks: 8,
                    buffer_offset: 0,
                },
                Some(&mut buffer),
                5,
            )
            .unwrap();
        let (buffer, error) = backend.release(1, buffer).err().unwrap();
        assert_eq!(error, BlockError::BufferBusy);
        backend.control(|q| q.device_mut().hold_completions(false)).unwrap();
        assert_eq!(lane1.wait().map(|c| (c.cookie, c.result)), Some((5, Ok(()))));
        assert!(backend.release(1, buffer).is_ok());
    }

    /// A client's own window rides the engine beside the lanes: a read
    /// grant over part of the disk streams into a client target, an idle
    /// pump parks the ring so the next publish owes a notice, the grant
    /// refuses what it does not cover, and the ring retires only once
    /// detached and its target released. A lane's ring is never reachable
    /// through the client calls.
    #[test]
    fn a_client_ring_streams_through_its_grant_and_parks_when_idle() {
        use kore_ring::stor::{Op, Request, StorClient, client, format};

        let backend = backend();
        let mut lane0 = backend.claim(0).unwrap();
        // Lay a pattern at LBA 8.. through the filesystem lane.
        let mut staging = lane0.allocate_buffer(4096).unwrap();
        for (i, byte) in staging.bytes_mut().iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        lane0
            .submit(DeviceCommand::Write { lba: 8, blocks: 8, buffer_offset: 0 }, Some(&mut staging), 1)
            .unwrap();
        assert_eq!(lane0.wait().map(|c| c.result), Some(Ok(())));

        let shared = window();
        format(shared).unwrap();
        // SAFETY: the test plays both parties over one leaked window; each
        // side touches it only through the channel protocols.
        let alias = unsafe { core::slice::from_raw_parts_mut(shared.as_mut_ptr(), shared.len()) };
        let mut ring_client = StorClient::new(client(alias).unwrap());
        let ring = backend.attach_client(shared, ClassSet::DATA).unwrap();
        assert_eq!(backend.detach_client(backend.lanes[0].ring), Err(BlockError::Range));

        let grant = backend
            .install(
                ring,
                Grant::new(&[ExtentRun { lba: 8, blocks: 16 }], Rights::READ, ClassSet::DATA).unwrap(),
            )
            .unwrap();
        let target = backend.control(|q| q.allocate(4096)).unwrap().unwrap();
        let slot = backend.adopt_client(ring, target, TargetAccess::ReadInto).ok().unwrap();
        let read = |op, offset, user_data| Request {
            op,
            grant,
            offset,
            len: 4096,
            target: slot,
            target_offset: 0,
            user_data,
            timeline: 1,
            deadline_ns: 0,
        };
        let ticket = ring_client.submit(Class::Demand, &read(Op::Read, 0, 10)).unwrap().unwrap();
        let _ = ring_client.publish(Class::Demand);
        assert_eq!(backend.pump(), Pump::Parked);
        assert_eq!(ring_client.poll_ticket(ticket), Ok(true));
        assert_eq!(ring_client.reap().map(|c| c.result), Some(Ok(4096)));

        // Past the grant, and a write it carries no right for.
        ring_client.submit(Class::Demand, &read(Op::Read, 16 * 512, 11)).unwrap();
        ring_client.submit(Class::Demand, &read(Op::Write, 0, 12)).unwrap();
        assert!(ring_client.publish(Class::Demand), "the idle pump parked the ring");
        assert_eq!(backend.pump(), Pump::Parked);
        assert_eq!(ring_client.reap().map(|c| c.result), Some(Err(StorStatus::Range)));
        assert_eq!(ring_client.reap().map(|c| c.result), Some(Err(StorStatus::Rights)));

        assert_eq!(backend.retire_client(ring), Err(BlockError::BufferBusy));
        assert_eq!(backend.detach_client(ring), Ok(Settle::Now));
        assert_eq!(backend.retire_client(ring), Err(BlockError::BufferBusy), "the target is still adopted");
        let back = backend.release_client(ring, slot).unwrap().into_inner();
        assert!(
            back.bytes().iter().enumerate().all(|(i, &byte)| byte == (i % 251) as u8),
            "the client's target holds the pattern the lane wrote"
        );
        assert_eq!(backend.retire_client(ring), Ok(()));
        assert_eq!(backend.client_open(ring), None, "the retired id is stale");
    }
}
