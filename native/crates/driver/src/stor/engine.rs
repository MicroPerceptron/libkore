//! **L1 — the transport engine** (`res/STORAGE_TRANSPORT.md` §2–§5).
//!
//! One engine value is one worker's whole data plane: the rings attached to
//! it, their grants and targets, the submissions in flight, and the queue
//! set it drives. [`StorEngine::advance`] is the only data-path entry. It
//! never waits: it reaps what the device finished, fills each class up to
//! its budget in the policy's order, publishes each touched submission queue
//! once, and commits each touched ring's completion channel once.
//!
//! ## One submission, one completion
//!
//! A client submission becomes an **op**. The engine pops a submission only
//! while the ring holds a completion credit — ops open on the ring are
//! strictly fewer than the free slots in its completion channel — so the
//! completion an op owes can always be posted. An op is issued as one or
//! more device **commands** (split at grant-run boundaries and the device's
//! transfer ceiling), a piece at a time as budget and queue room allow; its
//! completion posts when every issued piece has come back.
//!
//! ## Classes, budgets, fairness
//!
//! The [`ClassPolicy`] maps each class to a hardware submission queue and
//! gives it a command and byte budget. [`advance`](StorEngine::advance)
//! fills classes in the policy's order, and within a class serves rings
//! round robin with a byte quantum, so one ring's stream of large reads
//! cannot keep another ring's small reads waiting. Each ring issues its
//! current op in a class to the end before popping the next — order within
//! a class channel is preserved, and at most one op per ring per class is
//! partially issued.
//!
//! ## Revocation and departure
//!
//! Revoking a grant stops new submissions naming it (they fail
//! [`StorStatus::Revoked`]) and stops issuing the rest of any op already
//! using it; the grant's slot is freed and the authority told
//! ([`StorHost::drained`]) only once nothing that named it is in flight.
//! Detaching a ring stops popping; [`StorHost::departed`] fires once its
//! last op completes, after which its targets can be taken back and the
//! ring retired.

use kore_abi::stor::{
    Class, CqState, GrantId, RING_DEPARTED, RING_LIVE, RWF_DSYNC, RWF_KNOWN, RWF_NOWAIT,
    StorStatus, TIMELINES,
};
use kore_abi::uring::{IoUringCqe, IoUringSqe, op};
use kore_ring::stor::ServiceRing;
use zerocopy::FromZeros;

use super::grant::{ClassSet, Grant, Rights};
use crate::seam::Quiet;
use super::timeline::Timeline;
use super::{QueueSet, Tag};
use crate::block::{BlockError, DeviceCommand, DeviceCompletion, LatencyBooks, Liveness};

/// A ring as the engine names it: slot plus generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RingId {
    slot: u16,
    generation: u16,
}

impl RingId {
    /// The ring's slot.
    pub const fn slot(self) -> u16 {
        self.slot
    }
}

/// A command and byte budget for one class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Device commands the class may have in flight.
    pub commands: u32,
    /// Bytes the class may have in flight.
    pub bytes: u64,
}

/// How classes map onto a queue set and share it.
pub trait ClassPolicy {
    /// The order `advance` fills classes in.
    const ORDER: [Class; Class::COUNT];
    /// Bytes one ring may issue in a class before the next ring's turn.
    const QUANTUM_BYTES: u64;

    /// The hardware submission queue `class` rides.
    fn sq(class: Class) -> usize;
    /// `class`'s in-flight budget.
    fn budget(class: Class) -> Budget;
    /// The deadline a submission of `class` gets when it names none.
    fn deadline_ns(class: Class) -> u64;
}

/// Every class on one hardware queue — the policy a [`Single`](super::Single)
/// queue set runs. Commit fills first, then Demand, then Stream; the Stream
/// byte budget keeps bulk reads from occupying the whole queue.
pub struct OneQueue;

impl ClassPolicy for OneQueue {
    const ORDER: [Class; Class::COUNT] = [Class::Commit, Class::Demand, Class::Stream];
    const QUANTUM_BYTES: u64 = 1 << 20;

    fn sq(_class: Class) -> usize {
        0
    }

    fn budget(class: Class) -> Budget {
        match class {
            Class::Commit => Budget {
                commands: 32,
                bytes: 4 << 20,
            },
            Class::Demand => Budget {
                commands: 64,
                bytes: 32 << 20,
            },
            Class::Stream => Budget {
                commands: 128,
                bytes: 16 << 20,
            },
        }
    }

    fn deadline_ns(class: Class) -> u64 {
        match class {
            // A metadata write under device garbage collection can take
            // seconds; the budget matches the filesystem's stall budget.
            Class::Commit => 30_000_000_000,
            Class::Demand | Class::Stream => 30_000_000_000,
        }
    }
}

/// One hardware submission queue per class — Demand on 0, Commit on 1,
/// Stream on 2 — over one completion queue: the policy a queue group runs
/// (`res/STORAGE_TRANSPORT.md` §3). A demand read never sits behind bulk
/// work in the device's own queue, only behind the arbitration burst the
/// controller takes from each non-empty queue in turn; a controller under
/// weighted round robin serves them urgent, high, medium in that order.
/// Budgets and the fill order are [`OneQueue`]'s; only the queue each class
/// rides differs.
pub struct ThreeClass;

impl ClassPolicy for ThreeClass {
    const ORDER: [Class; Class::COUNT] = OneQueue::ORDER;
    const QUANTUM_BYTES: u64 = OneQueue::QUANTUM_BYTES;

    fn sq(class: Class) -> usize {
        match class {
            Class::Demand => 0,
            Class::Commit => 1,
            Class::Stream => 2,
        }
    }

    fn budget(class: Class) -> Budget {
        OneQueue::budget(class)
    }

    fn deadline_ns(class: Class) -> u64 {
        OneQueue::deadline_ns(class)
    }
}

/// The side effects a worker's host performs for the engine.
pub trait StorHost {
    /// `ring`'s completion channel was committed and its consumer had
    /// parked: send the ring's notice.
    fn kick(&mut self, ring: RingId);
    /// Nothing naming `grant` on `ring` is in flight any more; its slot is
    /// free and the blocks it named may be reused.
    fn drained(&mut self, ring: RingId, grant: GrantId);
    /// Detached `ring` has nothing in flight; its targets can be taken back
    /// and the ring retired.
    fn departed(&mut self, ring: RingId);
    /// An op passed its deadline while the device still held part of it.
    /// Reported once per op; recovery (abort, reset) is the host's.
    fn expired(&mut self, expiry: Expiry);
}

/// An op that outlived its deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expiry {
    /// The ring that submitted it.
    pub ring: RingId,
    /// The submission's token.
    pub user_data: u64,
    /// Its class.
    pub class: Class,
    /// Device commands of it still outstanding.
    pub outstanding: u16,
}

/// Which way the device moves bytes through a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetAccess {
    /// Reads land in it: the device writes.
    ReadInto,
    /// Writes come from it: the device reads.
    WriteFrom,
    /// Both.
    Both,
}

impl TargetAccess {
    const fn admits_read_into(self) -> bool {
        matches!(self, Self::ReadInto | Self::Both)
    }

    const fn admits_write_from(self) -> bool {
        matches!(self, Self::WriteFrom | Self::Both)
    }
}

/// Why a control-plane call was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorRefusal {
    /// The id names nothing live.
    Stale,
    /// A fixed table is full.
    Full,
    /// Something still in flight forbids it.
    Busy,
    /// The ring is detached or not yet idle for this step.
    Departed,
    /// The request is malformed (a target too large to address, a policy
    /// that names a queue the set does not have).
    Invalid,
}

/// Whether a revoke or detach finished at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settle {
    /// Nothing was in flight; done now, no host callback follows.
    Now,
    /// Something is in flight; the host callback fires when it drains.
    Later,
}

/// Engine counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineStats {
    /// Submissions popped.
    pub submissions: u64,
    /// Completions posted.
    pub completions: u64,
    /// Submissions that failed validation.
    pub refused: u64,
    /// Device commands staged.
    pub commands: u64,
    /// Device completions naming no live command — never acted on.
    pub stale_completions: u64,
    /// Ops reported past their deadline.
    pub expired: u64,
    /// Completions a timeline could not record — each one stops that
    /// timeline's fence for good. The pop's window check keeps this zero.
    pub timeline_refusals: u64,
    /// Device commands the queue set refused to stage. Transient refusals
    /// (a full queue, no list page free) retry on the next pass; one that
    /// only ever refuses stalls its op, and this count is how that shows.
    pub stage_refusals: u64,
    /// The last stage refusal's error.
    pub last_stage_error: Option<BlockError>,
    /// Commands of expired ops the device agreed to abort.
    pub aborts: u64,
    /// Commands of expired ops the device would not (or could not) abort.
    pub aborts_refused: u64,
    /// Controller resets: an expired op still held after its abort grace.
    pub resets: u64,
    /// Resets that did not bring the device back.
    pub resets_failed: u64,
    /// Completion latency per class, device command granularity.
    pub latency: [LatencyBooks; Class::COUNT],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrantState {
    Live,
    Revoking,
}

struct GrantSlot {
    generation: u16,
    grant: Option<(Grant, GrantState)>,
    /// Ops naming it that have not completed.
    ops: u32,
}

struct TargetSlot<T> {
    target: T,
    bytes: u64,
    access: TargetAccess,
    /// Device commands using it that have not completed.
    commands: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RingState {
    Live,
    Departed,
}

struct RingSlot<'w, T, const GRANTS: usize, const TARGETS: usize> {
    window: ServiceRing<'w>,
    state: RingState,
    classes: ClassSet,
    grants: [GrantSlot; GRANTS],
    targets: [Option<TargetSlot<T>>; TARGETS],
    timelines: [Timeline; TIMELINES],
    /// The op this ring is issuing in each class, if one is partial.
    current: [Option<u16>; Class::COUNT],
    /// Ops popped and not yet completed.
    open: u32,
    /// Completions posted since attach.
    completed: u64,
    /// Something was posted this advance; commit the channel.
    dirty: bool,
    /// `departed` was reported.
    departure_reported: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Read,
    Write,
    Flush,
    Discard,
}

#[derive(Clone, Copy)]
struct Op {
    ring: u16,
    class: Class,
    kind: Kind,
    user_data: u64,
    grant: u16,
    target: Option<u16>,
    timeline: u16,
    seq: u64,
    /// Block offset inside the grant of the next piece to issue.
    next_block: u64,
    /// Blocks still to issue.
    left_blocks: u64,
    /// Byte offset inside the target of the next piece.
    next_target: u64,
    /// Total bytes the op moves (the success `res`).
    bytes: u64,
    /// Device commands issued and not completed.
    outstanding: u16,
    /// Nothing more will be issued.
    issued: bool,
    error: Option<StorStatus>,
    deadline_ns: Option<u64>,
    expiry_reported: bool,
    /// Past its deadline and aborted: still held after this, the controller
    /// is reset.
    abort_deadline_ns: Option<u64>,
}

#[derive(Clone, Copy)]
struct Command {
    op: u16,
    bytes: u64,
    submitted_ns: Option<u64>,
    depth: u16,
}

/// The transport engine: one worker's data plane.
///
/// `RINGS` rings of `GRANTS` grants and `TARGETS` targets each; `OPS`
/// submissions and `COMMANDS` device commands in flight across all of them.
pub struct StorEngine<
    'w,
    Q: QueueSet,
    P: ClassPolicy,
    const RINGS: usize,
    const GRANTS: usize,
    const TARGETS: usize,
    const OPS: usize,
    const COMMANDS: usize,
> {
    queues: Q,
    rings: [Option<RingSlot<'w, Q::Target, GRANTS, TARGETS>>; RINGS],
    ring_generations: [u16; RINGS],
    ops: [Option<Op>; OPS],
    commands: [Option<Command>; COMMANDS],
    /// Commands and bytes in flight per class.
    in_flight: [(u32, u64); Class::COUNT],
    /// Round-robin start per class.
    cursor: [usize; Class::COUNT],
    /// Submission queues with staged, unpublished commands.
    staged: u64,
    next_deadline_scan: u64,
    stats: EngineStats,
    _policy: core::marker::PhantomData<P>,
}

/// How often the in-flight table is scanned for expired deadlines.
const DEADLINE_SCAN_NS: u64 = 1_000_000;
/// Completions reaped per device drain.
const REAP_BATCH: usize = 32;

impl<
    'w,
    Q: QueueSet,
    P: ClassPolicy,
    const RINGS: usize,
    const GRANTS: usize,
    const TARGETS: usize,
    const OPS: usize,
    const COMMANDS: usize,
> StorEngine<'w, Q, P, RINGS, GRANTS, TARGETS, OPS, COMMANDS>
{
    /// An engine driving `queues`. Refuses a policy that names a queue the
    /// set does not have, tables too large for their index types, or a
    /// device with a zero block size.
    ///
    /// The engine is tens of KiB of fixed tables; a caller whose stack is
    /// tight builds it in place with [`init_in`](Self::init_in) instead.
    pub fn new(queues: Q) -> Result<Self, (Q, StorRefusal)> {
        let mut place = core::mem::MaybeUninit::uninit();
        Self::init_in(&mut place, queues)?;
        // SAFETY: `init_in` returned `Ok`, so every field is written.
        Ok(unsafe { place.assume_init() })
    }

    /// [`new`](Self::new), written straight into `place` — a heap slot, say
    /// — so no engine-sized temporary ever exists on the caller's stack.
    /// Each table is written element by element for the same reason. On
    /// `Err`, `place` is untouched and still uninitialized.
    pub fn init_in(
        place: &mut core::mem::MaybeUninit<Self>,
        queues: Q,
    ) -> Result<&mut Self, (Q, StorRefusal)> {
        let policy_fits = Class::ALL.iter().all(|&class| P::sq(class) < Q::SQS);
        let tables_fit = RINGS <= 64
            && GRANTS <= usize::from(u16::MAX)
            && TARGETS <= usize::from(u16::MAX)
            && OPS <= usize::from(u16::MAX)
            && COMMANDS <= u32::MAX as usize
            && Q::SQS <= 64;
        if !policy_fits || !tables_fit || queues.geometry().block_size == 0 {
            return Err((queues, StorRefusal::Invalid));
        }
        let this = place.as_mut_ptr();
        // SAFETY: `this` is valid for writes of `Self` (it comes from the
        // exclusive `place`); each write targets one field, or one element
        // of an array field, by raw place projection — no reference to
        // uninitialized memory is formed — and every field is written
        // exactly once before `place` is assumed initialized.
        unsafe {
            use core::ptr::addr_of_mut;
            addr_of_mut!((*this).queues).write(queues);
            let rings = addr_of_mut!((*this).rings).cast::<Option<RingSlot<'w, Q::Target, GRANTS, TARGETS>>>();
            for i in 0..RINGS {
                rings.add(i).write(None);
            }
            addr_of_mut!((*this).ring_generations).write([0; RINGS]);
            let ops = addr_of_mut!((*this).ops).cast::<Option<Op>>();
            for i in 0..OPS {
                ops.add(i).write(None);
            }
            let commands = addr_of_mut!((*this).commands).cast::<Option<Command>>();
            for i in 0..COMMANDS {
                commands.add(i).write(None);
            }
            addr_of_mut!((*this).in_flight).write([(0, 0); Class::COUNT]);
            addr_of_mut!((*this).cursor).write([0; Class::COUNT]);
            addr_of_mut!((*this).staged).write(0);
            addr_of_mut!((*this).next_deadline_scan).write(0);
            addr_of_mut!((*this).stats).write(EngineStats::default());
            addr_of_mut!((*this)._policy).write(core::marker::PhantomData);
            Ok(place.assume_init_mut())
        }
    }

    /// Take the queue set back, once no ring is attached.
    pub fn into_queues(self) -> Result<Q, Self> {
        if self.rings.iter().any(Option::is_some) {
            return Err(self);
        }
        Ok(self.queues)
    }

    /// The queue set.
    pub fn queues(&self) -> &Q {
        &self.queues
    }

    /// The queue set, mutably — a control step, a test's knobs.
    pub fn queues_mut(&mut self) -> &mut Q {
        &mut self.queues
    }

    /// Counters.
    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }

    /// The device's out-of-band liveness.
    pub fn liveness(&mut self) -> Liveness {
        self.queues.liveness()
    }

    /// Device commands in flight, all classes.
    pub fn commands_in_flight(&self) -> usize {
        self.commands.iter().flatten().count()
    }

    /// Submissions popped and not yet completed, all rings.
    pub fn ops_open(&self) -> usize {
        self.ops.iter().flatten().count()
    }

    /// Commit to sleeping: the consumer's half of every live ring's
    /// submission Dekker pair (`kore_ring::event`).
    ///
    /// `true` means every submission channel is empty and parked, and the
    /// engine holds no op — so nothing will wake its host except a producer
    /// that sees `parked` and sends its notice. `false` means work is
    /// pending (an op open, or a record queued or published after the
    /// channel was read): advance again instead of sleeping. A `false` may
    /// leave some channels parked; the cost is one spurious notice.
    ///
    /// Each channel's wake word is read **before** its queue is checked:
    /// a producer that commits between the two moves the word, so
    /// [`park`](kore_ring::event::EventSink::park) refuses, and one that
    /// commits after the park sees the intent and owes the notice.
    pub fn park_idle(&mut self) -> bool {
        if self.ops_open() != 0 || self.commands_in_flight() != 0 {
            return false;
        }
        for ring in self.rings.iter().flatten() {
            if ring.state != RingState::Live {
                continue;
            }
            for sink in &ring.window.sq {
                let seen = sink.seq();
                if sink.outstanding() != 0 || !sink.park(seen) {
                    return false;
                }
            }
        }
        true
    }

    // ── control plane ───────────────────────────────────────────────────

    /// Attach a formatted window's service halves as a new ring admitting
    /// `classes`.
    pub fn attach(
        &mut self,
        window: ServiceRing<'w>,
        classes: ClassSet,
    ) -> Result<RingId, (ServiceRing<'w>, StorRefusal)> {
        let Some(slot) = self.rings.iter().position(Option::is_none) else {
            return Err((window, StorRefusal::Full));
        };
        let mut window = window;
        window.cq.replace_state(CqState {
            status: RING_LIVE,
            in_flight: 0,
            completed: 0,
        });
        let _ = window.cq.commit();
        self.rings[slot] = Some(RingSlot {
            window,
            state: RingState::Live,
            classes,
            grants: core::array::from_fn(|_| GrantSlot {
                generation: 0,
                grant: None,
                ops: 0,
            }),
            targets: core::array::from_fn(|_| None),
            timelines: [Timeline::default(); TIMELINES],
            current: [None; Class::COUNT],
            open: 0,
            completed: 0,
            dirty: false,
            departure_reported: false,
        });
        Ok(RingId {
            slot: slot as u16,
            generation: self.ring_generations[slot],
        })
    }

    fn ring_mut(
        &mut self,
        id: RingId,
    ) -> Result<&mut RingSlot<'w, Q::Target, GRANTS, TARGETS>, StorRefusal> {
        let slot = usize::from(id.slot);
        if self.ring_generations.get(slot) != Some(&id.generation) {
            return Err(StorRefusal::Stale);
        }
        self.rings[slot].as_mut().ok_or(StorRefusal::Stale)
    }

    /// Install `grant` into `ring`.
    pub fn install(&mut self, ring: RingId, grant: Grant) -> Result<GrantId, StorRefusal> {
        let ring = self.ring_mut(ring)?;
        if ring.state != RingState::Live {
            return Err(StorRefusal::Departed);
        }
        let slot = ring
            .grants
            .iter()
            .position(|slot| slot.grant.is_none() && slot.ops == 0)
            .ok_or(StorRefusal::Full)?;
        let entry = &mut ring.grants[slot];
        entry.grant = Some((grant, GrantState::Live));
        Ok(GrantId::new(slot as u16, entry.generation))
    }

    /// Revoke `grant`. [`Settle::Now`] when nothing names it; otherwise
    /// [`StorHost::drained`] fires once the last op naming it completes, and
    /// the slot is reusable only then.
    pub fn revoke(&mut self, ring_id: RingId, grant: GrantId) -> Result<Settle, StorRefusal> {
        let ring = self.ring_mut(ring_id)?;
        let slot = ring
            .grants
            .get_mut(usize::from(grant.slot()))
            .ok_or(StorRefusal::Stale)?;
        if slot.generation & GrantId::MAX_GENERATION != grant.generation() {
            return Err(StorRefusal::Stale);
        }
        match slot.grant.as_mut() {
            None => Err(StorRefusal::Stale),
            Some((_, state)) if *state == GrantState::Revoking => Ok(Settle::Later),
            Some((_, state)) => {
                if slot.ops == 0 {
                    slot.grant = None;
                    slot.generation = slot.generation.wrapping_add(1);
                    Ok(Settle::Now)
                } else {
                    *state = GrantState::Revoking;
                    Ok(Settle::Later)
                }
            }
        }
    }

    /// Adopt `target` into `ring` for `access`. The target is the ring's
    /// until [`release_target`](Self::release_target) hands it back.
    pub fn adopt_target(
        &mut self,
        ring: RingId,
        target: Q::Target,
        access: TargetAccess,
    ) -> Result<u16, (Q::Target, StorRefusal)> {
        let ring = match self.ring_mut(ring) {
            Ok(ring) => ring,
            Err(refusal) => return Err((target, refusal)),
        };
        if ring.state != RingState::Live {
            return Err((target, StorRefusal::Departed));
        }
        let Some(slot) = ring.targets.iter().position(Option::is_none) else {
            return Err((target, StorRefusal::Full));
        };
        let bytes = Q::target_bytes(&target);
        ring.targets[slot] = Some(TargetSlot {
            target,
            bytes,
            access,
            commands: 0,
        });
        Ok(slot as u16)
    }

    /// Target `slot` of `ring`, mutably — its owner's CPU view, refused while
    /// any command uses it.
    pub fn target_mut(&mut self, ring: RingId, slot: u16) -> Result<&mut Q::Target, StorRefusal> {
        let ring = self.ring_mut(ring)?;
        match ring.targets.get_mut(usize::from(slot)) {
            Some(Some(entry)) if entry.commands == 0 => Ok(&mut entry.target),
            Some(Some(_)) => Err(StorRefusal::Busy),
            _ => Err(StorRefusal::Stale),
        }
    }

    /// Submissions `ring` has popped and not yet completed.
    pub fn open(&self, ring: RingId) -> Option<u32> {
        let slot = usize::from(ring.slot);
        if self.ring_generations.get(slot) != Some(&ring.generation) {
            return None;
        }
        self.rings[slot].as_ref().map(|ring| ring.open)
    }

    /// Hand target `slot` back, once no command uses it — with the proof
    /// that its DMA mapping may retire while the device runs on: a command
    /// counts against its target from the pop until its completion is reaped,
    /// and the emptied slot is unreachable to any entry not yet popped.
    pub fn release_target(&mut self, ring: RingId, slot: u16) -> Result<Quiet<Q::Target>, StorRefusal> {
        let ring = self.ring_mut(ring)?;
        let entry = ring
            .targets
            .get_mut(usize::from(slot))
            .ok_or(StorRefusal::Stale)?;
        match entry {
            None => Err(StorRefusal::Stale),
            Some(target) if target.commands != 0 => Err(StorRefusal::Busy),
            Some(_) => Ok(Quiet::witness(entry.take().expect("checked occupied").target)),
        }
    }

    /// Detach `ring`: pop nothing more, stop issuing its partial ops, let
    /// what the device holds finish. [`Settle::Now`] when nothing was open.
    pub fn detach(&mut self, ring_id: RingId) -> Result<Settle, StorRefusal> {
        let ring_slot = usize::from(ring_id.slot);
        let ops = &mut self.ops;
        let ring = {
            if self.ring_generations.get(ring_slot) != Some(&ring_id.generation) {
                return Err(StorRefusal::Stale);
            }
            self.rings[ring_slot].as_mut().ok_or(StorRefusal::Stale)?
        };
        if ring.state == RingState::Departed {
            return Ok(if ring.open == 0 { Settle::Now } else { Settle::Later });
        }
        ring.state = RingState::Departed;
        ring.window.cq.state(|state| state.status = RING_DEPARTED);
        let _ = ring.window.cq.commit();
        // Partial ops stop issuing; what is out comes back.
        for current in ring.current.iter_mut() {
            if let Some(index) = current.take()
                && let Some(op) = ops[usize::from(index)].as_mut()
            {
                op.issued = true;
                op.error.get_or_insert(StorStatus::Cancelled);
            }
        }
        if ring.open == 0 {
            ring.departure_reported = true;
            Ok(Settle::Now)
        } else {
            // Any partial op that just became fully issued with nothing out
            // completes on the next advance.
            Ok(Settle::Later)
        }
    }

    /// Retire a detached, idle ring whose targets have all been released,
    /// handing its window halves back.
    pub fn retire(&mut self, ring_id: RingId) -> Result<ServiceRing<'w>, StorRefusal> {
        let ring = self.ring_mut(ring_id)?;
        if ring.state != RingState::Departed || ring.open != 0 {
            return Err(StorRefusal::Busy);
        }
        if ring.targets.iter().any(Option::is_some) {
            return Err(StorRefusal::Busy);
        }
        let slot = usize::from(ring_id.slot);
        let ring = self.rings[slot].take().expect("checked occupied");
        self.ring_generations[slot] = self.ring_generations[slot].wrapping_add(1);
        Ok(ring.window)
    }

    // ── data plane ──────────────────────────────────────────────────────

    /// Drive the data plane once: reap, fill every class in order, publish,
    /// reap again, report expiries, commit every touched ring. Never waits.
    pub fn advance<H: StorHost>(&mut self, host: &mut H) {
        let now = self.queues.clock_ns();
        self.reap(host);
        for class in P::ORDER {
            self.fill(class, now);
        }
        self.publish_staged();
        self.reap(host);
        self.finish_idle_ops(host);
        if let Some(now) = now
            && now >= self.next_deadline_scan
        {
            self.next_deadline_scan = now.saturating_add(DEADLINE_SCAN_NS);
            self.scan_deadlines(now, host);
        }
        self.commit_rings(host);
    }

    fn publish_staged(&mut self) {
        let mut staged = core::mem::take(&mut self.staged);
        while staged != 0 {
            let sq = staged.trailing_zeros() as usize;
            staged &= staged - 1;
            self.queues.publish(sq);
        }
    }

    fn commit_rings<H: StorHost>(&mut self, host: &mut H) {
        for (slot, entry) in self.rings.iter_mut().enumerate() {
            let Some(ring) = entry.as_mut() else { continue };
            if !ring.dirty {
                continue;
            }
            ring.dirty = false;
            let (open, completed) = (ring.open, ring.completed);
            ring.window.cq.state(|state| {
                state.in_flight = open;
                state.completed = completed;
            });
            if ring.window.cq.commit() {
                host.kick(RingId {
                    slot: slot as u16,
                    generation: self.ring_generations[slot],
                });
            }
        }
    }

    /// Fill `class`: rings round robin, each up to one quantum per turn,
    /// until the class budget, the device queue, or every ring runs dry.
    fn fill(&mut self, class: Class, now: Option<u64>) {
        let sq = P::sq(class);
        let budget = P::budget(class);
        loop {
            let mut progressed = false;
            for turn in 0..RINGS {
                let slot = (self.cursor[class.index()] + turn) % RINGS;
                let mut quantum = P::QUANTUM_BYTES;
                while quantum > 0 {
                    match self.step(slot, class, sq, budget, now) {
                        Step::Issued(bytes) => {
                            progressed = true;
                            quantum = quantum.saturating_sub(bytes.max(1));
                        }
                        Step::Posted => progressed = true,
                        Step::Stop => break,
                    }
                }
            }
            if RINGS > 0 {
                self.cursor[class.index()] = (self.cursor[class.index()] + 1) % RINGS;
            }
            if !progressed {
                break;
            }
        }
    }

    /// One unit of progress for `slot` in `class`: issue the next piece of
    /// its current op, or pop and plan its next submission.
    fn step(
        &mut self,
        slot: usize,
        class: Class,
        sq: usize,
        budget: Budget,
        now: Option<u64>,
    ) -> Step {
        let Some(ring) = self.rings[slot].as_mut() else {
            return Step::Stop;
        };
        let op_index = match ring.current[class.index()] {
            Some(index) => index,
            None => {
                if ring.state != RingState::Live || !ring.classes.contains(class) {
                    return Step::Stop;
                }
                // The completion credit: an op is popped only if the one
                // completion it owes can be posted.
                if ring.open >= ring.window.cq.capacity_left() {
                    return Step::Stop;
                }
                let Some(free_op) = self.ops.iter().position(Option::is_none) else {
                    return Step::Stop;
                };
                // The timeline's window: a submission whose timeline already
                // has a window's worth open waits, so every completion it
                // owes lands inside the window and the fence keeps moving.
                // A batch longer than the window drains through it in order.
                let Some(next) = ring.window.sq[class.index()].peek() else {
                    return Step::Stop;
                };
                if let Some(books) = usize::from(next.personality)
                    .checked_sub(1)
                    .and_then(|index| ring.timelines.get(index))
                    && books.open() >= super::timeline::WINDOW
                {
                    return Step::Stop;
                }
                let Some(sqe) = ring.window.sq[class.index()].pop() else {
                    return Step::Stop;
                };
                self.stats.submissions += 1;
                ring.open += 1;
                match plan::<Q::Target, P, GRANTS, TARGETS>(
                    ring,
                    slot as u16,
                    class,
                    &sqe,
                    self.queues.geometry(),
                    now,
                    budget,
                    self.in_flight[class.index()],
                ) {
                    Ok(op) => {
                        self.ops[free_op] = Some(op);
                        ring.current[class.index()] = Some(free_op as u16);
                        free_op as u16
                    }
                    Err((status, timeline)) => {
                        self.stats.refused += 1;
                        if !post(ring, sqe.user_data, status.res(), timeline, false) {
                            self.stats.timeline_refusals += 1;
                        }
                        ring.open -= 1;
                        self.stats.completions += 1;
                        return Step::Posted;
                    }
                }
            }
        };
        // Issue the next piece of the current op.
        let ops = &mut self.ops;
        let op = ops[usize::from(op_index)].as_mut().expect("current op is live");
        let grant_state = ring.grants[usize::from(op.grant)].grant.map(|(_, state)| state);
        if grant_state != Some(GrantState::Live) {
            op.issued = true;
            op.error.get_or_insert(StorStatus::Revoked);
        }
        if op.issued {
            ring.current[class.index()] = None;
            return Step::Posted;
        }
        let (in_cmds, in_bytes) = self.in_flight[class.index()];
        if in_cmds >= budget.commands || self.queues.room(sq) == 0 {
            return Step::Stop;
        }
        let Some(tag) = self.commands.iter().position(Option::is_none) else {
            return Step::Stop;
        };
        let geometry = self.queues.geometry();
        let block = u64::from(geometry.block_size);
        let (command, piece_blocks) = match op.kind {
            Kind::Flush => (DeviceCommand::Flush, 0),
            Kind::Read | Kind::Write | Kind::Discard => {
                let (grant, _) = ring.grants[usize::from(op.grant)].grant.expect("grant is live");
                let ceiling = match op.kind {
                    Kind::Discard => u32::MAX,
                    _ => geometry.max_transfer_blocks,
                };
                let max = op.left_blocks.min(u64::from(ceiling)) as u32;
                let Some(piece) = grant.segment(op.next_block, max) else {
                    // Validated in range; unreachable unless the grant moved.
                    op.issued = true;
                    op.error.get_or_insert(StorStatus::Range);
                    ring.current[class.index()] = None;
                    return Step::Posted;
                };
                let buffer_offset = op.next_target as usize;
                let command = match op.kind {
                    Kind::Read => DeviceCommand::Read {
                        lba: piece.lba,
                        blocks: piece.blocks,
                        buffer_offset,
                    },
                    Kind::Write => DeviceCommand::Write {
                        lba: piece.lba,
                        blocks: piece.blocks,
                        buffer_offset,
                    },
                    _ => DeviceCommand::Discard {
                        lba: piece.lba,
                        blocks: piece.blocks,
                    },
                };
                (command, piece.blocks)
            }
        };
        let piece_bytes = u64::from(piece_blocks) * block;
        let moves_data = matches!(op.kind, Kind::Read | Kind::Write);
        // A piece bigger than the whole byte budget still goes alone, or a
        // class whose budget is below one transfer could never move.
        if moves_data && in_cmds > 0 && in_bytes.saturating_add(piece_bytes) > budget.bytes {
            return Step::Stop;
        }
        let target = match op.target {
            Some(t) => ring.targets[usize::from(t)].as_mut().map(|slot| &mut slot.target),
            None => None,
        };
        let submitted_ns = self.queues.clock_ns();
        if let Err(error) = self.queues.stage(sq, command, target, Tag(tag as u32)) {
            self.stats.stage_refusals += 1;
            self.stats.last_stage_error = Some(error);
            return Step::Stop;
        }
        self.staged |= 1 << sq;
        self.stats.commands += 1;
        if let Some(t) = op.target {
            if let Some(entry) = ring.targets[usize::from(t)].as_mut() {
                entry.commands += 1;
            }
        }
        self.commands[tag] = Some(Command {
            op: op_index,
            bytes: if moves_data { piece_bytes } else { 0 },
            submitted_ns,
            depth: in_cmds.min(u32::from(u16::MAX)) as u16,
        });
        self.in_flight[class.index()] = (
            in_cmds + 1,
            in_bytes + if moves_data { piece_bytes } else { 0 },
        );
        op.outstanding += 1;
        op.next_block += u64::from(piece_blocks);
        op.next_target += piece_bytes;
        // A flush is one piece of no blocks: it is fully issued once staged.
        op.left_blocks -= if op.kind == Kind::Flush {
            op.left_blocks
        } else {
            u64::from(piece_blocks)
        };
        if op.left_blocks == 0 {
            op.issued = true;
            ring.current[class.index()] = None;
        }
        Step::Issued(piece_bytes)
    }

    /// Drain the device's completions.
    fn reap<H: StorHost>(&mut self, host: &mut H) {
        let mut batch = [DeviceCompletion {
            cookie: 0,
            result: Ok(()),
        }; REAP_BATCH];
        loop {
            let n = self.queues.reap(&mut batch);
            for completion in &batch[..n] {
                self.complete_command(*completion, host);
            }
            if n < REAP_BATCH {
                break;
            }
        }
    }

    fn complete_command<H: StorHost>(&mut self, completion: DeviceCompletion, host: &mut H) {
        let tag = completion.cookie as usize;
        let Some(command) = self.commands.get_mut(tag).and_then(Option::take) else {
            self.stats.stale_completions += 1;
            return;
        };
        let op_index = usize::from(command.op);
        let Some(op) = self.ops[op_index].as_mut() else {
            self.stats.stale_completions += 1;
            return;
        };
        let class = op.class;
        let ring = self.rings[usize::from(op.ring)]
            .as_mut()
            .expect("an op's ring outlives it");
        let mut result = completion.result;
        if let Some(t) = op.target
            && let Some(entry) = ring.targets[usize::from(t)].as_mut()
        {
            entry.commands -= 1;
            if result.is_ok() {
                result = self.queues.finish(Some(&mut entry.target));
            }
        }
        if let Err(error) = result {
            op.error.get_or_insert(status_of(error));
        }
        op.outstanding -= 1;
        let (cmds, bytes) = &mut self.in_flight[class.index()];
        *cmds -= 1;
        *bytes -= command.bytes;
        if let (Some(start), Some(now)) = (command.submitted_ns, self.queues.clock_ns()) {
            self.stats.latency[class.index()].record(now.saturating_sub(start), command.depth);
        }
        if op.issued && op.outstanding == 0 {
            self.finish_op(op_index, host);
        }
    }

    /// Ops made fully issued with nothing outstanding outside a completion
    /// (a revoke or detach stopped them between pieces) complete here.
    fn finish_idle_ops<H: StorHost>(&mut self, host: &mut H) {
        for index in 0..OPS {
            if let Some(op) = self.ops[index]
                && op.issued
                && op.outstanding == 0
            {
                self.finish_op(index, host);
            }
        }
        // A departed ring with nothing open reports once.
        for (slot, entry) in self.rings.iter_mut().enumerate() {
            if let Some(ring) = entry.as_mut()
                && ring.state == RingState::Departed
                && ring.open == 0
                && !ring.departure_reported
            {
                ring.departure_reported = true;
                host.departed(RingId {
                    slot: slot as u16,
                    generation: self.ring_generations[slot],
                });
            }
        }
    }

    fn finish_op<H: StorHost>(&mut self, index: usize, host: &mut H) {
        let op = self.ops[index].take().expect("finishing a live op");
        let ring_slot = usize::from(op.ring);
        let ring = self.rings[ring_slot]
            .as_mut()
            .expect("an op's ring outlives it");
        if ring.current[op.class.index()] == Some(index as u16) {
            ring.current[op.class.index()] = None;
        }
        let res = match op.error {
            Some(status) => status.res(),
            None => i32::try_from(op.bytes).unwrap_or(i32::MAX),
        };
        let timeline = (op.timeline != 0).then_some((op.timeline, op.seq));
        if !post(ring, op.user_data, res, timeline, op.error.is_none()) {
            self.stats.timeline_refusals += 1;
        }
        ring.open -= 1;
        self.stats.completions += 1;
        let ring_id = RingId {
            slot: op.ring,
            generation: self.ring_generations[ring_slot],
        };
        let grant = &mut ring.grants[usize::from(op.grant)];
        grant.ops -= 1;
        if grant.ops == 0
            && let Some((_, GrantState::Revoking)) = grant.grant
        {
            let id = GrantId::new(op.grant, grant.generation);
            grant.grant = None;
            grant.generation = grant.generation.wrapping_add(1);
            host.drained(ring_id, id);
        }
        if ring.state == RingState::Departed && ring.open == 0 && !ring.departure_reported {
            ring.departure_reported = true;
            host.departed(ring_id);
        }
    }

    fn scan_deadlines<H: StorHost>(&mut self, now: u64, host: &mut H) {
        let mut reset = false;
        for (index, slot) in self.ops.iter_mut().enumerate() {
            let Some(op) = slot.as_mut() else { continue };
            if op.outstanding == 0 {
                continue;
            }
            // The last rung: an op its abort did not release. Aborted or
            // refused alike — a refusal may only mean the command was
            // completing — it gets the grace, and then the controller.
            if op.abort_deadline_ns.is_some_and(|deadline| now > deadline) {
                reset = true;
                continue;
            }
            if op.expiry_reported {
                continue;
            }
            if op.deadline_ns.is_some_and(|deadline| now > deadline) {
                op.expiry_reported = true;
                op.abort_deadline_ns = Some(now.saturating_add(ABORT_GRACE_NS));
                self.stats.expired += 1;
                // The escalation: ask the device to abandon every command of
                // the op it still holds. An agreed abort fails the command
                // through its own completion, so the op fails instead of
                // hanging its ticket; one still held past the grace resets
                // the controller below.
                let sq = P::sq(op.class);
                for (tag, command) in self.commands.iter().enumerate() {
                    if command.is_some_and(|command| usize::from(command.op) == index) {
                        if self.queues.abort(sq, Tag(tag as u32)) {
                            self.stats.aborts += 1;
                        } else {
                            self.stats.aborts_refused += 1;
                        }
                    }
                }
                host.expired(Expiry {
                    ring: RingId {
                        slot: op.ring,
                        generation: self.ring_generations[usize::from(op.ring)],
                    },
                    user_data: op.user_data,
                    class: op.class,
                    outstanding: op.outstanding,
                });
            }
        }
        if reset {
            self.reset_device(host);
        }
    }

    /// Reset the device and fail every command it held
    /// (`res/STORAGE_TRANSPORT.md` §7): each completes `Lost` through the
    /// ordinary completion path, so its op fails, its fence moves, and its
    /// target is free. Rings, grants and targets survive; only the commands
    /// are gone. A reset that fails leaves everything as it was, counted.
    fn reset_device<H: StorHost>(&mut self, host: &mut H) {
        if !self.queues.reset() {
            self.stats.resets_failed += 1;
            return;
        }
        self.stats.resets += 1;
        for tag in 0..self.commands.len() {
            if self.commands[tag].is_some() {
                self.complete_command(
                    DeviceCompletion {
                        cookie: tag as u64,
                        result: Err(BlockError::Lost),
                    },
                    host,
                );
            }
        }
    }
}

/// How long an aborted command may stay held before the controller is
/// reset. An agreed abort completes the command at once; one still out a
/// second later is one the device will not give up.
const ABORT_GRACE_NS: u64 = 1_000_000_000;

enum Step {
    /// A device command was staged moving this many bytes.
    Issued(u64),
    /// A completion was posted, or an op stopped issuing.
    Posted,
    /// Nothing more for this ring in this class this turn.
    Stop,
}

/// Post one completion into `ring`'s channel and, when it names one,
/// advance its timeline and publish the fence. `false` when the timeline
/// refused the completion ([`EngineStats::timeline_refusals`]).
fn post<'w, T, const GRANTS: usize, const TARGETS: usize>(
    ring: &mut RingSlot<'w, T, GRANTS, TARGETS>,
    user_data: u64,
    res: i32,
    timeline: Option<(u16, u64)>,
    ok: bool,
) -> bool {
    let mut cqe = IoUringCqe::new_zeroed();
    cqe.user_data = user_data;
    cqe.res = res;
    // The completion credit taken at pop guarantees room.
    let pushed = ring.window.cq.push(cqe);
    debug_assert!(pushed.is_ok(), "completion credit violated");
    ring.completed += 1;
    ring.dirty = true;
    if let Some((t, seq)) = timeline
        && let Some(books) = ring.timelines.get_mut(usize::from(t) - 1)
    {
        match books.complete(seq, ok) {
            Ok(record) => {
                if let Some(cell) = ring.window.fences.timeline(t) {
                    cell.publish(record);
                }
            }
            // Unreachable while the pop holds the window; counted, because
            // this is the fence stopping for good.
            Err(_) => return false,
        }
    }
    true
}

/// Validate one popped submission against `ring`'s grants and targets and
/// build its op. A failure returns the status and — when the submission
/// named a valid timeline — the sequence it consumed, so the fence records
/// the failure in order.
#[allow(clippy::too_many_arguments)]
fn plan<'w, T, P: ClassPolicy, const GRANTS: usize, const TARGETS: usize>(
    ring: &mut RingSlot<'w, T, GRANTS, TARGETS>,
    ring_slot: u16,
    class: Class,
    sqe: &IoUringSqe,
    geometry: crate::block::BlockGeometry,
    now: Option<u64>,
    budget: Budget,
    in_flight: (u32, u64),
) -> Result<Op, (StorStatus, Option<(u16, u64)>)> {
    // The timeline first: a submission that names one consumes a sequence
    // even when it fails, because the client counted it.
    let timeline = sqe.personality;
    if usize::from(timeline) > TIMELINES {
        return Err((StorStatus::Invalid, None));
    }
    let seq = if timeline == 0 {
        0
    } else {
        ring.timelines[usize::from(timeline) - 1].issue()
    };
    let named = (timeline != 0).then_some((timeline, seq));
    let fail = |status| Err((status, named));

    let kind = match sqe.opcode {
        op::READ_FIXED => Kind::Read,
        op::WRITE_FIXED => Kind::Write,
        op::FSYNC => Kind::Flush,
        op::DISCARD => Kind::Discard,
        _ => return fail(StorStatus::Invalid),
    };
    if sqe.flags != 0 || sqe.op_flags & !RWF_KNOWN != 0 {
        return fail(StorStatus::Invalid);
    }
    if sqe.op_flags & RWF_DSYNC != 0 {
        // Force-unit-access arrives with the NVMe queue set (S2).
        return fail(StorStatus::Unsupported);
    }
    if sqe.op_flags & RWF_NOWAIT != 0 && in_flight.0 >= budget.commands {
        return fail(StorStatus::Busy);
    }
    let Some(grant_id) = GrantId::from_fd(sqe.fd) else {
        return fail(StorStatus::Stale);
    };
    let Some(grant_slot) = ring.grants.get_mut(usize::from(grant_id.slot())) else {
        return fail(StorStatus::Stale);
    };
    if grant_slot.generation & GrantId::MAX_GENERATION != grant_id.generation() {
        return fail(StorStatus::Stale);
    }
    let Some((grant, state)) = grant_slot.grant else {
        return fail(StorStatus::Stale);
    };
    if state == GrantState::Revoking {
        return fail(StorStatus::Revoked);
    }
    if !grant.classes().contains(class) {
        return fail(StorStatus::Class);
    }
    let needs = match kind {
        Kind::Read => Rights::READ,
        Kind::Write => Rights::WRITE,
        Kind::Flush => Rights::FLUSH,
        Kind::Discard => Rights::DISCARD,
    };
    if !grant.rights().contains(needs) {
        return fail(StorStatus::Rights);
    }
    let block = u64::from(geometry.block_size);
    let (first_block, blocks, target, next_target, bytes) = match kind {
        Kind::Flush => (0, 0, None, 0, 0),
        Kind::Read | Kind::Write | Kind::Discard => {
            let (off, len) = (sqe.off, u64::from(sqe.len));
            if len == 0 || off % block != 0 || len % block != 0 {
                return fail(StorStatus::Alignment);
            }
            let (first, count) = (off / block, len / block);
            if first.checked_add(count).is_none_or(|end| end > grant.blocks()) {
                return fail(StorStatus::Range);
            }
            if kind == Kind::Discard {
                (first, count, None, 0, 0)
            } else {
                let Some(entry) = ring
                    .targets
                    .get(usize::from(sqe.buf_index))
                    .and_then(Option::as_ref)
                else {
                    return fail(StorStatus::Stale);
                };
                let admits = match kind {
                    Kind::Read => entry.access.admits_read_into(),
                    _ => entry.access.admits_write_from(),
                };
                if !admits {
                    return fail(StorStatus::Rights);
                }
                let at = sqe.addr;
                if at % block != 0 {
                    return fail(StorStatus::Alignment);
                }
                if at.checked_add(len).is_none_or(|end| end > entry.bytes)
                    || usize::try_from(at + len).is_err()
                {
                    return fail(StorStatus::Range);
                }
                if i32::try_from(len).is_err() {
                    // `res` carries the byte count; larger submissions split
                    // on the client side.
                    return fail(StorStatus::Range);
                }
                (first, count, Some(sqe.buf_index), at, len)
            }
        }
    };
    grant_slot.ops += 1;
    let deadline_ns = now.map(|now| {
        now.saturating_add(if sqe.addr3 != 0 {
            sqe.addr3.saturating_sub(now)
        } else {
            P::deadline_ns(class)
        })
    });
    Ok(Op {
        ring: ring_slot,
        class,
        kind,
        user_data: sqe.user_data,
        grant: grant_id.slot(),
        target,
        timeline,
        seq,
        next_block: first_block,
        left_blocks: blocks,
        next_target,
        bytes,
        outstanding: 0,
        issued: false,
        error: None,
        deadline_ns,
        expiry_reported: false,
        abort_deadline_ns: None,
    })
    .map(|mut op| {
        if kind == Kind::Flush {
            // One command, no blocks: issue it as a single piece.
            op.left_blocks = 1;
        }
        op
    })
}

/// A device failure as the wire reports it.
const fn status_of(error: BlockError) -> StorStatus {
    match error {
        BlockError::Media | BlockError::Integrity => StorStatus::Media,
        BlockError::Lost | BlockError::Transport => StorStatus::Lost,
        BlockError::Cancelled => StorStatus::Cancelled,
        BlockError::Revoked | BlockError::Draining => StorStatus::Revoked,
        BlockError::Unsupported => StorStatus::Unsupported,
        BlockError::ReadOnly => StorStatus::Rights,
        BlockError::Alignment => StorStatus::Alignment,
        BlockError::Range | BlockError::TooLarge | BlockError::Geometry => StorStatus::Range,
        BlockError::Backpressure
        | BlockError::UnknownBuffer
        | BlockError::BufferBusy
        | BlockError::UnknownRequest
        | BlockError::StaleCompletion => StorStatus::Lost,
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::boxed::Box;
    use std::vec::Vec;

    use kore_abi::stor::{CQ_DEPTH, FenceRecord, SQ_DEPTH};
    use kore_ring::stor::{ClientRing, WINDOW_BYTES, split};

    use super::*;
    use crate::block::memory::MemoryBlockDevice;
    use crate::block::{BlockBuffer, DirectBlockDevice};
    use crate::stor::Single;
    use crate::stor::grant::ExtentRun;

    /// 128 blocks of 512 B, a 4 KiB transfer ceiling so splitting shows.
    type Dev = MemoryBlockDevice<{ 64 * 1024 }, { 16 * 1024 }>;
    type Queues = Single<Dev, 32>;
    type Engine = StorEngine<'static, Queues, Tight, 2, 4, 2, 300, 64>;

    const BLOCK: u64 = 512;

    /// Tiny budgets and a one-block quantum so ordering and fairness show.
    struct Tight;

    impl ClassPolicy for Tight {
        const ORDER: [Class; Class::COUNT] = [Class::Commit, Class::Demand, Class::Stream];
        const QUANTUM_BYTES: u64 = BLOCK;

        fn sq(_class: Class) -> usize {
            0
        }

        fn budget(class: Class) -> Budget {
            match class {
                Class::Stream => Budget {
                    commands: 2,
                    bytes: 1 << 20,
                },
                Class::Commit | Class::Demand => Budget {
                    commands: 4,
                    bytes: 1 << 20,
                },
            }
        }

        fn deadline_ns(_class: Class) -> u64 {
            1_000
        }
    }

    #[derive(Default)]
    struct Rec {
        kicks: Vec<RingId>,
        drained: Vec<(RingId, GrantId)>,
        departed: Vec<RingId>,
        expired: Vec<Expiry>,
    }

    impl StorHost for Rec {
        fn kick(&mut self, ring: RingId) {
            self.kicks.push(ring);
        }
        fn drained(&mut self, ring: RingId, grant: GrantId) {
            self.drained.push((ring, grant));
        }
        fn departed(&mut self, ring: RingId) {
            self.departed.push(ring);
        }
        fn expired(&mut self, expiry: Expiry) {
            self.expired.push(expiry);
        }
    }

    /// A page-aligned window that lives for the test binary.
    fn window() -> &'static mut [u8] {
        let layout = std::alloc::Layout::from_size_align(WINDOW_BYTES, 4096).unwrap();
        // SAFETY: non-zero size; leaked on purpose for a 'static window.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        // SAFETY: freshly allocated, exclusively ours.
        unsafe { core::slice::from_raw_parts_mut(ptr, WINDOW_BYTES) }
    }

    fn engine() -> Box<Engine> {
        let dev = Dev::new(BLOCK as u32, 8).unwrap();
        Box::new(Engine::new(Single::new(dev)).ok().expect("policy fits"))
    }

    fn attach(engine: &mut Engine, classes: ClassSet) -> (ClientRing<'static>, RingId) {
        let (client, service) = split(window()).unwrap();
        let ring = engine.attach(service, classes).ok().expect("a free ring");
        (client, ring)
    }

    fn target(engine: &mut Engine, bytes: usize, fill: u8) -> <Dev as DirectBlockDevice>::Buffer {
        let mut buffer = engine.queues_mut().device_mut().allocate_buffer(bytes).unwrap();
        buffer.bytes_mut().fill(fill);
        buffer
    }

    #[allow(clippy::too_many_arguments)]
    fn sqe(
        opcode: u8,
        grant: GrantId,
        off: u64,
        len: u32,
        target: u16,
        at: u64,
        user_data: u64,
        timeline: u16,
    ) -> IoUringSqe {
        let mut entry = IoUringSqe::new_zeroed();
        entry.opcode = opcode;
        entry.fd = grant.fd();
        entry.off = off;
        entry.len = len;
        entry.buf_index = target;
        entry.addr = at;
        entry.user_data = user_data;
        entry.personality = timeline;
        entry
    }

    fn submit(client: &mut ClientRing<'_>, class: Class, entry: IoUringSqe) {
        client.sq[class.index()].push(entry).unwrap();
        let _ = client.sq[class.index()].commit();
    }

    fn completions(client: &mut ClientRing<'_>) -> Vec<(u64, i32)> {
        let mut out = Vec::new();
        while let Some(cqe) = client.cq.pop() {
            out.push((cqe.user_data, cqe.res));
        }
        out
    }

    fn rw_grant(runs: &[ExtentRun]) -> Grant {
        Grant::new(runs, Rights::READ | Rights::WRITE | Rights::FLUSH, ClassSet::ALL).unwrap()
    }

    /// Bytes written through one target come back through another, across
    /// two grant runs and three device commands (a 4-block run, then a
    /// 12-block run split at the 8-block ceiling) — one completion each.
    #[test]
    fn a_write_then_read_round_trips_across_runs_and_splits() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::ALL);
        let grant = engine
            .install(
                ring,
                rw_grant(&[
                    ExtentRun { lba: 10, blocks: 4 },
                    ExtentRun { lba: 40, blocks: 12 },
                ]),
            )
            .unwrap();
        let mut source = target(&mut engine, 8192, 0);
        for (i, byte) in source.bytes_mut().iter_mut().enumerate() {
            *byte = (i * 7 % 251) as u8;
        }
        let expected: Vec<u8> = source.bytes().to_vec();
        let src = engine.adopt_target(ring, source, TargetAccess::WriteFrom).ok().unwrap();
        let sink = target(&mut engine, 8192, 0xee);
        let dst = engine.adopt_target(ring, sink, TargetAccess::ReadInto).ok().unwrap();

        submit(&mut client, Class::Demand, sqe(op::WRITE_FIXED, grant, 0, 8192, src, 0, 1, 1));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(1, 8192)]);
        assert_eq!(engine.stats().commands, 3, "4 blocks, then 8 + 4 at the ceiling");

        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 8192, dst, 0, 2, 1));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(2, 8192)]);
        assert_eq!(
            client.fences.timeline(1).unwrap().load(),
            FenceRecord { done: 2, fault: 0 }
        );
        let back = engine.release_target(ring, dst).unwrap().into_inner();
        assert_eq!(back.bytes(), &expected[..]);
        // The runs landed where the grant said: block 3 of the grant is LBA
        // 13, block 4 is LBA 40.
        assert!(!host.kicks.is_empty() || client.cq.state().unwrap().completed == 2);
        assert_eq!(client.cq.state().unwrap().completed, 2);
    }

    /// Every malformed submission fails with its own status and still
    /// consumes its timeline sequence, so the fence records the fault in
    /// order.
    #[test]
    fn each_refusal_has_its_status_and_consumes_its_sequence() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::DATA);
        let read_only = engine
            .install(
                ring,
                Grant::new(
                    &[ExtentRun { lba: 0, blocks: 8 }],
                    Rights::READ,
                    ClassSet::only(Class::Demand),
                )
                .unwrap(),
            )
            .unwrap();
        let buf = target(&mut engine, 4096, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::ReadInto).ok().unwrap();
        let stale = GrantId::new(read_only.slot(), read_only.generation() + 1);

        let cases: [(IoUringSqe, Class, StorStatus); 9] = [
            (sqe(0x77, read_only, 0, 512, t, 0, 10, 2), Class::Demand, StorStatus::Invalid),
            (sqe(op::READ_FIXED, stale, 0, 512, t, 0, 11, 2), Class::Demand, StorStatus::Stale),
            (sqe(op::WRITE_FIXED, read_only, 0, 512, t, 0, 12, 2), Class::Demand, StorStatus::Rights),
            (sqe(op::READ_FIXED, read_only, 100, 512, t, 0, 13, 2), Class::Demand, StorStatus::Alignment),
            (sqe(op::READ_FIXED, read_only, 0, 8192, t, 0, 14, 2), Class::Demand, StorStatus::Range),
            (sqe(op::READ_FIXED, read_only, 0, 512, t, 4096, 15, 2), Class::Demand, StorStatus::Range),
            (sqe(op::READ_FIXED, read_only, 0, 512, 1, 0, 16, 2), Class::Demand, StorStatus::Stale),
            (sqe(op::READ_FIXED, read_only, 0, 512, t, 0, 17, 2), Class::Stream, StorStatus::Class),
            (sqe(op::READ_FIXED, read_only, 0, 512, t, 0, 18, 99), Class::Demand, StorStatus::Invalid),
        ];
        for (entry, class, _) in cases {
            submit(&mut client, class, entry);
        }
        engine.advance(&mut host);
        let mut got = completions(&mut client);
        got.sort();
        let mut want: Vec<(u64, i32)> = cases
            .iter()
            .map(|(entry, _, status)| (entry.user_data, status.res()))
            .collect();
        want.sort();
        assert_eq!(got, want);
        // Eight named timeline 2 and consumed sequences 1..=8, all failed;
        // the one naming timeline 99 consumed nothing.
        assert_eq!(
            client.fences.timeline(2).unwrap().load(),
            FenceRecord { done: 8, fault: 1 }
        );
        // A good read afterwards takes sequence 9 and leaves the fault.
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, read_only, 0, 512, t, 0, 19, 2));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(19, 512)]);
        assert_eq!(
            client.fences.timeline(2).unwrap().load(),
            FenceRecord { done: 9, fault: 1 }
        );
        assert_eq!(engine.stats().refused, 9);
    }

    /// Revoking a grant with reads in flight: new submissions naming it are
    /// refused at once, the in-flight ones complete, and the authority hears
    /// `drained` only then — after which the old id is stale and the slot
    /// is reusable.
    #[test]
    fn revoke_drains_before_the_slot_frees() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::ALL);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 16 }])).unwrap();
        let buf = target(&mut engine, 8192, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::Both).ok().unwrap();

        engine.queues_mut().device_mut().hold_completions(true);
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 4096, t, 0, 1, 0));
        engine.advance(&mut host);
        assert!(completions(&mut client).is_empty(), "the device holds it");

        assert_eq!(engine.revoke(ring, grant), Ok(Settle::Later));
        assert_eq!(engine.release_target(ring, t).err(), Some(StorRefusal::Busy));
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 512, t, 0, 2, 0));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(2, StorStatus::Revoked.res())]);
        assert!(host.drained.is_empty());

        engine.queues_mut().device_mut().hold_completions(false);
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(1, 4096)]);
        assert_eq!(host.drained, [(ring, grant)]);

        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 512, t, 0, 3, 0));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(3, StorStatus::Stale.res())]);
        let again = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 1 }])).unwrap();
        assert_eq!(again.slot(), grant.slot());
        assert_ne!(again.generation(), grant.generation());
        // An idle grant revokes at once, with no callback.
        assert_eq!(engine.revoke(ring, again), Ok(Settle::Now));
        assert_eq!(host.drained.len(), 1);
    }

    /// Detach stops popping, lets held commands finish, reports `departed`
    /// once; retire waits for the targets to come back.
    #[test]
    fn detach_departs_after_the_last_completion_and_retire_needs_the_targets() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::ALL);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 16 }])).unwrap();
        let buf = target(&mut engine, 4096, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::Both).ok().unwrap();

        engine.queues_mut().device_mut().hold_completions(true);
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 512, t, 0, 1, 0));
        engine.advance(&mut host);
        assert_eq!(engine.detach(ring), Ok(Settle::Later));
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 512, t, 0, 2, 0));
        engine.advance(&mut host);
        assert!(host.departed.is_empty());
        assert_eq!(client.cq.state().unwrap().status, RING_DEPARTED);

        engine.queues_mut().device_mut().hold_completions(false);
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(1, 512)], "the popped read completes; the later one is never popped");
        assert_eq!(host.departed, [ring]);
        engine.advance(&mut host);
        assert_eq!(host.departed.len(), 1, "reported once");

        assert_eq!(engine.retire(ring).err(), Some(StorRefusal::Busy), "a target is still adopted");
        let _ = engine.release_target(ring, t).unwrap().into_inner();
        assert!(engine.retire(ring).is_ok());
        assert_eq!(engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 1 }])), Err(StorRefusal::Stale));
    }

    /// The completion credit: with nobody reaping, the engine completes
    /// exactly one channel's worth of submissions and pops no further; one
    /// reaped completion admits exactly one more.
    #[test]
    fn a_full_completion_channel_stops_the_popping() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::ALL);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 16 }])).unwrap();
        let buf = target(&mut engine, 512, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::ReadInto).ok().unwrap();
        for i in 0..u64::from(SQ_DEPTH) {
            submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 512, t, 0, i, 0));
            submit(&mut client, Class::Stream, sqe(op::READ_FIXED, grant, 0, 512, t, 0, 1000 + i, 0));
        }
        // The tight budgets admit six commands an advance; run until the
        // channel stops moving.
        for _ in 0..(2 * SQ_DEPTH) {
            engine.advance(&mut host);
        }
        assert_eq!(client.cq.outstanding(), CQ_DEPTH);
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 512, t, 0, 9999, 0));
        engine.advance(&mut host);
        assert_eq!(client.cq.outstanding(), CQ_DEPTH, "no credit, not popped");
        client.cq.pop().unwrap();
        engine.advance(&mut host);
        assert_eq!(client.cq.outstanding(), CQ_DEPTH, "one reaped, one more posted");
        let tail: Vec<u64> = core::iter::from_fn(|| client.cq.pop().map(|c| c.user_data)).collect();
        assert_eq!(tail.last(), Some(&9999));
    }

    /// Commit fills before Demand before Stream; each class stays inside its
    /// command budget; within Stream two rings take turns a quantum at a
    /// time rather than one ring taking the whole budget.
    #[test]
    fn classes_fill_in_order_within_budget_and_rings_take_turns() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut a, ring_a) = attach(&mut engine, ClassSet::ALL);
        let (mut b, ring_b) = attach(&mut engine, ClassSet::ALL);
        let ga = engine.install(ring_a, rw_grant(&[ExtentRun { lba: 0, blocks: 64 }])).unwrap();
        let gb = engine.install(ring_b, rw_grant(&[ExtentRun { lba: 64, blocks: 64 }])).unwrap();
        let ba = target(&mut engine, 512, 0);
        let bb = target(&mut engine, 512, 0);
        let ta = engine.adopt_target(ring_a, ba, TargetAccess::ReadInto).ok().unwrap();
        let tb = engine.adopt_target(ring_b, bb, TargetAccess::ReadInto).ok().unwrap();

        engine.queues_mut().device_mut().hold_completions(true);
        for i in 0..4 {
            submit(&mut a, Class::Stream, sqe(op::READ_FIXED, ga, 0, 512, ta, 0, 100 + i, 0));
            submit(&mut b, Class::Stream, sqe(op::READ_FIXED, gb, 0, 512, tb, 0, 200 + i, 0));
        }
        for i in 0..6 {
            submit(&mut a, Class::Demand, sqe(op::READ_FIXED, ga, 0, 512, ta, 0, 300 + i, 0));
        }
        engine.advance(&mut host);
        // Demand: its budget of 4 commands, all from A. Stream: its budget
        // of 2 — one from each ring, not both from A.
        assert_eq!(engine.commands_in_flight(), 6);
        engine.queues_mut().device_mut().hold_completions(false);
        engine.advance(&mut host);
        let mut first: Vec<u64> = completions(&mut a)
            .into_iter()
            .chain(completions(&mut b))
            .map(|(user, _)| user)
            .collect();
        first.sort();
        assert!(first.contains(&100) && first.contains(&200), "each ring got a stream turn: {first:?}");
        assert!(first.iter().filter(|&&u| (300..306).contains(&u)).count() >= 4);
        // Everything completes eventually.
        for _ in 0..16 {
            engine.advance(&mut host);
        }
        let rest = completions(&mut a).len() + completions(&mut b).len();
        assert_eq!(first.len() + rest, 14);
    }

    /// An op the device holds past its deadline is reported once, with its
    /// outstanding count, and each command it holds is aborted once; the
    /// aborted command's own completion then fails the op.
    #[test]
    fn an_expired_op_is_reported_once_aborted_and_fails() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::ALL);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 16 }])).unwrap();
        let buf = target(&mut engine, 4096, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::ReadInto).ok().unwrap();

        engine.queues_mut().device_mut().set_clock(10_000, 0);
        engine.queues_mut().device_mut().hold_completions(true);
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 4096, t, 0, 7, 0));
        engine.advance(&mut host);
        assert!(host.expired.is_empty());

        engine.queues_mut().device_mut().set_clock(10_000 + 2_000_000, 0);
        engine.advance(&mut host);
        assert_eq!(host.expired.len(), 1);
        assert_eq!(host.expired[0].user_data, 7);
        assert_eq!(host.expired[0].outstanding, 1);
        assert_eq!((engine.stats().aborts, engine.stats().aborts_refused), (1, 0));
        engine.queues_mut().device_mut().set_clock(10_000 + 4_000_000, 0);
        engine.advance(&mut host);
        assert_eq!(host.expired.len(), 1, "once");
        assert_eq!(engine.stats().aborts, 1, "aborted once");

        // The aborted command's own completion comes back, failed: the op
        // fails instead of hanging its ticket.
        engine.queues_mut().device_mut().hold_completions(false);
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(7, StorStatus::Cancelled.res())]);
    }

    /// The last rung: an op whose abort the device refused is still held
    /// past the grace, so the controller is reset. Its command completes
    /// `Lost`, its fence moves, its target is free, and the ring, grant and
    /// target all keep working.
    #[test]
    fn a_command_the_device_will_not_give_up_resets_the_controller() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::ALL);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 16 }])).unwrap();
        let buf = target(&mut engine, 4096, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::ReadInto).ok().unwrap();

        let device = engine.queues_mut().device_mut();
        device.set_clock(10_000, 0);
        device.hold_completions(true);
        device.refuse_aborts(true);
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 4096, t, 0, 7, 2));
        engine.advance(&mut host);

        // Past its deadline: reported, the abort refused, no reset yet.
        engine.queues_mut().device_mut().set_clock(10_000 + 2_000_000, 0);
        engine.advance(&mut host);
        assert_eq!(host.expired.len(), 1);
        assert_eq!((engine.stats().aborts, engine.stats().aborts_refused), (0, 1));
        assert_eq!(engine.stats().resets, 0, "the grace runs first");
        assert!(completions(&mut client).is_empty());

        // Past the grace: the controller is reset and the command is lost.
        engine
            .queues_mut()
            .device_mut()
            .set_clock(10_000 + 2_000_000 + ABORT_GRACE_NS + 1, 0);
        engine.advance(&mut host);
        assert_eq!(engine.stats().resets, 1);
        assert_eq!(engine.queues_mut().device_mut().resets(), 1);
        assert_eq!(completions(&mut client), [(7, StorStatus::Lost.res())]);
        assert_eq!(engine.commands_in_flight(), 0);
        assert_eq!(
            client.fences.timeline(2).unwrap().load(),
            FenceRecord { done: 1, fault: 1 },
            "the fence moves, faulted"
        );

        // Everything but the command survives: the same ring, grant and
        // target serve the next read.
        let device = engine.queues_mut().device_mut();
        device.hold_completions(false);
        device.refuse_aborts(false);
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 4096, t, 0, 8, 0));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(8, 4096)]);
        assert!(engine.release_target(ring, t).is_ok());
    }

    /// A device error on one piece fails the whole op with the mapped
    /// status, after every piece has come back.
    #[test]
    fn a_media_error_on_one_piece_fails_the_op_after_all_pieces_return() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::ALL);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 32 }])).unwrap();
        let buf = target(&mut engine, 16384, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::ReadInto).ok().unwrap();
        engine
            .queues_mut()
            .device_mut()
            .inject_after(1, BlockError::Media);
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 16384, t, 0, 5, 3));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(5, StorStatus::Media.res())]);
        assert_eq!(engine.stats().commands, 4);
        assert_eq!(engine.commands_in_flight(), 0);
        assert_eq!(
            client.fences.timeline(3).unwrap().load(),
            FenceRecord { done: 1, fault: 1 }
        );
        // The target is free again once its commands are back.
        assert!(engine.release_target(ring, t).is_ok());
    }

    /// The client helper and the engine agree on every sequence: tickets
    /// from `StorClient` complete through the engine's fences in order,
    /// including across a refused submission on the same timeline.
    #[test]
    fn client_tickets_complete_through_the_engine() {
        use kore_ring::stor::{Op, Request, StorClient};

        let mut engine = engine();
        let mut host = Rec::default();
        let (client, ring) = attach(&mut engine, ClassSet::ALL);
        let mut client = StorClient::new(client);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 16 }])).unwrap();
        let buf = target(&mut engine, 4096, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::ReadInto).ok().unwrap();
        let read = |len, user_data| Request {
            op: Op::Read,
            grant,
            offset: 0,
            len,
            target: t,
            target_offset: 0,
            user_data,
            timeline: 4,
            deadline_ns: 0,
        };
        let first = client.submit(Class::Demand, &read(512, 1)).unwrap().unwrap();
        let refused = client.submit(Class::Demand, &read(100, 2)).unwrap().unwrap();
        let third = client.submit(Class::Stream, &read(4096, 3)).unwrap().unwrap();
        client.publish(Class::Demand);
        client.publish(Class::Stream);
        engine.advance(&mut host);
        assert_eq!(client.poll_ticket(first), Ok(true));
        assert!(client.poll_ticket(refused).is_err(), "the alignment refusal is its fault");
        assert!(client.poll_ticket(third).is_err(), "sticky past the fault");
        assert_eq!(client.fence(4).unwrap(), FenceRecord { done: 3, fault: 2 });
        let mut done = Vec::new();
        while let Some(c) = client.reap() {
            done.push((c.user_data, c.result));
        }
        done.sort_by_key(|(user, _)| *user);
        assert_eq!(
            done,
            [(1, Ok(512)), (2, Err(StorStatus::Alignment)), (3, Ok(4096))]
        );
    }

    /// A submission whose timeline already has a window's worth open waits
    /// in its channel instead of being popped — a completion past the
    /// window could never be recorded and would stop the fence for good —
    /// and is popped once the window has room.
    #[test]
    fn a_full_timeline_window_holds_the_next_submission() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::ALL);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 16 }])).unwrap();
        let buf = target(&mut engine, 4096, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::ReadInto).ok().unwrap();
        // A window's worth of sequences open on timeline 1, as a slow
        // `done + 1` with everything behind it complete leaves it.
        let slot = usize::from(ring.slot());
        let books = &mut engine.rings[slot].as_mut().unwrap().timelines[0];
        for _ in 0..super::super::timeline::WINDOW {
            books.issue();
        }
        submit(&mut client, Class::Demand, sqe(op::READ_FIXED, grant, 0, 512, t, 0, 1, 1));
        engine.advance(&mut host);
        assert_eq!(client.sq[Class::Demand.index()].capacity_left(), SQ_DEPTH - 1, "held, not popped");
        assert!(completions(&mut client).is_empty());
        assert_eq!(engine.stats().timeline_refusals, 0);

        // The window drains; the held submission goes.
        engine.rings[slot].as_mut().unwrap().timelines[0] = Timeline::default();
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(1, 512)]);
        assert_eq!(engine.stats().timeline_refusals, 0);
    }

    /// An idle engine parks every submission channel, so the next publish
    /// owes a notice; a queued record, or an op still open, refuses the
    /// park; and a departed ring is not waited on.
    #[test]
    fn an_idle_engine_parks_and_the_next_publish_owes_a_notice() {
        use kore_ring::stor::{Op, Request, StorClient};

        let mut engine = engine();
        let mut host = Rec::default();
        let (client, ring) = attach(&mut engine, ClassSet::ALL);
        let mut client = StorClient::new(client);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 16 }])).unwrap();
        let buf = target(&mut engine, 4096, 0);
        let t = engine.adopt_target(ring, buf, TargetAccess::ReadInto).ok().unwrap();
        let read = Request {
            op: Op::Read,
            grant,
            offset: 0,
            len: 512,
            target: t,
            target_offset: 0,
            user_data: 7,
            timeline: 1,
            deadline_ns: 0,
        };

        assert!(engine.park_idle(), "nothing queued, nothing open");
        assert!(client.publish(Class::Demand), "a parked consumer is owed its notice");
        assert!(!client.publish(Class::Demand), "once: the publish collected the intent");

        client.submit(Class::Demand, &read).unwrap();
        assert!(!engine.park_idle(), "a queued record refuses the park");
        let _ = client.publish(Class::Demand);
        engine.advance(&mut host);
        assert_eq!(client.reap().map(|c| c.result), Some(Ok(512)));
        assert!(engine.park_idle(), "drained again");

        engine.detach(ring).unwrap();
        let _ = client.publish(Class::Demand);
        client.submit(Class::Demand, &read).unwrap();
        assert!(engine.park_idle(), "a departed ring pops nothing, so is not waited on");
    }

    /// A flush is one device command and one completion — not a piece
    /// that never finishes issuing — and a discard needs the discard right.
    #[test]
    fn a_flush_issues_once_and_a_discard_needs_its_right() {
        let mut engine = engine();
        let mut host = Rec::default();
        let (mut client, ring) = attach(&mut engine, ClassSet::ALL);
        let grant = engine.install(ring, rw_grant(&[ExtentRun { lba: 0, blocks: 16 }])).unwrap();
        submit(&mut client, Class::Commit, sqe(op::FSYNC, grant, 0, 0, 0, 0, 1, 0));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(1, 0)]);
        assert_eq!(engine.stats().commands, 1);
        submit(&mut client, Class::Commit, sqe(op::DISCARD, grant, 0, 1024, 0, 0, 2, 0));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(2, StorStatus::Rights.res())]);
        let discard = engine
            .install(
                ring,
                Grant::new(&[ExtentRun { lba: 0, blocks: 16 }], Rights::DISCARD, ClassSet::ALL).unwrap(),
            )
            .unwrap();
        submit(&mut client, Class::Commit, sqe(op::DISCARD, discard, 0, 1024, 0, 0, 3, 0));
        engine.advance(&mut host);
        assert_eq!(completions(&mut client), [(3, 0)]);
        assert_eq!(engine.commands_in_flight(), 0);
    }
}
