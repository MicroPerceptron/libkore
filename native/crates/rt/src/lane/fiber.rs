//! **The fiber escape** (`res/SPMD.md` §13.3, G9) — the scalar hatch
//! for divergence too irreducible to mask: the worker spills its lanes
//! to W **stackful fibers**, each a real continuation on its own
//! audited arena strip, runs them round-robin to the reconvergence
//! point, and folds back into lockstep.  Fibers are the *divergence
//! mechanism of the lane tier*, not a third general concurrency model —
//! `#[kore_rt::task]` futures remain how ordinary code concurs, and the
//! stackless executor is untouched.
//!
//! # The quad word — one packed word of scheduler state
//!
//! ```text
//! 64-bit word:  [ lane3 ][ lane2 ][ lane1 ][ lane0 ]
//!     field:    [ SP:13 | status:3 ] × 4     ×8 granule → 64 KiB strips
//! ```
//!
//! A suspended SP is 8-aligned on the ABI — and never 16-aligned: the
//! seven-word frame (return address + six callee-saved) leaves
//! `sp ≡ 8 mod 16`, so the granule is ×8, not the ×16 a first glance
//! at "aligned stacks" suggests.  The three spare low bits carry the
//! per-lane status (five states).  `2^13 × 8 = 65536`: the field
//! covers a 64 KiB strip **exactly**, and
//! `fiber_base(lane) = deck_base + (lane << 16)` keeps
//! the hot resume a shift-and-add (rule 4: scalar resume; any
//! vectorized unpack is quad-wide tooling, none of which v1 ships).
//!
//! The four rules, held as spec:
//!
//! 1. **Context-on-own-stack** — the switch pushes the full
//!    callee-saved set onto the fiber's strip; the packed SP is the
//!    sole root ([`switch`](super::switch)).
//! 2. **Single owner; hand-off is a CAS** — only the owning worker
//!    mutates the word; [`QuadWord::try_claim_parked`] transfers an
//!    all-parked quad in one CAS.  The LAW lands fully host-race-proven;
//!    **no production caller steals in v1** (a stealing policy needs
//!    scheduler context this tier does not own, and a stolen quad also
//!    requires a quiescent room — stated, not hidden).
//! 3. **The yield-time audit, not the wrap** — the pack path
//!    range-checks `sp - base` against the strip and POISONS the status
//!    nibble on failure; a truncated overflow laundered into plausible
//!    bookkeeping is the one failure a diagnostic must not have.
//!    Between yields the GPU contract holds: lane bodies promise
//!    bounded frames.
//! 4. Audited strips are guardless by construction (hardware protection
//!    is page-granular), so rule 3 plus the bounded-frame promise IS
//!    the safety story — stated rather than implied.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::stack_arena::StripPlan;

/// The fiber window (`res/SPMD.md` §13.4): a pure VA reservation
/// directly above the SPMD stack window, proven disjoint from every
/// band at compile time beside [`BANDS`](crate::thread::BANDS).  16 MiB
/// = 4096 granules; a deck carves 64 (four 64 KiB quarters) in one run.
pub(crate) const FIBER_WINDOW_BASE: u64 = 0x7fff_e100_0000;

/// The window's granule capacity.
pub(crate) const FIBER_WINDOW_GRANULES: u32 = 4096;

/// **The process's fiber plan** — audited (guardless) strips; a deck owns
/// its carve, released at teardown.  The VA window starts unbacked:
/// [`carve_backed`] maps retained tails through the runtime's
/// [`MemSource`](crate::mem_source::MemSource) seam before publishing them to
/// this ledger.
pub(crate) static FIBERS: StripPlan<64> =
    StripPlan::unbacked(FIBER_WINDOW_BASE, FIBER_WINDOW_GRANULES);

/// Backing-grow quantum: two decks (512 KiB).  The metal room witness
/// established this region size while 2 MiB could refuse; two decks per
/// slow-path trap still amortises growth without turning the arena into
/// one fragile 16 MiB contiguous allocation.
#[cfg(target_os = "none")]
const BACKING_QUANTUM_GRANULES: u32 = 128;

/// Bounded waits when another worker is mapping the same plan's next
/// tail.  Growth is a rare session crossing; four 4 ms slices avoid
/// converting a wedged mapper into an unbounded boot delay.
#[cfg(target_os = "none")]
const BACKING_WAITS: usize = 4;

/// A fixed-address `MemSource` for one fiber-arena tail.
///
/// The retained `Region` capability stays in the initial process
/// thread's arena for the process lifetime; strip teardown recycles
/// geometry, while process-root teardown remains the sole backing
/// revoke.  This is what makes a released strip reusable without
/// remapping an already-live VA.
///
/// Only band slot 0 may invoke this source: a worker's result slot and
/// holdings die with that worker, so accepting a map there would publish
/// process-wide geometry over worker-lifetime pages.
#[cfg(target_os = "none")]
struct FiberArenaSource {
    base: u64,
    bytes: usize,
}

#[cfg(target_os = "none")]
// SAFETY: every successful map installs exactly the caller-named, page-aligned
// tail once; the retained Region capability remains in process-root slot land
// until process teardown, and `BackingClaim` serialises/publishes disjoint tails.
unsafe impl crate::mem_source::MemSource for FiberArenaSource {
    fn map_extent(&self, min_bytes: usize) -> Option<core::ptr::NonNull<[u8]>> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        if crate::entry::geometry::shard_index() != Some(0)
            || min_bytes == 0
            || min_bytes != self.bytes
            || self.base % crate::mem_source::ALIGN as u64 != 0
            || min_bytes % crate::stack_arena::GRANULE != 0
        {
            return None;
        }
        let self_space = crate::entry::geometry::self_space_slot()?;
        let result = crate::entry::geometry::acquire_cap_slot()?;
        let rights = kore_abi::cap::Rights::ACCESS
            | kore_abi::cap::Rights::MODIFY
            | kore_abi::cap::Rights::REVOKE;
        let mut b = SessionBuilder::new();
        if b.map_region(
            self_space,
            rights,
            self.base,
            min_bytes as u64,
            0,
            0,
            result,
        )
        .is_err()
        {
            crate::entry::geometry::release_cap_slot(result);
            return None;
        }
        let committed = match TrapTransport::new().commit_now(&b) {
            Ok(committed) => committed,
            Err(_) => {
                crate::entry::geometry::release_cap_slot(result);
                return None;
            }
        };
        if committed.status() != kore_abi::session::Status::Ok {
            crate::entry::geometry::release_cap_slot(result);
            return None;
        }

        // `result` deliberately remains reserved and bound in the
        // process root's arena.  Its Region owns this mapping until root
        // teardown; no Rust-side owner may release it while ledger
        // strips still name the pages.
        Some(core::ptr::NonNull::slice_from_raw_parts(
            core::ptr::NonNull::new(self.base as *mut u8)?,
            min_bytes,
        ))
    }
}

/// Wait for the exact in-progress generation to finish.  The map owner
/// always wakes after publishing or rolling back; timeout is a bounded
/// refusal if that owner is wedged.
#[cfg(target_os = "none")]
fn wait_for_backing(observed: u32, waits: &mut usize) -> Option<()> {
    use core::time::Duration;

    if *waits == BACKING_WAITS {
        return None;
    }
    *waits += 1;
    if matches!(
        crate::futex::wait(
            FIBERS.backing_word(),
            observed,
            Some(Duration::from_millis(4)),
        ),
        crate::futex::WaitOutcome::Unsupported
    ) {
        return None;
    }
    Some(())
}

/// Commit one already-reserved tail through the root-owned source.
#[cfg(target_os = "none")]
fn back_claim(claim: crate::stack_arena::BackingClaim<'_, 64>) -> Option<()> {
    let bytes = (claim.granules() as usize).checked_mul(crate::stack_arena::GRANULE)?;
    let source = FiberArenaSource {
        base: claim.base(),
        bytes,
    };
    claim.back_with(&source)
}

/// Add one retained tail.  Mapping authority belongs to the initial
/// process thread; workers can consume its published pages but cannot
/// make worker-lifetime `Region`s look process-owned.
#[cfg(target_os = "none")]
fn grow_backing(needed: u32) -> Option<()> {
    if crate::entry::geometry::shard_index() != Some(0) {
        return None;
    }
    let claim = FIBERS
        .reserve_backing(needed, BACKING_QUANTUM_GRANULES)
        .ok()?;
    back_claim(claim)
}

/// Carve only from live pages, growing the mapped prefix on a miss.
///
/// The allocation fast path remains exactly the lock-free ledger scan.
/// The process owner takes a rare affine growth claim; workers park on
/// its per-plan word for a bounded interval and otherwise refuse a miss.
#[cfg(target_os = "none")]
fn carve_backed(granules: u32) -> Option<crate::stack_arena::Strip> {
    let mut waits = 0usize;
    loop {
        if let Some(strip) = FIBERS.carve(granules) {
            return Some(strip);
        }
        if crate::entry::geometry::shard_index() != Some(0) {
            let observed = FIBERS.backing_in_progress()?;
            wait_for_backing(observed, &mut waits)?;
            continue;
        }
        match FIBERS.reserve_backing(granules, BACKING_QUANTUM_GRANULES) {
            Ok(claim) => {
                back_claim(claim)?;
                waits = 0;
            }
            Err(crate::stack_arena::BackingRefused::Busy { observed }) => {
                wait_for_backing(observed, &mut waits)?;
            }
            Err(
                crate::stack_arena::BackingRefused::Full
                | crate::stack_arena::BackingRefused::Invalid,
            ) => return None,
        }
    }
}

/// Granules per deck: four 64 KiB fiber quarters.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
pub(crate) const DECK_GRANULES: u32 = 64;

/// Bytes per fiber quarter — what the 13-bit ×8 field covers exactly.
pub(crate) const FIBER_BYTES: u64 = 64 * 1024;

/// A lane's status field (3 bits of the packed lane u16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FiberStatus {
    /// No continuation — a fresh or drained slot.
    Vacant = 0,
    /// Running (its packed SP is stale by definition — the live SP is a
    /// register; rule 2's atomic snapshot covers *parked* lanes only).
    Live = 1,
    /// Suspended; the packed SP is the continuation's sole root.
    Parked = 2,
    /// Ran to its reconvergence point.
    Done = 3,
    /// Failed the yield audit — never resumed, never stolen.
    Poisoned = 4,
}

impl FiberStatus {
    const fn of(bits: u8) -> Self {
        match bits & 0x7 {
            0 => Self::Vacant,
            1 => Self::Live,
            2 => Self::Parked,
            3 => Self::Done,
            _ => Self::Poisoned,
        }
    }
}

/// A parked quad's four continuations, yielded whole by a successful
/// steal claim — the stacks travel for free as shared memory.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub struct StolenQuad {
    /// The four packed `(sp_off8, status)` fields, lane order.
    pub fields: [u16; 4],
}

/// Why a steal claim refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StealRefused {
    /// Some lane is Live, Poisoned, or Vacant — not an all-parked quad.
    NotAllParked,
    /// The word moved under the CAS — the owner (or a rival) got there
    /// first.
    Raced,
}

/// The packed quad TCB — rule 2's single owner mutates; rule 2's CAS
/// transfers.
#[repr(transparent)]
pub struct QuadWord(AtomicU64);

impl QuadWord {
    /// An all-vacant quad.
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Pack one lane field.  The SP field is ×8-granular: a suspended
    /// frame is 8-aligned but NEVER 16-aligned (seven pushed words —
    /// the return address plus six callee-saved — leave `sp ≡ 8 mod
    /// 16`), so a ×16 field could not represent a single real park.
    /// 13 bits × 8 covers the 64 KiB quarter exactly; the five
    /// statuses fit the remaining 3.
    pub const fn pack_field(sp_off8: u16, status: FiberStatus) -> u16 {
        (sp_off8 << 3) | status as u16
    }

    /// One lane's `(sp_off8, status)` out of a word snapshot.
    pub const fn field(word: u64, lane: usize) -> (u16, FiberStatus) {
        let f = (word >> (lane * 16)) as u16;
        (f >> 3, FiberStatus::of(f as u8))
    }

    /// The current word — an atomic snapshot, consistent only for
    /// parked lanes (rule 2).
    pub fn load(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    /// Owner-only field store.
    fn set_field(&self, lane: usize, field: u16) {
        let shift = lane * 16;
        let mut word = self.0.load(Ordering::Relaxed);
        loop {
            let next = (word & !(0xFFFFu64 << shift)) | ((field as u64) << shift);
            match self
                .0
                .compare_exchange_weak(word, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return,
                Err(seen) => word = seen,
            }
        }
    }

    /// Owner-only status store, SP field preserved.
    pub fn set_status(&self, lane: usize, status: FiberStatus) {
        let (sp, _) = Self::field(self.load(), lane);
        self.set_field(lane, Self::pack_field(sp, status));
    }

    /// **THE yield audit** (rule 3): park `lane` at `sp` against its
    /// quarter's `base`.  A misaligned or out-of-strip SP poisons the
    /// nibble and refuses — the overflow is never truncated into
    /// plausible bookkeeping.
    pub fn park(&self, lane: usize, sp: u64, base: u64) -> Result<(), ()> {
        let off = sp.wrapping_sub(base);
        if off % 8 != 0 || off >= FIBER_BYTES {
            self.set_field(lane, Self::pack_field(0, FiberStatus::Poisoned));
            return Err(());
        }
        self.set_field(lane, Self::pack_field((off / 8) as u16, FiberStatus::Parked));
        Ok(())
    }

    /// A parked lane's absolute SP (`base + off×8`), or `None` for any
    /// other status — a poisoned or done lane never resumes.
    pub fn resume_sp(&self, lane: usize, base: u64) -> Option<u64> {
        let (sp, status) = Self::field(self.load(), lane);
        (status == FiberStatus::Parked).then(|| base + (sp as u64) * 8)
    }

    /// **Quad-granular stealing** (rule 2's law): one CAS transfers an
    /// ALL-PARKED quad to the claimer, leaving the word vacant.  Any
    /// Live/Poisoned/Vacant lane refuses; a racing owner refuses.
    pub fn try_claim_parked(&self) -> Result<StolenQuad, StealRefused> {
        let word = self.load();
        let mut fields = [0u16; 4];
        for (lane, field) in fields.iter_mut().enumerate() {
            let (sp, status) = Self::field(word, lane);
            if status != FiberStatus::Parked {
                return Err(StealRefused::NotAllParked);
            }
            *field = Self::pack_field(sp, FiberStatus::Parked);
        }
        match self
            .0
            .compare_exchange(word, 0, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Ok(StolenQuad { fields }),
            Err(_) => Err(StealRefused::Raced),
        }
    }
}

impl Default for QuadWord {
    fn default() -> Self {
        Self::new()
    }
}

/// Why an escape (or its provisioning) refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiberFault {
    /// No proven switch on this target, no strip to carve, or the
    /// mapping session refused — the honest floor.
    Unsupported,
    /// A lane failed the yield audit; it was never resumed and the
    /// escape reports rather than reconverging over a hole.
    Poisoned {
        /// The first poisoned lane.
        lane: u32,
    },
}

/// A fiber body: runs on its own strip, yields through
/// [`FiberCtx::yield_now`], returns at its reconvergence point.
pub type FiberFn = fn(&mut FiberCtx);

/// What one deck resume came back with.
#[cfg(target_os = "none")]
enum Event {
    /// The fiber yielded; its stack pointer at the switch is in
    /// [`EscapeSession::fiber_sp`] (the switch writes it there — a
    /// fiber-stack local would be invisible to the deck).
    Yielded,
    /// The fiber's body returned — it is done.
    Finished,
}

/// The live half of one `escape` — on the escape's own frame, addressed
/// by raw pointer from the fibers (single worker, single core: the
/// deck and its fibers alternate on one instruction stream, so the
/// aliasing is a schedule, never a race).
#[cfg(target_os = "none")]
struct EscapeSession {
    /// Where a yielding fiber switches back to (the deck's saved SP).
    deck_sp: u64,
    /// The yielding fiber's saved SP — written BY the switch itself,
    /// audited by the deck on the way back.
    fiber_sp: u64,
    /// The lane currently running.
    lane: usize,
    /// The body every lane runs.
    body: FiberFn,
    /// The resume's outcome, written by the fiber side.
    event: Event,
    /// The deck, for the ctx's mail delegates.
    deck: *mut FiberDeck,
}

/// One worker's fiber deck (`res/SPMD.md` §13.3): the quad word, the
/// carved quarters, and the worker's [`Room`](crate::room::Room).
pub struct FiberDeck {
    quad: QuadWord,
    base: u64,
    room: crate::room::Room,
    #[cfg(target_os = "none")]
    strip: Option<crate::stack_arena::Strip>,
}

impl FiberDeck {
    /// This deck's bank name — the terminal reports' `bank=` field.
    pub fn bank_name() -> &'static str {
        crate::room::Room::bank_name()
    }

    /// Borrow the room — the microbench's measurement surface; fiber
    /// bodies reach it through [`FiberCtx`] instead.
    pub fn room(&mut self) -> &mut crate::room::Room {
        &mut self.room
    }

    /// A lane quarter's base.
    fn quarter_base(&self, lane: usize) -> u64 {
        self.base + (lane as u64) * FIBER_BYTES
    }

    /// Ensure room for `decks` simultaneous fiber quads before workers
    /// launch.  Only the initial process thread can succeed: it retains
    /// every backing `Region` for the process lifetime.  A worker may
    /// carve this capacity but may never grow it under a worker-lifetime
    /// capability root.
    #[cfg(target_os = "none")]
    pub fn prewarm(decks: u32) -> bool {
        if decks == 0 || crate::entry::geometry::shard_index() != Some(0) {
            return false;
        }
        let Some(target) = decks.checked_mul(DECK_GRANULES) else {
            return false;
        };
        if target > FIBER_WINDOW_GRANULES {
            return false;
        }
        while FIBERS.backed() < target {
            let remaining = target - FIBERS.backed();
            if grow_backing(remaining.min(BACKING_QUANTUM_GRANULES)).is_none() {
                return false;
            }
        }
        true
    }

    /// No process-owned mapping authority exists off metal.
    #[cfg(not(target_os = "none"))]
    pub fn prewarm(_decks: u32) -> bool {
        false
    }

    /// Provision a deck on the metal target: one audited carve (four
    /// 64 KiB quarters) from the retained pool, growing that pool on
    /// the process-root slow path when necessary.  `None` — no proven
    /// switch, no live strip, or a refused grow — is the honest floor:
    /// lockstep code simply keeps masking.
    #[cfg(target_os = "none")]
    pub fn provision() -> Option<FiberDeck> {
        use super::switch::{FiberSwitch, Switch};
        if !<Switch as FiberSwitch>::available() {
            return None;
        }
        let strip = carve_backed(DECK_GRANULES)?;
        Some(FiberDeck {
            quad: QuadWord::new(),
            base: strip.base(),
            room: crate::room::Room::new(),
            strip: Some(strip),
        })
    }

    /// Off the metal targets: the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn provision() -> Option<FiberDeck> {
        None
    }

    /// **The escape** (§13.2): spill the mask's active lanes of one
    /// quad group (`group` = 0 or 1; a W=8 worker escapes its two
    /// groups sequentially) to fibers, run them round-robin to their
    /// reconvergence points, fold back.  The deck audits every yield
    /// (rule 3); a poisoned lane is never resumed and the escape
    /// reports it.
    #[cfg(target_os = "none")]
    pub fn escape(
        &mut self,
        ls: &super::Lockstep,
        group: usize,
        body: FiberFn,
    ) -> Result<(), FiberFault> {
        use super::switch::{FiberSwitch, Switch};
        if !<Switch as FiberSwitch>::available() {
            return Err(FiberFault::Unsupported);
        }
        let mut session = EscapeSession {
            deck_sp: 0,
            fiber_sp: 0,
            lane: 0,
            body,
            event: Event::Finished,
            deck: self as *mut _,
        };
        let arg = (&mut session) as *mut EscapeSession as usize;
        // Seed every active lane of the group with a fresh suspended
        // frame; the seed SP passes the audit by construction.
        let mut seeded = false;
        for lane in 0..4usize {
            let global = group * 4 + lane;
            if global < super::W && ls.active().test(global) {
                let top = self.quarter_base(lane) + FIBER_BYTES;
                // SAFETY: the quarter is this deck's mapped, exclusively
                // owned memory; `fiber_entry` never returns.
                let sp = unsafe { Switch::seed(top, fiber_entry, arg) };
                let _ = self.quad.park(lane, sp, self.quarter_base(lane));
                seeded = true;
            }
        }
        if !seeded {
            return Ok(());
        }
        // Round-robin to reconvergence: resume each parked lane in
        // order until a full sweep finds none.
        loop {
            let mut progressed = false;
            for lane in 0..4usize {
                let Some(sp) = self.quad.resume_sp(lane, self.quarter_base(lane)) else {
                    continue;
                };
                self.quad.set_status(lane, FiberStatus::Live);
                session.lane = lane;
                // SAFETY: `sp` came from this deck's own seed/park on a
                // live quarter; `deck_sp` is re-adopted by the fiber's
                // next yield on this same instruction stream.
                unsafe { Switch::switch(&mut session.deck_sp as *mut u64, sp) };
                match session.event {
                    Event::Yielded => {
                        // THE audit (rule 3) — a bad SP poisons here.
                        let _ =
                            self.quad.park(lane, session.fiber_sp, self.quarter_base(lane));
                    }
                    Event::Finished => self.quad.set_status(lane, FiberStatus::Done),
                }
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        // Reconverged — unless a lane poisoned, which is a report, not
        // a silent hole in the fold.
        for lane in 0..4usize {
            if QuadWord::field(self.quad.load(), lane).1 == FiberStatus::Poisoned {
                return Err(FiberFault::Poisoned { lane: lane as u32 });
            }
        }
        Ok(())
    }

    /// Off the metal targets: the escape cannot run — same refusal as
    /// [`provision`](Self::provision).
    #[cfg(not(target_os = "none"))]
    pub fn escape(
        &mut self,
        _ls: &super::Lockstep,
        _group: usize,
        _body: FiberFn,
    ) -> Result<(), FiberFault> {
        Err(FiberFault::Unsupported)
    }
}

#[cfg(target_os = "none")]
impl Drop for FiberDeck {
    fn drop(&mut self) {
        // Only geometry returns.  The pool's retained Region mapping
        // stays live until process teardown, so this exact strip can be
        // carved again without an overlapping remap.
        if let Some(strip) = self.strip.take() {
            FIBERS.release(strip);
        }
    }
}

/// Carve and map a bench scratch from the fiber window (the `spmd room`
/// microbench's eviction buffer): arena-carved so no 2 MiB static ever
/// bloats an image's load segments — a fat `.bss` member measurably
/// drags the archive loader's zero-fill at boot.  The strip is
/// deliberately leaked (a bench tool exits; the window is per-process
/// address space).  Returns the mapped base.
#[cfg(target_os = "none")]
pub fn map_bench_scratch(bytes: u64) -> Option<u64> {
    let granules = bytes.div_ceil(crate::stack_arena::GRANULE as u64) as u32;
    let strip = carve_backed(granules)?;
    let base = strip.base();
    core::mem::forget(strip);
    Some(base)
}

/// No target-side mapping source exists off metal.
#[cfg(not(target_os = "none"))]
pub fn map_bench_scratch(_bytes: u64) -> Option<u64> {
    None
}

/// A running fiber's surface: its lane, the yield, and the quad mail
/// (§13.6 — the deck hands this only to the fiber it just resumed, so
/// only the RUNNING fiber ever touches the room).
pub struct FiberCtx {
    #[cfg(target_os = "none")]
    session: *mut EscapeSession,
    lane: usize,
}

impl FiberCtx {
    /// This fiber's quad lane index.
    pub fn lane(&self) -> u32 {
        self.lane as u32
    }

    /// This fiber's own quad lane, for the mail surface.
    fn quad_lane(&self) -> crate::room::QuadLane {
        crate::room::QuadLane::of(self.lane).expect("a fiber lane is a quad lane")
    }

    /// Post into a sibling's mailbox (§13.6 — the ITC): the displaced
    /// message returns.  Only the running fiber holds a `FiberCtx`, so
    /// only the running fiber ever touches the room.
    #[cfg(target_os = "none")]
    pub fn post(
        &mut self,
        dst: crate::room::QuadLane,
        msg: core::num::NonZeroU64,
    ) -> Option<core::num::NonZeroU64> {
        // SAFETY: one stream; the deck outlives the escape.
        unsafe { (*(*self.session).deck).room.post(dst, msg) }
    }

    /// Read-and-clear OWN mailbox.
    #[cfg(target_os = "none")]
    pub fn take(&mut self) -> Option<core::num::NonZeroU64> {
        let me = self.quad_lane();
        // SAFETY: as `post`.
        unsafe { (*(*self.session).deck).room.take(me) }
    }

    /// Publish to the broadcast slot.
    #[cfg(target_os = "none")]
    pub fn publish(&mut self, value: u64) {
        // SAFETY: as `post`.
        unsafe { (*(*self.session).deck).room.publish(value) }
    }

    /// Read the broadcast slot.
    #[cfg(target_os = "none")]
    pub fn snoop(&self) -> u64 {
        // SAFETY: as `post`.
        unsafe { (*(*self.session).deck).room.snoop() }
    }

    /// Yield to the deck; resumes when the round-robin comes back.
    #[cfg(target_os = "none")]
    pub fn yield_now(&mut self) {
        use super::switch::{FiberSwitch, Switch};
        // SAFETY: single instruction stream — the session outlives the
        // escape and the deck resumes this exact save.  The switch
        // writes this fiber's SP into the SESSION (never a fiber-stack
        // local, which the deck could not see), and the deck audits it
        // on the way back.
        unsafe {
            let session = &mut *self.session;
            session.event = Event::Yielded;
            Switch::switch(&mut session.fiber_sp as *mut u64, session.deck_sp);
        }
    }

    /// Off metal a fiber never runs; the surface still compiles.
    #[cfg(not(target_os = "none"))]
    pub fn yield_now(&mut self) {}

    /// Off metal — the surface compiles, a fiber never runs.
    #[cfg(not(target_os = "none"))]
    pub fn post(
        &mut self,
        _dst: crate::room::QuadLane,
        _msg: core::num::NonZeroU64,
    ) -> Option<core::num::NonZeroU64> {
        None
    }

    /// Off metal — as [`post`](Self::post).
    #[cfg(not(target_os = "none"))]
    pub fn take(&mut self) -> Option<core::num::NonZeroU64> {
        None
    }

    /// Off metal — as [`post`](Self::post).
    #[cfg(not(target_os = "none"))]
    pub fn publish(&mut self, _value: u64) {}

    /// Off metal — as [`post`](Self::post).
    #[cfg(not(target_os = "none"))]
    pub fn snoop(&self) -> u64 {
        0
    }
}

/// The fiber entry: run the body, mark done, return to the deck
/// forever.
#[cfg(target_os = "none")]
extern "C" fn fiber_entry(arg: usize) -> ! {
    use super::switch::{FiberSwitch, Switch};
    // SAFETY: the deck seeded `arg` as its live session; one stream.
    let session = unsafe { &mut *(arg as *mut EscapeSession) };
    let lane = session.lane;
    let mut ctx = FiberCtx {
        session: session as *mut _,
        lane,
    };
    (session.body)(&mut ctx);
    session.event = Event::Finished;
    let mut scratch = 0u64;
    loop {
        // SAFETY: as `yield_now`; a Done fiber is never resumed, and if
        // it ever were, looping the switch keeps it inert.
        unsafe { Switch::switch(&mut scratch as *mut u64, session.deck_sp) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packing round-trips, ×8 scaling, field isolation.
    #[test]
    fn packing_laws() {
        let q = QuadWord::new();
        q.set_field(2, QuadWord::pack_field(0xABC, FiberStatus::Parked));
        let (sp, status) = QuadWord::field(q.load(), 2);
        assert_eq!((sp, status), (0xABC, FiberStatus::Parked));
        for lane in [0usize, 1, 3] {
            assert_eq!(
                QuadWord::field(q.load(), lane),
                (0, FiberStatus::Vacant),
                "lane {lane} untouched"
            );
        }
        // The max in-range offset packs and scales back exactly —
        // and the REAL frame parity (`≡ 8 mod 16`) is representable:
        // the seed itself parks at `top − 56`.
        let base = 0x10_0000u64;
        q.park(1, base + FIBER_BYTES - 56, base).expect("in range");
        assert_eq!(q.resume_sp(1, base), Some(base + FIBER_BYTES - 56));
        q.park(1, base + FIBER_BYTES - 8, base).expect("in range");
        assert_eq!(q.resume_sp(1, base), Some(base + FIBER_BYTES - 8));
    }

    /// The audit poisons — misalignment and overflow — and a poisoned
    /// lane never resumes.  The overflow is REFUSED, not truncated.
    #[test]
    fn the_audit_poisons_and_never_launders() {
        let base = 0x10_0000u64;
        let q = QuadWord::new();
        assert!(q.park(0, base + 4, base).is_err(), "misaligned SP poisons");
        assert_eq!(QuadWord::field(q.load(), 0).1, FiberStatus::Poisoned);
        let q = QuadWord::new();
        assert!(
            q.park(0, base + FIBER_BYTES, base).is_err(),
            "one past the strip poisons"
        );
        assert_eq!(q.resume_sp(0, base), None, "poisoned lanes never resume");
        // The would-have-truncated shape: an overflow whose low 13 (+3)
        // bits alias an in-range offset must still refuse.
        let q = QuadWord::new();
        assert!(q.park(0, base + FIBER_BYTES + 32, base).is_err());
    }

    /// The steal law: all-parked transfers once; Live/Vacant/Poisoned
    /// refuse; racing claimers get exactly one winner.
    #[test]
    fn steal_is_all_parked_exactly_once() {
        let base = 0u64;
        let q = QuadWord::new();
        for lane in 0..4 {
            q.park(lane, base + 32 * (lane as u64 + 1), base).expect("parks");
        }
        q.set_status(3, FiberStatus::Live);
        assert!(matches!(q.try_claim_parked(), Err(StealRefused::NotAllParked)));
        q.set_status(3, FiberStatus::Parked);

        let q = std::sync::Arc::new(q);
        let winners = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = std::vec::Vec::new();
        for _ in 0..8 {
            let q = q.clone();
            let winners = winners.clone();
            handles.push(std::thread::spawn(move || {
                if q.try_claim_parked().is_ok() {
                    winners.fetch_add(1, Ordering::SeqCst);
                }
            }));
        }
        for h in handles {
            h.join().expect("no panics");
        }
        assert_eq!(winners.load(Ordering::SeqCst), 1, "exactly one claimer wins");
        assert_eq!(
            QuadWord::field(q.load(), 0).1,
            FiberStatus::Vacant,
            "the claimed word is vacant"
        );
        assert!(matches!(q.try_claim_parked(), Err(StealRefused::NotAllParked)));
    }
}
