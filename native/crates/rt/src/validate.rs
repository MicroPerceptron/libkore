//! **Metal self-validation** — compiled only under the `validate`
//! feature, run by a binary whose gate demands the witness line.
//!
//! The machinery this crate ships is host-proven only to the depth a
//! host can see: orderings over mocks, staging, codecs, typed refusals
//! (`no mock world` — the crate docs).  What a host can never see is
//! the metal composition — a real founding, real threads on real CPUs
//! meeting at a barrier, a real join.  Each exercise here IS that
//! composition, as a function: it runs the full path and returns a
//! **typed report**, and it never prints — the console belongs to the
//! consuming binary, which turns the report into the one witness line
//! its boot/terminal gate greps (`app-store: seeded 20/20 tree-load: ok`
//! is the pattern).
//!
//! Off by default: the exercise statics below (a region ctx, its
//! counters) have no business in a production image.

use core::sync::atomic::{AtomicU32, Ordering};

use crate::barrier::Arrival;
use crate::spmd::{SpmdCtx, spawn_spmd};

/// Workers the SPMD exercise launches.  Small on purpose: the
/// validating binary has threads of its own, and the exercise must
/// fit any band plan (`WORKERS + its host's threads ≤ MAX_THREADS`),
/// not saturate one.
const WORKERS: usize = 4;

/// Barrier phases each worker runs — enough that a reuse bug (a phase
/// released early, a generation skipped) has room to show, small
/// enough to cost nothing.
const PHASES: usize = 3;

/// The exercise's launch anchor — a linear one-tile-per-worker grid;
/// this exercise witnesses the sync fabric, not the striding.
static CTX: SpmdCtx = SpmdCtx::new(WORKERS as u32);

/// Arrivals seen per phase; each must equal [`WORKERS`] after the join
/// — the witness that every worker really ran every phase.
static ARRIVALS: [AtomicU32; PHASES] = [const { AtomicU32::new(0) }; PHASES];

/// Leader elections seen per phase; each must equal exactly 1 — the
/// witness that the barrier's election is unique per release.
static LEADERS: [AtomicU32; PHASES] = [const { AtomicU32::new(0) }; PHASES];

static HINT_VALID: [AtomicU32; WORKERS] = [const { AtomicU32::new(0) }; WORKERS];
static HINT_ALIAS: [core::sync::atomic::AtomicUsize; WORKERS] =
    [const { core::sync::atomic::AtomicUsize::new(0) }; WORKERS];

/// How many [`join_all`](crate::spmd::Spmd::join_all) refusals the
/// exercise absorbs before calling the join itself broken.  Transient
/// (`Retryable`/`NoSlots`) refusals on freshly-exited threads settle in
/// a turn or two; a witness that spins forever would hang its gate.
const JOIN_ATTEMPTS: usize = 16;

/// What the SPMD exercise proved, for the consumer's witness line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpmdWitness {
    /// Workers launched, barriered, and joined.
    pub workers: u32,
    /// Barrier phases every worker completed.
    pub phases: u32,
    /// Whether the doorbell star stood for the region (§3) — `false`
    /// is the degrade ladder's honest floor, not a failure.
    pub star: bool,
}

/// Where the SPMD exercise failed — one variant per witnessed law, so
/// the gate line can say *what* broke, not just that something did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpmdFault {
    /// `spawn_spmd` refused: a host build, an exhausted band plan, or
    /// a second call (the exercise is one-shot — its gate is spent).
    Launch,
    /// The join kept refusing past [`JOIN_ATTEMPTS`].
    Join,
    /// A worker exited nonzero or statusless.
    Status,
    /// A phase saw the wrong arrival count: some worker skipped it.
    Arrivals { phase: u32, seen: u32 },
    /// A phase elected the wrong number of leaders.
    Leaders { phase: u32, seen: u32 },
    /// Missing/invalid task advice or a shared-space sibling's arena alias.
    ExecutionHint,
}

/// Every worker's body: count in, meet the barrier, count the crown.
fn body(ctx: &'static SpmdCtx, worker: u32) {
    #[cfg(target_os = "none")]
    if let Some(va) =
        crate::entry::geometry::slot_record_va(kore_abi::session::SlotRef::from_wire(0))
    {
        HINT_ALIAS[worker as usize].store(va, Ordering::Relaxed);
    }
    for phase in 0..PHASES {
        if crate::entry::execution_hint().is_some() {
            HINT_VALID[worker as usize].fetch_add(1, Ordering::Relaxed);
        }
        ARRIVALS[phase].fetch_add(1, Ordering::SeqCst);
        if ctx.barrier().arrive_and_wait() == Arrival::Leader {
            LEADERS[phase].fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// **The SPMD exercise** (`res/SPMD.md` §11, the G1/G2
/// metal proof): launch [`WORKERS`] real threads through the gate, run
/// [`PHASES`] barrier phases across them, join them all, and check the
/// laws — everyone ran every phase, exactly one leader per phase,
/// every exit clean.
///
/// One-shot per boot: the launch spends the ctx's gate, so a second
/// call reports [`SpmdFault::Launch`] rather than re-running.
pub fn spmd() -> Result<SpmdWitness, SpmdFault> {
    let mut region = spawn_spmd::<WORKERS>(&CTX, body).ok_or(SpmdFault::Launch)?;
    let star = region.star_active();
    let statuses = 'join: {
        for _ in 0..JOIN_ATTEMPTS {
            match region.join_all() {
                Ok(statuses) => break 'join statuses,
                Err(refused) => region = refused.spmd,
            }
        }
        return Err(SpmdFault::Join);
    };
    if statuses.iter().any(|s| *s != Some(0)) {
        return Err(SpmdFault::Status);
    }
    for i in 0..WORKERS {
        let alias = HINT_ALIAS[i].load(Ordering::Relaxed);
        if HINT_VALID[i].load(Ordering::Relaxed) != PHASES as u32
            || alias == 0
            || (0..i).any(|j| HINT_ALIAS[j].load(Ordering::Relaxed) == alias)
        {
            return Err(SpmdFault::ExecutionHint);
        }
    }
    for phase in 0..PHASES {
        let seen = ARRIVALS[phase].load(Ordering::SeqCst);
        if seen != WORKERS as u32 {
            return Err(SpmdFault::Arrivals {
                phase: phase as u32,
                seen,
            });
        }
        let seen = LEADERS[phase].load(Ordering::SeqCst);
        if seen != 1 {
            return Err(SpmdFault::Leaders {
                phase: phase as u32,
                seen,
            });
        }
    }
    Ok(SpmdWitness {
        workers: WORKERS as u32,
        phases: PHASES as u32,
        star,
    })
}

/// What the register-room exercise proved (`res/SPMD.md` §13.6, R7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoomWitness {
    /// Kernel round-trips the pattern survived — each one crossed the
    /// ExtState choke-point (`XRSTOR` at enter_user) at least twice.
    pub round_trips: u32,
}

/// Where the register-room exercise failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoomFault {
    /// Not x86_64 — the bank does not exist here; the strip word is
    /// this target's room (§13.6 rule 4's floor).
    Unsupported,
    /// No park ever succeeded, so the pattern never crossed the kernel
    /// — a pass would be vacuous, and the host suite proves exactly
    /// this arm (parks refuse off the metal).
    NoRoundTrip,
    /// A slot came back wrong: component 0 did not round-trip.
    Corrupt { slot: u32, got: u64 },
}

/// The MM bank accessors — the witness's tenants (the production room
/// lives in [`crate::room`]).  `mmx_reg` is clobber-only in `asm!`, so
/// the mnemonics name the registers and the clobber list declares the
/// writes; a fixed-register `movq` is baseline x86-64 encoding, so no
/// feature re-admission exists or is needed (rustc's `target_feature`
/// whitelist has no `mmx` entry — the crate-wide `-mmx` spec
/// reservation bars only the ALLOCATOR from the bank, §13.6 rule 3).
#[cfg(target_arch = "x86_64")]
mod mm {
    /// Fill all eight MM registers.
    ///
    /// # Safety
    /// Requires MMX (baseline on every x86_64; the target feature is
    /// profile-disabled, not absent).  No x87 arithmetic may run
    /// between MMX use and any future x87 use — vacuous under R7's
    /// forbid.
    pub(super) unsafe fn write(v: &[u64; 8]) {
        unsafe {
            core::arch::asm!(
                "movq mm0, {a}", "movq mm1, {b}", "movq mm2, {c}",
                "movq mm3, {d}", "movq mm4, {e}", "movq mm5, {f}",
                "movq mm6, {g}", "movq mm7, {h}",
                a = in(reg) v[0], b = in(reg) v[1], c = in(reg) v[2],
                d = in(reg) v[3], e = in(reg) v[4], f = in(reg) v[5],
                g = in(reg) v[6], h = in(reg) v[7],
                out("mm0") _, out("mm1") _, out("mm2") _, out("mm3") _,
                out("mm4") _, out("mm5") _, out("mm6") _, out("mm7") _,
                options(nomem, nostack, preserves_flags)
            );
        }
    }

    /// Read all eight MM registers back.
    ///
    /// # Safety
    /// As [`write`]; reads only.
    pub(super) unsafe fn read() -> [u64; 8] {
        let (mut a, mut b, mut c, mut d): (u64, u64, u64, u64);
        let (mut e, mut f, mut g, mut h): (u64, u64, u64, u64);
        unsafe {
            core::arch::asm!(
                "movq {a}, mm0", "movq {b}, mm1", "movq {c}, mm2",
                "movq {d}, mm3", "movq {e}, mm4", "movq {f}, mm5",
                "movq {g}, mm6", "movq {h}, mm7",
                a = out(reg) a, b = out(reg) b, c = out(reg) c,
                d = out(reg) d, e = out(reg) e, f = out(reg) f,
                g = out(reg) g, h = out(reg) h,
                options(nomem, nostack, preserves_flags)
            );
        }
        [a, b, c, d, e, f, g, h]
    }
}

/// **The register-room exercise** (R7's proof instrument): park a
/// pattern in the x87/MMX bank, cross the kernel repeatedly — each
/// timed park is a descheduling plus at least two trips through the
/// ExtState choke-point — and read the bank back.  A pass is the
/// metal fact that component 0 rides the per-task round-trip, i.e.
/// that the register room needs **no kernel change at all**.
///
/// A pass requires at least one real park: a build whose parks refuse
/// (every host) reports [`RoomFault::NoRoundTrip`] rather than passing
/// vacuously — the same honesty rule as every witness here.
#[cfg(target_arch = "x86_64")]
pub fn register_room() -> Result<RoomWitness, RoomFault> {
    use core::sync::atomic::AtomicU32;
    use core::time::Duration;

    /// Parks to attempt; a few quanta's worth of descheduling.
    const PARKS: u32 = 8;

    // A pattern no init-state or junk image reproduces: slot-salted,
    // bit-dense, distinct per register.
    let mut pattern = [0u64; 8];
    let mut i = 0u64;
    while i < 8 {
        pattern[i as usize] = 0x9e37_79b9_7f4a_7c15u64
            .wrapping_mul(i + 1)
            .rotate_left((i * 11) as u32)
            | 1 << i;
        i += 1;
    }

    // SAFETY: x86_64 has the MM bank architecturally; R7's forbid bans
    // x87 arithmetic everywhere, so tag-word state never matters.
    unsafe { mm::write(&pattern) };

    static NEVER: AtomicU32 = AtomicU32::new(0);
    let mut round_trips = 0u32;
    for _ in 0..PARKS {
        // A timed park on a word nobody stores: the timeout is the
        // point — deschedule, let the core run someone else, come back
        // through the choke-point.
        match crate::futex::wait(&NEVER, 0, Some(Duration::from_millis(1))) {
            crate::futex::WaitOutcome::TimedOut | crate::futex::WaitOutcome::Woken => {
                round_trips += 1;
            }
            _ => {}
        }
    }
    if round_trips == 0 {
        return Err(RoomFault::NoRoundTrip);
    }

    // SAFETY: as above.
    let got = unsafe { mm::read() };
    for (slot, (want, have)) in pattern.iter().zip(got.iter()).enumerate() {
        if want != have {
            return Err(RoomFault::Corrupt {
                slot: slot as u32,
                got: *have,
            });
        }
    }
    Ok(RoomWitness { round_trips })
}

/// The floor: no MM bank on this architecture — the strip word is the
/// room here, and the witness says so rather than pretending.
#[cfg(not(target_arch = "x86_64"))]
pub fn register_room() -> Result<RoomWitness, RoomFault> {
    Err(RoomFault::Unsupported)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host half of the exercise: the launch is a typed refusal, so
    /// the witness reports `Launch` — never a hang, never a panic.  The
    /// laws themselves are checkable only where threads exist, which is
    /// the entire reason this module ships behind `validate`.
    #[test]
    fn the_exercise_refuses_typed_off_metal() {
        assert_eq!(spmd(), Err(SpmdFault::Launch));
    }

    /// The room witness never passes where it proved nothing: off
    /// x86_64 the bank is absent; on an x86_64 host the bank exists
    /// and round-trips registers, but no park ever succeeds, so the
    /// verdict is `NoRoundTrip` — the vacuous-pass hole is closed on
    /// both arms.
    #[test]
    fn the_room_witness_refuses_vacuous_passes() {
        #[cfg(target_arch = "x86_64")]
        assert_eq!(register_room(), Err(RoomFault::NoRoundTrip));
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(register_room(), Err(RoomFault::Unsupported));
    }
}

// ── The lane-tier exercise (G8/G9 — the fiber composition) ─────────────

/// What the lane exercise proved: the deck ran a full quad through
/// yields, mail, and reconvergence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneWitness {
    /// Lanes that ran to Done.
    pub lanes: u32,
    /// Total yields the deck audited.
    pub yields: u32,
}

/// Why the lane exercise refused or failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneFault {
    /// No fiber deck on this target (the switch floor, or provisioning
    /// refused) — the honest refusal the host suite pins.
    Unsupported,
    /// The escape reported (a poisoned lane, or a refused spill).
    Escape,
    /// A lane's mail chain broke: the expected neighbour value was
    /// absent or wrong.
    Mail { lane: u32 },
    /// The round-robin order broke: a lane's first run was out of
    /// sequence.
    Order { lane: u32 },
}

/// Per-lane observation cells for [`lane`] — the exercise records, the
/// checker reads (never printed; the consuming binary owns output).
static LANE_FIRST_RUN: [AtomicU32; 4] = [const { AtomicU32::new(u32::MAX) }; 4];
static LANE_MAIL_OK: [AtomicU32; 4] = [const { AtomicU32::new(0) }; 4];
static LANE_SEQ: AtomicU32 = AtomicU32::new(0);
static LANE_YIELDS: AtomicU32 = AtomicU32::new(0);

/// The fiber body: record arrival order, yield, mail the next lane,
/// yield, take own mail and verify the ring, done.
fn lane_body(ctx: &mut crate::lane::fiber::FiberCtx) {
    let lane = ctx.lane() as usize;
    let seq = LANE_SEQ.fetch_add(1, Ordering::SeqCst);
    LANE_FIRST_RUN[lane].store(seq, Ordering::SeqCst);
    LANE_YIELDS.fetch_add(1, Ordering::SeqCst);
    ctx.yield_now();
    // Mail the ring: lane → lane+1 carries (lane + 1) as the payload.
    let next = crate::room::QuadLane::of((lane + 1) % 4).expect("quad ring");
    let msg = core::num::NonZeroU64::new(lane as u64 + 1).expect("non-zero");
    let _ = ctx.post(next, msg);
    LANE_YIELDS.fetch_add(1, Ordering::SeqCst);
    ctx.yield_now();
    // My box should now carry my predecessor's value.
    let want = ((lane + 3) % 4) as u64 + 1;
    if ctx.take().map(|m| m.get()) == Some(want) {
        LANE_MAIL_OK[lane].store(1, Ordering::SeqCst);
    }
}

/// **The lane exercise** (G8/G9): provision a deck, spill a full quad
/// whose fibers interleave yields and pass mail around the ring, and
/// verify the round-robin order, the mail, and the reconvergence.
///
/// One-shot per boot (the observation cells are spent); the floor
/// refuses [`LaneFault::Unsupported`] — the vacuous-pass rule every
/// witness here follows.
pub fn lane() -> Result<LaneWitness, LaneFault> {
    let mut deck = crate::lane::fiber::FiberDeck::provision().ok_or(LaneFault::Unsupported)?;
    let ls = crate::lane::Lockstep::full();
    if deck.escape(&ls, 0, lane_body).is_err() {
        return Err(LaneFault::Escape);
    }
    // The first sweep runs lanes in index order — the round-robin's own
    // determinism, checked rather than assumed.
    for lane in 0..4u32 {
        if LANE_FIRST_RUN[lane as usize].load(Ordering::SeqCst) != lane {
            return Err(LaneFault::Order { lane });
        }
    }
    for lane in 0..4u32 {
        if LANE_MAIL_OK[lane as usize].load(Ordering::SeqCst) != 1 {
            return Err(LaneFault::Mail { lane });
        }
    }
    Ok(LaneWitness {
        lanes: 4,
        yields: LANE_YIELDS.load(Ordering::SeqCst),
    })
}

// ── The oversubscription exercise (G7 — forced cohort rotation) ────────

/// What the oversubscribed run proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotationWitness {
    /// The region's width — chosen past the CI gate's core count, so
    /// quantum rotations are forced by construction.
    pub workers: u32,
    /// Busy phases each worker completed.
    pub phases: u32,
}

/// The oversubscribed region's shape: eight workers (the gate boots
/// 2..4 cores), three barrier phases, each phase spinning long enough
/// that slices expire and the G7 epoch machinery rotates the cohort.
const ROTATION_WORKERS: usize = 8;
const ROTATION_PHASES: usize = 3;
static ROTATION_CTX: SpmdCtx = SpmdCtx::new(ROTATION_WORKERS as u32);
static ROTATION_DONE: AtomicU32 = AtomicU32::new(0);

fn rotation_body(ctx: &'static SpmdCtx, _worker: u32) {
    for _ in 0..ROTATION_PHASES {
        // Busy work sized to outlive a slice floor (1 ms) under TCG:
        // the spin is the point — a parked worker never rotates.
        let mut x = 0u64;
        for i in 0..2_000_000u64 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
        }
        core::hint::black_box(x);
        ctx.barrier().arrive_and_wait();
    }
    ROTATION_DONE.fetch_add(1, Ordering::SeqCst);
}

/// **The oversubscription exercise** (G7's contention proof): a region
/// wider than the machine, every phase spinning past the slice floor —
/// completion under forced rotation IS the verdict (a wedged cohort
/// never joins).  One-shot per boot.
pub fn spmd_oversubscribed() -> Result<RotationWitness, SpmdFault> {
    let mut region =
        spawn_spmd::<ROTATION_WORKERS>(&ROTATION_CTX, rotation_body).ok_or(SpmdFault::Launch)?;
    let statuses = 'join: {
        for _ in 0..JOIN_ATTEMPTS {
            match region.join_all() {
                Ok(statuses) => break 'join statuses,
                Err(refused) => region = refused.spmd,
            }
        }
        return Err(SpmdFault::Join);
    };
    if statuses.iter().any(|s| *s != Some(0)) {
        return Err(SpmdFault::Status);
    }
    if ROTATION_DONE.load(Ordering::SeqCst) != ROTATION_WORKERS as u32 {
        return Err(SpmdFault::Arrivals {
            phase: 0,
            seen: ROTATION_DONE.load(Ordering::SeqCst),
        });
    }
    Ok(RotationWitness {
        workers: ROTATION_WORKERS as u32,
        phases: ROTATION_PHASES as u32,
    })
}

// ── The auto-dispatch exercise (the parallel-for sizes itself) ─────────

/// What the dispatch exercise proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchWitness {
    /// The width the runtime CHOSE for the machine (one worker per
    /// exclusive best-class core) — the number a dev never wrote.
    pub width: u32,
    /// Tiles reduced.
    pub tiles: u32,
}

/// Where the dispatch exercise failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchFault {
    /// [`dispatch`](crate::spmd::dispatch)/[`reduce`](crate::spmd::reduce)
    /// refused: a host build, or a launch the machine could not place.
    Launch,
    /// The reduction returned the wrong total — a worker skipped or
    /// double-counted its slice.
    Sum {
        /// What the fold produced.
        got: u64,
        /// What the closed form says.
        want: u64,
    },
}

/// Tiles the dispatch exercise reduces — enough that the fit width
/// (best-lane) is the binding cap, not the tile count.
const DISPATCH_TILES: u32 = 4096;

/// **The auto-dispatch exercise**: reduce `0..DISPATCH_TILES` with the
/// runtime picking the width off the machine, and check the total
/// against the closed form `n·(n−1)/2`.  Proves the whole path —
/// `machine()` read, width fit, block split, per-worker fold,
/// cross-worker combine — computes exactly the serial answer, on metal,
/// with no width the exercise chose.  The honest floor
/// ([`DispatchFault::Launch`]) is the host refusal every witness here
/// follows.  Not one-shot: `dispatch` re-stages its region, so this may
/// run beside the other exercises and again from a terminal.
pub fn dispatch() -> Result<DispatchWitness, DispatchFault> {
    let n = DISPATCH_TILES as u64;
    let want = n * (n - 1) / 2;
    let got = crate::spmd::reduce(DISPATCH_TILES, 0, |tile| tile as u64, |a, b| a + b)
        .ok_or(DispatchFault::Launch)?;
    if got != want {
        return Err(DispatchFault::Sum { got, want });
    }
    let width = crate::spmd::machine()
        .map(|m| m.fitting_width().min(DISPATCH_TILES))
        .unwrap_or(1);
    Ok(DispatchWitness {
        width,
        tiles: DISPATCH_TILES,
    })
}

// ── The resident-region exercise (a founded gang, a stream of jobs) ────

/// What the resident exercise proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentWitness {
    /// The gang founded ONCE and reused across every job.
    pub workers: u32,
    /// Jobs the one gang served in turn (each fenced by the completion
    /// word, none re-founding the gang).
    pub jobs: u32,
}

/// The resident exercise's shape: a small gang founded once, several
/// jobs submitted over it in turn.  Small on purpose — like the barrier
/// exercise, it must fit beside the witness's own threads, and it proves
/// REUSE, not width.
const RESIDENT_WORKERS: usize = 2;
const RESIDENT_JOBS: u32 = 3;
static RESIDENT_CTX: SpmdCtx = SpmdCtx::new(RESIDENT_WORKERS as u32);

/// Jobs each worker has run — a resident worker bumps its own cell once
/// per job, so after job `g` every cell reads `g` (no miss, no re-run).
static RESIDENT_MARKS: [AtomicU32; RESIDENT_WORKERS] =
    [const { AtomicU32::new(0) }; RESIDENT_WORKERS];

/// A resident worker's entry: the loop, over the marking body.  Founded
/// once by [`spmd_resident`]; run for every job until the region retires.
/// Target-neutral like [`resident_run`] it wraps — off metal it is named
/// but never founded (`found_resident` refuses), never invoked.
fn resident_entry(ctx: &'static SpmdCtx, worker: u32) {
    crate::spmd::resident_run(ctx, worker, resident_body);
}

/// The per-job body: record that this worker ran one more job.
fn resident_body(_ctx: &'static SpmdCtx, worker: u32) {
    RESIDENT_MARKS[worker as usize].fetch_add(1, Ordering::SeqCst);
}

/// **The resident-region exercise** (R9, `res/SPMD.md` §14): found a gang
/// ONCE, submit [`RESIDENT_JOBS`] jobs over it in turn — each fenced by
/// the completion word with no join — and check after each that every
/// worker has run exactly that many jobs.  A gang that re-founded per job,
/// a completion word that reported early, or a worker that missed or
/// re-ran a job all fail the count; a retire that stranded a worker hangs
/// the join it ends with.
///
/// The metal proof the resident engine composes — a real founding, real
/// threads parked between real jobs, woken by a real submit, fenced by a
/// real memory word, and reaped by a real retire — that a host, with no
/// gang to found, can only refuse ([`SpmdFault::Launch`]).
pub fn spmd_resident() -> Result<ResidentWitness, SpmdFault> {
    for cell in RESIDENT_MARKS.iter() {
        cell.store(0, Ordering::SeqCst);
    }
    let mut region = crate::spmd::found_resident::<RESIDENT_WORKERS>(
        &RESIDENT_CTX,
        resident_entry,
        RESIDENT_WORKERS,
    )
    .ok_or(SpmdFault::Launch)?;

    for job in 1..=RESIDENT_JOBS {
        // SAFETY: `resident_body` reads no job data off `env`, so the null
        // pointer is trivially valid for the job's whole life.
        let ticket = unsafe { region.submit(RESIDENT_WORKERS as u32, 0, core::ptr::null_mut()) }
            .ok_or(SpmdFault::Launch)?;
        region.wait(ticket).map_err(|_| SpmdFault::Launch)?;
        // Every worker has now run exactly `job` jobs — the witness that the
        // one founded gang served this job on every worker, and that the
        // completion word did not report before they finished.
        for cell in RESIDENT_MARKS.iter() {
            let seen = cell.load(Ordering::SeqCst);
            if seen != job {
                return Err(SpmdFault::Arrivals { phase: job, seen });
            }
        }
    }

    // Retire the gang: a clean join is the witness that every worker
    // observed the teardown and exited its resident loop.
    region.retire().map_err(|_| SpmdFault::Join)?;
    Ok(ResidentWitness {
        workers: RESIDENT_WORKERS as u32,
        jobs: RESIDENT_JOBS,
    })
}

// ── The fault-upcall exercise (`res/WASM_PLATFORM.md` §12) ──────────────

/// What the fault-upcall exercise proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UpcallWitness {
    /// Faults delivered to the handler and recovered from: a load from an
    /// unmapped page, an invalid instruction, and on x86-64 a division by
    /// zero (AArch64 and RISC-V do not trap on one).
    pub delivered: u32,
}

/// Where the fault-upcall exercise failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpcallFault {
    /// Installing the handler or arming the thread refused — always, off
    /// the kore target.
    Arm(crate::upcall::Refusal),
    /// No unmapped page to fault on: the probe's guarded stack refused.
    Target,
    /// A probe ran to completion: its fault was never delivered.
    NotDelivered { probe: u32 },
    /// The handler was given the wrong `pc`, address or kind.
    Evidence { probe: u32 },
}

/// **The fault-upcall exercise**: install a handler, arm this thread, and
/// raise each fault a guarded guest raises — a load from an unmapped page,
/// an invalid instruction, a division by zero — checking that each reaches
/// the handler on the upcall stack with its exact evidence (the faulting
/// instruction, the unmapped address, the kind), and that the handler's
/// jump back resumes this thread on its own stack.  A wrong delivery is a
/// wrong answer, never a missing line; a delivery the kernel refused ends
/// the process, which the boot gate reads as the line never printed.
///
/// What a run cannot show in-process is a fault the upcall itself raises, or
/// a declined one: either ends the process.  The kernel's policy holds both
/// to account on the host (`runtime::sched::upcall`).
#[cfg(target_os = "none")]
pub fn upcall() -> Result<UpcallWitness, UpcallFault> {
    use kore_abi::crash::FaultKind;

    crate::upcall::install(upcall_probe::handler).map_err(UpcallFault::Arm)?;
    let armed = crate::upcall::arm_thread().map_err(UpcallFault::Arm)?;
    // A page no mapping backs: a guarded stack's guard page.
    let target = crate::fiber_stack::GuardedStack::new(1).ok_or(UpcallFault::Target)?;
    let unmapped = target.guard().start;
    upcall_probe::TARGET.store(unmapped as usize, Ordering::Relaxed);

    let probes = upcall_probe::probes();
    let mut delivered = 0;
    for (index, probe) in probes.iter().enumerate() {
        let index = index as u32;
        let seen =
            upcall_probe::run(probe.code).ok_or(UpcallFault::NotDelivered { probe: index })?;
        let start = probe.code as usize;
        let expected_address = match probe.kind {
            FaultKind::Memory => Some(unmapped as usize),
            _ => None,
        };
        let right = (start..start + upcall_probe::PROBE_BYTES).contains(&seen.pc)
            && seen.kind == probe.kind as usize
            && seen.address == expected_address;
        if !right {
            return Err(UpcallFault::Evidence { probe: index });
        }
        delivered += 1;
    }
    drop(target);
    drop(armed);
    Ok(UpcallWitness { delivered })
}

/// Off the kore target there is no kernel to deliver a fault.
#[cfg(not(target_os = "none"))]
pub fn upcall() -> Result<UpcallWitness, UpcallFault> {
    Err(UpcallFault::Arm(crate::upcall::Refusal::Unsupported))
}

/// The exercise's machinery: the probes that fault, a minimal catch and
/// resume (the callee-saved registers, stack and return address, as a C
/// `setjmp`/`longjmp` pair keeps them), and the handler that records what
/// it was given and resumes the catch.
#[cfg(target_os = "none")]
mod upcall_probe {
    use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

    use kore_abi::crash::FaultKind;

    /// Every probe faults within its first this-many bytes.
    pub(super) const PROBE_BYTES: usize = 16;

    /// The unmapped address the memory probe loads from.
    pub(super) static TARGET: AtomicUsize = AtomicUsize::new(0);

    /// The catch a fault resumes; null while no probe runs, so a fault
    /// outside one is declined and ends the process as it would have.
    static ACTIVE: AtomicPtr<Saved> = AtomicPtr::new(core::ptr::null_mut());

    static SEEN_PC: AtomicUsize = AtomicUsize::new(0);
    static SEEN_HAS_ADDRESS: AtomicUsize = AtomicUsize::new(0);
    static SEEN_ADDRESS: AtomicUsize = AtomicUsize::new(0);
    static SEEN_KIND: AtomicUsize = AtomicUsize::new(0);

    /// A catch's saved state: room for the largest callee-saved set (RISC-V's
    /// twelve integer and twelve FP registers, `ra` and `sp`).
    type Saved = [u64; 32];

    /// One probe: its code, and the fault it raises.
    pub(super) struct Probe {
        pub code: unsafe extern "C" fn(),
        pub kind: FaultKind,
    }

    /// What the handler was given.
    pub(super) struct Seen {
        pub pc: usize,
        pub address: Option<usize>,
        pub kind: usize,
    }

    const LOAD: Probe = Probe {
        code: probe_load,
        kind: FaultKind::Memory,
    };
    const INVALID: Probe = Probe {
        code: probe_invalid,
        kind: FaultKind::InvalidInstruction,
    };

    /// The probes this ISA can raise: a division by zero traps on x86-64
    /// only (AArch64 and RISC-V define its result).
    #[cfg(target_arch = "x86_64")]
    const PROBES: &[Probe] = &[
        LOAD,
        INVALID,
        Probe {
            code: probe_divide,
            kind: FaultKind::Arithmetic,
        },
    ];
    #[cfg(not(target_arch = "x86_64"))]
    const PROBES: &[Probe] = &[LOAD, INVALID];

    pub(super) fn probes() -> &'static [Probe] {
        PROBES
    }

    /// Run `probe` under a catch: what the handler was given if it faulted
    /// and was resumed, `None` if it ran to completion.
    #[inline(never)]
    pub(super) fn run(probe: unsafe extern "C" fn()) -> Option<Seen> {
        let mut saved: Saved = [0; 32];
        ACTIVE.store(&mut saved, Ordering::Release);
        // SAFETY: `catch` saves this frame's callee-saved state into
        // `saved`, which outlives every resume of it: the handler resumes
        // it only while ACTIVE names it, and ACTIVE is cleared below before
        // `saved` goes out of scope.  Nothing live across the call is held
        // outside the state `catch` restores.
        let caught = unsafe { catch(probe, &mut saved) };
        ACTIVE.store(core::ptr::null_mut(), Ordering::Release);
        (caught == 1).then(|| Seen {
            pc: SEEN_PC.load(Ordering::Relaxed),
            address: (SEEN_HAS_ADDRESS.load(Ordering::Relaxed) != 0)
                .then(|| SEEN_ADDRESS.load(Ordering::Relaxed)),
            kind: SEEN_KIND.load(Ordering::Relaxed),
        })
    }

    /// The exercise's handler: record, then resume the catch.  With no
    /// catch armed (a fault outside a probe, or a second one), it declines.
    pub(super) unsafe extern "C" fn handler(
        pc: usize,
        _fp: usize,
        has_address: usize,
        address: usize,
        kind: usize,
    ) {
        let saved = ACTIVE.swap(core::ptr::null_mut(), Ordering::AcqRel);
        if saved.is_null() {
            return;
        }
        SEEN_PC.store(pc, Ordering::Relaxed);
        SEEN_HAS_ADDRESS.store(has_address, Ordering::Relaxed);
        SEEN_ADDRESS.store(address, Ordering::Relaxed);
        SEEN_KIND.store(kind, Ordering::Relaxed);
        // SAFETY: `saved` is the live catch `run` armed, still on its frame.
        unsafe { resume(saved) }
    }

    // ── x86-64 ──────────────────────────────────────────────────────────

    /// `catch(probe, saved) -> 0` when the probe returns, `1` when resumed:
    /// saves `rbx`, `rbp`, `r12`–`r15`, the caller's stack pointer and the
    /// return address.
    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "C" fn catch(_probe: unsafe extern "C" fn(), _saved: *mut Saved) -> u64 {
        core::arch::naked_asm!(
            "mov [rsi + 0], rbx",
            "mov [rsi + 8], rbp",
            "mov [rsi + 16], r12",
            "mov [rsi + 24], r13",
            "mov [rsi + 32], r14",
            "mov [rsi + 40], r15",
            "lea rax, [rsp + 8]",
            "mov [rsi + 48], rax",
            "mov rax, [rsp]",
            "mov [rsi + 56], rax",
            "sub rsp, 8",
            "call rdi",
            "add rsp, 8",
            "xor eax, eax",
            "ret",
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "C" fn resume(_saved: *const Saved) -> ! {
        core::arch::naked_asm!(
            "mov rbx, [rdi + 0]",
            "mov rbp, [rdi + 8]",
            "mov r12, [rdi + 16]",
            "mov r13, [rdi + 24]",
            "mov r14, [rdi + 32]",
            "mov r15, [rdi + 40]",
            "mov rsp, [rdi + 48]",
            "mov eax, 1",
            "jmp qword ptr [rdi + 56]",
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "C" fn probe_load() {
        core::arch::naked_asm!(
            "mov rax, qword ptr [rip + {target}]",
            "mov rax, qword ptr [rax]",
            "ret",
            target = sym TARGET,
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "C" fn probe_invalid() {
        core::arch::naked_asm!("ud2", "ret")
    }

    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "C" fn probe_divide() {
        core::arch::naked_asm!(
            "xor ecx, ecx",
            "xor edx, edx",
            "mov eax, 1",
            "div ecx",
            "ret"
        )
    }

    // ── AArch64 ─────────────────────────────────────────────────────────

    /// Saves `x19`–`x30`, `sp` and `d8`–`d15`; resumed, it returns `1`
    /// through the saved link register.
    #[cfg(target_arch = "aarch64")]
    #[unsafe(naked)]
    unsafe extern "C" fn catch(_probe: unsafe extern "C" fn(), _saved: *mut Saved) -> u64 {
        core::arch::naked_asm!(
            "stp x19, x20, [x1, #0]",
            "stp x21, x22, [x1, #16]",
            "stp x23, x24, [x1, #32]",
            "stp x25, x26, [x1, #48]",
            "stp x27, x28, [x1, #64]",
            "stp x29, x30, [x1, #80]",
            "mov x9, sp",
            "str x9, [x1, #96]",
            "stp d8, d9, [x1, #104]",
            "stp d10, d11, [x1, #120]",
            "stp d12, d13, [x1, #136]",
            "stp d14, d15, [x1, #152]",
            "stp x29, x30, [sp, #-16]!",
            "mov x29, sp",
            "blr x0",
            "ldp x29, x30, [sp], #16",
            "mov x0, #0",
            "ret",
        )
    }

    #[cfg(target_arch = "aarch64")]
    #[unsafe(naked)]
    unsafe extern "C" fn resume(_saved: *const Saved) -> ! {
        core::arch::naked_asm!(
            "ldp x19, x20, [x0, #0]",
            "ldp x21, x22, [x0, #16]",
            "ldp x23, x24, [x0, #32]",
            "ldp x25, x26, [x0, #48]",
            "ldp x27, x28, [x0, #64]",
            "ldp x29, x30, [x0, #80]",
            "ldr x9, [x0, #96]",
            "mov sp, x9",
            "ldp d8, d9, [x0, #104]",
            "ldp d10, d11, [x0, #120]",
            "ldp d12, d13, [x0, #136]",
            "ldp d14, d15, [x0, #152]",
            "mov x0, #1",
            "ret",
        )
    }

    #[cfg(target_arch = "aarch64")]
    #[unsafe(naked)]
    unsafe extern "C" fn probe_load() {
        core::arch::naked_asm!(
            "adrp x9, {target}",
            "ldr x9, [x9, :lo12:{target}]",
            "ldr x9, [x9]",
            "ret",
            target = sym TARGET,
        )
    }

    #[cfg(target_arch = "aarch64")]
    #[unsafe(naked)]
    unsafe extern "C" fn probe_invalid() {
        core::arch::naked_asm!("udf #0", "ret")
    }

    // ── RISC-V ──────────────────────────────────────────────────────────

    /// Saves `s0`–`s11`, `ra`, `sp` and `fs0`–`fs11`; resumed, it returns
    /// `1` through the saved return address.
    #[cfg(target_arch = "riscv64")]
    #[unsafe(naked)]
    unsafe extern "C" fn catch(_probe: unsafe extern "C" fn(), _saved: *mut Saved) -> u64 {
        core::arch::naked_asm!(
            "sd s0, 0(a1)",
            "sd s1, 8(a1)",
            "sd s2, 16(a1)",
            "sd s3, 24(a1)",
            "sd s4, 32(a1)",
            "sd s5, 40(a1)",
            "sd s6, 48(a1)",
            "sd s7, 56(a1)",
            "sd s8, 64(a1)",
            "sd s9, 72(a1)",
            "sd s10, 80(a1)",
            "sd s11, 88(a1)",
            "sd ra, 96(a1)",
            "sd sp, 104(a1)",
            "fsd fs0, 112(a1)",
            "fsd fs1, 120(a1)",
            "fsd fs2, 128(a1)",
            "fsd fs3, 136(a1)",
            "fsd fs4, 144(a1)",
            "fsd fs5, 152(a1)",
            "fsd fs6, 160(a1)",
            "fsd fs7, 168(a1)",
            "fsd fs8, 176(a1)",
            "fsd fs9, 184(a1)",
            "fsd fs10, 192(a1)",
            "fsd fs11, 200(a1)",
            "addi sp, sp, -16",
            "sd ra, 8(sp)",
            "jalr a0",
            "ld ra, 8(sp)",
            "addi sp, sp, 16",
            "li a0, 0",
            "ret",
        )
    }

    #[cfg(target_arch = "riscv64")]
    #[unsafe(naked)]
    unsafe extern "C" fn resume(_saved: *const Saved) -> ! {
        core::arch::naked_asm!(
            "ld s0, 0(a0)",
            "ld s1, 8(a0)",
            "ld s2, 16(a0)",
            "ld s3, 24(a0)",
            "ld s4, 32(a0)",
            "ld s5, 40(a0)",
            "ld s6, 48(a0)",
            "ld s7, 56(a0)",
            "ld s8, 64(a0)",
            "ld s9, 72(a0)",
            "ld s10, 80(a0)",
            "ld s11, 88(a0)",
            "ld ra, 96(a0)",
            "ld sp, 104(a0)",
            "fld fs0, 112(a0)",
            "fld fs1, 120(a0)",
            "fld fs2, 128(a0)",
            "fld fs3, 136(a0)",
            "fld fs4, 144(a0)",
            "fld fs5, 152(a0)",
            "fld fs6, 160(a0)",
            "fld fs7, 168(a0)",
            "fld fs8, 176(a0)",
            "fld fs9, 184(a0)",
            "fld fs10, 192(a0)",
            "fld fs11, 200(a0)",
            "li a0, 1",
            "ret",
        )
    }

    #[cfg(target_arch = "riscv64")]
    #[unsafe(naked)]
    unsafe extern "C" fn probe_load() {
        core::arch::naked_asm!(
            "lla t0, {target}",
            "ld t0, 0(t0)",
            "ld t0, 0(t0)",
            "ret",
            target = sym TARGET,
        )
    }

    #[cfg(target_arch = "riscv64")]
    #[unsafe(naked)]
    unsafe extern "C" fn probe_invalid() {
        core::arch::naked_asm!("unimp", "ret")
    }
}

#[cfg(test)]
mod lane_tests {
    use super::*;

    /// Off metal there is no deck: the lane exercise refuses typed —
    /// never a vacuous pass.
    #[test]
    fn the_lane_exercise_refuses_off_metal() {
        assert_eq!(lane().unwrap_err(), LaneFault::Unsupported);
    }

    /// Off metal there is no gang to found: the resident exercise refuses
    /// rather than pretending an engine it never stood up.
    #[test]
    fn the_resident_exercise_refuses_off_metal() {
        assert_eq!(spmd_resident().unwrap_err(), SpmdFault::Launch);
    }

    /// Off metal there is no spawn: the rotation exercise refuses.
    #[test]
    fn the_rotation_exercise_refuses_off_metal() {
        assert_eq!(spmd_oversubscribed().unwrap_err(), SpmdFault::Launch);
    }

    /// Off metal the parallel-for cannot found workers: the dispatch
    /// exercise refuses rather than folding single-threaded.
    #[test]
    fn the_dispatch_exercise_refuses_off_metal() {
        assert_eq!(dispatch().unwrap_err(), DispatchFault::Launch);
    }

    /// Off metal there is no kernel to deliver a fault: the upcall exercise
    /// refuses rather than reporting deliveries it never saw.
    #[test]
    fn the_upcall_exercise_refuses_off_metal() {
        assert_eq!(
            upcall().unwrap_err(),
            UpcallFault::Arm(crate::upcall::Refusal::Unsupported)
        );
    }
}
