//! **SPMD** — one body, many workers, launched as one region.
//!
//! The launch spine of `res/SPMD.md` §5: [`spawn_spmd`] founds
//! `N` workers of one entry over the caller's [`SpmdCtx`], every worker is
//! entered as `entry(ctx, worker)` with its dense region-local index, and
//! the affine [`Spmd`] ticket joins (or tears down) all of them.  The
//! `(x,y,z)` tile geometry is deliberately NOT here — geometry is the
//! macro layer's compile-time affair (§1/§6); this module knows only
//! *workers*.
//!
//! # The context is the launch argument
//!
//! Workers share the address space, so shared state needs no provisioning
//! — it needs an *address*.  The caller declares a `static SpmdCtx`
//! (the macro emits one; a runtime caller writes their own) and every
//! worker receives `&'static SpmdCtx` in its start record, the same way
//! a CUDA kernel receives its launch arguments: explicitly, not through
//! an ambient global the body must know by name.  The ctx carries the
//! [`SpmdBarrier`] the workers tile-sync on and the launch gate below;
//! user data rides wherever the user's own statics or buffers put it.
//!
//! # All-or-nothing, behind the gate
//!
//! A founding can refuse at any worker (an exhausted band plan, arena
//! slots, a kernel refusal), and a founded worker starts running
//! immediately — the kernel has no "create suspended".  A partial gang
//! that reached the barrier would wait for siblings that never came.  So
//! no worker may touch user code until the whole launch is decided: the
//! runtime holds every worker at the ctx's **gate** ([`enter`], called by
//! the thread trampoline's Rust half, not by user code — unforgettable
//! by construction), and [`spawn_spmd`] releases the gate only once
//! all `N` foundings succeeded.  On a partial founding it aborts
//! instead: staged workers exit without running the entry, their tickets
//! are joined, and the caller gets back everything the attempt cost.
//! The barrier's party count therefore stays the compile-time constant
//! the geometry declared — a smaller machine wants a smaller *region*,
//! never a half-launched one (§12 keeps a `min_workers` floor as a
//! future knob on this same gate).

use core::sync::atomic::Ordering;

use crate::barrier::{Line, SpmdBarrier, wait_while};
use crate::stack_arena::StripPlan;

/// The SPMD stack window (`res/SPMD.md` §8, G4b): an rt-side VA
/// reservation, the thread stack band's own precedent — NOT a
/// `boot.toml` row.  It sits in the high-half gap between the stub
/// window and the per-thread stack band, and the proof beside
/// [`BANDS`](crate::thread::BANDS) holds it disjoint from every band
/// at compile time.
pub(crate) const STACK_WINDOW_BASE: u64 = 0x7fff_e000_0000;

/// The window's capacity: 16 MiB of *reservation* (4096 granules).  No
/// byte is mapped by declaring it — a guarded strip's pages arrive
/// with its worker's founding session and leave with its teardown, so
/// an idle window costs address space only.
pub(crate) const STACK_WINDOW_GRANULES: u32 = 4096;

/// Usable granules per worker stack — 64 KiB, parity with the band
/// stacks this replaces (the guard granule is carved on top by
/// [`carve_guarded`](StripPlan::carve_guarded)).
// Consumed by the metal spawn arm; host builds keep the declaration so
// the geometry stays one set of numbers everywhere.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
pub(crate) const STACK_GRANULES: u32 = 16;

/// **The task's SPMD stack plan** — worker stacks are carved here
/// (`spawn_worker`), not tiled by `MAX_THREADS`.  Task-wide like every
/// band; a *region* owns its strips (its workers' teardown gives them
/// back), never the window.
// Carved by the metal spawn arm, released by join; host builds keep it
// so the plan's compile-time geometry asserts run in `make test` too.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
pub(crate) static STACKS: StripPlan<64> = StripPlan::new(STACK_WINDOW_BASE, STACK_WINDOW_GRANULES);

/// A worker's entry: the shared context and this worker's dense index.
///
/// A plain `fn`, not a closure — the worker's whole context IS the two
/// arguments, which is what lets the start record carry it with no
/// allocation and no captured environment to keep alive.
pub type SpmdEntry = fn(&'static SpmdCtx, u32);

/// **What the machine will run** — the CPU facts the kernel stamped onto
/// this task's boot page (`kore_abi::machine::MachineBlock`), read with
/// no IPC.  [`dispatch`] reads it to size a launch; a terminal tool prints
/// it so a dev sees what they would otherwise guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Machine {
    /// Logical CPUs the catalog enumerated.
    pub logical_cpus: u32,
    /// The **best lane**: how many cores of this task's founding node's
    /// best performance class an SPMD region is granted exclusively —
    /// the G6c reservation population (P-cores on a hybrid box, all
    /// cores on a homogeneous one).  The width [`dispatch`] fits to.
    pub best_lane: u32,
    /// Performance-class cores machine-wide (Intel P / AMD Zen / Arm big).
    pub performance_cores: u8,
    /// Efficiency-class cores machine-wide (Intel E / AMD Zen-c / Arm LITTLE).
    pub efficient_cores: u8,
    /// Standard-class cores machine-wide — a homogeneous part counts every
    /// core here.  All three zero on a founder that predates the census.
    pub standard_cores: u8,
    /// Whether a hardware feedback source (Intel HFI) is live and steering
    /// the kernel's per-CPU capacity coefficient.
    pub hardware_feedback: bool,
}

impl Machine {
    /// The width a region should found to run one worker per exclusive
    /// best-class core: the best lane, clamped to at least one and to
    /// the runtime's ceiling.  The fit policy [`dispatch`] defaults to.
    pub fn fitting_width(&self) -> u32 {
        self.best_lane.clamp(1, crate::thread::MAX_THREADS - 1)
    }
}

/// This task's machine facts, or `None` off metal / on a founder that
/// predates the machine block (fail-closed, like every boot-page read).
///
/// Cheap: a read of the already-mapped, immutable boot page — no trap,
/// no service.  Cached would be premature; the page never changes and a
/// caller that wants it once reads it once.
pub fn machine() -> Option<Machine> {
    let rt = crate::entry::thread_context()?;
    let block = rt.machine()?;
    let (performance_cores, efficient_cores, standard_cores) = block.class_census()?;
    Some(Machine {
        logical_cpus: block.logical_cpus()?,
        best_lane: block.best_lane()?,
        performance_cores,
        efficient_cores,
        standard_cores,
        hardware_feedback: block.hardware_feedback(),
    })
}

/// The gate is staged: workers founded so far are parked at [`enter`].
const STAGED: u32 = 0;
/// The launch is complete: every worker runs its entry.  `pub(crate)`:
/// the star's hub parks on this very word and needs the value to wait
/// against.
pub(crate) const RUN: u32 = 1;
/// The launch failed partway: staged workers exit without entering.
const ABORT: u32 = 2;
/// The region is over (every worker joined): the hub's exit signal —
/// stored by the join path before the star tears down, never observed
/// by a worker (they all joined first) and never admitted by [`enter`].
const DONE: u32 = 3;

/// **A region's shared anchor** — what every worker can reach by address.
///
/// One per region, `'static` (the launch macro emits it beside the worker
/// geometry; a runtime caller declares it once).  Construction fixes the
/// worker count and sizes the barrier to it; the gate begins staged.
/// State, not authority: holding `&SpmdCtx` conveys the right to wait
/// and nothing else, exactly as the barrier alone does.
pub struct SpmdCtx {
    /// The launch gate. Cold after release — its own line, so gate
    /// traffic during launch never touches the barrier's hot words.
    gate: Line,
    /// The barrier every worker tile-syncs on; `parties == workers`.
    barrier: SpmdBarrier,
    /// The **capacity**: the worker count this ctx was constructed for,
    /// and the ceiling a runtime-width launch may found up to. For a
    /// const-width region (the `#[kore_rt::spmd]` macro) it is also the
    /// launched width; a runtime-width one ([`dispatch`]) founds
    /// [`active`](Self::active) `≤` capacity.
    capacity: u32,
    /// The **launched width** — 0 until a launch arms it, then the
    /// number of workers actually founded (== capacity for a const
    /// launch, `k ≤ capacity` for a runtime one). Set behind the gate,
    /// so a worker reads it only after `enter` and never races the set.
    active: core::sync::atomic::AtomicU32,
    /// The **iteration span** a [`dispatch`] launch strides — the tile
    /// count the runtime splits across workers. 0 means "use the tile
    /// grid" (the macro path); nonzero is the dispatch path's `[0, span)`.
    span: core::sync::atomic::AtomicU32,
    /// The **dispatch body** — a thin pointer to the launcher's
    /// `&dyn Fn` (the environment the start record has no room for), set
    /// behind the gate and read after `enter`. Null for a macro launch.
    /// Sound because workers `SHARE_SPACE` and the launcher outlives the
    /// region (it blocks on join), so the pointer stays dereferenceable
    /// for every read. See [`dispatch`].
    env: core::sync::atomic::AtomicPtr<()>,
    /// The `(x, y, z)` tile grid the workers stride (R1: geometry is
    /// tiles, not threads) — queryable at runtime like CUDA's
    /// `gridDim`, so a body never re-names its launch literals.
    tiles: [u32; 3],
    /// **Resident job generation** (R9, `res/SPMD.md` §14).  `0` at
    /// founding; each [`ResidentRegion::submit`] steps it, and a resident
    /// worker parks on it ([`resident_run`]) until it leaves the
    /// generation that worker last ran.  The one-shot launch path never
    /// touches it (it stays `0`), so a macro region carries the word inert
    /// — the same way it carries `active`/`span`/`env` unset.  Its own
    /// line: resident workers monitor-spin on it, so a hot counter's
    /// traffic must never evict the line they sleep on.
    job: Line,
    /// **Resident completion down-counter** (R9).  [`ResidentRegion::submit`]
    /// seeds it to the full founded width; every worker decrements it after
    /// either running the job body or sitting that job out.  The worker that
    /// drives it to zero publishes [`complete`](Self::complete), fencing every
    /// worker's `active`/`env` observation before the next submit.  Its own
    /// line — every founded worker touches it once per job.
    done: Line,
    /// **Resident completion generation** (R9).  The worker that zeroes
    /// [`done`](Self::done) stores the job's generation here; the launcher
    /// observes it ([`ResidentRegion::poll`]/[`wait`](ResidentRegion::wait))
    /// to learn the job retired, with no join.  Its own line so the
    /// launcher's read never bounces the workers' `done` traffic — the
    /// barrier's phase/pending isolation, applied to the launch↔completion
    /// pair.
    complete: Line,
    /// Monotonic identity source for opaque [`ResidentTicket`]s.
    ///
    /// Workers never touch this word: their compact, monitored protocol stays
    /// on [`job`](Self::job_word)/[`complete`](Self::complete_word) as `u32`.
    /// The resident owner increments this once per admitted job so a public
    /// ticket never reuses an identity when those physical generations wrap,
    /// or when this same static context later founds another resident epoch.
    /// It is deliberately not reset by [`restage`](Self::restage).
    resident_sequence: core::sync::atomic::AtomicU64,
}

impl SpmdCtx {
    /// A context for exactly `workers` workers over a linear grid of
    /// one tile per worker — the runtime-caller default.  A launch
    /// whose tile grid outnumbers its workers declares it with
    /// [`with_grid`](Self::with_grid) (what the `#[kore_rt::spmd]`
    /// macro emits).
    ///
    /// # Panics
    /// `workers == 0` (compile-time when const-evaluated), as the
    /// barrier it sizes.
    pub const fn new(workers: u32) -> Self {
        Self::with_grid(workers, [1, 1, workers])
    }

    /// A context for `workers` workers striding the `(x, y, z)` tile
    /// grid.
    ///
    /// # Panics
    /// A zero dimension, a tile count exceeding `u32::MAX`, or fewer
    /// tiles than workers (a worker with no tile is a worker the
    /// geometry never asked for) — all compile-time when
    /// const-evaluated.
    pub const fn with_grid(workers: u32, tiles: [u32; 3]) -> Self {
        let count = tiles[0] as u128 * tiles[1] as u128 * tiles[2] as u128;
        assert!(
            tiles[0] > 0 && tiles[1] > 0 && tiles[2] > 0,
            "a tile grid has no zero dimension"
        );
        assert!(count <= u32::MAX as u128, "tile count exceeds u32");
        assert!(count >= workers as u128, "more workers than tiles");
        Self {
            gate: Line::new(STAGED),
            barrier: SpmdBarrier::new(workers),
            capacity: workers,
            active: core::sync::atomic::AtomicU32::new(0),
            span: core::sync::atomic::AtomicU32::new(0),
            env: core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
            tiles,
            job: Line::new(0),
            done: Line::new(0),
            complete: Line::new(0),
            resident_sequence: core::sync::atomic::AtomicU64::new(0),
        }
    }

    /// The launched worker count — what a worker strides the tile grid
    /// by. The value a launch armed (== the constructed count for a
    /// const-width region); before any launch, the constructed capacity.
    /// Read by worker bodies, which run only after the gate opens, so it
    /// never observes the arming store racing.
    pub fn workers(&self) -> u32 {
        let a = self.active.load(Ordering::Relaxed);
        if a == 0 { self.capacity } else { a }
    }

    /// The worker count this ctx was constructed for — the ceiling a
    /// runtime-width launch founds up to.
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Arm the ctx for a launch of `active` workers, striding `span`
    /// tiles (0 = the tile grid), with `env` the dispatch body pointer
    /// (null for a macro launch). Sets the barrier's party count and
    /// every runtime word BEFORE the caller opens the gate, so the gate's
    /// release edge publishes them all to the workers at once.
    pub(crate) fn arm(&self, active: u32, span: u32, env: *mut ()) {
        self.active.store(active, Ordering::Relaxed);
        self.span.store(span, Ordering::Relaxed);
        self.env.store(env, Ordering::Relaxed);
        self.barrier.set_parties(active);
    }

    /// Re-stage the gate for a fresh launch over this same `'static` ctx
    /// (a [`dispatch`] in a loop). Called only after `join_all` has
    /// reaped every worker and torn down the star, so no thread is still
    /// reading the gate. The DONE store is starless-conditional, so the
    /// prior value may be RUN or DONE; this stores STAGED unconditionally.
    pub(crate) fn restage(&self) {
        self.env.store(core::ptr::null_mut(), Ordering::Relaxed);
        self.gate.0.store(STAGED, Ordering::SeqCst);
    }

    /// The dispatch body pointer this launch armed, or null — read by a
    /// worker after `enter`, under the gate's acquire edge.
    ///
    /// Public because it is how a resident worker's body reads its job data
    /// (see [`resident_run`] / [`ResidentRegion::submit`]): a backend's
    /// [`SpmdEntry`] lives outside this crate, so the accessor it needs must
    /// be reachable there. The submitter owns whatever this points at and
    /// keeps it valid until completion is observed.
    pub fn env(&self) -> *mut () {
        self.env.load(Ordering::Relaxed)
    }

    /// The iteration span a dispatch launch strides (0 = the tile grid).
    ///
    /// Public for the same reason as [`env`](Self::env): an out-of-crate
    /// resident body strides its work by this.
    pub fn span(&self) -> u32 {
        self.span.load(Ordering::Relaxed)
    }

    /// The gate word itself — the star's hub parks on it (any notice
    /// wakes the hub; the region's end flips it).  Crate plumbing: the
    /// word is state, never authority, but its VALUES are this module's
    /// protocol.
    pub(crate) fn gate_word(&self) -> &core::sync::atomic::AtomicU32 {
        &self.gate.0
    }

    /// The resident job word — a resident worker parks on it, the launcher
    /// steps it per submit (R9).  Crate plumbing; the word is state, but
    /// its VALUES (a stepping generation) are this module's protocol.
    pub(crate) fn job_word(&self) -> &core::sync::atomic::AtomicU32 {
        &self.job.0
    }

    /// The resident completion down-counter (R9): the full founded width at
    /// submit, decremented once by every worker before completion publishes.
    pub(crate) fn done_word(&self) -> &core::sync::atomic::AtomicU32 {
        &self.done.0
    }

    /// The resident completion generation the launcher observes (R9).
    pub(crate) fn complete_word(&self) -> &core::sync::atomic::AtomicU32 {
        &self.complete.0
    }

    /// Mint the next nonzero logical resident-job identity.
    ///
    /// Relaxed ordering is sufficient: this atomic allocates identity only;
    /// the physical job publication and completion edges stay on the existing
    /// `job`/`complete` words. Exhaustion is permanent and refuses rather than
    /// wrapping to an identity an old ticket may still carry.
    fn next_resident_ticket(&'static self) -> Option<ResidentTicket> {
        let previous = self
            .resident_sequence
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |sequence| {
                sequence.checked_add(1)
            })
            .ok()?;
        ResidentTicket::new(self, previous + 1)
    }

    /// Last logical resident-job identity allocated from this static context.
    #[inline]
    fn resident_sequence(&self) -> u64 {
        self.resident_sequence.load(Ordering::Relaxed)
    }

    /// The launched width the current job runs — the raw `active` a submit
    /// (or launch) armed, read by a resident worker to sit out a job
    /// narrower than the founded gang.  Distinct from [`workers`](Self::workers),
    /// whose `0 ⇒ capacity` fallback is for a body reading its stride
    /// before any launch; a resident worker reads this only after a job is
    /// published, so the raw value is what it wants.
    pub(crate) fn active(&self) -> u32 {
        self.active.load(Ordering::Relaxed)
    }

    /// The `(x, y, z)` tile grid this launch covers.
    pub const fn tiles(&self) -> [u32; 3] {
        self.tiles
    }

    /// Total tiles in the grid — the iteration space a worker strides
    /// by [`workers`](Self::workers).  Fits `u32` by construction.
    pub const fn tile_count(&self) -> u32 {
        (self.tiles[0] as u64 * self.tiles[1] as u64 * self.tiles[2] as u64) as u32
    }

    /// The region's barrier — `arrive_and_wait` per tile boundary.
    pub const fn barrier(&self) -> &SpmdBarrier {
        &self.barrier
    }
}

/// **The gate a founded worker holds at** until the launch is decided.
///
/// Called by the thread trampoline's Rust half before the user entry —
/// runtime-enforced, so no worker can race a half-launched region into the
/// barrier.  `true` releases the worker into its entry; `false` is an
/// aborted launch and the worker exits without running user code.  The
/// wait is the barrier's own two-tier slice ([`wait_while`]), so a
/// staged worker costs a parked thread, not a hot core.
pub(crate) fn enter(ctx: &'static SpmdCtx) -> bool {
    wait_while(&ctx.gate.0, STAGED);
    ctx.gate.0.load(Ordering::SeqCst) == RUN
}

/// Decide the launch: store the verdict, then kick the parked workers.
///
/// Store-then-wake, the futex rule the barrier's release also follows.
fn decide(ctx: &'static SpmdCtx, verdict: u32) {
    ctx.gate.0.store(verdict, Ordering::SeqCst);
    crate::futex::wake_all(&ctx.gate.0);
}

/// One worker's join authority, as the region's fold sees it.
///
/// The seam that keeps [`Spmd`]'s lifecycle host-testable (the
/// [`WorkerTask`](crate::worker::WorkerTask) pattern): production workers
/// are [`Thread`](crate::thread::Thread) tickets; host tests drive the
/// fold with a scripted mock instead of simulating kernel joins.
pub trait WorkerTicket: Sized {
    /// One task-local landing held across this ticket set's complete join
    /// fold. Production uses an affine capability-slot reservation; host
    /// tests use `()` because no kernel report crosses there.
    type JoinScratch;

    /// Block until this worker ends; its exit status, or the ticket back
    /// inside the refusal (affine — a failed join must not leak the
    /// join authority). `scratch` was reserved before any worker founding
    /// and remains owned by the enclosing [`Spmd`] across retries.
    fn join(
        self,
        scratch: &mut Self::JoinScratch,
    ) -> Result<Option<i32>, (Self, crate::cap::TaskWaitError)>;

    /// Give up the join; the worker runs on detached.
    fn detach(self);
}

#[cfg(target_os = "none")]
impl WorkerTicket for crate::thread::Thread {
    type JoinScratch = crate::thread::JoinScratch;

    fn join(
        self,
        scratch: &mut Self::JoinScratch,
    ) -> Result<Option<i32>, (Self, crate::cap::TaskWaitError)> {
        crate::thread::join_reserved(self, scratch)
            .map_err(crate::thread::ThreadJoinError::into_parts)
    }

    fn detach(self) {
        crate::thread::Thread::detach(self);
    }
}

/// **A launched region's join ticket** — the affine owner of its `N`
/// worker tickets, generic over the ticket seam exactly as
/// [`WorkerSupervisor`](crate::worker::WorkerSupervisor) is over its
/// task.  Dropping it detaches every remaining worker ([`Thread`]'s own
/// drop-is-detach posture, lifted to the set); [`join_all`](Self::join_all)
/// is the deliberate collect.
///
/// [`Thread`]: crate::thread::Thread
#[must_use = "dropping a Spmd detaches every worker: their bands and stacks stay spent for the task's life"]
pub struct Spmd<T: WorkerTicket, const N: usize> {
    /// Worker `i`'s ticket; `None` once joined (or never founded — the
    /// abort path's teardown leaves no ticket behind).
    workers: [Option<T>; N],
    /// The shared anchor the workers were launched over.
    ctx: &'static SpmdCtx,
    /// The one report landing reserved before founding. A refused join hands
    /// this back inside the region together with the remaining affine tickets,
    /// so arena exhaustion can never remove the collector's progress resource.
    join_scratch: T::JoinScratch,
    /// The region's doorbell star (`res/SPMD.md` §3), when provisioning
    /// succeeded — torn down by [`join_all`](Self::join_all)'s complete
    /// join.  Dropping an unjoined region leaks it bounded, exactly as
    /// it leaks the workers' bands.
    star: Option<crate::star::Star>,
}

/// [`SpmdJoinError::worker`] sentinel for the region's relay hub.
///
/// This value is outside every dense compute-worker index. It lets a caller
/// distinguish a fully collected compute gang whose remaining affine teardown
/// failure belongs to the star from a refusal by an ordinary worker.
pub const RELAY_HUB: u32 = u32::MAX;

/// A [`Spmd::join_all`] that learned no terminal fact about some worker or
/// about the region's relay hub.
///
/// Carries the region back (affine: the un-joined tickets must not leak)
/// with the refusing worker and its cause; already-joined workers stay
/// joined, and a retry resumes where the fold stopped.
#[must_use = "a failed spmd join still owns the remaining worker tickets"]
pub struct SpmdJoinError<T: WorkerTicket, const N: usize> {
    // (Debug below is manual: a refusal report names the worker and the
    // cause; the tickets themselves have nothing legible to print and
    // requiring `T: Debug` would infect the seam for it.)
    /// The region, still owning every not-yet-joined ticket.
    pub spmd: Spmd<T, N>,
    /// The worker whose join refused, or [`RELAY_HUB`] when every
    /// compute worker joined but the relay hub's terminal observation failed.
    pub worker: u32,
    /// Why the blocking lifecycle observation did not complete.
    pub cause: crate::cap::TaskWaitError,
}

impl<T: WorkerTicket, const N: usize> core::fmt::Debug for SpmdJoinError<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpmdJoinError")
            .field("worker", &self.worker)
            .field("cause", &self.cause)
            .field("live", &self.spmd.live())
            .finish_non_exhaustive()
    }
}

impl<T: WorkerTicket, const N: usize> Spmd<T, N> {
    /// The shared anchor this region was launched over.
    pub const fn ctx(&self) -> &'static SpmdCtx {
        self.ctx
    }

    /// How many workers remain joinable.
    pub fn live(&self) -> usize {
        self.workers.iter().flatten().count()
    }

    /// Whether this region carries a live doorbell star (`res/SPMD.md`
    /// §3) — the bench's `star=` column: with it, barrier parks wake by
    /// notice; without it, they re-check on the futex slice.
    pub fn star_active(&self) -> bool {
        self.star.is_some()
    }

    /// **Collect every worker**: block on each in turn, yielding each
    /// worker's exit status in worker order — the region-launch analogue of
    /// [`thread::join`](crate::thread::join), which this folds.
    ///
    /// On a worker's refusal the fold stops and hands everything back in
    /// [`SpmdJoinError`]; workers joined before the refusal stay joined
    /// and are skipped when the fold resumes.  **A status is observed
    /// once, on the attempt that joined that worker**: each attempt's
    /// array reports only what it collected, so a `None` slot means
    /// "no status observed *this attempt*" — a worker that exited
    /// without a status ([`TaskWaitOutcome`](crate::cap::TaskWaitOutcome))
    /// or one already joined by an earlier attempt.  A caller that must
    /// distinguish them records the earlier attempt's array; the region
    /// itself deliberately stores no status (derive-don't-store).
    pub fn join_all(self) -> Result<[Option<i32>; N], SpmdJoinError<T, N>> {
        let mut group = self;
        let mut statuses = [const { None }; N];

        for worker in 0..N {
            let ticket = match group.workers[worker].take() {
                Some(ticket) => ticket,
                None => continue,
            };
            match ticket.join(&mut group.join_scratch) {
                Ok(status) => {
                    statuses[worker] = status;
                }
                Err((ticket, cause)) => {
                    group.workers[worker] = Some(ticket);
                    return Err(SpmdJoinError {
                        spmd: group,
                        worker: worker as u32,
                        cause,
                    });
                }
            }
        }

        // Every worker joined: the region is over.  Flip the gate off
        // `RUN` so the hub's loop exits, then tear the star down —
        // nothing is left to wake.
        if let Some(star) = group.star.take() {
            group.ctx.gate.0.store(DONE, Ordering::SeqCst);
            if let Err((star, cause)) = star.finish() {
                group.star = Some(star);
                return Err(SpmdJoinError {
                    spmd: group,
                    worker: RELAY_HUB,
                    cause,
                });
            }
        }
        Ok(statuses)
    }
}

/// **Launch a region**: found `N` workers of `entry` over `ctx`, all
/// gated; release the gate only when every founding succeeded.
///
/// `None` gives back everything a partial attempt cost: the gate is
/// aborted (staged workers exit without entering user code) and each
/// founded worker is joined so its band, stack, and arena slots return.
/// Refusals that survive the teardown detach honestly rather than hang
/// the launch — a band held by a worker that will not exit is a caller's
/// bug to see, not this function's to hide.
///
/// `ctx.workers()` must equal `N` — the barrier was sized by it, and a
/// mismatched launch would deadlock at the first tile boundary; it is a
/// refusal here, not a wedge there.  The gate must be freshly staged
/// (one launch per [`SpmdCtx`] until a joined region re-stages it —
/// v1 keeps relaunch out of scope).
#[cfg(target_os = "none")]
pub fn spawn_spmd<const N: usize>(
    ctx: &'static SpmdCtx,
    entry: SpmdEntry,
) -> Option<SpawnedSpmd<N>> {
    // The const-width launch: the whole region, no dispatch body, and the
    // relay hub a barrier-syncing one-shot region wants.
    launch::<N>(ctx, entry, N, 0, core::ptr::null_mut(), true)
}

/// The one metal launch, over a runtime `active ≤ N` width.
///
/// `N` is the storage capacity ([`Spmd`]'s array size); `active` is the
/// number of workers this launch founds — equal to `N` for a const
/// region ([`spawn_spmd`]), `k ≤ N` for a runtime one ([`dispatch`]).
/// `span`/`env` carry the dispatch iteration count and body pointer (0
/// and null for the macro path). The founding loop, the star, and the
/// hub are all bounded at `active`, so the unfounded `N − active` slots
/// stay `None` — which join/abort/`live` already tolerate.
///
/// `hub` founds the star's relay thread (`res/SPMD.md` §3) — `true` for a
/// one-shot region, whose workers barrier-sync and need the leader→member
/// cross-chunk relay.  A resident region ([`found_resident`], R9) passes
/// `false`: it still takes the star's parent sends (the founder kicks its
/// workers directly on every submit), but a hub that kicks every
/// `HUB_SLICE` would only spuriously wake a gang that is idle between
/// jobs.  The parent-send kick and the starless degrade are unchanged.
#[cfg(target_os = "none")]
fn launch<const N: usize>(
    ctx: &'static SpmdCtx,
    entry: SpmdEntry,
    active: usize,
    span: u32,
    env: *mut (),
    hub: bool,
) -> Option<SpawnedSpmd<N>> {
    if active == 0 || active > N || ctx.gate.0.load(Ordering::SeqCst) != STAGED {
        return None;
    }
    // Join progress is admitted before any worker can consume parent-arena
    // names. This affine landing travels inside the returned region and is
    // reused serially across the entire fold; a launch cannot succeed in a
    // state where collecting it later requires another allocation.
    let mut join_scratch = crate::thread::JoinScratch::reserve()?;
    // Arm the barrier's party count and the runtime words BEFORE any
    // founding — the gate's later RUN store publishes them to every
    // worker (the same release edge the launch itself rides).
    ctx.arm(active as u32, span, env);
    // The star first (`res/SPMD.md` §3): minted before any founding,
    // all-or-nothing, degrading to a starless launch rather than
    // refusing one.  Sized to the LAUNCHED width, not the capacity.
    let star_plan = crate::star::provision(active);
    // The gang founding (G5): whole chunks per session — one trap each
    // — until the region is founded.  A width the descriptor cannot
    // carry chunks rather than refuses.  A starred chunk is capped at
    // [`MESH_CHUNK`](crate::star::MESH_CHUNK): every member's endow
    // carries the intra-chunk kick mesh (built inside the spawn, where
    // the bands are known), so a barrier leader wakes a parked sibling
    // in one hop; starless chunks keep the descriptor's full width.
    let mut workers: [Option<crate::thread::Thread>; N] = [const { None }; N];
    let chunk = if star_plan.is_some() {
        crate::star::MESH_CHUNK
    } else {
        crate::session::MAX_SESSION_FOUNDS
    };
    let mut at = 0usize;
    let mut ok = true;
    while at < active {
        let end = (at + chunk).min(active);
        let founded = crate::thread::spawn_gang_workers(
            entry,
            ctx,
            at as u32,
            true,
            &[],
            star_plan.as_ref(),
            &mut workers[at..end],
        );
        if !founded {
            ok = false;
            break;
        }
        at = end;
    }
    // The hub last — its sends are named by the workers' BAND indices,
    // known only now.  A refused hub is the ruled degrade: the workers
    // keep their doorbells (parks bounded by the futex slice), no hub.
    let star = match (ok, star_plan) {
        (true, Some(plan)) if hub => {
            let mut bands = [(0usize, 0u32); N];
            let mut known = true;
            for (w, band) in bands.iter_mut().enumerate().take(active) {
                match workers[w].as_ref() {
                    Some(t) => *band = (w, t.band_index()),
                    None => {
                        known = false;
                        break;
                    }
                }
            }
            let mut hub_caps = [crate::thread::WorkerCap {
                role: 0,
                index: 0,
                subject: kore_abi::session::SlotRef::from_wire(0),
                kind: 0,
                rights: kore_abi::cap::Rights::empty(),
            }; crate::session::flows::MAX_THREAD_ENDOW];
            let mut hub_out: [Option<crate::thread::Thread>; 1] = [None];
            let founded_hub = known
                && plan
                    .hub_endow(&bands[..active], &mut hub_caps)
                    .map(|filled| {
                        crate::thread::spawn_gang_workers(
                            crate::star::hub_main,
                            ctx,
                            crate::star::HUB,
                            false,
                            &[&hub_caps[..filled]],
                            None,
                            &mut hub_out,
                        )
                    })
                    .unwrap_or(false);
            let hub = if founded_hub {
                hub_out[0].take()
            } else {
                // A refused hub founding never entered user code (the
                // gate is still staged and its found failed), so a
                // minted ticket names a vacant handle: join refuses
                // fast, detach leaves the claims spent — visible.
                if let Some(t) = hub_out[0].take() {
                    match crate::thread::join_reserved(t, &mut join_scratch) {
                        Ok(_) => {}
                        Err(e) => e.into_thread().detach(),
                    }
                }
                None
            };
            Some(plan.into_star(hub, join_scratch.report_slot()))
        }
        (_, plan) => plan.map(|p| p.into_star(None, join_scratch.report_slot())),
    };
    if !ok {
        // ── Abort: no worker has passed the gate, so every founded
        //    worker exits promptly once told.  A partially-founded chunk
        //    still minted tickets, so its survivors join here too; a
        //    ticket whose founding never happened refuses its join and
        //    detaches (the band stays spent — visible, never hidden). ──
        decide(ctx, ABORT);
        for staged in workers.iter_mut().filter_map(Option::take) {
            // A refused join here is a transient observation failure on
            // a worker that is already exiting; one bounded retry, then
            // an honest detach.
            match crate::thread::join_reserved(staged, &mut join_scratch) {
                Ok(_) => {}
                Err(e) => match crate::thread::join_reserved(e.into_thread(), &mut join_scratch) {
                    Ok(_) => {}
                    Err(e) => e.into_thread().detach(),
                },
            }
        }
        // The minted star never wired anything that ran: retire it.
        if let Some(star) = star {
            // This abort star has no founded hub, so the target
            // implementation cannot fail; keep the result explicit if that
            // invariant changes later.
            let _ = star.finish();
        }
        return None;
    }
    decide(ctx, RUN);
    // The release kicks: every worker (and the hub) parked at the gate
    // wakes over the parent-held sends — the launch never depends on
    // the hub it just founded.
    if let Some(star) = &star {
        star.kick_all();
    }
    Some(Spmd {
        workers,
        ctx,
        join_scratch,
        star,
    })
}

// ── The resident region: a founded-once engine, a stream of jobs (R9) ──

/// **Run a founded worker's resident loop** (R9, `res/SPMD.md` §14) — the
/// body a resident region's worker entry calls, turning a one-shot worker
/// into a lane that serves job after job.
///
/// A resident worker's entry is an ordinary [`SpmdEntry`] (founded by
/// [`found_resident`]) whose whole body is a call to this: it parks on the
/// region's [`job`](SpmdCtx::job_word) word, runs `body(ctx, worker)` for
/// each job [`ResidentRegion::submit`] publishes, signals completion, and
/// returns only when the region is retired (the founding gate leaves
/// `RUN`).  `body` reads the job's data off [`ctx.env()`](SpmdCtx::env) — a
/// pointer the submitter owns and keeps live until it observes completion —
/// exactly as [`dispatch_worker`] does; the loop is the one piece a backend
/// does not rewrite, so a device row and the witness share it.
///
/// Public and target-neutral so a backend's entry can name it and the host
/// suite can drive it against a scripted submitter; it founds nothing.
pub fn resident_run<F: Fn(&'static SpmdCtx, u32)>(ctx: &'static SpmdCtx, worker: u32, body: F) {
    // `ran` is the generation this worker last executed; 0 is the
    // founded-idle state, which no submit uses (they start at 1).
    let mut ran = 0u32;
    loop {
        // Park until a new job is published or the region is retired.  The
        // wait is the barrier's two-tier slice: a co-resident submit ends
        // the monitored spin the instant it stores `job`; a descheduled
        // worker rides `PARK_SLICE` and re-reads.  `kick_all` only trims
        // the tail — the word is the truth.
        crate::barrier::wait_while(ctx.job_word(), ran);
        // A retire flips the gate off RUN, THEN steps `job` to wake us; the
        // SeqCst total order makes the DONE visible once the step that woke
        // us is.
        if ctx.gate_word().load(Ordering::SeqCst) != RUN {
            return;
        }
        let g = ctx.job_word().load(Ordering::SeqCst);
        ran = g;
        // A job narrower than the founded gang wakes the whole gang but runs
        // its body on only `active` of it; the rest sit the job out.
        if worker < ctx.active() {
            body(ctx, worker);
        }
        // EVERY founded worker accounts for the job — an active one after its
        // body, one sitting out immediately — so `done` is armed to the FULL
        // gang width (see `submit`) and `complete` advances only once ALL of
        // them have passed the `active` check above. That is the fence on the
        // next `submit`: its physical-completion gate now waits for every
        // worker, so a worker that has loaded `g` cannot still be deciding
        // its participation (or reading `env`) when the next job overwrites
        // `active`/`env` — the generation race the per-job words would
        // otherwise carry. The one that zeroes `done` publishes the
        // completion generation; no wake is posted — no worker holds a send
        // to the launcher (the star's sends are founder→worker only), so the
        // launcher learns through its own monitored spin on `complete`
        // (co-resident: at once) or its slice re-read (descheduled: bounded).
        if ctx.done_word().fetch_sub(1, Ordering::AcqRel) == 1 {
            ctx.complete_word().store(g, Ordering::Release);
        }
    }
}

/// **A resident SPMD region** (R9) — a founded gang that outlives any one
/// job, dispatched by [`submit`](Self::submit) and fenced by
/// [`poll`](Self::poll)/[`wait`](Self::wait) rather than by founding a
/// fresh gang per launch.
///
/// This is the execution-engine shape the accel protocol's device rows
/// already have (`res/SPMD.md` R8): a scarce, founded-once *engine* that a
/// cheap stream of *jobs* runs over, its completion a memory word the
/// submitter reads with no join.  [`spawn_spmd`]/[`dispatch`] stay the
/// one-shot API; this is for a caller that submits many jobs to one gang —
/// the accel `CPU-SIMD` backend, a service's compute region.
///
/// Affine and single-submitter: the handle owns its gang, so `submit`
/// takes `&mut self` and exactly one job is in flight at a time (the next
/// `submit` refuses until the last completes).  Dropping it detaches the
/// workers — they stay resident for the task's life, exactly as a dropped
/// [`Spmd`] leaks its bands; [`retire`](Self::retire) is the deliberate
/// teardown that reaps them.
#[must_use = "dropping a ResidentRegion detaches its workers: they stay resident for the task's life"]
pub struct ResidentRegion<const N: usize> {
    /// The founded gang and its wake fabric — a hubless star, so the
    /// founder kicks workers directly on submit and an idle gang wakes
    /// nobody (`found_resident` passes `hub: false`).
    region: SpawnedSpmd<N>,
    /// Public-ticket identity and the current compact worker generation.
    tickets: ResidentTickets,
}

/// Opaque identity of one admitted resident job.
///
/// The worker protocol intentionally keeps its cache-line-isolated `u32`
/// generation, but that physical value wraps and is therefore not a lawful
/// long-lived public identity. A ticket instead carries a nonzero monotonic
/// sequence plus the static context that minted it. Its fields are private:
/// safe code cannot manufacture generation zero, an ahead-of-submit value, or
/// retarget a sequence to another context. Tickets are observation tokens, not
/// authority, so copying one is harmless.
#[derive(Clone, Copy)]
pub struct ResidentTicket {
    ctx: &'static SpmdCtx,
    sequence: core::num::NonZeroU64,
}

impl ResidentTicket {
    fn new(ctx: &'static SpmdCtx, sequence: u64) -> Option<Self> {
        Some(Self {
            ctx,
            sequence: core::num::NonZeroU64::new(sequence)?,
        })
    }
}

impl core::fmt::Debug for ResidentTicket {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ResidentTicket")
            .field("ctx", &(self.ctx as *const SpmdCtx))
            .field("sequence", &self.sequence)
            .finish()
    }
}

impl PartialEq for ResidentTicket {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.ctx, other.ctx) && self.sequence == other.sequence
    }
}

impl Eq for ResidentTicket {}

/// Why a resident ticket cannot be observed through this region handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentTicketError {
    /// The ticket was minted by a different [`SpmdCtx`].
    ForeignRegion,
    /// The ticket predates this resident epoch over the same static context.
    StaleEpoch,
    /// This epoch has not issued the ticket (including a value ahead of its
    /// latest admitted job).
    NotIssued,
}

/// Ticket classification after the context/epoch/issuance proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentTicketPosition {
    /// A later serial job was admitted, which proves this one completed first.
    Complete,
    /// The latest admitted job, named by its compact worker generation.
    Current(u32),
}

/// The owner-local bridge between non-reused public identities and the compact
/// physical generation resident workers monitor.
struct ResidentTickets {
    ctx: &'static SpmdCtx,
    /// Every sequence at or below this value belongs to an earlier resident
    /// epoch over the same static context.
    floor: u64,
    last: Option<ResidentTicket>,
    /// Physical generation of `last`; zero before this epoch's first submit.
    generation: u32,
}

impl ResidentTickets {
    fn new(ctx: &'static SpmdCtx) -> Self {
        Self {
            ctx,
            floor: ctx.resident_sequence(),
            last: None,
            generation: 0,
        }
    }

    /// Admit one logical identity and step the compact physical generation.
    /// Once identity allocation succeeds, no later submit step is fallible.
    fn issue(&mut self) -> Option<(ResidentTicket, u32)> {
        let ticket = self.ctx.next_resident_ticket()?;
        let generation = match self.generation.wrapping_add(1) {
            0 => 1,
            next => next,
        };
        self.last = Some(ticket);
        self.generation = generation;
        Some((ticket, generation))
    }

    fn position(
        &self,
        ticket: ResidentTicket,
    ) -> Result<ResidentTicketPosition, ResidentTicketError> {
        if !core::ptr::eq(ticket.ctx, self.ctx) {
            return Err(ResidentTicketError::ForeignRegion);
        }
        if ticket.sequence.get() <= self.floor {
            return Err(ResidentTicketError::StaleEpoch);
        }
        let Some(last) = self.last else {
            return Err(ResidentTicketError::NotIssued);
        };
        match ticket.sequence.get().cmp(&last.sequence.get()) {
            core::cmp::Ordering::Less => Ok(ResidentTicketPosition::Complete),
            core::cmp::Ordering::Equal => Ok(ResidentTicketPosition::Current(self.generation)),
            core::cmp::Ordering::Greater => Err(ResidentTicketError::NotIssued),
        }
    }

    fn poll(&self, ticket: ResidentTicket, complete: u32) -> Result<bool, ResidentTicketError> {
        Ok(match self.position(ticket)? {
            ResidentTicketPosition::Complete => true,
            ResidentTicketPosition::Current(generation) => complete == generation,
        })
    }
}

impl<const N: usize> ResidentRegion<N> {
    /// The shared anchor this region runs over.
    pub fn ctx(&self) -> &'static SpmdCtx {
        self.region.ctx()
    }

    /// The founded gang width — workers that exist; a job may run fewer.
    pub fn width(&self) -> usize {
        self.region.live()
    }

    /// Whether idle workers have the doorbells needed to park. A long-lived
    /// engine should retire a starless founding instead of keeping spinning
    /// workers resident for the lifetime of an otherwise idle application.
    pub fn star_active(&self) -> bool {
        self.region.star_active()
    }

    /// **Submit a job** to the resident gang: run the body `env` names on
    /// workers `0..active`, striding `span` (0 = none).  Non-blocking —
    /// returns the job's opaque ticket for
    /// [`poll`](Self::poll)/[`wait`](Self::wait), or `None` if the gang is
    /// not idle (a job is still in flight) or `active` exceeds the founded
    /// width. `None` also becomes the permanent answer if this static
    /// context's `u64` ticket identity space is exhausted; identities never
    /// wrap or reuse.
    ///
    /// `env` is the pointer a resident worker's body reads its job data
    /// from (what [`resident_run`] hands the body via [`ctx.env()`](SpmdCtx::env)):
    /// for a `&dyn Fn` body it is a pointer to that reference; for a
    /// data-descriptor body (the accel backend) it is a pointer to the
    /// descriptor.  Either way the submitter owns it.
    ///
    /// # Safety
    /// Whatever `env` points at — and anything it transitively borrows —
    /// must stay valid until this job's completion is observed by
    /// `poll(ticket)`/`wait(ticket)`. A blocking submitter satisfies this by
    /// waiting before dropping it; a non-blocking one (the accel backend) owns
    /// the descriptor in a slot it frees only once `poll` reports done.
    pub unsafe fn submit(
        &mut self,
        active: u32,
        span: u32,
        env: *mut (),
    ) -> Option<ResidentTicket> {
        if active == 0 || active as usize > self.region.live() {
            return None;
        }
        let ctx = self.region.ctx();
        // One job in flight: refuse until the last one retired.
        if ctx.complete_word().load(Ordering::Acquire) != self.tickets.generation {
            return None;
        }
        // Allocate the non-reused public identity before the physical
        // acceptance edge, then step the compact worker generation (skipping
        // its founded-idle zero). Nothing below this point can refuse.
        let (ticket, generation) = self.tickets.issue()?;
        // Arm the per-job words BEFORE publishing `job`: the step's release
        // edge is what makes `done`/`env`/`span`/parties visible to a waking
        // worker. `done` counts the FULL founded gang, not just this job's
        // `active` subset: every worker decrements it once (see
        // `resident_run`), which is what makes `complete` — and so the next
        // submit's physical-generation gate — wait for every worker to pass
        // this job's `active` check before its `active`/`env` may be
        // overwritten.
        ctx.done_word()
            .store(self.region.live() as u32, Ordering::Relaxed);
        ctx.arm(active, span, env);
        ctx.job_word().store(generation, Ordering::SeqCst);
        // Wake the gang over the founder's parent sends.  A co-resident
        // worker is already leaving its monitored spin on the store above;
        // this trims the descheduled tail.  A starless region has no sends
        // and rides the workers' slice re-read — correct, just slower.
        if let Some(star) = self.region.star.as_ref() {
            star.kick_all();
        }
        Some(ticket)
    }

    /// Whether `ticket`'s job has completed — the non-blocking fence a device
    /// row's `poll_event` wants.
    ///
    /// A retained ticket from this epoch stays complete forever: logical `u64`
    /// ordering proves it preceded the current serial job, independently of
    /// any wrapped/reused worker generation. A foreign-context, previous-epoch,
    /// or never-issued ticket returns an explicit [`ResidentTicketError`].
    pub fn poll(&self, ticket: ResidentTicket) -> Result<bool, ResidentTicketError> {
        let complete = self.region.ctx().complete_word().load(Ordering::Acquire);
        self.tickets.poll(ticket, complete)
    }

    /// Block until `ticket`'s job completes. Efficient for a co-resident gang
    /// (the completion store ends the monitored spin at once); a descheduled
    /// submitter rides the barrier's slice and re-reads, because no worker
    /// holds a send to wake it.
    ///
    /// Validation runs before any wait, so an invalid ticket can never park on
    /// a physical generation that this region will not publish.
    pub fn wait(&self, ticket: ResidentTicket) -> Result<(), ResidentTicketError> {
        match self.tickets.position(ticket)? {
            ResidentTicketPosition::Complete => Ok(()),
            ResidentTicketPosition::Current(generation) => {
                wait_for_resident_generation(self.region.ctx(), generation);
                Ok(())
            }
        }
    }

    /// Observe completion without blocking an executor's sibling futures or
    /// consuming its standing doorbell. The futex supplies a short monitored
    /// prelude; longer waits yield to the executor. Workers have no reverse
    /// completion send, so a bounded park is mandatory even with a live star.
    /// A timeout or spurious wake is never completion: re-read the ticket.
    ///
    /// Like `wait`, rejects foreign/stale/unissued tickets before any wait.
    /// Cancelling this future cancels only observation, never the CPU job or
    /// its descriptor-lifetime obligation.
    pub async fn wait_async(&self, ticket: ResidentTicket) -> Result<(), ResidentTicketError> {
        self.tickets.position(ticket)?;
        let complete = self.region.ctx().complete_word();
        loop {
            let observed = complete.load(Ordering::Acquire);
            if self.tickets.poll(ticket, observed)? {
                return Ok(());
            }
            let outcome = crate::futex::wait_async(
                complete,
                observed,
                Some(core::time::Duration::from_micros(250)),
            )
            .await;
            if outcome == crate::futex::WaitOutcome::Unsupported {
                crate::executor::next_wake_for(250_000).await;
            }
        }
    }

    /// **Retire the region**: wake the gang, tell it to exit, and join
    /// every worker — reclaiming its band, stack, and the star.  The
    /// deliberate teardown a dropped handle skips.
    pub fn retire(self) -> Result<(), SpmdJoinError<SpawnedTicket, N>> {
        // Drain any in-flight job before teardown. A retire that raced an
        // accepted job would flip the gate off RUN and step `job`; a worker
        // that had not yet reached its gate check would then wake, observe
        // DONE, and return WITHOUT decrementing `done` — so `join_all` would
        // succeed while that job's `complete` never reached its generation
        // and its `env` was never observed done, silently abandoning it.
        // Waiting for the last submitted job first makes the region idle, so
        // the teardown abandons nothing. (Idle already ⇒ this returns at once.)
        if self.tickets.last.is_some() {
            wait_for_resident_generation(self.region.ctx(), self.tickets.generation);
        }
        let ctx = self.region.ctx();
        // Tell the gang to exit: flip the gate off RUN, THEN step `job` to
        // wake every parked worker (each observes DONE and returns), and
        // kick so a parked worker need not ride its slice to see it.
        ctx.gate_word().store(DONE, Ordering::SeqCst);
        ctx.job_word().fetch_add(1, Ordering::SeqCst);
        if let Some(star) = self.region.star.as_ref() {
            star.kick_all();
        }
        // join_all reaps every worker and finishes the star; its own DONE
        // store is idempotent with ours.
        self.region.join_all().map(|_| ())
    }
}

/// Wait for the one compact physical generation already proved current by the
/// opaque ticket layer. Keeping this separate lets `retire` drain its own last
/// accepted job without manufacturing or re-validating a public token.
fn wait_for_resident_generation(ctx: &'static SpmdCtx, generation: u32) {
    let complete = ctx.complete_word();
    loop {
        let current = complete.load(Ordering::Acquire);
        if current == generation {
            return;
        }
        crate::barrier::wait_while(complete, current);
    }
}

/// **Found a resident region** of `width` workers running `entry` over
/// `ctx`, all parked and idle, ready for [`submit`](ResidentRegion::submit).
///
/// Reuses the one-shot founding path ([`launch`]) — the gang, the stack
/// arena, the class-best pins, the star's parent sends — but hubless and
/// with no job armed (span 0, env null): each worker's entry parks on `job`
/// at once and runs nothing until the first submit steps it.  `entry` is an
/// ordinary [`SpmdEntry`] whose body calls [`resident_run`] with the
/// backend's per-job body.  `ctx.gate` must be freshly staged and
/// `ctx.capacity() >= width`.
#[cfg(target_os = "none")]
pub fn found_resident<const N: usize>(
    ctx: &'static SpmdCtx,
    entry: SpmdEntry,
    width: usize,
) -> Option<ResidentRegion<N>> {
    let region = launch::<N>(ctx, entry, width, 0, core::ptr::null_mut(), false)?;
    Some(ResidentRegion {
        region,
        tickets: ResidentTickets::new(ctx),
    })
}

/// Off the metal userland targets — the honest typed refusal, as
/// [`spawn_spmd`]: no real workers, so no resident engine to found.
#[cfg(not(target_os = "none"))]
pub fn found_resident<const N: usize>(
    _ctx: &'static SpmdCtx,
    _entry: SpmdEntry,
    _width: usize,
) -> Option<ResidentRegion<N>> {
    None
}

// ── Auto-dispatch: the parallel-for that sizes itself (`res/SPMD.md` §6) ──

/// The capacity of the shared dispatch region — the runtime's worker
/// ceiling, so [`dispatch`] can fit any width the machine offers.
#[cfg(target_os = "none")]
const DISPATCH_CAP: usize = (crate::thread::MAX_THREADS - 1) as usize;

/// The one re-armable region every [`dispatch`] launches over. A single
/// `static` serves every call: each blocks to join before returning and
/// re-stages the ctx for the next, so a parallel-for in a loop reuses
/// it. A second CONCURRENT dispatch finds the gate un-staged and refuses
/// (`None`) rather than corrupting the region — v1 dispatch is one
/// region at a time per process, the same all-or-nothing gate every
/// launch respects.
#[cfg(target_os = "none")]
static DISPATCH_CTX: SpmdCtx = SpmdCtx::new(DISPATCH_CAP as u32);

/// What a [`dispatch`] launch chose — the width it fit to and the tile
/// span it covered, so a terminal tool can show the auto-decision the caller
/// never had to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
pub struct Dispatched {
    /// Workers founded — the fitting width for this machine and tile
    /// count (0 for an empty grid, which launches nothing).
    pub width: u32,
    /// Tiles covered.
    pub tiles: u32,
}

/// The fixed worker entry every dispatch launches: read the launcher's
/// type-erased body off the ctx and call it. One monomorphic
/// [`SpmdEntry`], so the start record carries no environment — the body
/// rides `ctx.env()` instead (the start record is full at 24 bytes).
#[cfg(target_os = "none")]
fn dispatch_worker(ctx: &'static SpmdCtx, worker: u32) {
    let env = ctx.env();
    if env.is_null() {
        return;
    }
    // SAFETY: `env` is a thin pointer to the launcher's `&dyn Fn`, stored
    // by `arm` before the gate opened (published to this worker by the
    // gate's release edge, read after `enter`) and live until
    // `join_all` returns — the launcher blocks on it, so the closure
    // outlives every worker's call. Workers `SHARE_SPACE`, so the
    // pointer is dereferenceable here; the body is `+ Sync`, so the
    // concurrent calls from every worker are sound.
    let body: &(dyn Fn(&'static SpmdCtx, u32) + Sync) =
        unsafe { *(env as *const &(dyn Fn(&'static SpmdCtx, u32) + Sync)) };
    body(ctx, worker);
}

/// **The width model** — every fact a dispatch width is chosen from, and
/// [`choose`](Self::choose) is the only place the choice is made.
///
/// The fit policy (ruling 2026-08-20) reads one fact: the best lane.  That
/// is the right default precisely when nothing else is known, and it is
/// what this model returns when nothing else is supplied.  What it cannot
/// see is that a gang costs more to *start* the wider it is, so a small
/// grid can finish sooner on fewer workers — measured on a Lunar Lake box
/// 2026-09-03 with `spmd lane <width>` (median of the per-round minimum,
/// 64 rounds, identical body at every width):
///
/// | width | median min | median avg |
/// |---|---|---|
/// | 2 | 13.5 µs | 259 µs |
/// | 3 | 35.7 µs | 444 µs |
/// | 4 | 109 µs  | 550 µs |
///
/// The body was fixed-size, so all of that is wake-and-drain: more parties
/// on the barrier's completion count, more cache-line traffic on the job
/// word, more workers that may be descheduled and riding `PARK_SLICE`
/// rather than ending a monitored spin.  Width 4 costs 2.1× the average
/// and 8× the floor of width 2 to dispatch one batch.
///
/// So the choice is an argmin over a two-term cost, not a machine fact:
///
/// ```text
/// T(w) = overhead_ns[w - 1] + ceil(tiles / w) * per_tile_ns
/// ```
///
/// Both terms have to be known for that to beat the fit policy, and they
/// come from different places: `overhead_ns` is a machine property
/// (workload-independent, calibrated once), `per_tile_ns` is a workload
/// property (per call site).  Absent either, `choose` returns the fit
/// width — today's answer, exactly — so this model is inert until both are
/// supplied.  `res/SPMD.md` §12 carries the open question of where
/// `per_tile_ns` comes from; nothing in the tree feeds it yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WidthModel<'a> {
    /// The fit width: one worker per exclusive best-class core
    /// ([`Machine::fitting_width`]), already clamped to the runtime
    /// ceiling by the caller.
    pub fit: u32,
    /// Measured per-batch dispatch overhead in nanoseconds, indexed by
    /// `width - 1` — the wake-and-drain cost `spmd lane <width>` prints.
    /// Empty (the only state in-tree today) means uncalibrated.
    pub overhead_ns: &'a [u32],
}

impl WidthModel<'_> {
    /// The width to launch `tiles` at, given what is known about the cost
    /// of one tile.
    ///
    /// Never exceeds the fit width or the tile count, and never returns
    /// zero — a width is a party count, and a launch of none is not a
    /// launch.  Ties go to the **narrower** gang: equal predicted time on
    /// fewer cores is strictly better (less energy, more machine left for
    /// the next tenant), and the prediction's error bar is smaller where
    /// the overhead term is smaller.
    ///
    /// Pure arithmetic — no clock, no machine read, no allocation — so the
    /// policy is host-proven rather than metal-only truth.
    pub fn choose(&self, tiles: u32, per_tile_ns: Option<u32>) -> u32 {
        // The fit policy's answer, and the ceiling every other answer
        // lives under: never more workers than tiles, never zero.
        let ceiling = self.fit.min(tiles).max(1);
        let Some(per_tile_ns) = per_tile_ns else {
            return ceiling;
        };
        if self.overhead_ns.len() < ceiling as usize {
            // Uncalibrated for some width in range: refuse to compare
            // widths on a partial table rather than guess the gaps.
            return ceiling;
        }
        let mut best = ceiling;
        let mut best_ns = u64::MAX;
        // Ascending, with a strict improvement test, so the narrowest
        // width achieving the minimum wins.
        for width in 1..=ceiling {
            let overhead = u64::from(self.overhead_ns[width as usize - 1]);
            let per_worker = u64::from(tiles.div_ceil(width));
            let total = overhead.saturating_add(per_worker * u64::from(per_tile_ns));
            if total < best_ns {
                best_ns = total;
                best = width;
            }
        }
        best
    }
}

/// Pick the launch width for `tiles` under the **fit policy** (ruling
/// 2026-08-20): one worker per exclusive best-class core (`best_lane`),
/// so the region runs co-resident in the monitored tier — never more
/// workers than tiles, never past the runtime ceiling, at least one. A
/// machine with no facts stamped falls back to a single worker (correct,
/// if unparallel — the honest floor rather than a guessed width).
///
/// (The adaptive fill-past-a-threshold policy the design weighed is a
/// future knob on exactly this one function — `res/SPMD.md` §12.)
#[cfg(target_os = "none")]
fn dispatch_width(tiles: u32) -> u32 {
    let lane = machine()
        .map(|m| m.best_lane)
        .filter(|&l| l > 0)
        .unwrap_or(1);
    // No calibration and no per-tile cost: the model returns the fit
    // width, which is what this function has always returned. The
    // equivalence is host-proven (`the_width_model_reproduces_the_fit_policy`),
    // so routing through the model changes no behavior — it only puts the
    // policy somewhere a ruling can reach.
    WidthModel {
        fit: lane.clamp(1, DISPATCH_CAP as u32),
        overhead_ns: &[],
    }
    .choose(tiles, None)
}

/// Even block split of `span` items across `workers`: worker `w` gets
/// the contiguous `[start, end)` slice, remainder spread over the first
/// workers. Contiguous (not strided) — a range body reads its slice in
/// order, which is the locality a per-slice consumer wants.  Pure
/// arithmetic (unconditional), so the partition laws are host-proven.
#[cfg_attr(all(not(target_os = "none"), not(test)), allow(dead_code))]
fn block_split(span: u32, workers: u32, worker: u32) -> (u32, u32) {
    let base = span / workers;
    let rem = span % workers;
    let start = worker * base + worker.min(rem);
    let len = base + if worker < rem { 1 } else { 0 };
    (start, start + len)
}

/// **The parallel-for** — run `body` over `0..tiles`, sized to the
/// machine, with no thread configuration at all.
///
/// The one call that replaces the hand-rolled launch: [`dispatch`] reads
/// [`machine`], picks a fitting width (one worker per exclusive
/// best-class core), founds that many workers over a shared re-armable
/// region, splits the tiles across them, runs `body` on each, and blocks
/// until every tile is done. `None` only if the launch itself refused
/// (a second concurrent dispatch, or an exhausted machine); an empty
/// grid is `Some` with width 0.
///
/// ```ignore
/// kore_rt::spmd::dispatch(rows, |row| shade(row));   // that's the whole thing
/// ```
///
/// `body` runs on several threads at once, so it takes `&self`-style
/// shared state (`Sync`); a reduction that needs per-worker partials and
/// a combine is [`reduce`]. For a body that wants its whole contiguous
/// slice at once (to vectorize, or amortize a setup), see
/// [`dispatch_slices`].
#[cfg(target_os = "none")]
pub fn dispatch<F: Fn(u32) + Sync>(tiles: u32, body: F) -> Option<Dispatched> {
    dispatch_slices(tiles, move |range: core::ops::Range<u32>| {
        for tile in range {
            body(tile);
        }
    })
}

/// [`dispatch`] handing each worker its **contiguous slice** instead of
/// one tile at a time — for a body that vectorizes or batches over a
/// range. Same auto-sizing, same block until done.
///
/// ```ignore
/// kore_rt::spmd::dispatch_slices(n, |range| simd_sum(&data[range]));
/// ```
#[cfg(target_os = "none")]
pub fn dispatch_slices<F: Fn(core::ops::Range<u32>) + Sync>(
    tiles: u32,
    body: F,
) -> Option<Dispatched> {
    if tiles == 0 {
        return Some(Dispatched { width: 0, tiles: 0 });
    }
    dispatch_slices_at(dispatch_width(tiles), tiles, body)
}

/// [`dispatch_slices`] at a caller-chosen `width` — the measurement form,
/// as [`reduce_at`] is for [`reduce`]. Clamped like the derived width.
#[cfg(target_os = "none")]
pub fn dispatch_slices_at<F: Fn(core::ops::Range<u32>) + Sync>(
    width: u32,
    tiles: u32,
    body: F,
) -> Option<Dispatched> {
    if tiles == 0 {
        return Some(Dispatched { width: 0, tiles: 0 });
    }
    let width = width.min(tiles).clamp(1, DISPATCH_CAP as u32);
    // Wrap the user's per-slice body into the fixed worker signature: the
    // worker reads its launched width and tile span off the ctx (armed by
    // `launch`), computes its block, and runs the body on it.
    let worker_body = move |ctx: &'static SpmdCtx, worker: u32| {
        let (start, end) = block_split(ctx.span(), ctx.workers(), worker);
        if start < end {
            body(start..end);
        }
    };
    dispatch_erased(tiles, width, &worker_body)
}

/// The shared launch+join+restage every dispatch shape funnels through,
/// over an already-erased worker body. The caller owns the closure the
/// `dyn Fn` borrows and MUST outlive this call — which it does, because
/// this blocks on `join_all` before returning.
///
/// `body` is the fixed [`SpmdEntry`]-shaped worker closure (it reads its
/// slice off the ctx); `dispatch_worker` reads it back off `env` and
/// calls it. One place owns the join-to-completion rule and the re-stage, so
/// `dispatch`, `dispatch_slices`, and `reduce` cannot drift on them.
#[cfg(target_os = "none")]
fn dispatch_erased(
    tiles: u32,
    width: u32,
    body: &(dyn Fn(&'static SpmdCtx, u32) + Sync),
) -> Option<Dispatched> {
    let env = &body as *const &(dyn Fn(&'static SpmdCtx, u32) + Sync) as *mut ();
    let region = launch::<DISPATCH_CAP>(
        &DISPATCH_CTX,
        dispatch_worker,
        width as usize,
        tiles,
        env,
        true,
    )?;
    // Once the gate released, `body` is borrowed by every worker through
    // `ctx.env()`.  A failed join learned no terminal fact and hands the
    // affine tickets back; returning (or re-staging) at that point would let
    // detached workers dereference this stack borrow after it died.  Retain
    // both the region and this frame until every ticket is collected.  `None`
    // therefore remains exactly the pre-launch refusal documented by
    // `dispatch`: post-launch lifecycle failure is fail-stopped here, never
    // converted into a serial replay over potentially live workers.
    join_dispatched(region);
    DISPATCH_CTX.restage();
    Some(Dispatched { width, tiles })
}

/// Collect a released dispatch without ever detaching one of its workers.
///
/// There is intentionally no retry budget.  [`WorkerTicket::join`] returns the
/// ticket when it learned no terminal fact, and the dispatch body is a borrow
/// from the caller's stack.  No finite number of refusals can make dropping
/// that ticket or returning the borrowed frame sound.
#[cfg(any(target_os = "none", test))]
fn join_dispatched<T: WorkerTicket, const N: usize>(mut region: Spmd<T, N>) {
    loop {
        match region.join_all() {
            Ok(_) => return,
            Err(refused) => {
                region = refused.spmd;
                dispatch_join_backoff();
            }
        }
    }
}

/// Pace a join refusal so bounded waiter/slot pressure cannot become a trap
/// storm while the dispatch retains its live tickets.
#[cfg(target_os = "none")]
fn dispatch_join_backoff() {
    if let Some(word) = crate::cap::verbs::sleep_until(1) {
        let now_ms = word >> 32;
        let _ = crate::cap::verbs::sleep_until((now_ms + 2) * 1_000_000);
    } else {
        core::hint::spin_loop();
    }
}

#[cfg(test)]
fn dispatch_join_backoff() {}

/// Per-worker partials for [`reduce`] — worker `w` owns cell `w`, so the
/// stores never alias and the launcher folds `[0, width)` after join.
#[cfg(target_os = "none")]
static REDUCE_PARTIALS: [core::sync::atomic::AtomicU64; DISPATCH_CAP] =
    [const { core::sync::atomic::AtomicU64::new(0) }; DISPATCH_CAP];

/// **The parallel reduction** — the canonical SPMD shape (stride,
/// partial, combine) as one call, sized to the machine.
///
/// Each worker folds its contiguous slice of `0..tiles` with `map` then
/// `combine`, starting from `identity`; the runtime folds the workers'
/// partials the same way and returns the total. No barrier, no leader
/// election, no partial cells, no join loop — the pattern
/// `spmd example` hand-writes, as a call. `None` only if the launch
/// refused; an empty grid returns `Some(identity)`.
///
/// ```ignore
/// let total = kore_rt::spmd::reduce(n, 0, |i| TABLE[i as usize] as u64, |a, b| a + b);
/// ```
///
/// `u64` is the fold type (sums, checksums, counts — the overwhelming
/// case); a general-`T` reduction is future scope (`res/SPMD.md` §12).
/// `combine` must be associative — the worker and cross-worker folds
/// group differently — which every reduction the shape is for already is.
#[cfg(target_os = "none")]
pub fn reduce<M, C>(tiles: u32, identity: u64, map: M, combine: C) -> Option<u64>
where
    M: Fn(u32) -> u64 + Sync,
    C: Fn(u64, u64) -> u64 + Sync,
{
    use core::sync::atomic::Ordering;
    if tiles == 0 {
        return Some(identity);
    }
    reduce_at(dispatch_width(tiles), tiles, identity, map, combine)
}

/// [`reduce`] at a caller-chosen `width`, for measuring the fit policy
/// rather than trusting it.
///
/// [`dispatch_width`] is one line of policy on which every parallel body
/// in the system depends, and its own note calls the adaptive version "a
/// future knob on exactly this one function". A knob nobody can turn is a
/// knob nobody can measure, so this is the turn: same reduction, same
/// partition, the width named instead of derived. Clamped to the runtime
/// ceiling exactly as the derived width is — an override chooses among
/// lawful widths, it does not escape the bound.
#[cfg(target_os = "none")]
pub fn reduce_at<M, C>(width: u32, tiles: u32, identity: u64, map: M, combine: C) -> Option<u64>
where
    M: Fn(u32) -> u64 + Sync,
    C: Fn(u64, u64) -> u64 + Sync,
{
    use core::sync::atomic::Ordering;
    if tiles == 0 {
        return Some(identity);
    }
    let width = width.min(tiles).clamp(1, DISPATCH_CAP as u32);
    for cell in REDUCE_PARTIALS.iter().take(width as usize) {
        cell.store(identity, Ordering::Relaxed);
    }
    // Borrow (not move) `map`/`combine`: the worker closure needs them by
    // shared reference — it is `Sync`, called from every worker — and the
    // final fold below needs `combine` again. Both outlive the region;
    // `dispatch_erased` blocks on join before this frame returns.
    let worker_body = |ctx: &'static SpmdCtx, worker: u32| {
        let (start, end) = block_split(ctx.span(), ctx.workers(), worker);
        let mut acc = identity;
        for tile in start..end {
            acc = combine(acc, map(tile));
        }
        // This worker's own cell — no other worker writes it.
        REDUCE_PARTIALS[worker as usize].store(acc, Ordering::Relaxed);
    };
    let dispatched = dispatch_erased(tiles, width, &worker_body)?;
    // Every worker's store happens-before its exit, which happens-before
    // this thread's join returns — so the partials are all visible here.
    let mut total = identity;
    for cell in REDUCE_PARTIALS.iter().take(dispatched.width as usize) {
        total = combine(total, cell.load(Ordering::Relaxed));
    }
    Some(total)
}

/// Off the metal userland targets — the honest typed refusal, as
/// [`thread::spawn`](crate::thread::spawn).
#[cfg(not(target_os = "none"))]
pub fn spawn_spmd<const N: usize>(
    _ctx: &'static SpmdCtx,
    _entry: SpmdEntry,
) -> Option<SpawnedSpmd<N>> {
    None
}

/// Off metal — the parallel-for cannot found real workers, so it
/// refuses rather than running the body single-threaded behind your
/// back (`res/USERLAND.md` §2.1: no simulated world).
#[cfg(not(target_os = "none"))]
pub fn dispatch<F: Fn(u32) + Sync>(_tiles: u32, _body: F) -> Option<Dispatched> {
    None
}

/// Off metal — as [`dispatch`].
#[cfg(not(target_os = "none"))]
pub fn dispatch_slices<F: Fn(core::ops::Range<u32>) + Sync>(
    _tiles: u32,
    _body: F,
) -> Option<Dispatched> {
    None
}

#[cfg(not(target_os = "none"))]
pub fn dispatch_slices_at<F: Fn(core::ops::Range<u32>) + Sync>(
    _width: u32,
    _tiles: u32,
    _body: F,
) -> Option<Dispatched> {
    None
}

/// Off metal — as [`dispatch`]: no real workers, so the reduction
/// refuses rather than folding single-threaded behind your back.
#[cfg(not(target_os = "none"))]
pub fn reduce<M, C>(_tiles: u32, _identity: u64, _map: M, _combine: C) -> Option<u64>
where
    M: Fn(u32) -> u64 + Sync,
    C: Fn(u64, u64) -> u64 + Sync,
{
    None
}

#[cfg(not(target_os = "none"))]
pub fn reduce_at<M, C>(
    _width: u32,
    _tiles: u32,
    _identity: u64,
    _map: M,
    _combine: C,
) -> Option<u64>
where
    M: Fn(u32) -> u64 + Sync,
    C: Fn(u64, u64) -> u64 + Sync,
{
    None
}

/// This target's spawned-worker ticket — what [`spawn_spmd`] launches.
///
/// The one `cfg` alias of this module (the `Neutral` selection pattern):
/// macro-emitted launchers name [`SpawnedSpmd`] and compile on every
/// target, metal founding real [`Thread`](crate::thread::Thread)s and
/// host builds keeping the uninhabited refusal.
#[cfg(target_os = "none")]
pub type SpawnedTicket = crate::thread::Thread;

/// This target's spawned-worker ticket — the typed refusal.
#[cfg(not(target_os = "none"))]
pub type SpawnedTicket = NoWorker;

/// The region [`spawn_spmd`] yields on this target.
pub type SpawnedSpmd<const N: usize> = Spmd<SpawnedTicket, N>;

/// The uninhabitable host worker — [`spawn_spmd`]'s host arm returns a
/// `Spmd` no value can populate, the typed-refusal twin of a metal
/// `Thread` (`res/USERLAND.md` §2.1's posture at the type level).
#[cfg(not(target_os = "none"))]
#[derive(Debug)]
pub enum NoWorker {}

#[cfg(not(target_os = "none"))]
impl WorkerTicket for NoWorker {
    type JoinScratch = ();

    fn join(
        self,
        _scratch: &mut Self::JoinScratch,
    ) -> Result<Option<i32>, (Self, crate::cap::TaskWaitError)> {
        match self {}
    }

    fn detach(self) {
        match self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    #[test]
    #[cfg(not(target_os = "none"))]
    fn resident_async_observation_validates_before_touching_the_executor() {
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        static CTX: SpmdCtx = SpmdCtx::new(1);
        static OTHER: SpmdCtx = SpmdCtx::new(1);
        // No workers are simulated: this is solely the ticket/fast-observation
        // law, which must resolve before an executor or wait is needed.
        let mut region = ResidentRegion {
            region: Spmd::<NoWorker, 1> {
                workers: [None],
                ctx: &CTX,
                join_scratch: (),
                star: None,
            },
            tickets: ResidentTickets::new(&CTX),
        };
        let (ticket, generation) = region.tickets.issue().unwrap();
        CTX.complete_word().store(generation, Ordering::Release);
        let mut cx = Context::from_waker(Waker::noop());
        assert_eq!(
            core::pin::pin!(region.wait_async(ticket))
                .as_mut()
                .poll(&mut cx),
            Poll::Ready(Ok(())),
        );
        let foreign = ResidentTicket::new(&OTHER, 1).unwrap();
        assert_eq!(
            core::pin::pin!(region.wait_async(foreign))
                .as_mut()
                .poll(&mut cx),
            Poll::Ready(Err(ResidentTicketError::ForeignRegion)),
        );
        let ahead = ResidentTicket::new(&CTX, ticket.sequence.get() + 1).unwrap();
        assert_eq!(
            core::pin::pin!(region.wait_async(ahead))
                .as_mut()
                .poll(&mut cx),
            Poll::Ready(Err(ResidentTicketError::NotIssued)),
        );
    }

    /// The block split covers `0..span` exactly once — contiguous,
    /// gapless, and remainder-balanced (no worker carries two more tiles
    /// than another).
    /// The one property that makes this model safe to land: with no
    /// per-tile cost it is the fit policy, tile for tile, over the whole
    /// grid of lanes and tile counts a dispatch can see. The old body was
    /// `lane.min(tiles).clamp(1, cap)`.
    #[test]
    fn the_width_model_reproduces_the_fit_policy() {
        const CAP: u32 = 7;
        for lane in 0..=12u32 {
            for tiles in 0..=20u32 {
                let legacy = lane.min(tiles).clamp(1, CAP);
                let model = WidthModel {
                    fit: lane.clamp(1, CAP),
                    overhead_ns: &[],
                }
                .choose(tiles, None);
                assert_eq!(model, legacy, "lane={lane} tiles={tiles}");
            }
        }
    }

    /// A partially calibrated table is not a licence to interpolate: if any
    /// width in range has no measured overhead, the model declines to
    /// compare and hands back the fit width.
    #[test]
    fn a_short_overhead_table_falls_back_to_the_fit_width() {
        let model = WidthModel {
            fit: 4,
            overhead_ns: &[1_000, 2_000],
        };
        assert_eq!(model.choose(64, Some(10)), 4);
    }

    /// The measured Lunar Lake overheads (2026-09-03 floors, ns) against a
    /// cheap tile: the work cannot pay for a wide gang, so the model picks
    /// the narrow one the machine fact alone would have missed.
    #[test]
    fn a_cheap_grid_chooses_the_narrow_gang() {
        let model = WidthModel {
            fit: 4,
            overhead_ns: &[6_000, 13_500, 35_700, 109_000],
        };
        // 64 tiles at 100 ns each = 6.4 µs of work; every width past 1
        // adds more overhead than it removes work.
        assert_eq!(model.choose(64, Some(100)), 1);
    }

    /// The same machine, a tile expensive enough to amortize the wake: the
    /// model widens on its own. This is the case the fit policy already
    /// gets right, and the model must not regress it.
    #[test]
    fn an_expensive_grid_still_fills_the_fit_width() {
        let model = WidthModel {
            fit: 4,
            overhead_ns: &[6_000, 13_500, 35_700, 109_000],
        };
        // 64 tiles at 50 µs each = 3.2 ms of work — four ways beats two
        // by ~800 µs, far past the 95 µs the fourth worker costs to wake.
        assert_eq!(model.choose(64, Some(50_000)), 4);
    }

    /// Equal predicted time on fewer cores is the better answer, so the
    /// argmin's tie-break must favour the narrower gang.
    #[test]
    fn ties_go_to_the_narrower_gang() {
        // width 1: 0 + 4*100 = 400.  width 2: 200 + 2*100 = 400.
        let model = WidthModel {
            fit: 2,
            overhead_ns: &[0, 200],
        };
        assert_eq!(model.choose(4, Some(100)), 1);
    }

    /// A width is a party count: zero tiles still yields a launchable
    /// width, and no input drives the choice above the tile count.
    #[test]
    fn the_chosen_width_is_never_zero_and_never_exceeds_the_tiles() {
        let model = WidthModel {
            fit: 8,
            overhead_ns: &[0; 8],
        };
        assert_eq!(model.choose(0, Some(1_000)), 1);
        for tiles in 1..=8u32 {
            assert!(model.choose(tiles, Some(1_000)) <= tiles);
            assert!(model.choose(tiles, Some(1_000)) >= 1);
        }
    }

    #[test]
    fn block_split_partitions_the_span_exactly() {
        for span in 0u32..40 {
            for workers in 1u32..=8 {
                let mut prev_end = 0u32;
                let mut lens = [0u32; 8];
                for w in 0..workers {
                    let (start, end) = block_split(span, workers, w);
                    assert_eq!(start, prev_end, "gap or overlap at worker {w}");
                    assert!(end >= start, "reversed slice");
                    lens[w as usize] = end - start;
                    prev_end = end;
                }
                assert_eq!(prev_end, span, "the split did not cover the whole span");
                let (min, max) = (
                    lens[..workers as usize].iter().min().copied().unwrap(),
                    lens[..workers as usize].iter().max().copied().unwrap(),
                );
                assert!(max - min <= 1, "unbalanced split: {min}..={max}");
            }
        }
    }

    /// A scripted worker: joins yield the scripted outcomes in order.
    struct MockWorker {
        outcomes: &'static [Result<Option<i32>, crate::cap::TaskWaitError>],
        cursor: &'static Cursor,
    }

    struct Cursor(Cell<usize>);
    // SAFETY: single-threaded tests only.
    unsafe impl Sync for Cursor {}

    struct MockScratch(Cell<usize>);

    impl WorkerTicket for MockWorker {
        type JoinScratch = MockScratch;

        fn join(
            self,
            scratch: &mut Self::JoinScratch,
        ) -> Result<Option<i32>, (Self, crate::cap::TaskWaitError)> {
            scratch.0.set(scratch.0.get() + 1);
            let at = self.cursor.0.get();
            self.cursor.0.set(at + 1);
            match self.outcomes[at] {
                Ok(status) => Ok(status),
                Err(cause) => Err((self, cause)),
            }
        }

        fn detach(self) {}
    }

    fn group_of<const N: usize>(
        outcomes: &'static [Result<Option<i32>, crate::cap::TaskWaitError>],
        cursor: &'static Cursor,
        ctx: &'static SpmdCtx,
    ) -> Spmd<MockWorker, N> {
        Spmd {
            workers: core::array::from_fn(|_| Some(MockWorker { outcomes, cursor })),
            ctx,
            join_scratch: MockScratch(Cell::new(0)),
            star: None,
        }
    }

    /// The whole-region happy path: every status lands in worker order.
    #[test]
    fn join_all_collects_every_lane_in_order() {
        static CTX: SpmdCtx = SpmdCtx::new(3);
        static CURSOR: Cursor = Cursor(Cell::new(0));
        static OUTCOMES: [Result<Option<i32>, crate::cap::TaskWaitError>; 3] =
            [Ok(Some(0)), Ok(Some(7)), Ok(None)];
        let g = group_of::<3>(&OUTCOMES, &CURSOR, &CTX);
        assert_eq!(g.join_all().ok(), Some([Some(0), Some(7), None]));
    }

    /// A refusing worker hands the region back with the fold resumable:
    /// the retry joins only what remained, and nothing is joined twice.
    #[test]
    fn a_refused_join_is_resumable_and_joins_nobody_twice() {
        static CTX: SpmdCtx = SpmdCtx::new(3);
        static CURSOR: Cursor = Cursor(Cell::new(0));
        static OUTCOMES: [Result<Option<i32>, crate::cap::TaskWaitError>; 4] = [
            Ok(Some(1)),
            Err(crate::cap::TaskWaitError::Retryable),
            Ok(Some(2)), // worker 1's second attempt
            Ok(Some(3)),
        ];
        let g = group_of::<3>(&OUTCOMES, &CURSOR, &CTX);
        let err = g.join_all().err().expect("worker 1 refuses once");
        assert_eq!(err.worker, 1);
        assert!(matches!(err.cause, crate::cap::TaskWaitError::Retryable));
        assert_eq!(err.spmd.live(), 2, "worker 0 joined; 1 and 2 remain");
        assert_eq!(
            err.spmd.join_scratch.0.get(),
            2,
            "the same pre-launch scratch remains inside the refused region"
        );
        let statuses = err.spmd.join_all().expect("the retry completes");
        // Worker 1 and 2 joined on the retry; how worker 0's earlier status
        // reads here is the fold's documented choice — the invariant
        // this test pins is only that the retry consumed exactly the
        // two remaining scripted joins.
        let _ = statuses;
        assert_eq!(CURSOR.0.get(), 4, "exactly four joins ever attempted");
    }

    /// Once a dispatch has released its workers, a finite retry budget cannot
    /// make returning its stack-borrowed body safe.  Exercise more refusals
    /// than the former budget before the same affine ticket finally joins.
    #[test]
    fn a_dispatched_region_joins_past_any_old_retry_budget() {
        static CTX: SpmdCtx = SpmdCtx::new(1);
        static CURSOR: Cursor = Cursor(Cell::new(0));
        static OUTCOMES: [Result<Option<i32>, crate::cap::TaskWaitError>; 18] = {
            let mut outcomes = [Err(crate::cap::TaskWaitError::Retryable); 18];
            outcomes[17] = Ok(Some(0));
            outcomes
        };

        join_dispatched(group_of::<1>(&OUTCOMES, &CURSOR, &CTX));
        assert_eq!(
            CURSOR.0.get(),
            OUTCOMES.len(),
            "the live ticket must be retained until a terminal fact is observed"
        );
    }

    /// The gate protocol: staged refuses nothing yet, a decided gate
    /// answers every worker, and enter() reports the verdict.
    #[test]
    fn the_gate_reports_the_verdict() {
        static RUN_CTX: SpmdCtx = SpmdCtx::new(2);
        static ABORT_CTX: SpmdCtx = SpmdCtx::new(2);
        decide_for_test(&RUN_CTX, RUN);
        decide_for_test(&ABORT_CTX, ABORT);
        assert!(enter(&RUN_CTX), "a released gate admits");
        assert!(!enter(&ABORT_CTX), "an aborted gate refuses");
    }

    fn decide_for_test(ctx: &'static SpmdCtx, verdict: u32) {
        // `decide` itself (store + wake) is exercised on metal; the host
        // half proven here is the verdict read on the far side.
        ctx.gate.0.store(verdict, Ordering::SeqCst);
    }

    /// The ctx accessors carry the construction, geometry included.
    #[test]
    fn the_ctx_carries_its_shape() {
        static CTX: SpmdCtx = SpmdCtx::new(5);
        assert_eq!(CTX.workers(), 5);
        assert_eq!(CTX.barrier().parties(), 5);
        // The runtime-caller default: a linear grid, one tile per worker.
        assert_eq!(CTX.tiles(), [1, 1, 5]);
        assert_eq!(CTX.tile_count(), 5);
    }

    /// A macro-shaped launch: more tiles than workers, geometry queryable.
    #[test]
    fn a_grid_outnumbering_its_workers_rides_the_ctx() {
        static CTX: SpmdCtx = SpmdCtx::with_grid(7, [4, 4, 64]);
        assert_eq!(CTX.workers(), 7);
        assert_eq!(CTX.tiles(), [4, 4, 64]);
        assert_eq!(CTX.tile_count(), 1024);
        assert_eq!(CTX.barrier().parties(), 7);
    }

    // ── The resident region (R9) ──────────────────────────────────────
    //
    // These drive [`resident_run`] — the target-neutral loop — on real
    // `std` threads against a bare ctx the test submits to by hand (the
    // same arm/step `ResidentRegion::submit` performs; the handle itself
    // needs a founded gang and a star, which are metal-only). What is
    // proven here is the loop's contract: a submitted job runs once per
    // active worker, the completion word retires it, generations do not
    // re-run, and a retire ends every worker. On host the workers spin
    // (no monitor, no kick) — the word is the truth, exactly as the
    // barrier's `wait_while` guarantees on every target.

    use core::sync::atomic::AtomicU32 as TestU32;

    /// Zero is the physical idle sentinel, never a forgeable public ticket.
    #[test]
    fn resident_ticket_zero_is_unrepresentable() {
        static CTX: SpmdCtx = SpmdCtx::new(1);

        assert!(ResidentTicket::new(&CTX, 0).is_none());
    }

    /// Context identity, resident epoch, and issuance are three distinct
    /// rejection facts; none may turn into a wait on an unrelated generation.
    #[test]
    fn resident_tickets_reject_foreign_stale_and_unissued_values() {
        static FIRST: SpmdCtx = SpmdCtx::new(1);
        static SECOND: SpmdCtx = SpmdCtx::new(1);

        let mut first = ResidentTickets::new(&FIRST);
        let (ticket, generation) = first.issue().expect("the first identity exists");
        assert_eq!(first.poll(ticket, 0), Ok(false));
        assert_eq!(first.poll(ticket, generation), Ok(true));

        let second = ResidentTickets::new(&SECOND);
        assert_eq!(
            second.poll(ticket, 0),
            Err(ResidentTicketError::ForeignRegion)
        );

        // A new owner epoch snapshots FIRST's advanced identity floor. The
        // former epoch's token cannot be reinterpreted through it.
        let next_epoch = ResidentTickets::new(&FIRST);
        assert_eq!(
            next_epoch.poll(ticket, 0),
            Err(ResidentTicketError::StaleEpoch)
        );

        let ahead = ResidentTicket::new(&FIRST, ticket.sequence.get() + 1).unwrap();
        assert_eq!(
            first.poll(ahead, generation),
            Err(ResidentTicketError::NotIssued)
        );
    }

    /// The public sequence, not the wrapping physical generation, decides
    /// whether an older serial job completed. A retained token therefore never
    /// regresses while its compact generation is reused by a current job.
    #[test]
    fn resident_ticket_stays_complete_across_physical_generation_wrap() {
        static CTX: SpmdCtx = SpmdCtx::new(1);
        let mut tickets = ResidentTickets::new(&CTX);

        // Prime the private physical counter at its boundary without running
        // billions of jobs; `issue` is the exact stepping code submit uses.
        tickets.generation = u32::MAX - 1;
        let (old, last_physical) = tickets.issue().expect("last physical id");
        assert_eq!(last_physical, u32::MAX);
        assert_eq!(tickets.poll(old, u32::MAX), Ok(true));

        let (current, wrapped_physical) = tickets.issue().expect("wrapped physical id");
        assert_eq!(wrapped_physical, 1);
        assert_eq!(
            tickets.poll(old, u32::MAX),
            Ok(true),
            "the retained completed ticket must not regress"
        );
        assert_eq!(tickets.poll(current, u32::MAX), Ok(false));
        assert_eq!(tickets.poll(current, 1), Ok(true));
        assert_eq!(tickets.poll(old, 1), Ok(true));
    }

    /// The former half-range rule misclassified a ticket once the stream moved
    /// `2^31` jobs ahead. Logical ordering has no such retention horizon.
    #[test]
    fn resident_ticket_has_no_u32_half_range_retention_limit() {
        static CTX: SpmdCtx = SpmdCtx::new(1);
        let old = ResidentTicket::new(&CTX, 1).unwrap();
        let latest = ResidentTicket::new(&CTX, 0x8000_0002).unwrap();
        let tickets = ResidentTickets {
            ctx: &CTX,
            floor: 0,
            last: Some(latest),
            generation: 7,
        };

        assert_eq!(tickets.poll(old, 0), Ok(true));
        assert_eq!(tickets.poll(latest, 0), Ok(false));
        assert_eq!(tickets.poll(latest, 7), Ok(true));
    }

    /// Logical identity exhaustion refuses permanently; it cannot wrap to a
    /// token retained by an arbitrarily old event.
    #[test]
    fn resident_ticket_identity_exhaustion_refuses_without_reuse() {
        static CTX: SpmdCtx = SpmdCtx::new(1);
        CTX.resident_sequence.store(u64::MAX, Ordering::Relaxed);
        let mut tickets = ResidentTickets::new(&CTX);

        assert!(tickets.issue().is_none());
        assert!(tickets.last.is_none());
        assert_eq!(CTX.resident_sequence(), u64::MAX);
    }

    /// Hand-drive one job of `active` workers over `ctx` and block until it
    /// retires — the arm/step/observe `submit`+`wait` perform.
    fn submit_and_wait(ctx: &'static SpmdCtx, last: u32, active: u32) -> u32 {
        let g = last.wrapping_add(1);
        // `done` counts every founded worker (each decrements once), not the
        // active subset — the tests spawn `0..capacity` threads, so that is
        // the gang width here. Mirrors `ResidentRegion::submit`.
        ctx.done_word().store(ctx.capacity(), Ordering::Relaxed);
        ctx.arm(active, 0, core::ptr::null_mut());
        ctx.job_word().store(g, Ordering::SeqCst);
        while ctx.complete_word().load(Ordering::Acquire) != g {
            std::thread::yield_now();
        }
        g
    }

    /// End a resident gang: gate off RUN, step `job` to wake every worker.
    fn retire_ctx(ctx: &'static SpmdCtx) {
        ctx.gate_word().store(DONE, Ordering::SeqCst);
        ctx.job_word().fetch_add(1, Ordering::SeqCst);
    }

    /// The whole resident lifecycle: a founded gang serves job after job
    /// over one ctx, each completion observed with no join, then retires.
    /// A re-run (a generation a worker executes twice) would over-count;
    /// a missed retire would hang the join.
    #[test]
    fn a_resident_gang_serves_many_jobs_then_retires() {
        static CTX: SpmdCtx = SpmdCtx::new(2);
        static RUNS: [TestU32; 2] = [TestU32::new(0), TestU32::new(0)];

        fn body(_ctx: &'static SpmdCtx, worker: u32) {
            RUNS[worker as usize].fetch_add(1, Ordering::SeqCst);
        }

        // `found_resident` would open the gate; on host we do it directly,
        // as there is no gang to found.
        CTX.gate_word().store(RUN, Ordering::SeqCst);
        let workers: std::vec::Vec<_> = (0..2)
            .map(|w| std::thread::spawn(move || resident_run(&CTX, w, body)))
            .collect();

        let mut last = 0;
        for _ in 0..5 {
            last = submit_and_wait(&CTX, last, 2);
        }
        // Every worker ran every job exactly once — no re-run, no miss.
        assert_eq!(RUNS[0].load(Ordering::SeqCst), 5);
        assert_eq!(RUNS[1].load(Ordering::SeqCst), 5);

        retire_ctx(&CTX);
        for w in workers {
            w.join().expect("a retired worker exits its loop");
        }
    }

    /// A job narrower than the founded gang wakes the whole gang but runs
    /// only `active` of it.  The rest skip the body but still decrement the
    /// completion counter, so the next submit cannot overwrite `active`/`env`
    /// until every worker has accounted for this generation.
    #[test]
    fn a_narrow_job_runs_only_its_active_workers() {
        static CTX: SpmdCtx = SpmdCtx::new(3);
        static RUNS: [TestU32; 3] = [TestU32::new(0), TestU32::new(0), TestU32::new(0)];

        fn body(_ctx: &'static SpmdCtx, worker: u32) {
            RUNS[worker as usize].fetch_add(1, Ordering::SeqCst);
        }

        CTX.gate_word().store(RUN, Ordering::SeqCst);
        let workers: std::vec::Vec<_> = (0..3)
            .map(|w| std::thread::spawn(move || resident_run(&CTX, w, body)))
            .collect();

        // Two jobs of width 2 over a gang of 3: worker 2 sits both out.
        let mut last = 0;
        last = submit_and_wait(&CTX, last, 2);
        let _ = submit_and_wait(&CTX, last, 2);

        assert_eq!(RUNS[0].load(Ordering::SeqCst), 2);
        assert_eq!(RUNS[1].load(Ordering::SeqCst), 2);
        assert_eq!(
            RUNS[2].load(Ordering::SeqCst),
            0,
            "worker 2 was never in a job"
        );

        retire_ctx(&CTX);
        for w in workers {
            w.join().expect("a retired worker exits its loop");
        }
    }
}
