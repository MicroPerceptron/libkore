//! # `thread` — spawn a THREAD of this task (#407, the unified task model).
//!
//! A thread is a **task founded into the caller's own address space**: to
//! the kernel it is an ordinary `RunTask` — its own cap arena, its own
//! parks, its own scheduler identity — sharing the parent's page-table
//! root.  Nothing here is a green thread: the kernel schedules and
//! preempts it exactly like any task, and `#[kore_rt::task]` keeps its
//! cooperative meaning WITHIN a thread's executor.
//!
//! ## What the parent provisions (`res/USERLAND.md` §4's per-task pieces,
//! one band per thread)
//!
//! - the **stack** (guard-gapped, mapped by the provisioning session);
//! - the **runtime block** page (`%fs:0`'s home — [`crate::entry`]'s
//!   `ThreadBlock`);
//! - the **boot page**, parent-written: `.kore.boot` naming the child's
//!   `role::ADDRESS_SPACE` self-view, `.kore.geom` naming its own
//!   scratch/slots aliases (kernel-installed at `Found`), its map band,
//!   and the runtime block.
//!
//! ## The band plan ([`BANDS`])
//!
//! Thread `n`'s windows sit at band base + `n` strides (thread 0 is the
//! task's own founder-stamped plan).  The parent picks them; nothing
//! kernel-side fixes these numbers — they ride the boot page like every
//! task's.
//!
//! This used to be a block of hand-picked `u64` constants — `SCRATCH_BAND`,
//! `MAP_STRIDE`, `STACK_BAND`, … — with the convention held together by
//! this comment.  Nothing checked that the bands did not overlap, that a
//! stride covered the window it strided, or that a thread index stayed
//! inside its band; a thread whose index ran off the end simply got another
//! band's addresses and corrupted a sibling.  The plan is now a table of
//! typed [`Band`]s:
//!
//! - a `Band` cannot be built with a stride narrower than its window, so
//!   consecutive threads' windows cannot overlap **by construction**;
//! - [`ThreadBands::windows`] returns `Option`, and it is the only way to
//!   name a thread's windows, so an index past the band is a refusal
//!   instead of a stray address;
//! - the band spans are proven pairwise disjoint at compile time below,
//!   and proven against the founder's own task plan over in
//!   `kore-svc-exec`'s `founder` module — the two crates carve one address
//!   space, and that agreement is now an `assert!`, not a coincidence of
//!   matching hex.
//!
//! ### Aliased vs. disjoint bands
//!
//! Four bands **alias** the task's own founder-mapped windows at slot 0
//! (`scratch`, `thread_block`, `slots`, and `map`) — the task *is* thread
//! 0, so its window has to be that band's first slot or thread 1 would
//! land on top of it.  The other three (`stage`, `boot`, `stack`) have no
//! counterpart in the founder's plan and must miss it entirely.
//!
//! The relation runs the other way too, and one founder window has no band
//! at all: `aperture`, where a `Display`-rooted framebuffer is installed.
//! A thread has no screen of its own — it is one huge fixed mapping, not a
//! per-thread fixture — so no band aliases it and **every** band must miss
//! it.  A new band added here that grew into it would not corrupt a
//! sibling thread, it would corrupt the display; the whole-space proof in
//! `kore-svc-exec` (which runs over the founder's `child_windows()`, the
//! aperture included) is what makes that a build error rather than a
//! discovery.
//!
//! ### The map band's ceiling, and one cursor per thread (#428)
//!
//! [`MAX_THREADS`] is not a taste call: the `map` band subdivides the
//! task's single founder-declared map reservation, so the reservation's
//! length over the stride is exactly how many threads this plan can carry.
//! Every other band is sized to match rather than declaring a second,
//! looser bound.
//!
//! The map band is the one band whose slots are also **kernel** state: each
//! thread's monotone map cursor is installed at `Found` from the `geom::MAP`
//! window its boot page was stamped with, and every delivered view the
//! session engine installs bumps that cursor.  So the band's slot boundaries
//! are only real if each cursor spans **its own stride and no more** — which
//! is exactly what both stampers now do: [`spawn`] below stamps thread `n`
//! with `w.map` (band slot `n`), and the founder stamps the task itself with
//! `LAYOUT.map_cursor()` (band slot 0, since the task *is* thread 0).
//!
//! It was the founder's half that was wrong.  It stamped the whole
//! reservation, so the kernel installed the task's cursor over all 8 strides
//! while slots 1.. were being handed to threads: a main thread that mapped
//! past one stride's worth of views walked straight into thread 1's map
//! window, two owners on one range with nothing anywhere to notice (#428).
//! What holds now is stronger than the equality that recorded the hazard:
//! `res/boot.toml` declares the map row as a **band** — reservation *and*
//! stride in one row — the founder asserts that band `same_as` this one
//! (base, slot length, stride and count, not merely the span they sum to),
//! and asserts that the cursor it installs precedes slot 1.  Exhausting a
//! thread's stride is the same typed refusal it always was; there is simply
//! no longer a neighbour on the far side of it.
//!
//! ## Lifecycle: exit, join, and giving the band back
//!
//! A thread entry is `fn()`.  Returning from it is the ordinary end: the
//! trampoline calls [`exit`]`(0)`, which presents the thread's own
//! `role::SELF` handle to the kernel's exit verb.  [`exit`] is also
//! callable directly for an early, non-zero end.
//!
//! Exit is *kernel* state, not a userland ledger — that is what makes
//! [`join`] a real block instead of a poll: the exiting task's status is
//! recorded on its `RunTask`, its joiners are woken, and it stops being
//! scheduled while staying resolvable so a joiner arriving afterwards
//! still reads the status.
//!
//! [`spawn`] hands back a [`Thread`] — the affine join ticket. A successful
//! [`join`] yields the child's status **and reclaims everything the spawn
//! spent**: the child (its handle revoked, which is the reap), the parent-held
//! stack / runtime-block / boot-staging regions, the parent arena slots, and
//! last of all the band index. A join which could not establish its bounded
//! kernel waiter returns [`ThreadJoinError`] with the ticket still inside, so
//! its caller can back off and retry without losing reclamation authority.
//! Dropping the ticket instead is [`Thread::detach`] — legitimate for a worker
//! that lives as long as its task, and the reason the type does not panic on
//! drop.
//!
//! Without the reclamation the join would be worth very little: the band
//! index used to be a saturating counter with no free list, so
//! [`MAX_THREADS`] was a *lifetime* cap rather than a concurrency one and
//! a task could spawn seven threads ever.

#![cfg_attr(not(target_os = "none"), allow(unused_imports))]

use core::sync::atomic::{AtomicU32, Ordering};

use kore_abi::session::SlotRef;
use kore_abi::task::layout::{Band, Window};
use kore_abi::task::{BootBlock, GeomBlock, geom, role};

use crate::exit::ExitCode;
use crate::session::{SessionBuilder, flows};
use crate::transport::{Transport, TrapTransport};

/// One page — every per-thread window's granule.
const PAGE: u64 = 4096;
/// Thread stacks: 64 KiB live, with a guard page below and the rest of the
/// stride left as gap.
const STACK_BYTES: u64 = 0x1_0000;
/// The cap-slot alias: page-rounded records plus one hint page. Derived, not
/// spelled: the founder's manifest reserves the same span, and the two are
/// asserted equal at compile time — a literal here is exactly the drift
/// that assert exists to catch.
const SLOTS_BYTES: u64 = kore_abi::execution::alias_bytes(crate::entry::ARENA_SLOT_CEILING);
/// Per-thread map windows: 16 MiB each.
const MAP_BYTES: u64 = 0x0100_0000;

/// **How many threads one task's address plan carries**, thread 0 (the task
/// itself) included — so `MAX_THREADS - 1` are spawnable.
///
/// Derived, not chosen: the `map` band subdivides the founder's single map
/// reservation, and 512 MiB / 16 MiB is 32 (the G4c relayout — the band
/// moved to `0x8000_0000` and quadrupled rather than shrinking its
/// stride, `res/SPMD.md` §8).  Every band is sized to this so that one
/// number bounds the whole plan; the founder asserts this whole band
/// against `res/boot.toml`'s declared one — count included — so a
/// manifest that restrides the reservation makes this number a build
/// error rather than a silently over-long plan.  32 is also the
/// [`THREAD_SLOTS`] word's exact capacity — the next widening pays for
/// a wider bitmap and a wider [`NUM_SHARDS`](crate::heap) together.
pub const MAX_THREADS: u32 = 32;

/// **The per-thread band plan** — one [`Band`] per window kind a thread
/// needs.  See the module docs for the aliased/disjoint split and for why
/// this is a table of typed bands rather than a page of `u64` constants.
///
/// Not `cfg`-gated even though only the x86 userland target can [`spawn`]:
/// the table is pure data, and it must be visible to host builds so the
/// founder's compile-time agreement proof can run in `make test` too.
#[derive(Clone, Copy, Debug)]
pub struct ThreadBands {
    /// Receiver-map windows — 16 MiB apart, tiling the task's map
    /// reservation.  Slot `n` is both thread `n`'s view-install area **and**
    /// the exact span of its kernel-side map cursor (#428).
    pub map: Band,
    /// R/W descriptor-scratch aliases (`geom::SCRATCH`), two pages each.
    pub scratch: Band,
    /// R/W runtime-block pages (`geom::THREAD_BLOCK`, `%fs:0`'s home).
    pub thread_block: Band,
    /// The parent-side **staging** VA a thread's boot page is written at
    /// before the founding delegates it to its final home in `boot`.
    pub stage: Band,
    /// The child-side boot pages (the delegation's destination — what the
    /// thread receives as `Found`'s `arg`).
    pub boot: Band,
    /// R/O cap-slot-array and hint aliases (`geom::SLOTS`), derived per band.
    pub slots: Band,
    /// Thread stacks: 64 KiB live in a 128 KiB stride, the gap holding the
    /// guard page.
    pub stack: Band,
}

/// The v1 per-thread band plan.  Bases are unchanged from the constants
/// this table replaced; what is new is that they are now *checked*.
pub const BANDS: ThreadBands = ThreadBands {
    map: Band::new(0x8000_0000, MAP_BYTES, MAP_BYTES, MAX_THREADS),
    scratch: Band::new(0x6000_0000, 0x2000, 0x2000, MAX_THREADS),
    thread_block: Band::new(0x6800_0000, PAGE, PAGE, MAX_THREADS),
    stage: Band::new(0x6900_0000, PAGE, PAGE, MAX_THREADS),
    boot: Band::new(0x6a00_0000, PAGE, PAGE, MAX_THREADS),
    slots: Band::new(0x7000_0000, SLOTS_BYTES, SLOTS_BYTES, MAX_THREADS),
    // 64 KiB live in a 128 KiB stride — the gap is where the guard page sits.
    stack: Band::new(0x7fff_fd00_0000, STACK_BYTES, 0x2_0000, MAX_THREADS),
};

/// **Every window one thread needs**, named — [`ThreadBands::windows`]'s
/// yield.  A named record rather than a tuple of `u64`s on purpose: seven
/// bare addresses in a row is precisely where a transposed argument reads
/// like working code.
#[derive(Clone, Copy, Debug)]
pub struct ThreadWindows {
    /// This thread's receiver-map window (`geom::MAP`) — stamped whole, so
    /// the cursor the kernel installs for this thread spans this stride and
    /// stops at the next thread's (#428).
    pub map: Window,
    /// This thread's descriptor scratch (`geom::SCRATCH`).
    pub scratch: Window,
    /// This thread's runtime block (`geom::THREAD_BLOCK`).
    pub thread_block: Window,
    /// Where the parent writes this thread's boot page before delegating.
    pub stage: Window,
    /// Where the delegated boot page lands in the thread's own view.
    pub boot: Window,
    /// This thread's cap-slot alias (`geom::SLOTS`).
    pub slots: Window,
    /// This thread's stack; its [`end`](Window::end) is the initial `rsp`.
    pub stack: Window,
}

impl ThreadBands {
    /// **Thread `n`'s windows, or `None` once the plan is full** — the only
    /// way to name them, so an index past [`MAX_THREADS`] cannot produce an
    /// address at all (it used to produce the *next band's*).
    pub const fn windows(&self, n: u32) -> Option<ThreadWindows> {
        let (
            Some(map),
            Some(scratch),
            Some(thread_block),
            Some(stage),
            Some(boot),
            Some(slots),
            Some(stack),
        ) = (
            self.map.nth(n),
            self.scratch.nth(n),
            self.thread_block.nth(n),
            self.stage.nth(n),
            self.boot.nth(n),
            self.slots.nth(n),
            self.stack.nth(n),
        )
        else {
            return None;
        };
        Some(ThreadWindows {
            map,
            scratch,
            thread_block,
            stage,
            boot,
            slots,
            stack,
        })
    }

    /// Each band's whole extent — the values a disjointness proof runs
    /// over, since a band's claim on the address space is its span rather
    /// than its first slot.
    pub const fn spans(&self) -> [Window; 7] {
        [
            self.map.span(),
            self.scratch.span(),
            self.thread_block.span(),
            self.stage.span(),
            self.boot.span(),
            self.slots.span(),
            self.stack.span(),
        ]
    }
}

/// No band may tread on another: with every band's `count` pinned to
/// [`MAX_THREADS`], a band that grows past its neighbour is a build error
/// here rather than a thread silently provisioned into a sibling's window.
const _: () = assert!(
    Window::all_disjoint(&BANDS.spans()),
    "kore-rt thread bands overlap: a thread would be provisioned on top of another's window"
);

/// The SPMD stack window (`spmd::STACKS`, G4b) claims address space the
/// same way a band does, so it enters the same proof: disjoint from
/// every band, or a worker's arena stack would be carved on top of a
/// sibling's window.
const _: () = {
    let spans = BANDS.spans();
    let arena = Window::new(
        crate::spmd::STACK_WINDOW_BASE,
        crate::spmd::STACK_WINDOW_GRANULES as u64 * crate::stack_arena::GRANULE as u64,
    );
    // The fiber window (G9, `lane::fiber::FIBERS`) claims address space
    // the same way and joins the same proof.
    let fibers = Window::new(
        crate::lane::fiber::FIBER_WINDOW_BASE,
        crate::lane::fiber::FIBER_WINDOW_GRANULES as u64 * crate::stack_arena::GRANULE as u64,
    );
    // So does the guarded fiber-stack window (`fiber_stack`), whose
    // stacks foreign fiber runtimes (Wasmtime's async) switch onto.
    let guarded = Window::new(
        crate::fiber_stack::WINDOW_BASE,
        crate::fiber_stack::WINDOW_GRANULES as u64 * crate::stack_arena::GRANULE as u64,
    );
    let mut all = [arena; 10];
    all[1] = fibers;
    all[2] = guarded;
    let mut i = 0;
    while i < spans.len() {
        all[i + 3] = spans[i];
        i += 1;
    }
    assert!(
        Window::all_disjoint(&all),
        "the SPMD stack window, the fiber windows, and the thread bands overlap"
    );
};

/// Every band window is page-granular at both ends — they are mapped as
/// stated, so a sub-page length or base could not be honoured.
const _: () = {
    let spans = BANDS.spans();
    assert!(
        Window::all_aligned(&spans, PAGE),
        "kore-rt thread bands must be page-granular at both ends"
    );
};

/// The stack band's stride must leave room for a guard page below each
/// stack — the gap is what the provisioning session's guard argument maps
/// into.
const _: () = assert!(
    BANDS.stack.gap() >= PAGE,
    "the thread stack stride leaves no room for a guard page"
);

/// **Which band slots are in use** — bit `n` set means thread `n`'s
/// windows are live.  Bit 0 is set from the start: slot 0 is the task's
/// own plan (the task *is* thread 0) and is never handed out.
///
/// A CAS bitmap rather than a counter, because a counter cannot say
/// "this one is free again".  That was the shape here — a saturating
/// `NEXT_THREAD` with no free list — and it quietly made [`MAX_THREADS`]
/// a *lifetime* budget: seven threads per task, ever, no matter how few
/// ran at once.  A join that could not give the band back would have been
/// half a feature.
///
/// A bare `AtomicU32` and not one of the in-tree structures on purpose:
/// `src/collections` is kernel-only (`src/`, not `crates/`) and so out of
/// reach here, and `crate::session::slots::SlotAlloc` — the runtime's own
/// bounded allocator — is `&mut self`, built for a single-owner founder
/// session rather than for threads claiming and releasing concurrently.
/// One word covers the whole plan; the assert below is what keeps that
/// true if the plan ever grows.
#[cfg(target_os = "none")]
static THREAD_SLOTS: AtomicU32 = AtomicU32::new(1);

/// One `AtomicU32` holds the whole band plan — a wider plan needs a wider
/// word, and must not silently lose its upper slots.
const _: () = assert!(
    MAX_THREADS <= u32::BITS,
    "the thread-slot bitmap is one u32; MAX_THREADS no longer fits it"
);

/// **Claim a free thread index**, or `None` once the band plan is full.
///
/// Lock-free, no global lock, and a full plan is an honest refusal like
/// every other bounded resource in this runtime.  The index is *reusable*:
/// what makes that safe is the release side's ordering, not this one — see
/// [`release_thread_index`].
#[cfg(target_os = "none")]
fn claim_thread_index() -> Option<u32> {
    let mut used = THREAD_SLOTS.load(Ordering::Relaxed);
    loop {
        let free = !used & (((1u64 << MAX_THREADS) - 1) as u32);
        if free == 0 {
            return None;
        }
        let n = free.trailing_zeros();
        match THREAD_SLOTS.compare_exchange_weak(
            used,
            used | (1 << n),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Some(n),
            Err(observed) => used = observed,
        }
    }
}

/// **Return a thread index to the plan.**
///
/// **Ordering is load-bearing**: this must run only after every mapping
/// that index authorized is actually gone — the stack, the runtime block,
/// the boot page, and the kernel-installed scratch/slots/map aliases the
/// child's own kill sweep removes.  All of those live in the *shared*
/// address space, so a fresh thread handed a still-mapped band would not
/// fault on them; it would map its stack over a corpse's and the two would
/// silently share memory.  [`join`] therefore revokes in order — the
/// child's handle (which reaps it and drains its holdings), then the
/// parent-held regions — and calls this last.
#[cfg(target_os = "none")]
fn release_thread_index(n: u32) {
    if n > 0 && n < MAX_THREADS {
        THREAD_SLOTS.fetch_and(!(1 << n), Ordering::Release);
    }
}

/// **How many thread bands are free now**: how many more threads this task
/// could found before its band plan is full.  An upper bound — a founding also
/// needs its founder's arena names and, for a strip stack, room in the stack
/// window — and a snapshot, which a concurrent founding or join moves.
#[cfg(target_os = "none")]
pub fn free_bands() -> u32 {
    let used = THREAD_SLOTS.load(Ordering::Acquire);
    (!used & (((1u64 << MAX_THREADS) - 1) as u32)).count_ones()
}

/// Off the metal no thread founds another: no band is free.
#[cfg(not(target_os = "none"))]
pub fn free_bands() -> u32 {
    0
}

/// How a spawned thread enters Rust: the plain `fn()` every service
/// worker uses, or an **spmd entry** carrying a region launch's context
/// and the index it assigned (`res/SPMD.md` §2) — the CUDA
/// shape: a worker's arguments ride the launch, never an ambient global.
///
/// The index is the *caller's* numbering — a region-local worker id, NOT
/// the thread's band slot.  A region's workers may occupy any subset of
/// the band plan, so the two disagree the moment a region is not alone in
/// its task; carrying the id here (rather than deriving it from the
/// band) is what keeps that a non-fact the worker never has to know.
#[cfg(target_os = "none")]
#[derive(Clone, Copy)]
enum ThreadEntry {
    /// Entered as `f()`.
    Plain(fn()),
    Argument(fn(usize), usize),
    /// Held at the group's launch gate, then entered as `f(ctx, worker)`.
    Spmd(crate::spmd::SpmdEntry, &'static crate::spmd::SpmdCtx, u32),
}

/// A spawned thread's start record, written by the parent at the top of
/// the child's stack; the trampoline reads it before Rust runs.
#[cfg(target_os = "none")]
struct ThreadStart {
    entry: ThreadEntry,
}

/// The record is staged in the `[stack_top - 24, stack_top)` words and
/// the trampolines' pointer slot sits at `stack_top - 32` — growing past
/// 24 bytes would overlap the next stage of the plan.  A compile error
/// here means the staging arithmetic in `spawn_entry` must move, not
/// that this assert should.
#[cfg(target_os = "none")]
const _: () = assert!(core::mem::size_of::<ThreadStart>() <= 24);

/// The child-arena slot each founding delivery lands in.  Slot 0 is the
/// arena's conventional origin and is never delivered into.
#[cfg(any(test, target_os = "none"))]
mod child_slot {
    /// The delivered self-view (`role::ADDRESS_SPACE`).
    pub(super) const SPACE: u16 = 1;
    /// The thread's own task handle (`role::SELF`) — [`super::exit`]'s
    /// subject.
    pub(super) const SELF: u16 = 2;
    /// Endowments follow, one per [`super::WorkerCap`].
    pub(super) const ENDOW_BASE: u16 = 3;
}

/// **A spawned thread's join ticket** — everything [`join`] needs to
/// collect the thread's status and give back what founding it cost.
///
/// Affine by convention rather than by a `Drop` panic: dropping it is
/// [`detach`](Self::detach), which is the right answer for a worker that
/// lives as long as its task (`SystemService::WORKERS`), and the wrong one
/// only if you wanted the band back.  The `#[must_use]` is the nudge.
#[must_use = "dropping a Thread detaches it: its band index and its stack stay spent for the task's life"]
#[derive(Debug)]
pub struct Thread {
    /// Process-unique creator arena identity; a slot is not portable authority.
    owner: usize,
    /// This thread's band index — released LAST, after its memory is gone.
    index: u32,
    /// The parent-arena slot holding the child's task handle: the join
    /// subject, and (revoked) the reap.
    handle: SlotRef,
    /// The parent-held `Region`s the provisioning session produced that
    /// outlive the founding: the stack and the runtime block.  Revoking them
    /// is what actually unmaps the band; the parent holds them precisely so
    /// that a dead thread cannot keep its own stack alive.  The third, the
    /// boot staging page, is the founding's delegation source, which its
    /// commit consumes ([`HELD_REGIONS`]).
    regions: [SlotRef; HELD_REGIONS],
    tls_region: Option<SlotRef>,
    /// Founding names whose return waits for [`join`]: always the staged
    /// space, whose founder record the `Found` leaves standing until the reap
    /// ([`STAGED_SPACE`]); and, for a gang founding that did not commit whole,
    /// whose outcome only the join learns, what a whole commit returns at
    /// once ([`release_committed`]: [`RELEASED_AT_COMMIT`] names and one per
    /// endowment).
    spent: [SlotRef; SPENT_CAPACITY],
    /// Live entries in [`spent`](Self::spent).
    spent_len: u8,
    /// The arena strip this stack was carved from — `Some` for SPMD
    /// workers (G4b), `None` for band-stacked plain threads.  Released
    /// by [`join`] after the stack's `Region` is revoked (memory gone
    /// before the VA is reusable), and — like the band index — kept
    /// spent for the task's life by a detach or a failed teardown.
    strip: Option<crate::stack_arena::Strip>,
}

/// A [`join`] attempt which did not complete waiting and reclamation.
///
/// The affine [`Thread`] ticket is returned inside the error: a bounded waiter
/// or teardown refusal must not lose reclamation authority. Inspect
/// [`cause`](Self::cause), back off when it is
/// [`TaskWaitError::Retryable`](crate::cap::TaskWaitError::Retryable), then
/// recover the ticket with [`into_thread`](Self::into_thread) for another
/// attempt.
#[must_use = "a failed thread join still owns the affine join ticket"]
#[derive(Debug)]
pub struct ThreadJoinError {
    thread: Thread,
    cause: crate::cap::TaskWaitError,
}

impl ThreadJoinError {
    /// Why the blocking lifecycle observation did not complete.
    pub const fn cause(&self) -> crate::cap::TaskWaitError {
        self.cause
    }

    /// Recover the unique ticket after applying caller-owned retry policy.
    pub fn into_thread(self) -> Thread {
        self.thread
    }

    /// Recover both the unique ticket and its failed wait result.
    pub fn into_parts(self) -> (Thread, crate::cap::TaskWaitError) {
        (self.thread, self.cause)
    }
}

impl Thread {
    /// The parent-arena slot holding the child's task handle — the
    /// kill/reap authority, for a caller that wants to act on the thread
    /// without joining it.
    pub fn handle(&self) -> SlotRef {
        self.handle
    }

    /// Probe this thread's kernel task without consuming the join ticket.
    ///
    /// This is the nonblocking half of [`join`]: a controller can wait for
    /// shared completion state, confirm the worker itself has crossed the
    /// exit boundary, and only then call `join` to reclaim its bands and
    /// provisioning without parking an executor thread.
    pub fn poll(&self) -> Result<crate::cap::TaskPollOutcome, crate::cap::TaskWaitError> {
        #[cfg(target_os = "none")]
        if crate::entry::geometry::thread_owner_identity() != Some(self.owner) {
            return Err(crate::cap::TaskWaitError::Refused(
                kore_abi::session::Status::Denied,
            ));
        }
        poll_task_handle(self.handle)
    }

    /// **Give up the join.**  The thread runs on; its band index, its
    /// stack, and its arena slots stay spent for the rest of the task's
    /// life.  Named rather than implicit so a detach reads as a decision.
    pub fn detach(self) {}
}

/// How many capabilities one [`spawn_with`] may endow a thread with.
pub const MAX_WORKER_CAPS: usize = flows::MAX_THREAD_ENDOW;

/// One capability a spawned thread is founded **holding** — what the
/// parent derives into the child's own arena at founding.
///
/// A thread needs its own holdings whenever it must *act*, not merely
/// read shared memory: park on an interrupt route, drive I/O ports,
/// answer a channel.  Sharing an address space shares memory and nothing
/// else — capability holdings are per task, and a thread is a task — so
/// the parent's tokens are unreachable from here no matter what pointer
/// the two threads share.
///
/// The child finds it through its own boot page, under
/// `(role, index)`, exactly as a founded service finds its endowment;
/// the possession verbs then work unchanged.
#[derive(Clone, Copy, Debug)]
pub struct WorkerCap {
    /// The boot-block role the child looks it up under.
    pub role: u16,
    /// Instance index within that role.
    pub index: u16,
    /// The **parent-held** subject to derive from.
    pub subject: SlotRef,
    /// The kind to derive (`ResourceKind as u8`) — which of the
    /// subject's derives is meant.
    pub kind: u8,
    /// Rights the child receives.
    pub rights: kore_abi::cap::Rights,
}

impl WorkerCap {
    /// **The process heap, endowed** — `role::HEAP@0`: the position a
    /// thread must hold for the global allocator to grow on it.
    ///
    /// The allocator is process-wide memory, but its grow is
    /// capability-gated per task: a thread that finds the shared heap
    /// window short carves its own extent off the supply it holds
    /// (`heap::source`), and sharing the parent's address space does not
    /// make the parent's holding reachable. Every thread that runs code
    /// which may allocate is founded holding this, whatever else it is
    /// endowed with. The parent keeps the supply and the child receives a
    /// whole-span sublease with carve and map authority only; REVOKE stays
    /// with the parent, which tears the sublease down.
    pub fn heap(supply: SlotRef) -> Self {
        use kore_abi::cap::{ResourceKind, Rights};
        Self {
            role: role::HEAP,
            index: 0,
            subject: supply,
            kind: ResourceKind::MemLease as u16 as u8,
            rights: Rights::ACCESS | Rights::MODIFY | Rights::DERIVE,
        }
    }
}

/// [`WorkerCap::heap`] over this thread's own `role::HEAP@0` — `None` when
/// the task was founded without a heap supply (its allocator cannot grow
/// on any thread), or before the thread is anchored.
#[cfg(target_os = "none")]
pub(crate) fn heap_endowment() -> Option<WorkerCap> {
    let rt = crate::entry::thread_context()?;
    Some(WorkerCap::heap(rt.bootstrap().handle(role::HEAP, 0)?))
}

/// The rights of a wake send a thread may pass on: the notice itself
/// (`ACCESS`, [`verbs::NOTICE_RIGHTS`](crate::cap::verbs::NOTICE_RIGHTS)) and
/// `DERIVE`, so the thread can found children holding the same end. A send
/// into a wake channel conveys nothing but the right to kick it.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
pub(crate) const WAKE_SEND_RIGHTS: kore_abi::cap::Rights =
    kore_abi::cap::Rights::ACCESS.union(kore_abi::cap::Rights::DERIVE);

/// [`WorkerCap`] re-deriving this thread's own `DOORBELL@PEER_BASE` send for a
/// child, so the child's [`kick_parent`] reaches the same thread this one's
/// does: a [`DoorbellWorker`]'s wake end, or one inherited the same way.
///
/// `None` when this thread holds no such send, or holds one it may not derive
/// from (a band peer's notice-only end, [`spmd`](crate::spmd)'s mesh): an
/// inheritance this thread cannot make is not offered, so it never fails the
/// spawn it would have rode on. The arena record decides, which is the
/// kernel's own word for the slot's kind and rights.
#[cfg(target_os = "none")]
pub(crate) fn wake_endowment() -> Option<WorkerCap> {
    use kore_abi::cap::{CapRecord, ResourceKind};
    let rt = crate::entry::thread_context()?;
    let send = rt
        .bootstrap()
        .handle(role::DOORBELL, crate::futex::PEER_BASE)?;
    let record_va = crate::entry::geometry::slot_record_va(send)?;
    // SAFETY: the slot alias is this thread's kernel-authored, read-only view
    // of its own arena record for a slot its boot block names.
    let record = unsafe { core::ptr::read_volatile(record_va as *const CapRecord) };
    if record.kind != ResourceKind::ChanSend as u16 || !record.rights.contains(WAKE_SEND_RIGHTS) {
        return None;
    }
    Some(WorkerCap {
        role: role::DOORBELL,
        index: crate::futex::PEER_BASE,
        subject: send,
        kind: ResourceKind::ChanSend as u16 as u8,
        rights: WAKE_SEND_RIGHTS,
    })
}

impl WorkerCap {
    /// **Channel space, endowed** — `role::CHANNEL_SPACE@0`: a revocable
    /// child of the mint grant `space`, so the thread can mint the channels
    /// that wake the threads *it* founds (a `DoorbellWorker` of its own, or
    /// the [`thread_controller`](crate::thread_controller) a C program's
    /// `pthread_create` starts on its main thread). The parent keeps its
    /// grant; revoking it ends the child's.
    pub fn channel_space(space: SlotRef) -> Self {
        use kore_abi::cap::ResourceKind;
        Self {
            role: role::CHANNEL_SPACE,
            index: 0,
            subject: space,
            kind: ResourceKind::ChannelSpace as u16 as u8,
            rights: crate::cap::CHANNEL_SPACE_RIGHTS,
        }
    }
}

/// [`WorkerCap::channel_space`] over this thread's own `role::CHANNEL_SPACE@0`
/// — `None` when it holds no mint grant.
#[cfg(target_os = "none")]
pub fn channel_space_endowment() -> Option<WorkerCap> {
    let rt = crate::entry::thread_context()?;
    Some(WorkerCap::channel_space(
        rt.bootstrap().handle(role::CHANNEL_SPACE, 0)?,
    ))
}

/// Off metal no thread founds another.
#[cfg(not(target_os = "none"))]
pub fn channel_space_endowment() -> Option<WorkerCap> {
    None
}

/// This thread's **own kick**: a send into its own doorbell it may derive
/// from, as a slot — `DOORBELL@`[`OWN_KICK`](crate::futex::OWN_KICK), or for
/// thread 0 the founder-carved self-kick at `DOORBELL@PEER_BASE`.
///
/// `None` when it holds none, or one it may not derive from: the authority a
/// thread hands a peer so the peer can wake it must itself be derivable. The
/// arena record decides, as for [`wake_endowment`].
#[cfg(target_os = "none")]
pub(crate) fn own_kick() -> Option<SlotRef> {
    use kore_abi::cap::{CapRecord, ResourceKind};
    let rt = crate::entry::thread_context()?;
    let boot = rt.bootstrap();
    let send = boot
        .handle(role::DOORBELL, crate::futex::OWN_KICK)
        .or_else(|| {
            (crate::entry::geometry::shard_index() == Some(0))
                .then(|| boot.handle(role::DOORBELL, crate::futex::PEER_BASE))
                .flatten()
        })?;
    let record_va = crate::entry::geometry::slot_record_va(send)?;
    // SAFETY: the slot alias is this thread's kernel-authored, read-only view
    // of its own arena record for a slot its boot block names.
    let record = unsafe { core::ptr::read_volatile(record_va as *const CapRecord) };
    (record.kind == ResourceKind::ChanSend as u16 && record.rights.contains(WAKE_SEND_RIGHTS))
        .then_some(send)
}

/// Build exactly the capability namespace a spawned worker receives.
///
/// This stays target-independent so host tests can pin a subtle shared-space
/// invariant: workers share the parent's address space, but never inherit its
/// holding coordinates. Only self handles plus capabilities explicitly
/// derived through `endow` appear here. In particular, an originating task's
/// non-derivable ProcessIo/stdio views and wake ends cannot be rediscovered by
/// reconstructing `Rt` in a worker.
#[cfg(any(test, target_os = "none"))]
fn worker_boot_block(endow: &[WorkerCap]) -> Option<BootBlock> {
    let mut block = BootBlock::new();
    if !block.push(
        role::ADDRESS_SPACE,
        0,
        SlotRef::from_wire(u32::from(child_slot::SPACE)),
    ) || !block.push(
        role::SELF,
        0,
        SlotRef::from_wire(u32::from(child_slot::SELF)),
    ) {
        return None;
    }
    for (i, cap) in endow.iter().enumerate() {
        if !block.push(
            cap.role,
            cap.index,
            SlotRef::from_wire(u32::from(child_slot::ENDOW_BASE) + i as u32),
        ) {
            return None;
        }
    }
    // A boot-named vacant landing stays excluded from the dynamic allocator
    // whether or not an explicit endowment supplies DOORBELL@0.
    if !block.push(
        role::DOORBELL,
        1,
        SlotRef::from_wire(u32::from(child_slot::ENDOW_BASE) + endow.len() as u32),
    ) {
        return None;
    }
    Some(block)
}

/// Add the worker-local and process-root runtime-block coordinates.
///
/// A worker's `%fs` block is private, but safe runtime attachments are
/// process-wide because every worker shares one address space.  Instance 1
/// therefore names thread zero's founder-mapped block for the lifetime of the
/// process.  This is geometry only: it conveys no capability or arena holding.
#[cfg(any(test, target_os = "none"))]
fn push_worker_thread_blocks(gblock: &mut GeomBlock, local: Window) -> bool {
    gblock.push_window(geom::THREAD_BLOCK, 0, local)
        && gblock.push_window(geom::THREAD_BLOCK, 1, BANDS.thread_block.first())
}

/// Spawn a thread executing `entry`, holding nothing but its own space
/// view.  [`spawn_with`] when the thread must hold capabilities.
#[cfg(target_os = "none")]
pub fn spawn(entry: fn()) -> Option<Thread> {
    spawn_entry(
        ThreadEntry::Plain(entry),
        &[],
        WorkerStack::Band,
        WorkerArena::Task,
    )
}

// (The per-worker `spawn_worker` shim retired with G5: workers found as a
// gang — `spawn_gang_workers` below — so a region's numbering, gating,
// and founding all ride the batch.)

/// Spawn a thread executing `entry`, founded holding one derived
/// capability per [`WorkerCap`].  Returns the [`Thread`] join ticket, or
/// `None` on any refusal — an exhausted band plan, too many endowments,
/// provisioning, stamping, or founding (each session's status is the
/// kernel's word; nothing here retries).
///
/// `entry` returns rather than diverging: the trampoline turns a return
/// into [`exit`]`(0)`.
#[cfg(target_os = "none")]
pub fn spawn_with(entry: fn(), endow: &[WorkerCap]) -> Option<Thread> {
    spawn_entry(
        ThreadEntry::Plain(entry),
        endow,
        WorkerStack::Band,
        WorkerArena::Task,
    )
}

/// Where a spawned worker's stack lives.
///
/// A plain thread's stack is its band slot by default: `STACK_BYTES`
/// (64 KiB) live under a guard page, tiled by [`MAX_THREADS`] — what a
/// service worker's bounded frames need.  A synchronous engine whose
/// recursion depth follows its *input* — a C compiler's expression and block
/// nesting — needs more than a slot holds, so it asks for a **guarded
/// strip** of the task's stack window instead (`res/SPMD.md` §8's
/// kernel-thread class, the shape SPMD gang workers already take: usable
/// pages with one guard granule below, mapped by the founding and given
/// back by `join`).  Overflow takes a fault either way, never a
/// neighbour's memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerStack {
    /// The thread's band slot.
    Band,
    /// A guarded strip of at least this many bytes, rounded up to whole
    /// pages.  The strip comes from the task's one stack window (16 MiB of
    /// reservation, shared with SPMD gangs); a request the window cannot
    /// place — or a zero-byte one — refuses the spawn.
    Bytes(usize),
}

/// How many capability slots a spawned thread's arena holds: the one sizing
/// question a founding leaves to its founder (`FoundReq.arena_slots`, which
/// the kernel accepts in `1..=ARENA_SLOTS`).
///
/// Every thread's arena holds its boot names, the runtime's serve scratch,
/// report and carve slots, and whatever it reserves while it runs. A thread
/// that **founds** holds more, for as long as each child lives: the child's
/// handle and the regions its stack, runtime block and boot page live in
/// (what [`join`] revokes), and, while a founding is in flight, that
/// founding's whole result plan.  A thread that founds a gang, or a thread
/// controller that founds a process's threads, therefore needs an arena that
/// scales with its children, the case the kernel's founding granule is sized
/// for ("founders are the arena's real sizing case", `ARENA_SLOTS`). A thread
/// controller founded with a task's 64 slots held two live threads; with the
/// founding granule it holds over two dozen, within a couple of the band
/// plan's own limit ([`MAX_THREADS`]).
///
/// Spawned arenas cannot grow: only thread 0 carries the extension window, so
/// the size is chosen once, here.
#[cfg(any(test, target_os = "none"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkerArena {
    /// A thread that founds nothing: [`TASK_ARENA_SLOTS`].
    Task,
    /// A thread that founds others: the founding granule,
    /// [`ARENA_SLOT_CEILING`](crate::entry::ARENA_SLOT_CEILING).
    Founder,
}

/// A task's arena: one word of the runtime's slot ledger.
///
/// It holds the widest endowment a worker can take, the serve scratch, the
/// report and carve slots behind it, and the short-lived reservations of a
/// thread that founds nothing (a channel mint, a join landing). The assert
/// below proves the first three fit; a widening of [`MAX_WORKER_CAPS`] or of
/// the serve scratch that broke that would fail the build here, not a
/// founding at run time.
#[cfg(any(test, target_os = "none"))]
pub(crate) const TASK_ARENA_SLOTS: u32 = u64::BITS;

#[cfg(any(test, target_os = "none"))]
const _: () = {
    // The highest boot name a worker can carry, then the runtime layout the
    // entry derives after it: serve scratch, the report slot, the carve slot.
    let top = child_slot::ENDOW_BASE as usize + MAX_WORKER_CAPS;
    let carve = top + crate::entry::SERVE_SCRATCH_SLOTS + 2;
    assert!(carve < TASK_ARENA_SLOTS as usize);
    assert!(TASK_ARENA_SLOTS <= crate::entry::ARENA_SLOT_CEILING);
    // Both sizes fit the per-thread slot alias band: the slot alias of a
    // founder's arena is the band's whole width.
    assert!(kore_abi::execution::alias_bytes(crate::entry::ARENA_SLOT_CEILING) <= SLOTS_BYTES);
};

#[cfg(any(test, target_os = "none"))]
impl WorkerArena {
    /// The `FoundReq.arena_slots` this arena is founded with.
    pub(crate) const fn slots(self) -> u32 {
        match self {
            Self::Task => TASK_ARENA_SLOTS,
            Self::Founder => crate::entry::ARENA_SLOT_CEILING,
        }
    }
}

/// The regions a founder provisions for each thread: the stack, the runtime
/// block and the boot staging page (a TLS block is one more, apart).
const PROVISIONED_REGIONS: usize = 3;

/// The provisioned regions a founder holds until [`join`] revokes them: the
/// stack and the runtime block ([`Thread`]'s `regions`).  The boot staging
/// page is the founding's delegation source: the commit moves its page into
/// the child and retires its founder record, so its name is free from then
/// on, a dead record any later landing or result replaces.
const HELD_REGIONS: usize = 2;

/// The staged space a thread's `Found` consumes, whose founder name stays
/// spent until [`join`]'s reap.  The commit moves the space into the child but
/// scrubs from the founder only what it delivers and delegates, so the
/// founder's record of the space stands — an `AddrSpace` with every right,
/// bound to nothing the founder holds — until the reap retires it.  A name
/// returned before that is not vacant: the next post landing there is refused
/// as occupied, and the next result written there overwrites a live record.
const STAGED_SPACE: usize = 1;

/// The founding results a committed `Found` scrubs from the founder, beside
/// one per endowment: the child's delivered view and the boot delegation's.
const SCRUBBED_RESULTS: usize = 2;

/// The founding names a committed `Found` leaves without a live record,
/// beside one per endowment: the consumed staging page and the scrubbed
/// results ([`release_committed`]).
const RELEASED_AT_COMMIT: usize = PROVISIONED_REGIONS - HELD_REGIONS + SCRUBBED_RESULTS;

/// [`Thread`]'s `spent`: the staged space, and what a gang founding that did
/// not commit whole could not return.
const SPENT_CAPACITY: usize = STAGED_SPACE + RELEASED_AT_COMMIT + MAX_WORKER_CAPS;

/// Arena names one thread's founding holds in its founder while it is in
/// flight, beside one per endowment: its provisioned regions, the staged
/// space, the results the commit scrubs, and the handle.  One more with a TLS
/// block.
#[cfg_attr(all(not(target_os = "none"), not(test)), allow(dead_code))]
pub(crate) const FOUNDING_SLOTS: usize = PROVISIONED_REGIONS + STAGED_SPACE + SCRUBBED_RESULTS + 1;

/// Arena names a founded thread keeps in its founder until [`join`]: its
/// handle, its held regions and its staged space (one more with a TLS block).
/// The rest of the founding's plan returns at commit ([`release_committed`]).
#[cfg_attr(all(not(target_os = "none"), not(test)), allow(dead_code))]
pub(crate) const RETAINED_SLOTS: usize = 1 + HELD_REGIONS + STAGED_SPACE;

// Every founding name is kept until the reap or returned at commit, never
// both and never neither: the launch budgets ([`crate::spmd`]) count the two
// halves apart.
const _: () = assert!(FOUNDING_SLOTS == RETAINED_SLOTS + RELEASED_AT_COMMIT);

/// The arena names a gang session founding `workers` threads holds at its
/// peak, just before its founding commits: every worker's claims and every
/// endowment result.
#[cfg_attr(all(not(target_os = "none"), not(test)), allow(dead_code))]
pub(crate) const fn gang_founding_slots(workers: usize, tls: bool, endowments: usize) -> usize {
    workers * (FOUNDING_SLOTS + tls as usize) + endowments
}

/// A worker that can **park and be kicked**, and kick back: two channels
/// minted off the task's `role::CHANNEL_SPACE` — the worker's own
/// doorbell (its recv; the parent keeps the send that kicks it) and a
/// **wake** channel the other way (the worker holds the send at
/// `DOORBELL@PEER_BASE`; the parent parks on the recv) — plus a heap
/// position over the task's supply so it may allocate, and the worker's
/// own kick, a derivable send into its doorbell it can hand a thread it
/// founds.
///
/// The wake channel exists because a launched task holds no self-kick:
/// the founder carves `DOORBELL@2` for service rows that declare workers,
/// but exec gives an application only its doorbell recv and landing, so
/// there is no send into thread 0's doorbell for a worker to inherit.
/// Minting one is the star's rule (`res/SPMD.md` §3) applied to a single
/// peer.
///
/// This is the transport a synchronous engine on a worker thread needs — a
/// C interpreter whose `require` must hand a request to the executor
/// and wait for the answer without spinning (`res/DAGFS.md` §4): the
/// worker `futex::wait`s on a word and parks on its doorbell; the
/// executor answers, stores, and [`kick`](Self::kick)s; the worker
/// stores and [`kick_parent`]s; the executor parks on
/// [`wake_recv`](Self::wake_recv) through `next_wake_on`.  An engine does
/// not speak that protocol itself: [`bridge::WorkerHost`](crate::bridge)
/// founds the worker on this and owns the word, its ordering and the
/// worker's end, so every engine shares one copy of it.
#[cfg(target_os = "none")]
#[must_use]
pub struct DoorbellWorker {
    thread: Thread,
    kick: crate::cap::ChanSendToken,
    /// The pair's revocation pivot — retired with the worker.
    root: crate::cap::ChannelRoot,
    /// The wake channel's recv end: what the parent parks on.
    wake_recv: crate::cap::ChanRecvToken,
    /// The parent's copy of the wake send (the worker's is a founding
    /// derive of the root); retired with the worker.
    wake_send: crate::cap::ChanSendToken,
    wake_root: crate::cap::ChannelRoot,
    /// The claimed mint; held so both pairs' lineage stays lawful.
    _space: crate::cap::ChannelSpaceToken,
}

#[cfg(target_os = "none")]
impl DoorbellWorker {
    /// Post one notice into the worker's doorbell — a park that is armed
    /// returns; the worker re-tests its word.  `false` on a refused post.
    pub fn kick(&self) -> bool {
        crate::cap::verbs::post_notice(self.kick.slot(), crate::futex::COOKIE_FUTEX, 0)
    }

    /// The join ticket.
    pub fn thread(&self) -> &Thread {
        &self.thread
    }

    /// The wake channel's recv slot — build a `NoticeRx` over it and park
    /// with `executor::next_wake_on(rx.standing_post())`.  Re-check the
    /// word after the post stands and before parking: the worker stores,
    /// then kicks, and a kick that found no standing post was refused, so
    /// the store it announced is already visible.
    pub fn wake_recv(&self) -> SlotRef {
        self.wake_recv.slot()
    }

    /// Join the worker, then retire both pairs (sends, then roots).
    pub fn join(self) -> Result<Option<i32>, ThreadJoinError> {
        let Self {
            thread,
            kick,
            root,
            wake_recv,
            wake_send,
            wake_root,
            _space,
        } = self;
        let joined = join(thread);
        let _ = kick.try_retire();
        let _ = root.try_retire();
        let _ = wake_send.try_retire();
        let _ = wake_recv.try_retire();
        let _ = wake_root.try_retire();
        joined
    }
}

/// The calling thread's C `errno` word — what a C runtime's
/// `__errno_location` answers.
///
/// Per thread, as ISO C requires (C11 7.5p2): the word lives in this
/// thread's runtime block rather than an ELF TLS block, so an image that
/// carries C code needs no TLS segment for it. Zero when the thread starts.
/// The pointer is valid for the calling thread's life, and only that thread
/// may read or write through it.
#[cfg(target_os = "none")]
pub fn c_errno() -> *mut i32 {
    crate::entry::geometry::c_errno()
}

/// Worker side of [`DoorbellWorker`]: post one notice into the parent's
/// wake channel (the send founded at `DOORBELL@PEER_BASE`).  `false` when
/// this thread holds no such send or the post was refused — the latter
/// means no post stood, which the parent's stand-then-load discipline
/// makes harmless (the store this kick announces is already visible).
#[cfg(target_os = "none")]
pub fn kick_parent() -> bool {
    let Some(rt) = crate::entry::thread_context() else {
        return false;
    };
    let Some(slot) = rt
        .bootstrap()
        .handle(role::DOORBELL, crate::futex::PEER_BASE)
    else {
        return false;
    };
    crate::cap::verbs::post_notice(slot, crate::futex::COOKIE_FUTEX, 0)
}

/// Spawn `entry` on a worker founded holding its own doorbell, a send
/// into thread 0's doorbell, its own kick ([`own_kick`]), a heap position,
/// and `extra`.  `None` when the task holds no `CHANNEL_SPACE` (its profile
/// lacks `channels`), the mint or a derive refuses, or [`spawn_with`] does.
///
/// The worker takes a founder's arena, the founding granule of capability
/// slots rather than a task's 64: a doorbell worker is where a host runs a
/// program's own code (libkore's C `main`, a
/// language engine), and that code founds threads of its own — its first
/// `pthread_create` starts the [thread controller](crate::thread_controller)
/// and an SPMD dispatch founds a gang, each holding the founder's arena for
/// as long as its children live.
#[cfg(target_os = "none")]
pub fn spawn_with_doorbell(entry: fn(), extra: &[WorkerCap]) -> Option<DoorbellWorker> {
    spawn_with_doorbell_on(entry, extra, WorkerStack::Band)
}

/// [`spawn_with_doorbell`] on the stack `stack` names — a guarded strip
/// for an engine whose depth its input decides ([`WorkerStack`]).  `None`
/// also when the stack window cannot place the strip.
#[cfg(target_os = "none")]
pub fn spawn_with_doorbell_on(
    entry: fn(),
    extra: &[WorkerCap],
    stack: WorkerStack,
) -> Option<DoorbellWorker> {
    use kore_abi::cap::{ResourceKind, Rights};

    use crate::cap::verbs;

    let rt = crate::entry::thread_context()?;
    let space_slot = rt.bootstrap().handle(role::CHANNEL_SPACE, 0)?;
    let space = verbs::claim_channel_space(space_slot)?;
    let pair = verbs::mint_channel(&space).ok()?;
    // The parent's recv is surplus: the worker's recv is a fresh derive
    // of the root at founding (the star's rule).
    let _ = pair.recv.try_retire();
    // The wake channel the other way: the parent keeps the recv, the
    // worker is founded with a derive of the send at `DOORBELL@PEER_BASE`.
    let Ok(wake) = verbs::mint_channel(&space) else {
        let _ = pair.send.try_retire();
        let _ = pair.root.try_retire();
        return None;
    };
    let Some(heap) = heap_endowment() else {
        let _ = pair.send.try_retire();
        let _ = pair.root.try_retire();
        let _ = wake.send.try_retire();
        let _ = wake.recv.try_retire();
        let _ = wake.root.try_retire();
        return None;
    };

    let mut caps = [WorkerCap {
        role: 0,
        index: 0,
        subject: SlotRef::from_wire(0),
        kind: 0,
        rights: Rights::empty(),
    }; MAX_WORKER_CAPS];
    const OWN: usize = 4;
    if OWN + extra.len() > caps.len() {
        let _ = pair.send.try_retire();
        let _ = pair.root.try_retire();
        let _ = wake.send.try_retire();
        let _ = wake.recv.try_retire();
        let _ = wake.root.try_retire();
        return None;
    }
    caps[0] = WorkerCap {
        role: role::DOORBELL,
        index: 0,
        subject: pair.root.slot(),
        kind: ResourceKind::ChanRecv as u16 as u8,
        rights: crate::cap::CHANNEL_ENDPOINT_RIGHTS,
    };
    // Derivable, so the threads this worker founds through the
    // [`thread_controller`](crate::thread_controller) inherit a kick into the
    // same wake channel ([`wake_endowment`]).
    caps[1] = WorkerCap {
        role: role::DOORBELL,
        index: crate::futex::PEER_BASE,
        subject: wake.send.slot(),
        kind: ResourceKind::ChanSend as u16 as u8,
        rights: WAKE_SEND_RIGHTS,
    };
    // The worker's own kick ([`own_kick`]): a derivable send into its own
    // doorbell, so a thread it founds — the controller its `pthread_create`
    // starts — can be handed the means to wake it.
    caps[2] = WorkerCap {
        role: role::DOORBELL,
        index: crate::futex::OWN_KICK,
        subject: pair.root.slot(),
        kind: ResourceKind::ChanSend as u16 as u8,
        rights: WAKE_SEND_RIGHTS,
    };
    caps[3] = heap;
    caps[OWN..OWN + extra.len()].copy_from_slice(extra);
    let Some(thread) = spawn_entry(
        ThreadEntry::Plain(entry),
        &caps[..OWN + extra.len()],
        stack,
        WorkerArena::Founder,
    ) else {
        let _ = pair.send.try_retire();
        let _ = pair.root.try_retire();
        let _ = wake.send.try_retire();
        let _ = wake.recv.try_retire();
        let _ = wake.root.try_retire();
        return None;
    };
    Some(DoorbellWorker {
        thread,
        kick: pair.send,
        root: pair.root,
        wake_recv: wake.recv,
        wake_send: wake.send,
        wake_root: wake.root,
        _space: space,
    })
}

/// Spawn with an inert argument and the same capability endowment as
/// `spawn_with`. The argument grants no authority. A pointer encoded by a
/// caller must remain valid until the returned thread is joined.
#[cfg(target_os = "none")]
pub fn spawn_with_arg(entry: fn(usize), argument: usize, endow: &[WorkerCap]) -> Option<Thread> {
    spawn_entry(
        ThreadEntry::Argument(entry, argument),
        endow,
        WorkerStack::Band,
        WorkerArena::Task,
    )
}

/// [`spawn_with_arg`] founding a thread that founds others, with a founder's
/// arena ([`WorkerArena::Founder`]): the thread controller, and the threads
/// it founds, any of which may found a gang or ask for more threads.
#[cfg(target_os = "none")]
pub(crate) fn spawn_founder_with_arg(
    entry: fn(usize),
    argument: usize,
    endow: &[WorkerCap],
) -> Option<Thread> {
    spawn_entry(
        ThreadEntry::Argument(entry, argument),
        endow,
        WorkerStack::Band,
        WorkerArena::Founder,
    )
}
#[cfg(not(target_os = "none"))]
pub fn spawn_with_arg(_entry: fn(usize), _argument: usize, _endow: &[WorkerCap]) -> Option<Thread> {
    None
}

/// Return to the slot ledger the founding names a committed `Found` left
/// without a live record.
///
/// A `Found` that commits delivers the child's view and every endowment out
/// of the founder (the delivery map scrubs each one's founder slot), clears
/// the founder's delegated `Region` results, the boot page's among them, and
/// retires the tag of the staging page its delegation consumed. No live
/// record is left under these names, and none can have been copied: the
/// founder was inside its own trap from apply to scrub. They are free the
/// moment the commit reports `Ok`, and only that proves it: a refused or
/// unreadable commit keeps its names (the founding's leak-on-refusal rule),
/// as a gang founding that did not commit whole keeps them in its tickets for
/// `join`. The staged space is not among them: the commit consumes it without
/// scrubbing its founder record ([`STAGED_SPACE`]).
#[cfg(target_os = "none")]
fn release_committed(slots: impl IntoIterator<Item = SlotRef>) {
    for slot in slots {
        crate::entry::geometry::release_cap_slot(slot);
    }
}

/// The one founding every public spawn shape resolves to — the entry
/// shape is the only thing the shapes disagree on, and it rides the
/// start record untouched by everything below.
#[cfg(target_os = "none")]
fn spawn_entry(
    entry: ThreadEntry,
    endow: &[WorkerCap],
    stack: WorkerStack,
    arena: WorkerArena,
) -> Option<Thread> {
    if endow.len() > MAX_WORKER_CAPS {
        return None;
    }
    let template = crate::entry::geometry::tls_template();
    if (crate::entry::geometry::has_static_tls() && template.is_none())
        || template.is_some_and(|t| !t.fits_band(MAP_BYTES))
    {
        return None;
    }
    // A controller is a repeat caller with a bounded capability arena. Refuse
    // ordinary slot exhaustion before spending a band or provisioning memory,
    // and leave a report slot available to poll/join existing children.
    // Session A owns three regions (+ TLS); B owns four results + endowments.
    let slots_needed = FOUNDING_SLOTS + usize::from(template.is_some()) + endow.len() + 1;
    if crate::entry::geometry::available_cap_slots() < slots_needed {
        return None;
    }
    // The caller's own ADDRESS_SPACE slot, cached in its ThreadBlock at
    // anchor time — every thread of the task can spawn further threads.
    let self_space = crate::entry::geometry::self_space_slot()?;
    // The band plan hands out this thread's windows, or refuses.  Holding
    // a `ThreadWindows` is proof that every one of them fits its band and
    // touches no sibling's — the arithmetic that used to be open-coded
    // here, seven times, against nothing.
    let owner = crate::entry::geometry::thread_owner_identity()?;
    let index = claim_thread_index()?;
    // The index may have belonged to a dead sibling: its noted windows must
    // not become this thread's (`cap::mapped::forget_thread`).
    crate::cap::mapped::forget_thread(index as usize);
    let mut w = BANDS.windows(index)?;
    let tls = match template {
        Some(t) => Some(t.place(w.map)?),
        None => None,
    };
    if let Some(p) = tls {
        w.map = p.remaining;
    }

    // ── The stack: an arena strip for SPMD workers (G4b) and for a
    //    plain thread that asked for one ([`WorkerStack::Bytes`]), the
    //    band slot otherwise.  Carved here and — exactly like the
    //    band index above — left spent if a later founding step
    //    refuses: the teardown that gives both back is join's, never
    //    spawn's, and the leak-on-refusal is the safe direction for
    //    the same reason the band's is. ──
    let strip = match (&entry, stack) {
        (ThreadEntry::Spmd(..), _) => {
            Some(crate::spmd::STACKS.carve_guarded(crate::spmd::STACK_GRANULES)?)
        }
        (ThreadEntry::Plain(_) | ThreadEntry::Argument(..), WorkerStack::Band) => None,
        (ThreadEntry::Plain(_) | ThreadEntry::Argument(..), WorkerStack::Bytes(bytes)) => {
            let granules = u32::try_from(bytes.div_ceil(crate::stack_arena::GRANULE)).ok()?;
            Some(crate::spmd::STACKS.carve_guarded(granules)?)
        }
    };
    let (stack_base, stack_len) = match &strip {
        Some(s) => (s.base(), s.bytes() as u64),
        None => (w.stack.base(), w.stack.len()),
    };

    // ── Session A: provision the shared-space pieces — the stack
    //    (guarded), the runtime-block page, and the boot page at its
    //    staging VA (all fresh zeroed extents, parent-held Regions). ──
    let stack_res = crate::entry::geometry::acquire_cap_slot()?;
    let tblock_res = crate::entry::geometry::acquire_cap_slot()?;
    let stage_res = crate::entry::geometry::acquire_cap_slot()?;
    let tls_region = if tls.is_some() {
        Some(crate::entry::geometry::acquire_cap_slot()?)
    } else {
        None
    };
    let transport = TrapTransport::new();
    let mut a = SessionBuilder::new();
    // REVOKE rides so the boot page's Region can be DELEGATED (the
    // delegate demands it on the source) and the pages reclaimed with
    // the thread's teardown.
    let rights = kore_abi::cap::Rights::ACCESS
        | kore_abi::cap::Rights::MODIFY
        | kore_abi::cap::Rights::REVOKE;
    // One guard page below the stack.  For a band stack it lands in the
    // stride gap the const assert above proves is at least a page wide;
    // for an arena strip it lands in the guard granule carve_guarded
    // reserved directly below the usable span — either way, claimed VA
    // nobody else can be handed.
    a.map_region(self_space, rights, stack_base, stack_len, 1, 0, stack_res)
        .ok()?;
    a.map_region(
        self_space,
        rights,
        w.thread_block.base(),
        w.thread_block.len(),
        0,
        0,
        tblock_res,
    )
    .ok()?;
    a.map_region(
        self_space,
        rights,
        w.stage.base(),
        w.stage.len(),
        0,
        0,
        stage_res,
    )
    .ok()?;
    if let (Some(p), Some(slot)) = (tls, tls_region) {
        a.map_region(
            self_space,
            rights,
            p.mapping.base(),
            p.mapping.len(),
            0,
            0,
            slot,
        )
        .ok()?;
    }
    let committed = transport.commit_now(&a).ok()?;
    if committed.status() != kore_abi::session::Status::Ok {
        return None;
    }

    // ── Stamp the boot page (at its staging VA — the delegation moves
    //    it to `boot_va`, which is what the child receives as `arg`). ──
    let arena_slots = arena.slots();
    if let (Some(t), Some(p)) = (template, tls) {
        unsafe { t.initialize(p.mapping.base()) };
    }
    stamp_stage(
        &w,
        endow,
        entry,
        stack_base,
        stack_len,
        arena_slots,
        tls.map(|p| p.mapping),
    )?;
    let stack_top = stack_base + stack_len;

    // ── Session B: the founding — views + boot delegation + Found
    //    (SHARE_SPACE, NO_AUTHORITY). ─────────────────────────────────
    let space_res = crate::entry::geometry::acquire_cap_slot()?;
    let view_res = crate::entry::geometry::acquire_cap_slot()?;
    let boot_res = crate::entry::geometry::acquire_cap_slot()?;
    let handle_res = crate::entry::geometry::acquire_cap_slot()?;
    // One parent-arena SSA result per endowment.  Pre-filled with
    // `boot_res` so the array holds only real slots; entries past
    // `endow.len()` are never read.
    let mut staged = [flows::ThreadEndow {
        subject: self_space,
        rights: kore_abi::cap::Rights::empty(),
        kind: 0,
        result: boot_res,
        child_slot: 0,
    }; MAX_WORKER_CAPS];
    for (i, cap) in endow.iter().enumerate() {
        staged[i] = flows::ThreadEndow {
            subject: cap.subject,
            rights: cap.rights,
            kind: cap.kind,
            result: crate::entry::geometry::acquire_cap_slot()?,
            child_slot: child_slot::ENDOW_BASE + i as u16,
        };
    }
    let mut bld = SessionBuilder::new();
    flows::spawn_thread(
        &mut bld,
        self_space,
        thread_trampoline as *const () as usize as u64,
        stack_top - 32,
        matches!(entry, ThreadEntry::Spmd(..)),
        stage_res,
        w.boot.base(),
        w.boot.len(),
        arena_slots,
        space_res,
        view_res,
        boot_res,
        handle_res,
        child_slot::SPACE,
        child_slot::SELF,
        &staged[..endow.len()],
    )
    .ok()?;
    let committed = transport.commit_now(&bld).ok()?;
    if committed.status() != kore_abi::session::Status::Ok {
        return None;
    }
    // The founding's plan is spent, and the names it left without a live
    // record come back now: the commit moved the view and every endowment
    // into the child, scrubbing their founder slots, cleared the boot
    // delegation's, and consumed the staging page (`release_committed`).  The
    // staged space's record stands until the reap ([`STAGED_SPACE`]), so its
    // name rides the join ticket beside what `join` revokes.
    release_committed(
        [stage_res, view_res, boot_res]
            .into_iter()
            .chain(staged[..endow.len()].iter().map(|e| e.result)),
    );
    Some(Thread {
        owner,
        index,
        handle: handle_res,
        regions: [stack_res, tblock_res],
        tls_region,
        spent: [space_res; SPENT_CAPACITY],
        spent_len: STAGED_SPACE as u8,
        strip,
    })
}

/// Stamp one child's staging page — the boot block (the child's arena
/// map under roles), the geometry block (the child's own windows), the
/// clock-page copy, and the start record at the stack top.  Shared by
/// the single spawn and the gang founding so the two cannot drift.
/// `None` = a block refused a push, or the slots alias missed its band
/// window.
#[cfg(target_os = "none")]
fn stamp_stage(
    w: &ThreadWindows,
    endow: &[WorkerCap],
    entry: ThreadEntry,
    stack_base: u64,
    stack_len: u64,
    arena_slots: u32,
    tls: Option<Window>,
) -> Option<()> {
    let block = worker_boot_block(endow)?;
    let mut gblock = GeomBlock::new();
    // The cap-slot alias spans exactly the arena it aliases, and must fit
    // the band's window — the one thread window whose length is computed
    // rather than declared, hence the only one that can be wrong.
    let slots = Window::try_new(
        w.slots.base(),
        kore_abi::execution::alias_bytes(arena_slots),
    )
    .filter(|s| w.slots.contains(s))?;
    // Each window is pushed WHOLE, so the stamped span is the planned span:
    // no call site here can pair one window's base with another's length.
    //
    // `geom::APERTURE` is deliberately NOT among them: a thread has no
    // framebuffer of its own, and stamping it the task's window would put a
    // second independent cursor on one span — #428 exactly, in the one
    // place the map band's stride cannot help.  The kernel leaves an
    // unstamped aperture uninstalled, so a thread that tries to map a
    // display gets a typed refusal rather than its parent's screen.
    if !gblock.push_window(geom::SCRATCH, 0, w.scratch)
        || !gblock.push_window(geom::SLOTS, 0, slots)
        || !gblock.push_window(geom::MAP, 0, w.map)
        // Every worker shares the process-root runtime block. Naming it as a
        // second geometry instance gives the worker's anchor an immutable,
        // founder-shaped coordinate for process-wide attachment claims; it
        // does not install or transfer any capability.
        || !push_worker_thread_blocks(&mut gblock, w.thread_block)
    {
        return None;
    }
    if let Some(window) = tls {
        if !gblock.push_window(geom::THREAD_BLOCK, 2, window) {
            return None;
        }
    }
    // `geom::RESERVE` is the one task-level window a thread IS stamped: its
    // placement is exact and keeps no cursor, so there is no second cursor
    // to put on one span (#428), and the stamp is what makes the kernel keep
    // this thread's exact-VA shares out of the task's reservations too, as
    // it does the task's own (`refuse_reserve_window`).  Copied from the
    // spawning thread's geometry, which carries it in turn.
    if let Some(reserve) = crate::entry::thread_context()
        .and_then(|rt| rt.geom())
        .and_then(|g| g.window_of(geom::RESERVE))
    {
        if !gblock.push_window(geom::RESERVE, 0, reserve) {
            return None;
        }
    }
    let stage_va = w.stage.base();
    let stack_top = stack_base + stack_len;
    // SAFETY: the provisioning session just mapped the whole `w.stage`
    // window R/W zeroed in this task's own space, exclusively ours until
    // the delegation; both blocks fit inside it (`GEOM_BLOCK_AT` is a
    // boot-page offset, and the stage window is a boot page).
    unsafe {
        core::ptr::copy_nonoverlapping(
            (&block as *const BootBlock).cast::<u8>(),
            stage_va as *mut u8,
            core::mem::size_of::<BootBlock>(),
        );
        core::ptr::copy_nonoverlapping(
            (&gblock as *const GeomBlock).cast::<u8>(),
            (stage_va as usize + crate::entry::GEOM_BLOCK_AT) as *mut u8,
            core::mem::size_of::<GeomBlock>(),
        );
        if let Some(clock) = crate::entry::current_clock_page() {
            core::ptr::copy_nonoverlapping(
                (clock as *const kore_abi::clock::ClockPage).cast::<u8>(),
                (stage_va as usize + crate::entry::CLOCK_BLOCK_AT) as *mut u8,
                core::mem::size_of::<kore_abi::clock::ClockPage>(),
            );
        }
        // The start record, at the top of the child's stack; the
        // trampoline pops it before Rust runs.  The record itself lives
        // in the stack's top words (the child never unwinds past its
        // entry, so nothing reclaims them): the record in
        // [stack_top - 24, stack_top), its pointer at stack_top - 32 —
        // the [sp] the trampolines read (the size assert at the record's
        // declaration polices this exact layout).
        let start_at = (stack_top - 24) as *mut ThreadStart;
        start_at.write(ThreadStart { entry });
        ((stack_top - 32) as *mut u64).write(start_at as u64);
    }
    Some(())
}

/// The widest planned gang endowment: the star's mesh at the widest chunk
/// (own recv, hub send, the founder's send, and a send per chunk sibling —
/// `MESH_CEILING + 2`), closed with the heap.
#[cfg(any(target_os = "none", test))]
const PLANNED_CAPS: usize = crate::star::MESH_CEILING + 3;

/// The fill pattern of an endowment buffer, never read past its length.
#[cfg(any(target_os = "none", test))]
const VACANT_CAP: WorkerCap = WorkerCap {
    role: 0,
    index: 0,
    subject: SlotRef::from_wire(0),
    kind: 0,
    rights: kore_abi::cap::Rights::empty(),
};

/// Close the `filled` capabilities planned in `out` with `heap`, the
/// process heap when the task holds one: the endowment's new length, or
/// `None` when the heap does not fit.
#[cfg(any(target_os = "none", test))]
fn close_with_heap(out: &mut [WorkerCap], filled: usize, heap: Option<WorkerCap>) -> Option<usize> {
    let Some(heap) = heap else {
        return Some(filled);
    };
    *out.get_mut(filled)? = heap;
    filled.checked_add(1)
}

/// **The gang founding** (G5 — `res/SPMD.md` §7): found `out.len()` SPMD
/// workers of one region in TWO sessions total — one provisioning
/// session mapping every worker's stack, runtime-block, and staging
/// pages, then one founding session staging every worker's legs and the
/// Found tail ([`flows::spawn_gang`]) — where the per-worker path pays
/// two per worker.  Workers are numbered `first_worker..` in the
/// region's dense order; `endows` is empty or one `WorkerCap` list per
/// worker (the star's doorbells), and `pin` selects `SPMD_WORKER` — true
/// for compute workers, false for the star's hub, which must neither
/// occupy a class-best pin nor count in a reservation batch
/// (`spawn_spmd` chunks a wide launch across calls).
///
/// `heap` closes every **planned** endowment — none, or the star's mesh —
/// with the process heap ([`WorkerCap::heap`]): a compute worker runs
/// caller code, and caller code allocates. Without it a worker can only
/// bump what thread 0 already mapped into the shared heap window, and its
/// first allocation that finds the window short is refused (#1316). An
/// explicit `endows` list is the caller's whole endowment and takes no
/// heap: the star's hub, its one caller, runs only `star::hub_main`,
/// which posts notices and parks and never allocates.
///
/// Returns whether EVERY worker founded (`Status::Ok`).  Once the
/// founding session has committed, `out` is populated with a join
/// ticket per worker in either case — on a partial founding
/// (`Unfounded`) a failed worker's ticket names a vacant handle, whose
/// join refuses and detaches in the caller's abort arm — so no founded
/// worker is ever orphaned.  A refusal before the founding commit
/// returns `false` with `out` untouched; whatever indices, strips, and
/// slots were claimed stay spent (the single spawn's leak-on-refusal
/// rule: join is the only giver-back, and these have no joiner).
#[cfg(target_os = "none")]
pub(crate) fn spawn_gang_workers(
    entry: crate::spmd::SpmdEntry,
    ctx: &'static crate::spmd::SpmdCtx,
    first_worker: u32,
    pin: bool,
    endows: &[&[WorkerCap]],
    star: Option<&crate::star::StarPlan>,
    heap: Option<WorkerCap>,
    out: &mut [Option<Thread>],
) -> bool {
    use crate::session::MAX_SESSION_FOUNDS;
    let n = out.len();
    if n == 0 {
        return true;
    }
    let template = crate::entry::geometry::tls_template();
    if n > MAX_SESSION_FOUNDS
        || (crate::entry::geometry::has_static_tls() && template.is_none())
        || template.is_some_and(|t| !t.fits_band(MAP_BYTES))
    {
        return false;
    }
    // Endowments come one of three ways: absent, exactly one explicit
    // list per worker (the hub founding's shape), or a star plan —
    // whose per-worker kick-mesh endows name PEER BANDS and therefore
    // cannot exist yet: they are built below, after the claims loop
    // fixes every member's band.  A planned endowment (absent or the
    // mesh) is closed with `heap`; an explicit one is complete as given.
    // A meshed chunk is invocation-bounded at
    // [`MESH_CEILING`](crate::star::MESH_CEILING) (n·(n+7) invocations,
    // the heap included).
    if !endows.is_empty() && (endows.len() != n || star.is_some() || heap.is_some()) {
        return false;
    }
    if star.is_some() && n > crate::star::MESH_CEILING {
        return false;
    }
    // Refuse a session this arena cannot finish before claiming anything:
    // past here a refusal leaks what it claimed.  A planned endowment is the
    // star's mesh closed with the heap; an explicit one is as given.
    let endowments = if endows.is_empty() {
        n * (star.map_or(0, |plan| plan.mesh_caps(n)) + usize::from(heap.is_some()))
    } else {
        endows.iter().map(|caps| caps.len()).sum()
    };
    if crate::entry::geometry::available_cap_slots()
        < gang_founding_slots(n, template.is_some(), endowments)
    {
        return false;
    }
    let Some(self_space) = crate::entry::geometry::self_space_slot() else {
        return false;
    };
    let Some(owner) = crate::entry::geometry::thread_owner_identity() else {
        return false;
    };
    /// One worker's claims between the two sessions.
    struct Claim {
        index: u32,
        w: ThreadWindows,
        strip: crate::stack_arena::Strip,
        stack_res: SlotRef,
        tblock_res: SlotRef,
        stage_res: SlotRef,
        space_res: SlotRef,
        view_res: SlotRef,
        boot_res: SlotRef,
        handle_res: SlotRef,
        tls: Option<crate::tls::Placement>,
        tls_region: Option<SlotRef>,
    }
    let mut claims: [Option<Claim>; MAX_SESSION_FOUNDS] = [const { None }; MAX_SESSION_FOUNDS];
    // A gang worker runs a region's body and founds nothing.
    let arena_slots = WorkerArena::Task.slots();
    let transport = TrapTransport::new();
    // ── Session A′: provision every worker's three pages, one commit. ──
    let mut a = SessionBuilder::new();
    let rights = kore_abi::cap::Rights::ACCESS
        | kore_abi::cap::Rights::MODIFY
        | kore_abi::cap::Rights::REVOKE;
    for slot in claims.iter_mut().take(n) {
        let Some(index) = claim_thread_index() else {
            return false;
        };
        let Some(mut w) = BANDS.windows(index) else {
            return false;
        };
        let tls = match template {
            Some(t) => {
                let Some(p) = t.place(w.map) else {
                    return false;
                };
                Some(p)
            }
            None => None,
        };
        if let Some(p) = tls {
            w.map = p.remaining;
        }
        let tls_region = if tls.is_some() {
            let Some(slot) = crate::entry::geometry::acquire_cap_slot() else {
                return false;
            };
            Some(slot)
        } else {
            None
        };
        let Some(strip) = crate::spmd::STACKS.carve_guarded(crate::spmd::STACK_GRANULES) else {
            return false;
        };
        let Some(stack_res) = crate::entry::geometry::acquire_cap_slot() else {
            return false;
        };
        let Some(tblock_res) = crate::entry::geometry::acquire_cap_slot() else {
            return false;
        };
        let Some(stage_res) = crate::entry::geometry::acquire_cap_slot() else {
            return false;
        };
        let Some(space_res) = crate::entry::geometry::acquire_cap_slot() else {
            return false;
        };
        let Some(view_res) = crate::entry::geometry::acquire_cap_slot() else {
            return false;
        };
        let Some(boot_res) = crate::entry::geometry::acquire_cap_slot() else {
            return false;
        };
        let Some(handle_res) = crate::entry::geometry::acquire_cap_slot() else {
            return false;
        };
        if a.map_region(
            self_space,
            rights,
            strip.base(),
            strip.bytes() as u64,
            1,
            0,
            stack_res,
        )
        .is_err()
            || a.map_region(
                self_space,
                rights,
                w.thread_block.base(),
                w.thread_block.len(),
                0,
                0,
                tblock_res,
            )
            .is_err()
            || a.map_region(
                self_space,
                rights,
                w.stage.base(),
                w.stage.len(),
                0,
                0,
                stage_res,
            )
            .is_err()
        {
            return false;
        }
        if let (Some(p), Some(slot)) = (tls, tls_region) {
            if a.map_region(
                self_space,
                rights,
                p.mapping.base(),
                p.mapping.len(),
                0,
                0,
                slot,
            )
            .is_err()
            {
                return false;
            }
        }
        *slot = Some(Claim {
            tls,
            tls_region,
            index,
            w,
            strip,
            stack_res,
            tblock_res,
            stage_res,
            space_res,
            view_res,
            boot_res,
            handle_res,
        });
    }
    let Ok(committed) = transport.commit_now(&a) else {
        return false;
    };
    if committed.status() != kore_abi::session::Status::Ok {
        return false;
    }
    // ── The planned endowments, buildable only now: every member's band
    //    is claimed, so the star's peer sends can be NAMED
    //    (`DOORBELL@PEER_BASE+band` — the futex lookup's arithmetic),
    //    and each plan is closed with the heap.  One vacant WorkerCap is
    //    the fill pattern; the lens bound what is read. ──
    let mut planned = [[VACANT_CAP; PLANNED_CAPS]; MAX_SESSION_FOUNDS];
    let mut planned_lens = [0usize; MAX_SESSION_FOUNDS];
    if endows.is_empty() {
        let mut bands = [(0usize, 0u32); MAX_SESSION_FOUNDS];
        for (k, claim) in claims.iter().take(n).enumerate() {
            let Some(c) = claim else { return false };
            bands[k] = (first_worker as usize + k, c.index);
        }
        for k in 0..n {
            let meshed = match star {
                Some(plan) => match plan.worker_endow_mesh(
                    first_worker as usize + k,
                    &bands[..n],
                    &mut planned[k],
                ) {
                    Some(filled) => filled,
                    None => return false,
                },
                None => 0,
            };
            let Some(filled) = close_with_heap(&mut planned[k], meshed, heap) else {
                return false;
            };
            planned_lens[k] = filled;
        }
    }
    let endow_for = |k: usize| -> &[WorkerCap] {
        if endows.is_empty() {
            &planned[k][..planned_lens[k]]
        } else {
            endows[k]
        }
    };
    // ── Stamp every worker's boot page. ──
    for (k, claim) in claims.iter().take(n).enumerate() {
        let Some(c) = claim else { return false };
        if let (Some(t), Some(p)) = (template, c.tls) {
            unsafe { t.initialize(p.mapping.base()) };
        }
        if stamp_stage(
            &c.w,
            endow_for(k),
            ThreadEntry::Spmd(entry, ctx, first_worker + k as u32),
            c.strip.base(),
            c.strip.bytes() as u64,
            arena_slots,
            c.tls.map(|p| p.mapping),
        )
        .is_none()
        {
            return false;
        }
    }
    // ── Endowment staging inputs: one parent-arena SSA result per cap,
    //    the child slot mirroring the stamped boot-block name. ──
    let dummy_endow = flows::ThreadEndow {
        subject: self_space,
        rights: kore_abi::cap::Rights::empty(),
        kind: 0,
        result: self_space,
        child_slot: 0,
    };
    let mut endow_store = [[dummy_endow; MAX_WORKER_CAPS]; MAX_SESSION_FOUNDS];
    let mut endow_lens = [0usize; MAX_SESSION_FOUNDS];
    for k in 0..n {
        let caps = endow_for(k);
        if caps.len() > MAX_WORKER_CAPS {
            return false;
        }
        for (i, cap) in caps.iter().enumerate() {
            let Some(result) = crate::entry::geometry::acquire_cap_slot() else {
                return false;
            };
            endow_store[k][i] = flows::ThreadEndow {
                subject: cap.subject,
                rights: cap.rights,
                kind: cap.kind,
                result,
                child_slot: child_slot::ENDOW_BASE + i as u16,
            };
        }
        endow_lens[k] = caps.len();
    }
    // ── Session B′: every leg run, then the Found tail. ──
    let mut ws = {
        let Some(first) = claims[0].as_ref() else {
            return false;
        };
        let filler = flows::GangWorker {
            entry: thread_trampoline as *const () as usize as u64,
            stack: 0,
            space_slot: first.space_res,
            child_view_slot: first.view_res,
            boot_region: first.stage_res,
            boot_va: first.w.boot.base(),
            boot_bytes: first.w.boot.len(),
            boot_result: first.boot_res,
            handle_slot: first.handle_res,
            endow: &[],
        };
        [filler; MAX_SESSION_FOUNDS]
    };
    for (k, claim) in claims.iter().take(n).enumerate() {
        let Some(c) = claim else { return false };
        let stack_top = c.strip.base() + c.strip.bytes() as u64;
        ws[k] = flows::GangWorker {
            entry: thread_trampoline as *const () as usize as u64,
            stack: stack_top - 32,
            space_slot: c.space_res,
            child_view_slot: c.view_res,
            boot_region: c.stage_res,
            boot_va: c.w.boot.base(),
            boot_bytes: c.w.boot.len(),
            boot_result: c.boot_res,
            handle_slot: c.handle_res,
            endow: &endow_store[k][..endow_lens[k]],
        };
    }
    let mut b = SessionBuilder::new();
    if flows::spawn_gang(
        &mut b,
        self_space,
        pin,
        arena_slots,
        child_slot::SPACE,
        child_slot::SELF,
        &ws[..n],
    )
    .is_err()
    {
        return false;
    }
    let Ok(committed) = transport.commit_now(&b) else {
        return false;
    };
    let founded_all = committed.status() == kore_abi::session::Status::Ok;
    // Mint every ticket now that the founding committed: whichever
    // workers exist are joinable, and a vacant handle's join refuses in
    // the caller's abort arm rather than orphaning anyone.  A whole commit
    // returns what it left of each worker's founding without a live record
    // now (`release_committed`); a partial one leaves those names to the
    // tickets, for the join that learns which workers exist.  Either way the
    // staged space's name waits for the reap ([`STAGED_SPACE`]).
    for (k, claim) in claims.iter_mut().take(n).enumerate() {
        let Some(c) = claim.take() else { return false };
        let endowed = &endow_store[k][..endow_lens[k]];
        let committed = [c.stage_res, c.view_res, c.boot_res]
            .into_iter()
            .chain(endowed.iter().map(|e| e.result));
        let mut spent = [c.space_res; SPENT_CAPACITY];
        let spent_len = if founded_all {
            release_committed(committed);
            STAGED_SPACE
        } else {
            for (name, slot) in spent[STAGED_SPACE..].iter_mut().zip(committed) {
                *name = slot;
            }
            STAGED_SPACE + RELEASED_AT_COMMIT + endowed.len()
        };
        out[k] = Some(Thread {
            owner,
            index: c.index,
            handle: c.handle_res,
            regions: [c.stack_res, c.tblock_res],
            tls_region: c.tls_region,
            spent,
            spent_len: spent_len as u8,
            strip: Some(c.strip),
        });
    }
    founded_all
}

/// The band slot this thread landed on — the futex fabric's peer index
/// (`role::DOORBELL`, `PEER_BASE + band`), which is how the star names
/// the hub's send for this worker.  Deliberately distinct from the dense
/// region-local worker id.
#[cfg(target_os = "none")]
impl Thread {
    pub(crate) fn band_index(&self) -> u32 {
        self.index
    }
}

/// Off the metal userland targets — host builds: the honest typed
/// refusal (`res/USERLAND.md` §2.1), never a panic or a simulated
/// thread.
#[cfg(not(target_os = "none"))]
pub fn spawn(_entry: fn()) -> Option<Thread> {
    None
}

/// Off the metal userland targets — the honest typed refusal, as
/// [`spawn`].
#[cfg(not(target_os = "none"))]
pub fn spawn_with(_entry: fn(), _endow: &[WorkerCap]) -> Option<Thread> {
    None
}

/// The thread's ring-3 entry (x86_64): `rdi` = the boot-page VA
/// (`Found`'s `arg`, from the delegation), `[rsp]` = the parent-written
/// [`ThreadStart`] pointer.  Normalize the frame, recover both, and
/// enter the Rust half.
#[cfg(all(target_os = "none", target_arch = "x86_64"))]
#[unsafe(naked)]
extern "C" fn thread_trampoline() -> ! {
    core::arch::naked_asm!(
        "xor rbp, rbp",   // terminate the frame chain
        "mov rsi, [rsp]", // the ThreadStart pointer the parent wrote
        "and rsp, -16",   // ABI-normalize before calling into Rust
        "call {start}",   // rdi (the boot page) rides through untouched
        "ud2",            // start diverges; trap if it ever returns
        start = sym thread_start,
    )
}

/// The thread's EL0 entry (aarch64): `x0` = the boot-page VA, `[sp]` =
/// the parent-written [`ThreadStart`] pointer.  The pointer is read
/// through a copy of `sp` rather than `sp`-based addressing: the
/// parent's slot placement is the staging arithmetic's business
/// (`stack_top - 32` today, only 8-alignment promised), and
/// `SCTLR_EL1.SA0` would fault an `sp`-based access the moment that
/// arithmetic parks it off 16 — reading through `x9` keeps this
/// trampoline correct under any 8-aligned placement.
#[cfg(all(target_os = "none", target_arch = "aarch64"))]
#[unsafe(naked)]
extern "C" fn thread_trampoline() -> ! {
    core::arch::naked_asm!(
        "mov x29, xzr",     // terminate the frame-record chain
        "mov x30, xzr",     // no return address
        "mov x9, sp",       // the parent-staged sp (stack_top - 24)
        "ldr x1, [x9]",     // the ThreadStart pointer the parent wrote
        "and x9, x9, #-16", // ABI-normalize (16-align downward)
        "mov sp, x9",
        "bl {start}",       // x0 (the boot page) rides through untouched
        "brk #0",           // start diverges; trap if it ever returns
        start = sym thread_start,
    )
}

/// The thread's U-mode entry (riscv64): `a0` = the boot-page VA,
/// `[sp]` = the parent-written [`ThreadStart`] pointer.
#[cfg(all(target_os = "none", target_arch = "riscv64"))]
#[unsafe(naked)]
extern "C" fn thread_trampoline() -> ! {
    core::arch::naked_asm!(
        "mv fp, zero",    // terminate the frame chain
        "mv ra, zero",    // no return address
        "ld a1, 0(sp)",   // the ThreadStart pointer the parent wrote
        "andi sp, sp, -16", // ABI-normalize before calling into Rust
        "call {start}",   // a0 (the boot page) rides through untouched
        "unimp",          // start diverges; trap if it ever returns
        start = sym thread_start,
    )
}

/// The Rust half of the thread's crt0: anchor `%fs` at THIS thread's
/// runtime block (the first act, exactly like the task's own crt0),
/// install its geometry, run the entry, and **end the thread** when it
/// returns.  A thread that cannot anchor parks forever — it has no
/// lawful session path, and with no geometry it cannot even reach the
/// exit verb.
#[cfg(target_os = "none")]
extern "C" fn thread_start(page: *const u8, start: *const ThreadStart) -> ! {
    // SAFETY: `page` is the delegated boot page (`Found`'s arg — the
    // spawn contract above), live for the thread's whole life.
    let page: &'static [u8] =
        unsafe { core::slice::from_raw_parts(page, crate::entry::BOOT_PAGE_BYTES) };
    if !crate::entry::anchor_boot_thread(page) {
        crate::entry::refuse();
    }
    let Ok(rt) = crate::entry::boot_context(page) else {
        crate::entry::refuse();
    };
    let _ = rt; // the geometry installs are boot_context's side effect
    if !crate::entry::install_child_tls(page) {
        crate::entry::refuse();
    }
    // SAFETY: the parent wrote the start record at the stack top and it
    // outlives the thread (nothing unwinds past the entry).
    let entry = unsafe { (*start).entry };
    match entry {
        ThreadEntry::Plain(f) => f(),
        ThreadEntry::Argument(f, argument) => f(argument),
        // A worker holds at the group's launch gate FIRST — enforced here,
        // in the runtime, so no worker can race a half-launched group into
        // the barrier — and an aborted launch exits without ever running
        // user code (`res/SPMD.md` §5).
        ThreadEntry::Spmd(f, ctx, worker) => {
            if crate::spmd::enter(ctx) {
                f(ctx, worker);
            }
        }
    }
    // The unwind that used not to exist.  A worker's entry may now simply
    // return; this is where that return becomes a status its joiner reads.
    exit(ExitCode::OK)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExitCommitAction {
    Retry,
    Backoff,
    Refuse,
}

fn exit_commit_action(committed: kore_abi::session::Committed) -> ExitCommitAction {
    use kore_abi::session::Status;

    if committed.failing_index() != 0 {
        return ExitCommitAction::Refuse;
    }
    match committed.status() {
        Status::Conflict => ExitCommitAction::Retry,
        Status::WouldBlock => ExitCommitAction::Backoff,
        _ => ExitCommitAction::Refuse,
    }
}

/// Commit one exit attempt in its own frame. The ordinary builder is sized
/// for a full batch; keeping it out of the retry loop releases that storage
/// before a WouldBlock sleep builds another session. Inlining both builders
/// into `exit` consumed 18 KiB in the installed PTL input service.
#[cfg(target_os = "none")]
#[inline(never)]
fn exit_attempt(
    slot: kore_abi::session::SlotRef,
    code: ExitCode,
) -> Result<kore_abi::session::Committed, crate::transport::TrapUnavailable> {
    let mut b = SessionBuilder::new();
    b.exit(slot, code)
        .map_err(|_| crate::transport::TrapUnavailable)?;
    TrapTransport::new().commit_now(&b)
}

/// Keep the sleep descriptor out of both the terminal caller and the exit
/// attempt. Its frame is reused after the unsuccessful exit commit returned.
#[cfg(target_os = "none")]
#[inline(never)]
fn exit_backoff() {
    if let Some(word) = crate::cap::verbs::sleep_until(1) {
        let now_ms = word >> 32;
        let _ = crate::cap::verbs::sleep_until((now_ms + 1) * 1_000_000);
    }
}

/// **End the calling thread** with `code` — the kernel exit verb, over
/// the thread's own `role::SELF` handle.
///
/// The status must be a standard [`ExitCode`] or a named application-reserved
/// slot declared through [`exit_codes!`](crate::exit_codes). There is no
/// conversion from an arbitrary integer.
///
/// Does not return, and no code of this thread runs after the commit: the
/// kernel marks the exit, wakes every joiner, and parks the task with
/// nothing that can ever re-admit it.  The thread's memory stays mapped
/// until someone [`join`]s it (or revokes its handle) — that is what lets
/// a joiner arriving afterwards still read the status.
///
/// A context that cannot present its own SELF handle [`refuse`]s instead;
/// a fully founded task, including thread zero, receives that authority.
///
/// [`refuse`]: crate::entry::refuse
#[cfg(target_os = "none")]
pub fn exit(code: ExitCode) -> ! {
    let Some(rt) = crate::entry::thread_context() else {
        crate::entry::refuse();
    };
    // A thread whose end is awaited announces it first ([`on_exit`]): every
    // way a thread ends that runs code — its entry returning, a panic, an
    // explicit exit — comes through here. Taken, so a hook that itself ends
    // the thread runs once.
    if let Some((call, argument)) = crate::entry::geometry::take_exit_hook() {
        call(argument);
    }
    let Some(slot) = rt.bootstrap().handle(role::SELF, 0) else {
        crate::entry::refuse();
    };
    loop {
        // Each OCC retry rebuilds a complete descriptor. The previous
        // attempt's builder no longer occupies this worker's stack.
        let Ok(committed) = exit_attempt(slot, code) else {
            crate::entry::refuse();
        };
        // Success never returns. Conflict retries immediately; WouldBlock
        // retains the existing paced retry. Other outcomes stay terminal.
        match exit_commit_action(committed) {
            ExitCommitAction::Retry => continue,
            ExitCommitAction::Backoff => exit_backoff(),
            ExitCommitAction::Refuse => crate::entry::refuse(),
        }
    }
}

/// Off the metal userland targets: no session path, so no exit either.
#[cfg(not(target_os = "none"))]
pub fn exit(_code: ExitCode) -> ! {
    crate::entry::refuse()
}

/// Have this thread's [`exit`] run `call(argument)` first, on this thread,
/// before the exit commits — how the thread that reaps it learns it is
/// ending without polling for it (the
/// [`thread_controller`](crate::thread_controller)'s children; a
/// [`bridge`](crate::bridge) worker's host). The last hook set stands, so the
/// hook is the thread's entry's to set. `call` runs after the thread's own code, so it must not block,
/// and must not end the thread itself.
///
/// The kernel's exit commits after the hook returns, so a reaper woken by
/// the hook can still find the thread live for a moment; it parks on the
/// thread's end for that moment (`futex::wait_or_end`), which no store could
/// announce.
#[cfg(target_os = "none")]
pub(crate) fn on_exit(call: fn(usize), argument: usize) {
    crate::entry::geometry::set_exit_hook(call, argument);
}

/// One caller-owned report landing reserved for a complete worker-join fold.
///
/// The slot is affine for the same reason a [`Thread`] is: once worker
/// founding starts, the collector must retain enough authority to observe a
/// terminal fact even if the rest of its arena is consumed by those workers.
/// A region reserves this before founding and reuses it serially for every
/// worker; dropping the region returns the name to the task-local bitmap.
#[cfg(target_os = "none")]
#[must_use = "dropping join scratch releases the report landing"]
pub struct JoinScratch {
    report: SlotRef,
}

#[cfg(target_os = "none")]
impl JoinScratch {
    pub(crate) fn reserve() -> Option<Self> {
        crate::entry::geometry::acquire_cap_slot().map(|report| Self { report })
    }

    /// The reserved landing's task-local name.
    ///
    /// The doorbell star records this name while it remains nested inside the
    /// [`Spmd`](crate::spmd::Spmd) that owns `self`; it never releases or
    /// outlives the reservation.
    pub(crate) const fn report_slot(&self) -> SlotRef {
        self.report
    }
}

#[cfg(target_os = "none")]
impl Drop for JoinScratch {
    fn drop(&mut self) {
        crate::entry::geometry::release_cap_slot(self.report);
    }
}

/// **Block until `thread` ends, then give back everything it cost.**
///
/// `Ok(Some(status))` is the thread's own exit status. `Ok(None)` means it was
/// reaped rather than exited (its handle was revoked out from under the join).
/// `Err` means the lifecycle observation did not complete and returns the
/// affine ticket intact; no teardown or arena release has run.
///
/// A proper park, not a poll: the commit parks on the kernel's join arm. A
/// transient wake may re-park, but `WouldBlock` is returned as a typed retry
/// rather than immediately trapping again: it can mean the joinee's bounded
/// waiter set is full, and a tight retry could otherwise monopolize a CPU.
///
/// ## The reclamation, in the one order that is safe
///
/// 1. **the child** — revoking its handle reaps the task, and its own
///    kill sweep drains its holdings: the delivered self-view, the
///    kernel-installed scratch/slots/map aliases, the delegated boot page;
/// 2. **the parent-held regions** — stack, runtime block, boot staging.
///    These are the parent's precisely so a dead thread cannot keep its
///    own stack alive;
/// 3. **the arena slots** — the held regions, the staged space, whose
///    founder record the reap has just retired, whatever a partial gang
///    commit left in the ticket, and the handle;
/// 4. **the band index — last.**  Every window in that band is now
///    unmapped, so the next [`spawn`] can be handed it.  Released any
///    earlier and a new thread would map its stack over a corpse's: the
///    band lives in the *shared* address space, so nothing would fault —
///    the two threads would just quietly share memory.
#[cfg(target_os = "none")]
pub fn join(thread: Thread) -> Result<Option<i32>, ThreadJoinError> {
    let Some(mut scratch) = JoinScratch::reserve() else {
        return Err(ThreadJoinError {
            thread,
            cause: crate::cap::TaskWaitError::NoSlots,
        });
    };
    join_reserved(thread, &mut scratch)
}

/// Join with a report landing that was reserved before worker founding.
///
/// `scratch` remains owned by the enclosing region across every refusal; this
/// function never allocates another arena slot and never releases the landing.
#[cfg(target_os = "none")]
pub(crate) fn join_reserved(
    thread: Thread,
    scratch: &mut JoinScratch,
) -> Result<Option<i32>, ThreadJoinError> {
    join_reserved_at(thread, scratch.report)
}

/// Join through a landing whose [`JoinScratch`] owner is held by the enclosing
/// region.
///
/// This raw-name half exists only for the region's doorbell star: the star is
/// nested inside the same [`Spmd`](crate::spmd::Spmd), records the owner's
/// slot before launch, and finishes before that owner drops. Other callers
/// use [`join_reserved`] so the affine reservation stays visible in the type.
#[cfg(target_os = "none")]
pub(crate) fn join_reserved_at(
    mut thread: Thread,
    report: SlotRef,
) -> Result<Option<i32>, ThreadJoinError> {
    use kore_abi::session::Status;

    use crate::session::SessionBuilder;
    use crate::transport::TrapTransport;

    if crate::entry::geometry::thread_owner_identity() != Some(thread.owner) {
        return Err(ThreadJoinError {
            thread,
            cause: crate::cap::TaskWaitError::Refused(Status::Denied),
        });
    }
    let transport = TrapTransport::new();
    let status = match join_park(&transport, thread.handle, report) {
        Ok(crate::cap::TaskWaitOutcome::Exited(status)) => Some(status),
        Ok(crate::cap::TaskWaitOutcome::Reaped) => None,
        Err(cause) => return Err(ThreadJoinError { thread, cause }),
    };
    // ── Reclaim, in order ────────────────────────────────────────────
    let mut b = SessionBuilder::new();
    // The handle first: its revoke IS the reap, and the reap is what
    // drains the child's own holdings.
    if status.is_some() && b.revoke(thread.handle).is_err() {
        return Err(ThreadJoinError {
            thread,
            cause: crate::cap::TaskWaitError::Transport,
        });
    }
    // The stack and runtime-block regions are parent-held and die here.
    // The STAGE region is deliberately not among them: the founding's
    // delegate consumed it (the boot page re-rooted under the child, its
    // parent record retired with the move), so revoking it refuses the
    // WHOLE session as `Revoked` — which un-revokes the handle too and
    // silently leaves `torn_down` false.  That exact shape leaked every
    // joined thread's strip and band index since #407 (bounded, and
    // invisible until the G7 retirement witness demanded a real reap).
    // Its name came back when the founding committed, or rides `spent`
    // from a gang founding that did not commit whole.
    for slot in thread
        .regions
        .into_iter()
        .map(Some)
        .chain([thread.tls_region])
        .flatten()
    {
        if b.revoke(slot).is_err() {
            return Err(ThreadJoinError {
                thread,
                cause: crate::cap::TaskWaitError::Transport,
            });
        }
    }
    let completion = match transport.commit_now(&b) {
        Ok(c) => c,
        Err(_) => {
            return Err(ThreadJoinError {
                thread,
                cause: crate::cap::TaskWaitError::Transport,
            });
        }
    };
    if completion.status() != Status::Ok {
        return Err(ThreadJoinError {
            thread,
            cause: crate::cap::TaskWaitError::Refused(completion.status()),
        });
    }
    if let Some(slot) = thread.tls_region {
        crate::entry::geometry::release_cap_slot(slot);
    }
    for slot in thread.regions {
        crate::entry::geometry::release_cap_slot(slot);
    }
    for slot in &thread.spent[..thread.spent_len as usize] {
        crate::entry::geometry::release_cap_slot(*slot);
    }
    crate::entry::geometry::release_cap_slot(thread.handle);
    // The band (and any arena strip), last. Every failure above retained
    // the complete ticket; no reusable slot or address is published until
    // the reclamation session succeeds.
    {
        if let Some(strip) = thread.strip.take() {
            crate::spmd::STACKS.release(strip);
        }
        release_thread_index(thread.index);
    }
    Ok(status)
}

/// The shared blocking lifecycle engine behind [`join`] and
/// [`crate::cap::TaskToken::wait`].
///
/// Keeping this at the task-handle seam matters: an exec-owned process and a
/// spawned thread are both ordinary kernel tasks, so they must not grow two
/// subtly different park/report decoders merely because their userland
/// provisioning differs.
#[cfg(target_os = "none")]
pub(crate) fn wait_task_handle(
    handle: SlotRef,
) -> Result<crate::cap::TaskWaitOutcome, crate::cap::TaskWaitError> {
    let transport = TrapTransport::new();
    join_wait(&transport, handle)
}

/// Off the metal userland targets there is no live session transport. Host code
/// gets the same honest typed refusal as [`join`].
#[cfg(not(target_os = "none"))]
pub(crate) fn wait_task_handle(
    _handle: SlotRef,
) -> Result<crate::cap::TaskWaitOutcome, crate::cap::TaskWaitError> {
    Err(crate::cap::TaskWaitError::Transport)
}

/// The nonblocking lifecycle probe behind [`crate::cap::TaskToken::poll`].
///
/// The temporary report name never escapes this function: it is released
/// after every success or refusal, while the caller keeps the borrowed affine
/// task token.
#[cfg(target_os = "none")]
pub(crate) fn poll_task_handle(
    handle: SlotRef,
) -> Result<crate::cap::TaskPollOutcome, crate::cap::TaskWaitError> {
    use crate::transport::TrapTransport;

    let transport = TrapTransport::new();
    let report =
        crate::entry::geometry::acquire_cap_slot().ok_or(crate::cap::TaskWaitError::NoSlots)?;
    let answer = poll_task_handle_in(&transport, handle, report);
    crate::entry::geometry::release_cap_slot(report);
    answer
}

/// Off the metal userland targets there is no live session transport.  Refuse
/// honestly rather than simulating a task lifecycle.
#[cfg(not(target_os = "none"))]
pub(crate) fn poll_task_handle(
    _handle: SlotRef,
) -> Result<crate::cap::TaskPollOutcome, crate::cap::TaskWaitError> {
    Err(crate::cap::TaskWaitError::Transport)
}

/// The transport-injected half of [`wait_task_handle`]: park on the kernel's
/// join arm until the named task ends, and decode its report.
#[cfg(target_os = "none")]
fn join_wait(
    transport: &crate::transport::TrapTransport,
    handle: SlotRef,
) -> Result<crate::cap::TaskWaitOutcome, crate::cap::TaskWaitError> {
    // A slot of our own rather than the founder-stamped report slot: a
    // spawned thread's geometry carries no report slot, and a thread must
    // be able to join a sibling.  Returned below on every path.
    let report =
        crate::entry::geometry::acquire_cap_slot().ok_or(crate::cap::TaskWaitError::NoSlots)?;
    let answer = join_park(transport, handle, report);
    crate::entry::geometry::release_cap_slot(report);
    answer
}

/// [`join_wait`]'s loop, factored so its single owned slot is returned on
/// every path.
#[cfg(target_os = "none")]
fn join_park(
    transport: &crate::transport::TrapTransport,
    handle: SlotRef,
    report: SlotRef,
) -> Result<crate::cap::TaskWaitOutcome, crate::cap::TaskWaitError> {
    use kore_abi::session::Status;

    use crate::session::SessionBuilder;
    use crate::transport::ParkTransport;

    // Probe before the first park: the common collector case joins
    // workers that already finished, and a plain probe needs no waiter
    // slot, no park, and no exposure to the park-vs-exit window (a park
    // staged in the instant the joinee ends has been observed to strand
    // the joiner under TCG timing; the probe-first shape simply never
    // opens that window for an already-ended joinee).
    {
        use crate::transport::Transport;
        let mut b = SessionBuilder::new();
        b.join(handle, report)
            .map_err(|_| crate::cap::TaskWaitError::Transport)?;
        if let Ok(committed) =
            <crate::transport::TrapTransport as Transport>::commit_now(transport, &b)
        {
            validate_task_completion(committed)?;
            let status = committed.status();
            if status == Status::Exited {
                let rec = read_task_report(report)?;
                if let TaskWaitStep::Complete(outcome) = classify_task_wait(status, Some(&rec))? {
                    return Ok(outcome);
                }
            }
        }
    }
    let mut hot_retries = 0u32;
    loop {
        let mut b = SessionBuilder::new();
        b.join(handle, report)
            .map_err(|_| crate::cap::TaskWaitError::Transport)?;
        b.wait_task();
        let committed = transport
            .commit_join_park(&b)
            .map_err(|_| crate::cap::TaskWaitError::Transport)?;
        validate_task_completion(committed)?;
        let status = committed.status();
        let rec = if status == Status::Exited {
            Some(read_task_report(report)?)
        } else {
            None
        };
        match classify_task_wait(status, rec.as_ref())? {
            TaskWaitStep::Retry => {
                // A joiner whose park cannot engage — a doorbell-less
                // thread (a service worker joining a sibling: the join
                // park's doorbell arm has nothing to stand on) — would
                // otherwise HOT-loop this commit until the joinee ends:
                // an unbounded trap storm the serial log renders as a
                // WouldBlock refusal spray.  The first retry is free (a
                // spurious resume on a working park re-probes at once);
                // past it, pace the loop on the deadline park every
                // thread CAN take, the #407 witness's own idiom.  A
                // doorbelled joiner is unaffected in its steady state —
                // its park holds and Retry stays rare.
                hot_retries += 1;
                if hot_retries > 1 {
                    if let Some(word) = crate::cap::verbs::sleep_until(1) {
                        let now_ms = word >> 32;
                        let _ = crate::cap::verbs::sleep_until((now_ms + 2) * 1_000_000);
                    }
                }
                continue;
            }
            TaskWaitStep::Complete(outcome) => return Ok(outcome),
        }
    }
}

/// [`Thread::poll`] through a landing reserved for it: a repeat observer —
/// the [`thread_controller`](crate::thread_controller), between a child's
/// announced end and its reap — probes without spending an arena slot each
/// time, and so without the one refusal that is about the observer rather
/// than the thread.
#[cfg(target_os = "none")]
pub(crate) fn poll_reserved(
    thread: &Thread,
    scratch: &JoinScratch,
) -> Result<crate::cap::TaskPollOutcome, crate::cap::TaskWaitError> {
    poll_task_handle_in(&TrapTransport::new(), thread.handle, scratch.report)
}

/// Issue one unflagged join probe.  `WouldBlock` is the one positive live-task
/// result; unlike the blocking path, an OCC conflict or unrelated park status
/// is surfaced so a nonblocking caller controls its own retry policy.
#[cfg(target_os = "none")]
fn poll_task_handle_in(
    transport: &crate::transport::TrapTransport,
    handle: SlotRef,
    report: SlotRef,
) -> Result<crate::cap::TaskPollOutcome, crate::cap::TaskWaitError> {
    use kore_abi::session::Status;

    use crate::session::SessionBuilder;
    use crate::transport::Transport;

    let mut b = SessionBuilder::new();
    b.join(handle, report)
        .map_err(|_| crate::cap::TaskWaitError::Transport)?;
    let committed = transport
        .commit_now(&b)
        .map_err(|_| crate::cap::TaskWaitError::Transport)?;
    validate_task_completion(committed)?;
    let status = committed.status();
    let rec = if status == Status::Exited {
        Some(read_task_report(report)?)
    } else {
        None
    };
    classify_task_poll(status, rec.as_ref())
}

/// Read one kernel-written, bind-inert task report through the founder-mapped
/// read-only slots alias.
#[cfg(target_os = "none")]
fn read_task_report(
    report: SlotRef,
) -> Result<kore_abi::cap::CapRecord, crate::cap::TaskWaitError> {
    let rptr = crate::entry::geometry::slot_record_va(report)
        .ok_or(crate::cap::TaskWaitError::MalformedReport)?
        as *const kore_abi::cap::CapRecord;
    // SAFETY: `report` is reserved by the caller from its own bounded arena;
    // the founder mapped this alias read-only, and the kernel completed the
    // join write before returning `Status::Exited`.  Volatile prevents the
    // compiler from reusing a pre-trap vacant-record read.
    Ok(unsafe { core::ptr::read_volatile(rptr) })
}

/// Validate the exact completion envelope of a one-invocation join before
/// interpreting its status. The ABI's fail-closed `status()` maps unknown raw
/// values to `Malformed`; retaining the raw check prevents a corrupt transport
/// word from masquerading as an engine-authored refusal.
#[cfg(any(target_os = "none", test))]
fn validate_task_completion(
    committed: kore_abi::session::Committed,
) -> Result<(), crate::cap::TaskWaitError> {
    if committed.failing_index() != 0
        || committed.raw() as u16 > kore_abi::session::Status::AltDelivery as u16
    {
        Err(crate::cap::TaskWaitError::MalformedCompletion)
    } else {
        Ok(())
    }
}

/// Preserve the full status after the shared ABI decoder validates the
/// canonical report. Every lifecycle consumer uses that same decoder.
#[cfg(any(target_os = "none", test))]
fn decode_task_exit_report(
    rec: &kore_abi::cap::CapRecord,
) -> Result<i32, crate::cap::TaskWaitError> {
    kore_abi::session::ExitReport::decode(rec)
        .map(|report| report.status)
        .ok_or(crate::cap::TaskWaitError::MalformedReport)
}

/// One decoded step of the blocking wait loop.  Kept pure so every status in
/// the closed ABI vocabulary is host-testable without simulating a transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(any(target_os = "none", test))]
enum TaskWaitStep {
    Retry,
    Complete(crate::cap::TaskWaitOutcome),
}

#[cfg(any(target_os = "none", test))]
fn classify_task_wait(
    status: kore_abi::session::Status,
    report: Option<&kore_abi::cap::CapRecord>,
) -> Result<TaskWaitStep, crate::cap::TaskWaitError> {
    use kore_abi::session::Status;

    match status {
        Status::Exited => {
            decode_task_exit_report(report.ok_or(crate::cap::TaskWaitError::MalformedReport)?)
                .map(crate::cap::TaskWaitOutcome::Exited)
                .map(TaskWaitStep::Complete)
        }
        // Reaped rather than exited — no status was ever recorded.
        Status::Revoked => Ok(TaskWaitStep::Complete(crate::cap::TaskWaitOutcome::Reaped)),
        // Some other arm of a composite park won. Nothing was learned about
        // the end, but this pass really parked and may safely re-establish the
        // wait without becoming a syscall-only loop.
        Status::Ok | Status::Deadline | Status::Irq => Ok(TaskWaitStep::Retry),
        // No waiter was installed, or the OCC commit lost. `WouldBlock` covers
        // both a full bounded joiner set and a caller which could not establish
        // its park, so userland cannot distinguish them. An immediate trap
        // retry can spin forever; return the borrowed handle to caller-owned
        // backoff.
        Status::Conflict | Status::WouldBlock => Err(crate::cap::TaskWaitError::Retryable),
        // A refusal (a bad handle or malformed session) cannot make progress
        // by spinning in the blocking loop.
        status => Err(crate::cap::TaskWaitError::Refused(status)),
    }
}

/// Decode one unflagged task probe.  `WouldBlock` is the only affirmative
/// live-task result; all other nonterminal statuses are surfaced so callers
/// choose their own nonblocking retry policy.
#[cfg(any(target_os = "none", test))]
fn classify_task_poll(
    status: kore_abi::session::Status,
    report: Option<&kore_abi::cap::CapRecord>,
) -> Result<crate::cap::TaskPollOutcome, crate::cap::TaskWaitError> {
    use kore_abi::session::Status;

    match status {
        Status::WouldBlock => Ok(crate::cap::TaskPollOutcome::Running),
        Status::Exited => {
            decode_task_exit_report(report.ok_or(crate::cap::TaskWaitError::MalformedReport)?)
                .map(crate::cap::TaskPollOutcome::Exited)
        }
        Status::Revoked => Ok(crate::cap::TaskPollOutcome::Reaped),
        status => Err(crate::cap::TaskWaitError::Refused(status)),
    }
}

/// Off the metal userland targets: nothing was ever spawned, so nothing
/// joins — the honest typed refusal.
#[cfg(not(target_os = "none"))]
pub fn join(thread: Thread) -> Result<Option<i32>, ThreadJoinError> {
    Err(ThreadJoinError {
        thread,
        cause: crate::cap::TaskWaitError::Transport,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_exit_separates_occ_retry_from_task_backoff() {
        use kore_abi::session::{Committed, Status};

        assert_eq!(
            exit_commit_action(Committed::from_raw(Status::Conflict.encode(0))),
            ExitCommitAction::Retry
        );
        assert_eq!(
            exit_commit_action(Committed::from_raw(Status::WouldBlock.encode(0))),
            ExitCommitAction::Backoff
        );
        for status in [Status::Conflict, Status::WouldBlock] {
            assert_eq!(
                exit_commit_action(Committed::from_raw(status.encode(1))),
                ExitCommitAction::Refuse
            );
        }
        for status in [
            Status::Ok,
            Status::Denied,
            Status::Forged,
            Status::Malformed,
            Status::Revoked,
            Status::Exited,
            Status::Deadline,
            Status::Irq,
        ] {
            assert_eq!(
                exit_commit_action(Committed::from_raw(status.encode(0))),
                ExitCommitAction::Refuse
            );
        }
    }

    #[test]
    fn worker_bootstrap_does_not_inherit_process_attachments() {
        use kore_abi::process::{
            PROCESS_IO_INDEX, PROCESS_IO_ROLE, PROCESS_OUTPUT_WAKE_INDEX, PROCESS_OUTPUT_WAKE_ROLE,
            PROCESS_STDIO_HOST_CHANNEL_INDEX, PROCESS_STDIN_WAKE_INDEX, PROCESS_STDIN_WAKE_ROLE,
            PROCESS_STDIO_HOST_INDEX, PROCESS_STDIO_HOST_ROLE, PROCESS_STDIO_INDEX,
            PROCESS_STDIO_ROLE, PROCESS_STDIO_WAKE_RIGHTS,
        };

        assert_eq!(
            PROCESS_STDIO_WAKE_RIGHTS,
            kore_abi::cap::Rights::ACCESS | kore_abi::cap::Rights::REVOKE
        );
        assert!(!PROCESS_STDIO_WAKE_RIGHTS.contains(kore_abi::cap::Rights::DERIVE));

        let block = worker_boot_block(&[]).expect("an unendowed worker boot block fits");
        let bootstrap = kore_abi::task::Bootstrap::of(&block).expect("worker block validates");
        assert_eq!(bootstrap.handles().count(), 3);
        for (role, index) in [
            (PROCESS_IO_ROLE, PROCESS_IO_INDEX),
            (PROCESS_STDIO_ROLE, PROCESS_STDIO_INDEX),
            (PROCESS_STDIN_WAKE_ROLE, PROCESS_STDIN_WAKE_INDEX),
            (PROCESS_OUTPUT_WAKE_ROLE, PROCESS_OUTPUT_WAKE_INDEX),
            (PROCESS_STDIO_HOST_ROLE, PROCESS_STDIO_HOST_INDEX),
            (PROCESS_STDIO_HOST_ROLE, PROCESS_STDIO_HOST_CHANNEL_INDEX),
        ] {
            assert_eq!(
                bootstrap.handle(role, index),
                None,
                "shared address space must not imply inherited holding {role:#x}@{index}"
            );
        }
    }

    /// The heap endowment is a carve-and-map sublease of the parent's
    /// supply, under the name the allocator's grow looks up: never the
    /// teardown right, which stays with the parent.
    #[test]
    fn the_heap_endowment_is_a_carve_and_map_sublease() {
        use kore_abi::cap::{ResourceKind, Rights};

        let heap = WorkerCap::heap(SlotRef::from_wire(71));
        assert_eq!((heap.role, heap.index), (role::HEAP, 0));
        assert_eq!(heap.subject, SlotRef::from_wire(71));
        assert_eq!(heap.kind, ResourceKind::MemLease as u16 as u8);
        assert_eq!(heap.rights, Rights::ACCESS | Rights::MODIFY | Rights::DERIVE);
        assert!(!heap.rights.contains(Rights::REVOKE));
    }

    /// A planned gang endowment — none, or the star's mesh — is closed
    /// with the heap, so every compute worker can grow the allocator; a
    /// task without a heap adds nothing, and a heap that does not fit is
    /// refused rather than dropped (#1316).
    #[test]
    fn a_planned_gang_endowment_closes_with_the_heap() {
        let heap = WorkerCap::heap(SlotRef::from_wire(71));
        let doorbell = WorkerCap {
            role: role::DOORBELL,
            ..VACANT_CAP
        };

        let mut starless = [VACANT_CAP; PLANNED_CAPS];
        assert_eq!(close_with_heap(&mut starless, 0, Some(heap)), Some(1));
        assert_eq!((starless[0].role, starless[0].index), (role::HEAP, 0));

        // The widest mesh still leaves the heap its place.
        let widest = 2 + crate::star::MESH_CEILING;
        let mut meshed = [doorbell; PLANNED_CAPS];
        assert_eq!(close_with_heap(&mut meshed, widest, Some(heap)), Some(widest + 1));
        assert_eq!(meshed[widest].role, role::HEAP);
        assert!(meshed[..widest].iter().all(|cap| cap.role == role::DOORBELL));

        let mut unendowed = [VACANT_CAP; PLANNED_CAPS];
        assert_eq!(close_with_heap(&mut unendowed, 3, None), Some(3));
        assert_eq!(unendowed[3].role, 0);

        let mut full = [doorbell; 2];
        assert_eq!(close_with_heap(&mut full, 2, Some(heap)), None);
    }

    /// A meshed worker's planned endowment, the heap included, is a lawful
    /// founding endowment at the widest chunk.
    #[test]
    fn the_widest_planned_endowment_fits_a_founding() {
        assert!(PLANNED_CAPS <= MAX_WORKER_CAPS);
        let mut plan = [VACANT_CAP; PLANNED_CAPS];
        for (i, cap) in plan.iter_mut().enumerate() {
            *cap = WorkerCap {
                role: role::DOORBELL,
                index: crate::futex::PEER_BASE + i as u16,
                ..VACANT_CAP
            };
        }
        let heap = WorkerCap::heap(SlotRef::from_wire(71));
        assert_eq!(close_with_heap(&mut plan, PLANNED_CAPS - 1, Some(heap)), Some(PLANNED_CAPS));
        let block = worker_boot_block(&plan).expect("the widest planned endowment fits a boot block");
        let bootstrap = kore_abi::task::Bootstrap::of(&block).expect("worker block validates");
        assert!(bootstrap.handle(role::HEAP, 0).is_some());
    }

    #[test]
    fn worker_bootstrap_installs_an_explicit_heap_supply() {
        let parent_heap = SlotRef::from_wire(71);
        let block = worker_boot_block(&[WorkerCap::heap(parent_heap)])
            .expect("one runtime heap endowment fits");
        let bootstrap = kore_abi::task::Bootstrap::of(&block).expect("worker block validates");

        assert_eq!(
            bootstrap.handle(role::HEAP, 0),
            Some(SlotRef::from_wire(u32::from(child_slot::ENDOW_BASE)))
        );
        assert_ne!(
            bootstrap.handle(role::HEAP, 0),
            Some(parent_heap),
            "the child names its delivered holding, never the parent's slot coordinate"
        );
    }

    #[test]
    fn worker_geometry_names_one_shared_process_root_block() {
        let local = BANDS
            .windows(7)
            .expect("the tested worker lies inside the bounded band")
            .thread_block;
        let mut geom_block = GeomBlock::new();
        assert!(push_worker_thread_blocks(&mut geom_block, local));
        assert_eq!(
            geom_block.window_indexed(geom::THREAD_BLOCK, 0),
            Some((local.base(), local.len()))
        );
        assert_eq!(
            geom_block.window_indexed(geom::THREAD_BLOCK, 1),
            Some((
                BANDS.thread_block.first().base(),
                BANDS.thread_block.first().len()
            )),
            "every worker must resolve the same VAS-local attachment word"
        );
    }

    #[test]
    fn task_join_and_poll_preserve_all_exit_status_bits() {
        use kore_abi::session::{ExitReport, Status};
        for status in [0, 255, 256, 257, 65536, -1, -256, i32::MIN, i32::MAX] {
            let report = ExitReport { status }.encode();
            assert_eq!(
                classify_task_wait(Status::Exited, Some(&report)),
                Ok(TaskWaitStep::Complete(crate::cap::TaskWaitOutcome::Exited(
                    status
                )))
            );
            assert_eq!(
                classify_task_poll(Status::Exited, Some(&report)),
                Ok(crate::cap::TaskPollOutcome::Exited(status))
            );
        }
    }

    #[test]
    fn task_exit_report_requires_the_exact_bind_inert_encoding() {
        let canonical = kore_abi::session::ExitReport { status: -37 }.encode();
        assert_eq!(decode_task_exit_report(&canonical), Ok(-37));

        assert_eq!(
            decode_task_exit_report(&kore_abi::cap::CapRecord::vacant()),
            Err(crate::cap::TaskWaitError::MalformedReport)
        );

        // A stale capability-shaped field must not be accepted merely because
        // the report markers happen to match.
        let mut noncanonical = canonical;
        noncanonical.rights = kore_abi::cap::Rights::ACCESS;
        assert_eq!(
            decode_task_exit_report(&noncanonical),
            Err(crate::cap::TaskWaitError::MalformedReport)
        );

        // A truncated rather than sign-extended payload decodes to the same
        // low i32 through a permissive cast, but is not what ExitReport encoded.
        let mut truncated = canonical;
        truncated.scope_rid = u64::from(u32::MAX - 36);
        assert_eq!(
            decode_task_exit_report(&truncated),
            Err(crate::cap::TaskWaitError::MalformedReport)
        );
    }

    #[test]
    fn blocking_task_wait_reparks_only_after_an_established_wait() {
        use kore_abi::session::Status;

        for status in [Status::Ok, Status::Deadline, Status::Irq] {
            assert_eq!(classify_task_wait(status, None), Ok(TaskWaitStep::Retry));
        }
        for status in [Status::Conflict, Status::WouldBlock] {
            assert_eq!(
                classify_task_wait(status, None),
                Err(crate::cap::TaskWaitError::Retryable),
                "a non-parking result must return control for backoff"
            );
        }

        let exited = kore_abi::session::ExitReport { status: 73 }.encode();
        assert_eq!(
            classify_task_wait(Status::Exited, Some(&exited)),
            Ok(TaskWaitStep::Complete(crate::cap::TaskWaitOutcome::Exited(
                73
            )))
        );
        assert_eq!(
            classify_task_wait(Status::Revoked, None),
            Ok(TaskWaitStep::Complete(crate::cap::TaskWaitOutcome::Reaped))
        );
        assert_eq!(
            classify_task_wait(Status::Exited, None),
            Err(crate::cap::TaskWaitError::MalformedReport)
        );
        assert_eq!(
            classify_task_wait(Status::Denied, None),
            Err(crate::cap::TaskWaitError::Refused(Status::Denied))
        );
    }

    #[test]
    fn nonblocking_task_poll_has_a_closed_lifecycle_result() {
        use kore_abi::session::Status;

        assert_eq!(
            classify_task_poll(Status::WouldBlock, None),
            Ok(crate::cap::TaskPollOutcome::Running)
        );
        let exited = kore_abi::session::ExitReport { status: -9 }.encode();
        assert_eq!(
            classify_task_poll(Status::Exited, Some(&exited)),
            Ok(crate::cap::TaskPollOutcome::Exited(-9))
        );
        assert_eq!(
            classify_task_poll(Status::Revoked, None),
            Ok(crate::cap::TaskPollOutcome::Reaped)
        );

        for status in [
            Status::Ok,
            Status::Conflict,
            Status::Denied,
            Status::Forged,
            Status::Malformed,
            Status::Deadline,
            Status::Irq,
        ] {
            assert_eq!(
                classify_task_poll(status, None),
                Err(crate::cap::TaskWaitError::Refused(status))
            );
        }
        assert_eq!(
            classify_task_poll(Status::Exited, None),
            Err(crate::cap::TaskWaitError::MalformedReport)
        );
    }

    #[test]
    fn task_join_rejects_impossible_completion_envelopes() {
        use kore_abi::session::{Committed, Status};

        assert_eq!(
            validate_task_completion(Committed::from_raw(Status::Exited.encode(0))),
            Ok(())
        );
        assert_eq!(
            validate_task_completion(Committed::from_raw(Status::WouldBlock.encode(1))),
            Err(crate::cap::TaskWaitError::MalformedCompletion)
        );
        assert_eq!(
            validate_task_completion(Committed::from_raw(0xffff)),
            Err(crate::cap::TaskWaitError::MalformedCompletion)
        );
    }

    /// Every window thread `n` is provisioned with, in one list — the shape
    /// a collision proof has to run over.
    fn all_windows(n: u32) -> [Window; 7] {
        let w = BANDS.windows(n).expect("thread index within the plan");
        [
            w.map,
            w.scratch,
            w.thread_block,
            w.stage,
            w.boot,
            w.slots,
            w.stack,
        ]
    }

    /// The band spans do not tread on each other — the runtime mirror of
    /// the module's `const _: () = assert!(…)`, so the property shows up as
    /// a named failing test and not only as a build error.
    #[test]
    fn the_bands_are_pairwise_disjoint() {
        assert!(Window::all_disjoint(&BANDS.spans()));
    }

    /// The plan is finite and says so.  This is the bound that did not
    /// exist: the old counter handed out `BAND + n * STRIDE` for any `n`.
    #[test]
    fn the_band_plan_refuses_past_its_ceiling() {
        assert!(BANDS.windows(MAX_THREADS - 1).is_some());
        assert!(BANDS.windows(MAX_THREADS).is_none());
        assert!(BANDS.windows(u32::MAX).is_none());
    }

    /// No thread's window overlaps any other thread's — the property the
    /// stride convention only ever asserted in a comment.
    #[test]
    fn no_two_threads_share_a_window() {
        for a in 0..MAX_THREADS {
            for b in (a + 1)..MAX_THREADS {
                for (i, wa) in all_windows(a).iter().enumerate() {
                    for (j, wb) in all_windows(b).iter().enumerate() {
                        assert!(
                            !wa.overlaps(wb),
                            "thread {a} window {i} overlaps thread {b} window {j}"
                        );
                    }
                }
            }
        }
    }

    /// A single thread's own seven windows are mutually disjoint too — a
    /// band placed on top of a sibling band would corrupt one thread all by
    /// itself.
    #[test]
    fn one_threads_windows_do_not_collide() {
        for n in 0..MAX_THREADS {
            assert!(
                Window::all_disjoint(&all_windows(n)),
                "thread {n}'s own windows collide"
            );
        }
    }

    /// Every window a thread receives lies inside the band it came from —
    /// the containment that makes the band spans a sufficient proof.
    #[test]
    fn every_thread_window_stays_inside_its_band() {
        for n in 0..MAX_THREADS {
            let w = BANDS.windows(n).unwrap();
            for (span, win) in BANDS.spans().iter().zip(all_windows(n)) {
                assert!(span.contains(&win), "thread {n} escaped its band");
            }
            // Slot n really is the n-th stride, not merely *somewhere*
            // inside the band.
            assert_eq!(
                w.stack.base(),
                BANDS.stack.first().base() + u64::from(n) * BANDS.stack.stride()
            );
        }
    }

    /// Thread 0 is the task itself, so the aliased bands must start exactly
    /// where the founder maps the task's own windows.  (The founder holds
    /// the compile-time half of this against its declared plan; this end
    /// pins the addresses the rt believes in.)
    #[test]
    fn slot_zero_is_the_tasks_own_plan() {
        assert_eq!(BANDS.scratch.first().base(), 0x6000_0000);
        assert_eq!(BANDS.thread_block.first().base(), 0x6800_0000);
        assert_eq!(BANDS.slots.first().base(), 0x7000_0000);
        // The map band aliases the task's plan by its RESERVATION (the span
        // no other window may tread on); slot 0 is the task's own cursor.
        assert_eq!(BANDS.map.span(), Window::new(0x8000_0000, 0x2000_0000));
        assert_eq!(BANDS.map.first(), Window::new(0x8000_0000, 0x0100_0000));
    }

    /// Every thread recovers its own identity from the block window it was
    /// handed, and no two recover the same one.
    ///
    /// This is the allocator's injectivity premise stated over the *real*
    /// band rather than a synthetic one: `current_shard` reads this slot out
    /// of the anchored runtime block, and the owner-only free-lists are
    /// sound exactly while distinct live threads report distinct indices.
    #[test]
    fn every_thread_block_slot_names_exactly_its_own_thread() {
        let mut seen = [false; MAX_THREADS as usize];
        for n in 0..MAX_THREADS {
            let base = BANDS
                .thread_block
                .nth(n)
                .expect("every slot below MAX_THREADS exists")
                .base();
            assert_eq!(
                BANDS.thread_block.slot_of(base),
                Some(n),
                "thread {n}'s block window must name thread {n}"
            );
            assert!(!seen[n as usize], "two threads claimed one shard index");
            seen[n as usize] = true;
        }
        // That each of these slots also gets its *own* shard worker is the
        // `MAX_THREADS <= NUM_SHARDS` assertion in `heap::shard`, held at
        // compile time beside the array it constrains.
    }

    /// Each stack sits in a stride wide enough for the guard page mapped
    /// below the next one.
    #[test]
    fn every_stack_has_room_for_its_guard_page() {
        assert!(BANDS.stack.gap() >= PAGE);
        for n in 1..MAX_THREADS {
            let prev = BANDS.stack.nth(n - 1).unwrap();
            let this = BANDS.stack.nth(n).unwrap();
            assert!(this.base() - prev.end() >= PAGE, "no guard room below {n}");
        }
    }

    /// The ceiling is derived from the reservation the map band subdivides,
    /// not picked: changing either the reservation or the stride must move it.
    #[test]
    fn max_threads_is_the_map_bands_capacity() {
        assert_eq!(MAX_THREADS, BANDS.map.count());
        assert_eq!(
            BANDS.map.span().len() / BANDS.map.stride(),
            u64::from(MAX_THREADS)
        );
    }

    /// **#428, the property this whole band exists to buy**: the span a
    /// thread's *map cursor* is installed over is its own stride, so the
    /// cursor cannot reach the next thread's window however much that thread
    /// maps.  Thread 0 is the case that was broken — its cursor came from the
    /// founder and spanned all eight strides — so it is checked first and by
    /// name, but the property is every thread's.
    ///
    /// The cursor span IS the stamped window (`geom::MAP`), which is why this
    /// is checkable here at all: `spawn` pushes `w.map` whole, and the
    /// founder pushes `LAYOUT.map_cursor()` — band slot 0, asserted against
    /// `BANDS.map.first()` over in `kore-svc-exec`.
    #[test]
    fn a_threads_map_cursor_cannot_reach_the_next_threads_window() {
        let zero = BANDS.windows(0).unwrap().map;
        let one = BANDS.windows(1).unwrap().map;
        assert!(
            zero.precedes(&one),
            "thread 0's cursor runs into thread 1's map window"
        );
        // Not just the pair: no thread's cursor reaches any later window,
        // and the band's own span is the only thing that bounds the last.
        for n in 0..MAX_THREADS {
            let cursor = BANDS.windows(n).unwrap().map;
            assert_eq!(cursor.len(), BANDS.map.stride(), "a cursor is one stride");
            assert!(BANDS.map.span().contains(&cursor));
            for m in (n + 1)..MAX_THREADS {
                let later = BANDS.windows(m).unwrap().map;
                assert!(
                    cursor.precedes(&later),
                    "thread {n}'s cursor reaches thread {m}'s map window"
                );
            }
        }
        // And the whole plan's cursors together are exactly the reservation:
        // no stride is unowned, none is owned twice.
        assert_eq!(
            BANDS.map.stride() * u64::from(MAX_THREADS),
            BANDS.map.span().len()
        );
    }

    /// **The band index is reusable** — the property join exists to buy.
    /// It used to be a saturating counter with no free list, which made
    /// [`MAX_THREADS`] a *lifetime* budget: seven threads per task, ever.
    ///
    /// Kept off the `#[cfg]`-gated allocator so it runs in `make test`: the
    /// bitmap is pure data, and this is the arithmetic that decides whether
    /// two live threads can share a band.
    #[test]
    fn the_band_plan_is_a_live_population_not_a_lifetime_budget() {
        // A local mirror of `claim_thread_index`/`release_thread_index`
        // over the same word — the allocator itself is userland-target
        // only, and what is being pinned here is its arithmetic.
        fn claim(slots: &AtomicU32) -> Option<u32> {
            let mut used = slots.load(Ordering::Relaxed);
            loop {
                let free = !used & (((1u64 << MAX_THREADS) - 1) as u32);
                if free == 0 {
                    return None;
                }
                let n = free.trailing_zeros();
                match slots.compare_exchange_weak(
                    used,
                    used | (1 << n),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Some(n),
                    Err(observed) => used = observed,
                }
            }
        }
        fn release(slots: &AtomicU32, n: u32) {
            if n > 0 && n < MAX_THREADS {
                slots.fetch_and(!(1 << n), Ordering::Release);
            }
        }

        let slots = AtomicU32::new(1);
        // Slot 0 is the task's own plan and is never handed out — the old
        // wrapping counter eventually would have, handing a thread the
        // task's own windows.
        let claimed: std::vec::Vec<u32> = (0..MAX_THREADS - 1)
            .map(|_| claim(&slots).expect("within the plan"))
            .collect();
        assert!(
            claimed.iter().all(|&n| n != 0),
            "slot 0 is never handed out"
        );
        assert_eq!(claimed.len(), (MAX_THREADS - 1) as usize);
        assert!(claim(&slots).is_none(), "a full plan refuses");
        // The whole point: give one back and it comes round again.
        release(&slots, claimed[3]);
        assert_eq!(
            claim(&slots),
            Some(claimed[3]),
            "a released band is reusable"
        );
        assert!(claim(&slots).is_none());
        // Releasing slot 0 is a no-op — the task's own plan is not a
        // resource anyone may hand out.
        release(&slots, 0);
        assert!(claim(&slots).is_none(), "slot 0 stays the task's own");
        // Every index the plan can hand out has windows; the ceiling and
        // the bitmap agree.
        for n in 1..MAX_THREADS {
            assert!(BANDS.windows(n).is_some());
        }
    }

    /// A full arena's alias fits the slots window — the one per-thread
    /// window whose length is computed rather than declared, and so the
    /// only one `spawn` has to check.
    ///
    /// Derived from [`ARENA_SLOT_CEILING`], never spelled: this test read
    /// `64 * 128` until the ceiling moved, at which point it asserted a
    /// 65-slot alias does not fit a window that had just grown to hold
    /// 128.  A literal here tests the number, not the invariant.
    ///
    /// [`ARENA_SLOT_CEILING`]: crate::entry::ARENA_SLOT_CEILING
    #[test]
    fn a_full_arena_alias_fits_its_slots_window() {
        const CEILING: u64 = crate::entry::ARENA_SLOT_CEILING as u64;
        let w = BANDS.windows(1).unwrap();
        let alias = Window::new(
            w.slots.base(),
            kore_abi::execution::alias_bytes(CEILING as u32),
        );
        assert!(w.slots.contains(&alias));
        // One slot more would not fit — exactly what `spawn` refuses on.
        let over = Window::new(
            w.slots.base(),
            kore_abi::execution::alias_bytes((CEILING + 1) as u32),
        );
        assert!(!w.slots.contains(&over));
    }
}
