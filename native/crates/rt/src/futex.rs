//! The futex — **block until another thread changes a word**.
//!
//! Everything else in this runtime waits on something the *kernel* knows
//! about: a delivery, an interrupt route, a deadline, a task's death. A futex
//! waits on something the kernel has never heard of — a `u32` in this task's
//! own memory — and that is the whole difficulty. The kernel cannot wake a
//! waiter it does not know is waiting, so the *bookkeeping* has to live here,
//! in memory the threads already share, and only the final "wake that thread"
//! step goes through capability authority.
//!
//! # Why this needs no kernel verb
//!
//! Threads of one task share an address space (`SpawnFlags::SHARE_SPACE`), so
//! a `static` here is one object seen by every thread at one address — which
//! is exactly the registry a futex needs, for free. What sharing memory does
//! *not* give is the right to wake a sibling: holdings are per task and a
//! thread is a task, so waking thread B means posting a notice on a
//! `ChanSend` into B's doorbell, held in the waker's own arena and found
//! through this thread's own boot block at `role::DOORBELL@(2 + B)`.
//!
//! Both halves already existed. The park is the executor's
//! ([`park_standing`](crate::session::SessionBuilder::park_standing) with an
//! optional folded deadline) and the wake is the relay thread's
//! ([`verbs::post_notice`](crate::cap::verbs::post_notice)). This module is
//! the registry between them, and not one line of kernel.
//!
//! # The lost wakeup, and why the order below is the order
//!
//! The hazard every futex has to answer: thread A checks the word, decides to
//! sleep, and thread B changes the word and wakes before A is actually
//! parked. A sleeps forever on a condition that already happened.
//!
//! Two mechanisms rule it out here, and both are orderings rather than locks:
//!
//! 1. **Arm before you check.** [`wait`] publishes its doorbell's standing
//!    post *first*, then registers in `WAITING_ON`, then reads the word.
//!    The kernel resolves a park by asking whether the post was consumed —
//!    "a consumed post means delivery ([`Status::Ok`])" — so a notice landing
//!    any time after the post stands resolves the *next* park immediately
//!    rather than being dropped. There is no window in which a notice arrives
//!    "too early": early is the same as on time. A notice arriving before the
//!    post exists is the one that vanishes, which is why the post comes
//!    first.
//!
//! 2. **Register before you read; write before you scan.** A waiter claims a
//!    registry slot holding its address and *then* loads the word; [`wake`]'s
//!    caller stores the new word and *then* scans the registry. Both pairs
//!    are `SeqCst`, so the two cannot both miss each other: if the waiter's
//!    store lands before the waker's scan the waker sees it and kicks, and if
//!    it lands after, the waiter's load necessarily sees the waker's new
//!    value and never parks. This is Dekker's argument, and it is why the
//!    `SeqCst` on those four accesses is load-bearing rather than
//!    conservative — `Acquire`/`Release` permit exactly the store-load
//!    reordering that reopens the hole.
//!
//!    The argument is per *waiter*, so each needs its own slot. A registry
//!    with one entry per thread would satisfy it only while a thread had one
//!    waiter — true of the blocking form, false the moment two futures on one
//!    executor both wait.
//!
//! [`Status::Ok`]: kore_abi::session::Status::Ok
//!
//! # Who can wake whom, today
//!
//! Waiting needs only this thread's own doorbell, so **any thread the runtime
//! will run a service on can [`wait`]** — `service_ready` already refuses a
//! thread without one, so that is not a new requirement.
//!
//! Waking is the asymmetric half, because a send end has to be *held*. Peer
//! `i`'s send end is looked up at `role::DOORBELL@(2 + i)`, and index 2 is
//! the self-kick a founder already carves — so a thread endowed with it can
//! wake **thread 0** with nothing added, which is the endowment the input
//! service's relay thread already takes. Higher indices are the peer mesh,
//! and `thread::spawn_with` does not provision one yet: [`wake`] returns
//! `false` for a peer this thread holds no send end for, rather than
//! reporting a kick it did not deliver.
//!
//! That asymmetry is a provisioning gap, not a design one — filling it is
//! minting `MAX_THREADS` doorbell pairs in the spawning thread and endowing
//! each child with the ends, which needs a `role::CHANNEL_SPACE` grant in the
//! task that spawns. Nothing in the API changes when it lands.
//!
//! # Two forms of the wait, one registry
//!
//! [`wait`] blocks the thread in the kernel; [`wait_async`] yields to the
//! executor. They share `WAITING_ON`, share the orderings above, and are
//! indistinguishable to a waker — [`wake`] neither knows nor needs to know
//! which one its target used.
//!
//! The split exists because "stop running" means two different things here. A
//! bare thread stops by parking, and must arm its own post first. A future
//! stops by returning `Pending`, and must *not* touch the post at all: the
//! executor owns it, keeps it standing, and re-posts it after every notice it
//! consumes. A future that parked in the kernel instead would freeze its
//! sibling futures and swallow an edge one of them was owed — so the rule is
//! simply that async code uses the async form, and it is a correctness rule
//! rather than a stylistic one.
//!
//! # What a caller must still do
//!
//! **Re-check in a loop.** A return from [`wait`] is never a promise that the
//! condition holds: the doorbell is shared with everything else that kicks
//! this thread (an executor's peer, a service reply, another futex on another
//! address), and the cookie is a hint rather than a filter. Spurious wakeups
//! are therefore ordinary and frequent, not a rare kernel quirk. Rust's own
//! futex contract says the same thing, which is why `std`'s `Mutex` and
//! `Condvar` both loop; this module is shaped to be the platform half of
//! exactly that contract.

use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use core::time::Duration;

use crate::thread::MAX_THREADS;

/// How many waits one thread can have outstanding at once.
///
/// A *thread* can only be inside one blocking [`wait`] — it is one call stack.
/// But a thread running an executor can have the root future plus every
/// admitted task suspended inside [`wait_async`] simultaneously, each on its
/// own address, because a suspended future is not a call frame. So the bound
/// is the executor's task capacity plus the root, which is the most futures
/// one thread can ever have parked at once.
///
/// Sized rather than assumed-away because the alternative is silent: with one
/// entry per thread, a second waiter overwrites the first's registration, and
/// a later [`wake`] finds nothing to kick and reports an ordinary `false` —
/// a lost wakeup with no diagnostic anywhere.
const WAITS_PER_THREAD: usize = crate::executor::MAX_TASKS + 1;

/// The registry: **which addresses each thread has waiters on**, or
/// [`NOT_WAITING`].
///
/// A flat array indexed by thread identity, then by wait slot: no hashing, no
/// allocation, no collision chains, and a scan bounded by [`MAX_THREADS`] ×
/// [`WAITS_PER_THREAD`]. Both bounds are ones the tree already lives under —
/// the thread bands and the executor's task slab — rather than new ceilings
/// invented here.
///
/// Only thread `t` ever writes `WAITING_ON[t]`, so the claim needs no
/// cross-thread mutual exclusion; peers only ever read it, scanning for an
/// address to kick.
///
/// Shared because threads share an address space. This is the one piece of
/// futex state that must be *process*-global rather than per-thread, and it
/// is also the only piece that can be: a `SlotRef` is arena-local and means
/// nothing in a sibling's arena, so capabilities can never live here.
static WAITING_ON: [[AtomicUsize; WAITS_PER_THREAD]; MAX_THREADS as usize] =
    [const { [const { AtomicUsize::new(NOT_WAITING) }; WAITS_PER_THREAD] }; MAX_THREADS as usize];

/// Claim a wait slot for this thread on `addr`, or `None` if all are taken.
///
/// Exhaustion is structurally unreachable — [`WAITS_PER_THREAD`] is sized to
/// the most waiters one thread can host — but it is reported rather than
/// asserted, because the cost of being wrong about that is a fail-closed
/// refusal here and a lost wakeup if it were ignored.
fn claim_wait(thread: usize, addr: usize) -> Option<usize> {
    for (slot, cell) in WAITING_ON[thread].iter().enumerate() {
        if cell
            .compare_exchange(NOT_WAITING, addr, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            return Some(slot);
        }
    }
    None
}

/// A claimed wait slot, released when the waiter stops waiting **however it
/// stops**.
///
/// The reason this is a guard and not a pair of calls: [`wait_async`] returns
/// a future, and a future can be dropped while pending — a `select`, a
/// timeout, a cancelled task. A plain release after the loop never runs on
/// that path, so the slot stays claimed forever, holding an address nobody is
/// waiting on. Repeat it and the thread exhausts [`WAITS_PER_THREAD`], after
/// which every later wait on that thread refuses; before that, [`wake`] can
/// match the stale entry and report having woken a waiter that does not exist.
///
/// Cancellation is not an exotic path for a future, so the release has to be
/// tied to the value's lifetime rather than to one exit of one function.
struct WaitSlot {
    thread: usize,
    slot: usize,
}

impl Drop for WaitSlot {
    fn drop(&mut self) {
        release_wait(self.thread, self.slot);
    }
}

/// Release the wait slot this waiter claimed — **its own**, by index.
///
/// By index rather than by address: two waiters on one thread may hold the
/// same address, and clearing "the entry that matches" could retire a sibling
/// waiter's registration instead of this one's.
fn release_wait(thread: usize, slot: usize) {
    WAITING_ON[thread][slot].store(NOT_WAITING, Ordering::SeqCst);
}

/// Whether `thread` has any waiter registered on `addr`.
///
/// One kick per thread is enough however many of its futures are waiting: the
/// executor's drain is total, so a single notice re-polls all of them.
fn thread_waits_on(thread: usize, addr: usize) -> bool {
    WAITING_ON[thread]
        .iter()
        .any(|cell| cell.load(Ordering::SeqCst) == addr)
}

/// The vacant registry entry.
///
/// Zero, and safely so: the sentinel has to be an address no futex word can
/// have, and address 0 is unmapped in every kore task (the null page is never
/// mapped, so a `&AtomicU32` at 0 cannot exist). Using a separate `Option`
/// word instead would need the flag and the address to be written
/// atomically *together*, which is precisely the tearing this avoids.
const NOT_WAITING: usize = 0;

/// How [`wait`] came back.
///
/// A typed answer rather than `std`'s bare `bool` because three of these four
/// are useful to a caller and the fourth must never be mistaken for a
/// timeout: a runtime that cannot park at all has to be distinguishable from
/// one whose deadline elapsed, or a `Mutex` built on top would spin at full
/// speed while reporting that it was politely waiting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitOutcome {
    /// The word did not hold `expected` — the caller's condition had already
    /// changed, and no park was attempted. The common fast path under
    /// contention, and not a failure.
    Retry,
    /// The thread parked and was woken. **Says nothing about the word**: it
    /// may be a matching [`wake`], an unrelated kick on the shared doorbell,
    /// or a bare scheduler resumption. Re-check the condition.
    Woken,
    /// The park's deadline elapsed with no wake. Only ever returned when the
    /// caller passed a timeout.
    TimedOut,
    /// **This thread cannot wait.** It has no doorbell to park on, no
    /// identity in the thread band, or the build is not on metal. Not a
    /// transient condition and not a spurious wakeup — a caller that loops on
    /// this burns a core forever, so it is a distinct answer rather than a
    /// second flavour of [`Woken`](Self::Woken).
    Unsupported,
}

impl WaitOutcome {
    /// `std`'s `futex_wait` shape: `false` **only** on timeout.
    ///
    /// Provided so the eventual `sys::pal` shim is a rename rather than a
    /// reinterpretation. [`Unsupported`](Self::Unsupported) maps to `true`
    /// deliberately — `std`'s callers treat `false` as "the deadline passed"
    /// and would otherwise convert a refusal into a fabricated timeout.
    pub const fn waited(self) -> bool {
        !matches!(self, Self::TimedOut)
    }
}

/// The cookie a futex wake carries.
///
/// Distinct from the service cookies so a doorbell drain can *say* what woke
/// it, which is the difference between a legible trace and a mystery on a
/// machine with one serial line. It carries no meaning the receiver acts on:
/// the drain is total and the word is the truth, exactly as
/// [`COOKIE_REQUEST`](crate::svc::COOKIE_REQUEST) is a hint about a ring.
pub const COOKIE_FUTEX: u64 = 0x6B72_5F66_7574_5800; // "kr_futX"

/// Where the peer mesh begins in `role::DOORBELL`'s index space —
/// indices 0/1 are a thread's own recv and landing slot, and band `b`'s
/// send sits at `PEER_BASE + b` (the metal `imp` documents why peer 0
/// and the self-kick coincide).  Module-level so the star names hub
/// sends by the same arithmetic `kick` looks them up with.
pub(crate) const PEER_BASE: u16 = 2;

/// The **relay peer** — the star's hub thread (`res/SPMD.md` §3), the one
/// worker that holds a send into every sibling's doorbell.  A waker that
/// finds a registered waiter it holds no peer send for posts ONE notice
/// here instead; the hub kicks everyone it holds.  Numbered one past the
/// band peers so it can never collide with one.
pub(crate) const RELAY: u16 = PEER_BASE + MAX_THREADS as u16;

/// Wait until another thread wakes this one, or `timeout` elapses.
///
/// Parks **only if** `word` still holds `expected` at the moment the wait is
/// armed; otherwise returns [`Retry`](WaitOutcome::Retry) having done
/// nothing. That conditional check is the entire reason a futex is not just a
/// sleep: it closes the gap between "I decided to wait" and "I am waiting",
/// during which the thing being waited for may already have happened.
///
/// `timeout` is relative, and `None` waits indefinitely.
///
/// # Spurious wakeups are normal here
///
/// Call this in a loop that re-tests the real condition. See the module docs:
/// this thread's doorbell is shared with every other reason it might be
/// kicked, so [`Woken`](WaitOutcome::Woken) is a prompt to look, never a
/// proof.
///
/// # Not from inside a future — use [`wait_async`]
///
/// This blocks the calling *thread* in the kernel, so calling it from a task
/// running under an [`Executor`](crate::executor::Executor) stalls every
/// other future on that executor until the wake arrives — and worse, the
/// notice it consumes may be the one a sibling future was waiting for, which
/// this path has no way to hand back (the executor's own
/// `settle_inline_doorbell` exists for exactly that hand-back and needs the
/// running `ExecCtx`). [`wait_async`] is the same wait for that setting, over
/// the same registry; this one is for bare threads and for the blocking shims
/// `std`'s locks will be built from, the same disposition
/// [`block_on`](crate::block::block_on) takes.
pub fn wait(word: &AtomicU32, expected: u32, timeout: Option<Duration>) -> WaitOutcome {
    wait_inner(word, expected, timeout, true, LOCK_SPIN)
}

/// [`wait`] with the **barrier's** longer pre-park monitored budget.
///
/// A lock's counterparty releases in hundreds of nanoseconds, so
/// [`wait`] spins [`LOCK_SPIN`] and parks. A barrier's counterparty is
/// a **co-scheduled sibling** that arrives within a scheduling-jitter
/// window — microseconds to tens of them, not sub-µs — and the monitor
/// makes the wait cheap either way (`UMWAIT` sleeps the core on the
/// watched line; the sibling's own store ends it the instant it
/// arrives, so a longer budget costs nothing when the sibling is
/// prompt and only bounds the tail when it is late). Parking instead
/// trades that for two syscalls and a kick that can miss and ride
/// [`PARK_SLICE`] — the exact w≤4 regression the terminal bench showed as
/// ~500 parks over 512 phases. So the barrier spins [`BARRIER_SPIN`]
/// (a jitter window) before it parks, and resolves the co-resident
/// case in the monitored tier where it belongs.
pub(crate) fn wait_barrier(
    word: &AtomicU32,
    expected: u32,
    timeout: Option<Duration>,
) -> WaitOutcome {
    wait_inner(word, expected, timeout, true, BARRIER_SPIN)
}

/// [`wait`] without the [`stats`] accounting — for the star's hub,
/// whose housekeeping slices would otherwise pollute the very numbers
/// a terminal bench reads to judge the WORKERS' wait behavior.
pub(crate) fn wait_quiet(
    word: &AtomicU32,
    expected: u32,
    timeout: Option<Duration>,
) -> WaitOutcome {
    wait_inner(word, expected, timeout, false, LOCK_SPIN)
}

fn wait_inner(
    word: &AtomicU32,
    expected: u32,
    timeout: Option<Duration>,
    counted: bool,
    spin: SpinBudget,
) -> WaitOutcome {
    let Some(me) = current_thread() else {
        return WaitOutcome::Unsupported;
    };
    let addr = address_of(word);

    // ── 1 · Arm ─────────────────────────────────────────────────────────
    // The standing post must exist before this thread is discoverable as a
    // waiter, or a notice sent in the window between registering and parking
    // would find no post and be dropped (`WouldBlock`, on the floor). Arming
    // first makes "too early" indistinguishable from "on time".
    if !imp::arm() {
        return WaitOutcome::Unsupported;
    }

    // ── 2 · Register, then check ────────────────────────────────────────
    // This store must be visible before the load below, and `SeqCst` on both
    // is what forbids the store-load reordering that would let this thread
    // and a waker miss each other (module docs, mechanism 2).
    let Some(slot) = claim_wait(me, addr) else {
        return WaitOutcome::Unsupported;
    };
    let _slot = WaitSlot { thread: me, slot };
    if word.load(Ordering::SeqCst) != expected {
        return WaitOutcome::Retry;
    }

    // ── 3 · Spin on the line, briefly ───────────────────────────────────
    if let Some(end) = spin_briefly(word, expected, spin) {
        return end;
    }

    // ── 4 · Park ────────────────────────────────────────────────────────
    // `_slot` releases on the way out: a waiter that is awake must not stay
    // findable, or the next `wake` spends its kick on nobody and reports
    // having woken someone.
    if counted {
        STAT_PARKS.fetch_add(1, Ordering::Relaxed);
    }
    let outcome = imp::park(timeout);
    if counted {
        match outcome {
            WaitOutcome::Woken => STAT_WOKEN.fetch_add(1, Ordering::Relaxed),
            WaitOutcome::TimedOut => STAT_RODE.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
    }
    outcome
}

/// Cumulative process-wide wait/wake accounting — the terminal's window
/// into behavior no serial can show (`res/SPMD.md` §11's doctrine).
/// Relaxed counters: telemetry, never synchronization.  `rode` is the
/// number that adjudicates a wake mesh: a parked waiter its kick
/// reached wakes as `woken`; one the mesh missed rides its slice out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaitStats {
    /// Kernel parks entered (past the spin and monitored tiers).
    pub parks: u32,
    /// Parks ended by a delivered kick.
    pub woken: u32,
    /// Parks that rode their full slice to the deadline.
    pub rode: u32,
    /// Peer kicks posted (the caller held the peer's send).
    pub kicks: u32,
    /// Registered waiters the caller held no send for.
    pub missed: u32,
    /// Relay notices posted toward the hub.
    pub relays: u32,
}

static STAT_PARKS: AtomicU32 = AtomicU32::new(0);
static STAT_WOKEN: AtomicU32 = AtomicU32::new(0);
static STAT_RODE: AtomicU32 = AtomicU32::new(0);
static STAT_KICKS: AtomicU32 = AtomicU32::new(0);
static STAT_MISSED: AtomicU32 = AtomicU32::new(0);
static STAT_RELAYS: AtomicU32 = AtomicU32::new(0);

/// Snapshot the counters — a caller differences two snapshots around
/// the window it measures.
pub fn stats() -> WaitStats {
    WaitStats {
        parks: STAT_PARKS.load(Ordering::Relaxed),
        woken: STAT_WOKEN.load(Ordering::Relaxed),
        rode: STAT_RODE.load(Ordering::Relaxed),
        kicks: STAT_KICKS.load(Ordering::Relaxed),
        missed: STAT_MISSED.load(Ordering::Relaxed),
        relays: STAT_RELAYS.load(Ordering::Relaxed),
    }
}

/// [`wait`] for a future: **yield** to the executor instead of blocking the
/// thread in the kernel.
///
/// Same registry, same wake path, same orderings — a waker cannot tell which
/// form its target used, and does not need to. What differs is only *how the
/// waiter stops running*, and that difference is the whole reason this exists:
/// the blocking form parks the thread inside the kernel, which on an executor
/// thread stalls every sibling future and consumes a doorbell notice that may
/// have belonged to one of them.
///
/// Here the executor keeps owning its doorbell. This never arms a post and
/// never consumes a notice; it registers, then waits for the executor's next
/// wake and re-checks. A futex kick lands in the executor's standing post like
/// any other notice, and its **total drain** re-polls this future along with
/// everything else — so the wake costs nothing extra and steals nothing. That
/// is also why no `settle_inline_doorbell` hand-back is needed: nothing was
/// taken to hand back.
///
/// `timeout` is relative; `None` waits until woken. Spurious wakeups are, if
/// anything, more common than in the blocking form — every executor wake from
/// any source re-polls this — so the caller's re-check loop is not optional.
///
/// # Panics
/// Polled outside the kore executor, as [`next_wake`](crate::executor::next_wake):
/// the wake it waits on is the executor's own.
pub async fn wait_async(word: &AtomicU32, expected: u32, timeout: Option<Duration>) -> WaitOutcome {
    let Some(me) = current_thread() else {
        return WaitOutcome::Unsupported;
    };
    let addr = address_of(word);

    // No arm step, unlike `wait`: the executor's standing post is already up
    // and stays up — it is re-posted after every delivery it consumes. Arming
    // again here would refuse `WouldBlock` against the executor's own live
    // post, and consuming would take an edge the executor is accounting for.
    //
    // The registration must still precede the first load, and both stay
    // `SeqCst`: the Dekker argument against a waker's store-then-scan is
    // identical whether this thread parks in the kernel or parks on the
    // executor.
    let Some(slot) = claim_wait(me, addr) else {
        return WaitOutcome::Unsupported;
    };
    // Held across every await below, so a cancelled future releases its slot
    // on drop rather than leaving it claimed forever.
    let _slot = WaitSlot { thread: me, slot };

    let deadline = match timeout {
        Some(d) => Some(
            crate::executor::clock_ns()
                .await
                .saturating_add(nanos_of(d)),
        ),
        None => None,
    };

    // The same bet as the blocking form, and taken before the first yield for
    // the same reason: a handoff that resolves within a turn or two costs
    // neither a re-poll nor a park. Bounded by `SPIN_TURNS * SPIN_CAP`, which
    // is the whole budget this may stall its sibling futures for — a few
    // microseconds, against a park that costs more than that on its own.
    // The async lane keeps the lock budget: an executor's sibling futures
    // must not be stalled for a barrier-length jitter window.
    if let Some(end) = spin_briefly(word, expected, LOCK_SPIN) {
        return end;
    }

    loop {
        if word.load(Ordering::SeqCst) != expected {
            return WaitOutcome::Retry;
        }
        match deadline {
            None => {
                crate::executor::next_wake().await;
            }
            Some(at) => {
                let now = crate::executor::clock_ns().await;
                if now >= at {
                    return WaitOutcome::TimedOut;
                }
                // Either a wake or the remaining budget, whichever comes
                // first; both simply re-run the check above. The deadline is
                // recomputed each pass rather than held as a delta, so a
                // spurious wake cannot extend the total wait.
                crate::executor::next_wake_for(at - now).await;
            }
        }
    }
}

/// `Duration` to whole nanoseconds, saturating.
///
/// `as_nanos` is `u128`; a futex timeout that overflowed `u64` would be ~585
/// years, and saturating there is the honest answer rather than a wrap into a
/// deadline in the past.
fn nanos_of(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// How many monitored turns a waiter spends before it stops running.
///
/// A pre-park monitored-spin budget: how many turns, and the cap on one.
///
/// The waiter's counterparty decides it. A lock's is running and hands
/// off in hundreds of nanoseconds ([`LOCK_SPIN`]); a barrier's is a
/// co-scheduled sibling that arrives within a scheduling-jitter window
/// ([`BARRIER_SPIN`]). Both are cheap because [`hint::wait_on`] sleeps
/// the core on the line — the cap bounds only the tail, when the store
/// never comes.
#[derive(Clone, Copy)]
struct SpinBudget {
    turns: u32,
    cap: Duration,
}

/// The lock-handoff budget — small, because this is a *bet*: each turn
/// costs a bounded sleep on the line, and the payoff is skipping a park
/// entirely. A waiter that has spent four microseconds is waiting on
/// something slower than a handoff and should stop occupying its core,
/// which is what the park does and the monitor deliberately does not.
/// The cap is a microsecond: comfortably past a handoff's hundreds of
/// nanoseconds, invisible to a human, and a bound on a monitor that
/// never fires (four turns = four microseconds, then the park).
const LOCK_SPIN: SpinBudget = SpinBudget {
    turns: 4,
    cap: Duration::from_micros(1),
};

/// The barrier budget — a **scheduling-jitter window** before the park.
///
/// A co-scheduled sibling arrives within microseconds to tens of them,
/// and the monitor makes the wait free until it does (its store ends
/// the `UMWAIT` the instant it arrives). Eight turns of eight
/// microseconds — 64 µs of ceiling — covers that jitter so the
/// co-resident case resolves here instead of paying a park + a kick
/// that can miss and ride [`PARK_SLICE`](crate::barrier). Fewer,
/// longer turns than the lock budget: same ceiling economy, less
/// per-turn monitor re-arm overhead. A sibling later than 64 µs is
/// genuinely descheduled and the park (with the kick mesh behind it)
/// is the right tool.
const BARRIER_SPIN: SpinBudget = SpinBudget {
    turns: 8,
    cap: Duration::from_micros(8),
};

/// Sleep on the line for a few turns, hoping to skip the park entirely.
///
/// `Some` when the word moved — the caller is done and never blocks. `None`
/// when the turns are spent and the caller should go on to its real wait.
///
/// **This is the futex's only spin, and it is why one is affordable.** A bare
/// spin trades a core for latency, which is why this runtime had none. A
/// monitored turn does not: the core is asleep on the watched cache line and
/// the waker's own store is what ends it, so the common case costs neither a
/// syscall nor a busy core. On hardware without a monitor
/// ([`hint::wait_on`](crate::hint::wait_on) reports `Spun`) the turns degrade
/// to ordinary spin hints bounded by the same calibrated clock. Without a
/// readable clock only the finite probe count is available.
///
/// It also **does not need the waker to reach us**. A doorbell wake requires a
/// send end the waker holds; a store to the futex word requires nothing. So on
/// a machine with a monitor, short contention resolves between threads the
/// peer mesh cannot yet connect — which is most of them.
fn spin_briefly(word: &AtomicU32, expected: u32, budget: SpinBudget) -> Option<WaitOutcome> {
    spin_briefly_with(
        word,
        expected,
        budget,
        || crate::clock::uptime().ok().map(|time| time.as_nanos()),
        |cap| {
            crate::hint::wait_on(word, expected, cap);
        },
    )
}

/// Spend a wall-clock jitter window even when the ISA has no bounded monitor.
/// Spurious monitor returns do not consume the entire budget in a few hints.
/// Without a published clock use the existing finite probe bound; if a live
/// clock disappears, stop rather than silently turning the deadline into a spin.
fn spin_briefly_with(
    word: &AtomicU32,
    expected: u32,
    budget: SpinBudget,
    mut now: impl FnMut() -> Option<u64>,
    mut pause: impl FnMut(Duration),
) -> Option<WaitOutcome> {
    let cap_ns = nanos_of(budget.cap);
    let deadline =
        now().map(|start| start.saturating_add(cap_ns.saturating_mul(u64::from(budget.turns))));
    let mut probes = budget.turns;
    loop {
        if word.load(Ordering::SeqCst) != expected {
            return Some(WaitOutcome::Retry);
        }
        let cap = if let Some(deadline) = deadline {
            let current = now()?;
            if current >= deadline {
                return None;
            }
            Duration::from_nanos(cap_ns.min(deadline - current))
        } else {
            if probes == 0 {
                return None;
            }
            probes -= 1;
            budget.cap
        };
        pause(cap);
    }
}

/// A completion handoff uses the short lock budget before paying for a post.
pub(crate) fn spin_completion(word: &AtomicU32, expected: u32) {
    let _ = spin_briefly(word, expected, LOCK_SPIN);
}

/// Block on a caller-owned standing post without consuming its landing or
/// the executor's doorbell. The owner rechecks its shared word before this
/// call and drains/reposts its NoticeRx afterwards. Never called by a future.
pub(crate) fn wait_standing(post: crate::transport::ParkPost, timeout: Duration) -> WaitOutcome {
    STAT_PARKS.fetch_add(1, Ordering::Relaxed);
    let outcome = imp::park_post(post, Some(timeout));
    match outcome {
        WaitOutcome::Woken => {
            STAT_WOKEN.fetch_add(1, Ordering::Relaxed);
        }
        WaitOutcome::TimedOut => {
            STAT_RODE.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
    outcome
}

/// The star's relay endpoint is also the hubless resident's reverse
/// completion channel. The publisher must store the completion word first;
/// a refused/coalesced notice is only a hint and never completion evidence.
pub(crate) fn notify_resident_completion() -> bool {
    imp::relay()
}

/// Wake **one** thread waiting on `word`; `true` if a kick was delivered.
///
/// Not quite "a waiter existed". A registered waiter whose kick refuses —
/// most often `WouldBlock`, because its doorbell post was already consumed by
/// some other notice — reports `false` here even though it was found. That is
/// the honest reading and also the harmless one: a target whose post is spent
/// is already awake and will re-check the word, so the kick was unnecessary
/// rather than lost. `false` therefore means "no waiter was reached", which is
/// weaker than "no waiter existed" and is what a caller can actually rely on.
///
/// `std`'s `futex_wake` documents the same latitude — its contract permits a
/// platform that always answers `false` — so this is within the shape the
/// eventual shim needs, not a deviation from it.
///
/// The caller must have already published whatever the waiter will re-check —
/// store the word, *then* call this. The reverse order is the lost-wakeup bug
/// this module's ordering rules exist to forbid, and no amount of care inside
/// here can recover it.
///
/// "One" is the first registered waiter found, not the longest-waiting: this
/// is a wake primitive, not a queue, and fairness is the caller's to build on
/// top (as `std`'s `Mutex` does). A waiter that is registered but not yet
/// parked is still woken correctly — its post already stands, so the notice
/// resolves the park it is about to enter.
pub fn wake(word: &AtomicU32) -> bool {
    let addr = address_of(word);
    let mut unreachable_waiter = false;
    for thread in 0..MAX_THREADS as usize {
        if thread_waits_on(thread, addr) {
            if imp::kick(thread) {
                STAT_KICKS.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            unreachable_waiter = true;
        }
    }
    // A registered waiter this caller holds no peer send for is the
    // hub's to reach (`res/SPMD.md` §3 — the star): one relay notice,
    // counted as a wake because a notice is now in flight toward the
    // waiter.  No hub wired → today's honest miss.
    if unreachable_waiter {
        STAT_MISSED.fetch_add(1, Ordering::Relaxed);
        if imp::relay() {
            STAT_RELAYS.fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    false
}

/// Wake **every** thread waiting on `word`; returns how many were kicked.
///
/// The count is of notices successfully posted, not of threads that will
/// observe the condition — a woken thread may find the word changed again
/// before it looks. Callers use it for telemetry and tests, never as a
/// synchronization fact.
pub fn wake_all(word: &AtomicU32) -> usize {
    let addr = address_of(word);
    let mut woken = 0;
    let mut missed = false;
    for thread in 0..MAX_THREADS as usize {
        if thread_waits_on(thread, addr) {
            if imp::kick(thread) {
                STAT_KICKS.fetch_add(1, Ordering::Relaxed);
                woken += 1;
            } else {
                STAT_MISSED.fetch_add(1, Ordering::Relaxed);
                missed = true;
            }
        }
    }
    // The relay hop (the star, `res/SPMD.md` §3): every waiter this
    // caller could not kick is the hub's job — one notice covers them
    // all, counted once.  With the intra-chunk kick mesh this is the
    // exception path (cross-chunk waiters), not the release path.
    if missed && imp::relay() {
        STAT_RELAYS.fetch_add(1, Ordering::Relaxed);
        woken += 1;
    }
    woken
}

/// Whether this thread can [`wait`] at all — a doorbell to park on and an
/// identity in the thread band.
///
/// Lets a caller choose a different strategy up front instead of discovering
/// [`Unsupported`](WaitOutcome::Unsupported) from inside a loop it cannot
/// leave, the same courtesy [`random::available`](crate::random::available)
/// extends.
pub fn available() -> bool {
    current_thread().is_some() && imp::can_park()
}

/// The futex word's address — the registry key.
///
/// Threads share an address space, so this is the same integer in every
/// thread, which is the property that makes the shared registry work at all.
fn address_of(word: &AtomicU32) -> usize {
    word as *const AtomicU32 as usize
}

/// This thread's band index — its identity in [`WAITING_ON`].
///
/// The allocator's shard index, deliberately: it is already stamped at
/// anchor, already proved injective across live threads (the `MAX_THREADS <=
/// NUM_SHARDS` assertion in `heap::shard` and the band's `slot_of` inverse),
/// and a second independent numbering would be a second thing to get wrong.
///
/// The bound is re-checked rather than assumed: an out-of-band index would
/// index the registry out of range, and a thread that cannot name itself must
/// refuse to wait rather than borrow a neighbour's entry.
pub(crate) fn current_thread() -> Option<usize> {
    let index = imp::thread_index()?;
    (index < MAX_THREADS as usize).then_some(index)
}

/// The metal half: arm, park, kick.
///
/// Split out so the registry and its ordering — the part that is subtle and
/// the part that is testable without hardware — are ordinary code above,
/// rather than being locked behind `cfg(target_os = "none")` where the host
/// suite cannot see them.
#[cfg(target_os = "none")]
mod imp {
    use core::time::Duration;

    use kore_abi::session::Status;
    use kore_abi::task::role;

    use super::{COOKIE_FUTEX, WaitOutcome};
    use crate::cap::verbs;
    use crate::executor::Doorbell;
    use crate::session::SessionBuilder;
    use crate::transport::{ParkSpec, ParkTransport, Transport, TrapTransport};

    /// The [`role::DOORBELL`] index of the send end into **thread 0's**
    /// doorbell; peer `i`'s is `PEER_BASE + i`.
    ///
    /// Two, which is not a free choice — it is the convention that already
    /// exists. Index 0 is the held `ChanRecv` and index 1 its landing slot,
    /// and index 2 is what the founder carves as a service's **self-kick**:
    /// a `ChanSend` into its own doorbell. Thread 0 is the founded thread, so
    /// "the self-kick" and "a send end into thread 0's doorbell" name the
    /// same capability, seen from the task that holds it and from a worker
    /// that derived it — which is exactly what the input service's relay
    /// thread does to announce a keystroke. Numbering peer 0 at index 2
    /// therefore makes the existing endowment the mesh's first entry rather
    /// than a special case beside it, and the arithmetic lookup needs no
    /// exception for a thread's own index.
    ///
    /// **This rests on the founded thread's identity being 0**, since that is
    /// the peer number the self-kick answers for. It holds because the
    /// founder maps that thread's runtime block at the band's first slot, and
    /// the founder is where the two are actually cross-checked
    /// (`bands.thread_block.first().same_as(&LAYOUT.thread_block)` in
    /// `services/exec/src/founder`). Nothing here can restate that usefully:
    /// a check written against this crate's own band constant would only
    /// compare the band with itself and pass unconditionally, which is worse
    /// than no check because it reads like one. If that founder assertion
    /// ever goes, every wake quietly kicks a role index nobody holds and
    /// `kick` reports it as an ordinary `false`.
    pub(super) const PEER_BASE: u16 = super::PEER_BASE;

    /// This thread's doorbell, re-resolved from its own boot page.
    ///
    /// Re-parsed per call rather than cached in the `ThreadBlock`, because
    /// `thread_context` already does exactly this for every other per-thread
    /// lookup a worker makes and the page is immutable for the thread's life.
    /// A futex wait is a syscall either way; a boot-block scan beside it is
    /// not the cost worth adding a second source of truth to avoid.
    fn doorbell() -> Option<Doorbell> {
        crate::entry::thread_context()?.doorbell()
    }

    /// Publish this thread's standing doorbell post.
    ///
    /// Idempotent by the kernel's own answer: `Ok` means the post now stands
    /// and `WouldBlock` means one already did, and the caller wants the same
    /// thing in both cases. That is the identical rule the executor's
    /// `repost` follows, and for the identical reason.
    fn post(db: &Doorbell) -> bool {
        let mut s = SessionBuilder::new();
        if s.post_recv(db.recv, db.land).is_err() {
            return false;
        }
        TrapTransport::new()
            .commit_now(&s)
            .is_ok_and(|c| matches!(c.status(), Status::Ok | Status::WouldBlock))
    }

    pub(super) fn arm() -> bool {
        doorbell().is_some_and(|db| post(&db))
    }

    /// Park on the standing post, with `timeout` folded in as a deadline.
    ///
    /// The deadline the kernel wants is an **absolute** monotonic instant,
    /// not the caller's relative duration, so it is computed here against the
    /// zero-trap clock. A machine whose clock page is uncalibrated cannot
    /// bound a park at all; that refuses rather than parking forever on a
    /// caller that asked not to.
    pub(super) fn park(timeout: Option<Duration>) -> WaitOutcome {
        let Some(db) = doorbell() else {
            return WaitOutcome::Unsupported;
        };
        let standing = crate::transport::ParkPost {
            chan: db.chan,
            land: db.land,
            recv_tag: db.recv_tag,
        };
        let Some(status) = park_status(standing, timeout) else {
            return WaitOutcome::Unsupported;
        };
        match status {
            // A notice landed. Consume it and re-stand the post *before*
            // returning: `take_notice` deliberately does not clear the
            // landing slot — the next `post_recv` scrubs it kernel-side — so
            // skipping the re-post would leave both a dirty slot and no post
            // for the next kick to land in.
            Status::Ok => {
                let _ = verbs::take_notice(db.land);
                post(&db);
                WaitOutcome::Woken
            }
            // The deadline won; the post is left standing, so nothing to
            // re-arm.
            Status::Deadline => WaitOutcome::TimedOut,
            // A stray scheduler resumption with the post still standing.
            // Reported as a wake rather than retried here: the caller is
            // already required to re-test its condition in a loop, and
            // spinning inside the park would hide a livelock instead of
            // letting the caller's own loop bound it.
            Status::WouldBlock => WaitOutcome::Woken,
            // The recv end was reaped — this thread's wake edge is gone and
            // no future wait can succeed.
            _ => WaitOutcome::Unsupported,
        }
    }

    /// Shared parking transport for the fixed thread doorbell and an affine
    /// alternate receiver. Only the caller owning the landing may consume it.
    fn park_status(post: crate::transport::ParkPost, timeout: Option<Duration>) -> Option<Status> {
        let deadline = match timeout {
            Some(d) => Some((crate::clock::uptime().ok()? + d).as_nanos()),
            None => None,
        };
        let mut p = SessionBuilder::new();
        p.park_standing(post.chan, post.land.index() as u16, post.recv_tag);
        if let Some(at) = deadline {
            p.wait_sleep(at);
        }
        let spec = ParkSpec {
            chan: post.chan,
            land: post.land,
            alternate: None,
            deadline_ns: deadline,
            irq: false,
        };
        Some(
            TrapTransport::new()
                .commit_and_park(&p, &spec)
                .ok()?
                .status(),
        )
    }

    pub(super) fn park_post(
        post: crate::transport::ParkPost,
        timeout: Option<Duration>,
    ) -> WaitOutcome {
        match park_status(post, timeout) {
            Some(Status::Ok | Status::WouldBlock) => WaitOutcome::Woken,
            Some(Status::Deadline) => WaitOutcome::TimedOut,
            _ => WaitOutcome::Unsupported,
        }
    }

    /// Post a notice into thread `index`'s doorbell.
    ///
    /// `false` when this thread holds no send end for that peer (a mesh that
    /// was never provisioned) or the notice refused — including the
    /// `WouldBlock` that means the target's post was already consumed by
    /// another kick, which needs no retry: the target is awake and will
    /// re-check.
    pub(super) fn kick(index: usize) -> bool {
        let Some(rt) = crate::entry::thread_context() else {
            return false;
        };
        let Ok(peer) = u16::try_from(index) else {
            return false;
        };
        let Some(send) = rt.bootstrap().handle(role::DOORBELL, PEER_BASE + peer) else {
            return false;
        };
        verbs::post_notice(send, COOKIE_FUTEX, index as u64)
    }

    // The relay slot sits exactly one past the last band peer — the
    // module-level constant and this mesh's arithmetic must agree.
    const _: () = assert!(super::RELAY == PEER_BASE + super::MAX_THREADS as u16);

    /// Post one notice on the caller's hub send (`role::DOORBELL`,
    /// [`RELAY`](super::RELAY)) — the star's second hop.  `false` when no
    /// hub is wired (no such boot entry) or the notice refused —
    /// including the `WouldBlock` that means the hub's post is already
    /// consumed, which needs no retry: the hub is awake and kicks its
    /// peers on every re-check.
    pub(super) fn relay() -> bool {
        let Some(rt) = crate::entry::thread_context() else {
            return false;
        };
        let Some(send) = rt.bootstrap().handle(role::DOORBELL, super::RELAY) else {
            return false;
        };
        verbs::post_notice(send, COOKIE_FUTEX, u64::from(super::RELAY))
    }

    pub(super) fn can_park() -> bool {
        doorbell().is_some()
    }

    pub(super) fn thread_index() -> Option<usize> {
        crate::entry::geometry::shard_index()
    }
}

/// Off metal there is no doorbell and no sibling to wake: the honest typed
/// refusal (`res/USERLAND.md` §2.1), never a simulated park.
///
/// A host build that silently used `std::thread::park` here would make the
/// host suite pass for a reason the target never reproduces, which is the
/// failure mode this whole file's structure is arranged to avoid.
#[cfg(not(target_os = "none"))]
mod imp {
    pub(super) fn arm() -> bool {
        false
    }

    pub(super) fn park(_timeout: Option<core::time::Duration>) -> super::WaitOutcome {
        super::WaitOutcome::Unsupported
    }

    pub(super) fn park_post(
        _post: crate::transport::ParkPost,
        _timeout: Option<core::time::Duration>,
    ) -> super::WaitOutcome {
        super::WaitOutcome::Unsupported
    }

    pub(super) fn kick(_index: usize) -> bool {
        false
    }

    pub(super) fn relay() -> bool {
        false
    }

    pub(super) fn can_park() -> bool {
        false
    }

    /// No thread band off metal, so no identity — and therefore no wait.
    pub(super) fn thread_index() -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn jitter_window_is_time_bounded_without_a_monitor() {
        use core::cell::Cell;
        let time = Cell::new(0u64);
        let pauses = Cell::new(0u32);
        let word = AtomicU32::new(1);
        let out = spin_briefly_with(
            &word,
            1,
            BARRIER_SPIN,
            || Some(time.get()),
            |_| {
                time.set(time.get() + 1_000);
                pauses.set(pauses.get() + 1);
                if pauses.get() == 16 {
                    word.store(2, Ordering::SeqCst);
                }
            },
        );
        assert_eq!(out, Some(WaitOutcome::Retry));
        assert_eq!(pauses.get(), 16); // beyond the former eight-hint fallback
        let word = AtomicU32::new(1);
        time.set(0);
        let out = spin_briefly_with(
            &word,
            1,
            LOCK_SPIN,
            || Some(time.get()),
            |cap| {
                assert!(cap <= LOCK_SPIN.cap);
                time.set(time.get() + 1_000);
            },
        );
        assert_eq!(out, None);
        assert_eq!(time.get(), 4_000);
        let reads = Cell::new(0);
        assert_eq!(
            spin_briefly_with(
                &word,
                1,
                LOCK_SPIN,
                || {
                    reads.set(reads.get() + 1);
                    (reads.get() == 1).then_some(0)
                },
                |_| panic!("lost clock must end the wait")
            ),
            None
        );
    }

    use super::*;

    /// Clear **one thread's** row.
    ///
    /// Per-row rather than whole-registry, and each test below owns a distinct
    /// row, because the host suite runs tests in parallel over this one shared
    /// static — exactly as the threads it is built for share it. A global
    /// reset let one test wipe a registration another was mid-way through
    /// asserting on, and a test that fills every slot of a row starved when a
    /// neighbour claimed one first. Disjoint rows remove the race rather than
    /// serialising around it.
    fn reset_thread(thread: usize) {
        for cell in &WAITING_ON[thread] {
            cell.store(NOT_WAITING, Ordering::SeqCst);
        }
    }

    #[test]
    fn the_registry_covers_every_thread() {
        // The no-allocation claim: identity indexes the registry directly, so
        // every thread the band can name has its own row and no scan can run
        // off the end of one.
        assert_eq!(WAITING_ON.len(), MAX_THREADS as usize);
    }

    #[test]
    fn the_vacant_sentinel_is_not_a_reachable_address() {
        // `NOT_WAITING` must be a value no `&AtomicU32` can hash to, or a
        // real waiter would read as absent. The null page is never mapped.
        let word = AtomicU32::new(0);
        assert_ne!(address_of(&word), NOT_WAITING);
    }

    #[test]
    fn distinct_words_have_distinct_registry_keys() {
        // Two futexes must not alias, or a wake on one would spend itself on
        // the other's waiters.
        let a = AtomicU32::new(0);
        let b = AtomicU32::new(0);
        assert_ne!(address_of(&a), address_of(&b));
    }

    #[test]
    fn a_wake_finds_only_waiters_on_its_own_address() {
        // Rows 0 and 1 — this one needs two.
        reset_thread(0);
        reset_thread(1);
        let target = AtomicU32::new(0);
        let other = AtomicU32::new(0);
        claim_wait(0, address_of(&target)).expect("a free wait slot");
        claim_wait(1, address_of(&other)).expect("a free wait slot");

        // Off metal `kick` refuses, so `wake` cannot report a delivery — but
        // the *selection* is the shared half, identical on target, and it is
        // what must never spill across addresses.
        let hits = (0..MAX_THREADS as usize)
            .filter(|&t| thread_waits_on(t, address_of(&target)))
            .count();
        assert_eq!(hits, 1, "the scan matched a waiter on another word");
        assert!(
            thread_waits_on(0, address_of(&target)),
            "the matching waiter is the one registered against this word"
        );
        reset_thread(0);
        reset_thread(1);
    }

    #[test]
    fn only_a_timeout_reports_not_waited() {
        // The `std` mapping: `false` means "the deadline passed" and nothing
        // else. A refusal reported as `false` would become a fabricated
        // timeout in `Condvar::wait_timeout`.
        assert!(WaitOutcome::Woken.waited());
        assert!(WaitOutcome::Retry.waited());
        assert!(WaitOutcome::Unsupported.waited());
        assert!(!WaitOutcome::TimedOut.waited());
    }

    #[test]
    fn off_metal_wait_refuses_rather_than_pretending() {
        reset_thread(7);
        let word = AtomicU32::new(1);
        // No thread identity and no doorbell: the answer must be the typed
        // refusal, not a spin that looks like a wait.
        assert!(!available());
        assert_eq!(wait(&word, 1, None), WaitOutcome::Unsupported);
    }

    #[test]
    fn off_metal_async_wait_refuses_without_touching_the_executor() {
        reset_thread(7);
        let word = AtomicU32::new(1);
        // The identity check must come FIRST, before any `next_wake` — those
        // panic outside a kore executor, and a host test has none. Completing
        // on the first poll is the proof: it never reached an await point.
        let out = crate::block::try_block_on(wait_async(&word, 1, None), 1);
        assert_eq!(out, Some(WaitOutcome::Unsupported));
    }

    #[test]
    fn an_async_wait_leaves_no_registration_behind() {
        reset_thread(7);
        let word = AtomicU32::new(1);
        let _ = crate::block::try_block_on(wait_async(&word, 1, None), 1);
        // A thread that is not waiting must not be findable as one, or the
        // next `wake` spends its kick on nobody and reports success.
        assert_eq!(wake_all(&word), 0, "a refused wait left a live waiter");
    }

    #[test]
    fn two_waiters_on_one_thread_both_stay_findable() {
        reset_thread(2);
        // The regression this registry shape exists for. One executor thread
        // can have many futures suspended inside `wait_async` at once, each
        // on its own address. With a single entry per thread the second
        // registration silently erased the first, and the first's `wake`
        // then found nothing to kick and returned an ordinary `false` — a
        // lost wakeup with no diagnostic anywhere.
        let first = AtomicU32::new(0);
        let second = AtomicU32::new(0);
        let a = claim_wait(2, address_of(&first)).expect("a free wait slot");
        let b = claim_wait(2, address_of(&second)).expect("a second free wait slot");
        assert_ne!(a, b, "the second waiter reused the first's slot");

        assert!(thread_waits_on(2, address_of(&first)), "first waiter lost");
        assert!(
            thread_waits_on(2, address_of(&second)),
            "second waiter lost"
        );

        // Releasing one must not retire the other.
        release_wait(2, b);
        assert!(
            thread_waits_on(2, address_of(&first)),
            "releasing the second waiter retired the first"
        );
        assert!(!thread_waits_on(2, address_of(&second)));
        reset_thread(2);
    }

    #[test]
    fn releasing_by_index_does_not_retire_a_twin_on_the_same_address() {
        reset_thread(3);
        // Two futures may legitimately wait on the SAME word. Releasing by
        // matching address instead of by index would retire whichever entry
        // was found first — possibly the sibling's, leaving a waiter that is
        // awake registered and one that is asleep forgotten.
        let word = AtomicU32::new(0);
        let a = claim_wait(3, address_of(&word)).expect("a free wait slot");
        let _b = claim_wait(3, address_of(&word)).expect("a second free wait slot");
        release_wait(3, a);
        assert!(
            thread_waits_on(3, address_of(&word)),
            "releasing one waiter retired its twin on the same address"
        );
        reset_thread(3);
    }

    #[test]
    fn a_thread_can_host_every_future_an_executor_admits() {
        reset_thread(4);
        // The sizing claim: root + MAX_TASKS futures, each in its own wait.
        let word = AtomicU32::new(0);
        for i in 0..WAITS_PER_THREAD {
            assert!(
                claim_wait(4, address_of(&word)).is_some(),
                "ran out of wait slots at {i}, below the executor's capacity"
            );
        }
        assert_eq!(
            WAITS_PER_THREAD,
            crate::executor::MAX_TASKS + 1,
            "the registry no longer tracks the executor's task capacity"
        );
        reset_thread(4);
    }

    #[test]
    fn a_dropped_slot_guard_releases_its_registration() {
        reset_thread(5);
        // The cancellation leak. `wait_async` is a future, and a future can be
        // dropped while pending — a select, a timeout, a cancelled task. A
        // release placed after the loop never runs on that path, so the slot
        // stayed claimed forever holding an address nobody waits on: enough
        // repeats exhaust the thread and every later wait refuses, and before
        // that `wake` matches the stale entry and reports a waiter that is not
        // there.
        let word = AtomicU32::new(0);
        {
            let slot = claim_wait(5, address_of(&word)).expect("a free wait slot");
            let _guard = WaitSlot { thread: 5, slot };
            assert!(thread_waits_on(5, address_of(&word)));
        }
        assert!(
            !thread_waits_on(5, address_of(&word)),
            "a dropped waiter stayed registered"
        );
        reset_thread(5);
    }

    #[test]
    fn cancelling_repeatedly_does_not_exhaust_the_thread() {
        reset_thread(6);
        // The consequence the guard exists to prevent, driven past the point
        // where a leak would have shown: more cancellations than the thread
        // has slots.
        let word = AtomicU32::new(0);
        for _ in 0..(WAITS_PER_THREAD * 3) {
            let slot = claim_wait(6, address_of(&word)).expect("slots must be reusable");
            drop(WaitSlot { thread: 6, slot });
        }
        assert!(!thread_waits_on(6, address_of(&word)));
        reset_thread(6);
    }

    #[test]
    fn a_saturating_timeout_never_wraps_into_the_past() {
        // `Duration::as_nanos` is u128; a deadline computed from a wrapped
        // conversion would be in the past and time out instantly.
        assert_eq!(nanos_of(Duration::from_nanos(1_000)), 1_000);
        assert_eq!(nanos_of(Duration::MAX), u64::MAX);
    }

    #[test]
    fn a_brief_spin_reports_a_word_that_already_moved() {
        let word = AtomicU32::new(9);
        assert_eq!(spin_briefly(&word, 7, LOCK_SPIN), Some(WaitOutcome::Retry));
    }

    #[test]
    fn a_brief_spin_gives_up_and_lets_the_caller_block() {
        // The bet has to be bounded, or the futex has quietly become a
        // spinlock. Off metal every turn degrades to a plain spin hint, so
        // this also proves the fallback terminates rather than parking on a
        // monitor that is not there.
        let word = AtomicU32::new(7);
        assert_eq!(spin_briefly(&word, 7, LOCK_SPIN), None);
        // The barrier's longer budget still terminates rather than parking
        // on a monitor that never fires (the off-metal fallback path).
        assert_eq!(spin_briefly(&word, 7, BARRIER_SPIN), None);
    }

    #[test]
    fn the_spin_budget_stays_a_bet_rather_than_a_wait() {
        // If either budget's ceiling grows, the "occupies a core" cost stops
        // being negligible and the park it is trying to avoid becomes the
        // cheaper option. The lock budget is a handoff (µs); the barrier
        // budget is a scheduling-jitter window (tens of µs) — both far below
        // the park slice they replace, so both stay optimizations.
        assert!(LOCK_SPIN.turns <= 8, "lock spin grew into a real wait");
        assert!(LOCK_SPIN.cap <= Duration::from_micros(10));
        let barrier_ceiling = BARRIER_SPIN.cap.as_micros() * BARRIER_SPIN.turns as u128;
        assert!(
            barrier_ceiling <= 128,
            "barrier spin ceiling grew past a jitter window"
        );
    }

    #[test]
    fn waking_an_unwaited_word_reports_nobody() {
        reset_thread(7);
        let word = AtomicU32::new(0);
        assert!(!wake(&word));
        assert_eq!(wake_all(&word), 0);
    }
}
