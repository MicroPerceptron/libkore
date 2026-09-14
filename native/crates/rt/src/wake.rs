//! The **wake bell** — tell a parked executor there is work, without a
//! syscall, and find out whether it even needed telling.
//!
//! A ring handoff in this runtime costs three session commits today: the
//! producer's notice ([`post_notice`](crate::cap::verbs::post_notice)), and
//! the consuming executor's park plus its re-post. Two of those three are
//! paid on the assumption that the consumer is asleep — and on a busy
//! executor it usually is not. `res/USERLAND.md` §3.1 already rules the kick
//! **droppable** when the consumer is awake ("a consumed post means this
//! executor is awake and about to drain"); what has been missing is any way
//! for the producer to learn that without asking the kernel, which is the
//! very syscall the ruling would let it skip.
//!
//! This module is that answer, and it is one word per thread.
//!
//! # One word, one RMW, no ordering argument to get wrong
//!
//! Each thread owns a bell: an `AtomicU32` holding a **generation** and one
//! **`AWAKE`** bit.
//!
//! ```text
//!   bit 0     AWAKE       set while the owning executor is running,
//!                         draining, or monitoring — clear only while it is
//!                         parked in the kernel
//!   bits 1..  generation  bumped by every ring
//! ```
//!
//! A peer with news performs exactly one operation:
//!
//! ```ignore
//! if wake::ring(target) {
//!     verbs::post_notice(kick, cookie, 0);   // only when it was parked
//! }
//! ```
//!
//! [`ring`] is a single `fetch_add`, and that is the whole trick. The bump
//! **is** the wake edge (a store to the watched line ends the target's
//! monitor — [`hint`](crate::hint): "the store *is* the event"), and the
//! value it returns **is** the target's disposition at the instant of the
//! bump. One indivisible operation both delivers the news and reports whether
//! anything further is owed.
//!
//! That collapses the usual futex lost-wakeup problem instead of solving it.
//! [`futex`](crate::futex) needs a Dekker argument across four `SeqCst`
//! accesses precisely because its waiter's registration and its waker's scan
//! touch *different* locations and may be reordered. Here both parties touch
//! **one** location, and the modification order of a single atomic is total
//! by construction, so the two interleavings are the only two there are:
//!
//! - the peer's `fetch_add` lands **before** the executor's arm — then the
//!   executor's own RMW returns a generation that has moved, and it does not
//!   park;
//! - the peer's `fetch_add` lands **after** — then it observes `AWAKE` clear
//!   and posts the notice, which lands in a post that already stands, and
//!   "early is the same as on time".
//!
//! There is no third case and no fence to misplace.
//!
//! # Why the zero value is `parked`
//!
//! `AWAKE` is a *set* bit rather than a `PARKED` one so that an all-zero
//! bell — a thread that never ran an executor, a bell never initialised, a
//! future cross-task window before anyone formats it — reads as "parked: post
//! the notice". Every failure to know degrades to exactly today's behaviour,
//! which is the only degradation that is safe: a spurious notice costs a
//! syscall, and a suppressed one strands a message forever.
//!
//! # Threads only — for now
//!
//! The bells are a `static`, so they are one object at one address for every
//! thread of a task (`SpawnFlags::SHARE_SPACE`), exactly as
//! [`futex`](crate::futex)'s registry is. That covers every intra-task
//! producer — a relay thread feeding its service's executor, one worker
//! handing off to another — and covers it with no mm work, no lease, and no
//! endowment.
//!
//! It does **not** yet cover a cross-task peer, whose bell would have to live
//! in a shared window. The layout here is deliberately the layout that form
//! wants, but the sharing has an authority question this one does not: a
//! window a peer may write is a window a peer may *lie* in, and a false
//! `AWAKE` would strand a third party's message. The answer is to split the
//! two facts by who may state them — the generation is any peer's to bump
//! (shared `R|W`), the `AWAKE` bit is the executor's alone to publish (shared
//! `R`) — which the existing lease verbs already express
//! (`SHARE_ACCESS_R`). That is the next slice, not this one.

use core::sync::atomic::{AtomicU32, Ordering};

use crate::barrier::Line;
use crate::thread::MAX_THREADS;

/// The bell's low bit: **the owning executor is not parked**.
///
/// Clear means parked, and so does an untouched bell — see the module docs on
/// why the safe value is the zero one.
const AWAKE: u32 = 1;

/// What one ring adds. The generation occupies every bit above [`AWAKE`], so
/// a bump can never disturb the flag and a flag transition can never be
/// mistaken for a ring.
const GEN_STEP: u32 = 2;

/// One bell per thread, each on its own destructive-interference span.
///
/// The padding is not incidental: the entire point is that a peer's store
/// reaches a *monitored* line, and an unpadded array would put several
/// threads' bells on one line, so ringing thread 3 would wake thread 4's core
/// to learn nothing. [`Line`](crate::barrier::Line) is the span the barrier
/// already isolates against, reused rather than re-derived.
static BELLS: [Line; MAX_THREADS as usize] = [const { Line::new(0) }; MAX_THREADS as usize];

/// Bell counters — how often the fast path actually paid off.
///
/// Telemetry, never synchronization. These are the numbers that decide
/// whether this module earns its bytes on real silicon: `skipped` against
/// `rings` is the syscall win, and `resolved` against `armed` is the park
/// win. Both are invisible without counting, and a monitored tier that
/// silently never fires looks exactly like one that always does.
///
/// **Sharded per writing thread.** A single global set would land every
/// thread's `fetch_add` on one line — the cross-core bounce this whole module
/// exists to remove, reintroduced by the accounting *for* it, on the same hot
/// path (`ring` / `arm_park`). Each thread bumps only its own [`Counts`] line;
/// [`stats`] sums the shards. The four counters are co-resident on that line
/// on purpose: one thread owns them all, so co-residence costs it nothing and
/// isolates it from every peer. Padded like [`Line`] — one thread's counters
/// on their own 128-byte span, for the reason stated there.
#[repr(align(128))]
struct Counts {
    rings: AtomicU32,
    skipped: AtomicU32,
    armed: AtomicU32,
    resolved: AtomicU32,
}

impl Counts {
    const fn new() -> Self {
        Self {
            rings: AtomicU32::new(0),
            skipped: AtomicU32::new(0),
            armed: AtomicU32::new(0),
            resolved: AtomicU32::new(0),
        }
    }
}

/// One counter line per thread, plus a spillover slot (the `+ 1`) for a writer
/// that cannot name itself — `current() == None`: a host build, or a metal
/// thread before its anchor. Summed, never read individually, so the spillover
/// losing its per-thread isolation is immaterial.
static STATS: [Counts; MAX_THREADS as usize + 1] =
    [const { Counts::new() }; MAX_THREADS as usize + 1];

/// The counter shard a write is charged to: the named thread's own line, or
/// the spillover slot for an out-of-band index. Attribution is by the
/// **writer**, so `ring` charges the producer (`current`), never the target it
/// rings; `arm_park` / `note_resolved` charge the executor's own bell index,
/// which is that same thread.
#[inline]
fn counts(index: usize) -> &'static Counts {
    &STATS[if index < MAX_THREADS as usize {
        index
    } else {
        MAX_THREADS as usize
    }]
}

/// A snapshot of the bell counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Bells rung — every [`ring`] call, whatever it returned.
    pub rings: u32,
    /// Rings that needed **no** notice because the target was awake — the
    /// producer-side syscall this module exists to remove.
    pub skipped: u32,
    /// Idle windows an executor armed for parking after monitoring.
    pub armed: u32,
    /// Armed windows a ring resolved before the park committed — the
    /// consumer-side park + re-post pair removed.
    pub resolved: u32,
}

/// Read the bell counters — the sum across every thread's shard.
///
/// Best-effort like the counts themselves: the shards are read one at a time
/// under no lock, so a concurrent bump may or may not be included, and the
/// sum wraps exactly as each `u32` shard does (`fetch_add` already wraps).
pub fn stats() -> Stats {
    let mut s = Stats::default();
    for shard in &STATS {
        s.rings = s.rings.wrapping_add(shard.rings.load(Ordering::Relaxed));
        s.skipped = s
            .skipped
            .wrapping_add(shard.skipped.load(Ordering::Relaxed));
        s.armed = s.armed.wrapping_add(shard.armed.load(Ordering::Relaxed));
        s.resolved = s
            .resolved
            .wrapping_add(shard.resolved.load(Ordering::Relaxed));
    }
    s
}

/// The generation half of a bell word — the part a ring moves.
#[inline]
const fn generation(word: u32) -> u32 {
    word & !AWAKE
}

/// Whether a bell was rung between two readings of it.
///
/// Compares generations rather than whole words, so an executor's own
/// `AWAKE` transitions are not mistaken for a peer's news.
#[inline]
pub(crate) const fn rang(before: u32, after: u32) -> bool {
    generation(before) != generation(after)
}

/// Thread `index`'s bell, or `None` for an index outside the band.
///
/// Bounds-checked rather than asserted for the same reason
/// [`futex`](crate::futex)'s registry claim is: a thread that cannot name
/// itself must decline the fast path, not borrow a neighbour's word.
#[inline]
pub(crate) fn bell(index: usize) -> Option<&'static AtomicU32> {
    BELLS.get(index).map(|line| &line.0)
}

/// **Ring thread `index`'s bell** — returns whether a notice is still owed.
///
/// `true` means the target was parked at the instant of the bump and the
/// caller must still deliver its notice the ordinary way; `false` means the
/// target is running, draining, or asleep on this very word, and the store
/// just performed is the entire wake.
///
/// # The caller publishes first
///
/// Ring **after** the ring push, never before, and for the same reason
/// [`futex::wake`](crate::futex::wake) demands the word be stored first: the
/// bump is what licenses the target to go look, and a target that looks
/// before the record is visible finds an empty queue and parks again. The
/// `AcqRel` here is what makes the push visible to whoever the bump wakes.
///
/// An out-of-band `index` conservatively answers `true`: unknown means
/// notice-required, the same direction an untouched bell answers in.
pub fn ring(index: usize) -> bool {
    // Charged to the producer (this thread), not the target `index` — see
    // [`counts`]. Resolved once so the skip below reuses the same shard.
    let acct = counts(current().unwrap_or(MAX_THREADS as usize));
    acct.rings.fetch_add(1, Ordering::Relaxed);
    let Some(bell) = bell(index) else {
        return true;
    };
    // One indivisible operation: the bump that wakes, and the read that says
    // whether waking was enough. See the module docs — this is why there is
    // no ordering argument here beyond the release of the caller's publish.
    let prior = bell.fetch_add(GEN_STEP, Ordering::AcqRel);
    if prior & AWAKE != 0 {
        acct.skipped.fetch_add(1, Ordering::Relaxed);
        return false;
    }
    true
}

/// Read thread `index`'s bell without disturbing it — the executor's
/// snapshot before it begins monitoring.
#[inline]
pub(crate) fn observe(index: usize) -> u32 {
    bell(index).map_or(0, |bell| bell.load(Ordering::Acquire))
}

/// Whether thread `index`'s bell has **ever** been rung.
///
/// The self-calibration that keeps the monitored tier from costing anything
/// on an executor no peer talks to. A generation still at zero means no
/// producer has ever announced itself here, so there is nothing to monitor
/// *for*, and a budget spent looking would be pure waste — clock-gated waste
/// that still withholds the core from the scheduler for its duration.
///
/// One ring flips it for the life of the task, which is the right hysteresis
/// in both directions: a peer that rang once will ring again, and an executor
/// that has never been rung does not start paying a budget on the chance that
/// it might be. The seam therefore costs *nothing* until it can pay, and
/// needs no configuration to find that out.
#[inline]
pub(crate) fn ever_rung(index: usize) -> bool {
    generation(observe(index)) != 0
}

/// Publish **awake** for thread `index`.
///
/// Called by an executor whenever it is not parked: once on entry, and again
/// the moment a park returns. Idempotent.
#[inline]
pub(crate) fn set_awake(index: usize) {
    if let Some(bell) = bell(index) {
        bell.fetch_or(AWAKE, Ordering::Release);
    }
}

/// Publish **parked** for thread `index` without counting an arm.
///
/// Retirement rather than protocol: an executor that is going away leaves its
/// bell in the direction that costs a syscall instead of the one that loses a
/// message.
#[inline]
pub(crate) fn clear_awake(index: usize) {
    if let Some(bell) = bell(index) {
        bell.fetch_and(!AWAKE, Ordering::Release);
    }
}

/// Publish **parked** for thread `index`, and report the bell as it was.
///
/// The commit point of the whole protocol: after this returns, every peer
/// that rings will post its notice. The caller compares the returned word
/// against its pre-monitor snapshot with [`rang`] — a moved generation means
/// a peer rang during the window and the park must not be entered.
#[inline]
pub(crate) fn arm_park(index: usize) -> u32 {
    counts(index).armed.fetch_add(1, Ordering::Relaxed);
    bell(index).map_or(0, |bell| bell.fetch_and(!AWAKE, Ordering::AcqRel))
}

/// Count an armed window that a ring resolved before the park committed.
///
/// `index` is the resolving executor's own bell — the same thread that armed
/// it — so the count lands on that thread's shard, beside its `armed`.
#[inline]
pub(crate) fn note_resolved(index: usize) {
    counts(index).resolved.fetch_add(1, Ordering::Relaxed);
}

/// **A park in progress** — clears this thread's `AWAKE` for the duration and
/// restores exactly what it cleared.
///
/// The choke point that makes the bell's meaning *true*. An executor's idle
/// park is not the only way a thread stops running: it may block in
/// [`futex::wait`](crate::futex::wait), fuse a kick with a reply park inside a
/// service call, join a thread, or park a relay on its interrupt route. Every
/// one of those leaves the thread unable to look at its sources, and every one
/// of them would — without this — leave a **stale `AWAKE`** behind for a peer
/// to read and skip the very notice that would have ended the park.
///
/// So the guard goes where every park must pass: the three
/// [`ParkTransport`](crate::transport::ParkTransport) entries, which are
/// sealed precisely so that nothing else can claim park authority. One site,
/// and the property holds for parks that do not exist yet.
///
/// # It restores what it cleared, and nothing more
///
/// `was_awake` is not bookkeeping — it is the whole correctness of the guard.
/// A thread that never runs an executor (a relay thread, a bare worker) has a
/// bell that reads parked *because nothing on it will do a total drain*, and
/// blanket-setting `AWAKE` on every park return would make peers skip notices
/// for a thread that is never going to look. Restoring only a flag this guard
/// itself cleared keeps `AWAKE` meaning what it must: **an executor is running
/// here and will drain its sources.**
///
/// It also composes with the executor's own arm. [`Executor::idle`] clears the
/// bell itself, because there the clear and the generation check must be one
/// RMW to be a decision at all — so by the time this guard runs, `AWAKE` is
/// already clear, `was_awake` is false, and the executor keeps ownership of
/// its own restore. Two arms, no conflict, and neither has to know about the
/// other.
///
/// [`Executor::idle`]: crate::executor::Executor
pub(crate) struct Parked {
    index: usize,
    was_awake: bool,
}

impl Parked {
    /// Arm for a park, or `None` off the thread band (where there is no bell
    /// and nothing to keep honest).
    #[inline]
    pub(crate) fn enter() -> Option<Self> {
        Self::enter_for(current()?)
    }

    /// [`enter`](Self::enter) for an explicit bell index.
    ///
    /// Split out for the same reason [`futex`](crate::futex) splits its
    /// registry from its `imp`: off metal a thread cannot name itself, so a
    /// guard that could only be built from [`current`] would be a guard the
    /// host suite could never exercise — and the restore rule is the part
    /// that must not drift.
    #[inline]
    fn enter_for(index: usize) -> Option<Self> {
        let prior = bell(index)?.fetch_and(!AWAKE, Ordering::AcqRel);
        Some(Self {
            index,
            was_awake: prior & AWAKE != 0,
        })
    }
}

impl Drop for Parked {
    #[inline]
    fn drop(&mut self) {
        if self.was_awake {
            set_awake(self.index);
        }
    }
}

/// Ring the bell of the thread a `role::DOORBELL` **peer index** names.
///
/// The doorbell mesh numbers peers `PEER_BASE + thread`, so a kick site that
/// knows the send end it holds already knows the bell that matches it. Naming
/// both with one arithmetic is deliberate — it is the same discipline the
/// star takes with its hub sends ("the star's send naming and the futex's
/// peer lookup are one arithmetic"), and the alternative is two numberings
/// that agree until they quietly do not.
///
/// A doorbell index below [`PEER_BASE`](crate::futex) names this thread's own
/// recv or landing slot rather than a peer, and answers conservatively.
pub fn ring_doorbell_peer(doorbell_index: u16) -> bool {
    match doorbell_index.checked_sub(crate::futex::PEER_BASE) {
        Some(thread) => ring(thread as usize),
        None => true,
    }
}

/// This thread's bell index, or `None` off the thread band.
///
/// Deliberately [`futex`](crate::futex)'s numbering rather than a second one:
/// that module already argues why the allocator's shard index is the identity
/// to use, and two independent numberings would be two things to keep in
/// agreement.
#[inline]
pub(crate) fn current() -> Option<usize> {
    crate::futex::current_thread()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bells are per-thread and never alias — the property the padding and
    /// the index bound exist for.
    #[test]
    fn each_index_has_its_own_bell() {
        let a = bell(0).expect("index 0 is in band");
        let b = bell(1).expect("index 1 is in band");
        assert!(!core::ptr::eq(a, b), "distinct threads share no bell");
        assert!(
            bell(MAX_THREADS as usize).is_none(),
            "an out-of-band index has no bell"
        );
    }

    /// An untouched bell reads as parked, so an uninitialised thread costs a
    /// notice rather than losing one.
    #[test]
    fn an_untouched_bell_demands_the_notice() {
        // Index chosen high and unused by the other tests: the statics are
        // process-global and the suite is threaded.
        let idx = 20;
        assert_eq!(observe(idx) & AWAKE, 0, "zero means parked");
        assert!(ring(idx), "a parked target is still owed its notice");
    }

    /// An out-of-band target answers conservatively rather than silently
    /// dropping the caller's news.
    #[test]
    fn an_unknown_target_demands_the_notice() {
        assert!(
            ring(MAX_THREADS as usize + 7),
            "unknown target must not be reported as awake"
        );
    }

    /// The whole producer-side win: an awake target needs no syscall.
    #[test]
    fn an_awake_target_needs_no_notice() {
        let idx = 21;
        set_awake(idx);
        assert!(!ring(idx), "an awake target is fully woken by the store");
        assert!(!ring(idx), "and stays awake across rings");
    }

    /// A ring moves the generation without disturbing the flag, and a flag
    /// transition never looks like a ring.
    #[test]
    fn rings_and_flags_occupy_disjoint_halves() {
        let idx = 22;
        set_awake(idx);
        let before = observe(idx);
        ring(idx);
        let after = observe(idx);
        assert!(rang(before, after), "a ring moves the generation");
        assert_eq!(after & AWAKE, AWAKE, "and leaves the flag alone");

        let armed = arm_park(idx);
        assert!(!rang(after, observe(idx)), "parking is not a ring");
        assert_eq!(armed & AWAKE, AWAKE, "arm reports the flag it cleared");
        assert_eq!(observe(idx) & AWAKE, 0, "and the target now reads parked");
    }

    /// The race the protocol exists to close, in the order that matters: a
    /// peer that rings before the arm is seen by the arm itself.
    #[test]
    fn a_ring_before_the_arm_is_caught_by_the_arm() {
        let idx = 23;
        set_awake(idx);
        let snapshot = observe(idx);

        // The peer's bump lands while the executor still reads awake, so the
        // peer skips its notice — the executor MUST therefore not park.
        assert!(!ring(idx), "target was awake, so no notice was posted");

        let armed = arm_park(idx);
        assert!(
            rang(snapshot, armed),
            "the arm observes the ring the peer chose not to announce"
        );
    }

    /// The other interleaving: a ring after the arm finds a parked target and
    /// posts the notice the executor's standing post will resolve.
    #[test]
    fn a_ring_after_the_arm_posts_the_notice() {
        let idx = 24;
        set_awake(idx);
        let snapshot = observe(idx);
        let armed = arm_park(idx);
        assert!(!rang(snapshot, armed), "nothing rang during the window");
        assert!(ring(idx), "a parked target is owed its notice");
    }

    /// A park on a thread whose executor is running clears `AWAKE` for the
    /// duration and puts it back — so a peer that rings mid-park is told to
    /// post its notice, and one that rings after the return is not.
    #[test]
    fn a_park_hides_awake_for_exactly_its_duration() {
        let idx = 26;
        set_awake(idx);
        {
            let _parked = Parked::enter_for(idx).expect("in band");
            assert_eq!(observe(idx) & AWAKE, 0, "a parked thread reads parked");
            assert!(ring(idx), "and is owed its notice");
        }
        assert_eq!(observe(idx) & AWAKE, AWAKE, "the return restores awake");
        assert!(!ring(idx), "so the next ring needs no notice");
    }

    /// The rule that makes the guard safe on a thread with no executor: it
    /// restores only what it cleared.
    ///
    /// A relay thread's bell reads parked *because nothing on it does a total
    /// drain*. Blanket-setting `AWAKE` when its interrupt park returns would
    /// make peers skip notices for a thread that is never going to look.
    #[test]
    fn a_park_on_a_thread_with_no_executor_restores_nothing() {
        let idx = 27;
        assert_eq!(observe(idx) & AWAKE, 0, "never had an executor");
        {
            let _parked = Parked::enter_for(idx).expect("in band");
            assert_eq!(observe(idx) & AWAKE, 0, "still parked");
        }
        assert_eq!(
            observe(idx) & AWAKE,
            0,
            "a guard must not invent an awake it never cleared"
        );
        assert!(ring(idx), "so peers keep posting their notices");
    }

    /// Nested is not a case the runtime produces — a thread parks once — but
    /// the restore rule must still compose rather than leak an awake.
    #[test]
    fn park_guards_compose_without_inventing_awake() {
        let idx = 28;
        set_awake(idx);
        {
            let _outer = Parked::enter_for(idx).expect("in band");
            {
                let _inner = Parked::enter_for(idx).expect("in band");
                assert_eq!(observe(idx) & AWAKE, 0, "parked");
            }
            assert_eq!(
                observe(idx) & AWAKE,
                0,
                "the inner guard cleared nothing, so it restores nothing"
            );
        }
        assert_eq!(observe(idx) & AWAKE, AWAKE, "the outer guard owns the flag");
    }

    /// A doorbell peer index and the bell it names are one arithmetic.
    #[test]
    fn a_doorbell_peer_index_names_its_own_bell() {
        let idx = 29;
        let peer = crate::futex::PEER_BASE + idx as u16;
        set_awake(idx);
        assert!(
            !ring_doorbell_peer(peer),
            "the peer index resolves to the bell that thread owns"
        );

        // Indices below the peer base are this thread's own recv/landing
        // slots, not a peer — conservatively owed a notice.
        assert!(
            ring_doorbell_peer(0),
            "a non-peer doorbell index must not be read as a bell"
        );
        assert!(ring_doorbell_peer(1), "nor the landing slot");
    }

    /// Counters move in the directions their names claim.
    #[test]
    fn stats_count_the_two_wins_separately() {
        let idx = 25;
        set_awake(idx);
        let before = stats();
        ring(idx);
        let after = stats();
        assert!(after.rings > before.rings, "every ring counts");
        assert!(
            after.skipped > before.skipped,
            "an awake target counts a skip"
        );
    }
}
