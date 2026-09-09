//! **The SPMD barrier** — N workers meet at a phase line, cheaply.
//!
//! The sync primitive a data-parallel SPMD region (`res/SPMD.md`
//! §3) barriers on between tiles: every worker calls
//! [`arrive_and_wait`](SpmdBarrier::arrive_and_wait), the last arrival
//! releases the phase, and exactly one worker per phase is told it was the
//! [`Leader`](Arrival::Leader) — the election a reduction epilogue wants,
//! for free.
//!
//! # Why this is not a spinlock
//!
//! The wait rides [`futex::wait_barrier`](crate::futex::wait_barrier),
//! the two-tier wait this barrier needs with a budget tuned to its own
//! counterparty: a run of **monitored** turns asleep on the phase
//! word's cache line ([`hint::wait_on`](crate::hint::wait_on) — the
//! releaser's plain store is the wake, no syscall, no capability), then
//! a real park for a straggler. The monitored budget is a
//! **scheduling-jitter window** (tens of microseconds), not a lock's
//! sub-µs handoff: a barrier's counterparty is a co-scheduled sibling
//! that arrives within that window, and the monitor makes the wait free
//! until it does (the store ends the `UMWAIT` the instant it lands). So
//! a co-resident region resolves its barrier in the monitored tier — no
//! park, no kick — and a descheduled worker past the window stops
//! occupying its core. That split is the whole optimization; tuning the
//! spin budget to the barrier (not the lock default) is what the first
//! metal terminal round proved necessary — the shorter lock budget parked
//! ~500 times over 512 phases at w≤4, each release paying a syscall and
//! a kick that could miss.
//!
//! # The slow path is a sliced park; the star makes it a woken one
//!
//! A parked waiter is released by a doorbell kick — which needs a send
//! end the releaser holds.  A starred region (`res/SPMD.md` §3) endows
//! every member the intra-chunk kick MESH: the leader's
//! [`wake_all`](crate::futex::wake_all) kicks a parked chunk sibling in
//! ONE hop, and only a waiter the caller holds no send for (a
//! cross-chunk member of a >MESH_CHUNK region) routes through the hub,
//! leader → hub → member.  The two-hop relay was the original release
//! path, and the metal terminal priced it: the hub is unpinned and parked,
//! so a w≤4 release of a parked member averaged ~½·[`PARK_SLICE`] —
//! the wake mesh must not have a scheduler hop in the common case.
//! The waiter STILL never parks unbounded: it parks in [`PARK_SLICE`]
//! slices and re-reads the phase word between them, so a starless
//! region (provisioning degraded, no `CHANNEL_SPACE`) keeps the exact
//! pre-star behavior and a lost notice costs one slice, never a wedge.
//! The word is the truth; the kick is a latency optimization. A
//! monitored-tier wake is instant either way — the slice only bounds
//! how stale a *parked* straggler can run.
//!
//! [`Unsupported`](crate::futex::WaitOutcome::Unsupported) — a thread
//! that cannot park at all, and every off-metal build — degrades the same
//! turn to a monitored spin instead, so the barrier is correct (never
//! merely hot) everywhere, exactly as `hint`'s `NoMonitor` floor spins
//! where no monitor exists.
//!
//! # Reuse without a second allocation
//!
//! The barrier is **phase-counting**: `phase` is a generation that only
//! ever steps forward, `pending` counts the current phase's outstanding
//! arrivals. One barrier serves every tile boundary of a launch — workers
//! re-enter `arrive_and_wait` immediately, and the accounting must
//! tolerate a worker racing into phase *g+1* while the phase-*g* release is
//! still in flight (see `arrive_and_wait` on the reset/flip order).
//!
//! Both words live on their own destructive-interference span so a worker
//! hammering `pending` does not steal the line every waiter sleeps on.

use core::sync::atomic::AtomicU32;
use core::time::Duration;

use crate::futex;

/// One atomic on its own destructive-interference span.
///
/// 128, not 64: Intel's spatial prefetcher pulls line *pairs*, and the
/// large-core aarch64 targets use 128-byte lines outright — the same span
/// crossbeam isolates against. The cost is bytes in a type constructed
/// once per region, not per worker.
#[repr(align(128))]
pub(crate) struct Line(pub(crate) AtomicU32);

impl Line {
    /// A line holding `v` — the shape shared state built from these
    /// (this barrier, the region's gate) constructs with.
    pub(crate) const fn new(v: u32) -> Self {
        Self(AtomicU32::new(v))
    }
}

/// How long one parked slice of the slow path may last.
///
/// The backstop for a kick that misses — with the mesh AND the barrier's
/// jitter-window spin (`futex::wait_barrier`) a parked sibling is
/// normally either woken by a kick or resolved in the monitored tier
/// before parking at all, so this bounds only the residual: a
/// cross-chunk waiter, a coalesced notice, a send the caller lacks. A
/// quarter-millisecond, down from the scheduler's 1 ms quantum floor:
/// once parks are rare the re-check traffic is negligible, and a shorter
/// slice cuts the tail cost of the misses that do happen. The terminal
/// bench's `parks=P/W/T` triple prices exactly this — `T` (rode the
/// slice) is the number the mesh and the spin exist to hold at zero.
///
/// (A rider observed costing well past one slice on metal points at the
/// kernel deadline itself firing late on an otherwise-idle pinned AP —
/// tracked separately; shortening the slice here does not depend on it,
/// and the jitter-window spin makes it moot at fitting widths.)
const PARK_SLICE: Duration = Duration::from_micros(250);

/// What a worker learns from [`SpmdBarrier::arrive_and_wait`].
///
/// Exactly one worker per phase is the leader. Which worker is deliberately
/// unspecified (it is the last *arrival*, not a fixed index): callers use
/// it to elect one worker for a phase epilogue — a reduction, a progress
/// tick — never to identify a worker. `PartialEq` so `== Leader` reads at
/// the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrival {
    /// This worker's arrival completed the phase and performed the release.
    Leader,
    /// This worker waited for the phase to be released by another.
    Member,
}

/// A reusable N-party phase barrier over the futex.
///
/// Construction fixes the party count; every phase requires exactly
/// `parties` arrivals. The barrier is `Sync` state, not authority — it
/// lives in memory every worker can reach (a shared band window or a
/// `static`), and holding a reference conveys nothing but the right to
/// wait, which is the correct amount of nothing.
pub struct SpmdBarrier {
    /// The generation word — the futex the wait tier sleeps on. Stepped
    /// (wrapping) once per released phase; compared only for equality, so
    /// wrap is harmless.
    phase: Line,
    /// Arrivals still outstanding in the current phase. Reset to
    /// `parties` by the leader *before* the phase steps (see
    /// [`arrive_and_wait`](Self::arrive_and_wait)).
    pending: Line,
    /// The party count. Fixed at construction for a const-width region;
    /// **set once at launch, behind the gate** for a runtime-width one
    /// ([`set_parties`](Self::set_parties)) — every worker's first
    /// `arrive_and_wait` is causally after that set through the gate's
    /// release edge, so the leader's per-phase read never races it. An
    /// atomic (not a bare `u32`) is exactly what lets one `static`
    /// barrier serve any width `≤` its constructed capacity.
    parties: AtomicU32,
}

impl SpmdBarrier {
    /// A barrier for exactly `parties` workers.
    ///
    /// `const` so a region's barrier can be a `static` beside the worker
    /// storage the launch macro sizes (`res/SPMD.md` §6) — no
    /// allocation, matching every fixed-shape structure in this crate.
    /// A runtime-width region constructs it at its capacity and narrows
    /// it per launch with [`set_parties`](Self::set_parties).
    ///
    /// # Panics
    /// `parties == 0` is a shape with no phases, refused at construction
    /// (compile-time, when the call is const-evaluated).
    pub const fn new(parties: u32) -> Self {
        assert!(parties > 0, "a barrier needs at least one party");
        Self {
            phase: Line(AtomicU32::new(0)),
            pending: Line(AtomicU32::new(parties)),
            parties: AtomicU32::new(parties),
        }
    }

    /// The party count this barrier currently releases on.
    pub fn parties(&self) -> u32 {
        self.parties.load(core::sync::atomic::Ordering::Relaxed)
    }

    /// **Set the width for the next launch** — a runtime-width region's
    /// `k`, stored before any worker passes the gate. Re-seeds `pending`
    /// to match (the previous launch left it at the previous width), so
    /// the first phase counts down from `k`. Sequenced before the gate's
    /// `RUN` store by the caller, so it is published to every worker by
    /// the same release edge that publishes the launch itself; a
    /// same-width relaunch may skip it (the leader already left `pending
    /// == parties`).
    pub fn set_parties(&self, parties: u32) {
        use core::sync::atomic::Ordering;
        debug_assert!(parties > 0, "a barrier needs at least one party");
        self.parties.store(parties, Ordering::Relaxed);
        self.pending.0.store(parties, Ordering::Relaxed);
    }

    /// Arrive at the current phase; return once every party has.
    ///
    /// Exactly one arrival per phase reports [`Arrival::Leader`], and it
    /// returns only after performing the release; every other arrival
    /// reports [`Arrival::Member`] after observing it. Calling with more
    /// than `parties` concurrent arrivals is a caller bug the accounting
    /// does not defend against (debug builds catch the underflow).
    pub fn arrive_and_wait(&self) -> Arrival {
        use core::sync::atomic::{Ordering, fence};

        // Relaxed is enough, and not by luck: the flip this load could
        // race with is causally gated on this worker's own `fetch_sub`
        // below. The load is sequenced before that RMW, the RMW
        // happens-before the flip (release sequence → the leader's
        // acquire fence → the store), and a load never reads from a
        // store that happens-after it — so the generation cannot be read
        // from the future. Nor from the past: a re-entering worker already
        // observed the previous flip on this same word, and read-read
        // coherence forbids going back. Do not "harden" this to SeqCst —
        // it would buy nothing and cost every arrival.
        let expected = self.phase.0.load(Ordering::Relaxed);
        let previous = self.pending.0.fetch_sub(1, Ordering::Release);
        debug_assert!(previous > 0, "barrier arrivals exceeded party count");

        if previous == 1 {
            // Acquire the release sequence formed by every arrival. Only the
            // leader pays this cost; members acquire the phase flip below.
            fence(Ordering::Acquire);
            // Reset BEFORE the flip: the instant the phase steps, a
            // released worker may re-enter for the next tile and decrement
            // — an unreset counter underflows. The Relaxed store is
            // ordered anyway: re-entrants acquire the flip first.
            self.pending.0.store(self.parties.load(Ordering::Relaxed), Ordering::Relaxed);
            // Store, THEN wake (the futex module's rule): a waiter
            // kicked before the flip is visible re-reads the old
            // generation and sleeps through its own release.
            self.phase
                .0
                .store(expected.wrapping_add(1), Ordering::SeqCst);
            futex::wake_all(&self.phase.0);
            Arrival::Leader
        } else {
            self.wait_for_flip(expected);
            Arrival::Member
        }
    }

    /// Wait until the phase word leaves `expected` — the member's half,
    /// [`wait_while`] on the phase line.
    fn wait_for_flip(&self, expected: u32) {
        wait_while(&self.phase.0, expected);
    }
}

/// Wait until `word` leaves `expected` — the two-tier wait the barrier's
/// member half and the region's launch gate share.
///
/// Each turn is [`futex::wait`], which spends its monitored spin budget
/// on the line before parking, capped at [`PARK_SLICE`] so a parked
/// waiter the waker's kick cannot reach (module docs) re-reads the word
/// once per slice. Every outcome — woken, timed out, the condition
/// already changed — means the same thing: *look at the word*.
/// [`Unsupported`](futex::WaitOutcome::Unsupported) alone must not
/// re-enter the park (its contract: the park will never work), so that
/// turn becomes a monitored spin instead — the same wait the futex's own
/// fast tier would have taken, and on a monitorless build the plain
/// capped spin the host suite exercises.
pub(crate) fn wait_while(word: &AtomicU32, expected: u32) {
    use core::sync::atomic::Ordering;
    // SeqCst throughout: the registry's Dekker discipline (a waiter's
    // check and a releaser's store must not be reorderable past one
    // another), matched rather than locally re-derived.
    while word.load(Ordering::SeqCst) == expected {
        match futex::wait_barrier(word, expected, Some(PARK_SLICE)) {
            futex::WaitOutcome::Unsupported => {
                crate::hint::wait_on(word, expected, SPIN_SLICE);
            }
            // Retry: the word already moved — the loop head sees it.
            // Woken / TimedOut: a kick, a slice boundary, or noise —
            // the word is the truth either way.
            _ => {}
        }
    }
}

/// The monitored-turn cap for a thread that cannot park.
///
/// The futex's own spin tier uses 1 µs turns for the same reason (its
/// `SPIN_CAP`): comfortably past a handoff, invisible to a human, and a
/// bound on how long a monitor that never fires can hold the core.
const SPIN_SLICE: Duration = Duration::from_micros(1);

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU32, Ordering};
    use std::thread;

    /// The property a barrier exists for: nothing published before an
    /// arrival is unobserved after the release. Each worker counts itself
    /// in before arriving; after returning, it must see everyone.
    #[test]
    fn a_release_observes_every_arrival() {
        const LANES: u32 = 4;
        const ROUNDS: u32 = 32;
        static ARRIVED: AtomicU32 = AtomicU32::new(0);
        static BARRIER: SpmdBarrier = SpmdBarrier::new(LANES);

        thread::scope(|s| {
            for _ in 0..LANES {
                s.spawn(|| {
                    for round in 0..ROUNDS {
                        ARRIVED.fetch_add(1, Ordering::SeqCst);
                        BARRIER.arrive_and_wait();
                        let seen = ARRIVED.load(Ordering::SeqCst);
                        // Everyone from this round, nobody from the next
                        // beyond the workers that already raced ahead — the
                        // floor is the full round's count.
                        assert!(
                            seen >= (round + 1) * LANES,
                            "released before every arrival: saw {seen} in round {round}"
                        );
                        // A second barrier makes the count exact between
                        // rounds, which is what proves REUSE: a pending
                        // counter the leader forgot to reset (or reset
                        // after the flip) underflows or wedges here.
                        BARRIER.arrive_and_wait();
                    }
                });
            }
        });
        assert_eq!(ARRIVED.load(Ordering::SeqCst), LANES * ROUNDS);
    }

    /// Exactly one leader per phase, however the arrivals race.
    #[test]
    fn each_phase_elects_exactly_one_leader() {
        const LANES: u32 = 3;
        const ROUNDS: u32 = 64;
        static LEADERS: AtomicU32 = AtomicU32::new(0);
        static BARRIER: SpmdBarrier = SpmdBarrier::new(LANES);

        thread::scope(|s| {
            for _ in 0..LANES {
                s.spawn(|| {
                    for _ in 0..ROUNDS {
                        if BARRIER.arrive_and_wait() == Arrival::Leader {
                            LEADERS.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                });
            }
        });
        assert_eq!(LEADERS.load(Ordering::SeqCst), ROUNDS);
    }

    /// A single-party barrier is a no-op that still leads: the shape a
    /// one-worker degenerate launch produces, and it must not wait.
    #[test]
    fn a_single_party_barrier_never_waits() {
        let b = SpmdBarrier::new(1);
        for _ in 0..8 {
            assert_eq!(b.arrive_and_wait(), Arrival::Leader);
        }
    }

    /// The generation compare is equality-only, so stepping must survive
    /// wrap. Exercised by seeding the phase near the top rather than
    /// running four billion rounds.
    #[test]
    fn the_generation_survives_wrap() {
        let b = SpmdBarrier::new(1);
        b.phase.0.store(u32::MAX, Ordering::SeqCst);
        assert_eq!(b.arrive_and_wait(), Arrival::Leader);
        assert_eq!(b.phase.0.load(Ordering::SeqCst), 0, "wrapped forward");
        assert_eq!(b.arrive_and_wait(), Arrival::Leader);
    }

    /// Zero parties is refused at construction, not discovered at the
    /// first phase that can never complete.
    #[test]
    #[should_panic(expected = "at least one party")]
    fn zero_parties_is_refused() {
        let _ = SpmdBarrier::new(0);
    }

    /// The two words must not share a destructive-interference span:
    /// arrival traffic on `pending` would otherwise evict the line every
    /// monitored waiter sleeps on.
    #[test]
    fn the_words_are_isolated() {
        let b = SpmdBarrier::new(2);
        let phase = core::ptr::from_ref(&b.phase.0) as usize;
        let pending = core::ptr::from_ref(&b.pending.0) as usize;
        assert!(
            phase.abs_diff(pending) >= 128,
            "phase and pending share an interference span"
        );
    }
}
