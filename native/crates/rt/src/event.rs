//! The event channel's runtime half — **the wait ladder and the
//! conditional kick** (`res/EVENT.md`).
//!
//! [`kore_ring::event`] owns the memory: the seqlock'd state, the record
//! ring, the wake word, the park intent.  This module owns the two verbs
//! that need what a contract crate may not name — an executor to park on
//! and a send end to kick with:
//!
//! - [`wait_past`] — the consumer's one wait.  The ladder inside it is
//!   nobody's configuration: **check** (the word already moved — return),
//!   **monitor** (sleep the core on the word's cache line for a few
//!   microseconds; the producer's ordinary store is the wake, no syscall
//!   on either side), **park on the word** (rung 3′ — the kernel
//!   address-wait, `res/EVENT.md` §6: the executor's idle park carries
//!   the wake word by the *authority* that mapped it, and the producer's
//!   `wake_word` — or the word being seen moved at park time — resolves
//!   it), else **park on the doorbell** (rung 3 — the executor's standing
//!   post with the deadline folded in; the producer's conditional notice
//!   resolves it).  Rung 3′ is taken exactly when the channel's window is
//!   one this task mapped through a noting verb
//!   ([`cap::mapped::locate`](crate::cap::mapped::locate)) and the
//!   executor's one word arm is free; otherwise rung 3, unchanged.  No
//!   caller changed for it to land.
//! - [`commit_and_kick`] — the producer's one batch end.  Advances the
//!   wake word, and wakes the consumer **only if it had committed to
//!   waiting** — by `wake_word` on the memory when the consumer parked on
//!   the word and this side can name it, by the notice otherwise — so a
//!   running consumer costs the producer zero syscalls, which is the
//!   whole point at load.
//!
//! Every return is a hint; the memory is the truth.  Callers re-check in
//! a loop, exactly as [`futex`](crate::futex) and the ring teach.  What
//! this ladder adds over `futex::wait_async` is the *cross-task* shape:
//! the word lives in a shared lease, the intent word makes the producer's
//! kick conditional, and the monitor rung works against any agent that
//! stores to the line — another task, or a device's DMA.

use core::sync::atomic::AtomicU32;
use core::time::Duration;

use kore_abi::session::Notice;
use kore_ring::event::{EventSink, EventSource, EventState, PARK_WORD};
use kore_ring::spsc::RingRecord;

use crate::cap::mapped;
use crate::cap::verbs;
use crate::chan::ChanTx;
use crate::transport::Transport;

/// The event kick's notice cookie — "kr_event".  A hint for diagnostics,
/// never a filter: the consumer re-checks its channels on every wake
/// whatever the cookie says (the doorbell mux contract, §3.1).
pub const COOKIE_EVENT: u64 = 0x6B72_5F65_7665_6E74;

/// How an event wait ended. Every answer means "recheck your sources":
/// a deadline does not promise nothing changed (the change may have
/// landed as the wait expired), and `Changed` does not promise the
/// caller's condition holds — only that the word moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventWait {
    /// The wake word moved past `seen`.
    Changed,
    /// The deadline elapsed first.
    TimedOut,
    /// Another executor event woke a composed waiter; recheck all inputs.
    Woken,
}

/// Pre-park monitor budget: how many monitored turns, and the cap on
/// one.  The total is the whole stall this may cost an executor's
/// sibling futures — a few microseconds, against a park that costs more
/// than that on its own (the same bet, and the same bound, as
/// `futex::wait_async`'s pre-park spin).  On silicon without a monitor
/// each turn degrades to a spin hint and the budget still holds.
const MONITOR_TURNS: usize = 2;
const MONITOR_CAP: Duration = Duration::from_micros(2);

/// Bound of one word park when the caller asked for no deadline: the
/// kernel address-wait is never unbounded, so an open-ended wait is a
/// loop of bounded parks, each re-checking the word — a slice long enough
/// that an idle consumer is not paced by it, short enough that a lost
/// wake (a producer that could name neither word nor doorbell) costs one
/// slice, not forever.
const WORD_PARK_SLICE_NS: u64 = 250_000_000;

/// Wait until the channel's wake word moves past `seen` (a value the
/// caller read with [`EventSink::seq`] *before* its last check), or
/// until `timeout` elapses; `None` waits until woken.
///
/// The caller's loop is the contract:
///
/// ```ignore
/// loop {
///     let seen = sink.seq();
///     while let Some(record) = sink.pop() { /* … */ }
///     let state = sink.state();
///     if condition_met(state) { break; }
///     event::wait_past(&sink, seen, deadline).await;
/// }
/// ```
///
/// Reading `seen` before the checks is what makes the wait lossless: a
/// batch landing between the checks and the wait moves the word past
/// `seen`, so the wait returns instead of sleeping on it.
///
/// A cancelled wait (the future dropped mid-park) withdraws its park
/// intent on drop, so a departed waiter is never charged a notice.
///
/// The word rung (3′) never lengthens a wait: it re-checks `seq` on every
/// wake exactly as the doorbell rung does, keeps the same deadline, and
/// keeps the doorbell post standing beneath it — a producer that cannot
/// name the word still reaches this waiter with its notice.
///
/// # Panics
/// Polled outside the kore executor, as
/// [`next_wake`](crate::executor::next_wake): the park rung is the
/// executor's own.
pub async fn wait_past<S: EventState, R: RingRecord>(
    sink: &EventSink<'_, S, R>,
    seen: u32,
    timeout: Option<Duration>,
) -> EventWait {
    wait_with(sink, seen, timeout, false).await
}

/// Run the event wait ladder once, returning on any executor wake. Use this
/// for a task that also handles input, cancellation, or other channels. Read
/// `seen` before draining every source; a return never promises which changed.
/// `TimedOut` is reported only when the deadline had already passed before
/// the park; after one park the answer is `Changed` or `Woken`, and the caller
/// keeps its own deadline.  Dropping this future withdraws the park intent.
pub async fn wait_once<S: EventState, R: RingRecord>(
    sink: &EventSink<'_, S, R>,
    seen: u32,
    timeout: Option<Duration>,
) -> EventWait {
    wait_with(sink, seen, timeout, true).await
}

async fn wait_with<S: EventState, R: RingRecord>(
    sink: &EventSink<'_, S, R>,
    seen: u32,
    timeout: Option<Duration>,
    once: bool,
) -> EventWait {
    // Rung 1 — check.
    if sink.seq() != seen {
        return EventWait::Changed;
    }
    // Rung 2 — monitor the word's cache line.  The producer's `commit`
    // store wakes the core directly; bounded so a slow producer costs a
    // few microseconds of this core, never more.
    for _ in 0..MONITOR_TURNS {
        crate::hint::wait_on(sink.seq_word(), seen, MONITOR_CAP);
        if sink.seq() != seen {
            return EventWait::Changed;
        }
    }
    // Rung 3 — park.  Publish the intent first (the Dekker half the
    // channel proves), then sleep on the executor; the producer's
    // conditional notice — or any other wake, spurious included —
    // re-runs the check.
    if !sink.park(seen) {
        return EventWait::Changed;
    }
    // Withdraw the intent however this wait ends — deadline, wake, or
    // the future dropped mid-park — so a later batch is not charged a
    // notice for a waiter that left.  A harmlessly-early withdrawal
    // (the producer already collected the intent) is a no-op store.
    struct Withdraw<'w>(&'w AtomicU32);
    impl Drop for Withdraw<'_> {
        fn drop(&mut self) {
            self.0.store(0, core::sync::atomic::Ordering::SeqCst);
        }
    }
    let _withdraw = Withdraw(sink.parked_word());
    let deadline = match timeout {
        Some(t) => Some(
            crate::executor::clock_ns()
                .await
                .saturating_add(u64::try_from(t.as_nanos()).unwrap_or(u64::MAX)),
        ),
        None => None,
    };
    // Rung 3′ — park on the word itself, when this task can name it by
    // the authority that mapped it.  The intent is restated as a word park
    // before every re-park (the value tells the producer which wake
    // reaches us; the Dekker pair above is untouched — the kernel makes
    // the final `seq` check against the physical word at park time).  The
    // arm can be refused (the executor's one word arm is owned) or evicted
    // (an alternate delivery arm arrived); either way the wait falls back
    // to rung 3 by re-publishing the doorbell intent through `park`, whose
    // own re-check covers the gap.
    let mut word = mapped::locate(sink.seq_word() as *const AtomicU32 as *const u8);
    loop {
        if sink.seq() != seen {
            return EventWait::Changed;
        }
        if let Some((subject, offset)) = word {
            // An unbounded wait still bounds each word park (the kernel
            // requires it); the loop's re-check makes the bound invisible.
            let bound = match deadline {
                Some(at) => {
                    if crate::executor::clock_ns().await >= at {
                        return EventWait::TimedOut;
                    }
                    at
                }
                None => crate::executor::clock_ns()
                    .await
                    .saturating_add(WORD_PARK_SLICE_NS),
            };
            sink.park_on_word();
            match crate::executor::next_wake_on_word(subject, offset, seen, bound).await {
                Some(_) if once => {
                    // A word park also carries the executor's doorbell. A
                    // composed waiter must drain its other inputs on that
                    // wake, even when this channel's word stayed unchanged.
                    return if sink.seq() != seen {
                        EventWait::Changed
                    } else {
                        EventWait::Woken
                    };
                }
                Some(_) => continue,
                None => {
                    word = None;
                    if !sink.park(seen) {
                        return EventWait::Changed;
                    }
                    continue;
                }
            }
        }
        match deadline {
            None => {
                let _ = crate::executor::next_wake().await;
            }
            Some(at) => {
                let now = crate::executor::clock_ns().await;
                if now >= at {
                    return EventWait::TimedOut;
                }
                // Recomputed each pass, so a spurious wake cannot extend
                // the total wait (the futex rule).
                crate::executor::next_wake_for(at - now).await;
            }
        }
        if once {
            // One park is the composed waiter's whole turn: say whether the
            // word moved and let the caller recheck its other inputs and its
            // own deadline — no second clock read for an answer nobody acts on.
            return if sink.seq() != seen {
                EventWait::Changed
            } else {
                EventWait::Woken
            };
        }
    }
}

/// End a producer batch: advance the wake word, and wake the consumer
/// **only if it had committed to waiting** — a `wake_word` on the memory
/// when it parked on the word and this side can name it (`res/EVENT.md`
/// §6), the doorbell notice otherwise.
///
/// Returns whether a wake went out.  `Err(WouldBlock)` from the send
/// is treated as delivered-enough and dropped: a full doorbell means the
/// consumer has an unconsumed wake standing, and the §3.1 order (the
/// channel writes happen-before this kick) guarantees its next drain
/// sees this batch.  Real transport failures also resolve to `false`;
/// the deadline rung of the consumer's ladder is the recovery, exactly
/// as for a lost kick today.
pub async fn commit_and_kick<S: EventState, R: RingRecord, T: Transport>(
    source: &mut EventSource<'_, S, R>,
    kick: &ChanTx<'_, T>,
) -> bool {
    let intent = source.commit_intent();
    if intent == 0 {
        return false;
    }
    // The consumer parked on the word (rung 3′): wake it on the memory,
    // by this side's own mapping of it — no send end needed, one trap.
    // A word this side cannot name, or a refused wake, falls back to the
    // notice: the consumer's word park carries its doorbell post too.
    if intent == PARK_WORD
        && let Some((subject, offset)) =
            mapped::locate(source.seq_word() as *const AtomicU32 as *const u8)
        && verbs::wake_word(subject, offset, u32::MAX)
    {
        return true;
    }
    kick.send(Notice {
        cookie: COOKIE_EVENT,
        payload: 0,
    })
    .await
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::session::SlotRef;
    use kore_ring::event::{PARK_DOORBELL, SEQ_AT, split};
    use std::vec;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable)]
    struct State {
        serial: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable)]
    struct Record {
        value: u32,
        _pad: u32,
    }

    /// Rung selection is a fact about the window, not the caller: a channel
    /// in memory nobody noted keeps the doorbell rung, and once its window
    /// is noted the wake word names itself as `(slot, SEQ_AT)` — the exact
    /// pair the `WordWaitReq` / `WakeWordReq` carry — without the consumer
    /// or producer changing a line.
    #[test]
    fn the_word_rung_is_taken_exactly_when_the_window_is_noted() {
        let bytes = kore_ring::event::region_bytes::<State, Record>(4) + 64;
        let mut backing = vec![0u8; bytes];
        let off = backing.as_ptr().align_offset(64);
        let window = &mut backing[off..];
        let (base, len) = (window.as_ptr(), window.len());
        let (mut source, sink) = split::<State, Record>(window, 4).unwrap();

        let word = sink.seq_word() as *const AtomicU32 as *const u8;
        assert_eq!(mapped::locate(word), None, "an unnoted window: rung 3");

        let slot = SlotRef::from_wire(21);
        mapped::note(base, len, slot);
        assert_eq!(mapped::locate(word), Some((slot, SEQ_AT as u32)));
        assert_eq!(
            mapped::locate(source.seq_word() as *const AtomicU32 as *const u8),
            Some((slot, SEQ_AT as u32)),
            "both halves name the same word"
        );

        // The intent word tells the producer which wake reaches the
        // consumer: a plain park is the doorbell, restated as the word park.
        let seen = sink.seq();
        assert!(sink.park(seen));
        assert_eq!(sink.parked_word().load(core::sync::atomic::Ordering::SeqCst), PARK_DOORBELL);
        sink.park_on_word();
        assert_eq!(sink.parked_word().load(core::sync::atomic::Ordering::SeqCst), PARK_WORD);
        assert_eq!(source.commit_intent(), PARK_WORD, "collected as a word park");
        assert_eq!(source.commit_intent(), 0, "collected once");

        mapped::forget(base);
        assert_eq!(mapped::locate(word), None, "an unmapped window: rung 3 again");
    }

    use core::future::Future;
    use core::sync::atomic::Ordering;
    use core::task::{Context, Poll, Waker};

    #[test]
    fn cancelling_a_composed_wait_withdraws_its_intent() {
        const BYTES: usize = kore_ring::event::region_bytes::<u32, kore_abi::svc::OutChunk>(1);
        #[repr(align(64))]
        struct Window([u8; BYTES]);
        let mut window = Window([0; BYTES]);
        let (mut source, sink) =
            kore_ring::event::split::<u32, kore_abi::svc::OutChunk>(&mut window.0, 1).unwrap();
        let seen = sink.seq();
        {
            let mut wait = core::pin::pin!(wait_once(&sink, seen, None));
            let mut cx = Context::from_waker(Waker::noop());
            // The first poll arms the event and yields before entering any
            // executor-specific wake handling. Cancellation is a real owner
            // dropping that Pending future, as an input-select would do.
            assert_eq!(wait.as_mut().poll(&mut cx), Poll::Pending);
            assert_eq!(sink.parked_word().load(Ordering::SeqCst), 1);
        }
        assert_eq!(sink.parked_word().load(Ordering::SeqCst), 0);
        assert!(!source.commit(), "a cancelled waiter is owed no notice");
        assert!(
            !sink.park(seen),
            "publication still invalidates the old snapshot"
        );
    }
}
