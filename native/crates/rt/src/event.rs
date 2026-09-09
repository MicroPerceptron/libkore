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
//!   on either side), **park** (the executor's standing doorbell post
//!   with the deadline folded in; the producer's conditional notice
//!   resolves it).  A kernel address-wait slots between the last two
//!   when it lands (`res/EVENT.md` §6) without changing a caller.
//! - [`commit_and_kick`] — the producer's one batch end.  Advances the
//!   wake word, and sends the notice **only if the consumer had
//!   committed to waiting** — a running consumer costs the producer
//!   zero syscalls, which is the whole point at load.
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
use kore_ring::event::{EventSink, EventSource, EventState};
use kore_ring::spsc::RingRecord;

use crate::chan::ChanTx;
use crate::transport::Transport;

/// The event kick's notice cookie — "kr_event".  A hint for diagnostics,
/// never a filter: the consumer re-checks its channels on every wake
/// whatever the cookie says (the doorbell mux contract, §3.1).
pub const COOKIE_EVENT: u64 = 0x6B72_5F65_7665_6E74;

/// How a [`wait_past`] ended.  Both answers mean "look at the channel":
/// a deadline does not promise nothing changed (the change may have
/// landed as the wait expired), and `Changed` does not promise the
/// caller's condition holds — only that the word moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventWait {
    /// The wake word moved past `seen`.
    Changed,
    /// The deadline elapsed first.
    TimedOut,
}

/// Pre-park monitor budget: how many monitored turns, and the cap on
/// one.  The total is the whole stall this may cost an executor's
/// sibling futures — a few microseconds, against a park that costs more
/// than that on its own (the same bet, and the same bound, as
/// `futex::wait_async`'s pre-park spin).  On silicon without a monitor
/// each turn degrades to a spin hint and the budget still holds.
const MONITOR_TURNS: usize = 2;
const MONITOR_CAP: Duration = Duration::from_micros(2);

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
/// # Panics
/// Polled outside the kore executor, as
/// [`next_wake`](crate::executor::next_wake): the park rung is the
/// executor's own.
pub async fn wait_past<S: EventState, R: RingRecord>(
    sink: &EventSink<'_, S, R>,
    seen: u32,
    timeout: Option<Duration>,
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
    loop {
        if sink.seq() != seen {
            return EventWait::Changed;
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
    }
}

/// End a producer batch: advance the wake word, and kick the consumer's
/// doorbell **only if it had committed to waiting**.
///
/// Returns whether a notice went out.  `Err(WouldBlock)` from the send
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
    if !source.commit() {
        return false;
    }
    kick.send(Notice {
        cookie: COOKIE_EVENT,
        payload: 0,
    })
    .await
    .is_ok()
}
