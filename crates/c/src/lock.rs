//! The futex mutex this runtime's threads take turns with: the streams'
//! lock, and the callers of thread 0's memory cell.
//!
//! The classic three states. A waiter's park is bounded by the runtime's
//! slice, because a holder's wake may not reach it: a thread holds a send
//! into a peer's doorbell only where the thread controller relays one.

use core::sync::atomic::{AtomicU32, Ordering};
use core::time::Duration;

use kore_std::runtime::futex::{self, WaitOutcome};
use kore_std::runtime::pace;

const FREE: u32 = 0;
const HELD: u32 = 1;
const CONTENDED: u32 = 2;

/// How long a waiter parks before it tries again: the runtime's slice for a
/// waiter its waker cannot reach (the thread controller's `UNREACHED_SLICE`).
const SLICE: Duration = Duration::from_nanos(pace::CEDE_NS);

/// A mutex over one futex word. Not reentrant: a holder that needs to be
/// counts its own depth ([`crate::stdio`]).
pub(crate) struct Lock(AtomicU32);

impl Lock {
    pub(crate) const fn new() -> Self {
        Self(AtomicU32::new(FREE))
    }

    pub(crate) fn acquire(&self) {
        if self
            .0
            .compare_exchange(FREE, HELD, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
        while self.0.swap(CONTENDED, Ordering::Acquire) != FREE {
            // Bounded: the holder's wake may not reach this thread.
            if futex::wait(&self.0, CONTENDED, Some(SLICE)) == WaitOutcome::Unsupported {
                // No doorbell to park on: the deadline park any thread can.
                let _ = pace::cede();
            }
        }
    }

    pub(crate) fn release(&self) {
        // SeqCst, as the futex contract asks of the store before a wake.
        if self.0.swap(FREE, Ordering::SeqCst) == CONTENDED {
            futex::wake(&self.0);
        }
    }
}
