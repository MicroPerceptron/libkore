//! **Sleep the core on a cache line** — this runtime's view of
//! [`kore_hint`], the shared hardware address monitor.
//!
//! The mechanism lives in its own crate because the **kernel** needs it too:
//! ring 3 cannot issue `MONITOR`/`MWAIT`, so this runtime was forced to build
//! the WAITPKG / `MONITORX` seam properly, while the kernel could always fall
//! back to `hlt` and so never grew a *timed* equivalent. One implementation,
//! two callers.
//!
//! What stays here is the only thing that is genuinely runtime-specific: where
//! the counter rate comes from. Userland reads the founder-published clock
//! page; the kernel supplies its own TSC calibration. The shared crate takes
//! the rate as an argument rather than reaching for either, which is what lets
//! it sit below both.

pub use kore_hint::{AddressMonitor, Monitor, NoMonitor, WaitEnd, available, name};

use core::sync::atomic::AtomicU32;
use core::time::Duration;

/// This task's monotonic counter rate, from the founder-published clock page.
///
/// `0` when no calibrated page is anchored — the pre-`Rt` window, and any
/// machine whose counter userland may not read. The shared engine treats that
/// as "cannot bound this wait" and spins, which is always correct.
#[inline]
fn counter_hz() -> u64 {
    match crate::clock::calibration() {
        Some(page) if page.validate() && page.user_counter_readable() => page.tick_hz,
        _ => 0,
    }
}

/// [`kore_hint::wait_on`] over this task's clock.
///
/// The one call sites use. Spins when the machine has no monitor, so it is
/// always a correct replacement for a bare `core::hint::spin_loop()` and never
/// a behaviour change a caller has to reason about.
#[inline]
pub fn wait_on(word: &AtomicU32, expected: u32, cap: Duration) -> WaitEnd {
    kore_hint::wait_on(word, expected, cap, counter_hz())
}

/// [`kore_hint::pause_until`] over this task's clock — a timed pause with no
/// watched word.
#[inline]
pub fn pause_until(cap: Duration) -> WaitEnd {
    kore_hint::pause_until(cap, counter_hz())
}
