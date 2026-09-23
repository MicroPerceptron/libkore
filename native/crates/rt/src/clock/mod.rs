//! Zero-trap user clock over the founder-published [`ClockPage`].
//!
//! Reading time is ambient data.  It samples the architecture's monotonic
//! counter directly and projects it through immutable calibration in this
//! thread's boot page.  Arming remains a separate operation: executor timers
//! fold into the one `WAIT_DELIVERY | WAIT_SLEEP` park, while the time service
//! multiplexes asynchronous alarms for clients that need an event stream.

use kore_abi::clock::{ClockPage, TimerError, TimerResult, UnixTime, Uptime};

#[cfg(target_os = "none")]
trait UserCounter {
    fn ticks() -> u64;
}

/// This thread's immutable founder-published calibration block.
#[inline]
pub fn calibration() -> Option<&'static ClockPage> {
    crate::entry::current_clock_page()
}

#[cfg(all(target_os = "none", target_arch = "x86_64"))]
mod x86_64;
#[cfg(all(target_os = "none", target_arch = "x86_64"))]
use x86_64::Counter;

#[cfg(all(target_os = "none", target_arch = "aarch64"))]
mod aarch64;
#[cfg(all(target_os = "none", target_arch = "aarch64"))]
use aarch64::Counter;

#[cfg(all(target_os = "none", target_arch = "riscv64"))]
mod riscv64;
#[cfg(all(target_os = "none", target_arch = "riscv64"))]
use riscv64::Counter;

/// Monotonic uptime with no trap.
#[inline]
pub fn uptime() -> TimerResult<Uptime> {
    let page = calibration().ok_or(TimerError::Uncalibrated)?;
    uptime_from(page)
}

/// Calendar time with no trap once the boot wall anchor is available.
#[inline]
pub fn wall() -> TimerResult<UnixTime> {
    let page = calibration().ok_or(TimerError::Uncalibrated)?;
    wall_from(page)
}

/// Project a live counter sample through an explicitly supplied page.  This
/// is the composable form for a time-service client holding a later disciplined
/// shared page.
#[inline]
pub fn uptime_from(page: &ClockPage) -> TimerResult<Uptime> {
    if !page.validate() || !page.user_counter_readable() {
        return Err(TimerError::Uncalibrated);
    }
    #[cfg(target_os = "none")]
    {
        page.uptime(Counter::ticks())
    }
    #[cfg(not(target_os = "none"))]
    {
        let _ = page;
        Err(TimerError::Unsupported)
    }
}

/// [`uptime_from`]'s wall-clock projection.
#[inline]
pub fn wall_from(page: &ClockPage) -> TimerResult<UnixTime> {
    if !page.validate() || !page.user_counter_readable() {
        return Err(TimerError::Uncalibrated);
    }
    #[cfg(target_os = "none")]
    {
        page.wall(Counter::ticks())
    }
    #[cfg(not(target_os = "none"))]
    {
        let _ = page;
        Err(TimerError::Unsupported)
    }
}

/// Fast monotonic nanoseconds for the executor's local anchor.  `None` keeps
/// host tests event-driven and lets a legacy/uncalibrated target fall back to
/// the trap-return millisecond stamp.
#[inline]
pub(crate) fn monotonic_nanos() -> Option<u64> {
    uptime().ok().map(Uptime::as_nanos)
}

/// Advance an absolute millisecond anchor from a wrapping kernel stamp.
/// Prefer the full calibrated uptime when available; otherwise observations
/// must be less than half a stamp period (about 24.9 days) apart. A zero
/// anchor has no known epoch, so its first stamp initializes the low word.
/// A caller handing time to a new executor must preserve the full anchor.
/// Stale samples never move time backward or manufacture a future epoch.
pub fn anchor_millis(previous_ms: u64, stamp_ms: u32) -> u64 {
    let previous_ms = monotonic_nanos()
        .map(|ns| previous_ms.max(ns / 1_000_000))
        .unwrap_or(previous_ms);
    extend_millis(previous_ms, stamp_ms)
}

fn extend_millis(previous_ms: u64, stamp_ms: u32) -> u64 {
    if previous_ms == 0 {
        return u64::from(stamp_ms);
    }
    let delta = stamp_ms.wrapping_sub(previous_ms as u32);
    if delta < (1 << 31) {
        previous_ms.saturating_add(u64::from(delta))
    } else {
        previous_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn millisecond_anchor_crosses_wrap_and_rejects_stale_samples() {
        let period = 1u64 << 32;
        let mut now = extend_millis(0, u32::MAX - 2);
        assert_eq!(now, period - 3);
        for (stamp, expected) in [
            (u32::MAX, period - 1),
            (0, period),
            (1, period + 1),
            (u32::MAX, period + 1),
            (0, period + 1),
            (7, period + 7),
            (7, period + 7),
        ] {
            now = extend_millis(now, stamp);
            assert_eq!(now, expected);
        }
        assert_eq!(extend_millis(3 * period + 9, 12), 3 * period + 12);
        assert_eq!(extend_millis(3 * period + 9, 8), 3 * period + 9);
        assert_eq!(extend_millis(91, 91 + (1 << 31)), 91);
        assert_eq!(extend_millis(u64::MAX - 1, 0), u64::MAX);
    }

    #[test]
    fn malformed_or_privileged_only_pages_fail_closed() {
        let mut page = ClockPage::new(1_000_000_000, 7, 0, false);
        assert_eq!(uptime_from(&page), Err(TimerError::Uncalibrated));
        page.flags = u16::MAX;
        assert!(!page.validate());
        assert_eq!(uptime_from(&page), Err(TimerError::Uncalibrated));
    }
}
