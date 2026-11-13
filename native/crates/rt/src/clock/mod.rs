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

/// Monotonic nanoseconds now, for a thread that names a deadline to the
/// kernel: the calibrated counter where the task has one, and the kernel's
/// own stamp where it has none.
///
/// A target without a calibrated user clock (an emulator with no invariant
/// counter, a thread founded before the clock page) can still park against
/// a deadline, because the kernel keeps the time: a sleep to an instant
/// already past resolves at once and returns the kernel's clock, in whole
/// milliseconds. That costs a trap and is a millisecond coarse, so the
/// answer may be up to a millisecond behind; a deadline made from it is
/// never early by more than that. `None` only where the thread cannot trap
/// at all (a host build).
pub fn deadline_base_nanos() -> Option<u64> {
    monotonic_nanos().or_else(|| {
        crate::cap::verbs::sleep_until(1).map(|stamp| (stamp >> 32).saturating_mul(1_000_000))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_or_privileged_only_pages_fail_closed() {
        let mut page = ClockPage::new(1_000_000_000, 7, 0, false);
        assert_eq!(uptime_from(&page), Err(TimerError::Uncalibrated));
        page.flags = u16::MAX;
        assert!(!page.validate());
        assert_eq!(uptime_from(&page), Err(TimerError::Uncalibrated));
    }
}
