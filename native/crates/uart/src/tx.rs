//! Bounded transmit polling shared by kernel and service UART consumers.

use crate::{RegisterAccess, Uart};

/// Shared per-byte wait for kernel and service console consumers.
///
/// Two milliseconds covers a 9600-baud 8N1 character (about 1.04 ms), the
/// slowest enumerated SPCR line rate, with room for polling overhead. A 1 ms
/// timeout can reject that healthy line. Consumers of slower custom rates
/// may supply a larger timeout to [`transmit_byte`]. This is a byte-wait
/// allowance, not a hard latency bound on a whole record or register access.
pub const DEFAULT_TX_TIMEOUT_NS: u64 = 2_000_000;

/// Transmit one byte, or report that the transmitter remained unavailable.
///
/// `timeout_ns` bounds the wait against the caller's monotonic clock. Before
/// calibration (`None` at entry), `unclocked_probes` bounds unsuccessful polls
/// instead. Once a clocked wait starts, loss of the clock ends it; it must not
/// silently turn into an unbounded spin. A ready transmitter always gets one
/// opportunity, even with a zero budget.
///
/// The clock and register access are injected: kernel port I/O, kernel MMIO,
/// and service MMIO share this loop without importing one another's clocks,
/// scheduling, or address-monitor authority. `pause` is injected for the same
/// reason: production callers provide their native monitored relaxation while
/// a bootstrap path without a calibrated monitor supplies one spin. This
/// cannot interrupt a register access that itself never returns, and it does
/// not claim to bound an entire multi-byte record. A best-effort console
/// should stop that record after the first `false`; a reliable output queue
/// should leave the unsuccessful byte unconsumed.
#[inline]
pub fn transmit_byte<U: Uart + ?Sized>(
    uart: &U,
    regs: &dyn RegisterAccess,
    byte: u8,
    timeout_ns: u64,
    unclocked_probes: u32,
    mut now_ns: impl FnMut() -> Option<u64>,
    mut pause: impl FnMut(),
) -> bool {
    let deadline = now_ns().map(|now| now.saturating_add(timeout_ns));
    let mut remaining = unclocked_probes;
    while !uart.tx_ready(regs) {
        let spent = match deadline {
            Some(deadline) => now_ns().is_none_or(|now| now >= deadline),
            None => {
                remaining = remaining.saturating_sub(1);
                remaining == 0
            }
        };
        if spent {
            return false;
        }
        pause();
    }
    uart.tx_byte(regs, byte);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Uart16550;
    use core::cell::Cell;

    struct Transmitter {
        polls: Cell<u32>,
        ready_after: Option<u32>,
        writes: Cell<u32>,
        last: Cell<u8>,
    }

    impl Transmitter {
        fn new(ready_after: Option<u32>) -> Self {
            Self {
                polls: Cell::new(0),
                ready_after,
                writes: Cell::new(0),
                last: Cell::new(0),
            }
        }
    }

    impl RegisterAccess for Transmitter {
        fn read8(&self, offset: usize) -> u8 {
            assert_eq!(offset, 5);
            self.polls.set(self.polls.get() + 1);
            if self.ready_after.is_some_and(|n| self.polls.get() > n) {
                0x20
            } else {
                0
            }
        }

        fn write8(&self, offset: usize, value: u8) {
            assert_eq!(offset, 0);
            self.writes.set(self.writes.get() + 1);
            self.last.set(value);
        }

        fn read32(&self, _: usize) -> u32 {
            panic!("byte-spaced 16550")
        }

        fn write32(&self, _: usize, _: u32) {
            panic!("byte-spaced 16550")
        }
    }

    #[test]
    fn a_ready_byte_does_not_need_a_wait_budget() {
        let regs = Transmitter::new(Some(0));
        assert!(transmit_byte(
            &Uart16550::new(1),
            &regs,
            b'x',
            0,
            0,
            || None,
            || panic!("a ready transmitter must not pause"),
        ));
        assert_eq!(regs.polls.get(), 1);
        assert_eq!(regs.writes.get(), 1);
        assert_eq!(regs.last.get(), b'x');
    }

    #[test]
    fn a_stalled_transmitter_spends_only_its_time_budget() {
        let regs = Transmitter::new(None);
        let mut now = 0;
        assert!(!transmit_byte(
            &Uart16550::new(1),
            &regs,
            b'x',
            1_000_000,
            100_000,
            || {
                let sample = now;
                now += 250_000;
                Some(sample)
            },
            || {},
        ));
        assert_eq!(regs.polls.get(), 4);
        assert_eq!(regs.writes.get(), 0);
    }

    #[test]
    fn losing_a_clock_does_not_remove_the_wait_bound() {
        let regs = Transmitter::new(None);
        let mut samples = 0;
        assert!(!transmit_byte(
            &Uart16550::new(1),
            &regs,
            b'x',
            1_000_000,
            100_000,
            || {
                samples += 1;
                (samples == 1).then_some(0)
            },
            || {},
        ));
        assert_eq!(samples, 2);
        assert_eq!(regs.polls.get(), 1);
        assert_eq!(regs.writes.get(), 0);
    }

    #[test]
    fn an_uncalibrated_clock_keeps_the_probe_bound() {
        let regs = Transmitter::new(None);
        assert!(!transmit_byte(
            &Uart16550::new(1),
            &regs,
            b'x',
            1_000_000,
            3,
            || None,
            || {},
        ));
        assert_eq!(regs.polls.get(), 3);
        assert_eq!(regs.writes.get(), 0);
    }

    #[test]
    fn a_live_slow_transmitter_is_not_cut_off_by_the_fallback_count() {
        let regs = Transmitter::new(Some(20));
        let mut now = 0;
        assert!(transmit_byte(
            &Uart16550::new(1),
            &regs,
            b'x',
            1_000_000,
            1,
            || {
                let sample = now;
                now += 25_000;
                Some(sample)
            },
            || {},
        ));
        assert_eq!(regs.polls.get(), 21);
        assert_eq!(regs.writes.get(), 1);
    }

    #[test]
    fn the_shared_budget_accepts_9600_baud_characters() {
        // Eleven 100 us not-ready polls model 1.1 ms per character: just
        // slower than 9600-baud 8N1, and beyond the old 1 ms byte budget.
        let regs = Transmitter::new(Some(11));
        let mut now = 0;
        for &byte in b"9600 baud\n" {
            regs.polls.set(0);
            assert!(transmit_byte(
                &Uart16550::new(1),
                &regs,
                byte,
                DEFAULT_TX_TIMEOUT_NS,
                1,
                || {
                    let sample = now;
                    now += 100_000;
                    Some(sample)
                },
                || {},
            ));
            assert_eq!(regs.polls.get(), 12);
        }
        assert_eq!(regs.writes.get(), b"9600 baud\n".len() as u32);
        assert!(now > DEFAULT_TX_TIMEOUT_NS);
    }

    #[test]
    fn overflowing_deadlines_do_not_expire_before_the_first_wait() {
        let regs = Transmitter::new(Some(2));
        let mut now = u64::MAX - 10;
        assert!(transmit_byte(
            &Uart16550::new(1),
            &regs,
            b'x',
            100,
            0,
            || {
                let sample = now;
                now = now.saturating_add(1);
                Some(sample)
            },
            || {},
        ));
        assert_eq!(regs.writes.get(), 1);
    }

    #[test]
    fn each_unsuccessful_poll_uses_the_callers_pause() {
        let regs = Transmitter::new(Some(3));
        let pauses = Cell::new(0u32);
        assert!(transmit_byte(
            &Uart16550::new(1),
            &regs,
            b'x',
            0,
            8,
            || None,
            || pauses.set(pauses.get() + 1),
        ));
        assert_eq!(pauses.get(), 3);
    }
}
