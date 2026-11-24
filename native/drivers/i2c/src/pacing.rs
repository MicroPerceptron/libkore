//! Elapsed-time pacing over a caller-owned clock and bounded wait.

use crate::master::{Pacing, RateBudget};

#[derive(Clone, Copy)]
struct Armed {
    deadline_ns: u64,
    last_ns: u64,
}

/// An absolute transfer deadline, independent of how many interrupts fire.
///
/// The clock and wait belong to the caller: no runtime or interrupt authority
/// is introduced here. `wait_until` receives an absolute monotonic deadline
/// and must return after an event, that deadline, or refusal. `true` grants
/// the engine another poll, including when a wait's deadline expired; `false`
/// spends this allowance immediately (for example on revoke or park refusal).
///
/// Only [`Pacing::reset`] and [`Pacing::reset_recovery`] arm a new deadline.
/// Repeated wakes never extend it. Missing or regressing clock readings fail
/// closed until the next reset; equal samples are valid because several real
/// interrupts may arrive within one clock tick. As with other elapsed-time
/// budgets, the clock must advance to establish a time bound: this is not a
/// frozen-clock detector and does not impose an arbitrary interrupt quota.
pub struct DeadlinePacing<N, W> {
    bus_hz: u32,
    poll_interval_ns: u64,
    now_ns: N,
    wait_until: W,
    armed: Option<Armed>,
}

impl<N: FnMut() -> Option<u64>, W: FnMut(u64) -> bool> DeadlinePacing<N, W> {
    /// Construct an unarmed pacer; the engine resets it for each transfer.
    /// A zero polling interval becomes one nanosecond, not a zero-length slice.
    pub fn new(bus_hz: u32, poll_interval_ns: u64, now_ns: N, wait_until: W) -> Self {
        Self {
            bus_hz,
            poll_interval_ns: poll_interval_ns.max(1),
            now_ns,
            wait_until,
            armed: None,
        }
    }

    fn arm_ms(&mut self, duration_ms: u32) {
        self.armed = None;
        let Some(now) = (self.now_ns)() else { return };
        let Some(deadline_ns) = now.checked_add(u64::from(duration_ms) * 1_000_000) else {
            return;
        };
        self.armed = Some(Armed {
            deadline_ns,
            last_ns: now,
        });
    }
}

impl<N: FnMut() -> Option<u64>, W: FnMut(u64) -> bool> Pacing for DeadlinePacing<N, W> {
    fn step(&mut self) -> bool {
        let Some(mut armed) = self.armed else {
            return false;
        };
        let Some(now) = (self.now_ns)() else {
            self.armed = None;
            return false;
        };
        if now < armed.last_ns || now >= armed.deadline_ns {
            self.armed = None;
            return false;
        }
        // Clip the duration before adding, so even a maximal interval cannot
        // overflow or move the wait beyond this transfer's fixed deadline.
        let until = now + self.poll_interval_ns.min(armed.deadline_ns - now);
        armed.last_ns = now;
        self.armed = Some(armed);
        if !(self.wait_until)(until) {
            self.armed = None;
            return false;
        }
        // Poll after every successful wait, even at the final deadline. The
        // engine polls before calling step again, so expiry cannot discard a
        // completion that arrived without a delivered interrupt.
        true
    }

    fn reset(&mut self, transfer_bytes: usize) {
        self.arm_ms(RateBudget::<fn()>::deadline_ms(self.bus_hz, transfer_bytes));
    }

    fn reset_recovery(&mut self) {
        self.arm_ms(1_000);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::{Cell, RefCell};

    const MS: u64 = 1_000_000;

    #[test]
    fn more_than_eight_wakes_do_not_spend_or_extend_the_deadline() {
        let now = Cell::new(0);
        let waits = Cell::new(0);
        let mut pacing = DeadlinePacing::new(
            400_000,
            20 * MS,
            || Some(now.get()),
            |_| {
                waits.set(waits.get() + 1);
                now.set(now.get() + 1_000);
                true
            },
        );
        pacing.reset(193);
        for _ in 0..64 {
            assert!(pacing.step());
        }
        assert_eq!(waits.get(), 64);
        now.set(520 * MS);
        assert!(!pacing.step(), "wakes did not move the original deadline");
        assert!(!pacing.step());
        assert_eq!(waits.get(), 64);
    }

    #[test]
    fn equal_clock_samples_allow_many_genuine_wakes() {
        let now = Cell::new(10);
        let waits = Cell::new(0);
        let mut pacing = DeadlinePacing::new(
            400_000,
            20 * MS,
            || Some(now.get()),
            |_| {
                waits.set(waits.get() + 1);
                true
            },
        );
        pacing.reset(193);
        for _ in 0..64 {
            assert!(pacing.step());
        }
        assert_eq!(waits.get(), 64);
        now.set(10 + 520 * MS);
        assert!(!pacing.step());
    }

    #[test]
    fn no_progress_expires_after_clipped_waits_and_one_final_poll() {
        let now = Cell::new(0);
        let waits = Cell::new(0usize);
        let deadlines = RefCell::new([0u64; 26]);
        let mut pacing = DeadlinePacing::new(
            400_000,
            20 * MS,
            || Some(now.get()),
            |until| {
                deadlines.borrow_mut()[waits.get()] = until;
                waits.set(waits.get() + 1);
                now.set(until);
                true // A deadline wake still grants the engine its next poll.
            },
        );
        pacing.reset(9);
        for _ in 0..26 {
            assert!(pacing.step());
        }
        assert_eq!(now.get(), 504 * MS);
        assert!(!pacing.step());
        assert!(!pacing.step());
        assert_eq!(waits.get(), 26);
        let deadlines = deadlines.borrow();
        for (i, deadline) in deadlines[..25].iter().enumerate() {
            assert_eq!(*deadline, (i as u64 + 1) * 20 * MS);
        }
        assert_eq!(deadlines[25], 504 * MS);
    }

    #[test]
    fn absent_clock_at_reset_stays_spent_until_reset() {
        let now = Cell::new(None);
        let waits = Cell::new(0);
        let mut pacing = DeadlinePacing::new(
            400_000,
            20 * MS,
            || now.get(),
            |_| {
                waits.set(waits.get() + 1);
                true
            },
        );
        assert!(!pacing.step(), "new pacers are unarmed");
        pacing.reset(9);
        now.set(Some(0));
        assert!(!pacing.step());
        assert_eq!(waits.get(), 0);
        pacing.reset(9);
        assert!(pacing.step());
        assert_eq!(waits.get(), 1);
    }

    #[test]
    fn disappearing_or_regressing_clock_spends_the_allowance() {
        for broken in [None, Some(9)] {
            let now = Cell::new(Some(10));
            let waits = Cell::new(0);
            let mut pacing = DeadlinePacing::new(
                400_000,
                20 * MS,
                || now.get(),
                |_| {
                    waits.set(waits.get() + 1);
                    true
                },
            );
            pacing.reset(9);
            assert!(pacing.step());
            now.set(broken);
            assert!(!pacing.step());
            now.set(Some(11));
            assert!(!pacing.step());
            assert_eq!(waits.get(), 1);
            pacing.reset(9);
            assert!(pacing.step());
            assert_eq!(waits.get(), 2);
        }
    }

    #[test]
    fn refused_wait_spends_the_allowance_until_reset() {
        let permitted = Cell::new(false);
        let waits = Cell::new(0);
        let mut pacing = DeadlinePacing::new(
            400_000,
            20 * MS,
            || Some(0),
            |_| {
                waits.set(waits.get() + 1);
                permitted.get()
            },
        );
        pacing.reset(9);
        assert!(!pacing.step());
        permitted.set(true);
        assert!(!pacing.step());
        assert_eq!(waits.get(), 1);
        pacing.reset(9);
        assert!(pacing.step());
        assert_eq!(waits.get(), 2);
    }

    #[test]
    fn transfer_and_recovery_resets_each_arm_their_own_deadline() {
        let now = Cell::new(0);
        let deadline = Cell::new(0);
        let mut pacing = DeadlinePacing::new(
            400_000,
            u64::MAX,
            || Some(now.get()),
            |until| {
                deadline.set(until);
                true
            },
        );
        pacing.reset(9);
        assert!(pacing.step());
        assert_eq!(deadline.get(), 504 * MS);
        now.set(504 * MS);
        assert!(!pacing.step());
        now.set(3_000 * MS);
        pacing.reset_recovery();
        assert!(pacing.step());
        assert_eq!(deadline.get(), 4_000 * MS);
        now.set(4_000 * MS);
        assert!(!pacing.step());
        now.set(5_000 * MS);
        pacing.reset(193);
        assert!(pacing.step());
        assert_eq!(deadline.get(), 5_520 * MS);
    }

    #[test]
    fn zero_interval_and_overflow_have_explicit_bounded_behavior() {
        let until = Cell::new(0);
        let mut tiny = DeadlinePacing::new(
            400_000,
            0,
            || Some(100),
            |deadline| {
                until.set(deadline);
                true
            },
        );
        tiny.reset(9);
        assert!(tiny.step());
        assert_eq!(until.get(), 101);

        let now = Cell::new(u64::MAX - 1);
        let waits = Cell::new(0);
        let mut pacing = DeadlinePacing::new(
            400_000,
            u64::MAX,
            || Some(now.get()),
            |deadline| {
                waits.set(waits.get() + 1);
                until.set(deadline);
                true
            },
        );
        pacing.reset(193);
        assert!(!pacing.step(), "an overflowing deadline must not wrap");
        now.set(0);
        assert!(!pacing.step(), "overflow stays spent until reset");
        assert_eq!(waits.get(), 0);
        now.set(u64::MAX - 520 * MS);
        pacing.reset(193);
        assert!(pacing.step(), "the final representable deadline is valid");
        assert_eq!(until.get(), u64::MAX);
        now.set(u64::MAX);
        assert!(!pacing.step());
    }
}
