//! The **completion-wait seam** — one place, shared by every driver, for the
//! "wait for the next hardware completion" loop.
//!
//! Every SQ/CQ driver does the same thing: probe a device-written source
//! (`CompRing`'s phase bit, an SMMUv3 `CMDQ_CONS == PROD` register compare, an
//! NVMe CQE), and if it's empty, wait.
//!
//! This module splits the two concerns:
//!   * [`Completion`] — the driver's per-device, non-blocking *probe* (the part
//!     that stays in the driver: `CompRing::poll`, a register compare, …).
//!   * [`spin_wait`] — the explicitly bounded strategy for synchronous
//!     bring-up and kernel-internal boot tests.
//!
//! There is deliberately no generic async wrapper that merely returns
//! `Pending`: without a route-backed wake registration that shape silently
//! becomes executor polling. The userland completion path must wait on its
//! affine IRQ route and re-probe after a real kernel wake.

use core::task::Poll;

use crate::block::CompletionPacer;
use crate::ring::{CompRing, CqEntry};
use crate::seam::{Dma, Regs};

/// A non-blocking probe of a device-written completion source.
///
/// `poll_once` returns `Ready(v)` when it consumed one completion, `Pending`
/// when the source is still empty — the exact shape of `CompRing::poll` and the
/// register-compare loops drivers already write, lifted to a trait so the wait
/// strategy can be chosen by the caller.
pub trait Completion {
    /// What one completion yields (a CQE, a `()` drain acknowledgement, …).
    type Output;

    /// Probe once, consuming a completion if one is ready.
    fn poll_once(&mut self) -> Poll<Self::Output>;
}

/// **How long a bounded wait may last**, and how that is measured.
///
/// # Why this is not an iteration count
///
/// Every wait here ends in a verdict: `Ack::TimedOut` becomes a refusal that
/// names a device or a bring-up rung as broken. A budget counted in loop
/// iterations cannot support that verdict, because the same count is a
/// different amount of time on every machine it runs on — and the error is
/// unbounded in *both* directions:
///
/// - **Too short.** A budget of 1 000 probes against a controller that answers
///   in 20 ms is generous at 20 µs a probe and hopeless at 200 ns. The faster
///   the core, the sooner a healthy device is declared dead. The bound tightens
///   exactly as hardware improves, which is the opposite of what a timeout is
///   for.
/// - **Too long.** When each probe is a bus cycle nothing claims — an
///   unpopulated I2C segment, a legacy port over LPC/eSPI, a controller behind
///   a subtractive-decode timeout — a probe costs tens of microseconds instead
///   of tens of nanoseconds, and a 500 000-probe budget is minutes, not
///   milliseconds. Nothing preempts it: the core is stalled inside the access,
///   not spinning in a loop a scheduler could interrupt. This is how a
///   not-present device stalls a boot for ten seconds while every budget in the
///   path is nominally "bounded".
///
/// `crate::block::SpinPacer` already states this conclusion for the completion
/// path — *"its budget is retired instructions, not time, so it must not be the
/// input to a liveness decision"* — and names the clocked pacers a verdict has
/// to go through instead. This is that same rule, arriving at the seam every
/// bring-up, reset and power sequence in the tree is built from.
///
/// # The clock is injected
///
/// As a `fn` pointer rather than a direct call, for the reason
/// `crate::host::DeadlinePacer` gives: the host has no clock page, so a budget
/// hard-wired to one could only ever be tested in its clock-unavailable
/// posture, and the comparison deciding whether a device is declared dead would
/// ship unverified. It also keeps this crate's dependency posture — the kernel
/// builds `kore-driver` without `kore-rt`, so the seam cannot name a runtime
/// clock even if it wanted to.
///
/// # The clockless machine
///
/// A spin count remains the honest bound where **no time bound is expressible**
/// — before the boot counter is calibrated, or on a machine with no readable
/// counter at all. So a budget carries both, and which one is live is decided
/// once, at construction, by asking the clock. [`Budget::spins`] is the
/// explicit unclocked form; it is deliberately *not* the default, so every
/// remaining iteration-count budget in the tree is greppable rather than hidden
/// behind a parameter that merely looks like a timeout.
pub struct Budget {
    now_ns: fn() -> Option<u64>,
    /// Absolute monotonic deadline. Meaningful only when `clocked`.
    deadline_ns: u64,
    /// Whether the clock answered when this budget was taken out.
    clocked: bool,
    /// Probes left, on a machine whose clock cannot answer.
    remaining: u32,
}

impl Budget {
    /// Wait at most `budget_ns`, falling back to `spins_if_unclocked` probes on
    /// a machine whose clock cannot answer.
    ///
    /// The clock is read once, here. A clock that answers now is a clock that
    /// answers for the rest of this wait — a calibrated counter does not
    /// de-calibrate — so the fallback is chosen once rather than re-tested per
    /// probe.
    pub fn new(budget_ns: u64, spins_if_unclocked: u32, now_ns: fn() -> Option<u64>) -> Self {
        match now_ns() {
            Some(now) => Self {
                now_ns,
                deadline_ns: now.saturating_add(budget_ns),
                clocked: true,
                remaining: 0,
            },
            None => Self {
                now_ns,
                deadline_ns: 0,
                clocked: false,
                remaining: spins_if_unclocked,
            },
        }
    }

    /// **An explicitly unclocked budget** of `probes`.
    ///
    /// For the two callers that have no clock to offer and conclude nothing
    /// from expiry: the host tests, where a mock completes synchronously and
    /// the budget is never approached, and a bring-up running before any
    /// counter is calibrated.
    pub fn spins(probes: u32) -> Self {
        Self {
            now_ns: || None,
            deadline_ns: 0,
            clocked: false,
            remaining: probes,
        }
    }

    /// Whether this budget is bounded by time rather than by probe count —
    /// i.e. whether a timeout it reports is evidence about the *device*.
    pub fn is_clocked(&self) -> bool {
        self.clocked
    }

    /// **Whether another wait is allowed**, consuming one probe of an
    /// unclocked budget.
    ///
    /// Public because not every bounded wait fits [`spin_wait`]: a loop whose
    /// probe borrows its owner mutably (xHCI draining its event ring through
    /// `&mut self`) cannot be expressed as a [`Completion`] without fighting
    /// the borrow checker, and rewriting it to do so would be a worse change
    /// than letting it ask the budget directly. `while budget.may_wait()` is
    /// then the same loop with a real bound — the shape
    /// `crate::block::CompletionPacer::pace` already has on the completion
    /// side.
    ///
    /// A clocked budget whose clock stops answering mid-wait is spent: the
    /// machine has lost the timebase its bound was expressed in, and
    /// continuing would be an unbounded wait wearing a timeout's clothes.
    pub fn may_wait(&mut self) -> bool {
        if self.clocked {
            return matches!((self.now_ns)(), Some(now) if now < self.deadline_ns);
        }
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        true
    }
}

/// **Spin strategy** (bring-up / the kernel boot-test): probe until `budget` is
/// spent, running `delay` between probes — a ~50 µs spin on metal, a no-op in a
/// mock. Returns `None` if still `Pending` after the budget. This is the
/// `pause`/`delay` loop every driver threads today, in one place.
///
/// Probed once more after the budget, so a device that answers exactly at the
/// deadline is not reported as a timeout.
pub fn spin_wait<C: Completion>(
    c: &mut C,
    mut budget: Budget,
    mut delay: impl FnMut(),
) -> Option<C::Output> {
    while budget.may_wait() {
        if let Poll::Ready(v) = c.poll_once() {
            return Some(v);
        }
        delay();
    }
    match c.poll_once() {
        Poll::Ready(v) => Some(v),
        Poll::Pending => None,
    }
}

/// How a **register handshake** ended.
///
/// Three outcomes rather than two, because "the device never answered" and
/// "the device told us it was broken" call for different handling: a timeout
/// may be worth retrying, a fault never is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ack {
    /// The device acknowledged within the budget.
    Acked,
    /// The device raised its fault condition — the wait was abandoned early
    /// rather than burning the rest of the budget on a controller that has
    /// already given up.
    Faulted,
    /// The budget ran out with no acknowledgement and no fault.
    TimedOut,
}

impl Ack {
    /// Whether the device acknowledged.
    #[inline]
    pub fn is_acked(self) -> bool {
        matches!(self, Ack::Acked)
    }

    /// Collapse to a `Result`, naming the two failures separately.
    #[inline]
    pub fn ok_or<E>(self, timed_out: E, faulted: E) -> Result<(), E> {
        match self {
            Ack::Acked => Ok(()),
            Ack::TimedOut => Err(timed_out),
            Ack::Faulted => Err(faulted),
        }
    }
}

/// A register handshake as a [`Completion`]: poke a control, poll a status
/// until it acknowledges — or until the device says it has failed.
///
/// Private because the free functions below are the whole interface; it exists
/// so the handshake rides the *same* [`spin_wait`] loop the completion rings
/// use rather than adding a second one.
struct RegProbe<'r, R: ?Sized, D, F> {
    regs: &'r R,
    done: D,
    fault: F,
}

impl<R: Regs + ?Sized, D: FnMut(&R) -> bool, F: FnMut(&R) -> bool> Completion
    for RegProbe<'_, R, D, F>
{
    type Output = Ack;

    #[inline]
    fn poll_once(&mut self) -> Poll<Ack> {
        // Fault first: a device that has faulted may also, spuriously, be
        // showing the bit we are waiting for.
        if (self.fault)(self.regs) {
            return Poll::Ready(Ack::Faulted);
        }
        if (self.done)(self.regs) {
            return Poll::Ready(Ack::Acked);
        }
        Poll::Pending
    }
}

/// **Wait for a device to acknowledge**, abandoning the wait early if it
/// raises `fault`.
///
/// This is the loop every bring-up, reset, and power sequence is built from:
/// write a control bit, then poll a status until the hardware agrees. Bounded
/// by `budget` with `delay` between probes (a short spin on metal, a no-op in
/// a mock), and probed once more after the budget — so a device that answers
/// exactly at the deadline is not reported as a timeout.
///
/// `fault` is the part worth having in one place. NVMe's `CSTS.CFS` and xHCI's
/// `USBSTS.HCE` mean the controller has given up; without a way to express
/// them, a driver either spins out the whole budget against dead silicon or
/// hand-rolls the check around its wait helper.
pub fn handshake<R: Regs + ?Sized>(
    regs: &R,
    budget: Budget,
    delay: impl FnMut(),
    done: impl FnMut(&R) -> bool,
    fault: impl FnMut(&R) -> bool,
) -> Ack {
    let mut probe = RegProbe { regs, done, fault };
    spin_wait(&mut probe, budget, delay).unwrap_or(Ack::TimedOut)
}

/// Wait for a register acknowledgement through the driver's common pacing
/// seam.
///
/// Unlike [`handshake`], the budget is chosen by the composed host: a counted
/// spin in deterministic tests, or an elapsed-time deadline on metal.  The
/// final probe after `pace` refuses is load-bearing — an acknowledgement that
/// arrived at the deadline remains an acknowledgement.
pub fn handshake_paced<R: Regs + ?Sized>(
    regs: &R,
    pacer: &mut dyn CompletionPacer,
    done: impl FnMut(&R) -> bool,
    fault: impl FnMut(&R) -> bool,
) -> Ack {
    let mut probe = RegProbe { regs, done, fault };
    loop {
        if let Poll::Ready(ack) = probe.poll_once() {
            return ack;
        }
        if !pacer.pace() {
            return match probe.poll_once() {
                Poll::Ready(ack) => ack,
                Poll::Pending => Ack::TimedOut,
            };
        }
    }
}

/// [`handshake`] for a device with no fault status to consult.
#[inline]
pub fn handshake_until<R: Regs + ?Sized>(
    regs: &R,
    budget: Budget,
    delay: impl FnMut(),
    done: impl FnMut(&R) -> bool,
) -> Ack {
    handshake(regs, budget, delay, done, |_| false)
}

/// [`handshake`] for the common case: one register field reaching one value.
///
/// `expected` is compared against the field *shifted down* to its own zero, so
/// a one-bit flag is `0` or `1` regardless of where it sits in the word.
#[inline]
pub fn handshake_field<R: Regs + ?Sized>(
    regs: &R,
    reg: usize,
    mask: u32,
    expected: u32,
    budget: Budget,
    delay: impl FnMut(),
) -> Ack {
    handshake_until(regs, budget, delay, |r: &R| {
        (r.read32(reg) & mask) >> mask.trailing_zeros() == expected
    })
}

/// Every device-writer completion ring is a [`Completion`]: its phase-bit
/// `poll` is the probe. So NVMe / xHCI / any `CompRing`-based driver adopts the
/// seam for free.
impl<E: CqEntry, B: Dma> Completion for CompRing<E, B> {
    type Output = E;

    #[inline]
    fn poll_once(&mut self) -> Poll<E> {
        match self.poll() {
            Some(entry) => Poll::Ready(entry),
            None => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A completion that is `Pending` for its first `n` probes, then `Ready(42)`.
    struct ReadyAfter {
        n: u32,
    }

    impl Completion for ReadyAfter {
        type Output = u32;
        fn poll_once(&mut self) -> Poll<u32> {
            if self.n == 0 {
                Poll::Ready(42)
            } else {
                self.n -= 1;
                Poll::Pending
            }
        }
    }

    #[test]
    fn spin_wait_drives_to_ready_and_reports_none_on_timeout() {
        let mut c = ReadyAfter { n: 3 };
        let mut ticks = 0u32;
        assert_eq!(
            spin_wait(&mut c, Budget::spins(10), || ticks += 1),
            Some(42)
        );
        assert_eq!(ticks, 3); // three Pending probes → three delays, then Ready

        let mut stuck = ReadyAfter { n: 100 };
        assert_eq!(spin_wait(&mut stuck, Budget::spins(4), || {}), None);
    }

    /// A register file whose status word can be scripted to change after N
    /// probes — the shape of a device acknowledging a control write.
    struct MockRegs {
        words: core::cell::RefCell<alloc::collections::BTreeMap<usize, u32>>,
        reads: core::cell::Cell<u32>,
    }
    impl MockRegs {
        fn new() -> Self {
            Self {
                words: core::cell::RefCell::new(alloc::collections::BTreeMap::new()),
                reads: core::cell::Cell::new(0),
            }
        }
        fn set(&self, off: usize, val: u32) {
            self.words.borrow_mut().insert(off, val);
        }
    }
    impl Regs for MockRegs {
        fn read32(&self, off: usize) -> u32 {
            self.reads.set(self.reads.get() + 1);
            self.words.borrow().get(&off).copied().unwrap_or(0)
        }
        fn write32(&self, off: usize, val: u32) {
            self.set(off, val);
        }
    }

    const STATUS: usize = 0x10;
    const READY: u32 = 1 << 4;
    const FATAL: u32 = 1 << 1;

    /// The ordinary case, and the timeout — plus the post-budget re-probe, so a
    /// device that answers exactly at the deadline still counts as acked.
    #[test]
    fn a_handshake_acks_times_out_and_re_probes_at_the_deadline() {
        let regs = MockRegs::new();
        let ticks = core::cell::Cell::new(0u32);
        let ack = handshake_until(
            &regs,
            Budget::spins(10),
            || {
                ticks.set(ticks.get() + 1);
                if ticks.get() == 3 {
                    regs.set(STATUS, READY);
                }
            },
            |r: &MockRegs| r.read32(STATUS) & READY != 0,
        );
        assert_eq!(ack, Ack::Acked);
        assert_eq!(ticks.get(), 3);

        let stuck = MockRegs::new();
        assert_eq!(
            handshake_until(
                &stuck,
                Budget::spins(4),
                || {},
                |r: &MockRegs| r.read32(STATUS) != 0
            ),
            Ack::TimedOut
        );

        // Ready only on the very last delay: the probe after the budget catches it.
        let late = MockRegs::new();
        let n = core::cell::Cell::new(0u32);
        assert_eq!(
            handshake_until(
                &late,
                Budget::spins(3),
                || {
                    n.set(n.get() + 1);
                    if n.get() == 3 {
                        late.set(STATUS, READY);
                    }
                },
                |r: &MockRegs| r.read32(STATUS) & READY != 0
            ),
            Ack::Acked,
            "an answer at the deadline is not a timeout"
        );
    }

    /// The capability this exists for: a controller that reports itself broken
    /// ends the wait immediately instead of burning the whole budget.
    #[test]
    fn a_fault_abandons_the_wait_early() {
        let regs = MockRegs::new();
        regs.set(STATUS, FATAL);
        let ticks = core::cell::Cell::new(0u32);

        let ack = handshake(
            &regs,
            Budget::spins(1_000_000),
            || ticks.set(ticks.get() + 1),
            |r: &MockRegs| r.read32(STATUS) & READY != 0,
            |r: &MockRegs| r.read32(STATUS) & FATAL != 0,
        );
        assert_eq!(ack, Ack::Faulted);
        assert_eq!(ticks.get(), 0, "faulted on the first probe, no spinning");
    }

    /// A device can be faulted *and* showing the bit we wanted; the fault must
    /// win, or a driver proceeds over broken silicon.
    #[test]
    fn a_fault_outranks_a_simultaneous_acknowledgement() {
        let regs = MockRegs::new();
        regs.set(STATUS, READY | FATAL);
        assert_eq!(
            handshake(
                &regs,
                Budget::spins(4),
                || {},
                |r: &MockRegs| r.read32(STATUS) & READY != 0,
                |r: &MockRegs| r.read32(STATUS) & FATAL != 0,
            ),
            Ack::Faulted
        );
    }

    /// The field form shifts the value down to its own zero, so a flag reads as
    /// 0/1 wherever it sits in the word.
    #[test]
    fn the_field_form_normalizes_the_shift() {
        let regs = MockRegs::new();
        regs.set(STATUS, READY);
        assert_eq!(
            handshake_field(&regs, STATUS, READY, 1, Budget::spins(2), || {}),
            Ack::Acked
        );
        // Waiting for the same field to clear.
        let clear = MockRegs::new();
        assert_eq!(
            handshake_field(&clear, STATUS, READY, 0, Budget::spins(2), || {}),
            Ack::Acked
        );
        // A multi-bit field compares against its shifted-down value.
        let multi = MockRegs::new();
        multi.set(STATUS, 3 << 4);
        assert_eq!(
            handshake_field(&multi, STATUS, 0x7 << 4, 3, Budget::spins(2), || {}),
            Ack::Acked
        );
        assert_eq!(
            handshake_field(&multi, STATUS, 0x7 << 4, 2, Budget::spins(2), || {}),
            Ack::TimedOut
        );
    }

    /// The two failures stay distinguishable when collapsed to a `Result`.
    #[test]
    fn ack_collapses_to_a_result_without_losing_which_failure() {
        assert_eq!(Ack::Acked.ok_or("t", "f"), Ok(()));
        assert_eq!(Ack::TimedOut.ok_or("t", "f"), Err("t"));
        assert_eq!(Ack::Faulted.ok_or("t", "f"), Err("f"));
        assert!(Ack::Acked.is_acked());
        assert!(!Ack::Faulted.is_acked());
    }

    // ── The clocked budget ────────────────────────────────────────────────
    //
    // A test clock must be a bare `fn` pointer — that is the seam's whole
    // point — so its scripted reading lives in a `static`. Each test declares
    // its own inside its body rather than sharing one: the harness runs these
    // in parallel, and a shared clock would make them race.

    use core::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

    fn dead_clock() -> Option<u64> {
        None
    }

    /// **The bug this type exists for**, in the direction that declares working
    /// hardware dead: a device needing 50 000 probes' worth of waiting must not
    /// be abandoned because some iteration count ran out first. A clocked
    /// budget has no iteration count to run out.
    #[test]
    fn a_clocked_budget_outlasts_any_probe_count_while_time_remains() {
        static NOW: AtomicU64 = AtomicU64::new(0);
        fn clock() -> Option<u64> {
            Some(NOW.load(Relaxed))
        }

        let regs = MockRegs::new();
        let probes = core::cell::Cell::new(0u32);

        // 1 ms of budget; each delay advances the clock 1 ns, so time runs out
        // only after ~1 000 000 probes — far past any of the iteration budgets
        // this replaced (i2c-dw's 1 000, i8042's 10 000, xHCI's 500 000).
        let ack = handshake_until(
            &regs,
            Budget::new(1_000_000, 8, clock),
            || {
                probes.set(probes.get() + 1);
                NOW.fetch_add(1, Relaxed);
                if probes.get() == 50_000 {
                    regs.set(STATUS, READY);
                }
            },
            |r: &MockRegs| r.read32(STATUS) & READY != 0,
        );
        assert_eq!(
            ack,
            Ack::Acked,
            "a clocked wait is not cut short by a count"
        );
        assert_eq!(probes.get(), 50_000);
    }

    /// And the direction that stalls a boot: when each probe is an unclaimed
    /// bus cycle costing tens of microseconds, the wait ends on the clock
    /// rather than after a probe count that is now minutes long.
    #[test]
    fn a_clocked_budget_ends_on_time_when_each_probe_is_expensive() {
        static NOW: AtomicU64 = AtomicU64::new(0);
        fn clock() -> Option<u64> {
            Some(NOW.load(Relaxed))
        }

        let regs = MockRegs::new();
        let probes = core::cell::Cell::new(0u32);

        // 25 µs a probe — a subtractive-decode timeout on a port nothing
        // claims — against a 1 ms budget: 40 probes, not the 500 000 the
        // iteration budget would have spent (12.5 seconds of them).
        let ack = handshake_until(
            &regs,
            Budget::new(1_000_000, 500_000, clock),
            || {
                probes.set(probes.get() + 1);
                NOW.fetch_add(25_000, Relaxed);
            },
            |r: &MockRegs| r.read32(STATUS) & READY != 0,
        );
        assert_eq!(ack, Ack::TimedOut);
        assert_eq!(probes.get(), 40, "40 × 25 µs = the 1 ms budget, exactly");
    }

    /// A machine whose clock cannot answer keeps the spin backstop — the one
    /// posture where a probe count is still the honest bound.
    #[test]
    fn an_unreadable_clock_falls_back_to_the_spin_count() {
        static NOW: AtomicU64 = AtomicU64::new(0);
        fn clock() -> Option<u64> {
            Some(NOW.load(Relaxed))
        }

        let budget = Budget::new(1_000_000_000, 6, dead_clock);
        assert!(!budget.is_clocked());

        let regs = MockRegs::new();
        let probes = core::cell::Cell::new(0u32);
        let ack = handshake_until(
            &regs,
            budget,
            || probes.set(probes.get() + 1),
            |r: &MockRegs| r.read32(STATUS) != 0,
        );
        assert_eq!(ack, Ack::TimedOut);
        assert_eq!(probes.get(), 6, "the fallback count, not the nanoseconds");

        // And the clocked case reports itself as such, so a caller can tell
        // whether a timeout it got back is evidence about the device.
        assert!(Budget::new(1_000, 6, clock).is_clocked());
    }

    /// A clock that stops answering mid-wait ends the wait. Continuing would
    /// be an unbounded wait wearing a timeout's clothes — there is no longer a
    /// timebase the bound is expressed in.
    #[test]
    fn a_clock_that_vanishes_mid_wait_spends_the_budget() {
        static LIVE: AtomicBool = AtomicBool::new(true);
        fn clock() -> Option<u64> {
            LIVE.load(Relaxed).then_some(0)
        }

        let regs = MockRegs::new();
        let probes = core::cell::Cell::new(0u32);
        let ack = handshake_until(
            &regs,
            Budget::new(u64::MAX, 0, clock),
            || {
                probes.set(probes.get() + 1);
                if probes.get() == 3 {
                    LIVE.store(false, Relaxed);
                }
            },
            |r: &MockRegs| r.read32(STATUS) != 0,
        );
        assert_eq!(ack, Ack::TimedOut);
        assert_eq!(probes.get(), 3);
    }

    /// A budget of no time at all still probes once — the post-budget re-probe
    /// that keeps "answered at the deadline" from reading as a timeout.
    #[test]
    fn a_spent_budget_still_probes_once() {
        fn clock() -> Option<u64> {
            Some(100)
        }

        let regs = MockRegs::new();
        regs.set(STATUS, READY);
        assert_eq!(
            handshake_until(
                &regs,
                Budget::new(0, 0, clock),
                || panic!("no waiting is allowed on a spent budget"),
                |r: &MockRegs| r.read32(STATUS) & READY != 0,
            ),
            Ack::Acked
        );
    }

    /// The deadline is absolute and computed by addition, so a budget that
    /// overflows the counter must saturate rather than wrap.
    ///
    /// Wrapping would land the deadline *behind* the current reading, and a
    /// wait whose budget is spent before its first probe is the worst of both
    /// worlds: a device is declared dead having never been waited for. The
    /// spin fallback cannot catch it either — the budget is clocked, so there
    /// is no fallback armed.
    #[test]
    fn an_overflowing_deadline_saturates_instead_of_wrapping() {
        static NOW: AtomicU64 = AtomicU64::new(u64::MAX - 8);
        fn clock() -> Option<u64> {
            Some(NOW.load(Relaxed))
        }

        let regs = MockRegs::new();
        let probes = core::cell::Cell::new(0u32);
        let ack = handshake_until(
            &regs,
            Budget::new(u64::MAX, 0, clock),
            || {
                probes.set(probes.get() + 1);
                NOW.fetch_add(1, Relaxed);
                if probes.get() == 3 {
                    regs.set(STATUS, READY);
                }
            },
            |r: &MockRegs| r.read32(STATUS) & READY != 0,
        );
        assert_eq!(ack, Ack::Acked, "a wrapped deadline would refuse to wait");
        assert_eq!(probes.get(), 3);
    }
}
