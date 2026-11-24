//! The **controller contract** — the one place per-IP variation lives.
//!
//! An impl of [`I2cMaster`] is a complete description of how one silicon
//! master is driven: how it identifies itself, how it is configured, how a
//! transfer is queued, and how it is pumped to completion. The generic
//! engine in [`crate::engine`] never names a register, and adding a second
//! IP adds an impl here, never a branch there.
//!
//! # Sans-IO, on purpose
//!
//! [`start`](I2cMaster::start) queues, [`poll`](I2cMaster::poll) steps, and
//! neither ever waits. All waiting belongs to the caller's [`Pacing`], for
//! one reason the whole arc turns on: **the shipped design parks on an
//! interrupt, and a driver that spins internally can never be made to do
//! that.** A core with a hidden spin loop looks identical to a correct one
//! in a mock and quietly becomes a polled design on metal.

use kore_driver::Regs;

use crate::bus::Xfer;
use crate::error::{AbortDetail, I2cError, Snapshot};

/// A bus speed, as the standard names it.
///
/// Only the two speeds real HID-over-I2C devices declare. High-speed mode
/// needs a master code and a different arbitration sequence; admitting the
/// variant without implementing that would be a silent wrong-timing
/// configuration, so it is absent until a device asks for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speed {
    /// 100 kHz.
    Standard,
    /// 400 kHz — what nearly every touchpad and keyboard declares.
    Fast,
}

impl Speed {
    /// The nominal bus frequency in Hz.
    pub const fn hz(self) -> u32 {
        match self {
            Self::Standard => 100_000,
            Self::Fast => 400_000,
        }
    }

    /// Classify a firmware-declared connection speed.
    ///
    /// Rounds **down** to the nearest mode this seam implements: running a
    /// 400 kHz-rated part at 100 kHz is slow but correct, while running a
    /// 100 kHz part at 400 kHz violates its timing. A device declaring
    /// something above fast mode is clamped to fast rather than refused,
    /// because the alternative is a machine with no pointer at all over a
    /// speed field.
    pub const fn from_hz(hz: u32) -> Self {
        if hz >= 400_000 {
            Self::Fast
        } else {
            Self::Standard
        }
    }
}

/// SCL high/low counts for one speed, as firmware computed them.
///
/// Firmware-supplied counts (`SSCN`/`FMCN` on an ACPI machine) always win
/// over anything this crate derives: they account for the *board's* trace
/// capacitance and rise time, which no formula here can know. The derived
/// path exists for the machine that declares none, and is documented as
/// the fallback it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counts {
    /// SCL high count.
    pub hcnt: u16,
    /// SCL low count.
    pub lcnt: u16,
}

/// Everything a controller needs to be configured for one bus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusConfig {
    /// The bus speed to program.
    pub speed: Speed,
    /// Firmware's timing counts, when it declared them for `speed`.
    pub counts: Option<Counts>,
    /// The controller's input clock in Hz, used only to derive counts
    /// when `counts` is `None`.
    ///
    /// Per-integration data, never a constant in the core: Intel LPSS,
    /// AMD's `AMDI0010`, and a Qualcomm serial engine are clocked
    /// differently, and a core that hardcoded one would mis-time the
    /// other two while looking perfectly correct.
    pub input_clock_hz: u32,
    /// SDA hold time in input-clock ticks, when firmware declared one.
    pub sda_hold: Option<u32>,
    /// Whether the controller should assert its interrupt as a transfer
    /// progresses, so the owning task can **park** instead of spinning.
    ///
    /// Neutral despite sounding like a register: every I2C master has an
    /// interrupt-enable, and the choice is the caller's discipline, not
    /// the IP's. It lives here because it is the difference between the
    /// design this arc ships and a polled one — a core left permanently
    /// masked can only ever be spun on, and that failure is invisible in
    /// a mock (#400 ruling 5).
    pub irq_driven: bool,
}

impl BusConfig {
    /// A fast-mode bus with no firmware timings, interrupt-driven — the
    /// shape a machine that declared nothing gets.
    pub const fn fast(input_clock_hz: u32) -> Self {
        Self {
            speed: Speed::Fast,
            counts: None,
            input_clock_hz,
            sda_hold: None,
            irq_driven: true,
        }
    }
}

/// How far a queued transfer has got.
///
/// Three outcomes, not two: "still working" and "finished" are obvious,
/// and "the controller gave up and said why" is the one that must not
/// collapse into either. Folding an abort into a timeout would discard the
/// abort word — which on a NACK is the difference between "no device at
/// this address" and "the bus is broken".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XferState<A> {
    /// In flight.
    Busy {
        /// Whether this call moved bytes or queued commands.
        ///
        /// Load-bearing, not decoration: the engine paces only when a
        /// poll made **no** progress, and counts those as idle polls.
        /// That is the number the epic's "zero empty polls in steady
        /// state" evidence is read from, so an impl that reported
        /// progress unconditionally would both spin the CPU and forge
        /// the acceptance criterion.
        ///
        /// Progress is bounded — a transfer moves at most
        /// [`MAX_XFER_BYTES`](crate::bus::MAX_XFER_BYTES) in each
        /// direction — so an unpaced progress loop always terminates.
        progressed: bool,
    },
    /// Complete, with `read` bytes placed in the caller's buffer.
    Done {
        /// How many bytes were read.
        read: usize,
    },
    /// The controller aborted. The transfer is over; the caller must
    /// [`recover`](I2cMaster::recover) before starting another.
    Aborted(A, Snapshot),
}

/// Result of restoring a controller after a failed transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryOutcome {
    Clean,
    Cleared,
    Cancelled,
    Aborted,
    Failed(Snapshot),
}

/// First recovery action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryStart {
    Complete(RecoveryOutcome),
    WaitCancel,
    WaitAbort,
}

/// The caller's **bounded wait** between polls.
///
/// The seam takes pacing as a trait rather than a delay closure plus an
/// attempt count so that the production implementation can be a real park
/// on an interrupt route while a test double is a pure countdown. Both are
/// bounded; neither is a spin the core chose for itself.
pub trait Pacing {
    /// Wait one step. Returns `false` when the budget is exhausted, which
    /// the engine turns into a timeout carrying the controller's
    /// registers.
    fn step(&mut self) -> bool;

    /// Re-arm for a new transfer.
    ///
    /// Exists so one pacing instance serves a whole bus rather than being
    /// rebuilt per transfer: a production implementation holds an
    /// interrupt route, and re-deriving that per transfer would be both
    /// wasteful and a place for the route to be silently dropped.
    fn reset(&mut self, transfer_bytes: usize);

    /// Re-arm for one cancellation or abort allowance.
    fn reset_recovery(&mut self) {
        self.reset(crate::bus::MAX_XFER_BYTES);
    }
}

/// A fixed number of steps with a caller-supplied delay — the bring-up
/// and host-test shape.
pub struct Budget<F> {
    steps: u32,
    remaining: u32,
    delay: F,
}

impl<F: FnMut()> Budget<F> {
    /// At most `steps` waits of `delay`, per transfer.
    pub fn new(steps: u32, delay: F) -> Self {
        Self {
            steps,
            remaining: steps,
            delay,
        }
    }
}

impl<F: FnMut()> Pacing for Budget<F> {
    fn step(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        (self.delay)();
        true
    }

    fn reset(&mut self, _transfer_bytes: usize) {
        self.remaining = self.steps;
    }
}

/// A transfer-length/rate-aware polling budget. The wire-time term scales
/// with the request while a 500 ms floor covers firmware scheduling and HID
/// reset latency; every transfer remains capped at Linux's one-second order
/// of magnitude. Recovery receives a separate full one-second allowance.
pub struct RateBudget<F> {
    bus_hz: u32,
    step_ms: u32,
    remaining: u32,
    delay: F,
}

impl<F: FnMut()> RateBudget<F> {
    pub fn new(bus_hz: u32, step_ms: u32, delay: F) -> Self {
        Self {
            bus_hz: bus_hz.max(1),
            step_ms: step_ms.max(1),
            remaining: 0,
            delay,
        }
    }

    pub const fn deadline_ms(bus_hz: u32, bytes: usize) -> u32 {
        let hz = if bus_hz == 0 { 1 } else { bus_hz };
        let bits = bytes.saturating_add(2).saturating_mul(9) as u64;
        let wire_ms = bits.saturating_mul(1_000).saturating_add(hz as u64 - 1) / hz as u64;
        let deadline = 500u64.saturating_add(wire_ms.saturating_mul(4));
        if deadline > 1_000 {
            1_000
        } else {
            deadline as u32
        }
    }

    fn arm_ms(&mut self, ms: u32) {
        self.remaining = ms.saturating_add(self.step_ms - 1) / self.step_ms;
    }
}

impl<F: FnMut()> Pacing for RateBudget<F> {
    fn step(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        (self.delay)();
        true
    }

    fn reset(&mut self, transfer_bytes: usize) {
        self.arm_ms(Self::deadline_ms(self.bus_hz, transfer_bytes));
    }

    fn reset_recovery(&mut self) {
        self.arm_ms(1_000);
    }
}

/// One I2C master controller IP.
///
/// # Contract
///
/// - [`probe`](Self::probe) must be called before anything else, and must
///   refuse a window that does not decode to this IP.
/// - [`configure`](Self::configure) leaves the controller idle and
///   disabled-safe; it may be called repeatedly.
/// - [`start`](Self::start) queues a transfer and returns immediately.
/// - [`poll`](Self::poll) is non-blocking and may be called any number of
///   times; once it returns [`Done`](XferState::Done) or
///   [`Aborted`](XferState::Aborted) the caller must not poll again before
///   the next `start`.
/// - [`recover`](Self::recover) clears a latched abort so the next
///   transfer starts clean.
///
/// Implementations are **not** required to be re-entrant and are driven by
/// one thread at a time — the relay worker that owns the controller.
pub trait I2cMaster {
    /// The register window this IP is driven through.
    ///
    /// Associated rather than a method generic so the engine can name
    /// `M::Regs` in its own signature — the `PageTableEntry::Flags`
    /// idiom: the per-impl encoding rides an associated type and the
    /// engine stays neutral.
    type Regs: Regs + ?Sized;

    /// This IP's decoded abort word.
    type Abort: AbortDetail;

    /// Human name for witness lines (`"designware"`, `"geni"`).
    const NAME: &'static str;

    /// Identity self-check. Fatal, and first: every later register read is
    /// meaningless if the window does not decode to this core.
    fn probe(&mut self, regs: &Self::Regs) -> Result<(), I2cError<Self::Abort>>;

    /// Bring the controller to a configured, idle state.
    fn configure(
        &mut self,
        regs: &Self::Regs,
        cfg: &BusConfig,
    ) -> Result<(), I2cError<Self::Abort>>;

    /// Queue `xfer`. Returns as soon as the controller is running.
    fn start(&mut self, regs: &Self::Regs, xfer: &Xfer<'_>) -> Result<(), I2cError<Self::Abort>>;

    /// Advance the in-flight transfer by whatever the FIFOs allow now,
    /// draining read bytes into `rx`. Never waits.
    ///
    /// `xfer` is the **same transfer** that was passed to
    /// [`start`](Self::start), handed back rather than retained: a
    /// controller whose TX FIFO is shallower than the write payload must
    /// keep queueing bytes as the FIFO drains, and the alternative — the
    /// impl copying the payload into a buffer of its own — would put a
    /// fixed cap on transfer size inside every impl and copy every byte
    /// twice. The engine owns the transfer for its whole life, so lending
    /// it costs nothing.
    ///
    /// Passing a different `xfer` than `start` received is a caller bug
    /// the seam does not defend against; the engine is the only caller.
    fn poll(&mut self, regs: &Self::Regs, xfer: &Xfer<'_>, rx: &mut [u8])
    -> XferState<Self::Abort>;

    /// Clear a latched abort and leave the controller ready.
    fn recover(&mut self, regs: &Self::Regs);

    /// Begin documented recovery. Controllers without an asynchronous
    /// cancellation protocol use their ordinary clear path.
    fn begin_cancel(&mut self, regs: &Self::Regs) -> RecoveryStart {
        self.recover(regs);
        RecoveryStart::Complete(RecoveryOutcome::Cleared)
    }

    fn cancel_complete(&mut self, _regs: &Self::Regs) -> bool {
        true
    }

    /// Escalate a timed-out cancellation. A controller which became idle
    /// between the last cancel poll and this call reports the settled
    /// outcome instead of claiming an abort it never issued.
    fn begin_abort(&mut self, _regs: &Self::Regs) -> RecoveryStart {
        RecoveryStart::Complete(RecoveryOutcome::Cleared)
    }

    fn abort_complete(&mut self, _regs: &Self::Regs) -> bool {
        true
    }

    fn finish_recovery(&mut self, _regs: &Self::Regs) {}

    /// Registers worth photographing when a transfer times out.
    fn timeout_snapshot(&self, regs: &Self::Regs) -> Snapshot;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_rounds_down_to_an_implemented_mode() {
        assert_eq!(Speed::from_hz(400_000), Speed::Fast);
        assert_eq!(
            Speed::from_hz(1_000_000),
            Speed::Fast,
            "clamped, not refused"
        );
        assert_eq!(Speed::from_hz(399_999), Speed::Standard, "down, never up");
        assert_eq!(Speed::from_hz(100_000), Speed::Standard);
        assert_eq!(Speed::from_hz(0), Speed::Standard);
        assert_eq!((Speed::Standard.hz(), Speed::Fast.hz()), (100_000, 400_000));
    }

    #[test]
    fn a_budget_is_bounded_and_counts_its_delays() {
        use core::cell::Cell;
        let n = Cell::new(0u32);
        let mut b = Budget::new(3, || n.set(n.get() + 1));
        assert!(b.step() && b.step() && b.step());
        assert!(!b.step(), "the budget is spent");
        assert!(!b.step(), "and stays spent");
        assert_eq!(n.get(), 3, "exactly one delay per granted step");
        b.reset(0);
        assert!(b.step(), "reset re-arms the whole budget");
        assert_eq!(n.get(), 4);
    }

    #[test]
    fn a_zero_budget_grants_nothing() {
        let mut b = Budget::new(0, || unreachable!("no step was granted"));
        assert!(!b.step());
    }

    #[test]
    fn rate_budget_scales_with_wire_time_and_is_bounded() {
        assert_eq!(RateBudget::<fn()>::deadline_ms(400_000, 9), 504);
        assert_eq!(RateBudget::<fn()>::deadline_ms(400_000, 193), 520);
        assert_eq!(RateBudget::<fn()>::deadline_ms(100_000, 4096), 1_000);
    }
}
