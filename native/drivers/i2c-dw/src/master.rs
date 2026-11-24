//! The DesignWare transfer state machine — sans-IO, and the whole of what
//! is DesignWare-specific about driving an I2C bus in kore.

use core::marker::PhantomData;

use kore_driver::{Ack, Budget, Regs, handshake_field};
use kore_i2c::{BusConfig, Counts, I2cError, I2cMaster, Snapshot, Speed, Xfer, XferState};

use crate::abort::DwAbort;
use crate::integration::{self, Integration};
use crate::regs::{self, cmd, con, enable, intr, status};
use crate::timing;

/// How long an enable/disable handshake gets before the core is declared
/// stuck. The core answers in a handful of input clocks; 10 ms is orders of
/// magnitude beyond that, and bounded because an unbounded wait on a
/// register that never changes is a wedged boot with no witness.
///
/// Wall clock rather than a probe count, because the count could not mean
/// what it said in either direction: 1 000 probes is under a microsecond of
/// real waiting on a fast core reading a live register, and minutes when the
/// LPSS core is power-gated and each read is a stalled bus cycle. Neither is
/// the millisecond-scale bound the comment above promised.
const HANDSHAKE_BUDGET_NS: u64 = 10_000_000;

/// The clockless fallback for [`HANDSHAKE_BUDGET_NS`] — what this driver
/// shipped with, and all that is expressible before a counter is calibrated.
const HANDSHAKE_ATTEMPTS: u32 = 1_000;

/// The in-flight transfer's bookkeeping.
///
/// Replaced wholesale by `start`, never merged: carrying a half-updated
/// transfer into the next one is the shape of bug where a second read
/// returns the first read's tail.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Pending {
    write_len: usize,
    /// Write bytes pushed into the TX FIFO.
    write_queued: usize,
    read_len: usize,
    /// Read *commands* pushed into the TX FIFO.
    read_queued: usize,
    /// Read *bytes* popped from the RX FIFO into the caller's buffer.
    read_done: usize,
    /// Read commands queued whose bytes have not been popped — the RX
    /// FIFO's occupancy as this driver believes it. Queuing beyond the
    /// FIFO depth overflows it and loses bytes silently, so this is
    /// checked before every read command rather than trusted.
    rx_outstanding: usize,
    /// Whether the read phase still owes a repeated START.
    restart_owed: bool,
    /// Whether the transfer has been observed to reach its STOP.
    stopped: bool,
}

/// A Synopsys DesignWare I2C master.
///
/// Generic over its register window rather than holding `&dyn Regs`: the
/// window is fixed for the life of the driver, so type instantiation is
/// the seam's own selection mechanism and costs nothing at runtime.
pub struct DwMaster<R: Regs + ?Sized> {
    integration: Integration,
    /// TX FIFO depth, from `IC_COMP_PARAM_1`.
    tx_depth: usize,
    /// RX FIFO depth, from `IC_COMP_PARAM_1`.
    rx_depth: usize,
    /// Whether this core implements `IC_SDA_HOLD`.
    sda_hold_supported: bool,
    /// Whether the bus was configured to raise interrupts, so a transfer
    /// arms the mask that lets its owner park. Recorded at `configure`
    /// because it is a property of the bus, and re-applied on every
    /// `start` because reprogramming `IC_TAR` requires masking first.
    irq_driven: bool,
    /// Caller-selected relaxation between handshake probes.
    delay: fn(),
    /// This runtime's monotonic nanosecond reading, or `None` where no
    /// counter is calibrated — what every handshake's [`Budget`] is built
    /// from. Defaults to unclocked so a host test needs no clock, and the
    /// production caller installs the real one with
    /// [`with_clock`](Self::with_clock).
    clock: fn() -> Option<u64>,
    pending: Pending,
    _regs: PhantomData<fn(&R)>,
}

impl<R: Regs + ?Sized> DwMaster<R> {
    /// A master for one integration. Nothing is touched until `probe`.
    pub fn new(integration: Integration, delay: fn()) -> Self {
        Self {
            integration,
            // Conservative until `probe` reads the real depths: a depth of
            // one is always safe (it just queues slowly), whereas an
            // optimistic guess overflows a FIFO on the first transfer.
            tx_depth: 1,
            rx_depth: 1,
            sda_hold_supported: false,
            irq_driven: false,
            delay,
            clock: || None,
            pending: Pending::default(),
            _regs: PhantomData,
        }
    }

    /// Install this runtime's clock, so handshakes are bounded in time rather
    /// than in probes.
    ///
    /// `kore_driver::host::clock_now_ns` in a service. Left unset, every
    /// handshake falls back to [`HANDSHAKE_ATTEMPTS`] — correct for a mock,
    /// which answers on the first probe and never approaches either bound.
    pub fn with_clock(mut self, clock: fn() -> Option<u64>) -> Self {
        self.clock = clock;
        self
    }

    /// The integration this master drives.
    pub fn integration(&self) -> &Integration {
        &self.integration
    }

    /// The FIFO depths the core reported, as `(tx, rx)`.
    pub fn fifo_depths(&self) -> (usize, usize) {
        (self.tx_depth, self.rx_depth)
    }

    /// Whether transfers are configured to raise progress interrupts.
    pub fn is_irq_driven(&self) -> bool {
        self.irq_driven
    }

    /// Disable the controller and wait for it to agree.
    ///
    /// Configuration registers are writable only while disabled, so every
    /// caller of this is about to reconfigure; a half-honoured disable
    /// means those writes land nowhere and the controller keeps running
    /// its old configuration with no error anywhere.
    fn disable(&self, regs: &R) -> Result<(), I2cError<DwAbort>> {
        regs.write32(regs::IC_ENABLE, 0);
        self.await_enable_state(regs, 0)
    }

    /// Enable the controller and wait for it to agree.
    fn enable(&self, regs: &R) -> Result<(), I2cError<DwAbort>> {
        regs.write32(regs::IC_ENABLE, enable::ENABLE);
        self.await_enable_state(regs, 1)
    }

    fn await_enable_state(&self, regs: &R, want: u32) -> Result<(), I2cError<DwAbort>> {
        match handshake_field(
            regs,
            regs::IC_ENABLE_STATUS,
            enable::ENABLE,
            want,
            Budget::new(HANDSHAKE_BUDGET_NS, HANDSHAKE_ATTEMPTS, self.clock),
            self.delay,
        ) {
            Ack::Acked => Ok(()),
            // The snapshot pairs enable-status with raw-interrupt: the
            // first says what the core thinks its state is, the second
            // whether it is stuck mid-transfer, and one photograph of the
            // pair is usually the whole diagnosis.
            _ => Err(I2cError::EnableHandshake(Snapshot(
                regs.read32(regs::IC_ENABLE_STATUS),
                regs.read32(regs::IC_RAW_INTR_STAT),
            ))),
        }
    }

    /// The next command word to queue, or `None` when everything is out.
    ///
    /// Pure given the bookkeeping, so the STOP/RESTART placement — the
    /// part that silently produces a *different transaction* when wrong —
    /// is testable without a controller.
    fn next_command(p: &Pending) -> Option<u32> {
        if p.write_queued < p.write_len {
            // Write phase. STOP rides the last byte only when no read
            // follows; otherwise the read phase owns the STOP.
            let last = p.write_queued + 1 == p.write_len && p.read_len == 0;
            return Some(if last { cmd::STOP } else { 0 });
        }
        if p.read_queued < p.read_len {
            let mut c = cmd::READ;
            // The repeated START belongs to the *first* read command, and
            // only when a write preceded it: a pure read's opening START
            // is implied by enabling with IC_TAR already set.
            if p.restart_owed && p.read_queued == 0 {
                c |= cmd::RESTART;
            }
            if p.read_queued + 1 == p.read_len {
                c |= cmd::STOP;
            }
            return Some(c);
        }
        None
    }

    /// Whether the next command is a read (and so needs RX FIFO room).
    fn next_is_read(p: &Pending) -> bool {
        p.write_queued >= p.write_len && p.read_queued < p.read_len
    }

    /// Queue write bytes while the TX FIFO has room, then read commands.
    fn pump(&mut self, regs: &R, write: &[u8]) -> bool {
        let mut progressed = false;
        while self.pending.write_queued < self.pending.write_len
            && regs.read32(regs::IC_STATUS) & status::TFNF != 0
        {
            let Some(base) = Self::next_command(&self.pending) else {
                break;
            };
            // Indexing is bounded by `write_len`, which `start` set from
            // this same slice; `get` rather than `[]` so a caller that
            // hands `poll` a different transfer than it handed `start`
            // stalls instead of panicking inside a driver.
            let Some(&byte) = write.get(self.pending.write_queued) else {
                break;
            };
            regs.write32(regs::IC_DATA_CMD, base | byte as u32);
            self.pending.write_queued += 1;
            progressed = true;
        }
        progressed | self.pump_reads(regs)
    }

    /// Queue read commands, bounded by **both** FIFOs.
    ///
    /// The RX bound is the one that matters and the one that is easy to
    /// forget: a read command consumes a TX slot now and an RX slot when
    /// its byte arrives, so queueing against TX room alone overflows RX
    /// on any read longer than the RX FIFO — a bug that appears only on
    /// the largest reports, which is to say only on real hardware.
    fn pump_reads(&mut self, regs: &R) -> bool {
        let mut progressed = false;
        while Self::next_is_read(&self.pending)
            && self.pending.rx_outstanding < self.rx_depth
            && regs.read32(regs::IC_STATUS) & status::TFNF != 0
        {
            let Some(c) = Self::next_command(&self.pending) else {
                break;
            };
            regs.write32(regs::IC_DATA_CMD, c);
            self.pending.read_queued += 1;
            self.pending.rx_outstanding += 1;
            progressed = true;
        }
        progressed
    }

    /// Request only interrupts that can advance this transfer. TX_EMPTY
    /// is a level, not a completion event: once the last command is queued
    /// it stays asserted until the next transfer. It is also unproductive
    /// while read issuance is waiting for RX credits; RX_FULL wakes that
    /// case and `poll` drains RX before issuing more commands.
    fn arm_progress_interrupts(&self, regs: &R) {
        let mut mask = 0;
        if self.irq_driven {
            mask = intr::MASTER_MASK & !intr::TX_EMPTY;
            if self.pending.write_queued < self.pending.write_len
                || (self.pending.read_queued < self.pending.read_len
                    && self.pending.rx_outstanding < self.rx_depth)
            {
                mask |= intr::TX_EMPTY;
            }
        }
        // Write the complete mask: some integrations do not read it back,
        // so a read-modify-write would silently lose RX/STOP/abort wakeups.
        regs.write32(regs::IC_INTR_MASK, mask);
    }
}

impl<R: Regs + ?Sized> I2cMaster for DwMaster<R> {
    type Regs = R;
    type Abort = DwAbort;
    const NAME: &'static str = "designware";

    fn probe(&mut self, regs: &R) -> Result<(), I2cError<DwAbort>> {
        // The LPSS reset comes first, before *any* DesignWare register is
        // read: on an integration that has one, the core does not answer
        // until its function reset is released, so an identity check
        // performed first would read zeroes and report the wrong fault
        // — "this is not a DesignWare core" instead of "it is still in
        // reset".
        if self.integration.lpss_reset {
            regs.write32(integration::LPSS_RESETS, 0);
            regs.write32(
                integration::LPSS_RESETS,
                integration::LPSS_RESETS_FUNC | integration::LPSS_RESETS_IDMA,
            );
        }

        let found = regs.read32(regs::IC_COMP_TYPE);
        if found != regs::COMP_TYPE_VALUE {
            return Err(I2cError::Identity {
                found,
                expected: regs::COMP_TYPE_VALUE,
            });
        }

        // The window is now known to be DesignWare. Quiet any firmware
        // interrupt state before more setup; a failed identity must not
        // write a DesignWare offset into a foreign device.
        regs.write32(regs::IC_INTR_MASK, 0);

        // FIFO depths are encoded as depth-1, so the +1 is the databook's,
        // not a fudge. Getting this wrong by one overflows the RX FIFO on
        // the largest transfer only — the classic bug that survives every
        // small test.
        let param1 = regs.read32(regs::IC_COMP_PARAM_1);
        self.rx_depth = (((param1 >> 8) & 0xff) as usize) + 1;
        self.tx_depth = (((param1 >> 16) & 0xff) as usize) + 1;

        self.sda_hold_supported = regs.read32(regs::IC_COMP_VERSION) >= regs::SDA_HOLD_MIN_VERSION;
        Ok(())
    }

    fn configure(&mut self, regs: &R, cfg: &BusConfig) -> Result<(), I2cError<DwAbort>> {
        // Even a refused disable handshake must leave the source quiet.
        regs.write32(regs::IC_INTR_MASK, 0);
        self.disable(regs)?;

        // Firmware's counts win over anything derived; firmware's clock
        // wins over the integration's. Both are data, and neither is a
        // constant inside the state machine.
        let counts = cfg.counts.unwrap_or_else(|| {
            let hz = if cfg.input_clock_hz != 0 {
                cfg.input_clock_hz
            } else {
                self.integration.input_clock_hz
            };
            timing::derive(cfg.speed, hz)
        });
        let Counts { hcnt, lcnt } = counts;

        let speed_bits = match cfg.speed {
            Speed::Standard => {
                regs.write32(regs::IC_SS_SCL_HCNT, hcnt as u32);
                regs.write32(regs::IC_SS_SCL_LCNT, lcnt as u32);
                con::SPEED_STD
            }
            Speed::Fast => {
                regs.write32(regs::IC_FS_SCL_HCNT, hcnt as u32);
                regs.write32(regs::IC_FS_SCL_LCNT, lcnt as u32);
                con::SPEED_FAST
            }
        };

        // Writing IC_SDA_HOLD on a core that predates it writes a
        // reserved address, so the version gate is a correctness
        // condition, not an optimisation.
        if let Some(hold) = cfg.sda_hold
            && self.sda_hold_supported
        {
            regs.write32(regs::IC_SDA_HOLD, hold);
        }

        regs.write32(
            regs::IC_CON,
            con::MASTER | con::RESTART_EN | con::SLAVE_DISABLE | speed_bits,
        );

        // TX_TL=0 signals an EMPTY TX FIFO, not merely one free slot. Refill
        // then, and mask TX_EMPTY whenever there is no command we can issue
        // (including RX-credit stalls). RX_FULL and STOP_DET report progress.
        //
        // RX_TL is only seeded here; `arm_progress_interrupts` reprograms it
        // per transfer from the reads actually in flight, so a multi-byte read
        // costs one wake rather than one per byte.
        regs.write32(regs::IC_RX_TL, 0);
        regs.write32(regs::IC_TX_TL, 0);

        self.irq_driven = cfg.irq_driven;

        // Leave the controller masked and disabled: `start` owns turning
        // it on, and a configured-but-live controller between transfers is
        // a window in which a stale command could issue.
        regs.write32(regs::IC_INTR_MASK, 0);
        Ok(())
    }

    fn start(&mut self, regs: &R, xfer: &Xfer<'_>) -> Result<(), I2cError<DwAbort>> {
        // Mask before any refusal or reconfiguration, not just before
        // enabling: an inherited abort or failed handshake is terminal
        // for this request and must not leave a level interrupt asserted.
        regs.write32(regs::IC_INTR_MASK, 0);
        // A latched abort from a previous transfer would be read as this
        // one's cause. Refuse rather than clear silently: a driver that
        // never noticed its own abort would otherwise mis-attribute every
        // failure after it.
        if regs.read32(regs::IC_TX_ABRT_SOURCE) != 0 {
            return Err(I2cError::AbortPending);
        }

        self.disable(regs)?;
        regs.write32(regs::IC_TAR, xfer.addr as u32);
        // Wake once per FIFO-load of reads, not once per byte.
        //
        // `RX_TL = 0` raises `RX_FULL` the moment a single byte lands, so a
        // 32-byte I2C-HID report costs an interrupt per byte. Measured on a
        // Yoga's touchpad relay that was ~10 controller completion interrupts
        // for every GPIO-ready signal (`status input i2c`, the `c/g/w` row).
        //
        // Programmed here because `IC_RX_TL` is a configuration register: a
        // write while the core is enabled is *silently ignored* by real
        // hardware, so this has to ride the disabled window `IC_TAR` already
        // uses rather than be re-tuned mid-transfer.
        //
        // Clamped to the FIFO because a threshold the FIFO cannot reach would
        // never raise `RX_FULL` at all; a read longer than the FIFO then wakes
        // once per FIFO-load, which is exactly the drain cadence `pump_reads`
        // is credit-limited to anyway.
        //
        // Safe to raise because `poll` drains on `RFNE`, not on the threshold,
        // and `STOP_DET` stays armed as the terminal wake — so a short, NAK'd
        // or sub-threshold read still has its tail collected.
        let threshold = xfer.read_len.min(self.rx_depth).saturating_sub(1);
        regs.write32(regs::IC_RX_TL, threshold as u32);
        self.enable(regs)?;

        // The Bay Trail erratum: one read of IC_ENABLE_STATUS after
        // enabling, or the register can latch stale. Harmless elsewhere,
        // required there, and placed after the enable handshake because
        // that is where the erratum places it.
        let _ = regs.read32(regs::IC_ENABLE_STATUS);
        // Read-to-clear every stale condition before arming.
        let _ = regs.read32(regs::IC_CLR_INTR);

        self.pending = Pending {
            write_len: xfer.write.len(),
            read_len: xfer.read_len,
            restart_owed: !xfer.write.is_empty() && xfer.read_len > 0,
            ..Pending::default()
        };

        // Queue what the FIFOs will take now; `poll` queues the rest as
        // they drain. Writing the whole stream here would overflow the TX
        // FIFO on any transfer longer than its depth.
        let _ = self.pump(regs, xfer.write);
        // Arm only AFTER the initial fill. Otherwise an empty enabled
        // FIFO can interrupt before the first command is even issued.
        self.arm_progress_interrupts(regs);
        Ok(())
    }

    fn poll(&mut self, regs: &R, xfer: &Xfer<'_>, rx: &mut [u8]) -> XferState<DwAbort> {
        // Service level conditions with the source masked. Raw status
        // still reports all progress; only a continuing transfer rearms
        // below. Done/Aborted therefore return quiet even without an
        // engine recovery call (the successful path has no such call).
        regs.write32(regs::IC_INTR_MASK, 0);
        let raw = regs.read32(regs::IC_RAW_INTR_STAT);
        if raw & intr::TX_ABRT != 0 {
            let src = regs.read32(regs::IC_TX_ABRT_SOURCE);
            return XferState::Aborted(
                DwAbort::decode(src),
                Snapshot(src, regs.read32(regs::IC_STATUS)),
            );
        }

        let mut progressed = false;

        // Drain RX before queueing more: the RX FIFO is the scarcer
        // resource, and freeing a slot is what lets the next read command
        // be queued at all.
        while self.pending.read_done < self.pending.read_len
            && regs.read32(regs::IC_STATUS) & status::RFNE != 0
        {
            let byte = (regs.read32(regs::IC_DATA_CMD) & cmd::DATA_MASK) as u8;
            // `get_mut` rather than `[]`: the engine validates the buffer
            // against `read_len` before `start`, but `poll` is a public
            // method and an out-of-bounds write here would be a panic
            // inside a driver rather than a refusal at the seam.
            let Some(slot) = rx.get_mut(self.pending.read_done) else {
                break;
            };
            *slot = byte;
            self.pending.read_done += 1;
            self.pending.rx_outstanding = self.pending.rx_outstanding.saturating_sub(1);
            progressed = true;
        }

        // Keep queueing: a write payload deeper than the TX FIFO is only
        // fully queued across several polls, which is why `xfer` is lent
        // back to this method.
        progressed |= self.pump(regs, xfer.write);

        if raw & intr::STOP_DET != 0 && !self.pending.stopped {
            self.pending.stopped = true;
            let _ = regs.read32(regs::IC_CLR_STOP_DET);
            progressed = true;
        }

        // Done needs all three: the STOP seen (the bus is released), every
        // read byte collected, and every write byte queued. Dropping any
        // one of them completes a transfer that is still on the wire.
        if self.pending.stopped
            && self.pending.read_done == self.pending.read_len
            && self.pending.write_queued == self.pending.write_len
        {
            return XferState::Done {
                read: self.pending.read_done,
            };
        }

        self.arm_progress_interrupts(regs);
        XferState::Busy { progressed }
    }

    fn recover(&mut self, regs: &R) {
        regs.write32(regs::IC_INTR_MASK, 0);
        // Read-to-clear the abort, then park the controller disabled so
        // the next `start` reconfigures from a known state rather than
        // inheriting a half-finished command stream.
        let _ = regs.read32(regs::IC_CLR_TX_ABRT);
        let _ = regs.read32(regs::IC_CLR_INTR);
        regs.write32(regs::IC_ENABLE, 0);
        self.pending = Pending::default();
    }

    fn timeout_snapshot(&self, regs: &R) -> Snapshot {
        Snapshot(regs.read32(regs::IC_STATUS), regs.read32(regs::IC_RXFLR))
    }
}
