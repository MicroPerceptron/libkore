//! **`MockDwI2c`** — a DesignWare register file with side effects.
//!
//! Not a transcript recorder. The mock *behaves* like the core: commands
//! written to `IC_DATA_CMD` accumulate into a real command stream, and when
//! that stream reaches its STOP the mock decodes it back into an I2C
//! transaction and hands it to an [`I2cTarget`]. So a framing bug — a
//! missing repeated START, a STOP in the middle, a read queued past the RX
//! FIFO — changes what the *device* sees, which is the only place such a
//! bug shows up on real hardware.
//!
//! Alongside that it raises [`Violation`]s for the sequencing rules the
//! databook states and a register file cannot enforce on its own. Those are
//! the rules that, when broken, produce a controller that works in a mock
//! and hangs on metal.
//!
//! Heapless and `no_std`: fixed capacities, with overflow itself recorded as
//! a violation rather than silently truncating.

use core::cell::RefCell;

use kore_driver::Regs;
use kore_i2c::mock::{I2cTarget, TargetReply};

use crate::integration::{self, Integration};
use crate::regs::{self, cmd, enable, intr, status};

/// Write-accumulator, reply, and RX-FIFO capacity. Well beyond any single
/// HID transfer, including a full report descriptor.
const MAX_READ: usize = 512;
/// Recorded violations before the mock stops recording (and flags that).
const MAX_VIOLATIONS: usize = 16;

/// A databook rule the driver broke.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Violation {
    /// A DesignWare register was touched while the LPSS function was
    /// still in reset. On metal this reads zeroes and the driver
    /// misdiagnoses a live core as a missing one.
    RegisterBeforeLpssReset(usize),
    /// A configuration register was written while the core was enabled.
    /// The write is ignored by real hardware, so the controller keeps its
    /// old configuration with no error anywhere.
    ConfigWhileEnabled(usize),
    /// A command was queued while `IC_TX_ABRT_SOURCE` still stood.
    CommandOverPendingAbort,
    /// A write-then-read stream whose first read command carried no
    /// repeated START — two transactions on the wire instead of one, with
    /// the bus released in between.
    MissingRestart,
    /// A STOP appeared on a command that was not the last.
    StopBeforeEnd(usize),
    /// More read commands were outstanding than the RX FIFO can hold, so
    /// real hardware would drop bytes.
    RxFifoOverflow { outstanding: usize, depth: usize },
    /// More commands were queued than the TX FIFO had room for.
    TxFifoOverflow { queued: usize, depth: usize },
    /// The core was enabled and a command queued without the intervening
    /// `IC_ENABLE_STATUS` read the Bay Trail erratum requires.
    MissingErratumRead,
    /// A transfer began without the enable handshake being observed.
    StartWithoutEnableHandshake,
    /// The mock's own capacity was exceeded.
    Overflow,
}

/// The mock's mutable half.
struct State<T> {
    // ─── register file ───
    con: u32,
    tar: u32,
    ss_hcnt: u32,
    ss_lcnt: u32,
    fs_hcnt: u32,
    fs_lcnt: u32,
    intr_mask: u32,
    rx_tl: u32,
    tx_tl: u32,
    sda_hold: u32,
    enabled: bool,
    raw_intr: u32,
    abrt_source: u32,
    /// Times arming the mask made the empty TX FIFO request an interrupt.
    tx_empty_assertions: usize,

    // ─── LPSS private block ───
    lpss_released: bool,

    // ─── in-flight command stream ───
    /// Write bytes accumulated for the current transaction.
    write_buf: [u8; MAX_READ],
    n_write: usize,
    /// Commands seen in the current stream (for framing checks only —
    /// this core processes each command as it is written, so the TX FIFO
    /// never backs up).
    n_commands: usize,
    /// Whether a STOP has been seen; any command after it is framing
    /// damage.
    stop_seen: bool,
    /// Whether the device transaction has already run for this stream.
    queried: bool,
    /// Whether a read command has been seen in this stream.
    seen_read: bool,
    /// The device's answer, and how much of it has been clocked out.
    reply_buf: [u8; MAX_READ],
    reply_len: usize,
    reply_pos: usize,
    /// The RX FIFO proper: bytes clocked in and not yet popped.
    rx_fifo: [u8; MAX_READ],
    rx_head: usize,
    rx_len: usize,
    /// Read commands issued whose bytes have not been popped.
    rx_outstanding: usize,

    // ─── erratum / handshake bookkeeping ───
    enable_status_read_since_enable: bool,
    enable_handshake_seen: bool,

    // ─── violations ───
    violations: [Option<Violation>; MAX_VIOLATIONS],
    n_violations: usize,

    target: T,
}

/// A DesignWare core, faked, with a device behind it.
pub struct MockDwI2c<T> {
    integration: Integration,
    tx_depth: usize,
    rx_depth: usize,
    comp_type: u32,
    comp_version: u32,
    /// `IC_ENABLE_STATUS` never follows `IC_ENABLE` — a wedged core.
    stuck_enable: bool,
    /// Transactions run, but the STOP is never reported — the shape a
    /// transfer that hangs with the bus held takes.
    swallow_stop: bool,
    state: RefCell<State<T>>,
}

impl<T: I2cTarget> MockDwI2c<T> {
    /// A core with the given integration and device, FIFO depths of 16.
    pub fn new(integration: Integration, target: T) -> Self {
        Self::with_fifos(integration, target, 16, 16)
    }

    /// A core with explicit FIFO depths — the knob the overflow tests
    /// turn, because a shallow FIFO is what exposes multi-poll queueing.
    pub fn with_fifos(
        integration: Integration,
        target: T,
        tx_depth: usize,
        rx_depth: usize,
    ) -> Self {
        Self {
            integration,
            tx_depth,
            rx_depth,
            comp_type: regs::COMP_TYPE_VALUE,
            comp_version: regs::SDA_HOLD_MIN_VERSION,
            stuck_enable: false,
            swallow_stop: false,
            state: RefCell::new(State {
                con: 0,
                tar: 0,
                ss_hcnt: 0,
                ss_lcnt: 0,
                fs_hcnt: 0,
                fs_lcnt: 0,
                intr_mask: 0,
                rx_tl: 0,
                tx_tl: 0,
                sda_hold: 0,
                enabled: false,
                raw_intr: 0,
                abrt_source: 0,
                tx_empty_assertions: 0,
                // A core with no LPSS block behaves as though already
                // released — there is nothing to release.
                lpss_released: !integration.lpss_reset,
                write_buf: [0; MAX_READ],
                n_write: 0,
                n_commands: 0,
                stop_seen: false,
                queried: false,
                seen_read: false,
                reply_buf: [0; MAX_READ],
                reply_len: 0,
                reply_pos: 0,
                rx_fifo: [0; MAX_READ],
                rx_head: 0,
                rx_len: 0,
                rx_outstanding: 0,
                enable_status_read_since_enable: false,
                enable_handshake_seen: false,
                violations: [None; MAX_VIOLATIONS],
                n_violations: 0,
                target,
            }),
        }
    }

    /// Make the identity register read as something else — the
    /// mis-mapped-window case.
    pub fn with_comp_type(mut self, v: u32) -> Self {
        self.comp_type = v;
        self
    }

    /// Make the core predate `IC_SDA_HOLD`.
    pub fn with_comp_version(mut self, v: u32) -> Self {
        self.comp_version = v;
        self
    }

    /// Make `IC_ENABLE_STATUS` never follow `IC_ENABLE` — the core that
    /// accepts the write and never comes up.
    pub fn stuck_enable(mut self) -> Self {
        self.stuck_enable = true;
        self
    }

    /// Run transactions but never report the STOP, so a transfer can
    /// never complete — a controller wedged with the bus held.
    pub fn swallow_stop(mut self) -> Self {
        self.swallow_stop = true;
        self
    }

    /// Every violation recorded so far.
    pub fn violations(&self) -> heapless_vec::Violations {
        let s = self.state.borrow();
        let mut out = heapless_vec::Violations::default();
        for v in s.violations.iter().take(s.n_violations).flatten() {
            out.push(*v);
        }
        out
    }

    /// Whether any violation was recorded.
    pub fn clean(&self) -> bool {
        self.state.borrow().n_violations == 0
    }

    /// The programmed target address.
    pub fn target_address(&self) -> u8 {
        self.state.borrow().tar as u8
    }

    /// The programmed interrupt mask — how the owner would be woken.
    /// The RX FIFO threshold currently programmed — `IC_RX_TL`. The driver
    /// reprograms it per transfer, so this is how a test sees that a
    /// multi-byte read asks for one wake instead of one per byte.
    pub fn rx_threshold(&self) -> u32 {
        self.state.borrow().rx_tl
    }

    pub fn interrupt_mask(&self) -> u32 {
        self.state.borrow().intr_mask
    }

    /// Whether the modeled controller currently asserts its interrupt.
    pub fn interrupt_asserted(&self) -> bool {
        let s = self.state.borrow();
        s.raw_interrupts() & s.intr_mask != 0
    }

    /// Empty-FIFO IRQ assertions while arming, including transient ones
    /// before the initial command fill that would otherwise be invisible
    /// after a synchronous mock transfer finished.
    pub fn tx_empty_assertions(&self) -> usize {
        self.state.borrow().tx_empty_assertions
    }

    /// The programmed control register.
    pub fn control(&self) -> u32 {
        self.state.borrow().con
    }

    /// The programmed fast-mode counts.
    pub fn fast_counts(&self) -> (u32, u32) {
        let s = self.state.borrow();
        (s.fs_hcnt, s.fs_lcnt)
    }

    /// The programmed standard-mode counts.
    pub fn standard_counts(&self) -> (u32, u32) {
        let s = self.state.borrow();
        (s.ss_hcnt, s.ss_lcnt)
    }

    /// The programmed SDA hold time.
    pub fn sda_hold(&self) -> u32 {
        self.state.borrow().sda_hold
    }

    /// Stage an abort: the next transaction aborts with `source`.
    pub fn force_abort(&self, source: u32) {
        let mut s = self.state.borrow_mut();
        s.abrt_source = source;
        s.raw_intr |= intr::TX_ABRT;
    }

    /// Whether the core is currently enabled.
    pub fn enabled(&self) -> bool {
        self.state.borrow().enabled
    }
}

impl<T> State<T> {
    fn raw_interrupts(&self) -> u32 {
        // Commands are processed synchronously, so TX is always empty.
        // TX_EMPTY is a LEVEL: read-to-clear cannot clear it while the
        // enabled FIFO remains empty. Modeling it only as a latched event
        // hides an interrupt storm after the final command/completion.
        self.raw_intr | if self.enabled { intr::TX_EMPTY } else { 0 }
    }

    fn flag(&mut self, v: Violation) {
        if self.n_violations < MAX_VIOLATIONS {
            self.violations[self.n_violations] = Some(v);
            self.n_violations += 1;
        }
    }
}

impl<T: I2cTarget> MockDwI2c<T> {
    /// Ask the device for this stream's answer.
    ///
    /// Runs at the **first read command**, not at the STOP, because that
    /// is when real hardware turns the bus around: the target begins
    /// clocking bytes out as soon as the read is issued. Deferring it to
    /// the STOP would make a transfer deeper than the RX FIFO impossible
    /// to model — the driver would be waiting for bytes that the mock was
    /// waiting for the driver to finish asking for.
    fn query(&self, s: &mut State<T>) {
        if s.queried {
            return;
        }
        s.queried = true;
        let addr = s.tar as u8;
        let mut out = [0u8; MAX_READ];
        let n_write = s.n_write;
        let mut write = [0u8; MAX_READ];
        write[..n_write].copy_from_slice(&s.write_buf[..n_write]);
        match s.target.transact(addr, &write[..n_write], &mut out) {
            TargetReply::Acked(got) => {
                let got = got.min(MAX_READ);
                s.reply_buf[..got].copy_from_slice(&out[..got]);
                s.reply_len = got;
                s.reply_pos = 0;
            }
            TargetReply::NackAddr => {
                s.abrt_source = 1 << 0;
                s.raw_intr |= intr::TX_ABRT;
            }
            TargetReply::NackData(_) => {
                s.abrt_source = 1 << 3;
                s.raw_intr |= intr::TX_ABRT;
            }
        }
    }

    /// Clock one byte into the RX FIFO for a read command just issued.
    fn clock_in(&self, s: &mut State<T>) {
        // A device that has run out of bytes releases SDA: the bus reads
        // as all-ones, which is what a real short read looks like.
        let byte = if s.reply_pos < s.reply_len {
            let b = s.reply_buf[s.reply_pos];
            s.reply_pos += 1;
            b
        } else {
            0xff
        };
        if s.rx_len < MAX_READ {
            let tail = (s.rx_head + s.rx_len) % MAX_READ;
            s.rx_fifo[tail] = byte;
            s.rx_len += 1;
            s.raw_intr |= intr::RX_FULL;
        } else {
            s.raw_intr |= intr::RX_OVER;
        }
    }

    /// Finish the stream at its STOP.
    fn finish(&self, s: &mut State<T>) {
        // A pure write never issued a read command, so the device has not
        // been asked yet.
        self.query(s);
        if !self.swallow_stop {
            s.raw_intr |= intr::STOP_DET;
        }
        s.n_write = 0;
        s.n_commands = 0;
        s.queried = false;
        s.seen_read = false;
        s.stop_seen = false;
        s.reply_len = 0;
        s.reply_pos = 0;
    }
}

impl<T: I2cTarget> Regs for MockDwI2c<T> {
    fn read32(&self, off: usize) -> u32 {
        let mut s = self.state.borrow_mut();

        // Every DesignWare register reads as zero while the function is
        // held in reset — which is exactly why reading one first
        // misdiagnoses the core.
        if !s.lpss_released && off < integration::LPSS_PRIVATE {
            s.flag(Violation::RegisterBeforeLpssReset(off));
            return 0;
        }

        match off {
            regs::IC_COMP_TYPE => self.comp_type,
            regs::IC_COMP_VERSION => self.comp_version,
            regs::IC_COMP_PARAM_1 => {
                (((self.tx_depth - 1) as u32) << 16) | (((self.rx_depth - 1) as u32) << 8)
            }
            regs::IC_CON => s.con,
            regs::IC_TAR => s.tar,
            regs::IC_INTR_MASK => {
                // An integration whose mask does not read back returns
                // zero, so a driver that read-modify-writes the mask
                // silently disarms itself.
                if self.integration.intr_mask_reads_back {
                    s.intr_mask
                } else {
                    0
                }
            }
            regs::IC_ENABLE => u32::from(s.enabled),
            regs::IC_ENABLE_STATUS => {
                s.enable_status_read_since_enable = true;
                s.enable_handshake_seen = true;
                // A stuck core reports "disabled" forever, including
                // while it is being asked to disable — which is what
                // makes both handshakes time out rather than only one.
                if self.stuck_enable {
                    u32::from(!s.enabled)
                } else {
                    u32::from(s.enabled)
                }
            }
            regs::IC_RAW_INTR_STAT => s.raw_interrupts(),
            regs::IC_INTR_STAT => s.raw_interrupts() & s.intr_mask,
            regs::IC_TX_ABRT_SOURCE => s.abrt_source,
            // Each command is processed as it is written, so the TX FIFO
            // never backs up on this model; RX is the FIFO that applies
            // real back-pressure and is modelled properly.
            regs::IC_TXFLR => 0,
            regs::IC_RXFLR => s.rx_len as u32,
            regs::IC_SDA_HOLD => s.sda_hold,
            regs::IC_STATUS => {
                let mut v = status::TFNF | status::TFE;
                if s.rx_len > 0 {
                    v |= status::RFNE;
                }
                if s.rx_len >= self.rx_depth {
                    v |= status::RFF;
                }
                if s.n_commands > 0 {
                    v |= status::ACTIVITY | status::MST_ACTIVITY;
                }
                v
            }
            regs::IC_DATA_CMD => {
                if s.rx_len > 0 {
                    let b = s.rx_fifo[s.rx_head] as u32;
                    s.rx_head = (s.rx_head + 1) % MAX_READ;
                    s.rx_len -= 1;
                    s.rx_outstanding = s.rx_outstanding.saturating_sub(1);
                    if s.rx_len == 0 {
                        s.raw_intr &= !intr::RX_FULL;
                    }
                    b
                } else {
                    // Popping an empty RX FIFO is a driver bug: it means
                    // the driver believed a byte was there.
                    s.raw_intr |= intr::RX_UNDER;
                    0
                }
            }
            // Read-to-clear registers.
            regs::IC_CLR_INTR => {
                s.raw_intr = 0;
                s.abrt_source = 0;
                0
            }
            regs::IC_CLR_TX_ABRT => {
                s.raw_intr &= !intr::TX_ABRT;
                s.abrt_source = 0;
                0
            }
            regs::IC_CLR_STOP_DET => {
                s.raw_intr &= !intr::STOP_DET;
                0
            }
            _ => 0,
        }
    }

    fn write32(&self, off: usize, val: u32) {
        let mut s = self.state.borrow_mut();

        // The LPSS private block is reachable while the function is in
        // reset — releasing it is the whole point.
        if off == integration::LPSS_RESETS {
            s.lpss_released = val & (integration::LPSS_RESETS_FUNC | integration::LPSS_RESETS_IDMA)
                == (integration::LPSS_RESETS_FUNC | integration::LPSS_RESETS_IDMA);
            return;
        }
        if !s.lpss_released {
            s.flag(Violation::RegisterBeforeLpssReset(off));
            return;
        }

        // Configuration registers are writable only while disabled. Real
        // hardware drops the write; the mock drops it too, and says so.
        let config_reg = matches!(
            off,
            regs::IC_CON
                | regs::IC_TAR
                | regs::IC_SS_SCL_HCNT
                | regs::IC_SS_SCL_LCNT
                | regs::IC_FS_SCL_HCNT
                | regs::IC_FS_SCL_LCNT
                | regs::IC_SDA_HOLD
                | regs::IC_RX_TL
                | regs::IC_TX_TL
        );
        if config_reg && s.enabled {
            s.flag(Violation::ConfigWhileEnabled(off));
            return;
        }

        match off {
            regs::IC_CON => s.con = val,
            regs::IC_TAR => s.tar = val,
            regs::IC_SS_SCL_HCNT => s.ss_hcnt = val,
            regs::IC_SS_SCL_LCNT => s.ss_lcnt = val,
            regs::IC_FS_SCL_HCNT => s.fs_hcnt = val,
            regs::IC_FS_SCL_LCNT => s.fs_lcnt = val,
            regs::IC_SDA_HOLD => s.sda_hold = val,
            regs::IC_RX_TL => s.rx_tl = val,
            regs::IC_TX_TL => s.tx_tl = val,
            regs::IC_INTR_MASK => {
                s.intr_mask = val;
                if s.raw_interrupts() & val & intr::TX_EMPTY != 0 {
                    s.tx_empty_assertions += 1;
                }
            }
            regs::IC_ENABLE => {
                let want = val & enable::ENABLE != 0;
                if want && !s.enabled {
                    s.enable_status_read_since_enable = false;
                    s.enable_handshake_seen = false;
                }
                if !want {
                    // Disabling mid-stream discards the queued commands
                    // and the FIFOs, as the core does.
                    s.n_commands = 0;
                    s.n_write = 0;
                    s.rx_outstanding = 0;
                    s.rx_len = 0;
                    s.rx_head = 0;
                    s.queried = false;
                    s.seen_read = false;
                    s.stop_seen = false;
                    s.reply_len = 0;
                    s.reply_pos = 0;
                }
                s.enabled = want;
            }
            regs::IC_DATA_CMD => {
                if s.abrt_source != 0 {
                    s.flag(Violation::CommandOverPendingAbort);
                }
                if !s.enable_handshake_seen {
                    s.flag(Violation::StartWithoutEnableHandshake);
                }
                if !s.enable_status_read_since_enable {
                    s.flag(Violation::MissingErratumRead);
                }
                if s.stop_seen {
                    let at = s.n_commands;
                    s.flag(Violation::StopBeforeEnd(at));
                }
                s.n_commands += 1;

                if val & cmd::READ != 0 {
                    // The first read after a write owes a repeated START;
                    // without it the core releases the bus and starts a
                    // fresh transaction, which the device sees as two.
                    if !s.seen_read {
                        s.seen_read = true;
                        if s.n_write > 0 && val & cmd::RESTART == 0 {
                            s.flag(Violation::MissingRestart);
                        }
                        self.query(&mut s);
                    }
                    s.rx_outstanding += 1;
                    let outstanding = s.rx_outstanding;
                    if outstanding > self.rx_depth {
                        s.flag(Violation::RxFifoOverflow {
                            outstanding,
                            depth: self.rx_depth,
                        });
                    }
                    if s.abrt_source == 0 {
                        self.clock_in(&mut s);
                    }
                } else {
                    if s.seen_read {
                        // A write after reads began is a shape this seam
                        // cannot express.
                        s.flag(Violation::MissingRestart);
                    }
                    let n = s.n_write;
                    if n < MAX_READ {
                        s.write_buf[n] = (val & cmd::DATA_MASK) as u8;
                        s.n_write = n + 1;
                    } else {
                        s.flag(Violation::Overflow);
                    }
                }

                if val & cmd::STOP != 0 {
                    s.stop_seen = true;
                    if s.abrt_source == 0 {
                        self.finish(&mut s);
                    }
                }
            }
            _ => {}
        }
    }
}

/// A tiny fixed-capacity vector, so `violations()` can be returned by
/// value from a `no_std` crate with no `alloc`.
pub mod heapless_vec {
    use super::{MAX_VIOLATIONS, Violation};

    /// Up to [`MAX_VIOLATIONS`] recorded violations.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Violations {
        items: [Option<Violation>; MAX_VIOLATIONS],
        len: usize,
    }

    impl Violations {
        /// Append, saturating.
        pub fn push(&mut self, v: Violation) {
            if self.len < MAX_VIOLATIONS {
                self.items[self.len] = Some(v);
                self.len += 1;
            }
        }

        /// How many were recorded.
        pub fn len(&self) -> usize {
            self.len
        }

        /// Whether none were recorded.
        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// Whether `v` was recorded.
        pub fn contains(&self, v: Violation) -> bool {
            self.items.iter().take(self.len).flatten().any(|&x| x == v)
        }

        /// The first recorded violation.
        pub fn first(&self) -> Option<Violation> {
            self.items.first().copied().flatten()
        }
    }
}
