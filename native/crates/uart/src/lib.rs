//! # kore-uart — UART register-layout backends
//!
//! A serial console factors into two orthogonal axes;
//! this crate owns the **register-layout** axis (16550 vs PL011) and the
//! **access** seam ([`RegisterAccess`]) that abstracts the mechanism axis
//! (MMIO load/store vs x86 port `in`/`out` vs a capability-mediated call).
//! A backend's register logic is written once against `RegisterAccess`, so
//! the same 16550 code drives an MMIO part, a port-I/O COM UART, and a
//! ring-3 service that reaches its registers through a capability.
//!
//! Selection is **data, not `cfg(target_arch)`**: [`SerialBackend`] is
//! chosen at runtime from a [`BackendKind`] (mirroring the `IommuDriver`
//! enum-dispatch pattern), so one binary drives whichever UART the machine
//! turns out to have.
//!
//! ## Why this is a crate and not a kernel module
//!
//! The register logic is identical on both sides of the ring boundary; only
//! the *mechanism* differs, and that is exactly what `RegisterAccess`
//! abstracts.  Keeping it here — dependency-free, naming no kernel type and
//! no firmware table — lets the kernel console and a userland serial service
//! share one metal-tested implementation without the service linking against
//! the kernel or reaching for its logger.
//!
//! Firmware glue (decoding a backend and line rate out of an ACPI SPCR
//! descriptor) deliberately stays in the kernel: it is a *discovery*
//! concern, not a register-layout one, and the catalog `Serial` resource
//! already carries the decoded fields to consumers.

#![no_std]

use core::num::NonZeroU32;

mod geni;
mod pl011;
mod tx;
mod uart16550;

pub use geni::Geni;
pub use pl011::Pl011;
pub use tx::{DEFAULT_TX_TIMEOUT_NS, transmit_byte};
pub use uart16550::Uart16550;

/// Volatile access to a UART's register block, abstracting the mechanism
/// (MMIO load/store, x86 port `in`/`out`, or a mediated call) behind one
/// seam.  Offsets are **byte** offsets into the block.
///
/// Reads on a dead or out-of-range access yield `0` and writes are dropped:
/// a debug console must never fault its owner over a UART.  The access
/// object never owns the register window or its claim — it only borrows the
/// means to touch it.
pub trait RegisterAccess {
    fn read8(&self, offset: usize) -> u8;
    fn write8(&self, offset: usize, value: u8);
    fn read32(&self, offset: usize) -> u32;
    fn write32(&self, offset: usize, value: u32);
}

/// Line parameters for (re)programming a UART.
///
/// When `baud` **and** `clock` are both known the backend recomputes the
/// divisor; otherwise it leaves the existing configuration untouched — a
/// firmware-described console was already initialised by firmware, and
/// reprogramming a divisor from a guessed clock would garble it.
#[derive(Clone, Copy, Debug)]
pub struct LineConfig {
    pub baud: Option<NonZeroU32>,
    pub clock: Option<NonZeroU32>,
}

impl LineConfig {
    /// Line parameters that instruct the backend to trust whatever setup it
    /// finds — the "unknown rate" case, spelled explicitly.
    pub const TRUST_EXISTING: Self = Self {
        baud: None,
        clock: None,
    };
}

/// A UART register-layout backend.  Operates over a [`RegisterAccess`];
/// both directions are polling (IRQ-driven RX is a service-level concern,
/// never a register-layout one).
pub trait Uart {
    /// **Is a real UART actually answering at these registers?**
    ///
    /// For the one case where the address is a *guess* rather than a
    /// firmware statement: an architectural default
    /// (`arch::fallback_serial`). A firmware-described console needs no
    /// probe — firmware naming it IS the evidence, and second-guessing
    /// that would silence a console someone is reading.
    ///
    /// The guess must be verified because an unpopulated address is not
    /// inert. Writing a boot log to a port with nothing behind it still
    /// costs a bus cycle per byte, and on a modern chipset an unclaimed
    /// legacy I/O cycle is routed over LPC/eSPI and completes only after a
    /// subtractive-decode timeout — microseconds to tens of microseconds,
    /// *non-posted*, so the issuing core stalls on every one. A hundred
    /// kilobytes of boot log then costs seconds of machine-wide,
    /// unpreemptible stall for output nothing can ever read.
    fn probe_present(&self, regs: &dyn RegisterAccess) -> bool;
    /// (Re)program line parameters; see [`LineConfig`] for the
    /// trust-what-you-find-when-unknown rule.
    fn init(&self, regs: &dyn RegisterAccess, cfg: LineConfig);
    /// `true` when the transmit holding register can accept a byte.
    fn tx_ready(&self, regs: &dyn RegisterAccess) -> bool;
    /// Push one byte into the transmit holding register.  The caller has
    /// confirmed [`tx_ready`](Self::tx_ready).
    fn tx_byte(&self, regs: &dyn RegisterAccess, byte: u8);
    /// `true` when the receive holding register holds a byte.
    fn rx_ready(&self, regs: &dyn RegisterAccess) -> bool;
    /// Take one byte out of the receive holding register.  The caller has
    /// confirmed [`rx_ready`](Self::rx_ready).
    fn rx_byte(&self, regs: &dyn RegisterAccess) -> u8;
}

/// Which register layout a UART presents.  The `interface_type`-driven axis,
/// named independently of how it was discovered — ACPI
/// SPCR on one machine, a device tree or a hard-coded endowment on another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// 16550 / 16450 family — the classic 8-register PC UART.
    Uart16550,
    /// Arm PL011 and the SBSA generic UARTs.
    Pl011,
    /// Qualcomm GENI serial engine in UART personality — a command
    /// sequencer rather than a holding register (see [`Geni`]).  The only
    /// serial port Snapdragon laptop firmware designates.
    Geni,
}

/// The runtime-selected register-layout backend.  Mirrors the
/// `IommuDriver { Vtd, AmdVi }` enum-dispatch idiom: one variant per layout,
/// dispatched by `match`, no `dyn`.
pub enum SerialBackend {
    Uart16550(Uart16550),
    Pl011(Pl011),
    Geni(Geni),
}

impl SerialBackend {
    /// Build the backend for a register layout + MMIO register stride (port
    /// UARTs are byte-spaced → stride `1`).  The *mechanism* axis
    /// ([`RegisterAccess`]) is orthogonal and supplied separately.
    pub const fn new(kind: BackendKind, stride: u8) -> Self {
        match kind {
            BackendKind::Uart16550 => Self::Uart16550(Uart16550::new(stride)),
            BackendKind::Pl011 => Self::Pl011(Pl011),
            BackendKind::Geni => Self::Geni(Geni::new()),
        }
    }
}

impl Uart for SerialBackend {
    #[inline]
    fn probe_present(&self, regs: &dyn RegisterAccess) -> bool {
        match self {
            Self::Uart16550(u) => u.probe_present(regs),
            Self::Pl011(p) => p.probe_present(regs),
            Self::Geni(g) => g.probe_present(regs),
        }
    }

    #[inline]
    fn init(&self, regs: &dyn RegisterAccess, cfg: LineConfig) {
        match self {
            Self::Uart16550(u) => u.init(regs, cfg),
            Self::Pl011(p) => p.init(regs, cfg),
            Self::Geni(g) => g.init(regs, cfg),
        }
    }

    #[inline]
    fn tx_ready(&self, regs: &dyn RegisterAccess) -> bool {
        match self {
            Self::Uart16550(u) => u.tx_ready(regs),
            Self::Pl011(p) => p.tx_ready(regs),
            Self::Geni(g) => g.tx_ready(regs),
        }
    }

    #[inline]
    fn tx_byte(&self, regs: &dyn RegisterAccess, byte: u8) {
        match self {
            Self::Uart16550(u) => u.tx_byte(regs, byte),
            Self::Pl011(p) => p.tx_byte(regs, byte),
            Self::Geni(g) => g.tx_byte(regs, byte),
        }
    }

    #[inline]
    fn rx_ready(&self, regs: &dyn RegisterAccess) -> bool {
        match self {
            Self::Uart16550(u) => u.rx_ready(regs),
            Self::Pl011(p) => p.rx_ready(regs),
            Self::Geni(g) => g.rx_ready(regs),
        }
    }

    #[inline]
    fn rx_byte(&self, regs: &dyn RegisterAccess) -> u8 {
        match self {
            Self::Uart16550(u) => u.rx_byte(regs),
            Self::Pl011(p) => p.rx_byte(regs),
            Self::Geni(g) => g.rx_byte(regs),
        }
    }
}

/// RAM-backed [`RegisterAccess`] for tests: a flat byte buffer with
/// little-endian 32-bit access, exercising backend register logic without
/// hardware (the real MMIO / port impls are QEMU-verified).
///
/// Available to downstream crates under the `mock` feature so a new
/// `RegisterAccess` impl can be tested against the same model.
#[cfg(any(test, feature = "mock"))]
mod mock {
    use super::RegisterAccess;

    const MOCK_LEN: usize = 0x1000;

    pub struct MockRegisters {
        bytes: [core::cell::Cell<u8>; MOCK_LEN],
    }

    impl MockRegisters {
        pub fn new() -> Self {
            Self {
                bytes: core::array::from_fn(|_| core::cell::Cell::new(0)),
            }
        }

        /// Read a byte without going through the trait — the assertion side
        /// of a test, so a broken `read8` cannot mask a broken `write8`.
        pub fn peek(&self, offset: usize) -> u8 {
            self.bytes[offset].get()
        }
    }

    impl Default for MockRegisters {
        fn default() -> Self {
            Self::new()
        }
    }

    impl RegisterAccess for MockRegisters {
        fn read8(&self, offset: usize) -> u8 {
            self.bytes.get(offset).map_or(0, core::cell::Cell::get)
        }

        fn write8(&self, offset: usize, value: u8) {
            if let Some(cell) = self.bytes.get(offset) {
                cell.set(value);
            }
        }

        fn read32(&self, offset: usize) -> u32 {
            let b = |i: usize| self.read8(offset + i) as u32;
            b(0) | (b(1) << 8) | (b(2) << 16) | (b(3) << 24)
        }

        fn write32(&self, offset: usize, value: u32) {
            for i in 0..4 {
                self.write8(offset + i, (value >> (8 * i)) as u8);
            }
        }
    }
}

#[cfg(any(test, feature = "mock"))]
pub use mock::MockRegisters;

#[cfg(test)]
mod tests {
    use super::*;

    // 16550 register indices and bits (mirrored from uart16550.rs).
    const THR: usize = 0;
    const DLM: usize = 1;
    const FCR: usize = 2;
    const LCR: usize = 3;
    const LSR: usize = 5;

    fn known() -> LineConfig {
        // 1.8432 MHz / 115200 ⇒ divisor 1.
        LineConfig {
            baud: NonZeroU32::new(115_200),
            clock: NonZeroU32::new(1_843_200),
        }
    }

    #[test]
    fn uart16550_tx_ready_reads_lsr_thre() {
        let regs = MockRegisters::new();
        let uart = Uart16550::new(1);
        assert!(!uart.tx_ready(&regs));
        regs.write8(LSR, 0x20); // THRE
        assert!(uart.tx_ready(&regs));
    }

    #[test]
    fn uart16550_tx_byte_writes_thr() {
        let regs = MockRegisters::new();
        Uart16550::new(1).tx_byte(&regs, b'K');
        assert_eq!(regs.peek(THR), b'K');
    }

    #[test]
    fn uart16550_init_programs_divisor_8n1_fifo() {
        let regs = MockRegisters::new();
        Uart16550::new(1).init(&regs, known());
        assert_eq!(regs.peek(THR), 1); // DLL = divisor low
        assert_eq!(regs.peek(DLM), 0); // DLM = divisor high
        assert_eq!(regs.peek(LCR), 0x03); // 8N1, DLAB cleared
        assert_eq!(regs.peek(FCR), 0x07); // FIFO enabled + cleared
    }

    #[test]
    fn uart16550_init_trusts_firmware_when_rate_unknown() {
        let regs = MockRegisters::new();
        Uart16550::new(1).init(&regs, LineConfig::TRUST_EXISTING);
        // Nothing touched — the existing setup is preserved.
        assert_eq!(regs.peek(LCR), 0);
        assert_eq!(regs.peek(FCR), 0);
    }

    #[test]
    fn uart16550_dword_stride_uses_spaced_registers() {
        let regs = MockRegisters::new();
        let uart = Uart16550::new(4);
        // LSR lives at index 5 × stride 4 = byte offset 20.
        regs.write32(5 * 4, 0x20);
        assert!(uart.tx_ready(&regs));
        uart.tx_byte(&regs, b'Z');
        assert_eq!(regs.peek(0), b'Z'); // THR at index 0
    }

    #[test]
    fn uart16550_rx_ready_reads_lsr_dr_and_rx_byte_reads_rbr() {
        let regs = MockRegisters::new();
        let uart = Uart16550::new(1);
        assert!(!uart.rx_ready(&regs));
        regs.write8(THR, b'k'); // RBR shares index 0 with THR
        regs.write8(LSR, 0x01); // DR
        assert!(uart.rx_ready(&regs));
        assert_eq!(uart.rx_byte(&regs), b'k');
    }

    #[test]
    fn uart16550_dword_stride_rx_uses_spaced_registers() {
        let regs = MockRegisters::new();
        let uart = Uart16550::new(4);
        regs.write32(0, u32::from(b'q')); // RBR at index 0
        regs.write32(5 * 4, 0x01); // LSR.DR
        assert!(uart.rx_ready(&regs));
        assert_eq!(uart.rx_byte(&regs), b'q');
    }

    #[test]
    fn pl011_tx_ready_reads_fr_txff() {
        let regs = MockRegisters::new();
        let uart = Pl011;
        assert!(uart.tx_ready(&regs)); // FR=0 ⇒ TX not full
        regs.write32(0x18, 0x20); // FR.TXFF
        assert!(!uart.tx_ready(&regs));
    }

    #[test]
    fn pl011_tx_byte_writes_dr() {
        let regs = MockRegisters::new();
        Pl011.tx_byte(&regs, b'Q');
        assert_eq!(regs.read32(0x00), u32::from(b'Q'));
    }

    #[test]
    fn pl011_rx_ready_reads_fr_rxfe_and_rx_byte_reads_dr() {
        let regs = MockRegisters::new();
        let uart = Pl011;
        assert!(uart.rx_ready(&regs)); // FR=0 ⇒ RX not empty
        regs.write32(0x18, 0x10); // FR.RXFE
        assert!(!uart.rx_ready(&regs));
        regs.write32(0x18, 0x00); // a byte arrives
        regs.write32(0x00, u32::from(b'p'));
        assert!(uart.rx_ready(&regs));
        assert_eq!(uart.rx_byte(&regs), b'p');
    }

    #[test]
    fn pl011_init_programs_baud_divisors() {
        let regs = MockRegisters::new();
        // 24 MHz / 115200: div×64 = 96_000_000 / 115_200 = 833 ⇒
        // IBRD = 833 >> 6 = 13, FBRD = 833 & 0x3F = 1.
        let cfg = LineConfig {
            baud: NonZeroU32::new(115_200),
            clock: NonZeroU32::new(24_000_000),
        };
        Pl011.init(&regs, cfg);
        assert_eq!(regs.read32(0x24), 13); // IBRD
        assert_eq!(regs.read32(0x28), 1); // FBRD
        assert_eq!(regs.read32(0x2C), 0x70); // LCR_H: WLEN=8 + FEN
        assert_eq!(regs.read32(0x30), 0x301); // CR: UARTEN|TXE|RXE
    }

    #[test]
    fn pl011_init_trusts_firmware_when_rate_unknown() {
        let regs = MockRegisters::new();
        Pl011.init(&regs, LineConfig::TRUST_EXISTING);
        assert_eq!(regs.read32(0x30), 0); // CR untouched
    }

    #[test]
    fn new_selects_pl011() {
        assert!(matches!(
            SerialBackend::new(BackendKind::Pl011, 1),
            SerialBackend::Pl011(_)
        ));
    }

    #[test]
    fn serial_backend_dispatches_tx() {
        let regs = MockRegisters::new();
        SerialBackend::new(BackendKind::Uart16550, 1).tx_byte(&regs, b'A');
        assert_eq!(regs.peek(0), b'A');

        let regs = MockRegisters::new();
        SerialBackend::new(BackendKind::Pl011, 1).tx_byte(&regs, b'B');
        assert_eq!(regs.read32(0x00), u32::from(b'B'));
    }

    /// A floating bus — every register all-ones, which is what an
    /// unclaimed x86 I/O port reads.
    struct FloatingBus;
    impl RegisterAccess for FloatingBus {
        fn read8(&self, _: usize) -> u8 {
            0xFF
        }
        fn write8(&self, _: usize, _: u8) {}
        fn read32(&self, _: usize) -> u32 {
            0xFFFF_FFFF
        }
        fn write32(&self, _: usize, _: u32) {}
    }

    /// A decode that answers but stores nothing — the other empty shape.
    struct StuckLow;
    impl RegisterAccess for StuckLow {
        fn read8(&self, _: usize) -> u8 {
            0x00
        }
        fn write8(&self, _: usize, _: u8) {}
        fn read32(&self, _: usize) -> u32 {
            0
        }
        fn write32(&self, _: usize, _: u32) {}
    }

    #[test]
    fn an_absent_port_reads_as_transmit_ready_which_is_why_it_needs_a_probe() {
        // The whole reason the probe has to exist: a UART that is not there
        // never announces itself.  All-ones sets `LSR.THRE`, so the byte
        // loop sees "ready", never spins, never times out, never drops —
        // and writes the entire boot log into an unclaimed port.
        let u = SerialBackend::new(BackendKind::Uart16550, 1);
        assert!(
            u.tx_ready(&FloatingBus),
            "an absent port must look READY — if this ever fails, the \
             silent-cost argument for probing has changed",
        );
        assert!(!u.probe_present(&FloatingBus), "all-ones is not a UART");
    }

    #[test]
    fn a_decode_that_stores_nothing_is_also_absent() {
        let u = SerialBackend::new(BackendKind::Uart16550, 1);
        assert!(!u.probe_present(&StuckLow), "all-zeros is not a UART");
    }

    #[test]
    fn a_real_uart_passes_the_probe_with_its_scratch_byte_restored() {
        let regs = MockRegisters::new();
        regs.write8(5, 0x60); // LSR: THRE|TEMT — a plausible idle UART
        regs.write8(7, 0x3C); // SCR: whatever firmware left there
        let u = SerialBackend::new(BackendKind::Uart16550, 1);
        assert!(u.probe_present(&regs));
        assert_eq!(
            regs.peek(7),
            0x3C,
            "the probe must hand a firmware-owned scratch byte back untouched",
        );
    }

    #[test]
    fn a_pl011_is_never_probed_away() {
        // It is only ever adopted from an explicit firmware description, so
        // the probe has no guess to verify and must not veto one.
        assert!(SerialBackend::new(BackendKind::Pl011, 1).probe_present(&FloatingBus));
    }

    #[test]
    fn a_geni_uart_is_never_probed_away() {
        // Like PL011, GENI is selected only from explicit firmware data. A
        // probe must not invent a speculative MMIO transaction or veto that
        // firmware-owned console.
        assert!(SerialBackend::new(BackendKind::Geni, 1).probe_present(&FloatingBus));
    }

    #[test]
    fn serial_backend_dispatches_rx() {
        let regs = MockRegisters::new();
        regs.write8(0, b'a'); // RBR
        regs.write8(5, 0x01); // LSR.DR
        let u = SerialBackend::new(BackendKind::Uart16550, 1);
        assert!(u.rx_ready(&regs));
        assert_eq!(u.rx_byte(&regs), b'a');

        let regs = MockRegisters::new();
        regs.write32(0x00, u32::from(b'b')); // DR
        let p = SerialBackend::new(BackendKind::Pl011, 1);
        assert!(p.rx_ready(&regs));
        assert_eq!(p.rx_byte(&regs), b'b');
    }

    // ── Qualcomm GENI serial engine ─────────────────────────────────
    // Offsets mirrored from geni.rs, which took them from Linux's
    // geni-se.h / qcom_geni_serial.c.
    const G_STATUS: usize = 0x40;
    const G_DMA_MODE_EN: usize = 0x258;
    const G_TX_TRANS_LEN: usize = 0x270;
    const G_M_CMD0: usize = 0x600;
    const G_M_IRQ_CLEAR: usize = 0x618;
    const G_S_CMD0: usize = 0x630;
    const G_TX_FIFO: usize = 0x700;
    const G_RX_FIFO: usize = 0x780;
    const G_RX_FIFO_STATUS: usize = 0x804;
    const G_M_CMD_ACTIVE: u32 = 1 << 0;
    const G_S_CMD_ACTIVE: u32 = 1 << 12;
    const G_START: u32 = 0x1 << 27;

    #[test]
    fn geni_tx_ready_tracks_the_master_sequencer() {
        let regs = MockRegisters::new();
        let uart = Geni::new();
        assert!(uart.tx_ready(&regs));
        regs.write32(G_STATUS, G_M_CMD_ACTIVE);
        assert!(!uart.tx_ready(&regs));
    }

    #[test]
    fn geni_tx_byte_issues_a_one_byte_transaction() {
        let regs = MockRegisters::new();
        Geni::new().tx_byte(&regs, b'K');
        // The latched done bit is acknowledged first, or the next
        // transaction would look finished before it started.
        assert_eq!(regs.read32(G_M_IRQ_CLEAR), 1);
        assert_eq!(regs.read32(G_TX_TRANS_LEN), 1);
        assert_eq!(regs.read32(G_M_CMD0), G_START);
        assert_eq!(regs.read32(G_TX_FIFO), u32::from(b'K'));
    }

    #[test]
    fn geni_init_forces_fifo_mode_and_arms_receive() {
        let regs = MockRegisters::new();
        regs.write32(G_DMA_MODE_EN, 1);
        Geni::new().init(&regs, LineConfig::TRUST_EXISTING);
        assert_eq!(regs.read32(G_DMA_MODE_EN), 0, "FIFO path needs DMA off");
        assert_eq!(regs.read32(G_S_CMD0), G_START, "receive must be armed");
    }

    #[test]
    fn geni_init_leaves_a_live_receive_command_alone() {
        let regs = MockRegisters::new();
        regs.write32(G_STATUS, G_S_CMD_ACTIVE);
        Geni::new().init(&regs, LineConfig::TRUST_EXISTING);
        // Re-issuing START_READ over a live command is how a receive path
        // loses its framing.
        assert_eq!(regs.read32(G_S_CMD0), 0);
    }

    #[test]
    fn geni_init_never_programs_a_rate() {
        let regs = MockRegisters::new();
        // A GENI's baud lives in the SoC clock tree, which no firmware table
        // describes — a known rate must still not be acted on.
        Geni::new().init(&regs, known());
        assert_eq!(regs.read32(G_TX_TRANS_LEN), 0);
        assert_eq!(regs.read32(G_M_CMD0), 0);
    }

    #[test]
    fn geni_rx_is_empty_when_the_fifo_holds_no_word() {
        let regs = MockRegisters::new();
        assert!(!Geni::new().rx_ready(&regs));
    }

    #[test]
    fn geni_rx_delivers_a_full_word_least_significant_first() {
        let regs = MockRegisters::new();
        let uart = Geni::new();
        regs.write32(G_RX_FIFO_STATUS, 2); // two words held, none final
        regs.write32(G_RX_FIFO, 0x4443_4241); // "ABCD"
        assert!(uart.rx_ready(&regs));
        for expect in [b'A', b'B', b'C', b'D'] {
            assert_eq!(uart.rx_byte(&regs), expect);
        }
    }

    #[test]
    fn geni_rx_honours_a_partial_trailing_word() {
        let regs = MockRegisters::new();
        let uart = Geni::new();
        // One word left, it is the last, and only two bytes of it are real.
        regs.write32(G_RX_FIFO_STATUS, 1 | (1 << 31) | (2 << 28));
        regs.write32(G_RX_FIFO, 0xffff_4241);
        assert!(uart.rx_ready(&regs));
        assert_eq!(uart.rx_byte(&regs), b'A');
        assert_eq!(uart.rx_byte(&regs), b'B');
        // The padding bytes are not delivered.  Drain the mock's status
        // first: a flat register file does not decrement its own word count
        // the way the FIFO does, so without this the backend would correctly
        // pull the *same* word again and the assertion would test the mock.
        regs.write32(G_RX_FIFO_STATUS, 0);
        assert!(!uart.rx_ready(&regs));
    }

    #[test]
    fn geni_rx_reads_a_zero_valid_count_as_a_full_word() {
        let regs = MockRegisters::new();
        let uart = Geni::new();
        // The one encoding here that reads backwards: RX_LAST with a valid
        // count of zero means four bytes, not none.
        regs.write32(G_RX_FIFO_STATUS, 1 | (1 << 31));
        regs.write32(G_RX_FIFO, 0x4443_4241);
        assert!(uart.rx_ready(&regs));
        for expect in [b'A', b'B', b'C', b'D'] {
            assert_eq!(uart.rx_byte(&regs), expect);
        }
    }
}
