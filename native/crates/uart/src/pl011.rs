//! Arm PL011 (and SBSA generic) UART backend.
//!
//! The PrimeCell PL011 — the UART on Arm `virt` platforms.  Registers are
//! at fixed byte offsets and accessed 32-bit-wide.  The SBSA generic UART
//! (`ArmSBSAGeneric` / `…32bit`) is a register-compatible subset, so it
//! shares this backend.

use super::{LineConfig, RegisterAccess, Uart};

const DR: usize = 0x00; // data
const FR: usize = 0x18; // flag
const IBRD: usize = 0x24; // integer baud-rate divisor
const FBRD: usize = 0x28; // fractional baud-rate divisor
const LCR_H: usize = 0x2C; // line control
const CR: usize = 0x30; // control

const FR_RXFE: u32 = 1 << 4; // receive FIFO empty
const FR_TXFF: u32 = 1 << 5; // transmit FIFO full
const LCRH_WLEN8: u32 = 0b11 << 5; // 8 data bits
const LCRH_FEN: u32 = 1 << 4; // enable FIFOs
const CR_UARTEN: u32 = 1 << 0;
const CR_TXE: u32 = 1 << 8;
const CR_RXE: u32 = 1 << 9;

/// Arm PL011 / SBSA generic UART.  Stateless — all state lives in the
/// register block reached via [`RegisterAccess`].
pub struct Pl011;

impl Uart for Pl011 {
    /// Always present, and that is a decision rather than a stub.
    ///
    /// A PL011 is only ever adopted from an explicit firmware description
    /// — SPCR, or a devicetree `/chosen stdout-path`. Neither aarch64 nor
    /// riscv64 has an architectural UART address to guess at, so both
    /// `fallback_serial` implementations return `None` and this probe has
    /// no guess to verify. Firmware naming the console is the evidence;
    /// answering anything but `true` here could only silence a console
    /// firmware promised is there.
    fn probe_present(&self, _regs: &dyn RegisterAccess) -> bool {
        true
    }

    fn init(&self, regs: &dyn RegisterAccess, cfg: LineConfig) {
        let (Some(baud), Some(clock)) = (cfg.baud, cfg.clock) else {
            // Unknown rate ⇒ trust firmware's existing configuration.
            return;
        };
        // BAUDDIV is 22.6 fixed-point: divisor = UARTCLK / (16 × baud).
        // `divisor × 64 = clock × 4 / baud`, so IBRD/FBRD fall straight
        // out.  `saturating_mul` guards an implausibly high clock.
        let div_x64 = clock.get().saturating_mul(4) / baud.get();
        regs.write32(CR, 0); // disable while reprogramming
        regs.write32(IBRD, div_x64 >> 6);
        regs.write32(FBRD, div_x64 & 0x3F);
        regs.write32(LCR_H, LCRH_WLEN8 | LCRH_FEN);
        regs.write32(CR, CR_UARTEN | CR_TXE | CR_RXE);
    }

    fn tx_ready(&self, regs: &dyn RegisterAccess) -> bool {
        regs.read32(FR) & FR_TXFF == 0
    }

    fn tx_byte(&self, regs: &dyn RegisterAccess, byte: u8) {
        regs.write32(DR, u32::from(byte));
    }

    fn rx_ready(&self, regs: &dyn RegisterAccess) -> bool {
        regs.read32(FR) & FR_RXFE == 0
    }

    fn rx_byte(&self, regs: &dyn RegisterAccess) -> u8 {
        regs.read32(DR) as u8
    }
}
