//! 16550 / 16450-family UART backend.
//!
//! The classic 8-register PC UART, reachable over MMIO or x86 port I/O.
//! Registers are accessed at `index × stride`: byte-spaced (`stride == 1`)
//! on port UARTs and most MMIO parts; dword-spaced (`stride == 4`) on the
//! MMIO variants that advertise 32-bit access, where each 8-bit register
//! occupies a 32-bit slot and is touched 32-bit-wide (value in the low
//! byte).

use super::{LineConfig, RegisterAccess, Uart};

// Register indices (pre-stride).  Index 0/1 are the data/IER registers,
// or the baud divisor low/high when LCR.DLAB is set.
const REG_RBR: usize = 0; // receive buffer (read; DLL when DLAB)
const REG_THR: usize = 0; // transmit holding (write; DLL when DLAB)
const REG_IER: usize = 1; // interrupt enable (DLM when DLAB)
const REG_FCR: usize = 2; // FIFO control (write)
const REG_LCR: usize = 3; // line control
const REG_LSR: usize = 5; // line status
const REG_SCR: usize = 7; // scratch (16550A and later; absent on a 16450)

const LCR_8N1: u8 = 0x03; // 8 data bits, no parity, 1 stop; clears DLAB
const LCR_DLAB: u8 = 0x80; // divisor-latch access
const FCR_ENABLE_CLEAR: u8 = 0x07; // enable FIFOs + clear RX/TX
const LSR_DR: u8 = 0x01; // data ready (receive holding register non-empty)
const LSR_THRE: u8 = 0x20; // transmit holding register empty

/// 16550/16450 UART with a known MMIO register `stride`.
pub struct Uart16550 {
    stride: u8,
}

impl Uart16550 {
    #[inline]
    pub const fn new(stride: u8) -> Self {
        Self { stride }
    }

    #[inline]
    fn offset(&self, index: usize) -> usize {
        index * self.stride as usize
    }

    #[inline]
    fn read_reg(&self, regs: &dyn RegisterAccess, index: usize) -> u8 {
        let off = self.offset(index);
        // Dword-spaced parts must be accessed 32-bit-wide; the register
        // value is the low byte.
        if self.stride >= 4 {
            regs.read32(off) as u8
        } else {
            regs.read8(off)
        }
    }

    #[inline]
    fn write_reg(&self, regs: &dyn RegisterAccess, index: usize, value: u8) {
        let off = self.offset(index);
        if self.stride >= 4 {
            regs.write32(off, u32::from(value));
        } else {
            regs.write8(off, value);
        }
    }
}

impl Uart for Uart16550 {
    /// Two independent tests, because the two ways an address can be empty
    /// look nothing alike.
    ///
    /// **Floating bus.** An unclaimed x86 I/O port reads all-ones, and
    /// `LSR == 0xFF` is not a state a real UART is ever in: it asserts
    /// overrun *and* parity *and* framing *and* break *and* FIFO-error at
    /// the same instant as both "transmitter empty" bits. Note this is
    /// exactly the reading that makes an absent port look *healthy* to
    /// [`tx_ready`](Self::tx_ready) — `LSR.THRE` is set in all-ones — so
    /// the byte loop never blocks, never times out, and cheerfully writes
    /// the whole boot log into a hole.
    ///
    /// **Scratch round-trip.** The 16550A's scratch register holds any
    /// byte and means nothing, which is what makes it the standard
    /// presence test (Linux's `serial8250` autoconfig uses it). Two
    /// complementary patterns separate a real register from both failure
    /// shapes at once: all-ones (nothing decodes) and all-zeros (a decode
    /// that answers but stores nothing). The prior value is restored, so a
    /// UART firmware is mid-use of is handed back untouched.
    ///
    /// A genuine 16450 has no scratch register and is therefore reported
    /// absent. That is the right trade for the only caller: this decides
    /// whether to *adopt a guess*, and a machine old enough to ship a
    /// 16450 is not one this kernel boots.
    fn probe_present(&self, regs: &dyn RegisterAccess) -> bool {
        if self.read_reg(regs, REG_LSR) == 0xFF {
            return false;
        }
        let saved = self.read_reg(regs, REG_SCR);
        let present = [0xA5u8, 0x5A].into_iter().all(|pattern| {
            self.write_reg(regs, REG_SCR, pattern);
            self.read_reg(regs, REG_SCR) == pattern
        });
        self.write_reg(regs, REG_SCR, saved);
        present
    }

    fn init(&self, regs: &dyn RegisterAccess, cfg: LineConfig) {
        let (Some(baud), Some(clock)) = (cfg.baud, cfg.clock) else {
            // Unknown rate ⇒ trust firmware's existing configuration.
            return;
        };
        // 16× oversampling; clamp to a valid non-zero divisor.
        let divisor = (clock.get() / (16 * baud.get())).max(1);
        self.write_reg(regs, REG_LCR, LCR_DLAB);
        self.write_reg(regs, REG_THR, (divisor & 0xFF) as u8); // DLL
        self.write_reg(regs, REG_IER, ((divisor >> 8) & 0xFF) as u8); // DLM
        self.write_reg(regs, REG_LCR, LCR_8N1); // 8N1, clears DLAB
        self.write_reg(regs, REG_FCR, FCR_ENABLE_CLEAR);
    }

    fn tx_ready(&self, regs: &dyn RegisterAccess) -> bool {
        self.read_reg(regs, REG_LSR) & LSR_THRE != 0
    }

    fn tx_byte(&self, regs: &dyn RegisterAccess, byte: u8) {
        self.write_reg(regs, REG_THR, byte);
    }

    fn rx_ready(&self, regs: &dyn RegisterAccess) -> bool {
        self.read_reg(regs, REG_LSR) & LSR_DR != 0
    }

    fn rx_byte(&self, regs: &dyn RegisterAccess) -> u8 {
        self.read_reg(regs, REG_RBR)
    }
}
