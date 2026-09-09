//! Qualcomm **GENI serial engine** UART backend.
//!
//! The UART on Snapdragon laptops.  ACPI names it twice on the target
//! machine — SPCR `interface_type = 0x13` and DBG2 subtype `0x0013`, both
//! pointing at the same 4 KiB window (`\_SB.UARD`) — and it is the *only*
//! serial port that firmware designates, so a kernel that cannot drive it has
//! no console on that machine at all.
//!
//! # Why this looks nothing like [`Pl011`](super::Pl011)
//!
//! A 16550 or a PL011 is a holding register plus a status flag: write a byte,
//! poll a bit.  A GENI serial engine is a **command sequencer**.  Transmitting
//! means declaring a transfer length, issuing a `START_TX` opcode to the
//! *master* sequencer, pushing data words into a FIFO, and waiting for the
//! command to report done.  Receiving runs on a second, independent
//! *secondary* sequencer that must be armed with `START_READ` before anything
//! arrives.
//!
//! That maps onto [`Uart`]'s ready/take protocol without a blocking wait
//! inside a write, and deliberately so:
//!
//! - [`tx_ready`](Uart::tx_ready) asks whether the master sequencer is idle —
//!   i.e. whether the *previous* byte's command has retired.
//! - [`tx_byte`](Uart::tx_byte) starts a one-byte transaction and returns
//!   without waiting for it.  The wait is the next `tx_ready`, which is where
//!   the caller already expects to spin.
//!
//! So a byte costs one command, which is slower than a FIFO burst and exactly
//! right for a console: this is the path that has to work when nothing else
//! does, and a transaction per byte cannot desynchronise the sequencer.
//!
//! # What this backend does not do
//!
//! **It does not program the line rate.**  A GENI's baud comes from
//! `GENI_SER_M_CLK_CFG` divided out of the serial engine's parent clock,
//! which lives in the SoC clock controller — and neither ACPI nor a
//! devicetree `stdout-path` describes that tree.  Firmware has already
//! configured this port (both SPCR and DBG2 declare it as the live debug
//! console), so [`init`](Uart::init) inherits that configuration rather than
//! guessing at a divisor.  This is the same "trust what you find when
//! unknown" rule [`LineConfig`] states, arrived at from the other direction:
//! the 16550 and PL011 backends trust firmware when the *table* omits a rate,
//! and this one trusts it because the *hardware* keeps the rate somewhere the
//! table cannot reach.
//!
//! Register offsets and bit positions are transcribed from Linux's
//! `include/linux/soc/qcom/geni-se.h` and `drivers/tty/serial/
//! qcom_geni_serial.c`, whose polled `poll_put_char` / `get_char` paths are
//! the same job this backend does.

use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};

use super::{LineConfig, RegisterAccess, Uart};

// ── Serial-engine registers (geni-se.h) ─────────────────────────────
const SE_GENI_STATUS: usize = 0x40;
const SE_GENI_DMA_MODE_EN: usize = 0x258;
const SE_GENI_M_CMD0: usize = 0x600;
const SE_GENI_M_IRQ_STATUS: usize = 0x610;
const SE_GENI_M_IRQ_CLEAR: usize = 0x618;
const SE_GENI_S_CMD0: usize = 0x630;
const SE_GENI_S_IRQ_STATUS: usize = 0x640;
const SE_GENI_S_IRQ_CLEAR: usize = 0x648;
const SE_GENI_TX_FIFON: usize = 0x700;
const SE_GENI_RX_FIFON: usize = 0x780;
const SE_GENI_RX_FIFO_STATUS: usize = 0x804;

// ── UART-protocol registers (qcom_geni_serial.c) ────────────────────
const SE_UART_TX_TRANS_LEN: usize = 0x270;

// ── Bits ────────────────────────────────────────────────────────────
/// `GENI_STATUS`: the master (TX) sequencer has a command in flight.
const M_GENI_CMD_ACTIVE: u32 = 1 << 0;
/// `GENI_STATUS`: the secondary (RX) sequencer is armed.
const S_GENI_CMD_ACTIVE: u32 = 1 << 12;
/// Opcode field position, shared by both sequencers' `CMD0`.
const OPCODE_SHIFT: u32 = 27;
/// `M_IRQ_STATUS`: the master command retired.
const M_CMD_DONE_EN: u32 = 1 << 0;
/// UART master opcode: begin a transmit of `TX_TRANS_LEN` bytes.
const UART_START_TX: u32 = 0x1;
/// UART secondary opcode: begin receiving.
const UART_START_READ: u32 = 0x1;
/// `RX_FIFO_STATUS`: words currently held.
const RX_FIFO_WC_MASK: u32 = 0x01ff_ffff;
/// `RX_FIFO_STATUS`: this word is the last of a transfer, so it may be
/// partial — `RX_LAST_BYTE_VALID` says how partial.
const RX_LAST: u32 = 1 << 31;
const RX_LAST_BYTE_VALID_MASK: u32 = 0x7 << 28;
const RX_LAST_BYTE_VALID_SHIFT: u32 = 28;
/// A FIFO word carries four bytes, least-significant first.
const BYTES_PER_FIFO_WORD: u8 = 4;

/// Qualcomm GENI serial engine in UART personality.
///
/// Holds one FIFO word of receive state, which the register layout forces: a
/// read pops **four** bytes at once, and [`Uart::rx_byte`] hands back one.
/// Atomics rather than a `Cell` so the backend stays `Sync` like its
/// siblings; the console serialises access anyway, and a torn cache would
/// cost a console byte, not correctness elsewhere.
pub struct Geni {
    /// The undelivered bytes of the last FIFO word, least-significant next.
    rx_word: AtomicU32,
    /// How many of those bytes remain.
    rx_left: AtomicU8,
}

impl Geni {
    /// A backend with an empty receive cache.
    pub const fn new() -> Self {
        Self {
            rx_word: AtomicU32::new(0),
            rx_left: AtomicU8::new(0),
        }
    }

    /// Retire whatever the master sequencer last did, so a fresh command can
    /// be issued.  Acknowledging is required: `M_CMD_DONE` latches, and an
    /// unacknowledged done bit makes the *next* transaction look finished
    /// before it starts.
    fn ack_master(regs: &dyn RegisterAccess) {
        regs.write32(SE_GENI_M_IRQ_CLEAR, M_CMD_DONE_EN);
    }
}

impl Default for Geni {
    fn default() -> Self {
        Self::new()
    }
}

impl Uart for Geni {
    /// Always present because this backend is selected only from an explicit
    /// SPCR/DBG2 or devicetree console description. Qualcomm GENI has no
    /// architectural fallback address to probe, so firmware naming the serial
    /// engine is the presence evidence and speculative register writes would
    /// weaken the guarded-MMIO boundary.
    fn probe_present(&self, _regs: &dyn RegisterAccess) -> bool {
        true
    }

    /// Inherit firmware's line setup, put the engine in FIFO mode, and arm
    /// the receive sequencer.  `cfg` is deliberately unused — see the module
    /// note on why the rate is not ours to program.
    fn init(&self, regs: &dyn RegisterAccess, _cfg: LineConfig) {
        // This backend moves data through the FIFOs, so a firmware that left
        // the engine in DMA mode would have its descriptors driven by writes
        // meant for a FIFO.  Clearing the bit is safe here and only here: the
        // console is quiescent at handover, with no transfer in flight.
        regs.write32(SE_GENI_DMA_MODE_EN, 0);

        // Nothing arrives until the secondary sequencer is running.  Arm it
        // only if firmware has not already — re-issuing `START_READ` over a
        // live command is how a receive path loses its framing.
        if regs.read32(SE_GENI_STATUS) & S_GENI_CMD_ACTIVE == 0 {
            regs.write32(SE_GENI_S_CMD0, UART_START_READ << OPCODE_SHIFT);
        }
    }

    /// `true` when the master sequencer is idle — the previous byte's
    /// command has retired and a new one may be issued.
    fn tx_ready(&self, regs: &dyn RegisterAccess) -> bool {
        regs.read32(SE_GENI_STATUS) & M_GENI_CMD_ACTIVE == 0
    }

    /// Start a one-byte transmit and return.  The caller has confirmed
    /// [`tx_ready`](Uart::tx_ready), so the sequencer is idle; the command
    /// retires on its own and the next `tx_ready` observes it.
    fn tx_byte(&self, regs: &dyn RegisterAccess, byte: u8) {
        Self::ack_master(regs);
        regs.write32(SE_UART_TX_TRANS_LEN, 1);
        regs.write32(SE_GENI_M_CMD0, UART_START_TX << OPCODE_SHIFT);
        regs.write32(SE_GENI_TX_FIFON, u32::from(byte));
    }

    /// `true` when a byte is available — from the cached word if one is part
    /// spent, otherwise by pulling the next word out of the FIFO.
    ///
    /// Pulling here rather than in [`rx_byte`](Uart::rx_byte) is what keeps
    /// that method total: the trait promises a byte once this returned true,
    /// and the FIFO read is the only step that can fail to produce one.
    fn rx_ready(&self, regs: &dyn RegisterAccess) -> bool {
        if self.rx_left.load(Ordering::Relaxed) > 0 {
            return true;
        }
        // Acknowledge both sequencers before reading the FIFO, mirroring the
        // upstream polled path: a latched status bit otherwise re-reports a
        // watermark that has already been consumed.
        let m = regs.read32(SE_GENI_M_IRQ_STATUS);
        regs.write32(SE_GENI_M_IRQ_CLEAR, m);
        let s = regs.read32(SE_GENI_S_IRQ_STATUS);
        regs.write32(SE_GENI_S_IRQ_CLEAR, s);

        let status = regs.read32(SE_GENI_RX_FIFO_STATUS);
        if status & RX_FIFO_WC_MASK == 0 {
            return false;
        }
        // A trailing partial word declares how many of its bytes are real.
        // Zero in that field means a *full* word, not an empty one — the one
        // encoding here that reads backwards.
        let mut bytes = BYTES_PER_FIFO_WORD;
        if status & RX_FIFO_WC_MASK == 1 && status & RX_LAST != 0 {
            let valid = ((status & RX_LAST_BYTE_VALID_MASK) >> RX_LAST_BYTE_VALID_SHIFT) as u8;
            if valid != 0 {
                bytes = valid;
            }
        }
        self.rx_word
            .store(regs.read32(SE_GENI_RX_FIFON), Ordering::Relaxed);
        self.rx_left.store(bytes, Ordering::Relaxed);
        true
    }

    /// Take the next byte out of the cached word, least-significant first.
    fn rx_byte(&self, _regs: &dyn RegisterAccess) -> u8 {
        let word = self.rx_word.load(Ordering::Relaxed);
        let left = self.rx_left.load(Ordering::Relaxed);
        if left == 0 {
            // The caller skipped `rx_ready`; there is no byte to invent.
            return 0;
        }
        self.rx_word.store(word >> 8, Ordering::Relaxed);
        self.rx_left.store(left - 1, Ordering::Relaxed);
        word as u8
    }
}
