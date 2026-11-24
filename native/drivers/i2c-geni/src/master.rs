use core::marker::PhantomData;

use kore_driver::Regs;
use kore_i2c::{
    BusConfig, I2cError, I2cMaster, RecoveryOutcome, RecoveryStart, Snapshot, Speed, Xfer,
    XferState,
};

use crate::{GeniAbort, regs};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    Write,
    Read,
}

#[derive(Clone, Copy, Debug, Default)]
struct Pending {
    tx_done: usize,
    rx_done: usize,
    write_len: usize,
    read_len: usize,
    phase: Phase,
    /// `M_CMD_DONE` is edge-like software state even though the controller
    /// exposes it through a write-one-to-clear status bit.  A long RX can
    /// report command completion before software has drained every FIFO
    /// batch; remember the completion after acknowledging the IRQ so a later
    /// drain can finish the transfer.
    command_done: bool,
}

/// Qualcomm GENI serial engine in I2C personality, FIFO/PIO mode.
pub struct GeniMaster<R: Regs + ?Sized> {
    tx_depth: usize,
    rx_depth: usize,
    pending: Pending,
    _regs: PhantomData<fn(&R)>,
}

impl<R: Regs + ?Sized> GeniMaster<R> {
    pub const fn new() -> Self {
        Self {
            tx_depth: 1,
            rx_depth: 1,
            pending: Pending {
                tx_done: 0,
                rx_done: 0,
                write_len: 0,
                read_len: 0,
                phase: Phase::Idle,
                command_done: false,
            },
            _regs: PhantomData,
        }
    }

    fn fifo_depth(word: u32) -> Option<usize> {
        // Linux selects the 6- or 8-bit field from the QUP wrapper version.
        // This capability maps only the 0x4000-byte child serial element, not
        // the wrapper. Refuse a depth which needs the unavailable version
        // instead of reading outside authority or guessing. The ThinkBook's
        // depth (16) has the same encoding under both versions.
        if word & regs::FIFO_DEPTH_EXTENDED_MASK != 0 {
            return None;
        }
        let depth = ((word & regs::FIFO_DEPTH_MASK) >> regs::FIFO_DEPTH_SHIFT) as usize;
        (depth != 0).then_some(depth)
    }

    fn packing() -> (u32, u32) {
        // Linux's I2C path calls geni_se_config_packing(8, 4, true): four
        // 8-bit protocol words per 32-bit FIFO entry, each starting at its
        // most-significant bit and walking toward bit zero.  The direction
        // bit is part of every ten-bit vector.  Omitting it (and starting at
        // 0/8/16/24) does not describe the byte stream written to TX_FIFOn;
        // it programs a different bit ordering and a real target sees
        // corrupted register addresses and commands.
        let vector = |start: u32, stop: bool| (start << 5) | (1 << 4) | (7 << 1) | u32::from(stop);
        let cfg0 = vector(7, false) | (vector(15, false) << 10);
        let cfg1 = vector(23, false) | (vector(31, true) << 10);
        (cfg0, cfg1)
    }

    fn snapshot(regs: &R) -> Snapshot {
        Snapshot(
            regs.read32(regs::SE_GENI_STATUS),
            regs.read32(regs::SE_GENI_IOS),
        )
    }

    fn issue(regs: &R, opcode: u32, addr: u8, stop_stretch: bool) {
        let params = (u32::from(addr) << regs::SLAVE_ADDR_SHIFT)
            | if stop_stretch { regs::STOP_STRETCH } else { 0 };
        regs.write32(
            regs::SE_GENI_M_CMD0,
            (opcode << regs::M_OPCODE_SHIFT) | params,
        );
    }

    fn pump_tx(&mut self, regs: &R, write: &[u8]) -> bool {
        let occupied =
            (regs.read32(regs::SE_GENI_TX_FIFO_STATUS) & regs::TX_FIFO_WORDS_MASK) as usize;
        let mut room = self.tx_depth.saturating_sub(occupied);
        let mut progressed = false;
        while self.pending.tx_done < self.pending.write_len && room > 0 {
            let mut word = 0u32;
            let mut bytes = 0usize;
            while bytes < 4 && self.pending.tx_done < self.pending.write_len {
                let Some(byte) = write.get(self.pending.tx_done) else {
                    break;
                };
                word |= u32::from(*byte) << (bytes * 8);
                self.pending.tx_done += 1;
                bytes += 1;
            }
            if bytes == 0 {
                break;
            }
            regs.write32(regs::SE_GENI_TX_FIFO, word);
            room -= 1;
            progressed = true;
        }
        if self.pending.tx_done == self.pending.write_len {
            regs.write32(regs::SE_GENI_TX_WATERMARK, 0);
        }
        progressed
    }

    fn drain_rx(&mut self, regs: &R, rx: &mut [u8]) -> bool {
        let status = regs.read32(regs::SE_GENI_RX_FIFO_STATUS);
        let words = (status & regs::RX_FIFO_WORDS_MASK) as usize;
        let mut progressed = false;
        for word_index in 0..words {
            let mut word = regs.read32(regs::SE_GENI_RX_FIFO);
            let mut valid = 4usize;
            if word_index + 1 == words && status & regs::RX_LAST != 0 {
                let tail =
                    ((status & regs::RX_LAST_VALID_MASK) >> regs::RX_LAST_VALID_SHIFT) as usize;
                if tail != 0 {
                    valid = tail;
                }
            }
            for _ in 0..valid {
                if self.pending.rx_done == self.pending.read_len {
                    break;
                }
                let Some(slot) = rx.get_mut(self.pending.rx_done) else {
                    break;
                };
                *slot = word as u8;
                word >>= 8;
                self.pending.rx_done += 1;
                progressed = true;
            }
        }
        progressed
    }
}

impl<R: Regs + ?Sized> Default for GeniMaster<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Regs + ?Sized> I2cMaster for GeniMaster<R> {
    type Regs = R;
    type Abort = GeniAbort;
    const NAME: &'static str = "geni";

    fn probe(&mut self, regs: &R) -> Result<(), I2cError<GeniAbort>> {
        let revision = regs.read32(regs::GENI_FW_REVISION_RO);
        let protocol = (revision & regs::FW_PROTOCOL_MASK) >> regs::FW_PROTOCOL_SHIFT;
        if protocol != regs::PROTOCOL_I2C {
            return Err(I2cError::Identity {
                found: protocol,
                expected: regs::PROTOCOL_I2C,
            });
        }
        let tx = regs.read32(regs::SE_HW_PARAM_0);
        let rx = regs.read32(regs::SE_HW_PARAM_1);
        // FIFO packing below assumes at least a 32-bit entry.
        let tx_width = (tx & regs::FIFO_WIDTH_MASK) >> regs::FIFO_WIDTH_SHIFT;
        let rx_width = (rx & regs::FIFO_WIDTH_MASK) >> regs::FIFO_WIDTH_SHIFT;
        if tx_width < 32 || rx_width < 32 {
            return Err(I2cError::Identity {
                found: (tx_width << 16) | rx_width,
                expected: 0x0020_0020,
            });
        }
        self.tx_depth = Self::fifo_depth(tx).ok_or(I2cError::Identity {
            found: tx,
            expected: regs::FIFO_DEPTH_MASK,
        })?;
        self.rx_depth = Self::fifo_depth(rx).ok_or(I2cError::Identity {
            found: rx,
            expected: regs::FIFO_DEPTH_MASK,
        })?;
        Ok(())
    }

    fn configure(&mut self, regs: &R, cfg: &BusConfig) -> Result<(), I2cError<GeniAbort>> {
        if regs.read32(regs::SE_GENI_STATUS) & regs::M_CMD_ACTIVE != 0 {
            return Err(I2cError::EnableHandshake(Self::snapshot(regs)));
        }
        // Qualcomm's GENI FIFO-mode initialization sequence. ACPI firmware
        // keeps the clocks/power domain live, but the serial engine's own
        // routing and clock-gate registers remain the driver's responsibility.
        regs.write32(regs::SE_GSI_EVENT_EN, 0);
        regs.write32(regs::SE_GENI_M_IRQ_CLEAR, u32::MAX);
        regs.write32(regs::SE_GENI_S_IRQ_CLEAR, u32::MAX);
        // M_IRQ_EN below controls event generation/latching. SE_IRQ_EN is a
        // distinct route from those events toward the CPU-facing interrupt.
        // Leave that route shut in polling mode: the worker has no interrupt
        // capability and may not invent one.
        regs.write32(
            regs::SE_IRQ_EN,
            if cfg.irq_driven {
                regs::PRIMARY_IRQ_ROUTE
            } else {
                0
            },
        );
        regs.write32(regs::SE_GENI_DMA_MODE_EN, 0);
        let cgc = regs.read32(regs::SE_GENI_CGC_CTRL) | regs::DEFAULT_CGC_ENABLE;
        regs.write32(regs::SE_GENI_CGC_CTRL, cgc);
        let dma_cfg = regs.read32(regs::SE_DMA_GENERAL_CFG) | regs::DMA_INTERFACE_CLOCKS;
        regs.write32(regs::SE_DMA_GENERAL_CFG, dma_cfg);
        regs.write32(regs::GENI_OUTPUT_CTRL, regs::DEFAULT_IO_OUTPUT_ENABLE);
        regs.write32(regs::GENI_FORCE_DEFAULT_REG, regs::FORCE_DEFAULT);
        let (cfg0, cfg1) = Self::packing();
        for (offset, value) in [
            (regs::SE_GENI_TX_PACKING_CFG0, cfg0),
            (regs::SE_GENI_TX_PACKING_CFG1, cfg1),
            (regs::SE_GENI_RX_PACKING_CFG0, cfg0),
            (regs::SE_GENI_RX_PACKING_CFG1, cfg1),
        ] {
            regs.write32(offset, value);
        }
        regs.write32(regs::SE_GENI_BYTE_GRAN, 0);

        // Qualcomm's published tables for the two clock rates used by GENI.
        // Refuse any other exact firmware rate: treating every non-32 MHz
        // clock as 19.2 MHz silently mis-times the wire.
        let (div, high, low, cycle) = match (cfg.input_clock_hz, cfg.speed) {
            (32_000_000, Speed::Standard) => (8, 14, 18, 38),
            (32_000_000, Speed::Fast) => (4, 3, 9, 19),
            (19_200_000, Speed::Standard) => (7, 10, 12, 26),
            (19_200_000, Speed::Fast) => (2, 5, 11, 22),
            (hz, _) => return Err(I2cError::UnsupportedInputClock(hz)),
        };
        regs.write32(regs::SE_GENI_CLK_SEL, 0);
        regs.write32(
            regs::GENI_SER_M_CLK_CFG,
            (div << regs::CLK_DIV_SHIFT) | regs::SER_CLK_ENABLE,
        );
        regs.write32(
            regs::SE_I2C_SCL_COUNTERS,
            (high << regs::SCL_HIGH_SHIFT) | (low << regs::SCL_LOW_SHIFT) | cycle,
        );
        // Match geni_i2c_init -> geni_se_init: Qualcomm's I2C FIFO path uses
        // the discovered TX geometry for both receive thresholds, with the
        // watermark one word below the ready-for-receive threshold.
        regs.write32(regs::SE_GENI_RX_WATERMARK, (self.tx_depth - 1) as u32);
        regs.write32(regs::SE_GENI_RX_RFR_WATERMARK, self.tx_depth as u32);
        self.pending = Pending::default();
        Ok(())
    }

    fn start(&mut self, regs: &R, xfer: &Xfer<'_>) -> Result<(), I2cError<GeniAbort>> {
        let status = regs.read32(regs::SE_GENI_STATUS);
        let mut stale = regs.read32(regs::SE_GENI_M_IRQ_STATUS) & regs::IRQ_ERRORS;
        // Recovery is asynchronous in the hardware: ABORT completion may
        // latch after `recover` returned. Once the sequencer is idle that
        // administrative completion belongs to the previous transfer and is
        // safe to retire here; treating it as a fresh bus abort wedges every
        // later transfer behind `AbortPending`.
        let administrative = stale & (regs::IRQ_CMD_CANCEL | regs::IRQ_CMD_ABORT);
        if status & regs::M_CMD_ACTIVE == 0 && administrative != 0 {
            regs.write32(regs::SE_GENI_M_IRQ_CLEAR, administrative);
            stale &= !administrative;
        }
        if stale != 0 {
            return Err(I2cError::AbortPending);
        }
        if status & regs::M_CMD_ACTIVE != 0 {
            return Err(I2cError::EnableHandshake(Self::snapshot(regs)));
        }
        regs.write32(regs::SE_GENI_M_IRQ_CLEAR, u32::MAX);
        // These bits make FIFO progress sources latch in M_IRQ_STATUS. They
        // are required even when software polls that status. CPU delivery is
        // controlled separately by SE_IRQ_EN in configure().
        regs.write32(
            regs::SE_GENI_M_IRQ_EN,
            regs::IRQ_ENABLE | regs::IRQ_TRANSFER,
        );
        self.pending = Pending {
            tx_done: 0,
            rx_done: 0,
            write_len: xfer.write.len(),
            read_len: xfer.read_len,
            phase: if xfer.write.is_empty() {
                Phase::Read
            } else {
                Phase::Write
            },
            command_done: false,
        };
        regs.write32(regs::SE_I2C_TX_TRANS_LEN, xfer.write.len() as u32);
        regs.write32(
            regs::SE_I2C_RX_TRANS_LEN,
            if xfer.write.is_empty() {
                xfer.read_len as u32
            } else {
                0
            },
        );
        if xfer.write.is_empty() {
            Self::issue(regs, regs::I2C_READ, xfer.addr, false);
        } else {
            // A combined HID register read is two FIFO commands joined by
            // STOP_STRETCH, exactly the upstream FIFO path. The unused
            // WRITE_READ opcode is deliberately not trusted here.
            Self::issue(regs, regs::I2C_WRITE, xfer.addr, xfer.read_len != 0);
            regs.write32(regs::SE_GENI_TX_WATERMARK, 1);
            let _ = self.pump_tx(regs, xfer.write);
        }
        Ok(())
    }

    fn poll(&mut self, regs: &R, xfer: &Xfer<'_>, rx: &mut [u8]) -> XferState<GeniAbort> {
        let irq = regs.read32(regs::SE_GENI_M_IRQ_STATUS);
        if irq & regs::IRQ_ERRORS != 0 {
            let abort = GeniAbort::decode(
                irq,
                regs.read32(regs::SE_GENI_M_GP_LENGTH),
                self.pending.phase == Phase::Write,
            );
            regs.write32(regs::SE_GENI_TX_WATERMARK, 0);
            regs.write32(regs::SE_GENI_M_IRQ_CLEAR, irq);
            return XferState::Aborted(abort, Snapshot(irq, regs.read32(regs::SE_GENI_STATUS)));
        }
        let mut progressed = self.drain_rx(regs, rx);
        progressed |= self.pump_tx(regs, xfer.write);
        if irq & regs::IRQ_CMD_DONE != 0 {
            self.pending.command_done = true;
        }
        if irq != 0 {
            regs.write32(regs::SE_GENI_M_IRQ_CLEAR, irq);
        }
        if self.pending.command_done && self.pending.tx_done == self.pending.write_len {
            if self.pending.phase == Phase::Write && self.pending.read_len != 0 {
                regs.write32(regs::SE_I2C_TX_TRANS_LEN, 0);
                regs.write32(regs::SE_I2C_RX_TRANS_LEN, self.pending.read_len as u32);
                self.pending.phase = Phase::Read;
                self.pending.command_done = false;
                Self::issue(regs, regs::I2C_READ, xfer.addr, false);
                return XferState::Busy { progressed: true };
            }
            if self.pending.rx_done == self.pending.read_len {
                self.pending.phase = Phase::Idle;
                return XferState::Done {
                    read: self.pending.rx_done,
                };
            }
        }
        XferState::Busy { progressed }
    }

    fn recover(&mut self, regs: &R) {
        regs.write32(regs::SE_GENI_TX_WATERMARK, 0);
        if regs.read32(regs::SE_GENI_STATUS) & regs::M_CMD_ACTIVE != 0 {
            regs.write32(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_ABORT);
        } else {
            // No command can complete after this point, so every currently
            // latched source is old and may be retired synchronously.
            regs.write32(regs::SE_GENI_M_IRQ_CLEAR, u32::MAX);
        }
        self.pending = Pending::default();
    }

    fn begin_cancel(&mut self, regs: &R) -> RecoveryStart {
        regs.write32(regs::SE_GENI_TX_WATERMARK, 0);
        self.pending = Pending::default();
        if regs.read32(regs::SE_GENI_STATUS) & regs::M_CMD_ACTIVE == 0 {
            let stale = regs.read32(regs::SE_GENI_M_IRQ_STATUS);
            if stale != 0 {
                regs.write32(regs::SE_GENI_M_IRQ_CLEAR, stale);
            }
            RecoveryStart::Complete(RecoveryOutcome::Clean)
        } else {
            regs.write32(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_CANCEL);
            RecoveryStart::WaitCancel
        }
    }

    fn cancel_complete(&mut self, regs: &R) -> bool {
        let irq = regs.read32(regs::SE_GENI_M_IRQ_STATUS);
        let done = irq & regs::IRQ_CMD_CANCEL != 0
            || regs.read32(regs::SE_GENI_STATUS) & regs::M_CMD_ACTIVE == 0;
        if done && irq != 0 {
            regs.write32(regs::SE_GENI_M_IRQ_CLEAR, irq);
        }
        done
    }

    fn begin_abort(&mut self, regs: &R) -> RecoveryStart {
        if regs.read32(regs::SE_GENI_STATUS) & regs::M_CMD_ACTIVE == 0 {
            // The cancel request quiesced the sequencer without leaving a
            // completion source. No abort was issued, so preserve that
            // distinction in the settled diagnostic.
            RecoveryStart::Complete(RecoveryOutcome::Cancelled)
        } else {
            regs.write32(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_ABORT);
            RecoveryStart::WaitAbort
        }
    }

    fn abort_complete(&mut self, regs: &R) -> bool {
        let irq = regs.read32(regs::SE_GENI_M_IRQ_STATUS);
        let done = irq & regs::IRQ_CMD_ABORT != 0
            || regs.read32(regs::SE_GENI_STATUS) & regs::M_CMD_ACTIVE == 0;
        if done && irq != 0 {
            regs.write32(regs::SE_GENI_M_IRQ_CLEAR, irq);
        }
        done
    }

    fn finish_recovery(&mut self, regs: &R) {
        regs.write32(regs::SE_GENI_TX_WATERMARK, 0);
        let stale = regs.read32(regs::SE_GENI_M_IRQ_STATUS);
        if stale != 0 {
            regs.write32(regs::SE_GENI_M_IRQ_CLEAR, stale);
        }
        self.pending = Pending::default();
    }

    fn timeout_snapshot(&self, regs: &R) -> Snapshot {
        // A bounded-wait failure needs the event the pump was waiting for,
        // not merely the generic active bit. Keep the FIFO word beside it
        // for host/debug consumers; the input service's compact witness uses
        // the first word verbatim.
        Snapshot(
            regs.read32(regs::SE_GENI_M_IRQ_STATUS),
            regs.read32(regs::SE_GENI_RX_FIFO_STATUS),
        )
    }
}
