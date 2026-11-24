use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::vec::Vec;

use kore_driver::Regs;
use kore_i2c::{
    Budget, BusConfig, I2cEngine, I2cError, I2cMaster, Pacing, RecoveryOutcome, Speed, Xfer,
    XferState,
};

use crate::{GeniAbort, GeniAbortCause, GeniMaster, regs};

#[derive(Default)]
struct FakeRegs {
    values: RefCell<BTreeMap<usize, u32>>,
    reads: RefCell<BTreeMap<usize, VecDeque<u32>>>,
    writes: RefCell<Vec<(usize, u32)>>,
}

impl FakeRegs {
    fn set(&self, offset: usize, value: u32) {
        self.values.borrow_mut().insert(offset, value);
    }

    /// Queue destructive reads for a register.  This is load-bearing for the
    /// RX FIFO: a map-backed fake returns the same word forever and therefore
    /// cannot model a descriptor larger than one FIFO entry.
    fn queue_reads(&self, offset: usize, values: impl IntoIterator<Item = u32>) {
        self.reads
            .borrow_mut()
            .entry(offset)
            .or_default()
            .extend(values);
    }

    fn last_write(&self, offset: usize) -> Option<u32> {
        self.writes
            .borrow()
            .iter()
            .rev()
            .find_map(|(at, value)| (*at == offset).then_some(*value))
    }

    fn wrote(&self, offset: usize, value: u32) -> bool {
        self.writes.borrow().contains(&(offset, value))
    }
}

impl Regs for FakeRegs {
    fn read32(&self, offset: usize) -> u32 {
        if let Some(value) = self
            .reads
            .borrow_mut()
            .get_mut(&offset)
            .and_then(VecDeque::pop_front)
        {
            return value;
        }
        self.values.borrow().get(&offset).copied().unwrap_or(0)
    }

    fn write32(&self, offset: usize, value: u32) {
        if offset == regs::SE_GENI_M_IRQ_CLEAR {
            let mut values = self.values.borrow_mut();
            let status = values.entry(regs::SE_GENI_M_IRQ_STATUS).or_default();
            *status &= !value;
            values.insert(offset, value);
        } else {
            self.values.borrow_mut().insert(offset, value);
        }
        self.writes.borrow_mut().push((offset, value));
    }
}

fn packed_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0u32, |word, (shift, byte)| {
                word | (u32::from(*byte) << (shift * 8))
            })
        })
        .collect()
}

fn probed() -> (GeniMaster<FakeRegs>, FakeRegs) {
    let regs_io = FakeRegs::default();
    regs_io.set(
        regs::GENI_FW_REVISION_RO,
        regs::PROTOCOL_I2C << regs::FW_PROTOCOL_SHIFT,
    );
    let params = (32 << regs::FIFO_WIDTH_SHIFT) | (16 << regs::FIFO_DEPTH_SHIFT);
    regs_io.set(regs::SE_HW_PARAM_0, params);
    regs_io.set(regs::SE_HW_PARAM_1, params);
    let mut master = GeniMaster::new();
    master.probe(&regs_io).expect("GENI I2C identity");
    (master, regs_io)
}

/// Test pacing which keeps the transfer deadline distinct from each
/// recovery allowance. A shared callback would inject administrative
/// completion into the still-running transfer and forge an abort instead
/// of exercising cancel/abort recovery.
struct RecoveryBudget<F> {
    normal_steps: u32,
    recovery_steps: u32,
    remaining: u32,
    recovery_round: u32,
    delay: F,
}

impl<F: FnMut(u32)> RecoveryBudget<F> {
    fn new(normal_steps: u32, recovery_steps: u32, delay: F) -> Self {
        Self {
            normal_steps,
            recovery_steps,
            remaining: normal_steps,
            recovery_round: 0,
            delay,
        }
    }
}

impl<F: FnMut(u32)> Pacing for RecoveryBudget<F> {
    fn step(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        (self.delay)(self.recovery_round);
        true
    }

    fn reset(&mut self, _transfer_bytes: usize) {
        self.remaining = self.normal_steps;
        self.recovery_round = 0;
    }

    fn reset_recovery(&mut self) {
        self.remaining = self.recovery_steps;
        self.recovery_round += 1;
    }
}

#[test]
fn configure_replays_the_complete_fifo_mode_initialization() {
    let (mut master, regs_io) = probed();
    master
        .configure(
            &regs_io,
            &BusConfig {
                speed: Speed::Fast,
                counts: None,
                input_clock_hz: 19_200_000,
                sda_hold: None,
                irq_driven: true,
            },
        )
        .expect("idle controller configures");

    assert_eq!(regs_io.last_write(regs::SE_GSI_EVENT_EN), Some(0));
    assert_eq!(
        regs_io.last_write(regs::SE_IRQ_EN),
        Some(regs::PRIMARY_IRQ_ROUTE)
    );
    assert_eq!(
        regs_io.last_write(regs::SE_DMA_GENERAL_CFG),
        Some(regs::DMA_INTERFACE_CLOCKS)
    );
    assert_eq!(
        regs_io.last_write(regs::GENI_FORCE_DEFAULT_REG),
        Some(regs::FORCE_DEFAULT)
    );
    // These are the exact words produced by Linux
    // geni_se_config_packing(8, 4, true), and the values observed by the
    // FreeBSD QCOM0C10 ACPI driver on Snapdragon laptops.  A direction-bit
    // regression still looks plausible in a register transcript but puts
    // different bits on the wire.
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_TX_PACKING_CFG0),
        Some(0x0007_f8fe)
    );
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_TX_PACKING_CFG1),
        Some(0x000f_fefe)
    );
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_RX_PACKING_CFG0),
        Some(0x0007_f8fe)
    );
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_RX_PACKING_CFG1),
        Some(0x000f_fefe)
    );
    assert_eq!(
        regs_io.last_write(regs::GENI_SER_M_CLK_CFG),
        Some((2 << regs::CLK_DIV_SHIFT) | regs::SER_CLK_ENABLE)
    );
    assert_eq!(
        regs_io.last_write(regs::SE_I2C_SCL_COUNTERS),
        Some((5 << regs::SCL_HIGH_SHIFT) | (11 << regs::SCL_LOW_SHIFT) | 22)
    );
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_RX_WATERMARK),
        Some(15),
        "Linux programs discovered depth - 1"
    );
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_RX_RFR_WATERMARK),
        Some(16),
        "Linux programs the discovered depth"
    );
}

#[test]
fn configure_refuses_an_unproven_source_clock() {
    let (mut master, regs_io) = probed();
    assert_eq!(
        master.configure(&regs_io, &BusConfig::fast(24_000_000)),
        Err(I2cError::UnsupportedInputClock(24_000_000))
    );
}

#[test]
fn polling_keeps_progress_events_latched_but_closes_cpu_delivery() {
    let (mut master, regs_io) = probed();
    let cfg = BusConfig {
        irq_driven: false,
        ..BusConfig::fast(19_200_000)
    };
    master.configure(&regs_io, &cfg).expect("configure polling");
    assert_eq!(regs_io.last_write(regs::SE_IRQ_EN), Some(0));
    assert_eq!(
        regs::IRQ_COMMON,
        0x33c0_007e,
        "shared enable mask must remain bit-for-bit compatible with Linux"
    );

    let xfer = Xfer::read(0x3a, 9);
    master.start(&regs_io, &xfer).expect("start polling read");
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_M_IRQ_EN),
        Some(regs::IRQ_ENABLE | regs::IRQ_TRANSFER),
        "event/status production is independent of the CPU route"
    );
    assert_ne!(
        regs::IRQ_ENABLE & regs::IRQ_RX_FIFO_WR_ERR,
        0,
        "shared GENI FIFO faults remain observable in polling mode"
    );
    assert_eq!(
        regs::IRQ_ERRORS & regs::IRQ_TIMESTAMP,
        0,
        "an informational status source must not become a false I2C abort"
    );
}

#[test]
fn probe_refuses_version_ambiguous_or_zero_fifo_geometry() {
    let (mut master, regs_io) = probed();
    regs_io.set(
        regs::SE_HW_PARAM_0,
        (32 << regs::FIFO_WIDTH_SHIFT) | regs::FIFO_DEPTH_EXTENDED_MASK,
    );
    assert!(matches!(
        master.probe(&regs_io),
        Err(kore_i2c::I2cError::Identity { .. })
    ));

    regs_io.set(regs::SE_HW_PARAM_0, 32 << regs::FIFO_WIDTH_SHIFT);
    assert!(matches!(
        master.probe(&regs_io),
        Err(kore_i2c::I2cError::Identity { .. })
    ));
}

#[test]
fn write_read_uses_two_commands_joined_by_stop_stretch() {
    let (mut master, regs_io) = probed();
    master
        .configure(&regs_io, &BusConfig::fast(19_200_000))
        .expect("configure");
    let xfer = Xfer::write_read(0x3a, &[1, 0], 4);
    master.start(&regs_io, &xfer).expect("start write phase");

    assert_eq!(regs_io.last_write(regs::SE_I2C_TX_TRANS_LEN), Some(2));
    assert_eq!(regs_io.last_write(regs::SE_I2C_RX_TRANS_LEN), Some(0));
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_M_CMD0),
        Some(
            (regs::I2C_WRITE << regs::M_OPCODE_SHIFT)
                | (0x3a << regs::SLAVE_ADDR_SHIFT)
                | regs::STOP_STRETCH
        )
    );
    assert_eq!(regs_io.last_write(regs::SE_GENI_TX_FIFO), Some(1));

    regs_io.set(regs::SE_GENI_M_IRQ_STATUS, regs::IRQ_CMD_DONE);
    assert_eq!(
        master.poll(&regs_io, &xfer, &mut [0; 4]),
        XferState::Busy { progressed: true }
    );
    assert_eq!(regs_io.last_write(regs::SE_I2C_RX_TRANS_LEN), Some(4));
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_M_CMD0),
        Some((regs::I2C_READ << regs::M_OPCODE_SHIFT) | (0x3a << regs::SLAVE_ADDR_SHIFT))
    );
}

#[test]
fn read_completion_drains_packed_fifo_bytes() {
    let (mut master, regs_io) = probed();
    master
        .configure(&regs_io, &BusConfig::fast(32_000_000))
        .expect("configure");
    let xfer = Xfer::read(0x15, 3);
    master.start(&regs_io, &xfer).expect("start read");
    regs_io.set(
        regs::SE_GENI_RX_FIFO_STATUS,
        1 | regs::RX_LAST | (3 << regs::RX_LAST_VALID_SHIFT),
    );
    regs_io.set(regs::SE_GENI_RX_FIFO, 0x00_33_22_11);
    regs_io.set(
        regs::SE_GENI_M_IRQ_STATUS,
        regs::IRQ_CMD_DONE | regs::IRQ_RX_LAST,
    );
    let mut rx = [0u8; 3];
    assert_eq!(
        master.poll(&regs_io, &xfer, &mut rx),
        XferState::Done { read: 3 }
    );
    assert_eq!(rx, [0x11, 0x22, 0x33]);
}

#[test]
fn long_read_remembers_completion_until_the_last_fifo_batch_is_drained() {
    let (mut master, regs_io) = probed();
    master
        .configure(&regs_io, &BusConfig::fast(32_000_000))
        .expect("configure");
    let xfer = Xfer::read(0x3a, 7);
    master.start(&regs_io, &xfer).expect("start read");

    // Hardware may latch CMD_DONE with an RX batch still visible and expose
    // the tail on the next poll.  Acknowledging CMD_DONE must not forget that
    // the command itself is complete.
    regs_io.queue_reads(regs::SE_GENI_RX_FIFO_STATUS, [1]);
    regs_io.queue_reads(regs::SE_GENI_RX_FIFO, [0x44_33_22_11]);
    regs_io.set(
        regs::SE_GENI_M_IRQ_STATUS,
        regs::IRQ_CMD_DONE | regs::IRQ_RX_WATERMARK,
    );
    let mut rx = [0u8; 7];
    assert_eq!(
        master.poll(&regs_io, &xfer, &mut rx),
        XferState::Busy { progressed: true }
    );

    regs_io.set(regs::SE_GENI_M_IRQ_STATUS, 0);
    regs_io.queue_reads(
        regs::SE_GENI_RX_FIFO_STATUS,
        [1 | regs::RX_LAST | (3 << regs::RX_LAST_VALID_SHIFT)],
    );
    regs_io.queue_reads(regs::SE_GENI_RX_FIFO, [0x00_77_66_55]);
    assert_eq!(
        master.poll(&regs_io, &xfer, &mut rx),
        XferState::Done { read: 7 }
    );
    assert_eq!(rx, [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77]);
}

#[test]
fn report_descriptor_larger_than_the_fifo_drains_every_scripted_batch() {
    const LEN: usize = 193;
    const WORDS_PER_BATCH: usize = 12;

    let (master, regs_io) = probed();
    let mut engine = I2cEngine::new(master);
    engine
        .configure(&regs_io, &BusConfig::fast(19_200_000))
        .expect("configure");

    let mut expected = [0u8; LEN];
    for (index, byte) in expected.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
    }

    let words = packed_words(&expected);
    let batches = words.len().div_ceil(WORDS_PER_BATCH);

    // First poll completes the two-byte register-address write.  Each later
    // entry is one FIFO visibility edge from the read command.  The final
    // edge carries both RX_LAST and CMD_DONE, as real GENI does.
    let mut irqs = Vec::with_capacity(batches + 1);
    let mut fifo_status = Vec::with_capacity(batches + 1);
    // `start` first samples the register for a stale abort before issuing the
    // write command; the transcript's leading zero is that preflight read.
    irqs.push(0);
    irqs.push(regs::IRQ_CMD_DONE);
    fifo_status.push(0);
    for (batch_index, batch) in words.chunks(WORDS_PER_BATCH).enumerate() {
        let last = batch_index + 1 == batches;
        let mut status = batch.len() as u32;
        if last {
            let tail = (LEN % 4) as u32;
            status |= regs::RX_LAST | (tail << regs::RX_LAST_VALID_SHIFT);
            irqs.push(regs::IRQ_CMD_DONE | regs::IRQ_RX_LAST);
        } else {
            irqs.push(regs::IRQ_RX_WATERMARK);
        }
        fifo_status.push(status);
    }
    regs_io.queue_reads(regs::SE_GENI_M_IRQ_STATUS, irqs);
    regs_io.queue_reads(regs::SE_GENI_RX_FIFO_STATUS, fifo_status);
    regs_io.queue_reads(regs::SE_GENI_RX_FIFO, words);

    let xfer = Xfer::write_read(0x3a, &[0x20, 0], LEN);
    let mut actual = [0u8; LEN];
    let mut pacing = Budget::new(1, || panic!("every transcript edge makes progress"));
    let got = engine
        .transfer(&regs_io, &xfer, &mut actual, &mut pacing)
        .expect("complete multi-batch report descriptor");

    assert_eq!(got, LEN);
    assert_eq!(actual, expected);
    assert_eq!(engine.stats().idle_polls, 0);
    assert_eq!(engine.stats().transfers, 1);
}

#[test]
fn timeout_snapshot_captures_the_event_and_fifo_state_the_pump_waited_for() {
    let (master, regs_io) = probed();
    regs_io.set(regs::SE_GENI_M_IRQ_STATUS, regs::IRQ_RX_WATERMARK);
    regs_io.set(regs::SE_GENI_RX_FIFO_STATUS, 7 | regs::RX_LAST);

    assert_eq!(
        master.timeout_snapshot(&regs_io),
        kore_i2c::Snapshot(regs::IRQ_RX_WATERMARK, 7 | regs::RX_LAST)
    );
}

#[test]
fn nack_decode_distinguishes_address_from_write_data() {
    assert_eq!(
        GeniAbort::decode(regs::IRQ_GP_1, 0, true).cause,
        GeniAbortCause::AddressNack
    );
    assert_eq!(
        GeniAbort::decode(regs::IRQ_GP_1, 1, true).cause,
        GeniAbortCause::DataNack
    );
    assert_eq!(
        GeniAbort::decode(regs::IRQ_GP_1, 1, false).cause,
        GeniAbortCause::AddressNack
    );
}

#[test]
fn a_late_abort_completion_cannot_poison_the_next_transfer() {
    let (mut master, regs_io) = probed();
    master
        .configure(&regs_io, &BusConfig::fast(19_200_000))
        .expect("configure");
    regs_io.set(regs::SE_GENI_STATUS, regs::M_CMD_ACTIVE);
    master.recover(&regs_io);
    assert_eq!(
        regs_io.last_write(regs::SE_GENI_M_CMD_CTRL_REG),
        Some(regs::M_CMD_ABORT)
    );

    // Hardware retires the abort after recover returned. The next start sees
    // an idle sequencer, acknowledges that administrative completion, and
    // starts the new command instead of reporting AbortPending forever.
    regs_io.set(regs::SE_GENI_STATUS, 0);
    regs_io.set(regs::SE_GENI_M_IRQ_STATUS, regs::IRQ_CMD_ABORT);
    let xfer = Xfer::write(0x3a, &[1]);
    master
        .start(&regs_io, &xfer)
        .expect("late abort is retired");
    assert!(regs_io.wrote(regs::SE_GENI_M_IRQ_CLEAR, regs::IRQ_CMD_ABORT));
}

#[test]
fn failed_start_cancels_stale_command_before_a_sibling_transfer() {
    let (master, regs_io) = probed();
    regs_io.set(regs::SE_GENI_STATUS, regs::M_CMD_ACTIVE);
    let mut engine = I2cEngine::new(master);
    let mut pacing = RecoveryBudget::new(0, 1, |round| {
        assert_eq!(round, 1, "a failed start gets a cancel allowance");
        regs_io.set(regs::SE_GENI_M_IRQ_STATUS, regs::IRQ_CMD_CANCEL);
        regs_io.set(regs::SE_GENI_STATUS, 0);
    });

    let error = engine
        .transfer(&regs_io, &Xfer::write(0x3a, &[1]), &mut [], &mut pacing)
        .unwrap_err();
    assert_eq!(
        error,
        I2cError::EnableHandshake(kore_i2c::Snapshot(regs::M_CMD_ACTIVE, 0))
    );
    assert_eq!(pacing.recovery_round, 1);
    assert!(regs_io.wrote(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_CANCEL));
    assert!(!regs_io.wrote(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_ABORT));
    assert_eq!(regs_io.last_write(regs::SE_GENI_M_CMD0), None);
    assert_eq!(regs_io.read32(regs::SE_GENI_STATUS), 0);
    assert_eq!(engine.stats().refused, 1);
    assert_eq!(engine.stats().transfers, 0);

    // A child on the same bus must not encounter the previous child's
    // still-running administrative command. Start and complete immediately.
    regs_io.queue_reads(regs::SE_GENI_M_IRQ_STATUS, [0, regs::IRQ_CMD_DONE]);
    assert_eq!(
        engine.transfer(&regs_io, &Xfer::write(0x15, &[2]), &mut [], &mut pacing),
        Ok(0)
    );
    assert_eq!(engine.stats().refused, 1);
    assert_eq!(engine.stats().transfers, 1);
    assert_eq!(engine.stats().written, 1);
}

#[test]
fn failed_start_escalates_cancel_to_bounded_abort() {
    use core::cell::Cell;

    let (master, regs_io) = probed();
    regs_io.set(regs::SE_GENI_STATUS, regs::M_CMD_ACTIVE);
    let mut engine = I2cEngine::new(master);
    let steps = Cell::new([0u32; 2]);
    let mut pacing = RecoveryBudget::new(0, 2, |round| {
        assert!((1..=2).contains(&round));
        let mut counts = steps.get();
        counts[(round - 1) as usize] += 1;
        steps.set(counts);
        if round == 2 {
            regs_io.set(regs::SE_GENI_M_IRQ_STATUS, regs::IRQ_CMD_ABORT);
            regs_io.set(regs::SE_GENI_STATUS, 0);
        }
    });

    let error = engine
        .transfer(&regs_io, &Xfer::write(0x3a, &[1]), &mut [], &mut pacing)
        .unwrap_err();
    assert_eq!(
        error,
        I2cError::EnableHandshake(kore_i2c::Snapshot(regs::M_CMD_ACTIVE, 0))
    );
    assert_eq!(pacing.recovery_round, 2);
    assert_eq!(steps.get(), [2, 1]);
    assert!(regs_io.wrote(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_CANCEL));
    assert!(regs_io.wrote(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_ABORT));
    assert_eq!(regs_io.last_write(regs::SE_GENI_M_CMD0), None);
    assert_eq!(regs_io.read32(regs::SE_GENI_STATUS), 0);
    assert_eq!(engine.stats().refused, 1);
    assert_eq!(engine.stats().transfers, 0);
}

#[test]
fn failed_start_recovery_returns_after_both_allowances_expire() {
    use core::cell::Cell;

    let (master, regs_io) = probed();
    regs_io.set(regs::SE_GENI_STATUS, regs::M_CMD_ACTIVE);
    let mut engine = I2cEngine::new(master);
    let steps = Cell::new([0u32; 2]);
    let mut pacing = RecoveryBudget::new(0, 2, |round| {
        assert!((1..=2).contains(&round));
        let mut counts = steps.get();
        counts[(round - 1) as usize] += 1;
        steps.set(counts);
        // Neither cancel nor abort settles: both phases must be bounded.
    });

    let error = engine
        .transfer(&regs_io, &Xfer::write(0x3a, &[1]), &mut [], &mut pacing)
        .unwrap_err();
    assert_eq!(
        error,
        I2cError::EnableHandshake(kore_i2c::Snapshot(regs::M_CMD_ACTIVE, 0))
    );
    assert_eq!(pacing.recovery_round, 2);
    assert_eq!(steps.get(), [2, 2]);
    assert!(regs_io.wrote(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_CANCEL));
    assert!(regs_io.wrote(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_ABORT));
    assert_eq!(regs_io.last_write(regs::SE_GENI_M_CMD0), None);
    assert_eq!(regs_io.read32(regs::SE_GENI_STATUS), regs::M_CMD_ACTIVE);
    assert_eq!(engine.stats().refused, 1);
    assert_eq!(engine.stats().transfers, 0);
    assert_eq!(
        engine.stats().timeouts,
        0,
        "a start refusal is not a transfer timeout"
    );
}

#[test]
fn timeout_recovery_records_cancel_success() {
    let (master, regs_io) = probed();
    regs_io.queue_reads(regs::SE_GENI_STATUS, [0, regs::M_CMD_ACTIVE]);
    let mut engine = I2cEngine::new(master);
    let xfer = Xfer::read(0x3a, 9);
    let mut rx = [0u8; 9];
    let mut pacing = RecoveryBudget::new(0, 1, |round| {
        if round == 1 {
            regs_io.set(regs::SE_GENI_M_IRQ_STATUS, regs::IRQ_CMD_CANCEL);
            regs_io.set(regs::SE_GENI_STATUS, 0);
        }
    });
    let err = engine
        .transfer(&regs_io, &xfer, &mut rx, &mut pacing)
        .unwrap_err();
    assert!(matches!(
        err,
        I2cError::Timeout {
            recovery: RecoveryOutcome::Cancelled,
            ..
        }
    ));
    assert!(regs_io.wrote(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_CANCEL));
}

#[test]
fn cancel_timeout_escalates_to_abort_and_records_success() {
    use core::cell::Cell;

    let (master, regs_io) = probed();
    regs_io.queue_reads(
        regs::SE_GENI_STATUS,
        [
            0,
            regs::M_CMD_ACTIVE,
            regs::M_CMD_ACTIVE,
            regs::M_CMD_ACTIVE,
            regs::M_CMD_ACTIVE,
            0,
        ],
    );
    let wakes = Cell::new(0);
    let mut engine = I2cEngine::new(master);
    let xfer = Xfer::read(0x3a, 193);
    let mut rx = [0u8; 193];
    let mut pacing = RecoveryBudget::new(0, 1, |round| {
        wakes.set(wakes.get() + 1);
        if round == 2 {
            regs_io.set(regs::SE_GENI_M_IRQ_STATUS, regs::IRQ_CMD_ABORT);
            regs_io.set(regs::SE_GENI_STATUS, 0);
        }
    });
    let err = engine
        .transfer(&regs_io, &xfer, &mut rx, &mut pacing)
        .unwrap_err();
    assert!(matches!(
        err,
        I2cError::Timeout {
            recovery: RecoveryOutcome::Aborted,
            ..
        }
    ));
    assert!(regs_io.wrote(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_CANCEL));
    assert!(regs_io.wrote(regs::SE_GENI_M_CMD_CTRL_REG, regs::M_CMD_ABORT));
}

#[test]
fn cancel_and_abort_timeout_preserve_original_and_terminal_evidence() {
    let (master, regs_io) = probed();
    regs_io.queue_reads(
        regs::SE_GENI_STATUS,
        [
            0,
            regs::M_CMD_ACTIVE,
            regs::M_CMD_ACTIVE,
            regs::M_CMD_ACTIVE,
            regs::M_CMD_ACTIVE,
        ],
    );
    regs_io.set(regs::SE_GENI_RX_FIFO_STATUS, 7);
    let mut engine = I2cEngine::new(master);
    let xfer = Xfer::read(0x3a, 193);
    let mut rx = [0u8; 193];
    let mut pacing = RecoveryBudget::new(0, 0, |_| unreachable!());
    let err = engine
        .transfer(&regs_io, &xfer, &mut rx, &mut pacing)
        .unwrap_err();
    assert!(matches!(
        err,
        I2cError::Timeout {
            snapshot: kore_i2c::Snapshot(0, 7),
            recovery: RecoveryOutcome::Failed(kore_i2c::Snapshot(0, 7)),
        }
    ));
}
