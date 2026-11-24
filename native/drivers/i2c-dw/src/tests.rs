//! The DesignWare core against its own register file.
//!
//! Tests drive the **real** `DwMaster`, directly for interrupt lifecycle
//! checks and through the **real** `I2cEngine` for transfers, against
//! `MockDwI2c` with a device behind it. Nothing
//! asserts on a recorded register transcript: the assertions are about
//! what the device saw and what the driver got back, because those are the
//! only two things a bus is for. The mock's violation flags cover the
//! sequencing rules that have no observable effect in a mock and hang on
//! metal — which is exactly the class of bug a transcript test misses.

use core::cell::{Cell, RefCell};

use kore_driver::Regs;
use kore_i2c::mock::{I2cTarget, TargetReply};
use kore_i2c::{
    Budget, BusConfig, Counts, DeadlinePacing, I2cBus, I2cEngine, I2cError, I2cMaster, RateBudget,
    Speed, Xfer, XferState,
};

use crate::abort::DwAbortCause;
use crate::integration::Integration;
use crate::master::DwMaster;
use crate::mock::{MockDwI2c, Violation};
use crate::regs;

/// A device that answers reads from a canned buffer and remembers every
/// write it was given.
struct CannedDevice {
    addr: u8,
    reply: [u8; 64],
    reply_len: usize,
    /// The last write the device saw, and how long it was.
    seen: [u8; 64],
    seen_len: usize,
    /// How many transactions it answered.
    transactions: usize,
    /// Force an address NACK.
    nack_addr: bool,
    /// Force a data NACK at this index.
    nack_data: Option<usize>,
}

impl CannedDevice {
    fn new(addr: u8, reply: &[u8]) -> Self {
        let mut r = [0u8; 64];
        r[..reply.len()].copy_from_slice(reply);
        Self {
            addr,
            reply: r,
            reply_len: reply.len(),
            seen: [0; 64],
            seen_len: 0,
            transactions: 0,
            nack_addr: false,
            nack_data: None,
        }
    }
}

impl I2cTarget for CannedDevice {
    fn transact(&mut self, addr: u8, write: &[u8], out: &mut [u8]) -> TargetReply {
        self.transactions += 1;
        if addr != self.addr || self.nack_addr {
            return TargetReply::NackAddr;
        }
        if let Some(i) = self.nack_data {
            return TargetReply::NackData(i);
        }
        self.seen_len = write.len().min(self.seen.len());
        self.seen[..self.seen_len].copy_from_slice(&write[..self.seen_len]);
        let n = out.len().min(self.reply_len);
        out[..n].copy_from_slice(&self.reply[..n]);
        TargetReply::Acked(n)
    }
}

/// A no-op pacing budget generous enough that only a genuinely stuck
/// transfer times out.
fn budget() -> Budget<fn()> {
    Budget::new(64, || {})
}

/// A configured engine over a mock, ready to transfer.
fn bring_up<T: I2cTarget>(
    mock: &MockDwI2c<T>,
) -> Result<I2cEngine<DwMaster<MockDwI2c<T>>>, I2cError<crate::DwAbort>> {
    let mut engine = I2cEngine::new(DwMaster::new(Integration::INTEL_LPSS, || {}));
    engine.probe(mock)?;
    engine.configure(mock, &BusConfig::fast(100_000_000))?;
    Ok(engine)
}

fn configured_master<R: Regs>(regs: &R, integration: Integration) -> DwMaster<R> {
    let mut master = DwMaster::new(integration, || {});
    master.probe(regs).expect("probe");
    master
        .configure(regs, &BusConfig::fast(integration.input_clock_hz))
        .expect("configure");
    master
}

/// A four-command TX FIFO in front of the synchronous register model.
/// Only `drain_tx` advances the wire, so tests can inspect the IRQ line
/// while commands are queued, emptied, refilled, and finally completed.
struct BufferedTx<T> {
    core: MockDwI2c<T>,
    tx: RefCell<([u32; 4], usize)>,
}

impl<T: I2cTarget> BufferedTx<T> {
    fn new(target: T) -> Self {
        Self {
            core: MockDwI2c::with_fifos(Integration::INTEL_LPSS, target, 4, 4),
            tx: RefCell::new(([0; 4], 0)),
        }
    }

    fn drain_tx(&self) {
        let mut tx = self.tx.borrow_mut();
        for &word in &tx.0[..tx.1] {
            self.core.write32(regs::IC_DATA_CMD, word);
        }
        tx.1 = 0;
    }

    fn interrupt_asserted(&self) -> bool {
        self.read32(regs::IC_INTR_STAT) != 0
    }
}

impl<T: I2cTarget> Regs for BufferedTx<T> {
    fn read32(&self, off: usize) -> u32 {
        let tx = self.tx.borrow();
        match off {
            regs::IC_STATUS => {
                let mut status = self.core.read32(off) & !(regs::status::TFNF | regs::status::TFE);
                if tx.1 < tx.0.len() {
                    status |= regs::status::TFNF;
                }
                if tx.1 == 0 {
                    status |= regs::status::TFE;
                }
                status
            }
            regs::IC_TXFLR => tx.1 as u32,
            regs::IC_RAW_INTR_STAT | regs::IC_INTR_STAT => {
                let mut raw = self.core.read32(regs::IC_RAW_INTR_STAT);
                if tx.1 != 0 {
                    raw &= !regs::intr::TX_EMPTY;
                }
                if off == regs::IC_INTR_STAT {
                    raw & self.core.interrupt_mask()
                } else {
                    raw
                }
            }
            _ => self.core.read32(off),
        }
    }

    fn write32(&self, off: usize, val: u32) {
        if off == regs::IC_DATA_CMD {
            let mut tx = self.tx.borrow_mut();
            let next = tx.1;
            assert!(next < tx.0.len(), "TX FIFO overflow");
            tx.0[next] = val;
            tx.1 += 1;
        } else {
            if off == regs::IC_ENABLE && val & regs::enable::ENABLE == 0 {
                self.tx.borrow_mut().1 = 0;
            }
            self.core.write32(off, val);
        }
    }
}

#[test]
fn bring_up_releases_the_lpss_reset_before_touching_any_dw_register() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[]));
    let engine = bring_up(&mock).expect("bring-up");
    assert!(mock.clean(), "violations: {:?}", mock.violations());
    // The depths came from IC_COMP_PARAM_1, not from the conservative
    // defaults — proof the identity read happened after the reset.
    assert_eq!(engine.master().fifo_depths(), (16, 16));
}

#[test]
fn a_core_behind_an_unreleased_reset_would_be_misdiagnosed_and_the_mock_says_so() {
    // The failure the ordering exists to prevent: read identity first and
    // a live core reads back as absent.
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[]));
    let found = kore_driver::Regs::read32(&mock, regs::IC_COMP_TYPE);
    assert_eq!(found, 0, "a held-in-reset core reads as zero");
    assert!(
        mock.violations()
            .contains(Violation::RegisterBeforeLpssReset(regs::IC_COMP_TYPE)),
        "the mock names the ordering error"
    );
}

#[test]
fn a_plain_integration_needs_no_reset_and_gets_none() {
    let mock = MockDwI2c::new(Integration::AMD_PLAIN, CannedDevice::new(0x2c, &[]));
    let mut engine = I2cEngine::new(DwMaster::new(Integration::AMD_PLAIN, || {}));
    engine.probe(&mock).expect("probe");
    engine
        .configure(
            &mock,
            &BusConfig::fast(Integration::AMD_PLAIN.input_clock_hz),
        )
        .expect("configure");
    assert!(mock.clean(), "violations: {:?}", mock.violations());
    // AMD's clock derives different counts than Intel's — the whole
    // reason the clock is per-integration data.
    let (h, l) = mock.fast_counts();
    assert_eq!((h, l), (117, 212));
}

#[test]
fn a_window_that_is_not_a_designware_core_is_refused_by_identity() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[]))
        .with_comp_type(0xdead_beef);
    let Err(err) = bring_up(&mock) else {
        panic!("a foreign window was accepted")
    };
    assert_eq!(
        err,
        I2cError::Identity {
            found: 0xdead_beef,
            expected: regs::COMP_TYPE_VALUE
        }
    );
}

#[test]
fn a_core_that_never_enables_times_out_bounded_and_names_its_registers() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[])).stuck_enable();
    let mut engine = I2cEngine::new(DwMaster::new(Integration::INTEL_LPSS, || {}));
    engine.probe(&mock).expect("probe");
    // The handshake is bounded: this returns rather than hanging, which
    // is the entire point of routing it through the shared wait seam.
    let err = engine
        .configure(&mock, &BusConfig::fast(100_000_000))
        .unwrap_err();
    assert!(matches!(err, I2cError::EnableHandshake(_)), "{err:?}");
}

#[test]
fn configuration_lands_because_it_happens_while_disabled() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[]));
    let mut engine = I2cEngine::new(DwMaster::new(Integration::INTEL_LPSS, || {}));
    engine.probe(&mock).expect("probe");
    engine
        .configure(
            &mock,
            &BusConfig {
                speed: Speed::Standard,
                counts: Some(Counts {
                    hcnt: 0x1234,
                    lcnt: 0x5678,
                }),
                input_clock_hz: 100_000_000,
                sda_hold: Some(0x42),
                irq_driven: true,
            },
        )
        .expect("configure");
    assert!(mock.clean(), "violations: {:?}", mock.violations());
    // Firmware counts win over anything derived.
    assert_eq!(mock.standard_counts(), (0x1234, 0x5678));
    assert_eq!(mock.sda_hold(), 0x42);
    let con = mock.control();
    assert_ne!(con & regs::con::MASTER, 0, "master mode");
    assert_ne!(con & regs::con::RESTART_EN, 0, "restart enabled");
    assert_ne!(con & regs::con::SLAVE_DISABLE, 0, "slave block disabled");
    assert_eq!(con & regs::con::SPEED_MASK, regs::con::SPEED_STD);
    assert_eq!(con & regs::con::ADDR_10BIT_MASTER, 0, "never 10-bit");
}

#[test]
fn sda_hold_is_not_written_on_a_core_that_predates_the_register() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[]))
        .with_comp_version(regs::SDA_HOLD_MIN_VERSION - 1);
    let mut engine = I2cEngine::new(DwMaster::new(Integration::INTEL_LPSS, || {}));
    engine.probe(&mock).expect("probe");
    engine
        .configure(
            &mock,
            &BusConfig {
                speed: Speed::Fast,
                counts: None,
                input_clock_hz: 100_000_000,
                sda_hold: Some(0x42),
                irq_driven: true,
            },
        )
        .expect("configure");
    assert_eq!(mock.sda_hold(), 0, "the reserved address was left alone");
}

#[test]
fn a_write_then_read_reaches_the_device_as_one_transaction() {
    let mock = MockDwI2c::new(
        Integration::INTEL_LPSS,
        CannedDevice::new(0x2c, &[0xAA, 0xBB, 0xCC, 0xDD]),
    );
    let mut engine = bring_up(&mock).expect("bring-up");
    let mut rx = [0u8; 4];
    let n = engine
        .transfer(
            &mock,
            &Xfer::write_read(0x2c, &[0x01, 0x00], 4),
            &mut rx,
            &mut budget(),
        )
        .expect("transfer");

    assert_eq!(n, 4);
    assert_eq!(rx, [0xAA, 0xBB, 0xCC, 0xDD]);
    assert!(mock.clean(), "violations: {:?}", mock.violations());
    // One transaction, not two: a missing repeated START would have
    // released the bus and produced two.
    assert_eq!(mock.target_address(), 0x2c);
}

#[test]
fn a_multi_byte_read_asks_for_one_wake_not_one_per_byte() {
    // Regression: `IC_RX_TL` was left at 0 for the life of the controller, so
    // `RX_FULL` fired on every single received byte. On a Yoga's I2C-HID relay
    // that showed up as ~10 controller completion interrupts per GPIO-ready
    // signal — the whole reason I2C input cost visibly more than USB for the
    // same cursor.
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[0; 8]));
    let mut master = configured_master(&mock, Integration::INTEL_LPSS);
    let xfer = Xfer::read(0x2c, 8);
    master.start(&mock, &xfer).expect("start");

    // Eight reads are in flight and the FIFO is 16 deep, so the threshold
    // asks to be woken once, when all eight have landed.
    assert_eq!(
        mock.rx_threshold(),
        7,
        "RX_TL must track the reads in flight, not sit at 0"
    );
    // STOP_DET stays armed: it is the terminal wake that collects a short or
    // NAK'd read the threshold will never reach.
    assert_ne!(mock.interrupt_mask() & regs::intr::STOP_DET, 0);
}

#[test]
fn a_read_deeper_than_the_fifo_clamps_the_threshold_to_the_fifo() {
    // The threshold may never exceed what the FIFO can hold, or `RX_FULL`
    // could not fire before the hardware overran it.
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[0; 64]));
    let mut master = configured_master(&mock, Integration::INTEL_LPSS);
    let xfer = Xfer::read(0x2c, 64);
    master.start(&mock, &xfer).expect("start");

    let (_, rx_depth) = master.fifo_depths();
    assert_eq!(
        mock.rx_threshold() as usize,
        rx_depth - 1,
        "a read longer than the FIFO must clamp to the FIFO's depth"
    );
}

#[test]
fn a_deeper_read_still_delivers_every_byte() {
    // The threshold change must not lose the tail: drive a real transfer
    // longer than the FIFO and check every byte arrives.
    let payload: [u8; 40] = core::array::from_fn(|i| i as u8);
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &payload));
    let mut engine = bring_up(&mock).expect("bring-up");
    let mut rx = [0u8; 40];
    let n = engine
        .transfer(&mock, &Xfer::read(0x2c, 40), &mut rx, &mut budget())
        .expect("transfer");
    assert_eq!(n, 40);
    assert_eq!(rx, payload, "every byte survives the raised threshold");
    assert!(mock.clean(), "violations: {:?}", mock.violations());
}

#[test]
fn the_last_queued_command_keeps_rx_stop_and_abort_wakeups_but_not_tx_empty() {
    // Re-arming must work even when the mask register cannot read back.
    for integration in [Integration::INTEL_LPSS, Integration::AMD_PLAIN] {
        let mock = MockDwI2c::new(integration, CannedDevice::new(0x2c, &[1]));
        let mut master = configured_master(&mock, integration);
        let xfer = Xfer::read(0x2c, 1);
        master.start(&mock, &xfer).expect("start");
        let mask = mock.interrupt_mask();
        for bit in [
            regs::intr::STOP_DET,
            regs::intr::TX_ABRT,
            regs::intr::RX_FULL,
        ] {
            assert_ne!(mask & bit, 0, "mask {mask:#x} misses {bit:#x}");
        }
        assert_eq!(
            mask & regs::intr::TX_EMPTY,
            0,
            "all commands are already queued"
        );
        assert_eq!(
            mock.tx_empty_assertions(),
            0,
            "no IRQ before the initial fill"
        );
        assert!(
            mock.interrupt_asserted(),
            "the received byte/STOP wakes the owner"
        );
        let mut rx = [0; 1];
        assert!(matches!(
            master.poll(&mock, &xfer, &mut rx),
            XferState::Done { read: 1 }
        ));
        assert_eq!(rx, [1]);
        assert_eq!(mock.interrupt_mask(), 0, "completion quiesces every source");
        assert!(!mock.interrupt_asserted());
        // Completion does not race STOP by disabling a still-active core.
        assert!(mock.enabled());
        assert_ne!(
            mock.read32(regs::IC_RAW_INTR_STAT) & regs::intr::TX_EMPTY,
            0
        );
    }
}

#[test]
fn an_empty_tx_fifo_is_a_level_that_read_to_clear_cannot_acknowledge() {
    let mock = MockDwI2c::new(Integration::AMD_PLAIN, CannedDevice::new(0x2c, &[]));
    mock.write32(regs::IC_ENABLE, regs::enable::ENABLE);
    mock.write32(regs::IC_INTR_MASK, regs::intr::TX_EMPTY);
    assert!(mock.interrupt_asserted());
    let _ = mock.read32(regs::IC_CLR_INTR);
    assert!(
        mock.interrupt_asserted(),
        "an empty enabled FIFO keeps requesting IRQ"
    );
    mock.write32(regs::IC_INTR_MASK, 0);
    assert!(!mock.interrupt_asserted());
}

#[test]
fn a_successful_engine_transfer_leaves_no_idle_interrupt() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[1]));
    let mut engine = bring_up(&mock).expect("bring-up");
    assert_eq!(
        engine.transfer(&mock, &Xfer::read(0x2c, 1), &mut [0], &mut budget()),
        Ok(1)
    );
    assert!(mock.enabled(), "no abort recovery occurs after success");
    assert_eq!(mock.interrupt_mask(), 0);
    assert!(
        !mock.interrupt_asserted(),
        "the parked owner must not be interrupted forever"
    );
    assert_eq!(mock.tx_empty_assertions(), 0);
}

#[test]
fn rx_credit_stalls_keep_tx_empty_masked_until_rx_progress() {
    let reply = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mock = MockDwI2c::with_fifos(
        Integration::INTEL_LPSS,
        CannedDevice::new(0x2c, &reply),
        4,
        4,
    );
    let mut master = configured_master(&mock, Integration::INTEL_LPSS);
    let xfer = Xfer::read(0x2c, reply.len());
    master.start(&mock, &xfer).expect("start");
    let mut rx = [0; 9];
    for _ in 0..2 {
        assert_eq!(mock.interrupt_mask() & regs::intr::TX_EMPTY, 0);
        assert_ne!(mock.read32(regs::IC_INTR_STAT) & regs::intr::RX_FULL, 0);
        assert!(matches!(
            master.poll(&mock, &xfer, &mut rx),
            XferState::Busy { progressed: true }
        ));
    }
    assert_eq!(mock.interrupt_mask() & regs::intr::TX_EMPTY, 0);
    assert!(matches!(
        master.poll(&mock, &xfer, &mut rx),
        XferState::Done { read: 9 }
    ));
    assert_eq!(rx, reply);
    assert_eq!(
        mock.tx_empty_assertions(),
        0,
        "no interrupt for unavailable RX credit"
    );
    assert!(!mock.interrupt_asserted());
    assert!(mock.clean(), "violations: {:?}", mock.violations());
}

#[test]
fn tx_empty_rearms_only_until_the_final_fifo_fill_and_stop_still_wakes() {
    let mock = BufferedTx::new(CannedDevice::new(0x2c, &[]));
    let mut master = configured_master(&mock, Integration::INTEL_LPSS);
    let xfer = Xfer::write(0x2c, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    master.start(&mock, &xfer).expect("start");
    // Two full fills: more TX work remains, and emptying either one must
    // wake the owner to queue the next fill rather than deadlock the bus.
    for _ in 0..2 {
        assert_ne!(mock.core.interrupt_mask() & regs::intr::TX_EMPTY, 0);
        assert!(
            !mock.interrupt_asserted(),
            "a nonempty FIFO is below no TX threshold"
        );
        mock.drain_tx();
        assert_ne!(mock.read32(regs::IC_INTR_STAT) & regs::intr::TX_EMPTY, 0);
        assert!(matches!(
            master.poll(&mock, &xfer, &mut []),
            XferState::Busy { progressed: true }
        ));
    }
    // The final queued command carries STOP. TX empty alone has no more
    // work to request, but STOP must still be delivered after it executes.
    assert_eq!(mock.core.interrupt_mask() & regs::intr::TX_EMPTY, 0);
    assert!(!mock.interrupt_asserted());
    mock.drain_tx();
    assert_ne!(mock.read32(regs::IC_INTR_STAT) & regs::intr::STOP_DET, 0);
    assert!(matches!(
        master.poll(&mock, &xfer, &mut []),
        XferState::Done { read: 0 }
    ));
    assert_eq!(mock.core.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
    assert!(
        mock.core.clean(),
        "violations: {:?}",
        mock.core.violations()
    );
}

#[test]
fn a_valid_probe_masks_inherited_interrupts_before_configuration() {
    let mock = MockDwI2c::new(Integration::AMD_PLAIN, CannedDevice::new(0x2c, &[]));
    mock.write32(regs::IC_ENABLE, regs::enable::ENABLE);
    mock.write32(regs::IC_INTR_MASK, regs::intr::MASTER_MASK);
    assert!(mock.interrupt_asserted());
    let mut master = DwMaster::new(Integration::AMD_PLAIN, || {});
    master.probe(&mock).expect("probe");
    assert_eq!(mock.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
}

#[test]
fn a_failed_configure_handshake_still_masks_an_inherited_source() {
    let mock = MockDwI2c::new(Integration::AMD_PLAIN, CannedDevice::new(0x2c, &[])).stuck_enable();
    let mut master = DwMaster::new(Integration::AMD_PLAIN, || {});
    master.probe(&mock).expect("probe");
    mock.force_abort(1);
    mock.write32(regs::IC_INTR_MASK, regs::intr::MASTER_MASK);
    assert!(mock.interrupt_asserted());
    assert!(matches!(
        master.configure(&mock, &BusConfig::fast(133_000_000)),
        Err(I2cError::EnableHandshake(_))
    ));
    assert_eq!(mock.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
}

#[test]
fn start_refusal_masks_the_source_without_hiding_the_latched_abort() {
    let mock = MockDwI2c::new(Integration::AMD_PLAIN, CannedDevice::new(0x2c, &[]));
    let mut master = configured_master(&mock, Integration::AMD_PLAIN);
    mock.force_abort(1);
    mock.write32(regs::IC_INTR_MASK, regs::intr::MASTER_MASK);
    assert!(mock.interrupt_asserted());
    assert_eq!(
        master.start(&mock, &Xfer::read(0x2c, 1)),
        Err(I2cError::AbortPending)
    );
    assert_eq!(mock.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
    assert_eq!(
        mock.read32(regs::IC_TX_ABRT_SOURCE),
        1,
        "recovery still owns clearing the evidence"
    );
}

#[test]
fn an_aborted_poll_quiesces_before_the_engine_can_recover() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[1]));
    let mut master = configured_master(&mock, Integration::INTEL_LPSS);
    let xfer = Xfer::read(0x2c, 1);
    master.start(&mock, &xfer).expect("start");
    mock.force_abort(1 << 3);
    assert!(mock.interrupt_asserted());
    assert!(matches!(
        master.poll(&mock, &xfer, &mut [0]),
        XferState::Aborted(..)
    ));
    assert_eq!(mock.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
    assert_eq!(mock.read32(regs::IC_TX_ABRT_SOURCE), 1 << 3);
    master.recover(&mock);
    assert_eq!(mock.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
    assert_eq!(mock.read32(regs::IC_TX_ABRT_SOURCE), 0);
}

#[test]
fn a_bus_configured_without_interrupts_leaves_the_mask_clear() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[1]));
    let mut engine = I2cEngine::new(DwMaster::new(Integration::INTEL_LPSS, || {}));
    engine.probe(&mock).expect("probe");
    engine
        .configure(
            &mock,
            &BusConfig {
                irq_driven: false,
                ..BusConfig::fast(100_000_000)
            },
        )
        .expect("configure");
    let mut rx = [0u8; 1];
    engine
        .transfer(&mock, &Xfer::read(0x2c, 1), &mut rx, &mut budget())
        .expect("transfer");
    assert_eq!(mock.interrupt_mask(), 0);
}

#[test]
fn a_transfer_deeper_than_the_fifos_completes_across_several_polls() {
    // Depth 4 both ways against a 16-byte read and an 8-byte write: the
    // driver must keep queueing as the FIFOs drain, and must never queue
    // more read commands than the RX FIFO can hold.
    let reply: [u8; 16] = core::array::from_fn(|i| i as u8 + 1);
    let mock = MockDwI2c::with_fifos(
        Integration::INTEL_LPSS,
        CannedDevice::new(0x2c, &reply),
        4,
        4,
    );
    let mut engine = bring_up(&mock).expect("bring-up");
    let write: [u8; 8] = core::array::from_fn(|i| 0xf0 | i as u8);
    let mut rx = [0u8; 16];
    let n = engine
        .transfer(
            &mock,
            &Xfer::write_read(0x2c, &write, 16),
            &mut rx,
            &mut budget(),
        )
        .expect("transfer");

    assert_eq!(n, 16);
    assert_eq!(rx, reply);
    // No FIFO was overrun in either direction — the bug that only ever
    // appears on transfers larger than the FIFO.
    assert!(mock.clean(), "violations: {:?}", mock.violations());
}

#[test]
fn an_address_nack_is_distinguishable_from_every_other_abort() {
    let mut dev = CannedDevice::new(0x2c, &[]);
    dev.nack_addr = true;
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, dev);
    let mut engine = bring_up(&mock).expect("bring-up");
    let err = engine
        .transfer(&mock, &Xfer::write(0x2c, &[1]), &mut [], &mut budget())
        .unwrap_err();

    assert!(err.is_address_nack(), "{err:?}");
    match err {
        I2cError::Abort { detail, .. } => assert_eq!(detail.cause, DwAbortCause::AddrNack),
        other => panic!("{other:?}"),
    }
    assert_eq!(engine.stats().address_nacks, 1);
    assert_eq!(engine.stats().aborts, 0, "a NACK is not a fault");
}

#[test]
fn a_data_nack_is_a_fault_not_an_absent_device() {
    let mut dev = CannedDevice::new(0x2c, &[]);
    dev.nack_data = Some(1);
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, dev);
    let mut engine = bring_up(&mock).expect("bring-up");
    let err = engine
        .transfer(&mock, &Xfer::write(0x2c, &[1, 2]), &mut [], &mut budget())
        .unwrap_err();

    assert!(!err.is_address_nack(), "the device answered, then refused");
    assert_eq!(engine.stats().aborts, 1);
    assert_eq!(engine.stats().address_nacks, 0);
}

#[test]
fn an_abort_is_cleared_before_the_next_transfer_so_it_cannot_be_reread() {
    let mut dev = CannedDevice::new(0x2c, &[0x11]);
    dev.nack_addr = true;
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, dev);
    let mut engine = bring_up(&mock).expect("bring-up");

    // First transfer aborts.
    let _ = engine
        .transfer(&mock, &Xfer::write(0x2c, &[1]), &mut [], &mut budget())
        .unwrap_err();

    // The next transfer must reach the bus rather than being refused by
    // this driver's own uncleared abort, and must not queue a command
    // over a standing one. Both halves matter: `recover` clearing the
    // register is what makes the retry lawful, and the mock's violation
    // flag is what proves the clear actually happened rather than the
    // driver merely forgetting.
    let mut rx = [0u8; 1];
    let second = engine.transfer(&mock, &Xfer::read(0x2c, 1), &mut rx, &mut budget());
    assert!(
        !matches!(second, Err(I2cError::AbortPending)),
        "recover() left the abort latched; got {second:?}"
    );
    assert!(
        !mock
            .violations()
            .contains(Violation::CommandOverPendingAbort),
        "a command was queued over a standing abort: {:?}",
        mock.violations()
    );
    // This device NACKs unconditionally, so the retry NACKs too — the
    // point is that it was a *fresh* NACK, counted as a second one.
    assert_eq!(engine.stats().address_nacks, 2);
}

#[test]
fn a_start_refusal_is_reported_then_recovered_for_the_next_child() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[0x11]));
    let mut engine = bring_up(&mock).expect("bring-up");
    mock.force_abort(1); // 7-bit address NACK, already latched before start
    let mut rx = [0; 1];
    assert!(matches!(
        engine.transfer(&mock, &Xfer::read(0x2c, 1), &mut rx, &mut budget()),
        Err(I2cError::AbortPending)
    ));
    assert_eq!(engine.stats().refused, 1);
    assert_eq!(
        engine.transfer(&mock, &Xfer::read(0x2c, 1), &mut rx, &mut budget()),
        Ok(1)
    );
    assert_eq!(rx, [0x11]);
    assert!(
        !mock
            .violations()
            .contains(Violation::CommandOverPendingAbort)
    );
}

#[test]
fn a_transfer_that_never_stops_times_out_and_leaves_the_core_parked() {
    let mock =
        MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[0x11])).swallow_stop();
    let mut engine = bring_up(&mock).expect("bring-up");
    let mut rx = [0u8; 1];
    let err = engine
        .transfer(
            &mock,
            &Xfer::read(0x2c, 1),
            &mut rx,
            &mut Budget::new(4, || {}),
        )
        .unwrap_err();

    assert!(matches!(err, I2cError::Timeout { .. }), "{err:?}");
    assert_eq!(engine.stats().timeouts, 1);
    // A timed-out transfer must not leave the controller mid-sequence for
    // the next one to inherit.
    assert!(!mock.enabled(), "the core was parked disabled");
    assert_eq!(mock.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
}

#[test]
fn a_malformed_request_is_refused_before_any_register_is_touched() {
    let mock = MockDwI2c::new(Integration::INTEL_LPSS, CannedDevice::new(0x2c, &[]));
    let mut engine = bring_up(&mock).expect("bring-up");
    let before = mock.target_address();

    let err = engine
        .transfer(&mock, &Xfer::write(0x80, &[1]), &mut [], &mut budget())
        .unwrap_err();
    assert!(matches!(err, I2cError::Request(_)), "{err:?}");
    assert_eq!(mock.target_address(), before, "no register was written");
    assert_eq!(engine.stats().refused, 1);
    assert_eq!(engine.stats().transfers, 0);
}

#[test]
fn the_bus_trait_carries_the_same_transfers_a_protocol_driver_will_use() {
    let mock = MockDwI2c::new(
        Integration::INTEL_LPSS,
        CannedDevice::new(0x2c, &[0xDE, 0xAD]),
    );
    let mut engine = bring_up(&mock).expect("bring-up");
    let mut bus = engine.bus(&mock, budget());

    bus.write(0x2c, &[0x05, 0x00]).expect("write");
    let mut rx = [0u8; 2];
    let n = bus
        .write_read(0x2c, &[0x01, 0x00], &mut rx)
        .expect("write_read");
    assert_eq!((n, rx), (2, [0xDE, 0xAD]));
    let n = bus.read(0x2c, &mut rx).expect("read");
    assert_eq!(n, 2);

    assert_eq!(bus.stats().transfers, 3);
    assert!(mock.clean(), "violations: {:?}", mock.violations());
}

#[test]
fn steady_state_transfers_do_not_burn_idle_polls() {
    // The epic's "zero empty polls in steady state" criterion, as a unit
    // test: a transfer the FIFOs can satisfy immediately must complete
    // without the engine ever pacing.
    let mock = MockDwI2c::new(
        Integration::INTEL_LPSS,
        CannedDevice::new(0x2c, &[1, 2, 3, 4]),
    );
    let mut engine = bring_up(&mock).expect("bring-up");
    let mut rx = [0u8; 4];
    engine
        .transfer(
            &mock,
            &Xfer::write_read(0x2c, &[0x01], 4),
            &mut rx,
            // A zero-step budget: any pacing at all fails this.
            &mut Budget::new(0, || unreachable!("a steady-state transfer paced")),
        )
        .expect("transfer");
    assert_eq!(engine.stats().idle_polls, 0);
}

#[test]
fn deadline_pacing_allows_short_and_long_progress_producing_interrupt_streams() {
    let reply: [u8; 64] = core::array::from_fn(|i| i as u8 + 1);
    for read_len in [16usize, 64] {
        let mock = BufferedTx::new(CannedDevice::new(0x2c, &reply));
        let mut engine = I2cEngine::new(configured_master(&mock, Integration::INTEL_LPSS));
        let now = Cell::new(0u64);
        let fired = Cell::new(0usize);
        let mut pacing = DeadlinePacing::new(
            400_000,
            20_000_000,
            || Some(now.get()),
            |until| {
                let arrived = now.get() + 1_000_000;
                assert!(arrived < until, "this wake is Fired, not the deadline");
                now.set(arrived);
                assert_ne!(mock.read32(regs::IC_TXFLR), 0);
                mock.drain_tx();
                assert!(mock.interrupt_asserted(), "a real enabled DW source fired");
                fired.set(fired.get() + 1);
                true
            },
        );
        let mut rx = [0u8; 64];
        assert_eq!(
            engine.transfer(
                &mock,
                &Xfer::write_read(0x2c, &[0x20, 0], read_len),
                &mut rx[..read_len],
                &mut pacing,
            ),
            Ok(read_len),
        );
        assert_eq!(rx[..read_len], reply[..read_len]);
        assert_eq!(fired.get(), (read_len + 2).div_ceil(4));
        if read_len == 64 {
            assert!(fired.get() > 8, "healthy transfers may need many IRQs");
        }
        assert_eq!(engine.stats().timeouts, 0);
        assert_eq!(mock.core.interrupt_mask(), 0);
        assert!(!mock.interrupt_asserted());
        assert!(
            mock.core.clean(),
            "violations: {:?}",
            mock.core.violations()
        );
    }
}

#[test]
fn eight_wait_countdown_misclassifies_a_progressing_transfer_as_timeout() {
    // The former RoutePacing spent one of eight waits on every Fired wake.
    // Keep the counterexample beside the actual deadline-policy regression:
    // the very same wire is healthy, but eight callbacks move only 30/64 bytes.
    let reply: [u8; 64] = core::array::from_fn(|i| i as u8 + 1);
    let mock = BufferedTx::new(CannedDevice::new(0x2c, &reply));
    let mut engine = I2cEngine::new(configured_master(&mock, Integration::INTEL_LPSS));
    let fired = Cell::new(0usize);
    let mut pacing = Budget::new(8, || {
        assert_ne!(mock.read32(regs::IC_TXFLR), 0);
        mock.drain_tx();
        assert!(mock.interrupt_asserted());
        fired.set(fired.get() + 1);
    });
    let mut rx = [0u8; 64];
    assert!(matches!(
        engine.transfer(
            &mock,
            &Xfer::write_read(0x2c, &[0x20, 0], 64),
            &mut rx,
            &mut pacing,
        ),
        Err(I2cError::Timeout { .. })
    ));
    assert_eq!(fired.get(), 8);
    assert_eq!(rx[..30], reply[..30]);
    assert_eq!(rx[30..], [0; 34]);
    assert_eq!(engine.stats().idle_polls, 9);
    assert!(!mock.core.enabled());
    assert_eq!(mock.core.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
    assert!(
        mock.core.clean(),
        "violations: {:?}",
        mock.core.violations()
    );
}

#[test]
fn deadline_pacing_stops_no_progress_interrupts_at_the_original_deadline() {
    let mock = BufferedTx::new(CannedDevice::new(0x2c, &[0x11; 64]));
    let mut engine = I2cEngine::new(configured_master(&mock, Integration::INTEL_LPSS));
    let now = Cell::new(0u64);
    let fired = Cell::new(0u64);
    let deadline = u64::from(RateBudget::<fn()>::deadline_ms(400_000, 66)) * 1_000_000;
    let mut pacing = DeadlinePacing::new(
        400_000,
        20_000_000,
        || Some(now.get()),
        |until| {
            assert!(until <= deadline, "a Fired must not extend the deadline");
            let arrived = now.get() + 1_000_000;
            assert!(arrived <= until);
            now.set(arrived);
            fired.set(fired.get() + 1);
            // An unrelated/shared-route wake without this controller making
            // progress. No queued command is allowed onto the wire.
            true
        },
    );
    let mut rx = [0u8; 64];
    assert!(matches!(
        engine.transfer(
            &mock,
            &Xfer::write_read(0x2c, &[0x20, 0], 64),
            &mut rx,
            &mut pacing,
        ),
        Err(I2cError::Timeout { .. })
    ));
    assert_eq!(now.get(), deadline);
    assert_eq!(fired.get(), deadline / 1_000_000);
    assert_eq!(rx, [0; 64]);
    assert_eq!(engine.stats().timeouts, 1);
    assert!(!mock.core.enabled());
    assert_eq!(mock.core.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
    assert!(
        mock.core.clean(),
        "violations: {:?}",
        mock.core.violations()
    );
}

#[test]
fn deadline_pacing_polls_progress_when_the_interrupt_route_never_fires() {
    let reply: [u8; 64] = core::array::from_fn(|i| i as u8 + 1);
    let mock = BufferedTx::new(CannedDevice::new(0x2c, &reply));
    let mut engine = I2cEngine::new(configured_master(&mock, Integration::INTEL_LPSS));
    let now = Cell::new(0u64);
    let deadlines = Cell::new(0usize);
    let mut pacing = DeadlinePacing::new(
        400_000,
        20_000_000,
        || Some(now.get()),
        |until| {
            assert_eq!(until, now.get() + 20_000_000);
            now.set(until);
            mock.drain_tx();
            deadlines.set(deadlines.get() + 1);
            true // Deadline, not Fired: the engine must still inspect the FIFO.
        },
    );
    let mut rx = [0u8; 64];
    assert_eq!(
        engine.transfer(
            &mock,
            &Xfer::write_read(0x2c, &[0x20, 0], 64),
            &mut rx,
            &mut pacing,
        ),
        Ok(64)
    );
    assert_eq!(deadlines.get(), 17);
    assert_eq!(now.get(), 340_000_000);
    assert_eq!(rx, reply);
    assert_eq!(mock.core.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
    assert!(
        mock.core.clean(),
        "violations: {:?}",
        mock.core.violations()
    );
}

#[test]
fn deadline_pacing_inspects_final_completion_at_the_transfer_deadline() {
    let mock = BufferedTx::new(CannedDevice::new(0x2c, &[0x11, 0x22]));
    let mut engine = I2cEngine::new(configured_master(&mock, Integration::INTEL_LPSS));
    let now = Cell::new(0u64);
    let waits = Cell::new(0usize);
    let deadline = u64::from(RateBudget::<fn()>::deadline_ms(400_000, 4)) * 1_000_000;
    let mut pacing = DeadlinePacing::new(
        400_000,
        1_000_000_000,
        || Some(now.get()),
        |until| {
            assert_eq!(until, deadline, "poll interval is clamped to the budget");
            now.set(until);
            mock.drain_tx();
            waits.set(waits.get() + 1);
            true
        },
    );
    let mut rx = [0u8; 2];
    assert_eq!(
        engine.transfer(
            &mock,
            &Xfer::write_read(0x2c, &[0x20, 0], 2),
            &mut rx,
            &mut pacing,
        ),
        Ok(2),
        "a completed final poll wins over deadline expiration"
    );
    assert_eq!(waits.get(), 1);
    assert_eq!(rx, [0x11, 0x22]);
    assert_eq!(mock.core.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
    assert!(
        mock.core.clean(),
        "violations: {:?}",
        mock.core.violations()
    );
}

#[test]
fn deadline_pacing_wait_refusal_recovers_the_active_controller() {
    let mock = BufferedTx::new(CannedDevice::new(0x2c, &[0x11]));
    let mut engine = I2cEngine::new(configured_master(&mock, Integration::INTEL_LPSS));
    let waits = Cell::new(0usize);
    let mut pacing = DeadlinePacing::new(
        400_000,
        20_000_000,
        || Some(0),
        |_| {
            waits.set(waits.get() + 1);
            false
        },
    );
    assert!(matches!(
        engine.transfer(&mock, &Xfer::read(0x2c, 1), &mut [0], &mut pacing),
        Err(I2cError::Timeout { .. })
    ));
    assert_eq!(waits.get(), 1);
    assert_eq!(engine.stats().timeouts, 1);
    assert!(!mock.core.enabled());
    assert_eq!(mock.core.interrupt_mask(), 0);
    assert!(!mock.interrupt_asserted());
    assert!(
        mock.core.clean(),
        "violations: {:?}",
        mock.core.violations()
    );
}
