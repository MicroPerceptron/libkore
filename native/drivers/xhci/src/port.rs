//! The **root-hub port state machine** — PORTSC, driven directly (no fake
//! USB hub layer; the hardware exposes the ports as registers, so a
//! from-scratch driver treats them as registers).
//!
//! The one subtlety is **RW1CS**: PORTSC packs level bits (PP), a
//! write-1-to-disable bit (PED), the reset trigger (PR), and a cluster of
//! write-1-to-**clear** change bits.  Every write must leave PED / PR / the
//! change bits as 0 unless it means to act on them — [`PRESERVE`] is that
//! mask, and [`Controller::write_portsc`] the one safe writer.

use kore_driver::Budget;
use kore_driver::seam::{DmaPool, Regs64};

use crate::Controller;
use crate::device::Speed;

// PORTSC bit fields (xHCI 5.4.8).
const CCS: u32 = 1 << 0; // Current Connect Status (RO)
const PED: u32 = 1 << 1; // Port Enabled/Disabled (RW1CS: write 1 disables)
const PR: u32 = 1 << 4; // Port Reset (RW1S)
const SPEED_SHIFT: u32 = 10; // Port Speed [13:10] (RO)
const SPEED_MASK: u32 = 0xf << SPEED_SHIFT;
const CSC: u32 = 1 << 17; // Connect Status Change (RW1C)
const PEC: u32 = 1 << 18; // Port Enabled/Disabled Change (RW1C)
const WRC: u32 = 1 << 19; // Warm Reset Change (RW1C)
const OCC: u32 = 1 << 20; // Over-current Change (RW1C)
const PRC: u32 = 1 << 21; // Port Reset Change (RW1C)
const PLC: u32 = 1 << 22; // Port Link State Change (RW1C)
const CEC: u32 = 1 << 23; // Config Error Change (RW1C)

/// Every RW1C change bit — writing 1 to any clears it.
const CHANGES: u32 = CSC | PEC | WRC | OCC | PRC | PLC | CEC;

/// The bits a PORTSC write must carry through untouched writing **0**:
/// clearing this mask off the read value leaves PED / PR / the change bits
/// at 0 (no accidental disable / reset / change-clear), and preserves PP and
/// the rest.
const PRESERVE: u32 = !(PED | PR | CHANGES);

/// A **snapshot** of one root-hub port's PORTSC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Port {
    /// Port number (1-based, as slot contexts and the spec name it).
    pub number: u8,
    raw: u32,
}

impl Port {
    /// A device is attached.
    pub fn connected(&self) -> bool {
        self.raw & CCS != 0
    }
    /// The port is enabled (post-reset for USB2; auto for USB3).
    pub fn enabled(&self) -> bool {
        self.raw & PED != 0
    }
    /// A connect-status change is latched (needs clearing / debounce).
    pub fn connect_changed(&self) -> bool {
        self.raw & CSC != 0
    }
    /// The raw Port Speed field (protocol-speed-ID; 1=Full … 4=Super with
    /// the default PSI mapping).
    pub fn speed_id(&self) -> u8 {
        ((self.raw & SPEED_MASK) >> SPEED_SHIFT) as u8
    }
    /// The decoded [`Speed`].
    pub fn speed(&self) -> Speed {
        Speed::from_psi(self.speed_id())
    }
    /// A coarse [`PortState`] for reporting.
    pub fn state(&self) -> PortState {
        if !self.connected() {
            PortState::Disconnected
        } else if self.enabled() {
            PortState::Enabled
        } else {
            PortState::Connected
        }
    }
}

/// A port's coarse state (for boot-test reporting).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortState {
    /// Nothing attached.
    Disconnected,
    /// Attached but not yet enabled (USB2 pre-reset).
    Connected,
    /// Attached and enabled (ready to Address Device).
    Enabled,
}

impl<R: Regs64, P: DmaPool> Controller<R, P> {
    /// The number of root-hub ports.
    pub fn port_count(&self) -> u8 {
        self.caps().max_ports
    }

    /// Snapshot port `n` (1-based).
    pub fn port(&self, regs: &R, n: u8) -> Port {
        Port {
            number: n,
            raw: regs.read32(self.caps().portsc(n - 1)),
        }
    }

    /// Write PORTSC for port `n` with `action` bits, preserving PP and never
    /// disturbing PED / PR / the change bits (they stay 0 unless `action`
    /// sets them deliberately).
    fn write_portsc(&self, regs: &R, n: u8, action: u32) {
        let off = self.caps().portsc(n - 1);
        let base = regs.read32(off) & PRESERVE;
        regs.write32(off, base | action);
    }

    /// **Reset** port `n` and return the negotiated [`Speed`] once enabled.
    /// USB2 ports require the reset to enable + latch a valid speed; USB3
    /// ports typically self-train (already enabled at connect), and a reset
    /// is harmless recovery.  Clears the connect + reset change bits after.
    pub fn reset_port(
        &self,
        regs: &R,
        n: u8,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> crate::Result<Speed> {
        // Assert reset.
        self.write_portsc(regs, n, PR);
        // Wait for Port Reset Change (reset complete).
        //
        // Clocked, like every other wait in this driver: USB 2.0 §7.1.7.5 puts
        // reset signalling at 10-20 ms and recovery at 10 ms after it, so the
        // bound worth enforcing is a duration. The 500 000 probes this was
        // could be under a millisecond against a live PORTSC — short enough to
        // abandon a reset that was still running and report `NotReady` for a
        // device that is fine — and unbounded against a port whose register
        // reads stall.
        let off = self.caps().portsc(n - 1);
        let mut done = false;
        let mut budget = Budget::new(
            crate::controller::BRINGUP_BUDGET_NS,
            crate::controller::MAX_POLLS,
            clock,
        );
        while budget.may_wait() {
            if regs.read32(off) & PRC != 0 {
                done = true;
                break;
            }
            pause();
        }
        // The final paced interval may be the one the reset completes in;
        // sample once more before declaring the bounded wait exhausted.
        done |= regs.read32(off) & PRC != 0;
        if !done {
            return Err(crate::Error::NotReady);
        }
        // Acknowledge the reset + connect changes (RW1C).  Clearing CSC here
        // is what makes CSC a trustworthy "the attached device changed"
        // signal for the enumeration memo: after our own reset the port
        // reports PRC, never CSC, so a caller can tell its own side effect
        // apart from a real plug or unplug.
        self.write_portsc(regs, n, PRC | CSC);
        // Enabled now?
        let p = self.port(regs, n);
        if !p.enabled() {
            return Err(crate::Error::NotReady);
        }
        Ok(p.speed())
    }

    /// **Consume** a latched Connect Status Change on port `n` (RW1C).
    ///
    /// A change bit that is never cleared stays latched forever, so a caller
    /// that treats CSC as "a device was plugged or unplugged" must consume
    /// the edge once it has acted on it — otherwise the same plug is
    /// reported on every subsequent look.  Only CSC is written, so a reset
    /// or link-state change latched alongside it survives for whoever owns
    /// that bit.
    pub fn ack_connect_change(&self, regs: &R, n: u8) {
        self.write_portsc(regs, n, CSC);
    }
}
