//! **HID boot protocol** — the class driver that turns an enumerated,
//! endpoint-configured keyboard or mouse into decoded input events.
//!
//! The boot protocol is the scoping trick that avoids a full HID
//! report-descriptor parser: a device advertising interface class 3 /
//! subclass 1 (boot) / protocol 1 (keyboard) | 2 (mouse) speaks a **fixed**
//! report format (USB HID 1.11 Appendix B).  After SET_CONFIGURATION the
//! driver issues **SET_PROTOCOL(boot)** — devices default to *report*
//! protocol per §7.2.6, so a non-parsing driver must select boot to be
//! correct — then arms the interrupt IN endpoint and polls it.
//!
//! The transfer path reuses the substrate: arming stages a single Normal TRB
//! on the endpoint's [`SubRing`](kore_driver::SubRing) and rings the slot doorbell;
//! a completed report arrives as a Transfer Event on interrupter 0, polled
//! non-blocking ([`Controller::poll_interrupt`]).

use kore_driver::seam::{Dma, DmaPool, Regs64};

use xhci::ring::trb::transfer;

use crate::Controller;
use crate::device::{ControlSubmission, Device, EndpointKind, Setup};
use crate::{Error, Result};

/// bmRequestType for a HID class request to an interface (host-to-device).
const REQ_CLASS_INTERFACE_OUT: u8 = 0x21;
/// bRequest SET_REPORT (HID 1.11 §7.2.2).
const SET_REPORT: u8 = 0x09;
/// SET_REPORT's wValue for report type Output (high byte 2), report ID 0 —
/// the boot keyboard's one output report carries no ID.
const OUTPUT_REPORT_BOOT: u16 = 0x0200;
/// bRequest SET_IDLE (HID 1.11 §7.2.4).
const SET_IDLE: u8 = 0x0A;
/// bRequest SET_PROTOCOL (HID 1.11 §7.2.6).
const SET_PROTOCOL: u8 = 0x0B;
/// wValue for the boot protocol.
const PROTOCOL_BOOT: u16 = 0;
/// wValue for the report protocol.
const PROTOCOL_REPORT: u16 = 1;

/// A parsed **keyboard boot report** (8 bytes: modifier bitmap, a reserved
/// byte, then up to six concurrently-pressed usage codes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyboardReport {
    /// The modifier bitmap (bit 0 = LeftCtrl … bit 7 = RightGUI).
    pub modifiers: u8,
    /// Up to six pressed HID usage codes (0 = empty slot).
    pub keys: [u8; 6],
}

impl KeyboardReport {
    /// Parse an 8-byte boot report; `None` if too short.
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < 8 {
            return None;
        }
        Some(Self {
            modifiers: b[0],
            keys: [b[2], b[3], b[4], b[5], b[6], b[7]],
        })
    }

    /// The non-zero pressed usage codes.
    pub fn pressed(&self) -> impl Iterator<Item = u8> + '_ {
        self.keys.iter().copied().filter(|&k| k != 0)
    }
}

/// A parsed **mouse boot report** (3+ bytes: button bitmap, signed X/Y
/// deltas, and — beyond the 3-byte boot minimum — an optional wheel delta).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MouseReport {
    /// Button bitmap (bit 0 = left, bit 1 = right, bit 2 = middle).
    pub buttons: u8,
    /// Signed X movement.
    pub dx: i8,
    /// Signed Y movement.
    pub dy: i8,
    /// Signed wheel movement (0 if the report is the 3-byte minimum).
    pub wheel: i8,
}

impl MouseReport {
    /// Parse a boot report (≥ 3 bytes); `None` if too short.
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < 3 {
            return None;
        }
        Some(Self {
            buttons: b[0],
            dx: b[1] as i8,
            dy: b[2] as i8,
            wheel: if b.len() > 3 { b[3] as i8 } else { 0 },
        })
    }
}

impl<R: Regs64, P: DmaPool> Controller<R, P> {
    /// **SET_PROTOCOL** — select the boot (`true`) or report (`false`)
    /// protocol for `interface`.
    pub fn set_protocol(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        boot: bool,
        interface: u16,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<()> {
        let req = Setup {
            request_type: REQ_CLASS_INTERFACE_OUT,
            request: SET_PROTOCOL,
            value: if boot { PROTOCOL_BOOT } else { PROTOCOL_REPORT },
            index: interface,
        };
        self.control(regs, dev, req, 0, clock, pause).map(|_| ())
    }

    /// **SET_REPORT(Output)** carrying a boot keyboard's LED byte to
    /// `interface`, submitted without waiting
    /// ([`control_submit`](Self::control_submit)).
    ///
    /// The byte is the boot output report (HID 1.11 Appendix B.1): bit 0
    /// Num Lock, bit 1 Caps Lock, bit 2 Scroll Lock, bit 3 Compose, bit 4
    /// Kana. It goes over the default control pipe because a boot
    /// keyboard is not obliged to have an interrupt OUT endpoint, and the
    /// request is the one Linux 7.2.11 `drivers/hid/usbhid/usbkbd.c` sends
    /// (`bRequestType` class|interface, `bRequest` 0x09, `wValue` 0x0200,
    /// `wLength` 1).
    pub fn submit_set_leds(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        interface: u16,
        leds: u8,
    ) -> Result<ControlSubmission> {
        dev.write_data(&[leds])?;
        let req = Setup {
            request_type: REQ_CLASS_INTERFACE_OUT,
            request: SET_REPORT,
            value: OUTPUT_REPORT_BOOT,
            index: interface,
        };
        self.control_submit(regs, dev, req, 1)
    }

    /// **SET_IDLE** — `duration` in 4 ms units (0 = report only on change),
    /// for `interface`.
    pub fn set_idle(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        duration: u8,
        interface: u16,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<()> {
        let req = Setup {
            request_type: REQ_CLASS_INTERFACE_OUT,
            request: SET_IDLE,
            value: (duration as u16) << 8,
            index: interface,
        };
        self.control(regs, dev, req, 0, clock, pause).map(|_| ())
    }

    /// **Arm** the interrupt IN endpoint: stage one Normal TRB (the report
    /// buffer, `mps` bytes, interrupt-on-completion, routed to interrupter
    /// `target`) and ring the slot doorbell for the endpoint's DCI.  Re-arm
    /// after each report.  `target` = 0 is the controller's primary
    /// interrupter (poll via [`Controller::poll_interrupt`]); a secondary
    /// interrupter's index routes the completion to that core's private event
    /// ring (poll via [`Interrupter::poll`](crate::Interrupter::poll)).
    pub fn arm_interrupt(&mut self, regs: &R, dev: &mut Device<P>, target: u16) -> Result<()> {
        let dci = dev.int_ep().ok_or(Error::Unsupported)?.dci;
        self.arm_interrupt_at(regs, dev, dci, target).map(|_| ())
    }

    /// Arm one named interrupt-IN pipe of an affine, possibly composite,
    /// device. Returns the submitted TRB address for completion correlation;
    /// the caller retains it with the interface-to-DCI association.
    pub fn arm_interrupt_at(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        dci: u8,
        target: u16,
    ) -> Result<u64> {
        let slot_id = dev.slot_id;
        let int = dev.endpoint_mut(dci).ok_or(Error::Unsupported)?;
        if int.kind != EndpointKind::InterruptIn {
            return Err(Error::Unsupported);
        }
        // An interrupt endpoint always carries its report buffer; take the
        // address before staging so the immutable borrow of `data` ends
        // before the ring needs `int` mutably.
        let data = int.data.as_ref().ok_or(Error::Unsupported)?;
        if int.mps == 0 || usize::from(int.mps) > data.len() {
            return Err(Error::Unsupported);
        }
        let report = data.iova().0;
        let mut normal = transfer::Normal::new();
        normal
            .set_data_buffer_pointer(report)
            .set_trb_transfer_length(int.mps as u32)
            .set_interrupt_on_completion()
            .set_interrupter_target(target);
        let index = int.ring.stage(normal.into());
        let submitted = int.ring.slot_iova(index).0;
        let dci = int.dci;
        let db = self.caps().doorbell(slot_id);
        int.ring.ring(|_tail| regs.write32(db, dci as u32));
        Ok(submitted)
    }

    /// **Poll** (non-blocking) for a completed interrupt report.  `Some(len)`
    /// when the armed transfer finished — the report is in the endpoint's
    /// buffer ([`Device::read_report`]); re-arm to receive the next.  `None`
    /// when nothing has arrived.
    pub fn poll_interrupt(&mut self, regs: &R, dev: &Device<P>) -> Option<usize> {
        let event = self.poll_transfer(regs)?;
        if !event.data_completed() {
            return None;
        }
        let int = dev.int_ep()?;
        if event.slot_id == dev.slot_id && event.endpoint_id == int.dci {
            Some((int.mps as usize).saturating_sub(event.residual as usize))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyboard_report_parses_modifiers_and_keys() {
        // Left-Shift held (modifier bit 1), 'a' (usage 0x04) and 'b' (0x05).
        let raw = [0x02, 0x00, 0x04, 0x05, 0, 0, 0, 0];
        let r = KeyboardReport::parse(&raw).unwrap();
        assert_eq!(r.modifiers, 0x02);
        let pressed: alloc::vec::Vec<u8> = r.pressed().collect();
        assert_eq!(pressed, alloc::vec![0x04, 0x05]);
    }

    #[test]
    fn mouse_report_parses_buttons_and_deltas() {
        // Left button, +5 X, -3 Y.
        let raw = [0x01, 5, (-3i8) as u8];
        let r = MouseReport::parse(&raw).unwrap();
        assert_eq!(r.buttons, 0x01);
        assert_eq!(r.dx, 5);
        assert_eq!(r.dy, -3);
        assert_eq!(r.wheel, 0, "3-byte report has no wheel");
    }
}
