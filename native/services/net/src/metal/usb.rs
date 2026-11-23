//! **The USB composition** — the xHCI class binding over `kore-usbnet`, and
//! what its bound controller offers the row's generic device host.
//!
//! `kore-usbnet` owns every USB sequence and `service` owns every frame
//! decision; this module owns neither. It is the wiring between them and the
//! kernel: realize a claimed controller into the driver seams, find the
//! adaptor on it, and lend the backend one drive cycle at a time. Everything
//! here is exercised without hardware: the USB ladder by `kore-usbnet`'s
//! mock, the class binding by the driver host's own tests.

use kore_driver::host::{
    CapPool, CapRegs, ClassQuiesce, DeviceBinding, DeviceResources, DmaActive, DmaPrepared,
    InterruptBlock, InterruptPolicy, PciBindingContract,
};
use kore_usbnet::device::{Discovery, EcmNic, discover};
use kore_usbnet::{EcmDevice, Error as UsbNetError};
use kore_xhci::{Controller, Error as XhciError, PointerRegisterAccess, PreparedController};

use super::{NetDeviceHost, NicBound};
use crate::note;

/// Ports the founding walk will *examine* before giving up.
///
/// Bring-up talks to the device, so a hub of unresponsive ports must not be
/// able to hold the boot chain open behind this row.
const DISCOVERY_BUDGET: u8 = 8;

// ── the class binding ──────────────────────────────────────────────────

/// xHCI after its DMA graph is programmed while Run/Stop and interrupter
/// enable remain clear. The generic host owns the only transition from here.
pub struct XhciNetPrepared {
    controller: PreparedController<CapRegs, CapPool>,
}

/// The bound owner: the running controller, the adaptor found on it (when the
/// machine has one), and the capability resource set.
///
/// `ecm` is an `Option` rather than a bind failure on purpose. A controller
/// with no Ethernet adaptor plugged in is a *fact about the machine*, not a
/// wiring fault, and a driver row that refused to found on it would take the
/// boot chain down for a missing dongle.
pub struct XhciNetBound {
    ecm: Option<EcmDevice<CapPool>>,
    /// The root port the adaptor answered on — or, when one refused its
    /// bring-up ladder, the port it refused on.  Rides the readiness note's
    /// detail bits either way.
    port: u8,
    /// The core rung a connected Ethernet function refused at, when one did.
    ///
    /// `None` covers both "an adaptor came up" and "no port carries an
    /// Ethernet function"; [`has_adaptor`](Self::has_adaptor) separates
    /// those.  Keeping the rung is what lets the row note
    /// `BRING_UP_BASE + r` instead of `NO_ADAPTOR` — the walk is the only
    /// place that can still tell "nothing is plugged in" from "the thing
    /// plugged in would not come up".
    refused: Option<u64>,
    controller: Controller<CapRegs, CapPool>,
    resources: DeviceResources<DmaActive>,
}

impl XhciNetBound {
    /// Whether an adaptor was found and brought up.
    pub fn has_adaptor(&self) -> bool {
        self.ecm.is_some()
    }

    /// The core rung a connected Ethernet function refused its bring-up at,
    /// when one did.  See [`XhciNetBound::refused`].
    pub fn bring_up_rung(&self) -> Option<u64> {
        self.refused
    }
}

impl NicBound for XhciNetBound {
    type Backend<'a> = EcmNic<'a, CapRegs, CapPool>;

    /// Borrow the frame backend for one drive cycle.
    ///
    /// Three disjoint borrows of one owner: the registers out of the resource
    /// set, the controller, and the device's ECM state. That is exactly why
    /// the backend is a *view* — the controller is shared with whatever else
    /// is plugged into this xHC, and a backend that owned it would make those
    /// devices unreachable.
    fn backend(&mut self) -> Option<EcmNic<'_, CapRegs, CapPool>> {
        let regs = self.resources.bar(0)?;
        let ecm = self.ecm.as_mut()?;
        Some(EcmNic::new(
            &mut self.controller,
            regs,
            ecm,
            kore_driver::host::clock_now_ns,
            kore_driver::host::relax,
        ))
    }

    fn resources(&self) -> &DeviceResources<DmaActive> {
        &self.resources
    }

    /// Three outcomes, not two.  A controller whose ports carry no Ethernet
    /// function at all is `NO_ADAPTOR` — the cable is in the other port
    /// group, and moving it is the fix.  A controller carrying one that
    /// refused its ladder is a different fact, and telling an operator to
    /// move a cable that is already right is the worst answer available: the
    /// rung says where it stopped.
    fn founding_note(&self) -> u64 {
        if self.has_adaptor() {
            0
        } else if let Some(rung) = self.bring_up_rung() {
            note::BRING_UP_BASE + rung
        } else {
            note::NO_ADAPTOR
        }
    }

    /// The root port — the only way to tell one controller's ports from
    /// another's after the fact.
    fn detail(&self) -> u8 {
        self.port
    }
}

/// The frame engine's host over the xHCI class: the generic device host,
/// instantiated.
pub type EcmHost = NetDeviceHost<XhciNetBinding>;

/// The class declaration: one BAR, one interrupt block, DMA.
///
/// Identical to the input router's `XhciBinding` in everything the *host*
/// reads, because it is the same silicon — what differs is which core the
/// start hook composes over it. Two rows, two cores, one class.
pub struct XhciNetBinding;

impl PciBindingContract for XhciNetBinding {
    const BARS: kore_rt::service::PciBarMask = kore_rt::service::PciBarMask::BAR0;
    const INTERRUPTS: InterruptPolicy = InterruptPolicy::Required(InterruptBlock::SINGLE);
}

impl DeviceBinding for XhciNetBinding {
    type Pci = Self;
    type Config = PointerRegisterAccess;
    type Prepared = XhciNetPrepared;
    type Bound = XhciNetBound;
    type Error = XhciError;

    fn prepare(
        resources: &mut DeviceResources<DmaPrepared>,
        pointer_access: Self::Config,
    ) -> Result<Self::Prepared, XhciError> {
        let regs = resources.bar(0).ok_or(XhciError::Unsupported)?;
        let pool = resources.pool().ok_or(XhciError::Unsupported)?;
        let controller = Controller::prepare_with_pointer_access(
            regs,
            pool,
            pointer_access,
            kore_driver::host::clock_now_ns,
            kore_driver::host::relax,
        )?;
        Ok(XhciNetPrepared { controller })
    }

    fn start(
        prepared: Self::Prepared,
        resources: DeviceResources<DmaActive>,
    ) -> Result<Self::Bound, XhciError> {
        let regs = resources.bar(0).ok_or(XhciError::Unsupported)?;
        let pool = resources.pool().ok_or(XhciError::Unsupported)?;
        let mut controller = prepared.controller.start(
            regs,
            kore_driver::host::clock_now_ns,
            kore_driver::host::relax,
        )?;
        let found = discover(
            &mut controller,
            regs,
            pool,
            DISCOVERY_BUDGET,
            kore_driver::host::clock_now_ns,
            kore_driver::host::relax,
        );
        let (port, ecm, refused) = match found {
            Discovery::Found(port, ecm) => (port, Some(ecm), None),
            Discovery::Refused(port, error) => (port, None, Some(usbnet_rung(error))),
            Discovery::Absent => (0, None, None),
        };
        Ok(XhciNetBound {
            ecm,
            port,
            refused,
            controller,
            resources,
        })
    }

    fn quiesce(bound: &mut Self::Bound) -> Result<ClassQuiesce, XhciError> {
        let regs = bound.resources.bar(0).ok_or(XhciError::Unsupported)?;
        bound.controller.quiesce(
            regs,
            kore_driver::host::clock_now_ns,
            kore_driver::host::relax,
        )?;
        Ok(ClassQuiesce::Stopped)
    }

    fn unbind(bound: Self::Bound) -> DeviceResources<DmaActive> {
        let XhciNetBound {
            ecm,
            controller,
            resources,
            ..
        } = bound;
        drop(ecm);
        drop(controller);
        resources
    }
}

/// The USB core's refusal, widened for the row's note map.
pub fn usbnet_rung(error: UsbNetError) -> u64 {
    match error {
        UsbNetError::Probe(_) => 1,
        UsbNetError::Usb(_) => 2,
        UsbNetError::BadMacString => 3,
        UsbNetError::DmaAlloc => 4,
        UsbNetError::Exhausted => 5,
    }
}
