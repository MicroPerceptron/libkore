//! Capability-host binding for the fixed HostLink VirtIO console.
//!
//! The generic host maps exactly BAR4 and realizes MSI-X entries 0..=1. This
//! adapter observes the kernel-validated, redacted VirtIO PCI capability rows
//! through the still-held device token, then moves BAR/DMA seams into the safe
//! owning console. No PCI configuration address or physical BAR address is
//! exposed here.

use kore_abi::session::VirtioPciCapReport;
use kore_driver::host::{
    CapPool, CapRegs, ClassQuiesce, DeviceBinding, DeviceResources, DmaActive, DmaPrepared,
    InterruptBlock, InterruptPolicy, PciBindingContract,
};
use kore_rt::cap::{RouteToken, verbs};
use kore_rt::service::PciBarMask;

use crate::console::{ConsoleError, Plane};
use crate::owned::{FixedConsole, OwnedError, PAYLOAD_BYTES, PreparedConsole, QUEUE_SIZE};
use crate::pci::{MappedBars, ModernPciTransport, PciLayout, TransportError};
use crate::split::QueueError;

const BAR: u8 = 4;
const MAX_CAP_REPORTS: usize = 16;
const EMPTY_REPORT: VirtioPciCapReport = VirtioPciCapReport {
    cfg_type: 0,
    bar: 0,
    id: 0,
    offset: 0,
    length: 0,
    notify_multiplier: 0,
};

type Transport = ModernPciTransport<CapRegs>;
type Prepared = PreparedConsole<Transport, CapPool, QUEUE_SIZE, PAYLOAD_BYTES>;
type Console = FixedConsole<Transport, CapPool, QUEUE_SIZE, PAYLOAD_BYTES>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VirtioConsoleBindError {
    MissingDevice,
    MissingBar,
    MissingPool,
    TooManyCapabilityReports(u8),
    LayoutObservation(verbs::VirtioPciLayoutError),
    Transport(TransportError),
    Console(OwnedError),
}

impl From<TransportError> for VirtioConsoleBindError {
    fn from(value: TransportError) -> Self {
        Self::Transport(value)
    }
}

impl From<OwnedError> for VirtioConsoleBindError {
    fn from(value: OwnedError) -> Self {
        Self::Console(value)
    }
}

fn queue_refusal_code(error: QueueError) -> u8 {
    match error {
        QueueError::InvalidSize => 0x20,
        QueueError::AllocationFailed => 0x21,
        QueueError::MisalignedMemory => 0x22,
        QueueError::InvalidSpan => 0x23,
        QueueError::Saturated => 0x24,
        QueueError::StaleToken => 0x25,
        QueueError::InvalidUsedIndex => 0x26,
        QueueError::InvalidUsedId(_) => 0x27,
        QueueError::DuplicateCompletion(_) => 0x28,
        QueueError::UsedLength { .. } => 0x29,
        QueueError::Memory => 0x2a,
        QueueError::Poisoned => 0x2b,
    }
}

fn transport_refusal_code(error: TransportError) -> u8 {
    match error {
        TransportError::MissingBar(_) => 0x40,
        TransportError::InvalidRegion => 0x41,
        TransportError::ResetTimeout => 0x43,
        TransportError::MissingFeatures { required, offered } => {
            let missing = required & !offered;
            0x80 | u8::from(missing & crate::pci::F_CONSOLE_MULTIPORT != 0)
                | (u8::from(missing & crate::pci::F_VERSION_1 != 0) << 1)
                | (u8::from(missing & crate::pci::F_ACCESS_PLATFORM != 0) << 2)
        }
        TransportError::FeaturesRejected => 0x45,
        TransportError::QueueOutOfRange(_) => 0x46,
        TransportError::QueueUnavailable(_) => 0x47,
        TransportError::QueueTooSmall { .. } => 0x48,
        TransportError::QueueAlreadyEnabled(_) => 0x49,
        TransportError::QueueMemoryMisaligned(_) => 0x4a,
        TransportError::VectorRejected { .. } => 0x4b,
        TransportError::NotifyOutOfRange(_) => 0x4c,
        TransportError::ConfigUnstable => 0x4d,
        TransportError::DeviceFailed => 0x4e,
    }
}

fn console_refusal_code(error: ConsoleError) -> u8 {
    match error {
        ConsoleError::Truncated => 0x60,
        ConsoleError::MaxPorts(_) => 0x61,
        ConsoleError::NotReady => 0x62,
        ConsoleError::AlreadyReady => 0x63,
        ConsoleError::WrongDirection(_) => 0x64,
        ConsoleError::UnsupportedEvent(_) => 0x65,
        ConsoleError::UnexpectedPort(_) => 0x66,
        ConsoleError::DuplicatePort(_) => 0x67,
        ConsoleError::UnknownPort(_) => 0x68,
        ConsoleError::EventOrder { .. } => 0x69,
        ConsoleError::InvalidValue { .. } => 0x6a,
        ConsoleError::UnexpectedPayload(_) => 0x6b,
        ConsoleError::InvalidName => 0x6c,
        ConsoleError::UnexpectedName(_) => 0x6d,
        ConsoleError::DuplicateName(_) => 0x6e,
        ConsoleError::DuplicateEvent { .. } => 0x6f,
        ConsoleError::StaleHandle => 0x70,
    }
}

fn owned_refusal_code(error: OwnedError) -> u8 {
    match error {
        OwnedError::WouldBlock => 0x10,
        OwnedError::Backpressure => 0x11,
        OwnedError::Disconnected(Plane::Control) => 0x12,
        OwnedError::Disconnected(Plane::Bulk) => 0x13,
        OwnedError::InvalidBufferSize => 0x14,
        OwnedError::Queue(error) => queue_refusal_code(error),
        OwnedError::Transport(error) => transport_refusal_code(error),
        OwnedError::Console(error) => console_refusal_code(error),
        OwnedError::ResetRequired => 0x7f,
    }
}

pub struct VirtioConsolePrepared {
    console: Prepared,
}

/// Complete affine owner of the live console and the generic host resources
/// needed for IRQ parking and eventual containment.
pub struct VirtioConsoleBound {
    console: Console,
    resources: DeviceResources<DmaActive>,
}

impl VirtioConsoleBound {
    pub fn console_and_route(&mut self) -> (&mut Console, Option<&RouteToken>) {
        (&mut self.console, self.resources.route())
    }

    pub fn console(&mut self) -> &mut Console {
        &mut self.console
    }

    pub fn route(&self) -> Option<&RouteToken> {
        self.resources.route()
    }
}

pub struct VirtioConsoleBinding;

impl PciBindingContract for VirtioConsoleBinding {
    const BARS: PciBarMask = PciBarMask::BAR4;
    const INTERRUPTS: InterruptPolicy = InterruptPolicy::Required(InterruptBlock::MSIX_PAIR);
}

impl DeviceBinding for VirtioConsoleBinding {
    type Pci = Self;
    type Config = ();

    type Prepared = VirtioConsolePrepared;
    type Bound = VirtioConsoleBound;
    type Error = VirtioConsoleBindError;

    fn prepare(
        resources: &mut DeviceResources<DmaPrepared>,
        _config: (),
    ) -> Result<Self::Prepared, Self::Error> {
        let device = resources
            .device()
            .ok_or(VirtioConsoleBindError::MissingDevice)?;
        let count = verbs::virtio_pci_cap_count(device)
            .map_err(VirtioConsoleBindError::LayoutObservation)?;
        if usize::from(count) > MAX_CAP_REPORTS {
            return Err(VirtioConsoleBindError::TooManyCapabilityReports(count));
        }
        let mut reports = [EMPTY_REPORT; MAX_CAP_REPORTS];
        for index in 0..count {
            reports[usize::from(index)] = verbs::virtio_pci_cap(device, index)
                .map_err(VirtioConsoleBindError::LayoutObservation)?;
        }
        let layout = PciLayout::from_reports(&reports[..usize::from(count)])?;

        let bar = resources
            .take_bar(BAR)
            .ok_or(VirtioConsoleBindError::MissingBar)?;
        let Some(pool) = resources.take_pool() else {
            resources.restore_bar(BAR, bar);
            return Err(VirtioConsoleBindError::MissingPool);
        };
        let mut bars = MappedBars::new();
        let bar_len = bar.len();
        bars.insert(BAR, bar, bar_len)
            .map_err(|_| VirtioConsoleBindError::MissingBar)?;
        let transport = ModernPciTransport::new(bars, layout)?;
        match PreparedConsole::prepare(transport, pool) {
            Ok(console) => Ok(VirtioConsolePrepared { console }),
            Err((error, transport, pool)) => {
                let mut bars = transport.into_bars();
                if let Some(bar) = bars.take(BAR) {
                    resources.restore_bar(BAR, bar);
                }
                resources.restore_pool(pool);
                Err(error.into())
            }
        }
    }

    fn start(
        prepared: Self::Prepared,
        mut resources: DeviceResources<DmaActive>,
    ) -> Result<Self::Bound, Self::Error> {
        match prepared.console.start() {
            Ok(console) => Ok(VirtioConsoleBound { console, resources }),
            Err((error, transport, pool)) => {
                let mut bars = transport.into_bars();
                if let Some(bar) = bars.take(BAR) {
                    resources.restore_bar(BAR, bar);
                }
                resources.restore_pool(pool);
                Err(error.into())
            }
        }
    }

    fn quiesce(bound: &mut Self::Bound) -> Result<ClassQuiesce, Self::Error> {
        bound
            .console
            .quiesce()
            .map(|()| ClassQuiesce::Stopped)
            .map_err(Into::into)
    }

    fn unbind(bound: Self::Bound) -> DeviceResources<DmaActive> {
        let VirtioConsoleBound {
            console,
            mut resources,
        } = bound;
        let (transport, pool) = console.into_parts();
        let mut bars = transport.into_bars();
        resources.restore_bar(
            BAR,
            bars.take(BAR)
                .expect("VirtIO console lost its declared BAR owner"),
        );
        resources.restore_pool(pool);
        resources
    }

    fn refusal_code(error: &Self::Error) -> u8 {
        match error {
            VirtioConsoleBindError::MissingDevice => 1,
            VirtioConsoleBindError::MissingBar => 2,
            VirtioConsoleBindError::MissingPool => 3,
            VirtioConsoleBindError::TooManyCapabilityReports(_) => 4,
            VirtioConsoleBindError::LayoutObservation(_) => 5,
            VirtioConsoleBindError::Transport(error) => transport_refusal_code(*error),
            VirtioConsoleBindError::Console(error) => owned_refusal_code(*error),
        }
    }
}
