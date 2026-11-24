//! Capability-owned controller workers and fair, stepped child initialization.
//! No diagnostic on this path writes to UART or a synchronous console.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
pub use kore_abi::input_status::{
    MAX_I2C_CHILDREN as MAX_CHILDREN, MAX_I2C_CONTROLLERS as MAX_CONTROLLERS,
};
use kore_abi::{
    cap::{ResourceKind, Rights},
    session::I2cHidReport,
    svc::InputReport,
    task::role,
};
use kore_abi::{
    input_status::{I2cChildStatus, I2cInputOutcome, I2cProtocolPhase},
    session::I2cReadyRouteState,
};
use kore_driver::host::CapRegs;
use kore_i2c::{
    AbortDetail, BusConfig, Counts, DeadlinePacing, I2cBus, I2cEngine, I2cError, I2cMaster,
    MAX_XFER_BYTES, Pacing, RateBudget, RecoveryOutcome, Speed,
};
use kore_i2c_dw::{DwMaster, Integration};
use kore_i2c_geni::GeniMaster;
#[cfg(test)]
use kore_i2c_hid::parse::build_table;
use kore_i2c_hid::{
    bringup::{BringUp, BringUpError, Phase, Quirks, Step},
    features::{FeatureError, FeaturePhase, FeatureSetup},
    framing::{self, Frame},
    parse::{ParseError, build_device},
    table::{ExtractionTable, MAX_REPORTS},
};
use kore_ring::spsc;
use kore_rt::{
    cap::{RouteToken, verbs},
    service::Worker,
    thread::WorkerCap,
};

pub const SOURCE_I2C_BASE: u8 = 0x80;
pub const SOURCES_PER_CONTROLLER: usize = MAX_CHILDREN * MAX_REPORTS;
const _: () = assert!(SOURCE_I2C_BASE as usize + MAX_CONTROLLERS * SOURCES_PER_CONTROLLER <= 256);
const SERVICE_KICK_PEER: u16 = 2;
pub const RELAY_CAPACITY: u32 = 128;
const RELAY_BYTES: usize = spsc::region_bytes::<InputReport>(RELAY_CAPACITY);
/// Controller-completion liveness slice, not the HID report cadence.  The
/// child-ready GPIO owns input latency; a 1 ms slice visibly destabilized dense
/// trackpad delivery once GPIO delivery was live (metal, 2026-08-31).
const XFER_WAIT_MS: u64 = 20;
const INIT_BUDGET_MS: u64 = 2_000;
/// Controllers without an admitted HID-ready pad retain the proven bounded
/// polling path. This is also the bring-up cadence before child setup settles.
const GPIO_BACKSTOP_MS: u64 = 16;
/// A routed steady-state pad still receives a sparse poll. It recovers a
/// missed/disabled edge without paying the old 62.5 Hz controller traffic.
const GPIO_IRQ_WATCHDOG_MS: u64 = 250;
/// Once a watchdog drain proves the ready pad did not announce queued input,
/// poll at input latency until a real GPIO wake proves the line healthy again.
const GPIO_DEGRADED_POLL_MS: u64 = GPIO_BACKSTOP_MS;
/// Frames one wake may read from a child that keeps answering with decodable
/// input. Only a pass that fills this bound is a backlog ("saturated") and
/// earns the 1 ms re-poll; an empty frame or an undecodable one ends the pass
/// at the ordinary cadence — a device that never answers empty (metal,
/// 2026-09-03: report id 0xff, 4 bytes, forever) must not turn the bound into
/// a bus-speed poll storm that also bypasses the GPIO wait.
const MAX_DRAIN_FRAMES: usize = 16;
const REPORT_RESERVE: u32 = kore_abi::pointer::MAX_FRAME_RECORDS as u32;
const _: () = assert!(REPORT_RESERVE >= MAX_REPORTS as u32 && REPORT_RESERVE <= RELAY_CAPACITY);

/// Controller implementation and capability species travel together. PCI
/// diagnostic bits are never interpreted as a platform controller model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ControllerKind {
    PlatformDesignWare = 1,
    PlatformGeni = 2,
    PciDesignWare = 3,
}

impl ControllerKind {
    pub const fn from_platform_model(model: u32) -> Option<Self> {
        match model {
            kore_abi::device::DeviceRecord::PLATFORM_I2C_DESIGNWARE => {
                Some(Self::PlatformDesignWare)
            }
            kore_abi::device::DeviceRecord::PLATFORM_I2C_GENI => Some(Self::PlatformGeni),
            _ => None,
        }
    }

    const fn from_published(model: u32) -> Option<Self> {
        match model {
            1 => Some(Self::PlatformDesignWare),
            2 => Some(Self::PlatformGeni),
            3 => Some(Self::PciDesignWare),
            _ => None,
        }
    }
}

pub const fn source_base(controller: usize, child: usize) -> u8 {
    assert!(controller < MAX_CONTROLLERS && child < MAX_CHILDREN);
    (SOURCE_I2C_BASE as usize + controller * SOURCES_PER_CONTROLLER + child * MAX_REPORTS) as u8
}

static PREPARED: [AtomicBool; MAX_CONTROLLERS] =
    [const { AtomicBool::new(false) }; MAX_CONTROLLERS];
static MODELS: [AtomicU32; MAX_CONTROLLERS] = [const { AtomicU32::new(0) }; MAX_CONTROLLERS];
static MMIO_BASES: [AtomicU64; MAX_CONTROLLERS] = [const { AtomicU64::new(0) }; MAX_CONTROLLERS];
static MMIO_LENGTHS: [AtomicU64; MAX_CONTROLLERS] = [const { AtomicU64::new(0) }; MAX_CONTROLLERS];
static BINDING_COUNTS: [AtomicU32; MAX_CONTROLLERS] =
    [const { AtomicU32::new(0) }; MAX_CONTROLLERS];

#[repr(align(64))]
struct SharedBindings(core::cell::UnsafeCell<[Option<I2cHidReport>; MAX_CHILDREN]>);

// SAFETY: `prepare` is called by thread zero before workers spawn. It
// publishes the completed array with `PREPARED.store(Release)`; the sole
// worker reads that publication with `Acquire` and the array is never mutated
// again.
unsafe impl Sync for SharedBindings {}

static BINDINGS: [SharedBindings; MAX_CONTROLLERS] =
    [const { SharedBindings(core::cell::UnsafeCell::new([None; MAX_CHILDREN])) }; MAX_CONTROLLERS];

#[repr(align(64))]
struct SharedRegion(core::cell::UnsafeCell<[u8; RELAY_BYTES]>);
// SAFETY: each indexed slot is initialized once before its sole worker starts.
// Only that worker's producer and thread 0's consumer access its SPSC.
unsafe impl Sync for SharedRegion {}
impl SharedRegion {
    unsafe fn view(&self) -> &'static mut [u8] {
        unsafe { &mut *self.0.get() }
    }
}
static REGIONS: [SharedRegion; MAX_CONTROLLERS] =
    [const { SharedRegion(core::cell::UnsafeCell::new([0; RELAY_BYTES])) }; MAX_CONTROLLERS];

struct ChildStats {
    identity: AtomicU32, // address | phase << 8 | last_error << 16 | GPIO flag << 24
    polls: AtomicU32,
    reports: AtomicU32,
    errors: AtomicU32,
    protocol: AtomicU32,
    last_input: AtomicU32,
}
impl ChildStats {
    const fn new() -> Self {
        Self {
            identity: AtomicU32::new(0),
            polls: AtomicU32::new(0),
            reports: AtomicU32::new(0),
            errors: AtomicU32::new(0),
            protocol: AtomicU32::new(0),
            last_input: AtomicU32::new(0),
        }
    }
    fn phase(&self, phase: u8, error: u8) {
        let old = self.identity.load(Ordering::Relaxed);
        let error = if error == 0 { (old >> 16) as u8 } else { error };
        self.identity.store(
            (old & 0xff00_00ff) | (phase as u32) << 8 | (error as u32) << 16,
            Ordering::Release,
        );
    }
    fn snapshot(&self) -> I2cChildStatus {
        let id = self.identity.load(Ordering::Acquire);
        I2cChildStatus {
            address: id as u8,
            phase: (id >> 8) as u8,
            error: (id >> 16) as u8,
            flags: (id >> 24) as u8,
            polls: self.polls.load(Ordering::Relaxed).into(),
            reports: self.reports.load(Ordering::Relaxed).into(),
            errors: self.errors.load(Ordering::Relaxed).into(),
            protocol: self.protocol.load(Ordering::Relaxed).into(),
            last_input: self.last_input.load(Ordering::Relaxed).into(),
        }
    }

    /// This worker is the sole writer. Publish before entering a potentially
    /// parking call, so a call that never returns cannot masquerade as the
    /// previous completed protocol step. Restore its exact evidence afterward.
    fn during<T>(&self, phase: I2cProtocolPhase, call: impl FnOnce() -> T) -> T {
        let previous = self.protocol.load(Ordering::Relaxed);
        self.protocol.store(phase.pack(0, 0), Ordering::Release);
        let result = call();
        self.protocol.store(previous, Ordering::Release);
        result
    }

    fn observe_step(&self, child: &mut Child, result: Result<(), u8>) {
        if let Err(error) = result {
            if child.protocol >> 16 == 0 {
                // Ladder details have bit 15 clear. A relay category uses
                // the high bit, so truncation can never disguise one as the
                // other. Keep this exact publication in the host regressions.
                child.protocol |= u32::from(0x8000 | u16::from(error)) << 16;
            }
        }
        self.protocol.store(child.protocol, Ordering::Relaxed);
    }
}
struct RelayStats {
    rung: AtomicU32,
    state: AtomicU64,
    transfers: AtomicU64,
    count: AtomicU32,
    controller_irqs: AtomicU32,
    gpio_irqs: AtomicU32,
    watchdog_deadlines: AtomicU32,
    dropped: AtomicU64,
    children: [ChildStats; MAX_CHILDREN],
}
impl RelayStats {
    const fn new() -> Self {
        Self {
            rung: AtomicU32::new(rung::NO_DEVICE),
            state: AtomicU64::new(rung::NO_DEVICE as u64),
            transfers: AtomicU64::new(0),
            count: AtomicU32::new(0),
            controller_irqs: AtomicU32::new(0),
            gpio_irqs: AtomicU32::new(0),
            watchdog_deadlines: AtomicU32::new(0),
            dropped: AtomicU64::new(0),
            children: [const { ChildStats::new() }; MAX_CHILDREN],
        }
    }

    fn note_controller_irq(&self) {
        saturating_increment(&self.controller_irqs);
    }

    fn note_gpio_irq(&self) {
        saturating_increment(&self.gpio_irqs);
    }

    fn note_watchdog_deadline(&self) {
        saturating_increment(&self.watchdog_deadlines);
    }

    fn progress(&self, phase: u8, detail: u32) {
        // A successful sibling may advance independently, but must not erase
        // the last refusal from the legacy controller-wide diagnostic row.
        if ((self.state.load(Ordering::Relaxed) >> 16) & 0xff) != 0 {
            return;
        }
        self.state.store(
            state_word(
                self.rung.load(Ordering::Acquire),
                phase,
                failure::NONE,
                detail,
                0,
            ),
            Ordering::Release,
        );
    }

    fn record_failure(&self, phase: u8, class: u8, detail: u32, recovery: Option<RecoveryOutcome>) {
        self.state.store(
            state_word(
                self.rung.load(Ordering::Acquire),
                phase,
                class,
                detail,
                recovery_code(recovery),
            ),
            Ordering::Release,
        );
    }
}

fn saturating_increment(counter: &AtomicU32) {
    let _ = counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        Some(value.saturating_add(1))
    });
}
static RELAYS: [RelayStats; MAX_CONTROLLERS] = [const { RelayStats::new() }; MAX_CONTROLLERS];

/// Legacy summary rows; the paginated I2cStatus verb exposes all controllers
/// and children, including their independent failures.
pub fn books() -> [crate::service::I2cRelayBooks; 2] {
    core::array::from_fn(|index| {
        let stats = &RELAYS[index];
        crate::service::I2cRelayBooks {
            state: stats.state.load(Ordering::Acquire),
            transfers: stats.transfers.load(Ordering::Relaxed).min(u32::MAX as u64) as u32,
            reports: stats.children.iter().fold(0u32, |n, c| {
                n.saturating_add(c.reports.load(Ordering::Relaxed))
            }),
        }
    })
}

fn relay<const C: usize>() -> &'static RelayStats {
    &RELAYS[C]
}

pub mod rung {
    pub const UNSTARTED: u32 = 1;
    pub const NO_CONTEXT: u32 = 2;
    pub const NO_DEVICE: u32 = 3;
    pub const BAD_DEVICE: u32 = 4;
    pub const NO_BINDINGS: u32 = 5;
    pub const NO_RESOURCES: u32 = 6;
    pub const NO_KICK: u32 = 7;
    pub const NO_QUEUE: u32 = 8;
    pub const CONTROLLER: u32 = 9;
    pub const TOO_MANY_BINDINGS: u32 = 11;
    pub const CLAIM_FAILED: u32 = 12;
    /// The broker found matching controllers, but its claim ledger already
    /// owns them. Distinct from absent hardware (`NO_DEVICE`).
    pub const CLAIM_TAKEN: u32 = 13;
}
mod phase {
    pub const CLAIM: u8 = 1;
    pub const PROBE: u8 = 2;
    pub const CONFIGURE: u8 = 3;
    pub const DESCRIPTOR: u8 = 4;
    pub const POWER: u8 = 5;
    pub const RESET: u8 = 6;
    pub const RESET_ACK: u8 = 7;
    pub const REPORT_DESCRIPTOR: u8 = 8;
    pub const TABLE: u8 = 9;
    pub const STEADY: u8 = 10;
    pub const FEATURE_GET: u8 = 11;
    pub const FEATURE_SET: u8 = 12;
}

mod failure {
    pub const NONE: u8 = 0;
    pub const IDENTITY: u8 = 1;
    pub const ENABLE: u8 = 2;
    pub const ABORT: u8 = 3;
    pub const TIMEOUT: u8 = 4;
    pub const ABORT_PENDING: u8 = 5;
    pub const REQUEST: u8 = 6;
    pub const HID: u8 = 7;
    pub const SHORT: u8 = 8;
    pub const TABLE: u8 = 9;
    pub const CLOCK: u8 = 10;
    /// The owning task could not derive/map the controller window. Detail is
    /// [`kore_rt::cap::verbs::BarMapFault::code`].
    pub const MAP: u8 = 11;
    /// Pool-free register/route intake failed; detail is IntakeError's code.
    pub const INTAKE: u8 = 12;
}

const fn recovery_code(outcome: Option<RecoveryOutcome>) -> u8 {
    match outcome {
        None => 0,
        Some(RecoveryOutcome::Clean) => 1,
        Some(RecoveryOutcome::Cleared) => 2,
        Some(RecoveryOutcome::Cancelled) => 3,
        Some(RecoveryOutcome::Aborted) => 4,
        Some(RecoveryOutcome::Failed(_)) => 5,
    }
}

const fn state_word(rung: u32, phase: u8, failure: u8, detail: u32, recovery: u8) -> u64 {
    (rung as u64 & 0xff)
        | ((phase as u64) << 8)
        | ((failure as u64) << 16)
        | ((detail as u64) << 24)
        | ((recovery as u64) << 56)
}

fn publish_phase<const SLOT: usize>(phase: u8) {
    relay::<SLOT>().progress(phase, 0);
}

fn gpio_ready_detail(state: I2cReadyRouteState) -> u32 {
    match state {
        I2cReadyRouteState::NotDescribed => kore_abi::input_status::I2C_READY_POLL_NO_FIRMWARE_IRQ,
        I2cReadyRouteState::NoCommunity => kore_abi::input_status::I2C_READY_POLL_NO_COMMUNITY,
        I2cReadyRouteState::NoLayout => kore_abi::input_status::I2C_READY_POLL_NO_LAYOUT,
        I2cReadyRouteState::NoAperture => kore_abi::input_status::I2C_READY_POLL_NO_APERTURE,
        I2cReadyRouteState::NoParentLine => kore_abi::input_status::I2C_READY_POLL_NO_PARENT_LINE,
        I2cReadyRouteState::PadOutOfRange => {
            kore_abi::input_status::I2C_READY_POLL_PAD_OUT_OF_RANGE
        }
        I2cReadyRouteState::NoCatalogDevice => {
            kore_abi::input_status::I2C_READY_POLL_NO_CATALOG_DEVICE
        }
        I2cReadyRouteState::NoRegisterView => {
            kore_abi::input_status::I2C_READY_POLL_NO_REGISTER_VIEW
        }
        I2cReadyRouteState::QuiesceRefused => {
            kore_abi::input_status::I2C_READY_POLL_QUIESCE_REFUSED
        }
        I2cReadyRouteState::ParentRouteRefused => {
            kore_abi::input_status::I2C_READY_POLL_PARENT_ROUTE_REFUSED
        }
        I2cReadyRouteState::Ready => kore_abi::input_status::I2C_READY_POLL_PAD_BIND_REFUSED,
        I2cReadyRouteState::SetupUnavailable => {
            kore_abi::input_status::I2C_READY_POLL_SETUP_UNAVAILABLE
        }
        I2cReadyRouteState::PadConfigurationRefused => {
            kore_abi::input_status::I2C_READY_POLL_PAD_CONFIGURATION_REFUSED
        }
        I2cReadyRouteState::PadAliasRefused => {
            kore_abi::input_status::I2C_READY_POLL_PAD_ALIAS_REFUSED
        }
    }
}

fn publish_steady<const SLOT: usize>(devices: &[Option<Child>; MAX_CHILDREN], ready_route: bool) {
    let line = devices.iter().flatten().any(|device| {
        matches!(
            device.binding.ready_interrupt,
            Some(kore_abi::session::I2cReadyInterruptReport::Line { .. })
        )
    });
    let detail = if ready_route && line {
        kore_abi::input_status::I2C_READY_LINE_IRQ
    } else if ready_route {
        kore_abi::input_status::I2C_READY_GPIO_IRQ
    } else if line {
        kore_abi::input_status::I2C_READY_POLL_LINE_BIND_REFUSED
    } else if let Some(state) = devices
        .iter()
        .flatten()
        .find(|device| device.binding.ready_interrupt.is_some())
        .map(|device| device.binding.ready_route_state)
    {
        gpio_ready_detail(state)
    } else {
        kore_abi::input_status::I2C_READY_POLL_NO_FIRMWARE_IRQ
    };
    relay::<SLOT>().progress(phase::STEADY, detail);
}

fn publish_failure<const SLOT: usize>(
    phase: u8,
    class: u8,
    detail: u32,
    recovery: Option<RecoveryOutcome>,
) {
    relay::<SLOT>().record_failure(phase, class, detail, recovery);
}

trait CompactI2cError {
    fn publish<const SLOT: usize>(&self, phase: u8);
}

impl<A: AbortDetail> CompactI2cError for I2cError<A> {
    fn publish<const SLOT: usize>(&self, phase: u8) {
        let (class, detail) = match self {
            Self::Identity { found, .. } => (failure::IDENTITY, *found),
            Self::EnableHandshake(snapshot) => (failure::ENABLE, snapshot.0),
            Self::UnsupportedInputClock(hz) => (failure::CLOCK, *hz),
            Self::Abort { detail, .. } => (failure::ABORT, detail.raw()),
            Self::Timeout { snapshot, .. } => (failure::TIMEOUT, snapshot.0),
            Self::AbortPending => (failure::ABORT_PENDING, 0),
            Self::Request(_) => (failure::REQUEST, 0),
        };
        publish_failure::<SLOT>(phase, class, detail, self.recovery());
    }
}

pub const WORKERS: [Worker; MAX_CONTROLLERS] = [
    Worker {
        name: "i2c-hid-0",
        entry: run::<0>,
    },
    Worker {
        name: "i2c-hid-1",
        entry: run::<1>,
    },
    Worker {
        name: "i2c-hid-2",
        entry: run::<2>,
    },
    Worker {
        name: "i2c-hid-3",
        entry: run::<3>,
    },
];
pub fn note_claim(controller: usize, code: u32) {
    let stats = &RELAYS[controller];
    stats.rung.store(code, Ordering::Release);
    let state = stats.state.load(Ordering::Relaxed);
    stats
        .state
        .store((state & !0xff) | u64::from(code), Ordering::Release);
}
pub fn diagnostic(row: u8) -> Option<kore_abi::input_status::I2cDiagnostic> {
    use kore_abi::input_status::{I2C_STATUS_ROWS, I2cDiagnostic};
    if row >= I2C_STATUS_ROWS {
        return None;
    }
    let controller = row as usize / (MAX_CHILDREN + 1);
    let child = row as usize % (MAX_CHILDREN + 1);
    let stats = &RELAYS[controller];
    Some(if child == 0 {
        controller_diagnostic(row, stats)
    } else {
        let s = stats.children[child - 1].snapshot();
        I2cDiagnostic {
            row,
            state: s.phase,
            address_or_count: s.address,
            error: s.error,
            flags: s.flags,
            counters: [
                s.polls.get(),
                s.reports.get(),
                s.errors.get(),
                s.protocol.get(),
                s.last_input.get(),
            ],
        }
    })
}

fn controller_diagnostic(row: u8, stats: &RelayStats) -> kore_abi::input_status::I2cDiagnostic {
    use kore_abi::input_status::I2cDiagnostic;

    I2cDiagnostic {
        row,
        state: stats.rung.load(Ordering::Acquire) as u8,
        address_or_count: stats.count.load(Ordering::Relaxed) as u8,
        flags: 1,
        counters: [
            stats.controller_irqs.load(Ordering::Relaxed),
            stats.dropped.load(Ordering::Relaxed).min(u32::MAX as u64) as u32,
            // The rung names that a resource was refused; only the state
            // word says which. Carry its failure class and the low 24 bits
            // of that failure's detail — for INTAKE, the IntakeError class —
            // so a claim refusal stays attributable without a serial console.
            {
                let state = stats.state.load(Ordering::Acquire);
                let failure = (state >> 16) & 0xff;
                let detail = (state >> 24) & 0xff_ffff;
                (failure | (detail << 8)) as u32
            },
            stats.gpio_irqs.load(Ordering::Relaxed),
            stats.watchdog_deadlines.load(Ordering::Relaxed),
        ],
        ..I2cDiagnostic::default()
    }
}
pub fn endowment(device: kore_abi::session::SlotRef, controller: ControllerKind) -> WorkerCap {
    WorkerCap {
        role: role::DEVICE,
        index: 0,
        subject: device,
        kind: match controller {
            ControllerKind::PciDesignWare => ResourceKind::PciDevice as u16 as u8,
            ControllerKind::PlatformDesignWare | ControllerKind::PlatformGeni => {
                ResourceKind::PlatformDevice as u16 as u8
            }
        },
        rights: Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
    }
}
pub fn kick_endowment(kick: kore_abi::session::SlotRef) -> WorkerCap {
    WorkerCap {
        role: role::DOORBELL,
        index: SERVICE_KICK_PEER,
        subject: kick,
        kind: ResourceKind::ChanSend as u16 as u8,
        rights: verbs::NOTICE_RIGHTS,
    }
}
/// Every worker borrows its owner's mapped register window. DesignWare also
/// needs a derived controller capability for the worker's routed completion
/// line; GENI is paced without one.
pub const fn worker_needs_device(controller: ControllerKind) -> bool {
    !matches!(controller, ControllerKind::PlatformGeni)
}

/// Publish one controller's immutable binding and borrowed register view, then
/// form thread zero's queue half before the worker exists.
///
/// # Safety
/// Calls for a slot must be serialized by its sole owner, before spawning
/// that slot's worker. Once prepared, further calls refuse without touching
/// the queue or its published bindings.
/// Any mapping must remain owned and mapped until that worker has stopped.
pub unsafe fn prepare(
    index: usize,
    controller: ControllerKind,
    bindings: [Option<I2cHidReport>; MAX_CHILDREN],
    count: usize,
    mapping: Option<(*mut u8, u64)>,
) -> Option<spsc::Consumer<'static, InputReport>> {
    if index >= MAX_CONTROLLERS
        || count == 0
        || count > MAX_CHILDREN
        || PREPARED[index].load(Ordering::Acquire)
    {
        return None;
    }
    if mapping.is_none_or(|(base, len)| base.is_null() || len == 0) {
        note_claim(index, rung::NO_RESOURCES);
        return None;
    }
    RELAYS[index].count.store(count as u32, Ordering::Relaxed);
    let region = &REGIONS[index];
    // SAFETY: the sole owner calls before `spawn_workers`, and PREPARED's
    // guard refuses subsequent calls; no worker can read this array yet.
    unsafe { *BINDINGS[index].0.get() = bindings };
    BINDING_COUNTS[index].store(count as u32, Ordering::Relaxed);
    MODELS[index].store(controller as u32, Ordering::Relaxed);
    let (base, len) = mapping.map_or((core::ptr::null_mut(), 0), |value| value);
    MMIO_LENGTHS[index].store(len, Ordering::Relaxed);
    // Release publishes every scalar and the binding array together.
    MMIO_BASES[index].store(base as usize as u64, Ordering::Release);
    // SAFETY: called once by `start_on_metal`, before `spawn_workers`.
    let window = unsafe { region.view() };
    spsc::init::<InputReport>(window, RELAY_CAPACITY).ok()?;
    // SAFETY: this is the sole consumer handle.
    let consumer = spsc::consumer::<InputReport>(unsafe { region.view() }).ok()?;
    PREPARED[index].store(true, Ordering::Release);
    Some(consumer)
}

/// Preserve the exact owner-side mapping refusal for the shell photo.
pub fn map_failed(index: usize, code: u8) {
    if index >= MAX_CONTROLLERS {
        return;
    }
    let relay = &RELAYS[index];
    let rung = rung::NO_RESOURCES;
    relay.rung.store(rung, Ordering::Release);
    relay.state.store(
        state_word(rung, phase::CLAIM, failure::MAP, u32::from(code), 0),
        Ordering::Release,
    );
}

/// Freeze the published tail before popping. Producers publish whole frames
/// atomically, so this boundary cannot split one. Chasing a live producer up to
/// a record quota could leave a prefix pending when we switch controllers.
pub fn published_batch<'a>(
    rx: &'a mut spsc::Consumer<'_, InputReport>,
) -> impl Iterator<Item = InputReport> + 'a {
    assert!(rx.capacity() <= RELAY_CAPACITY);
    let count = rx.outstanding();
    (0..count).map_while(move |_| rx.pop())
}

/// IRQs are progress opportunities, not elapsed-time units: RX_TL=0 may
/// deliver one for every received byte. Keep one rate-aware transfer deadline
/// while retaining a short polling backstop for missed completion interrupts.
fn route_pacing<'a>(route: &'a RouteToken, stats: &'a RelayStats, bus_hz: u32) -> impl Pacing + 'a {
    DeadlinePacing::new(
        bus_hz,
        XFER_WAIT_MS * 1_000_000,
        monotonic_ns,
        move |deadline_ns| match verbs::wait_interrupt_until(route, deadline_ns) {
            Ok(verbs::InterruptWait::Fired) => {
                stats.note_controller_irq();
                true
            }
            // Let the engine poll once after a deadline wake, including the
            // final slice. The fixed deadline refuses the next empty wait.
            Ok(verbs::InterruptWait::Deadline) => true,
            _ => false,
        },
    )
}
fn now_ms() -> Option<u64> {
    // Reading time must not park the controller worker twice per drain pass.
    // The founder clock is the same monotonic domain as WAIT_IRQ deadlines.
    kore_rt::clock::uptime()
        .ok()
        .map(|time| time.as_nanos() / 1_000_000)
        .or_else(|| verbs::sleep_until(1).map(|word| word >> 32))
}

/// Monotonic nanoseconds for the bring-up ladder's own accounting.
///
/// The zero-trap user clock first — it is nanosecond-grained, which the
/// trap-return stamp is not, and this is read once per ladder step. The
/// millisecond stamp is the fallback for a thread founded without a
/// calibrated clock page, and `None` (a host build) means the ladder falls
/// back to its poll ceiling, exactly as it did before it could time
/// anything.
fn monotonic_ns() -> Option<u64> {
    if let Ok(uptime) = kore_rt::clock::uptime() {
        return Some(uptime.as_nanos());
    }
    now_ms().map(|ms| ms.saturating_mul(1_000_000))
}

/// The wall clock the bring-up ladder is charged against.
///
/// One anchor per child, reset on completed writes/reads/delays. Time spent
/// servicing siblings belongs to this child's reset budget too. A relay
/// without a clock reports zero; the ladder retains its poll-count ceiling.
struct Lap(Option<u64>);

impl Lap {
    /// Nanoseconds since the previous lap, re-anchoring here.
    fn take(&mut self, now: Option<u64>) -> u64 {
        let previous = core::mem::replace(&mut self.0, now);
        match (previous, now) {
            (Some(previous), Some(now)) => now.saturating_sub(previous),
            _ => 0,
        }
    }
}
fn config(binding: I2cHidReport, input_clock_hz: u32, irq_driven: bool) -> BusConfig {
    let speed = Speed::from_hz(binding.speed_hz);
    let timing = match speed {
        Speed::Standard => binding.standard_timing,
        Speed::Fast => binding.fast_timing,
    };
    BusConfig {
        speed,
        counts: timing.map(|t| Counts {
            hcnt: t.hcnt,
            lcnt: t.lcnt,
        }),
        input_clock_hz,
        sda_hold: None,
        irq_driven,
    }
}
fn dw_config(binding: I2cHidReport) -> Option<BusConfig> {
    let clock = binding
        .input_clock_hz
        .unwrap_or(Integration::INTEL_LPSS.input_clock_hz);
    (clock != 0).then(|| config(binding, clock, true))
}

fn geni_config(binding: I2cHidReport) -> Option<BusConfig> {
    let clock = binding.input_clock_hz.filter(|clock| *clock != 0)?;
    Some(config(binding, clock, false))
}

fn bus_error<A: kore_i2c::error::AbortDetail>(error: kore_i2c::error::I2cError<A>) -> u8 {
    if error.is_address_nack() {
        2
    } else if matches!(error, kore_i2c::error::I2cError::Timeout { .. }) {
        3
    } else {
        4
    }
}

struct DeviceState {
    descriptor: kore_i2c_hid::HidDescriptor,
    table: ExtractionTable,
    sequences: [u32; MAX_REPORTS],
}

/// Parsed input is not live input. Keep it behind the configuration ladder
/// until every advertised mode/control write has completed. No report bytes
/// or allocations survive parsing; both state machines are fixed-capacity.
struct PendingDevice {
    device: DeviceState,
    features: FeatureSetup,
}
#[derive(Default)]
struct Retry {
    failures: u8,
    ready_at: u64,
}
impl Retry {
    fn fail(&mut self, now: u64) {
        self.ready_at = now.saturating_add(250u64 << self.failures.min(4));
        self.failures = self.failures.saturating_add(1);
    }
    fn ready(&self, now: u64) -> bool {
        now >= self.ready_at
    }
}
struct Child {
    binding: I2cHidReport,
    bringup: Option<BringUp>,
    device: Option<DeviceState>,
    pending: Option<PendingDevice>,
    protocol: u32,
    retry: Retry,
    init_deadline: u64,
    lap: Lap,
    delay_pending: bool,
    generation: u32,
}
fn address_is_new(children: &[Option<Child>], address: u8) -> bool {
    !children
        .iter()
        .flatten()
        .any(|child| child.binding.slave_addr == address)
}
impl Child {
    fn new(binding: I2cHidReport) -> Self {
        Self {
            binding,
            bringup: None,
            device: None,
            pending: None,
            protocol: 0,
            retry: Retry::default(),
            init_deadline: 0,
            lap: Lap(None),
            delay_pending: false,
            generation: 1,
        }
    }
    fn begin(&mut self, now: u64) {
        self.pending = None;
        self.protocol = I2cProtocolPhase::Descriptor.pack(0, 0);
        self.bringup = Some(BringUp::new(
            self.binding.descriptor_register,
            Quirks {
                no_irq: self.binding.no_irq,
                no_power_after_reset: self.binding.no_power_after_reset,
            },
        ));
        self.init_deadline = now.saturating_add(INIT_BUDGET_MS);
        self.lap = Lap(None);
        self.delay_pending = false;
    }

    fn initializing(&self) -> bool {
        self.bringup.is_some() || self.pending.is_some()
    }

    fn check_init_deadline(&self, now: u64) -> Result<(), u8> {
        if self.device.is_none() && now >= self.init_deadline {
            Err(3)
        } else {
            Ok(())
        }
    }

    fn protocol_phase(&self) -> u8 {
        if let Some(pending) = &self.pending {
            match pending.features.phase() {
                FeaturePhase::Get => phase::FEATURE_GET,
                FeaturePhase::Set => phase::FEATURE_SET,
                FeaturePhase::Done => phase::STEADY,
            }
        } else {
            self.bringup.as_ref().map_or(phase::REPORT_DESCRIPTOR, |b| {
                initialization_phase(b.phase())
            })
        }
    }

    fn publish_online(&mut self) {
        self.device = self.pending.take().map(|pending| pending.device);
        self.bringup = None;
        self.protocol = I2cProtocolPhase::Steady.pack(0, 0);
    }

    fn fail(&mut self, now: u64, queue: &mut spsc::Producer<'_, InputReport>, source: u8) {
        // Once per online→failed transition, never one reset storm per poll.
        if self.device.take().is_some() {
            self.generation = self.generation.checked_add(1).unwrap_or(0);
            let resets: [_; MAX_REPORTS] = core::array::from_fn(|stream| {
                kore_abi::pointer::Record::Reset { all: false }.encode(source + stream as u8)
            });
            queue.push_slice_lossy(&resets);
        }
        self.bringup = None;
        // Reset/retry must reparse and replay configuration; a partially
        // configured device has never published input or acquired history.
        self.pending = None;
        self.retry.fail(now);
    }
}

/// Controller-wide sleeps belong to every child still initializing. This
/// bounded publication uses the existing per-child words; online and failed
/// children retain their input/failure evidence. No payload or authority moves.
fn during_initialization<T>(
    children: &[Option<Child>; MAX_CHILDREN],
    stats: &RelayStats,
    phase: I2cProtocolPhase,
    call: impl FnOnce() -> T,
) -> T {
    for (index, child) in children.iter().enumerate() {
        if child.as_ref().is_some_and(Child::initializing) {
            stats.children[index]
                .protocol
                .store(phase.pack(0, 0), Ordering::Release);
        }
    }
    let result = call();
    for (index, child) in children.iter().enumerate() {
        if let Some(child) = child.as_ref().filter(|child| child.initializing()) {
            stats.children[index]
                .protocol
                .store(child.protocol, Ordering::Release);
        }
    }
    result
}

/// One protocol step. Descriptors are parsed immediately, never retained in
/// shared scratch. Delay steps arm a per-child deadline without sleeping;
/// only a subsequent eligible pass acknowledges the measured delay.
fn init_step<const C: usize, B: I2cBus>(
    child: &mut Child,
    bus: &mut B,
    raw: &mut [u8],
    now: u64,
    mut clock: impl FnMut() -> Option<u64>,
    classify_error: impl Fn(B::Error) -> u8,
) -> Result<(), u8> {
    if !child.retry.ready(now) {
        return Ok(());
    }
    child.check_init_deadline(now)?;
    if child.pending.is_some() {
        return feature_step::<C, B>(child, bus, raw, classify_error);
    }
    let bringup = child.bringup.as_mut().ok_or(5u8)?;
    // Controller-wide diagnostics retain a sibling's last failure. They
    // must never substitute for this child's current protocol phase.
    let phase = initialization_phase(bringup.phase());
    child.protocol = I2cProtocolPhase::from_bits(phase).ok_or(5u8)?.pack(0, 0);
    let classify_hid = |error| bringup_error(relay::<C>(), phase, error);
    match bringup.step() {
        Step::Write { len } => {
            bus.write(child.binding.slave_addr, &bringup.out()[..len])
                .map_err(&classify_error)?;
            relay::<C>().transfers.fetch_add(1, Ordering::Relaxed);
            bringup.wrote().map_err(classify_hid)?;
            // A completed RESET write starts the budget, not the earlier
            // transfer start or the first poll on a later controller pass.
            child.lap = Lap(clock());
        }
        Step::WriteRead {
            write_len,
            read_len,
        } => {
            if read_len > raw.len() {
                return Err(5);
            }
            // A feature GET also uses WriteRead after the HID descriptor
            // exists. Its payload must never be parsed as a descriptor.
            let report_descriptor = bringup.phase() == Phase::ReportDescriptor;
            let got = bus
                .write_read(
                    child.binding.slave_addr,
                    &bringup.out()[..write_len],
                    &mut raw[..read_len],
                )
                .map_err(&classify_error)?;
            relay::<C>().transfers.fetch_add(1, Ordering::Relaxed);
            if got != read_len {
                return Err(5);
            }
            bringup
                .read(&raw[..got], child.lap.take(clock()))
                .map_err(classify_hid)?;
            if report_descriptor {
                let descriptor = *bringup.descriptor().ok_or(5u8)?;
                if !(2..=MAX_XFER_BYTES).contains(&(descriptor.w_max_input_length.get() as usize)) {
                    return Err(5);
                }
                child.protocol = I2cProtocolPhase::Table.pack(0, 0);
                let parsed = build_device(&raw[..got]).map_err(|error| {
                    if let ParseError::Feature(feature) = error {
                        let id = match feature {
                            FeatureError::Plan { report_id, .. } => report_id,
                            _ => 0,
                        };
                        child.protocol = I2cProtocolPhase::Table.pack(id, feature.diagnostic16());
                    }
                    publish_failure::<C>(phase::TABLE, failure::TABLE, error.diagnostic(), None);
                    6u8
                })?;
                let features = FeatureSetup::new(parsed.features, &descriptor).map_err(|error| {
                    child.protocol = I2cProtocolPhase::FeatureGet.pack(0, error.diagnostic16());
                    publish_failure::<C>(
                        phase::FEATURE_GET,
                        failure::HID,
                        error.diagnostic(),
                        None,
                    );
                    5u8
                });
                let features = features?;
                let complete = features.phase() == FeaturePhase::Done;
                child.pending = Some(PendingDevice {
                    device: DeviceState {
                        descriptor,
                        table: parsed.table,
                        sequences: [0; MAX_REPORTS],
                    },
                    features,
                });
                child.bringup = None;
                if complete {
                    child.publish_online();
                }
            }
        }
        Step::Read { len } => {
            if len > raw.len() {
                return Err(5);
            }
            let got = bus
                .read(child.binding.slave_addr, &mut raw[..len])
                .map_err(&classify_error)?;
            relay::<C>().transfers.fetch_add(1, Ordering::Relaxed);
            if got != len {
                return Err(5);
            }
            bringup
                .read(&raw[..got], child.lap.take(clock()))
                .map_err(classify_hid)?;
        }
        Step::Delay { ns } => {
            if child.delay_pending {
                bringup
                    .delayed(child.lap.take(clock()))
                    .map_err(classify_hid)?;
                child.delay_pending = false;
            } else {
                child.retry.ready_at = now.saturating_add(ns.div_ceil(1_000_000));
                child.delay_pending = true;
            }
        }
        Step::Done => return Err(5), // descriptor parse must have installed online state
    }
    Ok(())
}

/// Exactly one configuration transaction per eligible controller pass. A
/// sleeping/resetting/failing sibling cannot monopolize another child's
/// feature setup, and neither a failed GET nor SET advances the ladder.
fn feature_step<const C: usize, B: I2cBus>(
    child: &mut Child,
    bus: &mut B,
    raw: &mut [u8],
    classify_error: impl Fn(B::Error) -> u8,
) -> Result<(), u8> {
    let pending = child.pending.as_mut().ok_or(5u8)?;
    let report_id = pending.features.current_report_id().unwrap_or(0);
    let (phase, observation) = match pending.features.phase() {
        FeaturePhase::Get => (phase::FEATURE_GET, I2cProtocolPhase::FeatureGet),
        FeaturePhase::Set => (phase::FEATURE_SET, I2cProtocolPhase::FeatureSet),
        FeaturePhase::Done => {
            child.publish_online();
            return Ok(());
        }
    };
    child.protocol = observation.pack(report_id, 0);
    let completed = match pending.features.step() {
        Step::WriteRead {
            write_len,
            read_len,
        } => {
            if read_len > raw.len() {
                return Err(5);
            }
            let got = bus
                .write_read(
                    child.binding.slave_addr,
                    &pending.features.out()[..write_len],
                    &mut raw[..read_len],
                )
                .map_err(&classify_error)?;
            relay::<C>().transfers.fetch_add(1, Ordering::Relaxed);
            if got != read_len {
                publish_failure::<C>(phase, failure::SHORT, got as u32, None);
                return Err(5);
            }
            pending.features.read(&raw[..got])
        }
        Step::Write { len } => {
            bus.write(child.binding.slave_addr, &pending.features.out()[..len])
                .map_err(&classify_error)?;
            relay::<C>().transfers.fetch_add(1, Ordering::Relaxed);
            pending.features.wrote()
        }
        _ => return Err(5),
    };
    completed.map_err(|error| {
        let detail = error.diagnostic();
        child.protocol = observation.pack(report_id, error.diagnostic16());
        publish_failure::<C>(phase, failure::HID, detail, None);
        5u8
    })?;
    if pending.features.phase() == FeaturePhase::Done {
        child.publish_online();
    }
    Ok(())
}

const fn initialization_phase(phase: Phase) -> u8 {
    match phase {
        Phase::Descriptor => phase::DESCRIPTOR,
        Phase::Power => phase::POWER,
        Phase::Reset => phase::RESET,
        Phase::ResetAck => phase::RESET_ACK,
        Phase::ReportDescriptor | Phase::Done => phase::REPORT_DESCRIPTOR,
    }
}

fn bringup_error(stats: &RelayStats, phase: u8, error: BringUpError) -> u8 {
    let detail = match error {
        BringUpError::ResetTimeout { polls, .. } => polls,
        BringUpError::ReportDescriptorShort { want, got } => {
            ((want.min(u16::MAX as usize) as u32) << 16) | got.min(u16::MAX as usize) as u32
        }
        _ => 0,
    };
    stats.record_failure(phase, failure::HID, detail, None);
    match error {
        BringUpError::ResetTimeout { .. } => 3,
        _ => 5,
    }
}

/// HID input uses a plain receive (Linux i2c_hid_get_input), not an input-register
/// write. Descriptor fetches still use write/repeated-start/read.
fn read_input<B: I2cBus>(bus: &mut B, address: u8, raw: &mut [u8]) -> Result<usize, B::Error> {
    bus.read(address, raw)
}

fn poll_deadline(now_ms: u64, saturated: bool) -> u64 {
    let delay_ms = if saturated { 1 } else { GPIO_BACKSTOP_MS };
    now_ms.saturating_add(delay_ms).saturating_mul(1_000_000)
}

/// Adaptive deadline policy for an admitted HID-ready GPIO route.
///
/// A newly admitted line has not delivered anything yet, so it starts at the
/// input-latency fallback. A real interrupt proves the route and promotes its
/// deadline to the sparse liveness watchdog. A later report found after that
/// watchdog demotes it again because the GPIO failed to announce queued input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpioBackstop {
    degraded: bool,
    deadline_won: bool,
}

impl Default for GpioBackstop {
    fn default() -> Self {
        Self::new()
    }
}

impl GpioBackstop {
    const fn new() -> Self {
        Self {
            degraded: true,
            deadline_won: false,
        }
    }

    const fn period_ms(&self) -> u64 {
        if self.degraded {
            GPIO_DEGRADED_POLL_MS
        } else {
            GPIO_IRQ_WATCHDOG_MS
        }
    }

    fn parked(&mut self, fired: bool) {
        self.deadline_won = !fired;
        if fired {
            self.degraded = false;
        }
    }

    fn drained(&mut self, delivered: bool) {
        if delivered && self.deadline_won {
            self.degraded = true;
        }
    }
}

fn report_counts(stats: &RelayStats) -> [u32; MAX_CHILDREN] {
    core::array::from_fn(|index| stats.children[index].reports.load(Ordering::Relaxed))
}

fn child_order(first: usize) -> impl Iterator<Item = usize> {
    (0..MAX_CHILDREN).map(move |offset| (first + offset) % MAX_CHILDREN)
}

fn room_for_report(queue: &spsc::Producer<'_, InputReport>) -> bool {
    queue.capacity_left() >= REPORT_RESERVE
}

fn needs_report_kick(queue: &spsc::Producer<'_, InputReport>) -> bool {
    queue.capacity_left() < queue.capacity()
}

/// A held level source can request its next read by reasserting when the
/// kernel rearms it. Edge sources and unrouted children must drain to empty.
/// Firmware metadata alone is not proof: the owner must have admitted a route
/// and observed it firing. A deadline/refusal falls back to bounded draining.
fn rearm_after_report(binding: I2cHidReport, route_held: bool, backstop: &GpioBackstop) -> bool {
    use kore_abi::session::I2cReadyInterruptReport;
    route_held
        && !backstop.degraded
        && !backstop.deadline_won
        && matches!(
            binding.ready_interrupt,
            Some(I2cReadyInterruptReport::Line { edge: false, .. })
                | Some(I2cReadyInterruptReport::Gpio(
                    kore_abi::session::I2cGpioInterruptReport { edge: false, .. }
                ))
        )
}

/// Bound one source's drain while preserving whole pointer frames.
/// The same body serves GENI and DesignWare and is exercised without hardware.
///
/// Returns `true` only when the pass is a backlog the caller should revisit
/// before its idle cadence: the report queue lacked room, or (without a
/// rearmable level route) every one of
/// [`MAX_DRAIN_FRAMES`] reads decoded. An empty frame, an unknown report id
/// or an undecodable shape ends the pass with `false` — none of them is
/// evidence that input is queued, and reading on would only spend the bus.
fn drain_child<const C: usize, B: I2cBus>(
    child: &mut Child,
    level_ready: bool,
    bus: &mut B,
    raw: &mut [u8],
    decoded: &mut [InputReport],
    source: u8,
    books: &ChildStats,
    mut clock: impl FnMut() -> Option<u64>,
    classify_error: impl Fn(B::Error) -> u8,
    queue: &mut spsc::Producer<'_, InputReport>,
    mut published: impl FnMut(),
) -> Result<bool, u8> {
    for _ in 0..MAX_DRAIN_FRAMES {
        // Reserve before consuming a device packet, not after decoding it.
        // This is the sole producer; the consumer can only increase free
        // capacity during I/O. A full frame (or failure resets) will fit.
        if !room_for_report(queue) {
            return Ok(true);
        }
        let state = child.device.as_mut().ok_or(5u8)?;
        books.polls.fetch_add(1, Ordering::Relaxed);
        let want = state.descriptor.w_max_input_length.get() as usize;
        let got =
            read_input(bus, child.binding.slave_addr, &mut raw[..want]).map_err(&classify_error)?;
        let received_ns = clock();
        relay::<C>().transfers.fetch_add(1, Ordering::Relaxed);
        if got != want {
            books.last_input.store(
                I2cInputOutcome::Malformed.pack(0, got.min(u16::MAX as usize) as u16),
                Ordering::Relaxed,
            );
            publish_failure::<C>(phase::STEADY, failure::SHORT, got as u32, None);
            return Err(5);
        }
        let wire_length = u16::from_le_bytes([raw[0], raw[1]]);
        let payload = match framing::decode(&raw[..got], want as u16) {
            Frame::Report(payload) => payload,
            Frame::Empty => {
                // Idle polls must not erase the last actual report or
                // refusal immediately before a user reads `status input`.
                if books.last_input.load(Ordering::Relaxed) as u8 <= I2cInputOutcome::Empty as u8 {
                    books.last_input.store(
                        I2cInputOutcome::Empty.pack(0, wire_length),
                        Ordering::Relaxed,
                    );
                }
                child.retry.failures = 0;
                return Ok(false);
            }
            Frame::Refused(_) => {
                books.last_input.store(
                    I2cInputOutcome::Malformed.pack(0, wire_length),
                    Ordering::Relaxed,
                );
                publish_failure::<C>(phase::STEADY, failure::HID, 0, None);
                return Err(5);
            }
        };
        let report_id = if state.table.uses_report_ids() {
            payload.first().copied().unwrap_or(0)
        } else {
            0
        };
        state.table.validate_input_length(payload).map_err(|_| {
            books.last_input.store(
                I2cInputOutcome::DecodeRefused.pack(report_id, wire_length),
                Ordering::Relaxed,
            );
            5u8
        })?;
        let Some(stream) = state.table.stream_index(payload) else {
            books.last_input.store(
                if state.table.expected_input_length(payload).is_some() {
                    I2cInputOutcome::UnsupportedShape
                } else {
                    I2cInputOutcome::UnknownReport
                }
                .pack(report_id, wire_length),
                Ordering::Relaxed,
            );
            return Ok(false);
        };
        state.sequences[stream] = state.sequences[stream].wrapping_add(1);
        let n = state
            .table
            .decode_framed_into(
                payload,
                source,
                child.generation,
                state.sequences[stream],
                received_ns,
                decoded,
            )
            .map_err(|_| {
                books.last_input.store(
                    I2cInputOutcome::DecodeRefused.pack(report_id, wire_length),
                    Ordering::Relaxed,
                );
                5u8
            })?;
        books.last_input.store(
            if n == 0 {
                I2cInputOutcome::UnsupportedShape
            } else {
                I2cInputOutcome::Decoded
            }
            .pack(report_id, wire_length),
            Ordering::Relaxed,
        );
        child.retry.failures = 0;
        if n == 0 {
            return Ok(false);
        }
        if queue.push_slice_lossy(&decoded[..n]) {
            books.reports.fetch_add(1, Ordering::Relaxed);
            // A following read may park for controller completion. Deliver
            // this whole frame now instead of holding it behind that read.
            published();
        }
        if level_ready {
            // Rearm the kernel's mask-on-delivery source now. A queued frame
            // keeps the level asserted; an empty device costs no speculative
            // second transfer. Do not take the saturated sleep for this case.
            return Ok(false);
        }
    }

    Ok(true)
}

fn run<const C: usize>() {
    // Persistent books, not a parked task or console print, witness refusal.
    if let Err(code) = drive::<C>() {
        note_claim(C, code);
    }
}
fn drive<const C: usize>() -> Result<(), u32> {
    if !PREPARED[C].load(Ordering::Acquire) {
        return Err(rung::NO_BINDINGS);
    }
    let rt = kore_rt::entry::thread_context().ok_or(rung::NO_CONTEXT)?;
    let boot = rt.bootstrap();
    let kick = boot
        .handle(role::DOORBELL, SERVICE_KICK_PEER)
        .ok_or(rung::NO_KICK)?;
    let controller = ControllerKind::from_published(MODELS[C].load(Ordering::Relaxed))
        .ok_or(rung::BAD_DEVICE)?;
    match controller {
        ControllerKind::PlatformDesignWare => {
            let slot = boot.handle(role::DEVICE, 0).ok_or(rung::NO_DEVICE)?;
            let token = verbs::claim_platform_device(slot).ok_or(rung::BAD_DEVICE)?;
            let route = verbs::establish_line(&token).ok_or_else(|| {
                register_intake_error::<C>(kore_driver::host::IntakeError::InterruptUnavailable)
            })?;
            let ready_route = (BINDING_COUNTS[C].load(Ordering::Relaxed) == 1)
                .then(|| verbs::establish_i2c_hid_ready(&token, 0))
                .flatten();
            let regs = published_regs::<C>()?;
            drive_dw::<C>(&regs, &route, ready_route.as_ref(), kick)
        }
        ControllerKind::PciDesignWare => {
            let slot = boot.handle(role::DEVICE, 0).ok_or(rung::NO_DEVICE)?;
            let token = verbs::claim_device(slot).ok_or(rung::BAD_DEVICE)?;
            // The owning task already mapped BAR0 in the shared VAS. This
            // worker holds only the controller's firmware INTx route; that
            // route is distinct from the HID child's child-ready signal.
            let route = verbs::establish_pci_line(&token).ok_or_else(|| {
                register_intake_error::<C>(kore_driver::host::IntakeError::InterruptUnavailable)
            })?;
            let ready_route = (BINDING_COUNTS[C].load(Ordering::Relaxed) == 1)
                .then(|| verbs::establish_i2c_hid_ready(&token, 0))
                .flatten();
            let regs = published_regs::<C>()?;
            drive_dw::<C>(&regs, &route, ready_route.as_ref(), kick)
        }
        ControllerKind::PlatformGeni => {
            let regs = published_regs::<C>()?;
            drive_bus::<C, _, _, _>(
                I2cEngine::new(GeniMaster::<CapRegs>::new()),
                &regs,
                kick,
                None,
                geni_config,
                |hz| {
                    RateBudget::new(hz, 1, || {
                        if let Some(now) = monotonic_ns() {
                            let _ = verbs::sleep_until(now.saturating_add(1_000_000));
                        }
                    })
                },
            )
        }
    }
}

/// Reconstitute the borrowed register view published by the owning task.
///
/// The worker shares that task's address space but owns no mapping authority.
/// `PREPARED`'s release/acquire edge publishes the geometry only after the
/// owner retained the affine `CapRegs` mapping for the service lifetime.
fn published_regs<const C: usize>() -> Result<CapRegs, u32> {
    let base = MMIO_BASES[C].load(Ordering::Relaxed) as usize as *mut u8;
    let len = MMIO_LENGTHS[C].load(Ordering::Relaxed);
    if base.is_null() || len == 0 {
        return Err(rung::NO_RESOURCES);
    }
    // SAFETY: the input task retains the mapping for its lifetime; this sole
    // worker shares its VAS and receives checked, immutable geometry.
    Ok(unsafe { CapRegs::from_process_mapping(base, len) })
}

fn drive_dw<const C: usize>(
    regs: &CapRegs,
    route: &RouteToken,
    ready_route: Option<&RouteToken>,
    kick: kore_abi::session::SlotRef,
) -> Result<(), u32> {
    drive_bus::<C, _, _, _>(
        I2cEngine::new(
            DwMaster::new(Integration::INTEL_LPSS, kore_driver::host::relax)
                .with_clock(kore_driver::host::clock_now_ns),
        ),
        regs,
        kick,
        ready_route,
        dw_config,
        |hz| route_pacing(route, &RELAYS[C], hz),
    )
}

fn register_intake_error<const C: usize>(error: kore_driver::host::IntakeError) -> u32 {
    // RegisterDevice emits BAR-index, mapping, or IRQ failures, whose complete
    // stable diagnostic fits in this field. Refuse a future wider encoding
    // visibly rather than silently truncating its high bits.
    let detail = u32::try_from(error.diagnostic_code()).unwrap_or(u32::MAX);
    publish_failure::<C>(phase::CLAIM, failure::INTAKE, detail, None);
    rung::NO_RESOURCES
}

/// One child state machine and framing path for both controller families.
/// Only configuration and transfer pacing are supplied by the backend.
///
/// Keep this as a real call boundary. Its persistent child/parser state is a
/// large but bounded frame; inlining one monomorphization into `drive` while
/// retaining the DesignWare wrapper made two copies live across an interrupt
/// park and overflowed an ordinary worker stack. The boundary lets every
/// controller carry exactly one state-machine frame.
#[inline(never)]
fn drive_bus<const C: usize, M, P, F>(
    mut engine: I2cEngine<M>,
    regs: &CapRegs,
    kick: kore_abi::session::SlotRef,
    ready_route: Option<&RouteToken>,
    configuration: fn(I2cHidReport) -> Option<BusConfig>,
    pacing: F,
) -> Result<(), u32>
where
    M: I2cMaster<Regs = CapRegs>,
    P: Pacing,
    F: Fn(u32) -> P,
{
    let stats = &RELAYS[C];
    let count = BINDING_COUNTS[C].load(Ordering::Relaxed) as usize;
    if count == 0 || count > MAX_CHILDREN {
        return Err(rung::NO_BINDINGS);
    }
    // SAFETY: PREPARED's Acquire observed the owner's completed publication;
    // the bounded records are immutable thereafter and contain no authority.
    let bindings = unsafe { *BINDINGS[C].0.get() };
    let mut children: [Option<Child>; MAX_CHILDREN] = core::array::from_fn(|_| None);
    for (i, binding) in bindings.into_iter().take(count).enumerate() {
        let binding = binding.ok_or(rung::NO_BINDINGS)?;
        stats.children[i].identity.store(
            binding.slave_addr as u32 | 1 << 8 | u32::from(binding.ready_interrupt.is_some()) << 24,
            Ordering::Release,
        );
        if address_is_new(&children[..i], binding.slave_addr) {
            children[i] = Some(Child::new(binding));
        } else {
            stats.children[i].phase(4, 9);
            stats.children[i].errors.store(1, Ordering::Relaxed);
        }
    }
    publish_phase::<C>(phase::PROBE);
    engine.probe(regs).map_err(|e| {
        e.publish::<C>(phase::PROBE);
        rung::CONTROLLER
    })?;
    // SAFETY: prepare(C, ...) initialized this indexed ring before spawning this worker.
    let mut queue =
        spsc::producer::<InputReport>(unsafe { REGIONS[C].view() }).map_err(|_| rung::NO_QUEUE)?;
    note_claim(C, 0);
    let mut raw = [0u8; MAX_XFER_BYTES];
    let mut decoded = [InputReport::mouse(0, 0, 0, 0, 0); kore_abi::pointer::MAX_FRAME_RECORDS];
    let mut active_config = None;
    let mut first_child = 0;
    let mut ready_backstop = GpioBackstop::new();
    loop {
        let reports_before = report_counts(stats);
        let mut saturated = false;
        // A busy child must not exhaust every pass's ring budget before its
        // siblings get a turn, including siblings still being initialized.
        for i in child_order(first_child) {
            let Some(child) = children[i].as_mut() else {
                continue;
            };
            let books = &stats.children[i];
            let now = if child.device.is_none() {
                books.during(I2cProtocolPhase::InitClock, now_ms)
            } else {
                now_ms()
            }
            .ok_or(rung::CONTROLLER)?;
            if !child.retry.ready(now) || child.generation == 0 {
                continue;
            }
            if child.device.is_some() && !room_for_report(&queue) {
                // Do not even configure an online child without room for
                // its failure resets. Initialization emits no input yet.
                saturated = true;
                continue;
            }
            if child.device.is_none() && !child.initializing() {
                child.begin(now);
                books.phase(1, 0);
            }
            let result = (|| -> Result<(), u8> {
                child.check_init_deadline(now)?;
                child.protocol = I2cProtocolPhase::Configure.pack(0, 0);
                if child.device.is_none() {
                    books.protocol.store(child.protocol, Ordering::Release);
                }
                let needed = configuration(child.binding).ok_or_else(|| {
                    publish_failure::<C>(phase::CONFIGURE, failure::CLOCK, 0, None);
                    1u8
                })?;
                if active_config != Some(needed) {
                    engine.configure(regs, &needed).map_err(|e| {
                        e.publish::<C>(phase::CONFIGURE);
                        1u8
                    })?;
                    active_config = Some(needed);
                }
                let mut bus = engine.bus(regs, pacing(needed.speed.hz()));
                if child.device.is_none() {
                    let phase = child.protocol_phase();
                    publish_phase::<C>(phase);
                    // Publish the operation we are ABOUT TO enter. The final
                    // observe_step still records its exact refusal or result.
                    let report_id = child
                        .pending
                        .as_ref()
                        .and_then(|p| p.features.current_report_id())
                        .unwrap_or(0);
                    books.protocol.store(
                        u32::from(phase) | u32::from(report_id) << 8,
                        Ordering::Release,
                    );
                    init_step::<C, _>(child, &mut bus, &mut raw, now, monotonic_ns, |e| {
                        e.publish::<C>(phase);
                        bus_error(e)
                    })?;
                    if child.device.is_some() {
                        books.phase(2, 0);
                    }
                    return Ok(());
                }
                child.protocol = I2cProtocolPhase::Steady.pack(0, 0);
                let level_ready =
                    rearm_after_report(child.binding, ready_route.is_some(), &ready_backstop);
                saturated |= drain_child::<C, _>(
                    child,
                    level_ready,
                    &mut bus,
                    &mut raw,
                    &mut decoded,
                    source_base(C, i),
                    books,
                    monotonic_ns,
                    |e| {
                        e.publish::<C>(phase::STEADY);
                        bus_error(e)
                    },
                    &mut queue,
                    || {
                        if kore_rt::wake::ring_doorbell_peer(SERVICE_KICK_PEER) {
                            let _ = verbs::post_notice(kick, kore_rt::svc::COOKIE_REPORT, 0);
                        }
                    },
                )?;
                Ok(())
            })();
            // Publish only bounded protocol metadata, never HID payloads or
            // synchronous console writes. The child keeps its own witness
            // even if a sibling overwrites the legacy controller summary.
            books.observe_step(child, result);
            if let Err(error) = result {
                // Abort/timeout recovery invalidates the configuration cache;
                // the next healthy sibling must not inherit that stale state.
                active_config = None;
                child.fail(now_ms().unwrap_or(now), &mut queue, source_base(C, i));
                books.errors.fetch_add(1, Ordering::Relaxed);
                if child.generation == 0 {
                    books.phase(4, 8);
                } else {
                    books.phase(3, error);
                }
            }
        }
        first_child = (first_child + 1) % MAX_CHILDREN;
        if children
            .iter()
            .flatten()
            .any(|child| child.device.is_some())
        {
            publish_steady::<C>(&children, ready_route.is_some());
        }
        stats.dropped.store(queue.dropped(), Ordering::Relaxed);
        if needs_report_kick(&queue) {
            // Re-kick an already full queue even if this pass could not
            // read anything new: the consumer may still be parked.
            // Same-task relay, same peer index as i8042: the bell store is
            // sufficient while the root executor is awake. A parked or unknown
            // receiver still needs the owned-channel notice.
            if kore_rt::wake::ring_doorbell_peer(SERVICE_KICK_PEER) {
                let _ = verbs::post_notice(kick, kore_rt::svc::COOKIE_REPORT, 0);
            }
        }
        // A saturated report pass is useful work, not a register probe:
        // ceding() would wait 4,096 passes before actually parking. Yield
        // after every bounded batch, while retrying backlog sooner than the
        // idle child-readiness watchdog.
        let now = during_initialization(&children, stats, I2cProtocolPhase::InitClock, now_ms)
            .ok_or(rung::CONTROLLER)?;
        let online = children
            .iter()
            .flatten()
            .any(|child| child.device.is_some());
        // Consume the report witness before any backlog/initialization branch.
        // A full bounded drain is precisely where an unannounced input burst
        // is most likely; skipping it could leave the next idle park on the
        // sparse watchdog even though the readiness IRQ missed useful work.
        if online && ready_route.is_some() {
            ready_backstop.drained(report_counts(stats) != reports_before);
        }
        if saturated || !online || ready_route.is_none() {
            during_initialization(&children, stats, I2cProtocolPhase::InitSleep, || {
                verbs::sleep_until(poll_deadline(now, saturated))
            })
            .ok_or(rung::CONTROLLER)?;
        } else {
            let deadline = now
                .saturating_add(ready_backstop.period_ms())
                .saturating_mul(1_000_000);
            match verbs::wait_interrupt_until(ready_route.expect("checked above"), deadline) {
                Ok(verbs::InterruptWait::Fired) => {
                    stats.note_gpio_irq();
                    ready_backstop.parked(true);
                }
                Ok(verbs::InterruptWait::Deadline) => {
                    stats.note_watchdog_deadline();
                    ready_backstop.parked(false);
                }
                Ok(verbs::InterruptWait::Revoked) | Err(_) => {
                    // A refused route does not strand input. Retain the bounded
                    // polling path while teardown or route recovery proceeds.
                    // The next pass must drain backlog, not rely on rearming a
                    // route whose last wait failed.
                    ready_backstop.parked(false);
                    verbs::sleep_until(poll_deadline(now, false)).ok_or(rung::CONTROLLER)?;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_park_publishes_before_entry_and_preserves_child_failure_evidence() {
        let stats = RelayStats::new();
        let mut initializing = Child::new(binding(0x16));
        initializing.begin(0);
        initializing.protocol = I2cProtocolPhase::Power.pack(0, 0);
        let mut failed = Child::new(binding(0x2c));
        failed.protocol = I2cProtocolPhase::ResetAck.pack(0, 0x8003);
        let mut children: [Option<Child>; MAX_CHILDREN] = core::array::from_fn(|_| None);
        children[0] = Some(initializing);
        children[1] = Some(failed);
        for (index, child) in children.iter().enumerate() {
            if let Some(child) = child {
                stats.children[index]
                    .protocol
                    .store(child.protocol, Ordering::Relaxed);
            }
        }
        let before = stats.children[1].snapshot().protocol;
        let result = during_initialization(&children, &stats, I2cProtocolPhase::InitSleep, || {
            assert_eq!(
                stats.children[0].snapshot().protocol.get(),
                I2cProtocolPhase::InitSleep.pack(0, 0)
            );
            assert_eq!(stats.children[1].snapshot().protocol, before);
            None::<u64> // a refused transport must restore the prior evidence too
        });
        assert_eq!(result, None);
        assert_eq!(
            stats.children[0].snapshot().protocol.get(),
            I2cProtocolPhase::Power.pack(0, 0)
        );
        assert_eq!(stats.children[1].snapshot().protocol, before);
        stats.children[0].during(I2cProtocolPhase::InitClock, || {
            assert_eq!(
                stats.children[0].snapshot().protocol.get(),
                I2cProtocolPhase::InitClock.pack(0, 0)
            );
        });
        assert_eq!(
            stats.children[0].snapshot().protocol.get(),
            I2cProtocolPhase::Power.pack(0, 0)
        );
    }

    #[test]
    fn every_kernel_gpio_readiness_reason_keeps_a_distinct_poll_diagnostic() {
        use kore_abi::input_status::*;

        for (state, detail) in [
            (
                I2cReadyRouteState::NotDescribed,
                I2C_READY_POLL_NO_FIRMWARE_IRQ,
            ),
            (I2cReadyRouteState::NoCommunity, I2C_READY_POLL_NO_COMMUNITY),
            (I2cReadyRouteState::NoLayout, I2C_READY_POLL_NO_LAYOUT),
            (I2cReadyRouteState::NoAperture, I2C_READY_POLL_NO_APERTURE),
            (
                I2cReadyRouteState::NoParentLine,
                I2C_READY_POLL_NO_PARENT_LINE,
            ),
            (
                I2cReadyRouteState::PadOutOfRange,
                I2C_READY_POLL_PAD_OUT_OF_RANGE,
            ),
            (
                I2cReadyRouteState::NoCatalogDevice,
                I2C_READY_POLL_NO_CATALOG_DEVICE,
            ),
            (
                I2cReadyRouteState::NoRegisterView,
                I2C_READY_POLL_NO_REGISTER_VIEW,
            ),
            (
                I2cReadyRouteState::QuiesceRefused,
                I2C_READY_POLL_QUIESCE_REFUSED,
            ),
            (
                I2cReadyRouteState::ParentRouteRefused,
                I2C_READY_POLL_PARENT_ROUTE_REFUSED,
            ),
            (I2cReadyRouteState::Ready, I2C_READY_POLL_PAD_BIND_REFUSED),
            (
                I2cReadyRouteState::SetupUnavailable,
                I2C_READY_POLL_SETUP_UNAVAILABLE,
            ),
            (
                I2cReadyRouteState::PadConfigurationRefused,
                I2C_READY_POLL_PAD_CONFIGURATION_REFUSED,
            ),
            (
                I2cReadyRouteState::PadAliasRefused,
                I2C_READY_POLL_PAD_ALIAS_REFUSED,
            ),
        ] {
            assert_eq!(gpio_ready_detail(state), detail);
        }
    }

    #[test]
    fn backlog_parks_each_bounded_pass_without_the_idle_poll_delay() {
        assert_eq!(poll_deadline(100, true), 101_000_000);
        assert_eq!(poll_deadline(100, false), 116_000_000);
        assert!(poll_deadline(100, true) > 100_999_999);
        assert_eq!(poll_deadline(u64::MAX, true), u64::MAX);
    }

    #[test]
    fn ready_backstop_degrades_only_when_a_deadline_missed_input() {
        let mut backstop = GpioBackstop::new();
        assert_eq!(
            backstop.period_ms(),
            GPIO_DEGRADED_POLL_MS,
            "an unproven line must start at input latency"
        );

        backstop.parked(true);
        assert_eq!(backstop.period_ms(), GPIO_IRQ_WATCHDOG_MS);

        backstop.parked(false);
        backstop.drained(false);
        assert_eq!(
            backstop.period_ms(),
            GPIO_IRQ_WATCHDOG_MS,
            "an idle device is not evidence of a failed ready line"
        );

        backstop.parked(false);
        backstop.drained(true);
        assert_eq!(backstop.period_ms(), GPIO_DEGRADED_POLL_MS);

        backstop.parked(false);
        backstop.drained(false);
        assert_eq!(
            backstop.period_ms(),
            GPIO_DEGRADED_POLL_MS,
            "stay responsive until an interrupt proves recovery"
        );
    }

    #[test]
    fn ready_interrupt_immediately_restores_the_sparse_watchdog() {
        let mut backstop = GpioBackstop::new();
        assert_eq!(backstop.period_ms(), GPIO_DEGRADED_POLL_MS);

        backstop.parked(true);
        assert_eq!(backstop.period_ms(), GPIO_IRQ_WATCHDOG_MS);
        backstop.drained(true);
        assert_eq!(backstop.period_ms(), GPIO_IRQ_WATCHDOG_MS);
    }

    #[test]
    fn preparation_rejects_missing_geometry_and_never_reinitializes_a_live_queue() {
        // This test alone prepares slot 3; it never creates a worker or
        // touches MMIO. Other protocol tests use slots 0, 1, and 2.
        const SLOT: usize = 3;
        let mut bindings = [None; MAX_CHILDREN];
        bindings[0] = Some(binding(0x2c));
        // SAFETY: no worker exists and each refused call publishes no queue
        // half. The admitted call receives inert, non-null test geometry but
        // never creates a worker or touches it.
        unsafe {
            assert_eq!(ControllerKind::from_platform_model(u32::MAX), None);
            assert!(!PREPARED[SLOT].load(Ordering::Acquire));
            assert!(prepare(SLOT, ControllerKind::PlatformGeni, bindings, 1, None).is_none());
            assert!(!PREPARED[SLOT].load(Ordering::Acquire));
            assert!(prepare(SLOT, ControllerKind::PciDesignWare, bindings, 1, None).is_none());
            assert!(!PREPARED[SLOT].load(Ordering::Acquire));
            let mut byte = 0u8;
            let mapping = Some((core::ptr::addr_of_mut!(byte), 1));
            let mut rx = prepare(SLOT, ControllerKind::PciDesignWare, bindings, 1, mapping)
                .expect("one prepared consumer");
            assert!(PREPARED[SLOT].load(Ordering::Acquire));
            let mut tx = spsc::producer::<InputReport>(REGIONS[SLOT].view()).unwrap();
            let report = InputReport::keyboard(source_base(SLOT, 0), 0, [4, 0, 0, 0, 0, 0]);
            tx.push(report).unwrap();
            assert!(
                prepare(
                    SLOT,
                    ControllerKind::PlatformDesignWare,
                    bindings,
                    1,
                    mapping,
                )
                .is_none(),
                "a duplicate call must not manufacture another consumer"
            );
            assert_eq!(
                rx.pop(),
                Some(report),
                "duplicate prepare must not reset the ring"
            );
            assert_eq!(rx.pop(), None);
        }
    }

    #[test]
    fn controller_gpio_and_watchdog_wakes_publish_independently() {
        let stats = RelayStats::new();
        stats.rung.store(0, Ordering::Relaxed);
        stats.count.store(2, Ordering::Relaxed);
        stats.note_controller_irq();
        stats.note_controller_irq();
        stats.note_gpio_irq();
        stats.note_watchdog_deadline();
        stats.note_watchdog_deadline();
        stats.note_watchdog_deadline();

        let row = controller_diagnostic(0, &stats);
        assert_eq!(row.address_or_count, 2);
        assert_eq!(row.counters, [2, 0, 0, 1, 3]);
        assert!(row.valid());

        stats.controller_irqs.store(u32::MAX, Ordering::Relaxed);
        stats.gpio_irqs.store(u32::MAX, Ordering::Relaxed);
        stats.watchdog_deadlines.store(u32::MAX, Ordering::Relaxed);
        stats.note_controller_irq();
        stats.note_gpio_irq();
        stats.note_watchdog_deadline();
        assert_eq!(
            controller_diagnostic(0, &stats).counters,
            [u32::MAX, 0, 0, u32::MAX, u32::MAX]
        );
    }

    #[test]
    fn healthy_sibling_progress_preserves_the_legacy_failure_witness() {
        let stats = RelayStats::new();
        stats.rung.store(0, Ordering::Relaxed);
        stats.progress(
            phase::STEADY,
            kore_abi::input_status::I2C_READY_POLL_NO_GPIO_ROUTE,
        );
        assert_eq!(
            stats.state.load(Ordering::Relaxed),
            state_word(
                0,
                phase::STEADY,
                failure::NONE,
                kore_abi::input_status::I2C_READY_POLL_NO_GPIO_ROUTE,
                0
            ),
        );
        let refusal = state_word(0, phase::REPORT_DESCRIPTOR, failure::TIMEOUT, 0x1234, 3);
        stats.state.store(refusal, Ordering::Relaxed);
        stats.progress(phase::POWER, 0);
        stats.progress(
            phase::STEADY,
            kore_abi::input_status::I2C_READY_POLL_NO_GPIO_ROUTE,
        );
        assert_eq!(stats.state.load(Ordering::Relaxed), refusal);
    }

    #[test]
    fn a_new_hid_failure_uses_its_child_phase_not_a_sticky_sibling_phase() {
        let stats = RelayStats::new();
        stats.rung.store(0, Ordering::Relaxed);
        stats.record_failure(phase::DESCRIPTOR, failure::TIMEOUT, 0x1234, None);
        stats.progress(phase::RESET_ACK, 0);
        assert_eq!(
            (stats.state.load(Ordering::Relaxed) >> 8) as u8,
            phase::DESCRIPTOR
        );
        assert_eq!(
            bringup_error(
                &stats,
                initialization_phase(Phase::ResetAck),
                BringUpError::ResetTimeout {
                    polls: 2,
                    spent_ns: 1_000_000_000
                },
            ),
            3
        );
        assert_eq!(
            stats.state.load(Ordering::Relaxed),
            state_word(0, phase::RESET_ACK, failure::HID, 2, 0)
        );
    }

    #[test]
    fn both_controller_families_are_admitted_without_fabricated_clocks_or_irqs() {
        use kore_abi::device::DeviceRecord;
        use kore_rt::service::{DeviceSelector, SystemService};
        let cfg = geni_config(binding(0x3a)).unwrap();
        assert_eq!(cfg.input_clock_hz, 19_200_000);
        assert!(!cfg.irq_driven);
        let mut absent = binding(0x15);
        absent.input_clock_hz = None;
        assert!(geni_config(absent).is_none());
        assert!(dw_config(absent).unwrap().irq_driven);
        assert!(!worker_needs_device(ControllerKind::PlatformGeni));
        assert!(worker_needs_device(ControllerKind::PlatformDesignWare));
        assert!(worker_needs_device(ControllerKind::PciDesignWare));
        assert_eq!(
            ControllerKind::from_platform_model(DeviceRecord::PLATFORM_I2C_GENI),
            Some(ControllerKind::PlatformGeni),
        );
        assert_eq!(
            ControllerKind::from_platform_model(DeviceRecord::PLATFORM_I2C_DESIGNWARE),
            Some(ControllerKind::PlatformDesignWare),
        );
        let requests = <super::super::InputService as SystemService>::DEVICES;
        assert_eq!(
            requests[2].selector,
            DeviceSelector::PlatformI2cHid { irq: false }
        );
        assert!(matches!(
            requests[3].selector,
            DeviceSelector::Platform {
                irq: true,
                dma: true,
                ..
            }
        ));
        assert_eq!(
            requests[super::super::device_index::PCI_I2C_HID].selector,
            DeviceSelector::PciI2cHid { irq: true },
        );
        assert_eq!(WORKERS.len(), MAX_CONTROLLERS);
    }

    #[test]
    fn designware_preserves_explicit_firmware_clock_and_timing() {
        let mut binding = binding(0x2c);
        binding.input_clock_hz = Some(133_000_000);
        binding.speed_hz = 100_000;
        binding.standard_timing = Some(kore_abi::session::I2cTimingReport {
            hcnt: 600,
            lcnt: 700,
        });
        let config = dw_config(binding).unwrap();
        assert_eq!(config.input_clock_hz, 133_000_000);
        assert_eq!(
            config.counts,
            Some(Counts {
                hcnt: 600,
                lcnt: 700
            })
        );
        binding.input_clock_hz = None;
        assert_eq!(
            dw_config(binding).unwrap().input_clock_hz,
            Integration::INTEL_LPSS.input_clock_hz
        );
        binding.input_clock_hz = Some(0);
        assert!(dw_config(binding).is_none());
        assert!(geni_config(binding).is_none());
    }

    /// A child that answers every read with a frame the table knows but does
    /// not decode (a vendor report id, here 0xff) is not a backlog. One read
    /// books the outcome and ends the pass unsaturated, so the caller keeps
    /// its GPIO wait or 16 ms backstop instead of the 1 ms re-poll — the
    /// metal symptom was ~740 reads/s and a never-observed ready line.
    #[test]
    fn an_undecodable_frame_ends_the_pass_without_saturation() {
        // Keyboard (id 0x32) plus a vendor collection (id 0xff, one byte).
        const DESCRIPTOR: &[u8] = &[
            0x05, 0x01, 0x09, 0x06, 0xa1, 0x01, 0x85, 0x32, 0x05, 0x07, 0x19, 0xe0, 0x29, 0xe7,
            0x15, 0, 0x25, 1, 0x75, 1, 0x95, 8, 0x81, 2, 0x95, 1, 0x75, 8, 0x81, 1, 0x95, 6, 0x75,
            8, 0x15, 0, 0x26, 0xdd, 0, 0x05, 7, 0x19, 0, 0x2a, 0xdd, 0, 0x81, 0, 0xc0, 0x06, 0x00,
            0xff, 0x09, 0x01, 0xa1, 0x01, 0x85, 0xff, 0x09, 0x02, 0x15, 0, 0x26, 0xff, 0, 0x75, 8,
            0x95, 1, 0x81, 2, 0xc0,
        ];
        struct Bus {
            reads: usize,
        }
        impl I2cBus for Bus {
            type Error = u8;
            fn write(&mut self, _: u8, _: &[u8]) -> Result<(), u8> {
                panic!("input is receive-only")
            }
            fn write_read(
                &mut self,
                address: u8,
                write: &[u8],
                out: &mut [u8],
            ) -> Result<usize, u8> {
                assert_eq!(address, 0x3a);
                assert!(write.is_empty());
                assert_eq!(out.len(), 11);
                out.fill(0);
                // Length 4: two length bytes, report id 0xff, one vendor byte.
                out[0] = 4;
                out[2] = 0xff;
                out[3] = 0x5a;
                self.reads += 1;
                Ok(out.len())
            }
        }
        let mut descriptor = [0u8; 30];
        let words: [u16; 15] = [
            30,
            0x100,
            DESCRIPTOR.len() as u16,
            2,
            3,
            11,
            4,
            16,
            5,
            6,
            0,
            0,
            0,
            0,
            0,
        ];
        for (word, pair) in words.iter().zip(descriptor.chunks_exact_mut(2)) {
            pair.copy_from_slice(&word.to_le_bytes());
        }
        let mut child = Child::new(binding(0x3a));
        child.device = Some(DeviceState {
            descriptor: kore_i2c_hid::HidDescriptor::parse(&descriptor).unwrap(),
            table: build_table(DESCRIPTOR).unwrap(),
            sequences: [0; MAX_REPORTS],
        });
        let books = ChildStats::new();
        let mut bus = Bus { reads: 0 };
        let mut raw = [0u8; MAX_XFER_BYTES];
        let mut decoded =
            [InputReport::keyboard(0, 0, [0; 6]); kore_abi::pointer::MAX_FRAME_RECORDS];
        let mut ring = TestRing([0; RELAY_BYTES]);
        let (mut queue, mut rx) = spsc::split::<InputReport>(&mut ring.0, RELAY_CAPACITY).unwrap();
        let source = source_base(3, 0);
        for pass in 1..=3u32 {
            let saturated = drain_child::<3, _>(
                &mut child,
                false,
                &mut bus,
                &mut raw,
                &mut decoded,
                source,
                &books,
                || Some(10_000_000),
                |e| e,
                &mut queue,
                || {},
            )
            .unwrap();
            assert!(!saturated, "an undecodable frame is not a backlog");
            assert_eq!(bus.reads as u32, pass, "exactly one read per pass");
            assert_eq!(books.polls.load(Ordering::Relaxed), pass);
            assert_eq!(books.reports.load(Ordering::Relaxed), 0);
            assert_eq!(
                books.last_input.load(Ordering::Relaxed),
                I2cInputOutcome::UnsupportedShape.pack(0xff, 4)
            );
            assert_eq!(rx.pop(), None);
        }
    }

    #[test]
    fn shared_drain_preserves_keyboard_edges_and_is_bounded() {
        const DESCRIPTOR: &[u8] = &[
            0x05, 0x01, 0x09, 0x06, 0xa1, 0x01, 0x85, 0x32, 0x05, 0x07, 0x19, 0xe0, 0x29, 0xe7,
            0x15, 0, 0x25, 1, 0x75, 1, 0x95, 8, 0x81, 2, 0x95, 1, 0x75, 8, 0x81, 1, 0x95, 6, 0x75,
            8, 0x15, 0, 0x26, 0xdd, 0, 0x05, 7, 0x19, 0, 0x2a, 0xdd, 0, 0x81, 0, 0xc0,
        ];
        struct Bus {
            reads: usize,
            endless: bool,
            malformed: bool,
            fail: bool,
        }
        impl I2cBus for Bus {
            type Error = u8;
            fn write(&mut self, _: u8, _: &[u8]) -> Result<(), u8> {
                panic!("input is receive-only")
            }
            fn write_read(
                &mut self,
                address: u8,
                write: &[u8],
                out: &mut [u8],
            ) -> Result<usize, u8> {
                assert_eq!(address, 0x3a);
                assert!(write.is_empty());
                assert_eq!(out.len(), 11);
                if self.fail {
                    self.reads += 1;
                    return Err(2);
                }
                out.fill(0);
                if self.endless || self.reads < 5 {
                    out[0] = if self.malformed { 12 } else { 11 };
                    out[2] = if self.reads == 4 { 0x33 } else { 0x32 };
                    if self.reads < 3 {
                        out[5] = 4;
                    }
                    if self.reads == 2 {
                        out[3] = 2;
                    }
                }
                self.reads += 1;
                Ok(out.len())
            }
        }
        let mut descriptor = [0u8; 30];
        let words: [u16; 15] = [
            30,
            0x100,
            DESCRIPTOR.len() as u16,
            2,
            3,
            11,
            4,
            16,
            5,
            6,
            0,
            0,
            0,
            0,
            0,
        ];
        for (word, pair) in words.iter().zip(descriptor.chunks_exact_mut(2)) {
            pair.copy_from_slice(&word.to_le_bytes());
        }
        let mut child = Child::new(binding(0x3a));
        child.device = Some(DeviceState {
            descriptor: kore_i2c_hid::HidDescriptor::parse(&descriptor).unwrap(),
            table: build_table(DESCRIPTOR).unwrap(),
            sequences: [0; MAX_REPORTS],
        });
        let books = ChildStats::new();
        let mut bus = Bus {
            reads: 0,
            endless: false,
            malformed: false,
            fail: false,
        };
        let mut raw = [0u8; MAX_XFER_BYTES];
        let mut decoded =
            [InputReport::keyboard(0, 0, [0; 6]); kore_abi::pointer::MAX_FRAME_RECORDS];
        let mut ring = TestRing([0; RELAY_BYTES]);
        let (mut queue, mut rx) = spsc::split::<InputReport>(&mut ring.0, RELAY_CAPACITY).unwrap();
        let source = source_base(2, 0);
        RELAYS[2].transfers.store(0, Ordering::Relaxed);
        let saturated = drain_child::<2, _>(
            &mut child,
            false,
            &mut bus,
            &mut raw,
            &mut decoded,
            source,
            &books,
            || Some(10_000_000),
            |e| e,
            &mut queue,
            || {},
        )
        .unwrap();
        assert!(!saturated);
        // Four keyboard frames, then the unknown report id 0x33 ends the pass
        // (no evidence of queued input); the empty frame is never read.
        assert_eq!(bus.reads, 5);
        assert_eq!(RELAYS[2].transfers.load(Ordering::Relaxed), 5);
        let reports: [_; 4] = core::array::from_fn(|_| rx.pop().unwrap());
        assert_eq!(rx.pop(), None);
        assert_eq!(
            reports,
            [
                InputReport::keyboard(source, 0, [4, 0, 0, 0, 0, 0]),
                InputReport::keyboard(source, 0, [4, 0, 0, 0, 0, 0]),
                InputReport::keyboard(source, 2, [4, 0, 0, 0, 0, 0]),
                InputReport::keyboard(source, 0, [0; 6]),
            ]
        );
        bus.endless = true;
        let before = bus.reads;
        assert!(
            drain_child::<2, _>(
                &mut child,
                false,
                &mut bus,
                &mut raw,
                &mut decoded,
                source,
                &books,
                || Some(11_000_000),
                |e| e,
                &mut queue,
                || {},
            )
            .unwrap()
        );
        assert_eq!(bus.reads - before, MAX_DRAIN_FRAMES);
        assert_eq!(
            RELAYS[2].transfers.load(Ordering::Relaxed),
            bus.reads as u64
        );
        assert_eq!(published_batch(&mut rx).count(), MAX_DRAIN_FRAMES);
        bus.malformed = true;
        assert_eq!(
            drain_child::<2, _>(
                &mut child,
                false,
                &mut bus,
                &mut raw,
                &mut decoded,
                source,
                &books,
                || Some(12_000_000),
                |e| e,
                &mut queue,
                || {},
            ),
            Err(5)
        );
        assert_eq!(rx.pop(), None, "malformed frame was published");
        let completed = RELAYS[2].transfers.load(Ordering::Relaxed);
        assert_eq!(
            completed, bus.reads as u64,
            "malformed payload still completed I/O"
        );
        bus.fail = true;
        assert_eq!(
            drain_child::<2, _>(
                &mut child,
                false,
                &mut bus,
                &mut raw,
                &mut decoded,
                source,
                &books,
                || Some(13_000_000),
                |e| e,
                &mut queue,
                || {},
            ),
            Err(2)
        );
        assert_eq!(rx.pop(), None, "failed read was published");
        assert_eq!(RELAYS[2].transfers.load(Ordering::Relaxed), completed);
        assert_eq!(books.polls.load(Ordering::Relaxed) as usize, bus.reads);
    }

    #[repr(align(64))]
    struct TestRing([u8; RELAY_BYTES]);

    fn pointer_child(address: u8) -> Child {
        // A relative pointer with a wheel emits five records per packet.
        const REPORT: &[u8] = &[
            0x05, 1, 0x09, 2, 0xa1, 1, 0x09, 0x30, 0x09, 0x31, 0x09, 0x38, 0x15, 0x81, 0x25, 0x7f,
            0x75, 8, 0x95, 3, 0x81, 6, 0xc0,
        ];
        let mut descriptor = [0u8; 30];
        let words: [u16; 15] = [
            30,
            0x100,
            REPORT.len() as u16,
            2,
            3,
            5,
            4,
            3,
            5,
            6,
            0,
            0,
            0,
            0,
            0,
        ];
        for (word, pair) in words.iter().zip(descriptor.chunks_exact_mut(2)) {
            pair.copy_from_slice(&word.to_le_bytes());
        }
        let mut child = Child::new(binding(address));
        child.device = Some(DeviceState {
            descriptor: kore_i2c_hid::HidDescriptor::parse(&descriptor).unwrap(),
            table: build_table(REPORT).unwrap(),
            sequences: [0; MAX_REPORTS],
        });
        child
    }

    struct PointerBus<'a> {
        reads: [usize; 2],
        clock: &'a core::cell::Cell<u64>,
        fail: bool,
    }
    impl I2cBus for PointerBus<'_> {
        type Error = u8;
        fn write(&mut self, _: u8, _: &[u8]) -> Result<(), u8> {
            panic!("input is receive-only")
        }
        fn write_read(&mut self, address: u8, write: &[u8], out: &mut [u8]) -> Result<usize, u8> {
            assert!(write.is_empty());
            self.reads[usize::from(address - 0x2c)] += 1;
            self.clock.set(self.clock.get() + 500_000);
            if self.fail {
                return Err(2);
            }
            out.copy_from_slice(&[5, 0, 1, 1, 1]);
            Ok(out.len())
        }
    }

    #[test]
    fn full_queue_defers_device_reads_then_resumes_whole_timestamped_frames() {
        use kore_abi::pointer::Record;
        let clock = core::cell::Cell::new(10_000_000);
        let mut bus = PointerBus {
            reads: [0; 2],
            clock: &clock,
            fail: false,
        };
        let mut child = pointer_child(0x2c);
        let books = ChildStats::new();
        let mut raw = [0; MAX_XFER_BYTES];
        let mut decoded = [InputReport::keyboard(0, 0, [0; 6]); REPORT_RESERVE as usize];
        let mut ring = TestRing([0; RELAY_BYTES]);
        let (mut tx, mut rx) = spsc::split::<InputReport>(&mut ring.0, RELAY_CAPACITY).unwrap();
        for _ in 0..125 {
            tx.push(InputReport::keyboard(0, 0, [0; 6])).unwrap();
        }
        assert!(
            drain_child::<0, _>(
                &mut child,
                false,
                &mut bus,
                &mut raw,
                &mut decoded,
                source_base(0, 0),
                &books,
                || Some(clock.get()),
                |e| e,
                &mut tx,
                || {},
            )
            .unwrap()
        );
        assert_eq!(bus.reads, [0, 0], "no consume-then-drop under backpressure");
        assert_eq!(child.device.as_ref().unwrap().sequences, [0; MAX_REPORTS]);
        assert_eq!(books.polls.load(Ordering::Relaxed), 0);
        assert!(needs_report_kick(&tx), "an old backlog still needs a wake");
        assert_eq!(published_batch(&mut rx).count(), 125);
        assert!(
            drain_child::<0, _>(
                &mut child,
                false,
                &mut bus,
                &mut raw,
                &mut decoded,
                source_base(0, 0),
                &books,
                || Some(clock.get()),
                |e| e,
                &mut tx,
                || {},
            )
            .unwrap()
        );
        assert!(bus.reads[0] > 1 && bus.reads[0] < MAX_DRAIN_FRAMES);
        assert_eq!(tx.dropped(), 0);
        for sequence in 1..=bus.reads[0] as u32 {
            let frame: [_; 5] =
                core::array::from_fn(|_| Record::decode(rx.pop().unwrap()).unwrap());
            assert_eq!(
                frame[0],
                Record::Begin {
                    sequence,
                    received_ns: Some(10_000_000 + u64::from(sequence) * 500_000),
                    class: 0,
                }
            );
            assert!(
                matches!(frame[4], Record::End { sequence: s, generation: 1, complete: true, .. } if s == sequence)
            );
        }
        assert_eq!(rx.pop(), None);
        assert!(!needs_report_kick(&tx));
    }

    #[test]
    fn complete_pointer_frames_wake_before_the_next_bus_read() {
        use kore_abi::pointer::Record;
        let clock = core::cell::Cell::new(10_000_000);
        let mut bus = PointerBus {
            reads: [0; 2],
            clock: &clock,
            fail: false,
        };
        let mut child = pointer_child(0x2c);
        let books = ChildStats::new();
        let mut raw = [0; MAX_XFER_BYTES];
        let mut decoded = [InputReport::mouse(0, 0, 0, 0, 0); REPORT_RESERVE as usize];
        let mut ring = TestRing([0; RELAY_BYTES]);
        let (mut tx, mut rx) = spsc::split::<InputReport>(&mut ring.0, RELAY_CAPACITY).unwrap();
        let mut wakes = 0;
        assert!(drain_child::<0, _>(
            &mut child,
            false, &mut bus, &mut raw, &mut decoded, source_base(0, 0),
            &books, || Some(clock.get()), |e| e, &mut tx, || {
                wakes += 1;
                let frame: [_; 5] = core::array::from_fn(|_| Record::decode(rx.pop().unwrap()).unwrap());
                assert!(matches!(frame[0], Record::Begin { sequence, received_ns: Some(t), .. }
                    if sequence == wakes && t == clock.get()));
                assert!(matches!(frame[4], Record::End { sequence, complete: true, .. } if sequence == wakes));
                assert_eq!(clock.get(), 10_000_000 + u64::from(wakes) * 500_000,
                    "no later read may delay this complete frame's notification");
                assert_eq!(rx.pop(), None, "only the completed frame is published");
            },
        ).unwrap());
        assert_eq!(wakes as usize, MAX_DRAIN_FRAMES);
        assert_eq!(tx.dropped(), 0);
    }

    #[test]
    fn held_level_route_rearms_after_each_complete_frame_without_speculative_reads() {
        use kore_abi::pointer::Record;
        use kore_abi::session::I2cReadyInterruptReport;
        let clock = core::cell::Cell::new(0);
        let mut bus = PointerBus {
            reads: [0; 2],
            clock: &clock,
            fail: false,
        };
        let mut child = pointer_child(0x2c);
        child.binding.ready_interrupt = Some(I2cReadyInterruptReport::Line {
            gsi: 84,
            edge: false,
            active_high: false,
        });
        let mut backstop = GpioBackstop::new();
        assert!(
            !rearm_after_report(child.binding, true, &backstop),
            "route not proven yet"
        );
        backstop.parked(true);
        assert!(!rearm_after_report(child.binding, false, &backstop));
        backstop.parked(false);
        assert!(
            !rearm_after_report(child.binding, true, &backstop),
            "deadline/refusal must drain"
        );
        backstop.parked(true);
        let level_ready = rearm_after_report(child.binding, true, &backstop);
        assert!(level_ready);
        let books = ChildStats::new();
        let mut raw = [0; MAX_XFER_BYTES];
        let mut decoded = [InputReport::mouse(0, 0, 0, 0, 0); REPORT_RESERVE as usize];
        let mut ring = TestRing([0; RELAY_BYTES]);
        let (mut tx, mut rx) = spsc::split::<InputReport>(&mut ring.0, RELAY_CAPACITY).unwrap();
        // Pending input keeps the level asserted at every rearm. Each wake
        // delivers exactly one frame, including its timestamp and sequence.
        for sequence in 1..=MAX_DRAIN_FRAMES as u32 {
            let mut wakes = 0;
            assert!(
                !drain_child::<0, _>(
                    &mut child,
                    level_ready,
                    &mut bus,
                    &mut raw,
                    &mut decoded,
                    source_base(0, 0),
                    &books,
                    || Some(clock.get()),
                    |e| e,
                    &mut tx,
                    || {
                        wakes += 1;
                    },
                )
                .unwrap(),
                "must rearm, not take the saturated sleep"
            );
            assert_eq!(wakes, 1);
            assert_eq!(books.polls.load(Ordering::Relaxed), sequence);
            let frame: [_; 5] =
                core::array::from_fn(|_| Record::decode(rx.pop().unwrap()).unwrap());
            assert!(matches!(frame[0], Record::Begin { sequence: s, .. } if s == sequence));
            assert!(
                matches!(frame[4], Record::End { sequence: s, complete: true, .. } if s == sequence)
            );
            assert_eq!(rx.pop(), None);
        }
        child.binding.ready_interrupt = Some(I2cReadyInterruptReport::Line {
            gsi: 84,
            edge: true,
            active_high: false,
        });
        assert!(
            !rearm_after_report(child.binding, true, &backstop),
            "edge coalescing still needs a drain"
        );
        child.binding.ready_interrupt = None;
        assert!(!rearm_after_report(child.binding, true, &backstop));
        assert_eq!(tx.dropped(), 0);
    }

    #[test]
    fn saturated_children_rotate_first_access_to_the_shared_record_budget() {
        let clock = core::cell::Cell::new(0);
        let mut bus = PointerBus {
            reads: [0; 2],
            clock: &clock,
            fail: false,
        };
        let mut children = [pointer_child(0x2c), pointer_child(0x2d)];
        let books = [ChildStats::new(), ChildStats::new()];
        let mut raw = [0; MAX_XFER_BYTES];
        let mut decoded = [InputReport::keyboard(0, 0, [0; 6]); REPORT_RESERVE as usize];
        let mut ring = TestRing([0; RELAY_BYTES]);
        let (mut tx, mut rx) = spsc::split::<InputReport>(&mut ring.0, RELAY_CAPACITY).unwrap();
        for first in 0..2 {
            let before = bus.reads;
            for i in child_order(first).filter(|&i| i < 2) {
                assert!(
                    drain_child::<0, _>(
                        &mut children[i],
                        false,
                        &mut bus,
                        &mut raw,
                        &mut decoded,
                        source_base(0, i),
                        &books[i],
                        || Some(clock.get()),
                        |e| e,
                        &mut tx,
                        || {},
                    )
                    .unwrap()
                );
            }
            assert!(bus.reads[first] > before[first]);
            assert_eq!(bus.reads[1 - first], before[1 - first]);
            assert_eq!(tx.dropped(), 0);
            assert!(published_batch(&mut rx).all(|record| record.source == source_base(0, first)));
        }
    }

    #[test]
    fn failure_resets_reduce_the_next_siblings_admission_budget() {
        use kore_abi::pointer::Record;
        let clock = core::cell::Cell::new(0);
        let mut bus = PointerBus {
            reads: [0; 2],
            clock: &clock,
            fail: true,
        };
        let mut failed = pointer_child(0x2c);
        let mut sibling = pointer_child(0x2d);
        let books = ChildStats::new();
        let mut raw = [0; MAX_XFER_BYTES];
        let mut decoded = [InputReport::keyboard(0, 0, [0; 6]); REPORT_RESERVE as usize];
        let mut ring = TestRing([0; RELAY_BYTES]);
        let (mut tx, mut rx) = spsc::split::<InputReport>(&mut ring.0, RELAY_CAPACITY).unwrap();
        for _ in 0..RELAY_CAPACITY - REPORT_RESERVE {
            tx.push(InputReport::keyboard(0, 0, [0; 6])).unwrap();
        }
        assert!(room_for_report(&tx));
        assert_eq!(
            drain_child::<0, _>(
                &mut failed,
                false,
                &mut bus,
                &mut raw,
                &mut decoded,
                source_base(0, 0),
                &books,
                || Some(clock.get()),
                |e| e,
                &mut tx,
                || {},
            ),
            Err(2)
        );
        failed.fail(1, &mut tx, source_base(0, 0));
        assert!(!room_for_report(&tx));
        assert!(
            drain_child::<0, _>(
                &mut sibling,
                false,
                &mut bus,
                &mut raw,
                &mut decoded,
                source_base(0, 1),
                &books,
                || Some(clock.get()),
                |e| e,
                &mut tx,
                || {},
            )
            .unwrap()
        );
        assert_eq!(bus.reads, [1, 0]);
        assert_eq!(tx.dropped(), 0);
        for _ in 0..RELAY_CAPACITY - REPORT_RESERVE {
            rx.pop().unwrap();
        }
        for stream in 0..MAX_REPORTS {
            let reset = rx.pop().unwrap();
            assert_eq!(reset.source, source_base(0, 0) + stream as u8);
            assert_eq!(Record::decode(reset), Ok(Record::Reset { all: false }));
        }
        assert_eq!(rx.pop(), None);
    }

    fn binding(address: u8) -> I2cHidReport {
        I2cHidReport {
            slave_addr: address,
            descriptor_register: 1,
            speed_hz: 400_000,
            standard_timing: None,
            fast_timing: Some(kore_abi::session::I2cTimingReport {
                hcnt: 80,
                lcnt: 160,
            }),
            input_clock_hz: Some(19_200_000),
            ready_interrupt: None,
            ready_route_state: kore_abi::session::I2cReadyRouteState::NotDescribed,
            no_irq: false,
            no_power_after_reset: false,
        }
    }

    /// Small standard descriptors with the Yoga's configuration layout, not
    /// device-identity quirks. Input is one numbered finger/contact snapshot.
    fn digitizer_descriptor(screen: bool, mixed_mode: bool, out: &mut [u8]) -> usize {
        const INPUT: &[u8] = &[
            0x05, 0x0d, 0x09, 0x05, 0xa1, 1, 0x85, 3, 0x09, 0x22, 0xa1, 2, 0x09, 0x42, 0x09, 0x47,
            0x15, 0, 0x25, 1, 0x75, 1, 0x95, 2, 0x81, 2, 0x75, 6, 0x95, 1, 0x81, 3, 0x09, 0x51,
            0x15, 0, 0x25, 0x7f, 0x75, 8, 0x95, 1, 0x81, 2, 0x05, 1, 0x09, 0x30, 0x09, 0x31, 0x15,
            0, 0x26, 0xff, 0x7f, 0x75, 16, 0x95, 2, 0x81, 2, 0xc0, 0x05, 0x0d, 0x09, 0x54, 0x15, 0,
            0x25, 5, 0x75, 8, 0x95, 1, 0x81, 2, 0xc0,
        ];
        const PAD_FEATURES: &[u8] = &[
            0x05, 0x0d, 0x09, 0x0e, 0xa1, 1, 0x85, 4, 0x09, 0x52, 0x15, 0, 0x25, 10, 0x75, 8, 0x95,
            1, 0xb1, 2, 0x85, 6, 0x09, 0x57, 0x09, 0x58, 0x15, 0, 0x25, 1, 0x75, 1, 0x95, 2, 0xb1,
            2, 0x75, 6, 0x95, 1, 0xb1, 3, 0xc0,
        ];
        const PAD_FEATURES_MIXED: &[u8] = &[
            0x05, 0x0d, 0x09, 0x0e, 0xa1, 1, 0x85, 4, 0x09, 0x52, 0x15, 0, 0x25, 10, 0x75, 8, 0x95,
            1, 0xb1, 2, 0x06, 0, 0xff, 0x09, 1, 0x26, 0xff, 0, 0x75, 8, 0x95, 1, 0xb1, 2, 0x05,
            0x0d, 0x85, 6, 0x09, 0x57, 0x09, 0x58, 0x15, 0, 0x25, 1, 0x75, 1, 0x95, 2, 0xb1, 2,
            0x75, 6, 0x95, 1, 0xb1, 3, 0xc0,
        ];
        const SCREEN_FEATURES: &[u8] = &[
            0x05, 0x0d, 0x09, 0x0e, 0xa1, 1, 0x85, 0x0e, 0x09, 0x52, 0x09, 0x53, 0x15, 0, 0x25, 10,
            0x75, 8, 0x95, 2, 0xb1, 2, 0xc0,
        ];
        out[..INPUT.len()].copy_from_slice(INPUT);
        if screen {
            out[3] = 4; // Touchscreen application, not the touchpad application.
            out[7] = 0x1c;
        }
        let features = if screen {
            SCREEN_FEATURES
        } else if mixed_mode {
            PAD_FEATURES_MIXED
        } else {
            PAD_FEATURES
        };
        out[INPUT.len()..INPUT.len() + features.len()].copy_from_slice(features);
        INPUT.len() + features.len()
    }

    #[derive(Default)]
    struct DigitizerStub {
        mode: u8,
        switches: u8,
        identifier: u8,
        resets: u32,
        gets: u32,
        sets: u32,
        samples: usize,
        fail_mode_write: bool,
        report_descriptor_seen: bool,
        mode_read: bool,
        malformed_mode_reply: u8,
        invalid_mode_range: bool,
        invalid_data_register: bool,
    }

    struct DigitizerBus {
        screen: [bool; 2],
        devices: [DigitizerStub; 2],
    }

    impl DigitizerBus {
        fn new(screen: bool) -> Self {
            Self {
                screen: [screen; 2],
                devices: core::array::from_fn(|_| DigitizerStub::default()),
            }
        }
        fn index(address: u8) -> usize {
            assert!((0x2c..=0x2d).contains(&address));
            usize::from(address - 0x2c)
        }
    }

    impl I2cBus for DigitizerBus {
        type Error = u8;

        fn write(&mut self, address: u8, bytes: &[u8]) -> Result<(), u8> {
            let index = Self::index(address);
            let device = &mut self.devices[index];
            if bytes == [5, 0, 0, 8] {
                return Ok(());
            }
            if bytes == [5, 0, 0, 1] {
                device.resets += 1;
                device.mode = 0;
                device.switches = 0xa0;
                device.identifier = 7;
                device.samples = 0;
                device.report_descriptor_seen = false;
                device.mode_read = false;
                return Ok(());
            }
            assert!(
                device.resets > 0 && device.report_descriptor_seen,
                "feature writes must follow reset and report descriptor discovery"
            );
            let mode = if self.screen[index] { 2 } else { 3 };
            let id = if self.screen[index] { 0x0e } else { 4 };
            if bytes[2] == 0x30 | id {
                if self.screen[index] {
                    assert_eq!(bytes, [5, 0, 0x3e, 3, 6, 0, 5, 0, 0x0e, 2, 0]);
                } else if device.mode_read {
                    assert_eq!(bytes, [5, 0, 0x34, 3, 6, 0, 5, 0, 4, 3, 0xa5]);
                } else {
                    assert_eq!(bytes, [5, 0, 0x34, 3, 6, 0, 4, 0, 4, 3]);
                }
                if device.fail_mode_write {
                    return Err(2);
                }
                device.mode = mode;
                device.identifier = 0;
            } else {
                assert!(!self.screen[index]);
                assert_eq!(bytes, [5, 0, 0x36, 3, 6, 0, 4, 0, 6, 3]);
                device.switches = 3;
            }
            device.sets += 1;
            Ok(())
        }

        fn write_read(&mut self, address: u8, write: &[u8], out: &mut [u8]) -> Result<usize, u8> {
            let index = Self::index(address);
            let mut report = [0u8; 256];
            let mixed_mode = (1..=3).contains(&self.devices[index].malformed_mode_reply);
            let report_len = digitizer_descriptor(self.screen[index], mixed_mode, &mut report);
            if write == [1, 0] {
                assert_eq!(out.len(), 30);
                let mut words = [
                    30,
                    0x100,
                    report_len as u16,
                    2,
                    3,
                    10,
                    4,
                    16,
                    5,
                    6,
                    0,
                    0,
                    0,
                    0,
                    0,
                ];
                if self.devices[index].invalid_data_register {
                    words[9] = 0;
                }
                for (word, pair) in words.iter().zip(out.chunks_exact_mut(2)) {
                    pair.copy_from_slice(&word.to_le_bytes());
                }
            } else if write == [2, 0] {
                assert!(self.devices[index].resets > 0);
                if self.devices[index].invalid_mode_range {
                    let offset = report[..report_len]
                        .windows(6)
                        .position(|bytes| bytes == [0x09, 0x52, 0x15, 0, 0x25, 10])
                        .unwrap();
                    report[offset + 5] = 2; // cannot represent touchpad mode 3
                }
                out.copy_from_slice(&report[..report_len]);
                self.devices[index].report_descriptor_seen = true;
            } else if write.is_empty() {
                out.fill(0);
                if out.len() == 2 {
                    return Ok(2); // Reset ACK, never an input report.
                }
                assert_eq!(out.len(), 10);
                let device = &mut self.devices[index];
                assert_eq!(device.mode, if self.screen[index] { 2 } else { 3 });
                if !self.screen[index] {
                    assert_eq!(device.switches & 3, 3);
                }
                if device.samples < 3 {
                    let active = device.samples < 2;
                    let (x, y) = if device.samples == 0 {
                        (1000u16, 2000u16)
                    } else {
                        (2000, 2500)
                    };
                    out[..4].copy_from_slice(&[
                        10,
                        0,
                        if self.screen[index] { 0x1c } else { 3 },
                        if active { 3 } else { 0 },
                    ]);
                    out[4] = 7;
                    out[5..7].copy_from_slice(&x.to_le_bytes());
                    out[7..9].copy_from_slice(&y.to_le_bytes());
                    out[9] = u8::from(active);
                    device.samples += 1;
                }
            } else {
                let device = &mut self.devices[index];
                assert!(
                    device.resets > 0 && device.report_descriptor_seen,
                    "feature reads must follow reset and report descriptor discovery"
                );
                assert!(!self.screen[index]);
                assert_eq!(write, [5, 0, 0x34, 2, 6, 0]);
                out.copy_from_slice(&[5, 0, 4, device.mode, 0xa5]);
                device.mode_read = true;
                match device.malformed_mode_reply {
                    1 => out[2] = 6,
                    2 => out[..2].copy_from_slice(&u16::MAX.to_le_bytes()),
                    3 => return Ok(out.len() - 1),
                    _ => {}
                }
                device.gets += 1;
            }
            Ok(out.len())
        }
    }

    fn configure_digitizer(child: &mut Child, bus: &mut DigitizerBus, start: u64) {
        let mut raw = [0u8; MAX_XFER_BYTES];
        child.begin(start);
        for now in (start..start + 640).step_by(16) {
            if child.device.is_some() {
                return;
            }
            init_step::<3, _>(child, bus, &mut raw, now, || Some(now * 1_000_000), |e| e).unwrap();
            if child.pending.is_some() {
                assert!(child.device.is_none());
                assert!(
                    child.bringup.is_none(),
                    "feature replies must not reenter descriptor parsing"
                );
            }
        }
        assert!(
            child.device.is_some(),
            "bounded configuration failed to complete"
        );
    }

    #[test]
    fn configured_touchpad_and_touchscreen_frames_reach_the_real_pointer_seat() {
        use core::cell::Cell;
        use kore_axiom::pointer::{Dispatch, PointerSeat, Viewport, touchpad::TouchpadPolicy};
        for screen in [false, true] {
            let mut bus = DigitizerBus::new(screen);
            let mut child = Child::new(binding(0x2c));
            configure_digitizer(&mut child, &mut bus, 0);
            assert_eq!(bus.devices[0].mode, if screen { 2 } else { 3 });
            assert_eq!(bus.devices[0].identifier, 0);
            assert_eq!(bus.devices[0].gets, 0);
            assert_eq!(bus.devices[0].sets, if screen { 1 } else { 2 });
            assert!(child.pending.is_none());
            let books = ChildStats::new();
            let clock = Cell::new(640_000_000u64);
            let mut raw = [0u8; MAX_XFER_BYTES];
            let mut decoded =
                [InputReport::mouse(0, 0, 0, 0, 0); kore_abi::pointer::MAX_FRAME_RECORDS];
            let mut driver_ring = TestRing([0; RELAY_BYTES]);
            let (mut driver_tx, mut driver_rx) =
                spsc::split::<InputReport>(&mut driver_ring.0, RELAY_CAPACITY).unwrap();
            assert!(
                !drain_child::<3, _>(
                    &mut child,
                    false,
                    &mut bus,
                    &mut raw,
                    &mut decoded,
                    source_base(3, 0),
                    &books,
                    || {
                        clock.set(clock.get() + 1_000_000);
                        Some(clock.get())
                    },
                    |e| e,
                    &mut driver_tx,
                    || {},
                )
                .unwrap()
            );
            assert_eq!(books.reports.load(Ordering::Relaxed), 3);
            assert_eq!(
                books.last_input.load(Ordering::Relaxed),
                I2cInputOutcome::Decoded.pack(if screen { 0x1c } else { 3 }, 10)
            );

            // The pointer plane is an event channel, whose region carries the
            // state block and wake words on top of the ring — `TestRing` is
            // sized for a bare relay ring and cannot hold it.
            #[repr(align(64))]
            struct PointerWindow(
                [u8;
                    kore_ring::event::region_bytes::<kore_abi::pointer::PointerState, InputReport>(
                        RELAY_CAPACITY,
                    )],
            );
            let mut pointer_ring = PointerWindow(
                [0;
                    kore_ring::event::region_bytes::<kore_abi::pointer::PointerState, InputReport>(
                        RELAY_CAPACITY,
                    )],
            );
            let (tx, mut rx) = kore_ring::event::split::<
                kore_abi::pointer::PointerState,
                InputReport,
            >(&mut pointer_ring.0, RELAY_CAPACITY)
            .unwrap();
            let mut pointer = crate::service::PointerLane::default();
            pointer.attach(tx);
            // The compositor's own seat: gestures for the touchpad, the
            // compatibility policy for the touchscreen.
            let mut seat =
                PointerSeat::with_policy(Viewport::new(1000, 800).unwrap(), TouchpadPolicy::default());
            let initial = seat.position();
            for record in published_batch(&mut driver_rx) {
                assert_eq!(pointer.route(record), Some(true));
            }
            let mut frames = 0;
            while let Some(record) = rx.pop() {
                seat.feed_with(record, &mut |event| {
                    if let Dispatch::Frame(frame) = event {
                        frames += 1;
                        if frames <= 2 {
                            assert_eq!(frame.contacts[0].id, 7);
                            assert_eq!(frame.contacts[0].confidence, Some(true));
                        } else {
                            assert!(frame.contacts.is_empty(), "lift must clear the contact");
                        }
                    }
                });
            }
            assert_eq!(frames, 3);
            assert_eq!(seat.rejected_frames, 0);
            if screen {
                assert!(
                    seat.position().x < 100 && seat.position().y < 100,
                    "touchscreen must map absolute coordinates rather than accumulate deltas"
                );
            } else {
                assert!(
                    seat.position().x > initial.x && seat.position().y > initial.y,
                    "the second same-contact sample must move the touchpad pointer"
                );
            }
        }
    }

    #[test]
    fn mode_failure_never_goes_online_and_reset_replays_configuration() {
        let mut bus = DigitizerBus::new(false);
        bus.devices[0].fail_mode_write = true;
        let mut children = [Child::new(binding(0x2c)), Child::new(binding(0x2d))];
        let mut raw = [0u8; MAX_XFER_BYTES];
        let mut ring = TestRing([0; RELAY_BYTES]);
        let (mut tx, mut rx) = spsc::split::<InputReport>(&mut ring.0, RELAY_CAPACITY).unwrap();
        let mut refusal_seen = false;
        for now in (0..400).step_by(16) {
            for (index, child) in children.iter_mut().enumerate() {
                if child.device.is_some() || !child.retry.ready(now) {
                    continue;
                }
                if !child.initializing() {
                    child.begin(now);
                }
                if let Err(error) = init_step::<3, _>(
                    child,
                    &mut bus,
                    &mut raw,
                    now,
                    || Some(now * 1_000_000),
                    |e| e,
                ) {
                    assert_eq!(index, 0);
                    assert_eq!(error, 2);
                    assert!(child.device.is_none());
                    assert_eq!(
                        child.pending.as_ref().unwrap().features.phase(),
                        FeaturePhase::Set
                    );
                    child.fail(now, &mut tx, source_base(3, index));
                    assert!(!child.initializing());
                    assert_eq!(child.generation, 1);
                    refusal_seen = true;
                }
            }
        }
        assert!(refusal_seen);
        assert!(children[0].device.is_none());
        assert!(
            children[1].device.is_some(),
            "a refused mode must not starve the healthy sibling"
        );
        assert_eq!(
            rx.pop(),
            None,
            "unconfigured devices have no published history to reset"
        );
        let previous_sets = bus.devices[1].sets;
        let previous_resets = bus.devices[1].resets;
        children[1].fail(400, &mut tx, source_base(3, 1));
        assert_eq!(published_batch(&mut rx).count(), MAX_REPORTS);
        configure_digitizer(&mut children[1], &mut bus, 800);
        assert_eq!(bus.devices[1].resets, previous_resets + 1);
        assert_eq!(bus.devices[1].sets, previous_sets + 2);
        assert_eq!(bus.devices[1].mode, 3);
        assert_eq!(bus.devices[1].switches & 3, 3);
        assert_eq!(children[1].generation, 2);
    }

    #[test]
    fn feature_refusals_keep_per_child_phase_report_and_error_class() {
        use kore_abi::input_status::I2cProtocolObservation;
        for fault in 1..=5 {
            let mut bus = DigitizerBus::new(false);
            bus.devices[0].malformed_mode_reply = fault;
            bus.devices[0].invalid_mode_range = fault == 4;
            bus.devices[0].invalid_data_register = fault == 5;
            let mut child = Child::new(binding(0x2c));
            child.begin(0);
            let books = ChildStats::new();
            let mut raw = [0; MAX_XFER_BYTES];
            let mut observed = false;
            for now in (0..640).step_by(16) {
                let result = init_step::<3, _>(
                    &mut child,
                    &mut bus,
                    &mut raw,
                    now,
                    || Some(now * 1_000_000),
                    |e| e,
                );
                // The exact publisher used after drive_bus's child step.
                books.observe_step(&mut child, result);
                if result.is_err() {
                    assert!(
                        child.device.is_none(),
                        "refused configuration became online"
                    );
                    let protocol =
                        I2cProtocolObservation::from_bits(books.snapshot().protocol.get()).unwrap();
                    let (phase, report_id, detail) = match fault {
                        1 => (I2cProtocolPhase::FeatureGet, 4, 0x0606),
                        2 => (I2cProtocolPhase::FeatureGet, 4, 0x7f05),
                        3 => (I2cProtocolPhase::FeatureGet, 4, 0x8005),
                        4 => (I2cProtocolPhase::Table, 4, 0x0501),
                        5 => (I2cProtocolPhase::FeatureGet, 0, 2),
                        _ => unreachable!(),
                    };
                    assert_eq!(protocol.phase, phase);
                    assert_eq!(protocol.report_id, report_id);
                    assert_eq!(protocol.detail, detail);
                    assert_eq!(
                        bus.devices[0].sets, 0,
                        "invalid GET must never generate SET"
                    );
                    observed = true;
                    break;
                }
            }
            assert!(
                observed,
                "fault {fault} never reached the production refusal path"
            );
        }
    }

    #[test]
    fn pending_feature_configuration_obeys_the_original_child_deadline_without_io() {
        struct NoIo;
        impl I2cBus for NoIo {
            type Error = u8;
            fn write(&mut self, _: u8, _: &[u8]) -> Result<(), u8> {
                panic!("expired child must not write");
            }
            fn write_read(&mut self, _: u8, _: &[u8], _: &mut [u8]) -> Result<usize, u8> {
                panic!("expired child must not read");
            }
        }
        let mut bus = DigitizerBus::new(false);
        let mut child = Child::new(binding(0x2c));
        child.begin(0);
        let mut raw = [0; MAX_XFER_BYTES];
        for now in (0..640).step_by(16) {
            init_step::<3, _>(
                &mut child,
                &mut bus,
                &mut raw,
                now,
                || Some(now * 1_000_000),
                |e| e,
            )
            .unwrap();
            if child.pending.is_some() {
                break;
            }
        }
        assert!(child.pending.is_some());
        assert!(child.device.is_none());
        assert_eq!(child.check_init_deadline(INIT_BUDGET_MS - 1), Ok(()));
        assert_eq!(
            init_step::<3, _>(
                &mut child,
                &mut NoIo,
                &mut raw,
                INIT_BUDGET_MS,
                || Some(INIT_BUDGET_MS * 1_000_000),
                |e| e,
            ),
            Err(3)
        );
        assert!(child.device.is_none());
        assert_eq!(
            child.pending.as_ref().unwrap().features.phase(),
            FeaturePhase::Set
        );
    }

    fn awaiting_reset(at_ns: u64, no_irq: bool) -> Child {
        use kore_i2c_hid::bringup::POWER_SETTLE_NS;
        let mut binding = binding(0x2c);
        binding.no_irq = no_irq;
        let mut child = Child::new(binding);
        child.begin(at_ns / 1_000_000);
        let bringup = child.bringup.as_mut().unwrap();
        let mut descriptor = [0u8; 30];
        let words: [u16; 15] = [30, 0x100, 18, 2, 3, 3, 4, 3, 5, 6, 0, 0, 0, 0, 0];
        for (word, pair) in words.iter().zip(descriptor.chunks_exact_mut(2)) {
            pair.copy_from_slice(&word.to_le_bytes());
        }
        assert!(matches!(bringup.step(), Step::WriteRead { .. }));
        bringup.read(&descriptor, 0).unwrap();
        assert!(matches!(bringup.step(), Step::Write { .. }));
        bringup.wrote().unwrap();
        assert_eq!(
            bringup.step(),
            Step::Delay {
                ns: POWER_SETTLE_NS
            }
        );
        bringup.delayed(POWER_SETTLE_NS).unwrap();
        assert!(matches!(bringup.step(), Step::Write { .. }));
        bringup.wrote().unwrap();
        child.lap = Lap(Some(at_ns));
        child
    }

    #[test]
    fn reset_budget_includes_sibling_time_and_the_completed_bus_read() {
        use core::cell::Cell;
        use kore_i2c_hid::bringup::RESET_POLL_NS;
        struct SlowRead<'a>(&'a Cell<u64>);
        impl I2cBus for SlowRead<'_> {
            type Error = u8;
            fn write(&mut self, _: u8, _: &[u8]) -> Result<(), u8> {
                panic!("no write while awaiting the reset acknowledgement")
            }
            fn write_read(&mut self, _: u8, write: &[u8], out: &mut [u8]) -> Result<usize, u8> {
                assert!(write.is_empty());
                out.fill(0xff); // not an acknowledgement or a valid report
                self.0.set(self.0.get() + 75_000_000);
                Ok(out.len())
            }
        }
        let clock = Cell::new(10_000_000);
        let mut bus = SlowRead(&clock);
        let mut child = awaiting_reset(clock.get(), false);
        let mut raw = [0u8; 30];
        let step = |child: &mut Child, bus: &mut SlowRead<'_>, raw: &mut [u8], now| {
            init_step::<0, _>(child, bus, raw, now, || Some(clock.get()), |e| e)
        };
        step(&mut child, &mut bus, &mut raw, 10).unwrap();
        clock.set(10_900_000);
        step(&mut child, &mut bus, &mut raw, 10).unwrap();
        assert_eq!(
            child.bringup.as_mut().unwrap().step(),
            Step::Delay { ns: RESET_POLL_NS }
        );
        assert_eq!(child.bringup.as_ref().unwrap().reset_spent_ns(), 0);

        // Another child runs while this one is waiting. Its independent
        // anchor must not erase time from the first child's reset budget.
        clock.set(800_000_000);
        let mut sibling = awaiting_reset(clock.get(), false);
        step(&mut sibling, &mut bus, &mut raw, 800).unwrap();
        clock.set(900_000_000);
        step(&mut child, &mut bus, &mut raw, 900).unwrap();
        assert_eq!(
            child.bringup.as_ref().unwrap().reset_spent_ns(),
            890_000_000
        );
        assert_eq!(sibling.bringup.as_ref().unwrap().reset_spent_ns(), 0);

        clock.set(950_000_000);
        assert_eq!(step(&mut child, &mut bus, &mut raw, 950), Err(3));
        let bringup = child.bringup.as_ref().unwrap();
        assert_eq!(
            bringup.polls(),
            1,
            "wall time, not 1,000 nominal polls, exhausted the budget"
        );
        assert_eq!(bringup.reset_spent_ns(), 1_015_000_000);
    }

    #[test]
    fn no_irq_reset_delay_is_only_acknowledged_after_its_deadline() {
        use kore_i2c_hid::bringup::NO_IRQ_RESET_NS;
        struct NoIo;
        impl I2cBus for NoIo {
            type Error = u8;
            fn write(&mut self, _: u8, _: &[u8]) -> Result<(), u8> {
                panic!("delay must not touch the bus")
            }
            fn write_read(&mut self, _: u8, _: &[u8], _: &mut [u8]) -> Result<usize, u8> {
                panic!("no-IRQ reset must not poll")
            }
        }
        let mut child = awaiting_reset(10_000_000, true);
        for now in [10, 50, 109] {
            init_step::<0, _>(
                &mut child,
                &mut NoIo,
                &mut [],
                now,
                || Some(now * 1_000_000),
                |e| e,
            )
            .unwrap();
            assert_eq!(
                child.bringup.as_mut().unwrap().step(),
                Step::Delay {
                    ns: NO_IRQ_RESET_NS
                }
            );
            assert_eq!(child.bringup.as_ref().unwrap().reset_spent_ns(), 0);
        }
        init_step::<0, _>(
            &mut child,
            &mut NoIo,
            &mut [],
            110,
            || Some(110_000_000),
            |e| e,
        )
        .unwrap();
        assert!(matches!(
            child.bringup.as_mut().unwrap().step(),
            Step::Write { .. }
        ));
        assert_eq!(
            child.bringup.as_ref().unwrap().reset_spent_ns(),
            NO_IRQ_RESET_NS
        );
        assert!(!child.delay_pending);
    }

    #[test]
    fn second_power_settle_defers_descriptor_io_without_blocking_a_sibling() {
        use core::cell::Cell;
        use kore_i2c_hid::{
            bringup::POWER_SETTLE_NS,
            command::{Command, PowerState},
        };

        const MS: u64 = 1_000_000;
        const REPORT: &[u8] = &[
            0x05, 1, 0x09, 2, 0xa1, 1, 0x09, 0x30, 0x15, 0x81, 0x25, 0x7f, 0x75, 8, 0x95, 1, 0x81,
            6, 0xc0,
        ];
        // Fixed-state bus: the descriptor access itself enforces the same
        // compatibility interval as the ladder, not a test-only step count.
        struct PowerAwareBus<'a> {
            clock: &'a Cell<u64>,
            power_ons: u8,
            second_power_completed: Option<u64>,
            report_reads: u8,
            sibling_reads: u8,
        }
        impl I2cBus for PowerAwareBus<'_> {
            type Error = u8;
            fn write(&mut self, address: u8, bytes: &[u8]) -> Result<(), u8> {
                assert_eq!(address, 0x2c);
                if bytes == Command::set_power(5, PowerState::On).bytes() {
                    self.power_ons += 1;
                    if self.power_ons == 2 {
                        // Settlement is measured after success, not at the
                        // earlier start of a non-instantaneous transaction.
                        self.clock.set(self.clock.get() + 3 * MS);
                        self.second_power_completed = Some(self.clock.get());
                    }
                } else {
                    assert_eq!(bytes, Command::reset(5).bytes());
                }
                Ok(())
            }
            fn write_read(
                &mut self,
                address: u8,
                write: &[u8],
                out: &mut [u8],
            ) -> Result<usize, u8> {
                if address == 0x2d {
                    assert_eq!(write, [1, 0]);
                    self.sibling_reads += 1;
                } else {
                    assert_eq!(address, 0x2c);
                }
                if write == [1, 0] {
                    let words: [u16; 15] = [
                        30,
                        0x100,
                        REPORT.len() as u16,
                        2,
                        3,
                        3,
                        4,
                        3,
                        5,
                        6,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ];
                    assert_eq!(out.len(), 30);
                    for (word, pair) in words.iter().zip(out.chunks_exact_mut(2)) {
                        pair.copy_from_slice(&word.to_le_bytes());
                    }
                } else if write == [2, 0] {
                    let completed = self.second_power_completed.expect("second power succeeded");
                    assert!(
                        self.clock.get() >= completed + POWER_SETTLE_NS,
                        "report descriptor accessed before the second power settle"
                    );
                    self.report_reads += 1;
                    out.copy_from_slice(REPORT);
                } else {
                    assert!(write.is_empty(), "reset acknowledgement is a plain receive");
                    assert_eq!(out.len(), 2);
                    out.fill(0);
                }
                Ok(out.len())
            }
        }

        let clock = Cell::new(0u64);
        let mut bus = PowerAwareBus {
            clock: &clock,
            power_ons: 0,
            second_power_completed: None,
            report_reads: 0,
            sibling_reads: 0,
        };
        let mut child = Child::new(binding(0x2c));
        child.begin(0);
        let mut raw = [0u8; 30];
        let step = |child: &mut Child, bus: &mut PowerAwareBus<'_>, raw: &mut [u8]| {
            init_step::<0, _>(
                child,
                bus,
                raw,
                clock.get() / MS,
                || Some(clock.get()),
                |e| e,
            )
        };
        // Execute the real descriptor, power, scheduled settle, reset, and
        // acknowledgement steps. The second power completes at 70ms.
        for now in [0, 1, 2, 62, 63, 64, 65, 66, 67] {
            clock.set(now * MS);
            step(&mut child, &mut bus, &mut raw).unwrap();
        }
        assert_eq!(bus.power_ons, 2);
        assert_eq!(bus.second_power_completed, Some(70 * MS));
        let reset_spent = child.bringup.as_ref().unwrap().reset_spent_ns();
        step(&mut child, &mut bus, &mut raw).unwrap();
        assert_eq!(child.retry.ready_at, 130);
        assert!(child.delay_pending);
        assert_eq!(bus.report_reads, 0);

        clock.set(100 * MS);
        let mut sibling = Child::new(binding(0x2d));
        sibling.begin(100);
        step(&mut sibling, &mut bus, &mut raw).unwrap();
        assert_eq!(
            bus.sibling_reads, 1,
            "a delayed child did not stall its sibling"
        );
        assert!(sibling.bringup.as_ref().unwrap().descriptor().is_some());
        assert_eq!(child.retry.ready_at, 130);

        for now_ns in [110 * MS, 129 * MS, 130 * MS - 1] {
            clock.set(now_ns);
            step(&mut child, &mut bus, &mut raw).unwrap();
            assert_eq!(bus.report_reads, 0);
            assert_eq!(child.bringup.as_ref().unwrap().phase(), Phase::Power);
            assert_eq!(
                child.bringup.as_mut().unwrap().step(),
                Step::Delay {
                    ns: POWER_SETTLE_NS
                }
            );
        }
        clock.set(130 * MS);
        step(&mut child, &mut bus, &mut raw).unwrap();
        assert_eq!(bus.report_reads, 0, "one protocol step per controller pass");
        assert!(!child.delay_pending);
        assert_eq!(
            child.bringup.as_ref().unwrap().phase(),
            Phase::ReportDescriptor
        );
        assert_eq!(
            child.bringup.as_ref().unwrap().reset_spent_ns(),
            reset_spent,
            "post-reset power settling must not spend the acknowledgement budget"
        );
        step(&mut child, &mut bus, &mut raw).unwrap();
        assert_eq!(bus.report_reads, 1);
        assert!(
            child.device.is_some(),
            "descriptor parsing completed after settlement"
        );
    }

    #[test]
    fn real_init_steps_isolate_nack_and_never_sleep_a_healthy_child() {
        const REPORT: &[u8] = &[
            0x05, 1, 0x09, 2, 0xa1, 1, 0x09, 0x30, 0x15, 0x81, 0x25, 0x7f, 0x75, 8, 0x95, 1, 0x81,
            6, 0xc0,
        ];
        struct Bus {
            failed: u32,
            completed: u32,
        }
        impl I2cBus for Bus {
            type Error = u8;
            fn write(&mut self, _: u8, _: &[u8]) -> Result<(), u8> {
                self.completed += 1;
                Ok(())
            }
            fn write_read(
                &mut self,
                address: u8,
                write: &[u8],
                out: &mut [u8],
            ) -> Result<usize, u8> {
                if address == 0x2d {
                    self.failed += 1;
                    return Err(2);
                }
                if write == [1, 0] {
                    let words: [u16; 15] = [
                        30,
                        0x100,
                        REPORT.len() as u16,
                        2,
                        3,
                        3,
                        4,
                        3,
                        5,
                        6,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ];
                    for (word, pair) in words.iter().zip(out.chunks_exact_mut(2)) {
                        pair.copy_from_slice(&word.to_le_bytes());
                    }
                } else if write == [2, 0] {
                    out.copy_from_slice(REPORT);
                } else {
                    assert!(write.is_empty(), "reset acknowledgement is a plain receive");
                    out.fill(0);
                }
                self.completed += 1;
                Ok(out.len())
            }
        }
        #[repr(align(64))]
        struct Ring([u8; 4096]);
        let mut ring = Ring([0; 4096]);
        let (mut queue, mut rx) = spsc::split::<InputReport>(&mut ring.0, 128).unwrap();
        // This test alone drives slot 1; the other protocol tests use slot 0.
        // Keep the legacy transfer witness independent of steady report polls.
        RELAYS[1].transfers.store(0, Ordering::Relaxed);
        let mut bus = Bus {
            failed: 0,
            completed: 0,
        };
        let mut children = [Child::new(binding(0x2d)), Child::new(binding(0x2c))];
        let mut raw = [0u8; MAX_XFER_BYTES];
        // Both power commands now receive their scheduled compatibility
        // settle. Keep the real 16ms cadence through descriptor completion.
        for now in (0..304).step_by(16) {
            for (i, child) in children.iter_mut().enumerate() {
                if child.device.is_some() || !child.retry.ready(now) {
                    continue;
                }
                if !child.initializing() {
                    child.begin(now);
                }
                if init_step::<1, _>(
                    child,
                    &mut bus,
                    &mut raw,
                    now,
                    || Some(now * 1_000_000),
                    |e| e,
                )
                .is_err()
                {
                    child.fail(now, &mut queue, source_base(1, i));
                }
            }
        }
        assert_eq!(
            bus.failed, 2,
            "failed child retries once after its 250ms backoff, not each pass"
        );
        assert!(bus.completed > 0);
        assert_eq!(books()[1].transfers, bus.completed);
        assert!(children[0].device.is_none());
        assert!(
            children[1].device.is_some(),
            "sibling completed the actual protocol ladder"
        );
        assert_eq!(
            rx.pop(),
            None,
            "failed initialization has no live pointer history to reset"
        );
        children[1].fail(304, &mut queue, source_base(1, 1));
        for stream in 0..MAX_REPORTS {
            let record = rx.pop().unwrap();
            assert_eq!(record.source, source_base(1, 1) + stream as u8);
            assert_eq!(
                kore_abi::pointer::Record::decode(record),
                Ok(kore_abi::pointer::Record::Reset { all: false })
            );
        }
        children[1].fail(305, &mut queue, source_base(1, 1));
        assert_eq!(rx.pop(), None, "backoff does not generate repeated resets");
        assert_eq!(children[1].generation, 2);
        assert!(
            core::mem::size_of::<[Option<Child>; MAX_CHILDREN]>() + MAX_XFER_BYTES < 32 * 1024,
            "controller state must fit its bounded worker stack"
        );
    }

    #[test]
    fn per_child_firmware_timing_is_preserved() {
        assert_eq!(
            dw_config(binding(0x2c))
                .expect("valid fixture clock")
                .counts,
            Some(Counts {
                hcnt: 80,
                lcnt: 160
            })
        );
    }

    #[test]
    fn address_aliases_are_rejected_only_within_their_controller() {
        let first = [Some(Child::new(binding(0x2c)))];
        assert!(!address_is_new(&first, 0x2c));
        assert!(address_is_new(&first, 0x2d));
        assert!(
            address_is_new(&[], 0x2c),
            "same address on a different bus is legal"
        );
    }

    #[test]
    fn draining_a_live_producer_stops_at_the_original_complete_frame() {
        use kore_abi::pointer::Record;
        #[repr(align(64))]
        struct Ring([u8; RELAY_BYTES]);
        let mut ring = Ring([0; RELAY_BYTES]);
        let (mut tx, mut rx) = spsc::split::<InputReport>(&mut ring.0, RELAY_CAPACITY).unwrap();
        let frame = [
            Record::Begin {
                sequence: 1,
                received_ns: None,
                class: 0,
            },
            Record::Buttons(0),
            Record::End {
                sequence: 1,
                generation: 1,
                records: 1,
                reported_contacts: None,
                complete: true,
            },
        ]
        .map(|record| record.encode(source_base(0, 0)));
        assert!(tx.push_slice_lossy(&frame));
        {
            let mut batch = published_batch(&mut rx);
            assert_eq!(batch.next(), Some(frame[0]));
            // Model the worker running between consumer pops, without a
            // timing-dependent host thread or allocator-backed fixture.
            assert!(tx.push_slice_lossy(&frame));
            assert_eq!(batch.next(), Some(frame[1]));
            assert_eq!(batch.next(), Some(frame[2]));
            assert_eq!(batch.next(), None);
        }
        assert_eq!(rx.outstanding(), frame.len() as u32);
        assert!(published_batch(&mut rx).eq(frame));
    }
    #[test]
    fn controller_child_and_report_namespaces_are_disjoint() {
        let mut used = [false; 256];
        for c in 0..MAX_CONTROLLERS {
            for d in 0..MAX_CHILDREN {
                for r in 0..MAX_REPORTS {
                    let source = source_base(c, d) as usize + r;
                    assert!(source > super::super::SOURCE_LEGACY as usize);
                    assert!(!used[source]);
                    used[source] = true;
                }
            }
        }
        assert!(used[255]);
    }
    #[test]
    fn failing_child_backs_off_without_delaying_healthy_sibling() {
        let mut failed = Retry::default();
        let healthy = Retry::default();
        failed.fail(100);
        assert!(!failed.ready(349));
        assert!(failed.ready(350));
        assert!(healthy.ready(101));
        for _ in 0..200 {
            failed.fail(1_000);
        }
        assert!(!failed.ready(4_999));
        assert!(failed.ready(5_000));
    }
    #[test]
    fn input_receive_never_writes_a_register_address() {
        struct Bus;
        impl I2cBus for Bus {
            type Error = ();
            fn write(&mut self, _: u8, _: &[u8]) -> Result<(), ()> {
                panic!("input must not write")
            }
            fn write_read(&mut self, addr: u8, write: &[u8], read: &mut [u8]) -> Result<usize, ()> {
                assert_eq!(addr, 0x2c);
                assert!(write.is_empty());
                Ok(read.len())
            }
        }
        assert_eq!(read_input(&mut Bus, 0x2c, &mut [0; 20]), Ok(20));
    }
    #[test]
    fn worker_authority_matches_ps2_and_has_no_dma() {
        for (controller, resource) in [
            (
                ControllerKind::PlatformDesignWare,
                ResourceKind::PlatformDevice,
            ),
            (ControllerKind::PciDesignWare, ResourceKind::PciDevice),
        ] {
            let slot = kore_abi::session::SlotRef::from_wire(9);
            let dev = endowment(slot, controller);
            assert_eq!(dev.kind, resource as u16 as u8);
            assert_eq!(dev.subject, slot);
            assert_eq!(
                dev.rights,
                Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
            );
        }
        let kick = kick_endowment(kore_abi::session::SlotRef::from_wire(10));
        assert_eq!((kick.role, kick.index), (role::DOORBELL, SERVICE_KICK_PEER));
    }
}
