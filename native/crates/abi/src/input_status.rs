//! Bounded, inert input diagnostics. No MMIO addresses, capabilities or UART output.
//! The input owner publishes these books through its ordinary service reply;
//! input-stream carries them to the launcher for `status input`.

use zerocopy::byteorder::{LittleEndian, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub const MAX_I2C_CONTROLLERS: usize = 4;
pub const MAX_I2C_CHILDREN: usize = 4;
pub const I2C_STATUS_ROWS: u8 = (MAX_I2C_CONTROLLERS * (MAX_I2C_CHILDREN + 1)) as u8;
pub const MAX_USB_CONTROLLERS: usize = 4;
pub const USB_STATUS_BASE: u8 = I2C_STATUS_ROWS;
pub const USB_STATUS_PARTS: u8 = 3;
pub const POINTER_STATUS_BASE: u8 = USB_STATUS_BASE + MAX_USB_CONTROLLERS as u8 * USB_STATUS_PARTS;
pub const ADMISSION_STATUS_BASE: u8 = POINTER_STATUS_BASE + 2;
pub const INPUT_STATUS_ROWS: u8 = ADMISSION_STATUS_BASE + crate::device::I2C_ADMISSION_ROWS;

/// xHCI intake diagnostic detail: the controller is live and retained for
/// hotplug, but launch-time class discovery produced no HID source. The input
/// service carries this under bind stage 5; unlike other stage-5 details this
/// is an operating posture, not a bind refusal.
pub const USB_BOUND_NO_HID_DETAIL: u64 = 0x86;

/// xHCI stage-0 (running) posture bits.  Zero is the plain HID-backed bind;
/// these record what the firmware → OS handoff found in the controller kore
/// inherited — facts about the machine, not refusals.
///
/// A Debug Capability was running (`DCCTRL.DCE` set) at handoff and was
/// disabled before the first operational write.  A DbC survives HCRST and
/// keeps fetching its debug context from firmware's buffers under kore's
/// IOMMU context; the recurring single reason-6 read fault against the xHC
/// on Lunar Lake was its signature.
pub const USB_RUNNING_DBC_DISABLED_DETAIL: u64 = 1 << 0;
/// A running Debug Capability did not report stopped within the bring-up
/// budget; the controller was brought up anyway (the IOMMU refuses its
/// fetches, so the cost is a fault-census line, not memory).
pub const USB_RUNNING_DBC_STUCK_DETAIL: u64 = 1 << 1;

/// Last operation reached by one I2C-HID child, not the controller's startup rung.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum I2cProtocolPhase {
    #[default]
    None = 0,
    Claim = 1,
    Probe = 2,
    Configure = 3,
    Descriptor = 4,
    Power = 5,
    Reset = 6,
    ResetAck = 7,
    ReportDescriptor = 8,
    Table = 9,
    Steady = 10,
    FeatureGet = 11,
    FeatureSet = 12,
    /// Initialization worker is inside the kernel-clock sampling trap.
    InitClock = 13,
    /// Initialization worker is inside its deadline sleep trap.
    InitSleep = 14,
}

impl I2cProtocolPhase {
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0 => Some(Self::None),
            1 => Some(Self::Claim),
            2 => Some(Self::Probe),
            3 => Some(Self::Configure),
            4 => Some(Self::Descriptor),
            5 => Some(Self::Power),
            6 => Some(Self::Reset),
            7 => Some(Self::ResetAck),
            8 => Some(Self::ReportDescriptor),
            9 => Some(Self::Table),
            10 => Some(Self::Steady),
            11 => Some(Self::FeatureGet),
            12 => Some(Self::FeatureSet),
            13 => Some(Self::InitClock),
            14 => Some(Self::InitSleep),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Claim => "claim",
            Self::Probe => "probe",
            Self::Configure => "configure",
            Self::Descriptor => "descriptor",
            Self::Power => "power",
            Self::Reset => "reset",
            Self::ResetAck => "reset-ack",
            Self::ReportDescriptor => "report-descriptor",
            Self::Table => "table",
            Self::Steady => "steady",
            Self::FeatureGet => "feature-get",
            Self::FeatureSet => "feature-set",
            Self::InitClock => "init-clock",
            Self::InitSleep => "init-sleep",
        }
    }

    /// `None` is canonical only with zero report ID and detail.
    pub const fn pack(self, report_id: u8, detail: u16) -> u32 {
        self as u32 | (report_id as u32) << 8 | (detail as u32) << 16
    }
}

/// Bounded operation metadata; never HID payload bytes. A detail with bit 15
/// set is `0x8000 | I2cChildStatus::error`; otherwise it is the feature ladder's
/// diagnostic code. Keep it hexadecimal rather than guessing a shared enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I2cProtocolObservation {
    pub phase: I2cProtocolPhase,
    pub report_id: u8,
    pub detail: u16,
}

impl I2cProtocolObservation {
    pub const fn from_bits(bits: u32) -> Option<Self> {
        let Some(phase) = I2cProtocolPhase::from_bits(bits as u8) else {
            return None;
        };
        if matches!(phase, I2cProtocolPhase::None) && bits != 0 {
            return None;
        }
        Some(Self {
            phase,
            report_id: (bits >> 8) as u8,
            detail: (bits >> 16) as u16,
        })
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum I2cInputOutcome {
    #[default]
    Unseen = 0,
    Empty = 1,
    UnknownReport = 2,
    Decoded = 3,
    UnsupportedShape = 4,
    Malformed = 5,
    DecodeRefused = 6,
}

impl I2cInputOutcome {
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0 => Some(Self::Unseen),
            1 => Some(Self::Empty),
            2 => Some(Self::UnknownReport),
            3 => Some(Self::Decoded),
            4 => Some(Self::UnsupportedShape),
            5 => Some(Self::Malformed),
            6 => Some(Self::DecodeRefused),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unseen => "unseen",
            Self::Empty => "empty",
            Self::UnknownReport => "unknown-report",
            Self::Decoded => "decoded",
            Self::UnsupportedShape => "unsupported-shape",
            Self::Malformed => "malformed",
            Self::DecodeRefused => "decode-refused",
        }
    }

    /// `Unseen` is canonical only with zero report ID and wire length.
    pub const fn pack(self, report_id: u8, wire_length: u16) -> u32 {
        self as u32 | (report_id as u32) << 8 | (wire_length as u32) << 16
    }
}

/// Geometry of the last nonempty input or refusal; empty polls do not erase it.
/// Before any such observation, the first empty poll may replace `Unseen`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I2cInputObservation {
    pub outcome: I2cInputOutcome,
    pub report_id: u8,
    pub wire_length: u16,
}

impl I2cInputObservation {
    pub const fn from_bits(bits: u32) -> Option<Self> {
        let Some(outcome) = I2cInputOutcome::from_bits(bits as u8) else {
            return None;
        };
        if matches!(outcome, I2cInputOutcome::Unseen) && bits != 0 {
            return None;
        }
        Some(Self {
            outcome,
            report_id: (bits >> 8) as u8,
            wire_length: (bits >> 16) as u16,
        })
    }
}

/// One closed, indexed, data-only diagnostic page. The original I2C rows keep
/// their wire representation; USB, pointer-ring and firmware-admission books
/// occupy disjoint appended ranges. No page grants device or interrupt access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputDiagnostic {
    pub row: u8,
    pub state: u8,
    pub address_or_count: u8,
    pub error: u8,
    pub flags: u8,
    pub counters: [u32; 5],
}

impl InputDiagnostic {
    pub fn from_i2c(row: I2cDiagnostic) -> Self {
        Self {
            row: row.row,
            state: row.state,
            address_or_count: row.address_or_count,
            error: row.error,
            flags: row.flags,
            counters: row.counters,
        }
    }

    pub fn valid(&self) -> bool {
        if self.row < I2C_STATUS_ROWS {
            return I2cDiagnostic::from_parts(self.state, self.packed(), self.counters).is_some();
        }
        if self.address_or_count != 0 || self.error != 0 || self.flags != 0 {
            return false;
        }
        match self.row {
            USB_STATUS_BASE..POINTER_STATUS_BASE => {
                self.state == 0
                    && ((self.row - USB_STATUS_BASE) % USB_STATUS_PARTS != 2
                        || self.counters[2..] == [0; 3])
            }
            POINTER_STATUS_BASE => self.state <= 1,
            row if row == POINTER_STATUS_BASE + 1 => {
                self.state == 0 && self.counters[0] <= self.counters[1] && self.counters[4] == 0
            }
            ADMISSION_STATUS_BASE..INPUT_STATUS_ROWS => {
                self.state <= 1
                    && self.counters[2..] == [0; 3]
                    && crate::device::I2cAdmissionPage {
                        row: self.row - ADMISSION_STATUS_BASE,
                        words: [self.counters[0], self.counters[1]],
                    }
                    .valid()
            }
            _ => false,
        }
    }

    pub fn packed(&self) -> u32 {
        self.row as u32
            | (self.address_or_count as u32) << 8
            | (self.error as u32) << 16
            | (self.flags as u32) << 24
    }

    pub fn from_parts(state: u8, packed: u32, counters: [u32; 5]) -> Option<Self> {
        let row = Self {
            row: packed as u8,
            state,
            address_or_count: (packed >> 8) as u8,
            error: (packed >> 16) as u8,
            flags: (packed >> 24) as u8,
            counters,
        };
        row.valid().then_some(row)
    }
}

/// One indexed diagnostic reply, fitting the existing service record sizes.
/// Rows are controller, then its four child slots. Counter order is
/// controller: completion IRQs, drops, packed failure class and detail byte,
/// GPIO-ready IRQs, watchdog deadlines;
/// child: polls, reports, errors, packed protocol observation, packed last input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I2cDiagnostic {
    pub row: u8,
    pub state: u8,
    pub address_or_count: u8,
    pub error: u8,
    pub flags: u8,
    pub counters: [u32; 5],
}

/// A running I2C relay has firmware GPIOs but could not derive a pad route.
pub const I2C_READY_POLL_NO_GPIO_ROUTE: u32 = 1;
/// A running I2C relay has no firmware source to use as a ready interrupt.
pub const I2C_READY_POLL_NO_FIRMWARE_IRQ: u32 = 2;
/// A running I2C relay parks on its GPIO-ready route; polling is a watchdog.
pub const I2C_READY_GPIO_IRQ: u32 = 3;
pub const I2C_READY_POLL_NO_COMMUNITY: u32 = 4;
pub const I2C_READY_POLL_NO_LAYOUT: u32 = 5;
pub const I2C_READY_POLL_NO_APERTURE: u32 = 6;
pub const I2C_READY_POLL_NO_PARENT_LINE: u32 = 7;
pub const I2C_READY_POLL_PAD_OUT_OF_RANGE: u32 = 8;
pub const I2C_READY_POLL_NO_CATALOG_DEVICE: u32 = 9;
pub const I2C_READY_POLL_NO_REGISTER_VIEW: u32 = 10;
pub const I2C_READY_POLL_QUIESCE_REFUSED: u32 = 11;
pub const I2C_READY_POLL_PARENT_ROUTE_REFUSED: u32 = 12;
pub const I2C_READY_POLL_PAD_BIND_REFUSED: u32 = 13;
pub const I2C_READY_POLL_SETUP_UNAVAILABLE: u32 = 14;
pub const I2C_READY_POLL_PAD_CONFIGURATION_REFUSED: u32 = 15;
pub const I2C_READY_POLL_PAD_ALIAS_REFUSED: u32 = 16;
/// The child uses its firmware GSI through the ordinary line fabric.
pub const I2C_READY_LINE_IRQ: u32 = 17;
pub const I2C_READY_POLL_LINE_BIND_REFUSED: u32 = 18;

impl I2cDiagnostic {
    pub fn valid(&self) -> bool {
        self.row < I2C_STATUS_ROWS
            && self.flags & !1 == 0
            && self.error <= 9
            && if self.row as usize % (MAX_I2C_CHILDREN + 1) == 0 {
                self.state <= 13 && self.address_or_count as usize <= MAX_I2C_CHILDREN
            } else {
                self.state <= 4
                    && self.address_or_count <= 0x7f
                    && I2cProtocolObservation::from_bits(self.counters[3]).is_some()
                    && I2cInputObservation::from_bits(self.counters[4]).is_some()
            }
    }
    pub fn packed(&self) -> u32 {
        self.row as u32
            | (self.address_or_count as u32) << 8
            | (self.error as u32) << 16
            | (self.flags as u32) << 24
    }
    pub fn from_parts(state: u8, packed: u32, counters: [u32; 5]) -> Option<Self> {
        let row = Self {
            row: packed as u8,
            state,
            address_or_count: (packed >> 8) as u8,
            error: (packed >> 16) as u8,
            flags: (packed >> 24) as u8,
            counters,
        };
        row.valid().then_some(row)
    }
}

/// A child keeps its slot even when bring-up fails: sibling source IDs never move.
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct I2cChildStatus {
    pub address: u8,
    /// 0 absent, 1 initializing, 2 online, 3 retry backoff, 4 disabled.
    pub phase: u8,
    /// Last failure: 0 none, 1 config, 2 NACK, 3 timeout, 4 bus,
    /// 5 protocol/length, 6 descriptor parse, 7 clock, 8 generation exhausted,
    /// 9 duplicate slave address on this controller.
    pub error: u8,
    /// Bit 0: firmware described a child interrupt; NOT a claim that it is routed.
    pub flags: u8,
    /// All HID input reads, including IRQ-driven reads; not a fallback-poll count.
    pub polls: U32<LittleEndian>,
    pub reports: U32<LittleEndian>,
    pub errors: U32<LittleEndian>,
    /// See [`I2cProtocolObservation`]; carried in child diagnostic counter 3.
    pub protocol: U32<LittleEndian>,
    /// See [`I2cInputObservation`]; carried in child diagnostic counter 4.
    pub last_input: U32<LittleEndian>,
}

#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct I2cControllerStatus {
    /// Relay startup rung; 0 running, 3 unclaimed. Other values name refusals.
    pub rung: u8,
    pub children: u8,
    /// The refusal class behind `rung`, from the worker's published state word.
    /// A rung alone says a resource was refused; this says which kind — `11`
    /// the owner-side window map, `12` the pool-free register/route intake.
    /// `0` when the rung carries no failure of its own.
    pub failure: u8,
    /// That failure's detail, low 24 bits. When `failure == 0` and the relay
    /// is running, one of the `I2C_READY_*` constants instead names its
    /// child-ready wake mode.
    ///
    /// For intake (`failure == 12`) this is `IntakeError::diagnostic_code`:
    /// byte 0 the error class, and for the BAR-map class byte 1 the BAR index
    /// and byte 2 the stage-coded `BarMapFault` — its low three bits the stage
    /// (1 slot exhaustion, 2/3 claim build/commit, 4/5 map build/commit, 6
    /// geometry) and the rest the kernel status for the commit stages.
    ///
    /// The class alone cannot separate those stages, and they sit in
    /// different subsystems, so a metal refusal is only actionable with the
    /// whole word. 24 bits is what one wire counter has left beside the class
    /// byte; every intake code observed on this path fits it.
    pub detail: U32<LittleEndian>,
    /// Controller transfer-completion IRQs, not HID GPIO-ready IRQs.
    pub wakes: U32<LittleEndian>,
    pub dropped: U32<LittleEndian>,
    /// Child readiness route fires (GPIO or GSI), separate from controller transfer completion.
    pub gpio_wakes: U32<LittleEndian>,
    /// Child-route wait deadlines that fell through to the bounded polling watchdog.
    pub watchdog_deadlines: U32<LittleEndian>,
    pub child: [I2cChildStatus; MAX_I2C_CHILDREN],
}

/// One owned xHC (not a USB port). Event counters saturate at u32::MAX;
/// the packed admission stage/detail retains its full 64-bit value.
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct UsbControllerStatus {
    pub state: U64<LittleEndian>,
    pub identity: U32<LittleEndian>,
    pub sources: U32<LittleEndian>,
    pub discovery_errors: U32<LittleEndian>,
    pub irq_wakes: U32<LittleEndian>,
    pub irq_delivered: U32<LittleEndian>,
    pub irq_events: U32<LittleEndian>,
    pub irq_errors: U32<LittleEndian>,
    pub reports: U32<LittleEndian>,
    pub dropped: U32<LittleEndian>,
    pub source_exhausted: U32<LittleEndian>,
}

/// Observations at the input owner's existing SPSC producer. Consumption means
/// dequeued by the compositor, NOT accepted by its gesture parser or presented.
/// No extra backchannel, shared mutable telemetry or display authority is needed.
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct PointerStatus {
    pub attached: U32<LittleEndian>,
    pub reports: U32<LittleEndian>,
    pub publications: U64<LittleEndian>,
    pub dropped: U64<LittleEndian>,
    pub queued: U32<LittleEndian>,
    pub capacity: U32<LittleEndian>,
    pub rejected: U32<LittleEndian>,
    pub resets: U32<LittleEndian>,
}

impl PointerStatus {
    /// Advisory sampled count; the live consumer can advance between pages.
    pub fn consumed(&self) -> u64 {
        self.publications
            .get()
            .saturating_sub(self.dropped.get())
            .saturating_sub(u64::from(self.queued.get()))
    }
}

#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct InputSnapshot {
    /// Launcher's monotonic capture time. Rows are sampled, not atomically frozen.
    pub captured_ns: U64<LittleEndian>,
    /// 0 unavailable, 1 available. Empty topology is available, not a failure.
    pub available: U32<LittleEndian>,
    pub reserved: [u8; 4],
    pub i2c: [I2cControllerStatus; MAX_I2C_CONTROLLERS],
    pub usb: [UsbControllerStatus; MAX_USB_CONTROLLERS],
    pub pointer: PointerStatus,
    /// Separate from transport availability: a failed firmware observation is
    /// not an empty list and does not hide live USB/relay counters.
    pub admission_available: U32<LittleEndian>,
    pub admission: crate::device::I2cAdmissionSnapshot,
}

impl InputSnapshot {
    pub fn insert_diagnostic(&mut self, row: InputDiagnostic) -> bool {
        if !row.valid() {
            return false;
        }
        let w = row.counters;
        if row.row < I2C_STATUS_ROWS {
            return self.insert(I2cDiagnostic::from_parts(row.state, row.packed(), w).unwrap());
        }
        match row.row {
            USB_STATUS_BASE..POINTER_STATUS_BASE => {
                let offset = row.row - USB_STATUS_BASE;
                let c = &mut self.usb[usize::from(offset / USB_STATUS_PARTS)];
                match offset % USB_STATUS_PARTS {
                    0 => {
                        c.state.set(u64::from(w[0]) | u64::from(w[1]) << 32);
                        c.identity.set(w[2]);
                        c.sources.set(w[3]);
                        c.discovery_errors.set(w[4]);
                    }
                    1 => {
                        c.irq_wakes.set(w[0]);
                        c.irq_delivered.set(w[1]);
                        c.irq_events.set(w[2]);
                        c.irq_errors.set(w[3]);
                        c.reports.set(w[4]);
                    }
                    _ => {
                        c.dropped.set(w[0]);
                        c.source_exhausted.set(w[1]);
                    }
                }
            }
            POINTER_STATUS_BASE => {
                self.pointer.attached.set(u32::from(row.state));
                self.pointer.reports.set(w[0]);
                self.pointer
                    .publications
                    .set(u64::from(w[1]) | u64::from(w[2]) << 32);
                self.pointer
                    .dropped
                    .set(u64::from(w[3]) | u64::from(w[4]) << 32);
            }
            row if row == POINTER_STATUS_BASE + 1 => {
                self.pointer.queued.set(w[0]);
                self.pointer.capacity.set(w[1]);
                self.pointer.rejected.set(w[2]);
                self.pointer.resets.set(w[3]);
            }
            ADMISSION_STATUS_BASE..INPUT_STATUS_ROWS => {
                if !self.admission.insert(crate::device::I2cAdmissionPage {
                    row: row.row - ADMISSION_STATUS_BASE,
                    words: [w[0], w[1]],
                }) {
                    return false;
                }
                if row.row == ADMISSION_STATUS_BASE {
                    self.admission_available.set(u32::from(row.state));
                }
            }
            _ => return false,
        }
        true
    }

    pub fn insert(&mut self, row: I2cDiagnostic) -> bool {
        if !row.valid() {
            return false;
        }
        let index = row.row as usize;
        let c = &mut self.i2c[index / (MAX_I2C_CHILDREN + 1)];
        match index % (MAX_I2C_CHILDREN + 1) {
            0 => {
                c.rung = row.state;
                c.children = row.address_or_count;
                c.wakes.set(row.counters[0]);
                c.dropped.set(row.counters[1]);
                // The producer packs the state word's failure class and the
                // low 24 bits of its detail into counter 2. Counters 3 and 4
                // distinguish child-ready route fires from watchdog fallbacks.
                c.failure = row.counters[2] as u8;
                c.detail.set(row.counters[2] >> 8);
                c.gpio_wakes.set(row.counters[3]);
                c.watchdog_deadlines.set(row.counters[4]);
            }
            n => {
                c.child[n - 1] = I2cChildStatus {
                    address: row.address_or_count,
                    phase: row.state,
                    error: row.error,
                    flags: row.flags,
                    polls: row.counters[0].into(),
                    reports: row.counters[1].into(),
                    errors: row.counters[2].into(),
                    protocol: row.counters[3].into(),
                    last_input: row.counters[4].into(),
                };
            }
        }
        true
    }
    pub fn validate(&self) -> bool {
        self.available.get() <= 1
            && self.reserved == [0; 4]
            && self.admission_available.get() <= 1
            && self.admission.validate()
            && self.pointer.attached.get() <= 1
            && self.pointer.queued.get() <= self.pointer.capacity.get()
            && self.i2c.iter().all(|c| {
                c.rung <= 13
                    && c.children as usize <= MAX_I2C_CHILDREN
                    && c.failure <= 12
                    && c.child.iter().all(|d| {
                        d.address <= 0x7f
                            && d.phase <= 4
                            && d.error <= 9
                            && d.flags & !1 == 0
                            && I2cProtocolObservation::from_bits(d.protocol.get()).is_some()
                            && I2cInputObservation::from_bits(d.last_input.get()).is_some()
                    })
            })
    }
}

// Four controllers carry their two formerly-reserved wire counters as
// GPIO-ready and watchdog observations (+8 bytes apiece). This remains well
// inside the bounded process context asserted just below.
const _: () = assert!(core::mem::size_of::<InputSnapshot>() == 936);
const _: () =
    assert!(core::mem::size_of::<InputSnapshot>() <= crate::process::PROCESS_CONTEXT_BYTES);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_and_input_encodings_are_closed_and_preserve_full_widths() {
        for bits in 0..=u8::MAX {
            let phase = I2cProtocolPhase::from_bits(bits);
            assert_eq!(phase.is_some(), bits <= 14);
            if let Some(phase) = phase {
                let word = if phase == I2cProtocolPhase::None {
                    phase.pack(0, 0)
                } else {
                    phase.pack(u8::MAX, u16::MAX)
                };
                let observed = I2cProtocolObservation::from_bits(word).unwrap();
                assert_eq!(observed.phase, phase);
                assert_eq!(
                    observed.phase.pack(observed.report_id, observed.detail),
                    word
                );
                if phase != I2cProtocolPhase::None {
                    assert_eq!(word, 0xffff_ff00 | u32::from(bits));
                    assert_eq!((observed.report_id, observed.detail), (u8::MAX, u16::MAX));
                }
                assert!(!phase.as_str().is_empty());
            } else {
                assert!(I2cProtocolObservation::from_bits(u32::from(bits)).is_none());
            }
            let outcome = I2cInputOutcome::from_bits(bits);
            assert_eq!(outcome.is_some(), bits <= 6);
            if let Some(outcome) = outcome {
                let word = if outcome == I2cInputOutcome::Unseen {
                    outcome.pack(0, 0)
                } else {
                    outcome.pack(u8::MAX, u16::MAX)
                };
                let observed = I2cInputObservation::from_bits(word).unwrap();
                assert_eq!(observed.outcome, outcome);
                assert_eq!(
                    observed
                        .outcome
                        .pack(observed.report_id, observed.wire_length),
                    word
                );
                if outcome != I2cInputOutcome::Unseen {
                    assert_eq!(word, 0xffff_ff00 | u32::from(bits));
                    assert_eq!(
                        (observed.report_id, observed.wire_length),
                        (u8::MAX, u16::MAX)
                    );
                }
                assert!(!outcome.as_str().is_empty());
            } else {
                assert!(I2cInputObservation::from_bits(u32::from(bits)).is_none());
            }
        }
        // An unseen operation/report cannot nevertheless carry metadata.
        for word in [0x100, 0x1_0000, 0xffff_ff00] {
            assert!(I2cProtocolObservation::from_bits(word).is_none());
            assert!(I2cInputObservation::from_bits(word).is_none());
        }
    }

    #[test]
    fn all_child_observations_cross_indexed_rows_and_bounded_process_context() {
        use zerocopy::FromZeros;

        let protocol = I2cProtocolPhase::FeatureSet.pack(u8::MAX, u16::MAX);
        let input = I2cInputOutcome::Malformed.pack(u8::MAX, u16::MAX);
        let mut snapshot = InputSnapshot::default();
        for row in 0..I2C_STATUS_ROWS {
            let controller = usize::from(row) % (MAX_I2C_CHILDREN + 1) == 0;
            let page = InputDiagnostic::from_i2c(I2cDiagnostic {
                row,
                state: if controller { 12 } else { 3 },
                address_or_count: if controller {
                    MAX_I2C_CHILDREN as u8
                } else {
                    0x7f
                },
                error: if controller { 0 } else { 9 },
                flags: 1,
                counters: if controller {
                    [u32::MAX, u32::MAX, 0, u32::MAX, u32::MAX]
                } else {
                    [u32::MAX, u32::MAX, u32::MAX, protocol, input]
                },
            });
            let decoded =
                InputDiagnostic::from_parts(page.state, page.packed(), page.counters).unwrap();
            assert_eq!(decoded, page);
            assert!(snapshot.insert_diagnostic(decoded));
        }
        snapshot.available.set(1);
        assert!(snapshot.validate());
        let mut process = crate::process::ProcessIo::new_zeroed();
        process
            .initialize_with_context(
                b"input i2c",
                snapshot.as_bytes(),
                crate::monitor::TelemetrySnapshot::EMPTY,
            )
            .unwrap();
        let decoded = InputSnapshot::read_from_bytes(process.context().unwrap()).unwrap();
        assert_eq!(decoded, snapshot);
        assert!(decoded.validate());
        for controller in decoded.i2c {
            assert_eq!(controller.wakes.get(), u32::MAX);
            assert_eq!(controller.gpio_wakes.get(), u32::MAX);
            assert_eq!(controller.watchdog_deadlines.get(), u32::MAX);
            for child in controller.child {
                assert_eq!(child.protocol.get(), protocol);
                assert_eq!(child.last_input.get(), input);
                assert_eq!(child.polls.get(), u32::MAX);
                assert_eq!(child.reports.get(), u32::MAX);
                assert_eq!(child.errors.get(), u32::MAX);
                assert_eq!(child.error, 9);
            }
        }
    }

    #[test]
    fn controller_wake_counters_are_data_while_invalid_child_observations_refuse() {
        let mut snapshot = InputSnapshot::default();
        for slot in [3, 4] {
            let mut controller = I2cDiagnostic::default();
            controller.counters[slot] = u32::MAX;
            assert!(controller.valid());
            assert!(snapshot.insert(controller));
            if slot == 3 {
                assert_eq!(snapshot.i2c[0].gpio_wakes.get(), u32::MAX);
            } else {
                assert_eq!(snapshot.i2c[0].watchdog_deadlines.get(), u32::MAX);
            }
            snapshot = InputSnapshot::default();
            for word in [0xff, 0x100, 0x1_0000, u32::MAX] {
                let mut child = I2cDiagnostic {
                    row: 1,
                    ..Default::default()
                };
                child.counters[slot] = word;
                assert!(
                    I2cDiagnostic::from_parts(child.state, child.packed(), child.counters)
                        .is_none()
                );
                assert!(!snapshot.insert(child));
                assert!(!snapshot.insert_diagnostic(InputDiagnostic::from_i2c(child)));
                assert_eq!(snapshot, InputSnapshot::default());
                let d = &mut snapshot.i2c[MAX_I2C_CONTROLLERS - 1].child[MAX_I2C_CHILDREN - 1];
                if slot == 3 {
                    d.protocol.set(word);
                } else {
                    d.last_input.set(word);
                }
                assert!(!snapshot.validate());
                snapshot = InputSnapshot::default();
            }
        }
    }

    #[test]
    fn appended_diagnostic_rows_are_closed_and_preserve_wide_pointer_counts() {
        let mut snapshot = InputSnapshot::default();
        let row = InputDiagnostic::from_parts(1, u32::from(POINTER_STATUS_BASE), [3, 7, 2, 1, 0])
            .unwrap();
        assert!(snapshot.insert_diagnostic(row));
        let row =
            InputDiagnostic::from_parts(0, u32::from(POINTER_STATUS_BASE + 1), [2, 64, 0, 1, 0])
                .unwrap();
        assert!(snapshot.insert_diagnostic(row));
        assert_eq!(snapshot.pointer.consumed(), (2u64 << 32) + 4);
        assert!(snapshot.validate());
        assert!(InputDiagnostic::from_parts(0, u32::from(INPUT_STATUS_ROWS), [0; 5]).is_none());
        assert!(InputDiagnostic::from_parts(1, u32::from(USB_STATUS_BASE), [0; 5]).is_none());
        assert!(
            InputDiagnostic::from_parts(0, u32::from(USB_STATUS_BASE + 2), [0, 0, 1, 0, 0])
                .is_none()
        );
        assert!(
            InputDiagnostic::from_parts(0, u32::from(POINTER_STATUS_BASE + 1), [65, 64, 0, 0, 0])
                .is_none()
        );
        assert!(
            InputDiagnostic::from_parts(0, u32::from(ADMISSION_STATUS_BASE), [0xff, 0, 0, 0, 0])
                .is_none()
        );
    }

    #[test]
    fn appended_build_failure_crosses_input_pages_and_the_bounded_process_context() {
        use crate::device::{
            AmlBuildDiagnostic, AmlBuildErrorClass, AmlBuildStage, I2C_ADMISSION_BUILD_ROW,
            I2cInventoryState,
        };
        use crate::process::ProcessIo;
        use zerocopy::FromZeros;
        let mut snapshot = InputSnapshot::default();
        let summary = InputDiagnostic::from_parts(
            1,
            u32::from(ADMISSION_STATUS_BASE),
            [I2cInventoryState::BuildRefused as u32, 0, 0, 0, 0],
        )
        .unwrap();
        assert!(snapshot.insert_diagnostic(summary));
        let build = AmlBuildDiagnostic::failure(
            AmlBuildStage::SsdtLoad,
            AmlBuildErrorClass::Reference,
            u32::MAX,
            u16::MAX,
        )
        .unwrap();
        let [a, b] = build.words();
        let row = ADMISSION_STATUS_BASE + I2C_ADMISSION_BUILD_ROW;
        let detail = InputDiagnostic::from_parts(1, u32::from(row), [a, b, 0, 0, 0]).unwrap();
        assert!(snapshot.insert_diagnostic(detail));
        assert_eq!(snapshot.admission.build, build);
        assert!(InputDiagnostic::from_parts(1, u32::from(row), [a, b, 1, 0, 0]).is_none());
        assert!(snapshot.validate());
        assert_eq!(snapshot.as_bytes().len(), 936);
        let mut io = ProcessIo::new_zeroed();
        io.initialize_with_context(
            b"input",
            snapshot.as_bytes(),
            crate::monitor::TelemetrySnapshot::EMPTY,
        )
        .unwrap();
        let decoded = InputSnapshot::read_from_bytes(io.context().unwrap()).unwrap();
        assert!(decoded.validate());
        assert_eq!(decoded.admission.build, build);
    }

    #[test]
    fn invalid_rows_do_not_mutate_a_snapshot() {
        let mut snapshot = InputSnapshot::default();
        for row in [
            I2cDiagnostic {
                row: I2C_STATUS_ROWS,
                ..Default::default()
            },
            I2cDiagnostic {
                address_or_count: 5,
                ..Default::default()
            },
            I2cDiagnostic {
                row: 1,
                address_or_count: 0x80,
                ..Default::default()
            },
            I2cDiagnostic {
                row: 1,
                flags: 2,
                ..Default::default()
            },
            I2cDiagnostic {
                row: 1,
                error: 10,
                ..Default::default()
            },
        ] {
            assert!(!snapshot.insert(row));
            assert_eq!(snapshot, InputSnapshot::default());
        }
    }

    #[test]
    fn inserting_rows_does_not_advertise_a_complete_capture() {
        let mut snapshot = InputSnapshot::default();
        assert!(snapshot.insert(I2cDiagnostic {
            row: I2C_STATUS_ROWS - 1,
            state: 2,
            address_or_count: 0x2c,
            counters: [9, 8, 7, 0, 0],
            ..Default::default()
        }));
        assert_eq!(snapshot.available.get(), 0);
        assert_eq!(snapshot.i2c[3].child[3].reports.get(), 8);
        assert_eq!(snapshot.i2c[0], I2cControllerStatus::default());
        assert!(snapshot.validate());
        snapshot.reserved[0] = 1;
        assert!(!snapshot.validate());
        snapshot.reserved[0] = 0;
        snapshot.i2c[3].failure = 13;
        assert!(!snapshot.validate());
        snapshot.i2c[3].failure = 12;
        assert!(snapshot.validate());
    }

    /// A controller row's refusal must survive the wire, because a claim that
    /// refuses off-machine is diagnosable only through this byte pair. The
    /// packing rides counter 2, while counters 3 and 4 carry the GPIO-ready
    /// and watchdog split without changing the row size.
    #[test]
    fn a_controller_row_carries_its_failure_class_and_detail_byte() {
        let mut snapshot = InputSnapshot::default();
        // rung 6 (MMIO/route refused), failure 12 (intake), detail 8 — the
        // interrupt route could not be established.
        assert!(snapshot.insert(I2cDiagnostic {
            row: 0,
            state: 6,
            address_or_count: 1,
            flags: 1,
            counters: [0, 0, 12 | (0x04_0006 << 8), 7, 11],
            ..Default::default()
        }));
        assert_eq!(snapshot.i2c[0].rung, 6);
        assert_eq!(snapshot.i2c[0].failure, 12);
        // class 06 (BAR map), bar 00, stage-coded 04 (map build).
        assert_eq!(snapshot.i2c[0].detail.get(), 0x04_0006);
        assert_eq!(snapshot.i2c[0].gpio_wakes.get(), 7);
        assert_eq!(snapshot.i2c[0].watchdog_deadlines.get(), 11);
        assert!(snapshot.validate());
    }

    /// The rung alone cannot distinguish the two refusals that produce it, so
    /// a row that carries no failure must not invent one.
    #[test]
    fn a_running_controller_row_carries_no_failure() {
        let mut snapshot = InputSnapshot::default();
        assert!(snapshot.insert(I2cDiagnostic {
            row: 0,
            state: 0,
            address_or_count: 1,
            flags: 1,
            counters: [4, 0, 0, 0, 0],
            ..Default::default()
        }));
        assert_eq!(snapshot.i2c[0].failure, 0);
        assert_eq!(snapshot.i2c[0].detail.get(), 0);
        assert!(snapshot.validate());
    }
}
