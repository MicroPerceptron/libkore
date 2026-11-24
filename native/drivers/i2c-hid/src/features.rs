//! Bounded configuration of descriptor-declared digitizer features.
//!
//! The input parser supplies a validated plan. A report made entirely of the
//! selected controls and constant padding is synthesized and sent directly;
//! a mixed report uses GET_REPORT followed by a read-modify-write SET_REPORT
//! so unrelated fields retain their fetched values. Nothing here chooses a
//! controller, allocates, or treats a vendor ID as configuration.
//!
//! Protocol references (not copied implementations):
//! - [PTP configuration and reset semantics](https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/touchpad-configuration-collection)
//! - [Digitizer mode and Device Identifier](https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/using-report-descriptors-to-support-capability-discovery)
//! - [HID-over-I2C Get/Set Report](https://learn.microsoft.com/en-us/windows-hardware/drivers/hid/plug-and-play-support-and-power-management)
//! - [HID-over-I2C 1.0, HID Descriptor Format](https://download.microsoft.com/download/7/d/d/7dd44bb7-2a7a-4505-ac1c-7227d3d96d5b/hid-over-i2c-protocol-spec-v1-0.docx)
//!
//! A host-initiated reset invalidates configuration: create a fresh setup after
//! every successful reset and descriptor parse, not only at first discovery.

use crate::{Command, HidDescriptor, ReportType, Step};

/// Configuration capacity, independent of the number of input report IDs.
pub const MAX_FEATURE_REPORTS: usize = 4;
/// Includes a numbered report's ID and all padding, but not the I2C prefix.
/// Large unrelated vendor/certification reports never enter this budget.
pub const MAX_FEATURE_REPORT_BYTES: usize = 64;
/// Mode, optional Device Identifier, surface switch, and button switch.
pub const MAX_FEATURE_EDITS: usize = 4;
/// A GET response includes the two-byte I2C length prefix.
pub const MAX_FEATURE_RESPONSE_BYTES: usize = MAX_FEATURE_REPORT_BYTES + 2;
const MAX_FEATURE_WRITE_BYTES: usize = crate::command::MAX_COMMAND_LEN + MAX_FEATURE_RESPONSE_BYTES;

/// A descriptor's requested configuration was not safe to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PlanRefusal {
    TooManyReports = 1,
    TooManyFields = 2,
    ReportSize = 3,
    FieldBounds = 4,
    LogicalRange = 5,
    FieldAttributes = 6,
    AmbiguousMode = 7,
    DuplicateControl = 8,
    OverlappingFields = 9,
    InvalidCollection = 10,
    InvalidReportId = 11,
}

/// Configuration failures retain their cause without logging report contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureError {
    Plan {
        report_id: u8,
        reason: PlanRefusal,
    },
    /// HID-over-I2C's HID Descriptor Format forbids zero command/data
    /// registers. This does not restrict the separate ACPI-published address
    /// of the HID descriptor itself, which may be zero.
    InvalidRegisters,
    OutOfOrder,
    ResponseLength {
        expected: u16,
        received: usize,
    },
    LengthPrefix {
        expected: u16,
        received: u16,
    },
    ReportId {
        expected: u8,
        received: u8,
    },
}

impl FeatureError {
    /// Bounded service detail: class in the low byte, refusal reason or the
    /// received operand in the next seven bits. Operands saturate at 127,
    /// keeping bit 15 clear for the service's relay-error namespace. The
    /// requested report ID is separate [`FeatureSetup::current_report_id`]
    /// metadata; [`Self::diagnostic`] retains the full received report ID.
    pub const fn diagnostic16(self) -> u16 {
        let (class, operand) = match self {
            Self::Plan { reason, .. } => (1, reason as usize),
            Self::InvalidRegisters => (2, 0),
            Self::OutOfOrder => (3, 0),
            Self::ResponseLength { received, .. } => (4, received),
            Self::LengthPrefix { received, .. } => (5, received as usize),
            Self::ReportId { received, .. } => (6, received as usize),
        };
        let operand = if operand > 127 { 127 } else { operand as u16 };
        class | (operand << 8)
    }

    /// Compact, stable diagnostic: class in the low byte, operands above it.
    pub const fn diagnostic(self) -> u32 {
        match self {
            Self::Plan { report_id, reason } => {
                1 | ((reason as u32) << 8) | ((report_id as u32) << 16)
            }
            Self::InvalidRegisters => 2,
            Self::OutOfOrder => 3,
            Self::ResponseLength { expected, received } => {
                let received = if received > u16::MAX as usize {
                    u16::MAX as u32
                } else {
                    received as u32
                };
                4 | ((expected as u32) << 8) | (received << 16)
            }
            Self::LengthPrefix { expected, received } => {
                5 | ((expected as u32) << 8) | ((received as u32) << 16)
            }
            Self::ReportId { expected, received } => {
                6 | ((expected as u32) << 8) | ((received as u32) << 16)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Edit {
    offset: u16,
    width: u8,
    value: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReportPlan {
    /// None is an unnumbered report. Some(0) is rejected, never normalized.
    id: Option<u8>,
    bytes: u16,
    edits: [Edit; MAX_FEATURE_EDITS],
    len: usize,
    /// Mixed feature reports must be fetched so state outside the selected
    /// configuration fields survives the SET_REPORT. Fully described reports
    /// can be synthesized; some digitizers reject GET_REPORT for their
    /// write-only mode selector.
    preserve_unedited: bool,
}

impl ReportPlan {
    const EMPTY: Self = Self {
        id: None,
        bytes: 0,
        edits: [Edit {
            offset: 0,
            width: 0,
            value: 0,
        }; MAX_FEATURE_EDITS],
        len: 0,
        preserve_unedited: true,
    };

    pub(crate) fn new(id: Option<u8>, bytes: usize) -> Result<Self, FeatureError> {
        let report = Self { id, ..Self::EMPTY };
        if id == Some(0) {
            return Err(report.refuse(PlanRefusal::InvalidReportId));
        }
        if bytes <= usize::from(id.is_some()) || bytes > MAX_FEATURE_REPORT_BYTES {
            return Err(report.refuse(PlanRefusal::ReportSize));
        }
        Ok(Self {
            bytes: bytes as u16,
            ..report
        })
    }

    pub(crate) fn refuse(&self, reason: PlanRefusal) -> FeatureError {
        FeatureError::Plan {
            report_id: self.id.unwrap_or(0),
            reason,
        }
    }

    pub(crate) fn add(
        &mut self,
        offset: usize,
        width: usize,
        value: u32,
        minimum: i32,
        maximum: i32,
    ) -> Result<(), FeatureError> {
        if width == 0 || width > 32 || offset < usize::from(self.id.is_some()) * 8 {
            return Err(self.refuse(PlanRefusal::FieldBounds));
        }
        let end = offset
            .checked_add(width)
            .ok_or_else(|| self.refuse(PlanRefusal::FieldBounds))?;
        if end > usize::from(self.bytes) * 8 {
            return Err(self.refuse(PlanRefusal::FieldBounds));
        }
        let (wire_min, wire_max) = if minimum < 0 {
            (-(1i64 << (width - 1)), (1i64 << (width - 1)) - 1)
        } else {
            (0, (1i64 << width) - 1)
        };
        if minimum > maximum
            || i64::from(minimum) < wire_min
            || i64::from(maximum) > wire_max
            || i64::from(value) < i64::from(minimum)
            || i64::from(value) > i64::from(maximum)
        {
            return Err(self.refuse(PlanRefusal::LogicalRange));
        }
        for edit in &self.edits[..self.len] {
            let start = usize::from(edit.offset);
            if offset < start + usize::from(edit.width) && start < end {
                return Err(self.refuse(PlanRefusal::OverlappingFields));
            }
        }
        if self.len == self.edits.len() {
            return Err(self.refuse(PlanRefusal::TooManyFields));
        }
        self.edits[self.len] = Edit {
            offset: offset as u16,
            width: width as u8,
            value,
        };
        self.len += 1;
        Ok(())
    }

    pub(crate) fn covers(&self, offset: usize, width: usize) -> bool {
        let end = offset.saturating_add(width);
        self.edits[..self.len].iter().any(|edit| {
            let start = usize::from(edit.offset);
            start <= offset && end <= start + usize::from(edit.width)
        })
    }

    pub(crate) fn set_preserve_unedited(&mut self, preserve: bool) {
        self.preserve_unedited = preserve;
    }
}

/// Opaque, descriptor-validated configuration; no caller-supplied raw writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeaturePlan {
    reports: [ReportPlan; MAX_FEATURE_REPORTS],
    len: usize,
}

impl Default for FeaturePlan {
    fn default() -> Self {
        Self::empty()
    }
}

impl FeaturePlan {
    pub const fn empty() -> Self {
        Self {
            reports: [ReportPlan::EMPTY; MAX_FEATURE_REPORTS],
            len: 0,
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn push(&mut self, report: ReportPlan) -> Result<(), FeatureError> {
        if report.len == 0 {
            return Ok(());
        }
        if self.len == self.reports.len() {
            return Err(report.refuse(PlanRefusal::TooManyReports));
        }
        if self.reports[..self.len].iter().any(|r| r.id == report.id) {
            return Err(report.refuse(PlanRefusal::DuplicateControl));
        }
        self.reports[self.len] = report;
        self.len += 1;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeaturePhase {
    Get,
    Set,
    Done,
}

/// Sans-I/O setup. Only `read`/`wrote` acknowledge successful transactions.
/// If transport I/O fails, leave the operation unacknowledged: `step` repeats
/// it unchanged. A full device reset requires a new setup from the plan.
pub struct FeatureSetup {
    plan: FeaturePlan,
    index: usize,
    phase: FeaturePhase,
    command_register: u16,
    data_register: u16,
    out: [u8; MAX_FEATURE_WRITE_BYTES],
    out_len: usize,
}

impl FeatureSetup {
    pub fn new(plan: FeaturePlan, descriptor: &HidDescriptor) -> Result<Self, FeatureError> {
        // These two registers are explicitly nonzero in the protocol, unlike
        // the implementation-defined HID descriptor register from ACPI.
        if !plan.is_empty()
            && (descriptor.w_command_register.get() == 0 || descriptor.w_data_register.get() == 0)
        {
            return Err(FeatureError::InvalidRegisters);
        }
        let phase = if plan.is_empty() {
            FeaturePhase::Done
        } else if plan.reports[0].preserve_unedited {
            FeaturePhase::Get
        } else {
            FeaturePhase::Set
        };
        let mut setup = Self {
            plan,
            index: 0,
            phase,
            command_register: descriptor.w_command_register.get(),
            data_register: descriptor.w_data_register.get(),
            out: [0; MAX_FEATURE_WRITE_BYTES],
            out_len: 0,
        };
        if setup.phase == FeaturePhase::Set {
            setup.prepare_synthetic();
        }
        Ok(setup)
    }

    pub const fn phase(&self) -> FeaturePhase {
        self.phase
    }

    /// Some(0) here identifies an *unnumbered* wire report; None means Done.
    /// The descriptor parser rejects an explicitly declared Report ID zero.
    pub fn current_report_id(&self) -> Option<u8> {
        (self.index < self.plan.len).then(|| self.plan.reports[self.index].id.unwrap_or(0))
    }

    pub fn step(&mut self) -> Step {
        match self.phase {
            FeaturePhase::Get => {
                let report = &self.plan.reports[self.index];
                let command = Command::get_report(
                    self.command_register,
                    self.data_register,
                    ReportType::Feature,
                    report.id.unwrap_or(0),
                );
                self.out_len = command.bytes().len();
                self.out[..self.out_len].copy_from_slice(command.bytes());
                Step::WriteRead {
                    write_len: self.out_len,
                    read_len: usize::from(report.bytes) + 2,
                }
            }
            FeaturePhase::Set => Step::Write { len: self.out_len },
            FeaturePhase::Done => Step::Done,
        }
    }

    /// Bytes for the last demanded step, unchanged by a repeated `step`.
    pub fn out(&self) -> &[u8] {
        &self.out[..self.out_len]
    }

    pub fn read(&mut self, bytes: &[u8]) -> Result<(), FeatureError> {
        if self.phase != FeaturePhase::Get {
            return Err(FeatureError::OutOfOrder);
        }
        let report = self.plan.reports[self.index];
        let expected = report.bytes + 2;
        if bytes.len() != usize::from(expected) {
            return Err(FeatureError::ResponseLength {
                expected,
                received: bytes.len(),
            });
        }
        let received = u16::from_le_bytes([bytes[0], bytes[1]]);
        if received != expected {
            return Err(FeatureError::LengthPrefix { expected, received });
        }
        if let Some(id) = report.id
            && bytes[2] != id
        {
            return Err(FeatureError::ReportId {
                expected: id,
                received: bytes[2],
            });
        }
        let command = Command::set_report(
            self.command_register,
            self.data_register,
            ReportType::Feature,
            report.id.unwrap_or(0),
        );
        let header = command.bytes().len();
        self.out[..header].copy_from_slice(command.bytes());
        self.out[header..header + bytes.len()].copy_from_slice(bytes);
        apply_edits(&report, &mut self.out[header + 2..header + bytes.len()]);
        self.out_len = header + bytes.len();
        self.phase = FeaturePhase::Set;
        Ok(())
    }

    pub fn wrote(&mut self) -> Result<(), FeatureError> {
        if self.phase != FeaturePhase::Set {
            return Err(FeatureError::OutOfOrder);
        }
        self.index += 1;
        self.phase = if self.index == self.plan.len {
            FeaturePhase::Done
        } else if self.plan.reports[self.index].preserve_unedited {
            FeaturePhase::Get
        } else {
            FeaturePhase::Set
        };
        self.out_len = 0;
        if self.phase == FeaturePhase::Set {
            self.prepare_synthetic();
        }
        Ok(())
    }

    fn prepare_synthetic(&mut self) {
        let report = self.plan.reports[self.index];
        debug_assert!(!report.preserve_unedited);
        let command = Command::set_report(
            self.command_register,
            self.data_register,
            ReportType::Feature,
            report.id.unwrap_or(0),
        );
        let header = command.bytes().len();
        self.out[..header].copy_from_slice(command.bytes());
        let response_len = usize::from(report.bytes) + 2;
        self.out[header..header + response_len].fill(0);
        self.out[header..header + 2].copy_from_slice(&(report.bytes + 2).to_le_bytes());
        if let Some(id) = report.id {
            self.out[header + 2] = id;
        }
        apply_edits(&report, &mut self.out[header + 2..header + response_len]);
        self.out_len = header + response_len;
    }
}

fn apply_edits(report: &ReportPlan, payload: &mut [u8]) {
    for edit in &report.edits[..report.len] {
        for bit in 0..usize::from(edit.width) {
            let position = usize::from(edit.offset) + bit;
            let mask = 1u8 << (position % 8);
            let value = (((edit.value >> bit) & 1) as u8) << (position % 8);
            payload[position / 8] = (payload[position / 8] & !mask) | value;
        }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::parse::{ParseError, build_device, build_table};

    struct Descriptor {
        bytes: [u8; 1024],
        len: usize,
    }

    impl Descriptor {
        fn new(class: u8, numbered: bool) -> Self {
            let mut out = Self {
                bytes: [0; 1024],
                len: 0,
            };
            out.contact(class, numbered);
            out
        }
        fn append(&mut self, bytes: &[u8]) {
            self.bytes[self.len..self.len + bytes.len()].copy_from_slice(bytes);
            self.len += bytes.len();
        }
        fn contact(&mut self, class: u8, numbered: bool) {
            self.append(&[0x05, 0x0d, 0x09, class, 0xa1, 0x01]);
            if numbered {
                self.append(&[0x85, 1]);
            }
            self.append(&[
                0x09, 0x42, 0x15, 0, 0x25, 1, 0x75, 8, 0x95, 1, 0x81, 2, 0x05, 1, 0x09, 0x30, 0x09,
                0x31, 0x26, 0xff, 0x0f, 0x75, 16, 0x95, 2, 0x81, 2, 0xc0,
            ]);
        }
        fn config(&mut self, id: Option<u8>) {
            self.append(&[0x05, 0x0d, 0x09, 0x0e, 0xa1, 0x01]);
            if let Some(id) = id {
                self.append(&[0x85, id]);
            }
        }
        fn mode(&mut self, id: Option<u8>, maximum: u8, flags: u8) {
            self.config(id);
            self.append(&[
                0x09, 0x52, 0x15, 0, 0x25, maximum, 0x75, 8, 0x95, 1, 0xb1, flags, 0xc0,
            ]);
        }
        fn bytes(&self) -> &[u8] {
            &self.bytes[..self.len]
        }
        fn setup(&self) -> FeatureSetup {
            FeatureSetup::new(
                build_device(self.bytes()).unwrap().features,
                &hid_descriptor(),
            )
            .unwrap()
        }
    }

    fn hid_descriptor() -> HidDescriptor {
        let mut bytes = [0; 30];
        for (offset, value) in [
            (0, 30u16),
            (2, 0x0100),
            (4, 100),
            (6, 2),
            (8, 3),
            (10, 64),
            (16, 5),
            (18, 6),
        ] {
            bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }
        HidDescriptor::parse(&bytes).unwrap()
    }

    #[test]
    fn cf00_complete_configuration_reports_are_set_without_a_rejected_get() {
        // Minimal standard collections with the captured CF00 feature layout,
        // not a redistributed firmware descriptor or vendor-ID special case.
        let mut d = Descriptor::new(5, true);
        d.config(Some(4));
        d.append(&[
            0x09, 0x22, 0xa1, 2, 0x09, 0x52, 0x15, 0, 0x25, 10, 0x75, 8, 0x95, 1, 0xb1, 2, 0xc0,
            0xc0,
        ]);
        d.config(Some(6));
        d.append(&[
            0x09, 0x22, 0xa1, 0, 0x09, 0x57, 0x09, 0x58, 0x25, 1, 0x75, 1, 0x95, 2, 0xb1, 2, 0x75,
            6, 0x95, 1, 0xb1, 3, 0xc0, 0xc0,
        ]);
        // Contact Maximum / Button Type and a large certification feature are
        // deliberately not writable configuration, despite sharing the device.
        d.append(&[
            0x05, 0x0d, 0x09, 5, 0xa1, 1, 0x85, 8, 0x09, 0x55, 0x09, 0x59, 0x25, 15, 0x75, 4, 0x95,
            2, 0xb1, 2, 0x85, 7, 0x06, 0, 0xff, 0x09, 0xc5, 0x26, 0xff, 0, 0x75, 8, 0x96, 0, 1,
            0xb1, 2, 0xc0,
        ]);
        let parsed = build_device(d.bytes()).unwrap();
        assert_eq!(parsed.table, build_table(d.bytes()).unwrap());
        assert_eq!(parsed.features.len(), 2);
        let mut setup = FeatureSetup::new(parsed.features, &hid_descriptor()).unwrap();
        assert_eq!(setup.phase(), FeaturePhase::Set);
        assert_eq!(setup.current_report_id(), Some(4));
        assert_eq!(setup.step(), Step::Write { len: 10 });
        assert_eq!(setup.out(), &[5, 0, 0x34, 3, 6, 0, 4, 0, 4, 3]);
        setup.wrote().unwrap();
        assert_eq!(setup.current_report_id(), Some(6));
        assert_eq!(setup.step(), Step::Write { len: 10 });
        assert_eq!(setup.out(), &[5, 0, 0x36, 3, 6, 0, 4, 0, 6, 3]);
        setup.wrote().unwrap();
        assert_eq!(setup.phase(), FeaturePhase::Done);
        assert_eq!(setup.step(), Step::Done);
        assert_eq!(setup.current_report_id(), None);
    }

    #[test]
    fn wacom_53fe_configuration_sets_multi_input_and_the_paired_all_collections_selector() {
        let mut d = Descriptor::new(4, true);
        d.config(Some(14));
        d.append(&[
            0x09, 0x23, 0xa1, 2, 0x09, 0x52, 0x09, 0x53, 0x15, 0, 0x25, 10, 0x75, 8, 0x95, 2, 0xb1,
            2, 0xc0, 0xc0,
        ]);
        let mut setup = d.setup();
        assert_eq!(setup.step(), Step::Write { len: 11 });
        assert_eq!(setup.out(), &[5, 0, 0x3e, 3, 6, 0, 5, 0, 14, 2, 0]);
        setup.wrote().unwrap();
        assert_eq!(setup.step(), Step::Done);
    }

    #[test]
    fn report_ids_fourteen_fifteen_and_255_roundtrip_without_nibble_aliasing() {
        for id in [14, 15, 255] {
            let mut d = Descriptor::new(5, true);
            d.mode(Some(id), 10, 2);
            let mut setup = d.setup();
            let header = if id < 15 { 6 } else { 7 };
            assert_eq!(setup.step(), Step::Write { len: header + 4 });
            assert_eq!(&setup.out()[header..], &[4, 0, id, 3]);
            if id >= 15 {
                assert_eq!(&setup.out()[..header], &[5, 0, 0x3f, 3, id, 6, 0]);
            }
            setup.wrote().unwrap();
            assert_eq!(setup.step(), Step::Done);
        }
    }

    #[test]
    fn packed_feature_fields_preserve_vendor_bytes_and_padding_in_one_transaction() {
        let mut d = Descriptor::new(5, true);
        d.config(Some(14));
        d.append(&[
            0x06, 0, 0xff, 0x09, 1, 0x15, 0, 0x26, 0xff, 0, 0x75, 8, 0x95, 1, 0xb1, 2, 0x05, 0x0d,
            0x09, 0x52, 0x25, 7, 0x75, 3, 0xb1, 2, 0x75, 1, 0xb1, 3, 0x09, 0x57, 0x09, 0x58, 0x25,
            1, 0x95, 2, 0xb1, 2, 0x75, 2, 0x95, 1, 0xb1, 3, 0x06, 0, 0xff, 0x09, 2, 0x26, 0xff, 0,
            0x75, 8, 0xb1, 2, 0xc0,
        ]);
        let parsed = build_device(d.bytes()).unwrap();
        assert_eq!(parsed.features.len(), 1);
        let mut setup = FeatureSetup::new(parsed.features, &hid_descriptor()).unwrap();
        assert_eq!(
            setup.step(),
            Step::WriteRead {
                write_len: 6,
                read_len: 6
            }
        );
        setup.read(&[6, 0, 14, 0xa5, 0xc8, 0x5a]).unwrap();
        assert_eq!(
            setup.out(),
            &[5, 0, 0x3e, 3, 6, 0, 6, 0, 14, 0xa5, 0xfb, 0x5a]
        );
    }

    #[test]
    fn a_cross_byte_edit_preserves_every_unrelated_bit() {
        let mut report = ReportPlan::new(Some(4), 3).unwrap();
        report.add(15, 3, 3, 0, 7).unwrap();
        let mut plan = FeaturePlan::empty();
        plan.push(report).unwrap();
        let mut setup = FeatureSetup::new(plan, &hid_descriptor()).unwrap();
        setup.read(&[5, 0, 4, 0xff, 0xfc]).unwrap();
        assert_eq!(&setup.out()[6..], &[5, 0, 4, 0xff, 0xfd]);
    }

    #[test]
    fn unnumbered_reports_do_not_gain_an_id_byte() {
        let mut d = Descriptor::new(5, false);
        d.mode(None, 10, 2);
        let mut setup = d.setup();
        assert_eq!(setup.current_report_id(), Some(0));
        assert_eq!(setup.step(), Step::Write { len: 9 });
        assert_eq!(setup.out(), &[5, 0, 0x30, 3, 6, 0, 3, 0, 3]);
        assert!(matches!(
            ReportPlan::new(Some(0), 2),
            Err(FeatureError::Plan {
                reason: PlanRefusal::InvalidReportId,
                ..
            })
        ));
    }

    #[test]
    fn report_id_numbering_is_global_across_input_output_and_features() {
        // The Push/Pop restores an unnumbered global state after a numbered
        // Input. The underlying parser accepts this layout, but no Feature
        // write may assume it is an unnumbered device.
        let mut input = Descriptor {
            bytes: [0; 1024],
            len: 0,
        };
        input.append(&[0xa4]);
        input.contact(5, true);
        input.append(&[0xb4]);
        input.mode(None, 10, 2);
        assert!(build_table(input.bytes()).is_ok());
        assert!(matches!(
            build_device(input.bytes()),
            Err(ParseError::Feature(FeatureError::Plan {
                reason: PlanRefusal::InvalidReportId,
                ..
            }))
        ));

        // Output-only numbering is equally descriptor-wide. Its unrelated
        // vendor contents are not writable configuration, but its ID still
        // forbids silently treating the other report types as unnumbered.
        let mut output = Descriptor::new(5, false);
        output.mode(None, 10, 2);
        output.append(&[
            0x06, 0, 0xff, 0x09, 1, 0xa1, 1, 0x85, 9, 0x09, 1, 0x15, 0, 0x25, 1, 0x75, 8, 0x95, 1,
            0x91, 2, 0xc0,
        ]);
        assert!(matches!(
            build_device(output.bytes()),
            Err(ParseError::Feature(FeatureError::Plan {
                reason: PlanRefusal::InvalidReportId,
                ..
            }))
        ));

        let mut feature = Descriptor::new(5, false);
        feature.mode(Some(4), 10, 2);
        assert!(matches!(
            build_device(feature.bytes()),
            Err(ParseError::Feature(FeatureError::Plan {
                reason: PlanRefusal::InvalidReportId,
                ..
            }))
        ));
    }

    #[test]
    fn compact_diagnostics_preserve_refusal_class_and_received_operand() {
        assert_eq!(
            FeatureError::Plan {
                report_id: 255,
                reason: PlanRefusal::InvalidReportId,
            }
            .diagnostic16(),
            0x0b01
        );
        assert_eq!(FeatureError::InvalidRegisters.diagnostic16(), 2);
        assert_eq!(FeatureError::OutOfOrder.diagnostic16(), 3);
        assert_eq!(
            FeatureError::ResponseLength {
                expected: 66,
                received: usize::MAX,
            }
            .diagnostic16(),
            0x7f04
        );
        assert_eq!(
            FeatureError::LengthPrefix {
                expected: 66,
                received: u16::MAX,
            }
            .diagnostic16(),
            0x7f05
        );
        assert_eq!(
            FeatureError::ReportId {
                expected: 14,
                received: 255,
            }
            .diagnostic16(),
            0x7f06
        );
    }

    #[test]
    fn response_errors_do_not_advance_or_publish_a_set_and_transport_steps_are_repeatable() {
        let mut d = Descriptor::new(5, true);
        d.config(Some(4));
        d.append(&[
            0x09, 0x52, 0x15, 0, 0x25, 10, 0x75, 8, 0x95, 1, 0xb1, 2, 0x06, 0, 0xff, 0x09, 1, 0x26,
            0xff, 0, 0x75, 8, 0x95, 1, 0xb1, 2, 0xc0,
        ]);
        let mut setup = d.setup();
        assert_eq!(setup.wrote(), Err(FeatureError::OutOfOrder));
        let get = setup.step();
        for bytes in [&[][..], &[5, 0, 4][..], &[6, 0, 4, 0, 0, 0][..]] {
            assert!(matches!(
                setup.read(bytes),
                Err(FeatureError::ResponseLength { .. })
            ));
            assert_eq!(setup.phase(), FeaturePhase::Get);
            assert_eq!(setup.step(), get);
        }
        for prefix in [0u16, 1, 2, 3, 4, 6, u16::MAX] {
            let p = prefix.to_le_bytes();
            assert_eq!(
                setup.read(&[p[0], p[1], 4, 0, 0xa5]),
                Err(FeatureError::LengthPrefix {
                    expected: 5,
                    received: prefix
                })
            );
        }
        assert_eq!(
            setup.read(&[5, 0, 5, 0, 0xa5]),
            Err(FeatureError::ReportId {
                expected: 4,
                received: 5
            })
        );
        assert_eq!(setup.out(), &[5, 0, 0x34, 2, 6, 0]);
        setup.read(&[5, 0, 4, 0, 0xa5]).unwrap();
        let set = setup.step();
        assert_eq!(
            setup.read(&[5, 0, 4, 0, 0xa5]),
            Err(FeatureError::OutOfOrder)
        );
        assert_eq!(setup.step(), set); // no acknowledgement after a failed bus write
        assert_eq!(setup.out(), &[5, 0, 0x34, 3, 6, 0, 5, 0, 4, 3, 0xa5]);
        setup.wrote().unwrap();
        assert_eq!(setup.wrote(), Err(FeatureError::OutOfOrder));
        assert_eq!(setup.step(), Step::Done);
    }

    #[test]
    fn invalid_feature_plans_refuse_without_changing_input_only_compatibility() {
        for (maximum, flags, reason) in [
            (2, 2, PlanRefusal::LogicalRange),
            (10, 6, PlanRefusal::FieldAttributes),
        ] {
            let mut d = Descriptor::new(5, true);
            d.mode(Some(4), maximum, flags);
            assert!(build_table(d.bytes()).is_ok());
            assert!(
                matches!(build_device(d.bytes()), Err(ParseError::Feature(FeatureError::Plan { report_id: 4, reason: actual })) if actual == reason)
            );
        }
        let mut duplicate = Descriptor::new(5, true);
        duplicate.mode(Some(4), 10, 2);
        duplicate.mode(Some(5), 10, 2);
        assert!(matches!(
            build_device(duplicate.bytes()),
            Err(ParseError::Feature(FeatureError::Plan {
                reason: PlanRefusal::DuplicateControl,
                ..
            }))
        ));

        let mut huge = Descriptor::new(5, true);
        huge.config(Some(4));
        huge.append(&[
            0x09, 0x52, 0x15, 0, 0x25, 10, 0x75, 8, 0x95, 1, 0xb1, 2, 0x95, 64, 0xb1, 3, 0xc0,
        ]);
        assert!(matches!(
            build_device(huge.bytes()),
            Err(ParseError::Feature(FeatureError::Plan {
                reason: PlanRefusal::ReportSize,
                ..
            }))
        ));

        let mut ambiguous = Descriptor::new(5, true);
        // A second input report has a distinct touchscreen application.
        ambiguous.append(&[0x85, 2]);
        ambiguous.contact(4, false);
        ambiguous.mode(Some(4), 10, 2);
        assert!(matches!(
            build_device(ambiguous.bytes()),
            Err(ParseError::Feature(FeatureError::Plan {
                reason: PlanRefusal::AmbiguousMode,
                ..
            }))
        ));
    }

    #[test]
    fn plan_bounds_reject_id_overwrites_overlaps_and_unrepresentable_values() {
        let mut report = ReportPlan::new(Some(4), 3).unwrap();
        assert!(report.add(0, 8, 3, 0, 10).is_err());
        assert!(report.add(23, 2, 1, 0, 3).is_err());
        assert!(report.add(8, 0, 1, 0, 1).is_err());
        assert!(report.add(8, 33, 1, 0, 1).is_err());
        assert!(report.add(8, 1, 3, 0, 10).is_err());
        assert!(report.add(8, 2, 3, -2, 3).is_err());
        report.add(8, 8, 3, 0, 10).unwrap();
        assert!(matches!(
            report.add(15, 2, 1, 0, 3),
            Err(FeatureError::Plan {
                reason: PlanRefusal::OverlappingFields,
                ..
            })
        ));
    }

    #[test]
    fn only_variable_controls_inside_the_configuration_application_are_writable() {
        let mut vendor = Descriptor::new(5, true);
        vendor.append(&[
            0x06, 0, 0xff, 0x09, 0x0e, 0xa1, 1, 0x85, 4, 0x05, 0x0d, 0x09, 0x52, 0x15, 0, 0x25, 10,
            0x75, 8, 0x95, 1, 0xb1, 2, 0xc0,
        ]);
        assert!(matches!(
            build_device(vendor.bytes()),
            Err(ParseError::Feature(FeatureError::Plan {
                reason: PlanRefusal::InvalidCollection,
                ..
            }))
        ));
        let mut array = Descriptor::new(5, true);
        array.mode(Some(4), 10, 0);
        assert!(matches!(
            build_device(array.bytes()),
            Err(ParseError::Feature(FeatureError::Plan {
                reason: PlanRefusal::FieldAttributes,
                ..
            }))
        ));
        let mut explicit_zero = Descriptor::new(5, true);
        explicit_zero.mode(Some(0), 10, 2);
        assert!(build_device(explicit_zero.bytes()).is_err());
    }

    #[test]
    fn device_identifier_outside_the_modes_logical_group_is_preserved() {
        let mut d = Descriptor::new(4, true);
        d.config(Some(14));
        d.append(&[
            0x09, 0x23, 0xa1, 2, 0x09, 0x52, 0x15, 0, 0x25, 10, 0x75, 8, 0x95, 1, 0xb1, 2, 0xc0,
            0x09, 0x53, 0xb1, 2, 0xc0,
        ]);
        let mut setup = d.setup();
        setup.read(&[5, 0, 14, 0, 7]).unwrap();
        assert_eq!(&setup.out()[6..], &[5, 0, 14, 2, 7]);
    }

    #[test]
    fn maximum_configuration_report_uses_its_own_bound_not_max_input_length() {
        let mut report = ReportPlan::new(Some(255), MAX_FEATURE_REPORT_BYTES).unwrap();
        report.add(8, 8, 3, 0, 10).unwrap();
        let mut plan = FeaturePlan::empty();
        plan.push(report).unwrap();
        let mut hid = hid_descriptor();
        hid.w_max_input_length.set(2);
        let mut setup = FeatureSetup::new(plan, &hid).unwrap();
        assert_eq!(
            setup.step(),
            Step::WriteRead {
                write_len: 7,
                read_len: 66
            }
        );
        let mut response = [0xa5; MAX_FEATURE_RESPONSE_BYTES];
        response[..3].copy_from_slice(&[66, 0, 255]);
        setup.read(&response).unwrap();
        assert_eq!(setup.step(), Step::Write { len: 73 });
        assert_eq!(&setup.out()[7..11], &[66, 0, 255, 3]);
        assert!(setup.out()[11..].iter().all(|byte| *byte == 0xa5));
        assert!(core::mem::size_of::<FeatureSetup>() < 1024);

        let mut plan = FeaturePlan::empty();
        for id in 1..=MAX_FEATURE_REPORTS as u8 {
            let mut report = ReportPlan::new(Some(id), 2).unwrap();
            report.add(8, 8, 1, 0, 1).unwrap();
            plan.push(report).unwrap();
        }
        let mut extra = ReportPlan::new(Some(5), 2).unwrap();
        extra.add(8, 8, 1, 0, 1).unwrap();
        assert!(matches!(
            plan.push(extra),
            Err(FeatureError::Plan {
                reason: PlanRefusal::TooManyReports,
                ..
            })
        ));
    }

    #[test]
    fn a_keyboard_without_configuration_finishes_without_any_io_or_register_requirement() {
        let descriptor = [
            0x05, 1, 0x09, 6, 0xa1, 1, 0x05, 7, 0x09, 4, 0x15, 0, 0x25, 1, 0x75, 1, 0x95, 1, 0x81,
            2, 0xc0,
        ];
        let parsed = build_device(&descriptor).unwrap();
        assert!(parsed.features.is_empty());
        let mut hid = hid_descriptor();
        hid.w_command_register.set(0);
        hid.w_data_register.set(0);
        let mut setup = FeatureSetup::new(parsed.features, &hid).unwrap();
        assert_eq!(setup.step(), Step::Done);
        assert!(setup.out().is_empty());
        assert_eq!(setup.current_report_id(), None);
        let mut report = ReportPlan::new(Some(4), 2).unwrap();
        report.add(8, 8, 3, 0, 10).unwrap();
        let mut plan = FeaturePlan::empty();
        plan.push(report).unwrap();
        assert!(matches!(
            FeatureSetup::new(plan, &hid),
            Err(FeatureError::InvalidRegisters)
        ));
    }
}
