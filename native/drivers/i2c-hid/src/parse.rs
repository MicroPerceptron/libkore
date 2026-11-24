//! Report descriptor → [`ExtractionTable`], **once**, at bring-up.
//!
//! This is the only place `hidreport` is named, and the only place this
//! crate allocates. Both facts are deliberate: parsing a report descriptor
//! is genuinely intricate — nested collections, usage ranges, global item
//! state that carries across items — and hand-rolling it would be a large
//! surface for silent, wrong-by-one-field bugs. Reusing a maintained
//! parser and snapshotting its output is strictly better than either
//! hand-rolling the parse or calling into it per report.
//!
//! After this function returns, nothing in the steady-state path touches
//! `hidreport`, allocates, or looks at a descriptor. The table is the
//! whole interface.
//!
//! # Why this is feature-gated
//!
//! `hidreport` needs `alloc` against the global allocator. Userland
//! services have one; **the kernel deliberately does not**. The `alloc`
//! feature is off by default so a kernel build cannot link this path even
//! by accident.

use hidreport::{CollectionType, Field as HidField, FieldAttributes, Report, ReportDescriptor};

use crate::features::{FeatureError, FeaturePlan, PlanRefusal, ReportPlan};
use crate::table::{ExtractionTable, Field, TableError, page, usage};

/// How many distinct usages one contact slot may carry before the
/// repeat-detector stops tracking. Includes confidence, tip, in-range, ID,
/// X/Y, pressure, contact dimensions and orientation without splitting a slot.
const MAX_SLOT_USAGES: usize = 16;

/// Only usages consumed by the steady-state extractors belong in the
/// fixed table. A composite device can expose large vendor reports which
/// are valid input but never produce an InputReport. Their wire lengths,
/// report IDs and field bounds are still validated before filtering.
fn retained_usage(usage_page: u16, usage_id: u16) -> bool {
    match usage_page {
        page::GENERIC_DESKTOP => matches!(usage_id, usage::X | usage::Y | usage::WHEEL),
        page::KEYBOARD => true,
        page::BUTTON => (1..=8).contains(&usage_id),
        page::DIGITIZER => matches!(
            usage_id,
            usage::IN_RANGE
                | usage::TIP_SWITCH
                | usage::CONFIDENCE
                | usage::CONTACT_ID
                | usage::CONTACT_COUNT
                | usage::SCAN_TIME
                | usage::PRESSURE
                | usage::WIDTH
                | usage::HEIGHT
                | usage::AZIMUTH
        ),
        _ => false,
    }
}

/// Whether a usage belongs to the whole report rather than one contact.
///
/// The contact count describes the frame, and a clickpad's button is one
/// physical switch shared by every finger on it. Marking them global is
/// what lets each slot see them without any of them starting a new slot.
fn is_report_global(usage_page: u16, usage_id: u16) -> bool {
    (usage_page == page::DIGITIZER && matches!(usage_id, usage::CONTACT_COUNT | usage::SCAN_TIME))
        || usage_page == page::BUTTON
}

/// Whether a repeat of this usage means "a new contact starts here".
///
/// Restricted to the usages that actually identify a finger — the
/// digitizer page and the desktop axes. Without the restriction a
/// keyboard's six array slots, which all carry usage 0, would read as six
/// contacts, and a report with two padding fields would split in half.
fn is_contact_usage(usage_page: u16, usage_id: u16) -> bool {
    (usage_page == page::DIGITIZER && !is_report_global(usage_page, usage_id))
        || (usage_page == page::GENERIC_DESKTOP
            && matches!(usage_id, usage::X | usage::Y | usage::WHEEL))
}

/// Why a report descriptor could not be turned into a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// `hidreport` refused the descriptor.
    Malformed,
    /// The descriptor described more than the table can hold.
    Table(TableError),
    /// The descriptor parsed but described no input reports at all.
    NoInputReports,
    /// A field's bit range does not fit the table's compact
    /// representation.
    ///
    /// Refused rather than narrowed. The table stores an offset as `u16`
    /// and a width as `u8` because a real HID field is small, but casting
    /// a larger value into those types **wraps**: a 512-bit field becomes
    /// `bit_len == 0`, which then sails past
    /// [`TableError::FieldTooWide`] as though it were a legal zero-width
    /// field, and a 264-bit field becomes an 8-bit one that decodes
    /// silently wrong. A device is not trusted to declare a sane
    /// descriptor, so the conversion is checked.
    FieldOutOfRange {
        /// The field's first bit.
        start: usize,
        /// The field's width in bits.
        len: usize,
    },
    /// A complete report is too large for the compact table's byte count.
    ReportSizeOutOfRange(usize),
    /// A declared standard configuration cannot safely be applied.
    Feature(FeatureError),
}

impl ParseError {
    /// Stable, allocation-free detail for the metal bring-up witness.
    ///
    /// The low byte names the refusal class. The upper bytes retain the
    /// useful operand for range failures, so one framebuffer photograph can
    /// distinguish a hostile width from a parser or capacity problem.
    pub const fn diagnostic(self) -> u32 {
        match self {
            Self::Malformed => 1,
            Self::NoInputReports => 2,
            Self::Table(TableError::TooManyFields) => 3,
            Self::Table(TableError::TooManyReports) => 4,
            Self::Table(TableError::FieldTooWide(width)) => 5 | ((width as u32) << 8),
            Self::Table(TableError::FieldZeroWidth) => 6,
            Self::FieldOutOfRange { start, len } => {
                let operand = if start > u16::MAX as usize {
                    start
                } else {
                    len
                };
                let clipped = if operand > 0x00ff_ffff {
                    0x00ff_ffff
                } else {
                    operand
                };
                7 | ((clipped as u32) << 8)
            }
            Self::ReportSizeOutOfRange(bytes) => {
                let clipped = if bytes > 0x00ff_ffff {
                    0x00ff_ffff
                } else {
                    bytes
                };
                8 | ((clipped as u32) << 8)
            }
            Self::Table(TableError::InconsistentContactAxes) => 9,
            Self::Feature(error) => 10 | ((error.diagnostic() & 0x00ff_ffff) << 8),
        }
    }
}

/// Narrow a field's bit range into the table's representation, refusing
/// anything that would wrap.
fn field_bits(start: usize, len: usize) -> Result<(u16, u8), ParseError> {
    let err = || ParseError::FieldOutOfRange { start, len };
    Ok((
        u16::try_from(start).map_err(|_| err())?,
        u8::try_from(len).map_err(|_| err())?,
    ))
}

impl From<TableError> for ParseError {
    fn from(e: TableError) -> Self {
        Self::Table(e)
    }
}

impl From<FeatureError> for ParseError {
    fn from(error: FeatureError) -> Self {
        Self::Feature(error)
    }
}

/// Both parse-once products of a report descriptor. Feature setup must finish
/// before the caller exposes the extraction table as an online input source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedDevice {
    pub table: ExtractionTable,
    pub features: FeaturePlan,
}

/// Parse once into independent steady-state extraction and configuration plans.
/// Neither returned object retains parser allocations or descriptor borrows.
pub fn build_device(descriptor: &[u8]) -> Result<ParsedDevice, ParseError> {
    let descriptor = ReportDescriptor::try_from(descriptor).map_err(|_| ParseError::Malformed)?;
    let table = build_input_table(&descriptor)?;
    let features = build_features(&descriptor, &table)?;
    Ok(ParsedDevice { table, features })
}

const INPUT_MODE: u32 = 0x000d_0052;
const DEVICE_IDENTIFIER: u32 = 0x000d_0053;
const SURFACE_SWITCH: u32 = 0x000d_0057;
const BUTTON_SWITCH: u32 = 0x000d_0058;

fn is_configuration_control(usage: u32) -> bool {
    matches!(usage, INPUT_MODE | SURFACE_SWITCH | BUTTON_SWITCH)
}

fn configuration_collection(field: &hidreport::VariableField) -> Option<&hidreport::Collection> {
    // The top-level Application must be Device Configuration. A standard
    // usage nested inside a vendor application does not authorize a write.
    field
        .collections
        .iter()
        .find(|c| c.collection_type() == CollectionType::Application)
        .filter(|collection| {
            collection
                .usages()
                .iter()
                .any(|usage| u32::from(usage) == 0x000d_000e)
        })
}

fn build_features(
    descriptor: &ReportDescriptor,
    table: &ExtractionTable,
) -> Result<FeaturePlan, FeatureError> {
    use kore_abi::svc::InputReport;
    let classes = table.reports().iter().fold(0, |classes, report| {
        if report.shape != crate::table::ReportShape::Contact {
            return classes;
        }
        classes
            | match report.contact.class {
                InputReport::CONTACT_TOUCHPAD => 1,
                InputReport::CONTACT_TOUCHSCREEN => 2,
                _ => 0,
            }
    });
    let mut plan = FeaturePlan::empty();
    let mut seen = 0u8;
    // Report ID is descriptor-global, not local to the report type. In
    // particular, a numbered Input followed by a Push/Pop-created unnumbered
    // Feature cannot authorize a guessed ID-zero configuration transaction.
    let report_ids = || {
        descriptor
            .input_reports()
            .iter()
            .map(|report| report.report_id().as_ref().map(u8::from))
            .chain(
                descriptor
                    .output_reports()
                    .iter()
                    .map(|report| report.report_id().as_ref().map(u8::from)),
            )
            .chain(
                descriptor
                    .feature_reports()
                    .iter()
                    .map(|report| report.report_id().as_ref().map(u8::from)),
            )
    };
    if report_ids().any(|id| id.is_some()) && report_ids().any(|id| id.is_none() || id == Some(0)) {
        return Err(FeatureError::Plan {
            report_id: 0,
            reason: PlanRefusal::InvalidReportId,
        });
    }
    for report in descriptor.feature_reports() {
        let relevant = report.fields().iter().any(|field| match field {
            HidField::Variable(v) => is_configuration_control(u32::from(&v.usage)),
            HidField::Array(a) => a
                .usages()
                .iter()
                .any(|u| is_configuration_control(u32::from(u))),
            HidField::Constant(_) => false,
        });
        if !relevant {
            // Read-only ContactMaximum, static DeviceIdentifier and arbitrary
            // vendor features must not become blanket GET/SET transactions.
            continue;
        }
        let id = report.report_id().as_ref().map(u8::from);
        let mut selected = ReportPlan::new(id, report.size_in_bytes())?;
        let mode = report.fields().iter().find_map(|field| match field {
            HidField::Variable(v) if u32::from(&v.usage) == INPUT_MODE => Some(v),
            _ => None,
        });
        for field in report.fields() {
            let v = match field {
                HidField::Variable(v) => v,
                HidField::Array(a)
                    if a.usages()
                        .iter()
                        .any(|u| is_configuration_control(u32::from(u))) =>
                {
                    return Err(selected.refuse(PlanRefusal::FieldAttributes));
                }
                _ => continue,
            };
            let usage = u32::from(&v.usage);
            let (flag, value) = match usage {
                INPUT_MODE => (
                    1,
                    match classes {
                        1 => 3,
                        2 => 2,
                        _ => return Err(selected.refuse(PlanRefusal::AmbiguousMode)),
                    },
                ),
                DEVICE_IDENTIFIER => {
                    let Some(mode) = mode else { continue };
                    // DeviceIdentifier outside this exact configuration group
                    // is static capability information, not a mode selector.
                    if v.collections != mode.collections {
                        continue;
                    }
                    (2, 0)
                }
                SURFACE_SWITCH => (4, 1),
                BUTTON_SWITCH => (8, 1),
                _ => continue,
            };
            if configuration_collection(v).is_none() {
                return Err(selected.refuse(PlanRefusal::InvalidCollection));
            }
            if v.is_relative() || v.is_buffered_bytes() || v.wraps() || v.is_nonlinear() {
                return Err(selected.refuse(PlanRefusal::FieldAttributes));
            }
            if seen & flag != 0 {
                // No implicit first/last-wins routing across indistinguishable
                // configuration groups, especially with DeviceIdentifier=all.
                return Err(selected.refuse(PlanRefusal::DuplicateControl));
            }
            selected.add(
                v.bits.start,
                v.bits.len(),
                value,
                i32::from(&v.logical_minimum),
                i32::from(&v.logical_maximum),
            )?;
            seen |= flag;
        }
        selected.set_preserve_unedited(report.fields().iter().any(|field| match field {
            HidField::Constant(_) => false,
            HidField::Variable(field) => !selected.covers(field.bits.start, field.bits.len()),
            HidField::Array(field) => !selected.covers(field.bits.start, field.bits.len()),
        }));
        plan.push(selected)?;
    }
    Ok(plan)
}

/// HID 1.11 §6.2.2.7: logical span / (physical span × 10^exponent).
/// Only linear cm/inch units are admitted, and all arithmetic is checked.
/// Missing/unsupported calibration remains unknown, never guessed from pixels.
fn axis_resolution(field: &hidreport::VariableField) -> Option<(u32, u32)> {
    let unit = u32::from(&field.unit?);
    let (mm_num, mm_den) = match unit {
        0x11 => (10u64, 1u64),
        0x13 => (254, 10),
        _ => return None,
    };
    let logical = u64::try_from(
        i64::from(i32::from(&field.logical_maximum)) - i64::from(i32::from(&field.logical_minimum)),
    )
    .ok()?;
    let physical = u64::try_from(
        i64::from(i32::from(&field.physical_maximum?))
            - i64::from(i32::from(&field.physical_minimum?)),
    )
    .ok()?;
    if logical == 0 || physical == 0 {
        return None;
    }
    let exponent = field.unit_exponent.map_or(0, |e| e.exponent());
    let mut numerator = logical.checked_mul(mm_den)?;
    let mut denominator = physical.checked_mul(mm_num)?;
    let power = 10u64.checked_pow(u32::from(exponent.unsigned_abs()))?;
    if exponent < 0 {
        numerator = numerator.checked_mul(power)?;
    } else {
        denominator = denominator.checked_mul(power)?;
    }
    let (mut a, mut b) = (numerator, denominator);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    Some((
        u32::try_from(numerator / a).ok()?,
        u32::try_from(denominator / a).ok()?,
    ))
}

/// Parse a report descriptor into a table.
///
/// Only **input** reports are tabulated: output and feature reports are
/// written and read on demand through the data register, never decoded on
/// the hot path, so tabulating them would spend table capacity on fields
/// nothing extracts. Unsupported input usages likewise consume no field
/// slots, but retain their original wire offsets and report-size checks. Reports
/// with no retained fields likewise spend no routed stream slots.
pub fn build_table(descriptor: &[u8]) -> Result<ExtractionTable, ParseError> {
    let rd = ReportDescriptor::try_from(descriptor).map_err(|_| ParseError::Malformed)?;
    build_input_table(&rd)
}

fn build_input_table(rd: &ReportDescriptor) -> Result<ExtractionTable, ParseError> {
    let mut table = ExtractionTable::new();

    let inputs = rd.input_reports();
    if inputs.is_empty() {
        return Err(ParseError::NoInputReports);
    }

    // Whether reports carry a leading id byte is a property of the whole
    // descriptor, not of one report: a device either uses ids everywhere
    // or nowhere. Reading it from the first report and applying it to all
    // is what the specification prescribes.
    let uses_ids = inputs.iter().any(|r| r.report_id().is_some());
    table.set_uses_report_ids(uses_ids);

    for report in inputs {
        let id = report.report_id().map_or(0, |r| u8::from(&r));
        let input_bytes = u16::try_from(report.size_in_bytes())
            .map_err(|_| ParseError::ReportSizeOutOfRange(report.size_in_bytes()))?;
        table.record_input_length(id, input_bytes);
        let mut retained = false;

        // Finger logical collections are the authoritative contact boundary.
        // A touchscreen may publish both tip and center X/Y within one finger;
        // repeated axes there do not describe extra contacts. Descriptors
        // without Finger collections retain the repeated-usage fallback.
        let mut slot: u8 = 0;
        let mut seen: [(u16, u16); MAX_SLOT_USAGES] = [(0, 0); MAX_SLOT_USAGES];
        let mut n_seen = 0usize;

        let mut finger = None;
        let mut assign =
            |usage_page: u16, usage_id: u16, collection: Option<&hidreport::CollectionId>| -> u8 {
                if is_report_global(usage_page, usage_id) {
                    return Field::SLOT_GLOBAL;
                }
                if is_contact_usage(usage_page, usage_id) {
                    let next = if let Some(id) = collection {
                        let changed = finger.as_ref().is_some_and(|previous| previous != id);
                        finger = Some(id.clone());
                        changed
                    } else {
                        seen[..n_seen].contains(&(usage_page, usage_id))
                    };
                    if next {
                        slot = slot.saturating_add(1);
                        n_seen = 0;
                    }
                    if n_seen < MAX_SLOT_USAGES {
                        seen[n_seen] = (usage_page, usage_id);
                        n_seen += 1;
                    }
                }
                slot
            };

        for field in report.fields() {
            // Constant fields are padding by definition, and array fields
            // carry usage *sets* rather than one usage per bit range —
            // only the keyboard key array is of that shape, and it is
            // handled by the array arm below.
            match field {
                HidField::Variable(v) => {
                    let u = u32::from(&v.usage);
                    let (usage_page, usage_id) = ((u >> 16) as u16, (u & 0xffff) as u16);
                    let (bit_offset, bit_len) = field_bits(v.bits.start, v.bits.len())?;
                    let mut extracted = Field {
                        usage_page,
                        usage: usage_id,
                        bit_offset,
                        bit_len,
                        signed: v.is_signed(),
                        relative: v.is_relative(),
                        slot: 0,
                    };
                    extracted.validate()?;
                    if !retained_usage(usage_page, usage_id) {
                        continue;
                    }
                    if !retained {
                        table.begin_report_with_size(id, input_bytes)?;
                        retained = true;
                    }
                    if usage_page == page::GENERIC_DESKTOP
                        && matches!(usage_id, usage::X | usage::Y)
                    {
                        let class =
                            v.collections
                                .iter()
                                .flat_map(|c| c.usages())
                                .find_map(|usage| match u32::from(usage) {
                                    0x000d_0005 => {
                                        Some(kore_abi::svc::InputReport::CONTACT_TOUCHPAD)
                                    }
                                    0x000d_0004 => {
                                        Some(kore_abi::svc::InputReport::CONTACT_TOUCHSCREEN)
                                    }
                                    0x000d_0002 => Some(kore_abi::svc::InputReport::CONTACT_PEN),
                                    _ => None,
                                });
                        if let Some(class) = class {
                            table.set_contact_axis(
                                class,
                                usage_id,
                                i32::from(&v.logical_minimum),
                                i32::from(&v.logical_maximum),
                            )?;
                            table.set_contact_resolution(usage_id, axis_resolution(v))?;
                        }
                    }
                    let finger = v
                        .collections
                        .iter()
                        .rev()
                        .find(|collection| {
                            collection.collection_type() == CollectionType::Logical
                                && collection
                                    .usages()
                                    .iter()
                                    .any(|u| u32::from(u) == 0x000d_0022)
                        })
                        .map(|collection| collection.id());
                    extracted.slot = assign(usage_page, usage_id, finger);
                    table.push_field(extracted)?;
                }
                HidField::Array(a) => {
                    // A keyboard's key array: N slots of equal width, each
                    // carrying a usage *code* rather than a flag. The slots
                    // are recorded individually so the extractor can read
                    // each one; the usage page is taken from the array's
                    // own usages, which is what says "these are keys".
                    // `usages()` is the expanded inclusive set, whether
                    // the descriptor spelled it as a list or as a
                    // minimum/maximum range, so the first entry carries
                    // the page in both forms.
                    let array_page = a
                        .usages()
                        .first()
                        .map(|u| (u32::from(u) >> 16) as u16)
                        .unwrap_or(0);
                    let bits = &a.bits;
                    let count = usize::from(&a.report_count).max(1);
                    let width = (bits.len() / count).max(1);
                    for i in 0..count {
                        // Checked throughout: the slot offset is derived
                        // arithmetic, so it can overflow before it is even
                        // narrowed, and a wrapped offset reads a field
                        // from the wrong place rather than failing.
                        let start = i
                            .checked_mul(width)
                            .and_then(|o| bits.start.checked_add(o))
                            .ok_or(ParseError::FieldOutOfRange {
                                start: bits.start,
                                len: width,
                            })?;
                        let (bit_offset, bit_len) = field_bits(start, width)?;
                        let mut extracted = Field {
                            usage_page: array_page,
                            // An array slot has no single usage; zero
                            // marks it as "a code lives here", which the
                            // keyboard decoder reads by width, not usage.
                            usage: 0,
                            bit_offset,
                            bit_len,
                            signed: false,
                            relative: a.is_relative(),
                            slot: 0,
                        };
                        extracted.validate()?;
                        if array_page == page::KEYBOARD {
                            if !retained {
                                table.begin_report_with_size(id, input_bytes)?;
                                retained = true;
                            }
                            extracted.slot = assign(array_page, 0, None);
                            table.push_field(extracted)?;
                        }
                    }
                }
                HidField::Constant(_) => {}
            }
        }
    }

    table.classify();
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_bits_refuses_every_range_that_would_wrap() {
        // The boundaries themselves are lawful.
        assert_eq!(field_bits(0, 1), Ok((0, 1)));
        assert_eq!(
            field_bits(u16::MAX as usize, u8::MAX as usize),
            Ok((u16::MAX, u8::MAX))
        );

        // One past each is refused rather than narrowed. Before the check
        // these produced `bit_offset == 0` and `bit_len == 0` — a field
        // decoded from the wrong place, and a width that then slipped past
        // `TableError::FieldTooWide` as though it were legal.
        assert_eq!(
            field_bits(u16::MAX as usize + 1, 8),
            Err(ParseError::FieldOutOfRange {
                start: 65536,
                len: 8
            })
        );
        assert_eq!(
            field_bits(0, u8::MAX as usize + 1),
            Err(ParseError::FieldOutOfRange { start: 0, len: 256 })
        );

        // The specific value that motivated the check: a 512-bit field
        // wraps to exactly zero under `as u8`.
        assert_eq!(512_usize as u8, 0, "the wrap this refusal prevents");
        assert!(field_bits(0, 512).is_err());
    }
}
