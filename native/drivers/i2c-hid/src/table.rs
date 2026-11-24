//! The **extraction table** — the hot path, and the reason the report
//! descriptor is parsed exactly once.
//!
//! A report descriptor is a nested item stream that takes real work to
//! interpret. Doing that per report, sixty times a second per finger,
//! would be absurd; doing it once at bring-up and snapshotting the result
//! into a fixed table makes the steady state pure bit arithmetic over a
//! `[u8]`, with no allocation and no descriptor logic anywhere near it.
//!
//! The table is also the pointer seam. A keyboard, a mouse, and a
//! digitizer differ here only in which usages their fields carry and
//! which [`InputReport`] kind those usages assemble into — so supporting
//! a new class is a new [`ReportShape`] arm, not a new driver.

use kore_abi::svc::InputReport;

/// HID usage pages this driver understands.
pub mod page {
    /// Generic Desktop — X, Y, Wheel, and the mouse/keyboard collections.
    pub const GENERIC_DESKTOP: u16 = 0x01;
    /// Keyboard/Keypad — the usage codes a keyboard report carries.
    pub const KEYBOARD: u16 = 0x07;
    /// Button — mouse and touchpad buttons, numbered from 1.
    pub const BUTTON: u16 = 0x09;
    /// Digitizer — touchpads and touchscreens.
    pub const DIGITIZER: u16 = 0x0D;
}

/// Usage ids, within their pages.
pub mod usage {
    /// Generic Desktop: X.
    pub const X: u16 = 0x30;
    /// Generic Desktop: Y.
    pub const Y: u16 = 0x31;
    /// Generic Desktop: Wheel.
    pub const WHEEL: u16 = 0x38;
    /// Digitizer: in-range (hovering).
    pub const IN_RANGE: u16 = 0x32;
    /// Digitizer: tip switch (touching).
    pub const TIP_SWITCH: u16 = 0x42;
    /// Digitizer: confidence (not a palm).
    pub const CONFIDENCE: u16 = 0x47;
    /// Digitizer: contact identifier.
    pub const CONTACT_ID: u16 = 0x51;
    /// Digitizer: how many contacts are in this frame.
    pub const CONTACT_COUNT: u16 = 0x54;
    // USB HID Usage Tables, Digitizers page §16.
    pub const PRESSURE: u16 = 0x30;
    pub const WIDTH: u16 = 0x48;
    pub const HEIGHT: u16 = 0x49;
    pub const AZIMUTH: u16 = 0x3f;
    pub const SCAN_TIME: u16 = 0x56;
}

/// How many extracted fields one table can hold.
///
/// Sized for the worst real case rather than the common one: a ten-contact
/// touchscreen publishes five to seven fields *per contact*, and one table
/// holds the supported usages from every input report a device declares,
/// not just one. Vendor/unsupported usages still contribute to wire sizes
/// and descriptor validation, but consume no extraction slots. 64 fits a
/// five-finger touchpad and would silently refuse a ten-finger screen, so
/// the cap is set where a refusal means "this device is unreasonable"
/// rather than "this device is a touchscreen".
pub const MAX_FIELDS: usize = 128;

/// The most contacts one report may carry.
///
/// The Windows Precision Touchpad specification caps a frame at five
/// contacts; touchscreens go to ten. Sized for the larger, so a caller's
/// buffer of this length is never the reason a contact is dropped.
pub const MAX_CONTACTS: usize = 10;

/// How many input streams with retained fields one device may publish.
/// Vendor-only report IDs have wire lengths, but consume no stream slots.
pub const MAX_REPORTS: usize = 8;

/// One extracted field: where its bits are, and what they mean.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Field {
    /// The usage page.
    pub usage_page: u16,
    /// The usage within that page.
    pub usage: u16,
    /// Bit offset from the start of the **whole report, including the
    /// report-id byte** when the device uses one.
    ///
    /// Absolute rather than payload-relative on purpose: this is the
    /// convention HID report descriptors themselves use, so a table built
    /// from a descriptor needs no adjustment. Making it payload-relative
    /// would put an easily-forgotten `- 8` between the parser and the
    /// extractor, which is precisely the off-by-one that decodes every
    /// field one byte early.
    pub bit_offset: u16,
    /// How many bits wide.
    pub bit_len: u8,
    /// Whether the value is two's-complement.
    pub signed: bool,
    /// Whether the HID Input item declares a relative value. Absolute
    /// coordinates must never become mouse deltas merely because their
    /// usages are X and Y; the collection's advertised mode can differ
    /// between reports from the same device.
    pub relative: bool,
    /// Which **contact slot** this field belongs to, or
    /// [`SLOT_GLOBAL`](Self::SLOT_GLOBAL) for a report-wide field.
    ///
    /// A digitizer report repeats the same usages once per finger. Without
    /// a slot the extractor can only ever find the *first* occurrence of
    /// each usage, which decodes finger 0 and silently discards every
    /// other finger — a report that correctly says "two contacts" while
    /// delivering one.
    pub slot: u8,
}

impl Field {
    /// A field that belongs to the whole report rather than one contact:
    /// the contact count, and the clickpad button. Visible from every
    /// slot, because every contact in a frame shares them.
    pub const SLOT_GLOBAL: u8 = 0xFF;

    /// Validate even fields the parser does not retain. Filtering an
    /// unsupported usage must not make a malformed descriptor admissible.
    pub(crate) fn validate(&self) -> Result<(), TableError> {
        if self.bit_len == 0 {
            return Err(TableError::FieldZeroWidth);
        }
        if self.bit_len > 32 {
            return Err(TableError::FieldTooWide(self.bit_len));
        }
        Ok(())
    }
}

/// What kind of report an id carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReportShape {
    /// Nothing this driver decodes.
    #[default]
    Unknown,
    /// Relative pointer: buttons and deltas.
    Mouse,
    /// Absolute digitizer contact.
    Contact,
    /// Keyboard modifiers plus usage codes.
    Keyboard,
}

/// One report id's fields and shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReportEntry {
    /// The report id, or zero when the device publishes none.
    pub report_id: u8,
    /// Exact size of this HID report on the wire, including the report-id
    /// byte when the descriptor uses ids. Zero is reserved for tables
    /// assembled manually by tests which do not know the descriptor size.
    pub input_bytes: u16,
    /// What this report decodes to.
    pub shape: ReportShape,
    /// Index of this report's first field.
    pub first: u16,
    /// How many fields it has.
    pub count: u16,
    /// How many contact slots this report carries. Always 1 for a
    /// non-digitizer report.
    pub slots: u16,
    /// Descriptor-derived digitizer class and logical axis ranges.
    pub contact: ContactInfo,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContactInfo {
    pub class: u8,
    pub x: (i32, i32),
    pub y: (i32, i32),
    /// Logical units / millimeters, reduced rational; None is unknown.
    pub resolution: [Option<(u32, u32)>; 2],
    resolution_seen: [bool; 2],
}

impl ContactInfo {
    const EMPTY: Self = Self {
        class: 0,
        x: (0, 0),
        y: (0, 0),
        resolution: [None; 2],
        resolution_seen: [false; 2],
    };
    fn ranges(self) -> Option<(u16, u16)> {
        Some((
            u16::try_from(self.x.1.checked_sub(self.x.0)?)
                .ok()
                .filter(|n| *n != 0)?,
            u16::try_from(self.y.1.checked_sub(self.y.0)?)
                .ok()
                .filter(|n| *n != 0)?,
        ))
    }
}

/// The parsed-once snapshot of a device's report descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractionTable {
    fields: [Field; MAX_FIELDS],
    n_fields: usize,
    reports: [ReportEntry; MAX_REPORTS],
    n_reports: usize,
    /// Exact wire lengths indexed by the HID report-id byte (zero for
    /// unnumbered input). Vendor-only reports retain validation without
    /// consuming an extraction entry or a routed source ID. Zero means no
    /// size constraint, as for manually assembled size-less test tables.
    input_lengths: [u16; u8::MAX as usize + 1],
    /// Whether reports are prefixed with their id.
    uses_report_ids: bool,
}

impl Default for ExtractionTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a field or report could not be added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableError {
    /// Contact slots in one report must share coordinate calibration.
    InconsistentContactAxes,
    /// The table's field capacity was reached.
    TooManyFields,
    /// The table's report capacity was reached.
    TooManyReports,
    /// A field wider than 32 bits, which no usage this driver decodes
    /// ever is, and which the extractor cannot represent.
    FieldTooWide(u8),
    /// A field of zero width.
    ///
    /// Nothing can be extracted from it, so admitting one would put a
    /// field in the table that always reads zero and silently satisfies
    /// whatever usage it claims — a decode that looks like a device
    /// reporting nothing rather than like a malformed descriptor.
    FieldZeroWidth,
}

impl ExtractionTable {
    /// Stable descriptor-table stream index, independent of report arrival order.
    pub fn stream_index(&self, data: &[u8]) -> Option<usize> {
        self.selected(data).map(|(index, _, _)| index)
    }
    pub(crate) fn selected(&self, data: &[u8]) -> Option<(usize, &ReportEntry, &[Field])> {
        let id = if self.uses_report_ids {
            *data.first()?
        } else {
            0
        };
        let index = self.reports[..self.n_reports]
            .iter()
            .position(|r| r.report_id == id)?;
        let entry = &self.reports[index];
        Some((
            index,
            entry,
            &self.fields[entry.first as usize..(entry.first + entry.count) as usize],
        ))
    }

    pub fn set_contact_resolution(
        &mut self,
        axis: u16,
        resolution: Option<(u32, u32)>,
    ) -> Result<(), TableError> {
        if let Some(index) = self.n_reports.checked_sub(1) {
            let axis = match axis {
                usage::X => 0,
                usage::Y => 1,
                _ => return Ok(()),
            };
            let contact = &mut self.reports[index].contact;
            if contact.resolution_seen[axis] && contact.resolution[axis] != resolution {
                return Err(TableError::InconsistentContactAxes);
            }
            contact.resolution_seen[axis] = true;
            contact.resolution[axis] = resolution;
        }
        Ok(())
    }

    /// An empty table.
    pub const fn new() -> Self {
        Self {
            fields: [Field {
                usage_page: 0,
                usage: 0,
                bit_offset: 0,
                bit_len: 0,
                signed: false,
                relative: false,
                slot: 0,
            }; MAX_FIELDS],
            n_fields: 0,
            reports: [ReportEntry {
                report_id: 0,
                input_bytes: 0,
                shape: ReportShape::Unknown,
                first: 0,
                count: 0,
                slots: 1,
                contact: ContactInfo::EMPTY,
            }; MAX_REPORTS],
            n_reports: 0,
            input_lengths: [0; u8::MAX as usize + 1],
            uses_report_ids: false,
        }
    }

    /// Whether reports carry a leading id byte.
    pub fn uses_report_ids(&self) -> bool {
        self.uses_report_ids
    }

    /// Record whether reports carry a leading id byte.
    pub fn set_uses_report_ids(&mut self, yes: bool) {
        self.uses_report_ids = yes;
    }

    /// The reports this table can decode.
    pub fn reports(&self) -> &[ReportEntry] {
        &self.reports[..self.n_reports]
    }

    /// Every field, across all reports.
    pub fn fields(&self) -> &[Field] {
        &self.fields[..self.n_fields]
    }

    /// Begin a report. Fields pushed after this belong to it.
    pub fn begin_report(&mut self, report_id: u8) -> Result<(), TableError> {
        self.begin_report_with_size(report_id, 0)
    }

    /// Begin a report whose descriptor declares `input_bytes` bytes on the
    /// wire. The parser uses this form; the size-less wrapper remains useful
    /// for small hand-assembled extraction-table tests.
    pub fn begin_report_with_size(
        &mut self,
        report_id: u8,
        input_bytes: u16,
    ) -> Result<(), TableError> {
        if self.n_reports == MAX_REPORTS {
            return Err(TableError::TooManyReports);
        }
        self.record_input_length(report_id, input_bytes);
        self.reports[self.n_reports] = ReportEntry {
            report_id,
            input_bytes,
            shape: ReportShape::Unknown,
            first: self.n_fields as u16,
            count: 0,
            slots: 1,
            contact: ContactInfo::EMPTY,
        };
        self.n_reports += 1;
        Ok(())
    }

    /// Add a field to the report most recently begun.
    pub fn push_field(&mut self, f: Field) -> Result<(), TableError> {
        f.validate()?;
        if self.n_fields == MAX_FIELDS {
            return Err(TableError::TooManyFields);
        }
        self.fields[self.n_fields] = f;
        self.n_fields += 1;
        if let Some(r) = self.reports[..self.n_reports].last_mut() {
            r.count += 1;
        }
        Ok(())
    }

    /// Classify every report from the usages its fields carry.
    pub fn set_contact_axis(
        &mut self,
        class: u8,
        axis: u16,
        minimum: i32,
        maximum: i32,
    ) -> Result<(), TableError> {
        let Some(r) = self
            .n_reports
            .checked_sub(1)
            .and_then(|i| self.reports.get_mut(i))
        else {
            return Ok(());
        };
        if minimum >= maximum || (r.contact.class != 0 && r.contact.class != class) {
            return Err(TableError::InconsistentContactAxes);
        }
        r.contact.class = class;
        let target = match axis {
            usage::X => &mut r.contact.x,
            usage::Y => &mut r.contact.y,
            _ => return Ok(()),
        };
        if *target != (0, 0) && *target != (minimum, maximum) {
            return Err(TableError::InconsistentContactAxes);
        }
        *target = (minimum, maximum);
        Ok(())
    }

    /// Classify every report from the usages its fields carry.
    ///
    /// Deliberately after the fact rather than as fields arrive: a
    /// report's kind is a property of the whole set (relative X and Y
    /// make a mouse; absolute X, Y and a contact id make a digitizer).
    /// An absolute mouse/vendor report needs an explicit mapping that this
    /// driver does not implement; it is not a relative-pointer fallback.
    pub fn classify(&mut self) {
        for i in 0..self.n_reports {
            let r = self.reports[i];
            let fields = &self.fields[r.first as usize..(r.first + r.count) as usize];

            // The slot count is the highest slot any field claims, plus
            // one. Global fields are excluded — the contact count is not
            // itself a contact.
            self.reports[i].slots = fields
                .iter()
                .filter(|f| f.slot != Field::SLOT_GLOBAL)
                .map(|f| f.slot as u16 + 1)
                .max()
                .unwrap_or(1);
            let has = |p: u16, u: u16| fields.iter().any(|f| f.usage_page == p && f.usage == u);

            let digitizer = has(page::DIGITIZER, usage::CONTACT_ID)
                || has(page::DIGITIZER, usage::TIP_SWITCH)
                || has(page::DIGITIZER, usage::CONTACT_COUNT);
            let pointer =
                has(page::GENERIC_DESKTOP, usage::X) && has(page::GENERIC_DESKTOP, usage::Y);
            let axes = || {
                fields.iter().filter(|f| {
                    f.usage_page == page::GENERIC_DESKTOP && matches!(f.usage, usage::X | usage::Y)
                })
            };
            let absolute = axes().all(|f| !f.relative);
            let relative = axes().all(|f| f.relative);
            let keyboard = fields.iter().any(|f| f.usage_page == page::KEYBOARD);

            self.reports[i].shape = if digitizer && pointer && absolute {
                ReportShape::Contact
            } else if !digitizer && pointer && relative {
                ReportShape::Mouse
            } else if keyboard {
                ReportShape::Keyboard
            } else {
                ReportShape::Unknown
            };
        }
    }

    pub(crate) fn record_input_length(&mut self, id: u8, input_bytes: u16) {
        self.input_lengths[usize::from(id)] = input_bytes;
    }

    /// Descriptor-declared wire length, including the ID byte. This also
    /// recognizes vendor-only reports which own no input stream. None means
    /// an unknown ID or a manually assembled table without wire sizes.
    pub fn expected_input_length(&self, data: &[u8]) -> Option<usize> {
        let id = if self.uses_report_ids {
            data.first().copied().unwrap_or(0)
        } else {
            0
        };
        let length = self.input_lengths[usize::from(id)];
        (length != 0).then_some(usize::from(length))
    }

    /// Find a report entry by id.
    pub fn report(&self, id: u8) -> Option<&ReportEntry> {
        self.reports[..self.n_reports]
            .iter()
            .find(|r| r.report_id == id)
    }

    /// Decode one input report into the wire form the router carries,
    /// writing **one [`InputReport`] per contact** into `out`.
    ///
    /// Returns how many were written. A mouse or keyboard report always
    /// writes one; a digitizer writes one per live contact, bounded by
    /// `out.len()`.
    ///
    /// `out` rather than a return value because a digitizer frame is
    /// genuinely plural: five fingers arrive in one report, and returning
    /// the first would be the bug this method exists to fix. Callers size
    /// `out` at [`MAX_CONTACTS`].
    ///
    /// Writes nothing and returns zero for a report this table cannot
    /// name — which is a normal thing for a device to send, not an error.
    pub fn decode_into(&self, data: &[u8], source: u8, out: &mut [InputReport]) -> usize {
        // The id byte selects the report; the *whole* buffer, id byte
        // included, is what the fields are offset against.
        let id = if self.uses_report_ids {
            match data.first() {
                Some(b) => *b,
                None => return 0,
            }
        } else {
            0
        };

        let Some(entry) = self.report(id) else {
            return 0;
        };
        // A field extractor is deliberately bounds-safe, but silently
        // zero-filling a truncated keyboard report would manufacture a
        // key-up. Refuse both short and overlong frames against the exact
        // descriptor-declared size before interpreting any field.
        if entry.input_bytes != 0 && data.len() != usize::from(entry.input_bytes) {
            return 0;
        }
        let fields = &self.fields[entry.first as usize..(entry.first + entry.count) as usize];
        if out.is_empty() {
            return 0;
        }

        match entry.shape {
            ReportShape::Mouse => {
                out[0] = decode_mouse(fields, data, source);
                1
            }
            ReportShape::Keyboard => {
                out[0] = decode_keyboard(fields, data, source);
                1
            }
            ReportShape::Contact => {
                let slots = (entry.slots as usize).min(out.len()).min(MAX_CONTACTS);

                // The contact count bounds how many slots carry live data.
                // Slots past it hold stale values the device did not
                // refresh, and emitting them would replay a finger that is
                // no longer down.
                let live = match find(
                    fields,
                    Field::SLOT_GLOBAL,
                    page::DIGITIZER,
                    usage::CONTACT_COUNT,
                ) {
                    // A count of zero is not "nothing to report" — it is
                    // "every finger lifted", which the consumer must see
                    // to release the cursor. So slot 0 is always emitted;
                    // its own tip bit carries the lifted state.
                    Some(f) => {
                        (value(f, data).clamp(0, MAX_CONTACTS as i32) as usize).clamp(1, slots)
                    }
                    // No count field: the device publishes a fixed set of
                    // slots and marks liveness per contact.
                    None => slots,
                };

                for (i, slot) in (0..live).enumerate() {
                    out[i] = decode_contact(fields, slot as u8, data, source, entry.contact);
                }
                live
            }
            ReportShape::Unknown => 0,
        }
    }
}

/// Read `bit_len` bits at `bit_offset`, LSB-first within each byte.
///
/// HID packs fields little-endian at the bit level, so bit `n` of a field
/// is bit `(offset + n) % 8` of byte `(offset + n) / 8`. Reading past the
/// end of `data` yields zeroes rather than panicking: a device is free to
/// send a report shorter than its descriptor promised, and a driver that
/// panicked on one would be a device that can crash its host.
pub fn extract(data: &[u8], bit_offset: usize, bit_len: usize) -> u32 {
    let mut v: u32 = 0;
    for i in 0..bit_len.min(32) {
        let bit = bit_offset + i;
        let byte = bit / 8;
        if byte >= data.len() {
            break;
        }
        if (data[byte] >> (bit % 8)) & 1 != 0 {
            v |= 1 << i;
        }
    }
    v
}

/// Sign-extend a `bits`-wide two's-complement value.
pub fn sign_extend(v: u32, bits: usize) -> i32 {
    if bits == 0 || bits >= 32 {
        return v as i32;
    }
    let sign = 1u32 << (bits - 1);
    if v & sign != 0 {
        (v | !((1u32 << bits) - 1)) as i32
    } else {
        v as i32
    }
}

/// Read one field's value, sign-extended if the descriptor said so.
pub(crate) fn value(f: &Field, data: &[u8]) -> i32 {
    let raw = extract(data, f.bit_offset as usize, f.bit_len as usize);
    if f.signed {
        sign_extend(raw, f.bit_len as usize)
    } else {
        raw as i32
    }
}

/// The first field matching a usage **within one contact slot**.
///
/// A field marked [`Field::SLOT_GLOBAL`] matches from every slot, because
/// the contact count and the clickpad button belong to the whole frame
/// rather than to one finger.
pub(crate) fn find(fields: &[Field], slot: u8, p: u16, u: u16) -> Option<&Field> {
    fields.iter().find(|f| {
        f.usage_page == p && f.usage == u && (f.slot == slot || f.slot == Field::SLOT_GLOBAL)
    })
}

/// Collapse every Button-page field into a bitmap, button 1 at bit 0.
pub(crate) fn buttons(fields: &[Field], slot: u8, data: &[u8]) -> u8 {
    let mut bits = 0u8;
    for f in fields.iter().filter(|f| {
        f.usage_page == page::BUTTON && (f.slot == slot || f.slot == Field::SLOT_GLOBAL)
    }) {
        if f.usage == 0 || f.usage > 8 {
            continue;
        }
        if value(f, data) != 0 {
            bits |= 1 << (f.usage - 1);
        }
    }
    bits
}

fn decode_mouse(fields: &[Field], data: &[u8], source: u8) -> InputReport {
    let get = |u: u16| {
        find(fields, 0, page::GENERIC_DESKTOP, u)
            .filter(|f| f.relative)
            .map_or(0, |f| value(f, data))
    };
    // Deltas are clamped rather than truncated: a wrapped delta sends the
    // cursor the wrong way, which is worse than a capped one.
    let clamp = |v: i32| v.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
    InputReport::mouse(
        source,
        buttons(fields, 0, data),
        clamp(get(usage::X)),
        clamp(get(usage::Y)),
        clamp(get(usage::WHEEL)),
    )
}

fn decode_contact(
    fields: &[Field],
    slot: u8,
    data: &[u8],
    source: u8,
    info: ContactInfo,
) -> InputReport {
    let flag = |p: u16, u: u16, bit: u8| -> u8 {
        match find(fields, slot, p, u) {
            Some(f) if value(f, data) != 0 => bit,
            _ => 0,
        }
    };
    let mut state = flag(page::DIGITIZER, usage::TIP_SWITCH, InputReport::CONTACT_TIP)
        | flag(
            page::DIGITIZER,
            usage::CONFIDENCE,
            InputReport::CONTACT_CONFIDENCE,
        )
        | flag(
            page::DIGITIZER,
            usage::IN_RANGE,
            InputReport::CONTACT_IN_RANGE,
        );
    if buttons(fields, slot, data) != 0 {
        state |= InputReport::CONTACT_BUTTON;
    }
    // Devices without a confidence usage cannot mark palms. Absence is not
    // a negative confidence report (which would suppress all their input).
    if find(fields, slot, page::DIGITIZER, usage::CONFIDENCE).is_none() {
        state |= InputReport::CONTACT_CONFIDENCE;
    }

    // Absolute coordinates stay in the digitizer's own logical units —
    // scaling belongs to the consumer, which is the only layer that knows
    // the surface geometry it is mapping onto.
    let coord = |u: u16| -> u16 {
        let value = find(fields, slot, page::GENERIC_DESKTOP, u).map_or(0, |f| value(f, data));
        if info.ranges().is_some() {
            let (min, max) = if u == usage::X { info.x } else { info.y };
            (value.clamp(min, max) - min) as u16
        } else {
            value.clamp(0, u16::MAX as i32) as u16
        }
    };
    let byte = |p: u16, u: u16| -> u8 {
        find(fields, slot, p, u).map_or(0, |f| value(f, data).clamp(0, u8::MAX as i32) as u8)
    };

    let report = InputReport::contact(
        source,
        state,
        coord(usage::X),
        coord(usage::Y),
        // A device that publishes no contact-identifier field still has
        // distinguishable fingers: the slot index is their identity.
        find(fields, slot, page::DIGITIZER, usage::CONTACT_ID)
            .map_or(slot, |f| value(f, data).clamp(0, u8::MAX as i32) as u8),
        if find(fields, slot, page::DIGITIZER, usage::CONTACT_COUNT).is_some() {
            byte(page::DIGITIZER, usage::CONTACT_COUNT)
        } else {
            u8::from(state & InputReport::CONTACT_TIP != 0)
        },
    );
    if let Some((x, y)) = info.ranges() {
        report.with_contact_axes(info.class, x, y)
    } else {
        report
    }
}

fn decode_keyboard(fields: &[Field], data: &[u8], source: u8) -> InputReport {
    let mut modifiers = 0u8;
    let mut keys = [0u8; 6];
    let mut n = 0;

    for f in fields.iter().filter(|f| f.usage_page == page::KEYBOARD) {
        let v = value(f, data);
        // A one-bit keyboard-page field is a modifier, reported by its
        // own usage; a wider one is an array slot carrying a usage code.
        if f.bit_len == 1 {
            if v != 0 && (0xE0..=0xE7).contains(&f.usage) {
                modifiers |= 1 << (f.usage - 0xE0);
            }
        } else if v != 0 && n < keys.len() {
            keys[n] = v.clamp(0, u8::MAX as i32) as u8;
            n += 1;
        }
    }

    InputReport::keyboard(source, modifiers, keys)
}
