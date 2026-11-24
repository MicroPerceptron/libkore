//! The protocol, end to end, against a device that speaks it.
//!
//! The decode tests are the ones that matter: canned report descriptors —
//! a boot mouse, a boot keyboard, and a precision touchpad — parsed
//! **once** into a table, then fed real report bytes and asserted equal to
//! the exact [`InputReport`] the router should carry. That is the whole
//! claim of this crate, checked without hardware.

use kore_abi::svc::InputReport;
use kore_i2c::I2cBus;
use kore_i2c::mock::{I2cTarget, MockBus, TargetReply};

use crate::bringup::{BringUp, POWER_SETTLE_NS, Quirks, Step};
use crate::command::{Command, PowerState};
use crate::descriptor::HidDescriptor;
use crate::framing::{Frame, decode};
use crate::table::{
    ExtractionTable, Field, MAX_CONTACTS, MAX_FIELDS, MAX_REPORTS, ReportShape, extract, page,
    sign_extend, usage,
};

/// Decode a report that should yield exactly one record — the mouse and
/// keyboard shape. Panics if the table produced more than one, so a test
/// written for a single record cannot silently ignore extra contacts.
fn decode1(t: &ExtractionTable, data: &[u8], source: u8) -> Option<InputReport> {
    let mut out = [InputReport::keyboard(0, 0, [0; 6]); MAX_CONTACTS];
    match t.decode_into(data, source, &mut out) {
        0 => None,
        1 => Some(out[0]),
        n => panic!("expected one record, got {n}"),
    }
}

// ─────────────────────────── canned descriptors ───────────────────────────

/// The USB HID boot-mouse report descriptor: 3 buttons, 5 bits padding,
/// then signed 8-bit X and Y. No report ids.
const MOUSE_DESC: &[u8] = &[
    0x05, 0x01, // Usage Page (Generic Desktop)
    0x09, 0x02, // Usage (Mouse)
    0xA1, 0x01, // Collection (Application)
    0x09, 0x01, //   Usage (Pointer)
    0xA1, 0x00, //   Collection (Physical)
    0x05, 0x09, //     Usage Page (Button)
    0x19, 0x01, //     Usage Minimum (1)
    0x29, 0x03, //     Usage Maximum (3)
    0x15, 0x00, //     Logical Minimum (0)
    0x25, 0x01, //     Logical Maximum (1)
    0x95, 0x03, //     Report Count (3)
    0x75, 0x01, //     Report Size (1)
    0x81, 0x02, //     Input (Data,Var,Abs)
    0x95, 0x01, //     Report Count (1)
    0x75, 0x05, //     Report Size (5)
    0x81, 0x03, //     Input (Const) — padding
    0x05, 0x01, //     Usage Page (Generic Desktop)
    0x09, 0x30, //     Usage (X)
    0x09, 0x31, //     Usage (Y)
    0x15, 0x81, //     Logical Minimum (-127)
    0x25, 0x7F, //     Logical Maximum (127)
    0x75, 0x08, //     Report Size (8)
    0x95, 0x02, //     Report Count (2)
    0x81, 0x06, //     Input (Data,Var,Rel)
    0xC0, //   End Collection
    0xC0, // End Collection
];

/// The USB HID boot-keyboard report descriptor: 8 modifier bits, one
/// reserved byte, then six 8-bit key-code array slots.
const KEYBOARD_DESC: &[u8] = &[
    0x05, 0x01, // Usage Page (Generic Desktop)
    0x09, 0x06, // Usage (Keyboard)
    0xA1, 0x01, // Collection (Application)
    0x05, 0x07, //   Usage Page (Keyboard)
    0x19, 0xE0, //   Usage Minimum (LeftControl)
    0x29, 0xE7, //   Usage Maximum (RightGUI)
    0x15, 0x00, //   Logical Minimum (0)
    0x25, 0x01, //   Logical Maximum (1)
    0x75, 0x01, //   Report Size (1)
    0x95, 0x08, //   Report Count (8)
    0x81, 0x02, //   Input (Data,Var,Abs) — modifiers
    0x95, 0x01, //   Report Count (1)
    0x75, 0x08, //   Report Size (8)
    0x81, 0x01, //   Input (Const) — reserved
    0x95, 0x06, //   Report Count (6)
    0x75, 0x08, //   Report Size (8)
    0x15, 0x00, //   Logical Minimum (0)
    0x25, 0x65, //   Logical Maximum (101)
    0x05, 0x07, //   Usage Page (Keyboard)
    0x19, 0x00, //   Usage Minimum (0)
    0x29, 0x65, //   Usage Maximum (101)
    0x81, 0x00, //   Input (Data,Array) — keys
    0xC0, // End Collection
];

/// Deterministic reconstruction of all five top-level collections exposed by
/// the live ThinkBook `QTEC0001` (VID:PID 048d:83db). Windows intentionally
/// exposes preparsed HID capabilities rather than the original descriptor
/// bytes; `winhiddump` reconstructed Col01..Col05 as report ids 32, 33, 34,
/// 5a (feature-only), and 35. Col01's preparsed keyboard caps additionally
/// identify the six 00..DD array slots which the generic reconstruction marks
/// as padding. Keeping that distinction explicit avoids calling this a raw
/// firmware capture while still replaying the complete live device shape.
const QTEC0001_COMPOSITE_RECONSTRUCTED_DESC: &[u8] = &[
    // Col01: keyboard, input report 0x32, nine bytes including report id.
    0x05, 0x01, // Usage Page (Generic Desktop)
    0x09, 0x06, // Usage (Keyboard)
    0xA1, 0x01, // Collection (Application)
    0x85, 0x32, //   Report ID (0x32)
    0x05, 0x07, //   Usage Page (Keyboard)
    0x19, 0xE0, //   Usage Minimum (LeftControl)
    0x29, 0xE7, //   Usage Maximum (RightGUI)
    0x15, 0x00, //   Logical Minimum (0)
    0x25, 0x01, //   Logical Maximum (1)
    0x75, 0x01, //   Report Size (1)
    0x95, 0x08, //   Report Count (8)
    0x81, 0x02, //   Input (Data,Var,Abs) -- modifiers
    0x95, 0x01, //   Report Count (1)
    0x75, 0x08, //   Report Size (8)
    0x81, 0x01, //   Input (Const) -- reserved
    0x95, 0x06, //   Report Count (6)
    0x75, 0x08, //   Report Size (8)
    0x15, 0x00, //   Logical Minimum (0)
    0x26, 0xDD, 0x00, // Logical Maximum (0x00dd)
    0x05, 0x07, //   Usage Page (Keyboard)
    0x19, 0x00, //   Usage Minimum (0)
    0x2A, 0xDD, 0x00, // Usage Maximum (0x00dd)
    0x81, 0x00, //   Input (Data,Array) -- keys
    0x05, 0x08, 0x19, 0x01, 0x29, 0x05, 0x95, 0x05, 0x75, 0x01, 0x91, 0x02, 0x95, 0x03, 0x91,
    0x03, // LEDs and output padding
    0xC0, // End Collection
    // Col02: consumer control, report 0x33 is two constant bytes.
    0x05, 0x0C, 0x09, 0x01, 0xA1, 0x01, 0x85, 0x33, 0x15, 0x00, 0x25, 0x01, 0x35, 0x00, 0x45, 0x01,
    0x65, 0x00, 0x55, 0x00, 0x75, 0x01, 0x95, 0x10, 0x81, 0x03, 0xC0,
    // Col03: wireless radio control, report 0x34 plus seven padding bits.
    0x05, 0x01, 0x09, 0x0C, 0xA1, 0x01, 0x85, 0x34, 0x09, 0xC6, 0x15, 0x00, 0x25, 0x01, 0x35, 0x00,
    0x45, 0x01, 0x65, 0x00, 0x55, 0x00, 0x75, 0x01, 0x95, 0x01, 0x81, 0x06, 0x95, 0x07, 0x81, 0x03,
    0xC0, // Col04: vendor feature-only report 0x5a, deliberately not tabled.
    0x06, 0x83, 0xFF, 0x09, 0x80, 0xA1, 0x01, 0x85, 0x5A, 0x15, 0x00, 0x25, 0x01, 0x35, 0x00, 0x45,
    0x01, 0x65, 0x00, 0x55, 0x00, 0x75, 0x01, 0x95, 0x80, 0xB1, 0x03, 0xC0,
    // Col05: system controller, report 0x35 plus five padding bits.
    0x05, 0x01, 0x09, 0x80, 0xA1, 0x01, 0x85, 0x35, 0x19, 0x81, 0x29, 0x83, 0x15, 0x00, 0x25, 0x01,
    0x35, 0x00, 0x45, 0x01, 0x65, 0x00, 0x55, 0x00, 0x75, 0x01, 0x95, 0x03, 0x81, 0x02, 0x95, 0x05,
    0x81, 0x03, 0xC0,
];

/// A precision-touchpad style digitizer: report id 1, confidence and tip
/// bits, a contact id, absolute 16-bit X and Y, and a contact count.
const TOUCHPAD_DESC: &[u8] = &[
    0x05, 0x0D, // Usage Page (Digitizer)
    0x09, 0x05, // Usage (Touch Pad)
    0xA1, 0x01, // Collection (Application)
    0x85, 0x01, //   Report ID (1)
    0x09, 0x22, //   Usage (Finger)
    0xA1, 0x02, //   Collection (Logical)
    0x09, 0x47, //     Usage (Confidence)
    0x09, 0x42, //     Usage (Tip Switch)
    0x15, 0x00, //     Logical Minimum (0)
    0x25, 0x01, //     Logical Maximum (1)
    0x75, 0x01, //     Report Size (1)
    0x95, 0x02, //     Report Count (2)
    0x81, 0x02, //     Input (Data,Var,Abs)
    0x95, 0x06, //     Report Count (6)
    0x81, 0x03, //     Input (Const) — padding
    0x75, 0x08, //     Report Size (8)
    0x09, 0x51, //     Usage (Contact Identifier)
    0x95, 0x01, //     Report Count (1)
    0x25, 0x7F, //     Logical Maximum (127)
    0x81, 0x02, //     Input (Data,Var,Abs)
    0x05, 0x01, //     Usage Page (Generic Desktop)
    0x26, 0xFF, 0x0F, // Logical Maximum (4095)
    0x75, 0x10, //     Report Size (16)
    0x09, 0x30, //     Usage (X)
    0x09, 0x31, //     Usage (Y)
    0x95, 0x02, //     Report Count (2)
    0x81, 0x02, //     Input (Data,Var,Abs)
    0xC0, //   End Collection
    0x05, 0x0D, //   Usage Page (Digitizer)
    0x09, 0x54, //   Usage (Contact Count)
    0x95, 0x01, //   Report Count (1)
    0x75, 0x08, //   Report Size (8)
    0x25, 0x7F, //   Logical Maximum (127)
    0x81, 0x02, //   Input (Data,Var,Abs)
    0xC0, // End Collection
];

/// A two-finger touchscreen: the same Finger collection twice, then a
/// contact count. The shape every multi-touch device actually publishes.
const TWO_FINGER_DESC: &[u8] = &[
    0x05, 0x0D, // Usage Page (Digitizer)
    0x09, 0x04, //   Usage (Touch Screen)
    0xA1, 0x01, // Collection (Application)
    0x85, 0x01, //   Report ID (1)
    // ── finger 0 ──
    0x09, 0x22, //   Usage (Finger)
    0xA1, 0x02, //   Collection (Logical)
    0x09, 0x42, //     Usage (Tip Switch)
    0x15, 0x00, 0x25, 0x01, 0x75, 0x01, 0x95, 0x01, 0x81, 0x02, 0x95, 0x07, 0x81,
    0x03, //     7 bits padding
    0x09, 0x51, //     Usage (Contact Identifier)
    0x75, 0x08, 0x95, 0x01, 0x25, 0x7F, 0x81, 0x02, 0x05,
    0x01, //     Usage Page (Generic Desktop)
    0x26, 0xFF, 0x0F, // Logical Maximum (4095)
    0x75, 0x10, 0x09, 0x30, 0x09, 0x31, 0x95, 0x02, 0x81, 0x02, 0xC0, //   End Collection
    // ── finger 1 ──
    0x05, 0x0D, 0x09, 0x22, 0xA1, 0x02, 0x09, 0x42, 0x15, 0x00, 0x25, 0x01, 0x75, 0x01, 0x95, 0x01,
    0x81, 0x02, 0x95, 0x07, 0x81, 0x03, 0x09, 0x51, 0x75, 0x08, 0x95, 0x01, 0x25, 0x7F, 0x81, 0x02,
    0x05, 0x01, 0x26, 0xFF, 0x0F, 0x75, 0x10, 0x09, 0x30, 0x09, 0x31, 0x95, 0x02, 0x81, 0x02,
    0xC0, //   End Collection
    0x05, 0x0D, //   Usage Page (Digitizer)
    0x09, 0x54, //   Usage (Contact Count)
    0x95, 0x01, 0x75, 0x08, 0x25, 0x7F, 0x81, 0x02, 0xC0, // End Collection
];

// ─────────────────────────── bit extraction ───────────────────────────

#[test]
fn bits_are_extracted_lsb_first_within_each_byte() {
    // HID packs fields little-endian at the bit level. Getting this
    // backwards decodes every multi-bit field to a mirrored value.
    let data = [0b1010_0101, 0b0000_1111];
    assert_eq!(extract(&data, 0, 1), 1);
    assert_eq!(extract(&data, 1, 1), 0);
    assert_eq!(extract(&data, 0, 4), 0b0101);
    assert_eq!(extract(&data, 4, 4), 0b1010);
    assert_eq!(extract(&data, 8, 8), 0b0000_1111);
    // A field straddling a byte boundary.
    assert_eq!(extract(&data, 6, 4), 0b1110);
    assert_eq!(extract(&data, 0, 16), 0x0FA5);
}

#[test]
fn reading_past_the_end_yields_zero_rather_than_panicking() {
    // A device is free to send a report shorter than its descriptor
    // promised; a driver that panicked on one could be crashed by its
    // own peripheral.
    let data = [0xff];
    assert_eq!(extract(&data, 0, 8), 0xff);
    assert_eq!(extract(&data, 8, 8), 0);
    assert_eq!(extract(&data, 4, 8), 0x0f);
    assert_eq!(extract(&[], 0, 8), 0);
}

#[test]
fn sign_extension_covers_every_width_the_table_admits() {
    assert_eq!(sign_extend(0x7f, 8), 127);
    assert_eq!(sign_extend(0x80, 8), -128);
    assert_eq!(sign_extend(0xff, 8), -1);
    assert_eq!(sign_extend(0x0f, 4), -1);
    assert_eq!(sign_extend(0x07, 4), 7);
    assert_eq!(sign_extend(1, 1), -1);
    assert_eq!(sign_extend(0, 1), 0);
    assert_eq!(sign_extend(0x8000, 16), -32768);
    assert_eq!(sign_extend(0xffff_ffff, 32), -1);
    assert_eq!(sign_extend(5, 0), 5, "a zero-width field is not extended");
}

// ─────────────────────────── decode equivalence ───────────────────────────

#[cfg(feature = "alloc")]
mod decode {
    use super::*;
    use crate::parse::build_table;

    /// Stack-only descriptor assembly for synthetic composites. These are
    /// protocol-shaped fixtures, not copies of a user's firmware dump.
    struct DescriptorBytes {
        bytes: [u8; 512],
        len: usize,
    }

    impl DescriptorBytes {
        fn new() -> Self {
            Self {
                bytes: [0; 512],
                len: 0,
            }
        }

        fn push(&mut self, bytes: &[u8]) {
            self.bytes[self.len..self.len + bytes.len()].copy_from_slice(bytes);
            self.len += bytes.len();
        }

        fn as_slice(&self) -> &[u8] {
            &self.bytes[..self.len]
        }
    }

    fn composite_with_large_vendor_reports() -> DescriptorBytes {
        let mut descriptor = DescriptorBytes::new();
        // ID 2: two buttons, padding, relative X/Y = four fields, four
        // wire bytes including ID. The button state itself is absolute.
        descriptor.push(&[
            0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x85, 0x02, 0x05, 0x09, 0x19, 0x01, 0x29, 0x02,
            0x15, 0x00, 0x25, 0x01, 0x75, 0x01, 0x95, 0x02, 0x81, 0x02, 0x75, 0x06, 0x95, 0x01,
            0x81, 0x03, 0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x15, 0x81, 0x25, 0x7f, 0x75, 0x08,
            0x95, 0x02, 0x81, 0x06, 0xc0,
        ]);
        // ID 3: five contacts (confidence, tip, 3-bit ID, X, Y), followed
        // by scan time, count, and a global button = 28 fields, 30 bytes.
        descriptor.push(&[0x05, 0x0d, 0x09, 0x05, 0xa1, 0x01, 0x85, 0x03]);
        for _ in 0..5 {
            descriptor.push(&[
                0x05, 0x0d, 0x09, 0x22, 0xa1, 0x02, 0x15, 0x00, 0x25, 0x01, 0x75, 0x01, 0x95, 0x02,
                0x09, 0x47, 0x09, 0x42, 0x81, 0x02, 0x09, 0x51, 0x25, 0x07, 0x75, 0x03, 0x95, 0x01,
                0x81, 0x02, 0x75, 0x03, 0x81, 0x03, 0x05, 0x01, 0x26, 0xff, 0x0f, 0x75, 0x10, 0x95,
                0x02, 0x09, 0x30, 0x09, 0x31, 0x81, 0x02, 0xc0,
            ]);
        }
        descriptor.push(&[
            0x05, 0x0d, 0x09, 0x56, 0x26, 0xff, 0x7f, 0x75, 0x10, 0x95, 0x01, 0x81, 0x02, 0x09,
            0x54, 0x25, 0x05, 0x75, 0x08, 0x81, 0x02, 0x05, 0x09, 0x09, 0x01, 0x25, 0x01, 0x75,
            0x01, 0x81, 0x02, 0x75, 0x07, 0x81, 0x03, 0xc0,
        ]);
        // IDs 11/12: 61 repeated vendor variables each. The descriptor
        // totals 154 nonconstant fields, but only 32 have extractors.
        for id in [11, 12] {
            descriptor.push(&[
                0x06, 0x00, 0xff, 0x09, 0x01, 0xa1, 0x01, 0x85, id, 0x09, 0x01, 0x15, 0x00, 0x26,
                0xff, 0x00, 0x75, 0x08, 0x95, 0x3d, 0x81, 0x02, 0xc0,
            ]);
        }
        descriptor
    }

    #[test]
    fn a_154_field_composite_retains_only_the_32_fields_it_extracts() {
        use hidreport::{Field as HidField, Report, ReportDescriptor};
        use kore_abi::pointer::{self, Record};

        let descriptor = composite_with_large_vendor_reports();
        let parsed = ReportDescriptor::try_from(descriptor.as_slice()).unwrap();
        let raw_fields: usize = parsed
            .input_reports()
            .iter()
            .map(|report| {
                report
                    .fields()
                    .iter()
                    .filter(|field| matches!(field, HidField::Variable(_)))
                    .count()
            })
            .sum();
        assert_eq!(raw_fields, 154);
        assert!(raw_fields > MAX_FIELDS);

        let table = build_table(descriptor.as_slice()).unwrap();
        assert_eq!(table.fields().len(), 32);
        assert_eq!(table.reports().len(), 2);
        for (id, bytes, fields, shape) in [
            (2, 4, 4, ReportShape::Mouse),
            (3, 30, 28, ReportShape::Contact),
        ] {
            let entry = table.report(id).unwrap();
            assert_eq!(
                (entry.input_bytes, entry.count, entry.shape),
                (bytes, fields, shape)
            );
        }
        assert_eq!(table.report(3).unwrap().slots, 5);
        assert_eq!(
            decode1(&table, &[2, 1, 5, (-3i8) as u8], 0x41),
            Some(InputReport::mouse(0x41, 1, 5, -3, 0))
        );

        let mut raw = [0; 30];
        raw[0] = 3;
        for slot in 0..5 {
            let start = 1 + slot * 5;
            raw[start] = 3 | ((slot as u8) << 2);
            raw[start + 1..start + 3].copy_from_slice(&(100 + slot as u16).to_le_bytes());
            raw[start + 3..start + 5].copy_from_slice(&(200 + slot as u16).to_le_bytes());
        }
        raw[26..28].copy_from_slice(&0x1234u16.to_le_bytes());
        raw[28] = 5;
        raw[29] = 1;
        let mut legacy = [InputReport::mouse(0, 0, 0, 0, 0); MAX_CONTACTS];
        assert_eq!(table.decode_into(&raw, 0x41, &mut legacy), 5);
        for (slot, contact) in legacy[..5].iter().enumerate() {
            assert_eq!(
                (contact.contact_x(), contact.contact_y()),
                (100 + slot as u16, 200 + slot as u16)
            );
        }
        let mut out = [InputReport::mouse(0, 0, 0, 0, 0); pointer::MAX_FRAME_RECORDS];
        let n = table
            .decode_framed_into(&raw, 128, 1, 9, Some(123), &mut out)
            .unwrap();
        for slot in 0..5 {
            assert!(out[..n].iter().any(|record| Record::decode(*record)
                == Ok(Record::Contact {
                    id: slot,
                    x: 100 + slot as i32,
                    y: 200 + slot as i32,
                    touching: true,
                    confidence: Some(true),
                    in_range: None,
                })));
        }
        assert!(
            out[..n]
                .iter()
                .any(|record| Record::decode(*record) == Ok(Record::ScanTime(0x1234)))
        );
        assert!(
            out[..n]
                .iter()
                .any(|record| Record::decode(*record) == Ok(Record::Buttons(1)))
        );
        assert!(matches!(
            Record::decode(out[n - 1]),
            Ok(Record::End {
                complete: true,
                reported_contacts: Some(5),
                ..
            })
        ));
    }

    #[test]
    fn ignored_vendor_reports_keep_exact_lengths_without_spending_stream_ids() {
        use crate::framed::DecodeError;
        use kore_abi::pointer;

        let table = build_table(composite_with_large_vendor_reports().as_slice()).unwrap();
        let sentinel = InputReport::keyboard(0x42, 2, [4, 5, 0, 0, 0, 0]);
        for id in [11, 12] {
            let mut raw = [0xa5; 63];
            raw[0] = id;
            assert_eq!(table.stream_index(&raw), None);
            for len in [1, 61, 62, 63] {
                let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
                assert_eq!(table.decode_into(&raw[..len], 128, &mut out), 0);
                assert_eq!(
                    table.decode_framed_into(&raw[..len], 128, 1, 0, None, &mut out),
                    if len == 62 {
                        Ok(0)
                    } else {
                        Err(DecodeError::Truncated)
                    }
                );
                assert_eq!(out, [sentinel; pointer::MAX_FRAME_RECORDS]);
            }
        }
    }

    #[test]
    fn filtering_vendor_fields_does_not_rebase_offsets_or_shorten_mixed_reports() {
        use crate::framed::DecodeError;
        use kore_abi::pointer::{self, Record};

        // ID, three vendor bytes, two mouse deltas, two more vendor bytes.
        let descriptor = [
            0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x85, 0x21, 0x06, 0x00, 0xff, 0x09, 0x01, 0x15,
            0x00, 0x26, 0xff, 0x00, 0x75, 0x08, 0x95, 0x03, 0x81, 0x02, 0x05, 0x01, 0x09, 0x30,
            0x09, 0x31, 0x15, 0x81, 0x25, 0x7f, 0x95, 0x02, 0x81, 0x06, 0x06, 0x00, 0xff, 0x09,
            0x02, 0x81, 0x02, 0xc0,
        ];
        let table = build_table(&descriptor).unwrap();
        assert_eq!(table.fields().len(), 2);
        assert_eq!(
            (table.fields()[0].bit_offset, table.fields()[1].bit_offset),
            (32, 40)
        );
        assert_eq!(table.report(0x21).unwrap().input_bytes, 8);
        let raw = [0x21, 0xee, 0xee, 0xee, 5, (-3i8) as u8, 0xab, 0xcd, 0xff];
        let sentinel = InputReport::keyboard(0x42, 2, [4, 5, 0, 0, 0, 0]);
        for len in [1, 6, 7, 9] {
            let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
            assert_eq!(table.decode_into(&raw[..len], 128, &mut out), 0);
            assert_eq!(
                table.decode_framed_into(&raw[..len], 128, 1, 0, None, &mut out),
                Err(DecodeError::Truncated)
            );
            assert_eq!(out, [sentinel; pointer::MAX_FRAME_RECORDS]);
        }
        assert_eq!(
            decode1(&table, &raw[..8], 0x41),
            Some(InputReport::mouse(0x41, 0, 5, -3, 0))
        );
        let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
        let n = table
            .decode_framed_into(&raw[..8], 128, 1, 0, None, &mut out)
            .unwrap();
        assert!(
            out[..n]
                .iter()
                .any(|record| Record::decode(*record) == Ok(Record::Motion { dx: 5, dy: -3 }))
        );
    }

    fn one_field_descriptor(page: u16, width: u16, count: u16, flags: u8) -> DescriptorBytes {
        let mut descriptor = DescriptorBytes::new();
        descriptor.push(&[
            0x06,
            page as u8,
            (page >> 8) as u8,
            0x09,
            0x01,
            0xa1,
            0x01,
            0x09,
            0x01,
            0x15,
            0x00,
            0x26,
            0xff,
            0x00,
            0x76,
            width as u8,
            (width >> 8) as u8,
            0x96,
            count as u8,
            (count >> 8) as u8,
            0x81,
            flags,
            0xc0,
        ]);
        descriptor
    }

    #[test]
    fn ignored_variable_and_array_fields_still_validate_widths() {
        use crate::parse::ParseError;
        use crate::table::TableError;

        for flags in [0x00, 0x02] {
            for page in [0xff00, page::GENERIC_DESKTOP] {
                let too_wide = one_field_descriptor(page, 33, 1, flags);
                assert_eq!(
                    build_table(too_wide.as_slice()),
                    Err(ParseError::Table(TableError::FieldTooWide(33)))
                );
                let wrapping = one_field_descriptor(page, 512, 1, flags);
                assert_eq!(
                    build_table(wrapping.as_slice()),
                    Err(ParseError::FieldOutOfRange { start: 0, len: 512 })
                );
            }
        }
    }

    #[test]
    fn ignored_variable_and_array_fields_still_validate_offsets() {
        use crate::parse::ParseError;

        for flags in [0x00, 0x02] {
            let mut descriptor = DescriptorBytes::new();
            // 8192 bytes of constant padding put the following vendor
            // field just past the compact table's largest bit offset.
            descriptor.push(&[
                0x06, 0x00, 0xff, 0x09, 0x01, 0xa1, 0x01, 0x15, 0x00, 0x26, 0xff, 0x00, 0x75, 0x08,
                0x96, 0x00, 0x20, 0x81, 0x03, 0x09, 0x01, 0x95, 0x01, 0x81, flags, 0xc0,
            ]);
            assert_eq!(
                build_table(descriptor.as_slice()),
                Err(ParseError::FieldOutOfRange {
                    start: 65_536,
                    len: 8
                })
            );
        }
    }

    #[test]
    fn filtering_does_not_relax_retained_field_or_report_limits() {
        use crate::parse::ParseError;
        use crate::table::TableError;

        // Keyboard array slots are all extracted, so their report count
        // must still consume exactly that many field-table slots.
        let fits = one_field_descriptor(page::KEYBOARD, 8, MAX_FIELDS as u16, 0);
        assert_eq!(
            build_table(fits.as_slice()).unwrap().fields().len(),
            MAX_FIELDS
        );
        let exceeds = one_field_descriptor(page::KEYBOARD, 8, MAX_FIELDS as u16 + 1, 0);
        assert_eq!(
            build_table(exceeds.as_slice()),
            Err(ParseError::Table(TableError::TooManyFields))
        );

        let mut descriptor = DescriptorBytes::new();
        descriptor.push(&[
            0x06, 0x09, 0x00, 0x09, 0x01, 0xa1, 0x01, 0x15, 0x00, 0x25, 0x7f, 0x75, 0x08, 0x95,
            0x01,
        ]);
        for id in 1..=MAX_REPORTS as u8 + 1 {
            descriptor.push(&[0x85, id, 0x09, 0x01, 0x81, 0x02]);
        }
        descriptor.push(&[0xc0]);
        assert_eq!(
            build_table(descriptor.as_slice()),
            Err(ParseError::Table(TableError::TooManyReports))
        );
    }

    #[test]
    fn an_ignored_report_still_must_fit_the_wire_size_representation() {
        use crate::parse::ParseError;

        // A constant-only report has no extracted fields, but its 65536
        // bytes still exceed the compact report entry's u16 wire length.
        let descriptor = [
            0x06, 0x00, 0xff, 0x09, 0x01, 0xa1, 0x01, 0x15, 0x00, 0x26, 0xff, 0x00, 0x75, 0x08,
            0x97, 0x00, 0x00, 0x01, 0x00, 0x81, 0x03, 0xc0,
        ];
        assert_eq!(
            build_table(&descriptor),
            Err(ParseError::ReportSizeOutOfRange(65_536))
        );
    }

    #[test]
    fn absolute_or_mixed_mouse_and_vendor_axes_never_publish_relative_motion() {
        use kore_abi::pointer;

        for application in [
            &[0x05, 0x01, 0x09, 0x02][..],
            &[0x06, 0x11, 0xff, 0x09, 0x01][..],
        ] {
            for relative in [[false, false], [true, false], [false, true]] {
                let mut descriptor = DescriptorBytes::new();
                descriptor.push(application);
                descriptor.push(&[
                    0xa1, 0x01, 0x05, 0x01, 0x15, 0x00, 0x26, 0xff, 0x7f, 0x75, 0x10, 0x95, 0x01,
                ]);
                for (axis, relative) in relative.into_iter().enumerate() {
                    descriptor.push(&[
                        0x09,
                        0x30 + axis as u8,
                        0x81,
                        if relative { 0x06 } else { 0x02 },
                    ]);
                }
                descriptor.push(&[0xc0]);
                let table = build_table(descriptor.as_slice()).unwrap();
                assert_eq!(table.fields().len(), 2);
                assert_eq!(
                    [table.fields()[0].relative, table.fields()[1].relative],
                    relative
                );
                assert_eq!(table.reports()[0].shape, ReportShape::Unknown);
                let sentinel = InputReport::keyboard(0x42, 2, [4, 5, 0, 0, 0, 0]);
                let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
                let raw = [0xff, 0x7f, 0xff, 0x7f];
                assert_eq!(table.decode_into(&raw, 128, &mut out), 0);
                assert_eq!(
                    table.decode_framed_into(&raw, 128, 1, 0, None, &mut out),
                    Ok(0)
                );
                assert_eq!(out, [sentinel; pointer::MAX_FRAME_RECORDS]);
            }
        }
    }

    #[test]
    fn an_absolute_wheel_is_not_a_relative_scroll_fallback() {
        use kore_abi::pointer::{self, Record};

        let mut descriptor = [0; MOUSE_DESC.len() + 6];
        let tail = MOUSE_DESC.len() - 2;
        descriptor[..tail].copy_from_slice(&MOUSE_DESC[..tail]);
        descriptor[tail..tail + 6].copy_from_slice(&[0x09, 0x38, 0x95, 0x01, 0x81, 0x02]);
        descriptor[tail + 6..].copy_from_slice(&MOUSE_DESC[tail..]);
        let table = build_table(&descriptor).unwrap();
        let raw = [1, 5, (-3i8) as u8, 100];
        assert_eq!(
            decode1(&table, &raw, 0x41),
            Some(InputReport::mouse(0x41, 1, 5, -3, 0))
        );
        let mut out = [InputReport::mouse(0, 0, 0, 0, 0); pointer::MAX_FRAME_RECORDS];
        let n = table
            .decode_framed_into(&raw, 128, 1, 0, None, &mut out)
            .unwrap();
        assert!(
            out[..n]
                .iter()
                .any(|record| Record::decode(*record) == Ok(Record::Motion { dx: 5, dy: -3 }))
        );
        assert!(
            !out[..n]
                .iter()
                .any(|record| matches!(Record::decode(*record), Ok(Record::Scroll { .. })))
        );
    }

    #[test]
    fn a_contact_collection_cannot_turn_relative_axes_into_absolute_contacts() {
        use kore_abi::pointer;

        let mut descriptor = [0; TOUCHPAD_DESC.len()];
        descriptor.copy_from_slice(TOUCHPAD_DESC);
        // Find the descriptor's sole 16-bit, two-axis Input item.
        let axis_input = descriptor
            .windows(10)
            .position(|bytes| bytes == [0x75, 0x10, 0x09, 0x30, 0x09, 0x31, 0x95, 0x02, 0x81, 0x02])
            .unwrap();
        descriptor[axis_input + 9] = 0x06;
        let table = build_table(&descriptor).unwrap();
        assert_eq!(table.report(1).unwrap().shape, ReportShape::Unknown);
        let raw = [1, 3, 7, 100, 0, 200, 0, 1];
        let sentinel = InputReport::keyboard(0x42, 2, [4, 5, 0, 0, 0, 0]);
        let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
        assert_eq!(table.decode_into(&raw, 128, &mut out), 0);
        assert_eq!(
            table.decode_framed_into(&raw, 128, 1, 0, None, &mut out),
            Ok(0)
        );
        assert_eq!(out, [sentinel; pointer::MAX_FRAME_RECORDS]);
    }

    #[test]
    fn framed_decode_keeps_calibration_contact_attributes_and_scan_time() {
        use kore_abi::pointer::{self, Record};
        // HID 1.11 linear centimeters: 10000 logical units / 100 cm =
        // 10 units/mm. Pressure/shape retain raw logical units.
        let descriptor = [
            0x05, 0x0d, 0x09, 0x05, 0xa1, 0x01, 0x85, 0x01, 0x09, 0x22, 0xa1, 0x02, 0x09, 0x42,
            0x15, 0x00, 0x25, 0x01, 0x75, 0x08, 0x95, 0x01, 0x81, 0x02, 0x09, 0x51, 0x26, 0xff,
            0x7f, 0x75, 0x10, 0x81, 0x02, 0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x26, 0x10, 0x27,
            0x35, 0x00, 0x46, 0x64, 0x00, 0x55, 0x00, 0x65, 0x11, 0x75, 0x10, 0x95, 0x02, 0x81,
            0x02, 0x05, 0x0d, 0x09, 0x30, 0x09, 0x48, 0x09, 0x49, 0x09, 0x3f, 0x45, 0x00, 0x65,
            0x00, 0x95, 0x04, 0x81, 0x02, 0xc0, 0x09, 0x54, 0x75, 0x08, 0x95, 0x01, 0x81, 0x02,
            0x09, 0x56, 0x75, 0x10, 0x81, 0x02, 0xc0,
        ];
        let table = build_table(&descriptor).unwrap();
        assert_eq!(
            table.reports()[0].slots,
            1,
            "scan time must not create another contact"
        );
        let mut raw = [
            1, 1, 0x34, 0x12, 100, 0, 200, 0, 123, 0, 45, 0, 67, 0, 89, 0, 1, 0xe7, 3,
        ];
        let mut out = [InputReport::mouse(0, 0, 0, 0, 0); pointer::MAX_FRAME_RECORDS];
        let n = table
            .decode_framed_into(&raw, 128, 7, 42, Some(123_456_789), &mut out)
            .unwrap();
        let has = |wanted| out[..n].iter().any(|r| Record::decode(*r) == Ok(wanted));
        assert!(has(Record::Resolution {
            axis: 0,
            units: 10,
            millimeters: 1
        }));
        assert!(has(Record::Resolution {
            axis: 1,
            units: 10,
            millimeters: 1
        }));
        assert!(has(Record::Contact {
            id: 0x1234,
            x: 100,
            y: 200,
            touching: true,
            confidence: None,
            in_range: None
        }));
        for (attribute, value) in [
            (pointer::PRESSURE, 123),
            (pointer::WIDTH, 45),
            (pointer::HEIGHT, 67),
            (pointer::ORIENTATION, 89),
        ] {
            assert!(has(Record::Attribute {
                id: 0x1234,
                attribute,
                value
            }));
        }
        assert!(has(Record::ScanTime(999)));
        assert_eq!(
            Record::decode(out[n - 1]),
            Ok(Record::End {
                sequence: 42,
                generation: 7,
                records: (n - 2) as u16,
                reported_contacts: Some(1),
                complete: true
            })
        );
        assert_eq!(
            table.decode_framed_into(&raw[..raw.len() - 1], 128, 7, 43, None, &mut out),
            Err(crate::framed::DecodeError::Truncated)
        );
        assert_eq!(
            table.decode_framed_into(&raw, 128, 7, 43, None, &mut out[..4]),
            Err(crate::framed::DecodeError::Capacity)
        );
        raw[16] = 2;
        let n = table
            .decode_framed_into(&raw, 128, 7, 43, None, &mut out)
            .unwrap();
        assert!(matches!(
            Record::decode(out[n - 1]),
            Ok(Record::End {
                complete: false,
                reported_contacts: Some(2),
                ..
            })
        ));
        raw[16] = 0;
        let n = table
            .decode_framed_into(&raw, 128, 7, 44, None, &mut out)
            .unwrap();
        assert!(
            matches!(
                Record::decode(out[n - 1]),
                Ok(Record::End {
                    complete: false,
                    ..
                })
            ),
            "hybrid continuation is not a full snapshot"
        );
        raw[1] = 0;
        let n = table
            .decode_framed_into(&raw, 128, 7, 45, None, &mut out)
            .unwrap();
        assert!(matches!(
            Record::decode(out[n - 1]),
            Ok(Record::End {
                complete: true,
                reported_contacts: Some(0),
                ..
            })
        ));
        assert!(
            !out[..n]
                .iter()
                .any(|r| matches!(Record::decode(*r), Ok(Record::Contact { .. })))
        );
    }

    #[test]
    fn framed_mouse_deltas_are_not_clamped_to_boot_protocol_width() {
        use kore_abi::pointer::{self, Record};
        let descriptor = [
            0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x09, 0x30, 0x09, 0x31, 0x17, 0x00, 0x00, 0x00,
            0x80, 0x27, 0xff, 0xff, 0xff, 0x7f, 0x75, 0x20, 0x95, 0x02, 0x81, 0x06, 0xc0,
        ];
        let table = build_table(&descriptor).unwrap();
        let mut raw = [0; 8];
        raw[..4].copy_from_slice(&100_000i32.to_le_bytes());
        raw[4..].copy_from_slice(&(-100_000i32).to_le_bytes());
        let mut out = [InputReport::mouse(0, 0, 0, 0, 0); pointer::MAX_FRAME_RECORDS];
        let n = table
            .decode_framed_into(&raw, 128, 1, 0, None, &mut out)
            .unwrap();
        assert!(out[..n].iter().any(|r| Record::decode(*r)
            == Ok(Record::Motion {
                dx: 100_000,
                dy: -100_000
            })));
    }

    #[test]
    fn framed_mouse_requires_the_complete_descriptor_length_including_padding() {
        use crate::framed::DecodeError;
        use kore_abi::pointer::{self, Record};

        // The last constant byte is not an extracted field. Field bounds alone
        // would accept a report missing it, as well as reports with extra bytes.
        let mut descriptor = [0; MOUSE_DESC.len() + 6];
        let tail = MOUSE_DESC.len() - 2;
        descriptor[..tail].copy_from_slice(&MOUSE_DESC[..tail]);
        descriptor[tail..tail + 6].copy_from_slice(&[0x75, 0x08, 0x95, 0x01, 0x81, 0x03]);
        descriptor[tail + 6..].copy_from_slice(&MOUSE_DESC[tail..]);
        let table = build_table(&descriptor).unwrap();
        assert!(!table.uses_report_ids());
        assert_eq!(table.reports()[0].input_bytes, 4);
        let raw = [1, 5, (-3i8) as u8, 0xa5, 0xff];
        let sentinel = InputReport::keyboard(0x42, 2, [4, 5, 0, 0, 0, 0]);

        for data in [&raw[..0], &raw[..1], &raw[..3], &raw[..5]] {
            let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
            assert_eq!(
                table.decode_framed_into(data, 128, 7, 42, None, &mut out),
                Err(DecodeError::Truncated)
            );
            assert_eq!(out, [sentinel; pointer::MAX_FRAME_RECORDS]);
        }

        // Constant padding need not be zero; its presence, not its value, is
        // the framing requirement. The normal mouse frame remains unchanged.
        let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
        let n = table
            .decode_framed_into(&raw[..4], 128, 7, 42, None, &mut out)
            .unwrap();
        let expected = [
            Record::Begin {
                sequence: 42,
                received_ns: None,
                class: 0,
            },
            Record::Motion { dx: 5, dy: -3 },
            Record::Buttons(1),
            Record::End {
                sequence: 42,
                generation: 7,
                records: 2,
                reported_contacts: None,
                complete: true,
            },
        ]
        .map(|record| record.encode(128));
        assert_eq!(&out[..n], &expected);
        assert!(out[n..].iter().all(|report| *report == sentinel));
    }

    #[test]
    fn framed_touchpad_checks_report_id_and_padding_before_writing_records() {
        use crate::framed::DecodeError;
        use kore_abi::pointer::{self, Record};

        let mut descriptor = [0; TOUCHPAD_DESC.len() + 6];
        let tail = TOUCHPAD_DESC.len() - 1;
        descriptor[..tail].copy_from_slice(&TOUCHPAD_DESC[..tail]);
        descriptor[tail..tail + 6].copy_from_slice(&[0x75, 0x08, 0x95, 0x01, 0x81, 0x03]);
        descriptor[tail + 6..].copy_from_slice(&TOUCHPAD_DESC[tail..]);
        let table = build_table(&descriptor).unwrap();
        assert!(table.uses_report_ids());
        assert_eq!(table.report(1).unwrap().input_bytes, 9);
        let raw = [1, 3, 7, 100, 0, 200, 0, 1, 0xa5, 0xff];
        let sentinel = InputReport::keyboard(0x42, 2, [4, 5, 0, 0, 0, 0]);

        // A known report ID with too few/many bytes, including a missing
        // trailing constant byte, must not manufacture a contact or release.
        for data in [&raw[..1], &raw[..7], &raw[..8], &raw[..10]] {
            let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
            assert_eq!(
                table.decode_framed_into(data, 128, 7, 42, None, &mut out),
                Err(DecodeError::Truncated)
            );
            assert_eq!(out, [sentinel; pointer::MAX_FRAME_RECORDS]);
        }

        let mut unknown_id = raw;
        unknown_id[0] = 0x7f;
        // Empty input, an absent ID (the flags become unknown ID 3), and an
        // unknown ID do not select any stream and must publish nothing.
        for data in [&raw[..0], &raw[1..9], &unknown_id[..9]] {
            let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
            assert_eq!(
                table.decode_framed_into(data, 128, 7, 42, None, &mut out),
                Ok(0)
            );
            assert_eq!(out, [sentinel; pointer::MAX_FRAME_RECORDS]);
        }

        let mut out = [sentinel; pointer::MAX_FRAME_RECORDS];
        let n = table
            .decode_framed_into(&raw[..9], 128, 7, 42, None, &mut out)
            .unwrap();
        let expected = [
            Record::Begin {
                sequence: 42,
                received_ns: None,
                class: 1,
            },
            Record::Axis {
                axis: 0,
                min: 0,
                max: 4095,
            },
            Record::Axis {
                axis: 1,
                min: 0,
                max: 4095,
            },
            Record::Contact {
                id: 7,
                x: 100,
                y: 200,
                touching: true,
                confidence: Some(true),
                in_range: None,
            },
            Record::Buttons(0),
            Record::End {
                sequence: 42,
                generation: 7,
                records: 4,
                reported_contacts: Some(1),
                complete: true,
            },
        ]
        .map(|record| record.encode(128));
        assert_eq!(&out[..n], &expected);
        assert!(out[n..].iter().all(|report| *report == sentinel));
    }

    #[test]
    fn a_boot_mouse_descriptor_parses_once_and_decodes_movement() {
        let table = build_table(MOUSE_DESC).expect("parse");
        assert!(!table.uses_report_ids(), "the boot mouse uses no ids");
        assert_eq!(table.reports().len(), 1);
        assert_eq!(table.reports()[0].shape, ReportShape::Mouse);

        // buttons = left, X = +5, Y = -3.
        let report = [0b0000_0001, 5, (-3i8) as u8];
        assert_eq!(
            decode1(&table, &report, 0x41),
            Some(InputReport::mouse(0x41, 0b001, 5, -3, 0))
        );

        // No buttons, full-scale negative movement — the sign path.
        let report = [0, (-127i8) as u8, (-127i8) as u8];
        assert_eq!(
            decode1(&table, &report, 0x41),
            Some(InputReport::mouse(0x41, 0, -127, -127, 0))
        );

        // Three buttons at once, no movement.
        let report = [0b0000_0111, 0, 0];
        assert_eq!(
            decode1(&table, &report, 0x41),
            Some(InputReport::mouse(0x41, 0b111, 0, 0, 0))
        );
    }

    #[test]
    fn a_boot_keyboard_descriptor_decodes_modifiers_and_keys() {
        let table = build_table(KEYBOARD_DESC).expect("parse");
        assert_eq!(table.reports()[0].shape, ReportShape::Keyboard);

        // Left shift (bit 1) held, 'a' (0x04) and 'b' (0x05) pressed.
        let report = [0b0000_0010, 0, 0x04, 0x05, 0, 0, 0, 0];
        assert_eq!(
            decode1(&table, &report, 0x42),
            Some(InputReport::keyboard(
                0x42,
                0b0000_0010,
                [0x04, 0x05, 0, 0, 0, 0]
            ))
        );

        // Nothing held.
        let report = [0u8; 8];
        assert_eq!(
            decode1(&table, &report, 0x42),
            Some(InputReport::keyboard(0x42, 0, [0; 6]))
        );

        // Full six-key rollover with both control keys.
        let report = [0b0001_0001, 0, 4, 5, 6, 7, 8, 9];
        assert_eq!(
            decode1(&table, &report, 0x42),
            Some(InputReport::keyboard(0x42, 0b0001_0001, [4, 5, 6, 7, 8, 9]))
        );
    }

    #[test]
    fn a_touchpad_descriptor_decodes_an_absolute_contact() {
        let table = build_table(TOUCHPAD_DESC).expect("parse");
        assert!(
            table.uses_report_ids(),
            "the touchpad publishes report id 1"
        );
        assert_eq!(table.report(1).map(|r| r.shape), Some(ReportShape::Contact));

        // id 1 | confidence+tip | contact 0 | X = 0x0123 | Y = 0x0456 | 1 contact
        let report = [1, 0b0000_0011, 0, 0x23, 0x01, 0x56, 0x04, 1];
        let got = decode1(&table, &report, 0x43).expect("decoded");
        assert_eq!(got.kind, InputReport::KIND_CONTACT);
        assert_eq!(got.source, 0x43);
        assert_eq!(got.contact_x(), 0x0123);
        assert_eq!(got.contact_y(), 0x0456);
        assert_eq!(got.contact_id(), 0);
        assert_eq!(got.contact_count(), 1);
        assert_eq!(
            got.contact_axes(),
            Some((InputReport::CONTACT_TOUCHPAD, 4095, 4095))
        );
        assert_ne!(got.state & InputReport::CONTACT_TIP, 0);
        assert_ne!(got.state & InputReport::CONTACT_CONFIDENCE, 0);

        // A finger lifted: tip clear, confidence still set.
        let report = [1, 0b0000_0001, 2, 0xFF, 0x0F, 0x00, 0x00, 1];
        let got = decode1(&table, &report, 0x43).expect("decoded");
        assert_eq!(got.state & InputReport::CONTACT_TIP, 0, "not touching");
        assert_ne!(got.state & InputReport::CONTACT_CONFIDENCE, 0);
        assert_eq!(got.contact_x(), 0x0FFF, "full-scale X");
        assert_eq!(got.contact_y(), 0);
        assert_eq!(got.contact_id(), 2);
    }

    #[test]
    fn cursor_motion_is_recoverable_from_successive_contacts() {
        // The actual unlock: two frames of a finger moving right and down
        // must yield a delta the consumer can turn into cursor motion.
        let table = build_table(TOUCHPAD_DESC).expect("parse");
        let frame = |x: u16, y: u16| {
            let mut r = [1u8, 0b0000_0011, 0, 0, 0, 0, 0, 1];
            r[3..5].copy_from_slice(&x.to_le_bytes());
            r[5..7].copy_from_slice(&y.to_le_bytes());
            decode1(&table, &r, 0x43).expect("decoded")
        };
        let a = frame(100, 200);
        let b = frame(140, 260);
        assert_eq!(b.contact_x() as i32 - a.contact_x() as i32, 40);
        assert_eq!(b.contact_y() as i32 - a.contact_y() as i32, 60);
    }

    #[test]
    fn an_unknown_report_id_is_ignored_rather_than_misdecoded() {
        let table = build_table(TOUCHPAD_DESC).expect("parse");
        // Report id 7 is not in the descriptor. Decoding it as report 1
        // would emit a contact at a garbage coordinate.
        let report = [7, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
        assert_eq!(decode1(&table, &report, 0x43), None);
        // An empty buffer names no report at all.
        assert_eq!(decode1(&table, &[], 0x43), None);
    }

    /// One frame of the two-finger descriptor.
    /// `[id, tip0, id0, x0lo, x0hi, y0lo, y0hi, tip1, id1, x1lo, x1hi, y1lo, y1hi, count]`
    fn frame(f0: (bool, u8, u16, u16), f1: (bool, u8, u16, u16), count: u8) -> [u8; 14] {
        let mut r = [0u8; 14];
        r[0] = 1;
        r[1] = f0.0 as u8;
        r[2] = f0.1;
        r[3..5].copy_from_slice(&f0.2.to_le_bytes());
        r[5..7].copy_from_slice(&f0.3.to_le_bytes());
        r[7] = f1.0 as u8;
        r[8] = f1.1;
        r[9..11].copy_from_slice(&f1.2.to_le_bytes());
        r[11..13].copy_from_slice(&f1.3.to_le_bytes());
        r[13] = count;
        r
    }

    #[test]
    fn two_contacts_in_one_frame_both_decode() {
        // The bug this slot machinery exists to fix: before it, `find`
        // returned the first match for each usage, so a two-finger frame
        // delivered finger 0 and silently dropped finger 1 — while its
        // own contact-count field correctly said two.
        let table = build_table(TWO_FINGER_DESC).expect("parse");
        assert_eq!(table.report(1).map(|r| r.slots), Some(2), "two slots found");

        let r = frame((true, 7, 0x111, 0x222), (true, 9, 0x333, 0x444), 2);
        let mut out = [InputReport::keyboard(0, 0, [0; 6]); MAX_CONTACTS];
        let n = table.decode_into(&r, 0x43, &mut out);

        assert_eq!(n, 2, "both fingers delivered");
        assert_eq!((out[0].contact_x(), out[0].contact_y()), (0x111, 0x222));
        assert_eq!(out[0].contact_id(), 7);
        assert_eq!((out[1].contact_x(), out[1].contact_y()), (0x333, 0x444));
        assert_eq!(out[1].contact_id(), 9);
        for c in &out[..2] {
            assert_eq!(c.kind, InputReport::KIND_CONTACT);
            assert_eq!(c.source, 0x43);
            assert_ne!(c.state & InputReport::CONTACT_TIP, 0);
            assert_eq!(
                c.contact_count(),
                2,
                "the frame's own count rides every record"
            );
        }
    }

    #[test]
    fn the_contact_count_bounds_how_many_slots_are_emitted() {
        // One finger down, but slot 1 still holds last frame's values —
        // the device does not clear stale slots. Emitting slot 1 would
        // replay a finger that is no longer touching.
        let table = build_table(TWO_FINGER_DESC).expect("parse");
        let r = frame((true, 1, 100, 200), (true, 2, 900, 900), 1);
        let mut out = [InputReport::keyboard(0, 0, [0; 6]); MAX_CONTACTS];
        let n = table.decode_into(&r, 0x43, &mut out);

        assert_eq!(n, 1, "the stale second slot is not replayed");
        assert_eq!((out[0].contact_x(), out[0].contact_y()), (100, 200));
    }

    #[test]
    fn a_count_of_zero_still_emits_the_lifted_finger() {
        // All fingers up. If this emitted nothing the consumer would never
        // learn the last finger left the surface, and the cursor would
        // stay stuck to it.
        let table = build_table(TWO_FINGER_DESC).expect("parse");
        let r = frame((false, 1, 100, 200), (false, 0, 0, 0), 0);
        let mut out = [InputReport::keyboard(0, 0, [0; 6]); MAX_CONTACTS];
        let n = table.decode_into(&r, 0x43, &mut out);

        assert_eq!(n, 1, "the lift is reported");
        assert_eq!(out[0].state & InputReport::CONTACT_TIP, 0, "not touching");
        assert_eq!(out[0].contact_count(), 0);
    }

    #[test]
    fn a_caller_buffer_shorter_than_the_frame_is_respected() {
        let table = build_table(TWO_FINGER_DESC).expect("parse");
        let r = frame((true, 1, 10, 20), (true, 2, 30, 40), 2);
        let mut one = [InputReport::keyboard(0, 0, [0; 6]); 1];
        assert_eq!(table.decode_into(&r, 0x43, &mut one), 1);
        assert_eq!((one[0].contact_x(), one[0].contact_y()), (10, 20));
        // A zero-length buffer writes nothing rather than panicking.
        assert_eq!(table.decode_into(&r, 0x43, &mut []), 0);
    }

    #[test]
    fn a_keyboard_array_does_not_read_as_contacts() {
        // The six key-array slots all carry usage 0. A naive "a repeated
        // usage starts a new contact" rule would read them as six
        // contacts and emit six bogus records per keystroke.
        let table = build_table(KEYBOARD_DESC).expect("parse");
        assert_eq!(table.reports()[0].slots, 1);
        assert_eq!(table.reports()[0].shape, ReportShape::Keyboard);
        let report = [0b0000_0010, 0, 0x04, 0x05, 0, 0, 0, 0];
        let mut out = [InputReport::keyboard(0, 0, [0; 6]); MAX_CONTACTS];
        assert_eq!(table.decode_into(&report, 0x42, &mut out), 1);
    }

    #[test]
    fn a_mouse_report_emits_exactly_one_record() {
        let table = build_table(MOUSE_DESC).expect("parse");
        assert_eq!(table.reports()[0].slots, 1);
        let mut out = [InputReport::keyboard(0, 0, [0; 6]); MAX_CONTACTS];
        assert_eq!(table.decode_into(&[0b1, 5, 251], 0x41, &mut out), 1);
        assert_eq!(out[0], InputReport::mouse(0x41, 1, 5, -5, 0));
    }

    #[test]
    fn a_single_contact_descriptor_still_reports_one_slot() {
        // The touchpad descriptor has one Finger collection; the slot
        // machinery must not invent a second.
        let table = build_table(TOUCHPAD_DESC).expect("parse");
        assert_eq!(table.report(1).map(|r| r.slots), Some(1));
    }

    #[test]
    fn an_oversized_field_is_refused_rather_than_wrapped() {
        // Report Size 512 through the two-byte global item. A real device
        // has no reason to declare this; a malfunctioning or hostile one
        // might, and the table's compact representation must refuse it
        // rather than narrow it.
        //
        // Before the checked conversion this parsed *successfully* into a
        // field with `bit_len == 0` — 512 truncated by `as u8` — which
        // then passed `TableError::FieldTooWide` unchallenged, leaving a
        // field in the hot path that always reads zero.
        const OVERSIZED: &[u8] = &[
            0x05, 0x01, // Usage Page (Generic Desktop)
            0x09, 0x02, // Usage (Mouse)
            0xA1, 0x01, // Collection (Application)
            0x09, 0x30, //   Usage (X)
            0x15, 0x00, //   Logical Minimum (0)
            0x26, 0xFF, 0x00, // Logical Maximum (255)
            0x76, 0x00, 0x02, //   Report Size (512)
            0x95, 0x01, //   Report Count (1)
            0x81, 0x02, //   Input (Data,Var,Abs)
            0xC0, // End Collection
        ];
        assert!(
            matches!(
                build_table(OVERSIZED),
                Err(crate::parse::ParseError::FieldOutOfRange { .. })
            ),
            "got {:?}",
            build_table(OVERSIZED)
        );
    }

    #[test]
    fn a_malformed_report_descriptor_is_refused_not_guessed_at() {
        // A truncated item stream.
        assert!(build_table(&[0x05]).is_err());
        // Nothing at all.
        assert!(build_table(&[]).is_err());
    }

    #[test]
    fn parse_refusals_have_stable_metal_diagnostics() {
        use crate::parse::ParseError;
        use crate::table::TableError;

        assert_eq!(ParseError::Malformed.diagnostic(), 1);
        assert_eq!(ParseError::NoInputReports.diagnostic(), 2);
        assert_eq!(ParseError::Table(TableError::TooManyFields).diagnostic(), 3);
        assert_eq!(
            ParseError::Table(TableError::TooManyReports).diagnostic(),
            4
        );
        assert_eq!(
            ParseError::Table(TableError::FieldTooWide(33)).diagnostic(),
            0x2105
        );
        assert_eq!(
            ParseError::Table(TableError::FieldZeroWidth).diagnostic(),
            6
        );
        assert_eq!(
            ParseError::Table(TableError::InconsistentContactAxes).diagnostic(),
            9
        );
        assert_eq!(
            ParseError::FieldOutOfRange {
                start: 70_000,
                len: 8
            }
            .diagnostic(),
            (70_000 << 8) | 7
        );
        assert_eq!(
            ParseError::FieldOutOfRange { start: 8, len: 300 }.diagnostic(),
            (300 << 8) | 7
        );
        assert_eq!(
            ParseError::ReportSizeOutOfRange(70_000).diagnostic(),
            (70_000 << 8) | 8
        );
    }
}

// ─────────────────────────── the table by hand ───────────────────────────

#[test]
fn classification_needs_the_whole_field_set_not_one_field() {
    // Absolute X and Y alone are unsupported; the same two plus a
    // contact id identify a digitizer. They must never become relative
    // mouse deltas while awaiting the rest of the report's fields.
    let mut t = ExtractionTable::new();
    t.begin_report(1).unwrap();
    t.push_field(Field {
        usage_page: page::GENERIC_DESKTOP,
        usage: usage::X,
        bit_offset: 8,
        bit_len: 16,
        signed: false,
        relative: false,
        slot: 0,
    })
    .unwrap();
    t.push_field(Field {
        usage_page: page::GENERIC_DESKTOP,
        usage: usage::Y,
        bit_offset: 24,
        bit_len: 16,
        signed: false,
        relative: false,
        slot: 0,
    })
    .unwrap();
    t.classify();
    assert_eq!(t.reports()[0].shape, ReportShape::Unknown);

    t.push_field(Field {
        usage_page: page::DIGITIZER,
        usage: usage::CONTACT_ID,
        bit_offset: 40,
        bit_len: 8,
        signed: false,
        relative: false,
        slot: 0,
    })
    .unwrap();
    t.classify();
    assert_eq!(t.reports()[0].shape, ReportShape::Contact);
}

#[test]
fn the_table_refuses_to_overflow_rather_than_truncating() {
    let mut t = ExtractionTable::new();
    t.begin_report(1).unwrap();
    let f = Field {
        usage_page: page::BUTTON,
        usage: 1,
        bit_offset: 0,
        bit_len: 1,
        signed: false,
        relative: false,
        slot: 0,
    };
    for _ in 0..crate::table::MAX_FIELDS {
        t.push_field(f).unwrap();
    }
    assert_eq!(
        t.push_field(f),
        Err(crate::table::TableError::TooManyFields)
    );
    // A field wider than the extractor can represent is refused, not
    // silently narrowed.
    let mut t = ExtractionTable::new();
    t.begin_report(1).unwrap();
    assert_eq!(
        t.push_field(Field { bit_len: 33, ..f }),
        Err(crate::table::TableError::FieldTooWide(33))
    );
    // And so is a zero-width one: nothing can be extracted from it, so it
    // would sit in the hot path always reading zero while claiming a
    // usage. This is the shape a truncated oversized field used to take.
    assert_eq!(
        t.push_field(Field { bit_len: 0, ..f }),
        Err(crate::table::TableError::FieldZeroWidth)
    );
}

// ─────────────────────────── the bring-up ladder ───────────────────────────

/// A device that speaks HID-over-I2C, recording the order it was driven
/// in so the ladder's sequencing can be asserted.
struct HidDevice {
    addr: u8,
    descriptor_register: u16,
    descriptor: [u8; 30],
    report_descriptor: &'static [u8],
    /// Command words seen, in order.
    seen: [u16; 16],
    n_seen: usize,
    /// How many reset-acknowledgement polls to answer non-empty before
    /// acknowledging — the slow-reset device.
    stall_acks: u32,
    /// Whether a reset is outstanding.
    reset_pending: bool,
}

impl HidDevice {
    fn new(report_descriptor: &'static [u8]) -> Self {
        Self::at(report_descriptor, 0x2c)
    }

    fn at(report_descriptor: &'static [u8], addr: u8) -> Self {
        let mut d = [0u8; 30];
        d[0..2].copy_from_slice(&30u16.to_le_bytes());
        d[2..4].copy_from_slice(&0x0100u16.to_le_bytes());
        d[4..6].copy_from_slice(&(report_descriptor.len() as u16).to_le_bytes());
        d[6..8].copy_from_slice(&0x0002u16.to_le_bytes()); // report desc register
        d[8..10].copy_from_slice(&0x0003u16.to_le_bytes()); // input register
        d[10..12].copy_from_slice(&0x0040u16.to_le_bytes()); // max input
        d[12..14].copy_from_slice(&0x0004u16.to_le_bytes());
        d[14..16].copy_from_slice(&0x0040u16.to_le_bytes());
        d[16..18].copy_from_slice(&0x0005u16.to_le_bytes()); // command register
        d[18..20].copy_from_slice(&0x0006u16.to_le_bytes()); // data register
        Self {
            addr,
            descriptor_register: 0x0001,
            descriptor: d,
            report_descriptor,
            seen: [0; 16],
            n_seen: 0,
            stall_acks: 0,
            reset_pending: false,
        }
    }

    fn commands(&self) -> &[u16] {
        &self.seen[..self.n_seen]
    }
}

#[cfg(feature = "alloc")]
#[test]
fn the_thinkbook_qtec0001_composite_shape_fits_and_decodes_a_key() {
    use crate::parse::build_table;

    const ADDR: u8 = 0x3a;
    let mut bus = MockBus::new(HidDevice::at(QTEC0001_COMPOSITE_RECONSTRUCTED_DESC, ADDR));
    let mut bring = BringUp::new(0x0001, Quirks::default());
    run_ladder(&mut bring, &mut bus, ADDR).expect("QTEC0001 bring-up");

    assert!(bring.is_done());
    assert!(
        bus.log.iter().all(|(addr, _, _)| *addr == ADDR),
        "every transaction must use ECKB's firmware address"
    );
    assert_eq!(
        bus.log.first().map(|(_, write, read)| (&write[..], *read)),
        Some((&[0x01, 0x00][..], 30)),
        "the first transaction fetches the _DSM descriptor register"
    );

    let table =
        build_table(QTEC0001_COMPOSITE_RECONSTRUCTED_DESC).expect("complete QTEC0001 report table");
    assert!(table.uses_report_ids());
    assert_eq!(
        table.reports().len(),
        1,
        "feature-only and vendor-only collections spend no routed input slots"
    );
    assert_eq!(
        table.fields().len(),
        14,
        "only keyboard fields spend slots; unsupported input reports retain their wire sizes"
    );
    assert!(
        table.reports().len() < MAX_REPORTS,
        "real composite has report-table headroom"
    );
    assert!(
        table.fields().len() < MAX_FIELDS,
        "real composite has field-table headroom"
    );
    assert_eq!(
        table.report(0x32).map(|r| (r.shape, r.input_bytes)),
        Some((ReportShape::Keyboard, 9))
    );
    for (id, bytes) in [(0x33, 3), (0x34, 2), (0x35, 2)] {
        assert_eq!(
            table.expected_input_length(&[id]),
            Some(bytes),
            "unsupported collection {id:#x} remains harmless"
        );
    }

    // I2C-HID prefixes the nine-byte HID report with its inclusive u16
    // length: report id 0x32, LeftShift, reserved, then six key slots.
    let raw = [11, 0, 0x32, 0x02, 0, 0x04, 0x05, 0, 0, 0, 0];
    let Frame::Report(payload) = decode(&raw, 64) else {
        panic!("well-formed QTEC0001 input frame refused");
    };
    assert_eq!(
        decode1(&table, payload, 0x50),
        Some(InputReport::keyboard(0x50, 0x02, [0x04, 0x05, 0, 0, 0, 0]))
    );
    assert_eq!(decode1(&table, &[0x34, 1], 0x50), None);
    assert_eq!(decode1(&table, &[0x7f, 1, 2, 3], 0x50), None);
    assert_eq!(
        decode1(&table, &[0x32, 0, 0, 4], 0x50),
        None,
        "short report"
    );
    assert_eq!(
        decode1(&table, &[0x32, 0, 0, 4, 0, 0, 0, 0, 0, 0], 0x50),
        None,
        "overlong report"
    );
}

#[cfg(feature = "alloc")]
#[test]
fn qtec0001_keyboard_sequence_preserves_down_up_modifiers_and_repeat() {
    use crate::parse::build_table;

    let table = build_table(QTEC0001_COMPOSITE_RECONSTRUCTED_DESC).expect("QTEC table");
    let sequence = [
        (
            [0x32, 0x00, 0, 0x04, 0, 0, 0, 0, 0],
            InputReport::keyboard(0x50, 0, [4, 0, 0, 0, 0, 0]),
        ),
        // Identical reports are not deduplicated in the decoder: if firmware
        // emits repeat, the input-stream policy—not an IRQ/poll race—owns it.
        (
            [0x32, 0x00, 0, 0x04, 0, 0, 0, 0, 0],
            InputReport::keyboard(0x50, 0, [4, 0, 0, 0, 0, 0]),
        ),
        (
            [0x32, 0x02, 0, 0x04, 0, 0, 0, 0, 0],
            InputReport::keyboard(0x50, 2, [4, 0, 0, 0, 0, 0]),
        ),
        (
            [0x32, 0x00, 0, 0x00, 0, 0, 0, 0, 0],
            InputReport::keyboard(0x50, 0, [0; 6]),
        ),
    ];
    for (payload, expected) in sequence {
        assert_eq!(decode1(&table, &payload, 0x50), Some(expected));
    }
}

impl I2cTarget for HidDevice {
    fn transact(&mut self, addr: u8, write: &[u8], out: &mut [u8]) -> TargetReply {
        if addr != self.addr {
            return TargetReply::NackAddr;
        }

        // A two-byte write followed by a read is a register fetch.
        if write.len() == 2 && !out.is_empty() {
            let reg = u16::from_le_bytes([write[0], write[1]]);
            let src: &[u8] = if reg == self.descriptor_register {
                &self.descriptor
            } else if reg == 0x0002 {
                self.report_descriptor
            } else {
                return TargetReply::NackData(0);
            };
            let n = out.len().min(src.len());
            out[..n].copy_from_slice(&src[..n]);
            return TargetReply::Acked(n);
        }

        // A four-byte write to the command register is a command.
        if write.len() >= 4 && u16::from_le_bytes([write[0], write[1]]) == 0x0005 {
            let word = u16::from_le_bytes([write[2], write[3]]);
            if self.n_seen < self.seen.len() {
                self.seen[self.n_seen] = word;
                self.n_seen += 1;
            }
            // Opcode is the high byte of the word as written.
            if write[3] == 0x01 {
                self.reset_pending = true;
            }
            return TargetReply::Acked(0);
        }

        // A bare read is an input-register poll.
        if write.is_empty() && out.len() >= 2 {
            if self.reset_pending {
                if self.stall_acks > 0 {
                    self.stall_acks -= 1;
                    // Not yet: report the bogus length a device that is
                    // still resetting produces.
                    out[0] = 0xff;
                    out[1] = 0xff;
                    return TargetReply::Acked(2);
                }
                self.reset_pending = false;
                out[0] = 0;
                out[1] = 0;
                return TargetReply::Acked(2);
            }
            out[0] = 0;
            out[1] = 0;
            return TargetReply::Acked(2);
        }

        TargetReply::Acked(0)
    }
}

/// What the caller's clock reports back to the ladder.
///
/// The ladder's reset budget is denominated in wall clock the *caller*
/// measures, so a test that wants to exercise that budget has to be able to
/// say what the caller's clock saw — including "nothing", which is a real
/// caller (a host build, a thread founded with no clock page) and the one
/// case in which the poll ceiling is the only bound left.
#[derive(Clone, Copy)]
struct Reported {
    /// Whether a demanded delay is reported as actually taken.
    delays: bool,
    /// Wall clock one bus transaction costs.
    transaction_ns: u64,
}

impl Reported {
    /// An honest sleeper on a bus whose transactions are free — what every
    /// ordering test wants, and what the ladder's own nominal accounting
    /// used to assume was true everywhere.
    const FREE: Self = Self {
        delays: true,
        transaction_ns: 0,
    };
    /// A caller with no clock at all.
    const BLIND: Self = Self {
        delays: false,
        transaction_ns: 0,
    };

    /// An honest sleeper on a bus where one transaction costs
    /// `transaction_ns` — the shape a controller whose completion line does
    /// not deliver actually has.
    const fn bus(transaction_ns: u64) -> Self {
        Self {
            delays: true,
            transaction_ns,
        }
    }
}

/// Drive a ladder to completion against a bus, counting the delays it
/// demanded. Returns the total nanoseconds asked for.
fn run_ladder<T: I2cTarget>(
    b: &mut BringUp,
    bus: &mut MockBus<T>,
    addr: u8,
) -> Result<u64, crate::BringUpError> {
    run_ladder_reporting(b, bus, addr, Reported::FREE)
}

/// [`run_ladder`] with the caller's clock spelled out.
fn run_ladder_reporting<T: I2cTarget>(
    b: &mut BringUp,
    bus: &mut MockBus<T>,
    addr: u8,
    clock: Reported,
) -> Result<u64, crate::BringUpError> {
    let mut buf = [0u8; 512];
    let mut delay_ns = 0u64;
    for _ in 0..4096 {
        match b.step() {
            Step::Done => return Ok(delay_ns),
            Step::Delay { ns } => {
                delay_ns += ns;
                b.delayed(if clock.delays { ns } else { 0 })?;
            }
            Step::Write { len } => {
                let mut out = [0u8; crate::bringup::OUT_LEN];
                out[..len].copy_from_slice(&b.out()[..len]);
                bus.write(addr, &out[..len]).expect("write");
                b.wrote()?;
            }
            Step::WriteRead {
                write_len,
                read_len,
            } => {
                let mut out = [0u8; crate::bringup::OUT_LEN];
                out[..write_len].copy_from_slice(&b.out()[..write_len]);
                let n = bus
                    .write_read(addr, &out[..write_len], &mut buf[..read_len])
                    .expect("write_read");
                b.read(&buf[..n], clock.transaction_ns)?;
            }
            Step::Read { len } => {
                let n = bus.read(addr, &mut buf[..len]).expect("read");
                b.read(&buf[..n], clock.transaction_ns)?;
            }
        }
    }
    panic!("the ladder did not terminate");
}

#[test]
fn the_ladder_runs_power_settle_reset_ack_power_then_fetches_the_descriptor() {
    let mut bus = MockBus::new(HidDevice::new(MOUSE_DESC));
    let mut b = BringUp::new(0x0001, Quirks::default());
    let delayed = run_ladder(&mut b, &mut bus, 0x2c).expect("bring-up");

    assert!(b.is_done());
    let d = b.descriptor().expect("descriptor read");
    assert_eq!(d.w_command_register.get(), 0x0005);

    // The command order is the protocol's, not an approximation of it:
    // power on, reset, power on again.
    let power_on = u16::from_le_bytes([
        Command::set_power(0x0005, PowerState::On).bytes()[2],
        Command::set_power(0x0005, PowerState::On).bytes()[3],
    ]);
    let reset = u16::from_le_bytes([
        Command::reset(0x0005).bytes()[2],
        Command::reset(0x0005).bytes()[3],
    ]);
    assert_eq!(bus.target().commands(), &[power_on, reset, power_on]);

    // Both power commands demand a settle, separate from reset polling.
    assert_eq!(delayed, 2 * POWER_SETTLE_NS + crate::bringup::RESET_POLL_NS);
}

#[test]
fn every_power_on_is_followed_by_a_settle_before_further_io() {
    use crate::bringup::{NO_IRQ_RESET_NS, Phase, RESET_POLL_NS};

    for no_irq in [false, true] {
        for no_power_after_reset in [false, true] {
            let mut b = BringUp::new(
                0x0001,
                Quirks {
                    no_irq,
                    no_power_after_reset,
                },
            );
            assert!(matches!(b.step(), Step::WriteRead { read_len: 30, .. }));
            b.read(&HidDevice::new(MOUSE_DESC).descriptor, 0).unwrap();
            assert!(matches!(b.step(), Step::Write { .. }));
            b.wrote().unwrap();
            assert_eq!(b.phase(), Phase::Power);
            assert_eq!(
                b.step(),
                Step::Delay {
                    ns: POWER_SETTLE_NS
                }
            );
            b.delayed(POWER_SETTLE_NS).unwrap();

            assert_eq!(b.phase(), Phase::Reset);
            assert!(matches!(b.step(), Step::Write { .. }));
            b.wrote().unwrap();
            let ack_delay = if no_irq {
                NO_IRQ_RESET_NS
            } else {
                RESET_POLL_NS
            };
            assert_eq!(b.step(), Step::Delay { ns: ack_delay });
            b.delayed(ack_delay).unwrap();
            if !no_irq {
                assert_eq!(b.step(), Step::Read { len: 2 });
                b.read(&[0, 0], 0).unwrap();
            }
            let reset_spent = b.reset_spent_ns();
            assert_eq!(reset_spent, ack_delay);

            if !no_power_after_reset {
                assert!(matches!(b.step(), Step::Write { .. }));
                assert_eq!(&b.out()[..4], Command::set_power(5, PowerState::On).bytes());
                b.wrote().unwrap();
                assert_eq!(b.phase(), Phase::Power);
                // Merely polling for work must not advance the wait, and
                // a premature descriptor result cannot satisfy it either.
                for _ in 0..2 {
                    assert_eq!(
                        b.step(),
                        Step::Delay {
                            ns: POWER_SETTLE_NS
                        }
                    );
                }
                assert_eq!(b.read(MOUSE_DESC, 0), Err(crate::BringUpError::OutOfOrder));
                assert_eq!(b.wrote(), Err(crate::BringUpError::OutOfOrder));
                b.delayed(POWER_SETTLE_NS).unwrap();
            }

            // The completed reset is not charged for the second power wait.
            // A quirk that omits the command also omits its settle.
            assert_eq!(b.reset_spent_ns(), reset_spent);
            assert_eq!(b.phase(), Phase::ReportDescriptor);
            assert_eq!(
                b.step(),
                Step::WriteRead {
                    write_len: 2,
                    read_len: MOUSE_DESC.len(),
                }
            );
            assert_eq!(&b.out()[..2], &[2, 0]);
            b.read(MOUSE_DESC, 0).unwrap();
            assert_eq!(b.step(), Step::Done);
        }
    }
}

#[test]
fn a_slow_reset_is_polled_within_budget_rather_than_abandoned() {
    let mut dev = HidDevice::new(MOUSE_DESC);
    dev.stall_acks = 25;
    let mut bus = MockBus::new(dev);
    let mut b = BringUp::new(0x0001, Quirks::default());
    run_ladder(&mut b, &mut bus, 0x2c).expect("bring-up");
    assert!(b.is_done());
    assert_eq!(b.polls(), 25, "one poll per stalled acknowledgement");
}

#[test]
fn a_reset_that_never_acknowledges_times_out_instead_of_hanging() {
    let mut dev = HidDevice::new(MOUSE_DESC);
    dev.stall_acks = u32::MAX;
    let mut bus = MockBus::new(dev);
    let mut b = BringUp::new(0x0001, Quirks::default());
    let err = run_ladder(&mut b, &mut bus, 0x2c).unwrap_err();
    let crate::BringUpError::ResetTimeout { polls, spent_ns } = err else {
        panic!("{err:?}");
    };
    // On a free bus the poll cadence *is* the wall clock, so the budget and
    // the ceiling fall together — the old behaviour, preserved exactly.
    assert_eq!(polls, crate::bringup::MAX_RESET_POLLS);
    assert_eq!(spent_ns, crate::bringup::RESET_TIMEOUT_NS);
}

#[test]
fn a_reset_budget_is_spent_in_wall_clock_not_in_poll_count() {
    // The gigatron shape: the device answers its descriptor, accepts power
    // and reset, and then never acknowledges — while *every bus
    // transaction* costs the caller a full transfer-pacing deadline,
    // because the controller's completion line does not deliver. The old
    // accounting charged each poll a nominal millisecond and let the ladder
    // run a thousand of them: one advertised second that was really twenty,
    // per child, spent inside the founding window with the boot splash
    // frozen behind it.
    const XFER_WAIT_NS: u64 = 20_000_000;

    let mut dev = HidDevice::new(MOUSE_DESC);
    dev.stall_acks = u32::MAX;
    let mut bus = MockBus::new(dev);
    let mut b = BringUp::new(0x0001, Quirks::default());
    let err =
        run_ladder_reporting(&mut b, &mut bus, 0x2c, Reported::bus(XFER_WAIT_NS)).unwrap_err();

    let crate::BringUpError::ResetTimeout { polls, spent_ns } = err else {
        panic!("{err:?}");
    };
    assert!(
        spent_ns >= crate::bringup::RESET_TIMEOUT_NS,
        "the budget must end on the clock, not before it: spent {spent_ns} ns",
    );
    // The bound that matters: the ladder gave up after about a second of
    // real time, not after a thousand polls costing twenty times that.
    assert!(
        spent_ns < 2 * crate::bringup::RESET_TIMEOUT_NS,
        "one poll of overshoot at most: spent {spent_ns} ns",
    );
    assert!(
        polls < crate::bringup::MAX_RESET_POLLS,
        "the wall clock, not the poll ceiling, must be what ended this: {polls} polls",
    );
}

#[test]
fn a_caller_with_no_clock_still_terminates_on_the_poll_ceiling() {
    // A host build, or a thread founded without a calibrated clock page,
    // reports nothing. The wall-clock budget can never fire for it, so the
    // count ceiling has to remain — a single-ceiling design would have
    // turned "no clock" into "no bound".
    let mut dev = HidDevice::new(MOUSE_DESC);
    dev.stall_acks = u32::MAX;
    let mut bus = MockBus::new(dev);
    let mut b = BringUp::new(0x0001, Quirks::default());
    let err = run_ladder_reporting(&mut b, &mut bus, 0x2c, Reported::BLIND).unwrap_err();
    let crate::BringUpError::ResetTimeout { polls, spent_ns } = err else {
        panic!("{err:?}");
    };
    assert_eq!(polls, crate::bringup::MAX_RESET_POLLS);
    assert_eq!(spent_ns, 0, "a blind caller reports no time at all");
}

#[test]
fn a_present_but_slow_device_keeps_its_whole_protocol_budget() {
    // The fail-closed half: nothing here may cut short a device that is
    // really there and really answering. This one takes almost the entire
    // second — 900 polls on a bus whose transactions are not free — and
    // must still come up.
    let mut dev = HidDevice::new(MOUSE_DESC);
    dev.stall_acks = 900;
    let mut bus = MockBus::new(dev);
    let mut b = BringUp::new(0x0001, Quirks::default());
    run_ladder_reporting(&mut b, &mut bus, 0x2c, Reported::bus(100_000)).expect("bring-up");
    assert!(b.is_done());
    assert_eq!(b.polls(), 900);
    assert!(b.reset_spent_ns() < crate::bringup::RESET_TIMEOUT_NS);
}

#[test]
fn a_no_irq_device_is_waited_out_rather_than_polled_forever() {
    // This device never acknowledges — polling it would time out a
    // perfectly healthy part. The quirk turns the poll into one sleep.
    let mut dev = HidDevice::new(MOUSE_DESC);
    dev.stall_acks = u32::MAX;
    let mut bus = MockBus::new(dev);
    let mut b = BringUp::new(
        0x0001,
        Quirks {
            no_irq: true,
            ..Quirks::default()
        },
    );
    let delayed = run_ladder(&mut b, &mut bus, 0x2c).expect("bring-up");

    assert!(b.is_done());
    assert_eq!(b.polls(), 0, "no polling at all");
    assert_eq!(
        delayed,
        2 * POWER_SETTLE_NS + crate::bringup::NO_IRQ_RESET_NS
    );
}

#[test]
fn the_second_power_command_can_be_suppressed_for_a_device_that_refuses_it() {
    let mut bus = MockBus::new(HidDevice::new(MOUSE_DESC));
    let mut b = BringUp::new(
        0x0001,
        Quirks {
            no_power_after_reset: true,
            ..Quirks::default()
        },
    );
    let delayed = run_ladder(&mut b, &mut bus, 0x2c).expect("bring-up");
    assert_eq!(
        bus.target().commands().len(),
        2,
        "power, reset — no second power"
    );
    assert_eq!(delayed, POWER_SETTLE_NS + crate::bringup::RESET_POLL_NS);
}

#[test]
fn a_device_whose_descriptor_is_malformed_fails_bring_up_at_the_first_rung() {
    struct Dead;
    impl I2cTarget for Dead {
        fn transact(&mut self, _a: u8, _w: &[u8], out: &mut [u8]) -> TargetReply {
            // The all-ones a device that is not driving the bus produces.
            out.fill(0xff);
            TargetReply::Acked(out.len())
        }
    }
    let mut bus = MockBus::new(Dead);
    let mut b = BringUp::new(0x0001, Quirks::default());
    let err = run_ladder(&mut b, &mut bus, 0x2c).unwrap_err();
    assert!(matches!(err, crate::BringUpError::Descriptor(_)), "{err:?}");
}

#[test]
fn a_result_delivered_for_a_step_that_was_not_asked_for_is_refused() {
    // The state machine is driven by a loop that could get out of step;
    // it must say so rather than silently advancing.
    let mut b = BringUp::new(0x0001, Quirks::default());
    assert_eq!(b.wrote(), Err(crate::BringUpError::OutOfOrder));
    assert_eq!(b.delayed(0), Err(crate::BringUpError::OutOfOrder));
}

#[test]
fn framing_and_the_descriptor_agree_on_the_maximum_input_length() {
    // The descriptor's `wMaxInputLength` is what framing refuses against,
    // so the two must be read from the same place — a driver that used a
    // constant here would accept a report its buffer cannot hold.
    let dev = HidDevice::new(MOUSE_DESC);
    let d = HidDescriptor::parse(&dev.descriptor).expect("valid");
    let max = d.w_max_input_length.get();
    assert_eq!(max, 0x40);
    let raw = [(max + 1) as u8, 0, 1, 2];
    assert!(matches!(decode(&raw, max), Frame::Refused(_)));
    let raw = [4, 0, 1, 2];
    assert_eq!(decode(&raw, max), Frame::Report(&[1, 2]));
}

#[cfg(feature = "alloc")]
#[test]
fn biggertron_live_descriptors_fit_and_keep_five_fingers_not_duplicate_axes() {
    use crate::parse::build_device;
    use kore_abi::pointer::{self, Record};
    let elan = build_device(include_bytes!(
        "../../../res/x86_64/firmware-msi-prestige16/HID/elan-4516-report.bin"
    ))
    .unwrap();
    let syna = build_device(include_bytes!(
        "../../../res/x86_64/firmware-msi-prestige16/HID/syna-d022-report.bin"
    ))
    .unwrap();
    assert_eq!(elan.table.reports().len(), 2); // touchscreen + pen
    assert_eq!(syna.table.reports().len(), 2); // mouse + touchpad
    assert_eq!(elan.table.report(1).unwrap().slots, 5);
    assert_eq!(syna.table.report(3).unwrap().slots, 5);
    assert_eq!(elan.table.report(1).unwrap().input_bytes, 61);
    // The first finger has two X and two Y values (tip, center). They all
    // belong to the SAME Logical Finger collection. Decode the primary pair.
    let mut frame = [0u8; 61];
    frame[0] = 1;
    frame[1] = 0x85; // touching, contact ID 1, confidence
    frame[2] = 3;
    frame[3] = 4;
    frame[4..6].copy_from_slice(&1234u16.to_le_bytes());
    frame[6..8].copy_from_slice(&1235u16.to_le_bytes());
    frame[8..10].copy_from_slice(&567u16.to_le_bytes());
    frame[10..12].copy_from_slice(&568u16.to_le_bytes());
    frame[60] = 1;
    let mut out = [InputReport::mouse(0, 0, 0, 0, 0); pointer::MAX_FRAME_RECORDS];
    let n = elan
        .table
        .decode_framed_into(&frame, 128, 1, 1, None, &mut out)
        .unwrap();
    assert_eq!(
        out[..n]
            .iter()
            .filter(|r| matches!(Record::decode(**r), Ok(Record::Contact { .. })))
            .count(),
        1
    );
    assert!(out[..n].iter().any(|r| Record::decode(*r)
        == Ok(Record::Contact {
            id: 1,
            x: 1234,
            y: 567,
            touching: true,
            confidence: Some(true),
            in_range: None,
        })));
    assert!(matches!(
        Record::decode(out[n - 1]),
        Ok(Record::End {
            complete: true,
            reported_contacts: Some(1),
            ..
        })
    ));
    // Repeat for all five actual Finger collections. Center coordinates
    // deliberately differ, so choosing them or crossing slots is observable.
    for slot in 0..5 {
        let start = 1 + slot * 11;
        frame[start] = 0x81 | ((slot as u8) << 2);
        frame[start + 3..start + 5].copy_from_slice(&(100 + slot as u16).to_le_bytes());
        frame[start + 5..start + 7].copy_from_slice(&900u16.to_le_bytes());
        frame[start + 7..start + 9].copy_from_slice(&(200 + slot as u16).to_le_bytes());
        frame[start + 9..start + 11].copy_from_slice(&800u16.to_le_bytes());
    }
    frame[60] = 5;
    let n = elan
        .table
        .decode_framed_into(&frame, 128, 1, 2, None, &mut out)
        .unwrap();
    for slot in 0..5 {
        assert!(out[..n].iter().any(|r| Record::decode(*r)
            == Ok(Record::Contact {
                id: slot,
                x: 100 + slot as i32,
                y: 200 + slot as i32,
                touching: true,
                confidence: Some(true),
                in_range: None,
            })));
    }
}

#[test]
fn elan_hid_identity_suppresses_post_reset_power_without_an_acpi_quirk() {
    for (vendor, powers) in [(0x04f3u16, 1), (0x06cb, 2)] {
        let mut device = HidDevice::new(MOUSE_DESC);
        device.descriptor[20..22].copy_from_slice(&vendor.to_le_bytes());
        let mut bus = MockBus::new(device);
        let mut bringup = BringUp::new(1, Quirks::default());
        run_ladder(&mut bringup, &mut bus, 0x2c).unwrap();
        assert!(bringup.is_done());
        assert_eq!(bus.target().commands().len(), powers + 1); // plus RESET
    }
}
