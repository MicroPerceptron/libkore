//! Version-1 framed pointer records on the existing 16-byte input ring.
//!
//! A frame is BEGIN, zero or more typed records, END. Records are contiguous
//! (never interleaved with another frame); END repeats the sequence and counts
//! its payload records. No consumer may apply a prefix. Unknown kinds, flags,
//! duplicate fields, truncation and overflow invalidate the entire frame.
//! Keyboard and legacy input layouts do not change. New versions get new
//! discriminants, not reinterpretations of these payloads.

use crate::svc::InputReport;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Records the pointer channel's ring holds — the value the founder formats
/// the shared window with, and the bound both ends attach against.
///
/// A frame is at most [`MAX_FRAME_RECORDS`] records and is published whole,
/// so the ring must hold several frames for a consumer that wakes once per
/// composite rather than once per frame.
pub const CHANNEL_CAPACITY: u32 = 128;

/// The pointer plane's **level state** — the input service's books, read from
/// memory rather than asked for (`res/EVENT.md` §1).
///
/// The plane's payload is edges: a pointer frame must not coalesce, so every
/// record rides the ring. What belongs here is what a consumer would otherwise
/// have no way to see — the producer's own accounting. Today these counters
/// reach only the input service's diagnostics, so a compositor cannot tell a
/// quiet pointer from a refused one, and `kore_ring`'s ring-level `dropped`
/// counter covers overflow but not malformed or unroutable reports.
///
/// Every field is monotone, so a consumer compares against what it last read
/// and needs no agreement about when either side zeroes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable,
)]
pub struct PointerState {
    /// Records the producer has attempted to publish.
    pub publications: u64,
    /// Reports refused as malformed, out of frame, or unroutable.
    pub rejected: u64,
    /// Frame resets published — a torn or abandoned frame the consumer must
    /// drop rather than apply as a prefix.
    pub resets: u64,
}

pub const MAX_FRAME_RECORDS: usize = 64;
pub const FIRST_KIND: u8 = 0x10;
pub const LAST_KIND: u8 = 0x1f;

pub const fn is_pointer_kind(kind: u8) -> bool {
    kind == InputReport::KIND_MOUSE
        || kind == InputReport::KIND_CONTACT
        || (kind >= FIRST_KIND && kind <= LAST_KIND)
}

/// Raw logical-unit attributes. Attribute calibration is not carried here;
/// pressure, dimensions and orientation must not be treated as physical units.
pub const PRESSURE: u8 = 0;
pub const WIDTH: u8 = 1;
pub const HEIGHT: u8 = 2;
pub const ORIENTATION: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Record {
    /// Class 0 mouse, 1 touchpad, 2 touchscreen, 3 pen, 4 virtual.
    /// Time is host RECEIPT time, not claimed hardware capture time.
    Begin {
        sequence: u32,
        received_ns: Option<u64>,
        class: u8,
    },
    End {
        sequence: u32,
        generation: u32,
        records: u16,
        reported_contacts: Option<u16>,
        complete: bool,
    },
    Axis {
        axis: u8,
        min: i32,
        max: i32,
    },
    Resolution {
        axis: u8,
        units: u32,
        millimeters: u32,
    },
    Contact {
        id: u32,
        x: i32,
        y: i32,
        touching: bool,
        confidence: Option<bool>,
        in_range: Option<bool>,
    },
    Attribute {
        id: u32,
        attribute: u8,
        value: i32,
    },
    Motion {
        dx: i32,
        dy: i32,
    },
    /// Unit 0 detent, 1 pixel, 2 line; denominator allows fractional units.
    Scroll {
        horizontal: i32,
        vertical: i32,
        denominator: u32,
        unit: u8,
    },
    Buttons(u64),
    ScanTime(u32),
    /// Explicit discontinuity. `all` is reserved for a router detecting loss
    /// in a multiplexed upstream ring. Generation is used on the next frame.
    Reset {
        all: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidRecord;

impl Record {
    pub fn encode(self, source: u8) -> InputReport {
        let mut p = [0u8; 14];
        let kind = match self {
            Self::Begin {
                sequence,
                received_ns,
                class,
            } => {
                put32(&mut p, 0, sequence);
                p[4..12].copy_from_slice(&received_ns.unwrap_or(0).to_le_bytes());
                p[12] = u8::from(received_ns.is_some());
                p[13] = class;
                0x10
            }
            Self::End {
                sequence,
                generation,
                records,
                reported_contacts,
                complete,
            } => {
                put32(&mut p, 0, sequence);
                put32(&mut p, 4, generation);
                p[8..10].copy_from_slice(&records.to_le_bytes());
                p[10..12].copy_from_slice(&reported_contacts.unwrap_or(0).to_le_bytes());
                p[12] = u8::from(complete) | (u8::from(reported_contacts.is_some()) << 1);
                0x11
            }
            Self::Axis { axis, min, max } => {
                p[0] = axis;
                put32(&mut p, 2, min as u32);
                put32(&mut p, 6, max as u32);
                0x12
            }
            Self::Resolution {
                axis,
                units,
                millimeters,
            } => {
                p[0] = axis;
                put32(&mut p, 2, units);
                put32(&mut p, 6, millimeters);
                0x13
            }
            Self::Contact {
                id,
                x,
                y,
                touching,
                confidence,
                in_range,
            } => {
                put32(&mut p, 0, id);
                put32(&mut p, 4, x as u32);
                put32(&mut p, 8, y as u32);
                p[12] = u8::from(touching)
                    | (optional_bool(confidence) << 1)
                    | (optional_bool(in_range) << 3);
                0x14
            }
            Self::Attribute {
                id,
                attribute,
                value,
            } => {
                put32(&mut p, 0, id);
                p[4] = attribute;
                put32(&mut p, 6, value as u32);
                0x15
            }
            Self::Motion { dx, dy } => {
                put32(&mut p, 0, dx as u32);
                put32(&mut p, 4, dy as u32);
                0x16
            }
            Self::Scroll {
                horizontal,
                vertical,
                denominator,
                unit,
            } => {
                put32(&mut p, 0, horizontal as u32);
                put32(&mut p, 4, vertical as u32);
                put32(&mut p, 8, denominator);
                p[12] = unit;
                0x17
            }
            Self::Buttons(buttons) => {
                p[..8].copy_from_slice(&buttons.to_le_bytes());
                0x18
            }
            Self::ScanTime(time) => {
                put32(&mut p, 0, time);
                0x19
            }
            Self::Reset { all } => {
                p[0] = u8::from(all);
                0x1a
            }
        };
        let mut report = InputReport::mouse(source, 0, 0, 0, 0);
        report.kind = kind;
        report.state = p[0];
        report.data.copy_from_slice(&p[1..7]);
        report.reserved.copy_from_slice(&p[7..]);
        report
    }

    pub fn decode(report: InputReport) -> Result<Self, InvalidRecord> {
        let mut p = [0; 14];
        p[0] = report.state;
        p[1..7].copy_from_slice(&report.data);
        p[7..].copy_from_slice(&report.reserved);
        let record = match report.kind {
            0x10 if p[13] <= 4 => Self::Begin {
                sequence: get32(&p, 0),
                received_ns: (p[12] == 1).then(|| u64::from_le_bytes(p[4..12].try_into().unwrap())),
                class: p[13],
            },
            0x11 => Self::End {
                sequence: get32(&p, 0),
                generation: get32(&p, 4),
                records: u16::from_le_bytes([p[8], p[9]]),
                reported_contacts: (p[12] & 2 != 0).then(|| u16::from_le_bytes([p[10], p[11]])),
                complete: p[12] & 1 != 0,
            },
            0x12 if p[0] < 2 && (get32(&p, 2) as i32) < (get32(&p, 6) as i32) => Self::Axis {
                axis: p[0],
                min: get32(&p, 2) as i32,
                max: get32(&p, 6) as i32,
            },
            0x13 if p[0] < 2 && get32(&p, 2) != 0 && get32(&p, 6) != 0 => Self::Resolution {
                axis: p[0],
                units: get32(&p, 2),
                millimeters: get32(&p, 6),
            },
            0x14 => Self::Contact {
                id: get32(&p, 0),
                x: get32(&p, 4) as i32,
                y: get32(&p, 8) as i32,
                touching: p[12] & 1 != 0,
                confidence: read_bool(p[12] >> 1),
                in_range: read_bool(p[12] >> 3),
            },
            0x15 if p[4] <= ORIENTATION => Self::Attribute {
                id: get32(&p, 0),
                attribute: p[4],
                value: get32(&p, 6) as i32,
            },
            0x16 => Self::Motion {
                dx: get32(&p, 0) as i32,
                dy: get32(&p, 4) as i32,
            },
            0x17 if get32(&p, 8) != 0 && p[12] <= 2 => Self::Scroll {
                horizontal: get32(&p, 0) as i32,
                vertical: get32(&p, 4) as i32,
                denominator: get32(&p, 8),
                unit: p[12],
            },
            0x18 => Self::Buttons(u64::from_le_bytes(p[..8].try_into().unwrap())),
            0x19 => Self::ScanTime(get32(&p, 0)),
            0x1a => Self::Reset { all: p[0] == 1 },
            _ => return Err(InvalidRecord),
        };
        // Canonical encoding is also the reserved-bit/unknown-option guard.
        if record.encode(report.source) != report {
            return Err(InvalidRecord);
        }
        Ok(record)
    }
}

fn put32(p: &mut [u8; 14], at: usize, n: u32) {
    p[at..at + 4].copy_from_slice(&n.to_le_bytes());
}
fn get32(p: &[u8; 14], at: usize) -> u32 {
    u32::from_le_bytes(p[at..at + 4].try_into().unwrap())
}
fn optional_bool(b: Option<bool>) -> u8 {
    b.map_or(0, |b| 1 | (u8::from(b) << 1))
}
fn read_bool(b: u8) -> Option<bool> {
    (b & 1 != 0).then_some(b & 2 != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_wire_is_canonical_and_keeps_full_width_values() {
        let records = [
            Record::Begin {
                sequence: u32::MAX,
                received_ns: Some(u64::MAX),
                class: 1,
            },
            Record::End {
                sequence: 4,
                generation: 8,
                records: 17,
                reported_contacts: Some(5),
                complete: false,
            },
            Record::Axis {
                axis: 1,
                min: i32::MIN,
                max: i32::MAX,
            },
            Record::Resolution {
                axis: 0,
                units: 170,
                millimeters: 3,
            },
            Record::Contact {
                id: u32::MAX,
                x: i32::MIN,
                y: i32::MAX,
                touching: true,
                confidence: None,
                in_range: Some(false),
            },
            Record::Attribute {
                id: 999,
                attribute: PRESSURE,
                value: i32::MAX,
            },
            Record::Motion {
                dx: 100_000,
                dy: -100_000,
            },
            Record::Scroll {
                horizontal: 1,
                vertical: -1,
                denominator: 120,
                unit: 0,
            },
            Record::Buttons(u64::MAX),
            Record::ScanTime(u32::MAX),
            Record::Reset { all: true },
        ];
        for record in records {
            assert_eq!(Record::decode(record.encode(42)), Ok(record));
        }
        let mut bad = Record::Motion { dx: 0, dy: 0 }.encode(2);
        bad.reserved[6] = 1;
        assert_eq!(Record::decode(bad), Err(InvalidRecord));
        for kind in 0x1b..=0x1f {
            bad.kind = kind;
            assert!(is_pointer_kind(kind));
            assert!(Record::decode(bad).is_err());
        }
        assert!(
            Record::decode(
                Record::Scroll {
                    horizontal: 0,
                    vertical: 0,
                    denominator: 0,
                    unit: 0
                }
                .encode(1)
            )
            .is_err()
        );
    }
}
