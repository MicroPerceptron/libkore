//! Full-width, frame-preserving decode for the production pointer lane.
//! Legacy `decode_into` remains the unchanged keyboard/compatibility API.

use crate::table::{
    ExtractionTable, Field, MAX_CONTACTS, ReportShape, buttons, find, page, usage, value,
};
use kore_abi::{
    pointer::{self, Record},
    svc::InputReport,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    Capacity,
    Descriptor,
    SourceRange,
}

impl ExtractionTable {
    /// Validate descriptor-declared wire length even for a vendor-only report
    /// that owns no routed input stream. Unknown IDs still produce no records.
    pub fn validate_input_length(&self, data: &[u8]) -> Result<(), DecodeError> {
        if self
            .expected_input_length(data)
            .is_some_and(|expected| data.len() != expected)
        {
            return Err(DecodeError::Truncated);
        }
        Ok(())
    }

    /// Decode one HID report without narrowing coordinates, IDs or deltas.
    /// `source_base` reserves MAX_REPORTS consecutive stream IDs per device:
    /// different report collections must not overwrite each other's descriptor.
    /// The caller publishes the returned slice atomically, or none of it.
    pub fn decode_framed_into(
        &self,
        data: &[u8],
        source_base: u8,
        generation: u32,
        sequence: u32,
        received_ns: Option<u64>,
        out: &mut [InputReport],
    ) -> Result<usize, DecodeError> {
        self.validate_input_length(data)?;
        let Some((index, entry, fields)) = self.selected(data) else {
            return Ok(0);
        };
        // The descriptor boundary includes constant padding and the report ID,
        // not just fields we extract. Enforce it before writing any records.
        if entry.input_bytes != 0 && data.len() != usize::from(entry.input_bytes) {
            return Err(DecodeError::Truncated);
        }
        let source = source_base
            .checked_add(index as u8)
            .ok_or(DecodeError::SourceRange)?;
        if fields.iter().any(|f| {
            usize::from(f.bit_offset) + usize::from(f.bit_len) > data.len().saturating_mul(8)
        }) {
            return Err(DecodeError::Truncated);
        }
        if entry.shape == ReportShape::Keyboard {
            return Ok(self.decode_into(data, source, out));
        }
        if entry.shape == ReportShape::Unknown {
            return Ok(0);
        }
        let class = if entry.shape == ReportShape::Contact {
            entry.contact.class
        } else {
            0
        };
        if entry.shape == ReportShape::Contact && !(1..=3).contains(&class) {
            return Err(DecodeError::Descriptor);
        }
        let mut writer = Writer {
            out,
            count: 0,
            source,
        };
        writer.push(Record::Begin {
            sequence,
            received_ns,
            class,
        })?;
        let get = |slot: u8, p: u16, u: u16| find(fields, slot, p, u).map(|f| value(f, data));
        let mut reported_contacts = None;
        let mut complete = true;
        if entry.shape == ReportShape::Mouse {
            let delta = |u: u16| {
                find(fields, 0, page::GENERIC_DESKTOP, u)
                    .filter(|f| f.relative)
                    .map(|f| value(f, data))
            };
            writer.push(Record::Motion {
                dx: delta(usage::X).unwrap_or(0),
                dy: delta(usage::Y).unwrap_or(0),
            })?;
            if let Some(vertical) = delta(usage::WHEEL) {
                writer.push(Record::Scroll {
                    horizontal: 0,
                    vertical,
                    denominator: 1,
                    unit: 0,
                })?;
            }
        } else {
            for (axis, (min, max)) in [entry.contact.x, entry.contact.y].into_iter().enumerate() {
                if min >= max {
                    return Err(DecodeError::Descriptor);
                }
                writer.push(Record::Axis {
                    axis: axis as u8,
                    min,
                    max,
                })?;
                if let Some((units, millimeters)) = entry.contact.resolution[axis] {
                    writer.push(Record::Resolution {
                        axis: axis as u8,
                        units,
                        millimeters,
                    })?;
                }
            }
            if usize::from(entry.slots) > MAX_CONTACTS {
                return Err(DecodeError::Capacity);
            }
            reported_contacts = get(Field::SLOT_GLOBAL, page::DIGITIZER, usage::CONTACT_COUNT)
                .map(|n| u16::try_from(n).map_err(|_| DecodeError::Descriptor))
                .transpose()?;
            // Hybrid reports advertise a total larger than this packet. Their
            // continuation packets often advertise zero. Preserve fragments;
            // never label a partial set as a complete gesture frame.
            let slots = usize::from(entry.slots);
            let count = reported_contacts.map_or(slots, |count| {
                if count == 0 {
                    slots
                } else {
                    slots.min(usize::from(count))
                }
            });
            complete &= reported_contacts.is_none_or(|n| usize::from(n) <= slots);
            for slot in 0..count {
                let slot = slot as u8;
                let touching = get(slot, page::DIGITIZER, usage::TIP_SWITCH).unwrap_or(0) != 0;
                if reported_contacts == Some(0) && touching {
                    complete = false;
                }
                // Empty complete snapshots need no stale X/Y/ID from dead slots.
                if reported_contacts == Some(0) && !touching {
                    continue;
                }
                let id = get(slot, page::DIGITIZER, usage::CONTACT_ID)
                    .map_or(u32::from(slot), |n| n as u32);
                writer.push(Record::Contact {
                    id,
                    x: get(slot, page::GENERIC_DESKTOP, usage::X).ok_or(DecodeError::Descriptor)?,
                    y: get(slot, page::GENERIC_DESKTOP, usage::Y).ok_or(DecodeError::Descriptor)?,
                    touching,
                    confidence: get(slot, page::DIGITIZER, usage::CONFIDENCE).map(|v| v != 0),
                    in_range: get(slot, page::DIGITIZER, usage::IN_RANGE).map(|v| v != 0),
                })?;
                for (usage, attribute) in [
                    (usage::PRESSURE, pointer::PRESSURE),
                    (usage::WIDTH, pointer::WIDTH),
                    (usage::HEIGHT, pointer::HEIGHT),
                    (usage::AZIMUTH, pointer::ORIENTATION),
                ] {
                    if let Some(value) = get(slot, page::DIGITIZER, usage) {
                        writer.push(Record::Attribute {
                            id,
                            attribute,
                            value,
                        })?;
                    }
                }
            }
            if let Some(time) = get(Field::SLOT_GLOBAL, page::DIGITIZER, usage::SCAN_TIME) {
                writer.push(Record::ScanTime(time as u32))?;
            }
        }
        writer.push(Record::Buttons(u64::from(buttons(fields, 0, data))))?;
        writer.push(Record::End {
            sequence,
            generation,
            records: (writer.count - 1) as u16,
            reported_contacts,
            complete,
        })?;
        Ok(writer.count)
    }
}

struct Writer<'a> {
    out: &'a mut [InputReport],
    count: usize,
    source: u8,
}
impl Writer<'_> {
    fn push(&mut self, record: Record) -> Result<(), DecodeError> {
        if self.count >= pointer::MAX_FRAME_RECORDS {
            return Err(DecodeError::Capacity);
        }
        let slot = self.out.get_mut(self.count).ok_or(DecodeError::Capacity)?;
        *slot = record.encode(self.source);
        self.count += 1;
        Ok(())
    }
}
