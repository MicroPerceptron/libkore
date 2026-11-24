//! Bounded wire assembly. Only a validated END exposes a borrowed frame.
use axinput::{DeviceId, InputTimestamp, PointerSource, ScrollUnit, SeatId, frame::*};
use kore_abi::pointer::{self, Record};

// This is a transport admission limit, not a limit in axinput's borrowed API.
pub const MAX_CONTACTS: usize = 32;

pub struct Assembly {
    pub pending: bool,
    pub source: u8,
    descriptor: DeviceDescriptor,
    sequence: u32,
    timestamp: Option<InputTimestamp>,
    axes: [Option<AxisRange>; 2],
    resolutions: [Option<AxisResolution>; 2],
    contacts: [ContactSample; MAX_CONTACTS],
    count: usize,
    records: u16,
    attributes: [u8; MAX_CONTACTS],
    seen: u8,
    buttons: u64,
    motion: [i32; 2],
    scroll: Option<ScrollSample>,
    scan_time: Option<u32>,
    reported: Option<u16>,
    complete: bool,
}

impl Default for Assembly {
    fn default() -> Self {
        Self {
            pending: false,
            source: 0,
            descriptor: DeviceDescriptor::new(
                SeatId(0),
                DeviceId(0),
                DeviceGeneration(0),
                PointerSource::Mouse,
            ),
            sequence: 0,
            timestamp: None,
            axes: [None; 2],
            resolutions: [None; 2],
            contacts: [ContactSample::new(0, 0, 0, false); MAX_CONTACTS],
            count: 0,
            records: 0,
            attributes: [0; MAX_CONTACTS],
            seen: 0,
            buttons: 0,
            motion: [0; 2],
            scroll: None,
            scan_time: None,
            reported: None,
            complete: false,
        }
    }
}

impl Assembly {
    /// Returns true only when a complete wire envelope (possibly carrying a
    /// hardware fragment) is ready. On error the caller discards this assembly.
    pub fn push(&mut self, source: u8, record: Record) -> Result<bool, FrameError> {
        if let Record::Begin {
            sequence,
            received_ns,
            class,
        } = record
        {
            *self = Self::default();
            self.pending = true;
            self.source = source;
            self.sequence = sequence;
            self.timestamp = received_ns.map(InputTimestamp);
            self.descriptor.device = DeviceId(u32::from(source));
            self.descriptor.source = match class {
                0 => PointerSource::Mouse,
                1 => PointerSource::Touchpad,
                2 => PointerSource::Touchscreen,
                3 => PointerSource::Pen,
                4 => PointerSource::Virtual,
                _ => return Err(FrameError::WrongDevice),
            };
            return Ok(false);
        }
        if !self.pending || self.source != source {
            return Err(FrameError::WrongDevice);
        }
        if let Record::End {
            sequence,
            generation,
            records,
            reported_contacts,
            complete,
        } = record
        {
            if sequence != self.sequence || records != self.records || self.seen & 1 == 0 {
                return Err(FrameError::StaleSequence);
            }
            self.descriptor.generation = DeviceGeneration(generation);
            self.descriptor.axes = match self.axes {
                [None, None] => {
                    self.descriptor.relative_resolution = self.resolutions;
                    None
                }
                [Some(x), Some(y)] => Some([
                    AbsoluteAxis {
                        range: x,
                        resolution: self.resolutions[0],
                    },
                    AbsoluteAxis {
                        range: y,
                        resolution: self.resolutions[1],
                    },
                ]),
                _ => return Err(FrameError::InvalidAxes),
            };
            if matches!(
                self.descriptor.source,
                PointerSource::Touchpad | PointerSource::Touchscreen | PointerSource::Pen
            ) && self.descriptor.axes.is_none()
            {
                return Err(FrameError::InvalidAxes);
            }
            if complete && reported_contacts.is_some_and(|n| usize::from(n) > self.count) {
                return Err(FrameError::Capacity);
            }
            self.reported = reported_contacts;
            self.complete = complete;
            self.view().validate()?;
            self.pending = false;
            return Ok(true);
        }
        self.records += 1;
        if usize::from(self.records) + 2 > pointer::MAX_FRAME_RECORDS {
            return Err(FrameError::Capacity);
        }
        match record {
            Record::Axis { axis, min, max } => {
                let slot = self
                    .axes
                    .get_mut(usize::from(axis))
                    .ok_or(FrameError::InvalidAxes)?;
                if slot.is_some() {
                    return Err(FrameError::InvalidAxes);
                }
                *slot = Some(AxisRange::new(min, max)?);
            }
            Record::Resolution {
                axis,
                units,
                millimeters,
            } => {
                let slot = self
                    .resolutions
                    .get_mut(usize::from(axis))
                    .ok_or(FrameError::InvalidAxes)?;
                if slot.is_some() {
                    return Err(FrameError::InvalidAxes);
                }
                *slot = Some(AxisResolution::new(units, millimeters)?);
            }
            Record::Contact {
                id,
                x,
                y,
                touching,
                confidence,
                in_range,
            } => {
                if self.contacts[..self.count].iter().any(|c| c.id == id) {
                    return Err(FrameError::DuplicateContact);
                }
                let slot = self
                    .contacts
                    .get_mut(self.count)
                    .ok_or(FrameError::Capacity)?;
                *slot = ContactSample::new(id, x, y, touching);
                slot.confidence = confidence;
                slot.in_range = in_range;
                self.count += 1;
            }
            Record::Attribute {
                id,
                attribute,
                value,
            } => {
                let index = self.contacts[..self.count]
                    .iter()
                    .position(|c| c.id == id)
                    .ok_or(FrameError::WrongDevice)?;
                if attribute > pointer::ORIENTATION
                    || self.attributes[index] & (1 << attribute) != 0
                {
                    return Err(FrameError::DuplicateContact);
                }
                self.attributes[index] |= 1 << attribute;
                let c = &mut self.contacts[index];
                match attribute {
                    pointer::PRESSURE => c.pressure = Some(value),
                    pointer::WIDTH => c.width = Some(value),
                    pointer::HEIGHT => c.height = Some(value),
                    _ => c.orientation = Some(value),
                }
            }
            Record::Buttons(buttons) => {
                self.once(1)?;
                self.buttons = buttons;
            }
            Record::Motion { dx, dy } => {
                self.once(2)?;
                self.motion = [dx, dy];
            }
            Record::Scroll {
                horizontal,
                vertical,
                denominator,
                unit,
            } => {
                self.once(4)?;
                if denominator == 0 {
                    return Err(FrameError::InvalidMotion);
                }
                let unit = match unit {
                    0 => ScrollUnit::Detent,
                    1 => ScrollUnit::Pixel,
                    2 => ScrollUnit::Line,
                    _ => return Err(FrameError::InvalidMotion),
                };
                self.scroll = Some(ScrollSample {
                    horizontal: f64::from(horizontal) / f64::from(denominator),
                    vertical: f64::from(vertical) / f64::from(denominator),
                    unit,
                });
            }
            Record::ScanTime(time) => {
                self.once(8)?;
                self.scan_time = Some(time);
            }
            _ => return Err(FrameError::WrongDevice),
        }
        Ok(false)
    }

    fn once(&mut self, flag: u8) -> Result<(), FrameError> {
        if self.seen & flag != 0 {
            return Err(FrameError::DuplicateContact);
        }
        self.seen |= flag;
        Ok(())
    }

    pub fn view(&self) -> PointerFrame<'_> {
        let mut frame = PointerFrame::new(
            &self.descriptor,
            self.sequence,
            &self.contacts[..self.count],
        );
        frame.timestamp = self.timestamp;
        frame.timestamp_origin = if self.timestamp.is_some() {
            TimestampOrigin::Received
        } else {
            TimestampOrigin::Unknown
        };
        frame.scan_time = self.scan_time;
        frame.completeness = if self.complete {
            FrameCompleteness::Complete
        } else {
            FrameCompleteness::Fragment
        };
        frame.reported_contacts = self.reported;
        frame.buttons = self.buttons;
        frame.relative = self.motion;
        frame.scroll = self.scroll;
        frame
    }
}
