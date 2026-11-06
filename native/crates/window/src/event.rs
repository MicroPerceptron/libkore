//! Ordered, bounded compositor-to-client events in the window's pair lease.
//! Keyboard reports are complete HID state snapshots, not lossy key edges.

/// The vocabulary an [`Event::Keyboard`] is read in — its usages, modifier
/// bits and lock state — so a client names keys without a second source.
pub use kore_abi::keyboard::{Locks, modifier, usage};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned, little_endian::U64};

pub const EVENT_CAP: u32 = 64;

#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct EventWire {
    kind: u8,
    flags: u8,
    reserved: [u8; 6],
    serial: U64,
    a: U64,
    b: U64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// Re-read Configure. Geometry is never inferred from a pointer event.
    Configure,
    Focus(bool),
    /// One complete keyboard snapshot: the HID modifier bitmap, up to six
    /// held Keyboard/Keypad-page usages, and the seat's lock state, so a
    /// client resolves keys exactly as the terminal discipline does.
    Keyboard {
        source: u8,
        modifiers: u8,
        keys: [u8; 6],
        locks: Locks,
    },
    /// Coordinates are relative to the configured content bounds.
    Pointer {
        x: i32,
        y: i32,
        buttons: u8,
        scroll: Scroll,
    },
    CloseRequested,
    /// Queue overflow or focus loss: release all client-held keys/buttons.
    Reset,
    /// Compositor chrome activated one of this window's published tabs.
    /// The index is into the client's own `TabModel`, never authority.
    TabActivated {
        index: u8,
    },
    /// The context menu this window showed as `serial` has ended: `chosen`
    /// is the id of the entry chosen, or zero if it was dismissed
    /// (`crate::menu`). An "Open With" entry reports its own id, if it has
    /// one, once the compositor has queued the open.
    Menu {
        serial: u32,
        chosen: u16,
    },
    /// The pick this window asked for as `serial` has ended (`crate::pick`):
    /// `object` is the chosen file's stable identity, data the window opens
    /// from its own root, or zero if the user cancelled. A chosen file's
    /// name arrived just before, in `PickedName` records.
    Picked {
        serial: u32,
        object: u64,
    },
    /// Part `part` of the name of the file chosen for pick `serial`: the
    /// name's bytes from `part * NAME_PART_BYTES`, zero-padded
    /// ([`crate::pick::NameParts`] puts them back together).
    PickedName {
        serial: u32,
        part: u8,
        bytes: [u8; crate::pick::NAME_PART_BYTES],
    },
}

/// How far one pointer event scrolls, in [`ScrollUnit`]s. Positive `y`
/// scrolls toward the top (a wheel turned away from the user: the view moves
/// back toward its start), positive `x` toward the right.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Scroll {
    pub x: i16,
    pub y: i16,
    /// What the steps measure. A scroll of zero has no unit: it encodes as
    /// [`ScrollUnit::Detent`].
    pub unit: ScrollUnit,
}

/// What a [`Scroll`]'s steps measure, numbered as the pointer transport's
/// scroll record numbers them (`kore_abi::pointer::Record::Scroll`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScrollUnit {
    /// Wheel detents: the view moves by its own wheel step per detent.
    #[default]
    Detent = 0,
    /// Pixels of the window's content at its scale: a precision touchpad's
    /// fingers, which the content should follow.
    Pixel = 1,
    /// The view's own lines or rows.
    Line = 2,
}

impl Scroll {
    /// No scrolling: a pointer event that only moves or clicks.
    pub const NONE: Self = Self {
        x: 0,
        y: 0,
        unit: ScrollUnit::Detent,
    };

    pub const fn is_none(self) -> bool {
        self.x == 0 && self.y == 0
    }

    /// Bits 8.. of a pointer event's second word: vertical, horizontal,
    /// then the unit. The vertical sits where the wheel always did, so a
    /// wheel event's encoding is unchanged.
    const fn bits(self) -> u64 {
        if self.is_none() {
            return 0;
        }
        (self.y as u16 as u64) << 8 | (self.x as u16 as u64) << 24 | (self.unit as u64) << 40
    }

    const fn from_bits(b: u64) -> Option<Self> {
        let unit = match (b >> 40) & 0b11 {
            0 => ScrollUnit::Detent,
            1 => ScrollUnit::Pixel,
            2 => ScrollUnit::Line,
            _ => return None,
        };
        Some(Self {
            x: (b >> 24) as u16 as i16,
            y: (b >> 8) as u16 as i16,
            unit,
        })
    }
}

/// Pointer scrolling as whole steps (rows, lines or list items down the
/// view; columns across) for a view that scrolls by them, positive toward
/// the top and the right as in [`Scroll`].
///
/// A detent is the view's wheel step, a line one step, and pixels fill
/// steps of the view's step size: what does not fill a step waits for the
/// next event, so slow fingers still scroll and fast ones scroll exactly as
/// far as they travel. A reversal starts from zero rather than first
/// unwinding the fraction left from the other way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScrollSteps {
    /// Pixels toward the next step on each axis.
    pending: [i64; 2],
}

impl ScrollSteps {
    /// The whole steps `scroll` moves on each axis (`[x, y]`). `step` is
    /// the view's step on each axis in its content pixels (zero is taken as
    /// one), and `per_detent` how many steps a wheel detent moves.
    pub fn steps(&mut self, scroll: Scroll, step: [u32; 2], per_detent: i32) -> [i32; 2] {
        let deltas = [i64::from(scroll.x), i64::from(scroll.y)];
        let mut out = [0i32; 2];
        for axis in 0..2 {
            let delta = deltas[axis];
            out[axis] = match scroll.unit {
                ScrollUnit::Detent | ScrollUnit::Line => {
                    if delta != 0 {
                        // A wheel between two strokes owes them nothing.
                        self.pending[axis] = 0;
                    }
                    let per = if scroll.unit == ScrollUnit::Line {
                        1
                    } else {
                        i64::from(per_detent)
                    };
                    clamp(delta.saturating_mul(per))
                }
                ScrollUnit::Pixel => {
                    let pending = &mut self.pending[axis];
                    if delta.signum() * pending.signum() < 0 {
                        *pending = 0;
                    }
                    *pending += delta;
                    let size = i64::from(step[axis].max(1));
                    let whole = *pending / size;
                    *pending -= whole * size;
                    clamp(whole)
                }
            };
        }
        out
    }
}

fn clamp(steps: i64) -> i32 {
    steps.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

impl EventWire {
    pub fn encode(serial: u64, event: Event) -> Self {
        let mut wire = Self {
            kind: 0,
            flags: 0,
            reserved: [0; 6],
            serial: serial.into(),
            a: 0.into(),
            b: 0.into(),
        };
        match event {
            Event::Configure => wire.kind = 1,
            Event::Focus(focused) => {
                wire.kind = 2;
                wire.flags = u8::from(focused);
            }
            Event::Keyboard {
                source,
                modifiers,
                keys,
                locks,
            } => {
                wire.kind = 3;
                wire.a = u64::from_le_bytes([
                    source, modifiers, keys[0], keys[1], keys[2], keys[3], keys[4], keys[5],
                ])
                .into();
                wire.b = u64::from(locks.bits()).into();
            }
            Event::Pointer {
                x,
                y,
                buttons,
                scroll,
            } => {
                wire.kind = 4;
                wire.a = (u64::from(x as u32) | (u64::from(y as u32) << 32)).into();
                wire.b = (u64::from(buttons) | scroll.bits()).into();
            }
            Event::CloseRequested => wire.kind = 5,
            Event::Reset => wire.kind = 6,
            Event::TabActivated { index } => {
                wire.kind = 7;
                wire.a = u64::from(index).into();
            }
            Event::Menu { serial, chosen } => {
                wire.kind = 8;
                wire.a = u64::from(serial).into();
                wire.b = u64::from(chosen).into();
            }
            Event::Picked { serial, object } => {
                wire.kind = 9;
                wire.a = u64::from(serial).into();
                wire.b = object.into();
            }
            // The name fills both words, so the pick's serial rides the
            // reserved bytes and the part the flags: this kind alone.
            Event::PickedName { serial, part, bytes } => {
                wire.kind = 10;
                wire.flags = part;
                wire.reserved[..4].copy_from_slice(&serial.to_le_bytes());
                let (low, high) = bytes.split_at(8);
                wire.a = u64::from_le_bytes(low.try_into().unwrap()).into();
                wire.b = u64::from_le_bytes(high.try_into().unwrap()).into();
            }
        }
        wire
    }

    pub fn decode(&self) -> Option<(u64, Event)> {
        let event = match self.kind {
            1 => Event::Configure,
            2 if self.flags <= 1 => Event::Focus(self.flags != 0),
            3 => {
                let [source, modifiers, a, b, c, d, e, f] = self.a.get().to_le_bytes();
                // A lock byte outside the representable locks is not a
                // keyboard snapshot; the re-encode below refuses the rest.
                let locks = u8::try_from(self.b.get())
                    .ok()
                    .and_then(Locks::from_bits)?;
                Event::Keyboard {
                    source,
                    modifiers,
                    keys: [a, b, c, d, e, f],
                    locks,
                }
            }
            4 => Event::Pointer {
                x: self.a.get() as u32 as i32,
                y: (self.a.get() >> 32) as u32 as i32,
                buttons: self.b.get() as u8,
                scroll: Scroll::from_bits(self.b.get())?,
            },
            5 => Event::CloseRequested,
            6 => Event::Reset,
            7 if self.a.get() < crate::control::MAX_TABS as u64 => Event::TabActivated {
                index: self.a.get() as u8,
            },
            8 if self.a.get() != 0 => Event::Menu {
                serial: u32::try_from(self.a.get()).ok()?,
                chosen: u16::try_from(self.b.get()).ok()?,
            },
            9 if self.a.get() != 0 => Event::Picked {
                serial: u32::try_from(self.a.get()).ok()?,
                object: self.b.get(),
            },
            10 if usize::from(self.flags) < crate::pick::NAME_PARTS => {
                let serial = u32::from_le_bytes(self.reserved[..4].try_into().unwrap());
                if serial == 0 {
                    return None;
                }
                let mut bytes = [0; crate::pick::NAME_PART_BYTES];
                bytes[..8].copy_from_slice(&self.a.get().to_le_bytes());
                bytes[8..].copy_from_slice(&self.b.get().to_le_bytes());
                Event::PickedName { serial, part: self.flags, bytes }
            }
            _ => return None,
        };
        let serial = self.serial.get();
        (serial != 0 && Self::encode(serial, event) == *self).then_some((serial, event))
    }
}

const _: () = assert!(core::mem::size_of::<EventWire>() == 32);

#[cfg(test)]
mod tests {
    use super::*;

    /// A wheel turned `y` detents (positive away from the user).
    const fn detents(y: i16) -> Scroll {
        Scroll {
            x: 0,
            y,
            unit: ScrollUnit::Detent,
        }
    }

    #[test]
    fn events_roundtrip_exactly_and_reject_unused_data() {
        for event in [
            Event::Configure,
            Event::Focus(false),
            Event::Focus(true),
            Event::Keyboard {
                source: 2,
                modifiers: 8,
                keys: [4, 0, 0, 0, 0, 0],
                locks: Locks::NONE,
            },
            Event::Keyboard {
                source: 2,
                modifiers: 0,
                keys: [0x39, 0, 0, 0, 0, 0],
                locks: Locks::CAPS.toggled(Locks::NUM),
            },
            Event::Pointer {
                x: -1,
                y: 1200,
                buttons: 3,
                scroll: detents(-2),
            },
            Event::Pointer {
                x: 4,
                y: 5,
                buttons: 0,
                scroll: Scroll {
                    x: -300,
                    y: i16::MAX,
                    unit: ScrollUnit::Pixel,
                },
            },
            Event::Pointer {
                x: 4,
                y: 5,
                buttons: 1,
                scroll: Scroll {
                    x: 2,
                    y: 0,
                    unit: ScrollUnit::Line,
                },
            },
            Event::CloseRequested,
            Event::Reset,
            Event::TabActivated { index: 0 },
            Event::TabActivated { index: 7 },
            Event::Menu {
                serial: 1,
                chosen: 0,
            },
            Event::Menu {
                serial: u32::MAX,
                chosen: u16::MAX,
            },
            Event::Picked { serial: 1, object: 0 },
            Event::Picked {
                serial: u32::MAX,
                object: u64::MAX,
            },
            Event::PickedName {
                serial: 1,
                part: 0,
                bytes: *b"beach.png\0\0\0\0\0\0\0",
            },
            Event::PickedName {
                serial: u32::MAX,
                part: (crate::pick::NAME_PARTS - 1) as u8,
                bytes: [0xff; crate::pick::NAME_PART_BYTES],
            },
        ] {
            let mut wire = EventWire::encode(17, event);
            assert_eq!(wire.decode(), Some((17, event)));
            wire.reserved[5] = 1;
            assert_eq!(wire.decode(), None);
            assert_eq!(EventWire::encode(0, event).decode(), None);
        }
        let mut pointer = EventWire::encode(
            1,
            Event::Pointer {
                x: 0,
                y: 0,
                buttons: 0,
                scroll: Scroll::NONE,
            },
        );
        // A wheel event keeps the encoding it always had.
        assert_eq!(
            EventWire::encode(
                1,
                Event::Pointer {
                    x: 0,
                    y: 0,
                    buttons: 2,
                    scroll: detents(-1),
                },
            )
            .b
            .get(),
            2 | 0xffff << 8
        );
        // An unknown unit, bits past the unit, and a unit on no scroll at all
        // are refused.
        for bad in [3u64 << 40, 1 << 42, 1 << 63, 1 << 40] {
            pointer.b = bad.into();
            assert_eq!(pointer.decode(), None, "{bad:#x}");
        }
        let mut keyboard = EventWire::encode(
            1,
            Event::Keyboard {
                source: 1,
                modifiers: 0,
                keys: [0; 6],
                locks: Locks::SCROLL,
            },
        );
        for bad in [0x08, 0x100] {
            keyboard.b = bad.into();
            assert_eq!(keyboard.decode(), None, "lock word {bad:#x}");
        }
        let mut tab = EventWire::encode(1, Event::TabActivated { index: 7 });
        tab.a = 8.into();
        assert_eq!(tab.decode(), None);
        tab.a = 0.into();
        tab.flags = 1;
        assert_eq!(tab.decode(), None);
        // A name's record names its pick and a part the name has.
        let mut name = EventWire::encode(1, Event::PickedName { serial: 3, part: 1, bytes: [7; 16] });
        name.flags = crate::pick::NAME_PARTS as u8;
        assert_eq!(name.decode(), None, "a part past the last");
        name.flags = 1;
        name.reserved[..4].fill(0);
        assert_eq!(name.decode(), None, "no pick's serial");
        let mut picked = EventWire::encode(1, Event::Picked { serial: 2, object: 5 });
        picked.reserved[0] = 2;
        assert_eq!(picked.decode(), None, "only a name's record carries a serial there");
    }

    #[test]
    fn scroll_steps_carry_pixels_into_whole_steps() {
        let pixels = |x, y| Scroll {
            x,
            y,
            unit: ScrollUnit::Pixel,
        };
        let mut steps = ScrollSteps::default();
        // 20-pixel rows: 12 pixels is nothing yet, 12 more is one row.
        assert_eq!(steps.steps(pixels(0, 12), [8, 20], 3), [0, 0]);
        assert_eq!(steps.steps(pixels(0, 12), [8, 20], 3), [0, 1]);
        assert_eq!(steps.steps(pixels(0, 36), [8, 20], 3), [0, 2], "4 + 36 = 40");
        // A reversal starts from zero instead of unwinding the 0 left over.
        assert_eq!(steps.steps(pixels(0, 10), [8, 20], 3), [0, 0]);
        assert_eq!(steps.steps(pixels(0, -19), [8, 20], 3), [0, 0]);
        assert_eq!(steps.steps(pixels(0, -1), [8, 20], 3), [0, -1]);
        // Across, the column width counts.
        assert_eq!(steps.steps(pixels(-17, 0), [8, 20], 3), [-2, 0]);
        // A detent is the view's wheel step and drops the pixel fraction; a
        // line is one step.
        assert_eq!(steps.steps(pixels(0, 15), [8, 20], 3), [0, 0]);
        assert_eq!(steps.steps(detents(-2), [8, 20], 3), [0, -6]);
        assert_eq!(steps.steps(pixels(0, 15), [8, 20], 3), [0, 0], "the 15 before the wheel is gone");
        let line = Scroll {
            x: 1,
            y: 2,
            unit: ScrollUnit::Line,
        };
        assert_eq!(steps.steps(line, [8, 20], 3), [1, 2]);
        // A zero step size is one pixel, never a division by zero.
        assert_eq!(steps.steps(pixels(0, 5), [0, 0], 3), [0, 5]);
        assert_eq!(
            steps.steps(detents(i16::MAX), [1, 1], i32::MAX),
            [0, i32::MAX],
            "saturates"
        );
    }
}
