//! Ordered, bounded compositor-to-client events in the window's pair lease.
//! Keyboard reports are complete HID state snapshots, not lossy key edges.

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
    Keyboard {
        source: u8,
        modifiers: u8,
        keys: [u8; 6],
    },
    /// Coordinates are relative to the configured content bounds.
    Pointer {
        x: i32,
        y: i32,
        buttons: u8,
        wheel: i16,
    },
    CloseRequested,
    /// Queue overflow or focus loss: release all client-held keys/buttons.
    Reset,
    /// Compositor chrome activated one of this window's published tabs.
    /// The index is into the client's own `TabModel`, never authority.
    TabActivated {
        index: u8,
    },
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
            } => {
                wire.kind = 3;
                wire.a = u64::from_le_bytes([
                    source, modifiers, keys[0], keys[1], keys[2], keys[3], keys[4], keys[5],
                ])
                .into();
            }
            Event::Pointer {
                x,
                y,
                buttons,
                wheel,
            } => {
                wire.kind = 4;
                wire.a = (u64::from(x as u32) | (u64::from(y as u32) << 32)).into();
                wire.b = (u64::from(buttons) | (u64::from(wheel as u16) << 8)).into();
            }
            Event::CloseRequested => wire.kind = 5,
            Event::Reset => wire.kind = 6,
            Event::TabActivated { index } => {
                wire.kind = 7;
                wire.a = u64::from(index).into();
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
                Event::Keyboard {
                    source,
                    modifiers,
                    keys: [a, b, c, d, e, f],
                }
            }
            4 => Event::Pointer {
                x: self.a.get() as u32 as i32,
                y: (self.a.get() >> 32) as u32 as i32,
                buttons: self.b.get() as u8,
                wheel: (self.b.get() >> 8) as u16 as i16,
            },
            5 => Event::CloseRequested,
            6 => Event::Reset,
            7 if self.a.get() < crate::control::MAX_TABS as u64 => Event::TabActivated {
                index: self.a.get() as u8,
            },
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
            },
            Event::Pointer {
                x: -1,
                y: 1200,
                buttons: 3,
                wheel: -2,
            },
            Event::CloseRequested,
            Event::Reset,
            Event::TabActivated { index: 0 },
            Event::TabActivated { index: 7 },
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
                wheel: 0,
            },
        );
        pointer.b = (1u64 << 24).into();
        assert_eq!(pointer.decode(), None);
        let mut tab = EventWire::encode(1, Event::TabActivated { index: 7 });
        tab.a = 8.into();
        assert_eq!(tab.decode(), None);
        tab.a = 0.into();
        tab.flags = 1;
        assert_eq!(tab.decode(), None);
    }
}
