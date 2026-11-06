//! A file the user chooses for a window (`res/FILE_PICKER.md`): the bounds
//! of a pick's answer, and its name carried back in event records.
//!
//! A window asks with `SurfaceRequest::PickObject` and the file types it
//! accepts. Its controller launches the picker, which claims the pick by
//! its token and answers with the chosen object's stable identity and its
//! name. The window hears the answer as [`NAME_PARTS`] `Event::PickedName`
//! records, then the `Event::Picked` that ends it. The identity is data the
//! window opens from its own root, never authority, and the name is only
//! what to call it.

/// The longest picked name: a filesystem name's width (a DAGFS name's 32
/// bytes, `kore_abi::svc::FS_NAME_BYTES`).
pub const PICK_NAME_BYTES: usize = 32;

/// The bytes one `Event::PickedName` record carries: its two event words.
pub const NAME_PART_BYTES: usize = 16;

/// The records a name takes.
pub const NAME_PARTS: usize = PICK_NAME_BYTES / NAME_PART_BYTES;

const _: () = assert!(NAME_PARTS * NAME_PART_BYTES == PICK_NAME_BYTES);

/// Whether `name` can answer a pick: one filesystem name a window can show
/// — UTF-8, 1 to [`PICK_NAME_BYTES`] long, without a `/`, a NUL or any
/// control character. A NUL would end it early in its zero-padded records.
pub fn valid_name(name: &[u8]) -> bool {
    core::str::from_utf8(name).is_ok_and(|name| {
        !name.is_empty()
            && name.len() <= PICK_NAME_BYTES
            && !name.contains('/')
            && !name.chars().any(char::is_control)
    })
}

/// Part `part` of `name`, zero-padded: what that `Event::PickedName`
/// record carries.
pub fn name_part(name: &[u8], part: usize) -> [u8; NAME_PART_BYTES] {
    let mut bytes = [0; NAME_PART_BYTES];
    let start = (part * NAME_PART_BYTES).min(name.len());
    let end = (start + NAME_PART_BYTES).min(name.len());
    bytes[..end - start].copy_from_slice(&name[start..end]);
    bytes
}

/// A picked file's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PickName {
    bytes: [u8; PICK_NAME_BYTES],
    len: u8,
}

impl PickName {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }

    /// The name; empty if its bytes did not form one (a compositor that
    /// broke the protocol).
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(self.as_bytes()).unwrap_or("")
    }
}

/// A picked name, reassembled from its `Event::PickedName` records as they
/// arrive: each record lands in its part, and the `Event::Picked` that
/// follows takes the whole. A record for another pick starts over, so a
/// pick that ended unheard leaves nothing behind.
#[derive(Clone, Copy, Debug, Default)]
pub struct NameParts {
    serial: u32,
    bytes: [u8; PICK_NAME_BYTES],
    /// One bit for each part heard.
    heard: u8,
}

impl NameParts {
    /// One `Event::PickedName` record. A part past the last is ignored.
    pub fn part(&mut self, serial: u32, part: u8, bytes: [u8; NAME_PART_BYTES]) {
        let part = usize::from(part);
        if part >= NAME_PARTS {
            return;
        }
        if serial != self.serial {
            *self = Self { serial, ..Self::default() };
        }
        self.bytes[part * NAME_PART_BYTES..(part + 1) * NAME_PART_BYTES].copy_from_slice(&bytes);
        self.heard |= 1 << part;
    }

    /// The name for the `Event::Picked` naming `serial`, once every part of
    /// that pick arrived, and a valid name; then nothing is held.
    pub fn take(&mut self, serial: u32) -> Option<PickName> {
        let whole = serial == self.serial && self.heard == (1 << NAME_PARTS) - 1;
        let bytes = self.bytes;
        *self = Self::default();
        let len = bytes.iter().position(|&byte| byte == 0).unwrap_or(PICK_NAME_BYTES);
        (whole && valid_name(&bytes[..len])).then_some(PickName { bytes, len: len as u8 })
    }

    /// Drop what is held: the event queue overflowed (`Event::Reset`).
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_splits_into_its_records_and_reassembles_whole() {
        let name = b"iss-night-aurora-over-the-sea.jpg";
        assert!(name.len() > PICK_NAME_BYTES, "longer than a pick carries");
        assert!(!valid_name(name));
        let name = &name[..PICK_NAME_BYTES];
        assert!(valid_name(name));
        let mut parts = NameParts::default();
        for part in 0..NAME_PARTS {
            parts.part(7, part as u8, name_part(name, part));
        }
        assert_eq!(parts.take(7).unwrap().as_bytes(), name);
        assert_eq!(parts.take(7), None, "taken once");

        // A short name pads its last record and its second with zeros.
        let mut parts = NameParts::default();
        parts.part(8, 0, name_part(b"beach.png", 0));
        assert_eq!(parts.take(8), None, "a part is missing");
        for part in 0..NAME_PARTS {
            parts.part(8, part as u8, name_part(b"beach.png", part));
        }
        assert_eq!(parts.take(8).unwrap().as_str(), "beach.png");
    }

    #[test]
    fn records_of_another_pick_start_over_and_bad_names_are_none() {
        let mut parts = NameParts::default();
        parts.part(1, 0, name_part(b"old.png", 0));
        parts.part(2, 1, name_part(b"new.png", 1));
        parts.part(2, 0, name_part(b"new.png", 0));
        assert_eq!(parts.take(1), None, "another pick's name");
        for (serial, name) in [(3, &b"a/b.png"[..]), (4, b"tab\t.png"), (5, b"\xff.png")] {
            for part in 0..NAME_PARTS {
                parts.part(serial, part as u8, name_part(name, part));
            }
            assert_eq!(parts.take(serial), None, "{name:?}");
        }
        parts.part(6, 0, name_part(b"x.png", 0));
        parts.clear();
        parts.part(6, 1, name_part(b"x.png", 1));
        assert_eq!(parts.take(6), None, "a reset dropped part 0");
    }
}
