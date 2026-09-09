//! Authority-free task-local command metadata.
//!
//! A launcher owns the executable namespace and serializes only inert display
//! metadata into this bounded record. A child such as `help` may enumerate the
//! record, but receives no executable endpoint and cannot turn a copied name
//! into launch authority.

use crate::image::{ImageName, NAME_BYTES};

/// Snapshot wire version.
pub const COMMAND_SNAPSHOT_ABI: u8 = 1;
/// The largest task-local command set accepted by the decoder.  Raised
/// from 32 when the desktop's `editor` and `power-ui` rows joined the two
/// native filesystem verbs; the byte budget (`PROCESS_CONTEXT_BYTES`) is
/// the other bound, proven by exec's host test over the real catalogue.
pub const MAX_COMMAND_SNAPSHOT_ENTRIES: usize = 40;
/// The launcher holds a power-control face and therefore handles these two
/// privileged controls in-process.
pub const COMMAND_SNAPSHOT_SYSTEM_POWER: u8 = 1 << 0;
const COMMAND_SNAPSHOT_KNOWN_FLAGS: u8 = COMMAND_SNAPSHOT_SYSTEM_POWER;
const MAGIC: [u8; 4] = *b"KCMD";
const HEADER_BYTES: usize = 8;
const ENTRY_HEADER_BYTES: usize = 4;

/// Why task-local command metadata could not be encoded or decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandSnapshotError {
    BufferTooSmall,
    TooManyEntries,
    FieldTooLong,
    InvalidText,
    DuplicateName,
    InvalidEncoding,
}

/// One authority-free command description borrowed from a validated snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandDescription<'a> {
    pub local_name: &'a str,
    pub usage: &'a str,
    pub summary: &'a str,
}

/// Incremental encoder over a caller-owned bounded buffer.
pub struct CommandSnapshotBuilder<'a> {
    bytes: &'a mut [u8],
    len: usize,
    count: u8,
}

impl<'a> CommandSnapshotBuilder<'a> {
    /// Begin a snapshot. Flags describe inert shell-local UI state and grant no
    /// authority to the eventual reader.
    pub fn new(bytes: &'a mut [u8], flags: u8) -> Result<Self, CommandSnapshotError> {
        if bytes.len() < HEADER_BYTES || flags & !COMMAND_SNAPSHOT_KNOWN_FLAGS != 0 {
            return Err(CommandSnapshotError::BufferTooSmall);
        }
        bytes[..HEADER_BYTES].fill(0);
        bytes[..4].copy_from_slice(&MAGIC);
        bytes[4] = COMMAND_SNAPSHOT_ABI;
        bytes[5] = flags;
        Ok(Self {
            bytes,
            len: HEADER_BYTES,
            count: 0,
        })
    }

    /// Append one local namespace row. The executable capability paired with
    /// this metadata deliberately is not accepted by this API.
    pub fn push(
        &mut self,
        local_name: &ImageName,
        usage: &[u8],
        summary: &[u8],
    ) -> Result<(), CommandSnapshotError> {
        if usize::from(self.count) >= MAX_COMMAND_SNAPSHOT_ENTRIES {
            return Err(CommandSnapshotError::TooManyEntries);
        }
        let name = local_name
            .as_str()
            .ok_or(CommandSnapshotError::InvalidText)?;
        validate_text(name.as_bytes(), true)?;
        validate_text(usage, false)?;
        validate_text(summary, false)?;
        if name.len() > u8::MAX as usize
            || usage.len() > u8::MAX as usize
            || summary.len() > u8::MAX as usize
        {
            return Err(CommandSnapshotError::FieldTooLong);
        }
        if self.contains_name(name.as_bytes())? {
            return Err(CommandSnapshotError::DuplicateName);
        }
        let needed = ENTRY_HEADER_BYTES + name.len() + usage.len() + summary.len();
        let end = self
            .len
            .checked_add(needed)
            .filter(|end| *end <= self.bytes.len() && *end <= u16::MAX as usize)
            .ok_or(CommandSnapshotError::BufferTooSmall)?;
        let row = &mut self.bytes[self.len..end];
        row[0] = name.len() as u8;
        row[1] = usage.len() as u8;
        row[2] = summary.len() as u8;
        row[3] = 0;
        let mut at = ENTRY_HEADER_BYTES;
        row[at..at + name.len()].copy_from_slice(name.as_bytes());
        at += name.len();
        row[at..at + usage.len()].copy_from_slice(usage);
        at += usage.len();
        row[at..at + summary.len()].copy_from_slice(summary);
        self.len = end;
        self.count += 1;
        Ok(())
    }

    /// Seal the live prefix and return its exact length.
    pub fn finish(self) -> usize {
        self.bytes[6] = self.count;
        self.bytes[7] = 0;
        // The ProcessIo context prefix already carries the exact byte length,
        // so the snapshot header needs no second independently mutable length.
        self.len
    }

    fn contains_name(&self, wanted: &[u8]) -> Result<bool, CommandSnapshotError> {
        let mut at = HEADER_BYTES;
        for _ in 0..self.count {
            let (description, next) = decode_entry(&self.bytes[..self.len], at)?;
            if description.local_name.as_bytes() == wanted {
                return Ok(true);
            }
            at = next;
        }
        Ok(false)
    }
}

/// Validated, immutable metadata snapshot.
pub struct CommandSnapshot<'a> {
    bytes: &'a [u8],
    flags: u8,
    count: u8,
}

impl<'a> CommandSnapshot<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, CommandSnapshotError> {
        if bytes.len() < HEADER_BYTES
            || bytes.len() > u16::MAX as usize
            || bytes[..4] != MAGIC
            || bytes[4] != COMMAND_SNAPSHOT_ABI
            || bytes[5] & !COMMAND_SNAPSHOT_KNOWN_FLAGS != 0
            || bytes[7] != 0
        {
            return Err(CommandSnapshotError::InvalidEncoding);
        }
        let count = bytes[6];
        if usize::from(count) > MAX_COMMAND_SNAPSHOT_ENTRIES {
            return Err(CommandSnapshotError::TooManyEntries);
        }
        let mut offsets = [0u16; MAX_COMMAND_SNAPSHOT_ENTRIES];
        let mut at = HEADER_BYTES;
        for index in 0..usize::from(count) {
            offsets[index] = at as u16;
            let (current, next) = decode_entry(bytes, at)?;
            for prior in offsets[..index].iter().copied() {
                let (seen, _) = decode_entry(bytes, usize::from(prior))?;
                if seen.local_name == current.local_name {
                    return Err(CommandSnapshotError::DuplicateName);
                }
            }
            at = next;
        }
        if at != bytes.len() {
            return Err(CommandSnapshotError::InvalidEncoding);
        }
        Ok(Self {
            bytes,
            flags: bytes[5],
            count,
        })
    }

    pub fn has_system_power(&self) -> bool {
        self.flags & COMMAND_SNAPSHOT_SYSTEM_POWER != 0
    }

    pub fn entries(&self) -> CommandDescriptions<'a> {
        CommandDescriptions {
            bytes: self.bytes,
            remaining: self.count,
            at: HEADER_BYTES,
        }
    }

    pub fn find(&self, local_name: &[u8]) -> Option<CommandDescription<'a>> {
        self.entries()
            .find(|description| description.local_name.as_bytes() == local_name)
    }
}

pub struct CommandDescriptions<'a> {
    bytes: &'a [u8],
    remaining: u8,
    at: usize,
}

impl<'a> Iterator for CommandDescriptions<'a> {
    type Item = CommandDescription<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let (entry, next) = decode_entry(self.bytes, self.at).ok()?;
        self.at = next;
        self.remaining -= 1;
        Some(entry)
    }
}

fn validate_text(bytes: &[u8], name: bool) -> Result<(), CommandSnapshotError> {
    if bytes.is_empty()
        || core::str::from_utf8(bytes).is_err()
        || bytes
            .iter()
            .any(|byte| *byte == 0 || *byte == b'\n' || *byte == b'\r')
        || (name && (bytes.len() > NAME_BYTES || bytes.contains(&b' ')))
    {
        return Err(CommandSnapshotError::InvalidText);
    }
    Ok(())
}

fn decode_entry(
    bytes: &[u8],
    at: usize,
) -> Result<(CommandDescription<'_>, usize), CommandSnapshotError> {
    let header = bytes
        .get(at..at + ENTRY_HEADER_BYTES)
        .ok_or(CommandSnapshotError::InvalidEncoding)?;
    if header[3] != 0 {
        return Err(CommandSnapshotError::InvalidEncoding);
    }
    let name_len = usize::from(header[0]);
    let usage_len = usize::from(header[1]);
    let summary_len = usize::from(header[2]);
    let end = at
        .checked_add(ENTRY_HEADER_BYTES)
        .and_then(|value| value.checked_add(name_len))
        .and_then(|value| value.checked_add(usage_len))
        .and_then(|value| value.checked_add(summary_len))
        .filter(|end| *end <= bytes.len())
        .ok_or(CommandSnapshotError::InvalidEncoding)?;
    let mut cursor = at + ENTRY_HEADER_BYTES;
    let name = &bytes[cursor..cursor + name_len];
    cursor += name_len;
    let usage = &bytes[cursor..cursor + usage_len];
    cursor += usage_len;
    let summary = &bytes[cursor..cursor + summary_len];
    validate_text(name, true)?;
    validate_text(usage, false)?;
    validate_text(summary, false)?;
    Ok((
        CommandDescription {
            local_name: core::str::from_utf8(name)
                .map_err(|_| CommandSnapshotError::InvalidText)?,
            usage: core::str::from_utf8(usage).map_err(|_| CommandSnapshotError::InvalidText)?,
            summary: core::str::from_utf8(summary)
                .map_err(|_| CommandSnapshotError::InvalidText)?,
        },
        end,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(name: &str) -> ImageName {
        ImageName::new(name).unwrap()
    }

    #[test]
    fn snapshot_round_trips_only_inert_metadata() {
        let mut bytes = [0u8; 256];
        let mut builder =
            CommandSnapshotBuilder::new(&mut bytes, COMMAND_SNAPSHOT_SYSTEM_POWER).unwrap();
        builder
            .push(&image("hello"), b"hello", b"dynamic-linker witness")
            .unwrap();
        builder
            .push(&image("ls"), b"ls [-l] [path]", b"list directory entries")
            .unwrap();
        let len = builder.finish();
        let snapshot = CommandSnapshot::parse(&bytes[..len]).unwrap();
        assert!(snapshot.has_system_power());
        let mut entries = snapshot.entries();
        assert_eq!(
            entries.next(),
            Some(CommandDescription {
                local_name: "hello",
                usage: "hello",
                summary: "dynamic-linker witness",
            })
        );
        assert_eq!(
            entries.next(),
            Some(CommandDescription {
                local_name: "ls",
                usage: "ls [-l] [path]",
                summary: "list directory entries",
            })
        );
        assert_eq!(entries.next(), None);
        assert_eq!(snapshot.find(b"ls").unwrap().usage, "ls [-l] [path]");
        assert!(snapshot.find(b"touch").is_none());
    }

    #[test]
    fn duplicate_and_malformed_rows_fail_closed() {
        let mut bytes = [0u8; 128];
        let mut builder = CommandSnapshotBuilder::new(&mut bytes, 0).unwrap();
        builder.push(&image("one"), b"one", b"first").unwrap();
        assert_eq!(
            builder.push(&image("one"), b"one", b"second"),
            Err(CommandSnapshotError::DuplicateName)
        );
        let len = builder.finish();
        bytes[7] = 1;
        assert!(CommandSnapshot::parse(&bytes[..len]).is_err());
    }

    #[test]
    fn capacity_refusal_never_publishes_a_partial_row() {
        let mut bytes = [0u8; 24];
        let mut builder = CommandSnapshotBuilder::new(&mut bytes, 0).unwrap();
        assert_eq!(
            builder.push(
                &image("long-command"),
                b"long-command <argument>",
                b"does not fit",
            ),
            Err(CommandSnapshotError::BufferTooSmall)
        );
        let len = builder.finish();
        let snapshot = CommandSnapshot::parse(&bytes[..len]).unwrap();
        assert_eq!(snapshot.entries().count(), 0);
    }
}
