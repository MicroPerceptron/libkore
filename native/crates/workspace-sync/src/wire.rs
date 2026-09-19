//! Frozen, append-only snapshot wire heads.

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::bounds::FORMAT_VERSION;

pub const SNAPSHOT_MAGIC: [u8; 4] = *b"KWS1";
pub const ENTRY_MAGIC: [u8; 4] = *b"KWE1";
pub const CHANGESET_MAGIC: [u8; 4] = *b"KWC1";

pub type ContentDigest = [u8; 32];
pub type SnapshotId = [u8; 32];
pub type SourceCorrelation = [u8; 16];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EntryKind {
    Directory = 1,
    File = 2,
}

impl EntryKind {
    pub const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Directory),
            2 => Some(Self::File),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EntryFlags(u8);

impl EntryFlags {
    pub const EXECUTABLE: u8 = 1;

    pub const fn new(executable: bool) -> Self {
        Self(if executable { Self::EXECUTABLE } else { 0 })
    }

    pub const fn executable(self) -> bool {
        self.0 & Self::EXECUTABLE != 0
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::EXECUTABLE == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq)]
#[repr(C)]
pub struct SnapshotHeader {
    pub magic: [u8; 4],
    pub version: U16,
    pub policy_version: U16,
    pub entry_count: U16,
    pub directory_count: U16,
    pub reserved: U32,
    pub total_bytes: U64,
    pub parent_snapshot: SnapshotId,
    pub policy_digest: ContentDigest,
    pub root_digest: ContentDigest,
    pub snapshot_id: SnapshotId,
}

impl SnapshotHeader {
    pub fn shape_is_valid(&self) -> bool {
        self.magic == SNAPSHOT_MAGIC
            && self.version.get() == FORMAT_VERSION
            && self.reserved.get() == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq)]
#[repr(C)]
pub struct EntryHeader {
    pub magic: [u8; 4],
    pub version: U16,
    pub index: U16,
    pub kind: u8,
    pub flags: u8,
    pub depth: u8,
    pub reserved0: u8,
    pub path_len: U16,
    pub reserved1: U16,
    pub content_len: U32,
    pub digest: ContentDigest,
    /// Root of the fixed 4 KiB range tree used by the live backend. Snapshot
    /// consumers may ignore it, but it remains committed by the manifest.
    pub range_root: ContentDigest,
    pub source: SourceCorrelation,
}

impl EntryHeader {
    pub fn shape_is_valid(&self) -> bool {
        self.magic == ENTRY_MAGIC
            && self.version.get() == FORMAT_VERSION
            && self.reserved0 == 0
            && self.reserved1.get() == 0
            && EntryKind::from_wire(self.kind).is_some()
            && EntryFlags::from_bits(self.flags).is_some()
    }
}

const _: () = {
    assert!(core::mem::size_of::<SnapshotHeader>() == 152);
    assert!(core::mem::size_of::<EntryHeader>() == 100);
    assert!(EntryKind::Directory as u8 == 1);
    assert!(EntryKind::File as u8 == 2);
};
