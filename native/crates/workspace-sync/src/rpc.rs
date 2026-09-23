//! Fixed, transport-neutral control records for the live workspace adapter.
//!
//! HostLink supplies authenticated route identity, sequencing, cancellation,
//! and bulk objects. These records contain only authority-neutral workspace
//! data; no pathname, descriptor, pointer, route, or shared-memory offset is
//! representable.

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::{
    Entry,
    bounds::{FORMAT_VERSION, MAX_ENTRIES, MAX_PATH_BYTES, MAX_RANGE_PROOF_DEPTH, RANGE_BYTES},
    wire::{ContentDigest, EntryHeader, SnapshotId, SourceCorrelation},
};

pub const ATTACH_MAGIC: [u8; 4] = *b"KWA1";
pub const ENTRY_REQUEST_MAGIC: [u8; 4] = *b"KWEQ";
pub const ENTRY_RESPONSE_MAGIC: [u8; 4] = *b"KWER";
pub const RANGE_REQUEST_MAGIC: [u8; 4] = *b"KWRQ";
pub const RANGE_RESPONSE_MAGIC: [u8; 4] = *b"KWRP";
pub const DELTA_POLL_MAGIC: [u8; 4] = *b"KWDP";
pub const DELTA_RESPONSE_MAGIC: [u8; 4] = *b"KWDR";
pub const DELTA_OP_REQUEST_MAGIC: [u8; 4] = *b"KWDQ";
pub const ERROR_RESPONSE_MAGIC: [u8; 4] = *b"KWE!";
pub const BLOB_STORED_MAGIC: [u8; 4] = *b"KWB1";
pub const TRANSACTION_RESULT_MAGIC: [u8; 4] = *b"KWT1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum WorkspaceMode {
    Snapshot = 1,
    Live = 2,
}

impl WorkspaceMode {
    #[must_use]
    pub const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Snapshot),
            2 => Some(Self::Live),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum WorkspaceStatus {
    Ok = 0,
    Unavailable = 1,
    Stale = 2,
    Conflict = 3,
    Refused = 4,
    Partial = 5,
}

impl WorkspaceStatus {
    #[must_use]
    pub const fn from_wire(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Ok),
            1 => Some(Self::Unavailable),
            2 => Some(Self::Stale),
            3 => Some(Self::Conflict),
            4 => Some(Self::Refused),
            5 => Some(Self::Partial),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RpcError {
    Malformed,
    UnsupportedVersion,
    InvalidMode,
    InvalidStatus,
    InvalidRange,
    Reserved,
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct AttachRequestV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub mode: u8,
    pub flags: u8,
    pub policy_digest: ContentDigest,
    pub known_snapshot: SnapshotId,
}

impl AttachRequestV1 {
    pub fn validate(&self) -> Result<WorkspaceMode, RpcError> {
        if self.magic != ATTACH_MAGIC {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        if self.flags != 0 {
            return Err(RpcError::Reserved);
        }
        WorkspaceMode::from_wire(self.mode).ok_or(RpcError::InvalidMode)
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct AttachResponseV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub status: U16,
    pub snapshot: SnapshotId,
    pub parent_snapshot: SnapshotId,
    pub root_digest: ContentDigest,
    pub policy_digest: ContentDigest,
    pub entry_count: U16,
    pub directory_count: U16,
    pub total_bytes: U64,
    pub delta_sequence: U64,
    /// Importer-private correlation for the exact standing host grant. This
    /// is authority-neutral wire data and never crosses the filesystem
    /// provider's agent-facing API. All zeros means read-only attachment.
    pub write_grant: [u8; 16],
    pub write_operations: u8,
    pub reserved0: u8,
    pub write_max_operations: U16,
    pub write_max_bytes: U64,
    pub write_expires_at: U64,
}

impl AttachResponseV1 {
    pub fn validate(&self) -> Result<WorkspaceStatus, RpcError> {
        if self.magic != ATTACH_MAGIC {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        let status =
            WorkspaceStatus::from_wire(self.status.get()).ok_or(RpcError::InvalidStatus)?;
        let operations =
            crate::GrantOperations::from_bits(self.write_operations).ok_or(RpcError::Malformed)?;
        let read_only = operations.bits() == 0;
        if usize::from(self.entry_count.get()) > MAX_ENTRIES
            || self.delta_sequence.get() == 0
            || self.reserved0 != 0
            || (read_only
                && (self.write_grant != [0; 16]
                    || self.write_max_operations.get() != 0
                    || self.write_max_bytes.get() != 0
                    || self.write_expires_at.get() != 0))
            || (!read_only
                && (self.write_grant == [0; 16]
                    || self.write_max_operations.get() == 0
                    || usize::from(self.write_max_operations.get()) > crate::bounds::MAX_CHANGES
                    || self.write_max_bytes.get() == 0
                    || self.write_max_bytes.get() > crate::bounds::MAX_OVERLAY_BYTES))
        {
            return Err(RpcError::Malformed);
        }
        Ok(status)
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct ManifestEntryRequestV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub index: U16,
    pub snapshot: SnapshotId,
}

impl ManifestEntryRequestV1 {
    pub fn validate(&self) -> Result<(), RpcError> {
        if self.magic != ENTRY_REQUEST_MAGIC {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        if usize::from(self.index.get()) >= MAX_ENTRIES || self.snapshot == [0; 32] {
            return Err(RpcError::Malformed);
        }
        Ok(())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct ManifestEntryResponseV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub status: U16,
    pub snapshot: SnapshotId,
    pub index: U16,
    pub path_len: U16,
    pub reserved: U32,
    pub entry_header: [u8; core::mem::size_of::<EntryHeader>()],
    pub path: [u8; MAX_PATH_BYTES],
}

impl ManifestEntryResponseV1 {
    pub fn from_entry(snapshot: SnapshotId, index: u16, entry: &Entry) -> Option<Self> {
        let path_len = entry.path.as_bytes().len();
        if snapshot == [0; 32] || path_len > MAX_PATH_BYTES {
            return None;
        }
        let mut path = [0; MAX_PATH_BYTES];
        path[..path_len].copy_from_slice(entry.path.as_bytes());
        let mut entry_header = [0; core::mem::size_of::<EntryHeader>()];
        entry_header.copy_from_slice(entry.header(index).as_bytes());
        Some(Self {
            magic: ENTRY_RESPONSE_MAGIC,
            version: U16::new(FORMAT_VERSION),
            status: U16::new(WorkspaceStatus::Ok as u16),
            snapshot,
            index: U16::new(index),
            path_len: U16::new(path_len as u16),
            reserved: U32::ZERO,
            entry_header,
            path,
        })
    }

    pub fn validate(&self) -> Result<(WorkspaceStatus, Option<Entry>), RpcError> {
        if self.magic != ENTRY_RESPONSE_MAGIC || self.reserved.get() != 0 {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        let status =
            WorkspaceStatus::from_wire(self.status.get()).ok_or(RpcError::InvalidStatus)?;
        if status != WorkspaceStatus::Ok {
            return Ok((status, None));
        }
        let path_len = usize::from(self.path_len.get());
        if self.snapshot == [0; 32]
            || usize::from(self.index.get()) >= MAX_ENTRIES
            || path_len == 0
            || path_len > MAX_PATH_BYTES
            || self.path[path_len..].iter().any(|byte| *byte != 0)
        {
            return Err(RpcError::Malformed);
        }
        let header =
            EntryHeader::read_from_bytes(&self.entry_header).map_err(|_| RpcError::Malformed)?;
        if header.index.get() != self.index.get() || usize::from(header.path_len.get()) != path_len
        {
            return Err(RpcError::Malformed);
        }
        let entry =
            Entry::from_wire(&header, &self.path[..path_len]).map_err(|_| RpcError::Malformed)?;
        Ok((status, Some(entry)))
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct RangeRequestV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub flags: U16,
    pub snapshot: SnapshotId,
    pub source: SourceCorrelation,
    pub version_digest: ContentDigest,
    pub range_root: ContentDigest,
    pub file_len: U32,
    pub offset: U32,
    pub length: U16,
    pub retry: u8,
    pub reserved: u8,
}

impl RangeRequestV1 {
    pub fn validate(&self) -> Result<(), RpcError> {
        if self.magic != RANGE_REQUEST_MAGIC {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        if self.flags.get() != 0 || self.reserved != 0 {
            return Err(RpcError::Reserved);
        }
        let offset = self.offset.get();
        let len = u32::from(self.length.get());
        let end = offset.checked_add(len).ok_or(RpcError::InvalidRange)?;
        if self.snapshot == [0; 32]
            || self.source == [0; 16]
            || self.version_digest == [0; 32]
            || self.range_root == [0; 32]
            || len == 0
            || len > RANGE_BYTES as u32
            || end > self.file_len.get()
            || offset / RANGE_BYTES as u32 != (end - 1) / RANGE_BYTES as u32
        {
            return Err(RpcError::InvalidRange);
        }
        Ok(())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct RangeResponseV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub status: U16,
    pub snapshot: SnapshotId,
    pub source: SourceCorrelation,
    pub version_digest: ContentDigest,
    pub bulk_id: [u8; 16],
    pub file_len: U32,
    pub range_index: U16,
    pub range_count: U16,
    pub proof_depth: u8,
    pub reserved: [u8; 3],
    pub siblings: [[u8; 32]; MAX_RANGE_PROOF_DEPTH],
}

impl RangeResponseV1 {
    pub fn validate(&self) -> Result<WorkspaceStatus, RpcError> {
        if self.magic != RANGE_RESPONSE_MAGIC {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        if self.reserved != [0; 3] {
            return Err(RpcError::Reserved);
        }
        let status =
            WorkspaceStatus::from_wire(self.status.get()).ok_or(RpcError::InvalidStatus)?;
        if status == WorkspaceStatus::Ok
            && (self.snapshot == [0; 32]
                || self.source == [0; 16]
                || self.version_digest == [0; 32]
                || self.bulk_id == [0; 16]
                || self.range_count.get() == 0
                || self.range_index.get() >= self.range_count.get()
                || usize::from(self.proof_depth) > MAX_RANGE_PROOF_DEPTH
                || self.siblings[usize::from(self.proof_depth)..]
                    .iter()
                    .any(|sibling| *sibling != [0; 32]))
        {
            return Err(RpcError::InvalidRange);
        }
        Ok(status)
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct DeltaPollV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub flags: U16,
    pub snapshot: SnapshotId,
    pub next_sequence: U64,
}

impl DeltaPollV1 {
    pub fn validate(&self) -> Result<(), RpcError> {
        if self.magic != DELTA_POLL_MAGIC
            || self.snapshot == [0; 32]
            || self.next_sequence.get() == 0
        {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        if self.flags.get() != 0 {
            return Err(RpcError::Reserved);
        }
        Ok(())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct DeltaResponseV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub status: U16,
    pub parent: SnapshotId,
    pub next: SnapshotId,
    pub sequence: U64,
    pub delta_id: ContentDigest,
    pub origin_transaction: ContentDigest,
    pub change_count: U16,
    pub reserved: U16,
}

impl DeltaResponseV1 {
    pub fn validate(&self) -> Result<WorkspaceStatus, RpcError> {
        if self.magic != DELTA_RESPONSE_MAGIC || self.reserved.get() != 0 {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        let status =
            WorkspaceStatus::from_wire(self.status.get()).ok_or(RpcError::InvalidStatus)?;
        if status == WorkspaceStatus::Ok
            && (self.parent == [0; 32]
                || self.next == [0; 32]
                || self.sequence.get() == 0
                || self.delta_id == [0; 32]
                || usize::from(self.change_count.get()) > crate::bounds::MAX_CHANGES)
        {
            return Err(RpcError::Malformed);
        }
        Ok(status)
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct DeltaOpRequestV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub index: U16,
    pub sequence: U64,
    pub delta_id: ContentDigest,
}

impl DeltaOpRequestV1 {
    pub fn validate(&self) -> Result<(), RpcError> {
        if self.magic != DELTA_OP_REQUEST_MAGIC
            || self.sequence.get() == 0
            || self.delta_id == [0; 32]
            || usize::from(self.index.get()) >= crate::bounds::MAX_CHANGES
        {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        Ok(())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct WorkspaceErrorV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub status: U16,
    pub request_magic: [u8; 4],
    pub reserved: U32,
}

impl WorkspaceErrorV1 {
    pub fn validate(&self) -> Result<WorkspaceStatus, RpcError> {
        if self.magic != ERROR_RESPONSE_MAGIC || self.reserved.get() != 0 {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        let status =
            WorkspaceStatus::from_wire(self.status.get()).ok_or(RpcError::InvalidStatus)?;
        if status == WorkspaceStatus::Ok {
            return Err(RpcError::InvalidStatus);
        }
        Ok(status)
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct BlobStoredV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub status: U16,
    pub bulk_id: [u8; 16],
    pub digest: ContentDigest,
    pub length: U32,
    pub reserved: U32,
}

impl BlobStoredV1 {
    pub fn validate(&self) -> Result<WorkspaceStatus, RpcError> {
        if self.magic != BLOB_STORED_MAGIC || self.reserved.get() != 0 {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        let status =
            WorkspaceStatus::from_wire(self.status.get()).ok_or(RpcError::InvalidStatus)?;
        if status == WorkspaceStatus::Ok
            && (self.bulk_id == [0; 16]
                || self.digest == [0; 32]
                || self.length.get() == 0
                || u64::from(self.length.get()) > crate::bounds::MAX_FILE_BYTES)
        {
            return Err(RpcError::Malformed);
        }
        Ok(status)
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct TransactionResultV1 {
    pub magic: [u8; 4],
    pub version: U16,
    pub status: U16,
    pub transaction: ContentDigest,
    pub changes: ContentDigest,
    pub snapshot: SnapshotId,
    pub applied: U16,
    pub reserved: U16,
}

impl TransactionResultV1 {
    pub fn validate(&self) -> Result<WorkspaceStatus, RpcError> {
        if self.magic != TRANSACTION_RESULT_MAGIC
            || self.reserved.get() != 0
            || self.transaction == [0; 32]
            || self.changes == [0; 32]
        {
            return Err(RpcError::Malformed);
        }
        if self.version.get() != FORMAT_VERSION {
            return Err(RpcError::UnsupportedVersion);
        }
        let status =
            WorkspaceStatus::from_wire(self.status.get()).ok_or(RpcError::InvalidStatus)?;
        if status == WorkspaceStatus::Ok && self.snapshot == [0; 32] {
            return Err(RpcError::Malformed);
        }
        Ok(status)
    }
}

const _: () = {
    assert!(core::mem::size_of::<AttachRequestV1>() <= 512);
    assert!(core::mem::size_of::<AttachResponseV1>() <= 512);
    assert!(core::mem::size_of::<ManifestEntryRequestV1>() <= 512);
    assert!(core::mem::size_of::<ManifestEntryResponseV1>() <= 512);
    assert!(core::mem::size_of::<RangeRequestV1>() <= 512);
    assert!(core::mem::size_of::<RangeResponseV1>() <= 512);
    assert!(core::mem::size_of::<DeltaPollV1>() <= 512);
    assert!(core::mem::size_of::<DeltaResponseV1>() <= 512);
    assert!(core::mem::size_of::<DeltaOpRequestV1>() <= 512);
    assert!(core::mem::size_of::<WorkspaceErrorV1>() <= 512);
    assert!(core::mem::size_of::<BlobStoredV1>() <= 512);
    assert!(core::mem::size_of::<TransactionResultV1>() <= 512);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_request_checks_overflow_boundary_and_cross_range_geometry() {
        let valid = RangeRequestV1 {
            magic: RANGE_REQUEST_MAGIC,
            version: U16::new(FORMAT_VERSION),
            flags: U16::ZERO,
            snapshot: [1; 32],
            source: [2; 16],
            version_digest: [3; 32],
            range_root: [4; 32],
            file_len: U32::new(8192),
            offset: U32::new(4090),
            length: U16::new(6),
            retry: 0,
            reserved: 0,
        };
        assert_eq!(valid.validate(), Ok(()));
        let crossing = RangeRequestV1 {
            length: U16::new(7),
            ..valid
        };
        assert_eq!(crossing.validate(), Err(RpcError::InvalidRange));
        let overflow = RangeRequestV1 {
            file_len: U32::new(u32::MAX),
            offset: U32::new(u32::MAX),
            length: U16::new(1),
            ..valid
        };
        assert_eq!(overflow.validate(), Err(RpcError::InvalidRange));
    }

    #[test]
    fn reserved_and_unknown_discriminants_fail_exactly() {
        let request = AttachRequestV1 {
            magic: ATTACH_MAGIC,
            version: U16::new(FORMAT_VERSION),
            mode: WorkspaceMode::Live as u8,
            flags: 0,
            policy_digest: [1; 32],
            known_snapshot: [0; 32],
        };
        assert_eq!(request.validate(), Ok(WorkspaceMode::Live));
        assert_eq!(
            AttachRequestV1 { mode: 9, ..request }.validate(),
            Err(RpcError::InvalidMode)
        );
        assert_eq!(
            AttachRequestV1 {
                flags: 1,
                ..request
            }
            .validate(),
            Err(RpcError::Reserved)
        );
    }
}
