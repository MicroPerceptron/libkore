//! Frozen v1 wire records from `res/AGENT_WORKFLOW.md`.

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub const HOSTLINK_MAGIC: u32 = u32::from_le_bytes(*b"KHL1");
pub const SCHEMA_MAJOR_V1: u16 = 1;
pub const FAMILY_LINK: u16 = 1;

pub const CONTROL_FRAME_MAX: usize = 64 * 1024;
pub const INLINE_MAX: usize = 4 * 1024;
pub const BULK_CHUNK_MAX: usize = 1024 * 1024;
/// The bounded founded-service profile aggregates 512-byte IPC fragments into
/// at most one 4 KiB HostLink wire chunk. Both peers use this shared limit so
/// a host response cannot exceed the guest's allocation-free staging bank.
pub const FOUNDED_BULK_WIRE_CHUNK: usize = 4 * 1024;
pub const BULK_FRAME_MAX: usize = ENVELOPE_LEN + BULK_CHUNK_MAX;
pub const BULK_OBJECT_MAX: usize = 256 * 1024 * 1024;
pub const BULK_CHUNKS_MAX: u16 = 256;

macro_rules! opaque_id {
    ($name:ident) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            Eq,
            FromBytes,
            Immutable,
            IntoBytes,
            KnownLayout,
            PartialEq,
            Unaligned,
        )]
        #[repr(transparent)]
        pub struct $name(pub [u8; 16]);

        impl $name {
            pub const ZERO: Self = Self([0; 16]);

            #[must_use]
            pub const fn is_zero(self) -> bool {
                let mut i = 0;
                while i < self.0.len() {
                    if self.0[i] != 0 {
                        return false;
                    }
                    i += 1;
                }
                true
            }
        }
    };
}

opaque_id!(LinkEpoch);
opaque_id!(RouteId);
opaque_id!(RequestId);
opaque_id!(BulkId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskGeneration(u64);

impl TaskGeneration {
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The common control envelope.  Its byte size is part of the HostLink v1 ABI.
#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct EnvelopeV1 {
    pub magic: U32,
    pub family: U16,
    pub schema_major: U16,
    pub schema_minor: U16,
    pub kind: U16,
    pub flags: U32,
    pub link_epoch: LinkEpoch,
    pub route_id: RouteId,
    pub task_generation: U64,
    pub sequence: U64,
    pub request_id: RequestId,
    pub payload_len: U32,
    pub reserved: U32,
}

/// Host-order fields used to construct one frozen v1 envelope. Grouping the
/// correlated routing identifiers makes call sites auditable without a long
/// positional-argument list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnvelopeFields {
    pub family: u16,
    pub schema_minor: u16,
    pub kind: u16,
    pub epoch: LinkEpoch,
    pub route: RouteId,
    pub generation: u64,
    pub sequence: u64,
    pub request: RequestId,
    pub payload_len: u32,
}

pub const ENVELOPE_LEN: usize = core::mem::size_of::<EnvelopeV1>();
const _: () = assert!(ENVELOPE_LEN == 88);

impl EnvelopeV1 {
    #[must_use]
    pub fn new(fields: EnvelopeFields) -> Self {
        Self {
            magic: U32::new(HOSTLINK_MAGIC),
            family: U16::new(fields.family),
            schema_major: U16::new(SCHEMA_MAJOR_V1),
            schema_minor: U16::new(fields.schema_minor),
            kind: U16::new(fields.kind),
            flags: U32::ZERO,
            link_epoch: fields.epoch,
            route_id: fields.route,
            task_generation: U64::new(fields.generation),
            sequence: U64::new(fields.sequence),
            request_id: fields.request,
            payload_len: U32::new(fields.payload_len),
            reserved: U32::ZERO,
        }
    }

    #[must_use]
    pub fn frame_len(&self) -> Option<usize> {
        ENVELOPE_LEN.checked_add(self.payload_len.get() as usize)
    }
}

/// A control-plane reference to a separately credited bulk object.
#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
pub struct BulkRefV1 {
    pub bulk_id: BulkId,
    pub total_len: U32,
    pub chunk_count: U16,
    pub content_kind: U16,
    pub digest_blake3: [u8; 32],
}

const _: () = assert!(core::mem::size_of::<BulkRefV1>() == 56);

impl BulkRefV1 {
    #[must_use]
    pub fn shape_is_valid(&self) -> bool {
        let total = self.total_len.get() as usize;
        let chunks = self.chunk_count.get();
        !self.bulk_id.is_zero()
            && total <= BULK_OBJECT_MAX
            && chunks <= BULK_CHUNKS_MAX
            && ((total == 0 && chunks == 0) || (total != 0 && chunks != 0))
    }
}

/// Derive the link epoch from both endpoint freshness values and the exact
/// negotiated transcript.  Domain separation prevents using this digest as a
/// content hash by accident.
#[must_use]
pub fn derive_link_epoch(
    host_nonce: &[u8; 32],
    guest_boot: &[u8; 32],
    negotiated_transcript: &[u8],
) -> LinkEpoch {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"kore.hostlink.epoch.v1\0");
    hasher.update(host_nonce);
    hasher.update(guest_boot);
    hasher.update(&(negotiated_transcript.len() as u64).to_le_bytes());
    hasher.update(negotiated_transcript);
    let digest = hasher.finalize();
    let mut epoch = [0; 16];
    epoch.copy_from_slice(&digest.as_bytes()[..16]);
    LinkEpoch(epoch)
}

/// Derive the shared-window incarnation from the authenticated link epoch.
///
/// The shared plane deliberately has a smaller 64-bit reset stamp; domain
/// separation prevents treating either representation as the other. A zero
/// link epoch has no live shared incarnation, and an extraordinarily unlikely
/// all-zero digest is mapped to one because the memory layout reserves zero.
#[must_use]
pub fn derive_shmem_epoch(epoch: LinkEpoch) -> Option<u64> {
    if epoch.is_zero() {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"kore.hostlink.shmem-epoch.v1\0");
    hasher.update(&epoch.0);
    let digest = hasher.finalize();
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&digest.as_bytes()[..8]);
    Some(match u64::from_le_bytes(raw) {
        0 => 1,
        value => value,
    })
}
