//! The **storage transport** wire vocabulary (`res/STORAGE_TRANSPORT.md` §4).
//!
//! A storage client talks to the block row through one client-funded window:
//! a submission channel per [`Class`], one completion channel, and a page of
//! [`FenceRecord`]s. Entries are the io_uring records of [`crate::uring`],
//! read with the field meanings below; nothing on this wire is a capability
//! (a [`GrantId`] is a generation-checked index into grants the ring's
//! authority installed, never authority itself).
//!
//! | `IoUringSqe` field | meaning |
//! |---|---|
//! | `opcode` | [`op::READ_FIXED`], [`op::WRITE_FIXED`], [`op::FSYNC`], [`op::DISCARD`] |
//! | `fd` | [`GrantId`] — slot + generation; a stale id fails closed |
//! | `off` | byte offset within the grant (block-aligned) |
//! | `addr` | byte offset within the target — never a pointer |
//! | `len` | bytes (a block multiple) |
//! | `buf_index` | the target slot |
//! | `op_flags` | [`RWF_DSYNC`] (force unit access), [`RWF_NOWAIT`] |
//! | `personality` | timeline `1..=`[`TIMELINES`]; 0 = completion entry only |
//! | `addr3` | absolute uptime deadline in ns; 0 = the class default |
//!
//! A completion's `res` is the byte count, or a negated [`StorStatus`].
//!
//! [`op::READ_FIXED`]: crate::uring::op::READ_FIXED
//! [`op::WRITE_FIXED`]: crate::uring::op::WRITE_FIXED
//! [`op::FSYNC`]: crate::uring::op::FSYNC
//! [`op::DISCARD`]: crate::uring::op::DISCARD

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Submission depth of the Demand and Stream channels.
pub const SQ_DEPTH: u32 = 128;
/// Submission depth of the Commit channel (the filesystem's metadata path).
pub const COMMIT_SQ_DEPTH: u32 = 32;
/// Completion depth. The row pops a submission only while it holds a
/// completion credit, so this bounds what may be in flight plus unreaped —
/// it need not equal the sum of the submission depths.
pub const CQ_DEPTH: u32 = 256;
/// Fence records per window.
pub const TIMELINES: usize = 16;
/// The fence page: its own 4 KiB page so it can be subleased to a device
/// that waits on a monotone word (`res/RESIDENCY.md` §6).
pub const FENCE_PAGE_BYTES: usize = 4096;

const _: () = assert!(SQ_DEPTH.is_power_of_two());
const _: () = assert!(COMMIT_SQ_DEPTH.is_power_of_two());
const _: () = assert!(CQ_DEPTH.is_power_of_two());
const _: () = assert!(TIMELINES * core::mem::size_of::<FenceRecord>() <= FENCE_PAGE_BYTES);

/// `op_flags`: force unit access — the write is durable when it completes.
/// Linux's `RWF_DSYNC`.
pub const RWF_DSYNC: u32 = 0x0000_0002;
/// `op_flags`: refuse with [`StorStatus::Busy`] rather than wait for class
/// budget. Linux's `RWF_NOWAIT`.
pub const RWF_NOWAIT: u32 = 0x0000_0008;
/// Every `op_flags` bit the transport knows. Any other bit is refused as
/// [`StorStatus::Invalid`], never ignored.
pub const RWF_KNOWN: u32 = RWF_DSYNC | RWF_NOWAIT;

/// The service class a submission channel carries.
///
/// Each class has its own channel in the window and its own hardware
/// submission queue where the controller offers enough, so no class waits
/// behind another's backlog (`res/STORAGE_TRANSPORT.md` §3).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// Interactive reads a task is blocked on.
    Demand = 0,
    /// Bulk and read-ahead traffic — inference weights, prefetch.
    Stream = 1,
    /// The filesystem's metadata writes, flushes and force-unit-access.
    Commit = 2,
}

impl Class {
    /// Every class, in channel order.
    pub const ALL: [Self; 3] = [Self::Demand, Self::Stream, Self::Commit];
    /// Number of classes.
    pub const COUNT: usize = 3;

    /// The class a channel index names.
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Demand),
            1 => Some(Self::Stream),
            2 => Some(Self::Commit),
            _ => None,
        }
    }

    /// This class's channel index.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Submission depth of this class's channel.
    pub const fn depth(self) -> u32 {
        match self {
            Self::Demand | Self::Stream => SQ_DEPTH,
            Self::Commit => COMMIT_SQ_DEPTH,
        }
    }
}

/// A grant as a submission names it: slot in bits 0..16, generation in
/// bits 16..31. Bit 31 is always clear, so the id is a non-negative `fd`.
///
/// Generation-checked, never authority: a forged id reaches only grants the
/// ring's authority installed into that same ring, and a revoked slot's
/// successor carries a different generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GrantId(u32);

impl GrantId {
    /// Largest generation the id carries (15 bits).
    pub const MAX_GENERATION: u16 = 0x7fff;

    /// Compose from a slot and a generation (masked to 15 bits).
    pub const fn new(slot: u16, generation: u16) -> Self {
        Self(slot as u32 | (((generation & Self::MAX_GENERATION) as u32) << 16))
    }

    /// Decode a submission's `fd`; `None` for a negative one.
    pub const fn from_fd(fd: i32) -> Option<Self> {
        if fd < 0 {
            None
        } else {
            Some(Self(fd as u32))
        }
    }

    /// The `fd` a submission carries.
    pub const fn fd(self) -> i32 {
        self.0 as i32
    }

    /// The grant slot.
    pub const fn slot(self) -> u16 {
        self.0 as u16
    }

    /// The slot's generation.
    pub const fn generation(self) -> u16 {
        (self.0 >> 16) as u16 & Self::MAX_GENERATION
    }
}

/// Why a submission failed. A completion carries it negated in `res`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorStatus {
    /// A range fell outside its grant or target.
    Range = 1,
    /// An offset or length was not a whole number of blocks.
    Alignment = 2,
    /// The grant does not carry the right the operation needs.
    Rights = 3,
    /// The grant is being or has been revoked.
    Revoked = 4,
    /// The grant or target id names nothing live.
    Stale = 5,
    /// The deadline passed before the device let go of the command.
    Deadline = 6,
    /// Cancelled before it was issued.
    Cancelled = 7,
    /// The medium reported an error.
    Media = 8,
    /// The device was lost; nothing it held completed.
    Lost = 9,
    /// The operation is not offered on this ring.
    Unsupported = 10,
    /// [`RWF_NOWAIT`] and the class had no budget.
    Busy = 11,
    /// The class is above the grant's ceiling or the ring's policy.
    Class = 12,
    /// A field the transport does not accept (an unknown flag or opcode).
    Invalid = 13,
}

impl StorStatus {
    /// The negated form a completion's `res` carries.
    pub const fn res(self) -> i32 {
        -(self as i32)
    }

    /// Decode a completion's `res`; `None` for a success (`res >= 0`) or an
    /// unknown code.
    pub const fn from_res(res: i32) -> Option<Self> {
        Some(match res {
            -1 => Self::Range,
            -2 => Self::Alignment,
            -3 => Self::Rights,
            -4 => Self::Revoked,
            -5 => Self::Stale,
            -6 => Self::Deadline,
            -7 => Self::Cancelled,
            -8 => Self::Media,
            -9 => Self::Lost,
            -10 => Self::Unsupported,
            -11 => Self::Busy,
            -12 => Self::Class,
            -13 => Self::Invalid,
            _ => return None,
        })
    }
}

/// One timeline's monotone completion record, in the window's fence page.
///
/// Sequences are implicit: the k-th submission naming timeline `t`, counted
/// by both sides from 1, has sequence k. `done` is the highest k such that
/// every sequence at or below it has completed; `fault` is the first failed
/// sequence (0 = none) and is sticky. The row stores `fault` before `done`,
/// both with release ordering, so a client that reads `done` with acquire
/// ordering and then `fault` never misses a fault at or below `done`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct FenceRecord {
    /// Contiguous completed prefix.
    pub done: u64,
    /// First failed sequence, or 0.
    pub fault: u64,
}

const _: () = assert!(core::mem::size_of::<FenceRecord>() == 16);

/// The level state of a submission channel. Reserved: the client publishes
/// nothing level-shaped today.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct SqState {
    /// Reserved, zero.
    pub reserved: u64,
}

/// The ring is serving.
pub const RING_LIVE: u32 = 0;
/// The ring was detached: nothing more is popped; in-flight work still
/// completes.
pub const RING_DEPARTED: u32 = 1;
/// The device behind the ring was lost; every submission answers
/// [`StorStatus::Lost`].
pub const RING_LOST: u32 = 2;

/// The level state of the completion channel, republished by the row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct CqState {
    /// [`RING_LIVE`], [`RING_DEPARTED`] or [`RING_LOST`].
    pub status: u32,
    /// Submissions accepted and not yet completed.
    pub in_flight: u32,
    /// Completions posted since attach.
    pub completed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grant_ids_round_trip_through_fd_and_stay_non_negative() {
        for (slot, generation) in [(0, 0), (7, 1), (u16::MAX, GrantId::MAX_GENERATION)] {
            let id = GrantId::new(slot, generation);
            assert!(id.fd() >= 0);
            let back = GrantId::from_fd(id.fd()).unwrap();
            assert_eq!((back.slot(), back.generation()), (slot, generation));
        }
        // A generation past 15 bits wraps inside its field, never into bit 31.
        assert!(GrantId::new(1, u16::MAX).fd() >= 0);
        assert_eq!(GrantId::from_fd(-1), None);
    }

    #[test]
    fn status_codes_round_trip_negated() {
        for status in [
            StorStatus::Range,
            StorStatus::Alignment,
            StorStatus::Rights,
            StorStatus::Revoked,
            StorStatus::Stale,
            StorStatus::Deadline,
            StorStatus::Cancelled,
            StorStatus::Media,
            StorStatus::Lost,
            StorStatus::Unsupported,
            StorStatus::Busy,
            StorStatus::Class,
            StorStatus::Invalid,
        ] {
            assert!(status.res() < 0);
            assert_eq!(StorStatus::from_res(status.res()), Some(status));
        }
        assert_eq!(StorStatus::from_res(0), None);
        assert_eq!(StorStatus::from_res(4096), None);
        assert_eq!(StorStatus::from_res(-99), None);
    }

    #[test]
    fn classes_index_their_channels() {
        for (index, class) in Class::ALL.into_iter().enumerate() {
            assert_eq!(class.index(), index);
            assert_eq!(Class::from_index(index), Some(class));
        }
        assert_eq!(Class::from_index(Class::COUNT), None);
        assert_eq!(Class::Commit.depth(), COMMIT_SQ_DEPTH);
    }
}
