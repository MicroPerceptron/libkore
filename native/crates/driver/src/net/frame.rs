//! **Frames on the wire and descriptors in a ring.**
//!
//! The two things a frame service moves: the Ethernet II header it must read
//! to classify a frame, and the fixed-size record that names a frame's slot
//! to the peer across a shared window.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::addr::{EtherType, MacAddr};

/// Bytes of Ethernet II header before the payload: destination, source, type.
pub const HEADER_LEN: usize = 14;

/// Largest frame this service carries — Ethernet II header plus a 1500-byte
/// MTU.  An **architectural constant**: it fixes the window's slot size, so
/// it is not a runtime knob.
pub const FRAME_MAX: usize = HEADER_LEN + 1500;

/// Smallest frame that can carry a header at all.  Anything shorter cannot be
/// classified, so it is refused rather than guessed at.
pub const FRAME_MIN: usize = HEADER_LEN;

/// The parsed head of an Ethernet II frame.
///
/// Only the header is decoded; the payload stays bytes.  This service is a
/// **frame** service, and every temptation to look further into the payload
/// is the beginning of a protocol stack living in the wrong place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHead {
    /// Destination address — what the classifier matches a vif filter against.
    pub dst: MacAddr,
    /// Source address.
    pub src: MacAddr,
    /// The type field.  A value below [`EtherType::MIN`] is an 802.3 length,
    /// not a protocol selector; [`FrameHead::parse`] still reports it, and
    /// filters simply will not match it.
    pub ethertype: EtherType,
}

impl FrameHead {
    /// Decode the header, or `None` when there are not [`HEADER_LEN`] bytes.
    ///
    /// Refused rather than zero-filled: a frame too short to classify must
    /// not be delivered to a vif that would then read a fabricated header.
    pub fn parse(frame: &[u8]) -> Option<Self> {
        if frame.len() < HEADER_LEN {
            return None;
        }
        let mut dst = [0u8; 6];
        let mut src = [0u8; 6];
        dst.copy_from_slice(&frame[..6]);
        src.copy_from_slice(&frame[6..12]);
        Some(Self {
            dst: MacAddr(dst),
            src: MacAddr(src),
            ethertype: EtherType::from_wire([frame[12], frame[13]]),
        })
    }

    /// Write this header into the first [`HEADER_LEN`] bytes of `out`.
    pub fn write(&self, out: &mut [u8]) -> Option<()> {
        if out.len() < HEADER_LEN {
            return None;
        }
        out[..6].copy_from_slice(&self.dst.0);
        out[6..12].copy_from_slice(&self.src.0);
        out[12..14].copy_from_slice(&self.ethertype.to_wire());
        Some(())
    }
}

/// Flags on a [`FrameDesc`].
pub mod flags {
    /// The frame was longer than the slot and was **not** delivered whole.
    /// A receiver must treat the slot as unusable rather than parse a prefix.
    pub const TRUNCATED: u16 = 1 << 0;
    /// The destination was the broadcast address.
    pub const BROADCAST: u16 = 1 << 1;
    /// The destination was a multicast group (broadcast excluded).
    pub const MULTICAST: u16 = 1 << 2;
}

/// One frame's descriptor on a vif's ring — 8 bytes.
///
/// # There is no slot field, deliberately
///
/// A descriptor's slot is **derived, not stored**: the producer's *n*-th push
/// occupies slot `n mod depth`, because the ring and the slot subregion have
/// the same depth and the same order.  Each side tracks only its own
/// monotonic count.  Storing the index as well would create a second source
/// of truth that a hostile or buggy peer could disagree with, and the
/// receiver would then have to decide which to believe.  Deriving it means
/// there is nothing to disagree about.
///
/// The record reinterprets from **any** bytes: the window is shared with an
/// untrusted peer, so every field is validated at use rather than trusted at
/// read (`len` against the slot size, `ethertype` against the vif's filter).
#[repr(C)]
// Not `Unaligned`: the fields are `u16`, and the ring's slot area is
// 64-byte aligned with an 8-byte record, so every descriptor lands naturally
// aligned by construction.  `RingRecord` does not ask for it.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable,
)]
pub struct FrameDesc {
    /// Frame length in bytes, including the header.  Validated against
    /// [`FRAME_MIN`]/[`FRAME_MAX`] and the slot size before any use.
    pub len: u16,
    /// [`flags`].
    pub flags: u16,
    /// The frame's EtherType, host-native — carried so a receiver can
    /// dispatch without re-reading the header, and so a transmitter states
    /// its intent where the service can check it against the vif's filter
    /// instead of inferring it from bytes the peer also wrote.
    pub ethertype: u16,
    /// Reserved; must be zero.  Frozen additive headroom.
    pub reserved: u16,
}

impl FrameDesc {
    /// A descriptor for a frame of `len` bytes carrying `ethertype`.
    pub fn new(len: u16, ethertype: EtherType) -> Self {
        Self {
            len,
            flags: 0,
            ethertype: ethertype.0,
            reserved: 0,
        }
    }

    /// The declared EtherType.
    pub fn ethertype(&self) -> EtherType {
        EtherType(self.ethertype)
    }

    /// Whether the declared length can be a frame at all.
    pub fn len_is_sane(&self) -> bool {
        (self.len as usize) >= FRAME_MIN && (self.len as usize) <= FRAME_MAX
    }
}
