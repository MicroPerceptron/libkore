//! **The build-identity block** — the seventh frozen boot-page block, after
//! the machine block (`kore-rt::entry::IDENTITY_BLOCK_AT`).
//!
//! Which build the running kernel is: the Makefile's `KORE_BUILD_STAMP`, a
//! twelve-digit commit plus `-dirty` for an uncommitted tree. The kernel
//! paints it at boot, but no console pixel survives the display handoff
//! (`res/NETWORK_SERVICE.md`). Operators' screenshots were rejected again
//! and again for lacking it. `status sys` and the screenshot writer (#1414)
//! read it here.
//!
//! The stamp names the whole image, not only the kernel: the tree at that
//! commit pins the sealed service ELFs and archives by digest
//! (`res/boot.toml`), and `-dirty` says the image may not be exactly a
//! commit.
//!
//! **Kernel testimony, like the founding and machine blocks.** The kernel
//! writes it at the founding site: a userland relayer could claim any build.
//! It is a compile-time constant, so a plain copy with no seqlock. A zeroed
//! region reads `abi == 0` and exposes nothing, so a page from a founder
//! that predates the block fails closed.

use zerocopy::byteorder::little_endian::U16;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Layout version. A zeroed block reads `abi == 0` and exposes nothing.
pub const IDENTITY_ABI: u16 = 1;

/// The longest stamp the block carries. The Makefile's is at most 18 bytes
/// (`git rev-parse --short=12` and `-dirty`); a kernel stamped with a longer
/// one refuses to build rather than testify to a truncated name.
pub const STAMP_BYTES: usize = 28;

/// The build-identity block — kernel-stamped per spawn. 32 bytes, frozen
/// ABI: evolution is additive into a new layout version, never a reorder.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct IdentityBlock {
    /// Layout version ([`IDENTITY_ABI`]); `0` = no record (fail closed).
    pub abi: U16,
    /// How many bytes of [`stamp`](Self::stamp) are the stamp.
    pub len: U16,
    /// The build stamp, printable ASCII, zero-padded.
    pub stamp: [u8; STAMP_BYTES],
}

impl IdentityBlock {
    /// The block for `stamp`, or `None` if it is empty, longer than
    /// [`STAMP_BYTES`] or not printable ASCII. `const`, so a kernel builds
    /// its own block at compile time and refuses a stamp it cannot carry.
    pub const fn new(stamp: &str) -> Option<Self> {
        let bytes = stamp.as_bytes();
        if bytes.is_empty() || bytes.len() > STAMP_BYTES {
            return None;
        }
        let mut out = [0u8; STAMP_BYTES];
        let mut i = 0;
        while i < bytes.len() {
            if !printable(bytes[i]) {
                return None;
            }
            out[i] = bytes[i];
            i += 1;
        }
        Some(Self {
            abi: U16::from_bytes(IDENTITY_ABI.to_le_bytes()),
            len: U16::from_bytes((bytes.len() as u16).to_le_bytes()),
            stamp: out,
        })
    }

    /// The build stamp, or `None` on an unstamped or malformed block.
    pub fn stamp(&self) -> Option<&str> {
        if self.abi.get() != IDENTITY_ABI {
            return None;
        }
        let stamp = self.stamp.get(..usize::from(self.len.get()))?;
        if stamp.is_empty() || !stamp.iter().all(|&b| printable(b)) {
            return None;
        }
        core::str::from_utf8(stamp).ok()
    }
}

/// Printable ASCII: what a build stamp is made of.
const fn printable(byte: u8) -> bool {
    byte >= 0x20 && byte <= 0x7e
}

const _: () = assert!(
    core::mem::size_of::<IdentityBlock>() == 32,
    "the identity block is frozen ABI"
);

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::FromZeros;

    #[test]
    fn a_stamp_reads_back_as_written() {
        for stamp in ["304bfef9ca08", "304bfef9ca08-dirty", "nostamp", &"x".repeat(STAMP_BYTES)] {
            let block = IdentityBlock::new(stamp).unwrap();
            assert_eq!(block.stamp(), Some(stamp));
            let (read, _) = IdentityBlock::read_from_prefix(block.as_bytes()).unwrap();
            assert_eq!(read.stamp(), Some(stamp), "through the wire bytes");
        }
    }

    #[test]
    fn a_stamp_it_cannot_carry_is_refused() {
        let long = "x".repeat(STAMP_BYTES + 1);
        for stamp in ["", long.as_str(), "tab\there", "café"] {
            assert!(IdentityBlock::new(stamp).is_none(), "{stamp:?}");
        }
    }

    /// A zeroed region, an unknown layout, an overlong length or a stray
    /// byte each read as no record: every one fails closed.
    #[test]
    fn a_damaged_or_absent_block_exposes_nothing() {
        assert_eq!(IdentityBlock::new_zeroed().stamp(), None);
        let good = IdentityBlock::new("304bfef9ca08").unwrap();
        let mut future = good;
        future.abi = U16::new(IDENTITY_ABI + 1);
        let mut overlong = good;
        overlong.len = U16::new(STAMP_BYTES as u16 + 1);
        let mut stray = good;
        stray.stamp[3] = 0x07;
        let mut empty = good;
        empty.len = U16::new(0);
        for block in [future, overlong, stray, empty] {
            assert_eq!(block.stamp(), None);
        }
    }
}
