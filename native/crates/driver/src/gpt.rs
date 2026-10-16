//! # `gpt` — **read-only** GPT discovery
//!
//! How a block driver learns which blocks are its: it reads the machine's
//! GUID Partition Table and finds the partition whose type GUID says
//! "this is kore's".  Nothing here writes, and nothing here is
//! kore-specific except [`KORE_SWAP_TYPE`] — the table is the standard
//! UEFI one, so an ordinary installer or `sgdisk` run creates what this
//! finds, and no future tool has to learn a kore-private on-disk format.
//!
//! **Why this exists.** A driver holding a namespace capability may drive
//! the *controller*; it says nothing about who owns the *blocks*.  A swap
//! store that assumed the whole device is what destroyed a real machine's
//! partition table (`res/DEVICE_SERVICE.md` §8.1) — so the designation
//! must come from outside, and a partition table is how humans and
//! installers already express exactly that.
//!
//! **The parsing is not hand-rolled.**  The header and entry layouts,
//! their endianness, and the GUID encoding come from
//! [`gpt_disk_types`] (`no_std`, no-alloc, MIT/Apache-2.0) — the
//! crate-selection rule applied to an on-disk format, for the same reason
//! it applies to hardware register bits: a plausible hand-rolled layout
//! is usually a wrong one, and the mistakes live exactly in the
//! endianness and CRC handling this delegates.
//!
//! **I/O stays with the caller.**  These are pure functions over bytes
//! the caller already read, so discovery is fully host-testable and this
//! module depends on no device, no DMA, and no allocator.

use core::mem::size_of;

use gpt_disk_types::{GptHeader, GptHeaderSignature, GptPartitionEntry, guid};

/// Re-exported so callers can name the partition type-GUID parameter of
/// [`find_partition`] (and hold the `KORE_*_TYPE` designations) without
/// depending on `gpt_disk_types` directly.
pub use gpt_disk_types::Guid;

/// **The kore swap partition type.**
///
/// Deliberately not the Linux swap type
/// (`0657FD6D-A4AB-43C4-84E5-0933C84B4F4F`): on a dual-boot disk the two
/// systems must never believe they own the same blocks — sharing one
/// partition would corrupt a Linux hibernation image, and "it happens to
/// work while only one runs at a time" is not an ownership model.
///
/// Create one with `tools/provision_swap.py`, or by hand:
///
/// ```text
/// sgdisk --new=0:0:+128G \
///        --typecode=0:A64B64C3-AE6E-4348-9776-F7786300FF2B \
///        --change-name=0:"kore swap" /dev/nvme0n1
/// ```
pub const KORE_SWAP_TYPE: Guid = guid!("A64B64C3-AE6E-4348-9776-F7786300FF2B");

/// The partition type GUID that designates **the kore filesystem's
/// region** — the durable filesystem's write authority on a shared disk,
/// exactly as [`KORE_SWAP_TYPE`] designates swap's.  A machine whose GPT
/// names no such partition boots with the filesystem in its bounded RAM
/// posture; it is never an invitation to claim the whole namespace.
///
/// The designation admits the region; *formatting* additionally requires
/// the region's reserved root blocks to be blank, because partitioning
/// tools do not zero old bytes and stale data is evidence, not consent.
/// Create one and blank its head:
///
/// ```text
/// sgdisk --new=0:0:+32G \
///        --typecode=0:D673CABE-2DF1-4F42-8B89-B9E56AC86714 \
///        --change-name=0:"kore fs" /dev/nvme0n1
/// dd if=/dev/zero of=/dev/nvme0n1pN bs=4096 count=2
/// ```
pub const KORE_FS_TYPE: Guid = guid!("D673CABE-2DF1-4F42-8B89-B9E56AC86714");

/// The partition type GUID that designates **the kore SYSTEM store's
/// region** — the build-owned seeded executable store (`app/`) and future
/// provisioned state, held apart from user data (which lives in the
/// [`KORE_FS_TYPE`] region).  Introduced by issue #626 so the two can carry
/// their opposite lifecycle policies: the user region recovers
/// conservatively and is never auto-reformatted, while this region is
/// **self-healing** — blank formats and seeds, an era-stamp match recovers
/// (the seeder's skip-by-name fast path is then correct), and a stamp
/// mismatch or foreign bytes reformat-and-reseed *this region only*.  The
/// store is rebuildable from the boot archive by construction, so a reformat
/// here is never data loss; the era stamp (the archive's identity) is what
/// makes that judgement, killing the frozen-stale-tree hazard.
///
/// A machine whose GPT names no such partition seeds from the archive in the
/// bounded posture (exactly a diskless machine today) and leaves the user
/// region untouched by the store.  Its `RegionDevice` clamp cannot address
/// user bytes — the containment is structural, per the destructive-ops
/// doctrine.  Create one and blank its head:
///
/// ```text
/// sgdisk --new=0:0:+256M \
///        --typecode=0:D9991ABE-2DF1-409E-8416-E05280489E79 \
///        --change-name=0:"kore system" /dev/nvme0n1
/// dd if=/dev/zero of=/dev/nvme0n1pN bs=4096 count=2
/// ```
pub const KORE_SYSTEM_TYPE: Guid = guid!("D9991ABE-2DF1-409E-8416-E05280489E79");

/// The partition type GUID that designates **a kore pack region** — an
/// immutable, installer-written image (model weights and similar bulk
/// read-mostly data) that the filesystem row streams at queue depth into
/// client-funded memory without a filesystem in between.
///
/// The designation grants *read* authority only: the pack lane's
/// `RegionDevice` clamp is read-only by construction, so no kore code path
/// can write, discard, or format these blocks. The layout inside the region
/// belongs to whoever produced the image; kore sees block ranges. Write one
/// from another OS and leave it:
///
/// ```text
/// sgdisk --new=0:0:+24G \
///        --typecode=0:55287D31-3555-44E8-A6E8-F9F74A225290 \
///        --change-name=0:"kore pack" /dev/nvme0n1
/// dd if=model.pack of=/dev/nvme0n1pN bs=4M oflag=direct
/// ```
pub const KORE_PACK_TYPE: Guid = guid!("55287D31-3555-44E8-A6E8-F9F74A225290");

/// The LBA the primary GPT header always occupies (LBA 0 is the
/// protective MBR).
pub const PRIMARY_HEADER_LBA: u64 = 1;

/// The smallest header the spec admits; a shorter one is malformed.
const MIN_HEADER_SIZE: u32 = 92;

/// Byte offset of `header_crc32` within the header — zeroed while the
/// header's own checksum is computed (the spec's rule).
const HEADER_CRC_OFFSET: usize = 16;

/// The largest entry array this reads, in bytes.  The spec's minimum
/// reservation is 16 KiB (128 entries × 128 bytes) and real tables use
/// exactly that; a table claiming more than this is refused rather than
/// partially scanned, because a partial scan could miss the very
/// partition that says "these blocks are not yours".
pub const MAX_ENTRY_ARRAY_BYTES: usize = 32 * 1024;

/// Why discovery refused.  Every arm is a reason to touch **no** blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptError {
    /// The block carries no `EFI PART` signature — an unpartitioned
    /// device, an MBR-only device, or not the header block.
    NotGpt,
    /// The header is a revision this does not parse.
    BadRevision,
    /// The header's own CRC32 does not match: the table is damaged.
    BadHeaderCrc,
    /// The entry array's CRC32 does not match the header's claim.
    BadEntryCrc,
    /// A field is self-inconsistent — a too-small header, an entry size
    /// below the spec floor, an array larger than
    /// [`MAX_ENTRY_ARRAY_BYTES`], or a partition outside the usable range.
    BadGeometry,
    /// The table is valid and holds no partition of the requested type:
    /// this machine has not designated one.  A hardware fact, not a fault.
    NoMatch,
}

/// A validated GPT header's facts, in host order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableGeometry {
    /// First LBA of the partition entry array.
    pub entry_lba: u64,
    /// Number of entries the array holds.
    pub entry_count: u32,
    /// Size of one entry, in bytes.
    pub entry_size: u32,
    /// CRC32 the header claims for the entry array.
    pub entry_array_crc32: u32,
    /// First LBA usable by partition data.
    pub first_usable_lba: u64,
    /// Last LBA usable by partition data.
    pub last_usable_lba: u64,
}

impl TableGeometry {
    /// Bytes the entry array occupies — what the caller must read.
    pub fn entry_array_bytes(&self) -> usize {
        self.entry_count as usize * self.entry_size as usize
    }
}

/// One discovered partition's extent, in LBAs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    /// First block, inclusive.
    pub start_lba: u64,
    /// Last block, **inclusive** (the GPT convention).
    pub end_lba: u64,
}

impl Extent {
    /// Length in blocks.
    pub fn blocks(&self) -> u64 {
        self.end_lba - self.start_lba + 1
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    // The IEEE 802.3 polynomial the GPT spec names — the same one zlib
    // and PNG use.  Delegated rather than hand-rolled for the reason the
    // module docs give.
    const ALGO: crc::Crc<u32> = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);
    ALGO.checksum(bytes)
}

/// **Parse and validate the primary GPT header** out of the block read at
/// [`PRIMARY_HEADER_LBA`].
///
/// Validates signature, revision, self-consistent sizes, and the header's
/// own CRC32 — so a caller that gets `Ok` is reading a table the firmware
/// and every other OS on the machine also consider valid.
pub fn parse_header(block: &[u8]) -> Result<TableGeometry, GptError> {
    if block.len() < size_of::<GptHeader>() {
        return Err(GptError::BadGeometry);
    }
    let header: GptHeader = bytemuck::pod_read_unaligned(&block[..size_of::<GptHeader>()]);
    if header.signature != GptHeaderSignature::EFI_COMPATIBLE_PARTITION_TABLE_HEADER {
        return Err(GptError::NotGpt);
    }
    if header.revision != gpt_disk_types::GptHeaderRevision::VERSION_1_0 {
        return Err(GptError::BadRevision);
    }
    let header_size = header.header_size.to_u32();
    if header_size < MIN_HEADER_SIZE || header_size as usize > block.len() {
        return Err(GptError::BadGeometry);
    }

    // The header's checksum covers `header_size` bytes with its own
    // `header_crc32` field taken as four zero bytes (the spec's rule).
    let claimed = header.header_crc32.0.to_u32();
    let mut scratch = [0u8; MIN_HEADER_SIZE as usize];
    let covered = header_size as usize;
    if covered > scratch.len() {
        // A header longer than the spec's fixed part carries only
        // reserved bytes; refuse rather than checksum a prefix.
        return Err(GptError::BadGeometry);
    }
    scratch[..covered].copy_from_slice(&block[..covered]);
    scratch[HEADER_CRC_OFFSET..HEADER_CRC_OFFSET + 4].fill(0);
    if crc32(&scratch[..covered]) != claimed {
        return Err(GptError::BadHeaderCrc);
    }

    let entry_size = header.size_of_partition_entry.to_u32();
    let entry_count = header.number_of_partition_entries.to_u32();
    if (entry_size as usize) < size_of::<GptPartitionEntry>() || !entry_size.is_multiple_of(8) {
        return Err(GptError::BadGeometry);
    }
    let array_bytes = entry_count as usize * entry_size as usize;
    if array_bytes == 0 || array_bytes > MAX_ENTRY_ARRAY_BYTES {
        return Err(GptError::BadGeometry);
    }
    let first_usable_lba = header.first_usable_lba.to_u64();
    let last_usable_lba = header.last_usable_lba.to_u64();
    if first_usable_lba == 0 || last_usable_lba < first_usable_lba {
        return Err(GptError::BadGeometry);
    }

    Ok(TableGeometry {
        entry_lba: header.partition_entry_lba.to_u64(),
        entry_count,
        entry_size,
        entry_array_crc32: header.partition_entry_array_crc32.0.to_u32(),
        first_usable_lba,
        last_usable_lba,
    })
}

/// **Find the first partition of `type_guid`** in the entry array the
/// caller read at [`TableGeometry::entry_lba`].
///
/// Verifies the array's CRC32 against the header's claim **before**
/// trusting any entry: an array that does not checksum is a damaged or
/// racing table, and a partition extent read out of one could name blocks
/// belonging to something else entirely.
///
/// The returned extent is additionally bounds-checked against the
/// header's usable range, so a malformed entry cannot hand back blocks
/// that overlap the tables themselves.
pub fn find_partition(
    entries: &[u8],
    geom: &TableGeometry,
    type_guid: Guid,
) -> Result<Extent, GptError> {
    let array_bytes = geom.entry_array_bytes();
    if entries.len() < array_bytes {
        return Err(GptError::BadGeometry);
    }
    if crc32(&entries[..array_bytes]) != geom.entry_array_crc32 {
        return Err(GptError::BadEntryCrc);
    }

    for index in 0..geom.entry_count as usize {
        let at = index * geom.entry_size as usize;
        let raw = &entries[at..at + size_of::<GptPartitionEntry>()];
        let entry: GptPartitionEntry = bytemuck::pod_read_unaligned(raw);
        // Copied out of the packed struct before comparison: a
        // reference into a 1-byte-aligned field is UB even unread.
        let entry_type = entry.partition_type_guid;
        if entry_type.0 != type_guid {
            continue;
        }
        let start_lba = entry.starting_lba.to_u64();
        let end_lba = entry.ending_lba.to_u64();
        // A matching entry with an impossible extent is refused, never
        // clamped: the table is the authority on which blocks are ours,
        // and a table we cannot read exactly is one we obey by not
        // writing at all.
        if start_lba < geom.first_usable_lba
            || end_lba > geom.last_usable_lba
            || end_lba < start_lba
        {
            return Err(GptError::BadGeometry);
        }
        return Ok(Extent { start_lba, end_lba });
    }
    Err(GptError::NoMatch)
}

/// **Synthetic tables for consumer conformance tests** — the same posture
/// as `kore_nvme::mock`: a test double shipped in the library so a service
/// composing [`find_partition`] can exercise its whole discovery path over
/// a memory device without hand-rolling on-disk GPT layout (which is
/// exactly where the endianness and CRC mistakes live).  Never a
/// production writer: kore *reads* partition tables, it does not author
/// them.
pub mod synthetic {
    use super::{GptHeaderSignature, GptPartitionEntry, Guid, crc32, size_of};
    use gpt_disk_types::{
        Crc32, GptHeader, GptHeaderRevision, GptPartitionAttributes, GptPartitionName,
        GptPartitionType, LbaLe, U32Le, guid,
    };

    /// Entries in the synthetic array — small enough that the array spans
    /// two 512-byte blocks, large enough to prove a real scan.
    pub const ENTRY_COUNT: u32 = 8;
    /// The spec-floor entry stride.
    pub const ENTRY_SIZE: u32 = 128;
    /// Bytes of the synthetic entry array.
    pub const ARRAY_BYTES: usize = (ENTRY_COUNT * ENTRY_SIZE) as usize;

    /// One valid single-partition table for a 512-byte-block device:
    /// the primary header block (for LBA 1, entry array at LBA 2) and the
    /// checksummed entry array holding one `type_guid` partition at
    /// `[start, end]` (inclusive), with the usable window closing at
    /// `last_usable`.
    pub fn table(
        type_guid: Guid,
        start: u64,
        end: u64,
        last_usable: u64,
    ) -> ([u8; 512], [u8; ARRAY_BYTES]) {
        table_pair((type_guid, start, end), None, last_usable)
    }

    /// One valid table carrying two disjoint designations. The optional
    /// second entry exists for consumers such as the filesystem split, where
    /// proving that two independent lookups recover their exact extents is
    /// the conformance question.
    pub fn table_pair(
        first: (Guid, u64, u64),
        second: Option<(Guid, u64, u64)>,
        last_usable: u64,
    ) -> ([u8; 512], [u8; ARRAY_BYTES]) {
        let mut entries = [0u8; ARRAY_BYTES];
        for (index, (type_guid, start, end, unique)) in [
            Some((first.0, first.1, first.2, 1u8)),
            second.map(|(kind, start, end)| (kind, start, end, 2u8)),
        ]
        .into_iter()
        .flatten()
        .enumerate()
        {
            let unique_partition_guid = if unique == 1 {
                guid!("11111111-2222-3333-4444-555555555555")
            } else {
                guid!("AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")
            };
            let entry = GptPartitionEntry {
                partition_type_guid: GptPartitionType(type_guid),
                unique_partition_guid,
                starting_lba: LbaLe::from_u64(start),
                ending_lba: LbaLe::from_u64(end),
                attributes: GptPartitionAttributes::default(),
                name: GptPartitionName::default(),
            };
            // Start at entry index 3: a passing consumer proves it walks the
            // array, and the pair occupies adjacent complete records.
            let at = (index + 3) * ENTRY_SIZE as usize;
            entries[at..at + size_of::<GptPartitionEntry>()]
                .copy_from_slice(bytemuck::bytes_of(&entry));
        }

        let mut header = GptHeader {
            signature: GptHeaderSignature::EFI_COMPATIBLE_PARTITION_TABLE_HEADER,
            revision: GptHeaderRevision::VERSION_1_0,
            header_size: U32Le::from_u32(92),
            header_crc32: Crc32(U32Le::from_u32(0)),
            reserved: U32Le::from_u32(0),
            my_lba: LbaLe::from_u64(super::PRIMARY_HEADER_LBA),
            alternate_lba: LbaLe::from_u64(last_usable + 1),
            first_usable_lba: LbaLe::from_u64(34),
            last_usable_lba: LbaLe::from_u64(last_usable),
            disk_guid: guid!("99999999-8888-7777-6666-555555555555"),
            partition_entry_lba: LbaLe::from_u64(2),
            number_of_partition_entries: U32Le::from_u32(ENTRY_COUNT),
            size_of_partition_entry: U32Le::from_u32(ENTRY_SIZE),
            partition_entry_array_crc32: Crc32(U32Le::from_u32(crc32(&entries))),
        };
        // The header CRC covers the header with its own CRC field zeroed —
        // which it still is here.
        let crc = crc32(&bytemuck::bytes_of(&header)[..92]);
        header.header_crc32 = Crc32(U32Le::from_u32(crc));

        let mut block = [0u8; 512];
        block[..size_of::<GptHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
        (block, entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_disk_types::{
        Crc32, GptHeaderRevision, GptPartitionAttributes, GptPartitionName, GptPartitionType,
        LbaLe, U32Le,
    };

    const BLOCK: usize = 512;
    const ENTRY_COUNT: u32 = 128;
    const ENTRY_SIZE: u32 = 128;

    /// Build a valid single-partition table: the header block and the
    /// entry array, both correctly checksummed.
    fn table(type_guid: Guid, start: u64, end: u64) -> ([u8; BLOCK], [u8; 16384]) {
        let mut entries = [0u8; 16384];
        let entry = GptPartitionEntry {
            partition_type_guid: GptPartitionType(type_guid),
            unique_partition_guid: guid!("11111111-2222-3333-4444-555555555555"),
            starting_lba: LbaLe::from_u64(start),
            ending_lba: LbaLe::from_u64(end),
            attributes: GptPartitionAttributes::default(),
            name: GptPartitionName::default(),
        };
        // The matching entry sits at index 3, so a passing test proves the
        // scan actually walks the array rather than reading entry 0.
        let at = 3 * ENTRY_SIZE as usize;
        entries[at..at + size_of::<GptPartitionEntry>()]
            .copy_from_slice(bytemuck::bytes_of(&entry));

        let mut header = GptHeader {
            signature: GptHeaderSignature::EFI_COMPATIBLE_PARTITION_TABLE_HEADER,
            revision: GptHeaderRevision::VERSION_1_0,
            header_size: U32Le::from_u32(92),
            header_crc32: Crc32(U32Le::from_u32(0)),
            reserved: U32Le::from_u32(0),
            my_lba: LbaLe::from_u64(1),
            alternate_lba: LbaLe::from_u64(0x10_0000),
            first_usable_lba: LbaLe::from_u64(34),
            last_usable_lba: LbaLe::from_u64(0xF_FFFF),
            disk_guid: guid!("99999999-8888-7777-6666-555555555555"),
            partition_entry_lba: LbaLe::from_u64(2),
            number_of_partition_entries: U32Le::from_u32(ENTRY_COUNT),
            size_of_partition_entry: U32Le::from_u32(ENTRY_SIZE),
            partition_entry_array_crc32: Crc32(U32Le::from_u32(crc32(&entries))),
        };
        // The header CRC is computed over the header with its own CRC
        // field zeroed — which it already is here.
        let crc = crc32(&bytemuck::bytes_of(&header)[..92]);
        header.header_crc32 = Crc32(U32Le::from_u32(crc));

        let mut block = [0u8; BLOCK];
        block[..size_of::<GptHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
        (block, entries)
    }

    #[test]
    fn a_valid_table_yields_the_matching_partitions_extent() {
        let (block, entries) = table(KORE_SWAP_TYPE, 2048, 2048 + 1000 - 1);
        let geom = parse_header(&block).expect("valid header");
        assert_eq!(geom.entry_lba, 2);
        assert_eq!(geom.entry_array_bytes(), 16384);

        let extent = find_partition(&entries, &geom, KORE_SWAP_TYPE).expect("finds ours");
        assert_eq!(extent.start_lba, 2048);
        assert_eq!(extent.blocks(), 1000);
        assert!(extent.start_lba > 0, "never LBA 0");
    }

    /// A disk partitioned for something else is `NoMatch` — a hardware
    /// fact, distinct from a damaged table, because the caller's response
    /// differs (serve degraded vs. refuse the device).
    #[test]
    fn a_table_without_our_type_is_no_match() {
        let (block, entries) = table(
            guid!("0657FD6D-A4AB-43C4-84E5-0933C84B4F4F"), // Linux swap
            2048,
            4096,
        );
        let geom = parse_header(&block).unwrap();
        assert_eq!(
            find_partition(&entries, &geom, KORE_SWAP_TYPE),
            Err(GptError::NoMatch)
        );
    }

    /// An unpartitioned device is `NotGpt`, not a parse crash.
    #[test]
    fn a_blank_or_mbr_only_device_is_not_gpt() {
        assert_eq!(parse_header(&[0u8; BLOCK]), Err(GptError::NotGpt));
        assert_eq!(parse_header(&[0xFFu8; BLOCK]), Err(GptError::NotGpt));
        assert_eq!(parse_header(&[]), Err(GptError::BadGeometry));
    }

    /// **A damaged table is never trusted.** One flipped byte in the
    /// header, or in the entry array, must refuse — not hand back an
    /// extent read out of corrupt bytes, which is how a driver would
    /// come to write over somebody else's partition.
    #[test]
    fn corruption_refuses_rather_than_yielding_an_extent() {
        let (mut block, mut entries) = table(KORE_SWAP_TYPE, 2048, 4096);
        let good = parse_header(&block).unwrap();

        block[40] ^= 0x01; // first_usable_lba — inside the CRC'd span
        assert_eq!(parse_header(&block), Err(GptError::BadHeaderCrc));

        entries[3 * ENTRY_SIZE as usize + 32] ^= 0x01; // starting_lba
        assert_eq!(
            find_partition(&entries, &good, KORE_SWAP_TYPE),
            Err(GptError::BadEntryCrc)
        );
    }

    /// A matching entry whose extent escapes the usable range is refused,
    /// never clamped — including one that would reach LBA 0.
    #[test]
    fn an_out_of_range_extent_is_refused() {
        // Below first_usable_lba (34): an extent overlapping the tables.
        let (block, entries) = table(KORE_SWAP_TYPE, 1, 4096);
        let geom = parse_header(&block).unwrap();
        assert_eq!(
            find_partition(&entries, &geom, KORE_SWAP_TYPE),
            Err(GptError::BadGeometry)
        );

        // Past last_usable_lba.
        let (block, entries) = table(KORE_SWAP_TYPE, 2048, 0x10_0000);
        let geom = parse_header(&block).unwrap();
        assert_eq!(
            find_partition(&entries, &geom, KORE_SWAP_TYPE),
            Err(GptError::BadGeometry)
        );
    }

    /// A table claiming an absurd entry array is refused whole, rather
    /// than scanned partially — a partial scan could miss the entry that
    /// says these blocks belong to somebody else.
    #[test]
    fn an_oversized_entry_array_is_refused() {
        let (mut block, _) = table(KORE_SWAP_TYPE, 2048, 4096);
        let mut header: GptHeader = bytemuck::pod_read_unaligned(&block[..size_of::<GptHeader>()]);
        header.number_of_partition_entries = U32Le::from_u32(1_000_000);
        header.header_crc32 = Crc32(U32Le::from_u32(0));
        let crc = crc32(&bytemuck::bytes_of(&header)[..92]);
        header.header_crc32 = Crc32(U32Le::from_u32(crc));
        block[..size_of::<GptHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
        assert_eq!(parse_header(&block), Err(GptError::BadGeometry));
    }
}
