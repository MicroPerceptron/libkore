//! The two contract traits a device-MMU page-table format implements — mirrors
//! of the kernel's `platform::paging::{PageTableEntry, PagingMetaData}`, with
//! the **output/leaf address generalised to an associated type** so a device
//! instance holds an `IovaAddr` leaf where the kernel holds a `PhysAddr`.

use core::fmt::Debug;

use super::addr::DeviceAddr;
use super::alloc::MmuResult;
use crate::common::flags::PageSize;

/// A present entry's decoded role at a particular level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntryKind {
    /// Points at the next lower-level table.
    Table,
    /// Terminates the walk and maps an output leaf/block.
    Leaf,
    /// Not valid at this level.
    Invalid,
}

impl EntryKind {
    /// Whether this is a leaf.
    #[inline]
    pub const fn is_leaf(self) -> bool {
        matches!(self, Self::Leaf)
    }

    /// Whether this is a table pointer.
    #[inline]
    pub const fn is_table(self) -> bool {
        matches!(self, Self::Table)
    }
}

/// One device-MMU page-table entry.
///
/// Every impl is `#[repr(transparent)]` over `u64` — the walker manipulates a
/// page-table frame as a `[u64]`. Unlike the kernel's `PageTableEntry`, the
/// output/leaf address is an **associated type** ([`OutputAddr`]): an SMMUv3
/// device instance stores an `IovaAddr` where the kernel platform IOMMU stores
/// a `PhysAddr`. `Flags` is fully opaque so a format may carry format-private
/// bits (such as a local-memory selector, sparse-mapping marker, or atomic
/// enable) the walker never inspects.
///
/// [`OutputAddr`]: PageTableEntry::OutputAddr
pub trait PageTableEntry: Copy + Debug + 'static {
    /// Per-format permission + memory-type bundle. Opaque to the walker; `Eq`
    /// so a future coalescer can compare sibling flags.
    type Flags: Copy + Debug + Eq;

    /// The output address a leaf/table entry points at (an IOVA for a device
    /// instance; the table tree lives in the same output space).
    type OutputAddr: DeviceAddr;

    /// Refuse any protection/attribute set this format cannot encode exactly.
    ///
    /// The generic walker calls this before publishing a map or protection
    /// change. Implementations must not normalize an unsupported request into
    /// broader hardware authority (for example, WRITE-only into READ|WRITE).
    fn validate_flags(flags: Self::Flags, size: PageSize) -> MmuResult;

    /// Construct a leaf mapping `out` at the given leaf size.
    fn new_leaf(out: Self::OutputAddr, flags: Self::Flags, size: PageSize) -> MmuResult<Self>;

    /// Construct an intermediate entry pointing at a child table. `level` is
    /// the level *of this entry* (the containing table's level), matching the
    /// kernel contract — some formats encode the child level in the pointer.
    fn new_table(out: Self::OutputAddr, level: u8) -> MmuResult<Self>;

    /// Replace a leaf's permission and memory attributes while preserving its
    /// output address. Called only for a present leaf at `size`.
    fn set_flags(&mut self, flags: Self::Flags, size: PageSize) -> MmuResult;

    /// Output address the entry points at, decoded for `size`.
    ///
    /// For a leaf, `size` is that leaf's native size. For an intermediate
    /// table descriptor, callers pass [`PagingMetaData::TABLE_FRAME_SIZE`].
    /// The distinction is load-bearing for formats such as Intel PPGTT where
    /// a bit belongs to the output-address field in a 4 KiB PTE but selects a
    /// memory attribute in a huge-page PDE/PDPE.
    fn out_addr(&self, size: PageSize) -> Self::OutputAddr;

    /// Current flags, decoded for the leaf's native `size`.
    fn flags(&self, size: PageSize) -> Self::Flags;

    /// Whether the entry is present (a valid mapping or valid table pointer).
    fn is_present(&self) -> bool;

    /// Decode the entry's role at `level`. Called only after
    /// [`is_present`](Self::is_present) returns true.
    fn entry_kind(&self, level: u8) -> EntryKind;

    /// Zero the entry.
    fn clear(&mut self);

    /// Raw 64-bit backing.
    fn bits(&self) -> u64;

    /// Reconstruct from raw 64-bit backing.
    fn from_bits(bits: u64) -> Self;

    /// Convenience: terminality at `level`.
    #[inline]
    fn is_leaf_at(&self, level: u8) -> bool {
        self.is_present() && self.entry_kind(level).is_leaf()
    }
}

/// Geometry of one device-MMU page-table format.
///
/// Levels are numbered **finest = 1** (matching the kernel's SMMUv3 meta), so a
/// 4 KiB leaf sits at level 1 and the walk starts at [`LEVELS`](Self::LEVELS)
/// and descends. `level_shift(1) == 12` for the 4 KiB granule.
pub trait PagingMetaData: 'static {
    /// Number of translation levels (48-bit 4 KiB = 4).
    const LEVELS: usize;

    /// Bits per table index (512-entry 4 KiB frame = 9).
    const INDEX_BITS: u32 = 9;

    /// Finest translation granule supported by this format.
    const BASE_PAGE_SIZE: PageSize = PageSize::Size4K;

    /// Size and required output-address alignment of a table frame.
    const TABLE_FRAME_SIZE: PageSize = Self::BASE_PAGE_SIZE;

    /// Maximum input (virtual) address bits.
    const VA_MAX_BITS: usize;

    /// Maximum output-address bits the leaf field can encode.
    const OUT_MAX_BITS: usize;

    /// Supported leaf sizes in **descending** order (for `best_leaf`).
    const LEAF_SIZES_DESC: &'static [PageSize];

    /// The input address type (a device VA newtype).
    type VirtAddr: DeviceAddr;

    /// Entries per page-table frame.
    #[inline]
    fn entries_per_table() -> usize {
        1usize << Self::INDEX_BITS
    }

    /// Entries used by a table at `level`. Formats with a narrowed root can
    /// override [`level_index_bits`](Self::level_index_bits).
    #[inline]
    fn entries_at_level(level: u8) -> usize {
        1usize << Self::level_index_bits(level)
    }

    /// Bit-shift for indexing the table at `level`.
    fn level_shift(level: u8) -> u32;

    /// Index-bit width at `level` (default: uniform [`INDEX_BITS`](Self::INDEX_BITS)).
    #[inline]
    fn level_index_bits(_level: u8) -> u32 {
        Self::INDEX_BITS
    }

    /// Whether a leaf of `size` may terminate at `level`.
    fn level_supports_leaf(level: u8, size: PageSize) -> bool;

    /// Native leaf size at `level`, if that level may terminate a walk.
    #[inline]
    fn leaf_size_at_level(level: u8) -> Option<PageSize> {
        let mut i = 0;
        while i < Self::LEAF_SIZES_DESC.len() {
            let size = Self::LEAF_SIZES_DESC[i];
            if Self::level_supports_leaf(level, size) {
                return Some(size);
            }
            i += 1;
        }
        None
    }

    /// Lowest level that natively supports `size`.
    #[inline]
    fn size_to_level(size: PageSize) -> Option<u8> {
        let mut level = 1u8;
        while (level as usize) <= Self::LEVELS {
            if Self::level_supports_leaf(level, size) {
                return Some(level);
            }
            level += 1;
        }
        None
    }

    /// Whether `va` is inside the legal input range.
    #[inline]
    fn va_is_valid(va: Self::VirtAddr) -> bool {
        Self::VA_MAX_BITS >= 64 || va.to_u64() < (1u64 << Self::VA_MAX_BITS)
    }

    /// Whether a raw output address fits the leaf's address field.
    #[inline]
    fn out_is_valid(out: u64) -> bool {
        Self::OUT_MAX_BITS >= 64 || out < (1u64 << Self::OUT_MAX_BITS)
    }
}
