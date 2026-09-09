//! SMMUv3 **stage-1** translation tables — the format a device-embedded SMMUv3
//! uses to translate a device VA → IOVA. New code: the
//! kernel's landed SMMUv3 only encodes *stage-2* (IOVA → PA). The VMSAv8-64
//! stage-1 leaf differs from stage-2 in three load-bearing places, verified
//! against Linux `drivers/iommu/io-pgtable-arm.c`:
//!   * permissions use **AP[2:1]** (bit6 = unprivileged-access, bit7 = read-only)
//!     rather than stage-2's independent S2AP read/write bits;
//!   * the memory type is an **AttrIndx[2:0]** (bits[4:2]) index into the context
//!     descriptor's MAIR, not a direct MemAttr field;
//!   * stage-1 leaves set **nG** (not-global, bit11).
//!
//! Levels are numbered finest = 1 (see [`PagingMetaData`]); a 4 KiB page sits at
//! level 1. Output addresses are [`IovaAddr`] — the device's leaves hold IOVAs,
//! fenced by the platform IOMMU, so no physical address ever enters this tree.

use kore_abi::device::IovaAddr;

use crate::common::flags::{AccessFlags, CachePolicy, MemoryAttributes, PageSize, Shareability};
use crate::common::pagetable::{
    DevVirtAddr, DeviceWalker, EntryKind, MmuError, MmuResult, PageTableEntry, PagingMetaData,
};

/// Input IOVA / VA geometry of one stage-1 (or stage-2) domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SmmuV3AddressWidth {
    /// 39-bit input, 3 levels.
    Bits39 = 39,
    /// 48-bit input, 4 levels.
    Bits48 = 48,
}

impl SmmuV3AddressWidth {
    /// Address bits.
    #[inline]
    pub const fn bits(self) -> u8 {
        self as u8
    }

    /// Translation levels.
    #[inline]
    pub const fn levels(self) -> usize {
        match self {
            Self::Bits39 => 3,
            Self::Bits48 => 4,
        }
    }

    /// `TCR.T0SZ` for this input width (64 − bits).
    #[inline]
    pub const fn t0sz(self) -> u8 {
        64 - self.bits()
    }
}

/// Maximum output-address size (`SMMU_IDR5.OAS` / `TCR.IPS` domain).
///
/// Deliberately distinct from [`SmmuV3AddressWidth`]: OAS constrains the emitted
/// output addresses while the domain width describes the input geometry — a
/// 48-bit input domain is valid on an implementation whose output is 44 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SmmuV3OutputAddressSize {
    Bits32 = 32,
    Bits36 = 36,
    Bits40 = 40,
    Bits42 = 42,
    Bits44 = 44,
    Bits48 = 48,
    Bits52 = 52,
}

impl SmmuV3OutputAddressSize {
    /// Output-address bits.
    #[inline]
    pub const fn bits(self) -> u8 {
        self as u8
    }

    /// Architected `VTCR.PS` / `TCR.IPS` encoding.
    #[inline]
    pub const fn ps_encoding(self) -> u8 {
        match self {
            Self::Bits32 => 0,
            Self::Bits36 => 1,
            Self::Bits40 => 2,
            Self::Bits42 => 3,
            Self::Bits44 => 4,
            Self::Bits48 => 5,
            Self::Bits52 => 6,
        }
    }

    /// Largest output size representable by the non-LPA2 leaf address field.
    #[inline]
    pub const fn without_lpa2(self) -> Self {
        match self {
            Self::Bits52 => Self::Bits48,
            size => size,
        }
    }
}

const VALID: u64 = 1 << 0;
const PAGE_OR_TABLE: u64 = 1 << 1;
const ATTR_INDX_SHIFT: u32 = 2; // bits [4:2]
const NS: u64 = 1 << 5;
const AP_UNPRIV: u64 = 1 << 6; // AP[1]
const AP_RDONLY: u64 = 1 << 7; // AP[2]
const SH_SHIFT: u32 = 8; // bits [9:8]
const AF: u64 = 1 << 10;
const NG: u64 = 1 << 11;
const XN: u64 = 0b11 << 53; // PXN | UXN
const ADDR_MASK_48: u64 = 0x0000_ffff_ffff_f000;

// AttrIndx values — indices into [`STAGE1_MAIR`].
const ATTR_IDX_WB: u64 = 0;
const ATTR_IDX_DEVICE: u64 = 1;
const ATTR_IDX_NC: u64 = 2;

/// The MAIR a context descriptor must carry for the [`AttrIndx`] values this
/// format emits: `Attr0 = 0xff` (Normal Inner/Outer WB), `Attr1 = 0x04`
/// (Device-nGnRE), `Attr2 = 0x44` (Normal Non-Cacheable). Byte order is
/// `Attr0` in bits[7:0], `Attr1` in [15:8], `Attr2` in [23:16].
///
/// [`AttrIndx`]: SmmuV3Stage1Flags
pub const STAGE1_MAIR: u64 = 0x0000_0000_0044_04ff;

/// Permission + memory-type bundle carried by an SMMUv3 stage-1 leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmmuV3Stage1Flags {
    access: AccessFlags,
    attrs: MemoryAttributes,
}

impl SmmuV3Stage1Flags {
    /// Write-back, inner-shareable with the given access permissions.
    #[inline]
    pub const fn new(access: AccessFlags) -> Self {
        Self {
            access,
            attrs: MemoryAttributes::writeback(),
        }
    }

    /// With the memory attributes replaced.
    #[inline]
    pub const fn with_attrs(mut self, attrs: MemoryAttributes) -> Self {
        self.attrs = attrs;
        self
    }

    /// Access permissions.
    #[inline]
    pub const fn access(self) -> AccessFlags {
        self.access
    }

    /// Memory attributes.
    #[inline]
    pub const fn attrs(self) -> MemoryAttributes {
        self.attrs
    }

    fn leaf_bits(self, size: PageSize) -> u64 {
        let mut bits = VALID | AF | NG | NS | AP_UNPRIV;
        if matches!(size, PageSize::Size4K) {
            bits |= PAGE_OR_TABLE;
        }
        if !self.access.contains(AccessFlags::WRITE) {
            bits |= AP_RDONLY;
        }
        if !self.access.contains(AccessFlags::EXECUTE) {
            bits |= XN;
        }
        let attr_idx = match self.attrs.cache() {
            CachePolicy::Writeback | CachePolicy::WriteThrough => ATTR_IDX_WB,
            CachePolicy::Uncached => ATTR_IDX_DEVICE,
            CachePolicy::WriteCombine => ATTR_IDX_NC,
        };
        bits |= attr_idx << ATTR_INDX_SHIFT;
        let sh = match self.attrs.shareability() {
            Shareability::NonShareable => 0,
            Shareability::Outer | Shareability::System => 2,
            Shareability::Inner => 3,
        };
        bits | (sh << SH_SHIFT)
    }

    fn from_bits(bits: u64) -> Self {
        // Stage-1 always grants read; AP[2] (RDONLY) only distinguishes RW vs RO.
        let mut access = AccessFlags::READ;
        if bits & AP_RDONLY == 0 {
            access |= AccessFlags::WRITE;
        }
        if bits & XN == 0 {
            access |= AccessFlags::EXECUTE;
        }
        let cache = match (bits >> ATTR_INDX_SHIFT) & 0x7 {
            ATTR_IDX_DEVICE => CachePolicy::Uncached,
            ATTR_IDX_NC => CachePolicy::WriteCombine,
            _ => CachePolicy::Writeback,
        };
        let shareability = match (bits >> SH_SHIFT) & 0x3 {
            0 => Shareability::NonShareable,
            2 => Shareability::Outer,
            _ => Shareability::Inner,
        };
        Self {
            access,
            attrs: MemoryAttributes::writeback()
                .with_cache(cache)
                .with_shareability(shareability),
        }
    }
}

/// One 64-bit VMSAv8-64 **stage-1** descriptor. Output addresses are IOVAs.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct SmmuV3Stage1Pte(u64);

impl core::fmt::Debug for SmmuV3Stage1Pte {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SmmuV3Stage1Pte")
            .field(&format_args!("{:#018x}", self.0))
            .finish()
    }
}

impl PageTableEntry for SmmuV3Stage1Pte {
    type Flags = SmmuV3Stage1Flags;
    type OutputAddr = IovaAddr;

    #[inline]
    fn validate_flags(flags: Self::Flags, _size: PageSize) -> MmuResult {
        // VMSAv8 stage-1 AP[2:1] has no no-read or write-only encoding: every
        // valid leaf is readable. Refuse rather than silently widen authority.
        if flags.access().contains(AccessFlags::READ) {
            Ok(())
        } else {
            Err(MmuError::UnsupportedAccess)
        }
    }

    #[inline]
    fn new_leaf(out: IovaAddr, flags: Self::Flags, size: PageSize) -> MmuResult<Self> {
        Self::validate_flags(flags, size)?;
        Ok(Self((out.0 & ADDR_MASK_48) | flags.leaf_bits(size)))
    }

    #[inline]
    fn new_table(out: IovaAddr, _level: u8) -> MmuResult<Self> {
        Ok(Self((out.0 & ADDR_MASK_48) | VALID | PAGE_OR_TABLE))
    }

    #[inline]
    fn set_flags(&mut self, flags: Self::Flags, size: PageSize) -> MmuResult {
        *self = Self::new_leaf(self.out_addr(size), flags, size)?;
        Ok(())
    }

    #[inline]
    fn out_addr(&self, _size: PageSize) -> IovaAddr {
        IovaAddr(self.0 & ADDR_MASK_48)
    }

    #[inline]
    fn flags(&self, _size: PageSize) -> Self::Flags {
        SmmuV3Stage1Flags::from_bits(self.0)
    }

    #[inline]
    fn is_present(&self) -> bool {
        self.0 & VALID != 0
    }

    #[inline]
    fn entry_kind(&self, level: u8) -> EntryKind {
        if !self.is_present() {
            EntryKind::Invalid
        } else if level == 1 || self.0 & PAGE_OR_TABLE == 0 {
            // At the finest level a "page" (bit1 set) is a leaf; at higher
            // levels a cleared bit1 is a block leaf, a set bit1 a table.
            EntryKind::Leaf
        } else {
            EntryKind::Table
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.0 = 0;
    }

    #[inline]
    fn bits(&self) -> u64 {
        self.0
    }

    #[inline]
    fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

macro_rules! define_stage1_meta {
    ($name:ident, $levels:expr, $va_bits:expr) => {
        /// Stage-1 table geometry (4 KiB granule).
        #[derive(Clone, Copy, Debug)]
        pub struct $name;

        impl PagingMetaData for $name {
            const LEVELS: usize = $levels;
            const VA_MAX_BITS: usize = $va_bits;
            const OUT_MAX_BITS: usize = 48;
            const LEAF_SIZES_DESC: &'static [PageSize] =
                &[PageSize::Size1G, PageSize::Size2M, PageSize::Size4K];

            type VirtAddr = DevVirtAddr;

            #[inline]
            fn level_shift(level: u8) -> u32 {
                12 + ((level as u32) - 1) * 9
            }

            #[inline]
            fn level_supports_leaf(level: u8, size: PageSize) -> bool {
                matches!(
                    (level, size),
                    (1, PageSize::Size4K) | (2, PageSize::Size2M) | (3, PageSize::Size1G)
                )
            }
        }
    };
}

define_stage1_meta!(SmmuV3Stage1Meta39, 3, 39);
define_stage1_meta!(SmmuV3Stage1Meta48, 4, 48);

/// A 39-bit-input stage-1 device page table.
pub type SmmuV3Stage1Table39 = DeviceWalker<SmmuV3Stage1Meta39, SmmuV3Stage1Pte>;
/// A 48-bit-input stage-1 device page table.
pub type SmmuV3Stage1Table48 = DeviceWalker<SmmuV3Stage1Meta48, SmmuV3Stage1Pte>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage1_leaf_encodes_ap_rw_and_writeback_attr_index() {
        let flags = SmmuV3Stage1Flags::new(AccessFlags::READ | AccessFlags::WRITE);
        let pte = SmmuV3Stage1Pte::new_leaf(IovaAddr(0x20_0000), flags, PageSize::Size2M).unwrap();

        // 2 MiB block: bit1 clear (block), present, AF, nG, NS, unprivileged.
        assert!(pte.is_present());
        assert_eq!(pte.entry_kind(2), EntryKind::Leaf);
        assert_eq!(pte.bits() & PAGE_OR_TABLE, 0);
        assert_ne!(pte.bits() & AF, 0);
        assert_ne!(pte.bits() & NG, 0);
        assert_ne!(pte.bits() & AP_UNPRIV, 0);
        // Writable → AP[2] (RDONLY) clear; AttrIndx 0 (write-back).
        assert_eq!(pte.bits() & AP_RDONLY, 0);
        assert_eq!((pte.bits() >> ATTR_INDX_SHIFT) & 0x7, ATTR_IDX_WB);
        assert_eq!(pte.out_addr(PageSize::Size2M), IovaAddr(0x20_0000));
    }

    #[test]
    fn stage1_readonly_noexec_device_leaf() {
        let flags = SmmuV3Stage1Flags::new(AccessFlags::READ)
            .with_attrs(MemoryAttributes::writeback().with_cache(CachePolicy::Uncached));
        let pte = SmmuV3Stage1Pte::new_leaf(IovaAddr(0x1000), flags, PageSize::Size4K).unwrap();

        // 4 KiB page: bit1 set (page). Read-only → AP[2] set. No execute → XN.
        assert_ne!(pte.bits() & PAGE_OR_TABLE, 0);
        assert_ne!(pte.bits() & AP_RDONLY, 0);
        assert_eq!(pte.bits() & XN, XN);
        assert_eq!((pte.bits() >> ATTR_INDX_SHIFT) & 0x7, ATTR_IDX_DEVICE);

        let decoded = pte.flags(PageSize::Size4K);
        assert!(decoded.access().contains(AccessFlags::READ));
        assert!(!decoded.access().contains(AccessFlags::WRITE));
        assert!(!decoded.access().contains(AccessFlags::EXECUTE));
        assert_eq!(decoded.attrs().cache(), CachePolicy::Uncached);
    }

    #[test]
    fn stage1_refuses_access_without_read_instead_of_widening_it() {
        for access in [
            AccessFlags::empty(),
            AccessFlags::WRITE,
            AccessFlags::EXECUTE,
        ] {
            assert_eq!(
                SmmuV3Stage1Pte::new_leaf(
                    IovaAddr(0x1000),
                    SmmuV3Stage1Flags::new(access),
                    PageSize::Size4K,
                ),
                Err(MmuError::UnsupportedAccess)
            );
        }
    }

    #[test]
    fn table_descriptor_is_present_and_holds_iova() {
        let pte = SmmuV3Stage1Pte::new_table(IovaAddr(0x4000), 4).unwrap();
        assert_eq!(pte.entry_kind(4), EntryKind::Table);
        assert_eq!(pte.out_addr(PageSize::Size4K), IovaAddr(0x4000));
    }

    #[test]
    fn geometry_widths_and_t0sz() {
        assert_eq!(SmmuV3AddressWidth::Bits48.levels(), 4);
        assert_eq!(SmmuV3AddressWidth::Bits48.t0sz(), 16);
        assert_eq!(SmmuV3AddressWidth::Bits39.levels(), 3);
        assert_eq!(SmmuV3AddressWidth::Bits39.t0sz(), 25);
    }
}
