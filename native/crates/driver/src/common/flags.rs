//! Neutral permission / memory-attribute / page-size types shared by every
//! device-MMU page-table format in [`crate::common`].
//!
//! Mirrored driver-side from the kernel's `platform::paging` neutral types so a
//! format impl's flags read identically on both sides of the seam — but kept to
//! only what the device-MMU formats need (no swap / guard / CPU-VM concerns).
//! The library is arch-neutral: nothing here is `#[cfg(target_arch)]`.

use bitflags::bitflags;

bitflags! {
    /// Access permissions carried by a leaf mapping.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct AccessFlags: u8 {
        /// Device may read the mapped bytes.
        const READ = 1 << 0;
        /// Device may write the mapped bytes.
        const WRITE = 1 << 1;
        /// Device may fetch/execute from the mapped bytes.
        const EXECUTE = 1 << 2;
    }
}

/// Hardware translation granule / leaf size.
///
/// Every size used by a supported device format is named unconditionally;
/// each [`PagingMetaData`](crate::common::pagetable::PagingMetaData) impl
/// decides which sizes are legal. Carrying unreachable variants has no runtime
/// cost and lets common code reason about heterogeneous devices without cfgs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u64)]
pub enum PageSize {
    /// 4 KiB — the base granule.
    Size4K = 0x1000,
    /// 16 KiB base granule.
    Size16K = 0x4000,
    /// 64 KiB base granule or GPU large page.
    Size64K = 0x1_0000,
    /// 2 MiB block.
    Size2M = 0x20_0000,
    /// 32 MiB block (AArch64 16 KiB granule).
    Size32M = 0x200_0000,
    /// 512 MiB block (AArch64 64 KiB granule).
    Size512M = 0x2000_0000,
    /// 1 GiB block.
    Size1G = 0x4000_0000,
    /// 64 GiB block (AArch64 16 KiB granule).
    Size64G = 0x10_0000_0000,
    /// 512 GiB block (five-level/RISC-V formats).
    Size512G = 0x80_0000_0000,
    /// 4 TiB block (AArch64 64 KiB granule).
    Size4T = 0x400_0000_0000,
}

impl PageSize {
    /// Size in bytes.
    #[inline]
    pub const fn bytes(self) -> usize {
        self as u64 as usize
    }

    /// Recover a named page size from an exact byte count.
    #[inline]
    pub const fn from_bytes(bytes: u64) -> Option<Self> {
        match bytes {
            0x1000 => Some(Self::Size4K),
            0x4000 => Some(Self::Size16K),
            0x1_0000 => Some(Self::Size64K),
            0x20_0000 => Some(Self::Size2M),
            0x200_0000 => Some(Self::Size32M),
            0x2000_0000 => Some(Self::Size512M),
            0x4000_0000 => Some(Self::Size1G),
            0x10_0000_0000 => Some(Self::Size64G),
            0x80_0000_0000 => Some(Self::Size512G),
            0x400_0000_0000 => Some(Self::Size4T),
            _ => None,
        }
    }

    /// Whether `addr` is aligned to this size.
    #[inline]
    pub const fn is_aligned(self, addr: u64) -> bool {
        addr & (self as u64 - 1) == 0
    }

    /// Offset of `addr` within a leaf of this size.
    #[inline]
    pub const fn offset(self, addr: u64) -> u64 {
        addr & (self as u64 - 1)
    }
}

/// Cacheability of a leaf's target memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CachePolicy {
    /// Device / strongly-ordered, non-cacheable.
    Uncached,
    /// Normal, non-cacheable (write-combining).
    WriteCombine,
    /// Normal, write-through cacheable.
    WriteThrough,
    /// Normal, write-back cacheable.
    #[default]
    Writeback,
}

/// Shareability domain of a leaf's target memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Shareability {
    /// Not shared.
    NonShareable,
    /// Outer shareable.
    Outer,
    /// Inner shareable.
    #[default]
    Inner,
    /// Full-system shareable (treated as outer in the encodings here).
    System,
}

/// Memory-type bundle (cache policy + shareability) carried alongside access
/// permissions in a leaf's flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryAttributes {
    cache: CachePolicy,
    shareability: Shareability,
}

impl MemoryAttributes {
    /// Inner-shareable write-back — the default for coherent system RAM.
    #[inline]
    pub const fn writeback() -> Self {
        Self {
            cache: CachePolicy::Writeback,
            shareability: Shareability::Inner,
        }
    }

    /// Cache policy.
    #[inline]
    pub const fn cache(self) -> CachePolicy {
        self.cache
    }

    /// Shareability domain.
    #[inline]
    pub const fn shareability(self) -> Shareability {
        self.shareability
    }

    /// With the cache policy replaced.
    #[inline]
    pub const fn with_cache(mut self, cache: CachePolicy) -> Self {
        self.cache = cache;
        self
    }

    /// With the shareability replaced.
    #[inline]
    pub const fn with_shareability(mut self, shareability: Shareability) -> Self {
        self.shareability = shareability;
        self
    }
}
