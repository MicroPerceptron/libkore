//! Memory ABI — the properties a userland driver states when asking the
//! kernel to allocate a (DMA-capable) buffer, and the cache/access
//! attributes of a mapping. Mirrors the kernel mm's allocation properties;
//! the kernel's manifold chooses the actual backing region from these.

use bitflags::bitflags;

/// NUMA node hint for placement, or [`NumaNode::ANY`] to let the manifold
/// decide.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NumaNode(pub u16);

impl NumaNode {
    /// No preference — the kernel's placement policy chooses.
    pub const ANY: Self = Self(u16::MAX);
}

/// A device's DMA addressing reach, which constrains where backing frames
/// may be placed.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DmaClass {
    /// CPU-only memory — not reachable by any device.
    None = 0,
    /// 32-bit DMA: backing frames must sit below 4 GiB.
    Dma32 = 1,
    /// Full 64-bit DMA.
    Dma64 = 2,
}

bitflags! {
    /// Cache + access attributes requested for a mapping.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct MemAttrs: u16 {
        /// Write-back cacheable (normal memory).
        const WRITE_BACK = 1 << 0;
        /// Write-combining (framebuffers / streaming DMA).
        const WRITE_COMBINE = 1 << 1;
        /// Uncached (MMIO device registers).
        const UNCACHED = 1 << 2;
        /// Readable by the holder.
        const READ = 1 << 3;
        /// Writable by the holder.
        const WRITE = 1 << 4;
    }
}

/// Properties for a buffer allocation. The driver states only its
/// requirements; the manifold picks the region (NUMA-local, DMA-class-fit,
/// contiguous if asked).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AllocProps {
    /// Requested size in bytes.
    pub size: u64,
    /// Required alignment in bytes (power of two).
    pub align: u64,
    /// NUMA placement hint.
    pub numa: NumaNode,
    /// DMA addressing class the backing must satisfy.
    pub dma: DmaClass,
    /// Cache/access attributes.
    pub attrs: MemAttrs,
    /// Require physically-contiguous backing (e.g. a single descriptor
    /// ring the device walks as one extent).
    pub contiguous: bool,
}

// Frozen wire pins (#81; the crate-header wire-freeze discipline).
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<AllocProps>() == 24);
    assert!(size_of::<NumaNode>() == 2);
    assert!(size_of::<DmaClass>() == 1);
    assert!(size_of::<MemAttrs>() == 2);
};
