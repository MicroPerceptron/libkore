//! The device-VA aperture allocator: page-granular first-fit over one
//! device address range, with the 2 MiB huge-page policy every device MMU
//! in the tree shares.
//!
//! Extracted from the iVPU driver's `VpuAperture` when the Xe runtime import
//! path needed the same allocator for PPGTT import addresses
//! (`res/ACCEL_MEMORY_SCALE.md` §4, slice 3d): two device MMUs, one carve
//! discipline.  Sorted live extents define the free gaps without a second
//! free-list, so storage scales with live mappings rather than the
//! aperture's potentially vast page count.  Mutation is serialized by the
//! owning device's aperture borrow.

use alloc::vec::Vec;

use kore_abi::device::{IovaAddr, IovaRange};

/// Page-granular first-fit allocation over one device-VA aperture.
///
/// Sorted live extents define the free gaps without a second free-list. Storage
/// scales with live mappings, rather than the aperture's potentially vast page
/// count. Mutation is serialized by the owning device's aperture borrow.
#[derive(Debug)]
pub struct DeviceAperture {
    base: u64,
    len: u64,
    live: Vec<IovaRange>,
}

impl DeviceAperture {
    /// The device MMU granule every carve is rounded to.
    pub const PAGE: u64 = 4096;
    /// The block-leaf size the huge-page policy targets: the 2 MiB block
    /// both the SMMUv3 stage-1 (iVPU) and the Xe PPGTT (its default leaf
    /// ceiling) coalesce to when the backing is equally aligned.
    pub const BLOCK: u64 = 2 << 20;

    /// An allocator over `range`.
    pub fn new(range: IovaRange) -> Self {
        Self {
            base: range.base.0,
            len: range.len,
            live: Vec::new(),
        }
    }

    /// The alignment a carve of `bytes`/`align` is actually granted —
    /// the requested alignment, floored at a page and raised to the
    /// block size for block-sized requests (the huge-page policy).
    pub fn granted_align(bytes: u64, align: u64) -> u64 {
        let mut align = align.max(Self::PAGE);
        if bytes >= Self::BLOCK {
            align = align.max(Self::BLOCK);
        }
        align
    }

    /// Carve `bytes` (rounded up to pages), aligned to `align` (a power
    /// of two, at least a page).  `None` when the aperture is exhausted
    /// — the caller's refusal, never a panic.
    ///
    /// **The huge-page policy lives here**: a carve of 2 MiB or more is
    /// aligned up to the 2 MiB block size regardless of the requested
    /// alignment, so that when its backing is equally aligned (the
    /// pool lease's `LEASE_ALIGN_2M` founder arm), the shared stage-1
    /// walker's largest-leaf coalescing emits block PTEs — 512× the
    /// TLB reach for the weight/activation buffers the NPU streams
    /// mid-inference.  Small carves stay page-aligned: rings and
    /// descriptors would only waste aperture and pool on block
    /// rounding.
    pub fn carve(&mut self, bytes: u64, align: u64) -> Option<u64> {
        let align = Self::granted_align(bytes, align);
        if !align.is_power_of_two() || bytes == 0 {
            return None;
        }
        let aperture_end = self.base.checked_add(self.len)?;
        if self.base % Self::PAGE != 0 || self.len % Self::PAGE != 0 {
            return None;
        }
        let rounded = bytes.checked_add(Self::PAGE - 1)? & !(Self::PAGE - 1);
        let mut at = self.base.checked_add(align - 1)? & !(align - 1);
        let mut index = 0;
        for extent in &self.live {
            if at.checked_add(rounded)? <= extent.base.0 {
                break;
            }
            at = extent
                .base
                .0
                .checked_add(extent.len)?
                .checked_add(align - 1)?
                & !(align - 1);
            index += 1;
        }
        if at.checked_add(rounded)? > aperture_end {
            return None;
        }
        // Admission must not publish an address whose bookkeeping can fail.
        self.live.try_reserve(1).ok()?;
        self.live.insert(
            index,
            IovaRange {
                base: IovaAddr(at),
                len: rounded,
            },
        );
        Some(at)
    }

    /// Live carves — the mapping count a driver's census or a test reads.
    pub fn live_extents(&self) -> usize {
        self.live.len()
    }

    /// The live carves, ascending by base.
    pub fn live_ranges(&self) -> &[IovaRange] {
        &self.live
    }

    /// Forget every live carve without a release witness.  Only a device
    /// owner that has just reset its whole MMU (every translation gone) may
    /// call this; tests use it to model exactly that.
    pub fn forget_live(&mut self) {
        self.live.clear();
    }

    /// Return exactly one live extent. The device owner must first remove its
    /// mappings and complete invalidation; partial or duplicate release refuses.
    pub fn release_quiesced(&mut self, base: u64, bytes: u64) -> bool {
        let Ok(index) = self
            .live
            .binary_search_by_key(&base, |extent| extent.base.0)
        else {
            return false;
        };
        if self.live[index].len != bytes {
            return false;
        }
        self.live.remove(index);
        true
    }
}
