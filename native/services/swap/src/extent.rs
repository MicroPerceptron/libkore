//! Bounded, generation-checked swap extent identities and page-slot ownership.
//!
//! The swap backends have different byte stores (packed compressed RAM versus
//! an NVMe partition), but they must agree on the lifecycle that a client can
//! observe.  [`ExtentTable`] is that common engine: a store claims one
//! contiguous run of logical 4 KiB slots, loads are read-only, and only an
//! explicit release returns the run.  The generation in [`SwapExtent`] makes a
//! stale handle refuse after the same start slot is reused.

use alloc::vec::Vec;
use core::ops::Range;

/// Maximum pages moved by one protocol operation (2 MiB at 4 KiB/page).
///
/// This is a scheduling bound, not a model shape.  It caps one synchronous
/// service turn and one rollback set while still admitting large residency
/// segments.  Clients learn the effective bound from `Attached` and may use
/// any smaller extent, including non-power-of-two lengths.
pub const MAX_EXTENT_PAGES: u32 = 512;

/// Opaque identity of one immutable stored extent.
///
/// `token` is backend-private.  Clients must return the complete value they
/// received from `Stored`; its exact page count accompanies every load and
/// release request and is authenticated against the backend's descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwapExtent(u32);

impl SwapExtent {
    /// Reconstruct a wire handle.  Both halves are one-based, so zero and
    /// half-zero wire values can never accidentally name live storage.
    pub const fn from_raw(token: u32) -> Option<Self> {
        // Low half is descriptor index + 1; high half is generation.  Both
        // are one-based, so malformed half-zero identities fail at decode
        // instead of consuming a backend lookup.
        if token as u16 == 0 || token >> 16 == 0 {
            None
        } else {
            Some(Self(token))
        }
    }

    /// Backend-private token, for the wire codec only.
    pub const fn token(self) -> u32 {
        self.0
    }
}

const MAX_DESCRIPTORS: usize = u16::MAX as usize;

#[derive(Clone, Copy, Default)]
struct Descriptor {
    start: u32,
    pages: u32,
    generation: u16,
    live: bool,
}

/// A bounded next-fit allocator plus the generation metadata that authenticates
/// [`SwapExtent`] handles.
///
/// Storage is reserved once at bind.  Claim, validate, and release allocate
/// nothing; a successful next-fit claim inspects each page in the chosen run
/// once.  The cursor makes sequential residency segments O(total pages) rather
/// than repeatedly scanning from slot zero.
pub struct ExtentTable {
    capacity: u32,
    claimed: Vec<u64>,
    descriptors: Vec<Descriptor>,
    cursor: u32,
    descriptor_cursor: usize,
    live_extents: u32,
    live_pages: u32,
}

impl ExtentTable {
    /// An empty table over `capacity` logical page slots.
    pub fn new(capacity: u32) -> Self {
        let words = (capacity as usize).div_ceil(64);
        Self {
            capacity,
            claimed: alloc::vec![0; words],
            descriptors: alloc::vec![Descriptor::default();
                (capacity as usize).min(MAX_DESCRIPTORS)],
            cursor: 0,
            descriptor_cursor: 0,
            live_extents: 0,
            live_pages: 0,
        }
    }

    /// Total logical page slots.
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Currently live extent count.
    pub const fn live_extents(&self) -> u32 {
        self.live_extents
    }

    /// Currently claimed page count.
    pub const fn live_pages(&self) -> u32 {
        self.live_pages
    }

    /// Whether `page` belongs to any live extent.
    pub fn is_claimed(&self, page: u32) -> bool {
        page < self.capacity && self.claimed[page as usize / 64] & (1_u64 << (page % 64)) != 0
    }

    /// Claim one contiguous logical extent.  No bytes become externally live
    /// until the caller has populated the backing and retained this handle;
    /// callers roll a failed store back with [`release`](Self::release).
    pub fn claim(&mut self, pages: u32) -> Option<SwapExtent> {
        if pages == 0 || pages > MAX_EXTENT_PAGES || pages > self.capacity {
            return None;
        }
        let descriptor = self.find_descriptor()?;
        let start = self
            .find_run(self.cursor, self.capacity, pages)
            .or_else(|| self.find_run(0, self.cursor, pages))?;
        let generation = self.descriptors[descriptor].generation.checked_add(1)?;
        let end = start + pages;
        for page in start..end {
            self.claimed[page as usize / 64] |= 1_u64 << (page % 64);
        }
        self.descriptors[descriptor] = Descriptor {
            start,
            pages,
            generation,
            live: true,
        };
        self.cursor = if end == self.capacity { 0 } else { end };
        self.descriptor_cursor = (descriptor + 1) % self.descriptors.len();
        self.live_extents += 1;
        self.live_pages += pages;
        let token = (u32::from(generation) << 16) | (descriptor as u32 + 1);
        SwapExtent::from_raw(token)
    }

    /// Resolve a live handle to its logical slot range.
    pub fn range(&self, extent: SwapExtent, pages: u32) -> Option<Range<usize>> {
        if pages == 0 || pages > MAX_EXTENT_PAGES {
            return None;
        }
        let descriptor = usize::from((extent.0 as u16).checked_sub(1)?);
        let generation = (extent.0 >> 16) as u16;
        let entry = *self.descriptors.get(descriptor)?;
        let end = entry.start.checked_add(entry.pages)?;
        if generation == 0
            || !entry.live
            || entry.generation != generation
            || entry.pages != pages
            || end > self.capacity
            || (entry.start..end).any(|page| !self.is_claimed(page))
        {
            return None;
        }
        Some(entry.start as usize..end as usize)
    }

    /// Release exactly one live extent.  The generation is deliberately kept;
    /// its next reuse increments it, so the old handle can never name the new
    /// bytes.
    pub fn release(&mut self, extent: SwapExtent, pages: u32) -> Option<Range<usize>> {
        let range = self.range(extent, pages)?;
        let descriptor = usize::from((extent.0 as u16).checked_sub(1)?);
        for page in range.clone() {
            self.claimed[page / 64] &= !(1_u64 << (page % 64));
        }
        self.descriptors[descriptor].live = false;
        self.live_extents -= 1;
        self.live_pages -= pages;
        Some(range)
    }

    fn find_descriptor(&self) -> Option<usize> {
        if self.descriptors.is_empty() {
            return None;
        }
        (self.descriptor_cursor..self.descriptors.len())
            .chain(0..self.descriptor_cursor)
            .find(|&index| {
                let entry = self.descriptors[index];
                !entry.live && entry.generation != u16::MAX
            })
    }

    fn find_run(&self, begin: u32, limit: u32, pages: u32) -> Option<u32> {
        let mut start = begin;
        while start.checked_add(pages).is_some_and(|end| end <= limit) {
            let end = start + pages;
            match (start..end).find(|&page| self.is_claimed(page)) {
                Some(claimed) => start = claimed + 1,
                None => return Some(start),
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_handle_refuses_after_exact_slot_reuse() {
        let mut table = ExtentTable::new(8);
        let first = table.claim(8).unwrap();
        assert_eq!(table.range(first, 8), Some(0..8));
        assert_eq!(table.release(first, 8), Some(0..8));
        let second = table.claim(8).unwrap();
        assert_eq!(table.range(second, 8), Some(0..8));
        assert_ne!(first.token(), second.token());
        assert_eq!(table.range(first, 8), None);
        assert_eq!(table.release(first, 8), None);
    }

    #[test]
    fn next_fit_admits_non_power_of_two_extents_and_reuses_holes() {
        let mut table = ExtentTable::new(1_024);
        let a = table.claim(387).unwrap();
        let b = table.claim(512).unwrap();
        assert_eq!(table.range(a, 387), Some(0..387));
        assert_eq!(table.range(b, 512), Some(387..899));
        assert_eq!(table.live_extents(), 2);
        assert_eq!(table.live_pages(), 899);
        assert!(table.claim(126).is_none());
        table.release(a, 387).unwrap();
        let c = table.claim(126).unwrap();
        assert_eq!(table.range(c, 126), Some(0..126));
    }

    #[test]
    fn malformed_and_oversized_claims_refuse() {
        let mut table = ExtentTable::new(2_048);
        assert!(table.claim(0).is_none());
        assert!(table.claim(MAX_EXTENT_PAGES + 1).is_none());
        assert!(SwapExtent::from_raw(0).is_none());
        assert!(SwapExtent::from_raw(1).is_none());
        assert!(SwapExtent::from_raw(1 << 16).is_none());
    }

    #[test]
    fn exact_page_count_is_part_of_extent_identity() {
        let mut table = ExtentTable::new(16);
        let extent = table.claim(7).unwrap();
        assert_eq!(table.range(extent, 6), None);
        assert_eq!(table.range(extent, 8), None);
        assert_eq!(table.release(extent, 6), None);
        assert_eq!(table.live_extents(), 1);
        assert_eq!(table.live_pages(), 7);
        assert_eq!(table.release(extent, 7), Some(0..7));
    }

    /// The illustrative model-neutral 1M geometry from Axiom is 792,576
    /// backing frames split into 2,048 extents of 387 frames.  Admission is
    /// exact and does not depend on a model identifier.
    #[test]
    fn one_million_context_frame_geometry_fits_the_bounded_extent_table() {
        const FRAMES: u32 = 792_576;
        const EXTENT_PAGES: u32 = 387;
        let mut table = ExtentTable::new(FRAMES);
        for _ in 0..2_048 {
            assert!(table.claim(EXTENT_PAGES).is_some());
        }
        assert_eq!(table.live_extents(), 2_048);
        assert_eq!(table.live_pages(), FRAMES);
        assert!(table.claim(1).is_none());
    }
}
