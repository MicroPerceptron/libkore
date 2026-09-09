//! Exact, typed mapping shapes returned by the device page-table walker.

use super::{DeviceAddr, PageTableEntry};
use crate::common::flags::PageSize;

/// A checked half-open device-address range `[start, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceRange<A: DeviceAddr> {
    /// Inclusive start.
    start: A,
    /// Exclusive end.
    end: A,
}

impl<A: DeviceAddr> DeviceRange<A> {
    /// Construct a non-empty range, rejecting overflow and inverted bounds.
    #[inline]
    pub fn from_start_len(start: A, len: u64) -> Option<Self> {
        if len == 0 {
            return None;
        }
        let end = start.to_u64().checked_add(len)?;
        Some(Self {
            start,
            end: start.with_u64(end),
        })
    }

    /// Length in bytes.
    #[inline]
    pub fn len(self) -> u64 {
        self.end.to_u64() - self.start.to_u64()
    }

    /// A [`DeviceRange`] is non-empty by construction.
    #[inline]
    pub const fn is_empty(self) -> bool {
        false
    }

    /// Inclusive start.
    #[inline]
    pub const fn start(self) -> A {
        self.start
    }

    /// Exclusive end.
    #[inline]
    pub const fn end(self) -> A {
        self.end
    }

    /// Whether `addr` is inside the half-open range.
    #[inline]
    pub fn contains(self, addr: A) -> bool {
        let raw = addr.to_u64();
        self.start.to_u64() <= raw && raw < self.end.to_u64()
    }
}

/// One resolved hardware leaf, including its exact input range and attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapping<Entry, V>
where
    Entry: PageTableEntry,
    V: DeviceAddr,
{
    /// Exact device-virtual range covered by the leaf.
    pub range: DeviceRange<V>,
    /// Aligned output base encoded in the leaf.
    pub output: Entry::OutputAddr,
    /// Decoded format-specific flags.
    pub flags: Entry::Flags,
    /// Hardware leaf size.
    pub size: PageSize,
}

impl<Entry, V> Mapping<Entry, V>
where
    Entry: PageTableEntry,
    V: DeviceAddr,
{
    /// Translate an address inside this leaf to its exact output address.
    #[inline]
    pub fn translate(self, addr: V) -> Option<Entry::OutputAddr> {
        if !self.range.contains(addr) {
            return None;
        }
        let offset = addr.to_u64() - self.range.start.to_u64();
        self.output
            .to_u64()
            .checked_add(offset)
            .map(|raw| self.output.with_u64(raw))
    }
}
