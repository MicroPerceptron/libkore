//! The address abstraction the device-MMU walker is generic over.
//!
//! Deliberately **not** `memory_addr::MemoryAddr`: the driver-side output
//! address is [`kore_abi::device::IovaAddr`], a foreign newtype we cannot
//! implement a foreign trait for (orphan rule). A *local* trait sidesteps that
//! — `impl DeviceAddr for IovaAddr` is a local trait on a foreign type, which
//! is allowed — and keeps the walker off any heavyweight address crate.

use kore_abi::device::IovaAddr;

/// An address the walker moves between its typed newtype form and raw 64 bits.
///
/// Used on both sides of a translation: [`PagingMetaData::VirtAddr`] on the
/// input (a device virtual address) and [`PageTableEntry::OutputAddr`] on the
/// leaf/table-pointer output. An output type may carry identity in addition to
/// its numeric component (for example, system memory versus local VRAM).
///
/// [`PagingMetaData::VirtAddr`]: super::entry::PagingMetaData::VirtAddr
/// [`PageTableEntry::OutputAddr`]: super::entry::PageTableEntry::OutputAddr
pub trait DeviceAddr: Copy + Eq + core::fmt::Debug {
    /// Raw 64-bit value.
    fn to_u64(self) -> u64;

    /// Reconstruct from a raw 64-bit value.
    fn from_u64(v: u64) -> Self;

    /// Replace only the numeric component while preserving any address-space
    /// identity carried by `self`.
    ///
    /// Plain newtypes use [`from_u64`](Self::from_u64). A heterogeneous output
    /// address overrides this so range walking and interior translation cannot
    /// silently retarget the same number into a different memory space.
    #[inline]
    fn with_u64(self, v: u64) -> Self {
        Self::from_u64(v)
    }

    /// Whether the address is aligned to `align` (a power of two).
    #[inline]
    fn is_aligned_to(self, align: u64) -> bool {
        debug_assert!(align.is_power_of_two());
        self.to_u64() & (align - 1) == 0
    }
}

impl DeviceAddr for IovaAddr {
    #[inline]
    fn to_u64(self) -> u64 {
        self.0
    }

    #[inline]
    fn from_u64(v: u64) -> Self {
        IovaAddr(v)
    }
}

/// A device **virtual** address — the input side of a device-MMU translation
/// (e.g. an SMMUv3 stage-1 device VA). A distinct newtype so an input VA is never
/// confused with an output IOVA at a call site.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, PartialOrd, Ord)]
pub struct DevVirtAddr(pub u64);

impl DeviceAddr for DevVirtAddr {
    #[inline]
    fn to_u64(self) -> u64 {
        self.0
    }

    #[inline]
    fn from_u64(v: u64) -> Self {
        DevVirtAddr(v)
    }
}
