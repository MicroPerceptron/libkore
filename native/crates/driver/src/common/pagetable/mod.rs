//! The driver-side generic multi-level page-table engine — a mirror of the
//! kernel's `platform::paging` seam (same trait *shapes*, so there is one mental
//! model), generalised over the **output address** so a device instance's leaves
//! hold an `IovaAddr` rather than a `PhysAddr`.
//!
//! A device format implements [`PageTableEntry`] + [`PagingMetaData`]; the
//! shared [`DeviceWalker`] does the tree mechanics unchanged. The format may
//! be an open architecture such as SMMUv3 or live under a vendor namespace.

pub mod addr;
pub mod alloc;
pub mod entry;
pub mod mapping;
pub mod walker;

pub use addr::{DevVirtAddr, DeviceAddr};
pub use alloc::{MmuError, MmuResult, TableSource};
pub use entry::{EntryKind, PageTableEntry, PagingMetaData};
pub use mapping::{DeviceRange, Mapping};
pub use walker::{DeviceWalker, DeviceWalkerView};

#[cfg(feature = "alloc")]
pub use alloc::DmaPoolTables;
