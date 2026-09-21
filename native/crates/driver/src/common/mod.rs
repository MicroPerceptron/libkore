//! `common` — the pluggable **device-MMU** library.
//!
//! Many accelerators embed their own MMU that translates a device VA into an
//! IOVA, sitting *behind* the platform IOMMU. The hard, error-prone, and
//! *shared* part of driving one is the multi-level page-table machinery —
//! identical across those devices bar the per-format PTE bit layout.
//!
//! This module factors that out, driver-side, exactly as the kernel's
//! `platform::paging` seam does kernel-side: a generic
//! [`DeviceWalker`](pagetable::DeviceWalker) over a
//! [`PageTableEntry`](pagetable::PageTableEntry) +
//! [`PagingMetaData`](pagetable::PagingMetaData) contract, with the output/leaf
//! address generalised to an associated type so a device instance's leaves hold
//! an `IovaAddr`. A vendor whose device shares the topology implements the
//! format traits and gets map / unmap / teardown for free.
//!
//! [`smmuv3`] lives here because SMMUv3 is a reusable architecture embedded by
//! devices from multiple vendors. Vendor-defined formats live in their
//! consuming vendor package instead; Intel PPGTT, for example, belongs to
//! `kore-xe`.
//!
//! The library is arch-neutral — pure data + algorithm, no `#[cfg(target_arch)]`
//! — so it compiles identically on x86_64, aarch64, and riscv64.

#[cfg(feature = "alloc")]
pub mod aperture;
pub mod flags;
pub mod pagetable;
pub mod smmuv3;
