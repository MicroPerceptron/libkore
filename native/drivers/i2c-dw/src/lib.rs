//! **Synopsys DesignWare I2C master** — one implementation of the
//! [`kore_i2c`] seam.
//!
//! The same IP is wired into two of the fleet's three machines: Intel's
//! LPSS controllers (PCI-mode on Lunar Lake) and AMD's `AMDI0010`
//! (ACPI-mode). What differs between them is not the transfer state
//! machine — that is bit-identical — but a reset block, an input clock,
//! and whether the interrupt mask reads back. Those three live in
//! [`Integration`] as data, so this crate has no `if intel` anywhere and
//! a third DesignWare integration is a constant, not a code path.
//!
//! Qualcomm's controller is **not** this IP (it is a GENI serial engine)
//! and gets its own crate implementing the same seam.
//!
//! # PIO only
//!
//! No `Dma` or `DmaPool` appears in any signature here. HID-over-I2C
//! transfers are tens of bytes; the LPSS integrated-DMA block would add a
//! descriptor ring, an IOMMU mapping, and a second completion path to move
//! payloads that fit in the FIFO twice over.
//!
//! [`Integration`]: integration::Integration

#![no_std]
#![forbid(unsafe_code)]

pub mod abort;
pub mod integration;
pub mod master;
pub mod regs;
pub mod timing;

#[cfg(any(test, feature = "mock"))]
pub mod mock;

pub use abort::{DwAbort, DwAbortCause};
pub use integration::Integration;
pub use master::DwMaster;

#[cfg(test)]
mod tests;
