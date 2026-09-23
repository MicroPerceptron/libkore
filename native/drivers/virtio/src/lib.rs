//! Modern VirtIO guest foundations for capability-owned kore drivers.
//!
//! This crate deliberately does not implement a static, process-global HAL.
//! Every split queue owns DMA memory allocated from the caller's concrete
//! [`kore_driver::DmaPool`], and the PCI transport operates only on already
//! mapped BAR windows plus a redacted, bounds-checked capability layout.  No
//! physical address or PCI configuration-space access enters the driver core.

#![no_std]

#[cfg(test)]
extern crate alloc;

pub mod console;
pub mod owned;
pub mod pci;
pub mod pump;
pub mod split;

#[cfg(feature = "service")]
pub mod binding;
#[cfg(feature = "service")]
pub mod claim;

pub use kore_driver::seam::Regs64;
pub use kore_driver::{Regs, RegsNarrow};
