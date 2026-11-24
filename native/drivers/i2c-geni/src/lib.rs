//! Qualcomm GENI I2C master, FIFO/PIO implementation of the generic
//! `kore-i2c` controller seam.  The protocol driver above this crate is
//! silicon-agnostic; selection happens by concrete `GeniMaster<R>` type.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(test)]
extern crate std;

pub mod abort;
pub mod master;
pub mod regs;

#[cfg(test)]
mod tests;

pub use abort::{GeniAbort, GeniAbortCause};
pub use master::GeniMaster;
