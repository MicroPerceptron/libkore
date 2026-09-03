//! `kore-addresses` — the address management support surface.

#![no_std]

#[cfg(test)]
extern crate std;

mod address;
mod context;
mod unsigned;

pub use address::MemoryAddress;
pub use context::MemoryContext;
pub use unsigned::{FromUsize, IntoUsize, Unsigned};
