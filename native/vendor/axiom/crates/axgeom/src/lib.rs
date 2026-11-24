#![no_std]
#![cfg_attr(feature = "graph", feature(allocator_api))]

//! Backend-neutral geometry and multidimensional field kernels.
//!
//! The default core is allocation-free and depends only on `axmath`. Optional
//! `.ax` emission lives behind the `graph` feature and allocates exclusively
//! through a caller-provided [`core::alloc::Allocator`]. The crate deliberately
//! has no provider, scheduler policy, cache, window, clock, thread,
//! environment, or filesystem dependency.

#[cfg(test)]
extern crate std;

pub mod algebra;
#[cfg(feature = "graph")]
pub mod ax_field;
#[cfg(feature = "graph")]
pub mod ax_precision;
pub mod black_hole;
pub mod field;
pub mod gaussian;
#[cfg(feature = "graph")]
pub mod graph;
pub mod grid;
pub mod integrate;
#[cfg(feature = "graph")]
pub mod kerr_graph;
pub mod precision;
pub mod wormhole;
