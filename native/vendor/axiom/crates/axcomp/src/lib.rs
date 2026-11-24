//! Portable compositor mechanics for Axiom.
//!
//! `axcomp` owns the reusable state between a UI frontend, `.ax` graph
//! execution, and a platform presenter. It deliberately does not own an event
//! loop, window, clock, environment, device provider, or allocation policy.
//! The final runtime supplies every allocator and drives all progress.

#![no_std]
#![feature(allocator_api)]
#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod cache;
mod cursor;
mod dart;
mod family;
mod frames;
mod glyph;
mod pack;
mod shapes;
mod targets;

pub use cache::{
    CacheError, PreparedCache, PreparedScene, SignatureCache, structural_graph_signature,
};
pub use cursor::{
    ARROW_HEIGHT, ARROW_HOTSPOT, ARROW_RGBA, ARROW_WIDTH, CompositeCursor, CursorController,
    CursorDamage, CursorRoute, CursorTransition, CursorVisual, RING_HEIGHT, RING_HOTSPOT,
    RING_RGBA, RING_WIDTH,
};
pub use dart::{
    DART_HEIGHT, DART_HOTSPOT, DART_PIXELS, DART_RGBA_BYTES, DART_WIDTH, DartCursor, DartPose,
    DartRasterError,
};
pub use family::{CursorShape, FAMILY_HEIGHT, FAMILY_WIDTH, FamilyCursor, FamilyPose};
pub use frames::{FrameError, FrameId, FrameQueue};
pub use pack::{
    LinearXrgbBytes, LinearXrgbWords, PackError, PlanarFormat, PlanarOutput, bounded_damage,
    pack_planar_xrgb_bytes, pack_planar_xrgb_words,
};
pub use targets::{TargetId, TargetRing, TargetRingError};
