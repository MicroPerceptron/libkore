//! Optional host Metal execution for a bounded `.ax` field graph.
//!
//! This backend compiles the canonical graph, including typed voxel and RGB
//! resources, into a single field dispatch. It does not change WGSL parsing or
//! the portable CPU/NPU paths. Callers must explicitly permit approximate
//! arithmetic: precise Metal math still differs from CPU arithmetic on some
//! extreme inputs and subnormal intermediates. Strict requests are rejected.

mod compile;

#[cfg(target_os = "macos")]
mod runtime;

#[cfg(target_os = "macos")]
pub use runtime::{MetalField, MetalResourceBinding, MetalTiming};
