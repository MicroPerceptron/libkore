//! Base-NEON elementwise vector tier.
//!
//! The portable vector kernels use 128-bit `core::simd` vectors, which map
//! directly to NEON on AArch64. Re-exporting the complete family here keeps
//! strict fusion helpers and ordinary vector operations on the selected
//! architecture path while leaving room for focused intrinsic overrides.

pub(crate) use crate::arch::portable::vec::*;
