//! Base-NEON reduction tier.
//!
//! The portable reduction kernels use 128-bit `core::simd` vectors, which map
//! directly to NEON on AArch64. Re-exporting the family here gives AArch64 a
//! distinct dispatch tier and a home for focused intrinsic overrides.

pub(crate) use crate::arch::portable::reduce::*;
