//! ARM NEON kernel overrides for operations that need target-specific lowering.
//!
//! The vec/reduce/mem families are rooted here so AArch64 callers go through a
//! NEON tier even when an operation currently shares its `core::simd` or slice
//! implementation with portable. This module also provides hand-tuned NEON
//! intrinsics where portable lowering is insufficient: integer widening
//! multiply chains for quantized dot-products, unsigned-float promotion, and
//! element-aligned bare-metal raster loads.

#[allow(dead_code)]
pub(crate) mod gemm;
pub(crate) mod mem;
pub(crate) mod qdot;
pub(crate) mod raster;
pub(crate) mod reduce;
pub(crate) mod uf;
pub(crate) mod vec;

#[allow(dead_code)]
pub(crate) mod f4;

#[cfg(target_feature = "fhm")]
#[allow(dead_code)]
pub(crate) mod f8;
