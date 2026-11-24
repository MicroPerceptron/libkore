//! Portable SIMD tier — `core::simd`-based kernels that compile
//! optimally on all targets (NEON, SSE, AVX2, …) without hand-written
//! intrinsics.
//!
//! Lane widths are selected at compile time via `target_feature`:
//!   - AArch64 NEON / x86_64 SSE2:  `W=4`  (128-bit)
//!   - x86_64 AVX2 / AVX-512:       `W=8`  (256-bit)
//!
//! Quantized dot-products (qdot) use a scalar fallback here; arch-
//! specific tiers (avx2, neon) override qdot with hand-tuned intrinsics
//! when enabled.

// Functions are consumed via selective re-export in arch/mod.rs.
#[allow(dead_code)]
pub(crate) mod act;

pub mod bitmat;
#[allow(dead_code)]
pub(crate) mod cast;
#[allow(dead_code)]
pub(crate) mod conv;
#[allow(dead_code)]
pub(crate) mod f4;
#[allow(dead_code)]
pub(crate) mod f8;
#[allow(dead_code)]
pub(crate) mod gemm;
#[allow(dead_code)]
pub(crate) mod gemv;
#[allow(dead_code)]
pub(crate) mod mem;
#[allow(dead_code)]
pub(crate) mod qdot;
pub mod radiation;
#[allow(dead_code)]
pub(crate) mod raster;
#[allow(dead_code)]
pub(crate) mod reduce;
#[allow(dead_code)]
pub(crate) mod rope;
#[allow(dead_code)]
pub(crate) mod sh;
#[allow(dead_code)]
pub(crate) mod splat;
#[allow(dead_code)]
pub(crate) mod vec;
#[allow(dead_code)]
pub mod voxel;

// ── Lane widths: selected by target ISA, not by guessing ────────────

/// F32 SIMD lane width — matches the target's native register width.
#[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
pub(super) const W: usize = 8;
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
pub(super) const W: usize = 4;

/// F64 SIMD lane width.
#[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
pub(super) const WD: usize = 4;
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
pub(super) const WD: usize = 2;
