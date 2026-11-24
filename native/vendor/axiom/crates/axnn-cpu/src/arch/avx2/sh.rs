//! SH → RGB (rendering "Kernel A"): the shared `core::simd` implementation.
//!
//! Its per-channel dot is `portable::f8::vec_dot_*`, which under
//! `+avx2,+f16c` lowers to F16C + AVX — LLVM's multi-accumulator schedule
//! measured at parity with (or ahead of) hand intrinsics — so this tier
//! reuses it rather than maintaining a second copy.
pub(crate) use crate::arch::portable::sh::{sh_eval_e4m3, sh_eval_e5m2, sh_eval_f16, sh_eval_f32};
