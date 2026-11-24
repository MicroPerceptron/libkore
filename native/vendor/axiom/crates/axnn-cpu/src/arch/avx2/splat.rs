// The splat compositor is `core::simd` end to end (quadratic-in-dx
// FMAs, the shared vectorized exp, masked blends); under `+avx2,+fma`
// it lowers to AVX directly, so this tier re-exports the portable
// kernels — same policy as `sh`.
pub(crate) use crate::arch::portable::splat::{
    PW, composite_strip_e4m3, composite_strip_f16, composite_strip_f32, composite_strip_uf8e4m4,
    composite_strip_uf8e5m3,
};
