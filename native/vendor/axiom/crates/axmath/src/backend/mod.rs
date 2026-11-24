//! Internal scalar math backend dispatch.
//!
//! Public scalar APIs route through this layer so architecture-specific
//! implementations can be introduced without reshaping dependents.

mod arch;
mod portable;
mod shared;

pub(crate) use arch::{
    abs_f32, abs_f64, acos_f32, acos_f64, asin_f32, asin_f64, atan_f32, atan_f64, atan2_f32,
    atan2_f64, cbrt_f32, cbrt_f64, ceil_f32, ceil_f64, copysign_f32, copysign_f64, cos_f32,
    cos_f64, exp_f32, exp_f64, exp2_f32, exp2_f64, expm1_f32, expm1_f64, floor_f32, floor_f64,
    frexp_f32, frexp_f64, gelu_f32, gelu_f64, hypot_f32, hypot_f64, log_f32, log_f64, log1p_f32,
    log1p_f64, log2_f32, log2_f64, log10_f32, log10_f64, powi_f32, powi_f64, round_f32, round_f64,
    rsqrt_f32_fast, scalbn_f32, scalbn_f64, sigmoid_f32, sigmoid_f64, silu_f32, silu_f64,
    sin_cos_f32, sin_cos_f64, sin_f32, sin_f64, sqrt_f32, sqrt_f64, tan_f32, tan_f64, tanh_f32,
    tanh_f64, trunc_f32, trunc_f64,
};
