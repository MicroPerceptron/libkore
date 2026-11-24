//! x86_64 exponential functions.
//!
//! The current x86_64 path shares the portable pure-Rust implementation.
//! If we later add x86-specific intrinsics or tuned kernels, this file
//! remains the place to do it without changing dispatch.

pub(crate) use crate::backend::portable::{
    exp_f32, exp_f64, exp2_f32, exp2_f64, expm1_f32, expm1_f64,
};
