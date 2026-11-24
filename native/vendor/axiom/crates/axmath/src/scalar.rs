//! Public scalar math functions.
//!
//! These wrappers form the stable scalar API for the rest of the
//! workspace while internal backends remain free to evolve.

#[inline(always)]
pub fn exp_f32(x: f32) -> f32 {
    crate::backend::exp_f32(x)
}

#[inline(always)]
pub fn abs_f32(x: f32) -> f32 {
    crate::backend::abs_f32(x)
}

#[inline(always)]
pub fn floor_f32(x: f32) -> f32 {
    crate::backend::floor_f32(x)
}

#[inline(always)]
pub fn ceil_f32(x: f32) -> f32 {
    crate::backend::ceil_f32(x)
}

#[inline(always)]
pub fn round_f32(x: f32) -> f32 {
    crate::backend::round_f32(x)
}

#[inline(always)]
pub fn trunc_f32(x: f32) -> f32 {
    crate::backend::trunc_f32(x)
}

#[inline(always)]
pub fn copysign_f32(x: f32, y: f32) -> f32 {
    crate::backend::copysign_f32(x, y)
}

#[inline(always)]
pub fn scalbn_f32(x: f32, n: i32) -> f32 {
    crate::backend::scalbn_f32(x, n)
}

#[inline(always)]
pub fn frexp_f32(x: f32) -> (f32, i32) {
    crate::backend::frexp_f32(x)
}

#[inline(always)]
pub fn powi_f32(x: f32, n: i32) -> f32 {
    crate::backend::powi_f32(x, n)
}

#[inline(always)]
pub fn cbrt_f32(x: f32) -> f32 {
    crate::backend::cbrt_f32(x)
}

#[inline(always)]
pub fn hypot_f32(x: f32, y: f32) -> f32 {
    crate::backend::hypot_f32(x, y)
}

#[inline(always)]
pub fn sin_f32(x: f32) -> f32 {
    crate::backend::sin_f32(x)
}

#[inline(always)]
pub fn cos_f32(x: f32) -> f32 {
    crate::backend::cos_f32(x)
}

/// Evaluate sine and cosine with one shared range reduction.
#[inline(always)]
pub fn sin_cos_f32(x: f32) -> (f32, f32) {
    crate::backend::sin_cos_f32(x)
}

#[inline(always)]
pub fn tan_f32(x: f32) -> f32 {
    crate::backend::tan_f32(x)
}

#[inline(always)]
pub fn atan2_f32(y: f32, x: f32) -> f32 {
    crate::backend::atan2_f32(y, x)
}

#[inline(always)]
pub fn atan_f32(x: f32) -> f32 {
    crate::backend::atan_f32(x)
}

#[inline(always)]
pub fn asin_f32(x: f32) -> f32 {
    crate::backend::asin_f32(x)
}

#[inline(always)]
pub fn acos_f32(x: f32) -> f32 {
    crate::backend::acos_f32(x)
}

#[inline(always)]
pub fn exp2_f32(x: f32) -> f32 {
    crate::backend::exp2_f32(x)
}

#[inline(always)]
pub fn expm1_f32(x: f32) -> f32 {
    crate::backend::expm1_f32(x)
}

#[inline(always)]
pub fn log_f32(x: f32) -> f32 {
    crate::backend::log_f32(x)
}

#[inline(always)]
pub fn log10_f32(x: f32) -> f32 {
    crate::backend::log10_f32(x)
}

#[inline(always)]
pub fn log1p_f32(x: f32) -> f32 {
    crate::backend::log1p_f32(x)
}

#[inline(always)]
pub fn log2_f32(x: f32) -> f32 {
    crate::backend::log2_f32(x)
}

#[inline(always)]
pub fn sqrt_f32(x: f32) -> f32 {
    crate::backend::sqrt_f32(x)
}

#[inline(always)]
pub fn tanh_f32(x: f32) -> f32 {
    crate::backend::tanh_f32(x)
}

#[inline(always)]
pub fn exp_f64(x: f64) -> f64 {
    crate::backend::exp_f64(x)
}

#[inline(always)]
pub fn abs_f64(x: f64) -> f64 {
    crate::backend::abs_f64(x)
}

#[inline(always)]
pub fn floor_f64(x: f64) -> f64 {
    crate::backend::floor_f64(x)
}

#[inline(always)]
pub fn ceil_f64(x: f64) -> f64 {
    crate::backend::ceil_f64(x)
}

#[inline(always)]
pub fn round_f64(x: f64) -> f64 {
    crate::backend::round_f64(x)
}

#[inline(always)]
pub fn trunc_f64(x: f64) -> f64 {
    crate::backend::trunc_f64(x)
}

#[inline(always)]
pub fn copysign_f64(x: f64, y: f64) -> f64 {
    crate::backend::copysign_f64(x, y)
}

#[inline(always)]
pub fn scalbn_f64(x: f64, n: i32) -> f64 {
    crate::backend::scalbn_f64(x, n)
}

#[inline(always)]
pub fn frexp_f64(x: f64) -> (f64, i32) {
    crate::backend::frexp_f64(x)
}

#[inline(always)]
pub fn powi_f64(x: f64, n: i32) -> f64 {
    crate::backend::powi_f64(x, n)
}

#[inline(always)]
pub fn cbrt_f64(x: f64) -> f64 {
    crate::backend::cbrt_f64(x)
}

#[inline(always)]
pub fn hypot_f64(x: f64, y: f64) -> f64 {
    crate::backend::hypot_f64(x, y)
}

#[inline(always)]
pub fn sin_f64(x: f64) -> f64 {
    crate::backend::sin_f64(x)
}

#[inline(always)]
pub fn cos_f64(x: f64) -> f64 {
    crate::backend::cos_f64(x)
}

/// Evaluate sine and cosine with one shared range reduction.
#[inline(always)]
pub fn sin_cos_f64(x: f64) -> (f64, f64) {
    crate::backend::sin_cos_f64(x)
}

#[inline(always)]
pub fn tan_f64(x: f64) -> f64 {
    crate::backend::tan_f64(x)
}

#[inline(always)]
pub fn atan2_f64(y: f64, x: f64) -> f64 {
    crate::backend::atan2_f64(y, x)
}

#[inline(always)]
pub fn atan_f64(x: f64) -> f64 {
    crate::backend::atan_f64(x)
}

#[inline(always)]
pub fn asin_f64(x: f64) -> f64 {
    crate::backend::asin_f64(x)
}

#[inline(always)]
pub fn acos_f64(x: f64) -> f64 {
    crate::backend::acos_f64(x)
}

#[inline(always)]
pub fn exp2_f64(x: f64) -> f64 {
    crate::backend::exp2_f64(x)
}

#[inline(always)]
pub fn expm1_f64(x: f64) -> f64 {
    crate::backend::expm1_f64(x)
}

#[inline(always)]
pub fn log_f64(x: f64) -> f64 {
    crate::backend::log_f64(x)
}

#[inline(always)]
pub fn log10_f64(x: f64) -> f64 {
    crate::backend::log10_f64(x)
}

#[inline(always)]
pub fn log1p_f64(x: f64) -> f64 {
    crate::backend::log1p_f64(x)
}

#[inline(always)]
pub fn log2_f64(x: f64) -> f64 {
    crate::backend::log2_f64(x)
}

#[inline(always)]
pub fn sqrt_f64(x: f64) -> f64 {
    crate::backend::sqrt_f64(x)
}

#[inline(always)]
pub fn tanh_f64(x: f64) -> f64 {
    crate::backend::tanh_f64(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_f32(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn close_f64(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn log_identity_points() {
        assert!(close_f32(log_f32(1.0), 0.0));
        assert!(close_f64(log_f64(1.0), 0.0));
        assert!(close_f32(log10_f32(1.0), 0.0));
        assert!(close_f64(log10_f64(1.0), 0.0));
        assert!(close_f32(log1p_f32(0.0), 0.0));
        assert!(close_f64(log1p_f64(0.0), 0.0));
        assert!(close_f32(log2_f32(1.0), 0.0));
        assert!(close_f64(log2_f64(1.0), 0.0));
    }

    #[test]
    fn expm1_identity_points() {
        assert!(close_f32(exp2_f32(0.0), 1.0));
        assert!(close_f64(exp2_f64(0.0), 1.0));
        assert!(close_f32(expm1_f32(0.0), 0.0));
        assert!(close_f64(expm1_f64(0.0), 0.0));
    }

    #[test]
    fn sqrt_identity_points() {
        assert!(close_f32(sqrt_f32(4.0), 2.0));
        assert!(close_f64(sqrt_f64(9.0), 3.0));
    }

    #[test]
    fn rounding_identity_points() {
        assert!(close_f32(floor_f32(1.75), 1.0));
        assert!(close_f64(ceil_f64(1.25), 2.0));
        assert!(close_f32(round_f32(1.5), 2.0));
        assert!(close_f64(trunc_f64(-1.75), -1.0));
    }

    #[test]
    fn float_utility_identity_points() {
        assert!(close_f32(copysign_f32(2.0, -1.0), -2.0));
        assert!(close_f64(scalbn_f64(0.75, 1), 1.5));
        let (frac32, exp32) = frexp_f32(6.0);
        let (frac64, exp64) = frexp_f64(6.0);
        assert!(close_f32(scalbn_f32(frac32, exp32), 6.0));
        assert!(close_f64(scalbn_f64(frac64, exp64), 6.0));
    }

    #[test]
    fn power_identity_points() {
        assert!(close_f32(abs_f32(-3.0), 3.0));
        assert!(close_f64(powi_f64(2.0, 5), 32.0));
        assert!(close_f32(cbrt_f32(8.0), 2.0));
        assert!(close_f64(hypot_f64(3.0, 4.0), 5.0));
    }

    #[test]
    fn trig_identity_points() {
        assert!(close_f32(sin_f32(0.0), 0.0));
        assert!(close_f64(cos_f64(0.0), 1.0));
        assert!(close_f32(sin_f32(core::f32::consts::FRAC_PI_2), 1.0));
        assert!(close_f64(cos_f64(core::f64::consts::PI), -1.0));
        assert!(close_f32(tan_f32(core::f32::consts::FRAC_PI_4), 1.0));
        assert!(close_f64(atan2_f64(1.0, 1.0), core::f64::consts::FRAC_PI_4));
        assert!(close_f64(atan_f64(1.0), core::f64::consts::FRAC_PI_4));
        assert!(close_f32(asin_f32(1.0), core::f32::consts::FRAC_PI_2));
        assert!(close_f64(acos_f64(-1.0), core::f64::consts::PI));
    }

    #[test]
    fn tanh_is_odd() {
        assert!(close_f32(tanh_f32(1.0), -tanh_f32(-1.0)));
        assert!(close_f64(tanh_f64(1.0), -tanh_f64(-1.0)));
    }
}
