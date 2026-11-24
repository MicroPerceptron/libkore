//! Portable exponential functions.

use crate::backend::shared::{
    exp_reduce_f32, exp_reduce_f64, exp2_reduce_f32, exp2_reduce_f64, expm1_kernel_f32,
    expm1_kernel_f64, pow2i_f32, pow2i_f64,
};

const EXP2_F32_OVERFLOW_INPUT: f32 = 88.722_84;
const EXP2_F32_UNDERFLOW_INPUT: f32 = -103.972_08;
const EXP2_BASE2_F32_OVERFLOW_INPUT: f32 = 128.0;
const EXP2_BASE2_F32_UNDERFLOW_INPUT: f32 = -150.0;
#[allow(dead_code)]
const EXP_F64_OVERFLOW_INPUT: f64 = 709.782_712_893_384;
#[allow(dead_code)]
const EXP_F64_UNDERFLOW_INPUT: f64 = -745.133_219_101_941_1;
#[allow(dead_code)]
const EXP2_BASE2_F64_OVERFLOW_INPUT: f64 = 1024.0;
#[allow(dead_code)]
const EXP2_BASE2_F64_UNDERFLOW_INPUT: f64 = -1075.0;

#[inline(always)]
pub(crate) fn exp_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x >= EXP2_F32_OVERFLOW_INPUT {
        return f32::INFINITY;
    }
    if x <= EXP2_F32_UNDERFLOW_INPUT {
        return 0.0;
    }

    let (n, r) = exp_reduce_f32(x);
    let fraction = 1.0 + expm1_kernel_f32(r);
    if n > 127 {
        // The rounded exponent can be 128 while fraction < 1, so the
        // exponential is still finite. Do not materialize 2^128 first.
        (fraction * pow2i_f32(n - 1)) * 2.0
    } else if n < -126 {
        // Round only the final result into the subnormal range. Constructing
        // a subnormal 2^n first loses the half-ulp underflow interval.
        (fraction * pow2i_f32(n + 24)) * pow2i_f32(-24)
    } else {
        fraction * pow2i_f32(n)
    }
}

#[inline(always)]
pub(crate) fn expm1_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    if x == f32::NEG_INFINITY {
        return -1.0;
    }
    if x >= EXP2_F32_OVERFLOW_INPUT {
        return f32::INFINITY;
    }
    if x <= EXP2_F32_UNDERFLOW_INPUT {
        return -1.0;
    }
    if x.abs() < 1.0e-4 {
        return expm1_kernel_f32(x);
    }

    let (n, r) = exp_reduce_f32(x);
    let em1_r = expm1_kernel_f32(r);

    if n == 0 {
        em1_r
    } else {
        let scale = pow2i_f32(n);
        (scale - 1.0) + scale * em1_r
    }
}

#[inline(always)]
pub(crate) fn exp2_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x >= EXP2_BASE2_F32_OVERFLOW_INPUT {
        return f32::INFINITY;
    }
    if x <= EXP2_BASE2_F32_UNDERFLOW_INPUT {
        return 0.0;
    }

    let (n, r) = exp2_reduce_f32(x);
    (1.0 + expm1_kernel_f32(r)) * pow2i_f32(n)
}

#[inline(always)]
pub(crate) fn exp_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x >= EXP_F64_OVERFLOW_INPUT {
        return f64::INFINITY;
    }
    if x <= EXP_F64_UNDERFLOW_INPUT {
        return 0.0;
    }

    let (n, r) = exp_reduce_f64(x);
    (1.0 + expm1_kernel_f64(r)) * pow2i_f64(n)
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) fn expm1_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == f64::NEG_INFINITY {
        return -1.0;
    }
    if x >= EXP_F64_OVERFLOW_INPUT {
        return f64::INFINITY;
    }
    if x <= EXP_F64_UNDERFLOW_INPUT {
        return -1.0;
    }
    if x.abs() < 1.0e-8 {
        return expm1_kernel_f64(x);
    }

    let (n, r) = exp_reduce_f64(x);
    let em1_r = expm1_kernel_f64(r);

    if n == 0 {
        em1_r
    } else {
        let scale = pow2i_f64(n);
        (scale - 1.0) + scale * em1_r
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(crate) fn exp2_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x >= EXP2_BASE2_F64_OVERFLOW_INPUT {
        return f64::INFINITY;
    }
    if x <= EXP2_BASE2_F64_UNDERFLOW_INPUT {
        return 0.0;
    }

    let (n, r) = exp2_reduce_f64(x);
    (1.0 + expm1_kernel_f64(r)) * pow2i_f64(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{assert_unary_f32_range, assert_unary_f64_range};

    #[test]
    fn exp_f32_special_cases() {
        assert!(exp_f32(f32::NAN).is_nan());
        assert_eq!(exp_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp_f32(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn exp_f32_matches_libm_on_samples() {
        assert_unary_f32_range("exp_f32", exp_f32, libm::expf, -80.0, 80.0, 640, 0.0, 1e-6);
    }

    #[test]
    fn exp_f32_preserves_the_finite_overflow_edge_and_subnormal_tail() {
        for x in [
            88.0, 88.4, 88.5, 88.7, -87.5, -88.0, -100.0, -103.5, -103.9, -104.0,
        ] {
            let actual = exp_f32(x);
            let expected = libm::expf(x);
            assert_eq!(actual.is_finite(), expected.is_finite(), "exp({x})");
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 2,
                "exp({x}): {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn exp2_matches_libm_on_samples() {
        assert_unary_f64_range(
            "exp2_f64",
            exp2_f64,
            libm::exp2,
            -100.0,
            100.0,
            800,
            0.0,
            1e-8,
        );
    }

    #[test]
    fn exp_f64_matches_libm_on_samples() {
        assert_unary_f64_range(
            "exp_f64",
            exp_f64,
            libm::exp,
            -700.0,
            700.0,
            1200,
            0.0,
            5e-13,
        );
    }

    #[test]
    fn expm1_special_cases() {
        assert!(expm1_f32(f32::NAN).is_nan());
        assert!(expm1_f64(f64::NAN).is_nan());
        assert_eq!(expm1_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(expm1_f64(f64::INFINITY), f64::INFINITY);
        assert_eq!(expm1_f32(f32::NEG_INFINITY), -1.0);
        assert_eq!(expm1_f64(f64::NEG_INFINITY), -1.0);
    }

    #[test]
    fn expm1_matches_libm_on_samples() {
        assert_unary_f64_range(
            "expm1_f64",
            expm1_f64,
            libm::expm1,
            -20.0,
            20.0,
            800,
            1e-12,
            1e-8,
        );
    }

    #[test]
    fn exp_f32_is_monotonic_on_small_grid() {
        let mut prev = exp_f32(-10.0);
        let mut x = -9.75f32;
        while x <= 10.0 {
            let cur = exp_f32(x);
            assert!(
                cur >= prev,
                "exp must be monotonic: prev={prev:e} cur={cur:e} x={x}"
            );
            prev = cur;
            x += 0.25;
        }
    }

    #[test]
    fn exp_f64_is_monotonic_on_small_grid() {
        let mut prev = exp_f64(-10.0);
        let mut x = -9.75f64;
        while x <= 10.0 {
            let cur = exp_f64(x);
            assert!(
                cur >= prev,
                "exp must be monotonic: prev={prev:e} cur={cur:e} x={x}"
            );
            prev = cur;
            x += 0.25;
        }
    }
}
