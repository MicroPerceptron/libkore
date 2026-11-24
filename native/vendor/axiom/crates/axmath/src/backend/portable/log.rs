//! Portable logarithm functions.

use crate::backend::shared::{
    LN2_F64, LOG2E_F64, LOG10_2_F64, LOG10E_F64, log_kernel_ln_f64, log_reduce_f64,
};

#[inline(always)]
pub(crate) fn log_f32(x: f32) -> f32 {
    log_f64(x as f64) as f32
}

#[inline(always)]
pub(crate) fn log1p_f32(x: f32) -> f32 {
    log1p_f64(x as f64) as f32
}

#[inline(always)]
pub(crate) fn log10_f32(x: f32) -> f32 {
    log10_f64(x as f64) as f32
}

#[inline(always)]
pub(crate) fn log2_f32(x: f32) -> f32 {
    log2_f64(x as f64) as f32
}

#[inline(always)]
pub(crate) fn log_f64(x: f64) -> f64 {
    match log_reduce_f64(x) {
        Ok((e, m)) => (e as f64) * LN2_F64 + log_kernel_ln_f64(m),
        Err(special) => special,
    }
}

#[inline(always)]
pub(crate) fn log1p_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    if x < -1.0 {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    if x.abs() < 0.125 {
        let mut term = x;
        let mut sum = term;
        let mut n = 2.0f64;

        while n <= 14.0 {
            term *= -x;
            sum += term / n;
            n += 1.0;
        }

        return sum;
    }

    let u = 1.0 + x;
    let correction = (u - 1.0) - x;
    let log_u = log_f64(u);

    if correction == 0.0 {
        log_u
    } else {
        log_u - correction / u
    }
}

#[inline(always)]
pub(crate) fn log10_f64(x: f64) -> f64 {
    match log_reduce_f64(x) {
        Ok((e, m)) => (e as f64) * LOG10_2_F64 + log_kernel_ln_f64(m) * LOG10E_F64,
        Err(special) => special,
    }
}

#[inline(always)]
pub(crate) fn log2_f64(x: f64) -> f64 {
    match log_reduce_f64(x) {
        Ok((e, m)) => (e as f64) + log_kernel_ln_f64(m) * LOG2E_F64,
        Err(special) => special,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::assert_unary_f64_range;

    #[test]
    fn log_special_cases() {
        assert!(log_f32(f32::NAN).is_nan());
        assert!(log_f64(f64::NAN).is_nan());
        assert_eq!(log_f32(0.0), f32::NEG_INFINITY);
        assert_eq!(log_f64(0.0), f64::NEG_INFINITY);
        assert!(log_f32(-1.0).is_nan());
        assert!(log_f64(-1.0).is_nan());
    }

    #[test]
    fn log2_special_cases() {
        assert!(log2_f32(f32::NAN).is_nan());
        assert!(log2_f64(f64::NAN).is_nan());
        assert_eq!(log2_f32(0.0), f32::NEG_INFINITY);
        assert_eq!(log2_f64(0.0), f64::NEG_INFINITY);
        assert!(log2_f32(-1.0).is_nan());
        assert!(log2_f64(-1.0).is_nan());
    }

    #[test]
    fn log_matches_known_points() {
        assert!((log_f32(1.0) - 0.0).abs() < 1e-6);
        assert!((log_f32(core::f32::consts::E) - 1.0).abs() < 1e-5);
        assert!((log_f64(1.0) - 0.0).abs() < 1e-12);
        assert!((log_f64(core::f64::consts::E) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log2_matches_known_points() {
        assert!((log2_f32(1.0) - 0.0).abs() < 1e-6);
        assert!((log2_f32(2.0) - 1.0).abs() < 1e-5);
        assert!((log2_f32(8.0) - 3.0).abs() < 1e-5);
        assert!((log2_f64(1.0) - 0.0).abs() < 1e-12);
        assert!((log2_f64(2.0) - 1.0).abs() < 1e-12);
        assert!((log2_f64(8.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn log1p_special_cases() {
        assert!(log1p_f32(f32::NAN).is_nan());
        assert!(log1p_f64(f64::NAN).is_nan());
        assert_eq!(log1p_f32(-1.0), f32::NEG_INFINITY);
        assert_eq!(log1p_f64(-1.0), f64::NEG_INFINITY);
        assert!(log1p_f32(-2.0).is_nan());
        assert!(log1p_f64(-2.0).is_nan());
    }

    #[test]
    fn log1p_matches_known_points() {
        assert!((log1p_f32(0.0) - 0.0).abs() < 1e-6);
        assert!((log1p_f32(1.0) - core::f32::consts::LN_2).abs() < 1e-5);
        assert!((log1p_f64(0.0) - 0.0).abs() < 1e-12);
        assert!((log1p_f64(1.0) - core::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn log10_matches_known_points() {
        assert!((log10_f32(1.0) - 0.0).abs() < 1e-6);
        assert!((log10_f32(10.0) - 1.0).abs() < 1e-5);
        assert!((log10_f64(1.0) - 0.0).abs() < 1e-12);
        assert!((log10_f64(10.0) - 1.0).abs() < 1e-12);
        assert!((log10_f64(100.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn log_matches_libm_on_samples() {
        assert_unary_f64_range(
            "log_f64",
            log_f64,
            libm::log,
            0.125,
            1.0e6,
            1200,
            1e-12,
            1e-12,
        );
    }

    #[test]
    fn log2_matches_libm_on_samples() {
        assert_unary_f64_range(
            "log2_f64",
            log2_f64,
            libm::log2,
            0.125,
            1.0e6,
            1200,
            2e-12,
            2e-12,
        );
    }

    #[test]
    fn log1p_matches_libm_on_samples() {
        assert_unary_f64_range(
            "log1p_f64",
            log1p_f64,
            libm::log1p,
            -0.999,
            100.0,
            1200,
            1e-12,
            1e-12,
        );
    }

    #[test]
    fn log10_matches_libm_on_samples() {
        assert_unary_f64_range(
            "log10_f64",
            log10_f64,
            libm::log10,
            0.125,
            1.0e6,
            1200,
            1e-12,
            1e-12,
        );
    }
}
