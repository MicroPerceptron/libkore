//! Portable square-root related functions.

use crate::backend::portable::{frexp_f64, scalbn_f64};

#[inline(always)]
pub(crate) fn rsqrt_f32_fast(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x == 0.0 {
        return f32::INFINITY;
    }
    if x < 0.0 {
        return f32::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }

    let x2 = x * 0.5;
    let mut y = x;
    let mut i = y.to_bits();
    i = 0x5f37_5a86u32 - (i >> 1);
    y = f32::from_bits(i);

    // Two Newton steps are still cheap and noticeably tighten accuracy.
    y = y * (1.5 - x2 * y * y);
    y = y * (1.5 - x2 * y * y);
    y
}

#[inline(always)]
pub(crate) fn sqrt_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x == 0.0 || x == f32::INFINITY {
        return x;
    }
    if x < 0.0 {
        return f32::NAN;
    }

    // Newton refinement alone may stop one ULP from the correctly rounded
    // result. Geometric normalization must agree with the SIMD sqrt path.
    // The existing no_std dependency uses hardware sqrt when available and
    // a rounding-aware software implementation elsewhere.
    libm::sqrtf(x)
}

#[inline(always)]
pub(crate) fn sqrt_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == 0.0 || x == f64::INFINITY {
        return x;
    }
    if x < 0.0 {
        return f64::NAN;
    }

    let (mut frac, mut exp) = frexp_f64(x);
    if (exp & 1) != 0 {
        frac *= 2.0;
        exp -= 1;
    }

    let mut y = 0.5 * (1.0 + frac);
    y = 0.5 * (y + frac / y);
    y = 0.5 * (y + frac / y);
    y = 0.5 * (y + frac / y);
    y = 0.5 * (y + frac / y);

    scalbn_f64(y, exp / 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{assert_unary_f32_range, assert_unary_f64_range};

    fn libm_sqrtf(x: f32) -> f32 {
        libm::sqrtf(x)
    }

    fn libm_sqrt(x: f64) -> f64 {
        libm::sqrt(x)
    }

    #[test]
    fn rsqrt_f32_fast_special_cases() {
        assert!(rsqrt_f32_fast(f32::NAN).is_nan());
        assert_eq!(rsqrt_f32_fast(0.0), f32::INFINITY);
        assert_eq!(rsqrt_f32_fast(f32::INFINITY), 0.0);
        assert!(rsqrt_f32_fast(-1.0).is_nan());
    }

    #[test]
    fn rsqrt_f32_fast_matches_reference_on_samples() {
        let samples = [0.125f32, 0.25, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1024.0];

        for x in samples {
            let got = rsqrt_f32_fast(x);
            let want = 1.0 / libm::sqrtf(x);
            let rel_err = (got - want).abs() / want;
            assert!(
                rel_err < 2e-3,
                "x={x} got={got:e} want={want:e} rel_err={rel_err:e}"
            );
        }
    }

    #[test]
    fn sqrt_special_cases() {
        assert!(sqrt_f32(f32::NAN).is_nan());
        assert_eq!(sqrt_f32(0.0), 0.0);
        assert_eq!(sqrt_f32(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(sqrt_f32(f32::INFINITY), f32::INFINITY);
        assert!(sqrt_f32(f32::NEG_INFINITY).is_nan());
        assert!(sqrt_f32(-1.0).is_nan());

        assert!(sqrt_f64(f64::NAN).is_nan());
        assert_eq!(sqrt_f64(0.0), 0.0);
        assert_eq!(sqrt_f64(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(sqrt_f64(f64::INFINITY), f64::INFINITY);
        assert!(sqrt_f64(f64::NEG_INFINITY).is_nan());
        assert!(sqrt_f64(-1.0).is_nan());
    }

    #[test]
    fn sqrt_matches_libm_on_samples() {
        let samples_f32 = [
            f32::from_bits(1),
            f32::from_bits(0x007f_ffff),
            f32::MIN_POSITIVE,
            0.125,
            0.5,
            1.0,
            2.0,
            3.0,
            10.0,
            1024.0,
            f32::MAX,
        ];

        for x in samples_f32 {
            let got = sqrt_f32(x);
            let want = libm::sqrtf(x);
            let rel_err = (got - want).abs() / want.abs().max(1.0);
            assert!(
                rel_err < 2e-7,
                "x={x:e} got={got:e} want={want:e} rel_err={rel_err:e}"
            );
        }

        let samples_f64 = [
            f64::from_bits(1),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MIN_POSITIVE,
            0.125,
            0.5,
            1.0,
            2.0,
            3.0,
            10.0,
            1024.0,
            f64::MAX,
        ];

        for x in samples_f64 {
            let got = sqrt_f64(x);
            let want = libm::sqrt(x);
            let rel_err = (got - want).abs() / want.abs().max(1.0);
            assert!(
                rel_err < 1e-14,
                "x={x:e} got={got:e} want={want:e} rel_err={rel_err:e}"
            );
        }
    }

    #[test]
    fn sqrt_matches_libm_on_ranges() {
        assert_unary_f32_range(
            "sqrt_f32", sqrt_f32, libm_sqrtf, 0.0, 1.0e6, 4096, 2e-7, 2e-7,
        );
        assert_unary_f64_range(
            "sqrt_f64", sqrt_f64, libm_sqrt, 0.0, 1.0e12, 4096, 1e-14, 1e-14,
        );
    }

    #[test]
    fn sqrt_f32_matches_host_rounding_across_finite_exponents() {
        // A one-ULP normalization change can alter a geometric boundary hit.
        // Compare to the host primitive, not the library used by the wrapper.
        let mut state = 0x82af_51c3_u32;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let input = f32::from_bits(state & 0x7fff_ffff);
            if input.is_finite() {
                assert_eq!(
                    sqrt_f32(input).to_bits(),
                    input.sqrt().to_bits(),
                    "input={:#010x}",
                    input.to_bits(),
                );
            }
        }
    }
}
