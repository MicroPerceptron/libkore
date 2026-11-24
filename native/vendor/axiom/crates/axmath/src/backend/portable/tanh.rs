//! Portable hyperbolic tangent functions.

#[inline(always)]
pub(crate) fn tanh_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return x.signum();
    }

    // The expm1-based form keeps the small-x region stable without
    // relying on exp(x) - 1 cancellation.
    if x >= 0.0 {
        let m = super::expm1_f32(-2.0 * x);
        -m / (2.0 + m)
    } else {
        let m = super::expm1_f32(2.0 * x);
        m / (2.0 + m)
    }
}

#[inline(always)]
pub(crate) fn tanh_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return x.signum();
    }

    if x >= 0.0 {
        let m = super::expm1_f64(-2.0 * x);
        -m / (2.0 + m)
    } else {
        let m = super::expm1_f64(2.0 * x);
        m / (2.0 + m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::assert_unary_f32_range;

    #[test]
    fn tanh_f32_special_cases() {
        assert!(tanh_f32(f32::NAN).is_nan());
        assert_eq!(tanh_f32(f32::INFINITY), 1.0);
        assert_eq!(tanh_f32(f32::NEG_INFINITY), -1.0);
    }

    #[test]
    fn tanh_f32_matches_libm_on_samples() {
        assert_unary_f32_range(
            "tanh_f32",
            tanh_f32,
            libm::tanhf,
            -10.0,
            10.0,
            800,
            2e-7,
            2e-7,
        );
    }
}
