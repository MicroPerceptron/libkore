//! Composite activation functions built on top of elementary math.
//!
//! These are primarily useful for ML-style consumers. They remain in
//! `axmath` for now, but are kept out of the crate root so the root API
//! stays focused on broadly useful scalar math primitives.

#[inline(always)]
pub fn sigmoid_f32(x: f32) -> f32 {
    crate::backend::sigmoid_f32(x)
}

#[inline(always)]
pub fn silu_f32(x: f32) -> f32 {
    crate::backend::silu_f32(x)
}

#[inline(always)]
pub fn gelu_f32(x: f32) -> f32 {
    crate::backend::gelu_f32(x)
}

#[inline(always)]
pub fn sigmoid_f64(x: f64) -> f64 {
    crate::backend::sigmoid_f64(x)
}

#[inline(always)]
pub fn silu_f64(x: f64) -> f64 {
    crate::backend::silu_f64(x)
}

#[inline(always)]
pub fn gelu_f64(x: f64) -> f64 {
    crate::backend::gelu_f64(x)
}

/// Above this, `softplus` returns its input (PyTorch's default threshold).
const SOFTPLUS_THRESHOLD: f32 = 20.0;

/// `log(1 + exp(x))`, and `x` itself above 20, as PyTorch's `softplus`
/// with its defaults computes it. NaN stays NaN.
#[inline(always)]
pub fn softplus_f32(x: f32) -> f32 {
    if x > SOFTPLUS_THRESHOLD {
        x
    } else {
        crate::log1p_f32(crate::exp_f32(x))
    }
}

/// [`softplus_f32`] in F64, with the same threshold.
#[inline(always)]
pub fn softplus_f64(x: f64) -> f64 {
    if x > f64::from(SOFTPLUS_THRESHOLD) {
        x
    } else {
        crate::log1p_f64(crate::exp_f64(x))
    }
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
    fn sigmoid_center_is_half() {
        assert!(close_f32(sigmoid_f32(0.0), 0.5));
        assert!(close_f64(sigmoid_f64(0.0), 0.5));
    }

    #[test]
    fn softplus_is_linear_past_its_threshold_and_keeps_nan() {
        assert!(close_f32(softplus_f32(0.0), core::f32::consts::LN_2));
        assert!(close_f64(softplus_f64(0.0), core::f64::consts::LN_2));
        assert_eq!(softplus_f32(20.5), 20.5);
        assert_eq!(softplus_f64(100.0), 100.0);
        assert!(close_f32(softplus_f32(-30.0), 0.0));
        assert_eq!(softplus_f32(f32::NEG_INFINITY), 0.0);
        assert_eq!(softplus_f32(f32::INFINITY), f32::INFINITY);
        assert!(softplus_f32(f32::NAN).is_nan());
        // Continuous across the threshold to within its rounding.
        assert!((softplus_f32(20.0) - 20.0).abs() < 1e-5);
    }
}
