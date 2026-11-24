//! Approximation-oriented scalar functions.
//!
//! Items in this module are explicitly opt-in and may trade away
//! accuracy or strict `libm`-style behavior for speed.

/// Fast approximate reciprocal square root.
///
/// This is intended for hot paths that explicitly opt into an
/// approximation-oriented primitive. It should not be treated as a
/// drop-in substitute for `1.0 / crate::sqrt_f32(x)` when strict
/// accuracy or exact edge behavior matters.
#[inline(always)]
pub fn rsqrt_f32_fast(x: f32) -> f32 {
    crate::backend::rsqrt_f32_fast(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_f32(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn rsqrt_fast_known_point() {
        assert!(close_f32(rsqrt_f32_fast(4.0), 0.5));
    }
}
