//! Portable trigonometric functions.

use crate::backend::portable::sqrt::sqrt_f64;

const FRAC_2_PI_F64: f64 = core::f64::consts::FRAC_2_PI; // 2 / π
const FRAC_PI_4_F64: f64 = core::f64::consts::FRAC_PI_4; // π / 4
const PIO2_HI_F64: f64 = core::f64::consts::FRAC_PI_2; // π / 2
const PIO2_LO_F64: f64 = 6.123_233_995_736_766e-17; // π / 2 - PIO2_HI_F64, used for extra precision in reduction
const PI_F64: f64 = core::f64::consts::PI; // π
const TAN_PI_8_F64: f64 = 0.414_213_562_373_095_03; // tan(π / 8)
const TAN_3_PI_8_F64: f64 = 2.414_213_562_373_095; // tan(3π / 8)

#[inline(always)]
fn round_nearest_i64(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        (x - 0.5) as i64
    }
}

#[inline(always)]
fn trig_reduce_f64(x: f64) -> (i32, f64) {
    let k = round_nearest_i64(x * FRAC_2_PI_F64);
    let kf = k as f64;
    let r = (x - kf * PIO2_HI_F64) - kf * PIO2_LO_F64;
    (k.rem_euclid(4) as i32, r)
}

#[inline(always)]
fn trig_reduce_f32(x: f32) -> (i32, f32) {
    let (quadrant, r) = trig_reduce_f64(x as f64);
    (quadrant, r as f32)
}

#[inline(always)]
fn sin_kernel_f32(x: f32) -> f32 {
    let x2 = x * x;
    let mut poly = -1.0 / 39_916_800.0;
    poly = 1.0 / 362_880.0 + x2 * poly;
    poly = -1.0 / 5_040.0 + x2 * poly;
    poly = 1.0 / 120.0 + x2 * poly;
    poly = -1.0 / 6.0 + x2 * poly;
    x + x * x2 * poly
}

#[inline(always)]
fn cos_kernel_f32(x: f32) -> f32 {
    let x2 = x * x;
    let mut poly = -1.0 / 3_628_800.0;
    poly = 1.0 / 40_320.0 + x2 * poly;
    poly = -1.0 / 720.0 + x2 * poly;
    poly = 1.0 / 24.0 + x2 * poly;
    poly = -0.5 + x2 * poly;
    1.0 + x2 * poly
}

#[inline(always)]
fn sin_kernel_f64(x: f64) -> f64 {
    let x2 = x * x;
    let mut poly = 1.0 / 6_227_020_800.0;
    poly = -1.0 / 39_916_800.0 + x2 * poly;
    poly = 1.0 / 362_880.0 + x2 * poly;
    poly = -1.0 / 5_040.0 + x2 * poly;
    poly = 1.0 / 120.0 + x2 * poly;
    poly = -1.0 / 6.0 + x2 * poly;
    x + x * x2 * poly
}

#[inline(always)]
fn cos_kernel_f64(x: f64) -> f64 {
    let x2 = x * x;
    let mut poly = 1.0 / 479_001_600.0;
    poly = -1.0 / 3_628_800.0 + x2 * poly;
    poly = 1.0 / 40_320.0 + x2 * poly;
    poly = -1.0 / 720.0 + x2 * poly;
    poly = 1.0 / 24.0 + x2 * poly;
    poly = -0.5 + x2 * poly;
    1.0 + x2 * poly
}

#[inline(always)]
fn sin_cos_reduced_f32(quadrant: i32, r: f32) -> (f32, f32) {
    let s = sin_kernel_f32(r);
    let c = cos_kernel_f32(r);
    match quadrant {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

#[inline(always)]
fn sin_cos_reduced_f64(quadrant: i32, r: f64) -> (f64, f64) {
    let s = sin_kernel_f64(r);
    let c = cos_kernel_f64(r);
    match quadrant {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

#[inline(always)]
fn atan_kernel_f32(x: f32) -> f32 {
    let z = x * x;
    let mut poly = -1.0 / 21.0;
    poly = 1.0 / 19.0 + z * poly;
    poly = -1.0 / 17.0 + z * poly;
    poly = 1.0 / 15.0 + z * poly;
    poly = -1.0 / 13.0 + z * poly;
    poly = 1.0 / 11.0 + z * poly;
    poly = -1.0 / 9.0 + z * poly;
    poly = 1.0 / 7.0 + z * poly;
    poly = -1.0 / 5.0 + z * poly;
    poly = 1.0 / 3.0 + z * poly;
    x - x * z * poly
}

#[inline(always)]
fn atan_kernel_f64(x: f64) -> f64 {
    let z = x * x;
    let mut poly = -1.0 / 31.0;
    poly = 1.0 / 29.0 + z * poly;
    poly = -1.0 / 27.0 + z * poly;
    poly = 1.0 / 25.0 + z * poly;
    poly = -1.0 / 23.0 + z * poly;
    poly = 1.0 / 21.0 + z * poly;
    poly = -1.0 / 19.0 + z * poly;
    poly = 1.0 / 17.0 + z * poly;
    poly = -1.0 / 15.0 + z * poly;
    poly = 1.0 / 13.0 + z * poly;
    poly = -1.0 / 11.0 + z * poly;
    poly = 1.0 / 9.0 + z * poly;
    poly = -1.0 / 7.0 + z * poly;
    poly = 1.0 / 5.0 + z * poly;
    poly = -1.0 / 3.0 + z * poly;
    x + x * z * poly
}

#[inline(always)]
pub(crate) fn atan_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x == 0.0 {
        return x;
    }
    if x.is_infinite() {
        return if x.is_sign_negative() {
            -core::f32::consts::FRAC_PI_2
        } else {
            core::f32::consts::FRAC_PI_2
        };
    }

    let ax = x.abs();
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    // Keep the polynomial argument within tan(π/8). Reciprocal reduction
    // just above one would feed a value near one into the short Taylor sum.
    let y = if ax > TAN_3_PI_8_F64 as f32 {
        core::f32::consts::FRAC_PI_2 - atan_kernel_f32(1.0 / ax)
    } else if ax > TAN_PI_8_F64 as f32 {
        core::f32::consts::FRAC_PI_4 + atan_kernel_f32((ax - 1.0) / (ax + 1.0))
    } else {
        atan_kernel_f32(ax)
    };

    sign * y
}

#[inline(always)]
pub(crate) fn atan_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == 0.0 {
        return x;
    }
    if x.is_infinite() {
        return if x.is_sign_negative() {
            -PIO2_HI_F64
        } else {
            PIO2_HI_F64
        };
    }

    let ax = x.abs();
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let y = if ax > TAN_3_PI_8_F64 {
        PIO2_HI_F64 - atan_kernel_f64(1.0 / ax)
    } else if ax > TAN_PI_8_F64 {
        FRAC_PI_4_F64 + atan_kernel_f64((ax - 1.0) / (ax + 1.0))
    } else {
        atan_kernel_f64(ax)
    };

    sign * y
}

#[inline(always)]
pub(crate) fn sin_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }

    let (quadrant, r) = trig_reduce_f32(x);
    sin_cos_reduced_f32(quadrant, r).0
}

#[inline(always)]
pub(crate) fn sin_cos_f32(x: f32) -> (f32, f32) {
    if !x.is_finite() {
        return (f32::NAN, f32::NAN);
    }

    let (quadrant, r) = trig_reduce_f32(x);
    sin_cos_reduced_f32(quadrant, r)
}

#[inline(always)]
pub(crate) fn cos_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }

    let (quadrant, r) = trig_reduce_f32(x);
    sin_cos_reduced_f32(quadrant, r).1
}

#[inline(always)]
pub(crate) fn sin_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }

    let (quadrant, r) = trig_reduce_f64(x);
    sin_cos_reduced_f64(quadrant, r).0
}

#[inline(always)]
pub(crate) fn sin_cos_f64(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }

    let (quadrant, r) = trig_reduce_f64(x);
    sin_cos_reduced_f64(quadrant, r)
}

#[inline(always)]
pub(crate) fn cos_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }

    let (quadrant, r) = trig_reduce_f64(x);
    sin_cos_reduced_f64(quadrant, r).1
}

#[inline(always)]
pub(crate) fn tan_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }

    let (quadrant, r) = trig_reduce_f32(x);
    let (s, c) = sin_cos_reduced_f32(quadrant, r);
    s / c
}

#[inline(always)]
pub(crate) fn tan_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }

    let (quadrant, r) = trig_reduce_f64(x);
    let (s, c) = sin_cos_reduced_f64(quadrant, r);
    s / c
}

#[inline(always)]
pub(crate) fn atan2_f32(y: f32, x: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }

    if x == 0.0 {
        if y > 0.0 {
            return core::f32::consts::FRAC_PI_2;
        }
        if y < 0.0 {
            return -core::f32::consts::FRAC_PI_2;
        }
        if x.is_sign_negative() {
            return if y.is_sign_negative() {
                -core::f32::consts::PI
            } else {
                core::f32::consts::PI
            };
        }
        return y;
    }

    if x.is_infinite() && y.is_infinite() {
        let angle = core::f32::consts::FRAC_PI_4;
        return match (y.is_sign_negative(), x.is_sign_negative()) {
            (false, false) => angle,
            (false, true) => core::f32::consts::PI - angle,
            (true, false) => -angle,
            (true, true) => angle - core::f32::consts::PI,
        };
    }

    if x.is_infinite() {
        if x.is_sign_positive() {
            return if y.is_sign_negative() { -0.0 } else { 0.0 };
        }
        return if y.is_sign_negative() {
            -core::f32::consts::PI
        } else {
            core::f32::consts::PI
        };
    }

    if y.is_infinite() {
        return if y.is_sign_negative() {
            -core::f32::consts::FRAC_PI_2
        } else {
            core::f32::consts::FRAC_PI_2
        };
    }

    let angle = atan_f32(y / x);
    if x.is_sign_positive() {
        angle
    } else if y.is_sign_negative() {
        angle - core::f32::consts::PI
    } else {
        angle + core::f32::consts::PI
    }
}

#[inline(always)]
pub(crate) fn atan2_f64(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    if x == 0.0 {
        if y > 0.0 {
            return PIO2_HI_F64;
        }
        if y < 0.0 {
            return -PIO2_HI_F64;
        }
        if x.is_sign_negative() {
            return if y.is_sign_negative() {
                -PI_F64
            } else {
                PI_F64
            };
        }
        return y;
    }

    if x.is_infinite() && y.is_infinite() {
        return match (y.is_sign_negative(), x.is_sign_negative()) {
            (false, false) => FRAC_PI_4_F64,
            (false, true) => PI_F64 - FRAC_PI_4_F64,
            (true, false) => -FRAC_PI_4_F64,
            (true, true) => FRAC_PI_4_F64 - PI_F64,
        };
    }

    if x.is_infinite() {
        if x.is_sign_positive() {
            return if y.is_sign_negative() { -0.0 } else { 0.0 };
        }
        return if y.is_sign_negative() {
            -PI_F64
        } else {
            PI_F64
        };
    }

    if y.is_infinite() {
        return if y.is_sign_negative() {
            -PIO2_HI_F64
        } else {
            PIO2_HI_F64
        };
    }

    let angle = atan_f64(y / x);
    if x.is_sign_positive() {
        angle
    } else if y.is_sign_negative() {
        angle - PI_F64
    } else {
        angle + PI_F64
    }
}

#[inline(always)]
pub(crate) fn asin_f32(x: f32) -> f32 {
    asin_f64(x as f64) as f32
}

#[inline(always)]
pub(crate) fn acos_f32(x: f32) -> f32 {
    acos_f64(x as f64) as f32
}

#[inline(always)]
pub(crate) fn asin_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 1.0 {
        return PIO2_HI_F64;
    }
    if x == -1.0 {
        return -PIO2_HI_F64;
    }

    let y = sqrt_f64((1.0 - x) * (1.0 + x));
    2.0 * atan_f64(x / (1.0 + y))
}

#[inline(always)]
pub(crate) fn acos_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 1.0 {
        return 0.0;
    }
    if x == -1.0 {
        return PI_F64;
    }

    if x >= 0.0 {
        2.0 * atan_f64(sqrt_f64((1.0 - x) / (1.0 + x)))
    } else {
        PI_F64 - 2.0 * atan_f64(sqrt_f64((1.0 + x) / (1.0 - x)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{assert_unary_f32_range, assert_unary_f64_range};

    #[test]
    fn trig_special_cases() {
        assert!(sin_f32(f32::NAN).is_nan());
        assert!(cos_f32(f32::INFINITY).is_nan());
        assert!(sin_f64(f64::NEG_INFINITY).is_nan());
        assert!(cos_f64(f64::NAN).is_nan());
        assert!(tan_f32(f32::INFINITY).is_nan());
        assert!(tan_f64(f64::NEG_INFINITY).is_nan());
        assert!(atan2_f32(f32::NAN, 1.0).is_nan());
        assert!(atan2_f64(1.0, f64::NAN).is_nan());
        assert!(asin_f32(2.0).is_nan());
        assert!(acos_f64(-2.0).is_nan());
        let (sin, cos) = sin_cos_f32(f32::INFINITY);
        assert!(sin.is_nan());
        assert!(cos.is_nan());
    }

    #[test]
    fn sin_matches_known_points() {
        assert!(sin_f32(0.0).abs() < 1e-6);
        assert!((sin_f32(core::f32::consts::FRAC_PI_2) - 1.0).abs() < 1e-6);
        assert!(sin_f64(0.0).abs() < 1e-12);
        assert!((sin_f64(core::f64::consts::FRAC_PI_2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cos_matches_known_points() {
        assert!((cos_f32(0.0) - 1.0).abs() < 1e-6);
        assert!(cos_f32(core::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert!((cos_f64(0.0) - 1.0).abs() < 1e-12);
        assert!(cos_f64(core::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn paired_sine_cosine_matches_the_scalar_paths() {
        for x in [-10_000.0_f32, -3.0, 0.0, 1.25, 10_000.0] {
            let (sin, cos) = sin_cos_f32(x);
            assert_eq!(sin, sin_f32(x));
            assert_eq!(cos, cos_f32(x));
        }
        for x in [-10_000.0_f64, -3.0, 0.0, 1.25, 10_000.0] {
            let (sin, cos) = sin_cos_f64(x);
            assert_eq!(sin, sin_f64(x));
            assert_eq!(cos, cos_f64(x));
        }
    }

    #[test]
    fn sin_matches_libm_on_samples() {
        assert_unary_f32_range(
            "sin_f32",
            sin_f32,
            libm::sinf,
            -10_000.0,
            10_000.0,
            4000,
            0.0,
            2e-6,
        );
        assert_unary_f64_range(
            "sin_f64",
            sin_f64,
            libm::sin,
            -10_000.0,
            10_000.0,
            4000,
            0.0,
            1e-12,
        );
    }

    #[test]
    fn cos_matches_libm_on_samples() {
        assert_unary_f32_range(
            "cos_f32",
            cos_f32,
            libm::cosf,
            -10_000.0,
            10_000.0,
            4000,
            0.0,
            2e-6,
        );
        assert_unary_f64_range(
            "cos_f64",
            cos_f64,
            libm::cos,
            -10_000.0,
            10_000.0,
            4000,
            0.0,
            1e-12,
        );
    }

    #[test]
    fn tan_matches_known_points() {
        assert!(tan_f32(0.0).abs() < 1e-6);
        assert!((tan_f32(core::f32::consts::FRAC_PI_4) - 1.0).abs() < 1e-5);
        assert!(tan_f64(0.0).abs() < 1e-12);
        assert!((tan_f64(core::f64::consts::FRAC_PI_4) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn atan2_matches_known_points() {
        assert!(atan2_f32(0.0, 1.0).abs() < 1e-6);
        assert!((atan2_f32(1.0, 0.0) - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((atan2_f64(1.0, 1.0) - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((atan2_f64(1.0, -1.0) - 3.0 * core::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn atan_matches_known_points() {
        assert!(atan_f32(0.0).abs() < 1e-6);
        assert!((atan_f32(1.0) - core::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert!(atan_f64(0.0).abs() < 1e-12);
        assert!((atan_f64(1.0) - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn atan_matches_libm_on_samples() {
        // The wide sweep below steps by 500 and misses the range-reduction
        // transitions around one. Exercise every reduction interval densely.
        assert_unary_f32_range(
            "atan_f32_reduction",
            atan_f32,
            libm::atanf,
            -3.0,
            3.0,
            6000,
            0.0,
            2e-6,
        );
        assert_unary_f64_range(
            "atan_f64_reduction",
            atan_f64,
            libm::atan,
            -3.0,
            3.0,
            6000,
            0.0,
            1e-12,
        );
        assert_unary_f32_range(
            "atan_f32",
            atan_f32,
            libm::atanf,
            -1.0e6,
            1.0e6,
            4000,
            0.0,
            2e-6,
        );
        assert_unary_f64_range(
            "atan_f64",
            atan_f64,
            libm::atan,
            -1.0e6,
            1.0e6,
            4000,
            0.0,
            1e-11,
        );
    }

    #[test]
    fn tan_matches_libm_on_samples() {
        let mut x = -100.0f32;
        while x <= 100.0 {
            let want = libm::tanf(x);
            if libm::cosf(x).abs() > 0.05 {
                let got = tan_f32(x);
                let rel_err = (got - want).abs() / want.abs().max(1.0);
                assert!(
                    rel_err <= 5e-6,
                    "tan_f32 x={x} got={got:e} want={want:e} rel_err={rel_err:e}"
                );
            }
            x += 0.05;
        }

        let mut x = -100.0f64;
        while x <= 100.0 {
            let want = libm::tan(x);
            if libm::cos(x).abs() > 0.05 {
                let got = tan_f64(x);
                let rel_err = (got - want).abs() / want.abs().max(1.0);
                assert!(
                    rel_err <= 1e-11,
                    "tan_f64 x={x} got={got:e} want={want:e} rel_err={rel_err:e}"
                );
            }
            x += 0.05;
        }
    }

    #[test]
    fn atan2_matches_libm_on_samples() {
        let xs_f32 = [
            -10.0f32,
            -1.001,
            -1.0,
            -0.25,
            -0.0,
            0.0,
            0.25,
            1.0,
            1.001,
            1.5,
            10.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let ys_f32 = [
            -10.0f32,
            -1.0,
            -0.25,
            -0.0,
            0.0,
            0.25,
            1.0,
            10.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for y in ys_f32 {
            for x in xs_f32 {
                let got = atan2_f32(y, x);
                let want = libm::atan2f(y, x);
                if want.is_nan() {
                    assert!(got.is_nan(), "atan2_f32 y={y} x={x} got={got:e}");
                } else {
                    let abs_err = (got - want).abs();
                    assert!(
                        abs_err <= 2e-6,
                        "atan2_f32 y={y} x={x} got={got:e} want={want:e} abs_err={abs_err:e}"
                    );
                }
            }
        }

        let xs_f64 = [
            -10.0f64,
            -1.001,
            -1.0,
            -0.25,
            -0.0,
            0.0,
            0.25,
            1.0,
            1.001,
            1.5,
            10.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let ys_f64 = [
            -10.0f64,
            -1.0,
            -0.25,
            -0.0,
            0.0,
            0.25,
            1.0,
            10.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for y in ys_f64 {
            for x in xs_f64 {
                let got = atan2_f64(y, x);
                let want = libm::atan2(y, x);
                if want.is_nan() {
                    assert!(got.is_nan(), "atan2_f64 y={y} x={x} got={got:e}");
                } else {
                    let abs_err = (got - want).abs();
                    assert!(
                        abs_err <= 1e-12,
                        "atan2_f64 y={y} x={x} got={got:e} want={want:e} abs_err={abs_err:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn asin_acos_matches_known_points() {
        assert!(asin_f32(0.0).abs() < 1e-6);
        assert!((asin_f32(1.0) - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((acos_f32(1.0) - 0.0).abs() < 1e-6);
        assert!((acos_f32(-1.0) - core::f32::consts::PI).abs() < 1e-5);
        assert!(asin_f64(0.0).abs() < 1e-12);
        assert!((asin_f64(-1.0) + core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((acos_f64(1.0) - 0.0).abs() < 1e-12);
        assert!((acos_f64(-1.0) - core::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn asin_matches_libm_on_samples() {
        assert_unary_f32_range(
            "asin_f32",
            asin_f32,
            libm::asinf,
            -1.0,
            1.0,
            4000,
            0.0,
            3e-6,
        );
        assert_unary_f64_range("asin_f64", asin_f64, libm::asin, -1.0, 1.0, 4000, 0.0, 1e-9);
    }

    #[test]
    fn acos_matches_libm_on_samples() {
        assert_unary_f32_range(
            "acos_f32",
            acos_f32,
            libm::acosf,
            -1.0,
            1.0,
            4000,
            0.0,
            3e-6,
        );
        assert_unary_f64_range("acos_f64", acos_f64, libm::acos, -1.0, 1.0, 4000, 0.0, 1e-9);
    }
}
