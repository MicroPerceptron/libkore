//! Portable power and norm-related functions.

use crate::backend::portable::float::{
    copysign_f32, copysign_f64, frexp_f32, frexp_f64, scalbn_f32, scalbn_f64,
};
use crate::backend::portable::sqrt::{sqrt_f32, sqrt_f64};

const F32_SIGN_MASK: u32 = 0x8000_0000;
const F64_SIGN_MASK: u64 = 0x8000_0000_0000_0000;
const CBRT2_F32: f32 = 1.259_921_1;
const CBRT4_F32: f32 = 1.587_401;
const CBRT2_F64: f64 = 1.259_921_049_894_873_2;
const CBRT4_F64: f64 = 1.587_401_051_968_199_4;

#[inline(always)]
pub(crate) fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & !F32_SIGN_MASK)
}

#[inline(always)]
pub(crate) fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & !F64_SIGN_MASK)
}

#[inline(always)]
pub(crate) fn powi_f32(x: f32, n: i32) -> f32 {
    if n == 0 {
        return 1.0;
    }
    if x.is_nan() {
        return x;
    }

    let mut base = x;
    let mut exp = n.unsigned_abs();
    let mut acc = 1.0f32;

    while exp != 0 {
        if exp & 1 != 0 {
            acc *= base;
        }
        exp >>= 1;
        if exp != 0 {
            base *= base;
        }
    }

    if n < 0 { 1.0 / acc } else { acc }
}

#[inline(always)]
pub(crate) fn powi_f64(x: f64, n: i32) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if x.is_nan() {
        return x;
    }

    let mut base = x;
    let mut exp = n.unsigned_abs();
    let mut acc = 1.0f64;

    while exp != 0 {
        if exp & 1 != 0 {
            acc *= base;
        }
        exp >>= 1;
        if exp != 0 {
            base *= base;
        }
    }

    if n < 0 { 1.0 / acc } else { acc }
}

#[inline(always)]
pub(crate) fn hypot_f32(x: f32, y: f32) -> f32 {
    if x.is_infinite() || y.is_infinite() {
        return f32::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }

    let mut ax = abs_f32(x);
    let mut ay = abs_f32(y);
    if ax < ay {
        core::mem::swap(&mut ax, &mut ay);
    }
    if ax == 0.0 {
        return 0.0;
    }

    let r = ay / ax;
    ax * sqrt_f32(1.0 + r * r)
}

#[inline(always)]
pub(crate) fn hypot_f64(x: f64, y: f64) -> f64 {
    if x.is_infinite() || y.is_infinite() {
        return f64::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    let mut ax = abs_f64(x);
    let mut ay = abs_f64(y);
    if ax < ay {
        core::mem::swap(&mut ax, &mut ay);
    }
    if ax == 0.0 {
        return 0.0;
    }

    let r = ay / ax;
    ax * sqrt_f64(1.0 + r * r)
}

#[inline(always)]
pub(crate) fn cbrt_f32(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }

    let ax = abs_f32(x);
    let (frac, exp) = frexp_f32(ax);
    let rem = exp.rem_euclid(3);
    let q = (exp - rem) / 3;
    let z = scalbn_f32(frac, rem);
    let mut y = match rem {
        0 => 1.0,
        1 => CBRT2_F32,
        _ => CBRT4_F32,
    };

    for _ in 0..5 {
        y = (2.0 * y + z / (y * y)) / 3.0;
    }

    copysign_f32(scalbn_f32(y, q), x)
}

#[inline(always)]
pub(crate) fn cbrt_f64(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }

    let ax = abs_f64(x);
    let (frac, exp) = frexp_f64(ax);
    let rem = exp.rem_euclid(3);
    let q = (exp - rem) / 3;
    let z = scalbn_f64(frac, rem);
    let mut y = match rem {
        0 => 1.0,
        1 => CBRT2_F64,
        _ => CBRT4_F64,
    };

    for _ in 0..7 {
        y = (2.0 * y + z / (y * y)) / 3.0;
    }

    copysign_f64(scalbn_f64(y, q), x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{assert_unary_f32_range, assert_unary_f64_range};

    #[test]
    fn abs_matches_known_points() {
        assert_eq!(abs_f32(-3.5), 3.5);
        assert_eq!(abs_f64(-0.0).to_bits(), 0.0f64.to_bits());
        assert!(abs_f32(f32::NAN).is_nan());
        assert!(abs_f64(f64::NAN).is_nan());
    }

    #[test]
    fn powi_matches_core_on_samples() {
        let exponents = [-9, -4, -1, 0, 1, 3, 8];
        let xs = [-3.0f32, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0];
        for n in exponents {
            for x in xs {
                let got = powi_f32(x, n);
                let want = x.powi(n);
                if want.is_nan() {
                    assert!(got.is_nan(), "powi_f32 x={x} n={n} got={got:e}");
                } else {
                    assert_eq!(got.to_bits(), want.to_bits(), "powi_f32 x={x} n={n}");
                }
            }
        }

        let xs = [-3.0f64, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0];
        for n in exponents {
            for x in xs {
                let got = powi_f64(x, n);
                let want = x.powi(n);
                if want.is_nan() {
                    assert!(got.is_nan(), "powi_f64 x={x} n={n} got={got:e}");
                } else {
                    assert_eq!(got.to_bits(), want.to_bits(), "powi_f64 x={x} n={n}");
                }
            }
        }
    }

    #[test]
    fn hypot_matches_libm_on_samples() {
        let pairs_f32 = [
            (0.0f32, 0.0f32),
            (3.0, 4.0),
            (1.0e-20, 2.0e-20),
            (1.0e10, 3.0e10),
            (-5.5, 12.25),
        ];
        for (x, y) in pairs_f32 {
            let got = hypot_f32(x, y);
            let want = libm::hypotf(x, y);
            let rel_err = abs_f32(got - want) / abs_f32(want).max(1.0);
            assert!(
                rel_err <= 1e-6,
                "hypot_f32 x={x} y={y} got={got:e} want={want:e} rel_err={rel_err:e}"
            );
        }

        let pairs_f64 = [
            (0.0f64, 0.0f64),
            (3.0, 4.0),
            (1.0e-200, 2.0e-200),
            (1.0e200, 3.0e200),
            (-5.5, 12.25),
        ];
        for (x, y) in pairs_f64 {
            let got = hypot_f64(x, y);
            let want = libm::hypot(x, y);
            let rel_err = abs_f64(got - want) / abs_f64(want).max(1.0);
            assert!(
                rel_err <= 1e-12,
                "hypot_f64 x={x} y={y} got={got:e} want={want:e} rel_err={rel_err:e}"
            );
        }
    }

    #[test]
    fn cbrt_special_cases() {
        assert_eq!(cbrt_f32(0.0), 0.0);
        assert_eq!(cbrt_f64(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(cbrt_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(cbrt_f64(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(cbrt_f32(f32::NAN).is_nan());
        assert!(cbrt_f64(f64::NAN).is_nan());
    }

    #[test]
    fn cbrt_matches_libm_on_samples() {
        assert_unary_f32_range(
            "cbrt_f32",
            cbrt_f32,
            libm::cbrtf,
            -1.0e12,
            1.0e12,
            1200,
            0.0,
            2e-6,
        );
        assert_unary_f64_range(
            "cbrt_f64",
            cbrt_f64,
            libm::cbrt,
            -1.0e300,
            1.0e300,
            1200,
            0.0,
            1e-12,
        );
    }
}
