//! Portable floating-point utility functions.

use crate::backend::shared::{pow2i_f32, pow2i_f64};

const F32_SIGN_MASK: u32 = 0x8000_0000;
const F32_EXP_MASK: u32 = 0x7f80_0000;
const F32_MANT_MASK: u32 = 0x007f_ffff;
const F64_SIGN_MASK: u64 = 0x8000_0000_0000_0000;
const F64_EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
const F64_MANT_MASK: u64 = 0x000f_ffff_ffff_ffff;

#[inline(always)]
pub(crate) fn copysign_f32(x: f32, y: f32) -> f32 {
    f32::from_bits((x.to_bits() & !F32_SIGN_MASK) | (y.to_bits() & F32_SIGN_MASK))
}

#[inline(always)]
pub(crate) fn copysign_f64(x: f64, y: f64) -> f64 {
    f64::from_bits((x.to_bits() & !F64_SIGN_MASK) | (y.to_bits() & F64_SIGN_MASK))
}

#[inline(always)]
pub(crate) fn trunc_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }

    let bits = x.to_bits();
    let exp = ((bits & F32_EXP_MASK) >> 23) as i32 - 127;
    if exp < 0 {
        return copysign_f32(0.0, x);
    }
    if exp >= 23 {
        return x;
    }

    let mask = (1u32 << (23 - exp)) - 1;
    if bits & mask == 0 {
        x
    } else {
        f32::from_bits(bits & !mask)
    }
}

#[inline(always)]
pub(crate) fn trunc_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }

    let bits = x.to_bits();
    let exp = ((bits & F64_EXP_MASK) >> 52) as i32 - 1023;
    if exp < 0 {
        return copysign_f64(0.0, x);
    }
    if exp >= 52 {
        return x;
    }

    let mask = (1u64 << (52 - exp)) - 1;
    if bits & mask == 0 {
        x
    } else {
        f64::from_bits(bits & !mask)
    }
}

#[inline(always)]
pub(crate) fn floor_f32(x: f32) -> f32 {
    let t = trunc_f32(x);
    if t == x || !x.is_finite() {
        t
    } else if x.is_sign_negative() {
        t - 1.0
    } else {
        t
    }
}

#[inline(always)]
pub(crate) fn floor_f64(x: f64) -> f64 {
    let t = trunc_f64(x);
    if t == x || !x.is_finite() {
        t
    } else if x.is_sign_negative() {
        t - 1.0
    } else {
        t
    }
}

#[inline(always)]
pub(crate) fn ceil_f32(x: f32) -> f32 {
    let t = trunc_f32(x);
    if t == x || !x.is_finite() {
        t
    } else if x.is_sign_positive() {
        t + 1.0
    } else {
        t
    }
}

#[inline(always)]
pub(crate) fn ceil_f64(x: f64) -> f64 {
    let t = trunc_f64(x);
    if t == x || !x.is_finite() {
        t
    } else if x.is_sign_positive() {
        t + 1.0
    } else {
        t
    }
}

#[inline(always)]
pub(crate) fn round_f32(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }

    if x.is_sign_negative() {
        ceil_f32(x - 0.5)
    } else {
        floor_f32(x + 0.5)
    }
}

#[inline(always)]
pub(crate) fn round_f64(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }

    if x.is_sign_negative() {
        ceil_f64(x - 0.5)
    } else {
        floor_f64(x + 0.5)
    }
}

#[inline(always)]
pub(crate) fn scalbn_f32(mut x: f32, mut n: i32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }

    while n > 127 {
        x *= pow2i_f32(127);
        n -= 127;
        if !x.is_finite() {
            return x;
        }
    }

    while n < -126 {
        x *= pow2i_f32(-126);
        n += 126;
        if x == 0.0 {
            return x;
        }
    }

    x * pow2i_f32(n)
}

#[inline(always)]
pub(crate) fn scalbn_f64(mut x: f64, mut n: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }

    while n > 1023 {
        x *= pow2i_f64(1023);
        n -= 1023;
        if !x.is_finite() {
            return x;
        }
    }

    while n < -1022 {
        x *= pow2i_f64(-1022);
        n += 1022;
        if x == 0.0 {
            return x;
        }
    }

    x * pow2i_f64(n)
}

#[inline(always)]
pub(crate) fn frexp_f32(x: f32) -> (f32, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }

    let bits = x.to_bits();
    let sign = bits & F32_SIGN_MASK;
    let exp_bits = ((bits & F32_EXP_MASK) >> 23) as i32;
    let mantissa = bits & F32_MANT_MASK;

    if exp_bits == 0 {
        let p = 31 - mantissa.leading_zeros() as i32;
        let normalized = mantissa << (23 - p);
        let frac = f32::from_bits(sign | (126u32 << 23) | (normalized & F32_MANT_MASK));
        (frac, p - 148)
    } else {
        let frac = f32::from_bits(sign | (126u32 << 23) | mantissa);
        (frac, exp_bits - 126)
    }
}

#[inline(always)]
pub(crate) fn frexp_f64(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }

    let bits = x.to_bits();
    let sign = bits & F64_SIGN_MASK;
    let exp_bits = ((bits & F64_EXP_MASK) >> 52) as i32;
    let mantissa = bits & F64_MANT_MASK;

    if exp_bits == 0 {
        let p = 63 - mantissa.leading_zeros() as i32;
        let normalized = mantissa << (52 - p);
        let frac = f64::from_bits(sign | (1022u64 << 52) | (normalized & F64_MANT_MASK));
        (frac, p - 1073)
    } else {
        let frac = f64::from_bits(sign | (1022u64 << 52) | mantissa);
        (frac, exp_bits - 1022)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{assert_unary_f32_range, assert_unary_f64_range};

    fn std_floor_f32(x: f32) -> f32 {
        x.floor()
    }

    fn std_ceil_f32(x: f32) -> f32 {
        x.ceil()
    }

    fn std_round_f32(x: f32) -> f32 {
        x.round()
    }

    fn std_trunc_f32(x: f32) -> f32 {
        x.trunc()
    }

    fn std_floor_f64(x: f64) -> f64 {
        x.floor()
    }

    fn std_ceil_f64(x: f64) -> f64 {
        x.ceil()
    }

    fn std_round_f64(x: f64) -> f64 {
        x.round()
    }

    fn std_trunc_f64(x: f64) -> f64 {
        x.trunc()
    }

    #[test]
    fn floor_matches_std_on_samples() {
        assert_unary_f32_range(
            "floor_f32",
            floor_f32,
            std_floor_f32,
            -1024.0,
            1024.0,
            4000,
            0.0,
            0.0,
        );
        assert_unary_f64_range(
            "floor_f64",
            floor_f64,
            std_floor_f64,
            -1024.0,
            1024.0,
            4000,
            0.0,
            0.0,
        );
    }

    #[test]
    fn ceil_matches_std_on_samples() {
        assert_unary_f32_range(
            "ceil_f32",
            ceil_f32,
            std_ceil_f32,
            -1024.0,
            1024.0,
            4000,
            0.0,
            0.0,
        );
        assert_unary_f64_range(
            "ceil_f64",
            ceil_f64,
            std_ceil_f64,
            -1024.0,
            1024.0,
            4000,
            0.0,
            0.0,
        );
    }

    #[test]
    fn round_matches_std_on_samples() {
        assert_unary_f32_range(
            "round_f32",
            round_f32,
            std_round_f32,
            -1024.0,
            1024.0,
            4000,
            0.0,
            0.0,
        );
        assert_unary_f64_range(
            "round_f64",
            round_f64,
            std_round_f64,
            -1024.0,
            1024.0,
            4000,
            0.0,
            0.0,
        );
    }

    #[test]
    fn trunc_matches_std_on_samples() {
        assert_unary_f32_range(
            "trunc_f32",
            trunc_f32,
            std_trunc_f32,
            -1024.0,
            1024.0,
            4000,
            0.0,
            0.0,
        );
        assert_unary_f64_range(
            "trunc_f64",
            trunc_f64,
            std_trunc_f64,
            -1024.0,
            1024.0,
            4000,
            0.0,
            0.0,
        );
    }

    #[test]
    fn copysign_preserves_magnitude_and_takes_sign() {
        assert_eq!(copysign_f32(3.5, -1.0).to_bits(), (-3.5f32).to_bits());
        assert_eq!(copysign_f32(-3.5, 1.0).to_bits(), 3.5f32.to_bits());
        assert_eq!(copysign_f64(3.5, -1.0).to_bits(), (-3.5f64).to_bits());
        assert_eq!(copysign_f64(-3.5, 1.0).to_bits(), 3.5f64.to_bits());
    }

    #[test]
    fn scalbn_handles_large_shifts_without_spurious_overflow() {
        assert_eq!(scalbn_f32(f32::from_bits(1), 149).to_bits(), 0x3f80_0000);
        assert_eq!(
            scalbn_f64(f64::from_bits(1), 1074).to_bits(),
            0x3ff0_0000_0000_0000
        );
        assert_eq!(scalbn_f32(1.0, -149).to_bits(), 1);
        assert_eq!(scalbn_f64(1.0, -1074).to_bits(), 1);
    }

    #[test]
    fn frexp_reconstructs_normals_and_subnormals() {
        let f32_values = [1.0f32, -6.75, f32::MIN_POSITIVE, f32::from_bits(1)];
        for x in f32_values {
            let (frac, exp) = frexp_f32(x);
            assert_eq!(scalbn_f32(frac, exp).to_bits(), x.to_bits());
            assert!(frac.abs() >= 0.5 && frac.abs() < 1.0);
        }

        let f64_values = [1.0f64, -6.75, f64::MIN_POSITIVE, f64::from_bits(1)];
        for x in f64_values {
            let (frac, exp) = frexp_f64(x);
            assert_eq!(scalbn_f64(frac, exp).to_bits(), x.to_bits());
            assert!(frac.abs() >= 0.5 && frac.abs() < 1.0);
        }
    }

    #[test]
    fn frexp_special_cases() {
        assert_eq!(frexp_f32(0.0), (0.0, 0));
        assert_eq!(frexp_f64(-0.0), (-0.0, 0));
        let (nan_frac, nan_exp) = frexp_f32(f32::NAN);
        assert!(nan_frac.is_nan());
        assert_eq!(nan_exp, 0);
        let (inf_frac, inf_exp) = frexp_f64(f64::INFINITY);
        assert_eq!(inf_frac, f64::INFINITY);
        assert_eq!(inf_exp, 0);
    }
}
