//! Shared scalar math helpers reused across backends.
//!
//! These are the small building blocks that portable code and
//! ISA-specific overrides can both lean on without duplicating
//! bit-level range reduction or polynomial kernels.

pub(crate) const LN2_F32: f32 = core::f32::consts::LN_2;
pub(crate) const LOG2E_F32: f32 = core::f32::consts::LOG2_E;
pub(crate) const LN2_F64: f64 = core::f64::consts::LN_2;
pub(crate) const LOG2E_F64: f64 = core::f64::consts::LOG2_E;
pub(crate) const LOG10E_F64: f64 = core::f64::consts::LOG10_E;
pub(crate) const LOG10_2_F64: f64 = core::f64::consts::LOG10_2;

const LN2_HI_F64: f64 = 6.931_471_803_691_238e-1;
const LN2_LO_F64: f64 = 1.908_214_929_270_587_7e-10;
const LN2_HI_F32: f32 = 6.931_381_2e-1;
const LN2_LO_F32: f32 = 9.058_001e-6;
const SQRT2_F64: f64 = core::f64::consts::SQRT_2;
const TWO_POW_52_F64: f64 = 4_503_599_627_370_496.0;

#[inline(always)]
pub(crate) fn round_nearest_i32_f32(x: f32) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

#[inline(always)]
pub(crate) fn round_nearest_i32_f64(x: f64) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

#[inline(always)]
pub(crate) fn pow2i_f32(n: i32) -> f32 {
    if n > 127 {
        f32::INFINITY
    } else if n >= -126 {
        f32::from_bits(((n + 127) as u32) << 23)
    } else if n >= -149 {
        f32::from_bits(1u32 << (n + 149))
    } else {
        0.0
    }
}

#[inline(always)]
pub(crate) fn pow2i_f64(n: i32) -> f64 {
    if n > 1023 {
        f64::INFINITY
    } else if n >= -1022 {
        f64::from_bits(((n + 1023) as u64) << 52)
    } else if n >= -1074 {
        f64::from_bits(1u64 << (n + 1074))
    } else {
        0.0
    }
}

#[inline(always)]
pub(crate) fn exp_reduce_f32(x: f32) -> (i32, f32) {
    let n = round_nearest_i32_f32(x * LOG2E_F32);
    let nf = n as f32;
    let r = (x - nf * LN2_HI_F32) - nf * LN2_LO_F32;
    (n, r)
}

#[inline(always)]
pub(crate) fn exp_reduce_f64(x: f64) -> (i32, f64) {
    let n = round_nearest_i32_f64(x * LOG2E_F64);
    let nf = n as f64;

    // hi/lo split keeps the reduced remainder accurate for larger inputs.
    let r = (x - nf * LN2_HI_F64) - nf * LN2_LO_F64;
    (n, r)
}

#[inline(always)]
pub(crate) fn exp2_reduce_f32(x: f32) -> (i32, f32) {
    let n = round_nearest_i32_f32(x);
    let r = (x - n as f32) * LN2_F32;
    (n, r)
}

#[inline(always)]
pub(crate) fn exp2_reduce_f64(x: f64) -> (i32, f64) {
    let n = round_nearest_i32_f64(x);
    let r = (x - n as f64) * LN2_F64;
    (n, r)
}

#[inline(always)]
pub(crate) fn expm1_kernel_f32(r: f32) -> f32 {
    let mut poly = 1.0 / 362880.0;
    poly = 1.0 / 40320.0 + r * poly;
    poly = 1.0 / 5040.0 + r * poly;
    poly = 1.0 / 720.0 + r * poly;
    poly = 1.0 / 120.0 + r * poly;
    poly = 1.0 / 24.0 + r * poly;
    poly = 1.0 / 6.0 + r * poly;
    poly = 0.5 + r * poly;
    poly = 1.0 + r * poly;
    r * poly
}

#[inline(always)]
pub(crate) fn expm1_kernel_f64(r: f64) -> f64 {
    let mut poly = 1.0 / 479001600.0;
    poly = 1.0 / 39916800.0 + r * poly;
    poly = 1.0 / 3628800.0 + r * poly;
    poly = 1.0 / 362880.0 + r * poly;
    poly = 1.0 / 40320.0 + r * poly;
    poly = 1.0 / 5040.0 + r * poly;
    poly = 1.0 / 720.0 + r * poly;
    poly = 1.0 / 120.0 + r * poly;
    poly = 1.0 / 24.0 + r * poly;
    poly = 1.0 / 6.0 + r * poly;
    poly = 0.5 + r * poly;
    poly = 1.0 + r * poly;
    r * poly
}

#[inline(always)]
pub(crate) fn log_reduce_f64(x: f64) -> Result<(i32, f64), f64> {
    if x.is_nan() {
        return Err(x);
    }
    if x < 0.0 {
        return Err(f64::NAN);
    }
    if x == 0.0 {
        return Err(f64::NEG_INFINITY);
    }
    if x.is_infinite() {
        return Err(f64::INFINITY);
    }

    let bits = x.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;

    let (e, m_bits) = if exp_bits == 0 {
        let normalized = x * TWO_POW_52_F64;
        let norm_bits = normalized.to_bits();
        let e = (((norm_bits >> 52) & 0x7ff) as i32) - 1023 - 52;
        let m_bits = (norm_bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000;
        (e, m_bits)
    } else {
        let e = exp_bits - 1023;
        let m_bits = (bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000;
        (e, m_bits)
    };

    let m = f64::from_bits(m_bits);
    if m > SQRT2_F64 {
        Ok((e + 1, m * 0.5))
    } else {
        Ok((e, m))
    }
}

#[inline(always)]
pub(crate) fn log_kernel_ln_f64(m: f64) -> f64 {
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;

    let c21 = 1.0 / 21.0;
    let c19 = 1.0 / 19.0;
    let c17 = 1.0 / 17.0;
    let c15 = 1.0 / 15.0;
    let c13 = 1.0 / 13.0;
    let c11 = 1.0 / 11.0;
    let c9 = 1.0 / 9.0;
    let c7 = 1.0 / 7.0;
    let c5 = 1.0 / 5.0;
    let c3 = 1.0 / 3.0;

    let mut poly = c21;
    poly = c19 + s2 * poly;
    poly = c17 + s2 * poly;
    poly = c15 + s2 * poly;
    poly = c13 + s2 * poly;
    poly = c11 + s2 * poly;
    poly = c9 + s2 * poly;
    poly = c7 + s2 * poly;
    poly = c5 + s2 * poly;
    poly = c3 + s2 * poly;
    poly = 1.0 + s2 * poly;

    2.0 * s * poly
}
