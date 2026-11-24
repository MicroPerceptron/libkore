//! Element-wise vector operations — AVX2 + FMA.
//!
//! F32: 8 elements per 256-bit register.
//! F64: 4 elements per 256-bit register.
//! F16/BF16: widen to f32, compute, narrow back.
//!
//! Transcendental ops (exp, tanh, sigmoid, silu, gelu) use a fast
//! polynomial exp approximation (~1 ULP via Cephes decomposition).
//!
//! Raster primitives (cmpge_f32, and_masks, maskstore_u32): 8-wide
//! coverage test pipeline for Pineda edge-function rasterization.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

// The strict graph-fusion helpers use the portable SIMD implementation on
// AVX2. Re-export them so public elementwise entry points can consistently
// route through the selected `arch::vec` tier.
pub(crate) use crate::arch::portable::vec::{
    guarded_sqrt_f32_inplace, lerp_scalar_f32_inplace, linearized_clamp_f32_inplace, mul_add_f32,
    mul_add_f32_inplace, scale_add_f32, scale_add_f32_inplace, scale_add_scalar_f32,
    scale_add_scalar_f32_inplace, scale_interleave3_f32,
};

// ═══════════════════════════════════════════════════════════════════
//  Fast vectorised exp(x) — Cephes-style range reduction
// ═══════════════════════════════════════════════════════════════════
//
// exp(x) = 2^n · exp(r)  where n = round(x / ln2), r = x - n·ln2
// exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720
//
// Repair non-normal exponent ranges through the scalar contract.

#[inline(always)]
unsafe fn exp_f32x8(x: __m256) -> __m256 {
    let original = x;
    let ln2 = _mm256_set1_ps(core::f32::consts::LN_2);
    let ln2_inv = _mm256_set1_ps(1.0 / core::f32::consts::LN_2);
    let one = _mm256_set1_ps(1.0);

    // Polynomial coefficients (Horner form for exp(r))
    let c2 = _mm256_set1_ps(1.0 / 2.0);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let c6 = _mm256_set1_ps(1.0 / 720.0);

    // Keep exponent-bit construction within normal finite powers of two.
    let lo = _mm256_set1_ps(-87.0);
    let hi = _mm256_set1_ps(88.0);
    let x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);

    // n = round(x / ln2)
    let n = _mm256_round_ps(
        _mm256_mul_ps(x, ln2_inv),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    // r = x - n * ln2  (high precision via FMA)
    let r = _mm256_fnmadd_ps(n, ln2, x);

    // Polynomial: 1 + r(1 + r(c2 + r(c3 + r(c4 + r(c5 + r·c6)))))
    let mut p = _mm256_fmadd_ps(r, c6, c5);
    p = _mm256_fmadd_ps(r, p, c4);
    p = _mm256_fmadd_ps(r, p, c3);
    p = _mm256_fmadd_ps(r, p, c2);
    p = _mm256_fmadd_ps(r, p, one);
    p = _mm256_fmadd_ps(r, p, one);

    // 2^n via integer exponent manipulation
    let ni = _mm256_cvtps_epi32(n);
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ni, _mm256_set1_epi32(127)),
        23,
    ));

    let result = _mm256_mul_ps(p, pow2n);
    let normal = _mm256_and_ps(
        _mm256_cmp_ps(original, lo, _CMP_GE_OQ),
        _mm256_cmp_ps(original, hi, _CMP_LE_OQ),
    );
    if _mm256_movemask_ps(normal) != 0xff {
        let mut inputs = [0.0f32; 8];
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(inputs.as_mut_ptr(), original);
        _mm256_storeu_ps(lanes.as_mut_ptr(), result);
        for lane in 0..8 {
            if !(-87.0..=88.0).contains(&inputs[lane]) {
                lanes[lane] = axmath::exp_f32(inputs[lane]);
            }
        }
        _mm256_loadu_ps(lanes.as_ptr())
    } else {
        result
    }
}

/// Fast vectorised tanh(x) = (e^2x - 1) / (e^2x + 1)
#[inline(always)]
unsafe fn tanh_f32x8(x: __m256) -> __m256 {
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let e2x = exp_f32x8(_mm256_mul_ps(x, two));
    _mm256_div_ps(_mm256_sub_ps(e2x, one), _mm256_add_ps(e2x, one))
}

// ═══════════════════════════════════════════════════════════════════
//  Generic loop helpers — process wide chunks + scalar tail
// ═══════════════════════════════════════════════════════════════════

/// Apply a unary AVX2 kernel over `src` → `dst`, with a scalar fallback
/// for the tail (< 8 elements).
#[inline(always)]
unsafe fn map_f32(
    src: &[f32],
    dst: &mut [f32],
    mut simd_op: impl FnMut(__m256) -> __m256,
    mut scalar_op: impl FnMut(f32) -> f32,
) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let chunks = n / 8;
    let tail = n % 8;

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 8;
        let v = _mm256_loadu_ps(sp.add(off));
        let r = simd_op(v);
        _mm256_storeu_ps(dp.add(off), r);
    }

    let off = chunks * 8;
    for j in 0..tail {
        *dp.add(off + j) = scalar_op(*sp.add(off + j));
    }
}

/// Apply a binary AVX2 kernel over `(a, b)` → `dst`.
#[inline(always)]
unsafe fn zip_f32(
    a: &[f32],
    b: &[f32],
    dst: &mut [f32],
    mut simd_op: impl FnMut(__m256, __m256) -> __m256,
    mut scalar_op: impl FnMut(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let n = a.len();
    let chunks = n / 8;
    let tail = n % 8;

    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let dp = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 8;
        let va = _mm256_loadu_ps(ap.add(off));
        let vb = _mm256_loadu_ps(bp.add(off));
        let r = simd_op(va, vb);
        _mm256_storeu_ps(dp.add(off), r);
    }

    let off = chunks * 8;
    for j in 0..tail {
        *dp.add(off + j) = scalar_op(*ap.add(off + j), *bp.add(off + j));
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F32 — 8 elements per YMM register
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn relu_f32(src: &[f32], dst: &mut [f32]) {
    unsafe {
        let zero = _mm256_setzero_ps();
        map_f32(
            src,
            dst,
            |v| _mm256_max_ps(v, zero),
            |x| if x > 0.0 { x } else { 0.0 },
        );
    }
}

pub(crate) fn neg_f32(src: &[f32], dst: &mut [f32]) {
    unsafe {
        let sign_mask = _mm256_set1_ps(-0.0);
        map_f32(src, dst, |v| _mm256_xor_ps(v, sign_mask), |x| -x);
    }
}

pub(crate) fn sqrt_f32(src: &[f32], dst: &mut [f32]) {
    unsafe {
        map_f32(src, dst, |v| _mm256_sqrt_ps(v), axmath::sqrt_f32);
    }
}

pub(crate) fn exp_f32(src: &[f32], dst: &mut [f32]) {
    unsafe {
        map_f32(src, dst, |v| exp_f32x8(v), axmath::exp_f32);
    }
}

pub(crate) fn tanh_f32(src: &[f32], dst: &mut [f32]) {
    unsafe {
        map_f32(src, dst, |v| tanh_f32x8(v), axmath::tanh_f32);
    }
}

pub(crate) fn sigmoid_f32(src: &[f32], dst: &mut [f32]) {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let sign = _mm256_set1_ps(-0.0);
        map_f32(
            src,
            dst,
            |v| {
                let neg_v = _mm256_xor_ps(v, sign);
                let e = exp_f32x8(neg_v);
                _mm256_div_ps(one, _mm256_add_ps(one, e))
            },
            axmath::sigmoid_f32,
        );
    }
}

pub(crate) fn silu_f32(src: &[f32], dst: &mut [f32]) {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let sign = _mm256_set1_ps(-0.0);
        map_f32(
            src,
            dst,
            |v| {
                let neg_v = _mm256_xor_ps(v, sign);
                let e = exp_f32x8(neg_v);
                _mm256_div_ps(v, _mm256_add_ps(one, e))
            },
            axmath::silu_f32,
        );
    }
}

pub(crate) fn gelu_f32(src: &[f32], dst: &mut [f32]) {
    unsafe {
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let c = _mm256_set1_ps(0.797_884_56); // sqrt(2/π)
        let k = _mm256_set1_ps(0.044715);
        map_f32(
            src,
            dst,
            |v| {
                let x3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
                let inner = _mm256_fmadd_ps(k, x3, v);
                let t = tanh_f32x8(_mm256_mul_ps(c, inner));
                _mm256_mul_ps(half, _mm256_mul_ps(v, _mm256_add_ps(one, t)))
            },
            axmath::gelu_f32,
        );
    }
}

pub(crate) fn add_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    unsafe {
        zip_f32(a, b, dst, |va, vb| _mm256_add_ps(va, vb), |x, y| x + y);
    }
}

pub(crate) fn sub_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    unsafe {
        zip_f32(a, b, dst, |va, vb| _mm256_sub_ps(va, vb), |x, y| x - y);
    }
}

pub(crate) fn mul_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    unsafe {
        zip_f32(a, b, dst, |va, vb| _mm256_mul_ps(va, vb), |x, y| x * y);
    }
}

pub(crate) fn div_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    unsafe {
        zip_f32(a, b, dst, |va, vb| _mm256_div_ps(va, vb), |x, y| x / y);
    }
}

pub(crate) fn scale_f32(src: &[f32], dst: &mut [f32], s: f32) {
    unsafe {
        let vs = _mm256_set1_ps(s);
        map_f32(src, dst, |v| _mm256_mul_ps(v, vs), |x| x * s);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F64 — 4 elements per YMM register
// ═══════════════════════════════════════════════════════════════════

#[inline(always)]
unsafe fn map_f64(
    src: &[f64],
    dst: &mut [f64],
    mut simd_op: impl FnMut(__m256d) -> __m256d,
    mut scalar_op: impl FnMut(f64) -> f64,
) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let chunks = n / 4;
    let tail = n % 4;

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let v = _mm256_loadu_pd(sp.add(off));
        let r = simd_op(v);
        _mm256_storeu_pd(dp.add(off), r);
    }

    let off = chunks * 4;
    for j in 0..tail {
        *dp.add(off + j) = scalar_op(*sp.add(off + j));
    }
}

#[inline(always)]
unsafe fn zip_f64(
    a: &[f64],
    b: &[f64],
    dst: &mut [f64],
    mut simd_op: impl FnMut(__m256d, __m256d) -> __m256d,
    mut scalar_op: impl FnMut(f64, f64) -> f64,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let n = a.len();
    let chunks = n / 4;
    let tail = n % 4;

    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let dp = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let va = _mm256_loadu_pd(ap.add(off));
        let vb = _mm256_loadu_pd(bp.add(off));
        let r = simd_op(va, vb);
        _mm256_storeu_pd(dp.add(off), r);
    }

    let off = chunks * 4;
    for j in 0..tail {
        *dp.add(off + j) = scalar_op(*ap.add(off + j), *bp.add(off + j));
    }
}

/// Fast vectorised exp(x) for f64 — Cephes-style, 4-wide.
/// Uses 8th-order polynomial for f64 precision.
#[inline(always)]
unsafe fn exp_f64x4(x: __m256d) -> __m256d {
    let ln2 = _mm256_set1_pd(core::f64::consts::LN_2);
    let ln2_inv = _mm256_set1_pd(1.0 / core::f64::consts::LN_2);
    let one = _mm256_set1_pd(1.0);

    let c2 = _mm256_set1_pd(1.0 / 2.0);
    let c3 = _mm256_set1_pd(1.0 / 6.0);
    let c4 = _mm256_set1_pd(1.0 / 24.0);
    let c5 = _mm256_set1_pd(1.0 / 120.0);
    let c6 = _mm256_set1_pd(1.0 / 720.0);
    let c7 = _mm256_set1_pd(1.0 / 5040.0);
    let c8 = _mm256_set1_pd(1.0 / 40320.0);

    let lo = _mm256_set1_pd(-708.3);
    let hi = _mm256_set1_pd(709.7);
    let x = _mm256_min_pd(_mm256_max_pd(x, lo), hi);

    let n = _mm256_round_pd(
        _mm256_mul_pd(x, ln2_inv),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_pd(n, ln2, x);

    let mut p = _mm256_fmadd_pd(r, c8, c7);
    p = _mm256_fmadd_pd(r, p, c6);
    p = _mm256_fmadd_pd(r, p, c5);
    p = _mm256_fmadd_pd(r, p, c4);
    p = _mm256_fmadd_pd(r, p, c3);
    p = _mm256_fmadd_pd(r, p, c2);
    p = _mm256_fmadd_pd(r, p, one);
    p = _mm256_fmadd_pd(r, p, one);

    // 2^n via integer exponent — f64 bias 1023, shift 52
    let ni = _mm256_cvtpd_epi32(n);
    let ni64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(_mm256_castsi128_si256(ni)));
    let bias = _mm256_set1_epi64x(1023);
    let pow2n = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(ni64, bias), 52));

    _mm256_mul_pd(p, pow2n)
}

#[inline(always)]
unsafe fn tanh_f64x4(x: __m256d) -> __m256d {
    let two = _mm256_set1_pd(2.0);
    let one = _mm256_set1_pd(1.0);
    let e2x = exp_f64x4(_mm256_mul_pd(x, two));
    _mm256_div_pd(_mm256_sub_pd(e2x, one), _mm256_add_pd(e2x, one))
}

pub(crate) fn relu_f64(src: &[f64], dst: &mut [f64]) {
    unsafe {
        let zero = _mm256_setzero_pd();
        map_f64(
            src,
            dst,
            |v| _mm256_max_pd(v, zero),
            |x| if x > 0.0 { x } else { 0.0 },
        );
    }
}

pub(crate) fn neg_f64(src: &[f64], dst: &mut [f64]) {
    unsafe {
        let sign = _mm256_set1_pd(-0.0);
        map_f64(src, dst, |v| _mm256_xor_pd(v, sign), |x| -x);
    }
}

pub(crate) fn sqrt_f64(src: &[f64], dst: &mut [f64]) {
    unsafe {
        map_f64(src, dst, |v| _mm256_sqrt_pd(v), axmath::sqrt_f64);
    }
}

pub(crate) fn exp_f64(src: &[f64], dst: &mut [f64]) {
    unsafe {
        map_f64(src, dst, |v| exp_f64x4(v), axmath::exp_f64);
    }
}

pub(crate) fn tanh_f64(src: &[f64], dst: &mut [f64]) {
    unsafe {
        map_f64(src, dst, |v| tanh_f64x4(v), axmath::tanh_f64);
    }
}

pub(crate) fn sigmoid_f64(src: &[f64], dst: &mut [f64]) {
    unsafe {
        let one = _mm256_set1_pd(1.0);
        let sign = _mm256_set1_pd(-0.0);
        map_f64(
            src,
            dst,
            |v| {
                let neg_v = _mm256_xor_pd(v, sign);
                let e = exp_f64x4(neg_v);
                _mm256_div_pd(one, _mm256_add_pd(one, e))
            },
            axmath::sigmoid_f64,
        );
    }
}

pub(crate) fn silu_f64(src: &[f64], dst: &mut [f64]) {
    unsafe {
        let one = _mm256_set1_pd(1.0);
        let sign = _mm256_set1_pd(-0.0);
        map_f64(
            src,
            dst,
            |v| {
                let neg_v = _mm256_xor_pd(v, sign);
                let e = exp_f64x4(neg_v);
                _mm256_div_pd(v, _mm256_add_pd(one, e))
            },
            axmath::silu_f64,
        );
    }
}

pub(crate) fn gelu_f64(src: &[f64], dst: &mut [f64]) {
    unsafe {
        let half = _mm256_set1_pd(0.5);
        let one = _mm256_set1_pd(1.0);
        let c = _mm256_set1_pd(0.797_884_560_802_865_4);
        let k = _mm256_set1_pd(0.044715);
        map_f64(
            src,
            dst,
            |v| {
                let x3 = _mm256_mul_pd(_mm256_mul_pd(v, v), v);
                let inner = _mm256_fmadd_pd(k, x3, v);
                let t = tanh_f64x4(_mm256_mul_pd(c, inner));
                _mm256_mul_pd(half, _mm256_mul_pd(v, _mm256_add_pd(one, t)))
            },
            axmath::gelu_f64,
        );
    }
}

pub(crate) fn add_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    unsafe {
        zip_f64(a, b, dst, |va, vb| _mm256_add_pd(va, vb), |x, y| x + y);
    }
}

pub(crate) fn sub_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    unsafe {
        zip_f64(a, b, dst, |va, vb| _mm256_sub_pd(va, vb), |x, y| x - y);
    }
}

pub(crate) fn mul_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    unsafe {
        zip_f64(a, b, dst, |va, vb| _mm256_mul_pd(va, vb), |x, y| x * y);
    }
}

pub(crate) fn div_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    unsafe {
        zip_f64(a, b, dst, |va, vb| _mm256_div_pd(va, vb), |x, y| x / y);
    }
}

pub(crate) fn scale_f64(src: &[f64], dst: &mut [f64], s: f64) {
    unsafe {
        let vs = _mm256_set1_pd(s);
        map_f64(src, dst, |v| _mm256_mul_pd(v, vs), |x| x * s);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F16 (packed &[u8]) — widen via F16C, compute in f32, narrow back
// ═══════════════════════════════════════════════════════════════════
//
// F16C provides:
//   vcvtph2ps xmm → ymm  (8 × f16 → 8 × f32)
//   vcvtps2ph ymm → xmm  (8 × f32 → 8 × f16)

#[inline(always)]
unsafe fn load_f16x8(ptr: *const u8) -> __m256 {
    let bits = _mm_loadu_si128(ptr as *const __m128i);
    _mm256_cvtph_ps(bits)
}

#[inline(always)]
unsafe fn store_f16x8(ptr: *mut u8, v: __m256) {
    let bits = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(ptr as *mut __m128i, bits);
}

#[inline(always)]
unsafe fn map_f16(
    src: &[u8],
    dst: &mut [u8],
    mut simd_op: impl FnMut(__m256) -> __m256,
    mut scalar_op: impl FnMut(f32) -> f32,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 2, 0);
    let n = src.len() / 2;
    let chunks = n / 8;
    let tail = n % 8;

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 16;
        let v = load_f16x8(sp.add(off));
        let r = simd_op(v);
        store_f16x8(dp.add(off), r);
    }

    let off = chunks * 16;
    for j in 0..tail {
        let bo = off + j * 2;
        let x = crate::cast::read_f16(&src[bo..bo + 2]);
        crate::cast::write_f16(&mut dst[bo..bo + 2], scalar_op(x));
    }
}

#[inline(always)]
unsafe fn zip_f16(
    a: &[u8],
    b: &[u8],
    dst: &mut [u8],
    mut simd_op: impl FnMut(__m256, __m256) -> __m256,
    mut scalar_op: impl FnMut(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    debug_assert_eq!(a.len() % 2, 0);
    let n = a.len() / 2;
    let chunks = n / 8;
    let tail = n % 8;

    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let dp = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 16;
        let va = load_f16x8(ap.add(off));
        let vb = load_f16x8(bp.add(off));
        let r = simd_op(va, vb);
        store_f16x8(dp.add(off), r);
    }

    let off = chunks * 16;
    for j in 0..tail {
        let bo = off + j * 2;
        let x = crate::cast::read_f16(&a[bo..bo + 2]);
        let y = crate::cast::read_f16(&b[bo..bo + 2]);
        crate::cast::write_f16(&mut dst[bo..bo + 2], scalar_op(x, y));
    }
}

pub(crate) fn silu_f16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let sign = _mm256_set1_ps(-0.0);
        map_f16(
            src,
            dst,
            |v| {
                let e = exp_f32x8(_mm256_xor_ps(v, sign));
                _mm256_div_ps(v, _mm256_add_ps(one, e))
            },
            axmath::silu_f32,
        );
    }
}

pub(crate) fn gelu_f16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let c = _mm256_set1_ps(0.797_884_56);
        let k = _mm256_set1_ps(0.044715);
        map_f16(
            src,
            dst,
            |v| {
                let x3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
                let inner = _mm256_fmadd_ps(k, x3, v);
                let t = tanh_f32x8(_mm256_mul_ps(c, inner));
                _mm256_mul_ps(half, _mm256_mul_ps(v, _mm256_add_ps(one, t)))
            },
            axmath::gelu_f32,
        );
    }
}

pub(crate) fn relu_f16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let zero = _mm256_setzero_ps();
        map_f16(
            src,
            dst,
            |v| _mm256_max_ps(v, zero),
            |x| if x > 0.0 { x } else { 0.0 },
        );
    }
}

pub(crate) fn tanh_f16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        map_f16(src, dst, |v| tanh_f32x8(v), axmath::tanh_f32);
    }
}

pub(crate) fn sigmoid_f16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let sign = _mm256_set1_ps(-0.0);
        map_f16(
            src,
            dst,
            |v| {
                let e = exp_f32x8(_mm256_xor_ps(v, sign));
                _mm256_div_ps(one, _mm256_add_ps(one, e))
            },
            axmath::sigmoid_f32,
        );
    }
}

pub(crate) fn exp_f16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        map_f16(src, dst, |v| exp_f32x8(v), axmath::exp_f32);
    }
}

pub(crate) fn neg_f16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let sign = _mm256_set1_ps(-0.0);
        map_f16(src, dst, |v| _mm256_xor_ps(v, sign), |x| -x);
    }
}

pub(crate) fn sqrt_f16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        map_f16(src, dst, |v| _mm256_sqrt_ps(v), axmath::sqrt_f32);
    }
}

pub(crate) fn add_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    unsafe {
        zip_f16(a, b, dst, |va, vb| _mm256_add_ps(va, vb), |x, y| x + y);
    }
}

pub(crate) fn sub_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    unsafe {
        zip_f16(a, b, dst, |va, vb| _mm256_sub_ps(va, vb), |x, y| x - y);
    }
}

pub(crate) fn mul_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    unsafe {
        zip_f16(a, b, dst, |va, vb| _mm256_mul_ps(va, vb), |x, y| x * y);
    }
}

pub(crate) fn div_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    unsafe {
        zip_f16(a, b, dst, |va, vb| _mm256_div_ps(va, vb), |x, y| x / y);
    }
}

pub(crate) fn scale_f16(src: &[u8], dst: &mut [u8], s: f32) {
    unsafe {
        let vs = _mm256_set1_ps(s);
        map_f16(src, dst, |v| _mm256_mul_ps(v, vs), |x| x * s);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  BF16 (packed &[u8]) — shift-based conversion, compute in f32
// ═══════════════════════════════════════════════════════════════════
//
// BF16 ↔ F32 is just a 16-bit shift (no F16C needed):
//   f32_bits = bf16_bits << 16
//   bf16_bits = f32_bits >> 16  (truncate)

#[inline(always)]
unsafe fn load_bf16x8(ptr: *const u8) -> __m256 {
    let bits16 = _mm_loadu_si128(ptr as *const __m128i);
    let wide = _mm256_cvtepu16_epi32(bits16);
    _mm256_castsi256_ps(_mm256_slli_epi32(wide, 16))
}

#[inline(always)]
unsafe fn store_bf16x8(ptr: *mut u8, v: __m256) {
    let i32s = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
    let lo128 = _mm256_castsi256_si128(i32s);
    let hi128 = _mm256_extracti128_si256(i32s, 1);
    let packed = _mm_packus_epi32(lo128, hi128);
    _mm_storeu_si128(ptr as *mut __m128i, packed);
}

#[inline(always)]
unsafe fn map_bf16(
    src: &[u8],
    dst: &mut [u8],
    mut simd_op: impl FnMut(__m256) -> __m256,
    mut scalar_op: impl FnMut(f32) -> f32,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 2, 0);
    let n = src.len() / 2;
    let chunks = n / 8;
    let tail = n % 8;

    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 16;
        let v = load_bf16x8(sp.add(off));
        let r = simd_op(v);
        store_bf16x8(dp.add(off), r);
    }

    let off = chunks * 16;
    for j in 0..tail {
        let bo = off + j * 2;
        let x = crate::cast::read_bf16(&src[bo..bo + 2]);
        crate::cast::write_bf16(&mut dst[bo..bo + 2], scalar_op(x));
    }
}

#[inline(always)]
unsafe fn zip_bf16(
    a: &[u8],
    b: &[u8],
    dst: &mut [u8],
    mut simd_op: impl FnMut(__m256, __m256) -> __m256,
    mut scalar_op: impl FnMut(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    debug_assert_eq!(a.len() % 2, 0);
    let n = a.len() / 2;
    let chunks = n / 8;
    let tail = n % 8;

    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let dp = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 16;
        let va = load_bf16x8(ap.add(off));
        let vb = load_bf16x8(bp.add(off));
        let r = simd_op(va, vb);
        store_bf16x8(dp.add(off), r);
    }

    let off = chunks * 16;
    for j in 0..tail {
        let bo = off + j * 2;
        let x = crate::cast::read_bf16(&a[bo..bo + 2]);
        let y = crate::cast::read_bf16(&b[bo..bo + 2]);
        crate::cast::write_bf16(&mut dst[bo..bo + 2], scalar_op(x, y));
    }
}

pub(crate) fn silu_bf16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let sign = _mm256_set1_ps(-0.0);
        map_bf16(
            src,
            dst,
            |v| {
                let e = exp_f32x8(_mm256_xor_ps(v, sign));
                _mm256_div_ps(v, _mm256_add_ps(one, e))
            },
            axmath::silu_f32,
        );
    }
}

pub(crate) fn gelu_bf16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let c = _mm256_set1_ps(0.797_884_56);
        let k = _mm256_set1_ps(0.044715);
        map_bf16(
            src,
            dst,
            |v| {
                let x3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
                let inner = _mm256_fmadd_ps(k, x3, v);
                let t = tanh_f32x8(_mm256_mul_ps(c, inner));
                _mm256_mul_ps(half, _mm256_mul_ps(v, _mm256_add_ps(one, t)))
            },
            axmath::gelu_f32,
        );
    }
}

pub(crate) fn relu_bf16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let zero = _mm256_setzero_ps();
        map_bf16(
            src,
            dst,
            |v| _mm256_max_ps(v, zero),
            |x| if x > 0.0 { x } else { 0.0 },
        );
    }
}

pub(crate) fn tanh_bf16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        map_bf16(src, dst, |v| tanh_f32x8(v), axmath::tanh_f32);
    }
}

pub(crate) fn sigmoid_bf16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let sign = _mm256_set1_ps(-0.0);
        map_bf16(
            src,
            dst,
            |v| {
                let e = exp_f32x8(_mm256_xor_ps(v, sign));
                _mm256_div_ps(one, _mm256_add_ps(one, e))
            },
            axmath::sigmoid_f32,
        );
    }
}

pub(crate) fn exp_bf16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        map_bf16(src, dst, |v| exp_f32x8(v), axmath::exp_f32);
    }
}

pub(crate) fn neg_bf16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        let sign = _mm256_set1_ps(-0.0);
        map_bf16(src, dst, |v| _mm256_xor_ps(v, sign), |x| -x);
    }
}

pub(crate) fn sqrt_bf16(src: &[u8], dst: &mut [u8]) {
    unsafe {
        map_bf16(src, dst, |v| _mm256_sqrt_ps(v), axmath::sqrt_f32);
    }
}

pub(crate) fn add_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    unsafe {
        zip_bf16(a, b, dst, |va, vb| _mm256_add_ps(va, vb), |x, y| x + y);
    }
}

pub(crate) fn sub_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    unsafe {
        zip_bf16(a, b, dst, |va, vb| _mm256_sub_ps(va, vb), |x, y| x - y);
    }
}

pub(crate) fn mul_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    unsafe {
        zip_bf16(a, b, dst, |va, vb| _mm256_mul_ps(va, vb), |x, y| x * y);
    }
}

pub(crate) fn div_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    unsafe {
        zip_bf16(a, b, dst, |va, vb| _mm256_div_ps(va, vb), |x, y| x / y);
    }
}

pub(crate) fn scale_bf16(src: &[u8], dst: &mut [u8], s: f32) {
    unsafe {
        let vs = _mm256_set1_ps(s);
        map_bf16(src, dst, |v| _mm256_mul_ps(v, vs), |x| x * s);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Raster primitives — Pineda edge-function coverage pipeline
// ═══════════════════════════════════════════════════════════════════
//
// Inner loop of a software rasterizer processing 8 pixels per cycle:
//
//   cmpge_f32(e0, zero, m0)          // which pixels pass edge 0?
//   cmpge_f32(e1, zero, m1)          // ... edge 1 ...
//   cmpge_f32(e2, zero, m2)          // ... edge 2 ...
//   and_masks(m0, m1, m01)
//   and_masks(m01, m2, coverage)     // inside all three edges?
//   maskstore_u32(coverage, color, fb_tile)   // write covered pixels

/// For each lane: dst[i] = 0xFFFF_FFFF if a[i] >= b[i], else 0.
///
/// Uses `_CMP_GE_OQ` (ordered, quiet) — NaN produces 0 (not covered).
pub(crate) fn cmpge_f32(a: &[f32], b: &[f32], dst: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let n = a.len();
    let chunks = n / 8;
    let tail = n % 8;
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let dp = dst.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 8;
            let va = _mm256_loadu_ps(ap.add(off));
            let vb = _mm256_loadu_ps(bp.add(off));
            // _CMP_GE_OQ = 29: ordered, quiet >=
            let mask = _mm256_cmp_ps(va, vb, 29_i32);
            _mm256_storeu_si256(dp.add(off) as *mut __m256i, _mm256_castps_si256(mask));
        }
        let off = chunks * 8;
        for j in 0..tail {
            *dp.add(off + j) = if *ap.add(off + j) >= *bp.add(off + j) {
                0xFFFF_FFFF
            } else {
                0
            };
        }
    }
}

/// Bitwise AND of two u32 mask slices — intersect two edge coverage masks.
pub(crate) fn and_masks(a: &[u32], b: &[u32], dst: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let n = a.len();
    let chunks = n / 8;
    let tail = n % 8;
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let dp = dst.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 8;
            let va = _mm256_loadu_si256(ap.add(off) as *const __m256i);
            let vb = _mm256_loadu_si256(bp.add(off) as *const __m256i);
            _mm256_storeu_si256(dp.add(off) as *mut __m256i, _mm256_and_si256(va, vb));
        }
        let off = chunks * 8;
        for j in 0..tail {
            *dp.add(off + j) = *ap.add(off + j) & *bp.add(off + j);
        }
    }
}

/// Fused 3-edge coverage test + conditional framebuffer write — AVX2.
///
/// Equivalent to composing cmpge_f32 × 3 + and_masks × 2 + maskstore_u32,
/// but without materialising the three intermediate mask buffers.
/// All coverage masks stay in YMM registers between cmp and maskstore.
pub(crate) fn fused_edge_coverage(
    e0: &[f32],
    e1: &[f32],
    e2: &[f32],
    thresh: f32,
    color: &[u32],
    fb: &mut [u32],
) {
    let n = fb.len();
    debug_assert_eq!(e0.len(), n);
    debug_assert_eq!(e1.len(), n);
    debug_assert_eq!(e2.len(), n);
    debug_assert_eq!(color.len(), n);
    let chunks = n / 8;
    let tail = n % 8;
    unsafe {
        let vt = _mm256_set1_ps(thresh);
        let p0 = e0.as_ptr();
        let p1 = e1.as_ptr();
        let p2 = e2.as_ptr();
        let cp = color.as_ptr();
        let fp = fb.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 8;
            // Compare all three edge vectors against threshold
            let va0 = _mm256_loadu_ps(p0.add(off));
            let va1 = _mm256_loadu_ps(p1.add(off));
            let va2 = _mm256_loadu_ps(p2.add(off));
            let m0 = _mm256_castps_si256(_mm256_cmp_ps(va0, vt, 29_i32)); // GE_OQ
            let m1 = _mm256_castps_si256(_mm256_cmp_ps(va1, vt, 29_i32));
            let m2 = _mm256_castps_si256(_mm256_cmp_ps(va2, vt, 29_i32));
            // AND the masks in-register — no intermediate memory writes
            let coverage = _mm256_and_si256(_mm256_and_si256(m0, m1), m2);
            let vcolor = _mm256_loadu_si256(cp.add(off) as *const __m256i);
            _mm256_maskstore_epi32(fp.add(off) as *mut i32, coverage, vcolor);
        }
        let off = chunks * 8;
        for j in 0..tail {
            let k = off + j;
            if *p0.add(k) >= thresh && *p1.add(k) >= thresh && *p2.add(k) >= thresh {
                *fp.add(k) = *cp.add(k);
            }
        }
    }
}

/// Conditionally write src[i] → dst[i] where mask[i] != 0.
///
/// Uses `_mm256_maskstore_epi32` which checks the high bit of each 32-bit
/// mask lane — matches the all-ones/all-zeros pattern from `cmpge_f32`.
pub(crate) fn maskstore_u32(mask: &[u32], src: &[u32], dst: &mut [u32]) {
    debug_assert_eq!(mask.len(), src.len());
    debug_assert_eq!(mask.len(), dst.len());
    let n = mask.len();
    let chunks = n / 8;
    let tail = n % 8;
    unsafe {
        let mp = mask.as_ptr();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 8;
            let vmask = _mm256_loadu_si256(mp.add(off) as *const __m256i);
            let vsrc = _mm256_loadu_si256(sp.add(off) as *const __m256i);
            _mm256_maskstore_epi32(dp.add(off) as *mut i32, vmask, vsrc);
        }
        let off = chunks * 8;
        for j in 0..tail {
            if *mp.add(off + j) != 0 {
                *dp.add(off + j) = *sp.add(off + j);
            }
        }
    }
}
