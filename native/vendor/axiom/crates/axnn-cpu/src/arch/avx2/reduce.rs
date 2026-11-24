//! Reduction operations — AVX2 + FMA.
//!
//! Horizontal reductions accumulate in 256-bit registers, then reduce
//! to scalar via `hsum_f32x8` / `hsum_f64x4`.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

// ── Horizontal reduction helpers ─────────────────────────────────

#[inline(always)]
unsafe fn hsum_f32x8(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_movehdup_ps(sum64);
    _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
}

#[inline(always)]
unsafe fn hsum_f64x4(v: __m256d) -> f64 {
    let hi128 = _mm256_extractf128_pd(v, 1);
    let lo128 = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(lo128, hi128);
    let hi64 = _mm_unpackhi_pd(sum128, sum128);
    _mm_cvtsd_f64(_mm_add_sd(sum128, hi64))
}

#[inline(always)]
unsafe fn hmax_f32x8(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(max128, max128);
    let max64 = _mm_max_ps(max128, hi64);
    let hi32 = _mm_movehdup_ps(max64);
    _mm_cvtss_f32(_mm_max_ss(max64, hi32))
}

#[inline(always)]
unsafe fn hmax_f64x4(v: __m256d) -> f64 {
    let hi128 = _mm256_extractf128_pd(v, 1);
    let lo128 = _mm256_castpd256_pd128(v);
    let max128 = _mm_max_pd(lo128, hi128);
    let hi64 = _mm_unpackhi_pd(max128, max128);
    _mm_cvtsd_f64(_mm_max_sd(max128, hi64))
}

#[inline(always)]
unsafe fn arg_reduce_f32x8(values: __m256, indices: __m256i, want_max: bool) -> (f32, usize) {
    let mut vals = [0.0f32; 8];
    let mut idxs = [0i32; 8];
    _mm256_storeu_ps(vals.as_mut_ptr(), values);
    _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, indices);

    let mut best = vals[0];
    let mut best_idx = idxs[0] as usize;
    for i in 1..8 {
        let idx = idxs[i] as usize;
        let better = if want_max {
            vals[i] > best
        } else {
            vals[i] < best
        };
        if better || (vals[i] == best && idx < best_idx) {
            best = vals[i];
            best_idx = idx;
        }
    }
    (best, best_idx)
}

#[inline(always)]
unsafe fn arg_reduce_f64x4(values: __m256d, indices: __m256i, want_max: bool) -> (f64, usize) {
    let mut vals = [0.0f64; 4];
    let mut idxs = [0i64; 4];
    _mm256_storeu_pd(vals.as_mut_ptr(), values);
    _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, indices);

    let mut best = vals[0];
    let mut best_idx = idxs[0] as usize;
    for i in 1..4 {
        let idx = idxs[i] as usize;
        let better = if want_max {
            vals[i] > best
        } else {
            vals[i] < best
        };
        if better || (vals[i] == best && idx < best_idx) {
            best = vals[i];
            best_idx = idx;
        }
    }
    (best, best_idx)
}

// Inline exp_f32x8 (same as vec.rs — kept local to avoid cross-module dep)
#[inline(always)]
unsafe fn exp_f32x8(x: __m256) -> __m256 {
    let ln2 = _mm256_set1_ps(core::f32::consts::LN_2);
    let ln2_inv = _mm256_set1_ps(1.0 / core::f32::consts::LN_2);
    let one = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let c6 = _mm256_set1_ps(1.0 / 720.0);
    let lo = _mm256_set1_ps(-88.72284);
    let hi = _mm256_set1_ps(88.72284);
    let x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    let n = _mm256_round_ps(
        _mm256_mul_ps(x, ln2_inv),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_ps(n, ln2, x);
    let mut p = _mm256_fmadd_ps(r, c6, c5);
    p = _mm256_fmadd_ps(r, p, c4);
    p = _mm256_fmadd_ps(r, p, c3);
    p = _mm256_fmadd_ps(r, p, c2);
    p = _mm256_fmadd_ps(r, p, one);
    p = _mm256_fmadd_ps(r, p, one);
    let ni = _mm256_cvtps_epi32(n);
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ni, _mm256_set1_epi32(127)),
        23,
    ));
    _mm256_mul_ps(p, pow2n)
}

// ═══════════════════════════════════════════════════════════════════
//  F32
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn sum_f32(src: &[f32]) -> f32 {
    unsafe {
        let n = src.len();
        let chunks = n / 8;
        let tail = n % 8;
        let sp = src.as_ptr();

        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(sp.add(i * 8)));
        }

        let mut s = hsum_f32x8(acc);
        let off = chunks * 8;
        for j in 0..tail {
            s += *sp.add(off + j);
        }
        s
    }
}

pub(crate) fn max_f32(src: &[f32]) -> f32 {
    debug_assert!(!src.is_empty());
    unsafe {
        let n = src.len();
        let sp = src.as_ptr();

        if n < 8 {
            let mut m = *sp;
            for j in 1..n {
                let v = *sp.add(j);
                if v > m {
                    m = v;
                }
            }
            return m;
        }

        let chunks = n / 8;
        let tail = n % 8;
        let mut acc = _mm256_loadu_ps(sp);
        for i in 1..chunks {
            acc = _mm256_max_ps(acc, _mm256_loadu_ps(sp.add(i * 8)));
        }

        let mut m = hmax_f32x8(acc);
        let off = chunks * 8;
        for j in 0..tail {
            let v = *sp.add(off + j);
            if v > m {
                m = v;
            }
        }
        m
    }
}

pub(crate) fn argmin_f32(src: &[f32]) -> usize {
    debug_assert!(!src.is_empty());
    unsafe { arg_f32(src, false) }
}

pub(crate) fn argmax_f32(src: &[f32]) -> usize {
    debug_assert!(!src.is_empty());
    unsafe { arg_f32(src, true) }
}

unsafe fn arg_f32(src: &[f32], want_max: bool) -> usize {
    let n = src.len();
    let sp = src.as_ptr();
    if n < 8 {
        let mut idx = 0;
        let mut best = *sp;
        for j in 1..n {
            let v = *sp.add(j);
            if (want_max && v > best) || (!want_max && v < best) {
                best = v;
                idx = j;
            }
        }
        return idx;
    }

    let chunks = n / 8;
    let tail = n % 8;
    let base_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut best = _mm256_loadu_ps(sp);
    let mut best_idx = base_idx;

    for i in 1..chunks {
        let off = i * 8;
        let v = _mm256_loadu_ps(sp.add(off));
        let idx = _mm256_add_epi32(base_idx, _mm256_set1_epi32(off as i32));
        let mask = if want_max {
            _mm256_cmp_ps(v, best, _CMP_GT_OQ)
        } else {
            _mm256_cmp_ps(v, best, _CMP_LT_OQ)
        };
        best = _mm256_blendv_ps(best, v, mask);
        best_idx = _mm256_blendv_epi8(best_idx, idx, _mm256_castps_si256(mask));
    }

    let (mut best_v, mut out) = arg_reduce_f32x8(best, best_idx, want_max);
    let off = chunks * 8;
    for j in 0..tail {
        let idx = off + j;
        let v = *sp.add(idx);
        if (want_max && v > best_v) || (!want_max && v < best_v) {
            best_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn softmax_f32(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let m = max_f32(src);

    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let vm = _mm256_set1_ps(m);
        let chunks = n / 8;
        let tail = n % 8;

        let mut sum_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let off = i * 8;
            let e = exp_f32x8(_mm256_sub_ps(_mm256_loadu_ps(sp.add(off)), vm));
            _mm256_storeu_ps(dp.add(off), e);
            sum_acc = _mm256_add_ps(sum_acc, e);
        }

        let mut s = hsum_f32x8(sum_acc);
        let off = chunks * 8;
        for j in 0..tail {
            let e = axmath::exp_f32(*sp.add(off + j) - m);
            *dp.add(off + j) = e;
            s += e;
        }

        let inv = _mm256_set1_ps(1.0 / s);
        for i in 0..chunks {
            let off = i * 8;
            _mm256_storeu_ps(
                dp.add(off),
                _mm256_mul_ps(_mm256_loadu_ps(dp.add(off)), inv),
            );
        }
        let inv_s = 1.0 / s;
        for j in 0..tail {
            *dp.add(off + j) *= inv_s;
        }
    }
}

pub(crate) fn rms_norm_f32(src: &[f32], dst: &mut [f32], weight: &[f32], eps: f32) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());

    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let wp = weight.as_ptr();
        let chunks = n / 8;
        let tail = n % 8;

        let mut ss_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(sp.add(i * 8));
            ss_acc = _mm256_fmadd_ps(v, v, ss_acc);
        }
        let mut ss = hsum_f32x8(ss_acc);
        let off = chunks * 8;
        for j in 0..tail {
            let v = *sp.add(off + j);
            ss += v * v;
        }

        let rms_inv = 1.0 / axmath::sqrt_f32(ss / n as f32 + eps);
        let vrms = _mm256_set1_ps(rms_inv);

        for i in 0..chunks {
            let o = i * 8;
            let v = _mm256_loadu_ps(sp.add(o));
            let w = _mm256_loadu_ps(wp.add(o));
            _mm256_storeu_ps(dp.add(o), _mm256_mul_ps(_mm256_mul_ps(v, vrms), w));
        }
        for j in 0..tail {
            *dp.add(off + j) = *sp.add(off + j) * rms_inv * *wp.add(off + j);
        }
    }
}

pub(crate) fn layer_norm_f32(src: &[f32], dst: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());
    debug_assert_eq!(n, bias.len());

    let mean = sum_f32(src) / n as f32;

    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let wp = weight.as_ptr();
        let bp = bias.as_ptr();
        let chunks = n / 8;
        let tail = n % 8;
        let vmean = _mm256_set1_ps(mean);

        let mut var_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let d = _mm256_sub_ps(_mm256_loadu_ps(sp.add(i * 8)), vmean);
            var_acc = _mm256_fmadd_ps(d, d, var_acc);
        }
        let mut var = hsum_f32x8(var_acc);
        let off = chunks * 8;
        for j in 0..tail {
            let d = *sp.add(off + j) - mean;
            var += d * d;
        }

        let inv = 1.0 / axmath::sqrt_f32(var / n as f32 + eps);
        let vinv = _mm256_set1_ps(inv);

        for i in 0..chunks {
            let o = i * 8;
            let centered = _mm256_sub_ps(_mm256_loadu_ps(sp.add(o)), vmean);
            let normed = _mm256_mul_ps(centered, vinv);
            let r = _mm256_fmadd_ps(
                normed,
                _mm256_loadu_ps(wp.add(o)),
                _mm256_loadu_ps(bp.add(o)),
            );
            _mm256_storeu_ps(dp.add(o), r);
        }
        for j in 0..tail {
            let idx = off + j;
            *dp.add(idx) = (*sp.add(idx) - mean) * inv * *wp.add(idx) + *bp.add(idx);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F64
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn sum_f64(src: &[f64]) -> f64 {
    unsafe {
        let n = src.len();
        let chunks = n / 4;
        let tail = n % 4;
        let sp = src.as_ptr();

        let mut acc = _mm256_setzero_pd();
        for i in 0..chunks {
            acc = _mm256_add_pd(acc, _mm256_loadu_pd(sp.add(i * 4)));
        }

        let mut s = hsum_f64x4(acc);
        let off = chunks * 4;
        for j in 0..tail {
            s += *sp.add(off + j);
        }
        s
    }
}

pub(crate) fn max_f64(src: &[f64]) -> f64 {
    debug_assert!(!src.is_empty());
    unsafe {
        let n = src.len();
        let sp = src.as_ptr();

        if n < 4 {
            let mut m = *sp;
            for j in 1..n {
                let v = *sp.add(j);
                if v > m {
                    m = v;
                }
            }
            return m;
        }

        let chunks = n / 4;
        let tail = n % 4;
        let mut acc = _mm256_loadu_pd(sp);
        for i in 1..chunks {
            acc = _mm256_max_pd(acc, _mm256_loadu_pd(sp.add(i * 4)));
        }

        let mut m = hmax_f64x4(acc);
        let off = chunks * 4;
        for j in 0..tail {
            let v = *sp.add(off + j);
            if v > m {
                m = v;
            }
        }
        m
    }
}

pub(crate) fn argmin_f64(src: &[f64]) -> usize {
    debug_assert!(!src.is_empty());
    unsafe { arg_f64(src, false) }
}

pub(crate) fn argmax_f64(src: &[f64]) -> usize {
    debug_assert!(!src.is_empty());
    unsafe { arg_f64(src, true) }
}

unsafe fn arg_f64(src: &[f64], want_max: bool) -> usize {
    let n = src.len();
    let sp = src.as_ptr();
    if n < 4 {
        let mut idx = 0;
        let mut best = *sp;
        for j in 1..n {
            let v = *sp.add(j);
            if (want_max && v > best) || (!want_max && v < best) {
                best = v;
                idx = j;
            }
        }
        return idx;
    }

    let chunks = n / 4;
    let tail = n % 4;
    let base_idx = _mm256_setr_epi64x(0, 1, 2, 3);
    let mut best = _mm256_loadu_pd(sp);
    let mut best_idx = base_idx;

    for i in 1..chunks {
        let off = i * 4;
        let v = _mm256_loadu_pd(sp.add(off));
        let idx = _mm256_add_epi64(base_idx, _mm256_set1_epi64x(off as i64));
        let mask = if want_max {
            _mm256_cmp_pd(v, best, _CMP_GT_OQ)
        } else {
            _mm256_cmp_pd(v, best, _CMP_LT_OQ)
        };
        best = _mm256_blendv_pd(best, v, mask);
        best_idx = _mm256_blendv_epi8(best_idx, idx, _mm256_castpd_si256(mask));
    }

    let (mut best_v, mut out) = arg_reduce_f64x4(best, best_idx, want_max);
    let off = chunks * 4;
    for j in 0..tail {
        let idx = off + j;
        let v = *sp.add(idx);
        if (want_max && v > best_v) || (!want_max && v < best_v) {
            best_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn softmax_f64(src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    let m = max_f64(src);
    let mut s = 0.0f64;
    for (d, &v) in dst.iter_mut().zip(src) {
        let e = axmath::exp_f64(v - m);
        *d = e;
        s += e;
    }
    let inv = 1.0 / s;
    for d in dst.iter_mut() {
        *d *= inv;
    }
}

pub(crate) fn rms_norm_f64(src: &[f64], dst: &mut [f64], weight: &[f64], eps: f64) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());

    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let wp = weight.as_ptr();
        let chunks = n / 4;
        let tail = n % 4;

        let mut ss_acc = _mm256_setzero_pd();
        for i in 0..chunks {
            let v = _mm256_loadu_pd(sp.add(i * 4));
            ss_acc = _mm256_fmadd_pd(v, v, ss_acc);
        }
        let mut ss = hsum_f64x4(ss_acc);
        let off = chunks * 4;
        for j in 0..tail {
            let v = *sp.add(off + j);
            ss += v * v;
        }

        let rms_inv = 1.0 / axmath::sqrt_f64(ss / n as f64 + eps);
        let vrms = _mm256_set1_pd(rms_inv);

        for i in 0..chunks {
            let o = i * 4;
            let v = _mm256_loadu_pd(sp.add(o));
            let w = _mm256_loadu_pd(wp.add(o));
            _mm256_storeu_pd(dp.add(o), _mm256_mul_pd(_mm256_mul_pd(v, vrms), w));
        }
        for j in 0..tail {
            *dp.add(off + j) = *sp.add(off + j) * rms_inv * *wp.add(off + j);
        }
    }
}

pub(crate) fn layer_norm_f64(src: &[f64], dst: &mut [f64], weight: &[f64], bias: &[f64], eps: f64) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());
    debug_assert_eq!(n, bias.len());

    let mean = sum_f64(src) / n as f64;

    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let wp = weight.as_ptr();
        let bp = bias.as_ptr();
        let chunks = n / 4;
        let tail = n % 4;
        let vmean = _mm256_set1_pd(mean);

        let mut var_acc = _mm256_setzero_pd();
        for i in 0..chunks {
            let d = _mm256_sub_pd(_mm256_loadu_pd(sp.add(i * 4)), vmean);
            var_acc = _mm256_fmadd_pd(d, d, var_acc);
        }
        let mut var = hsum_f64x4(var_acc);
        let off = chunks * 4;
        for j in 0..tail {
            let d = *sp.add(off + j) - mean;
            var += d * d;
        }

        let inv = 1.0 / axmath::sqrt_f64(var / n as f64 + eps);
        let vinv = _mm256_set1_pd(inv);

        for i in 0..chunks {
            let o = i * 4;
            let centered = _mm256_sub_pd(_mm256_loadu_pd(sp.add(o)), vmean);
            let normed = _mm256_mul_pd(centered, vinv);
            let r = _mm256_fmadd_pd(
                normed,
                _mm256_loadu_pd(wp.add(o)),
                _mm256_loadu_pd(bp.add(o)),
            );
            _mm256_storeu_pd(dp.add(o), r);
        }
        for j in 0..tail {
            let idx = off + j;
            *dp.add(idx) = (*sp.add(idx) - mean) * inv * *wp.add(idx) + *bp.add(idx);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F16 (packed &[u8], scalar results in f32)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn sum_f16(src: &[u8]) -> f32 {
    debug_assert_eq!(src.len() % 2, 0);
    unsafe {
        let n = src.len() / 2;
        let chunks = n / 8;
        let tail = n % 8;
        let sp = src.as_ptr();

        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            acc = _mm256_add_ps(
                acc,
                _mm256_cvtph_ps(_mm_loadu_si128(sp.add(i * 16) as *const __m128i)),
            );
        }

        let mut s = hsum_f32x8(acc);
        let off = chunks * 16;
        for j in 0..tail {
            s += crate::cast::read_f16(&src[off + j * 2..]);
        }
        s
    }
}

pub(crate) fn max_f16(src: &[u8]) -> f32 {
    debug_assert!(src.len() >= 2);
    unsafe {
        let n = src.len() / 2;
        if n < 8 {
            let mut m = crate::cast::read_f16(&src[0..2]);
            for j in 1..n {
                let v = crate::cast::read_f16(&src[j * 2..]);
                if v > m {
                    m = v;
                }
            }
            return m;
        }

        let sp = src.as_ptr();
        let chunks = n / 8;
        let tail = n % 8;
        let mut acc = _mm256_cvtph_ps(_mm_loadu_si128(sp as *const __m128i));
        for i in 1..chunks {
            acc = _mm256_max_ps(
                acc,
                _mm256_cvtph_ps(_mm_loadu_si128(sp.add(i * 16) as *const __m128i)),
            );
        }

        let mut m = hmax_f32x8(acc);
        let off = chunks * 16;
        for j in 0..tail {
            let v = crate::cast::read_f16(&src[off + j * 2..]);
            if v > m {
                m = v;
            }
        }
        m
    }
}

pub(crate) fn argmin_f16(src: &[u8]) -> usize {
    debug_assert!(src.len() >= 2);
    unsafe { arg_f16(src, false) }
}

pub(crate) fn argmax_f16(src: &[u8]) -> usize {
    debug_assert!(src.len() >= 2);
    unsafe { arg_f16(src, true) }
}

unsafe fn arg_f16(src: &[u8], want_max: bool) -> usize {
    let n = src.len() / 2;
    let sp = src.as_ptr();
    if n < 8 {
        let mut idx = 0;
        let mut best = crate::cast::read_f16(src);
        for j in 1..n {
            let v = crate::cast::read_f16(&src[j * 2..]);
            if (want_max && v > best) || (!want_max && v < best) {
                best = v;
                idx = j;
            }
        }
        return idx;
    }

    let chunks = n / 8;
    let tail = n % 8;
    let base_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut best = _mm256_cvtph_ps(_mm_loadu_si128(sp as *const __m128i));
    let mut best_idx = base_idx;

    for i in 1..chunks {
        let off = i * 8;
        let v = _mm256_cvtph_ps(_mm_loadu_si128(sp.add(off * 2) as *const __m128i));
        let idx = _mm256_add_epi32(base_idx, _mm256_set1_epi32(off as i32));
        let mask = if want_max {
            _mm256_cmp_ps(v, best, _CMP_GT_OQ)
        } else {
            _mm256_cmp_ps(v, best, _CMP_LT_OQ)
        };
        best = _mm256_blendv_ps(best, v, mask);
        best_idx = _mm256_blendv_epi8(best_idx, idx, _mm256_castps_si256(mask));
    }

    let (mut best_v, mut out) = arg_reduce_f32x8(best, best_idx, want_max);
    let off = chunks * 8;
    for j in 0..tail {
        let idx = off + j;
        let v = crate::cast::read_f16(&src[idx * 2..]);
        if (want_max && v > best_v) || (!want_max && v < best_v) {
            best_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn softmax_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 2, 0);
    let m = max_f16(src);

    unsafe {
        let n = src.len() / 2;
        let chunks = n / 8;
        let tail = n % 8;
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let vm = _mm256_set1_ps(m);

        let mut sum_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let off = i * 16;
            let v = _mm256_cvtph_ps(_mm_loadu_si128(sp.add(off) as *const __m128i));
            let e = exp_f32x8(_mm256_sub_ps(v, vm));
            _mm_storeu_si128(
                dp.add(off) as *mut __m128i,
                _mm256_cvtps_ph(e, _MM_FROUND_TO_NEAREST_INT),
            );
            sum_acc = _mm256_add_ps(sum_acc, e);
        }

        let mut s = hsum_f32x8(sum_acc);
        let off = chunks * 16;
        for j in 0..tail {
            let bo = off + j * 2;
            let e = axmath::exp_f32(crate::cast::read_f16(&src[bo..]) - m);
            crate::cast::write_f16(&mut dst[bo..], e);
            s += e;
        }

        let inv = _mm256_set1_ps(1.0 / s);
        for i in 0..chunks {
            let off = i * 16;
            let v = _mm256_cvtph_ps(_mm_loadu_si128(dp.add(off) as *const __m128i));
            _mm_storeu_si128(
                dp.add(off) as *mut __m128i,
                _mm256_cvtps_ph(_mm256_mul_ps(v, inv), _MM_FROUND_TO_NEAREST_INT),
            );
        }
        let inv_s = 1.0 / s;
        for j in 0..tail {
            let bo = off + j * 2;
            let val = crate::cast::read_f16(&dst[bo..]) * inv_s;
            crate::cast::write_f16(&mut dst[bo..], val);
        }
    }
}

pub(crate) fn rms_norm_f16(src: &[u8], dst: &mut [u8], weight: &[u8], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    let n = src.len() / 2;

    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let wp = weight.as_ptr();
        let chunks = n / 8;
        let tail = n % 8;

        let mut ss_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_cvtph_ps(_mm_loadu_si128(sp.add(i * 16) as *const __m128i));
            ss_acc = _mm256_fmadd_ps(v, v, ss_acc);
        }
        let mut ss = hsum_f32x8(ss_acc);
        let off = chunks * 16;
        for j in 0..tail {
            let v = crate::cast::read_f16(&src[off + j * 2..]);
            ss += v * v;
        }

        let rms_inv = 1.0 / axmath::sqrt_f32(ss / n as f32 + eps);
        let vrms = _mm256_set1_ps(rms_inv);

        for i in 0..chunks {
            let o = i * 16;
            let v = _mm256_cvtph_ps(_mm_loadu_si128(sp.add(o) as *const __m128i));
            let w = _mm256_cvtph_ps(_mm_loadu_si128(wp.add(o) as *const __m128i));
            _mm_storeu_si128(
                dp.add(o) as *mut __m128i,
                _mm256_cvtps_ph(
                    _mm256_mul_ps(_mm256_mul_ps(v, vrms), w),
                    _MM_FROUND_TO_NEAREST_INT,
                ),
            );
        }
        for j in 0..tail {
            let bo = off + j * 2;
            let v = crate::cast::read_f16(&src[bo..]);
            let w = crate::cast::read_f16(&weight[bo..]);
            crate::cast::write_f16(&mut dst[bo..], v * rms_inv * w);
        }
    }
}

pub(crate) fn layer_norm_f16(src: &[u8], dst: &mut [u8], weight: &[u8], bias: &[u8], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    debug_assert_eq!(src.len(), bias.len());
    let n = src.len() / 2;
    let mean = sum_f16(src) / n as f32;

    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let wp = weight.as_ptr();
        let bp = bias.as_ptr();
        let chunks = n / 8;
        let tail = n % 8;
        let vmean = _mm256_set1_ps(mean);

        let mut var_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_cvtph_ps(_mm_loadu_si128(sp.add(i * 16) as *const __m128i));
            let d = _mm256_sub_ps(v, vmean);
            var_acc = _mm256_fmadd_ps(d, d, var_acc);
        }
        let mut var = hsum_f32x8(var_acc);
        let off = chunks * 16;
        for j in 0..tail {
            let d = crate::cast::read_f16(&src[off + j * 2..]) - mean;
            var += d * d;
        }

        let inv = 1.0 / axmath::sqrt_f32(var / n as f32 + eps);
        let vinv = _mm256_set1_ps(inv);

        for i in 0..chunks {
            let o = i * 16;
            let v = _mm256_cvtph_ps(_mm_loadu_si128(sp.add(o) as *const __m128i));
            let w = _mm256_cvtph_ps(_mm_loadu_si128(wp.add(o) as *const __m128i));
            let b = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(o) as *const __m128i));
            let r = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_sub_ps(v, vmean), vinv), w, b);
            _mm_storeu_si128(
                dp.add(o) as *mut __m128i,
                _mm256_cvtps_ph(r, _MM_FROUND_TO_NEAREST_INT),
            );
        }
        for j in 0..tail {
            let bo = off + j * 2;
            let v = crate::cast::read_f16(&src[bo..]);
            let w = crate::cast::read_f16(&weight[bo..]);
            let b = crate::cast::read_f16(&bias[bo..]);
            crate::cast::write_f16(&mut dst[bo..], (v - mean) * inv * w + b);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  BF16 (packed &[u8], scalar results in f32)
// ═══════════════════════════════════════════════════════════════════

#[inline(always)]
unsafe fn load_bf16x8(ptr: *const u8) -> __m256 {
    let bits16 = _mm_loadu_si128(ptr as *const __m128i);
    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits16), 16))
}

#[inline(always)]
unsafe fn store_bf16x8(ptr: *mut u8, v: __m256) {
    let i32s = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
    let lo128 = _mm256_castsi256_si128(i32s);
    let hi128 = _mm256_extracti128_si256(i32s, 1);
    _mm_storeu_si128(ptr as *mut __m128i, _mm_packus_epi32(lo128, hi128));
}

pub(crate) fn sum_bf16(src: &[u8]) -> f32 {
    debug_assert_eq!(src.len() % 2, 0);
    unsafe {
        let n = src.len() / 2;
        let chunks = n / 8;
        let tail = n % 8;
        let sp = src.as_ptr();

        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            acc = _mm256_add_ps(acc, load_bf16x8(sp.add(i * 16)));
        }

        let mut s = hsum_f32x8(acc);
        let off = chunks * 16;
        for j in 0..tail {
            s += crate::cast::read_bf16(&src[off + j * 2..]);
        }
        s
    }
}

pub(crate) fn max_bf16(src: &[u8]) -> f32 {
    debug_assert!(src.len() >= 2);
    unsafe {
        let n = src.len() / 2;
        if n < 8 {
            let mut m = crate::cast::read_bf16(&src[0..2]);
            for j in 1..n {
                let v = crate::cast::read_bf16(&src[j * 2..]);
                if v > m {
                    m = v;
                }
            }
            return m;
        }

        let sp = src.as_ptr();
        let chunks = n / 8;
        let tail = n % 8;
        let mut acc = load_bf16x8(sp);
        for i in 1..chunks {
            acc = _mm256_max_ps(acc, load_bf16x8(sp.add(i * 16)));
        }

        let mut m = hmax_f32x8(acc);
        let off = chunks * 16;
        for j in 0..tail {
            let v = crate::cast::read_bf16(&src[off + j * 2..]);
            if v > m {
                m = v;
            }
        }
        m
    }
}

pub(crate) fn argmin_bf16(src: &[u8]) -> usize {
    debug_assert!(src.len() >= 2);
    unsafe { arg_bf16(src, false) }
}

pub(crate) fn argmax_bf16(src: &[u8]) -> usize {
    debug_assert!(src.len() >= 2);
    unsafe { arg_bf16(src, true) }
}

unsafe fn arg_bf16(src: &[u8], want_max: bool) -> usize {
    let n = src.len() / 2;
    let sp = src.as_ptr();
    if n < 8 {
        let mut idx = 0;
        let mut best = crate::cast::read_bf16(src);
        for j in 1..n {
            let v = crate::cast::read_bf16(&src[j * 2..]);
            if (want_max && v > best) || (!want_max && v < best) {
                best = v;
                idx = j;
            }
        }
        return idx;
    }

    let chunks = n / 8;
    let tail = n % 8;
    let base_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut best = load_bf16x8(sp);
    let mut best_idx = base_idx;

    for i in 1..chunks {
        let off = i * 8;
        let v = load_bf16x8(sp.add(off * 2));
        let idx = _mm256_add_epi32(base_idx, _mm256_set1_epi32(off as i32));
        let mask = if want_max {
            _mm256_cmp_ps(v, best, _CMP_GT_OQ)
        } else {
            _mm256_cmp_ps(v, best, _CMP_LT_OQ)
        };
        best = _mm256_blendv_ps(best, v, mask);
        best_idx = _mm256_blendv_epi8(best_idx, idx, _mm256_castps_si256(mask));
    }

    let (mut best_v, mut out) = arg_reduce_f32x8(best, best_idx, want_max);
    let off = chunks * 8;
    for j in 0..tail {
        let idx = off + j;
        let v = crate::cast::read_bf16(&src[idx * 2..]);
        if (want_max && v > best_v) || (!want_max && v < best_v) {
            best_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn softmax_bf16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    let m = max_bf16(src);

    unsafe {
        let n = src.len() / 2;
        let chunks = n / 8;
        let tail = n % 8;
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let vm = _mm256_set1_ps(m);

        let mut sum_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let off = i * 16;
            let e = exp_f32x8(_mm256_sub_ps(load_bf16x8(sp.add(off)), vm));
            store_bf16x8(dp.add(off), e);
            sum_acc = _mm256_add_ps(sum_acc, e);
        }

        let mut s = hsum_f32x8(sum_acc);
        let off = chunks * 16;
        for j in 0..tail {
            let bo = off + j * 2;
            let e = axmath::exp_f32(crate::cast::read_bf16(&src[bo..]) - m);
            crate::cast::write_bf16(&mut dst[bo..], e);
            s += e;
        }

        let inv = _mm256_set1_ps(1.0 / s);
        for i in 0..chunks {
            let off = i * 16;
            store_bf16x8(dp.add(off), _mm256_mul_ps(load_bf16x8(dp.add(off)), inv));
        }
        let inv_s = 1.0 / s;
        for j in 0..tail {
            let bo = off + j * 2;
            let val = crate::cast::read_bf16(&dst[bo..]) * inv_s;
            crate::cast::write_bf16(&mut dst[bo..], val);
        }
    }
}

pub(crate) fn rms_norm_bf16(src: &[u8], dst: &mut [u8], weight: &[u8], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    let n = src.len() / 2;

    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let wp = weight.as_ptr();
        let chunks = n / 8;
        let tail = n % 8;

        let mut ss_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = load_bf16x8(sp.add(i * 16));
            ss_acc = _mm256_fmadd_ps(v, v, ss_acc);
        }
        let mut ss = hsum_f32x8(ss_acc);
        let off = chunks * 16;
        for j in 0..tail {
            let v = crate::cast::read_bf16(&src[off + j * 2..]);
            ss += v * v;
        }

        let rms_inv = 1.0 / axmath::sqrt_f32(ss / n as f32 + eps);
        let vrms = _mm256_set1_ps(rms_inv);

        for i in 0..chunks {
            let o = i * 16;
            store_bf16x8(
                dp.add(o),
                _mm256_mul_ps(
                    _mm256_mul_ps(load_bf16x8(sp.add(o)), vrms),
                    load_bf16x8(wp.add(o)),
                ),
            );
        }
        for j in 0..tail {
            let bo = off + j * 2;
            crate::cast::write_bf16(
                &mut dst[bo..],
                crate::cast::read_bf16(&src[bo..])
                    * rms_inv
                    * crate::cast::read_bf16(&weight[bo..]),
            );
        }
    }
}

pub(crate) fn layer_norm_bf16(src: &[u8], dst: &mut [u8], weight: &[u8], bias: &[u8], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    debug_assert_eq!(src.len(), bias.len());
    let n = src.len() / 2;
    let mean = sum_bf16(src) / n as f32;

    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let wp = weight.as_ptr();
        let bp = bias.as_ptr();
        let chunks = n / 8;
        let tail = n % 8;
        let vmean = _mm256_set1_ps(mean);

        let mut var_acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let d = _mm256_sub_ps(load_bf16x8(sp.add(i * 16)), vmean);
            var_acc = _mm256_fmadd_ps(d, d, var_acc);
        }
        let mut var = hsum_f32x8(var_acc);
        let off = chunks * 16;
        for j in 0..tail {
            let d = crate::cast::read_bf16(&src[off + j * 2..]) - mean;
            var += d * d;
        }

        let inv = 1.0 / axmath::sqrt_f32(var / n as f32 + eps);
        let vinv = _mm256_set1_ps(inv);

        for i in 0..chunks {
            let o = i * 16;
            let r = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_sub_ps(load_bf16x8(sp.add(o)), vmean), vinv),
                load_bf16x8(wp.add(o)),
                load_bf16x8(bp.add(o)),
            );
            store_bf16x8(dp.add(o), r);
        }
        for j in 0..tail {
            let bo = off + j * 2;
            let v = crate::cast::read_bf16(&src[bo..]);
            let w = crate::cast::read_bf16(&weight[bo..]);
            let b = crate::cast::read_bf16(&bias[bo..]);
            crate::cast::write_bf16(&mut dst[bo..], (v - mean) * inv * w + b);
        }
    }
}
