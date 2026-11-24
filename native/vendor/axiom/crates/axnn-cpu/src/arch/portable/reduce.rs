//! Reduction operations — portable `core::simd` tier.
//!
//! Horizontal reductions accumulate in `Simd<f32, W>` / `Simd<f64, WD>`
//! registers, then reduce to scalar via `reduce_sum()` / `reduce_max()`.

use core::simd::Select;
use core::simd::prelude::*;

use super::W;
const WD: usize = super::WD;

// ── SIMD exp helpers ─────────────────────────────────────────────

#[inline(always)]
fn exp_f32x(x: Simd<f32, W>) -> Simd<f32, W> {
    let ln2 = Simd::splat(core::f32::consts::LN_2);
    let ln2_inv = Simd::splat(1.0 / core::f32::consts::LN_2);
    let one = Simd::splat(1.0f32);
    let c2 = Simd::splat(0.5f32);
    let c3 = Simd::splat(1.0 / 6.0);
    let c4 = Simd::splat(1.0 / 24.0);
    let c5 = Simd::splat(1.0 / 120.0);
    let c6 = Simd::splat(1.0 / 720.0);

    let x = x
        .simd_min(Simd::splat(88.72284f32))
        .simd_max(Simd::splat(-88.72284f32));
    let magic_f32 = Simd::splat(12582912.0f32);
    let n = (x * ln2_inv + magic_f32) - magic_f32;
    let r = x - n * ln2;

    let mut p = r * c6 + c5;
    p = r * p + c4;
    p = r * p + c3;
    p = r * p + c2;
    p = r * p + one;
    p = r * p + one;

    let ni: Simd<i32, W> = n.cast();
    let pow2n =
        Simd::<f32, W>::from_bits((ni + Simd::splat(127i32)).cast::<u32>() << Simd::splat(23u32));
    p * pow2n
}

#[inline(always)]
fn exp_f64x(x: Simd<f64, WD>) -> Simd<f64, WD> {
    let ln2 = Simd::splat(core::f64::consts::LN_2);
    let ln2_inv = Simd::splat(1.0 / core::f64::consts::LN_2);
    let one = Simd::splat(1.0f64);
    let c2 = Simd::splat(0.5f64);
    let c3 = Simd::splat(1.0 / 6.0f64);
    let c4 = Simd::splat(1.0 / 24.0f64);
    let c5 = Simd::splat(1.0 / 120.0f64);
    let c6 = Simd::splat(1.0 / 720.0f64);
    let c7 = Simd::splat(1.0 / 5040.0f64);
    let c8 = Simd::splat(1.0 / 40320.0f64);

    let x = x
        .simd_min(Simd::splat(709.7f64))
        .simd_max(Simd::splat(-708.3f64));
    let magic_f64 = Simd::splat(6755399441055744.0f64);
    let n = (x * ln2_inv + magic_f64) - magic_f64;
    let r = x - n * ln2;

    let mut p = r * c8 + c7;
    p = r * p + c6;
    p = r * p + c5;
    p = r * p + c4;
    p = r * p + c3;
    p = r * p + c2;
    p = r * p + one;
    p = r * p + one;

    let ni: Simd<i64, WD> = n.cast();
    let pow2n =
        Simd::<f64, WD>::from_bits((ni + Simd::splat(1023i64)).cast::<u64>() << Simd::splat(52u64));
    p * pow2n
}

// ── F16/BF16 load/store helpers ──────────────────────────────────

#[inline(always)]
fn load_f16x(src: &[u8], byte_off: usize) -> Simd<f32, W> {
    Simd::from_array(core::array::from_fn(|i| {
        crate::cast::read_f16(&src[byte_off + i * 2..])
    }))
}

#[inline(always)]
fn store_f16x(dst: &mut [u8], byte_off: usize, v: Simd<f32, W>) {
    let a = v.to_array();
    for i in 0..W {
        crate::cast::write_f16(&mut dst[byte_off + i * 2..], a[i]);
    }
}

#[inline(always)]
fn load_bf16x(src: &[u8], byte_off: usize) -> Simd<f32, W> {
    Simd::from_array(core::array::from_fn(|i| {
        crate::cast::read_bf16(&src[byte_off + i * 2..])
    }))
}

#[inline(always)]
fn store_bf16x(dst: &mut [u8], byte_off: usize, v: Simd<f32, W>) {
    let a = v.to_array();
    for i in 0..W {
        crate::cast::write_bf16(&mut dst[byte_off + i * 2..], a[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F32
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn sum_f32(src: &[f32]) -> f32 {
    let n = src.len();
    let chunks = n / W;
    let tail = n % W;
    let mut acc = Simd::<f32, W>::splat(0.0);
    for i in 0..chunks {
        acc += Simd::from_slice(&src[i * W..]);
    }
    let mut s = acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        s += src[off + j];
    }
    s
}

pub(crate) fn max_f32(src: &[f32]) -> f32 {
    debug_assert!(!src.is_empty());
    let n = src.len();
    if n < W {
        let mut m = src[0];
        for &v in &src[1..] {
            if v > m {
                m = v;
            }
        }
        return m;
    }
    let chunks = n / W;
    let tail = n % W;
    let mut acc: Simd<f32, W> = Simd::from_slice(&src[0..]);
    for i in 1..chunks {
        acc = acc.simd_max(Simd::from_slice(&src[i * W..]));
    }
    let mut m = acc.reduce_max();
    let off = chunks * W;
    for j in 0..tail {
        let v = src[off + j];
        if v > m {
            m = v;
        }
    }
    m
}

pub(crate) fn argmin_f32(src: &[f32]) -> usize {
    debug_assert!(!src.is_empty());
    let n = src.len();
    if n < W {
        let mut idx = 0;
        let mut best = src[0];
        for (i, &v) in src.iter().enumerate().skip(1) {
            if v < best {
                best = v;
                idx = i;
            }
        }
        return idx;
    }

    let chunks = n / W;
    let tail = n % W;
    let mut best = Simd::<f32, W>::from_slice(&src[0..]);
    let mut best_idx = Simd::<usize, W>::from_array(core::array::from_fn(|i| i));
    for i in 1..chunks {
        let off = i * W;
        let v = Simd::<f32, W>::from_slice(&src[off..]);
        let idx = Simd::<usize, W>::from_array(core::array::from_fn(|j| off + j));
        let mask = v.simd_lt(best);
        best = mask.select(v, best);
        best_idx = mask.select(idx, best_idx);
    }

    let values = best.to_array();
    let indices = best_idx.to_array();
    let mut out = indices[0];
    let mut out_v = values[0];
    for i in 1..W {
        if values[i] < out_v || (values[i] == out_v && indices[i] < out) {
            out_v = values[i];
            out = indices[i];
        }
    }

    let off = chunks * W;
    for j in 0..tail {
        let idx = off + j;
        let v = src[idx];
        if v < out_v {
            out_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn argmax_f32(src: &[f32]) -> usize {
    debug_assert!(!src.is_empty());
    let n = src.len();
    if n < W {
        let mut idx = 0;
        let mut best = src[0];
        for (i, &v) in src.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                idx = i;
            }
        }
        return idx;
    }

    let chunks = n / W;
    let tail = n % W;
    let mut best = Simd::<f32, W>::from_slice(&src[0..]);
    let mut best_idx = Simd::<usize, W>::from_array(core::array::from_fn(|i| i));
    for i in 1..chunks {
        let off = i * W;
        let v = Simd::<f32, W>::from_slice(&src[off..]);
        let idx = Simd::<usize, W>::from_array(core::array::from_fn(|j| off + j));
        let mask = v.simd_gt(best);
        best = mask.select(v, best);
        best_idx = mask.select(idx, best_idx);
    }

    let values = best.to_array();
    let indices = best_idx.to_array();
    let mut out = indices[0];
    let mut out_v = values[0];
    for i in 1..W {
        if values[i] > out_v || (values[i] == out_v && indices[i] < out) {
            out_v = values[i];
            out = indices[i];
        }
    }

    let off = chunks * W;
    for j in 0..tail {
        let idx = off + j;
        let v = src[idx];
        if v > out_v {
            out_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn softmax_f32(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    if n == 0 {
        return;
    }

    let mx = max_f32(src);

    // exp(x - max) and accumulate sum
    let chunks = n / W;
    let tail = n % W;
    let mx_v = Simd::<f32, W>::splat(mx);
    let mut sum_acc = Simd::<f32, W>::splat(0.0);

    for i in 0..chunks {
        let off = i * W;
        let v = Simd::from_slice(&src[off..]) - mx_v;
        let e = exp_f32x(v);
        e.copy_to_slice(&mut dst[off..]);
        sum_acc += e;
    }
    let mut s = sum_acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        let e = axmath::exp_f32(src[off + j] - mx);
        dst[off + j] = e;
        s += e;
    }

    // normalize
    let inv = 1.0 / s;
    let inv_v = Simd::<f32, W>::splat(inv);
    for i in 0..chunks {
        let off = i * W;
        let v = Simd::from_slice(&dst[off..]) * inv_v;
        v.copy_to_slice(&mut dst[off..]);
    }
    for j in 0..tail {
        dst[chunks * W + j] *= inv;
    }
}

pub(crate) fn rms_norm_f32(src: &[f32], dst: &mut [f32], weight: &[f32], eps: f32) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());

    let chunks = n / W;
    let tail = n % W;

    // sum of squares
    let mut ss_acc = Simd::<f32, W>::splat(0.0);
    for i in 0..chunks {
        let v = Simd::from_slice(&src[i * W..]);
        ss_acc = v * v + ss_acc;
    }
    let mut ss = ss_acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        ss += src[off + j] * src[off + j];
    }

    let scale = 1.0 / axmath::sqrt_f32(ss / n as f32 + eps);
    let scale_v = Simd::<f32, W>::splat(scale);

    for i in 0..chunks {
        let o = i * W;
        let v = Simd::from_slice(&src[o..]) * scale_v * Simd::from_slice(&weight[o..]);
        v.copy_to_slice(&mut dst[o..]);
    }
    for j in 0..tail {
        dst[off + j] = src[off + j] * scale * weight[off + j];
    }
}

pub(crate) fn layer_norm_f32(src: &[f32], dst: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());
    debug_assert_eq!(n, bias.len());

    // mean
    let mean = sum_f32(src) / n as f32;

    // variance
    let chunks = n / W;
    let tail = n % W;
    let mean_v = Simd::<f32, W>::splat(mean);
    let mut var_acc = Simd::<f32, W>::splat(0.0);
    for i in 0..chunks {
        let d = Simd::from_slice(&src[i * W..]) - mean_v;
        var_acc = d * d + var_acc;
    }
    let mut var = var_acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        let d = src[off + j] - mean;
        var += d * d;
    }

    let inv_std = 1.0 / axmath::sqrt_f32(var / n as f32 + eps);
    let inv_v = Simd::<f32, W>::splat(inv_std);

    for i in 0..chunks {
        let o = i * W;
        let normed = (Simd::from_slice(&src[o..]) - mean_v) * inv_v;
        let out = normed * Simd::from_slice(&weight[o..]) + Simd::from_slice(&bias[o..]);
        out.copy_to_slice(&mut dst[o..]);
    }
    for j in 0..tail {
        let idx = off + j;
        dst[idx] = (src[idx] - mean) * inv_std * weight[idx] + bias[idx];
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F64
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn sum_f64(src: &[f64]) -> f64 {
    let n = src.len();
    let chunks = n / WD;
    let tail = n % WD;
    let mut acc = Simd::<f64, WD>::splat(0.0);
    for i in 0..chunks {
        acc += Simd::from_slice(&src[i * WD..]);
    }
    let mut s = acc.reduce_sum();
    let off = chunks * WD;
    for j in 0..tail {
        s += src[off + j];
    }
    s
}

pub(crate) fn max_f64(src: &[f64]) -> f64 {
    debug_assert!(!src.is_empty());
    let n = src.len();
    if n < WD {
        return src[0];
    }
    let chunks = n / WD;
    let tail = n % WD;
    let mut acc: Simd<f64, WD> = Simd::from_slice(&src[0..]);
    for i in 1..chunks {
        acc = acc.simd_max(Simd::from_slice(&src[i * WD..]));
    }
    let mut m = acc.reduce_max();
    let off = chunks * WD;
    for j in 0..tail {
        let v = src[off + j];
        if v > m {
            m = v;
        }
    }
    m
}

pub(crate) fn argmin_f64(src: &[f64]) -> usize {
    debug_assert!(!src.is_empty());
    let n = src.len();
    if n < WD {
        let mut idx = 0;
        let mut best = src[0];
        for (i, &v) in src.iter().enumerate().skip(1) {
            if v < best {
                best = v;
                idx = i;
            }
        }
        return idx;
    }

    let chunks = n / WD;
    let tail = n % WD;
    let mut best = Simd::<f64, WD>::from_slice(&src[0..]);
    let mut best_idx = Simd::<usize, WD>::from_array(core::array::from_fn(|i| i));
    for i in 1..chunks {
        let off = i * WD;
        let v = Simd::<f64, WD>::from_slice(&src[off..]);
        let idx = Simd::<usize, WD>::from_array(core::array::from_fn(|j| off + j));
        let mask = v.simd_lt(best);
        best = mask.select(v, best);
        best_idx = mask.select(idx, best_idx);
    }

    let values = best.to_array();
    let indices = best_idx.to_array();
    let mut out = indices[0];
    let mut out_v = values[0];
    for i in 1..WD {
        if values[i] < out_v || (values[i] == out_v && indices[i] < out) {
            out_v = values[i];
            out = indices[i];
        }
    }

    let off = chunks * WD;
    for j in 0..tail {
        let idx = off + j;
        let v = src[idx];
        if v < out_v {
            out_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn argmax_f64(src: &[f64]) -> usize {
    debug_assert!(!src.is_empty());
    let n = src.len();
    if n < WD {
        let mut idx = 0;
        let mut best = src[0];
        for (i, &v) in src.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                idx = i;
            }
        }
        return idx;
    }

    let chunks = n / WD;
    let tail = n % WD;
    let mut best = Simd::<f64, WD>::from_slice(&src[0..]);
    let mut best_idx = Simd::<usize, WD>::from_array(core::array::from_fn(|i| i));
    for i in 1..chunks {
        let off = i * WD;
        let v = Simd::<f64, WD>::from_slice(&src[off..]);
        let idx = Simd::<usize, WD>::from_array(core::array::from_fn(|j| off + j));
        let mask = v.simd_gt(best);
        best = mask.select(v, best);
        best_idx = mask.select(idx, best_idx);
    }

    let values = best.to_array();
    let indices = best_idx.to_array();
    let mut out = indices[0];
    let mut out_v = values[0];
    for i in 1..WD {
        if values[i] > out_v || (values[i] == out_v && indices[i] < out) {
            out_v = values[i];
            out = indices[i];
        }
    }

    let off = chunks * WD;
    for j in 0..tail {
        let idx = off + j;
        let v = src[idx];
        if v > out_v {
            out_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn softmax_f64(src: &[f64], dst: &mut [f64]) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    if n == 0 {
        return;
    }

    let mx = max_f64(src);

    let chunks = n / WD;
    let tail = n % WD;
    let mx_v = Simd::<f64, WD>::splat(mx);
    let mut sum_acc = Simd::<f64, WD>::splat(0.0);

    for i in 0..chunks {
        let off = i * WD;
        let v = Simd::from_slice(&src[off..]) - mx_v;
        let e = exp_f64x(v);
        e.copy_to_slice(&mut dst[off..]);
        sum_acc += e;
    }
    let mut s = sum_acc.reduce_sum();
    let off = chunks * WD;
    for j in 0..tail {
        let e = axmath::exp_f64(src[off + j] - mx);
        dst[off + j] = e;
        s += e;
    }

    let inv = 1.0 / s;
    let inv_v = Simd::<f64, WD>::splat(inv);
    for i in 0..chunks {
        let off = i * WD;
        let v = Simd::from_slice(&dst[off..]) * inv_v;
        v.copy_to_slice(&mut dst[off..]);
    }
    for j in 0..tail {
        dst[chunks * WD + j] *= inv;
    }
}

pub(crate) fn rms_norm_f64(src: &[f64], dst: &mut [f64], weight: &[f64], eps: f64) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());

    let chunks = n / WD;
    let tail = n % WD;

    let mut ss_acc = Simd::<f64, WD>::splat(0.0);
    for i in 0..chunks {
        let v = Simd::from_slice(&src[i * WD..]);
        ss_acc = v * v + ss_acc;
    }
    let mut ss = ss_acc.reduce_sum();
    let off = chunks * WD;
    for j in 0..tail {
        ss += src[off + j] * src[off + j];
    }

    let scale = 1.0 / axmath::sqrt_f64(ss / n as f64 + eps);
    let scale_v = Simd::<f64, WD>::splat(scale);

    for i in 0..chunks {
        let o = i * WD;
        let v = Simd::from_slice(&src[o..]) * scale_v * Simd::from_slice(&weight[o..]);
        v.copy_to_slice(&mut dst[o..]);
    }
    for j in 0..tail {
        dst[off + j] = src[off + j] * scale * weight[off + j];
    }
}

pub(crate) fn layer_norm_f64(src: &[f64], dst: &mut [f64], weight: &[f64], bias: &[f64], eps: f64) {
    let n = src.len();
    debug_assert_eq!(n, dst.len());
    debug_assert_eq!(n, weight.len());
    debug_assert_eq!(n, bias.len());

    let mean = sum_f64(src) / n as f64;

    let chunks = n / WD;
    let tail = n % WD;
    let mean_v = Simd::<f64, WD>::splat(mean);
    let mut var_acc = Simd::<f64, WD>::splat(0.0);
    for i in 0..chunks {
        let d = Simd::from_slice(&src[i * WD..]) - mean_v;
        var_acc = d * d + var_acc;
    }
    let mut var = var_acc.reduce_sum();
    let off = chunks * WD;
    for j in 0..tail {
        let d = src[off + j] - mean;
        var += d * d;
    }

    let inv_std = 1.0 / axmath::sqrt_f64(var / n as f64 + eps);
    let inv_v = Simd::<f64, WD>::splat(inv_std);

    for i in 0..chunks {
        let o = i * WD;
        let normed = (Simd::from_slice(&src[o..]) - mean_v) * inv_v;
        let out = normed * Simd::from_slice(&weight[o..]) + Simd::from_slice(&bias[o..]);
        out.copy_to_slice(&mut dst[o..]);
    }
    for j in 0..tail {
        let idx = off + j;
        dst[idx] = (src[idx] - mean) * inv_std * weight[idx] + bias[idx];
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F16  (packed &[u8], scalar results in f32)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn sum_f16(src: &[u8]) -> f32 {
    let n = src.len() / 2;
    let chunks = n / W;
    let tail = n % W;
    let mut acc = Simd::<f32, W>::splat(0.0);
    for i in 0..chunks {
        acc += load_f16x(src, i * W * 2);
    }
    let mut s = acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        s += crate::cast::read_f16(&src[(off + j) * 2..]);
    }
    s
}

pub(crate) fn max_f16(src: &[u8]) -> f32 {
    let n = src.len() / 2;
    debug_assert!(n > 0);
    if n < W {
        let mut m = crate::cast::read_f16(src);
        for j in 1..n {
            let v = crate::cast::read_f16(&src[j * 2..]);
            if v > m {
                m = v;
            }
        }
        return m;
    }
    let chunks = n / W;
    let tail = n % W;
    let mut acc = load_f16x(src, 0);
    for i in 1..chunks {
        acc = acc.simd_max(load_f16x(src, i * W * 2));
    }
    let mut m = acc.reduce_max();
    let off = chunks * W;
    for j in 0..tail {
        let v = crate::cast::read_f16(&src[(off + j) * 2..]);
        if v > m {
            m = v;
        }
    }
    m
}

pub(crate) fn argmin_f16(src: &[u8]) -> usize {
    let n = src.len() / 2;
    debug_assert!(n > 0);
    if n < W {
        let mut idx = 0;
        let mut best = crate::cast::read_f16(src);
        for j in 1..n {
            let v = crate::cast::read_f16(&src[j * 2..]);
            if v < best {
                best = v;
                idx = j;
            }
        }
        return idx;
    }
    let chunks = n / W;
    let tail = n % W;
    let mut best = load_f16x(src, 0);
    let mut best_idx = Simd::<usize, W>::from_array(core::array::from_fn(|i| i));
    for i in 1..chunks {
        let off = i * W;
        let v = load_f16x(src, off * 2);
        let idx = Simd::<usize, W>::from_array(core::array::from_fn(|j| off + j));
        let mask = v.simd_lt(best);
        best = mask.select(v, best);
        best_idx = mask.select(idx, best_idx);
    }
    let values = best.to_array();
    let indices = best_idx.to_array();
    let mut out = indices[0];
    let mut out_v = values[0];
    for i in 1..W {
        if values[i] < out_v || (values[i] == out_v && indices[i] < out) {
            out_v = values[i];
            out = indices[i];
        }
    }
    let off = chunks * W;
    for j in 0..tail {
        let idx = off + j;
        let v = crate::cast::read_f16(&src[idx * 2..]);
        if v < out_v {
            out_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn argmax_f16(src: &[u8]) -> usize {
    let n = src.len() / 2;
    debug_assert!(n > 0);
    if n < W {
        let mut idx = 0;
        let mut best = crate::cast::read_f16(src);
        for j in 1..n {
            let v = crate::cast::read_f16(&src[j * 2..]);
            if v > best {
                best = v;
                idx = j;
            }
        }
        return idx;
    }
    let chunks = n / W;
    let tail = n % W;
    let mut best = load_f16x(src, 0);
    let mut best_idx = Simd::<usize, W>::from_array(core::array::from_fn(|i| i));
    for i in 1..chunks {
        let off = i * W;
        let v = load_f16x(src, off * 2);
        let idx = Simd::<usize, W>::from_array(core::array::from_fn(|j| off + j));
        let mask = v.simd_gt(best);
        best = mask.select(v, best);
        best_idx = mask.select(idx, best_idx);
    }
    let values = best.to_array();
    let indices = best_idx.to_array();
    let mut out = indices[0];
    let mut out_v = values[0];
    for i in 1..W {
        if values[i] > out_v || (values[i] == out_v && indices[i] < out) {
            out_v = values[i];
            out = indices[i];
        }
    }
    let off = chunks * W;
    for j in 0..tail {
        let idx = off + j;
        let v = crate::cast::read_f16(&src[idx * 2..]);
        if v > out_v {
            out_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn softmax_f16(src: &[u8], dst: &mut [u8]) {
    let n = src.len() / 2;
    debug_assert_eq!(src.len(), dst.len());
    if n == 0 {
        return;
    }

    let mx = max_f16(src);
    let mx_v = Simd::<f32, W>::splat(mx);

    let chunks = n / W;
    let tail = n % W;
    let mut sum_acc = Simd::<f32, W>::splat(0.0);

    for i in 0..chunks {
        let boff = i * W * 2;
        let v = load_f16x(src, boff) - mx_v;
        let e = exp_f32x(v);
        store_f16x(dst, boff, e);
        sum_acc += e;
    }
    let mut s = sum_acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        let boff = (off + j) * 2;
        let e = axmath::exp_f32(crate::cast::read_f16(&src[boff..]) - mx);
        crate::cast::write_f16(&mut dst[boff..], e);
        s += e;
    }

    // normalize
    let inv = 1.0 / s;
    let inv_v = Simd::<f32, W>::splat(inv);
    for i in 0..chunks {
        let boff = i * W * 2;
        let v = load_f16x(dst, boff) * inv_v;
        store_f16x(dst, boff, v);
    }
    for j in 0..tail {
        let boff = (off + j) * 2;
        let v = crate::cast::read_f16(&dst[boff..]) * inv;
        crate::cast::write_f16(&mut dst[boff..], v);
    }
}

pub(crate) fn rms_norm_f16(src: &[u8], dst: &mut [u8], weight: &[u8], eps: f32) {
    let n = src.len() / 2;
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());

    let chunks = n / W;
    let tail = n % W;

    let mut ss_acc = Simd::<f32, W>::splat(0.0);
    for i in 0..chunks {
        let v = load_f16x(src, i * W * 2);
        ss_acc = v * v + ss_acc;
    }
    let mut ss = ss_acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        let v = crate::cast::read_f16(&src[(off + j) * 2..]);
        ss += v * v;
    }

    let scale = 1.0 / axmath::sqrt_f32(ss / n as f32 + eps);
    let scale_v = Simd::<f32, W>::splat(scale);

    for i in 0..chunks {
        let boff = i * W * 2;
        let v = load_f16x(src, boff) * scale_v * load_f16x(weight, boff);
        store_f16x(dst, boff, v);
    }
    for j in 0..tail {
        let boff = (off + j) * 2;
        let sv = crate::cast::read_f16(&src[boff..]);
        let wv = crate::cast::read_f16(&weight[boff..]);
        crate::cast::write_f16(&mut dst[boff..], sv * scale * wv);
    }
}

pub(crate) fn layer_norm_f16(src: &[u8], dst: &mut [u8], weight: &[u8], bias: &[u8], eps: f32) {
    let n = src.len() / 2;
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    debug_assert_eq!(src.len(), bias.len());

    let mean = sum_f16(src) / n as f32;

    let chunks = n / W;
    let tail = n % W;
    let mean_v = Simd::<f32, W>::splat(mean);

    let mut var_acc = Simd::<f32, W>::splat(0.0);
    for i in 0..chunks {
        let d = load_f16x(src, i * W * 2) - mean_v;
        var_acc = d * d + var_acc;
    }
    let mut var = var_acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        let d = crate::cast::read_f16(&src[(off + j) * 2..]) - mean;
        var += d * d;
    }

    let inv_std = 1.0 / axmath::sqrt_f32(var / n as f32 + eps);
    let inv_v = Simd::<f32, W>::splat(inv_std);

    for i in 0..chunks {
        let boff = i * W * 2;
        let normed = (load_f16x(src, boff) - mean_v) * inv_v;
        let out = normed * load_f16x(weight, boff) + load_f16x(bias, boff);
        store_f16x(dst, boff, out);
    }
    for j in 0..tail {
        let boff = (off + j) * 2;
        let sv = crate::cast::read_f16(&src[boff..]);
        let wv = crate::cast::read_f16(&weight[boff..]);
        let bv = crate::cast::read_f16(&bias[boff..]);
        crate::cast::write_f16(&mut dst[boff..], (sv - mean) * inv_std * wv + bv);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  BF16  (packed &[u8], scalar results in f32)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn sum_bf16(src: &[u8]) -> f32 {
    let n = src.len() / 2;
    let chunks = n / W;
    let tail = n % W;
    let mut acc = Simd::<f32, W>::splat(0.0);
    for i in 0..chunks {
        acc += load_bf16x(src, i * W * 2);
    }
    let mut s = acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        s += crate::cast::read_bf16(&src[(off + j) * 2..]);
    }
    s
}

pub(crate) fn max_bf16(src: &[u8]) -> f32 {
    let n = src.len() / 2;
    debug_assert!(n > 0);
    if n < W {
        let mut m = crate::cast::read_bf16(src);
        for j in 1..n {
            let v = crate::cast::read_bf16(&src[j * 2..]);
            if v > m {
                m = v;
            }
        }
        return m;
    }
    let chunks = n / W;
    let tail = n % W;
    let mut acc = load_bf16x(src, 0);
    for i in 1..chunks {
        acc = acc.simd_max(load_bf16x(src, i * W * 2));
    }
    let mut m = acc.reduce_max();
    let off = chunks * W;
    for j in 0..tail {
        let v = crate::cast::read_bf16(&src[(off + j) * 2..]);
        if v > m {
            m = v;
        }
    }
    m
}

pub(crate) fn argmin_bf16(src: &[u8]) -> usize {
    let n = src.len() / 2;
    debug_assert!(n > 0);
    if n < W {
        let mut idx = 0;
        let mut best = crate::cast::read_bf16(src);
        for j in 1..n {
            let v = crate::cast::read_bf16(&src[j * 2..]);
            if v < best {
                best = v;
                idx = j;
            }
        }
        return idx;
    }
    let chunks = n / W;
    let tail = n % W;
    let mut best = load_bf16x(src, 0);
    let mut best_idx = Simd::<usize, W>::from_array(core::array::from_fn(|i| i));
    for i in 1..chunks {
        let off = i * W;
        let v = load_bf16x(src, off * 2);
        let idx = Simd::<usize, W>::from_array(core::array::from_fn(|j| off + j));
        let mask = v.simd_lt(best);
        best = mask.select(v, best);
        best_idx = mask.select(idx, best_idx);
    }
    let values = best.to_array();
    let indices = best_idx.to_array();
    let mut out = indices[0];
    let mut out_v = values[0];
    for i in 1..W {
        if values[i] < out_v || (values[i] == out_v && indices[i] < out) {
            out_v = values[i];
            out = indices[i];
        }
    }
    let off = chunks * W;
    for j in 0..tail {
        let idx = off + j;
        let v = crate::cast::read_bf16(&src[idx * 2..]);
        if v < out_v {
            out_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn argmax_bf16(src: &[u8]) -> usize {
    let n = src.len() / 2;
    debug_assert!(n > 0);
    if n < W {
        let mut idx = 0;
        let mut best = crate::cast::read_bf16(src);
        for j in 1..n {
            let v = crate::cast::read_bf16(&src[j * 2..]);
            if v > best {
                best = v;
                idx = j;
            }
        }
        return idx;
    }
    let chunks = n / W;
    let tail = n % W;
    let mut best = load_bf16x(src, 0);
    let mut best_idx = Simd::<usize, W>::from_array(core::array::from_fn(|i| i));
    for i in 1..chunks {
        let off = i * W;
        let v = load_bf16x(src, off * 2);
        let idx = Simd::<usize, W>::from_array(core::array::from_fn(|j| off + j));
        let mask = v.simd_gt(best);
        best = mask.select(v, best);
        best_idx = mask.select(idx, best_idx);
    }
    let values = best.to_array();
    let indices = best_idx.to_array();
    let mut out = indices[0];
    let mut out_v = values[0];
    for i in 1..W {
        if values[i] > out_v || (values[i] == out_v && indices[i] < out) {
            out_v = values[i];
            out = indices[i];
        }
    }
    let off = chunks * W;
    for j in 0..tail {
        let idx = off + j;
        let v = crate::cast::read_bf16(&src[idx * 2..]);
        if v > out_v {
            out_v = v;
            out = idx;
        }
    }
    out
}

pub(crate) fn softmax_bf16(src: &[u8], dst: &mut [u8]) {
    let n = src.len() / 2;
    debug_assert_eq!(src.len(), dst.len());
    if n == 0 {
        return;
    }

    let mx = max_bf16(src);
    let mx_v = Simd::<f32, W>::splat(mx);

    let chunks = n / W;
    let tail = n % W;
    let mut sum_acc = Simd::<f32, W>::splat(0.0);

    for i in 0..chunks {
        let boff = i * W * 2;
        let v = load_bf16x(src, boff) - mx_v;
        let e = exp_f32x(v);
        store_bf16x(dst, boff, e);
        sum_acc += e;
    }
    let mut s = sum_acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        let boff = (off + j) * 2;
        let e = axmath::exp_f32(crate::cast::read_bf16(&src[boff..]) - mx);
        crate::cast::write_bf16(&mut dst[boff..], e);
        s += e;
    }

    let inv = 1.0 / s;
    let inv_v = Simd::<f32, W>::splat(inv);
    for i in 0..chunks {
        let boff = i * W * 2;
        let v = load_bf16x(dst, boff) * inv_v;
        store_bf16x(dst, boff, v);
    }
    for j in 0..tail {
        let boff = (off + j) * 2;
        let v = crate::cast::read_bf16(&dst[boff..]) * inv;
        crate::cast::write_bf16(&mut dst[boff..], v);
    }
}

pub(crate) fn rms_norm_bf16(src: &[u8], dst: &mut [u8], weight: &[u8], eps: f32) {
    let n = src.len() / 2;
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());

    let chunks = n / W;
    let tail = n % W;

    let mut ss_acc = Simd::<f32, W>::splat(0.0);
    for i in 0..chunks {
        let v = load_bf16x(src, i * W * 2);
        ss_acc = v * v + ss_acc;
    }
    let mut ss = ss_acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        let v = crate::cast::read_bf16(&src[(off + j) * 2..]);
        ss += v * v;
    }

    let scale = 1.0 / axmath::sqrt_f32(ss / n as f32 + eps);
    let scale_v = Simd::<f32, W>::splat(scale);

    for i in 0..chunks {
        let boff = i * W * 2;
        let v = load_bf16x(src, boff) * scale_v * load_bf16x(weight, boff);
        store_bf16x(dst, boff, v);
    }
    for j in 0..tail {
        let boff = (off + j) * 2;
        let sv = crate::cast::read_bf16(&src[boff..]);
        let wv = crate::cast::read_bf16(&weight[boff..]);
        crate::cast::write_bf16(&mut dst[boff..], sv * scale * wv);
    }
}

pub(crate) fn layer_norm_bf16(src: &[u8], dst: &mut [u8], weight: &[u8], bias: &[u8], eps: f32) {
    let n = src.len() / 2;
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), weight.len());
    debug_assert_eq!(src.len(), bias.len());

    let mean = sum_bf16(src) / n as f32;

    let chunks = n / W;
    let tail = n % W;
    let mean_v = Simd::<f32, W>::splat(mean);

    let mut var_acc = Simd::<f32, W>::splat(0.0);
    for i in 0..chunks {
        let d = load_bf16x(src, i * W * 2) - mean_v;
        var_acc = d * d + var_acc;
    }
    let mut var = var_acc.reduce_sum();
    let off = chunks * W;
    for j in 0..tail {
        let d = crate::cast::read_bf16(&src[(off + j) * 2..]) - mean;
        var += d * d;
    }

    let inv_std = 1.0 / axmath::sqrt_f32(var / n as f32 + eps);
    let inv_v = Simd::<f32, W>::splat(inv_std);

    for i in 0..chunks {
        let boff = i * W * 2;
        let normed = (load_bf16x(src, boff) - mean_v) * inv_v;
        let out = normed * load_bf16x(weight, boff) + load_bf16x(bias, boff);
        store_bf16x(dst, boff, out);
    }
    for j in 0..tail {
        let boff = (off + j) * 2;
        let sv = crate::cast::read_bf16(&src[boff..]);
        let wv = crate::cast::read_bf16(&weight[boff..]);
        let bv = crate::cast::read_bf16(&bias[boff..]);
        crate::cast::write_bf16(&mut dst[boff..], (sv - mean) * inv_std * wv + bv);
    }
}
