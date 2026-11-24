//! Rotary position embedding kernels — portable `core::simd`.
//!
//! Two layouts:
//!   Interleaved (Llama/Mistral): pairs [e,o,e,o,...] within each head.
//!     Pair structure doesn't map cleanly to portable SIMD without swizzle,
//!     so we use scalar pair processing (LLVM auto-vectorizes the inner loop).
//!   NeoX/Qwen (split halves): first half vs second half of each head.
//!     Perfectly SIMD-friendly — lo/hi are contiguous, just FMA.

use core::simd::prelude::*;

use super::W;
const WD: usize = super::WD;

// ═══════════════════════════════════════════════════════════════════
//  F16/BF16 SIMD load/store helpers
// ═══════════════════════════════════════════════════════════════════

#[inline(always)]
fn load_f16xw(src: &[u8], byte_off: usize) -> Simd<f32, W> {
    Simd::from_array(core::array::from_fn(|i| {
        crate::cast::read_f16(&src[byte_off + i * 2..])
    }))
}

#[inline(always)]
fn store_f16xw(dst: &mut [u8], byte_off: usize, v: Simd<f32, W>) {
    let a = v.as_array();
    for i in 0..W {
        crate::cast::write_f16(&mut dst[byte_off + i * 2..], a[i]);
    }
}

#[inline(always)]
fn load_bf16xw(src: &[u8], byte_off: usize) -> Simd<f32, W> {
    Simd::from_array(core::array::from_fn(|i| {
        crate::cast::read_bf16(&src[byte_off + i * 2..])
    }))
}

#[inline(always)]
fn store_bf16xw(dst: &mut [u8], byte_off: usize, v: Simd<f32, W>) {
    let a = v.as_array();
    for i in 0..W {
        crate::cast::write_bf16(&mut dst[byte_off + i * 2..], a[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Interleaved RoPE — F32
// ═══════════════════════════════════════════════════════════════════
//
// q[2i]   = q[2i] * cos − q[2i+1] * sin
// q[2i+1] = q[2i] * sin + q[2i+1] * cos
//
// Scalar pair processing — LLVM auto-vectorizes the inner loop.

pub(crate) fn rope_interleaved_f32(
    qk: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let ei = base + i * 2;
                let e = qk[ei];
                let o = qk[ei + 1];
                let c = cos[freq_base + i];
                let s = sin[freq_base + i];
                qk[ei] = e * c - o * s;
                qk[ei + 1] = e * s + o * c;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  NeoX / split-half RoPE — F32
// ═══════════════════════════════════════════════════════════════════
//
// q[i]        = q[i]        * cos − q[i + half] * sin
// q[i + half] = q[i + half] * cos + q[i]        * sin

pub(crate) fn rope_neox_f32(
    qk: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;
    let chunks = half / W;
    let tail = half % W;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;

            for ci in 0..chunks {
                let off = ci * W;
                let lo_off = base + off;
                let hi_off = base + half + off;
                let cos_v = Simd::<f32, W>::from_slice(&cos[freq_base + off..]);
                let sin_v = Simd::<f32, W>::from_slice(&sin[freq_base + off..]);
                let lo_v = Simd::<f32, W>::from_slice(&qk[lo_off..]);
                let hi_v = Simd::<f32, W>::from_slice(&qk[hi_off..]);

                // new_lo = lo*cos − hi*sin
                let new_lo = lo_v * cos_v - hi_v * sin_v;
                // new_hi = lo*sin + hi*cos
                let new_hi = lo_v * sin_v + hi_v * cos_v;

                new_lo.copy_to_slice(&mut qk[lo_off..]);
                new_hi.copy_to_slice(&mut qk[hi_off..]);
            }

            // Scalar tail
            let tail_start = chunks * W;
            let head = &mut qk[base..base + head_dim];
            let (lo, hi) = head.split_at_mut(half);
            for i in tail_start..tail_start + tail {
                let lv = lo[i];
                let hv = hi[i];
                let c = cos[freq_base + i];
                let s = sin[freq_base + i];
                lo[i] = lv * c - hv * s;
                hi[i] = hv * c + lv * s;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F64 variants
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn rope_interleaved_f64(
    qk: &mut [f64],
    cos: &[f64],
    sin: &[f64],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let ei = base + i * 2;
                let e = qk[ei];
                let o = qk[ei + 1];
                let c = cos[freq_base + i];
                let s = sin[freq_base + i];
                qk[ei] = e * c - o * s;
                qk[ei + 1] = e * s + o * c;
            }
        }
    }
}

pub(crate) fn rope_neox_f64(
    qk: &mut [f64],
    cos: &[f64],
    sin: &[f64],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;
    let chunks = half / WD;
    let tail = half % WD;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;

            for ci in 0..chunks {
                let off = ci * WD;
                let lo_off = base + off;
                let hi_off = base + half + off;
                let cos_v = Simd::<f64, WD>::from_slice(&cos[freq_base + off..]);
                let sin_v = Simd::<f64, WD>::from_slice(&sin[freq_base + off..]);
                let lo_v = Simd::<f64, WD>::from_slice(&qk[lo_off..]);
                let hi_v = Simd::<f64, WD>::from_slice(&qk[hi_off..]);

                let new_lo = lo_v * cos_v - hi_v * sin_v;
                let new_hi = lo_v * sin_v + hi_v * cos_v;

                new_lo.copy_to_slice(&mut qk[lo_off..]);
                new_hi.copy_to_slice(&mut qk[hi_off..]);
            }

            let tail_start = chunks * WD;
            let head = &mut qk[base..base + head_dim];
            let (lo, hi) = head.split_at_mut(half);
            for i in tail_start..tail_start + tail {
                let lv = lo[i];
                let hv = hi[i];
                let c = cos[freq_base + i];
                let s = sin[freq_base + i];
                lo[i] = lv * c - hv * s;
                hi[i] = hv * c + lv * s;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F16 (packed &[u8], cos/sin always f32)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn rope_interleaved_f16(
    qk: &mut [u8],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base_bytes = (t * n_heads + h) * head_dim * 2;
            for i in 0..half {
                let bo = base_bytes + i * 4; // 2 f16 = 4 bytes per pair
                let e = crate::cast::read_f16(&qk[bo..]);
                let o = crate::cast::read_f16(&qk[bo + 2..]);
                let c = cos[freq_base + i];
                let s = sin[freq_base + i];
                crate::cast::write_f16(&mut qk[bo..], e * c - o * s);
                crate::cast::write_f16(&mut qk[bo + 2..], e * s + o * c);
            }
        }
    }
}

pub(crate) fn rope_neox_f16(
    qk: &mut [u8],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;
    let chunks = half / W;
    let tail = half % W;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base_bytes = (t * n_heads + h) * head_dim * 2;

            for ci in 0..chunks {
                let off = ci * W;
                let lo_bo = base_bytes + off * 2;
                let hi_bo = base_bytes + half * 2 + off * 2;

                let lo_v = load_f16xw(qk, lo_bo);
                let hi_v = load_f16xw(qk, hi_bo);
                let cos_v = Simd::<f32, W>::from_slice(&cos[freq_base + off..]);
                let sin_v = Simd::<f32, W>::from_slice(&sin[freq_base + off..]);

                let new_lo = lo_v * cos_v - hi_v * sin_v;
                let new_hi = lo_v * sin_v + hi_v * cos_v;

                store_f16xw(qk, lo_bo, new_lo);
                store_f16xw(qk, hi_bo, new_hi);
            }

            let tail_start = chunks * W;
            for i in tail_start..tail_start + tail {
                let lo_bo = base_bytes + i * 2;
                let hi_bo = base_bytes + half * 2 + i * 2;
                let lv = crate::cast::read_f16(&qk[lo_bo..]);
                let hv = crate::cast::read_f16(&qk[hi_bo..]);
                let c = cos[freq_base + i];
                let s = sin[freq_base + i];
                crate::cast::write_f16(&mut qk[lo_bo..], lv * c - hv * s);
                crate::cast::write_f16(&mut qk[hi_bo..], hv * c + lv * s);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  BF16 (packed &[u8], cos/sin always f32)
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn rope_interleaved_bf16(
    qk: &mut [u8],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base_bytes = (t * n_heads + h) * head_dim * 2;
            for i in 0..half {
                let bo = base_bytes + i * 4;
                let e = crate::cast::read_bf16(&qk[bo..]);
                let o = crate::cast::read_bf16(&qk[bo + 2..]);
                let c = cos[freq_base + i];
                let s = sin[freq_base + i];
                crate::cast::write_bf16(&mut qk[bo..], e * c - o * s);
                crate::cast::write_bf16(&mut qk[bo + 2..], e * s + o * c);
            }
        }
    }
}

pub(crate) fn rope_neox_bf16(
    qk: &mut [u8],
    cos: &[f32],
    sin: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;
    let chunks = half / W;
    let tail = half % W;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base_bytes = (t * n_heads + h) * head_dim * 2;

            for ci in 0..chunks {
                let off = ci * W;
                let lo_bo = base_bytes + off * 2;
                let hi_bo = base_bytes + half * 2 + off * 2;

                let lo_v = load_bf16xw(qk, lo_bo);
                let hi_v = load_bf16xw(qk, hi_bo);
                let cos_v = Simd::<f32, W>::from_slice(&cos[freq_base + off..]);
                let sin_v = Simd::<f32, W>::from_slice(&sin[freq_base + off..]);

                let new_lo = lo_v * cos_v - hi_v * sin_v;
                let new_hi = lo_v * sin_v + hi_v * cos_v;

                store_bf16xw(qk, lo_bo, new_lo);
                store_bf16xw(qk, hi_bo, new_hi);
            }

            let tail_start = chunks * W;
            for i in tail_start..tail_start + tail {
                let lo_bo = base_bytes + i * 2;
                let hi_bo = base_bytes + half * 2 + i * 2;
                let lv = crate::cast::read_bf16(&qk[lo_bo..]);
                let hv = crate::cast::read_bf16(&qk[hi_bo..]);
                let c = cos[freq_base + i];
                let s = sin[freq_base + i];
                crate::cast::write_bf16(&mut qk[lo_bo..], lv * c - hv * s);
                crate::cast::write_bf16(&mut qk[hi_bo..], hv * c + lv * s);
            }
        }
    }
}
