//! Rotary position embedding kernels — AVX2.
//!
//! Two layouts:
//!   Interleaved (Llama/Mistral): pairs [e,o,e,o,...] within each head.
//!     SIMD: shuffle to swap pairs, XOR sign trick for negation.
//!   NeoX/Qwen (split halves): first half vs second half of each head.
//!     SIMD: straightforward — lo/hi are contiguous, just fmsub/fmadd.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

// ═══════════════════════════════════════════════════════════════════
//  Interleaved RoPE — F32
// ═══════════════════════════════════════════════════════════════════
//
// q[2i]   = q[2i] * cos − q[2i+1] * sin
// q[2i+1] = q[2i] * sin + q[2i+1] * cos
//
// Process 4 pairs (8 floats) per iteration:
//   v       = [e0,o0,e1,o1,e2,o2,e3,o3]
//   v_swap  = [o0,e0,o1,e1,...] via shuffle(0xB1)
//   cos_dup = [c0,c0,c1,c1,...] via unpack+insert
//   sin_neg = [-s0,s0,-s1,s1,...] via XOR sign mask
//   result  = fmadd(v, cos_dup, v_swap * sin_neg)

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
    let pairs_simd = half / 4; // 4 pairs = 8 f32 per YMM
    let pairs_tail = half % 4;

    unsafe {
        // Sign mask: negate even positions (the "e" slot in swapped pairs).
        let sign_mask = _mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

        for t in 0..n_tokens {
            let freq_base = t * half;
            let sin_base = t * half;
            for h in 0..n_heads {
                let base = (t * n_heads + h) * head_dim;

                for ci in 0..pairs_simd {
                    let vi = base + ci * 8;
                    let fi = freq_base + ci * 4;

                    let v = _mm256_loadu_ps(qk.as_ptr().add(vi));
                    let v_swap = _mm256_shuffle_ps(v, v, 0xB1);

                    // Duplicate cos/sin: [c0,c1,c2,c3] → [c0,c0,c1,c1,c2,c2,c3,c3]
                    let c4 = _mm_loadu_ps(cos.as_ptr().add(fi));
                    let c_lo = _mm_unpacklo_ps(c4, c4);
                    let c_hi = _mm_unpackhi_ps(c4, c4);
                    let cos_dup = _mm256_insertf128_ps(_mm256_castps128_ps256(c_lo), c_hi, 1);

                    let s4 = _mm_loadu_ps(sin.as_ptr().add(sin_base + ci * 4));
                    let s_lo = _mm_unpacklo_ps(s4, s4);
                    let s_hi = _mm_unpackhi_ps(s4, s4);
                    let sin_dup = _mm256_insertf128_ps(_mm256_castps128_ps256(s_lo), s_hi, 1);

                    // Negate even positions: [-s0,s0,-s1,s1,...]
                    let sin_neg = _mm256_xor_ps(sin_dup, sign_mask);

                    let result = _mm256_fmadd_ps(v, cos_dup, _mm256_mul_ps(v_swap, sin_neg));
                    _mm256_storeu_ps(qk.as_mut_ptr().add(vi), result);
                }

                // Scalar tail for remaining pairs.
                let tail_start = pairs_simd * 4;
                for i in tail_start..tail_start + pairs_tail {
                    let ei = base + i * 2;
                    let e = qk[ei];
                    let o = qk[ei + 1];
                    let c = cos[freq_base + i];
                    let s = sin[sin_base + i];
                    qk[ei] = e * c - o * s;
                    qk[ei + 1] = e * s + o * c;
                }
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
    let chunks = half / 8;
    let tail = half % 8;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;

            unsafe {
                for ci in 0..chunks {
                    let off = ci * 8;
                    let lo_ptr = qk.as_mut_ptr().add(base + off);
                    let hi_ptr = qk.as_mut_ptr().add(base + half + off);
                    let cos_v = _mm256_loadu_ps(cos.as_ptr().add(freq_base + off));
                    let sin_v = _mm256_loadu_ps(sin.as_ptr().add(freq_base + off));

                    let lo_v = _mm256_loadu_ps(lo_ptr);
                    let hi_v = _mm256_loadu_ps(hi_ptr);

                    // new_lo = lo*cos − hi*sin
                    let new_lo = _mm256_fmsub_ps(lo_v, cos_v, _mm256_mul_ps(hi_v, sin_v));
                    // new_hi = hi*cos + lo*sin
                    let new_hi = _mm256_fmadd_ps(lo_v, sin_v, _mm256_mul_ps(hi_v, cos_v));

                    _mm256_storeu_ps(lo_ptr, new_lo);
                    _mm256_storeu_ps(hi_ptr, new_hi);
                }
            }

            let tail_start = chunks * 8;
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
//  F64 variants (4 elements per YMM)
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
    let pairs_simd = half / 2; // 2 pairs = 4 f64 per YMM
    let pairs_tail = half % 2;

    unsafe {
        let sign_mask = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);

        for t in 0..n_tokens {
            let freq_base = t * half;
            for h in 0..n_heads {
                let base = (t * n_heads + h) * head_dim;

                for ci in 0..pairs_simd {
                    let vi = base + ci * 4;
                    let fi = freq_base + ci * 2;

                    let v = _mm256_loadu_pd(qk.as_ptr().add(vi));
                    let v_swap = _mm256_shuffle_pd(v, v, 0x5);

                    // Duplicate cos/sin: [c0,c1] → [c0,c0,c1,c1]
                    let c2 = _mm_loadu_pd(cos.as_ptr().add(fi));
                    let c_lo = _mm_unpacklo_pd(c2, c2);
                    let c_hi = _mm_unpackhi_pd(c2, c2);
                    let cos_dup = _mm256_insertf128_pd(_mm256_castpd128_pd256(c_lo), c_hi, 1);

                    let s2 = _mm_loadu_pd(sin.as_ptr().add(fi));
                    let s_lo = _mm_unpacklo_pd(s2, s2);
                    let s_hi = _mm_unpackhi_pd(s2, s2);
                    let sin_dup = _mm256_insertf128_pd(_mm256_castpd128_pd256(s_lo), s_hi, 1);

                    let sin_neg = _mm256_xor_pd(sin_dup, sign_mask);
                    let result = _mm256_fmadd_pd(v, cos_dup, _mm256_mul_pd(v_swap, sin_neg));
                    _mm256_storeu_pd(qk.as_mut_ptr().add(vi), result);
                }

                let tail_start = pairs_simd * 2;
                for i in tail_start..tail_start + pairs_tail {
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
    let chunks = half / 4;
    let tail = half % 4;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;

            unsafe {
                for ci in 0..chunks {
                    let off = ci * 4;
                    let lo_ptr = qk.as_mut_ptr().add(base + off);
                    let hi_ptr = qk.as_mut_ptr().add(base + half + off);
                    let cos_v = _mm256_loadu_pd(cos.as_ptr().add(freq_base + off));
                    let sin_v = _mm256_loadu_pd(sin.as_ptr().add(freq_base + off));

                    let lo_v = _mm256_loadu_pd(lo_ptr);
                    let hi_v = _mm256_loadu_pd(hi_ptr);

                    _mm256_storeu_pd(
                        lo_ptr,
                        _mm256_fmsub_pd(lo_v, cos_v, _mm256_mul_pd(hi_v, sin_v)),
                    );
                    _mm256_storeu_pd(
                        hi_ptr,
                        _mm256_fmadd_pd(lo_v, sin_v, _mm256_mul_pd(hi_v, cos_v)),
                    );
                }
            }

            let tail_start = chunks * 4;
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
    let pairs_simd = half / 4;
    let pairs_tail = half % 4;

    unsafe {
        let sign_mask = _mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

        for t in 0..n_tokens {
            let freq_base = t * half;
            for h in 0..n_heads {
                let base_bytes = (t * n_heads + h) * head_dim * 2;

                for ci in 0..pairs_simd {
                    let vi_bytes = base_bytes + ci * 16; // 8 f16 = 16 bytes
                    let fi = freq_base + ci * 4;

                    // Load 8 f16 → 8 f32
                    let bits = _mm_loadu_si128(qk.as_ptr().add(vi_bytes) as *const __m128i);
                    let v = _mm256_cvtph_ps(bits);
                    let v_swap = _mm256_shuffle_ps(v, v, 0xB1);

                    let c4 = _mm_loadu_ps(cos.as_ptr().add(fi));
                    let cos_dup = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm_unpacklo_ps(c4, c4)),
                        _mm_unpackhi_ps(c4, c4),
                        1,
                    );

                    let s4 = _mm_loadu_ps(sin.as_ptr().add(fi));
                    let sin_dup = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm_unpacklo_ps(s4, s4)),
                        _mm_unpackhi_ps(s4, s4),
                        1,
                    );
                    let sin_neg = _mm256_xor_ps(sin_dup, sign_mask);

                    let result = _mm256_fmadd_ps(v, cos_dup, _mm256_mul_ps(v_swap, sin_neg));

                    // Store 8 f32 → 8 f16
                    let out_bits = _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
                    _mm_storeu_si128(qk.as_mut_ptr().add(vi_bytes) as *mut __m128i, out_bits);
                }

                // Scalar tail
                let tail_start = pairs_simd * 4;
                let head_bytes = base_bytes;
                for i in tail_start..tail_start + pairs_tail {
                    let bo = head_bytes + i * 4;
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
    let chunks = half / 8;
    let tail = half % 8;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base_bytes = (t * n_heads + h) * head_dim * 2;

            unsafe {
                for ci in 0..chunks {
                    let off = ci * 8;
                    let lo_bptr = qk.as_mut_ptr().add(base_bytes + off * 2);
                    let hi_bptr = qk.as_mut_ptr().add(base_bytes + half * 2 + off * 2);

                    let lo_v = _mm256_cvtph_ps(_mm_loadu_si128(lo_bptr as *const __m128i));
                    let hi_v = _mm256_cvtph_ps(_mm_loadu_si128(hi_bptr as *const __m128i));
                    let cos_v = _mm256_loadu_ps(cos.as_ptr().add(freq_base + off));
                    let sin_v = _mm256_loadu_ps(sin.as_ptr().add(freq_base + off));

                    let new_lo = _mm256_fmsub_ps(lo_v, cos_v, _mm256_mul_ps(hi_v, sin_v));
                    let new_hi = _mm256_fmadd_ps(lo_v, sin_v, _mm256_mul_ps(hi_v, cos_v));

                    _mm_storeu_si128(
                        lo_bptr as *mut __m128i,
                        _mm256_cvtps_ph(new_lo, _MM_FROUND_TO_NEAREST_INT),
                    );
                    _mm_storeu_si128(
                        hi_bptr as *mut __m128i,
                        _mm256_cvtps_ph(new_hi, _MM_FROUND_TO_NEAREST_INT),
                    );
                }
            }

            let tail_start = chunks * 8;
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

#[inline(always)]
unsafe fn load_bf16x8(ptr: *const u8) -> __m256 {
    unsafe {
        let bits16 = _mm_loadu_si128(ptr as *const __m128i);
        let wide = _mm256_cvtepu16_epi32(bits16);
        _mm256_castsi256_ps(_mm256_slli_epi32(wide, 16))
    }
}

#[inline(always)]
unsafe fn store_bf16x8(ptr: *mut u8, v: __m256) {
    unsafe {
        let i32s = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
        let lo128 = _mm256_castsi256_si128(i32s);
        let hi128 = _mm256_extracti128_si256(i32s, 1);
        _mm_storeu_si128(ptr as *mut __m128i, _mm_packus_epi32(lo128, hi128));
    }
}

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
    let pairs_simd = half / 4;
    let pairs_tail = half % 4;

    unsafe {
        let sign_mask = _mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

        for t in 0..n_tokens {
            let freq_base = t * half;
            for h in 0..n_heads {
                let base_bytes = (t * n_heads + h) * head_dim * 2;

                for ci in 0..pairs_simd {
                    let vi_bytes = base_bytes + ci * 16;
                    let fi = freq_base + ci * 4;

                    let v = load_bf16x8(qk.as_ptr().add(vi_bytes));
                    let v_swap = _mm256_shuffle_ps(v, v, 0xB1);

                    let c4 = _mm_loadu_ps(cos.as_ptr().add(fi));
                    let cos_dup = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm_unpacklo_ps(c4, c4)),
                        _mm_unpackhi_ps(c4, c4),
                        1,
                    );

                    let s4 = _mm_loadu_ps(sin.as_ptr().add(fi));
                    let sin_dup = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm_unpacklo_ps(s4, s4)),
                        _mm_unpackhi_ps(s4, s4),
                        1,
                    );
                    let sin_neg = _mm256_xor_ps(sin_dup, sign_mask);

                    let result = _mm256_fmadd_ps(v, cos_dup, _mm256_mul_ps(v_swap, sin_neg));
                    store_bf16x8(qk.as_mut_ptr().add(vi_bytes), result);
                }

                let tail_start = pairs_simd * 4;
                let head_bytes = base_bytes;
                for i in tail_start..tail_start + pairs_tail {
                    let bo = head_bytes + i * 4;
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
    let chunks = half / 8;
    let tail = half % 8;

    for t in 0..n_tokens {
        let freq_base = t * half;
        for h in 0..n_heads {
            let base_bytes = (t * n_heads + h) * head_dim * 2;

            unsafe {
                for ci in 0..chunks {
                    let off = ci * 8;
                    let lo_bptr = qk.as_mut_ptr().add(base_bytes + off * 2);
                    let hi_bptr = qk.as_mut_ptr().add(base_bytes + half * 2 + off * 2);

                    let lo_v = load_bf16x8(lo_bptr as *const u8);
                    let hi_v = load_bf16x8(hi_bptr as *const u8);
                    let cos_v = _mm256_loadu_ps(cos.as_ptr().add(freq_base + off));
                    let sin_v = _mm256_loadu_ps(sin.as_ptr().add(freq_base + off));

                    let new_lo = _mm256_fmsub_ps(lo_v, cos_v, _mm256_mul_ps(hi_v, sin_v));
                    let new_hi = _mm256_fmadd_ps(lo_v, sin_v, _mm256_mul_ps(hi_v, cos_v));

                    store_bf16x8(lo_bptr, new_lo);
                    store_bf16x8(hi_bptr, new_hi);
                }
            }

            let tail_start = chunks * 8;
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
