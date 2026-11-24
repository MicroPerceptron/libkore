//! Quantized dot-product and GEMM kernels — SSE4.2 (SSSE3).
//!
//! 128-bit counterpart to the AVX2 tier.  Key intrinsics:
//!   - `_mm_maddubs_epi16`: 16 x (u8 x i8) -> 8 x i16    (SSSE3)
//!   - `_mm_madd_epi16`:    8 x i16 -> 4 x i32            (SSE2)
//! Signed Q8 uses split-sign dot products to cover the full i8 domain.
//!
//! Processes 16 i8 elements per iteration (vs 32 for AVX2).

use core::arch::x86_64::*;

use crate::cast::f16_to_f32;
use crate::quant::{
    Q4_BLOCK_BYTES, Q4_BLOCK_SIZE, Q4_D_OFFSET, Q4_DMIN_OFFSET, Q4_QS_OFFSET, Q4_SC_OFFSET,
    Q6_BLOCK_BYTES, Q6_BLOCK_SIZE, Q6_D_OFFSET, Q6_QH_OFFSET, Q6_QL_OFFSET, Q6_SC_OFFSET,
    Q8_BLOCK_BYTES, Q8_BLOCK_SIZE, Q8_QS_OFFSET, unpack_q4_scales,
};

// ── Helpers ──────────────────────────────────────────────────────────

/// Read an f16 scale from a byte slice at the given offset.
#[inline(always)]
fn read_f16(src: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
}

/// Horizontal sum of 4 x f32 lanes.
#[inline(always)]
unsafe fn hsum_f32x4(v: __m128) -> f32 {
    unsafe {
        let hi64 = _mm_movehl_ps(v, v);
        let sum64 = _mm_add_ps(v, hi64);
        let hi32 = _mm_movehdup_ps(sum64);
        _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
    }
}

/// Horizontal sum of 4 x i32 lanes.
#[inline(always)]
unsafe fn hsum_i32x4(v: __m128i) -> i32 {
    unsafe {
        let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        _mm_cvtsi128_si32(s)
    }
}

// ── Vec dot: Q8_0 x Q8_0 ────────────────────────────────────────────

/// Inner product of two Q8_0-quantized vectors.
///
/// `n` must be divisible by 32 (Q8 block size).
/// Both `x` and `y` are packed Q8_0 blocks.
pub(crate) fn vec_dot_q8_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q8_BLOCK_SIZE, 0);
    let nb = n / Q8_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q8_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * Q8_BLOCK_BYTES);

    unsafe {
        let mut acc = _mm_setzero_ps();

        for ib in 0..nb {
            let xb = &x[ib * Q8_BLOCK_BYTES..];
            let yb = &y[ib * Q8_BLOCK_BYTES..];

            let xd = read_f16(xb, 0);
            let yd = read_f16(yb, 0);
            let d = _mm_set1_ps(xd * yd);

            let xq = xb.as_ptr().add(Q8_QS_OFFSET);
            let yq = yb.as_ptr().add(Q8_QS_OFFSET);

            // Process 32 bytes in 2 iterations of 16.
            let mut sum = _mm_setzero_si128();
            for j in 0..2 {
                let off = j * 16;
                let qx = _mm_loadu_si128(xq.add(off) as *const __m128i);
                let qy = _mm_loadu_si128(yq.add(off) as *const __m128i);
                // signed(x) = (x & 127) - (x & 128). Each pair fits i16;
                // widen before subtracting so two -128*-128 products fit.
                let lo = _mm_and_si128(qx, _mm_set1_epi8(127));
                let hi = _mm_and_si128(qx, _mm_set1_epi8(-128));
                let positive = _mm_madd_epi16(_mm_set1_epi16(1), _mm_maddubs_epi16(lo, qy));
                let negative = _mm_madd_epi16(_mm_set1_epi16(1), _mm_maddubs_epi16(hi, qy));
                let exact = _mm_sub_epi32(positive, negative);
                sum = _mm_add_epi32(sum, exact);
            }

            acc = _mm_add_ps(acc, _mm_mul_ps(d, _mm_cvtepi32_ps(sum)));
        }

        hsum_f32x4(acc)
    }
}

// ── Vec dot: Q4_K_M x Q8_0 ──────────────────────────────────────────

/// Inner product of Q4_K_M weights x Q8_0 activations.
///
/// `n` must be divisible by 256 (Q4 block size).
/// `x` is packed Q4_K_M blocks, `y` is packed Q8_0 blocks.
pub(crate) fn vec_dot_q4_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q4_BLOCK_SIZE, 0);
    let nb = n / Q4_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q4_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q4_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);
        let ones_16 = _mm_set1_epi16(1);
        let ones_u8 = _mm_set1_epi8(1);

        let mut acc = _mm_setzero_ps();
        let mut acc_min = _mm_setzero_ps();

        for ib in 0..nb {
            let q4 = &x[ib * Q4_BLOCK_BYTES..][..Q4_BLOCK_BYTES];
            let d = read_f16(q4, Q4_D_OFFSET);
            let dmin = read_f16(q4, Q4_DMIN_OFFSET);
            let (scales, mins) = unpack_q4_scales(&q4[Q4_SC_OFFSET..Q4_SC_OFFSET + 12]);

            // 8 sub-blocks of 32 elements, each pairing with one Q8 block.
            for sb in 0..8 {
                let q8_idx = ib * 8 + sb;
                let q8 = &y[q8_idx * Q8_BLOCK_BYTES..][..Q8_BLOCK_BYTES];
                let q8d = read_f16(q8, 0);

                let q4_ptr = q4.as_ptr().add(Q4_QS_OFFSET + sb * 16);
                let q8_ptr = q8.as_ptr().add(Q8_QS_OFFSET);

                // Load 16 packed bytes containing 32 nibbles.
                let bits = _mm_loadu_si128(q4_ptr as *const __m128i);
                let lo_nib = _mm_and_si128(bits, mask_0f);
                let hi_nib = _mm_and_si128(_mm_srli_epi16(bits, 4), mask_0f);

                // Interleave to sequential element order.
                let first_16 = _mm_unpacklo_epi8(lo_nib, hi_nib);
                let second_16 = _mm_unpackhi_epi8(lo_nib, hi_nib);

                // Load 32 Q8 values as 2 x 16.
                let q8v_lo = _mm_loadu_si128(q8_ptr as *const __m128i);
                let q8v_hi = _mm_loadu_si128(q8_ptr.add(16) as *const __m128i);

                // nibble x q8 -> i32 via maddubs + madd.
                let p32_lo = _mm_madd_epi16(ones_16, _mm_maddubs_epi16(first_16, q8v_lo));
                let p32_hi = _mm_madd_epi16(ones_16, _mm_maddubs_epi16(second_16, q8v_hi));
                let p32 = _mm_add_epi32(p32_lo, p32_hi);

                let vd = _mm_set1_ps(d * scales[sb] as f32 * q8d);
                acc = _mm_add_ps(acc, _mm_mul_ps(vd, _mm_cvtepi32_ps(p32)));

                // Min offset: dmin * mins[sb] * q8d * sum(q8).
                let q8sum_lo = _mm_madd_epi16(_mm_maddubs_epi16(ones_u8, q8v_lo), ones_16);
                let q8sum_hi = _mm_madd_epi16(_mm_maddubs_epi16(ones_u8, q8v_hi), ones_16);
                let q8sum = _mm_add_epi32(q8sum_lo, q8sum_hi);
                let md = _mm_set1_ps(dmin * mins[sb] as f32 * q8d);
                acc_min = _mm_add_ps(acc_min, _mm_mul_ps(md, _mm_cvtepi32_ps(q8sum)));
            }
        }

        hsum_f32x4(acc) - hsum_f32x4(acc_min)
    }
}

// ── Vec dot: Q6_K x Q8_0 ────────────────────────────────────────────

/// Unpack 32 contiguous 6-bit values from a Q6_K block into two XMM
/// registers as unsigned bytes [0..63].
///
/// Returns (first_16, second_16) ready for the offset-32 trick.
#[inline(always)]
unsafe fn unpack_q6x32_sse(q6: &[u8], elem_base: usize) -> (__m128i, __m128i) {
    unsafe {
        let ql_off = Q6_QL_OFFSET + elem_base / 2;
        let qh_off = Q6_QH_OFFSET + elem_base / 4;

        // ql: 16 packed bytes -> 32 sequential 4-bit values.
        let ql_raw = _mm_loadu_si128(q6.as_ptr().add(ql_off) as *const __m128i);
        let m4 = _mm_set1_epi8(0x0F);
        let ql_even = _mm_and_si128(ql_raw, m4);
        let ql_odd = _mm_and_si128(_mm_srli_epi16(ql_raw, 4), m4);

        let ql_first = _mm_unpacklo_epi8(ql_even, ql_odd);
        let ql_second = _mm_unpackhi_epi8(ql_even, ql_odd);

        // qh: 8 bytes -> 32 x 2-bit values shifted to bits [5:4].
        let qh_ptr = q6.as_ptr().add(qh_off);
        let mut qh_arr = [0u8; 32];
        for k in 0..8 {
            let b = *qh_ptr.add(k);
            qh_arr[k * 4] = (b & 0x03) << 4;
            qh_arr[k * 4 + 1] = ((b >> 2) & 0x03) << 4;
            qh_arr[k * 4 + 2] = ((b >> 4) & 0x03) << 4;
            qh_arr[k * 4 + 3] = ((b >> 6) & 0x03) << 4;
        }
        let qh_first = _mm_loadu_si128(qh_arr.as_ptr() as *const __m128i);
        let qh_second = _mm_loadu_si128(qh_arr.as_ptr().add(16) as *const __m128i);

        // Combine: q6_unsigned = ql_nibble | (qh_2bits << 4).
        let q6u_first = _mm_or_si128(ql_first, qh_first);
        let q6u_second = _mm_or_si128(ql_second, qh_second);

        (q6u_first, q6u_second)
    }
}

/// Inner product of Q6_K weights x Q8_0 activations.
///
/// `n` must be divisible by 256 (Q6 block size).
/// `x` is packed Q6_K blocks, `y` is packed Q8_0 blocks.
pub(crate) fn vec_dot_q6_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q6_BLOCK_SIZE, 0);
    let nb = n / Q6_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q6_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q6_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    unsafe {
        let m32s = _mm_set1_epi8(32);
        let mut acc = _mm_setzero_ps();

        for ib in 0..nb {
            let q6 = &x[ib * Q6_BLOCK_BYTES..][..Q6_BLOCK_BYTES];
            let d6 = read_f16(q6, Q6_D_OFFSET);

            // 8 Q8 blocks per Q6 block (256 / 32 = 8).
            for q8b in 0..8 {
                let q8_off = (ib * 8 + q8b) * Q8_BLOCK_BYTES;
                let q8 = &y[q8_off..][..Q8_BLOCK_BYTES];
                let q8d = read_f16(q8, 0);

                let elem_base = q8b * 32;

                // Unpack 32 unsigned 6-bit values as two 128-bit vectors.
                let (q6u_lo, q6u_hi) = unpack_q6x32_sse(q6, elem_base);

                let q8_ptr = q8.as_ptr().add(Q8_QS_OFFSET);
                let q8v_lo = _mm_loadu_si128(q8_ptr as *const __m128i);
                let q8v_hi = _mm_loadu_si128(q8_ptr.add(16) as *const __m128i);

                // Offset-32 trick: maddubs(q6_u, q8) - maddubs(32, q8).
                let p16_lo = _mm_sub_epi16(
                    _mm_maddubs_epi16(q6u_lo, q8v_lo),
                    _mm_maddubs_epi16(m32s, q8v_lo),
                );
                let p16_hi = _mm_sub_epi16(
                    _mm_maddubs_epi16(q6u_hi, q8v_hi),
                    _mm_maddubs_epi16(m32s, q8v_hi),
                );

                // Sub-block scales: elements 0..15 -> scale[q8b*2],
                //                   elements 16..31 -> scale[q8b*2+1].
                let sc0 = q6[Q6_SC_OFFSET + q8b * 2] as i8 as i16;
                let sc1 = q6[Q6_SC_OFFSET + q8b * 2 + 1] as i8 as i16;

                let sc_vec_lo = _mm_set1_epi16(sc0);
                let sc_vec_hi = _mm_set1_epi16(sc1);

                let p32_lo = _mm_madd_epi16(sc_vec_lo, p16_lo);
                let p32_hi = _mm_madd_epi16(sc_vec_hi, p16_hi);
                let p32 = _mm_add_epi32(p32_lo, p32_hi);

                let vd = _mm_set1_ps(d6 * q8d);
                acc = _mm_add_ps(acc, _mm_mul_ps(vd, _mm_cvtepi32_ps(p32)));
            }
        }

        hsum_f32x4(acc)
    }
}

// ── GEMM: Quantized weight x Q8 activation -> F32 output ────────────
//
// Simple delegation to vec_dot. The SSE4.2 vec_dot handles the SIMD
// acceleration; GEMM-level tiling can be added later.

/// GEMM: Q8 weights x Q8 activations -> F32 output.
///
/// `C[i,j] += dot(w[i,:], x[j,:])` over `k` elements.
/// `k` must be divisible by 32.
pub(crate) fn gemm_q8_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q8_BLOCK_SIZE, 0);
    let row_bytes = k / Q8_BLOCK_SIZE * Q8_BLOCK_BYTES;

    for i in 0..m {
        let w_row = &w[i * row_bytes..][..row_bytes];
        for j in 0..n {
            let x_row = &x[j * row_bytes..][..row_bytes];
            c[i * n + j] += vec_dot_q8_q8(w_row, x_row, k);
        }
    }
}

/// GEMM: Q4_K_M weights x Q8 activations -> F32 output.
///
/// `C[i,j] += dot(w[i,:], x[j,:])` over `k` elements.
/// `k` must be divisible by 256.
pub(crate) fn gemm_q4_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q4_BLOCK_SIZE, 0);
    let w_row_bytes = k / Q4_BLOCK_SIZE * Q4_BLOCK_BYTES;
    let x_row_bytes = k / Q8_BLOCK_SIZE * Q8_BLOCK_BYTES;

    for i in 0..m {
        let w_row = &w[i * w_row_bytes..][..w_row_bytes];
        for j in 0..n {
            let x_row = &x[j * x_row_bytes..][..x_row_bytes];
            c[i * n + j] += vec_dot_q4_q8(w_row, x_row, k);
        }
    }
}

/// GEMM: Q6_K weights x Q8 activations -> F32 output.
///
/// `C[i,j] += dot(w[i,:], x[j,:])` over `k` elements.
/// `k` must be divisible by 256.
pub(crate) fn gemm_q6_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q6_BLOCK_SIZE, 0);
    let w_row_bytes = k / Q6_BLOCK_SIZE * Q6_BLOCK_BYTES;
    let x_row_bytes = k / Q8_BLOCK_SIZE * Q8_BLOCK_BYTES;

    for i in 0..m {
        let w_row = &w[i * w_row_bytes..][..w_row_bytes];
        for j in 0..n {
            let x_row = &x[j * x_row_bytes..][..x_row_bytes];
            c[i * n + j] += vec_dot_q6_q8(w_row, x_row, k);
        }
    }
}
