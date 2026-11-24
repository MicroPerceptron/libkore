//! Quantized dot-product and GEMM kernels — AVX-VNNI.
//!
//! Key intrinsic:
//!   - `_mm256_dpbusd_epi32`: fused u8×i8 → i32 dot-product accumulation
//!     (VEX-encoded `vpdpbusd`, no AVX-512 required).
//!
//! Replaces the `maddubs + madd` two-instruction chain from AVX2 with a
//! single `dpbusd`. Signed Q8 splits the sign bit and subtracts two dot
//! products, preserving the full i8 domain including -128.

use core::arch::x86_64::*;

use crate::cast::f16_to_f32;
use crate::quant::{
    Q4_BLOCK_BYTES, Q4_BLOCK_SIZE, Q4_D_OFFSET, Q4_DMIN_OFFSET, Q4_QS_OFFSET, Q4_SC_OFFSET,
    Q6_BLOCK_BYTES, Q6_BLOCK_SIZE, Q6_D_OFFSET, Q6_QH_OFFSET, Q6_QL_OFFSET, Q6_SC_OFFSET,
    Q8_BLOCK_BYTES, Q8_BLOCK_SIZE, Q8_QS_OFFSET, unpack_q4_scales,
};
// ── Horizontal reduction ──────────────────────────────────────────────

#[inline(always)]
unsafe fn hsum_f32x8(x: __m256) -> f32 {
    unsafe {
        let hi = _mm256_extractf128_ps(x, 1);
        let lo = _mm256_castps256_ps128(x);
        let s = _mm_add_ps(hi, lo);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }
}

#[inline(always)]
unsafe fn hsum_i32x8(x: __m256i) -> i32 {
    unsafe {
        let hi = _mm256_extracti128_si256(x, 1);
        let lo = _mm256_castsi256_si128(x);
        let s = _mm_add_epi32(hi, lo);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        _mm_cvtsi128_si32(s)
    }
}

// ── Vec dot: Q8_0 × Q8_0 ────────────────────────────────────────────

#[target_feature(enable = "avx2,avxvnni")]
unsafe fn vec_dot_q8_q8_inner(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q8_BLOCK_SIZE, 0);
    let nb = n / Q8_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q8_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * Q8_BLOCK_BYTES);

    unsafe {
        let mut sumf = 0.0f32;

        for ib in 0..nb {
            let xb = &x[ib * Q8_BLOCK_BYTES..];
            let yb = &y[ib * Q8_BLOCK_BYTES..];

            let xd = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
            let yd = f16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));

            // VNNI dot products consume unsigned by signed bytes.
            let qx = _mm256_loadu_si256(xb.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);
            let qy = _mm256_loadu_si256(yb.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);
            // Split the sign bit rather than negating signed bytes. This is
            // exact for -128 too: signed(x) = (x & 127) - (x & 128).
            let lo = _mm256_and_si256(qx, _mm256_set1_epi8(127));
            let hi = _mm256_and_si256(qx, _mm256_set1_epi8(-128));
            let positive = _mm256_dpbusd_epi32(_mm256_setzero_si256(), lo, qy);
            let negative = _mm256_dpbusd_epi32(_mm256_setzero_si256(), hi, qy);
            let sum = _mm256_sub_epi32(positive, negative);
            let sumi = hsum_i32x8(sum);
            sumf += xd * yd * sumi as f32;
        }

        sumf
    }
}

// ── Vec dot: Q4_K_M × Q8_0 ──────────────────────────────────────────
//
// Nibble extraction and interleaving identical to AVX2.  The maddubs+madd
// chain is replaced by a single dpbusd.  Min-offset byte sums also use
// dpbusd(zero, ones_u8, q8v) which sums groups of 4 bytes.

#[target_feature(enable = "avx2,avxvnni")]
unsafe fn vec_dot_q4_q8_inner(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q4_BLOCK_SIZE, 0);
    let nb = n / Q4_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q4_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q4_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let ones_u8 = _mm256_set1_epi8(1);

        let mut acc = _mm256_setzero_ps();
        let mut acc_min = _mm256_setzero_ps();

        for ib in 0..nb {
            let q4 = &x[ib * Q4_BLOCK_BYTES..][..Q4_BLOCK_BYTES];
            let d = f16_to_f32(u16::from_le_bytes([q4[Q4_D_OFFSET], q4[Q4_D_OFFSET + 1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([
                q4[Q4_DMIN_OFFSET],
                q4[Q4_DMIN_OFFSET + 1],
            ]));
            let (scales, mins) = unpack_q4_scales(&q4[Q4_SC_OFFSET..Q4_SC_OFFSET + 12]);

            // Pre-multiply d×scale and dmin×min once per super-block.
            let mut d_sc = [0.0f32; 8];
            let mut dm_mn = [0.0f32; 8];
            for sb in 0..8 {
                d_sc[sb] = d * scales[sb] as f32;
                dm_mn[sb] = dmin * mins[sb] as f32;
            }

            // 4 iterations: each loads 32 packed bytes covering 64 elements,
            // then re-interleaves to pair correctly with 2 Q8 blocks.
            for j in 0..4 {
                let sb_lo = j * 2;
                let sb_hi = j * 2 + 1;

                let q8_lo_off = (ib * 8 + sb_lo) * Q8_BLOCK_BYTES;
                let q8_hi_off = (ib * 8 + sb_hi) * Q8_BLOCK_BYTES;

                let q8d_lo = f16_to_f32(u16::from_le_bytes([y[q8_lo_off], y[q8_lo_off + 1]]));
                let q8d_hi = f16_to_f32(u16::from_le_bytes([y[q8_hi_off], y[q8_hi_off + 1]]));

                // Extract nibbles from 32 packed bytes.
                let q4bits =
                    _mm256_loadu_si256(q4.as_ptr().add(Q4_QS_OFFSET + j * 32) as *const __m256i);
                let q4_even = _mm256_and_si256(q4bits, m4);
                let q4_odd = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                // Re-interleave to sequential element order.
                let inter_lo = _mm256_unpacklo_epi8(q4_even, q4_odd);
                let inter_hi = _mm256_unpackhi_epi8(q4_even, q4_odd);

                // Cross-lane permute to get contiguous 32-element groups.
                let first_32 = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x20);
                let second_32 = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x31);

                // Load Q8 values.
                let q8v_lo =
                    _mm256_loadu_si256(y.as_ptr().add(q8_lo_off + Q8_QS_OFFSET) as *const __m256i);
                let q8v_hi =
                    _mm256_loadu_si256(y.as_ptr().add(q8_hi_off + Q8_QS_OFFSET) as *const __m256i);

                // VNNI: single dpbusd replaces maddubs+madd.
                // Q4 nibbles are unsigned [0..15], Q8 values are signed i8 —
                // dpbusd(src, unsigned, signed) is the correct operand order.
                let p32_lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(), first_32, q8v_lo);
                let p32_hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), second_32, q8v_hi);

                // Scale: pre-multiplied d_sc[sb] × q8d.
                let vd_lo = _mm256_set1_ps(d_sc[sb_lo] * q8d_lo);
                let vd_hi = _mm256_set1_ps(d_sc[sb_hi] * q8d_hi);
                acc = _mm256_fmadd_ps(vd_lo, _mm256_cvtepi32_ps(p32_lo), acc);
                acc = _mm256_fmadd_ps(vd_hi, _mm256_cvtepi32_ps(p32_hi), acc);

                // Min offset: dpbusd(zero, ones_u8, q8v) sums groups of 4
                // signed bytes (treating ones as unsigned 1s), giving the
                // byte sum as 8×i32.
                let q8sum_lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones_u8, q8v_lo);
                let q8sum_hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones_u8, q8v_hi);

                let md_lo = _mm256_set1_ps(dm_mn[sb_lo] * q8d_lo);
                let md_hi = _mm256_set1_ps(dm_mn[sb_hi] * q8d_hi);
                acc_min = _mm256_fmadd_ps(md_lo, _mm256_cvtepi32_ps(q8sum_lo), acc_min);
                acc_min = _mm256_fmadd_ps(md_hi, _mm256_cvtepi32_ps(q8sum_hi), acc_min);
            }
        }

        hsum_f32x8(acc) - hsum_f32x8(acc_min)
    }
}

// ── Vec dot: Q6_K × Q8_0 ────────────────────────────────────────────
//
// 6-bit K-quant: q6_unsigned = (ql_low4 | (qh_2bits << 4)), range [0..63].
// q6_centered = q6_unsigned − 32.
//
// Offset-32 trick with VNNI:
//   dpbusd(zero, q6_unsigned, q8) − dpbusd(zero, const_32, q8)
// Each dpbusd replaces the maddubs+madd pair from AVX2.

#[target_feature(enable = "avx2,avxvnni")]
unsafe fn vec_dot_q6_q8_inner(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q6_BLOCK_SIZE, 0);
    let nb = n / Q6_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q6_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q6_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    unsafe {
        let m32u = _mm256_set1_epi8(32);

        let mut acc = _mm256_setzero_ps();

        for ib in 0..nb {
            let q6 = &x[ib * Q6_BLOCK_BYTES..][..Q6_BLOCK_BYTES];
            let d6 = f16_to_f32(u16::from_le_bytes([q6[Q6_D_OFFSET], q6[Q6_D_OFFSET + 1]]));

            for q8b in 0..8 {
                let q8_off = (ib * 8 + q8b) * Q8_BLOCK_BYTES;
                let q8d = f16_to_f32(u16::from_le_bytes([y[q8_off], y[q8_off + 1]]));

                let elem_base = q8b * 32;

                let q6_unsigned = unpack_q6x32(q6, elem_base);

                let q8v =
                    _mm256_loadu_si256(y.as_ptr().add(q8_off + Q8_QS_OFFSET) as *const __m256i);

                // Offset-32: dpbusd(zero, q6_u, q8) − dpbusd(zero, 32, q8)
                let dot_q6 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q6_unsigned, q8v);
                let dot_32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), m32u, q8v);
                let p32 = _mm256_sub_epi32(dot_q6, dot_32);

                // Sub-block scales: elements 0..15 → scale[q8b*2],
                //                   elements 16..31 → scale[q8b*2+1].
                let sc0 = q6[Q6_SC_OFFSET + q8b * 2] as i8 as f32;
                let sc1 = q6[Q6_SC_OFFSET + q8b * 2 + 1] as i8 as f32;

                // Split the 8 i32 lanes: low 4 use sc0, high 4 use sc1.
                let p32_lo = _mm256_castsi256_si128(p32);
                let p32_hi = _mm256_extracti128_si256(p32, 1);

                let vd0 = _mm_set1_ps(d6 * q8d * sc0);
                let vd1 = _mm_set1_ps(d6 * q8d * sc1);

                let acc_lo = _mm256_castps256_ps128(acc);
                let acc_hi = _mm256_extractf128_ps(acc, 1);

                let new_lo = _mm_fmadd_ps(vd0, _mm_cvtepi32_ps(p32_lo), acc_lo);
                let new_hi = _mm_fmadd_ps(vd1, _mm_cvtepi32_ps(p32_hi), acc_hi);

                acc = _mm256_insertf128_ps(_mm256_castps128_ps256(new_lo), new_hi, 1);
            }
        }

        hsum_f32x8(acc)
    }
}

/// Unpack 32 contiguous 6-bit values from a Q6_K block into a YMM register.
///
/// Element layout in our Q6_K format:
///   ql[j/2]: low 4 bits, nibble selected by j%2
///   qh[j/4]: high 2 bits, bit-pair selected by j%4
///
/// ql extraction: SIMD mask + shift + unpacklo/hi interleave (XMM).
/// qh extraction: scalar loop (8 bytes → 32 bytes, stack-free via setr).
#[inline(always)]
unsafe fn unpack_q6x32(q6: &[u8], elem_base: usize) -> __m256i {
    unsafe {
        let ql_off = Q6_QL_OFFSET + elem_base / 2;
        let qh_off = Q6_QH_OFFSET + elem_base / 4;

        // ql: 16 packed bytes → 32 sequential 4-bit values via interleave.
        let ql_raw = _mm_loadu_si128(q6.as_ptr().add(ql_off) as *const __m128i);
        let m4_128 = _mm_set1_epi8(0x0F);
        let ql_even = _mm_and_si128(ql_raw, m4_128);
        let ql_odd = _mm_and_si128(_mm_srli_epi16(ql_raw, 4), m4_128);

        let ql_lo = _mm_unpacklo_epi8(ql_even, ql_odd);
        let ql_hi = _mm_unpackhi_epi8(ql_even, ql_odd);
        let ql_vec = _mm256_insertf128_si256(_mm256_castsi128_si256(ql_lo), ql_hi, 1);

        // qh: 8 bytes → 32 × 2-bit values shifted to [5:4].
        let qh_ptr = q6.as_ptr().add(qh_off);
        let mut qh_bytes = [0u8; 32];
        for k in 0..8 {
            let b = *qh_ptr.add(k);
            qh_bytes[k * 4] = (b & 3) << 4;
            qh_bytes[k * 4 + 1] = ((b >> 2) & 3) << 4;
            qh_bytes[k * 4 + 2] = ((b >> 4) & 3) << 4;
            qh_bytes[k * 4 + 3] = ((b >> 6) & 3) << 4;
        }
        let qh_vec = _mm256_loadu_si256(qh_bytes.as_ptr() as *const __m256i);

        _mm256_or_si256(ql_vec, qh_vec)
    }
}

// Safe crate-internal entry points. This module is selected only when
// AVX2+AVX-VNNI is part of the target contract (auto-detected target
// features or the explicit `avx_vnni` build feature).

pub(crate) fn vec_dot_q8_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    // SAFETY: module selection establishes the target-feature contract.
    unsafe { vec_dot_q8_q8_inner(x, y, n) }
}

pub(crate) fn vec_dot_q4_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    // SAFETY: module selection establishes the target-feature contract.
    unsafe { vec_dot_q4_q8_inner(x, y, n) }
}

pub(crate) fn vec_dot_q6_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    // SAFETY: module selection establishes the target-feature contract.
    unsafe { vec_dot_q6_q8_inner(x, y, n) }
}

// ── GEMM: Q × Q8 → F32 ─────────────────────────────────────────────
//
// Simple delegation to vec_dot.  The VNNI speedup comes from the fused
// dpbusd in the inner dot-product; the GEMM structure is identical to
// the AVX2 tier.

pub(crate) fn gemm_q8_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q8_BLOCK_SIZE, 0);
    let nb = k / Q8_BLOCK_SIZE;
    let row_bytes = nb * Q8_BLOCK_BYTES;

    for i in 0..m {
        let w_row = &w[i * row_bytes..][..row_bytes];
        for j in 0..n {
            let x_row = &x[j * row_bytes..][..row_bytes];
            c[i * n + j] += vec_dot_q8_q8(w_row, x_row, k);
        }
    }
}

pub(crate) fn gemm_q4_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q4_BLOCK_SIZE, 0);
    let nb = k / Q4_BLOCK_SIZE;
    let w_row_bytes = nb * Q4_BLOCK_BYTES;
    let x_row_bytes = nb * 8 * Q8_BLOCK_BYTES;

    for i in 0..m {
        let w_row = &w[i * w_row_bytes..][..w_row_bytes];
        for j in 0..n {
            let x_row = &x[j * x_row_bytes..][..x_row_bytes];
            c[i * n + j] += vec_dot_q4_q8(w_row, x_row, k);
        }
    }
}

pub(crate) fn gemm_q6_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q6_BLOCK_SIZE, 0);
    let nb = k / Q6_BLOCK_SIZE;
    let w_row_bytes = nb * Q6_BLOCK_BYTES;
    let x_row_bytes = nb * 8 * Q8_BLOCK_BYTES;

    for i in 0..m {
        let w_row = &w[i * w_row_bytes..][..w_row_bytes];
        for j in 0..n {
            let x_row = &x[j * x_row_bytes..][..x_row_bytes];
            c[i * n + j] += vec_dot_q6_q8(w_row, x_row, k);
        }
    }
}
