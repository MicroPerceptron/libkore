//! Quantized dot-product and GEMM kernels — AVX2.
//!
//! Key intrinsics:
//!   - `_mm256_maddubs_epi16`: 32 × (u8 × i8) → 16 × i16
//!   - `_mm256_madd_epi16`:    16 × i16 → 8 × i32
//! Signed Q8 uses split-sign dot products to cover the full i8 domain.
//!   - `_mm256_fmadd_ps`:      fused multiply-add for scale accumulation
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

use crate::quant::{
    Q4_BLOCK_BYTES, Q4_BLOCK_SIZE, Q4_D_OFFSET, Q4_DMIN_OFFSET, Q4_QS_OFFSET, Q4_SC_OFFSET,
    Q6_BLOCK_BYTES, Q6_BLOCK_SIZE, Q6_D_OFFSET, Q6_QH_OFFSET, Q6_QL_OFFSET, Q6_SC_OFFSET,
    Q8_BLOCK_BYTES, Q8_BLOCK_SIZE, Q8_QS_OFFSET, unpack_q4_scales,
};
#[cfg(feature = "alloc")]
use alloc::vec;

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

// ── Hardware F16 → F32 (single value) ─────────────────────────────────
//
// Replaces the software `crate::cast::f16_to_f32` (6 shifts + 3 branches)
// with a single `vcvtph2ps` instruction via F16C. ~3 cycles vs ~10.
// Called per-block for scale reads — the hottest scalar path in Q4/Q6.

#[inline(always)]
unsafe fn f16_to_f32_hw(bits: u16) -> f32 {
    unsafe {
        // Load u16 into the lowest 16 bits of an XMM register.
        let v = _mm_cvtsi32_si128(bits as i32);
        // vcvtph2ps: convert 4 f16 → 4 f32 (only lane 0 valid).
        _mm_cvtss_f32(_mm_cvtph_ps(v))
    }
}

/// Read an f16 value from a byte slice using hardware F16C conversion.
#[inline(always)]
unsafe fn read_f16_hw(src: &[u8], off: usize) -> f32 {
    unsafe {
        let bits = u16::from_le_bytes([*src.as_ptr().add(off), *src.as_ptr().add(off + 1)]);
        f16_to_f32_hw(bits)
    }
}

// ── Vec dot: Q8_0 × Q8_0 ────────────────────────────────────────────

pub(crate) fn vec_dot_q8_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q8_BLOCK_SIZE, 0);
    let nb = n / Q8_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q8_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * Q8_BLOCK_BYTES);

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for ib in 0..nb {
            let xb = &x[ib * Q8_BLOCK_BYTES..];
            let yb = &y[ib * Q8_BLOCK_BYTES..];

            let xd = read_f16_hw(xb, 0);
            let yd = read_f16_hw(yb, 0);
            let d = _mm256_set1_ps(xd * yd);

            let qx = _mm256_loadu_si256(xb.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);
            let qy = _mm256_loadu_si256(yb.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);

            // Split-sign dot products → 8×i32, convert to 8×f32,
            // then FMA-accumulate across all blocks. Single hsum at end.
            let lo = _mm256_and_si256(qx, _mm256_set1_epi8(127));
            let hi = _mm256_and_si256(qx, _mm256_set1_epi8(-128));
            let positive = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(lo, qy));
            let negative = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(hi, qy));
            let exact = _mm256_sub_epi32(positive, negative);
            let sum32 = exact;

            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sum32), acc);
        }

        hsum_f32x8(acc)
    }
}

// ── Vec dot: Q4_K_M × Q8_0 ──────────────────────────────────────────
//
// val_j = d * scale[sb] * nibble_j − dmin * min[sb]
//
// Our packing stores consecutive pairs: qs[k] = elem[2k] | (elem[2k+1] << 4).
// After nibble extraction, low/high nibbles give even/odd indexed elements.
// We re-interleave to sequential order before pairing with Q8 blocks.
//
// Min offset is accumulated in SIMD lanes (no horizontal reduction in the
// inner loop). The q8 sum stays as 8×i32, multiplied by broadcast scalars,
// and only reduced at the very end.

pub(crate) fn vec_dot_q4_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q4_BLOCK_SIZE, 0);
    let nb = n / Q4_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q4_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q4_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let ones_16 = _mm256_set1_epi16(1);
        let ones_u8 = _mm256_set1_epi8(1);

        let mut acc = _mm256_setzero_ps();
        let mut acc_min = _mm256_setzero_ps();

        for ib in 0..nb {
            let q4 = &x[ib * Q4_BLOCK_BYTES..][..Q4_BLOCK_BYTES];
            let d = read_f16_hw(q4, Q4_D_OFFSET);
            let dmin = read_f16_hw(q4, Q4_DMIN_OFFSET);
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

                let q8d_lo = read_f16_hw(y, q8_lo_off);
                let q8d_hi = read_f16_hw(y, q8_hi_off);

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

                // nibble × q8 → i32 via maddubs + madd.
                let p32_lo = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(first_32, q8v_lo));
                let p32_hi = _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(second_32, q8v_hi));

                // Scale: pre-multiplied d_sc[sb] × q8d (1 mul instead of 2).
                let vd_lo = _mm256_set1_ps(d_sc[sb_lo] * q8d_lo);
                let vd_hi = _mm256_set1_ps(d_sc[sb_hi] * q8d_hi);
                acc = _mm256_fmadd_ps(vd_lo, _mm256_cvtepi32_ps(p32_lo), acc);
                acc = _mm256_fmadd_ps(vd_hi, _mm256_cvtepi32_ps(p32_hi), acc);

                // Min offset: dm_mn[sb] × q8d × Σ(q8).
                let q8sum_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, q8v_lo), ones_16);
                let q8sum_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, q8v_hi), ones_16);

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
// val = d6 * scale[sub_block] * q6_centered.
//
// Offset-32 trick: maddubs(q6_u, q8) − maddubs(32, q8).

pub(crate) fn vec_dot_q6_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q6_BLOCK_SIZE, 0);
    let nb = n / Q6_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q6_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q6_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    unsafe {
        let m32s = _mm256_set1_epi8(32);

        let mut acc = _mm256_setzero_ps();

        for ib in 0..nb {
            let q6 = &x[ib * Q6_BLOCK_BYTES..][..Q6_BLOCK_BYTES];
            let d6 = read_f16_hw(q6, Q6_D_OFFSET);

            for q8b in 0..8 {
                let q8_off = (ib * 8 + q8b) * Q8_BLOCK_BYTES;
                let q8d = read_f16_hw(y, q8_off);

                let elem_base = q8b * 32;

                let q6_unsigned = unpack_q6x32(q6, elem_base);

                let q8v =
                    _mm256_loadu_si256(y.as_ptr().add(q8_off + Q8_QS_OFFSET) as *const __m256i);

                // Offset-32: maddubs(q6_u, q8) − maddubs(32, q8)
                let p16 = _mm256_sub_epi16(
                    _mm256_maddubs_epi16(q6_unsigned, q8v),
                    _mm256_maddubs_epi16(m32s, q8v),
                );

                // Sub-block scales: elements 0..15 → scale[q8b*2],
                //                   elements 16..31 → scale[q8b*2+1].
                let sc0 = q6[Q6_SC_OFFSET + q8b * 2] as i8 as i16;
                let sc1 = q6[Q6_SC_OFFSET + q8b * 2 + 1] as i8 as i16;
                let sc_vec = _mm256_set_epi16(
                    sc1, sc1, sc1, sc1, sc1, sc1, sc1, sc1, sc0, sc0, sc0, sc0, sc0, sc0, sc0, sc0,
                );
                let p32 = _mm256_madd_epi16(sc_vec, p16);

                let vd = _mm256_set1_ps(d6 * q8d);
                acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32), acc);
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
///
/// The scalar qh loop compiles to tight pipelineable code; SIMD alternatives
/// (blendv, pshufb+interleave) caused register pressure that degraded the
/// entire compilation unit. The f16→f32 hardware fix matters far more.
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

// ── GEMM: Q × Q8 → F32 (K-tiled) ───────────────────────────────────
//
// Weight metadata (f16 scales, packed sub-block scales/mins, Q6 high-bit
// extraction) is decoded once per weight row, then reused across all N
// activation columns.  This eliminates N−1 redundant header decodes per
// K-block — the dominant overhead for batched inference (N>1).

/// Max super-blocks per row (Q4/Q6: 256-elem blocks, supports k ≤ 16384).
const MAX_NB_256: usize = 64;
/// Max Q8 blocks per row (32-elem blocks, supports k ≤ 16384).
const MAX_NB_32: usize = 512;

pub(crate) fn gemm_q8_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q8_BLOCK_SIZE, 0);
    let nb = k / Q8_BLOCK_SIZE;
    let row_bytes = nb * Q8_BLOCK_BYTES;

    // Small-N fast path: vec_dot per (i,j) avoids pre-decode overhead.
    if n <= 4 {
        for i in 0..m {
            let w_row = &w[i * row_bytes..][..row_bytes];
            for j in 0..n {
                let x_row = &x[j * row_bytes..][..row_bytes];
                c[i * n + j] += vec_dot_q8_q8(w_row, x_row, k);
            }
        }
        return;
    }

    debug_assert!(nb <= MAX_NB_32);

    unsafe {
        for i in 0..m {
            let w_row = w.as_ptr().add(i * row_bytes);

            // Pre-decode weight scales for all K-blocks.
            let mut wd = [0.0f32; MAX_NB_32];
            for ib in 0..nb {
                wd[ib] = read_f16_hw(
                    core::slice::from_raw_parts(w_row.add(ib * Q8_BLOCK_BYTES), Q8_BLOCK_BYTES),
                    0,
                );
            }

            for j in 0..n {
                let x_row = x.as_ptr().add(j * row_bytes);
                let mut acc = _mm256_setzero_ps();

                for ib in 0..nb {
                    let wb = w_row.add(ib * Q8_BLOCK_BYTES);
                    let xb = x_row.add(ib * Q8_BLOCK_BYTES);

                    let xd = {
                        let bits = u16::from_le_bytes([*xb, *xb.add(1)]);
                        f16_to_f32_hw(bits)
                    };
                    let d = _mm256_set1_ps(wd[ib] * xd);

                    let qw = _mm256_loadu_si256(wb.add(Q8_QS_OFFSET) as *const __m256i);
                    let qx = _mm256_loadu_si256(xb.add(Q8_QS_OFFSET) as *const __m256i);

                    let lo = _mm256_and_si256(qw, _mm256_set1_epi8(127));
                    let hi = _mm256_and_si256(qw, _mm256_set1_epi8(-128));
                    let positive =
                        _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(lo, qx));
                    let negative =
                        _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(hi, qx));
                    let exact = _mm256_sub_epi32(positive, negative);
                    let sum32 = exact;

                    acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sum32), acc);
                }

                c[i * n + j] += hsum_f32x8(acc);
            }
        }
    }
}

#[cfg(feature = "alloc")]
pub(crate) fn gemm_q4_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q4_BLOCK_SIZE, 0);
    let nb = k / Q4_BLOCK_SIZE;
    let w_row_bytes = nb * Q4_BLOCK_BYTES;
    let x_row_bytes = nb * 8 * Q8_BLOCK_BYTES;

    // Small-N fast path: vec_dot per (i,j) avoids pre-decode overhead.
    if n <= 4 {
        for i in 0..m {
            let w_row = &w[i * w_row_bytes..][..w_row_bytes];
            for j in 0..n {
                let x_row = &x[j * x_row_bytes..][..x_row_bytes];
                c[i * n + j] += vec_dot_q4_q8(w_row, x_row, k);
            }
        }
        return;
    }

    debug_assert!(nb <= MAX_NB_256);

    // Total Q8 blocks per activation row.
    let nb8 = nb * 8;

    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let ones_16 = _mm256_set1_epi16(1);
        let ones_u8 = _mm256_set1_epi8(1);

        // Pre-compute (q8_scale × byte_sum) for every Q8 block across all
        // activation rows.  This lets us replace the entire SIMD min-offset
        // path (2×maddubs + 2×madd + 2×cvt + 2×fmadd per jj, 4 jj per ib)
        // with 8 scalar multiply-adds per (ib, j) — and frees registers for
        // the pre-extracted nibble arrays.
        let mut q8d_bsum = vec![0.0f32; n * nb8];
        for j in 0..n {
            for idx in 0..nb8 {
                let q8_ptr = x.as_ptr().add(j * x_row_bytes + idx * Q8_BLOCK_BYTES);
                let q8d = f16_to_f32_hw(u16::from_le_bytes([*q8_ptr, *q8_ptr.add(1)]));
                let q8v = _mm256_loadu_si256(q8_ptr.add(Q8_QS_OFFSET) as *const __m256i);
                let bsum = hsum_i32x8(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(ones_u8, q8v),
                    ones_16,
                ));
                q8d_bsum[j * nb8 + idx] = q8d * bsum as f32;
            }
        }

        for i in 0..m {
            let w_row = w.as_ptr().add(i * w_row_bytes);

            // Pre-decode all super-block headers for this weight row.
            let mut d_sc = [[0.0f32; 8]; MAX_NB_256];
            let mut dm_mn = [[0.0f32; 8]; MAX_NB_256];
            for ib in 0..nb {
                let q4 =
                    core::slice::from_raw_parts(w_row.add(ib * Q4_BLOCK_BYTES), Q4_BLOCK_BYTES);
                let d = read_f16_hw(q4, Q4_D_OFFSET);
                let dmin = read_f16_hw(q4, Q4_DMIN_OFFSET);
                let (sc, mn) = unpack_q4_scales(&q4[Q4_SC_OFFSET..Q4_SC_OFFSET + 12]);
                for sb in 0..8 {
                    d_sc[ib][sb] = d * sc[sb] as f32;
                    dm_mn[ib][sb] = dmin * mn[sb] as f32;
                }
            }

            // K-tiled (i, ib, j) ordering: pre-extract Q4 nibbles once per
            // super-block, then reuse across all N activation columns.
            // With the min path moved to scalar, the jj loop only needs:
            //   2 pre-extracted (first_32[jj], second_32[jj]) + 1 acc
            //   + 2 q8v + 2 p32 + 2 vd + 2 constants = 11 live regs.
            // The remaining 6 pre-extracted values fit in the 5 spare YMMs
            // with at most 1 spill — far better than the 3+ spills without
            // the bsum optimisation.
            for ib in 0..nb {
                let q4_qs = w_row.add(ib * Q4_BLOCK_BYTES + Q4_QS_OFFSET);

                // Pre-extract interleaved nibble pairs for all 4 sub-block
                // pairs.
                let mut first_32 = [_mm256_setzero_si256(); 4];
                let mut second_32 = [_mm256_setzero_si256(); 4];
                for jj in 0..4 {
                    let q4bits = _mm256_loadu_si256(q4_qs.add(jj * 32) as *const __m256i);
                    let q4_even = _mm256_and_si256(q4bits, m4);
                    let q4_odd = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                    let inter_lo = _mm256_unpacklo_epi8(q4_even, q4_odd);
                    let inter_hi = _mm256_unpackhi_epi8(q4_even, q4_odd);
                    first_32[jj] = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x20);
                    second_32[jj] = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x31);
                }

                for j in 0..n {
                    let q8_base = x.as_ptr().add(j * x_row_bytes + ib * 8 * Q8_BLOCK_BYTES);
                    let mut block_acc = _mm256_setzero_ps();

                    for jj in 0..4 {
                        let sb_lo = jj * 2;
                        let sb_hi = jj * 2 + 1;

                        let q8_lo = q8_base.add(sb_lo * Q8_BLOCK_BYTES);
                        let q8_hi = q8_base.add(sb_hi * Q8_BLOCK_BYTES);

                        let q8d_lo = {
                            let bits = u16::from_le_bytes([*q8_lo, *q8_lo.add(1)]);
                            f16_to_f32_hw(bits)
                        };
                        let q8d_hi = {
                            let bits = u16::from_le_bytes([*q8_hi, *q8_hi.add(1)]);
                            f16_to_f32_hw(bits)
                        };

                        let q8v_lo = _mm256_loadu_si256(q8_lo.add(Q8_QS_OFFSET) as *const __m256i);
                        let q8v_hi = _mm256_loadu_si256(q8_hi.add(Q8_QS_OFFSET) as *const __m256i);

                        // maddubs + madd → 8×i32 using pre-extracted nibbles.
                        let p32_lo =
                            _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(first_32[jj], q8v_lo));
                        let p32_hi =
                            _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(second_32[jj], q8v_hi));

                        // Scale (pre-multiplied d×scale).
                        let vd_lo = _mm256_set1_ps(d_sc[ib][sb_lo] * q8d_lo);
                        let vd_hi = _mm256_set1_ps(d_sc[ib][sb_hi] * q8d_hi);
                        block_acc = _mm256_fmadd_ps(vd_lo, _mm256_cvtepi32_ps(p32_lo), block_acc);
                        block_acc = _mm256_fmadd_ps(vd_hi, _mm256_cvtepi32_ps(p32_hi), block_acc);
                    }

                    // Scalar min offset using pre-computed (q8d × bsum).
                    let db = &q8d_bsum[j * nb8 + ib * 8..];
                    let dm = &dm_mn[ib];
                    let mut min_val = 0.0f32;
                    for sb in 0..8 {
                        min_val += dm[sb] * db[sb];
                    }

                    c[i * n + j] += hsum_f32x8(block_acc) - min_val;
                }
            }
        }
    }
}

#[cfg(not(feature = "alloc"))]
pub(crate) fn gemm_q4_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    crate::arch::portable::qdot::gemm_q4_q8(w, x, c, m, n, k)
}

pub(crate) fn gemm_q6_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q6_BLOCK_SIZE, 0);
    let nb = k / Q6_BLOCK_SIZE;
    let w_row_bytes = nb * Q6_BLOCK_BYTES;
    let x_row_bytes_q6 = nb * 8 * Q8_BLOCK_BYTES;

    // Small-N fast path: vec_dot per (i,j) avoids pre-unpack overhead.
    if n <= 4 {
        for i in 0..m {
            let w_row = &w[i * w_row_bytes..][..w_row_bytes];
            for j in 0..n {
                let x_row = &x[j * x_row_bytes_q6..][..x_row_bytes_q6];
                c[i * n + j] += vec_dot_q6_q8(w_row, x_row, k);
            }
        }
        return;
    }

    debug_assert!(nb <= MAX_NB_256);

    unsafe {
        let m32s = _mm256_set1_epi8(32);

        for i in 0..m {
            let w_row = &w[i * w_row_bytes..];

            // For Q6 we iterate (i, ib, j): the expensive unpack_q6x32 depends
            // only on weight data, so we pre-unpack all 8 sub-blocks per
            // super-block and reuse them across all N activation columns.
            for ib in 0..nb {
                let q6 = &w_row[ib * Q6_BLOCK_BYTES..][..Q6_BLOCK_BYTES];
                let d6 = read_f16_hw(q6, Q6_D_OFFSET);

                // Pre-unpack all 8 × 32 Q6 unsigned values.
                let mut q6u = [_mm256_setzero_si256(); 8];
                let mut sc = [0i16; 16];
                for q8b in 0..8 {
                    q6u[q8b] = unpack_q6x32(q6, q8b * 32);
                    sc[q8b * 2] = q6[Q6_SC_OFFSET + q8b * 2] as i8 as i16;
                    sc[q8b * 2 + 1] = q6[Q6_SC_OFFSET + q8b * 2 + 1] as i8 as i16;
                }

                for j in 0..n {
                    let q8_seg = x.as_ptr().add(j * x_row_bytes_q6 + ib * 8 * Q8_BLOCK_BYTES);
                    let mut block_acc = _mm256_setzero_ps();

                    for q8b in 0..8 {
                        let q8_ptr = q8_seg.add(q8b * Q8_BLOCK_BYTES);
                        let q8d = {
                            let bits = u16::from_le_bytes([*q8_ptr, *q8_ptr.add(1)]);
                            f16_to_f32_hw(bits)
                        };

                        let q8v = _mm256_loadu_si256(q8_ptr.add(Q8_QS_OFFSET) as *const __m256i);

                        // Offset-32: maddubs(q6_u, q8) − maddubs(32, q8).
                        let p16 = _mm256_sub_epi16(
                            _mm256_maddubs_epi16(q6u[q8b], q8v),
                            _mm256_maddubs_epi16(m32s, q8v),
                        );

                        let sc_vec = _mm256_set_epi16(
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                        );
                        let p32 = _mm256_madd_epi16(sc_vec, p16);

                        let vd = _mm256_set1_ps(d6 * q8d);
                        block_acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32), block_acc);
                    }

                    c[i * n + j] += hsum_f32x8(block_acc);
                }
            }
        }
    }
}
