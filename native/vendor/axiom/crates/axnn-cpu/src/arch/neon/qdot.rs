//! Quantized dot-product and GEMM kernels — ARM NEON (AArch64).
//!
//! Uses `vdotq_s32` (FEAT_DotProd) for the integer inner loops.
//! Both `vec_dot` and `gemm` functions match the signatures of the
//! scalar and AVX2 tiers exactly.
//!
//! ## Inner-loop strategy
//!
//! - **Q8 x Q8**: 2× `vdotq_s32` per 32-element block.  Each call
//!   processes 16 i8×i8 → i32 products in a single instruction.
//! - **Q4 x Q8**: Nibble extraction via `vandq_u8` + `vshrq_n_u8`,
//!   re-interleaved with `vzip1q_u8` / `vzip2q_u8`.  Nibbles are
//!   unsigned [0..15] but fit in signed i8, so `vdotq_s32` works
//!   directly.  Min offset accumulated via `vpaddlq_s8`.
//! - **Q6 x Q8**: Low-4 + high-2 bit combination, centered by -32.
//!   Values in [-32..31] are already signed i8, so `vdotq_s32`
//!   applies directly.

use core::arch::aarch64::*;

use crate::cast::f16_to_f32;
use crate::quant::{
    Q4_BLOCK_BYTES, Q4_BLOCK_SIZE, Q4_D_OFFSET, Q4_DMIN_OFFSET, Q4_QS_OFFSET, Q4_SC_OFFSET,
    Q4_SUB_BLOCK, Q6_BLOCK_BYTES, Q6_BLOCK_SIZE, Q6_D_OFFSET, Q6_QH_OFFSET, Q6_QL_OFFSET,
    Q6_SC_OFFSET, Q8_BLOCK_BYTES, Q8_BLOCK_SIZE, Q8_QS_OFFSET, unpack_q4_scales,
};

// ── Helpers ──────────────────────────────────────────────────────────

/// Read an f16 scale from a byte slice at the given offset.
#[inline(always)]
fn read_f16(src: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
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
        // 2 independent f32x4 accumulators for ILP — the OoO engine
        // can overlap FMA chains from even/odd blocks.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        let xp = x.as_ptr();
        let yp = y.as_ptr();

        // 2-block unroll: compute integer dot products for both blocks
        // before any float conversion, so fcvt latency is hidden behind
        // the second block's integer work.
        let pairs = nb / 2;
        for p in 0..pairs {
            let ib0 = p * 2;
            let ib1 = ib0 + 1;

            let xoff0 = ib0 * Q8_BLOCK_BYTES;
            let xoff1 = ib1 * Q8_BLOCK_BYTES;
            let yoff0 = ib0 * Q8_BLOCK_BYTES;
            let yoff1 = ib1 * Q8_BLOCK_BYTES;

            let xq0 = xp.add(xoff0 + Q8_QS_OFFSET) as *const i8;
            let yq0 = yp.add(yoff0 + Q8_QS_OFFSET) as *const i8;
            let xq1 = xp.add(xoff1 + Q8_QS_OFFSET) as *const i8;
            let yq1 = yp.add(yoff1 + Q8_QS_OFFSET) as *const i8;

            // Integer dot products — all 4 vdotq before any float work.
            let mut dot0 = vdupq_n_s32(0);
            dot0 = vdotq_s32(dot0, vld1q_s8(xq0), vld1q_s8(yq0));
            dot0 = vdotq_s32(dot0, vld1q_s8(xq0.add(16)), vld1q_s8(yq0.add(16)));

            let mut dot1 = vdupq_n_s32(0);
            dot1 = vdotq_s32(dot1, vld1q_s8(xq1), vld1q_s8(yq1));
            dot1 = vdotq_s32(dot1, vld1q_s8(xq1.add(16)), vld1q_s8(yq1.add(16)));

            // Scales — scalar lane form avoids dup instruction.
            let d0 = read_f16(x, xoff0) * read_f16(y, yoff0);
            let d1 = read_f16(x, xoff1) * read_f16(y, yoff1);

            // Float convert + FMA into independent accumulators.
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(dot0), d0);
            acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(dot1), d1);
        }

        // Handle odd trailing block.
        if nb & 1 != 0 {
            let ib = nb - 1;
            let xoff = ib * Q8_BLOCK_BYTES;
            let yoff = ib * Q8_BLOCK_BYTES;
            let xq = xp.add(xoff + Q8_QS_OFFSET) as *const i8;
            let yq = yp.add(yoff + Q8_QS_OFFSET) as *const i8;

            let mut dot = vdupq_n_s32(0);
            dot = vdotq_s32(dot, vld1q_s8(xq), vld1q_s8(yq));
            dot = vdotq_s32(dot, vld1q_s8(xq.add(16)), vld1q_s8(yq.add(16)));

            let d = read_f16(x, xoff) * read_f16(y, yoff);
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(dot), d);
        }

        vaddvq_f32(vaddq_f32(acc0, acc1))
    }
}

// ── Vec dot: Q4_K_M x Q8_0 ──────────────────────────────────────────

/// Extract 32 sequential nibble values from 16 packed bytes.
///
/// Input: 16 bytes where byte[i] = (elem[2i] & 0xF) | (elem[2i+1] << 4).
/// Output: (first_16, second_16) as int8x16_t with values in [0..15].
#[inline(always)]
unsafe fn extract_nibbles_16x2(ptr: *const u8) -> (int8x16_t, int8x16_t) {
    unsafe {
        let packed = vld1q_u8(ptr);
        let lo_nib = vandq_u8(packed, vdupq_n_u8(0x0F));
        let hi_nib = vshrq_n_u8(packed, 4);

        // Interleave to sequential order: elem[0], elem[1], elem[2], ...
        let first_16 = vzip1q_u8(lo_nib, hi_nib);
        let second_16 = vzip2q_u8(lo_nib, hi_nib);

        (
            vreinterpretq_s8_u8(first_16),
            vreinterpretq_s8_u8(second_16),
        )
    }
}

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
        let mut sumf = 0.0f32;

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

                let q4_ptr = q4.as_ptr().add(Q4_QS_OFFSET + sb * Q4_SUB_BLOCK / 2);
                let q8_ptr = q8.as_ptr().add(Q8_QS_OFFSET) as *const i8;

                // Extract 32 nibble values from 16 packed bytes.
                let (nib_lo, nib_hi) = extract_nibbles_16x2(q4_ptr);

                // Load 32 Q8 values as 2 x 16xi8.
                let q8v_lo = vld1q_s8(q8_ptr);
                let q8v_hi = vld1q_s8(q8_ptr.add(16));

                // nibble × q8 dot product via vdotq_s32
                // (nibbles are [0..15], safe as signed i8).
                let mut acc = vdupq_n_s32(0);
                acc = vdotq_s32(acc, nib_lo, q8v_lo);
                acc = vdotq_s32(acc, nib_hi, q8v_hi);
                let sumi = vaddvq_s32(acc);

                // Sum of Q8 values for the min offset.
                // vpaddlq_s8: pairwise add 16xi8 -> 8xi16
                // vpadalq_s16: pairwise accumulate 8xi16 -> 4xi32
                let q8_sum_lo = vpaddlq_s8(q8v_lo);
                let q8_sum_hi = vpaddlq_s8(q8v_hi);
                let q8_sum32 = vpadalq_s16(vpaddlq_s16(q8_sum_lo), q8_sum_hi);
                let sum_q8 = vaddvq_s32(q8_sum32);

                sumf += q8d
                    * (d * scales[sb] as f32 * sumi as f32
                        - dmin * mins[sb] as f32 * sum_q8 as f32);
            }
        }

        sumf
    }
}

// ── Vec dot: Q6_K x Q8_0 ────────────────────────────────────────────

/// Unpack 32 contiguous 6-bit values from a Q6_K block and center by -32.
///
/// Returns (first_16, second_16) as int8x16_t with values in [-32..31].
#[inline(always)]
unsafe fn unpack_q6x32_neon(q6: &[u8], elem_base: usize) -> (int8x16_t, int8x16_t) {
    unsafe {
        let ql_off = Q6_QL_OFFSET + elem_base / 2;
        let qh_off = Q6_QH_OFFSET + elem_base / 4;

        // Low 4 bits: 16 packed bytes -> 32 sequential nibbles.
        let ql_raw = vld1q_u8(q6.as_ptr().add(ql_off));
        let ql_even = vandq_u8(ql_raw, vdupq_n_u8(0x0F));
        let ql_odd = vshrq_n_u8(ql_raw, 4);
        let ql_first = vzip1q_u8(ql_even, ql_odd);
        let ql_second = vzip2q_u8(ql_even, ql_odd);

        // High 2 bits: 8 bytes → 32 × 2-bit values shifted to bits [5:4].
        // Each qh byte holds 4 elements at bit positions [1:0],[3:2],[5:4],[7:6].
        //
        // Strategy: load 8 bytes into low half of a vector, replicate each
        // byte 4× via vtbl, then mask-and-shift each 2-bit position.
        let qh_raw = vld1_u8(q6.as_ptr().add(qh_off)); // uint8x8_t (8 bytes)

        // Replicate each byte 4×: byte[k] → positions [4k, 4k+1, 4k+2, 4k+3]
        let expand_idx_lo: [u8; 16] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
        let expand_idx_hi: [u8; 16] = [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7];
        let qh_src = vcombine_u8(qh_raw, qh_raw);
        let qh_rep_lo = vqtbl1q_u8(qh_src, vld1q_u8(expand_idx_lo.as_ptr()));
        let qh_rep_hi = vqtbl1q_u8(qh_src, vld1q_u8(expand_idx_hi.as_ptr()));

        // Within each replicated group, extract the correct 2-bit field:
        //   position 0 in group: shift right 0, mask 0x03, shift left 4
        //   position 1 in group: shift right 2, mask 0x03, shift left 4
        //   position 2 in group: shift right 4, mask 0x03, shift left 4
        //   position 3 in group: shift right 6, mask 0x03, shift left 4
        // Combined: (byte >> [0,2,4,6]) & 0x03, then << 4
        let shift_pattern: [i8; 16] = [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6];
        let shifts = vld1q_s8(shift_pattern.as_ptr());
        let neg_shifts = vnegq_s8(shifts);
        let m03 = vdupq_n_u8(0x03);

        let qh_first = vshlq_n_u8(
            vandq_u8(
                vshlq_u8(
                    qh_rep_lo,
                    vreinterpretq_s8_u8(vreinterpretq_u8_s8(neg_shifts)),
                ),
                m03,
            ),
            4,
        );
        let qh_second = vshlq_n_u8(
            vandq_u8(
                vshlq_u8(
                    qh_rep_hi,
                    vreinterpretq_s8_u8(vreinterpretq_u8_s8(neg_shifts)),
                ),
                m03,
            ),
            4,
        );

        // Combine: q6_unsigned = ql_nibble | (qh_2bits << 4), range [0..63].
        let q6u_first = vorrq_u8(ql_first, qh_first);
        let q6u_second = vorrq_u8(ql_second, qh_second);

        // Center: q6_signed = q6_unsigned - 32, range [-32..31].
        let offset = vdupq_n_s8(32);
        let q6s_first = vsubq_s8(vreinterpretq_s8_u8(q6u_first), offset);
        let q6s_second = vsubq_s8(vreinterpretq_s8_u8(q6u_second), offset);

        (q6s_first, q6s_second)
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
        let mut sumf = 0.0f32;

        for ib in 0..nb {
            let q6 = &x[ib * Q6_BLOCK_BYTES..][..Q6_BLOCK_BYTES];
            let d6 = read_f16(q6, Q6_D_OFFSET);

            // 8 Q8 blocks per Q6 block (256 / 32 = 8).
            for q8b in 0..8 {
                let q8_idx = ib * 8 + q8b;
                let q8 = &y[q8_idx * Q8_BLOCK_BYTES..][..Q8_BLOCK_BYTES];
                let q8d = read_f16(q8, 0);

                let elem_base = q8b * 32;

                // Unpack 32 centered 6-bit values.
                let (q6s_lo, q6s_hi) = unpack_q6x32_neon(q6, elem_base);

                let q8_ptr = q8.as_ptr().add(Q8_QS_OFFSET) as *const i8;
                let q8v_lo = vld1q_s8(q8_ptr);
                let q8v_hi = vld1q_s8(q8_ptr.add(16));

                // Each Q8 block spans 2 Q6 sub-blocks of 16 elements.
                // Sub-block 0: elements [0..16), sub-block 1: elements [16..32).
                let sc0 = q6[Q6_SC_OFFSET + q8b * 2] as i8;
                let sc1 = q6[Q6_SC_OFFSET + q8b * 2 + 1] as i8;

                // Dot product for first 16 elements (sub-block 0).
                let sumi0 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q6s_lo, q8v_lo));

                // Dot product for second 16 elements (sub-block 1).
                let sumi1 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q6s_hi, q8v_hi));

                sumf += d6 * q8d * (sc0 as f32 * sumi0 as f32 + sc1 as f32 * sumi1 as f32);
            }
        }

        sumf
    }
}

// ── GEMM: Quantized weight x Q8 activation -> F32 output ────────────
//
// Standard inference layout:
//   W: [m, k] in Qx format     (weight matrix, rows are output features)
//   X: [n, k] in Q8_0 format   (activation matrix, rows are batch items)
//   C: [m, n] in f32            (output, C += W x X^T)
//
// C[i, j] = dot(W[i, :], X[j, :])
//
// Delegates to vec_dot for the inner product.  The NEON vec_dot already
// handles the SIMD acceleration; GEMM-level tiling can be added later.

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
