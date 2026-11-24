//! Quantized dot-product and GEMM kernels — portable fallback.
//!
//! The integer inner products (i8×i8 → i32) use scalar loops that
//! LLVM auto-vectorizes well. The f32 accumulation across blocks
//! uses explicit `Simd<f32, W>` to defer horizontal reduction —
//! same technique as the NEON/AVX2 tiers.
//!
//! Q4/Q6 bit-field extraction (nibbles, ql/qh) stays scalar because
//! `core::simd` lacks the widening multiply-accumulate instructions
//! needed to beat LLVM's output. The arch-specific tiers (avx2, neon)
//! override with `maddubs`/`vdotq` when enabled.

use core::simd::num::SimdFloat;
use core::simd::prelude::*;

use crate::cast::f16_to_f32;
use crate::quant::{
    Q4_BLOCK_BYTES, Q4_BLOCK_SIZE, Q4_D_OFFSET, Q4_DMIN_OFFSET, Q4_QS_OFFSET, Q4_SC_OFFSET,
    Q4_SUB_BLOCK, Q6_BLOCK_BYTES, Q6_BLOCK_SIZE, Q6_D_OFFSET, Q6_QH_OFFSET, Q6_QL_OFFSET,
    Q6_SC_OFFSET, Q8_BLOCK_BYTES, Q8_BLOCK_SIZE, Q8_QS_OFFSET, unpack_q4_scales,
};

use super::W;

// ── Vec dot: Q8_0 × Q8_0 ────────────────────────────────────────────

pub(crate) fn vec_dot_q8_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q8_BLOCK_SIZE, 0);
    let nb = n / Q8_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q8_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * Q8_BLOCK_BYTES);

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let xb = &x[ib * Q8_BLOCK_BYTES..][..Q8_BLOCK_BYTES];
        let yb = &y[ib * Q8_BLOCK_BYTES..][..Q8_BLOCK_BYTES];

        let xd = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let yd = f16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));

        let mut sumi = 0i32;
        for j in 0..Q8_BLOCK_SIZE {
            sumi += (xb[Q8_QS_OFFSET + j] as i8 as i32) * (yb[Q8_QS_OFFSET + j] as i8 as i32);
        }

        sumf += xd * yd * sumi as f32;
    }

    sumf
}

// ── Vec dot: Q4_K_M × Q8_0 ──────────────────────────────────────────

pub(crate) fn vec_dot_q4_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q4_BLOCK_SIZE, 0);
    let nb = n / Q4_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q4_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q4_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let q4 = &x[ib * Q4_BLOCK_BYTES..][..Q4_BLOCK_BYTES];

        let d = f16_to_f32(u16::from_le_bytes([q4[Q4_D_OFFSET], q4[Q4_D_OFFSET + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([
            q4[Q4_DMIN_OFFSET],
            q4[Q4_DMIN_OFFSET + 1],
        ]));
        let (scales, mins) = unpack_q4_scales(&q4[Q4_SC_OFFSET..Q4_SC_OFFSET + 12]);

        for sb in 0..8 {
            let q8_idx = ib * 8 + sb;
            let q8 = &y[q8_idx * Q8_BLOCK_BYTES..][..Q8_BLOCK_BYTES];
            let q8d = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));

            let mut sumi = 0i32;
            let mut sum_q8 = 0i32;

            for j in 0..Q4_SUB_BLOCK {
                let elem = sb * Q4_SUB_BLOCK + j;
                let qs_byte = q4[Q4_QS_OFFSET + elem / 2];
                let nibble = ((qs_byte >> (4 * (elem % 2))) & 0x0F) as i32;

                let q8val = q8[Q8_QS_OFFSET + j] as i8 as i32;

                sumi += nibble * q8val;
                sum_q8 += q8val;
            }

            sumf += q8d
                * (d * scales[sb] as f32 * sumi as f32 - dmin * mins[sb] as f32 * sum_q8 as f32);
        }
    }

    sumf
}

// ── Vec dot: Q6_K × Q8_0 ────────────────────────────────────────────

pub(crate) fn vec_dot_q6_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q6_BLOCK_SIZE, 0);
    let nb = n / Q6_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q6_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q6_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let q6 = &x[ib * Q6_BLOCK_BYTES..][..Q6_BLOCK_BYTES];
        let d6 = f16_to_f32(u16::from_le_bytes([q6[Q6_D_OFFSET], q6[Q6_D_OFFSET + 1]]));

        for q8b in 0..8 {
            let q8_idx = ib * 8 + q8b;
            let q8 = &y[q8_idx * Q8_BLOCK_BYTES..][..Q8_BLOCK_BYTES];
            let q8d = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));

            for half in 0..2u8 {
                let sc_idx = q8b * 2 + half as usize;
                let sc = q6[Q6_SC_OFFSET + sc_idx] as i8;

                let mut sumi = 0i32;

                for j in 0..16 {
                    let elem = q8b * 32 + half as usize * 16 + j;

                    let ql_byte = q6[Q6_QL_OFFSET + elem / 2];
                    let low4 = (ql_byte >> (4 * (elem % 2))) & 0x0F;
                    let qh_byte = q6[Q6_QH_OFFSET + elem / 4];
                    let high2 = (qh_byte >> (2 * (elem % 4))) & 0x03;
                    let q6_val = (low4 | (high2 << 4)) as i32 - 32;

                    let q8_val = q8[Q8_QS_OFFSET + half as usize * 16 + j] as i8 as i32;

                    sumi += q6_val * q8_val;
                }

                sumf += d6 * sc as f32 * q8d * sumi as f32;
            }
        }
    }

    sumf
}

// ── GEMM ─────────────────────────────────────────────────────────────

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
