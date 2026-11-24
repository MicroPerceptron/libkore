//! Quantized dot-product and GEMM kernels — scalar fallback.
//!
//! ## Hot path for quantized inference
//!
//! Weights stay in their compact format (Q4_K_M, Q6_K); activations
//! are quantized to Q8_0 on-the-fly before computing.  The inner
//! product accumulates in `i32` within each block to avoid float
//! rounding noise, then converts to `f32` once per block when
//! multiplying by the scale factors.
//!
//! ## Block alignment
//!
//! | Weight | Activation | Weight block | Q8 blocks per weight block |
//! |--------|------------|-------------|----------------------------|
//! | Q8_0   | Q8_0       | 32 elem     | 1                          |
//! | Q6_K   | Q8_0       | 256 elem    | 8                          |
//! | Q4_K_M | Q8_0       | 256 elem    | 8                          |
//!
//! ## SIMD upgrade notes
//!
//! Each `for j in 0..N` inner loop maps to a single SIMD instruction
//! or short sequence:
//!
//! - **AVX2**: `vpmaddubsw` (u8×i8 → i16 pairs) + `vpmaddwd` (i16 → i32)
//!   for 32 products in 2 instructions.  Horizontal sum via `vphaddw`.
//! - **AVX-512 VNNI**: `vpdpbusd` fuses the above into one instruction
//!   — 64 i8 products accumulated into 16 i32 lanes per cycle.
//! - **AVX-512BF16**: Not relevant here (float conversion, not int dot).

use crate::cast::f16_to_f32;
use crate::quant::{
    Q4_BLOCK_BYTES, Q4_BLOCK_SIZE, Q4_D_OFFSET, Q4_DMIN_OFFSET, Q4_QS_OFFSET, Q4_SC_OFFSET,
    Q4_SUB_BLOCK, Q6_BLOCK_BYTES, Q6_BLOCK_SIZE, Q6_D_OFFSET, Q6_QH_OFFSET, Q6_QL_OFFSET,
    Q6_SC_OFFSET, Q8_BLOCK_BYTES, Q8_BLOCK_SIZE, Q8_QS_OFFSET, unpack_q4_scales,
};

// ── Vec dot: Q8_0 × Q8_0 ────────────────────────────────────────────
//
// Symmetric × symmetric: both sides have f16 scale + i8[32] quants.
//
//   block_dot = (Σ x_qs[j] * y_qs[j])  ← i32 accumulator
//   sumf += x_d * y_d * block_dot       ← f32

/// Inner product of two Q8_0-quantized vectors.
///
/// `n` must be divisible by 32 (Q8 block size).
/// Both `x` and `y` are packed Q8_0 blocks.
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

        // Integer accumulator — the core SIMD target.
        // AVX2: vpmaddubsw + vpmaddwd on 32 bytes each.
        let mut sumi = 0i32;
        for j in 0..Q8_BLOCK_SIZE {
            sumi += (xb[Q8_QS_OFFSET + j] as i8 as i32) * (yb[Q8_QS_OFFSET + j] as i8 as i32);
        }

        sumf += xd * yd * sumi as f32;
    }

    sumf
}

// ── Vec dot: Q4_K_M × Q8_0 ──────────────────────────────────────────
//
// Asymmetric Q4 weights × symmetric Q8 activations.
//
// Per Q4 block (256 elem = 8 sub-blocks of 32):
//   val_j = d * scale[sb] * nibble_j − dmin * min[sb]
//
// So the dot with Q8 activation y is:
//   Σ val_j * y_j = Σ (d*scale[sb]*nibble_j − dmin*min[sb]) * q8d*q8_j
//                 = q8d * (d*scale[sb] * Σ(nibble_j * q8_j)
//                         − dmin*min[sb] * Σ(q8_j))
//
// The two inner sums (nibble×q8 and plain q8) stay in i32.

/// Inner product of Q4_K_M weights × Q8_0 activations.
///
/// `n` must be divisible by 256 (Q4 block size).
/// `x` is packed Q4_K_M blocks, `y` is packed Q8_0 blocks.
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

        // 8 sub-blocks of 32 elements, each pairing with one Q8 block.
        for sb in 0..8 {
            let q8_idx = ib * 8 + sb;
            let q8 = &y[q8_idx * Q8_BLOCK_BYTES..][..Q8_BLOCK_BYTES];
            let q8d = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));

            // i32 accumulators — SIMD replaces this with vpmaddubsw chain.
            let mut sumi = 0i32; // Σ nibble × q8
            let mut sum_q8 = 0i32; // Σ q8  (for min offset)

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
//
// 6-bit K-quant weights × symmetric Q8 activations.
//
// Per Q6 block (256 elem = 16 sub-blocks of 16):
//   q6_j  = (ql_low4 | (qh_2bits << 4)) − 32
//   val_j = d * scale[j/16] * q6_j
//
// Each Q8 block (32 elem) spans 2 Q6 sub-blocks (16 elem each).
// The i32 accumulator covers one sub-block (16 products).

/// Inner product of Q6_K weights × Q8_0 activations.
///
/// `n` must be divisible by 256 (Q6 block size).
/// `x` is packed Q6_K blocks, `y` is packed Q8_0 blocks.
pub(crate) fn vec_dot_q6_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q6_BLOCK_SIZE, 0);
    let nb = n / Q6_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q6_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q6_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let q6 = &x[ib * Q6_BLOCK_BYTES..][..Q6_BLOCK_BYTES];
        let d6 = f16_to_f32(u16::from_le_bytes([q6[Q6_D_OFFSET], q6[Q6_D_OFFSET + 1]]));

        // 8 Q8 blocks per Q6 block (256 / 32 = 8).
        for q8b in 0..8 {
            let q8_idx = ib * 8 + q8b;
            let q8 = &y[q8_idx * Q8_BLOCK_BYTES..][..Q8_BLOCK_BYTES];
            let q8d = f16_to_f32(u16::from_le_bytes([q8[0], q8[1]]));

            // Each Q8 block spans 2 Q6 sub-blocks (16 elements each).
            for half in 0..2u8 {
                let sc_idx = q8b * 2 + half as usize;
                let sc = q6[Q6_SC_OFFSET + sc_idx] as i8;

                // i32 accumulator — SIMD replaces with vpmaddubsw or vpdpbusd.
                let mut sumi = 0i32;

                for j in 0..16 {
                    let elem = q8b * 32 + half as usize * 16 + j;

                    // Decode 6-bit quantized value.
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

// ── GEMM: Quantized weight × Q8 activation → F32 output ─────────────
//
// Standard inference layout:
//   W: [m, k] in Qx format     (weight matrix, rows are output features)
//   X: [n, k] in Q8_0 format   (activation matrix, rows are batch items)
//   C: [m, n] in f32            (output, C += W × X^T)
//
// C[i, j] = dot(W[i, :], X[j, :])
//
// The scalar version is a simple double loop over vec_dot. SIMD tiers
// can tile and batch multiple output rows for cache efficiency.

/// GEMM: Q8 weights × Q8 activations → F32 output.
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

/// GEMM: Q4_K_M weights × Q8 activations → F32 output.
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

/// GEMM: Q6_K weights × Q8 activations → F32 output.
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

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{quantize_q4, quantize_q6, quantize_q8};
    use alloc::vec;

    /// Compute f32 reference dot product.
    fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut sum = 0.0f64; // f64 for reference accuracy
        for (&av, &bv) in a.iter().zip(b) {
            sum += av as f64 * bv as f64;
        }
        sum as f32
    }

    // ── Q8 × Q8 dot product ─────────────────────────────────────

    #[test]
    fn q8q8_dot_basic() {
        let n = 256;
        let a: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let b: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 * 0.7 - 90.0) * 0.01).collect();

        let nb = n / Q8_BLOCK_SIZE;
        let mut qa = vec![0u8; nb * Q8_BLOCK_BYTES];
        let mut qb = vec![0u8; nb * Q8_BLOCK_BYTES];
        quantize_q8(&a, &mut qa);
        quantize_q8(&b, &mut qb);

        let got = vec_dot_q8_q8(&qa, &qb, n);
        let expected = dot_f32(&a, &b);

        let rel_err = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(
            rel_err < 0.02,
            "q8×q8 dot: got={got}, expected={expected}, rel_err={rel_err}"
        );
    }

    #[test]
    fn q8q8_dot_zeros() {
        let n = 32;
        let a = vec![0.0f32; n];
        let b = vec![1.0f32; n];

        let mut qa = vec![0u8; Q8_BLOCK_BYTES];
        let mut qb = vec![0u8; Q8_BLOCK_BYTES];
        quantize_q8(&a, &mut qa);
        quantize_q8(&b, &mut qb);

        assert_eq!(vec_dot_q8_q8(&qa, &qb, n), 0.0);
    }

    #[test]
    fn q8q8_dot_self() {
        // dot(x, x) >= 0 always.
        let n = 128;
        let a: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.05).collect();

        let nb = n / Q8_BLOCK_SIZE;
        let mut qa = vec![0u8; nb * Q8_BLOCK_BYTES];
        quantize_q8(&a, &mut qa);

        let got = vec_dot_q8_q8(&qa, &qa, n);
        assert!(got > 0.0, "self-dot should be positive, got={got}");
    }

    // ── Q4 × Q8 dot product ─────────────────────────────────────

    #[test]
    fn q4q8_dot_basic() {
        let n = 256;
        let w: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 / 128.0 - 1.0) * 0.5).collect();
        let a: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 38.0) * 0.02).collect();

        let mut qw = vec![0u8; Q4_BLOCK_BYTES];
        let mut qa = vec![0u8; (n / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES];
        quantize_q4(&w, &mut qw);
        quantize_q8(&a, &mut qa);

        let got = vec_dot_q4_q8(&qw, &qa, n);
        let expected = dot_f32(&w, &a);

        let rel_err = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(
            rel_err < 0.15,
            "q4×q8 dot: got={got}, expected={expected}, rel_err={rel_err}"
        );
    }

    #[test]
    fn q4q8_dot_multi_block() {
        let n = 512;
        let w: alloc::vec::Vec<f32> = (0..n).map(|i| i as f32 / 256.0 - 1.0).collect();
        let a: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 / 256.0 - 1.0) * 0.5).collect();

        let nb4 = n / Q4_BLOCK_SIZE;
        let nb8 = n / Q8_BLOCK_SIZE;
        let mut qw = vec![0u8; nb4 * Q4_BLOCK_BYTES];
        let mut qa = vec![0u8; nb8 * Q8_BLOCK_BYTES];
        quantize_q4(&w, &mut qw);
        quantize_q8(&a, &mut qa);

        let got = vec_dot_q4_q8(&qw, &qa, n);
        let expected = dot_f32(&w, &a);

        let rel_err = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(
            rel_err < 0.15,
            "q4×q8 multi: got={got}, expected={expected}, rel_err={rel_err}"
        );
    }

    // ── Q6 × Q8 dot product ─────────────────────────────────────

    #[test]
    fn q6q8_dot_basic() {
        let n = 256;
        let w: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 / 128.0 - 1.0) * 2.0).collect();
        let a: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 * 0.5 - 64.0) * 0.01).collect();

        let mut qw = vec![0u8; Q6_BLOCK_BYTES];
        let mut qa = vec![0u8; (n / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES];
        quantize_q6(&w, &mut qw);
        quantize_q8(&a, &mut qa);

        let got = vec_dot_q6_q8(&qw, &qa, n);
        let expected = dot_f32(&w, &a);

        let rel_err = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(
            rel_err < 0.10,
            "q6×q8 dot: got={got}, expected={expected}, rel_err={rel_err}"
        );
    }

    #[test]
    fn q6q8_dot_multi_block() {
        let n = 512;
        let w: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 / 256.0 - 1.0) * 1.5).collect();
        let a: alloc::vec::Vec<f32> = (0..n).map(|i| (i as f32 / 256.0 - 1.0) * 0.8).collect();

        let nb6 = n / Q6_BLOCK_SIZE;
        let nb8 = n / Q8_BLOCK_SIZE;
        let mut qw = vec![0u8; nb6 * Q6_BLOCK_BYTES];
        let mut qa = vec![0u8; nb8 * Q8_BLOCK_BYTES];
        quantize_q6(&w, &mut qw);
        quantize_q8(&a, &mut qa);

        let got = vec_dot_q6_q8(&qw, &qa, n);
        let expected = dot_f32(&w, &a);

        let rel_err = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(
            rel_err < 0.10,
            "q6×q8 multi: got={got}, expected={expected}, rel_err={rel_err}"
        );
    }

    // ── GEMM correctness ────────────────────────────────────────

    #[test]
    fn gemm_q8q8_identity() {
        // 2×2 identity-ish: W = [[1,0,...], [0,1,...]], X = [[a,...],[b,...]]
        // Test with k=32 (minimum Q8 block).
        let k = 32;
        let m = 2;
        let n = 1;

        // W row 0: [1, 0, 0, ...], W row 1: [0, 1, 0, ...]
        let mut w0 = vec![0.0f32; k];
        let mut w1 = vec![0.0f32; k];
        w0[0] = 1.0;
        w1[1] = 1.0;

        let mut x = vec![0.0f32; k];
        x[0] = 3.0;
        x[1] = 7.0;

        let mut qw = vec![0u8; m * Q8_BLOCK_BYTES];
        let mut qx = vec![0u8; n * Q8_BLOCK_BYTES];
        quantize_q8(&w0, &mut qw[..Q8_BLOCK_BYTES]);
        quantize_q8(&w1, &mut qw[Q8_BLOCK_BYTES..]);
        quantize_q8(&x, &mut qx);

        let mut c = vec![0.0f32; m * n];
        gemm_q8_q8(&qw, &qx, &mut c, m, n, k);

        // The quantization of sparse one-hot vectors should be quite accurate.
        assert!((c[0] - 3.0).abs() < 0.1, "c[0]={}", c[0]);
        assert!((c[1] - 7.0).abs() < 0.1, "c[1]={}", c[1]);
    }

    #[test]
    fn gemm_q4q8_vs_reference() {
        let m = 2; // 2 output features
        let n = 1; // 1 batch
        let k = 256; // inner dim (Q4 block aligned)

        // Use correlated data that produces a sizable dot product
        // (avoid near-cancellation where Q4's coarse quant dominates).
        let w: alloc::vec::Vec<f32> = (0..m * k)
            .map(|i| (i as f32 / (m * k) as f32 - 0.3) * 2.0)
            .collect();
        let x: alloc::vec::Vec<f32> = (0..n * k)
            .map(|i| (i as f32 / k as f32 - 0.5) * 1.5)
            .collect();

        let nb4 = k / Q4_BLOCK_SIZE;
        let nb8 = k / Q8_BLOCK_SIZE;
        let mut qw = vec![0u8; m * nb4 * Q4_BLOCK_BYTES];
        let mut qx = vec![0u8; n * nb8 * Q8_BLOCK_BYTES];

        for row in 0..m {
            quantize_q4(
                &w[row * k..(row + 1) * k],
                &mut qw[row * nb4 * Q4_BLOCK_BYTES..(row + 1) * nb4 * Q4_BLOCK_BYTES],
            );
        }
        quantize_q8(&x, &mut qx);

        let mut c = vec![0.0f32; m * n];
        gemm_q4_q8(&qw, &qx, &mut c, m, n, k);

        // Reference in f32. Use combined abs + rel tolerance:
        // Q4 has 16 levels, so quantization noise is meaningful.
        for i in 0..m {
            let expected = dot_f32(&w[i * k..(i + 1) * k], &x);
            let abs_err = (c[i] - expected).abs();
            let tol = 0.20 * expected.abs() + 0.5; // rel + floor
            assert!(
                abs_err < tol,
                "gemm_q4q8 row {i}: got={}, expected={expected}, abs_err={abs_err}, tol={tol}",
                c[i]
            );
        }
    }

    #[test]
    fn gemm_q6q8_vs_reference() {
        let m = 2;
        let n = 1;
        let k = 256;

        let w: alloc::vec::Vec<f32> = (0..m * k).map(|i| ((i % 19) as f32 - 9.0) * 0.08).collect();
        let x: alloc::vec::Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32 - 5.0) * 0.04).collect();

        let nb6 = k / Q6_BLOCK_SIZE;
        let nb8 = k / Q8_BLOCK_SIZE;
        let mut qw = vec![0u8; m * nb6 * Q6_BLOCK_BYTES];
        let mut qx = vec![0u8; n * nb8 * Q8_BLOCK_BYTES];

        for row in 0..m {
            quantize_q6(
                &w[row * k..(row + 1) * k],
                &mut qw[row * nb6 * Q6_BLOCK_BYTES..(row + 1) * nb6 * Q6_BLOCK_BYTES],
            );
        }
        quantize_q8(&x, &mut qx);

        let mut c = vec![0.0f32; m * n];
        gemm_q6_q8(&qw, &qx, &mut c, m, n, k);

        for i in 0..m {
            let expected = dot_f32(&w[i * k..(i + 1) * k], &x);
            let rel_err = (c[i] - expected).abs() / expected.abs().max(1e-6);
            assert!(
                rel_err < 0.15,
                "gemm_q6q8 row {i}: got={}, expected={expected}, rel_err={rel_err}",
                c[i]
            );
        }
    }

    #[test]
    fn gemm_accumulates_into_c() {
        // Verify C += W × X (not C = W × X).
        let k = 32;
        let mut w = vec![0.0f32; k];
        w[0] = 1.0;
        let x = vec![1.0f32; k];

        let mut qw = vec![0u8; Q8_BLOCK_BYTES];
        let mut qx = vec![0u8; Q8_BLOCK_BYTES];
        quantize_q8(&w, &mut qw);
        quantize_q8(&x, &mut qx);

        let mut c = vec![100.0f32; 1];
        gemm_q8_q8(&qw, &qx, &mut c, 1, 1, k);

        // Should be ~101.0 (100 + dot ≈ 1.0).
        assert!(c[0] > 100.5, "expected accumulation, got c[0]={}", c[0]);
    }
}
