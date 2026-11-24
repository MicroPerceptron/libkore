//! Bulk dtype conversion kernels — scalar fallback.
//!
//! Row-at-a-time conversion for SIMD acceleration opportunities.
//! Element-wise scalar casts live in `crate::cast`; these are the
//! bulk paths that SIMD tiers (F16C, AVX-512BF16, VNNI) can accelerate.
//!
//! Quantized kernels delegate to `crate::quant` block-level helpers.
//! SIMD tiers will replace these with intrinsics-based implementations
//! that bypass the block abstraction entirely.

// ── Scalar float conversion ────────────────────────────────────────

/// F16 → F32 bulk conversion.
///
/// `src` is `n` packed f16 values (2 bytes each, little-endian).
pub(crate) fn f16_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n * 2);
    debug_assert!(dst.len() >= n);
    for (d, chunk) in dst[..n].iter_mut().zip(src[..n * 2].chunks_exact(2)) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        *d = crate::cast::f16_to_f32(bits);
    }
}

/// F32 → F16 bulk conversion.
///
/// `dst` receives `n` packed f16 values (2 bytes each, little-endian).
pub(crate) fn f32_to_f16(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n * 2);
    for (&s, chunk) in src[..n].iter().zip(dst[..n * 2].chunks_exact_mut(2)) {
        let bytes = crate::cast::f32_to_f16(s).to_le_bytes();
        chunk[0] = bytes[0];
        chunk[1] = bytes[1];
    }
}

/// BF16 → F32 bulk conversion.
///
/// `src` is `n` packed bf16 values (2 bytes each, little-endian).
pub(crate) fn bf16_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n * 2);
    debug_assert!(dst.len() >= n);
    for (d, chunk) in dst[..n].iter_mut().zip(src[..n * 2].chunks_exact(2)) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        *d = crate::cast::b16_to_f32(bits);
    }
}

/// F32 → BF16 bulk conversion.
///
/// `dst` receives `n` packed bf16 values (2 bytes each, little-endian).
pub(crate) fn f32_to_bf16(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n * 2);
    for (&s, chunk) in src[..n].iter().zip(dst[..n * 2].chunks_exact_mut(2)) {
        let bytes = crate::cast::f32_to_b16(s).to_le_bytes();
        chunk[0] = bytes[0];
        chunk[1] = bytes[1];
    }
}

// ── Dequantize (Qx → F32) ─────────────────────────────────────────

/// Dequantize `n` Q8_0 elements → f32.
///
/// `src` is packed Q8_0 blocks. `n` must be divisible by 32.
pub(crate) fn dequant_q8_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert_eq!(n % 32, 0);
    crate::quant::dequantize_q8(&src[..n / 32 * 34], &mut dst[..n]);
}

/// Dequantize `n` Q6_K elements → f32.
///
/// `src` is packed Q6_K blocks. `n` must be divisible by 256.
pub(crate) fn dequant_q6_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert_eq!(n % 256, 0);
    crate::quant::dequantize_q6(&src[..n / 256 * 210], &mut dst[..n]);
}

/// Dequantize `n` Q4_K_M elements → f32.
///
/// `src` is packed Q4_K_M blocks. `n` must be divisible by 256.
pub(crate) fn dequant_q4_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert_eq!(n % 256, 0);
    crate::quant::dequantize_q4(&src[..n / 256 * 144], &mut dst[..n]);
}

// ── Quantize (F32 → Qx) ───────────────────────────────────────────

/// Quantize `n` f32 elements → Q8_0.
///
/// `dst` receives packed Q8_0 blocks. `n` must be divisible by 32.
pub(crate) fn quant_f32_q8(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert_eq!(n % 32, 0);
    crate::quant::quantize_q8(&src[..n], &mut dst[..n / 32 * 34]);
}

/// Quantize `n` f32 elements → Q6_K.
///
/// `dst` receives packed Q6_K blocks. `n` must be divisible by 256.
pub(crate) fn quant_f32_q6(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert_eq!(n % 256, 0);
    crate::quant::quantize_q6(&src[..n], &mut dst[..n / 256 * 210]);
}

/// Quantize `n` f32 elements → Q4_K_M.
///
/// `dst` receives packed Q4_K_M blocks. `n` must be divisible by 256.
pub(crate) fn quant_f32_q4(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert_eq!(n % 256, 0);
    crate::quant::quantize_q4(&src[..n], &mut dst[..n / 256 * 144]);
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── F16 round-trip ───────────────────────────────────────

    #[test]
    fn f16_roundtrip_basic() {
        let vals = [0.0f32, 1.0, -1.0, 0.5, 65504.0];
        let mut f16_buf = [0u8; 10];
        let mut out = [0.0f32; 5];
        f32_to_f16(&vals, &mut f16_buf, 5);
        f16_to_f32(&f16_buf, &mut out, 5);
        for i in 0..5 {
            assert!(
                close(out[i], vals[i]),
                "mismatch at {i}: {} vs {}",
                out[i],
                vals[i]
            );
        }
    }

    #[test]
    fn f16_roundtrip_small() {
        let vals = [0.001, 0.01, 0.1f32];
        let mut f16_buf = [0u8; 6];
        let mut out = [0.0f32; 3];
        f32_to_f16(&vals, &mut f16_buf, 3);
        f16_to_f32(&f16_buf, &mut out, 3);
        for i in 0..3 {
            assert!(
                close(out[i], vals[i]),
                "mismatch at {i}: {} vs {}",
                out[i],
                vals[i]
            );
        }
    }

    // ── BF16 round-trip ──────────────────────────────────────

    #[test]
    fn bf16_roundtrip_basic() {
        let vals = [0.0f32, 1.0, -1.0, 3.14, 1e10];
        let mut bf16_buf = [0u8; 10];
        let mut out = [0.0f32; 5];
        f32_to_bf16(&vals, &mut bf16_buf, 5);
        bf16_to_f32(&bf16_buf, &mut out, 5);
        for i in 0..5 {
            let tol = if vals[i] == 0.0 {
                0.0
            } else {
                vals[i].abs() * 0.01
            };
            assert!(
                (out[i] - vals[i]).abs() <= tol,
                "mismatch at {i}: {} vs {}",
                out[i],
                vals[i]
            );
        }
    }

    #[test]
    fn bf16_exact_powers_of_two() {
        let vals = [1.0, 2.0, 4.0, 0.5, 0.25f32];
        let mut bf16_buf = [0u8; 10];
        let mut out = [0.0f32; 5];
        f32_to_bf16(&vals, &mut bf16_buf, 5);
        bf16_to_f32(&bf16_buf, &mut out, 5);
        assert_eq!(out, vals);
    }

    // ── Cross-format ─────────────────────────────────────────

    #[test]
    fn f16_preserves_sign() {
        let vals = [-42.0f32];
        let mut f16_buf = [0u8; 2];
        let mut out = [0.0f32; 1];
        f32_to_f16(&vals, &mut f16_buf, 1);
        f16_to_f32(&f16_buf, &mut out, 1);
        assert!(out[0] < 0.0);
        assert!(close(out[0], -42.0));
    }

    // ── Quant round-trips through arch dispatch ──────────────

    #[test]
    fn dequant_quant_q8_roundtrip() {
        let src: [f32; 32] = core::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
        let mut q = [0u8; 34];
        let mut out = [0.0f32; 32];

        quant_f32_q8(&src, &mut q, 32);
        dequant_q8_f32(&q, &mut out, 32);

        for i in 0..32 {
            let err = (src[i] - out[i]).abs();
            assert!(err < 0.02, "q8 elem {i}: err={err}");
        }
    }

    #[test]
    fn dequant_quant_q6_roundtrip() {
        let src: [f32; 256] = core::array::from_fn(|i| (i as f32 / 128.0 - 1.0) * 2.0);
        let mut q = [0u8; 210];
        let mut out = [0.0f32; 256];

        quant_f32_q6(&src, &mut q, 256);
        dequant_q6_f32(&q, &mut out, 256);

        let max_err = src
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.15, "q6 roundtrip max_err={max_err}");
    }

    #[test]
    fn dequant_quant_q4_roundtrip() {
        let src: [f32; 256] = core::array::from_fn(|i| i as f32 / 128.0 - 1.0);
        let mut q = [0u8; 144];
        let mut out = [0.0f32; 256];

        quant_f32_q4(&src, &mut q, 256);
        dequant_q4_f32(&q, &mut out, 256);

        let max_err = src
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.3, "q4 roundtrip max_err={max_err}");
    }
}
