//! Bulk dtype conversion kernels — portable `core::simd` tier.
//!
//! BF16 uses SIMD bit manipulation (shift-based, since BF16 is the top
//! 16 bits of f32). F16 uses scalar software conversion — the portable
//! tier has no F16C hardware; arch-specific tiers handle that.
//!
//! Quantized cast delegates to `crate::quant` (same as every other tier).

use core::simd::prelude::*;

use super::W;

// ── Float conversion ───────────────────────────────────────────────

/// F16 → F32 bulk conversion (scalar — no portable SIMD F16).
pub(crate) fn f16_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n * 2);
    debug_assert!(dst.len() >= n);

    for j in 0..n {
        let bo = j * 2;
        let bits = u16::from_le_bytes([src[bo], src[bo + 1]]);
        dst[j] = crate::cast::f16_to_f32(bits);
    }
}

/// F32 → F16 bulk conversion (scalar — no portable SIMD F16).
pub(crate) fn f32_to_f16(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n * 2);

    for j in 0..n {
        let bytes = crate::cast::f32_to_f16(src[j]).to_le_bytes();
        let bo = j * 2;
        dst[bo] = bytes[0];
        dst[bo + 1] = bytes[1];
    }
}

/// BF16 → F32 bulk conversion — shift-based, W-wide SIMD.
pub(crate) fn bf16_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n * 2);
    debug_assert!(dst.len() >= n);

    let chunks = n / W;
    let tail = n % W;

    for i in 0..chunks {
        let off = i * W;
        // Read W u16 values, widen to u32, shift left 16, reinterpret as f32
        let bits = Simd::<u32, W>::from_array(core::array::from_fn(|lane| {
            let bo = (off + lane) * 2;
            (u16::from_le_bytes([src[bo], src[bo + 1]]) as u32) << 16
        }));
        let f32s = Simd::<f32, W>::from_bits(bits);
        f32s.copy_to_slice(&mut dst[off..]);
    }

    let off = chunks * W;
    for j in 0..tail {
        let bo = (off + j) * 2;
        let bits = u16::from_le_bytes([src[bo], src[bo + 1]]);
        dst[off + j] = crate::cast::b16_to_f32(bits);
    }
}

/// F32 → BF16 bulk conversion — shift-based truncation, W-wide SIMD.
pub(crate) fn f32_to_bf16(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n * 2);

    let chunks = n / W;
    let tail = n % W;

    for i in 0..chunks {
        let off = i * W;
        let f32s = Simd::<f32, W>::from_slice(&src[off..]);
        let bits = f32s.to_bits() >> Simd::splat(16u32);
        let a = bits.as_array();
        for lane in 0..W {
            let bytes = (a[lane] as u16).to_le_bytes();
            let bo = (off + lane) * 2;
            dst[bo] = bytes[0];
            dst[bo + 1] = bytes[1];
        }
    }

    let off = chunks * W;
    for j in 0..tail {
        let bytes = crate::cast::f32_to_b16(src[off + j]).to_le_bytes();
        let bo = (off + j) * 2;
        dst[bo] = bytes[0];
        dst[bo + 1] = bytes[1];
    }
}

// ── Dequantize (Qx → F32) ─────────────────────────────────────────
//
// Quantized cast delegates to crate::quant — the block unpacking logic
// is complex (sub-block scales, nibble packing) and the cast path is
// not the hot loop. SIMD quant acceleration belongs in qdot.rs.

pub(crate) fn dequant_q8_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert_eq!(n % 32, 0);
    crate::quant::dequantize_q8(&src[..n / 32 * 34], &mut dst[..n]);
}

pub(crate) fn dequant_q6_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert_eq!(n % 256, 0);
    crate::quant::dequantize_q6(&src[..n / 256 * 210], &mut dst[..n]);
}

pub(crate) fn dequant_q4_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert_eq!(n % 256, 0);
    crate::quant::dequantize_q4(&src[..n / 256 * 144], &mut dst[..n]);
}

// ── Quantize (F32 → Qx) ───────────────────────────────────────────

pub(crate) fn quant_f32_q8(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert_eq!(n % 32, 0);
    crate::quant::quantize_q8(&src[..n], &mut dst[..n / 32 * 34]);
}

pub(crate) fn quant_f32_q6(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert_eq!(n % 256, 0);
    crate::quant::quantize_q6(&src[..n], &mut dst[..n / 256 * 210]);
}

pub(crate) fn quant_f32_q4(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert_eq!(n % 256, 0);
    crate::quant::quantize_q4(&src[..n], &mut dst[..n / 256 * 144]);
}
