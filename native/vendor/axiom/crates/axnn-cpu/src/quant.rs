//! Block-quantized dequantize / quantize operations.
//!
//! Each function operates on a single block of raw bytes (the on-disk /
//! in-memory format) and produces or consumes `f32` elements. Row-level
//! wrappers loop over blocks for the full tensor cast path.
//!
//! Block layouts match the ggml reference exactly:
//!
//! | Type | Block | Bytes | Layout                                          |
//! | ---- | ----- | ----- | ----------------------------------------------- |
//! | Q8_0 | 32    | 34    | `d:f16 (2B) + qs:[i8;32] (32B)`                 |
//! | Q6_K | 256   | 210   | `ql:[u8;128] + qh:[u8;64] + sc:[i8;16] + d:f16` |
//! | Q4_K | 256   | 144   | `d:f16 + dmin:f16 + sc:[u8;12] + qs:[u8;128]`   |

use crate::cast::{f16_to_f32, f32_to_f16};

// ── no_std math helpers ────────────────────────────────────────────
//
// f32::round() is std-only in older Rust. These cover the narrow
// range needed by quantization (values that fit in i32).

/// Round to nearest integer, ties away from zero.
#[inline(always)]
fn roundf(v: f32) -> f32 {
    if v >= 0.0 {
        (v + 0.5) as i32 as f32
    } else {
        (v - 0.5) as i32 as f32
    }
}

/// Clamp `v` to `[lo, hi]`.
#[inline(always)]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Q8_0 — Symmetric 8-bit, f16 per-block scale
// ═══════════════════════════════════════════════════════════════════
//
// Block: 32 elements
// Byte layout: [d: f16 LE (2B)] [qs: i8 × 32 (32B)]  = 34 bytes total
// Symmetric quantization: val = d * qs[i]

pub(crate) const Q8_BLOCK_SIZE: usize = 32;
pub(crate) const Q8_BLOCK_BYTES: usize = 34;

pub(crate) const Q8_D_OFFSET: usize = 0; // f16 scale
pub(crate) const Q8_QS_OFFSET: usize = 2; // 32 × i8

/// Dequantize one Q8_0 block (34 bytes) into 32 f32 values.
pub(crate) fn dequantize_q8_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), Q8_BLOCK_BYTES);
    debug_assert_eq!(dst.len(), Q8_BLOCK_SIZE);

    let d = f16_to_f32(u16::from_le_bytes([src[Q8_D_OFFSET], src[Q8_D_OFFSET + 1]]));

    for i in 0..Q8_BLOCK_SIZE {
        let qs = src[Q8_QS_OFFSET + i] as i8;
        dst[i] = d * qs as f32;
    }
}

/// Quantize 32 f32 values into one Q8_0 block (34 bytes).
///
/// Symmetric: find max |val|, scale = max / 127, round to nearest i8.
pub(crate) fn quantize_q8_block(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), Q8_BLOCK_SIZE);
    debug_assert_eq!(dst.len(), Q8_BLOCK_BYTES);

    let mut amax = 0.0f32;
    for &v in src.iter() {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }

    let d = amax / 127.0;
    let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };

    let d_bits = f32_to_f16(d);
    dst[Q8_D_OFFSET] = d_bits as u8;
    dst[Q8_D_OFFSET + 1] = (d_bits >> 8) as u8;

    for i in 0..Q8_BLOCK_SIZE {
        let q = clampf(roundf(src[i] * id), -127.0, 127.0) as i8;
        dst[Q8_QS_OFFSET + i] = q as u8;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Q6_K — 6-bit K-quant, super-block structure
// ═══════════════════════════════════════════════════════════════════
//
// Block: 256 elements, 16 sub-blocks of 16 elements.
// Byte layout (210 bytes total):
//   ql : [u8; 128]  — low 4 bits of each 6-bit value (2 per byte)
//   qh : [u8; 64]   — high 2 bits of each 6-bit value (4 per byte)
//   scales: [i8; 16] — per-sub-block scale factors
//   d  : f16 LE      — super-block scale
//
// Reconstruction:
//   q6 = (ql_nibble) | (qh_2bits << 4)   — unsigned [0..63]
//   val = d * scale[j/16] * (q6 - 32)    — center around zero

pub(crate) const Q6_BLOCK_SIZE: usize = 256;
pub(crate) const Q6_BLOCK_BYTES: usize = 210;
pub(crate) const Q6_SUB_BLOCK: usize = 16; // elements per sub-block

pub(crate) const Q6_QL_OFFSET: usize = 0; // 128 bytes: low nibbles
pub(crate) const Q6_QH_OFFSET: usize = 128; // 64 bytes: high 2 bits
pub(crate) const Q6_SC_OFFSET: usize = 192; // 16 bytes: i8 sub-block scales
pub(crate) const Q6_D_OFFSET: usize = 208; // 2 bytes: f16 super-block scale

/// Dequantize one Q6_K block (210 bytes) into 256 f32 values.
pub(crate) fn dequantize_q6_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), Q6_BLOCK_BYTES);
    debug_assert_eq!(dst.len(), Q6_BLOCK_SIZE);

    let d = f16_to_f32(u16::from_le_bytes([src[Q6_D_OFFSET], src[Q6_D_OFFSET + 1]]));

    for j in 0..Q6_BLOCK_SIZE {
        let ql_byte = src[Q6_QL_OFFSET + j / 2];
        let low4 = (ql_byte >> (4 * (j % 2))) & 0x0F;

        let qh_byte = src[Q6_QH_OFFSET + j / 4];
        let high2 = (qh_byte >> (2 * (j % 4))) & 0x03;

        let q6 = (low4 | (high2 << 4)) as i32 - 32;

        let sc = src[Q6_SC_OFFSET + j / 16] as i8;

        dst[j] = d * sc as f32 * q6 as f32;
    }
}

/// Quantize 256 f32 values into one Q6_K block (210 bytes).
///
/// Two-pass hierarchical quantization:
///   1. Per-sub-block (16 elems): find ideal scale from max |val|.
///   2. Super-block `d` normalizes sub-block scales into i8 range.
///   3. Each element → 6-bit unsigned q6, packed across ql + qh.
pub(crate) fn quantize_q6_block(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), Q6_BLOCK_SIZE);
    debug_assert_eq!(dst.len(), Q6_BLOCK_BYTES);

    // Zero the output block.
    for b in dst.iter_mut() {
        *b = 0;
    }

    // Pass 1: compute per-sub-block ideal scales.
    // Each sub-block of 16 elements maps to [-32..31] centered range.
    // Ideal scale = max_abs / 31.
    let mut sub_scales = [0.0f32; 16];
    for sb in 0..16 {
        let base = sb * Q6_SUB_BLOCK;
        let mut amax = 0.0f32;
        for j in 0..Q6_SUB_BLOCK {
            let a = src[base + j].abs();
            if a > amax {
                amax = a;
            }
        }
        sub_scales[sb] = amax / 31.0;
    }

    // Pass 2: super-block scale d normalizes sub-scales into i8 range.
    // d = max(sub_scales) / 127.
    let mut max_sub = 0.0f32;
    for &s in sub_scales.iter() {
        if s > max_sub {
            max_sub = s;
        }
    }

    let d = max_sub / 127.0;
    let id = if max_sub != 0.0 { 127.0 / max_sub } else { 0.0 };

    // Store super-block scale.
    let d_bits = f32_to_f16(d);
    dst[Q6_D_OFFSET] = d_bits as u8;
    dst[Q6_D_OFFSET + 1] = (d_bits >> 8) as u8;

    // Compute and store i8 sub-block scales.
    // Re-read d back from f16 so quantize and dequantize use the exact
    // same value (avoids f16 truncation mismatch).
    let d_actual = f16_to_f32(d_bits);

    let mut i8_scales = [0i8; 16];
    for sb in 0..16 {
        let sc = clampf(roundf(sub_scales[sb] * id), -127.0, 127.0) as i8;
        i8_scales[sb] = sc;
        dst[Q6_SC_OFFSET + sb] = sc as u8;
    }

    // Pass 3: quantize each element to 6-bit, pack into ql + qh.
    for j in 0..Q6_BLOCK_SIZE {
        let sb = j / Q6_SUB_BLOCK;
        let sc = i8_scales[sb] as f32;
        let effective_scale = d_actual * sc;

        let q6 = if effective_scale.abs() > 1e-15 {
            let centered = src[j] / effective_scale;
            (clampf(roundf(centered), -32.0, 31.0) as i32 + 32) as u8
        } else {
            32 // zero → centered 0
        };

        // Pack low 4 bits into ql.
        let low4 = q6 & 0x0F;
        let ql_idx = Q6_QL_OFFSET + j / 2;
        let ql_shift = 4 * (j % 2);
        dst[ql_idx] |= low4 << ql_shift;

        // Pack high 2 bits into qh.
        let high2 = (q6 >> 4) & 0x03;
        let qh_idx = Q6_QH_OFFSET + j / 4;
        let qh_shift = 2 * (j % 4);
        dst[qh_idx] |= high2 << qh_shift;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Q4_K_M — 4-bit K-quant medium, asymmetric with min offsets
// ═══════════════════════════════════════════════════════════════════
//
// Block: 256 elements, 8 sub-blocks of 32 elements.
// Byte layout (144 bytes total):
//   d    : f16 LE    — super-block scale
//   dmin : f16 LE    — super-block minimum offset
//   scales: [u8; 12] — packed 6-bit scale & min for each sub-block
//   qs   : [u8; 128] — packed 4-bit quantized values (2 per byte)
//
// Reconstruction:
//   nibble = (qs[j/2] >> (4*(j%2))) & 0x0F
//   val = d * scale[j/32] * nibble - dmin * min[j/32]

pub(crate) const Q4_BLOCK_SIZE: usize = 256;
pub(crate) const Q4_BLOCK_BYTES: usize = 144;
pub(crate) const Q4_SUB_BLOCK: usize = 32; // elements per sub-block

pub(crate) const Q4_D_OFFSET: usize = 0; // 2 bytes: f16 super-block scale
pub(crate) const Q4_DMIN_OFFSET: usize = 2; // 2 bytes: f16 super-block min
pub(crate) const Q4_SC_OFFSET: usize = 4; // 12 bytes: packed 6-bit scale/min
pub(crate) const Q4_QS_OFFSET: usize = 16; // 128 bytes: packed nibbles

/// Unpack Q4_K_M 6-bit scales and mins from the 12-byte packed field.
///
/// Returns `([scale; 8], [min; 8])` — both as u8 values in [0..63].
pub(crate) fn unpack_q4_scales(sc: &[u8]) -> ([u8; 8], [u8; 8]) {
    debug_assert_eq!(sc.len(), 12);
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for i in 0..4 {
        scales[i] = sc[i] & 0x3F;
        mins[i] = sc[4 + i] & 0x3F;
    }

    for i in 4..8 {
        scales[i] = (sc[i - 4] >> 6) | ((sc[i + 4] & 0x0F) << 2);
        mins[i] = (sc[i] >> 6) | ((sc[i + 4] >> 4) << 2);
    }

    (scales, mins)
}

/// Pack Q4_K_M 6-bit scales and mins into the 12-byte packed field.
///
/// Inverse of `unpack_q4_scales`. Both arrays must be in [0..63].
pub(crate) fn pack_q4_scales(scales: &[u8; 8], mins: &[u8; 8], sc: &mut [u8]) {
    debug_assert_eq!(sc.len(), 12);

    for b in sc.iter_mut() {
        *b = 0;
    }

    // Sub-blocks 0..3: low 6 bits into bytes 0..3 (scales) and 4..7 (mins).
    // High 2 bits of bytes 0..3 carry bits for scales[4..7].
    // High 2 bits of bytes 4..7 carry bits for mins[4..7].
    for i in 0..4 {
        sc[i] = (scales[i] & 0x3F) | ((scales[i + 4] & 0x03) << 6);
        sc[4 + i] = (mins[i] & 0x3F) | ((mins[i + 4] & 0x03) << 6);
    }

    // Bytes 8..11: remaining 4 bits for scales[4..7] and mins[4..7].
    for i in 0..4 {
        sc[8 + i] = ((scales[i + 4] >> 2) & 0x0F) | (((mins[i + 4] >> 2) & 0x0F) << 4);
    }
}

/// Dequantize one Q4_K_M block (144 bytes) into 256 f32 values.
pub(crate) fn dequantize_q4_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), Q4_BLOCK_BYTES);
    debug_assert_eq!(dst.len(), Q4_BLOCK_SIZE);

    let d = f16_to_f32(u16::from_le_bytes([src[Q4_D_OFFSET], src[Q4_D_OFFSET + 1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([
        src[Q4_DMIN_OFFSET],
        src[Q4_DMIN_OFFSET + 1],
    ]));

    let (scales, mins) = unpack_q4_scales(&src[Q4_SC_OFFSET..Q4_SC_OFFSET + 12]);

    for j in 0..Q4_BLOCK_SIZE {
        let qs_byte = src[Q4_QS_OFFSET + j / 2];
        let nibble = (qs_byte >> (4 * (j % 2))) & 0x0F;

        let sb = j / Q4_SUB_BLOCK;
        dst[j] = d * scales[sb] as f32 * nibble as f32 - dmin * mins[sb] as f32;
    }
}

/// Quantize 256 f32 values into one Q4_K_M block (144 bytes).
///
/// Asymmetric quantization with per-sub-block scale + min offsets.
///
///   1. Per-sub-block (32 elems): find min and max.
///   2. range = max - min, nibble = round((val - min) / range * 15).
///   3. Super-block `d` and `dmin` normalize sub-block scales/mins into
///      6-bit range [0..63].
pub(crate) fn quantize_q4_block(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), Q4_BLOCK_SIZE);
    debug_assert_eq!(dst.len(), Q4_BLOCK_BYTES);

    for b in dst.iter_mut() {
        *b = 0;
    }

    // Pass 1: per-sub-block min/max → ideal scale and offset.
    let mut sub_scale = [0.0f32; 8]; // range / 15
    let mut sub_min = [0.0f32; 8]; // min value (most negative)

    for sb in 0..8 {
        let base = sb * Q4_SUB_BLOCK;
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for j in 0..Q4_SUB_BLOCK {
            let v = src[base + j];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        // Clamp min to ≤ 0 (if all positive, min offset can still be 0).
        if lo > 0.0 {
            lo = 0.0;
        }
        sub_min[sb] = -lo; // store as positive offset
        let range = hi - lo;
        sub_scale[sb] = if range > 0.0 { range / 15.0 } else { 0.0 };
    }

    // Pass 2: super-block d and dmin normalize into [0..63].
    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;
    for sb in 0..8 {
        if sub_scale[sb] > max_scale {
            max_scale = sub_scale[sb];
        }
        if sub_min[sb] > max_min {
            max_min = sub_min[sb];
        }
    }

    let d = max_scale / 63.0;
    let dmin = max_min / 63.0;
    let inv_d = if max_scale != 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_dmin = if max_min != 0.0 { 63.0 / max_min } else { 0.0 };

    // Store super-block parameters.
    let d_bits = f32_to_f16(d);
    dst[Q4_D_OFFSET] = d_bits as u8;
    dst[Q4_D_OFFSET + 1] = (d_bits >> 8) as u8;

    let dmin_bits = f32_to_f16(dmin);
    dst[Q4_DMIN_OFFSET] = dmin_bits as u8;
    dst[Q4_DMIN_OFFSET + 1] = (dmin_bits >> 8) as u8;

    // Re-read from f16 for exact round-trip consistency.
    let d_actual = f16_to_f32(d_bits);
    let dmin_actual = f16_to_f32(dmin_bits);

    // Compute and pack 6-bit sub-block scales and mins.
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for sb in 0..8 {
        scales[sb] = clampf(roundf(sub_scale[sb] * inv_d), 0.0, 63.0) as u8;
        mins[sb] = clampf(roundf(sub_min[sb] * inv_dmin), 0.0, 63.0) as u8;
    }

    pack_q4_scales(&scales, &mins, &mut dst[Q4_SC_OFFSET..Q4_SC_OFFSET + 12]);

    // Pass 3: quantize each element to 4-bit nibble.
    for j in 0..Q4_BLOCK_SIZE {
        let sb = j / Q4_SUB_BLOCK;

        let eff_scale = d_actual * scales[sb] as f32;
        let eff_min = dmin_actual * mins[sb] as f32;

        let nibble = if eff_scale > 1e-15 {
            clampf(roundf((src[j] + eff_min) / eff_scale), 0.0, 15.0) as u8
        } else {
            0
        };

        // Pack nibble.
        let qs_idx = Q4_QS_OFFSET + j / 2;
        let shift = 4 * (j % 2);
        dst[qs_idx] |= nibble << shift;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Row-level wrappers
// ═══════════════════════════════════════════════════════════════════

/// Dequantize a contiguous buffer of Q8_0 blocks into f32.
pub(crate) fn dequantize_q8(src: &[u8], dst: &mut [f32]) {
    let num_blocks = src.len() / Q8_BLOCK_BYTES;
    debug_assert_eq!(src.len(), num_blocks * Q8_BLOCK_BYTES);
    debug_assert_eq!(dst.len(), num_blocks * Q8_BLOCK_SIZE);

    for b in 0..num_blocks {
        let s = b * Q8_BLOCK_BYTES;
        let d = b * Q8_BLOCK_SIZE;
        dequantize_q8_block(&src[s..s + Q8_BLOCK_BYTES], &mut dst[d..d + Q8_BLOCK_SIZE]);
    }
}

/// Quantize a contiguous f32 buffer into Q8_0 blocks.
pub(crate) fn quantize_q8(src: &[f32], dst: &mut [u8]) {
    let num_blocks = src.len() / Q8_BLOCK_SIZE;
    debug_assert_eq!(src.len(), num_blocks * Q8_BLOCK_SIZE);
    debug_assert_eq!(dst.len(), num_blocks * Q8_BLOCK_BYTES);

    for b in 0..num_blocks {
        let s = b * Q8_BLOCK_SIZE;
        let d = b * Q8_BLOCK_BYTES;
        quantize_q8_block(&src[s..s + Q8_BLOCK_SIZE], &mut dst[d..d + Q8_BLOCK_BYTES]);
    }
}

/// Dequantize a contiguous buffer of Q6_K blocks into f32.
pub(crate) fn dequantize_q6(src: &[u8], dst: &mut [f32]) {
    let num_blocks = src.len() / Q6_BLOCK_BYTES;
    debug_assert_eq!(src.len(), num_blocks * Q6_BLOCK_BYTES);
    debug_assert_eq!(dst.len(), num_blocks * Q6_BLOCK_SIZE);

    for b in 0..num_blocks {
        let s = b * Q6_BLOCK_BYTES;
        let d = b * Q6_BLOCK_SIZE;
        dequantize_q6_block(&src[s..s + Q6_BLOCK_BYTES], &mut dst[d..d + Q6_BLOCK_SIZE]);
    }
}

/// Quantize a contiguous f32 buffer into Q6_K blocks.
pub(crate) fn quantize_q6(src: &[f32], dst: &mut [u8]) {
    let num_blocks = src.len() / Q6_BLOCK_SIZE;
    debug_assert_eq!(src.len(), num_blocks * Q6_BLOCK_SIZE);
    debug_assert_eq!(dst.len(), num_blocks * Q6_BLOCK_BYTES);

    for b in 0..num_blocks {
        let s = b * Q6_BLOCK_SIZE;
        let d = b * Q6_BLOCK_BYTES;
        quantize_q6_block(&src[s..s + Q6_BLOCK_SIZE], &mut dst[d..d + Q6_BLOCK_BYTES]);
    }
}

/// Dequantize a contiguous buffer of Q4_K_M blocks into f32.
pub(crate) fn dequantize_q4(src: &[u8], dst: &mut [f32]) {
    let num_blocks = src.len() / Q4_BLOCK_BYTES;
    debug_assert_eq!(src.len(), num_blocks * Q4_BLOCK_BYTES);
    debug_assert_eq!(dst.len(), num_blocks * Q4_BLOCK_SIZE);

    for b in 0..num_blocks {
        let s = b * Q4_BLOCK_BYTES;
        let d = b * Q4_BLOCK_SIZE;
        dequantize_q4_block(&src[s..s + Q4_BLOCK_BYTES], &mut dst[d..d + Q4_BLOCK_SIZE]);
    }
}

/// Quantize a contiguous f32 buffer into Q4_K_M blocks.
pub(crate) fn quantize_q4(src: &[f32], dst: &mut [u8]) {
    let num_blocks = src.len() / Q4_BLOCK_SIZE;
    debug_assert_eq!(src.len(), num_blocks * Q4_BLOCK_SIZE);
    debug_assert_eq!(dst.len(), num_blocks * Q4_BLOCK_BYTES);

    for b in 0..num_blocks {
        let s = b * Q4_BLOCK_SIZE;
        let d = b * Q4_BLOCK_BYTES;
        quantize_q4_block(&src[s..s + Q4_BLOCK_SIZE], &mut dst[d..d + Q4_BLOCK_BYTES]);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // ── Q8_0 ───────────────────────────────────────────────────

    #[test]
    fn q8_roundtrip_identity() {
        let src: [f32; 32] = core::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
        let mut block = [0u8; 34];
        quantize_q8_block(&src, &mut block);

        let mut out = [0.0f32; 32];
        dequantize_q8_block(&block, &mut out);

        for i in 0..32 {
            let err = (src[i] - out[i]).abs();
            // Q8: d ≈ max_abs/127 ≈ 1.6/127 ≈ 0.013. Max error < d.
            assert!(
                err < 0.02,
                "q8 elem {i}: src={}, got={}, err={err}",
                src[i],
                out[i]
            );
        }
    }

    #[test]
    fn q8_zeros() {
        let src = [0.0f32; 32];
        let mut block = [0xFFu8; 34];
        quantize_q8_block(&src, &mut block);

        let mut out = [1.0f32; 32];
        dequantize_q8_block(&block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_preserves_sign() {
        let mut src = [0.0f32; 32];
        src[0] = 1.0;
        src[1] = -1.0;
        let mut block = [0u8; 34];
        quantize_q8_block(&src, &mut block);

        let mut out = [0.0f32; 32];
        dequantize_q8_block(&block, &mut out);
        assert!(out[0] > 0.0);
        assert!(out[1] < 0.0);
    }

    #[test]
    fn q8_scale_stored_as_f16() {
        let mut src = [0.0f32; 32];
        src[0] = 10.0;
        let mut block = [0u8; 34];
        quantize_q8_block(&src, &mut block);

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let expected = 10.0 / 127.0;
        assert!((d - expected).abs() < 0.001, "d={d}, expected={expected}");
    }

    #[test]
    fn q8_row_level() {
        let src: [f32; 64] = core::array::from_fn(|i| i as f32 * 0.5 - 16.0);
        let mut encoded = vec![0u8; 2 * Q8_BLOCK_BYTES];
        let mut decoded = vec![0.0f32; 64];

        quantize_q8(&src, &mut encoded);
        dequantize_q8(&encoded, &mut decoded);

        for i in 0..64 {
            let err = (src[i] - decoded[i]).abs();
            assert!(err < 0.15, "q8 row elem {i}: err={err}");
        }
    }

    // ── Q6_K ───────────────────────────────────────────────────

    #[test]
    fn q6_dequantize_zeros() {
        let block = [0u8; 210];
        let mut out = [1.0f32; 256];
        dequantize_q6_block(&block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q6_dequantize_known_values() {
        let mut block = [0u8; 210];

        // d = 1.0 as f16
        let d_bits = f32_to_f16(1.0);
        block[Q6_D_OFFSET] = d_bits as u8;
        block[Q6_D_OFFSET + 1] = (d_bits >> 8) as u8;

        // All sub-block scales = 1.
        for i in 0..16 {
            block[Q6_SC_OFFSET + i] = 1;
        }

        // All q6 = 33 → centered = 1.
        // q6 = low4 | (high2 << 4).  33 = 0b100001. low4 = 1, high2 = 2.
        for j in 0..256 {
            let ql_idx = Q6_QL_OFFSET + j / 2;
            let shift = 4 * (j % 2);
            block[ql_idx] |= 1 << shift;

            let qh_idx = Q6_QH_OFFSET + j / 4;
            let qh_shift = 2 * (j % 4);
            block[qh_idx] |= 2 << qh_shift;
        }

        let mut out = [0.0f32; 256];
        dequantize_q6_block(&block, &mut out);

        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "q6 known elem {i}: expected 1.0, got {v}"
            );
        }
    }

    #[test]
    fn q6_roundtrip() {
        // Generate a realistic distribution: values in [-2.0, 2.0].
        let src: [f32; 256] = core::array::from_fn(|i| {
            let t = i as f32 / 255.0; // [0, 1]
            (t - 0.5) * 4.0 // [-2, 2]
        });

        let mut block = [0u8; Q6_BLOCK_BYTES];
        quantize_q6_block(&src, &mut block);

        let mut out = [0.0f32; 256];
        dequantize_q6_block(&block, &mut out);

        // Q6: 6 bits → 64 levels over the range. Expect ~range/64 error.
        // Range ~4.0, so max error ~0.063. Allow some headroom for
        // hierarchical scale quantization noise.
        let mut max_err = 0.0f32;
        for i in 0..256 {
            let err = (src[i] - out[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 0.15, "q6 roundtrip max_err={max_err}");
    }

    #[test]
    fn q6_row_level() {
        let src: [f32; 512] = core::array::from_fn(|i| (i as f32 - 256.0) * 0.01);
        let mut encoded = vec![0u8; 2 * Q6_BLOCK_BYTES];
        let mut decoded = vec![0.0f32; 512];

        quantize_q6(&src, &mut encoded);
        dequantize_q6(&encoded, &mut decoded);

        let max_err = src
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.1, "q6 row max_err={max_err}");
    }

    // ── Q4_K_M ─────────────────────────────────────────────────

    #[test]
    fn q4_dequantize_zeros() {
        let block = [0u8; Q4_BLOCK_BYTES];
        let mut out = [1.0f32; 256];
        dequantize_q4_block(&block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q4_scale_pack_roundtrip() {
        let scales = [10u8, 20, 30, 40, 5, 15, 25, 35];
        let mins = [1u8, 2, 3, 4, 11, 22, 33, 44];

        let mut packed = [0u8; 12];
        pack_q4_scales(&scales, &mins, &mut packed);

        let (s2, m2) = unpack_q4_scales(&packed);
        assert_eq!(s2, scales, "scales mismatch");
        assert_eq!(m2, mins, "mins mismatch");
    }

    #[test]
    fn q4_scale_pack_roundtrip_edge() {
        // All 63s — max value for 6-bit.
        let scales = [63u8; 8];
        let mins = [63u8; 8];

        let mut packed = [0u8; 12];
        pack_q4_scales(&scales, &mins, &mut packed);

        let (s2, m2) = unpack_q4_scales(&packed);
        assert_eq!(s2, scales);
        assert_eq!(m2, mins);
    }

    #[test]
    fn q4_roundtrip() {
        // Generate values in [-1.0, 1.0].
        let src: [f32; 256] = core::array::from_fn(|i| {
            let t = i as f32 / 255.0;
            (t - 0.5) * 2.0
        });

        let mut block = [0u8; Q4_BLOCK_BYTES];
        quantize_q4_block(&src, &mut block);

        let mut out = [0.0f32; 256];
        dequantize_q4_block(&block, &mut out);

        // Q4: 4 bits → 16 levels. With range ~2.0, expect ~0.13 step.
        // Hierarchical scale overhead adds noise. Allow generous bound.
        let max_err = src
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.3, "q4 roundtrip max_err={max_err}");
    }

    #[test]
    fn q4_roundtrip_positive_only() {
        // All-positive range: tests that min-offset logic works.
        let src: [f32; 256] = core::array::from_fn(|i| i as f32 * 0.01);

        let mut block = [0u8; Q4_BLOCK_BYTES];
        quantize_q4_block(&src, &mut block);

        let mut out = [0.0f32; 256];
        dequantize_q4_block(&block, &mut out);

        let max_err = src
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.3, "q4 positive roundtrip max_err={max_err}");
    }

    #[test]
    fn q4_row_level() {
        let src: [f32; 512] = core::array::from_fn(|i| (i as f32 - 256.0) * 0.005);
        let mut encoded = vec![0u8; 2 * Q4_BLOCK_BYTES];
        let mut decoded = vec![0.0f32; 512];

        quantize_q4(&src, &mut encoded);
        dequantize_q4(&encoded, &mut decoded);

        let max_err = src
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.2, "q4 row max_err={max_err}");
    }
}
