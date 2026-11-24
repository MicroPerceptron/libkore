//! Bulk dtype conversion kernels — AVX2 + F16C.
//!
//! F16C hardware conversion: 8 elements per instruction.
//! BF16: shift-based conversion (no dedicated hardware on AVX2).
//! Quantized: delegate to `crate::quant` (same as scalar — the block
//! logic is complex enough that per-block scalar is fine for cast;
//! the hot path is qdot, not cast).

use core::arch::x86_64::*;

// ── Float conversion ───────────────────────────────────────────────

/// F16 → F32 bulk conversion using F16C `vcvtph2ps`.
pub(crate) fn f16_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n * 2);
    debug_assert!(dst.len() >= n);

    unsafe {
        let chunks = n / 8;
        let tail = n % 8;
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        for i in 0..chunks {
            let bits = _mm_loadu_si128(sp.add(i * 16) as *const __m128i);
            let f32s = _mm256_cvtph_ps(bits);
            _mm256_storeu_ps(dp.add(i * 8), f32s);
        }

        let off = chunks * 8;
        for j in 0..tail {
            let bo = (off + j) * 2;
            let bits = u16::from_le_bytes([*sp.add(bo), *sp.add(bo + 1)]);
            *dp.add(off + j) = crate::cast::f16_to_f32(bits);
        }
    }
}

/// F32 → F16 bulk conversion using F16C `vcvtps2ph`.
pub(crate) fn f32_to_f16(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n * 2);

    unsafe {
        let chunks = n / 8;
        let tail = n % 8;
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        for i in 0..chunks {
            let f32s = _mm256_loadu_ps(sp.add(i * 8));
            let bits = _mm256_cvtps_ph(f32s, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(dp.add(i * 16) as *mut __m128i, bits);
        }

        let off = chunks * 8;
        for j in 0..tail {
            let bytes = crate::cast::f32_to_f16(*sp.add(off + j)).to_le_bytes();
            let bo = (off + j) * 2;
            *dp.add(bo) = bytes[0];
            *dp.add(bo + 1) = bytes[1];
        }
    }
}

/// BF16 → F32 bulk conversion — shift-based, 8-wide.
pub(crate) fn bf16_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n * 2);
    debug_assert!(dst.len() >= n);

    unsafe {
        let chunks = n / 8;
        let tail = n % 8;
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        for i in 0..chunks {
            let bits16 = _mm_loadu_si128(sp.add(i * 16) as *const __m128i);
            let wide = _mm256_cvtepu16_epi32(bits16);
            let f32s = _mm256_castsi256_ps(_mm256_slli_epi32(wide, 16));
            _mm256_storeu_ps(dp.add(i * 8), f32s);
        }

        let off = chunks * 8;
        for j in 0..tail {
            let bo = (off + j) * 2;
            let bits = u16::from_le_bytes([*sp.add(bo), *sp.add(bo + 1)]);
            *dp.add(off + j) = crate::cast::b16_to_f32(bits);
        }
    }
}

/// F32 → BF16 bulk conversion — shift-based truncation, 8-wide.
pub(crate) fn f32_to_bf16(src: &[f32], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n * 2);

    unsafe {
        let chunks = n / 8;
        let tail = n % 8;
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        for i in 0..chunks {
            let f32s = _mm256_loadu_ps(sp.add(i * 8));
            let i32s = _mm256_srli_epi32(_mm256_castps_si256(f32s), 16);
            let lo128 = _mm256_castsi256_si128(i32s);
            let hi128 = _mm256_extracti128_si256(i32s, 1);
            let packed = _mm_packus_epi32(lo128, hi128);
            _mm_storeu_si128(dp.add(i * 16) as *mut __m128i, packed);
        }

        let off = chunks * 8;
        for j in 0..tail {
            let bytes = crate::cast::f32_to_b16(*sp.add(off + j)).to_le_bytes();
            let bo = (off + j) * 2;
            *dp.add(bo) = bytes[0];
            *dp.add(bo + 1) = bytes[1];
        }
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
