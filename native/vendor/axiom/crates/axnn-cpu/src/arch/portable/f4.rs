//! Portable `core::simd` E2M1 (OCP FP4) kernels with external block scales.
//!
//! NVFP4 stores two weights per byte — the even element in the low nibble —
//! beside an E4M3 scale per sixteen elements and one F32 scale for the whole
//! tensor. The tensor scale is the caller's (it folds into the per-expert
//! scale a kernel already applies); these kernels take the block scales.
//!
//! The widen needs no arithmetic. E2M1's eight magnitudes are 0, 0.5, 1,
//! 1.5, 2, 3, 4 and 6, and every one of them — with either sign — is an F16
//! pattern whose low byte is zero. So a code's F16 high byte comes from one
//! byte shuffle of a sixteen-entry table, `<< 8` completes the pattern, and
//! the F16→F32 cast is a single instruction wherever half-precision is
//! hardware (AArch64 v8.2+, x86 F16C). No exponent rebias, no mask, and no
//! encoding is special: FP4 has no infinity, no NaN and no subnormal that
//! this table does not already hold exactly.
//!
//! Every product is scaled before it is summed, so a row's total is
//! `Σ_i x_i · scale(i/16) · e2m1(w_i)` accumulated in lane order — one
//! accumulator per row, independent of how callers split the rows.

use super::W;
use core::simd::num::SimdFloat;
use core::simd::prelude::*;

/// Elements one E4M3 scale covers.
pub(crate) const BLOCK: usize = 16;

/// Activations biased to U8 for NVFP4 W4A8, one scale per block.
pub struct Quantized<'a> {
    pub biased: &'a [u8],
    pub scales: &'a [f32],
}

/// Bytes and scales a row of `k` activations quantizes into.
pub const fn quantized_len(k: usize) -> (usize, usize) {
    (k, k.div_ceil(BLOCK))
}

/// Quantize activations per block, independently of which VNNI width runs
/// the product. The byte grid is shared by 256-bit and 512-bit kernels.
pub fn quantize(x: &[f32], biased: &mut [u8], scales: &mut [f32]) {
    for (b, block) in x.chunks(BLOCK).enumerate() {
        let peak = block.iter().fold(0.0_f32, |a, v| a.max(v.abs()));
        let scale = peak / 127.0;
        scales[b] = scale;
        let recip = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (i, &v) in block.iter().enumerate() {
            let scaled = v * recip;
            let rounded = if scaled >= 0.0 {
                scaled + 0.5
            } else {
                scaled - 0.5
            } as i32;
            biased[b * BLOCK + i] = (rounded.clamp(-127, 127) + 128) as u8;
        }
    }
}

/// The sixteen E2M1 codes as F16 high bytes, sign in bit 3 of the code.
pub(crate) const F16_HIGH: [u8; 16] = [
    0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46, // +0, .5, 1, 1.5, 2, 3, 4, 6
    0x80, 0xB8, 0xBC, 0xBE, 0xC0, 0xC2, 0xC4, 0xC6, // the same, negated
];

/// Every E2M1 code's exact value: the same sixteen [`F16_HIGH`] names.
const VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// One E2M1 code's exact value.
#[inline(always)]
pub(crate) fn e2m1(code: u8) -> f32 {
    VALUES[usize::from(code & 0x0F)]
}

/// `n` packed E2M1 weights, unscaled, in element order.
pub(crate) fn e2m1_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() * 2 >= n && dst.len() >= n);
    for (i, value) in dst[..n].iter_mut().enumerate() {
        let byte = src[i / 2];
        *value = e2m1(if i % 2 == 0 { byte } else { byte >> 4 });
    }
}

/// Sixteen codes from eight packed bytes, in element order: the even
/// element's low nibble before the odd element's high one. Interleaving the
/// codes rather than their decoded values leaves one dynamic shuffle, or
/// none at all.
#[inline(always)]
fn codes(bytes: &[u8]) -> Simd<u8, BLOCK> {
    let mut packed = [0u8; BLOCK];
    packed[..8].copy_from_slice(&bytes[..8]);
    let packed = Simd::<u8, BLOCK>::from_array(packed);
    let (codes, _) = (packed & Simd::splat(0x0F)).interleave(packed >> Simd::splat(4));
    codes
}

/// Sixteen weights by table: each code's F16 high byte, then one widen.
#[inline(always)]
fn block(bytes: &[u8]) -> Simd<f32, BLOCK> {
    let table = Simd::<u8, BLOCK>::from_array(F16_HIGH);
    let high = table.swizzle_dyn(codes(bytes));
    Simd::<f16, BLOCK>::from_bits(high.cast::<u16>() << Simd::splat(8)).cast()
}

/// Sixteen weights by construction: E2M1 is regular enough to build the
/// F32 patterns outright, with no table and no dynamic shuffle.
///
/// The exponent field `e` and mantissa bit `m` of a magnitude are its bits
/// 2:1 and 0. A normal value is `(1 + m/2) · 2^(e-1)`, so its F32 exponent
/// is `126 + e` and `m` lands in mantissa bit 22. The one subnormal, `e = 0`
/// and `m = 1`, is 0.5 — which `126 << 23` already is, so only a zero
/// magnitude needs clearing. The sign moves from bit 3 to bit 31, which
/// keeps `-0` as `-0`.
#[inline(always)]
fn block_bits(bytes: &[u8]) -> Simd<f32, BLOCK> {
    let c = codes(bytes).cast::<u32>();
    let zero = Simd::<u32, BLOCK>::splat(0);
    let magnitude = c & Simd::splat(0x7);
    let exponent = magnitude >> Simd::splat(1);
    let mantissa = exponent
        .simd_ne(zero)
        .select((magnitude & Simd::splat(1)) << Simd::splat(22), zero);
    let positive = magnitude.simd_ne(zero).select(
        ((exponent + Simd::splat(126)) << Simd::splat(23)) | mantissa,
        zero,
    );
    Simd::from_bits(positive | ((c & Simd::splat(8)) << Simd::splat(28)))
}

/// `Σ_i x_i · scale(i / 16) · e2m1(w_i)` over `n` elements: one row of an
/// NVFP4 weight matrix against an F32 activation row. `w` packs two weights
/// per byte and `scales` holds one E4M3 byte per sixteen.
pub(crate) fn vec_dot_nvfp4_f32(w: &[u8], scales: &[u8], x: &[f32], n: usize) -> f32 {
    dot::<false>(w, scales, x, n)
}

/// [`vec_dot_nvfp4_f32`] with the table-free decoder: the same value for
/// every input, by construction rather than lookup.
pub(crate) fn vec_dot_nvfp4_bits_f32(w: &[u8], scales: &[u8], x: &[f32], n: usize) -> f32 {
    dot::<true>(w, scales, x, n)
}

/// Blocks whose scales are widened in one pass: 4096 elements of weights,
/// 1 KiB of F32 scales on the stack.
const SCALE_RUN: usize = 256;

#[inline(always)]
fn dot<const BITS: bool>(w: &[u8], scales: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(w.len() * 2 >= n && x.len() >= n);
    debug_assert!(scales.len() * BLOCK >= n);
    let mut acc = Simd::<f32, BLOCK>::splat(0.0);
    let whole = n / BLOCK;
    // Widening one E4M3 scale at a time is a branchy scalar conversion per
    // sixteen weights — more work than the weights themselves. Widen a run
    // of them at once instead, vectorized, and read the results back as
    // plain loads.
    let mut widened = [0.0_f32; SCALE_RUN];
    for run in (0..whole).step_by(SCALE_RUN) {
        let count = SCALE_RUN.min(whole - run);
        crate::arch::f8::e4m3_to_f32(&scales[run..], &mut widened, count);
        for (b, &scale) in (run..run + count).zip(&widened[..count]) {
            let bytes = &w[b * BLOCK / 2..];
            let decoded = if BITS {
                block_bits(bytes)
            } else {
                block(bytes)
            };
            acc += decoded * Simd::splat(scale) * Simd::from_slice(&x[b * BLOCK..]);
        }
    }
    let mut total = acc.reduce_sum();
    // A tail shorter than one block keeps the same order, scaled by the
    // block it falls in.
    let rest = n - whole * BLOCK;
    if rest != 0 {
        let scale = crate::fp8::e4m3_to_f32(scales[whole]);
        let mut tail = 0.0;
        for i in whole * BLOCK..n {
            let byte = w[i / 2];
            tail += x[i] * e2m1(if i % 2 == 0 { byte } else { byte >> 4 });
        }
        total += tail * scale;
    }
    total
}

/// One row against `T` activation rows at once: each block of sixteen
/// weights is widened and scaled once and used by every token, where `T`
/// separate dots would widen it `T` times. Each token's products are summed
/// in the same order [`vec_dot_nvfp4_f32`] uses, so `out[t]` is exactly what
/// that function returns for token `t`.
pub(crate) fn vec_dot_nvfp4_tokens<const T: usize>(
    w: &[u8],
    scales: &[u8],
    xs: &[&[f32]; T],
    n: usize,
    out: &mut [f32; T],
) {
    tokens::<T, false>(w, scales, xs, n, out)
}

/// [`vec_dot_nvfp4_tokens`] over either decoder.
#[inline(always)]
pub(crate) fn tokens<const T: usize, const BITS: bool>(
    w: &[u8],
    scales: &[u8],
    xs: &[&[f32]; T],
    n: usize,
    out: &mut [f32; T],
) {
    let mut acc = [Simd::<f32, BLOCK>::splat(0.0); T];
    let whole = n / BLOCK;
    let mut widened = [0.0_f32; SCALE_RUN];
    for run in (0..whole).step_by(SCALE_RUN) {
        let count = SCALE_RUN.min(whole - run);
        crate::arch::f8::e4m3_to_f32(&scales[run..], &mut widened, count);
        for (b, &scale) in (run..run + count).zip(&widened[..count]) {
            let bytes = &w[b * BLOCK / 2..];
            let decoded = if BITS {
                block_bits(bytes)
            } else {
                block(bytes)
            };
            let weights = decoded * Simd::splat(scale);
            for (acc, x) in acc.iter_mut().zip(xs) {
                *acc += weights * Simd::from_slice(&x[b * BLOCK..]);
            }
        }
    }
    for (out, acc) in out.iter_mut().zip(acc) {
        *out = acc.reduce_sum();
    }
    let rest = n - whole * BLOCK;
    if rest != 0 {
        let scale = crate::fp8::e4m3_to_f32(scales[whole]);
        for (out, x) in out.iter_mut().zip(xs) {
            let mut tail = 0.0;
            for i in whole * BLOCK..n {
                let byte = w[i / 2];
                tail += x[i] * e2m1(if i % 2 == 0 { byte } else { byte >> 4 });
            }
            *out += tail * scale;
        }
    }
}

/// NVFP4 weight rows against `x`: `out[r]` is row `r`'s dot. Rows are
/// `x.len()` elements each, packed two per byte, with `x.len() / 16` scales
/// per row.
pub(crate) fn nvfp4_f32(weights: &[u8], scales: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    if k == 0 {
        out.fill(0.0);
        return;
    }
    let (row_bytes, row_scales) = (k.div_ceil(2), k.div_ceil(BLOCK));
    debug_assert!(weights.len() >= out.len() * row_bytes);
    debug_assert!(scales.len() >= out.len() * row_scales);
    for (r, y) in out.iter_mut().enumerate() {
        *y = vec_dot_nvfp4_f32(&weights[r * row_bytes..], &scales[r * row_scales..], x, k);
    }
}

/// Lane width the tier compiles to, for callers sizing work.
#[allow(dead_code)]
pub(crate) const LANES: usize = W;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// The sixteen codes, by definition: magnitude from the exponent and
    /// mantissa, sign from bit 3.
    fn reference(code: u8) -> f32 {
        let magnitude = match code & 0x07 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        if code & 0x08 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    #[test]
    fn every_code_widens_exactly() {
        for code in 0..16u8 {
            assert_eq!(e2m1(code), reference(code), "code {code}");
        }
        // The high nibble is the odd element, the low nibble the even one.
        let packed = [0x21u8, 0xC3];
        let mut out = [0.0; 4];
        e2m1_to_f32(&packed, &mut out, 4);
        assert_eq!(out, [0.5, 1.0, 1.5, -2.0]);
    }

    #[test]
    fn both_decoders_agree_on_every_code() {
        // One block holding all sixteen codes, twice.
        let packed: Vec<u8> = (0..16u8).map(|i| (i & 0x0F) | (i << 4)).collect();
        let by_table = block(&packed);
        let by_bits = block_bits(&packed);
        assert_eq!(by_table.to_array(), by_bits.to_array());
        for (i, value) in by_bits.to_array().iter().enumerate() {
            let byte = packed[i / 2];
            let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            assert_eq!(value.to_bits(), reference(code).to_bits(), "code {code}");
        }
    }

    #[test]
    fn many_tokens_match_one_token_bit_for_bit() {
        let k = 2688;
        let packed: Vec<u8> = (0..k / 2).map(|i| (i * 53 % 251) as u8).collect();
        let scales: Vec<u8> = (0..k / BLOCK).map(|i| (58 + i % 15) as u8).collect();
        let rows: Vec<Vec<f32>> = (0..4)
            .map(|t| {
                (0..k)
                    .map(|i| ((i * (17 + t) % 97) as f32 - 48.0) / 48.0)
                    .collect()
            })
            .collect();
        let xs: [&[f32]; 4] = [&rows[0], &rows[1], &rows[2], &rows[3]];
        let mut together = [0.0_f32; 4];
        vec_dot_nvfp4_tokens::<4>(&packed, &scales, &xs, k, &mut together);
        for (t, x) in xs.iter().enumerate() {
            let alone = vec_dot_nvfp4_f32(&packed, &scales, x, k);
            assert_eq!(together[t], alone, "token {t}");
        }
    }

    #[test]
    fn dots_match_the_scalar_reference_and_split_the_same() {
        let k = 2688;
        let packed: Vec<u8> = (0..k / 2).map(|i| (i * 37 % 251) as u8).collect();
        let scales: Vec<u8> = (0..k / BLOCK).map(|i| (60 + i % 12) as u8).collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 29 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let mut expected = 0.0_f32;
        for (b, &byte) in scales.iter().enumerate() {
            let scale = crate::fp8::e4m3_to_f32(byte);
            let mut sum = 0.0_f32;
            for i in b * BLOCK..(b + 1) * BLOCK {
                let byte = packed[i / 2];
                let w = reference(if i % 2 == 0 { byte } else { byte >> 4 });
                sum += x[i] * w * scale;
            }
            expected += sum;
        }
        let actual = vec_dot_nvfp4_f32(&packed, &scales, &x, k);
        assert!(
            (actual - expected).abs() <= 1e-3 * (1.0 + expected.abs()),
            "{actual} vs {expected}"
        );

        // Every row is its own serial computation: a two-row call gives
        // each row exactly what a one-row call gives it.
        let rows = 2;
        let mut both = vec![0.0; rows];
        let weights: Vec<u8> = packed.iter().chain(packed.iter().rev()).copied().collect();
        let all_scales: Vec<u8> = scales.iter().chain(scales.iter().rev()).copied().collect();
        nvfp4_f32(&weights, &all_scales, &x, &mut both);
        for r in 0..rows {
            let mut alone = [0.0; 1];
            nvfp4_f32(
                &weights[r * (k / 2)..(r + 1) * (k / 2)],
                &all_scales[r * (k / BLOCK)..(r + 1) * (k / BLOCK)],
                &x,
                &mut alone,
            );
            assert_eq!(both[r], alone[0], "row {r}");
        }
    }
}
