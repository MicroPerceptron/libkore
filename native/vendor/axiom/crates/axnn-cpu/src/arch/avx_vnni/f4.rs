//! AVX-VNNI E2M1 GEMV with quantized activations (W4A8).
//!
//! Every E2M1 value is a multiple of one half — 0, 0.5, 1, 1.5, 2, 3, 4, 6 —
//! so `q = 2 · value` is an exact I8 and `vpshufb` decodes a nibble straight
//! into it, with the halving folded into the block scale. Quantizing the
//! activations to U8 as well lets `vpdpbusd` do the work: 32 products per
//! instruction against an F32 FMA's 8.
//!
//! `vpdpbusd` wants its left operand unsigned, so activations are stored
//! biased by 128 and the bias is taken back with a second `vpdpbusd` against
//! a vector of ones: `Σ (a+128)·q − 128·Σ q`. Both give one I32 lane per
//! four elements, and — the point of the arrangement — a block's sixteen
//! elements occupy four whole lanes, so its scale can multiply those lanes
//! where they sit. Nothing is reduced horizontally until the row ends, and
//! per 32 elements the loop is two `vpdpbusd`, a shift, a subtract, one
//! convert and one FMA.
//!
//! Unlike the float tiers this is an approximation: the weights stay exact,
//! the activations do not. Callers wanting the checkpoint's own arithmetic
//! use [`crate::arch::avx2::f4`].
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

use crate::arch::portable::f4::BLOCK;
pub use crate::arch::portable::f4::{Quantized, quantize, quantized_len};

/// `2 · e2m1(code)`, exactly, for each of the sixteen codes.
const DOUBLED: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// NVFP4 weight rows against quantized activations: `out[r]` is row `r`'s
/// dot. Rows are `x` elements each, two weights per byte, one E4M3 scale
/// per sixteen.
pub fn nvfp4_q8_f32(weights: &[u8], scales: &[u8], x: &Quantized<'_>, out: &mut [f32]) {
    let k = x.biased.len();
    if k == 0 {
        out.fill(0.0);
        return;
    }
    let (row_bytes, row_scales) = (k.div_ceil(2), k.div_ceil(BLOCK));
    debug_assert!(weights.len() >= out.len() * row_bytes);
    debug_assert!(scales.len() >= out.len() * row_scales);
    debug_assert!(x.scales.len() >= row_scales);
    // Two blocks per step; a trailing odd block falls to the scalar tail.
    let steps = k / (2 * BLOCK);
    for (r, y) in out.iter_mut().enumerate() {
        // SAFETY: the row holds `row_bytes` weights and `row_scales`
        // scales, and the step reads 32 activations per pair of blocks,
        // all within the lengths asserted above.
        let mut total = unsafe {
            row(
                weights.as_ptr().add(r * row_bytes),
                scales.as_ptr().add(r * row_scales),
                x,
                steps,
            )
        };
        for b in steps * 2..row_scales {
            let scale = crate::fp8::e4m3_to_f32(scales[r * row_scales + b]) * x.scales[b] / 2.0;
            let mut sum = 0.0;
            for i in b * BLOCK..k.min((b + 1) * BLOCK) {
                let byte = weights[r * row_bytes + i / 2];
                let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let a = i32::from(x.biased[i]) - 128;
                sum += (a * i32::from(DOUBLED[code as usize])) as f32;
            }
            total += sum * scale;
        }
        *y = total;
    }
}

#[target_feature(enable = "avx2,avxvnni,fma")]
unsafe fn row(w: *const u8, s: *const u8, x: &Quantized<'_>, steps: usize) -> f32 {
    let table = _mm_loadu_si128(DOUBLED.as_ptr() as *const __m128i);
    let nibble = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi8(1);
    let zero = _mm256_setzero_si256();
    // Four independent chains: one FMA per step would stall on the last
    // step's latency, as the float tiers' row blocking avoids.
    let mut chains = [_mm256_setzero_ps(); 4];
    for step in 0..steps {
        // Sixteen bytes are 32 weights: one lookup per nibble half, then
        // interleaved back into element order.
        let packed = _mm_loadu_si128(w.add(step * BLOCK) as *const __m128i);
        let low = _mm_shuffle_epi8(table, _mm_and_si128(packed, nibble));
        let high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16::<4>(packed), nibble));
        let weights = _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high));
        let biased = _mm256_loadu_si256(x.biased.as_ptr().add(step * 2 * BLOCK) as *const __m256i);
        // Σ (a + 128)·q − 128·Σ q, four elements to an I32 lane.
        let product = _mm256_dpbusd_epi32(zero, biased, weights);
        let bias = _mm256_dpbusd_epi32(zero, ones, weights);
        let exact = _mm256_sub_epi32(product, _mm256_slli_epi32::<7>(bias));
        // Each block owns four whole lanes, so its scale lands there.
        let (first, second) = (step * 2, step * 2 + 1);
        let scale = _mm256_set_m128(
            _mm_set1_ps(crate::fp8::e4m3_to_f32(*s.add(second)) * x.scales[second] * 0.5),
            _mm_set1_ps(crate::fp8::e4m3_to_f32(*s.add(first)) * x.scales[first] * 0.5),
        );
        let chain = &mut chains[step % 4];
        *chain = _mm256_fmadd_ps(_mm256_cvtepi32_ps(exact), scale, *chain);
    }
    let acc = _mm256_add_ps(
        _mm256_add_ps(chains[0], chains[1]),
        _mm256_add_ps(chains[2], chains[3]),
    );
    let sum = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Quantized activations cost accuracy, not correctness: the result
    /// tracks the float tier to the error a per-sixteen U8 grid allows.
    #[test]
    fn tracks_the_float_tier_within_the_activation_grid() {
        let (k, rows) = (2688, 5);
        let packed: Vec<u8> = (0..rows * k / 2).map(|i| (i * 37 % 251) as u8).collect();
        let scales: Vec<u8> = (0..rows * k / BLOCK).map(|i| (60 + i % 12) as u8).collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 29 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let (bytes, blocks) = quantized_len(k);
        let mut biased = vec![0u8; bytes];
        let mut activation_scales = vec![0.0; blocks];
        quantize(&x, &mut biased, &mut activation_scales);
        let quantized = Quantized {
            biased: &biased,
            scales: &activation_scales,
        };
        let mut actual = vec![0.0; rows];
        nvfp4_q8_f32(&packed, &scales, &quantized, &mut actual);
        for (r, got) in actual.iter().enumerate() {
            let want = crate::arch::portable::f4::vec_dot_nvfp4_f32(
                &packed[r * k / 2..],
                &scales[r * k / BLOCK..],
                &x,
                k,
            );
            // One part in 127 per element, averaged over 2688 of them.
            assert!(
                (got - want).abs() <= 0.02 * (1.0 + want.abs()),
                "row {r}: {got} vs {want}"
            );
        }
    }
}
