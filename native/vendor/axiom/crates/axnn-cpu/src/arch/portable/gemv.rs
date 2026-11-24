//! GEMV: many weight rows against one F32 activation row — the decode
//! shape of every projection.
//!
//! `out[r] = Σ_i decode(weights[r][i]) · x[i]` for each of `out.len()`
//! rows, packed back to back at `x.len()` elements each. The portable
//! tier is one dot product per row; tiers that block rows share each
//! activation load across the block and keep independent accumulator
//! chains per row. Every row is still one task's serial computation, so
//! a result never depends on how callers split the rows.

use core::simd::num::SimdFloat;
use core::simd::{Simd, f32x8};

/// E4M3 weights (unscaled) against `x`.
pub(crate) fn e4m3_f32(weights: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    if k == 0 {
        out.fill(0.0);
        return;
    }
    debug_assert!(weights.len() >= out.len() * k);
    for (row, y) in weights.chunks_exact(k).zip(out) {
        *y = crate::arch::f8::vec_dot_e4m3_f32(row, x, k);
    }
}

/// Little-endian BF16 weights against `x`.
pub(crate) fn bf16_f32(weights: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    if k == 0 {
        out.fill(0.0);
        return;
    }
    debug_assert!(weights.len() >= out.len() * 2 * k);
    for (row, y) in weights.chunks_exact(2 * k).zip(out) {
        *y = dot_bf16_f32(row, x);
    }
}

/// `acc[i] += a * e4m3(row[i])` over `acc.len()` elements: one transposed
/// weight row scaled into an accumulator.
pub(crate) fn axpy_e4m3_f32(row: &[u8], a: f32, acc: &mut [f32]) {
    debug_assert!(row.len() >= acc.len());
    let mut tile = [0.0_f32; 64];
    for (acc, row) in acc.chunks_mut(64).zip(row.chunks(64)) {
        let n = acc.len();
        crate::arch::f8::e4m3_to_f32(&row[..n], &mut tile[..n], n);
        for (value, &w) in acc.iter_mut().zip(&tile[..n]) {
            *value += a * w;
        }
    }
}

/// Decode one little-endian BF16 value exactly.
#[inline(always)]
pub(crate) fn bf16(bytes: [u8; 2]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes(bytes)) << 16)
}

/// `sum_i x[i] * bf16(w[i])` with eight F32 lanes, then the tail.
pub(crate) fn dot_bf16_f32(weights: &[u8], x: &[f32]) -> f32 {
    let (words, word_tail) = weights.as_chunks::<16>();
    let (lanes, lane_tail) = x.as_chunks::<8>();
    let mut sum = f32x8::splat(0.0);
    for (word, lane) in words.iter().zip(lanes) {
        let pairs = word.as_chunks::<2>().0;
        let bits: Simd<u32, 8> = Simd::from_array(core::array::from_fn(|i| {
            u32::from(u16::from_le_bytes(pairs[i]))
        }));
        sum += f32x8::from_bits(bits << 16) * f32x8::from_array(*lane);
    }
    let mut total = sum.reduce_sum();
    for (pair, value) in word_tail.as_chunks::<2>().0.iter().zip(lane_tail) {
        total += bf16(*pair) * value;
    }
    total
}
