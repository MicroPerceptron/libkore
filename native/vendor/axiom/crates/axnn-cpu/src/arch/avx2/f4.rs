//! AVX2 + FMA + F16C E2M1 (OCP FP4) GEMV: weight rows in blocks of [`ROWS`].
//!
//! Per 32-byte step a row holds 64 weights. `vpshufb` looks up each
//! nibble's F16 high byte — the whole decode, since every E2M1 value is an
//! F16 pattern with a zero low byte — and the two halves interleave back
//! into element order with `vpunpcklbw`/`vpunpckhbw`. Each group of sixteen
//! then widens with `vpmovzxbw`, `<< 8` and `vcvtph2ps`, and the group's
//! E4M3 block scale multiplies it before one `vfmadd` per eight elements.
//!
//! A block walks the activations once for all of its rows, and each row
//! keeps two accumulators, so a block runs `2 × ROWS` independent chains.
//! Rows past the last whole block run the same loop as a block of one, and
//! each row's last `k % STEP` elements use the single-row dot, so a row's
//! result never depends on which block it fell in — and so never on how
//! callers split rows between threads.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

use crate::arch::portable::f4::{BLOCK, F16_HIGH, vec_dot_nvfp4_f32};

/// Rows per block.
const ROWS: usize = 4;
/// Elements per row per step: 32 packed bytes.
const STEP: usize = 64;

#[inline(always)]
unsafe fn hsum256(x: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}

/// Sixteen weights from sixteen F16 high bytes, as two vectors of eight.
#[inline(always)]
unsafe fn widen(high: __m128i) -> (__m256, __m256) {
    let words = _mm256_slli_epi16::<8>(_mm256_cvtepu8_epi16(high));
    (
        _mm256_cvtph_ps(_mm256_castsi256_si128(words)),
        _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(words)),
    )
}

/// NVFP4 weight rows against `x`: `out[r]` is row `r`'s dot. Rows are
/// `x.len()` elements each, two weights per byte, with one E4M3 scale per
/// sixteen.
pub(crate) fn nvfp4_f32(weights: &[u8], scales: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    if k == 0 {
        out.fill(0.0);
        return;
    }
    let (row_bytes, row_scales) = (k.div_ceil(2), k.div_ceil(BLOCK));
    debug_assert!(weights.len() >= out.len() * row_bytes);
    debug_assert!(scales.len() >= out.len() * row_scales);
    let body = k - k % STEP;
    let (blocks, rest) = out.as_chunks_mut::<ROWS>();
    for (b, ys) in blocks.iter_mut().enumerate() {
        let (w, s) = (b * ROWS * row_bytes, b * ROWS * row_scales);
        rows::<ROWS>(
            &weights[w..],
            &scales[s..],
            x,
            row_bytes,
            row_scales,
            body,
            ys,
        );
    }
    let first = blocks.len() * ROWS;
    for (r, y) in (first..).zip(rest) {
        rows::<1>(
            &weights[r * row_bytes..],
            &scales[r * row_scales..],
            x,
            row_bytes,
            row_scales,
            body,
            core::array::from_mut(y),
        );
    }
}

/// Tokens [`nvfp4_f32_tokens`] serves per pass over a row's weights.
pub(crate) const TOKENS: usize = 4;

/// [`nvfp4_f32`] for up to [`TOKENS`] activation rows at once: `out[t][r]`
/// is row `r` against `xs[t]`. Each step of a row is widened and scaled
/// once and used by every token, where one-token calls widen it per token;
/// each token's products and sums are the ones [`nvfp4_f32`] makes, so
/// every result is the one-token result, bit for bit.
pub(crate) fn nvfp4_f32_tokens(
    weights: &[u8],
    scales: &[u8],
    xs: &[&[f32]],
    out: &mut [&mut [f32]],
) {
    match xs.len() {
        1 => tokens::<1>(weights, scales, xs, out),
        2 => tokens::<2>(weights, scales, xs, out),
        3 => tokens::<3>(weights, scales, xs, out),
        4 => tokens::<4>(weights, scales, xs, out),
        n => panic!("{n} tokens per pass"),
    }
}

fn tokens<const T: usize>(weights: &[u8], scales: &[u8], xs: &[&[f32]], out: &mut [&mut [f32]]) {
    let k = xs[0].len();
    let rows = out[0].len();
    debug_assert!(xs.iter().all(|x| x.len() == k) && out.iter().all(|o| o.len() == rows));
    if k == 0 {
        out.iter_mut().for_each(|o| o.fill(0.0));
        return;
    }
    let (row_bytes, row_scales) = (k.div_ceil(2), k.div_ceil(BLOCK));
    debug_assert!(weights.len() >= rows * row_bytes);
    debug_assert!(scales.len() >= rows * row_scales);
    let body = k - k % STEP;
    let x: [*const f32; T] = core::array::from_fn(|t| xs[t].as_ptr());
    for r in 0..rows {
        let (w, s) = (&weights[r * row_bytes..], &scales[r * row_scales..]);
        // SAFETY: `w` and `s` hold a whole row, each `xs[t]` `k` elements,
        // and the body reads the first `body <= k` of each, whole steps.
        let sums = unsafe { row_tokens::<T>(w.as_ptr(), s.as_ptr(), x, body) };
        for t in 0..T {
            out[t][r] = sums[t]
                + vec_dot_nvfp4_f32(&w[body / 2..], &s[body / BLOCK..], &xs[t][body..], k - body);
        }
    }
}

/// One row's body against `T` activation rows: [`block`]'s arithmetic for
/// a block of one row, per token.
#[inline(always)]
unsafe fn row_tokens<const T: usize>(
    w: *const u8,
    s: *const u8,
    x: [*const f32; T],
    body: usize,
) -> [f32; T] {
    let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(F16_HIGH.as_ptr() as *const __m128i));
    let nibble = _mm256_set1_epi8(0x0F);
    let mut even = [_mm256_setzero_ps(); T];
    let mut odd = [_mm256_setzero_ps(); T];
    let mut i = 0;
    while i < body {
        let packed = _mm256_loadu_si256(w.add(i / 2) as *const __m256i);
        let low = _mm256_shuffle_epi8(table, _mm256_and_si256(packed, nibble));
        let high = _mm256_shuffle_epi8(
            table,
            _mm256_and_si256(_mm256_srli_epi16::<4>(packed), nibble),
        );
        let first = _mm256_unpacklo_epi8(low, high);
        let second = _mm256_unpackhi_epi8(low, high);
        let groups = [
            _mm256_castsi256_si128(first),
            _mm256_castsi256_si128(second),
            _mm256_extracti128_si256::<1>(first),
            _mm256_extracti128_si256::<1>(second),
        ];
        for (g, group) in groups.into_iter().enumerate() {
            let scale = _mm256_set1_ps(crate::fp8::e4m3_to_f32(*s.add(i / BLOCK + g)));
            let (lo, hi) = widen(group);
            let (lo, hi) = (_mm256_mul_ps(lo, scale), _mm256_mul_ps(hi, scale));
            for t in 0..T {
                let base = x[t].add(i + g * BLOCK);
                even[t] = _mm256_fmadd_ps(lo, _mm256_loadu_ps(base), even[t]);
                odd[t] = _mm256_fmadd_ps(hi, _mm256_loadu_ps(base.add(8)), odd[t]);
            }
        }
        i += STEP;
    }
    core::array::from_fn(|t| hsum256(_mm256_add_ps(even[t], odd[t])))
}

/// One block's rows: the vector body, then each row's tail.
fn rows<const R: usize>(
    weights: &[u8],
    scales: &[u8],
    x: &[f32],
    row_bytes: usize,
    row_scales: usize,
    body: usize,
    ys: &mut [f32; R],
) {
    let k = x.len();
    // SAFETY: `weights` holds R rows of `row_bytes`, `scales` R rows of
    // `row_scales`, and `x` k elements; the body reads the first `body`
    // elements of each row, which is a whole number of steps.
    let sums = unsafe {
        block::<R>(
            weights.as_ptr(),
            scales.as_ptr(),
            row_bytes,
            row_scales,
            x.as_ptr(),
            body,
        )
    };
    for (r, y) in ys.iter_mut().enumerate() {
        *y = sums[r]
            + vec_dot_nvfp4_f32(
                &weights[r * row_bytes + body / 2..],
                &scales[r * row_scales + body / BLOCK..],
                &x[body..],
                k - body,
            );
    }
}

/// `R` rows over the first `body` elements, `body` a multiple of [`STEP`].
#[inline(always)]
unsafe fn block<const R: usize>(
    w: *const u8,
    s: *const u8,
    row_bytes: usize,
    row_scales: usize,
    x: *const f32,
    body: usize,
) -> [f32; R] {
    let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(F16_HIGH.as_ptr() as *const __m128i));
    let nibble = _mm256_set1_epi8(0x0F);
    let mut even = [_mm256_setzero_ps(); R];
    let mut odd = [_mm256_setzero_ps(); R];
    let mut i = 0;
    while i < body {
        let xs = x.add(i);
        for r in 0..R {
            let p = w.add(r * row_bytes + i / 2);
            let packed = _mm256_loadu_si256(p as *const __m256i);
            // Each byte's low nibble is the even element, its high nibble
            // the odd one; one lookup each, then interleaved back into
            // element order within every 128-bit lane.
            let low = _mm256_shuffle_epi8(table, _mm256_and_si256(packed, nibble));
            let high = _mm256_shuffle_epi8(
                table,
                _mm256_and_si256(_mm256_srli_epi16::<4>(packed), nibble),
            );
            let first = _mm256_unpacklo_epi8(low, high);
            let second = _mm256_unpackhi_epi8(low, high);
            // Elements 0..15, 16..31, 32..47 and 48..63 of this step, each
            // group with its own scale.
            let groups = [
                _mm256_castsi256_si128(first),
                _mm256_castsi256_si128(second),
                _mm256_extracti128_si256::<1>(first),
                _mm256_extracti128_si256::<1>(second),
            ];
            for (g, group) in groups.into_iter().enumerate() {
                let at = i / BLOCK + g;
                let scale = _mm256_set1_ps(crate::fp8::e4m3_to_f32(*s.add(r * row_scales + at)));
                let (lo, hi) = widen(group);
                let base = xs.add(g * BLOCK);
                even[r] = _mm256_fmadd_ps(_mm256_mul_ps(lo, scale), _mm256_loadu_ps(base), even[r]);
                odd[r] = _mm256_fmadd_ps(
                    _mm256_mul_ps(hi, scale),
                    _mm256_loadu_ps(base.add(8)),
                    odd[r],
                );
            }
        }
        i += STEP;
    }
    let mut sums = [0.0; R];
    for r in 0..R {
        sums[r] = hsum256(_mm256_add_ps(even[r], odd[r]));
    }
    sums
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    /// The AVX2 tier sums two accumulator chains where portable sums
    /// sixteen lanes, so the two agree to F32 association, and every row is
    /// still its own serial computation.
    #[test]
    fn matches_the_portable_tier_and_splits_the_same() {
        let (k, rows) = (2688, 6);
        let packed: Vec<u8> = (0..rows * k / 2).map(|i| (i * 37 % 251) as u8).collect();
        let scales: Vec<u8> = (0..rows * k / 16).map(|i| (60 + i % 12) as u8).collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 29 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let mut all = vec![0.0; rows];
        super::nvfp4_f32(&packed, &scales, &x, &mut all);
        for r in 0..rows {
            let portable = crate::arch::portable::f4::vec_dot_nvfp4_f32(
                &packed[r * k / 2..],
                &scales[r * k / 16..],
                &x,
                k,
            );
            assert!(
                (all[r] - portable).abs() <= 1e-3 * (1.0 + portable.abs()),
                "row {r}: {} vs {portable}",
                all[r]
            );
            let mut alone = [0.0; 1];
            super::nvfp4_f32(
                &packed[r * k / 2..(r + 1) * k / 2],
                &scales[r * k / 16..(r + 1) * k / 16],
                &x,
                &mut alone,
            );
            assert_eq!(all[r], alone[0], "row {r} moved with its block");
        }
    }

    /// Several tokens per pass give each token its one-token result, bit
    /// for bit, with and without a tail past the last whole step.
    #[test]
    fn tokens_match_one_token_calls() {
        for k in [2688, 2720] {
            let rows = 7;
            let packed: Vec<u8> = (0..rows * k / 2).map(|i| (i * 53 % 251) as u8).collect();
            let scales: Vec<u8> = (0..rows * k / 16).map(|i| (52 + i % 20) as u8).collect();
            let xs: Vec<Vec<f32>> = (0..4)
                .map(|t| {
                    (0..k)
                        .map(|i| ((i * 31 + t * 17) % 97) as f32 / 48.0 - 1.0)
                        .collect()
                })
                .collect();
            for count in 1..=super::TOKENS {
                let x: Vec<&[f32]> = xs[..count].iter().map(|x| x.as_slice()).collect();
                let mut outs = vec![vec![f32::NAN; rows]; count];
                let mut refs: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
                super::nvfp4_f32_tokens(&packed, &scales, &x, &mut refs);
                for (t, out) in outs.iter().enumerate() {
                    let mut alone = vec![0.0; rows];
                    super::nvfp4_f32(&packed, &scales, &xs[t], &mut alone);
                    let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(out), bits(&alone), "k {k}, token {t} of {count}");
                }
            }
        }
    }
}
