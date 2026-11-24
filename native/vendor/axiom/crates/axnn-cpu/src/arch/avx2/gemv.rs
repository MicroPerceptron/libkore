//! AVX2 + FMA + F16C GEMV: weight rows in blocks of [`ROWS`].
//!
//! A row-at-a-time dot is one FMA chain per row and reloads the
//! activations for every row. Here a block of rows walks the activations
//! together: each step loads its activations once for the whole block,
//! and every row keeps two accumulators, so the block runs `2 × ROWS`
//! independent chains. Rows past the last whole block run the same loop
//! as a block of one, so a row's result never depends on which block it
//! fell in, and so never on how callers split rows between threads. Each
//! row's last `k % STEP` elements use the single-row dots.
//!
//! * **E4M3** widens to the same F16 patterns `f8::vec_dot_e4m3_f32`
//!   makes (exactly value / 256, then `vcvtph2ps`), here as sign-extended
//!   words `<< 7` with bit 14 cleared, 32 bytes per row per step. The NaN patch leaves the loop: the block keeps a
//!   running byte maximum of `w & 0x7F`, and only when that reaches the
//!   NaN encoding are its rows checked one by one. A row holding it is
//!   NaN, as the patched dot would make it.
//! * **BF16** needs no convert at all: of each 32-bit pair of weights,
//!   `<< 16` is the even element's F32 pattern and `& 0xFFFF0000` the odd
//!   one's. The activations are split into even and odd lanes to match,
//!   once per step for the whole block.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

use crate::arch::f8;

/// Rows per block.
const ROWS: usize = 4;
/// `× 2^8`, folding out the E4M3 widen's `/ 256`.
const E4M3_SCALE: f32 = 256.0;
/// Clears the sign that sign extension leaves in bit 14 after `<< 7`.
const E4M3_CLR: i16 = 0xBFFFu16 as i16;

#[inline(always)]
unsafe fn hsum256(x: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}

/// E4M3 weights (unscaled) against `x`.
pub(crate) fn e4m3_f32(weights: &[u8], x: &[f32], out: &mut [f32]) {
    const STEP: usize = 32;
    let k = x.len();
    if k == 0 {
        out.fill(0.0);
        return;
    }
    assert!(weights.len() >= out.len() * k);
    let body = k - k % STEP;
    let (blocks, rest) = out.as_chunks_mut::<ROWS>();
    for (b, ys) in blocks.iter_mut().enumerate() {
        e4m3_rows(&weights[b * ROWS * k..(b + 1) * ROWS * k], x, body, ys);
    }
    let first = blocks.len() * ROWS;
    for (r, y) in (first..).zip(rest) {
        e4m3_rows::<1>(
            &weights[r * k..(r + 1) * k],
            x,
            body,
            core::array::from_mut(y),
        );
    }
}

/// `R` rows of `x.len()` bytes each, `body` of them in blocked steps.
#[inline(always)]
fn e4m3_rows<const R: usize>(block: &[u8], x: &[f32], body: usize, ys: &mut [f32; R]) {
    let k = x.len();
    // SAFETY: `block` holds R rows of `k` bytes and `x` holds `k`
    // elements; the loop reads the first `body <= k` of each.
    let (sums, nan) = unsafe { e4m3_block::<R>(block.as_ptr(), k, x.as_ptr(), body) };
    for (r, y) in ys.iter_mut().enumerate() {
        let row = &block[r * k..(r + 1) * k];
        *y = if nan && row[..body].iter().any(|&b| b & 0x7F == 0x7F) {
            f32::NAN
        } else {
            sums[r] * E4M3_SCALE + f8::vec_dot_e4m3_f32(&row[body..], &x[body..], k - body)
        };
    }
}

/// One block's sums over the first `body` elements (still `/ 256`), and
/// whether any of its bytes is the NaN encoding.
#[inline(always)]
unsafe fn e4m3_block<const R: usize>(
    w: *const u8,
    k: usize,
    x: *const f32,
    body: usize,
) -> ([f32; R], bool) {
    let clr = _mm256_set1_epi16(E4M3_CLR);
    let magnitude = _mm256_set1_epi8(0x7F);
    let mut even = [_mm256_setzero_ps(); R];
    let mut odd = [_mm256_setzero_ps(); R];
    let mut peak = _mm256_setzero_si256();
    let mut i = 0;
    while i < body {
        let xs = x.add(i);
        for r in 0..R {
            let p = w.add(r * k + i);
            // The NaN check takes the whole step at once.
            let step = _mm256_loadu_si256(p as *const __m256i);
            peak = _mm256_max_epu8(peak, _mm256_and_si256(step, magnitude));
            // Each half of the step as 16 sign-extended words, `<< 7` then
            // clear bit 14: exactly value / 256 as F16, in element order.
            // Loading each half on its own keeps one half live at a time,
            // where unpacking the 32 bytes in-lane holds both.
            for (half, acc) in [(0, &mut even[r]), (16, &mut odd[r])] {
                let words = _mm256_cvtepi8_epi16(_mm_loadu_si128(p.add(half) as *const __m128i));
                let f16 = _mm256_and_si256(_mm256_slli_epi16::<7>(words), clr);
                let lo = _mm256_cvtph_ps(_mm256_castsi256_si128(f16));
                let hi = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(f16));
                *acc = _mm256_fmadd_ps(lo, _mm256_loadu_ps(xs.add(half)), *acc);
                *acc = _mm256_fmadd_ps(hi, _mm256_loadu_ps(xs.add(half + 8)), *acc);
            }
        }
        i += 32;
    }
    let nan = _mm256_movemask_epi8(_mm256_cmpeq_epi8(peak, magnitude)) != 0;
    let mut sums = [0.0; R];
    for r in 0..R {
        sums[r] = hsum256(_mm256_add_ps(even[r], odd[r]));
    }
    (sums, nan)
}

/// Tokens [`e4m3_f32_tokens`] serves per pass over a row's weights.
pub(crate) const TOKENS: usize = 4;

/// [`e4m3_f32`] for up to [`TOKENS`] activation rows at once: `out[t][r]`
/// is row `r` against `xs[t]`. Each step of a row widens once for every
/// token, where one-token calls widen it per token; each token's products
/// and sums are the ones [`e4m3_f32`] makes, so every result is the
/// one-token result, bit for bit.
pub(crate) fn e4m3_f32_tokens(weights: &[u8], xs: &[&[f32]], out: &mut [&mut [f32]]) {
    match xs.len() {
        1 => e4m3_tokens::<1>(weights, xs, out),
        2 => e4m3_tokens::<2>(weights, xs, out),
        3 => e4m3_tokens::<3>(weights, xs, out),
        4 => e4m3_tokens::<4>(weights, xs, out),
        n => panic!("{n} tokens per pass"),
    }
}

fn e4m3_tokens<const T: usize>(weights: &[u8], xs: &[&[f32]], out: &mut [&mut [f32]]) {
    const STEP: usize = 32;
    let k = xs[0].len();
    let rows = out[0].len();
    debug_assert!(xs.iter().all(|x| x.len() == k) && out.iter().all(|o| o.len() == rows));
    if k == 0 {
        out.iter_mut().for_each(|o| o.fill(0.0));
        return;
    }
    assert!(weights.len() >= rows * k);
    let body = k - k % STEP;
    let x: [*const f32; T] = core::array::from_fn(|t| xs[t].as_ptr());
    for r in 0..rows {
        let row = &weights[r * k..(r + 1) * k];
        // SAFETY: `row` holds `k` bytes and each `xs[t]` `k` elements; the
        // loop reads the first `body <= k` of each.
        let (sums, nan) = unsafe { e4m3_row_tokens::<T>(row.as_ptr(), x, body) };
        let nan = nan && row[..body].iter().any(|&b| b & 0x7F == 0x7F);
        for t in 0..T {
            out[t][r] = if nan {
                f32::NAN
            } else {
                sums[t] * E4M3_SCALE + f8::vec_dot_e4m3_f32(&row[body..], &xs[t][body..], k - body)
            };
        }
    }
}

/// One row's body against `T` activation rows: [`e4m3_block`]'s
/// arithmetic for a block of one row, per token.
#[inline(always)]
unsafe fn e4m3_row_tokens<const T: usize>(
    w: *const u8,
    x: [*const f32; T],
    body: usize,
) -> ([f32; T], bool) {
    let clr = _mm256_set1_epi16(E4M3_CLR);
    let magnitude = _mm256_set1_epi8(0x7F);
    let mut even = [_mm256_setzero_ps(); T];
    let mut odd = [_mm256_setzero_ps(); T];
    let mut peak = _mm256_setzero_si256();
    let mut i = 0;
    while i < body {
        let p = w.add(i);
        let step = _mm256_loadu_si256(p as *const __m256i);
        peak = _mm256_max_epu8(peak, _mm256_and_si256(step, magnitude));
        for half in [0, 16] {
            let words = _mm256_cvtepi8_epi16(_mm_loadu_si128(p.add(half) as *const __m128i));
            let f16 = _mm256_and_si256(_mm256_slli_epi16::<7>(words), clr);
            let lo = _mm256_cvtph_ps(_mm256_castsi256_si128(f16));
            let hi = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(f16));
            for t in 0..T {
                let acc = if half == 0 { &mut even[t] } else { &mut odd[t] };
                let xs = x[t].add(i + half);
                *acc = _mm256_fmadd_ps(lo, _mm256_loadu_ps(xs), *acc);
                *acc = _mm256_fmadd_ps(hi, _mm256_loadu_ps(xs.add(8)), *acc);
            }
        }
        i += 32;
    }
    let nan = _mm256_movemask_epi8(_mm256_cmpeq_epi8(peak, magnitude)) != 0;
    (
        core::array::from_fn(|t| hsum256(_mm256_add_ps(even[t], odd[t]))),
        nan,
    )
}

/// Little-endian BF16 weights against `x`.
pub(crate) fn bf16_f32(weights: &[u8], x: &[f32], out: &mut [f32]) {
    const STEP: usize = 16;
    let k = x.len();
    if k == 0 {
        out.fill(0.0);
        return;
    }
    let row_bytes = 2 * k;
    assert!(weights.len() >= out.len() * row_bytes);
    let body = k - k % STEP;
    let (blocks, rest) = out.as_chunks_mut::<ROWS>();
    for (b, ys) in blocks.iter_mut().enumerate() {
        let block = &weights[b * ROWS * row_bytes..(b + 1) * ROWS * row_bytes];
        bf16_rows(block, x, body, ys);
    }
    let first = blocks.len() * ROWS;
    for (r, y) in (first..).zip(rest) {
        let row = &weights[r * row_bytes..(r + 1) * row_bytes];
        bf16_rows::<1>(row, x, body, core::array::from_mut(y));
    }
}

/// `R` rows of `2 * x.len()` bytes each, `body` elements in blocked steps.
#[inline(always)]
fn bf16_rows<const R: usize>(block: &[u8], x: &[f32], body: usize, ys: &mut [f32; R]) {
    let row_bytes = 2 * x.len();
    // SAFETY: `block` holds R rows of `row_bytes` bytes and `x` holds
    // `x.len()` elements; the loop reads the first `body` of each.
    let sums = unsafe { bf16_block::<R>(block.as_ptr(), row_bytes, x.as_ptr(), body) };
    for (r, y) in ys.iter_mut().enumerate() {
        let tail = &block[r * row_bytes + 2 * body..(r + 1) * row_bytes];
        *y = sums[r] + crate::arch::portable::gemv::dot_bf16_f32(tail, &x[body..]);
    }
}

#[inline(always)]
unsafe fn bf16_block<const R: usize>(
    w: *const u8,
    row_bytes: usize,
    x: *const f32,
    body: usize,
) -> [f32; R] {
    let high = _mm256_set1_epi32(0xFFFF_0000u32 as i32);
    let mut even = [_mm256_setzero_ps(); R];
    let mut odd = [_mm256_setzero_ps(); R];
    let mut i = 0;
    while i < body {
        // Regroup to x0..x3 | x8..x11 and x4..x7 | x12..x15, then split by
        // parity: x0 x2 x4 x6 | x8 .. x14 and x1 .. x7 | x9 .. x15, matching
        // the weights' pairs within each 128-bit lane.
        let a = _mm256_loadu_ps(x.add(i));
        let b = _mm256_loadu_ps(x.add(i + 8));
        let c = _mm256_permute2f128_ps::<0x20>(a, b);
        let d = _mm256_permute2f128_ps::<0x31>(a, b);
        let xe = _mm256_shuffle_ps::<0x88>(c, d);
        let xo = _mm256_shuffle_ps::<0xDD>(c, d);
        for r in 0..R {
            let pairs = _mm256_loadu_si256(w.add(r * row_bytes + 2 * i) as *const __m256i);
            let we = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(pairs));
            let wo = _mm256_castsi256_ps(_mm256_and_si256(pairs, high));
            even[r] = _mm256_fmadd_ps(we, xe, even[r]);
            odd[r] = _mm256_fmadd_ps(wo, xo, odd[r]);
        }
        i += 16;
    }
    let mut sums = [0.0; R];
    for r in 0..R {
        sums[r] = hsum256(_mm256_add_ps(even[r], odd[r]));
    }
    sums
}

/// `acc[i] += a * e4m3(row[i])` over `acc.len()` elements. Widened as the
/// block kernel widens (value / 256) against `a * 256`: the product is `a *
/// value` exactly, so each fused multiply-add rounds as the portable one
/// does. The NaN encoding widens to a finite value, so the loop keeps a
/// running byte maximum and, only when that reaches the encoding, sets the
/// affected elements to NaN afterwards, which is what adding a NaN product
/// makes them. An `a` whose `* 256` leaves the finite range takes the
/// portable path, as does the tail.
pub(crate) fn axpy_e4m3_f32(row: &[u8], a: f32, acc: &mut [f32]) {
    const STEP: usize = 16;
    let n = acc.len();
    assert!(row.len() >= n);
    let body = n - n % STEP;
    let scaled = a * E4M3_SCALE;
    if !scaled.is_finite() {
        return crate::arch::portable::gemv::axpy_e4m3_f32(row, a, acc);
    }
    // SAFETY: `row` and `acc` hold at least `body` elements, read and
    // written in whole steps below it.
    let nan = unsafe {
        let clr = _mm256_set1_epi16(E4M3_CLR);
        let magnitude = _mm_set1_epi8(0x7F);
        let mut peak = _mm_setzero_si128();
        let coefficient = _mm256_set1_ps(scaled);
        let (w, out) = (row.as_ptr(), acc.as_mut_ptr());
        let mut i = 0;
        while i < body {
            let bytes = _mm_loadu_si128(w.add(i) as *const __m128i);
            peak = _mm_max_epu8(peak, _mm_and_si128(bytes, magnitude));
            let words = _mm256_cvtepi8_epi16(bytes);
            let f16 = _mm256_and_si256(_mm256_slli_epi16::<7>(words), clr);
            let lo = _mm256_cvtph_ps(_mm256_castsi256_si128(f16));
            let hi = _mm256_cvtph_ps(_mm256_extracti128_si256::<1>(f16));
            let (p0, p1) = (out.add(i), out.add(i + 8));
            _mm256_storeu_ps(p0, _mm256_fmadd_ps(coefficient, lo, _mm256_loadu_ps(p0)));
            _mm256_storeu_ps(p1, _mm256_fmadd_ps(coefficient, hi, _mm256_loadu_ps(p1)));
            i += STEP;
        }
        _mm_movemask_epi8(_mm_cmpeq_epi8(peak, magnitude)) != 0
    };
    if nan {
        for (value, &b) in acc[..body].iter_mut().zip(row) {
            if b & 0x7F == 0x7F {
                *value = f32::NAN;
            }
        }
    }
    crate::arch::portable::gemv::axpy_e4m3_f32(&row[body..], a, &mut acc[body..]);
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    /// Several tokens per pass give each token its one-token result, bit
    /// for bit, with a tail past the last whole step and a NaN row.
    #[test]
    fn e4m3_tokens_match_one_token_calls() {
        let (rows, k) = (6, 2688 + 20);
        let mut weights: Vec<u8> = (0..rows * k).map(|i| (i * 37 % 251) as u8 & 0xF7).collect();
        weights[3 * k + 5] = 0x7F;
        let xs: Vec<Vec<f32>> = (0..4)
            .map(|t| {
                (0..k)
                    .map(|i| ((i * 29 + t * 11) % 101) as f32 / 50.0 - 1.0)
                    .collect()
            })
            .collect();
        for count in 1..=super::TOKENS {
            let x: Vec<&[f32]> = xs[..count].iter().map(|x| x.as_slice()).collect();
            let mut outs = vec![vec![0.0; rows]; count];
            let mut refs: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
            super::e4m3_f32_tokens(&weights, &x, &mut refs);
            for (t, out) in outs.iter().enumerate() {
                let mut alone = vec![0.0; rows];
                super::e4m3_f32(&weights, &xs[t], &mut alone);
                let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(out), bits(&alone), "token {t} of {count}");
                assert!(out[3].is_nan() && !out[2].is_nan());
            }
        }
    }
}
