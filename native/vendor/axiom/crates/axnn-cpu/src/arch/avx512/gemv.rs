//! AVX-512 BF16 weight rows for the Nemotron and DSpark projections.
//!
//! Sixteen BF16 words widen exactly to one ZMM of F32. Four rows share
//! each activation load, and each row keeps two chains across 32 elements.
//! E4M3 projection rows widen 64 bytes per step into four F32 vectors.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

pub(crate) const TOKENS: usize = 4;
const E4M3_STEP: usize = 64;
const E4M3_SCALE: f32 = 256.0;
const E4M3_CLR: i16 = 0xBFFFu16 as i16;

const ROWS: usize = 4;
const STEP: usize = 32;

/// Unscaled E4M3 weight rows against F32 activations.
pub(crate) fn e4m3_f32(weights: &[u8], x: &[f32], out: &mut [f32]) {
    let k = x.len();
    if k == 0 {
        out.fill(0.0);
        return;
    }
    assert!(weights.len() >= out.len() * k);
    let body = k - k % E4M3_STEP;
    let (blocks, rest) = out.as_chunks_mut::<ROWS>();
    for (b, ys) in blocks.iter_mut().enumerate() {
        e4m3_rows::<ROWS>(&weights[b * ROWS * k..(b + 1) * ROWS * k], x, body, ys);
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

fn e4m3_rows<const R: usize>(weights: &[u8], x: &[f32], body: usize, ys: &mut [f32; R]) {
    let k = x.len();
    // SAFETY: each row and activation has a complete `body` prefix.
    let (sums, nan) = unsafe { e4m3_block::<R>(weights.as_ptr(), k, x.as_ptr(), body) };
    for (r, y) in ys.iter_mut().enumerate() {
        let row = &weights[r * k..(r + 1) * k];
        *y = if nan && row[..body].iter().any(|&b| b & 0x7f == 0x7f) {
            f32::NAN
        } else {
            sums[r] * E4M3_SCALE
                + crate::arch::f8::vec_dot_e4m3_f32(&row[body..], &x[body..], k - body)
        };
    }
}

#[inline(always)]
unsafe fn e4m3_widen(bytes: __m128i, clear: __m256i) -> __m512 {
    let words = _mm256_cvtepi8_epi16(bytes);
    let half = _mm256_and_si256(_mm256_slli_epi16::<7>(words), clear);
    _mm512_cvtph_ps(half)
}

#[inline(always)]
unsafe fn e4m3_block<const R: usize>(
    w: *const u8,
    k: usize,
    x: *const f32,
    body: usize,
) -> ([f32; R], bool) {
    let clear = _mm256_set1_epi16(E4M3_CLR);
    let magnitude = _mm512_set1_epi8(0x7f);
    let mut peak = _mm512_setzero_si512();
    let mut even = [_mm512_setzero_ps(); R];
    let mut odd = [_mm512_setzero_ps(); R];
    let mut i = 0;
    while i < body {
        let xs = [
            _mm512_loadu_ps(x.add(i)),
            _mm512_loadu_ps(x.add(i + 16)),
            _mm512_loadu_ps(x.add(i + 32)),
            _mm512_loadu_ps(x.add(i + 48)),
        ];
        for r in 0..R {
            let row = w.add(r * k + i);
            let step = _mm512_loadu_si512(row.cast());
            peak = _mm512_max_epu8(peak, _mm512_and_si512(step, magnitude));
            for g in 0..4 {
                let bytes = _mm_loadu_si128(row.add(16 * g).cast());
                let weight = e4m3_widen(bytes, clear);
                let acc = if g % 2 == 0 {
                    &mut even[r]
                } else {
                    &mut odd[r]
                };
                *acc = _mm512_fmadd_ps(weight, xs[g], *acc);
            }
        }
        i += E4M3_STEP;
    }
    let nan = _mm512_cmpeq_epi8_mask(peak, magnitude) != 0;
    (
        core::array::from_fn(|r| _mm512_reduce_add_ps(_mm512_add_ps(even[r], odd[r]))),
        nan,
    )
}

/// Reuse each E4M3 row across up to four activation tokens.
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
    let k = xs[0].len();
    let rows = out[0].len();
    assert!(xs.iter().all(|x| x.len() == k) && out.iter().all(|o| o.len() == rows));
    if k == 0 {
        out.iter_mut().for_each(|o| o.fill(0.0));
        return;
    }
    assert!(weights.len() >= rows * k);
    let body = k - k % E4M3_STEP;
    let x: [*const f32; T] = core::array::from_fn(|t| xs[t].as_ptr());
    for r in 0..rows {
        let row = &weights[r * k..(r + 1) * k];
        // SAFETY: each activation has `k` elements and this row has `k` bytes.
        let (sums, nan) = unsafe { e4m3_row_tokens::<T>(row.as_ptr(), x, body) };
        let nan = nan && row[..body].iter().any(|&b| b & 0x7f == 0x7f);
        for t in 0..T {
            out[t][r] = if nan {
                f32::NAN
            } else {
                sums[t] * E4M3_SCALE
                    + crate::arch::f8::vec_dot_e4m3_f32(&row[body..], &xs[t][body..], k - body)
            };
        }
    }
}

#[inline(always)]
unsafe fn e4m3_row_tokens<const T: usize>(
    w: *const u8,
    x: [*const f32; T],
    body: usize,
) -> ([f32; T], bool) {
    let clear = _mm256_set1_epi16(E4M3_CLR);
    let magnitude = _mm512_set1_epi8(0x7f);
    let mut peak = _mm512_setzero_si512();
    let mut even = [_mm512_setzero_ps(); T];
    let mut odd = [_mm512_setzero_ps(); T];
    let mut i = 0;
    while i < body {
        let step = _mm512_loadu_si512(w.add(i).cast());
        peak = _mm512_max_epu8(peak, _mm512_and_si512(step, magnitude));
        for g in 0..4 {
            let bytes = _mm_loadu_si128(w.add(i + 16 * g).cast());
            let weight = e4m3_widen(bytes, clear);
            for t in 0..T {
                let acc = if g % 2 == 0 {
                    &mut even[t]
                } else {
                    &mut odd[t]
                };
                *acc = _mm512_fmadd_ps(weight, _mm512_loadu_ps(x[t].add(i + 16 * g)), *acc);
            }
        }
        i += E4M3_STEP;
    }
    (
        core::array::from_fn(|t| _mm512_reduce_add_ps(_mm512_add_ps(even[t], odd[t]))),
        _mm512_cmpeq_epi8_mask(peak, magnitude) != 0,
    )
}

/// `acc[i] += a * e4m3(row[i])` for the routed FP8 expert path.
pub(crate) fn axpy_e4m3_f32(row: &[u8], a: f32, acc: &mut [f32]) {
    let n = acc.len();
    assert!(row.len() >= n);
    let scaled = a * E4M3_SCALE;
    if !scaled.is_finite() {
        return crate::arch::portable::gemv::axpy_e4m3_f32(row, a, acc);
    }
    let body = n - n % E4M3_STEP;
    // SAFETY: the loop reads and writes whole 64-element steps below `n`.
    let nan = unsafe {
        let clear = _mm256_set1_epi16(E4M3_CLR);
        let magnitude = _mm512_set1_epi8(0x7f);
        let coefficient = _mm512_set1_ps(scaled);
        let mut peak = _mm512_setzero_si512();
        let (w, out) = (row.as_ptr(), acc.as_mut_ptr());
        let mut i = 0;
        while i < body {
            let bytes = _mm512_loadu_si512(w.add(i).cast());
            peak = _mm512_max_epu8(peak, _mm512_and_si512(bytes, magnitude));
            for g in 0..4 {
                let weight = e4m3_widen(_mm_loadu_si128(w.add(i + 16 * g).cast()), clear);
                let p = out.add(i + 16 * g);
                _mm512_storeu_ps(p, _mm512_fmadd_ps(coefficient, weight, _mm512_loadu_ps(p)));
            }
            i += E4M3_STEP;
        }
        _mm512_cmpeq_epi8_mask(peak, magnitude) != 0
    };
    if nan {
        for (value, &code) in acc[..body].iter_mut().zip(row) {
            if code & 0x7f == 0x7f {
                *value = f32::NAN;
            }
        }
    }
    crate::arch::portable::gemv::axpy_e4m3_f32(&row[body..], a, &mut acc[body..]);
}

pub(crate) fn bf16_f32(weights: &[u8], x: &[f32], out: &mut [f32]) {
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
        rows::<ROWS>(&weights[b * ROWS * row_bytes..], x, row_bytes, body, ys);
    }
    let first = blocks.len() * ROWS;
    for (r, y) in (first..).zip(rest) {
        rows::<1>(
            &weights[r * row_bytes..],
            x,
            row_bytes,
            body,
            core::array::from_mut(y),
        );
    }
}

fn rows<const R: usize>(
    weights: &[u8],
    x: &[f32],
    row_bytes: usize,
    body: usize,
    ys: &mut [f32; R],
) {
    // SAFETY: every row holds `row_bytes` bytes and the body is a complete
    // multiple of STEP inside the activation slice.
    let sums = unsafe { block::<R>(weights.as_ptr(), x.as_ptr(), row_bytes, body) };
    for (r, y) in ys.iter_mut().enumerate() {
        *y = sums[r]
            + crate::arch::portable::gemv::dot_bf16_f32(
                &weights[r * row_bytes + 2 * body..(r + 1) * row_bytes],
                &x[body..],
            );
    }
}

#[inline(always)]
unsafe fn block<const R: usize>(
    w: *const u8,
    x: *const f32,
    row_bytes: usize,
    body: usize,
) -> [f32; R] {
    let mut low = [_mm512_setzero_ps(); R];
    let mut high = [_mm512_setzero_ps(); R];
    let mut i = 0;
    while i < body {
        let (x0, x1) = (_mm512_loadu_ps(x.add(i)), _mm512_loadu_ps(x.add(i + 16)));
        for r in 0..R {
            let row = w.add(r * row_bytes + 2 * i);
            let w0 = _mm256_loadu_si256(row.cast());
            let w1 = _mm256_loadu_si256(row.add(32).cast());
            let f0 = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(w0)));
            let f1 = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(w1)));
            low[r] = _mm512_fmadd_ps(f0, x0, low[r]);
            high[r] = _mm512_fmadd_ps(f1, x1, high[r]);
        }
        i += STEP;
    }
    core::array::from_fn(|r| _mm512_reduce_add_ps(_mm512_add_ps(low[r], high[r])))
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn e4m3_axpy_matches_the_256_tier() {
        for n in [1_usize, 31, 63, 64, 65, 128, 2688, 2704] {
            let mut row: Vec<u8> = (0..n).map(|i| (i * 37 % 251) as u8 & 0xf7).collect();
            row[n / 2] = 0x7f;
            for a in [0.0, 0.125, -0.75, 3.0] {
                let mut got: Vec<f32> = (0..n).map(|i| i as f32 / 101.0).collect();
                let mut old = got.clone();
                super::axpy_e4m3_f32(&row, a, &mut got);
                crate::arch::avx2::gemv::axpy_e4m3_f32(&row, a, &mut old);
                for i in 0..n {
                    if old[i].is_nan() {
                        assert!(got[i].is_nan(), "n={n}, a={a}, i={i}");
                    } else {
                        assert_eq!(got[i].to_bits(), old[i].to_bits(), "n={n}, a={a}, i={i}");
                    }
                }
            }
        }
    }

    #[test]
    fn e4m3_rows_and_tokens_match_at_step_boundaries() {
        for k in [1_usize, 31, 32, 63, 64, 65, 2688, 2704] {
            let rows = 7;
            let mut weights: Vec<u8> = (0..rows * k).map(|i| (i * 37 % 251) as u8 & 0xf7).collect();
            weights[3 * k + k / 2] = 0x7f;
            let xs: Vec<Vec<f32>> = (0..4)
                .map(|t| {
                    (0..k)
                        .map(|i| ((i * 29 + t * 11) % 101) as f32 / 50.0 - 1.0)
                        .collect()
                })
                .collect();
            let mut single = vec![vec![0.0; rows]; 4];
            for (x, out) in xs.iter().zip(&mut single) {
                super::e4m3_f32(&weights, x, out);
            }
            for r in 0..rows {
                let mut alone = [0.0];
                super::e4m3_f32(&weights[r * k..(r + 1) * k], &xs[0], &mut alone);
                assert_eq!(single[0][r].to_bits(), alone[0].to_bits(), "k={k}, r={r}");
            }
            let mut reference = vec![0.0; rows];
            crate::arch::avx2::gemv::e4m3_f32(&weights, &xs[0], &mut reference);
            for r in 0..rows {
                if r == 3 {
                    assert!(single[0][r].is_nan() && reference[r].is_nan());
                } else {
                    assert!(
                        (single[0][r] - reference[r]).abs() <= 1e-3 * (1.0 + reference[r].abs()),
                        "k={k}, r={r}: {} vs {}",
                        single[0][r],
                        reference[r]
                    );
                }
            }
            for count in 1..=4 {
                let x: Vec<&[f32]> = xs[..count].iter().map(Vec::as_slice).collect();
                let mut multi = vec![vec![0.0; rows]; count];
                let mut outputs: Vec<&mut [f32]> =
                    multi.iter_mut().map(Vec::as_mut_slice).collect();
                super::e4m3_f32_tokens(&weights, &x, &mut outputs);
                for t in 0..count {
                    for r in 0..rows {
                        assert_eq!(
                            multi[t][r].to_bits(),
                            single[t][r].to_bits(),
                            "k={k}, t={t}, r={r}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bf16_matches_avx2_and_row_splits() {
        for k in [1_usize, 16, 32, 48, 2688, 2704] {
            let rows = 7;
            let weights: Vec<u8> = (0..rows * k)
                .flat_map(|i| {
                    let weight = ((i * 37 % 101) as f32 - 50.0) / 200.0;
                    (((weight.to_bits() >> 16) as u16).to_le_bytes()).into_iter()
                })
                .collect();
            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 29 % 97) as f32 - 48.0) / 49.0)
                .collect();
            let mut got = vec![0.0; rows];
            let mut old = vec![0.0; rows];
            super::bf16_f32(&weights, &x, &mut got);
            crate::arch::avx2::gemv::bf16_f32(&weights, &x, &mut old);
            for r in 0..rows {
                let mut alone = [0.0];
                super::bf16_f32(&weights[r * 2 * k..(r + 1) * 2 * k], &x, &mut alone);
                assert_eq!(got[r].to_bits(), alone[0].to_bits(), "k={k}, row={r}");
                assert!(
                    (got[r] - old[r]).abs() <= 1e-3 * (1.0 + old[r].abs()),
                    "k={k}, row={r}: {} vs {}",
                    got[r],
                    old[r],
                );
            }
        }
    }
}
