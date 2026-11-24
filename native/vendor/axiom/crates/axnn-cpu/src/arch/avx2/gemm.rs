//! Matrix multiplication micro-kernels — AVX2 + FMA.
//!
//! C += A × B  (row-major, register-tiled).
//!
//! Six rows × sixteen columns of C stay resident in YMM accumulators
//! across the complete k reduction. Every B vector feeds six FMA
//! chains and C is touched exactly once per tile.
//!
//! F16/BF16: mixed-precision (A,B in narrow format, C accumulates in f32).

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

/// L1D budget in bytes (Lion Cove P-core, same as scalar).
const L1_BYTES: usize = 49152;

/// Tile side so 3 tiles (A-strip + B-strip + C-tile) fit in L1D.
#[inline(always)]
fn tile_size(elem_bytes: usize) -> usize {
    let budget = L1_BYTES / (3 * elem_bytes);
    let mut t = 1usize;
    while (t + 1) * (t + 1) <= budget {
        t += 1;
    }
    t
}

// ── F32 ──────────────────────────────────────────────────────────────

const MR: usize = 6;
const NB: usize = 16;

pub(crate) fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // SAFETY: this module is selected only for an AVX2+FMA+F16C target
    // contract, either auto-detected or explicitly asserted by the build.
    unsafe { gemm_f32_avx2(a, b, c, m, n, k) }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemm_f32_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    let m_main = m - m % MR;
    let n_main = n - n % NB;
    for i0 in (0..m_main).step_by(MR) {
        for j0 in (0..n_main).step_by(NB) {
            let mut acc = [[_mm256_setzero_ps(); 2]; MR];
            for (r, row) in acc.iter_mut().enumerate() {
                let cp = c.as_ptr().add((i0 + r) * n + j0);
                row[0] = _mm256_loadu_ps(cp);
                row[1] = _mm256_loadu_ps(cp.add(8));
            }
            for p in 0..k {
                let bp = b.as_ptr().add(p * n + j0);
                let b0 = _mm256_loadu_ps(bp);
                let b1 = _mm256_loadu_ps(bp.add(8));
                for (r, row) in acc.iter_mut().enumerate() {
                    let av = _mm256_set1_ps(*a.get_unchecked((i0 + r) * k + p));
                    row[0] = _mm256_fmadd_ps(av, b0, row[0]);
                    row[1] = _mm256_fmadd_ps(av, b1, row[1]);
                }
            }
            for (r, row) in acc.iter().enumerate() {
                let cp = c.as_mut_ptr().add((i0 + r) * n + j0);
                _mm256_storeu_ps(cp, row[0]);
                _mm256_storeu_ps(cp.add(8), row[1]);
            }
        }
        if n_main < n {
            crate::arch::portable::gemm::axpy_strip_f32(a, b, c, i0, i0 + MR, n_main, n, n, k);
        }
    }
    if m_main < m {
        crate::arch::portable::gemm::axpy_strip_f32(a, b, c, m_main, m, 0, n, n, k);
    }
}

#[inline(always)]
fn scalar_strip_narrow(
    a: &[u8],
    b: &[u8],
    c: &mut [f32],
    rows: core::ops::Range<usize>,
    cols: core::ops::Range<usize>,
    n: usize,
    k: usize,
    read: fn(&[u8]) -> f32,
) {
    for i in rows {
        for j in cols.clone() {
            let mut sum = c[i * n + j];
            for p in 0..k {
                sum += read(&a[(i * k + p) * 2..]) * read(&b[(p * n + j) * 2..]);
            }
            c[i * n + j] = sum;
        }
    }
}

#[inline(always)]
unsafe fn load_bf16x8(p: *const u8) -> __m256 {
    let bits16 = _mm_loadu_si128(p as *const __m128i);
    let wide = _mm256_cvtepu16_epi32(bits16);
    _mm256_castsi256_ps(_mm256_slli_epi32(wide, 16))
}

#[inline(always)]
unsafe fn gemm_narrow_avx2(
    a: &[u8],
    b: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    read: fn(&[u8]) -> f32,
    load8: unsafe fn(*const u8) -> __m256,
) {
    let m_main = m - m % MR;
    let n_main = n - n % NB;
    for i0 in (0..m_main).step_by(MR) {
        for j0 in (0..n_main).step_by(NB) {
            let mut acc = [[_mm256_setzero_ps(); 2]; MR];
            for (r, row) in acc.iter_mut().enumerate() {
                let cp = c.as_ptr().add((i0 + r) * n + j0);
                row[0] = _mm256_loadu_ps(cp);
                row[1] = _mm256_loadu_ps(cp.add(8));
            }
            for p in 0..k {
                let bp = b.as_ptr().add((p * n + j0) * 2);
                let b0 = load8(bp);
                let b1 = load8(bp.add(16));
                for (r, row) in acc.iter_mut().enumerate() {
                    let av = _mm256_set1_ps(read(&a[((i0 + r) * k + p) * 2..]));
                    row[0] = _mm256_fmadd_ps(av, b0, row[0]);
                    row[1] = _mm256_fmadd_ps(av, b1, row[1]);
                }
            }
            for (r, row) in acc.iter().enumerate() {
                let cp = c.as_mut_ptr().add((i0 + r) * n + j0);
                _mm256_storeu_ps(cp, row[0]);
                _mm256_storeu_ps(cp.add(8), row[1]);
            }
        }
        if n_main < n {
            scalar_strip_narrow(a, b, c, i0..i0 + MR, n_main..n, n, k, read);
        }
    }
    if m_main < m {
        scalar_strip_narrow(a, b, c, m_main..m, 0..n, n, k, read);
    }
}

#[inline(always)]
unsafe fn load_f16x8(p: *const u8) -> __m256 {
    _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
}

// ── F64 ──────────────────────────────────────────────────────────────

pub(crate) fn gemm_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    let tile = tile_size(8);

    for i0 in (0..m).step_by(tile) {
        let i_end = (i0 + tile).min(m);
        for j0 in (0..n).step_by(tile) {
            let j_end = (j0 + tile).min(n);
            for p0 in (0..k).step_by(tile) {
                let p_end = (p0 + tile).min(k);

                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_val = a[i * k + p];
                        let c_row = &mut c[i * n..];
                        let b_row = &b[p * n..];

                        unsafe {
                            let a_broadcast = _mm256_set1_pd(a_val);
                            let span = j_end - j0;
                            let chunks = span / 4;
                            let tail = span % 4;

                            for ci in 0..chunks {
                                let j = j0 + ci * 4;
                                let bv = _mm256_loadu_pd(b_row.as_ptr().add(j));
                                let cv = _mm256_loadu_pd(c_row.as_ptr().add(j));
                                _mm256_storeu_pd(
                                    c_row.as_mut_ptr().add(j),
                                    _mm256_fmadd_pd(a_broadcast, bv, cv),
                                );
                            }
                            let j_tail = j0 + chunks * 4;
                            for t in 0..tail {
                                c_row[j_tail + t] += a_val * b_row[j_tail + t];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── F16 mixed-precision: A(f16) × B(f16) → C(f32) ──────────────────

pub(crate) fn gemm_f16(a: &[u8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k * 2);
    debug_assert_eq!(b.len(), k * n * 2);
    debug_assert_eq!(c.len(), m * n);
    // SAFETY: module selection establishes AVX2+FMA+F16C.
    unsafe { gemm_f16_avx2(a, b, c, m, n, k) }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemm_f16_avx2(a: &[u8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    gemm_narrow_avx2(a, b, c, m, n, k, crate::cast::read_f16, load_f16x8)
}

// ── BF16 mixed-precision: A(bf16) × B(bf16) → C(f32) ───────────────

pub(crate) fn gemm_bf16(a: &[u8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k * 2);
    debug_assert_eq!(b.len(), k * n * 2);
    debug_assert_eq!(c.len(), m * n);

    // SAFETY: module selection establishes AVX2+FMA+F16C.
    unsafe { gemm_bf16_avx2(a, b, c, m, n, k) }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemm_bf16_avx2(a: &[u8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    gemm_narrow_avx2(a, b, c, m, n, k, crate::cast::read_bf16, load_bf16x8)
}
