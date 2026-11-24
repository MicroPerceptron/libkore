//! Matrix multiplication micro-kernels — portable `core::simd`.
//!
//! C += A × B  (row-major, L1D-tiled).
//!
//! Uses `core::simd` for platform-agnostic SIMD: LLVM lowers to the
//! best available instruction set (NEON, SSE2, AVX2, …) automatically.
//!
//! F16/BF16: mixed-precision (A,B in narrow format, C accumulates in f32).

use core::simd::prelude::*;

use super::W;
const WD: usize = super::WD;

/// L1D budget in bytes.
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

/// Rows per register tile. With two `W`-lane column vectors this holds
/// `2·MR` accumulators live across the whole k extent — C is loaded and
/// stored exactly once per tile instead of once per (i, p), and every
/// loaded B vector feeds MR FMA chains (the multi-accumulator pattern;
/// the axpy sweep this replaced was memory-latency-pinned at ~11 GF/s).
const MR: usize = 4;

pub(crate) fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    const NB: usize = 2 * W;
    let m_main = m - m % MR;
    let n_main = n - n % NB;

    for i0 in (0..m_main).step_by(MR) {
        let a_rows: [&[f32]; MR] = core::array::from_fn(|r| &a[(i0 + r) * k..][..k]);
        for j0 in (0..n_main).step_by(NB) {
            let mut acc = [[Simd::<f32, W>::splat(0.0); 2]; MR];
            for (r, row) in acc.iter_mut().enumerate() {
                let c_row = &c[(i0 + r) * n + j0..];
                row[0] = Simd::from_slice(&c_row[..W]);
                row[1] = Simd::from_slice(&c_row[W..2 * W]);
            }
            for p in 0..k {
                let b_row = &b[p * n + j0..];
                let b0 = Simd::<f32, W>::from_slice(&b_row[..W]);
                let b1 = Simd::<f32, W>::from_slice(&b_row[W..2 * W]);
                for r in 0..MR {
                    let av = Simd::<f32, W>::splat(a_rows[r][p]);
                    acc[r][0] = av * b0 + acc[r][0];
                    acc[r][1] = av * b1 + acc[r][1];
                }
            }
            for (r, row) in acc.iter().enumerate() {
                let c_row = &mut c[(i0 + r) * n + j0..];
                row[0].copy_to_slice(&mut c_row[..W]);
                row[1].copy_to_slice(&mut c_row[W..2 * W]);
            }
        }
        if n_main < n {
            axpy_strip_f32(a, b, c, i0, i0 + MR, n_main, n, n, k);
        }
    }
    if m_main < m {
        axpy_strip_f32(a, b, c, m_main, m, 0, n, n, k);
    }
}

/// Edge fallback: the old broadcast-axpy sweep over a row/column strip.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn axpy_strip_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    n: usize,
    k: usize,
) {
    for i in i_start..i_end {
        for p in 0..k {
            let a_val = a[i * k + p];
            let a_broadcast = Simd::<f32, W>::splat(a_val);
            let c_row = &mut c[i * n..];
            let b_row = &b[p * n..];
            let span = j_end - j_start;
            let chunks = span / W;
            for ci in 0..chunks {
                let j = j_start + ci * W;
                let bv = Simd::<f32, W>::from_slice(&b_row[j..]);
                let cv = Simd::<f32, W>::from_slice(&c_row[j..]);
                (a_broadcast * bv + cv).copy_to_slice(&mut c_row[j..]);
            }
            for j in (j_start + chunks * W)..j_end {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
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

                        let a_broadcast = Simd::<f64, WD>::splat(a_val);
                        let span = j_end - j0;
                        let chunks = span / WD;
                        let tail = span % WD;

                        for ci in 0..chunks {
                            let j = j0 + ci * WD;
                            let bv = Simd::<f64, WD>::from_slice(&b_row[j..]);
                            let cv = Simd::<f64, WD>::from_slice(&c_row[j..]);
                            (a_broadcast * bv + cv).copy_to_slice(&mut c_row[j..]);
                        }
                        let j_tail = j0 + chunks * WD;
                        for t in 0..tail {
                            c_row[j_tail + t] += a_val * b_row[j_tail + t];
                        }
                    }
                }
            }
        }
    }
}

// ── F16 mixed-precision: A(f16) × B(f16) → C(f32) ──────────────────

const TILE_F16: usize = 64;

pub(crate) fn gemm_f16(a: &[u8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k * 2);
    debug_assert_eq!(b.len(), k * n * 2);
    debug_assert_eq!(c.len(), m * n);

    for i0 in (0..m).step_by(TILE_F16) {
        let i_end = (i0 + TILE_F16).min(m);
        for j0 in (0..n).step_by(TILE_F16) {
            let j_end = (j0 + TILE_F16).min(n);
            for p0 in (0..k).step_by(TILE_F16) {
                let p_end = (p0 + TILE_F16).min(k);

                for i in i0..i_end {
                    let c_row = &mut c[i * n..];
                    for p in p0..p_end {
                        let a_ip = crate::cast::read_f16(&a[(i * k + p) * 2..]);
                        let a_broadcast = Simd::<f32, W>::splat(a_ip);
                        let span = j_end - j0;
                        let chunks = span / W;
                        let tail = span % W;

                        for ci in 0..chunks {
                            let j = j0 + ci * W;
                            let bv = Simd::<f32, W>::from_array(core::array::from_fn(|lane| {
                                crate::cast::read_f16(&b[(p * n + j + lane) * 2..])
                            }));
                            let cv = Simd::<f32, W>::from_slice(&c_row[j..]);
                            (a_broadcast * bv + cv).copy_to_slice(&mut c_row[j..]);
                        }
                        let j_tail = j0 + chunks * W;
                        for t in 0..tail {
                            let b_pj = crate::cast::read_f16(&b[(p * n + j_tail + t) * 2..]);
                            c_row[j_tail + t] += a_ip * b_pj;
                        }
                    }
                }
            }
        }
    }
}

// ── BF16 mixed-precision: A(bf16) × B(bf16) → C(f32) ───────────────

pub(crate) fn gemm_bf16(a: &[u8], b: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k * 2);
    debug_assert_eq!(b.len(), k * n * 2);
    debug_assert_eq!(c.len(), m * n);

    for i0 in (0..m).step_by(TILE_F16) {
        let i_end = (i0 + TILE_F16).min(m);
        for j0 in (0..n).step_by(TILE_F16) {
            let j_end = (j0 + TILE_F16).min(n);
            for p0 in (0..k).step_by(TILE_F16) {
                let p_end = (p0 + TILE_F16).min(k);

                for i in i0..i_end {
                    let c_row = &mut c[i * n..];
                    for p in p0..p_end {
                        let a_ip = crate::cast::read_bf16(&a[(i * k + p) * 2..]);
                        let a_broadcast = Simd::<f32, W>::splat(a_ip);
                        let span = j_end - j0;
                        let chunks = span / W;
                        let tail = span % W;

                        for ci in 0..chunks {
                            let j = j0 + ci * W;
                            let bv = Simd::<f32, W>::from_array(core::array::from_fn(|lane| {
                                crate::cast::read_bf16(&b[(p * n + j + lane) * 2..])
                            }));
                            let cv = Simd::<f32, W>::from_slice(&c_row[j..]);
                            (a_broadcast * bv + cv).copy_to_slice(&mut c_row[j..]);
                        }
                        let j_tail = j0 + chunks * W;
                        for t in 0..tail {
                            let b_pj = crate::cast::read_bf16(&b[(p * n + j_tail + t) * 2..]);
                            c_row[j_tail + t] += a_ip * b_pj;
                        }
                    }
                }
            }
        }
    }
}
