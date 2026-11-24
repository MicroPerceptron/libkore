//! Matrix multiplication micro-kernels — scalar fallback.
//!
//! Cache-tiled for Lunar Lake P-core L1D (48 KB).

use crate::floats::Float;

/// L1D budget in bytes (Lion Cove P-core, pinned via kraph scheduler).
const L1_BYTES: usize = 49152;

/// Compute tile size so 3 tiles (A + B + C) fit in L1D.
/// LLVM const-folds this after monomorphization.
#[inline(always)]
fn tile_size<T>() -> usize {
    // budget = L1_BYTES / (3 * sizeof(T))
    // tile = isqrt(budget)
    let budget = L1_BYTES / (3 * core::mem::size_of::<T>());
    let mut t = 1usize;
    while (t + 1) * (t + 1) <= budget {
        t += 1;
    }
    t
}

/// General matrix multiply: C += A × B  (row-major, tiled)
///
/// A: [m, k]   B: [k, n]   C: [m, n]
fn gemm<T: Float>(a: &[T], b: &[T], c: &mut [T], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    let tile = tile_size::<T>();

    for i0 in (0..m).step_by(tile) {
        let i_end = (i0 + tile).min(m);
        for j0 in (0..n).step_by(tile) {
            let j_end = (j0 + tile).min(n);
            for p0 in (0..k).step_by(tile) {
                let p_end = (p0 + tile).min(k);

                for i in i0..i_end {
                    let c_row = &mut c[i * n..][..n];
                    for p in p0..p_end {
                        let a_ip = a[i * k + p];
                        let b_row = &b[p * n..][..n];
                        for (cj, &bj) in c_row[j0..j_end].iter_mut().zip(&b_row[j0..j_end]) {
                            *cj += a_ip * bj;
                        }
                    }
                }
            }
        }
    }
}

// ── F32 / F64 wrappers ───────────────────────────────────────────────

pub(crate) fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    gemm(a, b, c, m, n, k)
}
pub(crate) fn gemm_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    gemm(a, b, c, m, n, k)
}

// ── F16 mixed-precision ──────────────────────────────────────────────
//
// A(f16) × B(f16) → C(f32) — standard mixed-precision inference pattern.

/// Tile for F16: 2 × TILE² × 2 (A+B as f16) + TILE² × 4 (C as f32).
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
                    let c_row = &mut c[i * n..][..n];
                    for p in p0..p_end {
                        let a_ip = crate::cast::read_f16(&a[(i * k + p) * 2..]);
                        for j in j0..j_end {
                            let b_pj = crate::cast::read_f16(&b[(p * n + j) * 2..]);
                            c_row[j] += a_ip * b_pj;
                        }
                    }
                }
            }
        }
    }
}

// ── BF16 mixed-precision ─────────────────────────────────────────────
//
// A(bf16) × B(bf16) → C(f32) — same mixed-precision pattern as F16.

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
                    let c_row = &mut c[i * n..][..n];
                    for p in p0..p_end {
                        let a_ip = crate::cast::read_bf16(&a[(i * k + p) * 2..]);
                        for j in j0..j_end {
                            let b_pj = crate::cast::read_bf16(&b[(p * n + j) * 2..]);
                            c_row[j] += a_ip * b_pj;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matmul() {
        // I × [1,2;3,4] = [1,2;3,4]
        let eye = [1.0, 0.0, 0.0, 1.0f32];
        let b = [1.0, 2.0, 3.0, 4.0f32];
        let mut c = [0.0f32; 4];
        gemm_f32(&eye, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn known_2x3_3x2() {
        // A = [[1,2,3],[4,5,6]]  B = [[7,8],[9,10],[11,12]]
        // C = [[58,64],[139,154]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32];
        let mut c = [0.0f32; 4];
        gemm_f32(&a, &b, &mut c, 2, 2, 3);
        assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn accumulates_into_c() {
        // Verify C += A × B (not C = A × B).
        let a = [1.0, 0.0, 0.0, 1.0f32];
        let b = [1.0, 2.0, 3.0, 4.0f32];
        let mut c = [10.0, 20.0, 30.0, 40.0f32];
        gemm_f32(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn larger_than_tile() {
        // Matrix bigger than tile to exercise the tiling boundary.
        let t = tile_size::<f32>();
        let n = t + 7; // non-aligned to test remainder handling
        let a = alloc::vec![1.0f32; n * n];
        let b = alloc::vec![1.0f32; n * n];
        let mut c = alloc::vec![0.0f32; n * n];
        gemm_f32(&a, &b, &mut c, n, n, n);
        for &v in &c {
            assert!((v - n as f32).abs() < 1e-3);
        }
    }

    #[test]
    fn non_square() {
        // A: [1, 4]  B: [4, 1]  → C: [1, 1] = dot product
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [0.0f32; 1];
        gemm_f32(&a, &b, &mut c, 1, 1, 4);
        assert_eq!(c[0], 70.0); // 5+12+21+32
    }
}
