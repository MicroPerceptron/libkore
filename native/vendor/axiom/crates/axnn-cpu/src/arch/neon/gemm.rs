//! F32 GEMM micro-kernel — NEON fused-multiply-add tier.
//!
//! Same register-tile shape as the portable kernel (MR rows × 16
//! columns of C living in accumulators across the whole k extent, C
//! touched once per tile) but with explicit `vfmaq_f32`: `core::simd`'s
//! `a * b + c` lowers to separate `fmul`+`fadd` (fusion needs std's
//! `mul_add`, unavailable in no_std), which halves FMA throughput.
//! NEON is architecturally mandatory on AArch64, so this tier needs no
//! feature gate. Narrow-type and f64 GEMMs re-export from portable.

use core::arch::aarch64::*;

pub(crate) use crate::arch::portable::gemm::{gemm_bf16, gemm_f16, gemm_f64};

const MR: usize = 4;
const NB: usize = 16;

pub(crate) fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    let m_main = m - m % MR;
    let n_main = n - n % NB;

    for i0 in (0..m_main).step_by(MR) {
        for j0 in (0..n_main).step_by(NB) {
            // SAFETY: every pointer below is bounded by the row/column
            // block ranges checked against the slice-length asserts.
            unsafe {
                let mut acc = [[vdupq_n_f32(0.0); 4]; MR];
                for (r, row) in acc.iter_mut().enumerate() {
                    let c_row = c.as_ptr().add((i0 + r) * n + j0);
                    for (q, lane) in row.iter_mut().enumerate() {
                        *lane = vld1q_f32(c_row.add(4 * q));
                    }
                }
                for p in 0..k {
                    let b_row = b.as_ptr().add(p * n + j0);
                    let b0 = vld1q_f32(b_row);
                    let b1 = vld1q_f32(b_row.add(4));
                    let b2 = vld1q_f32(b_row.add(8));
                    let b3 = vld1q_f32(b_row.add(12));
                    for (r, row) in acc.iter_mut().enumerate() {
                        let av = vdupq_n_f32(*a.get_unchecked((i0 + r) * k + p));
                        row[0] = vfmaq_f32(row[0], av, b0);
                        row[1] = vfmaq_f32(row[1], av, b1);
                        row[2] = vfmaq_f32(row[2], av, b2);
                        row[3] = vfmaq_f32(row[3], av, b3);
                    }
                }
                for (r, row) in acc.iter().enumerate() {
                    let c_row = c.as_mut_ptr().add((i0 + r) * n + j0);
                    for (q, lane) in row.iter().enumerate() {
                        vst1q_f32(c_row.add(4 * q), *lane);
                    }
                }
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
