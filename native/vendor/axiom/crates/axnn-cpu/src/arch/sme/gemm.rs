//! F32 GEMM on the SME outer-product engine.
//!
//! `FMOPA za.s` accumulates a full SVL×SVL f32 outer product per
//! instruction — at SVL = 512 bits that is a 16×16 tile (256 FMAs).
//! This kernel drives a 2×2 arrangement of the four f32 ZA tiles: a
//! 32×32 block of C stays resident in ZA while the k extent streams
//! through as rank-1 updates, 4 FMOPA per k step.
//!
//! Safety architecture: each (row-block × col-block × k-block) pass is
//! ONE `asm!` block that brackets `smstart`/`smstop` itself — no Rust
//! code executes in streaming mode. Entering/leaving streaming mode
//! invalidates the NEON register file, so all v-registers are declared
//! clobbered; predicate registers are used by no non-SME code. The ZA
//! slice index register must be one of w12–w15 (architectural
//! constraint), hence the hardcoded w12.
//!
//! A is packed per row-block into a k-major panel (`at[p][32]`) so the
//! FMOPA left operand (a column of A) is a contiguous vector load; B
//! rows are already contiguous. The panel is a 32 KiB stack buffer
//! (the house scratch budget), which also bounds the k block.

use core::arch::asm;

pub(crate) use crate::arch::neon::gemm::{gemm_bf16, gemm_f16, gemm_f64};

/// Row/column block: 2 tiles × 16 f32 lanes (SVL = 512 bits).
const RB: usize = 32;
/// k block so the packed A panel is exactly the 32 KiB scratch budget.
const KB: usize = 256;

pub(crate) fn gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    // Outer-product setup only pays off with a full 32×32 ZA block and
    // a nontrivial reduction; everything else takes the NEON FMA tier.
    if m < RB || n < RB || k < 8 {
        return crate::arch::neon::gemm::gemm_f32(a, b, c, m, n, k);
    }

    let m_main = m - m % RB;
    let n_main = n - n % RB;
    let mut panel = [0f32; KB * RB];

    for i0 in (0..m_main).step_by(RB) {
        for p0 in (0..k).step_by(KB) {
            let kb = KB.min(k - p0);
            // Pack A[i0..i0+32, p0..p0+kb] k-major: panel[p][r].
            for (p, row) in panel.chunks_exact_mut(RB).take(kb).enumerate() {
                for (r, slot) in row.iter_mut().enumerate() {
                    *slot = a[(i0 + r) * k + p0 + p];
                }
            }
            for j0 in (0..n_main).step_by(RB) {
                // SAFETY: `feature = "sme"` asserts FEAT_SME hardware
                // (module doctrine). All pointers and strides are
                // bounded by the asserts and block ranges above; the
                // block enters and exits streaming mode itself and
                // declares every register it touches.
                unsafe {
                    za_block_f32(
                        c.as_mut_ptr().add(i0 * n + j0),
                        panel.as_ptr(),
                        b.as_ptr().add(p0 * n + j0),
                        kb,
                        n,
                    );
                }
            }
        }
        if n_main < n {
            crate::arch::portable::gemm::axpy_strip_f32(a, b, c, i0, i0 + RB, n_main, n, n, k);
        }
    }
    if m_main < m {
        crate::arch::portable::gemm::axpy_strip_f32(a, b, c, m_main, m, 0, n, n, k);
    }
}

/// One streaming-mode pass: `C[32×32] += At_panel × B_rows` over `kb`.
///
/// * `c` — top-left of the 32×32 C block, row stride `n` elements
/// * `at` — packed A panel, k-major `[kb][32]`, contiguous
/// * `b` — B[p0, j0], row stride `n` elements
///
/// # Safety
/// Requires FEAT_SME. Pointers must cover the full block extents.
#[inline(never)]
unsafe fn za_block_f32(c: *mut f32, at: *const f32, b: *const f32, kb: usize, n: usize) {
    let row_bytes = n * 4;
    unsafe {
        asm!(
            ".arch_extension sme",
            "smstart",
            "ptrue p0.s",
            // ── Load C block into ZA (four 16×16 tiles) ─────────────
            "mov x9, {c}",              // rows 0..16
            "add x10, {c}, {stride}, lsl #4", // rows 16..32 (16 * stride)
            "mov x11, #16",             // second-tile column offset (elems)
            "mov w12, #0",
            "2:",
            "ld1w {{za0h.s[w12, 0]}}, p0/z, [x9]",
            "ld1w {{za1h.s[w12, 0]}}, p0/z, [x9, x11, lsl #2]",
            "ld1w {{za2h.s[w12, 0]}}, p0/z, [x10]",
            "ld1w {{za3h.s[w12, 0]}}, p0/z, [x10, x11, lsl #2]",
            "add x9, x9, {stride}",
            "add x10, x10, {stride}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 2b",
            // ── k loop: 4 outer products per step ────────────────────
            "mov x9, {at}",
            "mov x10, {b}",
            "mov x13, {kb}",
            "3:",
            "ld1w {{z0.s}}, p0/z, [x9]",
            "ld1w {{z1.s}}, p0/z, [x9, #1, mul vl]",
            "ld1w {{z2.s}}, p0/z, [x10]",
            "ld1w {{z3.s}}, p0/z, [x10, #1, mul vl]",
            "add x9, x9, #128",         // 32 packed f32
            "add x10, x10, {stride}",
            "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
            "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
            "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
            "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
            "subs x13, x13, #1",
            "b.ne 3b",
            // ── Store ZA back to C ───────────────────────────────────
            "mov x9, {c}",
            "add x10, {c}, {stride}, lsl #4",
            "mov w12, #0",
            "4:",
            "st1w {{za0h.s[w12, 0]}}, p0, [x9]",
            "st1w {{za1h.s[w12, 0]}}, p0, [x9, x11, lsl #2]",
            "st1w {{za2h.s[w12, 0]}}, p0, [x10]",
            "st1w {{za3h.s[w12, 0]}}, p0, [x10, x11, lsl #2]",
            "add x9, x9, {stride}",
            "add x10, x10, {stride}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.ne 4b",
            "smstop",
            c = in(reg) c,
            at = in(reg) at,
            b = in(reg) b,
            kb = in(reg) kb,
            stride = in(reg) row_bytes,
            out("x9") _, out("x10") _, out("x11") _, out("x12") _, out("x13") _,
            // Streaming-mode entry/exit invalidates the SIMD file.
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
        );
    }
}

#[cfg(test)]
mod tests {
    /// SME blocks + NEON/axpy edges against the portable kernel, on
    /// sizes that exercise full tiles, k-blocking, and both remainders.
    #[test]
    fn matches_portable_reference() {
        const M: usize = 45;
        const N: usize = 39;
        const K: usize = 300; // > KB forces the k-block accumulate path
        let mut state = 0x5eedu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) as f32 / u32::MAX as f32 - 0.5
        };
        let mut a = [0f32; M * K];
        let mut b = [0f32; K * N];
        for v in a.iter_mut() {
            *v = next();
        }
        for v in b.iter_mut() {
            *v = next();
        }
        let mut c_sme = [1.0f32; M * N]; // nonzero: checks C += semantics
        let mut c_ref = [1.0f32; M * N];
        super::gemm_f32(&a, &b, &mut c_sme, M, N, K);
        crate::arch::portable::gemm::gemm_f32(&a, &b, &mut c_ref, M, N, K);
        for i in 0..M * N {
            let (got, want) = (c_sme[i], c_ref[i]);
            assert!(
                (got - want).abs() <= want.abs() * 1e-5 + 1e-5,
                "elem {i}: sme {got}, portable {want}"
            );
        }
    }
}
