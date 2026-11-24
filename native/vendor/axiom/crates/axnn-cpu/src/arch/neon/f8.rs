//! NEON FEAT_FHM FP8 kernels — the f16-domain tier for FP8×FP8.
//!
//! Where **both** operands are FP8, the F16→F32 convert (`fcvtl`) that
//! dominates the portable tier is unnecessary: FP8 bytes unpack to F16
//! bit patterns with pure integer ops (`ushll #8`, plus the E4M3
//! no-rebias trick), and `FMLAL`/`FMLAL2` multiply the F16 lanes and
//! accumulate straight into F32 — the product is formed in F32, so no
//! F16 rounding or underflow is introduced. Per 8 elements that is two
//! unpacks and two FMLALs, versus two unpacks, four `fcvtl`s and two
//! FMAs before.
//!
//! E4M3 unpacks in the `/256` domain (both operands), so every product
//! carries `/65536`, folded back with one multiply on the reduced sum;
//! the scalar tails decode exactly. Results match the portable tier
//! within FP association (the accumulation splits low/high lanes).
//!
//! Mixed FP8×F32 kernels stay portable — FMLAL needs both operands in
//! F16, and rounding F32 activations to F16 would lose precision.
//! Compiled only when `fhm` (+`fp16`) is in the target feature set —
//! Apple M-series and Armv8.4+; other builds keep the portable tier.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::*;

// Everything not overridden rides the portable tier untouched.
pub(crate) use crate::arch::portable::f8::{
    e4m3_to_f16, e4m3_to_f32, e5m2_to_f16, e5m2_to_f32, gemm_e4m3_f32, gemm_e5m2_f32,
    mad_e4m3_e4m3_bf16, mad_e4m3_e4m3_f16, mad_e5m2_e5m2_bf16, mad_e5m2_e5m2_f16, mul_e4m3_f32,
    mul_e5m2_f32, vec_dot_e4m3_f32, vec_dot_e5m2_f32,
};

/// `× 2^16`, undoing the two E4M3 operands' `/256` unpacks.
const E4M3_DOT_SCALE: f32 = 65536.0;

/// Unpack 8 E5M2 bytes at `p` to F16 lanes (exact; `byte << 8`).
#[inline(always)]
unsafe fn unpack_e5m2(p: *const u8) -> float16x8_t {
    vreinterpretq_f16_u16(vshll_n_u8::<8>(vld1_u8(p)))
}

/// Unpack 8 E4M3 bytes at `p` to F16 lanes scaled by `1/256`
/// (the no-rebias trick; NaN encodings patched to real NaNs).
#[inline(always)]
unsafe fn unpack_e4m3_over256(p: *const u8) -> float16x8_t {
    let raw = vshll_n_u8::<8>(vld1_u8(p));
    let sh = vreinterpretq_u16_s16(vshrq_n_s16::<1>(vreinterpretq_s16_u16(raw)));
    let v = vandq_u16(sh, vdupq_n_u16(0xBFFF));
    let m = vdupq_n_u16(0x3F80);
    let nan = vceqq_u16(vandq_u16(v, m), m);
    vreinterpretq_f16_u16(vorrq_u16(v, vandq_u16(nan, vdupq_n_u16(0x4000))))
}

/// Scalar E5M2 → F16 bits.
#[inline(always)]
fn bits_e5m2(b: u8) -> u16 {
    (b as u16) << 8
}

/// Scalar E4M3 → F16 bits in the `/256` domain (NaN patched).
#[inline(always)]
fn bits_e4m3_over256(b: u8) -> u16 {
    let sh = ((((b as u16) << 8) as i16) >> 1) as u16 & 0xBFFF;
    if sh & 0x3F80 == 0x3F80 {
        sh | 0x4000
    } else {
        sh
    }
}

// ── Reduction dot: Σ decode(a) · decode(b)  (FP8 × FP8) ────────────

macro_rules! dot_fp8_kernel {
    ($name:ident, $unpack:ident, $scale:expr, $decode:path) => {
        pub(crate) fn $name(a: &[u8], b: &[u8], n: usize) -> f32 {
            debug_assert!(a.len() >= n && b.len() >= n);
            unsafe {
                let (ap, bp) = (a.as_ptr(), b.as_ptr());
                let chunks = n / 8;
                let blocks = chunks / 4;
                let mut acc = [vdupq_n_f32(0.0); 8];
                for blk in 0..blocks {
                    for u in 0..4 {
                        let o = (blk * 4 + u) * 8;
                        let va = $unpack(ap.add(o));
                        let vb = $unpack(bp.add(o));
                        acc[u * 2] = vfmlalq_low_f16(acc[u * 2], va, vb);
                        acc[u * 2 + 1] = vfmlalq_high_f16(acc[u * 2 + 1], va, vb);
                    }
                }
                for c in (blocks * 4)..chunks {
                    let o = c * 8;
                    let va = $unpack(ap.add(o));
                    let vb = $unpack(bp.add(o));
                    acc[0] = vfmlalq_low_f16(acc[0], va, vb);
                    acc[1] = vfmlalq_high_f16(acc[1], va, vb);
                }
                let t0 = vaddq_f32(acc[0], acc[1]);
                let t1 = vaddq_f32(acc[2], acc[3]);
                let t2 = vaddq_f32(acc[4], acc[5]);
                let t3 = vaddq_f32(acc[6], acc[7]);
                let total = vaddq_f32(vaddq_f32(t0, t1), vaddq_f32(t2, t3));
                let mut sum = vaddvq_f32(total) * $scale;
                for i in (chunks * 8)..n {
                    sum += $decode(a[i]) * $decode(b[i]);
                }
                sum
            }
        }
    };
}

dot_fp8_kernel!(
    dot_e4m3_e4m3,
    unpack_e4m3_over256,
    E4M3_DOT_SCALE,
    crate::cast::e4m3_to_f32
);
dot_fp8_kernel!(dot_e5m2_e5m2, unpack_e5m2, 1.0f32, crate::cast::e5m2_to_f32);

// ── GEMM: D[i,j] = Σ_k decode(A[i,k])·decode(B[j,k]) + widen(C[i,j]) ──
//
// The blocked structure of the portable tier, upgraded to the F16
// domain: a BIAS_MR-row block of A unpacks to an F16 scratch — integer
// ops only, half the scratch traffic of f32 — and each B chunk unpacks
// once, FMLAL-ing against all BIAS_MR rows. Edge rows (m % BIAS_MR)
// delegate to the portable kernel over the row remainder.

/// A rows unpacked per block.
const BIAS_MR: usize = 8;
/// k-block width; the F16 A block is `BIAS_MR × BIAS_KB × 2 B` = 32 KiB.
const BIAS_KB: usize = 2048;

macro_rules! unpack_row {
    ($name:ident, $unpack:ident, $bits:ident) => {
        /// Unpack an FP8 row into F16-bit scratch.
        #[inline(always)]
        unsafe fn $name(src: &[u8], dst: *mut u16, n: usize) {
            let chunks = n / 8;
            for c in 0..chunks {
                let v = $unpack(src.as_ptr().add(c * 8));
                vst1q_u16(dst.add(c * 8), vreinterpretq_u16_f16(v));
            }
            for i in (chunks * 8)..n {
                *dst.add(i) = $bits(src[i]);
            }
        }
    };
}

unpack_row!(unpack_row_e4m3, unpack_e4m3_over256, bits_e4m3_over256);
unpack_row!(unpack_row_e5m2, unpack_e5m2, bits_e5m2);

macro_rules! gemm_bias_kernel {
    ($name:ident, $portable:path, $unpack:ident, $unpack_row:ident, $scale:expr,
     $decode:path, $sc:path) => {
        pub(crate) fn $name(
            a: &[u8],
            b: &[u8],
            c: &[u8],
            d: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) {
            debug_assert!(
                a.len() >= m * k && b.len() >= n * k && c.len() >= m * n * 2 && d.len() >= m * n
            );
            let m_main = (m / BIAS_MR) * BIAS_MR;

            // D starts as the widened bias; k-blocks accumulate on top.
            for idx in 0..m_main * n {
                d[idx] = $sc(&c[idx * 2..]);
            }

            let mut aw = [0u16; BIAS_MR * BIAS_KB];
            unsafe {
                for k0 in (0..k).step_by(BIAS_KB) {
                    let kb = BIAS_KB.min(k - k0);
                    let chunks = kb / 8;
                    let tail = chunks * 8;
                    for blk in 0..m_main / BIAS_MR {
                        let i0 = blk * BIAS_MR;
                        for r in 0..BIAS_MR {
                            $unpack_row(
                                &a[(i0 + r) * k + k0..][..kb],
                                aw.as_mut_ptr().add(r * BIAS_KB),
                                kb,
                            );
                        }
                        for j in 0..n {
                            let brow = &b[j * k + k0..][..kb];
                            let mut acc = [vdupq_n_f32(0.0); BIAS_MR];
                            for ck in 0..chunks {
                                let o = ck * 8;
                                let vb = $unpack(brow.as_ptr().add(o));
                                for r in 0..BIAS_MR {
                                    let va = vreinterpretq_f16_u16(vld1q_u16(
                                        aw.as_ptr().add(r * BIAS_KB + o),
                                    ));
                                    // Alternate low/high per chunk parity
                                    // would halve the chains; low+high per
                                    // chunk keeps one acc per row and the
                                    // register file happy at MR = 8.
                                    acc[r] = vfmlalq_low_f16(acc[r], va, vb);
                                    acc[r] = vfmlalq_high_f16(acc[r], va, vb);
                                }
                            }
                            for r in 0..BIAS_MR {
                                let mut s = vaddvq_f32(acc[r]) * $scale;
                                for t in tail..kb {
                                    s += $decode(a[(i0 + r) * k + k0 + t]) * $decode(brow[t]);
                                }
                                d[(i0 + r) * n + j] += s;
                            }
                        }
                    }
                }
            }

            // Rows past the last full block: the portable kernel over the
            // remainder slice (it initializes its own bias rows).
            if m_main < m {
                $portable(
                    &a[m_main * k..],
                    b,
                    &c[m_main * n * 2..],
                    &mut d[m_main * n..],
                    m - m_main,
                    n,
                    k,
                );
            }
        }
    };
}

gemm_bias_kernel!(
    gemm_e4m3_e4m3_f16,
    crate::arch::portable::f8::gemm_e4m3_e4m3_f16,
    unpack_e4m3_over256,
    unpack_row_e4m3,
    E4M3_DOT_SCALE,
    crate::cast::e4m3_to_f32,
    crate::cast::read_f16
);
gemm_bias_kernel!(
    gemm_e4m3_e4m3_bf16,
    crate::arch::portable::f8::gemm_e4m3_e4m3_bf16,
    unpack_e4m3_over256,
    unpack_row_e4m3,
    E4M3_DOT_SCALE,
    crate::cast::e4m3_to_f32,
    crate::cast::read_bf16
);
gemm_bias_kernel!(
    gemm_e5m2_e5m2_f16,
    crate::arch::portable::f8::gemm_e5m2_e5m2_f16,
    unpack_e5m2,
    unpack_row_e5m2,
    1.0f32,
    crate::cast::e5m2_to_f32,
    crate::cast::read_f16
);
gemm_bias_kernel!(
    gemm_e5m2_e5m2_bf16,
    crate::arch::portable::f8::gemm_e5m2_e5m2_bf16,
    unpack_e5m2,
    unpack_row_e5m2,
    1.0f32,
    crate::cast::e5m2_to_f32,
    crate::cast::read_bf16
);

#[cfg(test)]
mod tests {
    use super::*;

    // The FMLAL tier against the exact scalar reference — all 256
    // encodings appear across the operands.
    #[test]
    fn dot_matches_scalar_reference() {
        let a: [u8; 512] = core::array::from_fn(|i| (i % 256) as u8);
        let b: [u8; 512] = core::array::from_fn(|i| ((i * 7 + 3) % 256) as u8);
        // Finite-only slices (skip NaN/Inf encodings for the sum check).
        let fa: [u8; 200] = core::array::from_fn(|i| (i % 100) as u8 | ((i & 1) as u8) << 7);
        let fb: [u8; 200] = core::array::from_fn(|i| ((i * 3 + 5) % 100) as u8);

        let got = dot_e4m3_e4m3(&fa, &fb, 200);
        let mut want = 0.0f64;
        for i in 0..200 {
            want += crate::cast::e4m3_to_f32(fa[i]) as f64 * crate::cast::e4m3_to_f32(fb[i]) as f64;
        }
        assert!(
            (got as f64 - want).abs() <= 1e-3 * (1.0 + want.abs()),
            "e4m3 {got} vs {want}"
        );

        let got = dot_e5m2_e5m2(&fa, &fb, 200);
        let mut want = 0.0f64;
        for i in 0..200 {
            want += crate::cast::e5m2_to_f32(fa[i]) as f64 * crate::cast::e5m2_to_f32(fb[i]) as f64;
        }
        assert!(
            (got as f64 - want).abs() <= 1e-3 * (1.0 + want.abs()),
            "e5m2 {got} vs {want}"
        );

        // NaN-bearing operands propagate.
        assert!(dot_e4m3_e4m3(&a, &b, 512).is_nan() || dot_e4m3_e4m3(&a, &b, 512).is_infinite());
    }

    // FMLAL bias GEMM equals the portable kernel within FP association,
    // with dims off the block sizes so the delegation path runs.
    #[test]
    fn gemm_bias_matches_portable() {
        let (m, n, k) = (11usize, 5usize, 100usize);
        let a: alloc::vec::Vec<u8> = (0..m * k).map(|i| (i % 100) as u8).collect();
        let b: alloc::vec::Vec<u8> = (0..n * k).map(|i| ((i * 13 + 7) % 100) as u8).collect();
        let c: alloc::vec::Vec<u8> = (0..m * n)
            .flat_map(|i| crate::cast::f32_to_f16((i % 10) as f32 * 0.1).to_le_bytes())
            .collect();

        let mut got = alloc::vec![0.0f32; m * n];
        let mut want = alloc::vec![0.0f32; m * n];
        gemm_e4m3_e4m3_f16(&a, &b, &c, &mut got, m, n, k);
        crate::arch::portable::f8::gemm_e4m3_e4m3_f16(&a, &b, &c, &mut want, m, n, k);
        for i in 0..m * n {
            assert!(
                (got[i] - want[i]).abs() <= 1e-3 * (1.0 + want[i].abs()),
                "[{i}] {} vs {}",
                got[i],
                want[i]
            );
        }

        let mut got5 = alloc::vec![0.0f32; m * n];
        let mut want5 = alloc::vec![0.0f32; m * n];
        gemm_e5m2_e5m2_bf16(&a, &b, &c, &mut got5, m, n, k);
        crate::arch::portable::f8::gemm_e5m2_e5m2_bf16(&a, &b, &c, &mut want5, m, n, k);
        for i in 0..m * n {
            assert!(
                (got5[i] - want5[i]).abs() <= 1e-3 * (1.0 + want5[i].abs()),
                "e5m2 [{i}] {} vs {}",
                got5[i],
                want5[i]
            );
        }
    }
}
