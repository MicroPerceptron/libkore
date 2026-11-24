//! Spherical-harmonics → RGB evaluation for 3D Gaussian splatting
//! (rendering "Kernel A").
//!
//! Each Gaussian stores SH coefficients for view-dependent color: for a
//! caller-normalized view direction `d`, channel `c`'s color is
//! `Σ_k basis_k(d) · coeff[c][k]`, where the basis is the real spherical
//! harmonics up to `deg` (deg 3 = 16 coeffs, the 3DGS default). The basis
//! is direction-dependent but shared across the three channels, so it is
//! evaluated once per Gaussian and reused across R/G/B — the transpose of
//! the GEMM's weight-row reuse.
//!
//! The coefficients are the bandwidth term — 48 per Gaussian at deg 3,
//! dwarfing the geometry — so storing them in E4M3/E5M2 cuts the
//! per-Gaussian footprint ~4× over f32. Each per-channel reduction rides
//! the shared `f8` widen (F16C / `fcvtl`), monomorphized per SH degree so
//! the whole channel is one wide product + reduction.
//!
//! Output is the raw SH sum; the caller applies the 3DGS activation
//! `max(x + 0.5, 0)`.

use core::simd::prelude::*;

use super::f8;

/// Max SH coefficients (degree 3 → 16).
const NC_MAX: usize = 16;

// Real spherical-harmonics constants — identical to the 3DGS reference.
const SH_C0: f32 = 0.282_094_79;
const SH_C1: f32 = 0.488_602_5;
const SH_C2: [f32; 5] = [
    1.092_548_4,
    -1.092_548_4,
    0.315_391_57,
    -1.092_548_4,
    0.546_274_2,
];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_33,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

/// Coefficient count for an SH degree (0..=3): `(deg + 1)²`.
#[inline(always)]
pub(crate) const fn sh_coeff_count(deg: usize) -> usize {
    (deg + 1) * (deg + 1)
}

/// Evaluate the real-SH basis for a unit direction into `b[..NC]`
/// (entries past NC stay zero). Homogeneous in `(x,y,z)`, so `d` must be
/// normalized by the caller.
#[inline(always)]
fn sh_basis(x: f32, y: f32, z: f32, deg: usize) -> [f32; NC_MAX] {
    let mut b = [0.0f32; NC_MAX];
    b[0] = SH_C0;
    if deg >= 1 {
        b[1] = -SH_C1 * y;
        b[2] = SH_C1 * z;
        b[3] = -SH_C1 * x;
    }
    if deg >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, yz, xz) = (x * y, y * z, x * z);
        b[4] = SH_C2[0] * xy;
        b[5] = SH_C2[1] * yz;
        b[6] = SH_C2[2] * (2.0 * zz - xx - yy);
        b[7] = SH_C2[3] * xz;
        b[8] = SH_C2[4] * (xx - yy);
        if deg >= 3 {
            b[9] = SH_C3[0] * y * (3.0 * xx - yy);
            b[10] = SH_C3[1] * xy * z;
            b[11] = SH_C3[2] * y * (4.0 * zz - xx - yy);
            b[12] = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
            b[13] = SH_C3[4] * x * (4.0 * zz - xx - yy);
            b[14] = SH_C3[5] * z * (xx - yy);
            b[15] = SH_C3[6] * x * (xx - 3.0 * yy);
        }
    }
    b
}

// Every kernel dispatches `deg` once to a `(DEG, NC)`-monomorphized body,
// so each degree's three channel dots are straight-line 8-wide chunks
// (16-lane single products measured slower: past 8 lanes the ops stop
// lowering to the paired NEON forms), and `sh_basis` folds its branches
// under const DEG.

/// Dispatch `deg` (0..=3) to a monomorphized body via `$body::<DEG, NC>`.
macro_rules! sh_dispatch {
    ($body:ident, $deg:expr, $($arg:expr),+) => {
        match $deg {
            0 => $body::<0, 1>($($arg),+),
            1 => $body::<1, 4>($($arg),+),
            2 => $body::<2, 9>($($arg),+),
            _ => $body::<3, 16>($($arg),+),
        }
    };
}

/// Pairwise add (`faddp`): `[a0+a1, a2+a3, b0+b1, b2+b3]`.
#[inline(always)]
fn faddp4(a: Simd<f32, 4>, b: Simd<f32, 4>) -> Simd<f32, 4> {
    let even: Simd<f32, 4> = simd_swizzle!(a, b, [0, 2, 4, 6]);
    let odd: Simd<f32, 4> = simd_swizzle!(a, b, [1, 3, 5, 7]);
    even + odd
}

/// Fold 8 lanes to 4 (halving add).
#[inline(always)]
fn fold8(v: Simd<f32, 8>) -> Simd<f32, 4> {
    let a = v.to_array();
    Simd::<f32, 4>::from_slice(&a[..4]) + Simd::<f32, 4>::from_slice(&a[4..8])
}

// FP8 kernels: per channel, widen the coeff bytes 8 at a time through the
// shared `f8` unpack (`fcvtl`/`fcvtl2`, `vcvtph2ps` ×2), FMA against the
// basis, and fold to a 4-lane partial. The three channel partials merge
// through two `faddp` levels into one vector holding [R, G, B, ·], which
// is scaled once and stored with a single vector store — no per-channel
// vector→scalar extract (a ~6-cycle escape ×3 per Gaussian) survives.
macro_rules! sh_eval_fp8 {
    ($name:ident, $widen_n:ident, $scale:expr, $decode:path) => {
        /// `out[g*3+c] = Σ_k basis_k(dirs[g]) · decode(coeffs[(g*3+c)*NC + k])`.
        ///
        /// `dirs`: N×3 unit directions (caller-normalized). `coeffs`:
        /// N×3×NC packed FP8, channel-major. `out`: N×3. `deg` in 0..=3.
        /// Output is the raw SH sum — the caller applies `max(x + 0.5, 0)`.
        pub(crate) fn $name(dirs: &[f32], coeffs: &[u8], out: &mut [f32], n: usize, deg: usize) {
            debug_assert!(deg <= 3);

            /// One channel's 4-lane partial (lane sum = the channel dot,
            /// in the widen's domain — /256 for E4M3; scalar-remainder
            /// coeffs are handled in `rgb`, exactly, outside the scale).
            #[inline(always)]
            fn ch_partial<const NC: usize>(cf: &[u8], b: &[f32; NC_MAX]) -> Simd<f32, 4> {
                if NC == 16 {
                    let lo = f8::$widen_n::<8>(cf, 0) * Simd::<f32, 8>::from_slice(&b[..8]);
                    let s = f8::$widen_n::<8>(cf, 8) * Simd::<f32, 8>::from_slice(&b[8..16]) + lo;
                    fold8(s)
                } else if NC == 9 {
                    fold8(f8::$widen_n::<8>(cf, 0) * Simd::<f32, 8>::from_slice(&b[..8]))
                } else if NC == 4 {
                    f8::$widen_n::<4>(cf, 0) * Simd::<f32, 4>::from_slice(&b[..4])
                } else {
                    Simd::splat(0.0)
                }
            }

            /// All three channels of one Gaussian → `[R, G, B, ·]`.
            #[inline(always)]
            fn rgb<const NC: usize>(coeffs: &[u8], base: usize, b: &[f32; NC_MAX]) -> Simd<f32, 4> {
                let p0 = ch_partial::<NC>(&coeffs[base..][..NC], b);
                let p1 = ch_partial::<NC>(&coeffs[base + NC..][..NC], b);
                let p2 = ch_partial::<NC>(&coeffs[base + 2 * NC..][..NC], b);
                // [ΣR, ΣG, ΣB, ΣB], one scale multiply for all channels.
                let v = faddp4(faddp4(p0, p1), faddp4(p2, p2)) * Simd::splat($scale);
                if NC == 9 || NC == 1 {
                    // Coeffs past the vector part decode exactly (unscaled).
                    let t = NC - 1;
                    v + Simd::from_array([
                        $decode(coeffs[base + t]) * b[t],
                        $decode(coeffs[base + NC + t]) * b[t],
                        $decode(coeffs[base + 2 * NC + t]) * b[t],
                        0.0,
                    ])
                } else {
                    v
                }
            }

            #[inline(always)]
            fn body<const DEG: usize, const NC: usize>(
                dirs: &[f32],
                coeffs: &[u8],
                out: &mut [f32],
                n: usize,
            ) {
                debug_assert!(
                    dirs.len() >= n * 3 && coeffs.len() >= n * 3 * NC && out.len() >= n * 3
                );
                // Two Gaussians per iteration keep six independent
                // widen→FMA→fold chains in flight. The 4-lane RGB store
                // clobbers the next Gaussian's R, which its own store
                // then rewrites — so Gaussians at the array end (no slot
                // after their B) take the lane-extract path instead.
                let full = n.saturating_sub(1);
                let pairs = full / 2;
                for gp in 0..pairs {
                    let g = gp * 2;
                    let (d0, d1) = (&dirs[g * 3..], &dirs[(g + 1) * 3..]);
                    let b0 = sh_basis(d0[0], d0[1], d0[2], DEG);
                    let b1 = sh_basis(d1[0], d1[1], d1[2], DEG);
                    let v0 = rgb::<NC>(coeffs, g * 3 * NC, &b0);
                    let v1 = rgb::<NC>(coeffs, (g + 1) * 3 * NC, &b1);
                    v0.copy_to_slice(&mut out[g * 3..]);
                    v1.copy_to_slice(&mut out[(g + 1) * 3..]);
                }
                for g in (pairs * 2)..n {
                    let d = &dirs[g * 3..];
                    let b = sh_basis(d[0], d[1], d[2], DEG);
                    let v = rgb::<NC>(coeffs, g * 3 * NC, &b).to_array();
                    out[g * 3..g * 3 + 3].copy_from_slice(&v[..3]);
                }
            }

            sh_dispatch!(body, deg, dirs, coeffs, out, n)
        }
    };
}

sh_eval_fp8!(
    sh_eval_e4m3,
    widen_e4m3_over256_n,
    f8::E4M3_SCALE,
    crate::cast::e4m3_to_f32
);
sh_eval_fp8!(sh_eval_e5m2, widen_e5m2_n, 1.0f32, crate::cast::e5m2_to_f32);

/// f32-coefficient baseline (same basis, wide vector dots) — the
/// high-precision reference the FP8 kernels trade footprint against.
pub(crate) fn sh_eval_f32(dirs: &[f32], coeffs: &[f32], out: &mut [f32], n: usize, deg: usize) {
    debug_assert!(deg <= 3);

    #[inline(always)]
    fn ch_dot<const NC: usize>(cf: &[f32], b: &[f32; NC_MAX]) -> f32 {
        if NC == 16 {
            let lo = Simd::<f32, 8>::from_slice(&cf[..8]) * Simd::<f32, 8>::from_slice(&b[..8]);
            let s =
                Simd::<f32, 8>::from_slice(&cf[8..16]) * Simd::<f32, 8>::from_slice(&b[8..16]) + lo;
            f8::hsum(s)
        } else if NC == 9 {
            f8::hsum(Simd::<f32, 8>::from_slice(&cf[..8]) * Simd::<f32, 8>::from_slice(&b[..8]))
                + cf[8] * b[8]
        } else if NC == 4 {
            f8::hsum(Simd::<f32, 4>::from_slice(&cf[..4]) * Simd::<f32, 4>::from_slice(&b[..4]))
        } else {
            cf[0] * b[0]
        }
    }

    #[inline(always)]
    fn body<const DEG: usize, const NC: usize>(
        dirs: &[f32],
        coeffs: &[f32],
        out: &mut [f32],
        n: usize,
    ) {
        debug_assert!(dirs.len() >= n * 3 && coeffs.len() >= n * 3 * NC && out.len() >= n * 3);
        for g in 0..n {
            let d = &dirs[g * 3..];
            let b = sh_basis(d[0], d[1], d[2], DEG);
            let base = g * 3 * NC;
            for c in 0..3 {
                out[g * 3 + c] = ch_dot::<NC>(&coeffs[base + c * NC..][..NC], &b);
            }
        }
    }

    sh_dispatch!(body, deg, dirs, coeffs, out, n)
}

/// f16-coefficient variant — the hardware widen (`fcvtl` / `vcvtph2ps`,
/// one instruction, no bit-twiddling) at 2× storage vs f32. Often the CPU
/// sweet spot: near-f32 throughput for half the coefficient footprint.
pub(crate) fn sh_eval_f16(dirs: &[f32], coeffs: &[f16], out: &mut [f32], n: usize, deg: usize) {
    debug_assert!(deg <= 3);

    #[inline(always)]
    fn ch_dot<const NC: usize>(cf: &[f16], b: &[f32; NC_MAX]) -> f32 {
        if NC == 16 {
            let lo: Simd<f32, 8> = Simd::<f16, 8>::from_slice(&cf[..8]).cast();
            let hi: Simd<f32, 8> = Simd::<f16, 8>::from_slice(&cf[8..16]).cast();
            let s = hi * Simd::<f32, 8>::from_slice(&b[8..16])
                + lo * Simd::<f32, 8>::from_slice(&b[..8]);
            f8::hsum(s)
        } else if NC == 9 {
            let wf: Simd<f32, 8> = Simd::<f16, 8>::from_slice(&cf[..8]).cast();
            f8::hsum(wf * Simd::<f32, 8>::from_slice(&b[..8])) + cf[8] as f32 * b[8]
        } else if NC == 4 {
            let wf: Simd<f32, 4> = Simd::<f16, 4>::from_slice(&cf[..4]).cast();
            f8::hsum(wf * Simd::<f32, 4>::from_slice(&b[..4]))
        } else {
            cf[0] as f32 * b[0]
        }
    }

    #[inline(always)]
    fn body<const DEG: usize, const NC: usize>(
        dirs: &[f32],
        coeffs: &[f16],
        out: &mut [f32],
        n: usize,
    ) {
        debug_assert!(dirs.len() >= n * 3 && coeffs.len() >= n * 3 * NC && out.len() >= n * 3);
        for g in 0..n {
            let d = &dirs[g * 3..];
            let b = sh_basis(d[0], d[1], d[2], DEG);
            let base = g * 3 * NC;
            for c in 0..3 {
                out[g * 3 + c] = ch_dot::<NC>(&coeffs[base + c * NC..][..NC], &b);
            }
        }
    }

    sh_dispatch!(body, deg, dirs, coeffs, out, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sh_eval_matches_scalar() {
        const N: usize = 4;
        const DEG: usize = 3;
        let nc = sh_coeff_count(DEG);
        // Unit directions (no runtime sqrt): axes + the body diagonal.
        let dirs: [f32; N * 3] = [
            1.0,
            0.0,
            0.0, //
            0.0,
            1.0,
            0.0, //
            0.0,
            0.0,
            1.0, //
            0.577_350_3,
            0.577_350_3,
            0.577_350_3,
        ];
        // Finite E4M3 (bytes < 0x7C avoid inf/NaN), varied.
        let coeffs: [u8; N * 3 * NC_MAX] = core::array::from_fn(|i| ((i * 11 + 5) % 100) as u8);
        let mut out = [0.0f32; N * 3];
        sh_eval_e4m3(&dirs, &coeffs, &mut out, N, DEG);

        for g in 0..N {
            let b = sh_basis(dirs[g * 3], dirs[g * 3 + 1], dirs[g * 3 + 2], DEG);
            for c in 0..3 {
                let mut want = 0.0f32;
                for k in 0..nc {
                    want += crate::cast::e4m3_to_f32(coeffs[(g * 3 + c) * nc + k]) * b[k];
                }
                let got = out[g * 3 + c];
                assert!(
                    (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "g{g} c{c}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn sh_eval_f32_matches_fp8() {
        // The f32 baseline agrees with E4M3 once its coeffs are dequantized.
        const N: usize = 3;
        const DEG: usize = 2;
        let dirs: [f32; N * 3] = [
            0.0,
            0.0,
            1.0, //
            1.0,
            0.0,
            0.0, //
            0.577_350_3,
            0.577_350_3,
            0.577_350_3,
        ];
        let bytes: [u8; N * 3 * 9] = core::array::from_fn(|i| ((i * 7 + 3) % 100) as u8);
        let f32c: [f32; N * 3 * 9] = core::array::from_fn(|i| crate::cast::e4m3_to_f32(bytes[i]));
        let mut out_fp8 = [0.0f32; N * 3];
        let mut out_f32 = [0.0f32; N * 3];
        sh_eval_e4m3(&dirs, &bytes, &mut out_fp8, N, DEG);
        sh_eval_f32(&dirs, &f32c, &mut out_f32, N, DEG);
        for i in 0..N * 3 {
            assert!((out_fp8[i] - out_f32[i]).abs() <= 1e-4 * (1.0 + out_f32[i].abs()));
        }
    }

    #[test]
    fn sh_eval_f16_matches_f32() {
        // f16 coeffs agree with the f32 baseline fed the same widened values.
        const N: usize = 3;
        const DEG: usize = 3;
        let dirs: [f32; N * 3] = [
            0.0,
            0.0,
            1.0, //
            0.577_350_3,
            0.577_350_3,
            0.577_350_3, //
            -0.6,
            0.8,
            0.0,
        ];
        let f16c: [f16; N * 3 * NC_MAX] =
            core::array::from_fn(|i| (((i * 13 + 7) % 100) as f32 * 0.02 - 1.0) as f16);
        let f32c: [f32; N * 3 * NC_MAX] = core::array::from_fn(|i| f16c[i] as f32);
        let mut out16 = [0.0f32; N * 3];
        let mut out32 = [0.0f32; N * 3];
        sh_eval_f16(&dirs, &f16c, &mut out16, N, DEG);
        sh_eval_f32(&dirs, &f32c, &mut out32, N, DEG);
        for i in 0..N * 3 {
            assert!((out16[i] - out32[i]).abs() <= 1e-4 * (1.0 + out32[i].abs()));
        }
    }
}
