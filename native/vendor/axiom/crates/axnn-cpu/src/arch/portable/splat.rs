//! Gaussian-splat tile compositing for 3DGS (rendering "Kernel B").
//!
//! The counterpart of `sh` (Kernel A): after projection, tile binning
//! and a front-to-back depth sort — all upstream of this crate — each
//! tile owns a sorted list of splat indices. Per pixel, splats blend
//! front-to-back under transmittance `T`:
//!
//! ```text
//! power = -½·(a·dx² + c·dy²) - b·dx·dy         (conic = inverse 2D cov)
//! alpha = min(0.99, opacity · exp(power))       skip if power > 0
//! skip  if alpha < 1/255                        (reference thresholds)
//! stop  if T·(1-alpha) < 1e-4                   (before compositing)
//! C    += rgb · alpha · T;   T *= 1 - alpha
//! ```
//!
//! The kernel vectorizes across a strip of [`PW`] contiguous pixels in
//! one row (lane = pixel): `dy` is a per-splat scalar, `dx` a lane
//! ramp, so `power` is a quadratic in `dx` — two FMAs — followed by one
//! [`PW`]-wide `exp` and masked blends. A lane retires when its
//! transmittance crosses the 1e-4 floor; the splat loop exits when
//! every lane has retired, which is the early termination that makes
//! front-to-back order pay.
//!
//! Splat geometry (center, conic, opacity) stays f32 — it is loaded
//! once per splat per strip. The color table is where mixed precision
//! pays: colors are read three scalars per splat and broadcast, so the
//! decode cost is amortized over the strip while the underlying table
//! footprint shrinks 2× (F16) or 4× (E4M3/UF8). Whether that footprint
//! produces an additional cache win depends on the caller's binning and
//! working-set layout. Accumulation is always f32.

use core::simd::prelude::*;

use super::vec::exp_f32_n;

/// Pixels per strip — one row segment processed per call. Eight lanes
/// is this crate's measured sweet spot (see `f8`): wide enough to fill
/// 128-bit pipes twice over, narrow enough that early termination
/// still bites.
pub(crate) const PW: usize = 8;

/// Reference thresholds from the 3DGS rasterizer.
const ALPHA_MIN: f32 = 1.0 / 255.0;
const ALPHA_MAX: f32 = 0.99;
const T_EPS: f32 = 1e-4;

#[inline(always)]
fn color_f32(c: &[f32], i: usize) -> f32 {
    c[i]
}
#[inline(always)]
fn color_f16(c: &[u8], i: usize) -> f32 {
    crate::cast::read_f16(&c[i * 2..])
}
#[inline(always)]
fn color_e4m3(c: &[u8], i: usize) -> f32 {
    crate::cast::e4m3_to_f32(c[i])
}
#[inline(always)]
fn color_uf8e4m4(c: &[u8], i: usize) -> f32 {
    crate::cast::uf8e4m4_to_f32(c[i])
}
#[inline(always)]
fn color_uf8e5m3(c: &[u8], i: usize) -> f32 {
    crate::cast::uf8e5m3_to_f32(c[i])
}

macro_rules! composite_strip_kernel {
    ($name:ident, $cty:ty, $get:ident, $cstride:expr) => {
        /// Composite `ids` (front-to-back) over one `PW`-pixel strip at
        /// row `y`, pixels `x0 .. x0 + PW`.
        ///
        /// Splat SoA: `center` `[x,y]`-interleaved, `conic` `[a,b,c]`-
        /// interleaved, `opacity` scalar, `color` 3 channels per splat.
        /// `rgb` is planar `[r; PW][g; PW][b; PW]` and `t` per-pixel
        /// transmittance; both are read *and* written, so a caller can
        /// feed chunked splat lists — initialize `rgb = 0`, `t = 1`
        /// before the first chunk. Returns how many list entries were
        /// consumed (short on all-lanes early termination).
        pub(crate) fn $name(
            center: &[f32],
            conic: &[f32],
            opacity: &[f32],
            color: &[$cty],
            ids: &[u32],
            x0: f32,
            y: f32,
            rgb: &mut [f32],
            t: &mut [f32],
        ) -> usize {
            debug_assert!(rgb.len() >= 3 * PW && t.len() >= PW);
            let px =
                Simd::<f32, PW>::splat(x0) + Simd::from_array(core::array::from_fn(|l| l as f32));

            let mut acc_r = Simd::<f32, PW>::from_slice(&rgb[..PW]);
            let mut acc_g = Simd::<f32, PW>::from_slice(&rgb[PW..2 * PW]);
            let mut acc_b = Simd::<f32, PW>::from_slice(&rgb[2 * PW..3 * PW]);
            let mut tr = Simd::<f32, PW>::from_slice(&t[..PW]);

            let t_eps = Simd::splat(T_EPS);
            // A lane is live until its transmittance would cross T_EPS.
            let mut live = tr.simd_ge(t_eps);
            let mut consumed = ids.len();

            for (k, &id) in ids.iter().enumerate() {
                let s = id as usize;
                let (cx, cy) = (center[s * 2], center[s * 2 + 1]);
                let (a, b, c) = (conic[s * 3], conic[s * 3 + 1], conic[s * 3 + 2]);
                let dy = y - cy;

                // power = -½a·dx² - (b·dy)·dx - ½c·dy², two FMAs in dx.
                let dx = px - Simd::splat(cx);
                let qa = Simd::splat(-0.5 * a);
                let qb = Simd::splat(-b * dy);
                let qc = Simd::splat(-0.5 * c * dy * dy);
                let power = (qa * dx + qb) * dx + qc;

                let alpha = (Simd::splat(opacity[s]) * exp_f32_n::<PW>(power))
                    .simd_min(Simd::splat(ALPHA_MAX));
                let hit =
                    power.simd_le(Simd::splat(0.0)) & alpha.simd_ge(Simd::splat(ALPHA_MIN)) & live;

                if !hit.any() {
                    continue;
                }

                // Lanes whose T would cross the floor retire *without*
                // compositing this splat (reference semantics).
                let test_t = tr * (Simd::splat(1.0) - alpha);
                let blend = hit & test_t.simd_ge(t_eps);
                live = live & !(hit & test_t.simd_lt(t_eps));

                let w = blend.select(alpha * tr, Simd::splat(0.0));
                let cr = $get(color, s * $cstride);
                let cg = $get(color, s * $cstride + 1);
                let cb = $get(color, s * $cstride + 2);
                acc_r = Simd::splat(cr) * w + acc_r;
                acc_g = Simd::splat(cg) * w + acc_g;
                acc_b = Simd::splat(cb) * w + acc_b;
                tr = blend.select(test_t, tr);

                if !live.any() {
                    consumed = k + 1;
                    break;
                }
            }

            acc_r.copy_to_slice(&mut rgb[..PW]);
            acc_g.copy_to_slice(&mut rgb[PW..2 * PW]);
            acc_b.copy_to_slice(&mut rgb[2 * PW..3 * PW]);
            tr.copy_to_slice(&mut t[..PW]);
            consumed
        }
    };
}

// The `$cstride` is in *elements* of the color slice: 3 f32s, 3 packed
// f16 (2 bytes each, indexed by element), or 3 e4m3 bytes.
composite_strip_kernel!(composite_strip_f32, f32, color_f32, 3);
composite_strip_kernel!(composite_strip_f16, u8, color_f16, 3);
composite_strip_kernel!(composite_strip_e4m3, u8, color_e4m3, 3);
// Only the post-activation RGB table is unsigned. Geometry and the conic
// cross-term retain their signed f32 representation.
composite_strip_kernel!(composite_strip_uf8e4m4, u8, color_uf8e4m4, 3);
composite_strip_kernel!(composite_strip_uf8e5m3, u8, color_uf8e5m3, 3);

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    // Scalar reference — the 3DGS per-pixel loop, verbatim.
    fn reference_pixel(
        center: &[f32],
        conic: &[f32],
        opacity: &[f32],
        color: &[f32],
        ids: &[u32],
        px: f32,
        py: f32,
    ) -> ([f32; 3], f32) {
        let (mut c, mut t) = ([0.0f32; 3], 1.0f32);
        for &id in ids {
            let s = id as usize;
            let dx = px - center[s * 2];
            let dy = py - center[s * 2 + 1];
            let power = -0.5 * (conic[s * 3] * dx * dx + conic[s * 3 + 2] * dy * dy)
                - conic[s * 3 + 1] * dx * dy;
            if power > 0.0 {
                continue;
            }
            let alpha = (opacity[s] * axmath::exp_f32(power)).min(ALPHA_MAX);
            if alpha < ALPHA_MIN {
                continue;
            }
            let test_t = t * (1.0 - alpha);
            if test_t < T_EPS {
                break;
            }
            for ch in 0..3 {
                c[ch] += color[s * 3 + ch] * alpha * t;
            }
            t = test_t;
        }
        (c, t)
    }

    fn scene(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>) {
        let mut center = Vec::new();
        let mut conic = Vec::new();
        let mut opacity = Vec::new();
        let mut color = Vec::new();
        for i in 0..n {
            center.push((i * 7 % 16) as f32 + 0.3);
            center.push((i * 11 % 16) as f32 + 0.6);
            let inv_s2 = 1.0 / (1.0 + (i % 5) as f32); // 1/σ²
            conic.push(inv_s2);
            conic.push(0.1 * inv_s2 * if i % 2 == 0 { 1.0 } else { -1.0 });
            conic.push(1.3 * inv_s2);
            opacity.push(0.2 + 0.15 * (i % 4) as f32);
            for ch in 0..3 {
                color.push(((i * 3 + ch) % 10) as f32 * 0.1);
            }
        }
        let ids = (0..n as u32).collect();
        (center, conic, opacity, color, ids)
    }

    #[test]
    fn matches_scalar_reference() {
        let (center, conic, opacity, color, ids) = scene(64);
        let (x0, y) = (4.0f32, 7.0f32);
        let mut rgb = [0.0f32; 3 * PW];
        let mut t = [1.0f32; PW];
        composite_strip_f32(
            &center, &conic, &opacity, &color, &ids, x0, y, &mut rgb, &mut t,
        );

        for l in 0..PW {
            let (want_c, want_t) =
                reference_pixel(&center, &conic, &opacity, &color, &ids, x0 + l as f32, y);
            for ch in 0..3 {
                let got = rgb[ch * PW + l];
                assert!(
                    (got - want_c[ch]).abs() <= 2e-4 * (1.0 + want_c[ch].abs()),
                    "lane {l} ch {ch}: {got} vs {}",
                    want_c[ch]
                );
            }
            assert!(
                (t[l] - want_t).abs() <= 2e-4 * (1.0 + want_t.abs()),
                "lane {l} T"
            );
        }
    }

    #[test]
    fn fp8_and_f16_match_f32_on_decoded_colors() {
        let (center, conic, opacity, _, ids) = scene(48);
        // Colors born quantized, then decoded for the f32 kernel — the
        // variants must agree bit-for-bit (identical math post-decode).
        let ce4: Vec<u8> = (0..48 * 3).map(|i| (i % 100) as u8).collect();
        let cf16: Vec<u8> = (0..48 * 3)
            .flat_map(|i| crate::cast::f32_to_f16((i % 50) as f32 * 0.02).to_le_bytes())
            .collect();
        let ce4_f32: Vec<f32> = ce4.iter().map(|&b| crate::cast::e4m3_to_f32(b)).collect();
        let cf16_f32: Vec<f32> = (0..48 * 3)
            .map(|i| crate::cast::read_f16(&cf16[i * 2..]))
            .collect();

        for (qcolor_run, ref_colors) in [
            (0usize, &ce4_f32), // marker: 0 → e4m3, 1 → f16
            (1usize, &cf16_f32),
        ] {
            let mut rgb_q = [0.0f32; 3 * PW];
            let mut t_q = [1.0f32; PW];
            let mut rgb_r = [0.0f32; 3 * PW];
            let mut t_r = [1.0f32; PW];
            let n_q = if qcolor_run == 0 {
                composite_strip_e4m3(
                    &center, &conic, &opacity, &ce4, &ids, 2.0, 5.0, &mut rgb_q, &mut t_q,
                )
            } else {
                composite_strip_f16(
                    &center, &conic, &opacity, &cf16, &ids, 2.0, 5.0, &mut rgb_q, &mut t_q,
                )
            };
            let n_r = composite_strip_f32(
                &center, &conic, &opacity, ref_colors, &ids, 2.0, 5.0, &mut rgb_r, &mut t_r,
            );
            assert_eq!(n_q, n_r);
            assert_eq!(rgb_q, rgb_r);
            assert_eq!(t_q, t_r);
        }
    }

    #[test]
    fn unsigned_color_variants_match_f32_on_decoded_colors() {
        let (center, conic, opacity, _, ids) = scene(48);
        let raw: Vec<u8> = (0..48 * 3).map(|i| (i % 0xF0) as u8).collect();
        let refs = [
            raw.iter()
                .map(|&b| crate::cast::uf8e4m4_to_f32(b))
                .collect::<Vec<_>>(),
            raw.iter()
                .map(|&b| crate::cast::uf8e5m3_to_f32(b))
                .collect::<Vec<_>>(),
        ];
        for (layout, ref_color) in refs.iter().enumerate() {
            let mut rgb_q = [0.0f32; 3 * PW];
            let mut t_q = [1.0f32; PW];
            let mut rgb_r = [0.0f32; 3 * PW];
            let mut t_r = [1.0f32; PW];
            let consumed_q = if layout == 0 {
                composite_strip_uf8e4m4(
                    &center, &conic, &opacity, &raw, &ids, 2.0, 5.0, &mut rgb_q, &mut t_q,
                )
            } else {
                composite_strip_uf8e5m3(
                    &center, &conic, &opacity, &raw, &ids, 2.0, 5.0, &mut rgb_q, &mut t_q,
                )
            };
            let consumed_r = composite_strip_f32(
                &center, &conic, &opacity, ref_color, &ids, 2.0, 5.0, &mut rgb_r, &mut t_r,
            );
            assert_eq!(consumed_q, consumed_r);
            assert_eq!(rgb_q, rgb_r);
            assert_eq!(t_q, t_r);
        }
    }

    #[test]
    fn unsigned_fraction_bit_reduces_composited_color_error() {
        let (center, conic, opacity, _, ids) = scene(96);
        let color: Vec<f32> = (0..96 * 3)
            .map(|i| 0.005 + 15.0 * ((i * 101 % 1_009) as f32 / 1_009.0))
            .collect();
        let signed: Vec<u8> = color
            .iter()
            .map(|&v| crate::fp8::f32_to_e4m3_rne(v))
            .collect();
        let unsigned: Vec<u8> = color
            .iter()
            .map(|&v| crate::cast::f32_to_uf8e4m4_rne(v))
            .collect();
        let (mut signed_error, mut unsigned_error) = (0.0f64, 0.0f64);
        let (mut signed_max, mut unsigned_max) = (0.0f64, 0.0f64);

        for strip in 0..24 {
            let x = 0.25 + (strip % 6) as f32 * PW as f32;
            let y = 0.5 + (strip / 6) as f32 * 3.25;
            let mut reference_rgb = [0.0; 3 * PW];
            let mut signed_rgb = [0.0; 3 * PW];
            let mut unsigned_rgb = [0.0; 3 * PW];
            let mut reference_t = [1.0; PW];
            let mut signed_t = [1.0; PW];
            let mut unsigned_t = [1.0; PW];
            let reference_count = composite_strip_f32(
                &center,
                &conic,
                &opacity,
                &color,
                &ids,
                x,
                y,
                &mut reference_rgb,
                &mut reference_t,
            );
            let signed_count = composite_strip_e4m3(
                &center,
                &conic,
                &opacity,
                &signed,
                &ids,
                x,
                y,
                &mut signed_rgb,
                &mut signed_t,
            );
            let unsigned_count = composite_strip_uf8e4m4(
                &center,
                &conic,
                &opacity,
                &unsigned,
                &ids,
                x,
                y,
                &mut unsigned_rgb,
                &mut unsigned_t,
            );
            assert_eq!(signed_count, reference_count);
            assert_eq!(unsigned_count, reference_count);
            assert_eq!(signed_t, reference_t);
            assert_eq!(unsigned_t, reference_t);
            for lane in 0..3 * PW {
                let signed_delta = f64::from(signed_rgb[lane] - reference_rgb[lane]);
                let unsigned_delta = f64::from(unsigned_rgb[lane] - reference_rgb[lane]);
                signed_error += signed_delta * signed_delta;
                unsigned_error += unsigned_delta * unsigned_delta;
                signed_max = signed_max.max(signed_delta.abs());
                unsigned_max = unsigned_max.max(unsigned_delta.abs());
            }
        }
        assert!(
            unsigned_error < signed_error * 0.35,
            "composited color squared error: UF8={unsigned_error}, E4M3={signed_error}",
        );
        assert!(
            unsigned_max < signed_max,
            "composited color maximum error: UF8={unsigned_max}, E4M3={signed_max}",
        );
    }

    #[test]
    fn early_termination_short_circuits() {
        // A stack of near-opaque splats centered on the strip drives T
        // through the floor; the tail must not be consumed.
        let n = 64;
        let mut center = Vec::new();
        let mut conic = Vec::new();
        let mut opacity = Vec::new();
        let mut color = Vec::new();
        for i in 0..n {
            center.push(3.5);
            center.push(0.0);
            conic.push(0.001); // huge footprint → covers all lanes
            conic.push(0.0);
            conic.push(0.001);
            opacity.push(0.95);
            for _ in 0..3 {
                color.push(0.5 + (i % 2) as f32 * 0.1);
            }
        }
        let ids: Vec<u32> = (0..n as u32).collect();
        let mut rgb = [0.0f32; 3 * PW];
        let mut t = [1.0f32; PW];
        let consumed = composite_strip_f32(
            &center, &conic, &opacity, &color, &ids, 0.0, 0.0, &mut rgb, &mut t,
        );
        assert!(
            consumed < n,
            "expected early termination, consumed {consumed}"
        );
        // Every lane's T is parked just above/at the floor crossing.
        for l in 0..PW {
            assert!(t[l] >= 0.0 && t[l] < 0.01, "lane {l} T = {}", t[l]);
        }
    }

    #[test]
    fn chunked_lists_compose() {
        // Feeding the list in two chunks must equal one pass.
        let (center, conic, opacity, color, ids) = scene(40);
        let mut rgb1 = [0.0f32; 3 * PW];
        let mut t1 = [1.0f32; PW];
        composite_strip_f32(
            &center, &conic, &opacity, &color, &ids, 1.0, 3.0, &mut rgb1, &mut t1,
        );

        let mut rgb2 = [0.0f32; 3 * PW];
        let mut t2 = [1.0f32; PW];
        composite_strip_f32(
            &center,
            &conic,
            &opacity,
            &color,
            &ids[..17],
            1.0,
            3.0,
            &mut rgb2,
            &mut t2,
        );
        composite_strip_f32(
            &center,
            &conic,
            &opacity,
            &color,
            &ids[17..],
            1.0,
            3.0,
            &mut rgb2,
            &mut t2,
        );
        assert_eq!(rgb1, rgb2);
        assert_eq!(t1, t2);
    }
}
