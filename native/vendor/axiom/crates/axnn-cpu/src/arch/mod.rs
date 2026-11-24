//! ISA dispatch: compile-time tier selection via conditional re-export.
//!
//! ## Architecture
//!
//! Two layers:
//!
//! 1. **Portable tier** (`portable/`): `core::simd`-based kernels that
//!    compile optimally on all targets. Covers float ops (vec, reduce,
//!    gemm, conv, rope, cast, mem) and a scalar qdot fallback.
//!
//! 2. **Arch-specific qdot overrides**: Hand-tuned intrinsics where a
//!    specific instruction sequence matters for quantized dot products:
//!    - `amx/`:      Intel AMX tile matmul (placeholder for GEMM)
//!    - `avx512/`:   512-bit qdot, packed FP4, FP8, BF16, and attention
//!    - `avx_vnni/`: Fused `vpdpbusd` at 256-bit (Alder Lake+)
//!    - `avx2/`:     256-bit `maddubs` + full float override (F16C)
//!    - `sse42/`:    128-bit `maddubs` (SSSE3)
//!    - `sse2/`:     Delegates to portable (no maddubs)
//!    - `neon/`:     AArch64 widening multiply chains
//!
//! ## Tier selection
//!
//! Float kernels: `avx2` provides a full override; AArch64 supplies a NEON
//! tier for vec/reduce/mem plus focused overrides for GEMM, unsigned-float
//! expansion, and compact depth; remaining operations use the portable tier.
//!
//! qdot: highest-priority enabled feature wins.

pub mod portable;
pub(crate) mod scalar;

// ── Arch-specific modules (compiled only when feature-gated) ────────

#[cfg(all(feature = "amx", target_arch = "x86_64"))]
mod amx;

#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
mod avx512;

#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
pub(crate) use self::avx512::attention as avx512_attention;

#[cfg(all(feature = "avx_vnni", target_arch = "x86_64"))]
pub(crate) mod avx_vnni;

#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
pub(crate) mod avx2;

#[cfg(all(feature = "sse42", target_arch = "x86_64"))]
mod sse42;

#[cfg(all(feature = "sse2", target_arch = "x86_64"))]
mod sse2;

// Declared for the whole arch (not the `neon` qdot feature): the f8
// FHM tier inside participates by target feature alone.
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(all(target_arch = "aarch64", feature = "sme"))]
pub(crate) mod sme;

// ── The selected float tier, as a typed fact ────────────────────────
//
// `build.rs` picks the tiers from the target feature set and
// `crate::build_info` reports them as labels; `FLOAT_TIER` is the same
// float/raster choice as a type, so an integrator can `const`-assert that
// a target built for a SIMD class did not silently fall back to portable
// code (a string cannot be matched in a `const` block).  Its `name()` is
// `build_info::raster_isa()` verbatim; the qdot tier stays a label,
// `build_info::qdot_isa()`.

/// The float / raster / memory kernel tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatTier {
    /// AVX2 + FMA + F16C hand-tuned override (x86_64).
    Avx2,
    /// NEON tier (aarch64).
    Neon,
    /// `core::simd` portable kernels (the compiler's vector width).
    Portable,
}

impl FloatTier {
    /// The tier's name, for witness lines — the same label
    /// [`build_info::raster_isa`](crate::build_info::raster_isa) reports,
    /// so a witness line and a benchmark header never disagree.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
            Self::Portable => "portable-simd",
        }
    }
}

/// The float / raster tier this build selected.
pub const FLOAT_TIER: FloatTier = if cfg!(all(feature = "avx2", target_arch = "x86_64")) {
    FloatTier::Avx2
} else if cfg!(target_arch = "aarch64") {
    FloatTier::Neon
} else {
    FloatTier::Portable
};

// ── Float kernels ───────────────────────────────────────────────────
// avx2 provides a hand-tuned full override; AArch64 owns vec/reduce/mem in
// its NEON tier and replaces focused hot paths below. Other families use
// portable.

// bitmat (Q1 bit-packed ops) always uses the portable implementation.
// Arch-specific overrides can be added later for hot paths.
pub(crate) use self::portable::bitmat;

// voxel (DDA / SVO ray traversal) is branch-dominated scalar code —
// portable-only until a SIMD ray-packet tier exists.
pub(crate) use self::portable::voxel;

#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
pub(crate) use self::avx2::{cast, conv, mem, raster, reduce, rope, sh, splat, vec};

#[cfg(target_arch = "aarch64")]
pub(crate) use self::neon::{mem, reduce, vec};

#[cfg(not(any(all(feature = "avx2", target_arch = "x86_64"), target_arch = "aarch64")))]
pub(crate) use self::portable::{mem, reduce, vec};

#[cfg(not(all(feature = "avx2", target_arch = "x86_64")))]
pub(crate) use self::portable::{cast, conv, rope, sh, splat};

#[cfg(target_arch = "aarch64")]
pub(crate) use self::neon::raster;

#[cfg(not(any(target_arch = "aarch64", all(feature = "avx2", target_arch = "x86_64"))))]
pub(crate) use self::portable::raster;

// gemm: sme (opt-in) > avx2 full override > NEON FMA tier > portable.
#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
pub(crate) use self::avx2::gemm;

#[cfg(all(target_arch = "aarch64", feature = "sme"))]
pub(crate) use self::sme::gemm;

#[cfg(all(target_arch = "aarch64", not(feature = "sme")))]
pub(crate) use self::neon::gemm;

#[cfg(not(any(all(feature = "avx2", target_arch = "x86_64"), target_arch = "aarch64")))]
pub(crate) use self::portable::gemm;

// gemv: AVX-512 BF16 rows > AVX2 row blocks > portable dots.
#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
pub(crate) use self::avx512::gemv;

#[cfg(all(feature = "avx2", not(feature = "avx512"), target_arch = "x86_64"))]
pub(crate) use self::avx2::gemv;

#[cfg(not(any(
    all(feature = "avx512", target_arch = "x86_64"),
    all(feature = "avx2", target_arch = "x86_64")
)))]
pub(crate) use self::portable::gemv;

// ── f8: FP8 kernels — avx2 full override > NEON FHM tier > portable ──
//
// The NEON tier upgrades the FP8×FP8 kernels to FMLAL (FEAT_FHM) and
// re-exports the rest of the family from portable. Unlike the qdot
// tiers it needs no cargo feature: `fhm` in the target feature set is
// a compile-time fact (default on Apple M-series and Armv8.4+ targets)
// and emitting FMLAL is always sound there — so the fast tier is the
// default wherever the hardware exists.

#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
pub(crate) use self::avx2::f8;

#[cfg(all(target_arch = "aarch64", target_feature = "fhm"))]
pub(crate) use self::neon::f8;

#[cfg(not(any(
    all(feature = "avx2", target_arch = "x86_64"),
    all(target_arch = "aarch64", target_feature = "fhm"),
)))]
pub(crate) use self::portable::f8;

// Native NVFP4 GEMV follows the selected float tier. The checkpoint stays
// packed; this is a direct consumer, not a conversion path.
#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
pub(crate) use self::avx512::f4;

#[cfg(all(feature = "avx2", not(feature = "avx512"), target_arch = "x86_64"))]
pub(crate) use self::avx2::f4;

#[cfg(not(any(
    all(feature = "avx512", target_arch = "x86_64"),
    all(feature = "avx2", target_arch = "x86_64")
)))]
pub(crate) use self::portable::f4;

// W4A8 needs VNNI in addition to the AVX-512 float tier. A target with
// AVX-512BW but without AVX-512 VNNI still uses the AVX-VNNI 256-bit path.
#[cfg(all(
    feature = "avx512",
    feature = "nvfp4_vnni",
    target_feature = "avx512vnni",
    target_arch = "x86_64"
))]
pub(crate) use self::avx512::f4 as qf4;

#[cfg(all(
    feature = "avx_vnni",
    target_arch = "x86_64",
    not(all(feature = "avx512", target_feature = "avx512vnni"))
))]
pub(crate) use self::avx_vnni::f4 as qf4;

// Unsigned-float storage expansion is selected independently from signed
// FP8 FHM: base NEON materially improves native AArch64 load/store codegen,
// while AVX2 uses explicit F16C/integer sequences. Strict-alignment bare-metal
// targets use element-aligned LD1/ST1 shims so byte/halfword/word-aligned
// public slices keep vector memory traffic without stronger alignment.
#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
pub(crate) use self::avx2::f8 as uf;

#[cfg(target_arch = "aarch64")]
pub(crate) use self::neon::uf;

#[cfg(not(any(all(feature = "avx2", target_arch = "x86_64"), target_arch = "aarch64")))]
pub(crate) use self::portable::f8 as uf;

// ── qdot: highest-priority arch feature wins ────────────────────────
//
// Priority (x86_64): amx > avx512 > avx_vnni > avx2 > sse42 > sse2
// Priority (aarch64): neon
// Fallback: portable scalar

#[cfg(all(feature = "amx", target_arch = "x86_64"))]
pub(crate) use self::amx::qdot;

#[cfg(all(feature = "avx512", target_arch = "x86_64", not(feature = "amx"),))]
pub(crate) use self::avx512::qdot;

#[cfg(all(
    feature = "avx_vnni",
    target_arch = "x86_64",
    not(any(feature = "amx", feature = "avx512")),
))]
pub(crate) use self::avx_vnni::qdot;

#[cfg(all(
    feature = "avx2",
    target_arch = "x86_64",
    not(any(feature = "amx", feature = "avx512", feature = "avx_vnni")),
))]
pub(crate) use self::avx2::qdot;

#[cfg(all(
    feature = "sse42",
    target_arch = "x86_64",
    not(any(
        feature = "amx",
        feature = "avx512",
        feature = "avx_vnni",
        feature = "avx2"
    )),
))]
pub(crate) use self::sse42::qdot;

#[cfg(all(
    feature = "sse2",
    target_arch = "x86_64",
    not(any(
        feature = "amx",
        feature = "avx512",
        feature = "avx_vnni",
        feature = "avx2",
        feature = "sse42"
    )),
))]
pub(crate) use self::sse2::qdot;

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub(crate) use self::neon::qdot;

#[cfg(not(any(
    all(feature = "amx", target_arch = "x86_64"),
    all(feature = "avx512", target_arch = "x86_64"),
    all(feature = "avx_vnni", target_arch = "x86_64"),
    all(feature = "avx2", target_arch = "x86_64"),
    all(feature = "sse42", target_arch = "x86_64"),
    all(feature = "sse2", target_arch = "x86_64"),
    all(feature = "neon", target_arch = "aarch64"),
)))]
pub(crate) use self::portable::qdot;

#[cfg(test)]
mod tests {
    /// The tier the build selected follows the target feature set: a
    /// target that carries the AVX2 class (or is AArch64) must not have
    /// fallen back to portable kernels.  Run on such a host, this is the
    /// proof `build.rs` did its job; run elsewhere it proves the fallback.
    #[test]
    fn selected_tier_follows_the_target_features() {
        use super::{FLOAT_TIER, FloatTier};
        use crate::build_info;
        let avx2_class = cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "f16c"
        ));
        let expected = if avx2_class {
            FloatTier::Avx2
        } else if cfg!(target_arch = "aarch64") {
            FloatTier::Neon
        } else {
            FloatTier::Portable
        };
        assert_eq!(FLOAT_TIER, expected, "float tier vs target features");
        // One vocabulary: the typed tier names itself exactly as the
        // benchmark header does.
        assert_eq!(FLOAT_TIER.name(), build_info::raster_isa());
        if avx2_class {
            let qdot = build_info::qdot_isa();
            assert!(
                !(qdot.starts_with("sse") || qdot.starts_with("portable")),
                "qdot tier {qdot} below the AVX2 class"
            );
        }
    }

    use super::*;

    #[test]
    fn dispatch_fill_pattern() {
        let mut dst = [0u8; 12];
        mem::fill_pattern(&mut dst, &[0xAB, 0xCD, 0xEF], 4);
        assert_eq!(
            dst,
            [
                0xAB, 0xCD, 0xEF, 0xAB, 0xCD, 0xEF, 0xAB, 0xCD, 0xEF, 0xAB, 0xCD, 0xEF
            ]
        );
    }

    #[test]
    fn dispatch_copy_blocks() {
        let src = [1u8, 2, 3, 4, 5, 6];
        let mut dst = [0u8; 6];
        mem::copy_blocks(&src, &mut dst, 3, 2);
        assert_eq!(dst, [1, 2, 3, 4, 5, 6]);
    }

    // ── Raster primitive dispatch tests ──────────────────────────────

    #[test]
    fn dispatch_sub_f32_exact_chunk() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let mut dst = [0.0f32; 8];
        vec::sub_f32(&a, &b, &mut dst);
        assert_eq!(dst, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    }

    #[test]
    fn dispatch_sub_f32_with_tail() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [1.0f32; 9];
        let mut dst = [0.0f32; 9];
        vec::sub_f32(&a, &b, &mut dst);
        assert_eq!(dst[8], 8.0, "tail element must be correct");
        assert_eq!(dst[0], 0.0);
    }

    #[test]
    fn dispatch_cmpge_f32_exact_chunk() {
        let a: [f32; 8] = [2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
        let b: [f32; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut dst = [0u32; 8];
        raster::cmpge_f32(&a, &b, &mut dst);
        assert_eq!(
            dst,
            [
                0xFFFF_FFFF,
                0,
                0xFFFF_FFFF,
                0,
                0xFFFF_FFFF,
                0,
                0xFFFF_FFFF,
                0
            ]
        );
    }

    #[test]
    fn dispatch_cmpge_f32_with_tail() {
        let mut a = [2.0f32; 9];
        let mut b = [1.0f32; 9];
        a[8] = 0.0;
        b[8] = 1.0;
        let mut dst = [0u32; 9];
        raster::cmpge_f32(&a, &b, &mut dst);
        for i in 0..8 {
            assert_eq!(dst[i], 0xFFFF_FFFF, "chunk lane {i} wrong");
        }
        assert_eq!(dst[8], 0x0000_0000, "tail lane must be zero (0.0 < 1.0)");
    }

    #[test]
    fn dispatch_and_masks_with_tail() {
        let a = [0xFFFF_FFFFu32; 9];
        let mut b = [0xFFFF_FFFFu32; 9];
        b[8] = 0;
        let mut dst = [0u32; 9];
        raster::and_masks(&a, &b, &mut dst);
        for i in 0..8 {
            assert_eq!(dst[i], 0xFFFF_FFFF, "chunk lane {i} wrong");
        }
        assert_eq!(dst[8], 0, "tail lane must be zero");
    }

    #[test]
    fn dispatch_maskstore_u32_with_tail() {
        let mut mask = [0xFFFF_FFFFu32; 9];
        mask[8] = 0;
        let src = [0xBEEF_CAFEu32; 9];
        let mut dst = [0u32; 9];
        dst[8] = 0x1234_5678;
        raster::maskstore_u32(&mask, &src, &mut dst);
        for i in 0..8 {
            assert_eq!(dst[i], 0xBEEF_CAFE, "chunk lane {i} wrong");
        }
        assert_eq!(dst[8], 0x1234_5678, "tail sentinel must be preserved");
    }

    #[test]
    fn dispatch_fused_edge_coverage_correctness() {
        let e0 = [1.0f32; 9];
        let e1: [f32; 9] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let e2 = [1.0f32; 9];
        let color = [0xFFFF_FFFFu32; 9];
        let mut fb = [0u32; 9];

        raster::fused_edge_coverage(&e0, &e1, &e2, 0.0, &color, &mut fb);

        for i in 0..9 {
            if i % 2 == 0 {
                assert_eq!(fb[i], 0xFFFF_FFFF, "pixel {i} should be written");
            } else {
                assert_eq!(fb[i], 0x0000_0000, "pixel {i} should be black");
            }
        }
    }

    #[test]
    fn dispatch_fused_edge_coverage_tail_sentinel() {
        let e0 = [1.0f32; 9];
        let e1 = [1.0f32; 9];
        let mut e2 = [1.0f32; 9];
        e2[8] = -1.0;
        let color = [0xDEAD_BEEFu32; 9];
        let mut fb = [0u32; 9];
        fb[8] = 0x1234_5678;

        raster::fused_edge_coverage(&e0, &e1, &e2, 0.0, &color, &mut fb);

        for i in 0..8 {
            assert_eq!(fb[i], 0xDEAD_BEEF, "chunk lane {i} must be written");
        }
        assert_eq!(fb[8], 0x1234_5678, "tail sentinel must be preserved");
    }

    #[test]
    fn dispatch_raster_pipeline_correctness() {
        let e0 = [1.0f32; 9];
        let e1: [f32; 9] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let e2 = [1.0f32; 9];
        let zero = [0.0f32; 9];
        let color = [0xFFFF_FFFFu32; 9];

        let mut m0 = [0u32; 9];
        let mut m1 = [0u32; 9];
        let mut m2 = [0u32; 9];
        let mut coverage = [0u32; 9];
        let mut fb = [0u32; 9];

        raster::cmpge_f32(&e0, &zero, &mut m0);
        raster::cmpge_f32(&e1, &zero, &mut m1);
        raster::cmpge_f32(&e2, &zero, &mut m2);
        raster::and_masks(&m0, &m1, &mut coverage);
        raster::and_masks(&coverage, &m2, &mut m0);
        raster::maskstore_u32(&m0, &color, &mut fb);

        for i in 0..9 {
            if i % 2 == 0 {
                assert_eq!(fb[i], 0xFFFF_FFFF, "pixel {i} should be written");
            } else {
                assert_eq!(fb[i], 0x0000_0000, "pixel {i} should be black");
            }
        }
    }

    #[test]
    fn dispatch_rounded_xrgb_pack_matches_scalar_across_chunks_and_tail() {
        const N: usize = 19;
        let red: [f32; N] = core::array::from_fn(|i| (i as f32 - 2.0) / 16.0);
        let green: [f32; N] = core::array::from_fn(|i| (i as f32 + 0.5) / 255.0);
        let blue: [f32; N] = core::array::from_fn(|i| 1.125 - i as f32 / 20.0);
        let mut actual = [0x1234_5678; N];
        raster::xrgb_pack_rounded(&red, &green, &blue, &mut actual);
        let to_u8 = |v: f32| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u32;
        for i in 0..N {
            assert_eq!(
                actual[i],
                to_u8(red[i]) | to_u8(green[i]) << 8 | to_u8(blue[i]) << 16 | 0xff00_0000,
                "pixel {i}"
            );
        }
    }

    #[test]
    fn dispatch_u8_biased_xrgb_pack_matches_scalar_across_chunks_and_tail() {
        const N: usize = 67; // > 4·W chunks on every tier, plus a tail
        let red: [u8; N] = core::array::from_fn(|i| (i * 7) as u8);
        let green: [u8; N] = core::array::from_fn(|i| (i * 13 + 100) as u8);
        let blue: [u8; N] = core::array::from_fn(|i| (255 - i * 3) as u8);
        let mut actual = [0x1234_5678; N];
        raster::xrgb_pack_u8_biased(&red, &green, &blue, &mut actual);
        let mut want = [0u32; N];
        crate::arch::scalar::raster::xrgb_pack_u8_biased(&red, &green, &blue, &mut want);
        assert_eq!(actual, want);
        // Same lane order as `xrgb_pack`: first slice → bits 7:0.
        for i in 0..N {
            let ch = |v: u8| u32::from(v.wrapping_add(128));
            assert_eq!(
                actual[i],
                0xff00_0000 | ch(blue[i]) << 16 | ch(green[i]) << 8 | ch(red[i]),
                "pixel {i}"
            );
        }
    }

    // ── Shape coverage: the selected tier is bit-exact with scalar ──

    /// 67 lanes: several W-chunks on every tier plus a tail.
    const SHAPE_N: usize = 67;

    fn shape_distances() -> [f32; SHAPE_N] {
        // A row through the top corner arcs of a 40×40 box with radius 6
        // (2.5 px below its top edge), starting left of the box so the row
        // crosses outside → arc → inside → arc → outside.
        let mut d = [0f32; SHAPE_N];
        scalar::raster::rounded_sdf_row_f32(&mut d, -28.5, -17.5, 20.0, 20.0, 6.0);
        d
    }

    #[test]
    fn dispatch_rounded_sdf_row_matches_scalar_bit_exact() {
        for (px0, py, hw, hh, r) in [
            (-28.5f32, 3.5f32, 20.0f32, 20.0f32, 6.0f32),
            (-7.5, -17.5, 6.0, 20.0, 0.0),
            (-100.25, 199.5, 79.0, 100.0, 16.0),
            (0.5, 0.5, 0.5, 0.5, 0.0),
        ] {
            let mut want = [0f32; SHAPE_N];
            scalar::raster::rounded_sdf_row_f32(&mut want, px0, py, hw, hh, r);
            let mut got = [0f32; SHAPE_N];
            raster::rounded_sdf_row_f32(&mut got, px0, py, hw, hh, r);
            assert_eq!(
                got.map(f32::to_bits),
                want.map(f32::to_bits),
                "({px0}, {py}, r={r})"
            );
        }
    }

    #[test]
    fn dispatch_sdf_coverage_and_ring_match_scalar() {
        let outer = shape_distances();
        let inner: [f32; SHAPE_N] = core::array::from_fn(|i| outer[i] + 1.0);
        let (mut want, mut got) = ([0u8; SHAPE_N], [0u8; SHAPE_N]);
        scalar::raster::sdf_coverage_u8(&outer, &mut want);
        raster::sdf_coverage_u8(&outer, &mut got);
        assert_eq!(got, want);
        assert!(want.iter().any(|&c| c == 255) && want.iter().any(|&c| c == 0));
        assert!(
            want.iter().any(|&c| c != 0 && c != 255),
            "no edge lanes covered"
        );
        scalar::raster::sdf_ring_coverage_u8(&outer, &inner, &mut want);
        raster::sdf_ring_coverage_u8(&outer, &inner, &mut got);
        assert_eq!(got, want);
    }

    #[test]
    fn dispatch_sdf_falloff_matches_scalar_for_every_table_length() {
        let d = shape_distances();
        for len in 1..=32usize {
            let table: alloc::vec::Vec<u8> = (0..len)
                .map(|i| (255 - (i * 255) / len.max(2)) as u8)
                .collect();
            let (mut want, mut got) = ([0u8; SHAPE_N], [0u8; SHAPE_N]);
            scalar::raster::sdf_falloff_u8(&d, &table, &mut want);
            raster::sdf_falloff_u8(&d, &table, &mut got);
            assert_eq!(got, want, "table length {len}");
        }
    }

    fn coverage_pattern() -> [u8; SHAPE_N] {
        core::array::from_fn(|i| match i % 5 {
            0 => 0,
            1 => 255,
            2 => 1,
            3 => 254,
            _ => (i * 37) as u8,
        })
    }

    fn dst_pattern() -> [u32; SHAPE_N] {
        core::array::from_fn(|i| 0xff00_0000 | (i as u32 * 0x0103_0507))
    }

    #[test]
    fn dispatch_blend_row_alpha_matches_scalar() {
        let cov = coverage_pattern();
        for pixel in [0xff20_c0ff_u32, 0x80_10_60_80, 0x01_00_00_01, 0x0000_0000] {
            let (mut want, mut got) = (dst_pattern(), dst_pattern());
            scalar::raster::blend_row_alpha(&mut want, pixel, &cov);
            raster::blend_row_alpha(&mut got, pixel, &cov);
            assert_eq!(got, want, "pixel {pixel:#010x}");
        }
    }

    #[test]
    fn dispatch_coverage_blends_match_scalar() {
        let cov8 = coverage_pattern();
        let cov16: [u16; SHAPE_N] = core::array::from_fn(|i| match i % 4 {
            0 => u16::from(cov8[i]) * 257,
            1 => 0xffff,
            2 => 128,
            _ => (i as u16).wrapping_mul(0x9e37),
        });
        let cov32: [u32; SHAPE_N] =
            core::array::from_fn(|i| (u32::from(cov8[i]) << 24) | 0x0055_aa11);
        for (color, opacity) in [
            (0x00ff_8040_u32, 255u8),
            (0xee10_2030, 128),
            (0x0000_00ff, 1),
        ] {
            let (mut want, mut got) = (dst_pattern(), dst_pattern());
            scalar::raster::coverage_blend_u8(&cov8, color, opacity, &mut want);
            raster::coverage_blend_u8(&cov8, color, opacity, &mut got);
            assert_eq!(got, want, "u8 {color:#x} @ {opacity}");
            let (mut want, mut got) = (dst_pattern(), dst_pattern());
            scalar::raster::coverage_blend_u16(&cov16, color, opacity, &mut want);
            raster::coverage_blend_u16(&cov16, color, opacity, &mut got);
            assert_eq!(got, want, "u16 {color:#x} @ {opacity}");
            let (mut want, mut got) = (dst_pattern(), dst_pattern());
            scalar::raster::coverage_blend_a32(&cov32, color, opacity, &mut want);
            raster::coverage_blend_a32(&cov32, color, opacity, &mut got);
            assert_eq!(got, want, "a32 {color:#x} @ {opacity}");
        }
    }

    #[test]
    fn dispatch_blend_row_masked_matches_scalar() {
        let mask = coverage_pattern();
        let src: [u32; SHAPE_N] = core::array::from_fn(|i| match i % 3 {
            0 => 0xff00_0000 | (i as u32 * 0x0307_0b0d),
            1 => 0x8040_2010,
            _ => 0,
        });
        let (mut want, mut got) = (dst_pattern(), dst_pattern());
        scalar::raster::blend_row_masked(&src, &mask, &mut want);
        raster::blend_row_masked(&src, &mask, &mut got);
        assert_eq!(got, want);
    }

    #[test]
    fn dispatch_uf16_depth_matches_ordered_scalar_reference() {
        const N: usize = 19; // multiple SIMD chunks plus a tail on every tier
        let coverage: [u32; N] = core::array::from_fn(|i| match i % 4 {
            0 => u32::MAX,
            1 => 1, // the public contract is nonzero, not only high-bit masks
            _ => 0,
        });
        let frag: [u16; N] = core::array::from_fn(|i| match i % 7 {
            0 => crate::cast::f32_to_uf16e5m11_rne(0.25),
            1 => crate::cast::f32_to_uf16e5m11_rne(0.5),
            2 => crate::cast::f32_to_uf16e5m11_rne(1.0),
            3 => crate::cast::f32_to_uf16e5m11_rne(2.0),
            4 => 0xF800, // +infinity
            5 => 0xFC00, // canonical NaN
            _ => 1,      // minimum subnormal
        });
        let mut zbuf: [u16; N] = core::array::from_fn(|i| match i % 5 {
            0 => crate::cast::f32_to_uf16e5m11_rne(1.0),
            1 => crate::cast::f32_to_uf16e5m11_rne(0.5),
            2 => 0xF800,
            3 => 0xFC01,
            _ => crate::cast::f32_to_uf16e5m11_rne(4.0),
        });
        let mut want_z = zbuf;
        let mut want_mask = [0u32; N];
        for i in 0..N {
            let f_nan = frag[i] & 0xF800 == 0xF800 && frag[i] & 0x07FF != 0;
            let z_nan = want_z[i] & 0xF800 == 0xF800 && want_z[i] & 0x07FF != 0;
            if coverage[i] != 0 && frag[i] <= want_z[i] && !f_nan && !z_nan {
                want_z[i] = frag[i];
                want_mask[i] = u32::MAX;
            }
        }
        let mut got_mask = [0u32; N];
        raster::depth_test_uf16e5m11(&coverage, &frag, &mut zbuf, &mut got_mask);
        assert_eq!(zbuf, want_z);
        assert_eq!(got_mask, want_mask);
    }

    // ── FP8: the *selected* tier vs the exact scalar reference ──
    // On x86 with `--features avx2` this exercises the F16C widen path —
    // the independent trick-vs-reference check from the fp8 experiment.
    #[test]
    fn dispatch_fp8_dequant_matches_reference() {
        let src: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut dst = [0.0f32; 256];

        f8::e4m3_to_f32(&src, &mut dst, 256);
        for b in 0..256 {
            let want = crate::cast::e4m3_to_f32(b as u8);
            if want.is_nan() {
                assert!(dst[b].is_nan(), "e4m3 0x{b:02X}");
            } else {
                assert_eq!(dst[b], want, "e4m3 0x{b:02X}");
            }
        }

        f8::e5m2_to_f32(&src, &mut dst, 256);
        for b in 0..256 {
            let want = crate::cast::e5m2_to_f32(b as u8);
            if want.is_nan() {
                assert!(dst[b].is_nan(), "e5m2 0x{b:02X}");
            } else {
                assert_eq!(dst[b], want, "e5m2 0x{b:02X}");
            }
        }
    }

    #[test]
    fn dispatch_uf8_dequant_matches_reference() {
        let src: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut dst = [0.0f32; 256];
        for (kernel, decode, name) in [
            (
                uf::uf8e4m4_to_f32 as fn(&[u8], &mut [f32], usize),
                crate::cast::uf8e4m4_to_f32 as fn(u8) -> f32,
                "E4M4",
            ),
            (
                uf::uf8e5m3_to_f32 as fn(&[u8], &mut [f32], usize),
                crate::cast::uf8e5m3_to_f32 as fn(u8) -> f32,
                "E5M3",
            ),
        ] {
            kernel(&src, &mut dst, src.len());
            for raw in 0..256 {
                let want = decode(raw as u8);
                assert_eq!(dst[raw].to_bits(), want.to_bits(), "UF8 {name} 0x{raw:02X}");
            }
        }
    }

    #[test]
    fn dispatch_uf16_promotion_matches_reference_exhaustively() {
        let mut src: alloc::vec::Vec<u16> = (0u32..=u16::MAX as u32).map(|v| v as u16).collect();
        // Force a tail on W=4 and W=8 selected tiers.
        src.extend_from_slice(&[0, 1, 0xF800]);
        let mut dst = alloc::vec![0.0f32; src.len()];
        uf::uf16e5m11_to_f32(&src, &mut dst, src.len());
        for (i, &raw) in src.iter().enumerate() {
            let want = crate::cast::uf16e5m11_to_f32(raw);
            assert_eq!(dst[i].to_bits(), want.to_bits(), "UF16 E5M11 0x{raw:04X}");
        }

        let packed: alloc::vec::Vec<u8> = src.iter().flat_map(|raw| raw.to_le_bytes()).collect();
        let mut packed_dst = alloc::vec![0.0f32; src.len()];
        uf::uf16e5m11_le_to_f32(&packed, &mut packed_dst, src.len());
        for i in 0..src.len() {
            assert_eq!(
                packed_dst[i].to_bits(),
                dst[i].to_bits(),
                "packed UF16 lane {i}"
            );
        }
    }

    #[test]
    fn dispatch_fp8_dot() {
        let w = [0x38u8, 0x40, 0xC0, 0x38, 0x40, 0xC0, 0x38, 0x40]; // 1,2,−2,…
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let got = f8::vec_dot_e4m3_f32(&w, &x, w.len());
        let mut want = 0.0f32;
        for i in 0..w.len() {
            want += crate::cast::e4m3_to_f32(w[i]) * x[i];
        }
        assert_eq!(got, want);
    }

    /// The dispatched row AXPY: every element as an exact per-element
    /// multiply-add would make it, the NaN encoding making its element NaN.
    #[test]
    fn dispatch_axpy_matches_the_exact_update() {
        use alloc::vec::Vec;
        for n in [16, 67, 256, 2688] {
            let mut row: Vec<u8> = (0..n).map(|i| ((i * 131) % 256) as u8).collect();
            for b in &mut row {
                if *b & 0x7F == 0x7F {
                    *b ^= 1;
                }
            }
            let base: Vec<f32> = (0..n).map(|i| (i % 13) as f32 - 6.0).collect();
            for (a, nan_at) in [(0.75f32, None), (-3.5, Some(n / 3))] {
                let mut row = row.clone();
                if let Some(at) = nan_at {
                    row[at] = 0xFF;
                }
                let mut acc = base.clone();
                gemv::axpy_e4m3_f32(&row, a, &mut acc);
                for (i, (&got, &b)) in acc.iter().zip(&row).enumerate() {
                    let want = a.mul_add(crate::cast::e4m3_to_f32(b), base[i]);
                    if Some(i) == nan_at {
                        assert!(got.is_nan(), "element {i} holds NaN");
                    } else {
                        assert!(
                            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                            "n {n} element {i}"
                        );
                    }
                }
            }
        }
    }

    /// The dispatched GEMV: every row within float association of the
    /// exact sum, a row holding the E4M3 NaN encoding NaN, and each row's
    /// bits the same whether it ran in a block or alone.
    #[test]
    fn dispatch_gemv_rows_are_exact_and_independent() {
        use alloc::vec::Vec;
        // Widths with and without a partial step; row counts with and
        // without a partial block.
        for (rows, k) in [(9, 67), (8, 64), (5, 2688), (3, 31)] {
            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 29 % 23) as f32 - 11.0) / 7.0)
                .collect();
            let mut e4m3: Vec<u8> = (0..rows * k).map(|i| ((i * 131) % 256) as u8).collect();
            for b in &mut e4m3 {
                if *b & 0x7F == 0x7F {
                    *b ^= 1;
                }
            }
            e4m3[2 * k + k / 2] = 0x7F; // row 2 is NaN
            let bf16: Vec<u8> = (0..rows * k)
                .flat_map(|i| {
                    let v = ((i * 37 % 41) as f32 - 20.0) / 9.0;
                    ((v.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect();
            let mut out = alloc::vec![0.0f32; rows];
            let mut alone = [0.0f32];

            gemv::e4m3_f32(&e4m3, &x, &mut out);
            for r in 0..rows {
                let row = &e4m3[r * k..(r + 1) * k];
                gemv::e4m3_f32(row, &x, &mut alone);
                assert_eq!(
                    out[r].to_bits(),
                    alone[0].to_bits(),
                    "e4m3 row {r} of {rows}x{k}"
                );
                let exact: f64 = row
                    .iter()
                    .zip(&x)
                    .map(|(&b, &v)| f64::from(crate::cast::e4m3_to_f32(b)) * f64::from(v))
                    .sum();
                if r == 2 {
                    assert!(out[r].is_nan(), "row {r} holds NaN");
                } else {
                    let bound = 1e-5 * row.len() as f64 * exact.abs().max(1.0);
                    assert!((f64::from(out[r]) - exact).abs() <= bound, "e4m3 row {r}");
                }
            }

            gemv::bf16_f32(&bf16, &x, &mut out);
            for r in 0..rows {
                let row = &bf16[2 * r * k..2 * (r + 1) * k];
                gemv::bf16_f32(row, &x, &mut alone);
                assert_eq!(
                    out[r].to_bits(),
                    alone[0].to_bits(),
                    "bf16 row {r} of {rows}x{k}"
                );
                let exact: f64 = row
                    .chunks_exact(2)
                    .zip(&x)
                    .map(|(w, &v)| {
                        f64::from(f32::from_bits(
                            u32::from(u16::from_le_bytes([w[0], w[1]])) << 16,
                        )) * f64::from(v)
                    })
                    .sum();
                let bound = 1e-5 * k as f64 * exact.abs().max(1.0);
                assert!((f64::from(out[r]) - exact).abs() <= bound, "bf16 row {r}");
            }
        }
    }
}
