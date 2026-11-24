//! AVX2 raster primitives — coverage ops plus storage-specialized depth.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

// Coverage ops: AVX2-tuned from vec.rs
pub(crate) use super::vec::{and_masks, cmpge_f32, fused_edge_coverage, maskstore_u32};

// Fragment, packing, and blit ops: portable implementations
pub(crate) use crate::arch::portable::raster::{
    alpha_blend, bary_interp, blend_row_alpha, blend_row_masked, blit_blend, blit_nearest,
    blit_opaque, coverage_blend_a32, coverage_blend_u8, coverage_blend_u16, depth_test,
    fill_blend_u32, fill_rect_u32, rounded_sdf_row_f32, sdf_coverage_u8, sdf_falloff_u8,
    sdf_ring_coverage_u8, xrgb_pack, xrgb_pack_rounded, xrgb_pack_u8_biased, xrgb_unpack,
};

/// AVX2 unsigned-E5M11 depth test. Eight pixels consume one 256-bit
/// coverage vector but only 128 bits each of fragment and z-buffer traffic.
/// Raw UF ordering is numeric for all non-negative values; NaNs are masked to
/// preserve ordered-float comparison semantics.
pub(crate) fn depth_test_uf16e5m11(
    coverage: &[u32],
    frag_z: &[u16],
    zbuf: &mut [u16],
    out_mask: &mut [u32],
) {
    let n = coverage.len();
    debug_assert_eq!(frag_z.len(), n);
    debug_assert_eq!(zbuf.len(), n);
    debug_assert_eq!(out_mask.len(), n);
    unsafe {
        let chunks = n / 8;
        let zero = _mm256_setzero_si256();
        let infinity = _mm256_set1_epi32(0xF800);
        for c in 0..chunks {
            let off = c * 8;
            let cov = _mm256_loadu_si256(coverage.as_ptr().add(off) as *const __m256i);
            let covered = _mm256_xor_si256(_mm256_cmpeq_epi32(cov, zero), _mm256_set1_epi32(-1));

            let fragment16 = _mm_loadu_si128(frag_z.as_ptr().add(off) as *const __m128i);
            let old16 = _mm_loadu_si128(zbuf.as_ptr().add(off) as *const __m128i);
            let fragment = _mm256_cvtepu16_epi32(fragment16);
            let old = _mm256_cvtepu16_epi32(old16);

            let depth_pass = _mm256_cmpeq_epi32(_mm256_max_epu32(fragment, old), old);
            // In this unsigned format every NaN encoding is greater than
            // +infinity, so raw max comparisons implement ordered semantics.
            let fragment_ordered =
                _mm256_cmpeq_epi32(_mm256_max_epu32(fragment, infinity), infinity);
            let old_ordered = _mm256_cmpeq_epi32(_mm256_max_epu32(old, infinity), infinity);
            let pass = _mm256_and_si256(
                covered,
                _mm256_and_si256(depth_pass, _mm256_and_si256(fragment_ordered, old_ordered)),
            );

            let selected = _mm256_blendv_epi8(old, fragment, pass);
            let packed = _mm_packus_epi32(
                _mm256_castsi256_si128(selected),
                _mm256_extracti128_si256(selected, 1),
            );
            _mm_storeu_si128(zbuf.as_mut_ptr().add(off) as *mut __m128i, packed);
            _mm256_storeu_si256(out_mask.as_mut_ptr().add(off) as *mut __m256i, pass);
        }

        for i in chunks * 8..n {
            let f = *frag_z.get_unchecked(i);
            let z = *zbuf.get_unchecked(i);
            if *coverage.get_unchecked(i) != 0 && f <= z && f <= 0xF800 && z <= 0xF800 {
                *zbuf.get_unchecked_mut(i) = f;
                *out_mask.get_unchecked_mut(i) = u32::MAX;
            } else {
                *out_mask.get_unchecked_mut(i) = 0;
            }
        }
    }
}
