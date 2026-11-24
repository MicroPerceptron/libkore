//! Base-NEON raster overrides.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::*;
#[cfg(target_os = "none")]
use core::arch::asm;

pub(crate) use crate::arch::portable::raster::{
    alpha_blend, and_masks, bary_interp, blend_row_alpha, blend_row_masked, blit_blend,
    blit_nearest, blit_opaque, cmpge_f32, coverage_blend_a32, coverage_blend_u8,
    coverage_blend_u16, depth_test, fill_blend_u32, fill_rect_u32, fused_edge_coverage,
    maskstore_u32, rounded_sdf_row_f32, sdf_coverage_u8, sdf_falloff_u8, sdf_ring_coverage_u8,
    xrgb_pack, xrgb_pack_rounded, xrgb_pack_u8_biased, xrgb_unpack,
};

// The bare-metal AArch64 target is compiled with `strict-align`.  LD1/ST1
// structure operations are still lawful at their arranged element alignment,
// which is exactly what these typed slices guarantee.  Ordinary targets keep
// the intrinsics so LLVM can combine adjacent accesses (notably on Apple CPUs).

#[cfg(target_os = "none")]
#[inline(always)]
unsafe fn load_u16x8(ptr: *const u16) -> uint16x8_t {
    let value;
    asm!(
        "ld1 {{{value:v}.8h}}, [{ptr}]",
        value = lateout(vreg) value,
        ptr = in(reg) ptr,
        options(readonly, nostack, preserves_flags),
    );
    value
}

#[cfg(not(target_os = "none"))]
#[inline(always)]
unsafe fn load_u16x8(ptr: *const u16) -> uint16x8_t {
    vld1q_u16(ptr)
}

#[cfg(target_os = "none")]
#[inline(always)]
unsafe fn load_u32x4(ptr: *const u32) -> uint32x4_t {
    let value;
    asm!(
        "ld1 {{{value:v}.4s}}, [{ptr}]",
        value = lateout(vreg) value,
        ptr = in(reg) ptr,
        options(readonly, nostack, preserves_flags),
    );
    value
}

#[cfg(not(target_os = "none"))]
#[inline(always)]
unsafe fn load_u32x4(ptr: *const u32) -> uint32x4_t {
    vld1q_u32(ptr)
}

#[cfg(target_os = "none")]
#[inline(always)]
unsafe fn store_u16x8(ptr: *mut u16, value: uint16x8_t) {
    asm!(
        "st1 {{{value:v}.8h}}, [{ptr}]",
        value = in(vreg) value,
        ptr = in(reg) ptr,
        options(nostack, preserves_flags),
    );
}

#[cfg(not(target_os = "none"))]
#[inline(always)]
unsafe fn store_u16x8(ptr: *mut u16, value: uint16x8_t) {
    vst1q_u16(ptr, value);
}

#[cfg(target_os = "none")]
#[inline(always)]
unsafe fn store_u32x4(ptr: *mut u32, value: uint32x4_t) {
    asm!(
        "st1 {{{value:v}.4s}}, [{ptr}]",
        value = in(vreg) value,
        ptr = in(reg) ptr,
        options(nostack, preserves_flags),
    );
}

#[cfg(not(target_os = "none"))]
#[inline(always)]
unsafe fn store_u32x4(ptr: *mut u32, value: uint32x4_t) {
    vst1q_u32(ptr, value);
}

/// Eight-pixel raw unsigned-E5M11 depth test. Native raw ordering is numeric
/// for the non-negative format; the raw `<= +infinity` bound excludes every
/// NaN encoding and preserves ordered-float semantics without decoding.
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
        let zero32 = vdupq_n_u32(0);
        let infinity = vdupq_n_u16(0xF800);
        for c in 0..chunks {
            let off = c * 8;
            let cov0 = vmvnq_u32(vceqq_u32(load_u32x4(coverage.as_ptr().add(off)), zero32));
            let cov1 = vmvnq_u32(vceqq_u32(
                load_u32x4(coverage.as_ptr().add(off + 4)),
                zero32,
            ));
            let covered = vcombine_u16(vmovn_u32(cov0), vmovn_u32(cov1));

            let fragment = load_u16x8(frag_z.as_ptr().add(off));
            let old = load_u16x8(zbuf.as_ptr().add(off));
            let pass = vandq_u16(
                covered,
                vandq_u16(
                    vcleq_u16(fragment, old),
                    vandq_u16(vcleq_u16(fragment, infinity), vcleq_u16(old, infinity)),
                ),
            );
            store_u16x8(zbuf.as_mut_ptr().add(off), vbslq_u16(pass, fragment, old));

            let signed = vreinterpretq_s16_u16(pass);
            store_u32x4(
                out_mask.as_mut_ptr().add(off),
                vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(signed))),
            );
            store_u32x4(
                out_mask.as_mut_ptr().add(off + 4),
                vreinterpretq_u32_s32(vmovl_high_s16(signed)),
            );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(16))]
    struct Aligned<T>(T);

    #[test]
    fn uf16_depth_accepts_element_aligned_odd_offset_slices() {
        const N: usize = 19;
        let mut coverage = Aligned([0u32; N + 1]);
        let mut fragment = Aligned([0u16; N + 1]);
        let mut zbuf = Aligned([0u16; N + 1]);
        let mut out = Aligned([0u32; N + 1]);

        for i in 0..N {
            coverage.0[i + 1] = match i % 4 {
                0 => u32::MAX,
                1 => 1,
                _ => 0,
            };
            fragment.0[i + 1] = match i % 7 {
                0 => crate::cast::f32_to_uf16e5m11_rne(0.25),
                1 => crate::cast::f32_to_uf16e5m11_rne(0.5),
                2 => crate::cast::f32_to_uf16e5m11_rne(1.0),
                3 => crate::cast::f32_to_uf16e5m11_rne(2.0),
                4 => 0xF800,
                5 => 0xFC00,
                _ => 1,
            };
            zbuf.0[i + 1] = match i % 5 {
                0 => crate::cast::f32_to_uf16e5m11_rne(1.0),
                1 => crate::cast::f32_to_uf16e5m11_rne(0.5),
                2 => 0xF800,
                3 => 0xFC01,
                _ => crate::cast::f32_to_uf16e5m11_rne(4.0),
            };
        }

        let coverage = &coverage.0[1..];
        let fragment = &fragment.0[1..];
        let zbuf = &mut zbuf.0[1..];
        let out = &mut out.0[1..];
        assert_ne!((coverage.as_ptr() as usize) & 15, 0);
        assert_ne!((fragment.as_ptr() as usize) & 15, 0);
        assert_ne!((zbuf.as_ptr() as usize) & 15, 0);
        assert_ne!((out.as_ptr() as usize) & 15, 0);

        let mut want_z = [0u16; N];
        want_z.copy_from_slice(zbuf);
        let mut want_out = [0u32; N];
        for i in 0..N {
            let f = fragment[i];
            let z = want_z[i];
            if coverage[i] != 0 && f <= z && f <= 0xF800 && z <= 0xF800 {
                want_z[i] = f;
                want_out[i] = u32::MAX;
            }
        }

        depth_test_uf16e5m11(coverage, fragment, zbuf, out);
        assert_eq!(zbuf, want_z);
        assert_eq!(out, want_out);
    }
}
