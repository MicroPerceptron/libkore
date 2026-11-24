//! Base-NEON unsigned-float storage kernels.
//!
//! `core::simd` is the semantic source of truth, but current AArch64 codegen
//! scalarizes its slice assembly for these mixed lane-width conversions.
//! These overrides produce vector-native loads/stores on ordinary AArch64
//! targets and use element-aligned LD1/ST1 shims on strict-alignment bare
//! metal; no FEAT_FHM requirement is involved.

#![allow(unsafe_op_in_unsafe_fn)]

pub(crate) use crate::arch::portable::f8::finite_uf8e5m3_rgb;
use core::arch::aarch64::*;
#[cfg(target_os = "none")]
use core::arch::asm;

// `aarch64-unknown-none` enables LLVM's `strict-align` feature.  The standard
// load/store intrinsics become ordinary vector IR loads/stores, so LLVM must
// scalarize them when a slice only guarantees element alignment.  AArch64's
// single-structure LD1/ST1 forms perform one memory access per arranged
// element: `.16b` therefore needs byte alignment, `.8h` halfword alignment,
// and `.4s` word alignment.  Keep the workaround confined to bare metal so
// ordinary targets retain LLVM's profitable LDR/STR and load/store pairing.

#[cfg(target_os = "none")]
#[inline(always)]
unsafe fn load_u8x16(ptr: *const u8) -> uint8x16_t {
    let value;
    asm!(
        "ld1 {{{value:v}.16b}}, [{ptr}]",
        value = lateout(vreg) value,
        ptr = in(reg) ptr,
        options(readonly, nostack, preserves_flags),
    );
    value
}

#[cfg(not(target_os = "none"))]
#[inline(always)]
unsafe fn load_u8x16(ptr: *const u8) -> uint8x16_t {
    vld1q_u8(ptr)
}

#[cfg(target_os = "none")]
#[inline(always)]
unsafe fn load_u8x8(ptr: *const u8) -> uint8x8_t {
    let value;
    asm!(
        "ld1 {{{value:v}.8b}}, [{ptr}]",
        value = lateout(vreg) value,
        ptr = in(reg) ptr,
        options(readonly, nostack, preserves_flags),
    );
    value
}

#[cfg(not(target_os = "none"))]
#[inline(always)]
unsafe fn load_u8x8(ptr: *const u8) -> uint8x8_t {
    vld1_u8(ptr)
}

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
unsafe fn store_u8x16(ptr: *mut u8, value: uint8x16_t) {
    asm!(
        "st1 {{{value:v}.16b}}, [{ptr}]",
        value = in(vreg) value,
        ptr = in(reg) ptr,
        options(nostack, preserves_flags),
    );
}

#[cfg(not(target_os = "none"))]
#[inline(always)]
unsafe fn store_u8x16(ptr: *mut u8, value: uint8x16_t) {
    vst1q_u8(ptr, value);
}

#[cfg(target_os = "none")]
#[inline(always)]
unsafe fn store_f32x4(ptr: *mut f32, value: float32x4_t) {
    asm!(
        "st1 {{{value:v}.4s}}, [{ptr}]",
        value = in(vreg) value,
        ptr = in(reg) ptr,
        options(nostack, preserves_flags),
    );
}

#[cfg(not(target_os = "none"))]
#[inline(always)]
unsafe fn store_f32x4(ptr: *mut f32, value: float32x4_t) {
    vst1q_f32(ptr, value);
}

#[inline(always)]
unsafe fn patch_e5_special(raw: uint32x4_t, converted: float32x4_t) -> float32x4_t {
    let exp = vandq_u32(raw, vdupq_n_u32(0xF8));
    let special = vceqq_u32(exp, vdupq_n_u32(0xF8));
    let mant = vandq_u32(raw, vdupq_n_u32(0x07));
    let bits = vorrq_u32(vdupq_n_u32(0x7F80_0000), vshlq_n_u32::<20>(mant));
    vbslq_f32(special, vreinterpretq_f32_u32(bits), converted)
}

#[inline(always)]
unsafe fn decode_e5m3(bytes: uint8x8_t) -> (float32x4_t, float32x4_t) {
    let raw = vmovl_u8(bytes);
    let exp = vandq_u16(raw, vdupq_n_u16(0xF8));
    let special = vceqq_u16(exp, vdupq_n_u16(0xF8));
    // Keep exponent-all-ones encodings out of FCVTL so FPCR.AHP and sNaN
    // exception behavior cannot alter the UF contract. FCVTL widens half
    // subnormals exactly and does not apply half-precision flush controls.
    let safe_bits = vandq_u16(vshll_n_u8::<7>(bytes), vmvnq_u16(special));
    let half = vreinterpretq_f16_u16(safe_bits);
    let lo = vcvt_f32_f16(vget_low_f16(half));
    let hi = vcvt_high_f32_f16(half);
    (
        patch_e5_special(vmovl_u16(vget_low_u16(raw)), lo),
        patch_e5_special(vmovl_high_u16(raw), hi),
    )
}

#[inline(always)]
unsafe fn patch_e4_special(raw: uint32x4_t, converted: float32x4_t) -> float32x4_t {
    let exp = vandq_u32(raw, vdupq_n_u32(0xF0));
    let special = vceqq_u32(exp, vdupq_n_u32(0xF0));
    let mant = vandq_u32(raw, vdupq_n_u32(0x0F));
    let bits = vorrq_u32(vdupq_n_u32(0x7F80_0000), vshlq_n_u32::<19>(mant));
    vbslq_f32(special, vreinterpretq_f32_u32(bits), converted)
}

#[inline(always)]
unsafe fn decode_e4m4(bytes: uint8x8_t) -> (float32x4_t, float32x4_t) {
    let raw8 = vmovl_u8(bytes);
    let exp = vandq_u16(raw8, vdupq_n_u16(0xF0));
    let special = vceqq_u16(exp, vdupq_n_u16(0xF0));
    let half_bits = vandq_u16(vshll_n_u8::<6>(bytes), vmvnq_u16(special));
    let half = vreinterpretq_f16_u16(half_bits);
    let lo = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(half)), 256.0);
    let hi = vmulq_n_f32(vcvt_high_f32_f16(half), 256.0);
    (
        patch_e4_special(vmovl_u16(vget_low_u16(raw8)), lo),
        patch_e4_special(vmovl_high_u16(raw8), hi),
    )
}

macro_rules! dequant_uf8 {
    ($name:ident, $decode:ident, $scalar:path) => {
        pub(crate) fn $name(src: &[u8], dst: &mut [f32], n: usize) {
            debug_assert!(src.len() >= n && dst.len() >= n);
            unsafe {
                let chunks16 = n / 16;
                for c in 0..chunks16 {
                    let off = c * 16;
                    let bytes = load_u8x16(src.as_ptr().add(off));
                    let (a0, a1) = $decode(vget_low_u8(bytes));
                    let (b0, b1) = $decode(vget_high_u8(bytes));
                    store_f32x4(dst.as_mut_ptr().add(off), a0);
                    store_f32x4(dst.as_mut_ptr().add(off + 4), a1);
                    store_f32x4(dst.as_mut_ptr().add(off + 8), b0);
                    store_f32x4(dst.as_mut_ptr().add(off + 12), b1);
                }
                let mut off = chunks16 * 16;
                if off + 8 <= n {
                    let (lo, hi) = $decode(load_u8x8(src.as_ptr().add(off)));
                    store_f32x4(dst.as_mut_ptr().add(off), lo);
                    store_f32x4(dst.as_mut_ptr().add(off + 4), hi);
                    off += 8;
                }
                for i in off..n {
                    *dst.get_unchecked_mut(i) = $scalar(*src.get_unchecked(i));
                }
            }
        }
    };
}

dequant_uf8!(uf8e4m4_to_f32, decode_e4m4, crate::cast::uf8e4m4_to_f32);
dequant_uf8!(uf8e5m3_to_f32, decode_e5m3, crate::cast::uf8e5m3_to_f32);

#[inline(always)]
unsafe fn store_le_u16(dst: *mut u8, values: uint16x8_t) {
    #[cfg(target_endian = "little")]
    store_u8x16(dst, vreinterpretq_u8_u16(values));
    #[cfg(target_endian = "big")]
    store_u8x16(dst, vrev16q_u8(vreinterpretq_u8_u16(values)));
}

#[inline(always)]
unsafe fn load_le_u16(src: *const u8) -> uint16x8_t {
    #[cfg(target_endian = "little")]
    {
        vreinterpretq_u16_u8(load_u8x16(src))
    }
    #[cfg(target_endian = "big")]
    {
        vreinterpretq_u16_u8(vrev16q_u8(load_u8x16(src)))
    }
}

#[inline(always)]
unsafe fn pack_e4m4_f16(bytes: uint8x8_t) -> uint16x8_t {
    let raw = vmovl_u8(bytes);
    let exp = vandq_u16(raw, vdupq_n_u16(0xF0));
    let mant = vandq_u16(raw, vdupq_n_u16(0x0F));
    let base = vshlq_n_u16::<6>(raw);
    let normal = vaddq_u16(base, vdupq_n_u16(0x2000));
    let special = vaddq_u16(base, vdupq_n_u16(0x4000));

    let sub1 = vdupq_n_u16(0x1400);
    let sub2 = vorrq_u16(
        vdupq_n_u16(0x1800),
        vshlq_n_u16::<9>(vandq_u16(mant, vdupq_n_u16(1))),
    );
    let sub4 = vorrq_u16(
        vdupq_n_u16(0x1C00),
        vshlq_n_u16::<8>(vandq_u16(mant, vdupq_n_u16(3))),
    );
    let sub8 = vorrq_u16(
        vdupq_n_u16(0x2000),
        vshlq_n_u16::<7>(vandq_u16(mant, vdupq_n_u16(7))),
    );
    let sub = vbslq_u16(
        vcgeq_u16(mant, vdupq_n_u16(8)),
        sub8,
        vbslq_u16(
            vcgeq_u16(mant, vdupq_n_u16(4)),
            sub4,
            vbslq_u16(
                vcgeq_u16(mant, vdupq_n_u16(2)),
                sub2,
                vbslq_u16(vcgeq_u16(mant, vdupq_n_u16(1)), sub1, vdupq_n_u16(0)),
            ),
        ),
    );
    vbslq_u16(
        vceqq_u16(exp, vdupq_n_u16(0)),
        sub,
        vbslq_u16(vceqq_u16(exp, vdupq_n_u16(0xF0)), special, normal),
    )
}

pub(crate) fn uf8e4m4_to_f16(src: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n * 2);
    unsafe {
        let chunks16 = n / 16;
        for c in 0..chunks16 {
            let off = c * 16;
            let bytes = load_u8x16(src.as_ptr().add(off));
            store_le_u16(
                dst.as_mut_ptr().add(off * 2),
                pack_e4m4_f16(vget_low_u8(bytes)),
            );
            store_le_u16(
                dst.as_mut_ptr().add((off + 8) * 2),
                pack_e4m4_f16(vget_high_u8(bytes)),
            );
        }
        let mut off = chunks16 * 16;
        if off + 8 <= n {
            store_le_u16(
                dst.as_mut_ptr().add(off * 2),
                pack_e4m4_f16(load_u8x8(src.as_ptr().add(off))),
            );
            off += 8;
        }
        for i in off..n {
            let bytes = crate::cast::uf8e4m4_to_f16_bits(*src.get_unchecked(i)).to_le_bytes();
            *dst.get_unchecked_mut(i * 2) = bytes[0];
            *dst.get_unchecked_mut(i * 2 + 1) = bytes[1];
        }
    }
}

pub(crate) fn uf8e5m3_to_f16(src: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n * 2);
    unsafe {
        let chunks16 = n / 16;
        for c in 0..chunks16 {
            let off = c * 16;
            let bytes = load_u8x16(src.as_ptr().add(off));
            store_le_u16(
                dst.as_mut_ptr().add(off * 2),
                vshll_n_u8::<7>(vget_low_u8(bytes)),
            );
            store_le_u16(
                dst.as_mut_ptr().add((off + 8) * 2),
                vshll_high_n_u8::<7>(bytes),
            );
        }
        let mut off = chunks16 * 16;
        if off + 8 <= n {
            store_le_u16(
                dst.as_mut_ptr().add(off * 2),
                vshll_n_u8::<7>(load_u8x8(src.as_ptr().add(off))),
            );
            off += 8;
        }
        for i in off..n {
            let bytes = ((*src.get_unchecked(i) as u16) << 7).to_le_bytes();
            *dst.get_unchecked_mut(i * 2) = bytes[0];
            *dst.get_unchecked_mut(i * 2 + 1) = bytes[1];
        }
    }
}

#[inline(always)]
unsafe fn decode_e5m11_x4(raw: uint32x4_t) -> float32x4_t {
    let exp = vandq_u32(vshrq_n_u32::<11>(raw), vdupq_n_u32(0x1F));
    let mant = vandq_u32(raw, vdupq_n_u32(0x07FF));
    let normal_bits = vorrq_u32(
        vshlq_n_u32::<23>(vaddq_u32(exp, vdupq_n_u32(127 - 15))),
        vshlq_n_u32::<12>(mant),
    );
    let normal = vreinterpretq_f32_u32(normal_bits);
    let subnormal = vmulq_n_f32(vcvtq_f32_u32(mant), 0.000_000_029_802_322);
    let special =
        vreinterpretq_f32_u32(vorrq_u32(vdupq_n_u32(0x7F80_0000), vshlq_n_u32::<12>(mant)));
    let finite = vbslq_f32(vceqq_u32(exp, vdupq_n_u32(0)), subnormal, normal);
    vbslq_f32(vceqq_u32(exp, vdupq_n_u32(0x1F)), special, finite)
}

#[inline(always)]
unsafe fn store_e5m11_x8(raw: uint16x8_t, dst: *mut f32) {
    store_f32x4(dst, decode_e5m11_x4(vmovl_u16(vget_low_u16(raw))));
    store_f32x4(dst.add(4), decode_e5m11_x4(vmovl_high_u16(raw)));
}

pub(crate) fn uf16e5m11_to_f32(src: &[u16], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n);
    unsafe {
        let chunks = n / 8;
        for c in 0..chunks {
            let off = c * 8;
            store_e5m11_x8(load_u16x8(src.as_ptr().add(off)), dst.as_mut_ptr().add(off));
        }
        for i in chunks * 8..n {
            *dst.get_unchecked_mut(i) = crate::cast::uf16e5m11_to_f32(*src.get_unchecked(i));
        }
    }
}

pub(crate) fn uf16e5m11_le_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n * 2 && dst.len() >= n);
    unsafe {
        let chunks = n / 8;
        for c in 0..chunks {
            let off = c * 8;
            store_e5m11_x8(
                load_le_u16(src.as_ptr().add(off * 2)),
                dst.as_mut_ptr().add(off),
            );
        }
        for i in chunks * 8..n {
            let off = i * 2;
            let raw = u16::from_le_bytes([*src.get_unchecked(off), *src.get_unchecked(off + 1)]);
            *dst.get_unchecked_mut(i) = crate::cast::uf16e5m11_to_f32(raw);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(16))]
    struct Aligned<T>(T);

    fn assert_same_float(got: f32, want: f32, context: &str) {
        assert_eq!(got.to_bits(), want.to_bits(), "{context}");
    }

    #[test]
    fn odd_offset_slices_preserve_all_uf_memory_contracts() {
        const N: usize = 33;

        let mut src8 = Aligned([0u8; N + 1]);
        for i in 0..N {
            src8.0[i + 1] = (i * 37 + 11) as u8;
        }
        let src8 = &src8.0[1..];
        assert_ne!((src8.as_ptr() as usize) & 15, 0);

        for (kernel, decode, name) in [
            (
                uf8e4m4_to_f32 as fn(&[u8], &mut [f32], usize),
                crate::cast::uf8e4m4_to_f32 as fn(u8) -> f32,
                "UF8 E4M4",
            ),
            (
                uf8e5m3_to_f32 as fn(&[u8], &mut [f32], usize),
                crate::cast::uf8e5m3_to_f32 as fn(u8) -> f32,
                "UF8 E5M3",
            ),
        ] {
            let mut dst = Aligned([0.0f32; N + 1]);
            let dst = &mut dst.0[1..];
            assert_ne!((dst.as_ptr() as usize) & 15, 0);
            kernel(src8, dst, N);
            for i in 0..N {
                assert_same_float(dst[i], decode(src8[i]), name);
            }
        }

        for (kernel, encode, name) in [
            (
                uf8e4m4_to_f16 as fn(&[u8], &mut [u8], usize),
                crate::cast::uf8e4m4_to_f16_bits as fn(u8) -> u16,
                "UF8 E4M4 to F16",
            ),
            (
                uf8e5m3_to_f16 as fn(&[u8], &mut [u8], usize),
                (|raw| (raw as u16) << 7) as fn(u8) -> u16,
                "UF8 E5M3 to F16",
            ),
        ] {
            let mut dst = Aligned([0xA5u8; N * 2 + 2]);
            let dst = &mut dst.0[1..];
            assert_ne!((dst.as_ptr() as usize) & 15, 0);
            kernel(src8, dst, N);
            for i in 0..N {
                assert_eq!(
                    u16::from_le_bytes([dst[i * 2], dst[i * 2 + 1]]),
                    encode(src8[i]),
                    "{name}"
                );
            }
            assert_eq!(dst[N * 2], 0xA5, "{name} wrote past n");
        }

        let mut src16 = Aligned([0u16; N + 1]);
        for i in 0..N {
            src16.0[i + 1] = (i * 8_191 + 17) as u16;
        }
        let src16 = &src16.0[1..];
        assert_ne!((src16.as_ptr() as usize) & 15, 0);

        let mut native_dst = Aligned([0.0f32; N + 1]);
        let native_dst = &mut native_dst.0[1..];
        assert_ne!((native_dst.as_ptr() as usize) & 15, 0);
        uf16e5m11_to_f32(src16, native_dst, N);
        for i in 0..N {
            assert_same_float(
                native_dst[i],
                crate::cast::uf16e5m11_to_f32(src16[i]),
                "native UF16 E5M11",
            );
        }

        let mut packed = Aligned([0u8; N * 2 + 1]);
        for i in 0..N {
            let raw = src16[i].to_le_bytes();
            packed.0[i * 2 + 1] = raw[0];
            packed.0[i * 2 + 2] = raw[1];
        }
        let packed = &packed.0[1..];
        assert_ne!((packed.as_ptr() as usize) & 15, 0);

        let mut packed_dst = Aligned([0.0f32; N + 1]);
        let packed_dst = &mut packed_dst.0[1..];
        assert_ne!((packed_dst.as_ptr() as usize) & 15, 0);
        uf16e5m11_le_to_f32(packed, packed_dst, N);
        for i in 0..N {
            assert_same_float(packed_dst[i], native_dst[i], "packed UF16 E5M11");
        }
    }
}
