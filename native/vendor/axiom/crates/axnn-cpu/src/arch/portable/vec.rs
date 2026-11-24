//! Element-wise vector operations — portable `core::simd` tier.
//!
//! F32: 4 elements per 128-bit register (NEON / SSE2).
//! F64: 2 elements per 128-bit register.
//! F16/BF16: widen to f32, compute, narrow back via software conversion.
//!
//! Transcendental ops (exp, tanh, sigmoid, silu, gelu) use a fast
//! polynomial exp approximation (~1 ULP via Cephes decomposition).
//!
//! Raster primitives (cmpge_f32, and_masks, maskstore_u32): 4-wide
//! coverage test pipeline for Pineda edge-function rasterization.

use core::simd::num::{SimdFloat, SimdInt};
use core::simd::prelude::*;
use core::simd::{Select, Swizzle};

use super::{W, WD};

pub(crate) fn sub_mul_add_f32(inputs: [&[f32]; 4], output: &mut [f32]) {
    let mask = inputs
        .iter()
        .enumerate()
        .fold(0, |mask, (c, p)| mask | (usize::from(p.len() == 1) << c));
    macro_rules! select { ($($mask:literal),*) => { match mask { $($mask => sub_mul_add_pattern::<$mask>(inputs, output),)* _ => unreachable!() } }; }
    select!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

fn sub_mul_add_pattern<const MASK: usize>(inputs: [&[f32]; 4], output: &mut [f32]) {
    #[inline(always)]
    fn load<const MASK: usize, const C: usize>(inputs: [&[f32]; 4], i: usize) -> Simd<f32, W> {
        if MASK & (1 << C) != 0 {
            Simd::splat(inputs[C][0])
        } else {
            Simd::from_slice(&inputs[C][i..i + W])
        }
    }
    let vector_end = output.len() / W * W;
    for i in (0..vector_end).step_by(W) {
        let a = load::<MASK, 0>(inputs, i);
        let b = load::<MASK, 1>(inputs, i);
        let scale = load::<MASK, 2>(inputs, i);
        let addend = load::<MASK, 3>(inputs, i);
        let difference = a - b;
        let product = difference * scale;
        (product + addend).copy_to_slice(&mut output[i..i + W]);
    }
    for (i, out) in output.iter_mut().enumerate().skip(vector_end) {
        let [a, b, scale, addend] =
            core::array::from_fn(|c| inputs[c][if MASK & (1 << c) != 0 { 0 } else { i }]);
        *out = (a - b) * scale + addend;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Fast vectorised exp(x) — Cephes-style range reduction (f32)
// ═══════════════════════════════════════════════════════════════════

/// Lane-count-generic fast exp — shared with the splat compositor,
/// which runs it at pixel-strip width rather than `W`.
#[inline(always)]
pub(crate) fn exp_f32_n<const N: usize>(x: Simd<f32, N>) -> Simd<f32, N> {
    let original = x;
    let ln2 = Simd::splat(core::f32::consts::LN_2);
    let ln2_inv = Simd::splat(1.0 / core::f32::consts::LN_2);
    let one = Simd::splat(1.0f32);
    let c2 = Simd::splat(0.5f32);
    let c3 = Simd::splat(1.0 / 6.0);
    let c4 = Simd::splat(1.0 / 24.0);
    let c5 = Simd::splat(1.0 / 120.0);
    let c6 = Simd::splat(1.0 / 720.0);

    // The exponent-bit construction below is only valid for normal finite
    // powers of two. In particular n=-128 would construct NEGATIVE infinity,
    // not an underflowed exponential. Keep its working range normal and
    // repair subnormal/overflow/non-finite lanes with the scalar contract.
    let x = x
        .simd_min(Simd::splat(88.0f32))
        .simd_max(Simd::splat(-87.0f32));

    // n = round(x / ln2) via magic-number trick (no_std-safe)
    let magic_f32 = Simd::splat(12582912.0f32); // 2^23 + 2^22
    let n = (x * ln2_inv + magic_f32) - magic_f32;

    // r = x - n*ln2
    let r = x - n * ln2;

    // Horner polynomial: 1 + r(1 + r(c2 + r(c3 + r(c4 + r(c5 + r*c6)))))
    let mut p = r * c6 + c5;
    p = r * p + c4;
    p = r * p + c3;
    p = r * p + c2;
    p = r * p + one;
    p = r * p + one;

    // 2^n via integer exponent manipulation
    let ni: Simd<i32, N> = n.cast();
    let pow2n =
        Simd::<f32, N>::from_bits((ni + Simd::splat(127i32)).cast::<u32>() << Simd::splat(23u32));

    let result = p * pow2n;
    let exceptional = original.simd_lt(Simd::splat(-87.0))
        | original.simd_gt(Simd::splat(88.0))
        | original.is_nan();
    if exceptional.any() {
        let mut lanes = result.to_array();
        let inputs = original.to_array();
        let repair = exceptional.to_array();
        for lane in 0..N {
            if repair[lane] {
                lanes[lane] = axmath::exp_f32(inputs[lane]);
            }
        }
        Simd::from_array(lanes)
    } else {
        result
    }
}

#[inline(always)]
fn exp_f32x(x: Simd<f32, W>) -> Simd<f32, W> {
    exp_f32_n::<W>(x)
}

// ═══════════════════════════════════════════════════════════════════
//  Vector square root — hardware operation with a no_std scalar fallback
// ═══════════════════════════════════════════════════════════════════
//
// core::simd does not expose sqrt without std::simd::StdFloat in this
// toolchain. Use the hardware intrinsics where the target guarantees them;
// unlike a generic LLVM sqrt intrinsic, the fallback below cannot introduce
// an external libm dependency on targets without hardware square root.
// Two inverse-sqrt refinements are not a full-precision square root, and
// multiplying x by an rsqrt estimate also mishandles infinities/subnormals.

#[inline(always)]
fn sqrt_f32x(x: Simd<f32, W>) -> Simd<f32, W> {
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        // SAFETY: NEON is part of this target contract, and W=4 on AArch64.
        unsafe { core::arch::aarch64::vsqrtq_f32(x.into()).into() }
    }
    #[cfg(all(
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    {
        // SAFETY: AVX2/AVX-512 imply AVX and select W=8 in portable/mod.rs.
        unsafe { core::arch::x86_64::_mm256_sqrt_ps(x.into()).into() }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(any(target_feature = "avx2", target_feature = "avx512f"))
    ))]
    {
        // SAFETY: SSE2 is in the target contract and the portable width is 4.
        unsafe { core::arch::x86_64::_mm_sqrt_ps(x.into()).into() }
    }
    #[cfg(not(any(
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little"
        ),
        all(
            target_arch = "x86_64",
            any(
                target_feature = "sse2",
                target_feature = "avx2",
                target_feature = "avx512f"
            )
        ),
    )))]
    Simd::from_array(x.to_array().map(axmath::sqrt_f32))
}

#[inline(always)]
fn sqrt_f64x(x: Simd<f64, WD>) -> Simd<f64, WD> {
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        // SAFETY: NEON is part of this target contract, and WD=2 on AArch64.
        unsafe { core::arch::aarch64::vsqrtq_f64(x.into()).into() }
    }
    #[cfg(all(
        target_arch = "x86_64",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    {
        // SAFETY: AVX2/AVX-512 imply AVX and select WD=4 in portable/mod.rs.
        unsafe { core::arch::x86_64::_mm256_sqrt_pd(x.into()).into() }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(any(target_feature = "avx2", target_feature = "avx512f"))
    ))]
    {
        // SAFETY: SSE2 is in the target contract and the portable width is 2.
        unsafe { core::arch::x86_64::_mm_sqrt_pd(x.into()).into() }
    }
    #[cfg(not(any(
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little"
        ),
        all(
            target_arch = "x86_64",
            any(
                target_feature = "sse2",
                target_feature = "avx2",
                target_feature = "avx512f"
            )
        ),
    )))]
    Simd::from_array(x.to_array().map(axmath::sqrt_f64))
}

/// Fast vectorised tanh(x) = (e^2x - 1) / (e^2x + 1)
#[inline(always)]
fn tanh_f32x(x: Simd<f32, W>) -> Simd<f32, W> {
    let two = Simd::splat(2.0f32);
    let one = Simd::splat(1.0f32);
    let e2x = exp_f32x(x * two);
    (e2x - one) / (e2x + one)
}

// ═══════════════════════════════════════════════════════════════════
//  Fast vectorised exp(x) — Cephes-style range reduction (f64)
// ═══════════════════════════════════════════════════════════════════

#[inline(always)]
fn exp_f64x(x: Simd<f64, WD>) -> Simd<f64, WD> {
    let ln2 = Simd::splat(core::f64::consts::LN_2);
    let ln2_inv = Simd::splat(1.0 / core::f64::consts::LN_2);
    let one = Simd::splat(1.0f64);
    let c2 = Simd::splat(1.0 / 2.0);
    let c3 = Simd::splat(1.0 / 6.0);
    let c4 = Simd::splat(1.0 / 24.0);
    let c5 = Simd::splat(1.0 / 120.0);
    let c6 = Simd::splat(1.0 / 720.0);
    let c7 = Simd::splat(1.0 / 5040.0);
    let c8 = Simd::splat(1.0 / 40320.0);

    let x = x
        .simd_min(Simd::splat(709.7f64))
        .simd_max(Simd::splat(-708.3f64));

    // n = round(x / ln2) via magic-number trick (no_std-safe)
    let magic_f64 = Simd::splat(6755399441055744.0f64); // 2^52 + 2^51
    let n = (x * ln2_inv + magic_f64) - magic_f64;
    let r = x - n * ln2;

    // 8th-order Horner polynomial for f64 precision
    let mut p = r * c8 + c7;
    p = r * p + c6;
    p = r * p + c5;
    p = r * p + c4;
    p = r * p + c3;
    p = r * p + c2;
    p = r * p + one;
    p = r * p + one;

    // 2^n via integer exponent — f64 bias 1023, shift 52
    let ni: Simd<i64, WD> = n.cast();
    let pow2n =
        Simd::<f64, WD>::from_bits((ni + Simd::splat(1023i64)).cast::<u64>() << Simd::splat(52u64));

    p * pow2n
}

#[inline(always)]
fn tanh_f64x(x: Simd<f64, WD>) -> Simd<f64, WD> {
    let two = Simd::splat(2.0f64);
    let one = Simd::splat(1.0f64);
    let e2x = exp_f64x(x * two);
    (e2x - one) / (e2x + one)
}

// ═══════════════════════════════════════════════════════════════════
//  Generic loop helpers — process wide chunks + scalar tail
// ═══════════════════════════════════════════════════════════════════

#[inline(always)]
fn map_f32(
    src: &[f32],
    dst: &mut [f32],
    mut simd_op: impl FnMut(Simd<f32, W>) -> Simd<f32, W>,
    mut scalar_op: impl FnMut(f32) -> f32,
) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let chunks = n / W;
    let tail = n % W;

    for i in 0..chunks {
        let off = i * W;
        let v = Simd::from_slice(&src[off..off + W]);
        let r = simd_op(v);
        dst[off..off + W].copy_from_slice(r.as_array());
    }

    let off = chunks * W;
    for j in 0..tail {
        dst[off + j] = scalar_op(src[off + j]);
    }
}

#[inline(always)]
fn zip_f32(
    a: &[f32],
    b: &[f32],
    dst: &mut [f32],
    mut simd_op: impl FnMut(Simd<f32, W>, Simd<f32, W>) -> Simd<f32, W>,
    mut scalar_op: impl FnMut(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let n = a.len();
    let chunks = n / W;
    let tail = n % W;

    for i in 0..chunks {
        let off = i * W;
        let va = Simd::from_slice(&a[off..off + W]);
        let vb = Simd::from_slice(&b[off..off + W]);
        let r = simd_op(va, vb);
        dst[off..off + W].copy_from_slice(r.as_array());
    }

    let off = chunks * W;
    for j in 0..tail {
        dst[off + j] = scalar_op(a[off + j], b[off + j]);
    }
}

#[inline(always)]
fn map_f64(
    src: &[f64],
    dst: &mut [f64],
    mut simd_op: impl FnMut(Simd<f64, WD>) -> Simd<f64, WD>,
    mut scalar_op: impl FnMut(f64) -> f64,
) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let chunks = n / WD;
    let tail = n % WD;

    for i in 0..chunks {
        let off = i * WD;
        let v = Simd::from_slice(&src[off..off + WD]);
        let r = simd_op(v);
        dst[off..off + WD].copy_from_slice(r.as_array());
    }

    let off = chunks * WD;
    for j in 0..tail {
        dst[off + j] = scalar_op(src[off + j]);
    }
}

#[inline(always)]
fn zip_f64(
    a: &[f64],
    b: &[f64],
    dst: &mut [f64],
    mut simd_op: impl FnMut(Simd<f64, WD>, Simd<f64, WD>) -> Simd<f64, WD>,
    mut scalar_op: impl FnMut(f64, f64) -> f64,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let n = a.len();
    let chunks = n / WD;
    let tail = n % WD;

    for i in 0..chunks {
        let off = i * WD;
        let va = Simd::from_slice(&a[off..off + WD]);
        let vb = Simd::from_slice(&b[off..off + WD]);
        let r = simd_op(va, vb);
        dst[off..off + WD].copy_from_slice(r.as_array());
    }

    let off = chunks * WD;
    for j in 0..tail {
        dst[off + j] = scalar_op(a[off + j], b[off + j]);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F16/BF16 load/store helpers
// ═══════════════════════════════════════════════════════════════════

#[inline(always)]
fn load_f16x(src: &[u8], off: usize) -> Simd<f32, W> {
    Simd::from_array(core::array::from_fn(|i| {
        let bo = off + i * 2;
        crate::cast::read_f16(&src[bo..bo + 2])
    }))
}

#[inline(always)]
fn store_f16x(dst: &mut [u8], off: usize, v: Simd<f32, W>) {
    let a = v.as_array();
    for i in 0..W {
        let bo = off + i * 2;
        crate::cast::write_f16(&mut dst[bo..bo + 2], a[i]);
    }
}

#[inline(always)]
fn load_bf16x(src: &[u8], off: usize) -> Simd<f32, W> {
    Simd::from_array(core::array::from_fn(|i| {
        let bo = off + i * 2;
        crate::cast::read_bf16(&src[bo..bo + 2])
    }))
}

#[inline(always)]
fn store_bf16x(dst: &mut [u8], off: usize, v: Simd<f32, W>) {
    let a = v.as_array();
    for i in 0..W {
        let bo = off + i * 2;
        crate::cast::write_bf16(&mut dst[bo..bo + 2], a[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F16 loop helpers
// ═══════════════════════════════════════════════════════════════════

#[inline(always)]
fn map_f16(
    src: &[u8],
    dst: &mut [u8],
    mut simd_op: impl FnMut(Simd<f32, W>) -> Simd<f32, W>,
    mut scalar_op: impl FnMut(f32) -> f32,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 2, 0);
    let n = src.len() / 2;
    let chunks = n / W;
    let tail = n % W;

    for i in 0..chunks {
        let off = i * W * 2;
        let v = load_f16x(src, off);
        let r = simd_op(v);
        store_f16x(dst, off, r);
    }

    let off = chunks * W * 2;
    for j in 0..tail {
        let bo = off + j * 2;
        let x = crate::cast::read_f16(&src[bo..bo + 2]);
        crate::cast::write_f16(&mut dst[bo..bo + 2], scalar_op(x));
    }
}

#[inline(always)]
fn zip_f16(
    a: &[u8],
    b: &[u8],
    dst: &mut [u8],
    mut simd_op: impl FnMut(Simd<f32, W>, Simd<f32, W>) -> Simd<f32, W>,
    mut scalar_op: impl FnMut(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    debug_assert_eq!(a.len() % 2, 0);
    let n = a.len() / 2;
    let chunks = n / W;
    let tail = n % W;

    for i in 0..chunks {
        let off = i * W * 2;
        let va = load_f16x(a, off);
        let vb = load_f16x(b, off);
        let r = simd_op(va, vb);
        store_f16x(dst, off, r);
    }

    let off = chunks * W * 2;
    for j in 0..tail {
        let bo = off + j * 2;
        let x = crate::cast::read_f16(&a[bo..bo + 2]);
        let y = crate::cast::read_f16(&b[bo..bo + 2]);
        crate::cast::write_f16(&mut dst[bo..bo + 2], scalar_op(x, y));
    }
}

// ═══════════════════════════════════════════════════════════════════
//  BF16 loop helpers
// ═══════════════════════════════════════════════════════════════════

#[inline(always)]
fn map_bf16(
    src: &[u8],
    dst: &mut [u8],
    mut simd_op: impl FnMut(Simd<f32, W>) -> Simd<f32, W>,
    mut scalar_op: impl FnMut(f32) -> f32,
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 2, 0);
    let n = src.len() / 2;
    let chunks = n / W;
    let tail = n % W;

    for i in 0..chunks {
        let off = i * W * 2;
        let v = load_bf16x(src, off);
        let r = simd_op(v);
        store_bf16x(dst, off, r);
    }

    let off = chunks * W * 2;
    for j in 0..tail {
        let bo = off + j * 2;
        let x = crate::cast::read_bf16(&src[bo..bo + 2]);
        crate::cast::write_bf16(&mut dst[bo..bo + 2], scalar_op(x));
    }
}

#[inline(always)]
fn zip_bf16(
    a: &[u8],
    b: &[u8],
    dst: &mut [u8],
    mut simd_op: impl FnMut(Simd<f32, W>, Simd<f32, W>) -> Simd<f32, W>,
    mut scalar_op: impl FnMut(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    debug_assert_eq!(a.len() % 2, 0);
    let n = a.len() / 2;
    let chunks = n / W;
    let tail = n % W;

    for i in 0..chunks {
        let off = i * W * 2;
        let va = load_bf16x(a, off);
        let vb = load_bf16x(b, off);
        let r = simd_op(va, vb);
        store_bf16x(dst, off, r);
    }

    let off = chunks * W * 2;
    for j in 0..tail {
        let bo = off + j * 2;
        let x = crate::cast::read_bf16(&a[bo..bo + 2]);
        let y = crate::cast::read_bf16(&b[bo..bo + 2]);
        crate::cast::write_bf16(&mut dst[bo..bo + 2], scalar_op(x, y));
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F32 — 4 elements per 128-bit register
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn relu_f32(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let zero = Simd::<f32, W>::splat(0.0);
    let chunks = src.len() / W;
    let pairs = chunks / 2;
    for i in 0..pairs {
        let o0 = i * 2 * W;
        let o1 = o0 + W;
        let v0 = Simd::<f32, W>::from_slice(&src[o0..]);
        let v1 = Simd::<f32, W>::from_slice(&src[o1..]);
        v0.simd_max(zero).copy_to_slice(&mut dst[o0..]);
        v1.simd_max(zero).copy_to_slice(&mut dst[o1..]);
    }
    let off = pairs * 2 * W;
    if chunks & 1 != 0 {
        let v = Simd::<f32, W>::from_slice(&src[off..]);
        v.simd_max(zero).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * W..src.len() {
        dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 };
    }
}

pub(crate) fn neg_f32(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let chunks = src.len() / W;
    let pairs = chunks / 2;
    for i in 0..pairs {
        let o0 = i * 2 * W;
        let o1 = o0 + W;
        let v0 = Simd::<f32, W>::from_slice(&src[o0..]);
        let v1 = Simd::<f32, W>::from_slice(&src[o1..]);
        (-v0).copy_to_slice(&mut dst[o0..]);
        (-v1).copy_to_slice(&mut dst[o1..]);
    }
    let off = pairs * 2 * W;
    if chunks & 1 != 0 {
        let v = Simd::<f32, W>::from_slice(&src[off..]);
        (-v).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * W..src.len() {
        dst[j] = -src[j];
    }
}

pub(crate) fn sqrt_f32(src: &[f32], dst: &mut [f32]) {
    map_f32(src, dst, sqrt_f32x, axmath::sqrt_f32);
}

pub(crate) fn exp_f32(src: &[f32], dst: &mut [f32]) {
    map_f32(src, dst, exp_f32x, axmath::exp_f32);
}

pub(crate) fn tanh_f32(src: &[f32], dst: &mut [f32]) {
    map_f32(src, dst, tanh_f32x, axmath::tanh_f32);
}

pub(crate) fn sigmoid_f32(src: &[f32], dst: &mut [f32]) {
    let one = Simd::splat(1.0f32);
    map_f32(
        src,
        dst,
        |v| {
            let e = exp_f32x(-v);
            one / (one + e)
        },
        axmath::sigmoid_f32,
    );
}

pub(crate) fn silu_f32(src: &[f32], dst: &mut [f32]) {
    let one = Simd::splat(1.0f32);
    map_f32(
        src,
        dst,
        |v| {
            let e = exp_f32x(-v);
            v / (one + e)
        },
        axmath::silu_f32,
    );
}

pub(crate) fn gelu_f32(src: &[f32], dst: &mut [f32]) {
    let half = Simd::splat(0.5f32);
    let one = Simd::splat(1.0f32);
    let c = Simd::splat(0.797_884_56f32); // sqrt(2/pi)
    let k = Simd::splat(0.044715f32);
    map_f32(
        src,
        dst,
        |v| {
            let x3 = v * v * v;
            let inner = k * x3 + v;
            let t = tanh_f32x(c * inner);
            half * v * (one + t)
        },
        axmath::gelu_f32,
    );
}

pub(crate) fn add_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let chunks = a.len() / W;
    let pairs = chunks / 2;
    for i in 0..pairs {
        let o0 = i * 2 * W;
        let o1 = o0 + W;
        let va0 = Simd::<f32, W>::from_slice(&a[o0..]);
        let va1 = Simd::<f32, W>::from_slice(&a[o1..]);
        let vb0 = Simd::<f32, W>::from_slice(&b[o0..]);
        let vb1 = Simd::<f32, W>::from_slice(&b[o1..]);
        (va0 + vb0).copy_to_slice(&mut dst[o0..]);
        (va1 + vb1).copy_to_slice(&mut dst[o1..]);
    }
    let off = pairs * 2 * W;
    if chunks & 1 != 0 {
        let va = Simd::<f32, W>::from_slice(&a[off..]);
        let vb = Simd::<f32, W>::from_slice(&b[off..]);
        (va + vb).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * W..a.len() {
        dst[j] = a[j] + b[j];
    }
}

pub(crate) fn sub_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let chunks = a.len() / W;
    let pairs = chunks / 2;
    for i in 0..pairs {
        let o0 = i * 2 * W;
        let o1 = o0 + W;
        let va0 = Simd::<f32, W>::from_slice(&a[o0..]);
        let va1 = Simd::<f32, W>::from_slice(&a[o1..]);
        let vb0 = Simd::<f32, W>::from_slice(&b[o0..]);
        let vb1 = Simd::<f32, W>::from_slice(&b[o1..]);
        (va0 - vb0).copy_to_slice(&mut dst[o0..]);
        (va1 - vb1).copy_to_slice(&mut dst[o1..]);
    }
    let off = pairs * 2 * W;
    if chunks & 1 != 0 {
        let va = Simd::<f32, W>::from_slice(&a[off..]);
        let vb = Simd::<f32, W>::from_slice(&b[off..]);
        (va - vb).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * W..a.len() {
        dst[j] = a[j] - b[j];
    }
}

pub(crate) fn mul_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let chunks = a.len() / W;
    let pairs = chunks / 2;
    for i in 0..pairs {
        let o0 = i * 2 * W;
        let o1 = o0 + W;
        let va0 = Simd::<f32, W>::from_slice(&a[o0..]);
        let va1 = Simd::<f32, W>::from_slice(&a[o1..]);
        let vb0 = Simd::<f32, W>::from_slice(&b[o0..]);
        let vb1 = Simd::<f32, W>::from_slice(&b[o1..]);
        (va0 * vb0).copy_to_slice(&mut dst[o0..]);
        (va1 * vb1).copy_to_slice(&mut dst[o1..]);
    }
    let off = pairs * 2 * W;
    if chunks & 1 != 0 {
        let va = Simd::<f32, W>::from_slice(&a[off..]);
        let vb = Simd::<f32, W>::from_slice(&b[off..]);
        (va * vb).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * W..a.len() {
        dst[j] = a[j] * b[j];
    }
}

/// Pack the OFFSET..OFFSET+W portion of W xyz triples using two fixed
/// shuffles. The first supplies x/y lanes; the second fills the z lanes.
/// Indices are compile-time constants for both supported W=4 and W=8.
#[inline(always)]
fn pack_xyz<const OFFSET: usize>(
    x: Simd<f32, W>,
    y: Simd<f32, W>,
    z: Simd<f32, W>,
) -> Simd<f32, W> {
    struct Xy<const OFFSET: usize>;
    impl<const OFFSET: usize> Swizzle<W> for Xy<OFFSET> {
        const INDEX: [usize; W] = {
            let mut indices = [0; W];
            let mut lane = 0;
            while lane < W {
                let flat = OFFSET + lane;
                indices[lane] = if flat % 3 == 1 {
                    W + flat / 3
                } else {
                    flat / 3
                };
                lane += 1;
            }
            indices
        };
    }
    struct Xyz<const OFFSET: usize>;
    impl<const OFFSET: usize> Swizzle<W> for Xyz<OFFSET> {
        const INDEX: [usize; W] = {
            let mut indices = [0; W];
            let mut lane = 0;
            while lane < W {
                let flat = OFFSET + lane;
                indices[lane] = if flat % 3 == 2 { W + flat / 3 } else { lane };
                lane += 1;
            }
            indices
        };
    }
    Xyz::<OFFSET>::concat_swizzle(Xy::<OFFSET>::concat_swizzle(x, y), z)
}

/// Combine three scalar MULs with dense xyz packing. Each product is rounded
/// once to F32 before the lane permutation; no intermediate plane is stored.
pub(crate) fn scale_interleave3_f32(
    planes: [&[f32]; 3],
    scale: f32,
    dst: &mut [f32],
    scale_first: [bool; 3],
) {
    debug_assert!(planes.iter().all(|plane| plane.len() == planes[0].len()));
    debug_assert_eq!(planes[0].len().checked_mul(3), Some(dst.len()));
    // Resolve all authored operand orders once, outside the vector/tail loops.
    match scale_first {
        [false, false, false] => scale_interleave3_impl::<false, false, false>(planes, scale, dst),
        [false, false, true] => scale_interleave3_impl::<false, false, true>(planes, scale, dst),
        [false, true, false] => scale_interleave3_impl::<false, true, false>(planes, scale, dst),
        [false, true, true] => scale_interleave3_impl::<false, true, true>(planes, scale, dst),
        [true, false, false] => scale_interleave3_impl::<true, false, false>(planes, scale, dst),
        [true, false, true] => scale_interleave3_impl::<true, false, true>(planes, scale, dst),
        [true, true, false] => scale_interleave3_impl::<true, true, false>(planes, scale, dst),
        [true, true, true] => scale_interleave3_impl::<true, true, true>(planes, scale, dst),
    }
}

// Both vector chunks and tails retain the authored MUL operand order rather
// than applying the lint's commutative rewrite.
#[allow(clippy::if_same_then_else)]
fn scale_interleave3_impl<const X: bool, const Y: bool, const Z: bool>(
    planes: [&[f32]; 3],
    scale: f32,
    dst: &mut [f32],
) {
    let scale_vector = Simd::<f32, W>::splat(scale);
    let chunks = planes[0].len() / W;
    for chunk in 0..chunks {
        let offset = chunk * W;
        let x = Simd::<f32, W>::from_slice(&planes[0][offset..]);
        let y = Simd::<f32, W>::from_slice(&planes[1][offset..]);
        let z = Simd::<f32, W>::from_slice(&planes[2][offset..]);
        let x = if X {
            scale_vector * x
        } else {
            x * scale_vector
        };
        let y = if Y {
            scale_vector * y
        } else {
            y * scale_vector
        };
        let z = if Z {
            scale_vector * z
        } else {
            z * scale_vector
        };
        let packed = &mut dst[offset * 3..];
        pack_xyz::<0>(x, y, z).copy_to_slice(packed);
        pack_xyz::<W>(x, y, z).copy_to_slice(&mut packed[W..]);
        pack_xyz::<{ W * 2 }>(x, y, z).copy_to_slice(&mut packed[W * 2..]);
    }
    for index in chunks * W..planes[0].len() {
        let x = planes[0][index];
        let y = planes[1][index];
        let z = planes[2][index];
        dst[index * 3] = if X { scale * x } else { x * scale };
        dst[index * 3 + 1] = if Y { scale * y } else { y * scale };
        dst[index * 3 + 2] = if Z { scale * z } else { z * scale };
    }
}

/// Compute one strict plane `MUL -> ADD` region without materializing the
/// product plane. Ordinary SIMD multiply and add operators preserve the graph
/// arithmetic boundaries; this does not opt into FMA contraction.
pub(crate) fn mul_add_f32(
    left: &[f32],
    right: &[f32],
    addend: &[f32],
    output: &mut [f32],
    order: crate::elementwise::MulAddOrder,
) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), addend.len());
    debug_assert_eq!(left.len(), output.len());
    let chunks = left.len() / W;
    for index in 0..chunks {
        let offset = index * W;
        let left = Simd::<f32, W>::from_slice(&left[offset..]);
        let right = Simd::<f32, W>::from_slice(&right[offset..]);
        let addend = Simd::<f32, W>::from_slice(&addend[offset..]);
        let product = left * right;
        let result = match order {
            crate::elementwise::MulAddOrder::ProductAddend => product + addend,
            crate::elementwise::MulAddOrder::AddendProduct => addend + product,
        };
        result.copy_to_slice(&mut output[offset..]);
    }
    for index in chunks * W..left.len() {
        let product = left[index] * right[index];
        output[index] = match order {
            crate::elementwise::MulAddOrder::ProductAddend => product + addend[index],
            crate::elementwise::MulAddOrder::AddendProduct => addend[index] + product,
        };
    }
}

pub(crate) fn mul_add_f32_inplace(
    left: &mut [f32],
    right: &[f32],
    addend: &[f32],
    order: crate::elementwise::MulAddOrder,
) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), addend.len());
    let chunks = left.len() / W;
    for index in 0..chunks {
        let offset = index * W;
        let left_value = Simd::<f32, W>::from_slice(&left[offset..]);
        let right = Simd::<f32, W>::from_slice(&right[offset..]);
        let addend = Simd::<f32, W>::from_slice(&addend[offset..]);
        let product = left_value * right;
        let result = match order {
            crate::elementwise::MulAddOrder::ProductAddend => product + addend,
            crate::elementwise::MulAddOrder::AddendProduct => addend + product,
        };
        result.copy_to_slice(&mut left[offset..]);
    }
    for index in chunks * W..left.len() {
        let product = left[index] * right[index];
        left[index] = match order {
            crate::elementwise::MulAddOrder::ProductAddend => product + addend[index],
            crate::elementwise::MulAddOrder::AddendProduct => addend[index] + product,
        };
    }
}

/// Compute `dst = input * scale + addend` with separate IEEE multiply and
/// add operations. Keeping the three live vectors in one kernel removes the
/// graph executor's intermediate tensor without opting into FMA contraction.
pub(crate) fn scale_add_f32(
    input: &[f32],
    scale: f32,
    addend: &[f32],
    dst: &mut [f32],
    order: crate::elementwise::ScaleAddOrder,
) {
    debug_assert_eq!(input.len(), addend.len());
    debug_assert_eq!(input.len(), dst.len());
    match order {
        crate::elementwise::ScaleAddOrder::InputScaleAddend => scale_add_f32_impl(
            input,
            scale,
            addend,
            dst,
            |input, scale, addend| input * scale + addend,
            |input, scale, addend| input * scale + addend,
        ),
        crate::elementwise::ScaleAddOrder::ScaleInputAddend => scale_add_f32_impl(
            input,
            scale,
            addend,
            dst,
            |input, scale, addend| scale * input + addend,
            |input, scale, addend| scale * input + addend,
        ),
        crate::elementwise::ScaleAddOrder::AddendInputScale => scale_add_f32_impl(
            input,
            scale,
            addend,
            dst,
            |input, scale, addend| addend + input * scale,
            |input, scale, addend| addend + input * scale,
        ),
        crate::elementwise::ScaleAddOrder::AddendScaleInput => scale_add_f32_impl(
            input,
            scale,
            addend,
            dst,
            |input, scale, addend| addend + scale * input,
            |input, scale, addend| addend + scale * input,
        ),
    }
}

fn scale_add_f32_impl(
    input: &[f32],
    scale: f32,
    addend: &[f32],
    dst: &mut [f32],
    vector: impl Fn(Simd<f32, W>, Simd<f32, W>, Simd<f32, W>) -> Simd<f32, W>,
    scalar: impl Fn(f32, f32, f32) -> f32,
) {
    let scale_lanes = Simd::<f32, W>::splat(scale);
    let chunks = input.len() / W;
    for i in 0..chunks {
        let offset = i * W;
        let input = Simd::<f32, W>::from_slice(&input[offset..]);
        let addend = Simd::<f32, W>::from_slice(&addend[offset..]);
        vector(input, scale_lanes, addend).copy_to_slice(&mut dst[offset..]);
    }
    for i in chunks * W..input.len() {
        dst[i] = scalar(input[i], scale, addend[i]);
    }
}

/// In-place form of [`scale_add_f32`]. The old input lane is loaded before
/// the result overwrites it, so the multiply/add intermediate remains solely
/// in the SIMD register file.
pub(crate) fn scale_add_f32_inplace(
    input: &mut [f32],
    scale: f32,
    addend: &[f32],
    order: crate::elementwise::ScaleAddOrder,
) {
    debug_assert_eq!(input.len(), addend.len());
    match order {
        crate::elementwise::ScaleAddOrder::InputScaleAddend => scale_add_f32_inplace_impl(
            input,
            scale,
            addend,
            |input, scale, addend| input * scale + addend,
            |input, scale, addend| input * scale + addend,
        ),
        crate::elementwise::ScaleAddOrder::ScaleInputAddend => scale_add_f32_inplace_impl(
            input,
            scale,
            addend,
            |input, scale, addend| scale * input + addend,
            |input, scale, addend| scale * input + addend,
        ),
        crate::elementwise::ScaleAddOrder::AddendInputScale => scale_add_f32_inplace_impl(
            input,
            scale,
            addend,
            |input, scale, addend| addend + input * scale,
            |input, scale, addend| addend + input * scale,
        ),
        crate::elementwise::ScaleAddOrder::AddendScaleInput => scale_add_f32_inplace_impl(
            input,
            scale,
            addend,
            |input, scale, addend| addend + scale * input,
            |input, scale, addend| addend + scale * input,
        ),
    }
}

fn scale_add_f32_inplace_impl(
    input: &mut [f32],
    scale: f32,
    addend: &[f32],
    vector: impl Fn(Simd<f32, W>, Simd<f32, W>, Simd<f32, W>) -> Simd<f32, W>,
    scalar: impl Fn(f32, f32, f32) -> f32,
) {
    let scale_lanes = Simd::<f32, W>::splat(scale);
    let chunks = input.len() / W;
    for i in 0..chunks {
        let offset = i * W;
        let input_lanes = Simd::<f32, W>::from_slice(&input[offset..]);
        let addend_lanes = Simd::<f32, W>::from_slice(&addend[offset..]);
        vector(input_lanes, scale_lanes, addend_lanes).copy_to_slice(&mut input[offset..]);
    }
    for i in chunks * W..input.len() {
        input[i] = scalar(input[i], scale, addend[i]);
    }
}

/// Scalar-addend form of [`scale_add_f32`].
pub(crate) fn scale_add_scalar_f32(
    input: &[f32],
    scale: f32,
    addend: f32,
    dst: &mut [f32],
    order: crate::elementwise::ScaleAddOrder,
) {
    debug_assert_eq!(input.len(), dst.len());
    match order {
        crate::elementwise::ScaleAddOrder::InputScaleAddend => scale_add_scalar_f32_impl(
            input,
            scale,
            addend,
            dst,
            |input, scale, addend| input * scale + addend,
            |input, scale, addend| input * scale + addend,
        ),
        crate::elementwise::ScaleAddOrder::ScaleInputAddend => scale_add_scalar_f32_impl(
            input,
            scale,
            addend,
            dst,
            |input, scale, addend| scale * input + addend,
            |input, scale, addend| scale * input + addend,
        ),
        crate::elementwise::ScaleAddOrder::AddendInputScale => scale_add_scalar_f32_impl(
            input,
            scale,
            addend,
            dst,
            |input, scale, addend| addend + input * scale,
            |input, scale, addend| addend + input * scale,
        ),
        crate::elementwise::ScaleAddOrder::AddendScaleInput => scale_add_scalar_f32_impl(
            input,
            scale,
            addend,
            dst,
            |input, scale, addend| addend + scale * input,
            |input, scale, addend| addend + scale * input,
        ),
    }
}

fn scale_add_scalar_f32_impl(
    input: &[f32],
    scale: f32,
    addend: f32,
    dst: &mut [f32],
    vector: impl Fn(Simd<f32, W>, Simd<f32, W>, Simd<f32, W>) -> Simd<f32, W>,
    scalar: impl Fn(f32, f32, f32) -> f32,
) {
    let scale_lanes = Simd::<f32, W>::splat(scale);
    let addend_lanes = Simd::<f32, W>::splat(addend);
    let chunks = input.len() / W;
    for i in 0..chunks {
        let offset = i * W;
        let input = Simd::<f32, W>::from_slice(&input[offset..]);
        vector(input, scale_lanes, addend_lanes).copy_to_slice(&mut dst[offset..]);
    }
    for i in chunks * W..input.len() {
        dst[i] = scalar(input[i], scale, addend);
    }
}

/// In-place scalar-addend form of [`scale_add_scalar_f32`].
pub(crate) fn scale_add_scalar_f32_inplace(
    input: &mut [f32],
    scale: f32,
    addend: f32,
    order: crate::elementwise::ScaleAddOrder,
) {
    match order {
        crate::elementwise::ScaleAddOrder::InputScaleAddend => scale_add_scalar_f32_inplace_impl(
            input,
            scale,
            addend,
            |input, scale, addend| input * scale + addend,
            |input, scale, addend| input * scale + addend,
        ),
        crate::elementwise::ScaleAddOrder::ScaleInputAddend => scale_add_scalar_f32_inplace_impl(
            input,
            scale,
            addend,
            |input, scale, addend| scale * input + addend,
            |input, scale, addend| scale * input + addend,
        ),
        crate::elementwise::ScaleAddOrder::AddendInputScale => scale_add_scalar_f32_inplace_impl(
            input,
            scale,
            addend,
            |input, scale, addend| addend + input * scale,
            |input, scale, addend| addend + input * scale,
        ),
        crate::elementwise::ScaleAddOrder::AddendScaleInput => scale_add_scalar_f32_inplace_impl(
            input,
            scale,
            addend,
            |input, scale, addend| addend + scale * input,
            |input, scale, addend| addend + scale * input,
        ),
    }
}

/// In-place scalar-target linear interpolation. This is the strict SIMD form
/// of a graph SUB → MUL → ADD chain; keeping the old current lane in a
/// register removes both intermediate tensor planes.
pub(crate) fn lerp_scalar_f32_inplace(current: &mut [f32], target: f32, weight: &[f32]) {
    debug_assert_eq!(current.len(), weight.len());
    let target_lanes = Simd::<f32, W>::splat(target);
    let chunks = current.len() / W;
    for index in 0..chunks {
        let offset = index * W;
        let current_lanes = Simd::<f32, W>::from_slice(&current[offset..]);
        let weight_lanes = Simd::<f32, W>::from_slice(&weight[offset..]);
        let delta = target_lanes - current_lanes;
        (current_lanes + delta * weight_lanes).copy_to_slice(&mut current[offset..]);
    }
    for index in chunks * W..current.len() {
        current[index] = current[index] + (target - current[index]) * weight[index];
    }
}

/// Strict in-place two-axis linearized translation followed by scalar clamp.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linearized_clamp_f32_inplace(
    value: &mut [f32],
    derivative_x: &[f32],
    offset_x: f32,
    derivative_y: &[f32],
    offset_y: f32,
    lower: f32,
    upper: f32,
) {
    debug_assert_eq!(value.len(), derivative_x.len());
    debug_assert_eq!(value.len(), derivative_y.len());
    let offset_x_lanes = Simd::<f32, W>::splat(offset_x);
    let offset_y_lanes = Simd::<f32, W>::splat(offset_y);
    let lower_lanes = Simd::<f32, W>::splat(lower);
    let upper_lanes = Simd::<f32, W>::splat(upper);
    let chunks = value.len() / W;
    for index in 0..chunks {
        let offset = index * W;
        let value_lanes = Simd::<f32, W>::from_slice(&value[offset..]);
        let derivative_x = Simd::<f32, W>::from_slice(&derivative_x[offset..]);
        let derivative_y = Simd::<f32, W>::from_slice(&derivative_y[offset..]);
        let x = derivative_x * offset_x_lanes;
        let y = derivative_y * offset_y_lanes;
        let shifted = (value_lanes + x) + y;
        let shifted_nan = shifted.simd_ne(shifted);
        let lower_clamped = if lower.is_nan() {
            lower_lanes
        } else {
            shifted_nan.select(
                shifted,
                shifted.simd_gt(lower_lanes).select(shifted, lower_lanes),
            )
        };
        let lower_nan = lower_clamped.simd_ne(lower_clamped);
        let clamped = if upper.is_nan() {
            upper_lanes
        } else {
            lower_nan.select(
                lower_clamped,
                lower_clamped
                    .simd_lt(upper_lanes)
                    .select(lower_clamped, upper_lanes),
            )
        };
        clamped.copy_to_slice(&mut value[offset..]);
    }
    for index in chunks * W..value.len() {
        let x = derivative_x[index] * offset_x;
        let y = derivative_y[index] * offset_y;
        let shifted = (value[index] + x) + y;
        let lower_clamped = if shifted.is_nan() {
            shifted
        } else if lower.is_nan() {
            lower
        } else if shifted > lower {
            shifted
        } else {
            lower
        };
        value[index] = if lower_clamped.is_nan() {
            lower_clamped
        } else if upper.is_nan() {
            upper
        } else if lower_clamped < upper {
            lower_clamped
        } else {
            upper
        };
    }
}

/// In-place `input * rsqrt(maximum(input, floor))` with TOSA's propagating
/// NaN policy. The common positive-normal path evaluates axmath's square-root
/// refinement lane-wise in portable SIMD; exceptional chunks retain the
/// scalar reference behavior exactly.
pub(crate) fn guarded_sqrt_f32_inplace(input: &mut [f32], floor: f32) {
    let floor_lanes = Simd::<f32, W>::splat(floor);
    let zero = Simd::<f32, W>::splat(0.0);
    let one = Simd::<f32, W>::splat(1.0);
    let half = Simd::<f32, W>::splat(0.5);
    let chunks = input.len() / W;
    for index in 0..chunks {
        let offset = index * W;
        let value = Simd::<f32, W>::from_slice(&input[offset..]);
        let value_nan = value.simd_ne(value);
        let guarded = if floor.is_nan() {
            value_nan.select(value, floor_lanes)
        } else {
            value_nan.select(value, value.simd_gt(floor_lanes).select(value, floor_lanes))
        };
        let bits = guarded.to_bits();
        let exponent_bits = (bits & Simd::splat(0x7f80_0000)) >> Simd::splat(23);
        let positive_normal = guarded.simd_gt(zero)
            & exponent_bits.simd_ne(Simd::splat(0))
            & exponent_bits.simd_ne(Simd::splat(0xff));
        if positive_normal.all() {
            let mantissa = bits & Simd::splat(0x007f_ffff);
            let mut fraction = Simd::<f32, W>::from_bits(Simd::splat(126_u32 << 23) | mantissa);
            let mut exponent = exponent_bits.cast::<i32>() - Simd::splat(126);
            let odd = (exponent & Simd::splat(1)).simd_ne(Simd::splat(0));
            fraction = odd.select(fraction * Simd::splat(2.0), fraction);
            exponent = odd.select(exponent - Simd::splat(1), exponent);

            // Mirror axmath's three Newton refinements without contraction so
            // graph fusion does not silently alter arithmetic boundaries.
            let mut root = half * (one + fraction);
            root = half * (root + fraction / root);
            root = half * (root + fraction / root);
            root = half * (root + fraction / root);
            let scale_bits = (exponent >> Simd::splat(1)) + Simd::splat(127);
            let scale = Simd::<f32, W>::from_bits(scale_bits.cast::<u32>() << Simd::splat(23));
            root *= scale;
            let inverse = one / root;
            (value * inverse).copy_to_slice(&mut input[offset..]);
        } else {
            for lane in offset..offset + W {
                let guarded = if input[lane].is_nan() {
                    input[lane]
                } else if floor.is_nan() {
                    floor
                } else {
                    input[lane].max(floor)
                };
                input[lane] *= 1.0 / axmath::sqrt_f32(guarded);
            }
        }
    }
    for value in &mut input[chunks * W..] {
        let guarded = if value.is_nan() {
            *value
        } else if floor.is_nan() {
            floor
        } else {
            value.max(floor)
        };
        *value *= 1.0 / axmath::sqrt_f32(guarded);
    }
}

fn scale_add_scalar_f32_inplace_impl(
    input: &mut [f32],
    scale: f32,
    addend: f32,
    vector: impl Fn(Simd<f32, W>, Simd<f32, W>, Simd<f32, W>) -> Simd<f32, W>,
    scalar: impl Fn(f32, f32, f32) -> f32,
) {
    let scale_lanes = Simd::<f32, W>::splat(scale);
    let addend_lanes = Simd::<f32, W>::splat(addend);
    let chunks = input.len() / W;
    for i in 0..chunks {
        let offset = i * W;
        let input_lanes = Simd::<f32, W>::from_slice(&input[offset..]);
        vector(input_lanes, scale_lanes, addend_lanes).copy_to_slice(&mut input[offset..]);
    }
    for value in &mut input[chunks * W..] {
        *value = scalar(*value, scale, addend);
    }
}

pub(crate) fn div_f32(a: &[f32], b: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let chunks = a.len() / W;
    let pairs = chunks / 2;
    for i in 0..pairs {
        let o0 = i * 2 * W;
        let o1 = o0 + W;
        let va0 = Simd::<f32, W>::from_slice(&a[o0..]);
        let va1 = Simd::<f32, W>::from_slice(&a[o1..]);
        let vb0 = Simd::<f32, W>::from_slice(&b[o0..]);
        let vb1 = Simd::<f32, W>::from_slice(&b[o1..]);
        (va0 / vb0).copy_to_slice(&mut dst[o0..]);
        (va1 / vb1).copy_to_slice(&mut dst[o1..]);
    }
    let off = pairs * 2 * W;
    if chunks & 1 != 0 {
        let va = Simd::<f32, W>::from_slice(&a[off..]);
        let vb = Simd::<f32, W>::from_slice(&b[off..]);
        (va / vb).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * W..a.len() {
        dst[j] = a[j] / b[j];
    }
}

pub(crate) fn scale_f32(src: &[f32], dst: &mut [f32], s: f32) {
    debug_assert_eq!(src.len(), dst.len());
    let vs = Simd::<f32, W>::splat(s);
    let chunks = src.len() / W;
    let pairs = chunks / 2;
    for i in 0..pairs {
        let o0 = i * 2 * W;
        let o1 = o0 + W;
        let v0 = Simd::<f32, W>::from_slice(&src[o0..]);
        let v1 = Simd::<f32, W>::from_slice(&src[o1..]);
        (v0 * vs).copy_to_slice(&mut dst[o0..]);
        (v1 * vs).copy_to_slice(&mut dst[o1..]);
    }
    let off = pairs * 2 * W;
    if chunks & 1 != 0 {
        let v = Simd::<f32, W>::from_slice(&src[off..]);
        (v * vs).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * W..src.len() {
        dst[j] = src[j] * s;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F64 — 2 elements per 128-bit register
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn relu_f64(src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    let zero = Simd::<f64, WD>::splat(0.0);
    let chunks = src.len() / WD;
    for i in 0..chunks {
        let off = i * WD;
        let v = Simd::<f64, WD>::from_slice(&src[off..]);
        v.simd_max(zero).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * WD..src.len() {
        dst[j] = if src[j] > 0.0 { src[j] } else { 0.0 };
    }
}

pub(crate) fn neg_f64(src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    let chunks = src.len() / WD;
    for i in 0..chunks {
        let off = i * WD;
        let v = Simd::<f64, WD>::from_slice(&src[off..]);
        (-v).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * WD..src.len() {
        dst[j] = -src[j];
    }
}

pub(crate) fn sqrt_f64(src: &[f64], dst: &mut [f64]) {
    map_f64(src, dst, sqrt_f64x, axmath::sqrt_f64);
}

pub(crate) fn exp_f64(src: &[f64], dst: &mut [f64]) {
    map_f64(src, dst, exp_f64x, axmath::exp_f64);
}

pub(crate) fn tanh_f64(src: &[f64], dst: &mut [f64]) {
    map_f64(src, dst, tanh_f64x, axmath::tanh_f64);
}

pub(crate) fn sigmoid_f64(src: &[f64], dst: &mut [f64]) {
    let one = Simd::splat(1.0f64);
    map_f64(
        src,
        dst,
        |v| {
            let e = exp_f64x(-v);
            one / (one + e)
        },
        axmath::sigmoid_f64,
    );
}

pub(crate) fn silu_f64(src: &[f64], dst: &mut [f64]) {
    let one = Simd::splat(1.0f64);
    map_f64(
        src,
        dst,
        |v| {
            let e = exp_f64x(-v);
            v / (one + e)
        },
        axmath::silu_f64,
    );
}

pub(crate) fn gelu_f64(src: &[f64], dst: &mut [f64]) {
    let half = Simd::splat(0.5f64);
    let one = Simd::splat(1.0f64);
    let c = Simd::splat(0.797_884_560_802_865_4f64); // sqrt(2/pi)
    let k = Simd::splat(0.044715f64);
    map_f64(
        src,
        dst,
        |v| {
            let x3 = v * v * v;
            let inner = k * x3 + v;
            let t = tanh_f64x(c * inner);
            half * v * (one + t)
        },
        axmath::gelu_f64,
    );
}

pub(crate) fn add_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let chunks = a.len() / WD;
    for i in 0..chunks {
        let off = i * WD;
        let va = Simd::<f64, WD>::from_slice(&a[off..]);
        let vb = Simd::<f64, WD>::from_slice(&b[off..]);
        (va + vb).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * WD..a.len() {
        dst[j] = a[j] + b[j];
    }
}

pub(crate) fn sub_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let chunks = a.len() / WD;
    for i in 0..chunks {
        let off = i * WD;
        let va = Simd::<f64, WD>::from_slice(&a[off..]);
        let vb = Simd::<f64, WD>::from_slice(&b[off..]);
        (va - vb).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * WD..a.len() {
        dst[j] = a[j] - b[j];
    }
}

pub(crate) fn mul_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let chunks = a.len() / WD;
    for i in 0..chunks {
        let off = i * WD;
        let va = Simd::<f64, WD>::from_slice(&a[off..]);
        let vb = Simd::<f64, WD>::from_slice(&b[off..]);
        (va * vb).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * WD..a.len() {
        dst[j] = a[j] * b[j];
    }
}

pub(crate) fn div_f64(a: &[f64], b: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let chunks = a.len() / WD;
    for i in 0..chunks {
        let off = i * WD;
        let va = Simd::<f64, WD>::from_slice(&a[off..]);
        let vb = Simd::<f64, WD>::from_slice(&b[off..]);
        (va / vb).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * WD..a.len() {
        dst[j] = a[j] / b[j];
    }
}

pub(crate) fn scale_f64(src: &[f64], dst: &mut [f64], s: f64) {
    debug_assert_eq!(src.len(), dst.len());
    let vs = Simd::<f64, WD>::splat(s);
    let chunks = src.len() / WD;
    for i in 0..chunks {
        let off = i * WD;
        let v = Simd::<f64, WD>::from_slice(&src[off..]);
        (v * vs).copy_to_slice(&mut dst[off..]);
    }
    for j in chunks * WD..src.len() {
        dst[j] = src[j] * s;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  F16 (packed &[u8]) — widen to f32, compute, narrow back
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn relu_f16(src: &[u8], dst: &mut [u8]) {
    let zero = Simd::<f32, W>::splat(0.0);
    map_f16(
        src,
        dst,
        |v| v.simd_max(zero),
        |x| if x > 0.0 { x } else { 0.0 },
    );
}

pub(crate) fn neg_f16(src: &[u8], dst: &mut [u8]) {
    map_f16(src, dst, |v| -v, |x| -x);
}

pub(crate) fn sqrt_f16(src: &[u8], dst: &mut [u8]) {
    map_f16(src, dst, sqrt_f32x, axmath::sqrt_f32);
}

pub(crate) fn exp_f16(src: &[u8], dst: &mut [u8]) {
    map_f16(src, dst, exp_f32x, axmath::exp_f32);
}

pub(crate) fn tanh_f16(src: &[u8], dst: &mut [u8]) {
    map_f16(src, dst, tanh_f32x, axmath::tanh_f32);
}

pub(crate) fn sigmoid_f16(src: &[u8], dst: &mut [u8]) {
    let one = Simd::splat(1.0f32);
    map_f16(
        src,
        dst,
        |v| {
            let e = exp_f32x(-v);
            one / (one + e)
        },
        axmath::sigmoid_f32,
    );
}

pub(crate) fn silu_f16(src: &[u8], dst: &mut [u8]) {
    let one = Simd::splat(1.0f32);
    map_f16(
        src,
        dst,
        |v| {
            let e = exp_f32x(-v);
            v / (one + e)
        },
        axmath::silu_f32,
    );
}

pub(crate) fn gelu_f16(src: &[u8], dst: &mut [u8]) {
    let half = Simd::splat(0.5f32);
    let one = Simd::splat(1.0f32);
    let c = Simd::splat(0.797_884_56f32);
    let k = Simd::splat(0.044715f32);
    map_f16(
        src,
        dst,
        |v| {
            let x3 = v * v * v;
            let inner = k * x3 + v;
            let t = tanh_f32x(c * inner);
            half * v * (one + t)
        },
        axmath::gelu_f32,
    );
}

pub(crate) fn add_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    zip_f16(a, b, dst, |va, vb| va + vb, |x, y| x + y);
}

pub(crate) fn sub_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    zip_f16(a, b, dst, |va, vb| va - vb, |x, y| x - y);
}

pub(crate) fn mul_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    zip_f16(a, b, dst, |va, vb| va * vb, |x, y| x * y);
}

pub(crate) fn div_f16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    zip_f16(a, b, dst, |va, vb| va / vb, |x, y| x / y);
}

pub(crate) fn scale_f16(src: &[u8], dst: &mut [u8], s: f32) {
    let vs = Simd::<f32, W>::splat(s);
    map_f16(src, dst, |v| v * vs, |x| x * s);
}

// ═══════════════════════════════════════════════════════════════════
//  BF16 (packed &[u8]) — widen to f32, compute, narrow back
// ═══════════════════════════════════════════════════════════════════

pub(crate) fn relu_bf16(src: &[u8], dst: &mut [u8]) {
    let zero = Simd::<f32, W>::splat(0.0);
    map_bf16(
        src,
        dst,
        |v| v.simd_max(zero),
        |x| if x > 0.0 { x } else { 0.0 },
    );
}

pub(crate) fn neg_bf16(src: &[u8], dst: &mut [u8]) {
    map_bf16(src, dst, |v| -v, |x| -x);
}

pub(crate) fn sqrt_bf16(src: &[u8], dst: &mut [u8]) {
    map_bf16(src, dst, sqrt_f32x, axmath::sqrt_f32);
}

pub(crate) fn exp_bf16(src: &[u8], dst: &mut [u8]) {
    map_bf16(src, dst, exp_f32x, axmath::exp_f32);
}

pub(crate) fn tanh_bf16(src: &[u8], dst: &mut [u8]) {
    map_bf16(src, dst, tanh_f32x, axmath::tanh_f32);
}

pub(crate) fn sigmoid_bf16(src: &[u8], dst: &mut [u8]) {
    let one = Simd::splat(1.0f32);
    map_bf16(
        src,
        dst,
        |v| {
            let e = exp_f32x(-v);
            one / (one + e)
        },
        axmath::sigmoid_f32,
    );
}

pub(crate) fn silu_bf16(src: &[u8], dst: &mut [u8]) {
    let one = Simd::splat(1.0f32);
    map_bf16(
        src,
        dst,
        |v| {
            let e = exp_f32x(-v);
            v / (one + e)
        },
        axmath::silu_f32,
    );
}

pub(crate) fn gelu_bf16(src: &[u8], dst: &mut [u8]) {
    let half = Simd::splat(0.5f32);
    let one = Simd::splat(1.0f32);
    let c = Simd::splat(0.797_884_56f32);
    let k = Simd::splat(0.044715f32);
    map_bf16(
        src,
        dst,
        |v| {
            let x3 = v * v * v;
            let inner = k * x3 + v;
            let t = tanh_f32x(c * inner);
            half * v * (one + t)
        },
        axmath::gelu_f32,
    );
}

pub(crate) fn add_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    zip_bf16(a, b, dst, |va, vb| va + vb, |x, y| x + y);
}

pub(crate) fn sub_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    zip_bf16(a, b, dst, |va, vb| va - vb, |x, y| x - y);
}

pub(crate) fn mul_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    zip_bf16(a, b, dst, |va, vb| va * vb, |x, y| x * y);
}

pub(crate) fn div_bf16(a: &[u8], b: &[u8], dst: &mut [u8]) {
    zip_bf16(a, b, dst, |va, vb| va / vb, |x, y| x / y);
}

pub(crate) fn scale_bf16(src: &[u8], dst: &mut [u8], s: f32) {
    let vs = Simd::<f32, W>::splat(s);
    map_bf16(src, dst, |v| v * vs, |x| x * s);
}

// Raster primitives moved to arch/portable/raster.rs

#[cfg(test)]
mod scale_interleave3_tests {
    use super::W;
    use crate::elementwise::scale_interleave3_f32;

    #[test]
    fn ordered_scale_and_xyz_packing_cover_lanes_tails_and_exceptional_values() {
        const N: usize = 3 * W + 3;
        let cases = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(3),
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::from_bits(0x3f80_0001),
            -1.25,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_4321),
        ];
        let planes: [[f32; N]; 3] = core::array::from_fn(|axis| {
            core::array::from_fn(|lane| cases[(lane + axis * 3) % cases.len()])
        });
        for mask in 0..8 {
            let scale_first = core::array::from_fn(|axis| mask & (1 << axis) != 0);
            for scale in [
                0.0,
                -0.0,
                0.5,
                f32::from_bits(0x3f80_0001),
                f32::MIN_POSITIVE,
                f32::MAX,
                f32::INFINITY,
                f32::from_bits(0x7fc0_5678),
            ] {
                let scale = core::hint::black_box(scale);
                for length in 0..=N {
                    let mut actual = [-99.0_f32; 3 * N + 2];
                    scale_interleave3_f32(
                        planes.each_ref().map(|plane| &plane[..length]),
                        scale,
                        &mut actual[1..1 + length * 3],
                        scale_first,
                    );
                    assert_eq!(actual[0], -99.0);
                    assert!(actual[1 + length * 3..].iter().all(|&value| value == -99.0));
                    for lane in 0..length {
                        for axis in 0..3 {
                            let input = core::hint::black_box(planes[axis][lane]);
                            let expected = if scale_first[axis] {
                                scale * input
                            } else {
                                input * scale
                            };
                            let result = actual[1 + lane * 3 + axis];
                            if expected.is_nan() {
                                assert!(result.is_nan());
                            } else {
                                assert_eq!(
                                    result.to_bits(),
                                    expected.to_bits(),
                                    "mask={mask} length={length} lane={lane} axis={axis}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn public_helper_rejects_unequal_planes() {
        scale_interleave3_f32(
            [&[1.0, 2.0], &[3.0], &[4.0, 5.0]],
            2.0,
            &mut [0.0; 6],
            [false; 3],
        );
    }

    #[test]
    #[should_panic]
    fn public_helper_rejects_wrong_packed_length() {
        scale_interleave3_f32([&[1.0]; 3], 2.0, &mut [0.0; 4], [false; 3]);
    }
}

#[cfg(test)]
mod sqrt_tests {
    use super::*;

    fn assert_sqrt_f32(input: f32, actual: f32) {
        if input.is_nan() || input < 0.0 {
            assert!(actual.is_nan(), "sqrt({input:?}) = {actual:?}");
            return;
        }
        let expected = axmath::sqrt_f32(input);
        if input == 0.0 || input.is_infinite() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "sqrt({input:?})");
        } else {
            // Hardware sqrt rounds directly, while axmath's portable Newton
            // oracle can round once more during its final refinement.
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 1,
                "sqrt({input:e}) = {actual:e}, scalar = {expected:e}"
            );
        }
    }

    fn assert_sqrt_f64(input: f64, actual: f64) {
        if input.is_nan() || input < 0.0 {
            assert!(actual.is_nan(), "sqrt({input:?}) = {actual:?}");
            return;
        }
        let expected = axmath::sqrt_f64(input);
        if input == 0.0 || input.is_infinite() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "sqrt({input:?})");
        } else {
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 1,
                "sqrt({input:e}) = {actual:e}, scalar = {expected:e}"
            );
        }
    }

    #[test]
    fn sqrt_f32_lane_and_tail_boundaries() {
        let cases = [
            0.0,
            -0.0,
            f32::from_bits(1),
            f32::from_bits(2),
            f32::from_bits(0x007f_ffff),
            f32::MIN_POSITIVE,
            f32::from_bits(0x0080_0001),
            0.25,
            1.0,
            2.0,
            4.0,
            9.0,
            f32::MAX,
            f32::INFINITY,
            -1.0,
            -f32::from_bits(1),
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fa1_2345),
        ];
        for offset in 0..cases.len() {
            let inputs: [f32; 3 * W + 1] =
                core::array::from_fn(|lane| cases[(lane + offset) % cases.len()]);
            // Every case appears in each SIMD lane and in each scalar tail
            // length, including empty input and a final partial register.
            for length in 0..=inputs.len() {
                let mut outputs = [f32::NAN; 3 * W + 1];
                sqrt_f32(&inputs[..length], &mut outputs[..length]);
                for lane in 0..length {
                    assert_sqrt_f32(inputs[lane], outputs[lane]);
                }
            }
        }
        let mut exact = [0.0; W];
        sqrt_f32(&[1.0; W], &mut exact);
        assert_eq!(exact, [1.0; W]);
    }

    #[test]
    fn sqrt_f64_lane_and_tail_boundaries() {
        let cases = [
            0.0,
            -0.0,
            f64::from_bits(1),
            f64::from_bits(2),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MIN_POSITIVE,
            f64::from_bits(0x0010_0000_0000_0001),
            0.25,
            1.0,
            2.0,
            4.0,
            9.0,
            f64::MAX,
            f64::INFINITY,
            -1.0,
            -f64::from_bits(1),
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff1_2345_6789_abcd),
        ];
        for offset in 0..cases.len() {
            let inputs: [f64; 3 * WD + 1] =
                core::array::from_fn(|lane| cases[(lane + offset) % cases.len()]);
            for length in 0..=inputs.len() {
                let mut outputs = [f64::NAN; 3 * WD + 1];
                sqrt_f64(&inputs[..length], &mut outputs[..length]);
                for lane in 0..length {
                    assert_sqrt_f64(inputs[lane], outputs[lane]);
                }
            }
        }
        let mut exact = [0.0; WD];
        sqrt_f64(&[1.0; WD], &mut exact);
        assert_eq!(exact, [1.0; WD]);
    }

    #[test]
    fn sqrt_f32_covers_normal_exponents_and_subnormal_mantissas() {
        let mantissas = [0, 1, 0x003f_ffff, 0x007f_fffe, 0x007f_ffff];
        for exponent in 0..255_u32 {
            for offset in 0..mantissas.len() {
                let inputs: [f32; W] = core::array::from_fn(|lane| {
                    f32::from_bits((exponent << 23) | mantissas[(lane + offset) % mantissas.len()])
                });
                let outputs = sqrt_f32x(Simd::from_array(inputs)).to_array();
                for lane in 0..W {
                    assert_sqrt_f32(inputs[lane], outputs[lane]);
                }
            }
        }
    }

    #[test]
    fn sqrt_f64_covers_normal_exponents_and_subnormal_mantissas() {
        let mantissas = [
            0,
            1,
            0x0007_ffff_ffff_ffff,
            0x000f_ffff_ffff_fffe,
            0x000f_ffff_ffff_ffff,
        ];
        for exponent in 0..2047_u64 {
            for offset in 0..mantissas.len() {
                let inputs: [f64; WD] = core::array::from_fn(|lane| {
                    f64::from_bits((exponent << 52) | mantissas[(lane + offset) % mantissas.len()])
                });
                let outputs = sqrt_f64x(Simd::from_array(inputs)).to_array();
                for lane in 0..WD {
                    assert_sqrt_f64(inputs[lane], outputs[lane]);
                }
            }
        }
    }
}
