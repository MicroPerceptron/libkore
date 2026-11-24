//! Portable `core::simd` 8-bit float kernels — signed FP8 and unsigned UF8.
//!
//! The same widen-then-convert algorithm as the `avx2` tier, expressed in
//! `core::simd`: FP8 bytes are unpacked into F16 bit patterns with integer
//! ops, then `Simd::<f16, W>::cast::<f32>()` performs the widen. On targets
//! with hardware half-precision — AArch64 v8.2+ (`fcvtl`), x86 F16C — that
//! `cast` lowers to a single instruction; elsewhere it is a software
//! conversion. Exact on all 256 encodings (subnormals preserved, the two
//! E4M3 NaN encodings patched), matching the scalar reference and avx2.
//!
//! * **E5M2** shares F16's 5-bit exponent and bias-15, so `byte << 8` is
//!   already an exact F16 pattern.
//! * **E4M3** uses the no-rebias trick — `<<8`, arithmetic `>>1`, mask
//!   `0xBFFF` → F16 = value / 2^8 — and the `× 256` is folded back (into
//!   the dequant multiply, or applied once to the dot's scalar sum).

use core::simd::prelude::*;

/// Exact finite E5M3 RGB expansion for the validated volume consumer.
#[inline(always)]
pub(crate) fn finite_uf8e5m3_rgb(rgb: [u8; 3]) -> [f32; 4] {
    let raw = Simd::<u32, 4>::from_array([rgb[0] as u32, rgb[1] as u32, rgb[2] as u32, 0]);
    let normal = Simd::<f32, 4>::from_bits((raw << Simd::splat(20)) + Simd::splat(112 << 23));
    let subnormal = raw.cast::<f32>() * Simd::splat(0.000_007_629_394_5);
    raw.simd_lt(Simd::splat(8))
        .select(subnormal, normal)
        .to_array()
}

use super::W;

/// `× 2^8`, undoing the E4M3 trick's `/ 256`.
pub(super) const E4M3_SCALE: f32 = 256.0;

/// `× 2^8`, undoing the unsigned E4M4 `/256` relayout.
const UF8E4M4_SCALE: f32 = 256.0;

/// Double-width lane count — the working chunk of every kernel here:
/// the FP8→F16 integer unpack runs at `u16×2W` — a full 128-bit NEON op
/// per step instead of a half-filled one — and the F16→F32 widen lowers
/// to the paired `fcvtl`/`fcvtl2` (`vcvtph2ps` ×2 on x86).
const W2: usize = 2 * W;

/// Widen `N` E5M2 bytes at `w[off..]` to f32 (exact).
#[inline(always)]
pub(super) fn widen_e5m2_n<const N: usize>(w: &[u8], off: usize) -> Simd<f32, N> {
    let bytes: Simd<u16, N> = Simd::<u8, N>::from_slice(&w[off..off + N]).cast();
    let f16bits = bytes << Simd::splat(8);
    Simd::<f16, N>::from_bits(f16bits).cast()
}

/// Widen `N` E4M3 bytes at `w[off..]` to f32 scaled by `1/256`
/// (NaN encodings → NaN; the caller applies `× 256`).
#[inline(always)]
pub(super) fn widen_e4m3_over256_n<const N: usize>(w: &[u8], off: usize) -> Simd<f32, N> {
    let bytes: Simd<u16, N> = Simd::<u8, N>::from_slice(&w[off..off + N]).cast();
    let raw = bytes << Simd::splat(8);
    let shifted: Simd<u16, N> = (raw.cast::<i16>() >> Simd::splat(1)).cast();
    let f16bits = shifted & Simd::splat(0xBFFFu16);
    // Exact NaN: the sole NaN pattern (bits & 0x3F80 == 0x3F80) → F16 NaN.
    let m = Simd::splat(0x3F80u16);
    let is_nan = (f16bits & m).simd_eq(m);
    let f16bits = is_nan.select(f16bits | Simd::splat(0x4000u16), f16bits);
    Simd::<f16, N>::from_bits(f16bits).cast()
}

/// Widen `N` unsigned E5M3 bytes to f32 (exact). E5M3 has the same
/// exponent width and bias as F16; shifting by seven only moves its
/// three fraction bits into the F16 fraction field.
#[inline(always)]
fn widen_uf8e5m3_n<const N: usize>(w: &[u8], off: usize) -> Simd<f32, N> {
    let bytes: Simd<u16, N> = Simd::<u8, N>::from_slice(&w[off..off + N]).cast();
    let special = (bytes & Simd::splat(0x00F8u16)).simd_eq(Simd::splat(0x00F8u16));
    // Do not feed exponent-all-ones lanes to a hardware F16 converter: Arm
    // AHP can reinterpret them and signaling NaNs can raise Invalid. Half
    // subnormals are safe here: widening conversions handle them exactly and
    // do not apply half-precision flush controls.
    let normal_bits = special.select(Simd::splat(0), bytes << Simd::splat(7));
    let converted: Simd<f32, N> = Simd::<f16, N>::from_bits(normal_bits).cast();
    let mant = (bytes & Simd::splat(0x0007u16)).cast::<u32>();
    let special_value =
        Simd::<f32, N>::from_bits(Simd::splat(0x7F80_0000) | (mant << Simd::splat(20)));
    special.select(special_value, converted)
}

/// Widen `N` unsigned E4M4 bytes to f32 scaled by `1/256`.
///
/// `byte << 6` is an F16 pattern with the same exponent/fraction fields,
/// but bias 15 instead of 7. Exponent-all-ones encodings are kept out of
/// the F16 converter and reconstructed after the caller applies `× 256`.
#[inline(always)]
fn widen_uf8e4m4_over256_n<const N: usize>(w: &[u8], off: usize) -> Simd<f32, N> {
    let bytes: Simd<u16, N> = Simd::<u8, N>::from_slice(&w[off..off + N]).cast();
    let shifted = bytes << Simd::splat(6);
    let special = (bytes & Simd::splat(0x00F0u16)).simd_eq(Simd::splat(0x00F0u16));
    let normal_bits = special.select(Simd::splat(0), shifted);
    Simd::<f16, N>::from_bits(normal_bits).cast()
}

#[inline(always)]
fn patch_uf8e4m4_special_n<const N: usize>(
    w: &[u8],
    off: usize,
    normal: Simd<f32, N>,
) -> Simd<f32, N> {
    let bytes: Simd<u16, N> = Simd::<u8, N>::from_slice(&w[off..off + N]).cast();
    let exponent = bytes & Simd::splat(0x00F0u16);
    let special = exponent.simd_eq(Simd::splat(0x00F0u16));
    let mant = (bytes & Simd::splat(0x000Fu16)).cast::<u32>();
    let special_value =
        Simd::<f32, N>::from_bits(Simd::splat(0x7F80_0000) | (mant << Simd::splat(19)));
    special.select(special_value, normal)
}

/// Exact unsigned E4M4 → IEEE F16 bit relayout, including normalization
/// of the fifteen UF subnormal encodings. Integer-only, so it is independent
/// of Arm FPCR.FZ16/AHP and does not spend an F16/F32 conversion round trip.
#[inline(always)]
fn pack_uf8e4m4_f16bits_n<const N: usize>(src: &[u8], off: usize) -> Simd<u16, N> {
    let bytes: Simd<u16, N> = Simd::<u8, N>::from_slice(&src[off..off + N]).cast();
    let exp = bytes & Simd::splat(0x00F0);
    let mant = bytes & Simd::splat(0x000F);
    let base = bytes << Simd::splat(6);
    let normal = base + Simd::splat(0x2000);
    let special = base + Simd::splat(0x4000);

    let sub1 = Simd::splat(0x1400);
    let sub2 = Simd::splat(0x1800) | ((mant & Simd::splat(1)) << Simd::splat(9));
    let sub4 = Simd::splat(0x1C00) | ((mant & Simd::splat(3)) << Simd::splat(8));
    let sub8 = Simd::splat(0x2000) | ((mant & Simd::splat(7)) << Simd::splat(7));
    let subnormal = mant.simd_ge(Simd::splat(8)).select(
        sub8,
        mant.simd_ge(Simd::splat(4)).select(
            sub4,
            mant.simd_ge(Simd::splat(2)).select(
                sub2,
                mant.simd_ge(Simd::splat(1)).select(sub1, Simd::splat(0)),
            ),
        ),
    );
    exp.simd_eq(Simd::splat(0)).select(
        subnormal,
        exp.simd_eq(Simd::splat(0x00F0)).select(special, normal),
    )
}

/// Widen unsigned E5M11 words to f32 without passing through F16 (which
/// would discard the format's eleventh fraction bit).
#[inline(always)]
fn widen_uf16e5m11_n<const N: usize>(w: &[u16], off: usize) -> Simd<f32, N> {
    let raw = Simd::<u16, N>::from_slice(&w[off..off + N]).cast::<u32>();
    decode_uf16e5m11_n(raw)
}

#[inline(always)]
fn decode_uf16e5m11_n<const N: usize>(raw: Simd<u32, N>) -> Simd<f32, N> {
    let exp = (raw >> Simd::splat(11)) & Simd::splat(0x1F);
    let mant = raw & Simd::splat(0x07FF);
    let normal_bits =
        ((exp + Simd::splat(127 - 15)) << Simd::splat(23)) | (mant << Simd::splat(12));
    let normal = Simd::<f32, N>::from_bits(normal_bits);
    let subnormal = mant.cast::<f32>() * Simd::splat(0.000_000_029_802_322); // 2^-25
    let special = Simd::<f32, N>::from_bits(Simd::splat(0x7F80_0000) | (mant << Simd::splat(12)));
    let finite = exp.simd_eq(Simd::splat(0)).select(subnormal, normal);
    exp.simd_eq(Simd::splat(0x1F)).select(special, finite)
}

/// Chunks handled per iteration in the unrolled elementwise/reduction
/// loops — enough independent accumulator chains to cover FMA latency
/// (~4 cycles against 2–4 FMA pipes on the targets we care about).
const UNROLL: usize = 4;

// ── Horizontal sum: explicit halving tree ──────────────────────────
//
// `Simd::reduce_sum` on floats is the *strict-ordered* reduction
// (`simd_reduce_add_ordered`) — N−1 dependent scalar adds, a ~3·(N−1)
// cycle latency chain that no ISA can tree. These kernels own their
// summation order, so reduce by explicit halving: log2(N) deep, and the
// final step lowers to the paired-add forms (`faddp` / `haddps`-style).

#[inline(always)]
fn hsum4(v: Simd<f32, 4>) -> f32 {
    let s = v + simd_swizzle!(v, [2, 3, 0, 1]);
    s[0] + s[1]
}

#[inline(always)]
fn hsum8(v: Simd<f32, 8>) -> f32 {
    let a = v.to_array();
    hsum4(Simd::from_slice(&a[..4]) + Simd::from_slice(&a[4..8]))
}

/// Tree horizontal sum of an N-lane f32 vector (N ∈ {4, 8, 16}).
#[inline(always)]
pub(super) fn hsum<const N: usize>(v: Simd<f32, N>) -> f32 {
    let a = v.to_array();
    if N == 16 {
        let lo = Simd::<f32, 8>::from_slice(&a[..8]);
        let hi = Simd::<f32, 8>::from_slice(&a[8..16]);
        hsum8(lo + hi)
    } else if N == 8 {
        hsum8(Simd::<f32, 8>::from_slice(&a[..8]))
    } else if N == 4 {
        hsum4(Simd::<f32, 4>::from_slice(&a[..4]))
    } else {
        a[..N].iter().sum()
    }
}

/// Elementwise map: fill `d[0..n]` from an N-wide `chunk` + a scalar
/// `tail`. The chunk loop is unrolled ×`UNROLL` — stores have no
/// dependency chain, but the unroll amortizes loop and address overhead.
#[inline(always)]
fn map_simd<const N: usize>(
    n: usize,
    d: &mut [f32],
    mut chunk: impl FnMut(usize) -> Simd<f32, N>,
    mut tail: impl FnMut(usize) -> f32,
) {
    let chunks = n / N;
    let blocks = chunks / UNROLL;
    for blk in 0..blocks {
        for u in 0..UNROLL {
            let o = (blk * UNROLL + u) * N;
            chunk(o).copy_to_slice(&mut d[o..]);
        }
    }
    for c in (blocks * UNROLL)..chunks {
        chunk(c * N).copy_to_slice(&mut d[c * N..]);
    }
    for i in (chunks * N)..n {
        d[i] = tail(i);
    }
}

/// Reduction: `Σ` of an N-wide `chunk` + a scalar `tail`, accumulated
/// through `UNROLL` independent chains so back-to-back adds don't
/// serialize on FP latency.
#[inline(always)]
fn reduce_simd<const N: usize>(
    n: usize,
    mut chunk: impl FnMut(usize) -> Simd<f32, N>,
    mut tail: impl FnMut(usize) -> f32,
) -> f32 {
    let chunks = n / N;
    let blocks = chunks / UNROLL;
    let mut acc = [Simd::<f32, N>::splat(0.0); UNROLL];
    for blk in 0..blocks {
        for u in 0..UNROLL {
            acc[u] += chunk((blk * UNROLL + u) * N);
        }
    }
    for c in (blocks * UNROLL)..chunks {
        acc[0] += chunk(c * N);
    }
    let mut sum = hsum((acc[0] + acc[1]) + (acc[2] + acc[3]));
    for i in (chunks * N)..n {
        sum += tail(i);
    }
    sum
}

/// Widen `W2` E5M2 bytes at `w[off..]` to f32 (exact).
#[inline(always)]
fn widen_e5m2_x2(w: &[u8], off: usize) -> Simd<f32, W2> {
    widen_e5m2_n::<W2>(w, off)
}

/// Widen `W2` E4M3 bytes at `w[off..]` to f32 scaled by `1/256`.
#[inline(always)]
fn widen_e4m3_over256_x2(w: &[u8], off: usize) -> Simd<f32, W2> {
    widen_e4m3_over256_n::<W2>(w, off)
}

#[inline(always)]
fn widen_uf8e5m3_x2(w: &[u8], off: usize) -> Simd<f32, W2> {
    widen_uf8e5m3_n::<W2>(w, off)
}

#[inline(always)]
fn widen_uf8e4m4_over256_x2(w: &[u8], off: usize) -> Simd<f32, W2> {
    widen_uf8e4m4_over256_n::<W2>(w, off)
}

// ── Dequantize: FP8 → F32 ──────────────────────────────────────────

pub(crate) fn e5m2_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n);
    map_simd(
        n,
        dst,
        |o| widen_e5m2_x2(src, o),
        |i| crate::cast::e5m2_to_f32(src[i]),
    );
}

pub(crate) fn e4m3_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n);
    let scale = Simd::<f32, W2>::splat(E4M3_SCALE);
    map_simd(
        n,
        dst,
        |o| widen_e4m3_over256_x2(src, o) * scale,
        |i| crate::cast::e4m3_to_f32(src[i]),
    );
}

/// Decode unsigned E5M3 storage to f32. Finite values and infinity are exact;
/// NaNs retain their classification.
pub(crate) fn uf8e5m3_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n);
    map_simd(
        n,
        dst,
        |o| widen_uf8e5m3_x2(src, o),
        |i| crate::cast::uf8e5m3_to_f32(src[i]),
    );
}

/// Decode unsigned E4M4 storage to f32. Finite values and infinity are exact;
/// NaNs retain their classification.
pub(crate) fn uf8e4m4_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n);
    let scale = Simd::<f32, W2>::splat(UF8E4M4_SCALE);
    map_simd(
        n,
        dst,
        |o| patch_uf8e4m4_special_n(src, o, widen_uf8e4m4_over256_x2(src, o) * scale),
        |i| crate::cast::uf8e4m4_to_f32(src[i]),
    );
}

/// Promote unsigned E5M11 storage to f32 exactly. This is the lawful device
/// boundary for UF16; narrowing it to F16 would throw away the extra bit.
pub(crate) fn uf16e5m11_to_f32(src: &[u16], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n);
    let chunks = n / W2;
    for c in 0..chunks {
        let off = c * W2;
        widen_uf16e5m11_n::<W2>(src, off).copy_to_slice(&mut dst[off..off + W2]);
    }
    for i in chunks * W2..n {
        dst[i] = crate::cast::uf16e5m11_to_f32(src[i]);
    }
}

/// Packed-little-endian counterpart for serialized assets and device upload
/// buffers. The fixed-size array assembly folds into unaligned vector loads
/// while retaining correct behavior on big-endian hosts.
pub(crate) fn uf16e5m11_le_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n * 2 && dst.len() >= n);
    let chunks = n / W2;
    for c in 0..chunks {
        let off = c * W2;
        let bytes = &src[off * 2..off * 2 + W2 * 2];
        let raw = Simd::<u16, W2>::from_array(core::array::from_fn(|lane| {
            u16::from_le_bytes([bytes[lane * 2], bytes[lane * 2 + 1]])
        }))
        .cast::<u32>();
        decode_uf16e5m11_n(raw).copy_to_slice(&mut dst[off..off + W2]);
    }
    for i in chunks * W2..n {
        let off = i * 2;
        dst[i] = crate::cast::uf16e5m11_to_f32(u16::from_le_bytes([src[off], src[off + 1]]));
    }
}

// ── Dequantize: FP8 → F16 (native half output, packed LE bytes) ────
//
// On targets with hardware FP16 these never touch F32 at all: E5M2 is
// a pure integer relayout (`byte << 8` *is* the F16 pattern), and E4M3
// is the integer unpack plus one exact F16 `× 256` (a power-of-two
// scale whose results — 2⁻⁹ … 448 — are all normal F16 values, so the
// multiply is exact even where `core::simd` softens it).

/// Map `n` FP8 bytes to F16 bit chunks, ×`UNROLL` like `map_simd`.
#[inline(always)]
fn map_f16_bits(
    n: usize,
    dst: &mut [u8],
    mut chunk: impl FnMut(usize) -> Simd<u16, W2>,
    mut tail: impl FnMut(usize) -> u16,
) {
    let chunks = n / W2;
    // Iterator-sliced stores keep the loop free of per-chunk bounds
    // checks, letting LLVM fuse across the unroll.
    for (c, out) in dst[..chunks * W2 * 2].chunks_exact_mut(W2 * 2).enumerate() {
        chunk(c * W2).to_le_bytes().copy_to_slice(out);
    }
    for i in (chunks * W2)..n {
        dst[i * 2..i * 2 + 2].copy_from_slice(&tail(i).to_le_bytes());
    }
}

/// Dequantize `n` E5M2 bytes into packed little-endian F16.
pub(crate) fn e5m2_to_f16(src: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n * 2);
    map_f16_bits(
        n,
        dst,
        |o| {
            let bytes: Simd<u16, W2> = Simd::<u8, W2>::from_slice(&src[o..o + W2]).cast();
            bytes << Simd::splat(8)
        },
        |i| (src[i] as u16) << 8,
    );
}

/// Dequantize `n` E4M3 bytes into packed little-endian F16 (exact on
/// all 256 encodings; NaNs patched).
pub(crate) fn e4m3_to_f16(src: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n * 2);
    let scale = Simd::<f16, W2>::splat(256.0);
    map_f16_bits(
        n,
        dst,
        |o| (Simd::<f16, W2>::from_bits(unpack_e4m3_f16bits_x2(src, o)) * scale).to_bits(),
        |i| ((crate::cast::e4m3_to_f32(src[i])) as f16).to_bits(),
    );
}

/// Decode unsigned E5M3 into packed little-endian F16. The conversion is a
/// pure integer relayout and preserves every UF8 value exactly.
pub(crate) fn uf8e5m3_to_f16(src: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n * 2);
    map_f16_bits(
        n,
        dst,
        |o| {
            let bytes: Simd<u16, W2> = Simd::<u8, W2>::from_slice(&src[o..o + W2]).cast();
            bytes << Simd::splat(7)
        },
        |i| (src[i] as u16) << 7,
    );
}

/// Decode unsigned E4M4 into packed little-endian F16. All finite values are
/// exactly representable; infinity and NaNs retain their classification.
pub(crate) fn uf8e4m4_to_f16(src: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n * 2);
    map_f16_bits(
        n,
        dst,
        |o| pack_uf8e4m4_f16bits_n::<W2>(src, o),
        |i| crate::cast::uf8e4m4_to_f16_bits(src[i]),
    );
}

/// The E4M3 integer unpack, stopping at F16 bits in the `/256` domain.
#[inline(always)]
fn unpack_e4m3_f16bits_x2(w: &[u8], off: usize) -> Simd<u16, W2> {
    let bytes: Simd<u16, W2> = Simd::<u8, W2>::from_slice(&w[off..off + W2]).cast();
    let raw = bytes << Simd::splat(8);
    let shifted: Simd<u16, W2> = (raw.cast::<i16>() >> Simd::splat(1)).cast();
    let f16bits = shifted & Simd::splat(0xBFFFu16);
    let m = Simd::splat(0x3F80u16);
    let is_nan = (f16bits & m).simd_eq(m);
    is_nan.select(f16bits | Simd::splat(0x4000u16), f16bits)
}

// ── Fused dot: Σ decode(w[i]) * x[i] ───────────────────────────────
//
// A per-tensor quantization scale factors out of the sum, so the kernel
// itself is scale-free: the caller multiplies the scalar result.
// `W2` chunks halve the per-byte widen cost, and `UNROLL` independent
// accumulators keep the FMA pipes fed; the original single-chain W-wide
// form was pinned by FMA latency (one W-chunk per ~4 cycles).

macro_rules! vec_dot_kernel {
    ($name:ident, $widen:path, $scale:expr, $decode:path) => {
        pub(crate) fn $name(w: &[u8], x: &[f32], n: usize) -> f32 {
            debug_assert!(w.len() >= n);
            debug_assert!(x.len() >= n);
            let chunks = n / W2;
            let blocks = chunks / UNROLL;
            let mut acc = [Simd::<f32, W2>::splat(0.0); UNROLL];
            for blk in 0..blocks {
                for u in 0..UNROLL {
                    let o = (blk * UNROLL + u) * W2;
                    let xf = Simd::<f32, W2>::from_slice(&x[o..o + W2]);
                    acc[u] = $widen(w, o) * xf + acc[u];
                }
            }
            for c in (blocks * UNROLL)..chunks {
                let o = c * W2;
                let xf = Simd::<f32, W2>::from_slice(&x[o..o + W2]);
                acc[0] = $widen(w, o) * xf + acc[0];
            }
            // One scale multiply on the reduced sum (E4M3's ×256 fold);
            // the scalar tail decodes exactly and needs none.
            let mut sum = hsum((acc[0] + acc[1]) + (acc[2] + acc[3])) * $scale;
            for i in (chunks * W2)..n {
                sum += $decode(w[i]) * x[i];
            }
            sum
        }
    };
}

vec_dot_kernel!(
    vec_dot_e5m2_f32,
    widen_e5m2_x2,
    1.0f32,
    crate::cast::e5m2_to_f32
);
vec_dot_kernel!(
    vec_dot_e4m3_f32,
    widen_e4m3_over256_x2,
    E4M3_SCALE,
    crate::cast::e4m3_to_f32
);

// ── GEMV / GEMM: [m,k] FP8 weights × [n,k] f32 activations → [m,n] ──
//
// C[i,j] += Σ_k decode(w[i,k])·x[j,k].  The FP8 widen is the unit of
// cost, so the weight block is dequantized to an f32 scratch *once* per
// `BLK_MR` rows — the widen is paid O(m·k) total — and the inner loop is
// pure f32 FMA traffic: stream a pair of activation rows, reuse each
// scratch load across both columns and each activation load across all
// BLK_MR rows. k is swept in BLK_KB blocks so the scratch stays a fixed
// 32 KiB of stack (L1-resident) and any k is handled, with C
// accumulating across k-blocks. This replaced a register-tiled
// widen-in-loop form: paying the widen once beat re-widening the tile
// rows every column sweep, on top of cutting the inner loop to loads +
// FMAs. Both tiers share the kernel (F16C / `fcvtl` dequant, AVX / NEON
// FMAs); edge rows and columns fall back to `vec_dot_*` on the raw FP8
// row, whose UNROLL-chain association differs only in FP rounding.

/// Weight rows dequantized per block (mixed FP8×f32 GEMM).
const BLK_MR: usize = 4;
/// k-block width; the f32 weight block is `BLK_MR × BLK_KB` = 32 KiB.
const BLK_KB: usize = 2048;

/// A rows dequantized per block for the FP8×FP8 bias GEMM — taller than
/// the mixed GEMM's block because every extra row amortizes the dominant
/// cost, the per-chunk B widen, further. KB shrinks to hold the scratch
/// at `BIAS_MR × BIAS_KB` = 32 KiB.
const BIAS_MR: usize = 8;
const BIAS_KB: usize = 1024;

macro_rules! gemm_mixed_kernel {
    ($name:ident, $dequant:path, $vec_dot:path) => {
        /// `c[i,j] += Σ_k decode(w[i,k]) * x[j,k]`, accumulating into `c`.
        pub(crate) fn $name(w: &[u8], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
            debug_assert!(w.len() >= m * k && x.len() >= n * k && c.len() >= m * n);
            let mut aw = [0.0f32; BLK_MR * BLK_KB];
            let m_blocks = m / BLK_MR;
            let n_pairs = n / 2;
            for k0 in (0..k).step_by(BLK_KB) {
                let kb = BLK_KB.min(k - k0);
                let chunks = kb / W2;
                let tail = chunks * W2;
                for blk in 0..m_blocks {
                    let i0 = blk * BLK_MR;
                    for r in 0..BLK_MR {
                        $dequant(
                            &w[(i0 + r) * k + k0..][..kb],
                            &mut aw[r * BLK_KB..][..kb],
                            kb,
                        );
                    }
                    // Column pairs: one scratch load feeds two FMAs.
                    for jp in 0..n_pairs {
                        let j0 = jp * 2;
                        let x0 = &x[j0 * k + k0..][..kb];
                        let x1 = &x[(j0 + 1) * k + k0..][..kb];
                        let mut acc = [[Simd::<f32, W2>::splat(0.0); 2]; BLK_MR];
                        for ck in 0..chunks {
                            let o = ck * W2;
                            let xv0 = Simd::<f32, W2>::from_slice(&x0[o..o + W2]);
                            let xv1 = Simd::<f32, W2>::from_slice(&x1[o..o + W2]);
                            for r in 0..BLK_MR {
                                let av = Simd::<f32, W2>::from_slice(&aw[r * BLK_KB + o..][..W2]);
                                acc[r][0] = av * xv0 + acc[r][0];
                                acc[r][1] = av * xv1 + acc[r][1];
                            }
                        }
                        for r in 0..BLK_MR {
                            let (mut s0, mut s1) = (hsum(acc[r][0]), hsum(acc[r][1]));
                            for t in tail..kb {
                                s0 += aw[r * BLK_KB + t] * x0[t];
                                s1 += aw[r * BLK_KB + t] * x1[t];
                            }
                            c[(i0 + r) * n + j0] += s0;
                            c[(i0 + r) * n + j0 + 1] += s1;
                        }
                    }
                    // Odd trailing column.
                    for j in (n_pairs * 2)..n {
                        let xrow = &x[j * k + k0..][..kb];
                        let mut acc = [Simd::<f32, W2>::splat(0.0); BLK_MR];
                        for ck in 0..chunks {
                            let o = ck * W2;
                            let xv = Simd::<f32, W2>::from_slice(&xrow[o..o + W2]);
                            for r in 0..BLK_MR {
                                let av = Simd::<f32, W2>::from_slice(&aw[r * BLK_KB + o..][..W2]);
                                acc[r] = av * xv + acc[r];
                            }
                        }
                        for r in 0..BLK_MR {
                            let mut s = hsum(acc[r]);
                            for t in tail..kb {
                                s += aw[r * BLK_KB + t] * xrow[t];
                            }
                            c[(i0 + r) * n + j] += s;
                        }
                    }
                }
                // Rows past the last full BLK_MR block: fused dot on the
                // raw FP8 row (widen-per-row, ≤ BLK_MR − 1 rows).
                for i in (m_blocks * BLK_MR)..m {
                    for j in 0..n {
                        c[i * n + j] +=
                            $vec_dot(&w[i * k + k0..][..kb], &x[j * k + k0..][..kb], kb);
                    }
                }
            }
        }
    };
}

gemm_mixed_kernel!(gemm_e4m3_f32, e4m3_to_f32, vec_dot_e4m3_f32);
gemm_mixed_kernel!(gemm_e5m2_f32, e5m2_to_f32, vec_dot_e5m2_f32);

// ══════════════════════════════════════════════════════════════════
//  Mixed-precision MAC: FP8 · FP8 + {F16, BF16} → F32
// ══════════════════════════════════════════════════════════════════
//
// The tensor-core primitive in rendering terms: cheap FP8 operands, a
// mid-precision addend, a precise F32 result. Every operand widens to f32
// through the same hardware path (`fcvtl` / F16C), then one FMA. The
// FP8×FP8 product's ~8-bit significand is lossless into F16/BF16; the F32
// output keeps the sum exact.

/// Widen `W2` E4M3 bytes to their exact f32 values (folds in the ×256).
#[inline(always)]
fn widen_e4m3_x2(w: &[u8], off: usize) -> Simd<f32, W2> {
    widen_e4m3_over256_x2(w, off) * Simd::splat(E4M3_SCALE)
}

/// Read `2·W2` bytes at `c[2·off..]` as `W2` little-endian u16 lanes.
/// The fixed-size array hop drops per-byte bounds checks, so the lane
/// assembly folds into a single unaligned load on little-endian targets.
#[inline(always)]
fn load_u16_x2(c: &[u8], off: usize) -> Simd<u16, W2> {
    let s: &[u8; 2 * W2] = c[off * 2..off * 2 + 2 * W2].try_into().unwrap();
    Simd::from_array(core::array::from_fn(|l| {
        u16::from_le_bytes([s[l * 2], s[l * 2 + 1]])
    }))
}

/// Widen `W2` F16 elements (2 bytes each) at `c[2·off..]` to f32.
#[inline(always)]
fn widen_f16_x2(c: &[u8], off: usize) -> Simd<f32, W2> {
    Simd::<f16, W2>::from_bits(load_u16_x2(c, off)).cast()
}

/// Widen `W2` BF16 elements (2 bytes each) at `c[2·off..]` to f32
/// (BF16 is the top 16 bits of f32).
#[inline(always)]
fn widen_bf16_x2(c: &[u8], off: usize) -> Simd<f32, W2> {
    let bits: Simd<u32, W2> = load_u16_x2(c, off).cast::<u32>() << Simd::splat(16);
    Simd::<f32, W2>::from_bits(bits)
}

// ── Elementwise MAD: d[i] = decode(a[i])·decode(b[i]) + widen(c[i]) ──

macro_rules! mad_kernel {
    ($name:ident, $wa:ident, $wb:ident, $wc:ident, $sa:path, $sb:path, $sc:path) => {
        pub(crate) fn $name(a: &[u8], b: &[u8], c: &[u8], d: &mut [f32], n: usize) {
            debug_assert!(a.len() >= n && b.len() >= n && d.len() >= n);
            map_simd(
                n,
                d,
                |o| $wa(a, o) * $wb(b, o) + $wc(c, o),
                |i| $sa(a[i]) * $sb(b[i]) + $sc(&c[i * 2..]),
            );
        }
    };
}

mad_kernel!(
    mad_e4m3_e4m3_f16,
    widen_e4m3_x2,
    widen_e4m3_x2,
    widen_f16_x2,
    crate::cast::e4m3_to_f32,
    crate::cast::e4m3_to_f32,
    crate::cast::read_f16
);
mad_kernel!(
    mad_e4m3_e4m3_bf16,
    widen_e4m3_x2,
    widen_e4m3_x2,
    widen_bf16_x2,
    crate::cast::e4m3_to_f32,
    crate::cast::e4m3_to_f32,
    crate::cast::read_bf16
);
mad_kernel!(
    mad_e5m2_e5m2_f16,
    widen_e5m2_x2,
    widen_e5m2_x2,
    widen_f16_x2,
    crate::cast::e5m2_to_f32,
    crate::cast::e5m2_to_f32,
    crate::cast::read_f16
);
mad_kernel!(
    mad_e5m2_e5m2_bf16,
    widen_e5m2_x2,
    widen_e5m2_x2,
    widen_bf16_x2,
    crate::cast::e5m2_to_f32,
    crate::cast::e5m2_to_f32,
    crate::cast::read_bf16
);

// ── Elementwise multiply: d[i] = decode(a[i]) · b[i]  (FP8 × F32) ──

pub(crate) fn mul_e4m3_f32(a: &[u8], b: &[f32], d: &mut [f32], n: usize) {
    debug_assert!(a.len() >= n && b.len() >= n && d.len() >= n);
    map_simd(
        n,
        d,
        |o| widen_e4m3_x2(a, o) * Simd::<f32, W2>::from_slice(&b[o..o + W2]),
        |i| crate::cast::e4m3_to_f32(a[i]) * b[i],
    );
}

pub(crate) fn mul_e5m2_f32(a: &[u8], b: &[f32], d: &mut [f32], n: usize) {
    debug_assert!(a.len() >= n && b.len() >= n && d.len() >= n);
    map_simd(
        n,
        d,
        |o| widen_e5m2_x2(a, o) * Simd::<f32, W2>::from_slice(&b[o..o + W2]),
        |i| crate::cast::e5m2_to_f32(a[i]) * b[i],
    );
}

// ── Reduction dot: Σ decode(a) · decode(b)  (FP8 × FP8) ──

pub(crate) fn dot_e4m3_e4m3(a: &[u8], b: &[u8], n: usize) -> f32 {
    debug_assert!(a.len() >= n && b.len() >= n);
    reduce_simd(
        n,
        |o| widen_e4m3_x2(a, o) * widen_e4m3_x2(b, o),
        |i| crate::cast::e4m3_to_f32(a[i]) * crate::cast::e4m3_to_f32(b[i]),
    )
}

pub(crate) fn dot_e5m2_e5m2(a: &[u8], b: &[u8], n: usize) -> f32 {
    debug_assert!(a.len() >= n && b.len() >= n);
    reduce_simd(
        n,
        |o| widen_e5m2_x2(a, o) * widen_e5m2_x2(b, o),
        |i| crate::cast::e5m2_to_f32(a[i]) * crate::cast::e5m2_to_f32(b[i]),
    )
}

// ── GEMM: D[i,j] = Σ_k decode(A[i,k])·decode(B[j,k]) + widen(C[i,j]) ──
// A is [m,k] FP8, B is [n,k] FP8, C is [m,n] F16/BF16 bias, D is [m,n] F32.
//
// Both operands are FP8, so the widen — not the FMA — is the unit of
// cost. Dequantizing a BIAS_MR-row block of A once and reusing each B
// chunk widen across all of its rows pays A's widen O(m·k) and B's
// O((m/BIAS_MR)·n·k) — 4× fewer than the row-at-a-time form that
// re-widened B for every A row. k is swept in BIAS_KB blocks so the f32
// A block stays a fixed 32 KiB of stack (and any k is handled, with D
// accumulating across k-blocks on top of the bias).

macro_rules! gemm_bias_kernel {
    ($name:ident, $dequant:path, $widen:path, $scale:expr, $decode:path,
     $vec_dot:path, $wc:path, $sc:path) => {
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
            // D starts as the widened bias; every k-block accumulates on top.
            map_simd(m * n, d, |o| $wc(c, o), |i| $sc(&c[i * 2..]));

            let mut aw = [0.0f32; BIAS_MR * BIAS_KB];
            let m_blocks = m / BIAS_MR;
            for k0 in (0..k).step_by(BIAS_KB) {
                let kb = BIAS_KB.min(k - k0);
                let chunks = kb / W2;
                let tail = chunks * W2;
                for blk in 0..m_blocks {
                    let i0 = blk * BIAS_MR;
                    for r in 0..BIAS_MR {
                        $dequant(
                            &a[(i0 + r) * k + k0..][..kb],
                            &mut aw[r * BIAS_KB..][..kb],
                            kb,
                        );
                    }
                    for j in 0..n {
                        let brow = &b[j * k + k0..][..kb];
                        let mut acc = [Simd::<f32, W2>::splat(0.0); BIAS_MR];
                        for ck in 0..chunks {
                            let o = ck * W2;
                            let bw = $widen(brow, o);
                            for r in 0..BIAS_MR {
                                let av = Simd::<f32, W2>::from_slice(&aw[r * BIAS_KB + o..][..W2]);
                                acc[r] = bw * av + acc[r];
                            }
                        }
                        for r in 0..BIAS_MR {
                            let mut s = hsum(acc[r]) * $scale;
                            for t in tail..kb {
                                s += $decode(brow[t]) * aw[r * BIAS_KB + t];
                            }
                            d[(i0 + r) * n + j] += s;
                        }
                    }
                }
                // Rows past the last full BIAS_MR block: dequantize the row
                // and run the plain fused dot (B re-widened per row, but
                // only for ≤ BIAS_MR − 1 rows).
                for i in (m_blocks * BIAS_MR)..m {
                    $dequant(&a[i * k + k0..][..kb], &mut aw[..kb], kb);
                    for j in 0..n {
                        d[i * n + j] += $vec_dot(&b[j * k + k0..][..kb], &aw[..kb], kb);
                    }
                }
            }
        }
    };
}

gemm_bias_kernel!(
    gemm_e4m3_e4m3_f16,
    e4m3_to_f32,
    widen_e4m3_over256_x2,
    E4M3_SCALE,
    crate::cast::e4m3_to_f32,
    vec_dot_e4m3_f32,
    widen_f16_x2,
    crate::cast::read_f16
);
gemm_bias_kernel!(
    gemm_e4m3_e4m3_bf16,
    e4m3_to_f32,
    widen_e4m3_over256_x2,
    E4M3_SCALE,
    crate::cast::e4m3_to_f32,
    vec_dot_e4m3_f32,
    widen_bf16_x2,
    crate::cast::read_bf16
);
gemm_bias_kernel!(
    gemm_e5m2_e5m2_f16,
    e5m2_to_f32,
    widen_e5m2_x2,
    1.0f32,
    crate::cast::e5m2_to_f32,
    vec_dot_e5m2_f32,
    widen_f16_x2,
    crate::cast::read_f16
);
gemm_bias_kernel!(
    gemm_e5m2_e5m2_bf16,
    e5m2_to_f32,
    widen_e5m2_x2,
    1.0f32,
    crate::cast::e5m2_to_f32,
    vec_dot_e5m2_f32,
    widen_bf16_x2,
    crate::cast::read_bf16
);

#[cfg(test)]
mod tests {
    use super::*;

    // Known encodings, hand-derived from the OCP FP8 definitions — an
    // independent check on the scalar reference the kernels widen against.
    #[test]
    fn known_values() {
        assert_eq!(crate::cast::e4m3_to_f32(0x38), 1.0);
        assert_eq!(crate::cast::e4m3_to_f32(0x40), 2.0);
        assert_eq!(crate::cast::e4m3_to_f32(0x7E), 448.0);
        assert!(crate::cast::e4m3_to_f32(0x7F).is_nan());
        assert_eq!(crate::cast::e5m2_to_f32(0x3C), 1.0);
        assert!(crate::cast::e5m2_to_f32(0x7C).is_infinite());
    }

    // The f16-widen dequant matches the scalar reference on every encoding
    // — on this AArch64 host that runs against real `fcvtl` hardware.
    #[test]
    fn dequant_all_encodings() {
        let src: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut dst = [0.0f32; 256];

        e4m3_to_f32(&src, &mut dst, 256);
        for b in 0..256 {
            let want = crate::cast::e4m3_to_f32(b as u8);
            if want.is_nan() {
                assert!(dst[b].is_nan(), "e4m3 0x{b:02X}");
            } else {
                assert_eq!(dst[b], want, "e4m3 0x{b:02X}");
            }
        }

        e5m2_to_f32(&src, &mut dst, 256);
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
    fn unsigned_dequant_all_encodings() {
        let src: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut dst = [0.0f32; 256];

        uf8e4m4_to_f32(&src, &mut dst, src.len());
        for raw in 0..256 {
            let want = crate::cast::uf8e4m4_to_f32(raw as u8);
            assert_eq!(dst[raw].to_bits(), want.to_bits(), "UF8 E4M4 0x{raw:02X}");
        }

        uf8e5m3_to_f32(&src, &mut dst, src.len());
        for raw in 0..256 {
            let want = crate::cast::uf8e5m3_to_f32(raw as u8);
            assert_eq!(dst[raw].to_bits(), want.to_bits(), "UF8 E5M3 0x{raw:02X}");
        }
    }

    #[test]
    fn unsigned_f16_promotion_all_encodings() {
        let src: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut dst = [0u8; 512];

        for (promote, decode, name) in [
            (
                uf8e4m4_to_f16 as fn(&[u8], &mut [u8], usize),
                crate::cast::uf8e4m4_to_f32 as fn(u8) -> f32,
                "E4M4",
            ),
            (
                uf8e5m3_to_f16 as fn(&[u8], &mut [u8], usize),
                crate::cast::uf8e5m3_to_f32 as fn(u8) -> f32,
                "E5M3",
            ),
        ] {
            promote(&src, &mut dst, src.len());
            for raw in 0..256 {
                let got = f16::from_bits(u16::from_le_bytes([dst[raw * 2], dst[raw * 2 + 1]]));
                let want = decode(raw as u8);
                if want.is_nan() {
                    assert!(got.is_nan(), "UF8 {name} 0x{raw:02X}");
                } else {
                    assert_eq!(got as f32, want, "UF8 {name} 0x{raw:02X}");
                }
            }
        }
    }

    // The native F16 dequants match the scalar reference on every
    // encoding, bit for bit (NaNs → any F16 NaN).
    #[test]
    fn dequant_f16_all_encodings() {
        let src: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut dst = [0u8; 512];

        e4m3_to_f16(&src, &mut dst, 256);
        for b in 0..256 {
            let got = f16::from_bits(u16::from_le_bytes([dst[b * 2], dst[b * 2 + 1]]));
            let want = crate::cast::e4m3_to_f32(b as u8);
            if want.is_nan() {
                assert!(got.is_nan(), "e4m3 0x{b:02X}");
            } else {
                assert_eq!(got.to_bits(), (want as f16).to_bits(), "e4m3 0x{b:02X}");
            }
        }

        e5m2_to_f16(&src, &mut dst, 256);
        for b in 0..256 {
            let got = f16::from_bits(u16::from_le_bytes([dst[b * 2], dst[b * 2 + 1]]));
            let want = crate::cast::e5m2_to_f32(b as u8);
            if want.is_nan() {
                assert!(got.is_nan(), "e5m2 0x{b:02X}");
            } else {
                assert_eq!(got.to_bits(), (want as f16).to_bits(), "e5m2 0x{b:02X}");
            }
        }
    }

    // Fused dot equals the naive scalar dot (products chosen exact).
    #[test]
    fn dot_matches_scalar() {
        let w = [0x38u8, 0x40, 0xC0, 0x38, 0x40, 0xC0, 0x38, 0x40, 0xC0, 0x38]; // 1,2,-2,…
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let got = vec_dot_e4m3_f32(&w, &x, w.len());
        let mut want = 0.0f32;
        for i in 0..w.len() {
            want += crate::cast::e4m3_to_f32(w[i]) * x[i];
        }
        assert_eq!(got, want);
    }

    // ── Mixed-precision MAC ──

    #[test]
    fn mad_matches_scalar() {
        let a = [0x38u8, 0x40, 0xC0, 0x38, 0x40]; // 1, 2, -2, 1, 2
        let b = [0x40u8, 0x38, 0x40, 0xC0, 0x38]; // 2, 1, 2, -2, 1
        let c: [u8; 10] = [0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C]; // 1.0 ×5 (f16)
        let mut d = [0.0f32; 5];
        mad_e4m3_e4m3_f16(&a, &b, &c, &mut d, 5);
        for i in 0..5 {
            let want = crate::cast::e4m3_to_f32(a[i]) * crate::cast::e4m3_to_f32(b[i]) + 1.0;
            assert_eq!(d[i], want, "lane {i}");
        }
    }

    #[test]
    fn mul_fp8_f32_matches_scalar() {
        let a = [0x38u8, 0x40, 0xC0, 0x38, 0x40];
        let b = [1.5f32, 2.0, -3.0, 4.0, 0.5];
        let mut d = [0.0f32; 5];
        mul_e4m3_f32(&a, &b, &mut d, 5);
        for i in 0..5 {
            assert_eq!(d[i], crate::cast::e4m3_to_f32(a[i]) * b[i], "lane {i}");
        }
    }

    #[test]
    fn dot_fp8_fp8_matches_scalar() {
        let a = [0x38u8, 0x40, 0xC0, 0x38, 0x40, 0xC0, 0x38, 0x40, 0xC0, 0x38];
        let b = [0x40u8, 0x38, 0x40, 0xC0, 0x38, 0x40, 0xC0, 0x38, 0x40, 0xC0];
        let got = dot_e4m3_e4m3(&a, &b, 10);
        let mut want = 0.0f32;
        for i in 0..10 {
            want += crate::cast::e4m3_to_f32(a[i]) * crate::cast::e4m3_to_f32(b[i]);
        }
        assert_eq!(got, want);
    }

    #[test]
    fn gemm_bias_matches_scalar() {
        // 2×2 · 2×2 (k=2), C = all-ones F16 bias.
        let a = [0x38u8, 0x40, 0x40, 0x38]; // [[1,2],[2,1]]
        let b = [0x38u8, 0x38, 0x40, 0x40]; // [[1,1],[2,2]]
        let c: [u8; 8] = [0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C]; // 1.0 ×4
        let mut d = [0.0f32; 4];
        gemm_e4m3_e4m3_f16(&a, &b, &c, &mut d, 2, 2, 2);
        for i in 0..2 {
            for j in 0..2 {
                let mut want = 1.0f32;
                for k in 0..2 {
                    want += crate::cast::e4m3_to_f32(a[i * 2 + k])
                        * crate::cast::e4m3_to_f32(b[j * 2 + k]);
                }
                assert_eq!(d[i * 2 + j], want, "d[{i},{j}]");
            }
        }
    }

    #[test]
    fn gemm_mixed_tiled_matches_scalar() {
        // Dims deliberately off the MR/NR/W tiling so the edge paths run.
        let (m, n, k) = (7usize, 5usize, 37usize);
        // Finite for both formats (|byte| < 0x7C avoids inf/NaN), mixed sign.
        let w: alloc::vec::Vec<u8> = (0..m * k)
            .map(|i| {
                let mag = (i % 100) as u8;
                if i % 2 == 0 { mag } else { 0x80 | mag }
            })
            .collect();
        let x: alloc::vec::Vec<f32> = (0..n * k).map(|i| i as f32 * 0.013 - 0.4).collect();

        let mut got_e4 = alloc::vec![0.0f32; m * n];
        let mut got_e5 = alloc::vec![0.0f32; m * n];
        gemm_e4m3_f32(&w, &x, &mut got_e4, m, n, k);
        gemm_e5m2_f32(&w, &x, &mut got_e5, m, n, k);

        for i in 0..m {
            for j in 0..n {
                let (mut r4, mut r5) = (0.0f32, 0.0f32);
                for kk in 0..k {
                    r4 += crate::cast::e4m3_to_f32(w[i * k + kk]) * x[j * k + kk];
                    r5 += crate::cast::e5m2_to_f32(w[i * k + kk]) * x[j * k + kk];
                }
                let (g4, g5) = (got_e4[i * n + j], got_e5[i * n + j]);
                assert!(
                    (g4 - r4).abs() <= 1e-3 * (1.0 + r4.abs()),
                    "e4m3 [{i},{j}] {g4} vs {r4}"
                );
                assert!(
                    (g5 - r5).abs() <= 1e-3 * (1.0 + r5.abs()),
                    "e5m2 [{i},{j}] {g5} vs {r5}"
                );
            }
        }
    }
}
