//! AVX2 + F16C FP8 (E4M3 / E5M2) kernels — the benchmarked widen path.
//!
//! FP8 bytes are unpacked against zero into F16 bit patterns and widened
//! to f32 with one `vcvtph2ps` (F16C), consumed 8-wide:
//!
//! * **E5M2** shares F16's 5-bit exponent and bias-15, so `byte << 8`
//!   (an `unpacklo` against zero) is already an exact F16 pattern.
//! * **E4M3** needs no rebias: after `<< 8`, an arithmetic `sraw 1`
//!   + `and 0xBFFF` drops the 4-bit exponent into F16's 5-bit field,
//!   giving *exactly value / 2^8*; the `× 256` is folded back (into the
//!   dequant multiply, or applied once to the dot's scalar sum). The two
//!   E4M3 NaN encodings are patched to a real NaN so every tier agrees
//!   on all 256 encodings (the compare/or costs a few ALU ops that hide
//!   under the memory / FMA traffic of these kernels).
//!
//! Only compiled with `feature = "avx2"`; assumes FMA + F16C (true on
//! every real AVX2 CPU). Loads are unaligned — arena storage carries no
//! alignment guarantee. The scalar tail uses the exact `crate::cast`
//! reference, so results are bit-identical to the portable tier.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

/// `× 2^8`, folding out the E4M3 no-rebias trick's `/ 256`.
const E4M3_SCALE: f32 = 256.0;
/// Clears the sign duplicated by the arithmetic `sraw 1`.
const E4M3_CLR: i16 = 0xBFFFu16 as i16;
/// Post-widen mask: `(w & 0x3F80) == 0x3F80` iff the byte was E4M3 NaN.
const E4M3_NANM: i16 = 0x3F80;
/// OR to force the F16 exponent field to all-ones (bit 14; 13..10 set).
const E4M3_NANB: i16 = 0x4000;

#[inline(always)]
unsafe fn hsum256(x: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(x, 1);
    let lo = _mm256_castps256_ps128(x);
    let s = _mm_add_ps(hi, lo);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

/// Widen 8 packed E5M2 bytes at `p` to 8× f32 (exact).
#[inline(always)]
unsafe fn widen_e5m2_x8(p: *const u8) -> __m256 {
    let bytes = _mm_loadl_epi64(p as *const __m128i); // 8 bytes → low 64
    let f16 = _mm_unpacklo_epi8(_mm_setzero_si128(), bytes); // 8× (byte << 8)
    _mm256_cvtph_ps(f16)
}

/// Widen 8 packed E4M3 bytes at `p` to 8× f32 scaled by `1/256`
/// (NaN encodings → NaN; the caller applies the `× 256`).
#[inline(always)]
unsafe fn widen_e4m3_over256_x8(p: *const u8) -> __m256 {
    let bytes = _mm_loadl_epi64(p as *const __m128i);
    let raw = _mm_unpacklo_epi8(_mm_setzero_si128(), bytes); // byte << 8
    let f16 = _mm_and_si128(_mm_srai_epi16(raw, 1), _mm_set1_epi16(E4M3_CLR));
    // Exact NaN: force the sole NaN pattern to an F16 NaN.
    let nm = _mm_set1_epi16(E4M3_NANM);
    let is_nan = _mm_cmpeq_epi16(_mm_and_si128(f16, nm), nm);
    let f16 = _mm_or_si128(f16, _mm_and_si128(is_nan, _mm_set1_epi16(E4M3_NANB)));
    _mm256_cvtph_ps(f16) // = value / 256
}

#[inline(always)]
unsafe fn decode_uf8_half_x8<
    const EXP_MASK: i32,
    const MANT_MASK: i32,
    const MANT_SHIFT: i32,
    const SCALE_E4: bool,
>(
    raw16: __m128i,
    half16: __m128i,
) -> __m256 {
    let exp16 = _mm_and_si128(raw16, _mm_set1_epi16(EXP_MASK as i16));
    let special16 = _mm_cmpeq_epi16(exp16, _mm_set1_epi16(EXP_MASK as i16));

    // VCVTPH2PS widens half subnormals exactly and ignores MXCSR.DAZ. Mask
    // only exponent-all-ones lanes so signaling NaNs cannot raise Invalid;
    // their exact UF infinity/NaN bit patterns are rebuilt after conversion.
    let safe_half16 = _mm_andnot_si128(special16, half16);
    let mut normal = _mm256_cvtph_ps(safe_half16);
    if SCALE_E4 {
        normal = _mm256_mul_ps(normal, _mm256_set1_ps(256.0));
    }

    let raw = _mm256_cvtepu16_epi32(raw16);
    let mant = _mm256_and_si256(raw, _mm256_set1_epi32(MANT_MASK));
    let special = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_set1_epi32(0x7F80_0000),
        _mm256_slli_epi32(mant, MANT_SHIFT),
    ));
    _mm256_blendv_ps(
        normal,
        special,
        _mm256_castsi256_ps(_mm256_cvtepi16_epi32(special16)),
    )
}

#[inline(always)]
unsafe fn widen_uf8e5m3_x16(p: *const u8) -> (__m256, __m256) {
    let bytes = _mm_loadu_si128(p as *const __m128i);
    let lanes = _mm256_cvtepu8_epi16(bytes);
    let half = _mm256_slli_epi16(lanes, 7);
    (
        decode_uf8_half_x8::<0xF8, 0x07, 20, false>(
            _mm256_castsi256_si128(lanes),
            _mm256_castsi256_si128(half),
        ),
        decode_uf8_half_x8::<0xF8, 0x07, 20, false>(
            _mm256_extracti128_si256(lanes, 1),
            _mm256_extracti128_si256(half, 1),
        ),
    )
}

/// Validated finite RGB only: no fourth-byte overread and no expanded buffer.
#[inline(always)]
pub(crate) fn finite_uf8e5m3_rgb(rgb: [u8; 3]) -> [f32; 4] {
    // SAFETY: this module is selected only by the AVX2/FMA/F16C tier.
    // E5M3 shares binary16's exponent/bias, including subnormals.
    unsafe {
        let half = _mm_setr_epi16(
            (rgb[0] as i16) << 7,
            (rgb[1] as i16) << 7,
            (rgb[2] as i16) << 7,
            0,
            0,
            0,
            0,
            0,
        );
        core::mem::transmute::<__m128, [f32; 4]>(_mm_cvtph_ps(half))
    }
}

#[inline(always)]
unsafe fn widen_uf8e4m4_x16(p: *const u8) -> (__m256, __m256) {
    let bytes = _mm_loadu_si128(p as *const __m128i);
    let lanes = _mm256_cvtepu8_epi16(bytes);
    let half = _mm256_slli_epi16(lanes, 6);
    (
        decode_uf8_half_x8::<0xF0, 0x0F, 19, true>(
            _mm256_castsi256_si128(lanes),
            _mm256_castsi256_si128(half),
        ),
        decode_uf8_half_x8::<0xF0, 0x0F, 19, true>(
            _mm256_extracti128_si256(lanes, 1),
            _mm256_extracti128_si256(half, 1),
        ),
    )
}

/// Widen 8 unsigned E5M11 words to f32 exactly, retaining all eleven
/// fraction bits (so this deliberately does not use F16C).
#[inline(always)]
unsafe fn widen_uf16e5m11_x8(p: *const u16) -> __m256 {
    let raw16 = _mm_loadu_si128(p as *const __m128i);
    let raw = _mm256_cvtepu16_epi32(raw16);
    let exp = _mm256_and_si256(_mm256_srli_epi32(raw, 11), _mm256_set1_epi32(0x1F));
    let mant = _mm256_and_si256(raw, _mm256_set1_epi32(0x07FF));
    let normal_bits = _mm256_or_si256(
        _mm256_slli_epi32(_mm256_add_epi32(exp, _mm256_set1_epi32(127 - 15)), 23),
        _mm256_slli_epi32(mant, 12),
    );
    let normal = _mm256_castsi256_ps(normal_bits);
    let subnormal = _mm256_mul_ps(
        _mm256_cvtepi32_ps(mant),
        _mm256_set1_ps(0.000_000_029_802_322),
    );
    let special = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_set1_epi32(0x7F80_0000),
        _mm256_slli_epi32(mant, 12),
    ));
    let finite = _mm256_blendv_ps(
        normal,
        subnormal,
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(exp, _mm256_setzero_si256())),
    );
    _mm256_blendv_ps(
        finite,
        special,
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(exp, _mm256_set1_epi32(0x1F))),
    )
}

// ── Dequantize: FP8 → F32 ──────────────────────────────────────────

pub(crate) fn e5m2_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n);
    unsafe {
        let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
        let chunks = n / 8;
        for i in 0..chunks {
            _mm256_storeu_ps(dp.add(i * 8), widen_e5m2_x8(sp.add(i * 8)));
        }
        for j in (chunks * 8)..n {
            *dp.add(j) = crate::cast::e5m2_to_f32(*sp.add(j));
        }
    }
}

pub(crate) fn e4m3_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n);
    debug_assert!(dst.len() >= n);
    unsafe {
        let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
        let scale = _mm256_set1_ps(E4M3_SCALE);
        let chunks = n / 8;
        for i in 0..chunks {
            let v = _mm256_mul_ps(widen_e4m3_over256_x8(sp.add(i * 8)), scale);
            _mm256_storeu_ps(dp.add(i * 8), v);
        }
        for j in (chunks * 8)..n {
            *dp.add(j) = crate::cast::e4m3_to_f32(*sp.add(j));
        }
    }
}

pub(crate) fn uf8e5m3_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n);
    unsafe {
        let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
        let chunks = n / 16;
        for i in 0..chunks {
            let off = i * 16;
            let (lo, hi) = widen_uf8e5m3_x16(sp.add(off));
            _mm256_storeu_ps(dp.add(off), lo);
            _mm256_storeu_ps(dp.add(off + 8), hi);
        }
        for j in (chunks * 16)..n {
            *dp.add(j) = crate::cast::uf8e5m3_to_f32(*sp.add(j));
        }
    }
}

pub(crate) fn uf8e4m4_to_f32(src: &[u8], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n);
    unsafe {
        let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
        let chunks = n / 16;
        for i in 0..chunks {
            let off = i * 16;
            let (lo, hi) = widen_uf8e4m4_x16(sp.add(off));
            _mm256_storeu_ps(dp.add(off), lo);
            _mm256_storeu_ps(dp.add(off + 8), hi);
        }
        for j in (chunks * 16)..n {
            *dp.add(j) = crate::cast::uf8e4m4_to_f32(*sp.add(j));
        }
    }
}

pub(crate) fn uf16e5m11_to_f32(src: &[u16], dst: &mut [f32], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n);
    unsafe {
        let chunks = n / 8;
        for i in 0..chunks {
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i * 8),
                widen_uf16e5m11_x8(src.as_ptr().add(i * 8)),
            );
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
        for i in 0..chunks {
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i * 8),
                // x86 is little-endian and the intrinsic load is explicitly
                // unaligned, so packed bytes are the exact raw word lanes.
                widen_uf16e5m11_x8(src.as_ptr().add(i * 16) as *const u16),
            );
        }
        for i in chunks * 8..n {
            let off = i * 2;
            let raw = u16::from_le_bytes([*src.get_unchecked(off), *src.get_unchecked(off + 1)]);
            *dst.get_unchecked_mut(i) = crate::cast::uf16e5m11_to_f32(raw);
        }
    }
}

// ── Fused dot: Σ decode(w[i]) * x[i] ───────────────────────────────

pub(crate) fn vec_dot_e5m2_f32(w: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(w.len() >= n);
    debug_assert!(x.len() >= n);
    unsafe {
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let chunks = n / 8;
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let wf = widen_e5m2_x8(wp.add(i * 8));
            let xf = _mm256_loadu_ps(xp.add(i * 8));
            acc = _mm256_fmadd_ps(wf, xf, acc);
        }
        let mut sum = hsum256(acc);
        for j in (chunks * 8)..n {
            sum += crate::cast::e5m2_to_f32(*wp.add(j)) * *xp.add(j);
        }
        sum
    }
}

pub(crate) fn vec_dot_e4m3_f32(w: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(w.len() >= n);
    debug_assert!(x.len() >= n);
    unsafe {
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let chunks = n / 8;
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let wf = widen_e4m3_over256_x8(wp.add(i * 8)); // value / 256
            let xf = _mm256_loadu_ps(xp.add(i * 8));
            acc = _mm256_fmadd_ps(wf, xf, acc);
        }
        // Undo the /256 once on the reduced chunk sum; the scalar tail is
        // already exact.
        let mut sum = hsum256(acc) * E4M3_SCALE;
        for j in (chunks * 8)..n {
            sum += crate::cast::e4m3_to_f32(*wp.add(j)) * *xp.add(j);
        }
        sum
    }
}

// ── GEMV / GEMM: [m,k] FP8 × [n,k] f32 → [m,n] ─────────────────────

// The mixed FP8×f32 GEMM is the shared blocked `core::simd` kernel
// (see `portable::f8`): the weight block is dequantized to an f32
// scratch once — the widen (F16C here) is paid O(m·k) instead of per
// k-step — and the inner loop is pure load/FMA traffic. Its earlier
// shared register-blocked form already measured faster than this tier's
// hand-rolled dot-per-(i,j) (notably ~1.8× on E5M2, pinned by a
// single-accumulator FMA latency chain). One implementation, both tiers.
pub(crate) use crate::arch::portable::f8::{gemm_e4m3_f32, gemm_e5m2_f32};

// Native F16-output dequants are shared `core::simd`: the E4M3 `× 256`
// promotes through F16C on x86 (convert-only ISA), stays pure-half on
// AArch64, and is exact either way.
pub(crate) use crate::arch::portable::f8::{
    e4m3_to_f16, e5m2_to_f16, uf8e4m4_to_f16, uf8e5m3_to_f16,
};

// ══════════════════════════════════════════════════════════════════
//  Mixed-precision MAC: FP8 · FP8 + {F16, BF16} → F32
// ══════════════════════════════════════════════════════════════════

/// Widen 8 packed E4M3 bytes to their exact f32 values (folds in ×256).
#[inline(always)]
unsafe fn widen_e4m3_x8(p: *const u8) -> __m256 {
    _mm256_mul_ps(widen_e4m3_over256_x8(p), _mm256_set1_ps(E4M3_SCALE))
}

/// Widen 8 F16 elements (16 bytes) at `p` to f32 via F16C.
#[inline(always)]
unsafe fn widen_f16_x8(p: *const u8) -> __m256 {
    _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
}

/// Widen 8 BF16 elements (16 bytes) at `p` to f32 (top 16 bits of f32).
#[inline(always)]
unsafe fn widen_bf16_x8(p: *const u8) -> __m256 {
    let bits16 = _mm_loadu_si128(p as *const __m128i);
    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits16), 16))
}

// ── Elementwise MAD: d[i] = decode(a[i])·decode(b[i]) + widen(c[i]) ──

macro_rules! mad_kernel {
    ($name:ident, $wa:ident, $wb:ident, $wc:ident, $sa:path, $sb:path, $sc:path) => {
        pub(crate) fn $name(a: &[u8], b: &[u8], c: &[u8], d: &mut [f32], n: usize) {
            debug_assert!(a.len() >= n && b.len() >= n && c.len() >= n * 2 && d.len() >= n);
            unsafe {
                let chunks = n / 8;
                for i in 0..chunks {
                    let wa = $wa(a.as_ptr().add(i * 8));
                    let wb = $wb(b.as_ptr().add(i * 8));
                    let wc = $wc(c.as_ptr().add(i * 16));
                    _mm256_storeu_ps(d.as_mut_ptr().add(i * 8), _mm256_fmadd_ps(wa, wb, wc));
                }
                for j in (chunks * 8)..n {
                    d[j] = $sa(a[j]) * $sb(b[j]) + $sc(&c[j * 2..]);
                }
            }
        }
    };
}

mad_kernel!(
    mad_e4m3_e4m3_f16,
    widen_e4m3_x8,
    widen_e4m3_x8,
    widen_f16_x8,
    crate::cast::e4m3_to_f32,
    crate::cast::e4m3_to_f32,
    crate::cast::read_f16
);
mad_kernel!(
    mad_e4m3_e4m3_bf16,
    widen_e4m3_x8,
    widen_e4m3_x8,
    widen_bf16_x8,
    crate::cast::e4m3_to_f32,
    crate::cast::e4m3_to_f32,
    crate::cast::read_bf16
);
mad_kernel!(
    mad_e5m2_e5m2_f16,
    widen_e5m2_x8,
    widen_e5m2_x8,
    widen_f16_x8,
    crate::cast::e5m2_to_f32,
    crate::cast::e5m2_to_f32,
    crate::cast::read_f16
);
mad_kernel!(
    mad_e5m2_e5m2_bf16,
    widen_e5m2_x8,
    widen_e5m2_x8,
    widen_bf16_x8,
    crate::cast::e5m2_to_f32,
    crate::cast::e5m2_to_f32,
    crate::cast::read_bf16
);

// ── Elementwise multiply: d[i] = decode(a[i]) · b[i]  (FP8 × F32) ──

macro_rules! mul_kernel {
    ($name:ident, $wa:ident, $sa:path) => {
        pub(crate) fn $name(a: &[u8], b: &[f32], d: &mut [f32], n: usize) {
            debug_assert!(a.len() >= n && b.len() >= n && d.len() >= n);
            unsafe {
                let chunks = n / 8;
                for i in 0..chunks {
                    let wa = $wa(a.as_ptr().add(i * 8));
                    let bf = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                    _mm256_storeu_ps(d.as_mut_ptr().add(i * 8), _mm256_mul_ps(wa, bf));
                }
                for j in (chunks * 8)..n {
                    d[j] = $sa(a[j]) * b[j];
                }
            }
        }
    };
}

mul_kernel!(mul_e4m3_f32, widen_e4m3_x8, crate::cast::e4m3_to_f32);
mul_kernel!(mul_e5m2_f32, widen_e5m2_x8, crate::cast::e5m2_to_f32);

// ── Reduction dot: Σ decode(a) · decode(b)  (FP8 × FP8) ──

macro_rules! dot_kernel {
    ($name:ident, $wa:ident, $wb:ident, $sa:path, $sb:path) => {
        pub(crate) fn $name(a: &[u8], b: &[u8], n: usize) -> f32 {
            debug_assert!(a.len() >= n && b.len() >= n);
            unsafe {
                let chunks = n / 8;
                let mut acc = _mm256_setzero_ps();
                for i in 0..chunks {
                    let wa = $wa(a.as_ptr().add(i * 8));
                    let wb = $wb(b.as_ptr().add(i * 8));
                    acc = _mm256_fmadd_ps(wa, wb, acc);
                }
                let mut sum = hsum256(acc);
                for j in (chunks * 8)..n {
                    sum += $sa(a[j]) * $sb(b[j]);
                }
                sum
            }
        }
    };
}

dot_kernel!(
    dot_e4m3_e4m3,
    widen_e4m3_x8,
    widen_e4m3_x8,
    crate::cast::e4m3_to_f32,
    crate::cast::e4m3_to_f32
);
dot_kernel!(
    dot_e5m2_e5m2,
    widen_e5m2_x8,
    widen_e5m2_x8,
    crate::cast::e5m2_to_f32,
    crate::cast::e5m2_to_f32
);

// ── GEMM: D[i,j] = Σ_k decode(A[i,k])·decode(B[j,k]) + widen(C[i,j]) ──

// The double-FP8 bias GEMM is the shared blocked `core::simd` kernel
// (see `portable::f8`): a BIAS_MR-row block of A is dequantized once and
// every B chunk widen is reused across the whole block, so B pays
// O((m/BIAS_MR)·n·k) widens instead of the O(m·n·k) this tier's former
// row-at-a-time form paid — a structural saving independent of ISA
// (the widens still lower to F16C here). k-blocking bounds the scratch
// and removes the old very-large-k double-widen fallback.
pub(crate) use crate::arch::portable::f8::{
    gemm_e4m3_e4m3_bf16, gemm_e4m3_e4m3_f16, gemm_e5m2_e5m2_bf16, gemm_e5m2_e5m2_f16,
};
