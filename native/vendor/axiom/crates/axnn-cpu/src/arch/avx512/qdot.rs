//! Quantized dot-product and GEMM kernels — AVX-512.
//!
//! Key intrinsics (512-bit counterparts of the AVX2 tier):
//!   - `_mm512_maddubs_epi16`: 64 x (u8 x i8) -> 32 x i16
//!   - `_mm512_madd_epi16`:    32 x i16 -> 16 x i32
//!   - `_mm512_abs_epi8` + mask blend: sign trick replacement (no `_mm512_sign_epi8`)
//!   - `_mm512_reduce_add_ps` / `_mm512_reduce_add_epi32`: horizontal reduction
//!
//! Each Q8 block is 32 bytes of quants — one YMM. We pair two consecutive
//! blocks into a single ZMM (lower/upper 256-bit halves) and split the
//! per-block scale application via extract after the integer dot product.

use core::arch::x86_64::*;

use crate::quant::{
    Q4_BLOCK_BYTES, Q4_BLOCK_SIZE, Q4_D_OFFSET, Q4_DMIN_OFFSET, Q4_QS_OFFSET, Q4_SC_OFFSET,
    Q6_BLOCK_BYTES, Q6_BLOCK_SIZE, Q6_D_OFFSET, Q6_QH_OFFSET, Q6_QL_OFFSET, Q6_SC_OFFSET,
    Q8_BLOCK_BYTES, Q8_BLOCK_SIZE, Q8_QS_OFFSET, unpack_q4_scales,
};
#[cfg(feature = "alloc")]
use alloc::vec;

// ── Helpers ──────────────────────────────────────────────────────────

/// Hardware F16 -> F32 via `vcvtph2ps` (F16C, implied by AVX-512F).
#[inline(always)]
unsafe fn f16_to_f32_hw(bits: u16) -> f32 {
    unsafe {
        let v = _mm_cvtsi32_si128(bits as i32);
        _mm_cvtss_f32(_mm_cvtph_ps(v))
    }
}

/// Read an f16 value from a byte slice using hardware F16C conversion.
#[inline(always)]
unsafe fn read_f16_hw(src: &[u8], off: usize) -> f32 {
    unsafe {
        let bits = u16::from_le_bytes([*src.as_ptr().add(off), *src.as_ptr().add(off + 1)]);
        f16_to_f32_hw(bits)
    }
}

/// Horizontal sum of 8 x i32 in a YMM (reused for per-half extraction).
#[inline(always)]
unsafe fn hsum_i32x8(x: __m256i) -> i32 {
    unsafe {
        let hi = _mm256_extracti128_si256(x, 1);
        let lo = _mm256_castsi256_si128(x);
        let s = _mm_add_epi32(hi, lo);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        _mm_cvtsi128_si32(s)
    }
}

/// Horizontal sum of 8 x f32 in a YMM.
#[inline(always)]
unsafe fn hsum_f32x8(x: __m256) -> f32 {
    unsafe {
        let hi = _mm256_extractf128_ps(x, 1);
        let lo = _mm256_castps256_ps128(x);
        let s = _mm_add_ps(hi, lo);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }
}

// ── 512-bit signed x signed dot product ──────────────────────────────
//
// Split signed(x) into (x & 127) - (x & 128). Each unsigned-by-signed
// pair fits i16 exactly; widening before subtraction preserves +32768.

/// Return 16 exact i32 dot-product lanes over the full signed-byte domain.
#[inline(always)]
unsafe fn signed_dot_512(qx: __m512i, qy: __m512i) -> __m512i {
    unsafe {
        let lo = _mm512_and_si512(qx, _mm512_set1_epi8(127));
        let hi = _mm512_and_si512(qx, _mm512_set1_epi8(-128));
        let ones = _mm512_set1_epi16(1);
        let positive = _mm512_madd_epi16(ones, _mm512_maddubs_epi16(lo, qy));
        let negative = _mm512_madd_epi16(ones, _mm512_maddubs_epi16(hi, qy));
        _mm512_sub_epi32(positive, negative)
    }
}

// ── Vec dot: Q8_0 x Q8_0 ────────────────────────────────────────────
//
// Two Q8 blocks (2 x 32 = 64 bytes of quants) are loaded into the
// lower and upper 256-bit halves of a ZMM register. After the 512-bit
// integer dot product we extract per-half i32 sums and apply the
// per-block scales separately.

pub(crate) fn vec_dot_q8_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q8_BLOCK_SIZE, 0);
    let nb = n / Q8_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q8_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * Q8_BLOCK_BYTES);

    unsafe {
        let mut acc = 0.0f32;

        let pairs = nb / 2;
        let leftover = nb % 2;

        for ip in 0..pairs {
            let ib = ip * 2;
            let xb0 = &x[ib * Q8_BLOCK_BYTES..];
            let yb0 = &y[ib * Q8_BLOCK_BYTES..];
            let xb1 = &x[(ib + 1) * Q8_BLOCK_BYTES..];
            let yb1 = &y[(ib + 1) * Q8_BLOCK_BYTES..];

            let xd0 = read_f16_hw(xb0, 0);
            let yd0 = read_f16_hw(yb0, 0);
            let xd1 = read_f16_hw(xb1, 0);
            let yd1 = read_f16_hw(yb1, 0);

            // Load 32 bytes from each block into 256-bit halves of a ZMM.
            let qx_lo = _mm256_loadu_si256(xb0.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);
            let qx_hi = _mm256_loadu_si256(xb1.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);
            let qx = _mm512_inserti64x4(_mm512_castsi256_si512(qx_lo), qx_hi, 1);

            let qy_lo = _mm256_loadu_si256(yb0.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);
            let qy_hi = _mm256_loadu_si256(yb1.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);
            let qy = _mm512_inserti64x4(_mm512_castsi256_si512(qy_lo), qy_hi, 1);

            // Full-domain signed dot product -> 16 x i32.
            let sum32 = signed_dot_512(qx, qy);

            // Extract lower 8 and upper 8 i32 lanes for per-block scaling.
            let lo = _mm512_castsi512_si256(sum32);
            let hi = _mm512_extracti64x4_epi64(sum32, 1);
            let s0 = hsum_i32x8(lo);
            let s1 = hsum_i32x8(hi);

            acc += (xd0 * yd0) * s0 as f32 + (xd1 * yd1) * s1 as f32;
        }

        // Handle odd trailing block with scalar/256-bit fallback.
        if leftover != 0 {
            let ib = pairs * 2;
            let xb = &x[ib * Q8_BLOCK_BYTES..];
            let yb = &y[ib * Q8_BLOCK_BYTES..];
            let xd = read_f16_hw(xb, 0);
            let yd = read_f16_hw(yb, 0);

            let qx = _mm256_loadu_si256(xb.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);
            let qy = _mm256_loadu_si256(yb.as_ptr().add(Q8_QS_OFFSET) as *const __m256i);

            let lo = _mm256_and_si256(qx, _mm256_set1_epi8(127));
            let hi = _mm256_and_si256(qx, _mm256_set1_epi8(-128));
            let positive = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(lo, qy));
            let negative = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(hi, qy));
            let exact = _mm256_sub_epi32(positive, negative);
            let sum32 = exact;
            acc += (xd * yd) * hsum_i32x8(sum32) as f32;
        }

        acc
    }
}

// ── Vec dot: Q4_K_M x Q8_0 ──────────────────────────────────────────
//
// Same structure as AVX2 but we use 512-bit integer ops (maddubs + madd)
// to process two 32-byte sub-block pairs at once, then extract per-half
// sums for independent scale application. The 256-bit FMA accumulator
// keeps scale logic simple.

pub(crate) fn vec_dot_q4_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q4_BLOCK_SIZE, 0);
    let nb = n / Q4_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q4_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q4_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let ones_16 = _mm256_set1_epi16(1);
        let ones_u8 = _mm256_set1_epi8(1);
        let ones_16z = _mm512_set1_epi16(1);
        let ones_u8z = _mm512_set1_epi8(1);

        let mut acc = _mm256_setzero_ps();
        let mut acc_min = _mm256_setzero_ps();

        for ib in 0..nb {
            let q4 = &x[ib * Q4_BLOCK_BYTES..][..Q4_BLOCK_BYTES];
            let d = read_f16_hw(q4, Q4_D_OFFSET);
            let dmin = read_f16_hw(q4, Q4_DMIN_OFFSET);
            let (scales, mins) = unpack_q4_scales(&q4[Q4_SC_OFFSET..Q4_SC_OFFSET + 12]);

            let mut d_sc = [0.0f32; 8];
            let mut dm_mn = [0.0f32; 8];
            for sb in 0..8 {
                d_sc[sb] = d * scales[sb] as f32;
                dm_mn[sb] = dmin * mins[sb] as f32;
            }

            // 4 iterations matching AVX2 structure: each loads 32 packed Q4
            // bytes covering 64 elements (2 sub-blocks), pairs with 2 Q8 blocks.
            // The nibble extraction uses 256-bit ops (same as AVX2) and the
            // maddubs + madd use 512-bit by combining the two sub-block results.
            for j in 0..4 {
                let sb_lo = j * 2;
                let sb_hi = j * 2 + 1;

                let q8_lo_off = (ib * 8 + sb_lo) * Q8_BLOCK_BYTES;
                let q8_hi_off = (ib * 8 + sb_hi) * Q8_BLOCK_BYTES;

                let q8d_lo = read_f16_hw(y, q8_lo_off);
                let q8d_hi = read_f16_hw(y, q8_hi_off);

                // Extract nibbles from 32 packed bytes (256-bit, same as AVX2).
                let q4bits =
                    _mm256_loadu_si256(q4.as_ptr().add(Q4_QS_OFFSET + j * 32) as *const __m256i);
                let q4_even = _mm256_and_si256(q4bits, m4);
                let q4_odd = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                // Re-interleave to sequential element order.
                let inter_lo = _mm256_unpacklo_epi8(q4_even, q4_odd);
                let inter_hi = _mm256_unpackhi_epi8(q4_even, q4_odd);

                // Cross-lane permute to get contiguous 32-element groups.
                let first_32 = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x20);
                let second_32 = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x31);

                // Load Q8 values.
                let q8v_lo =
                    _mm256_loadu_si256(y.as_ptr().add(q8_lo_off + Q8_QS_OFFSET) as *const __m256i);
                let q8v_hi =
                    _mm256_loadu_si256(y.as_ptr().add(q8_hi_off + Q8_QS_OFFSET) as *const __m256i);

                // Combine into ZMM for 512-bit integer dot product.
                let nib_z = _mm512_inserti64x4(_mm512_castsi256_si512(first_32), second_32, 1);
                let q8v_z = _mm512_inserti64x4(_mm512_castsi256_si512(q8v_lo), q8v_hi, 1);

                // nibble x q8 -> 16 x i32 via maddubs + madd (512-bit).
                let p32z = _mm512_madd_epi16(ones_16z, _mm512_maddubs_epi16(nib_z, q8v_z));

                // Extract per-half 8 x i32, convert to f32, FMA-accumulate.
                let p32_lo = _mm512_castsi512_si256(p32z);
                let p32_hi = _mm512_extracti64x4_epi64(p32z, 1);

                let vd_lo = _mm256_set1_ps(d_sc[sb_lo] * q8d_lo);
                let vd_hi = _mm256_set1_ps(d_sc[sb_hi] * q8d_hi);
                acc = _mm256_fmadd_ps(vd_lo, _mm256_cvtepi32_ps(p32_lo), acc);
                acc = _mm256_fmadd_ps(vd_hi, _mm256_cvtepi32_ps(p32_hi), acc);

                // Min offset: dm_mn[sb] x q8d x sum(q8).
                let q8sum_z = _mm512_madd_epi16(_mm512_maddubs_epi16(ones_u8z, q8v_z), ones_16z);
                let q8sum_lo = _mm512_castsi512_si256(q8sum_z);
                let q8sum_hi = _mm512_extracti64x4_epi64(q8sum_z, 1);

                let md_lo = _mm256_set1_ps(dm_mn[sb_lo] * q8d_lo);
                let md_hi = _mm256_set1_ps(dm_mn[sb_hi] * q8d_hi);
                acc_min = _mm256_fmadd_ps(md_lo, _mm256_cvtepi32_ps(q8sum_lo), acc_min);
                acc_min = _mm256_fmadd_ps(md_hi, _mm256_cvtepi32_ps(q8sum_hi), acc_min);
            }
        }

        hsum_f32x8(acc) - hsum_f32x8(acc_min)
    }
}

// ── Vec dot: Q6_K x Q8_0 ────────────────────────────────────────────
//
// 6-bit K-quant: q6_unsigned = (ql_low4 | (qh_2bits << 4)), range [0..63].
// q6_centered = q6_unsigned - 32.
// Offset-32 trick: maddubs(q6_u, q8) - maddubs(32, q8).
//
// We process two Q8 sub-blocks at a time in 512-bit, extracting per-half
// results for independent scale application.

pub(crate) fn vec_dot_q6_q8(x: &[u8], y: &[u8], n: usize) -> f32 {
    debug_assert_eq!(n % Q6_BLOCK_SIZE, 0);
    let nb = n / Q6_BLOCK_SIZE;
    debug_assert!(x.len() >= nb * Q6_BLOCK_BYTES);
    debug_assert!(y.len() >= nb * (Q6_BLOCK_SIZE / Q8_BLOCK_SIZE) * Q8_BLOCK_BYTES);

    unsafe {
        let m32s_512 = _mm512_set1_epi8(32);

        let mut acc = _mm256_setzero_ps();

        for ib in 0..nb {
            let q6 = &x[ib * Q6_BLOCK_BYTES..][..Q6_BLOCK_BYTES];
            let d6 = read_f16_hw(q6, Q6_D_OFFSET);

            // Process pairs of Q8 sub-blocks: 4 iterations of 2.
            // 512-bit integer ops for the maddubs/sub/madd chain,
            // then extract per-half for scale application with 256-bit FMA.
            for q8p in 0..4 {
                let q8b0 = q8p * 2;
                let q8b1 = q8p * 2 + 1;

                let q8_off0 = (ib * 8 + q8b0) * Q8_BLOCK_BYTES;
                let q8_off1 = (ib * 8 + q8b1) * Q8_BLOCK_BYTES;
                let q8d0 = read_f16_hw(y, q8_off0);
                let q8d1 = read_f16_hw(y, q8_off1);

                // Unpack Q6 unsigned values for both sub-blocks.
                let q6u0 = unpack_q6x32(q6, q8b0 * 32);
                let q6u1 = unpack_q6x32(q6, q8b1 * 32);
                let q6u = _mm512_inserti64x4(_mm512_castsi256_si512(q6u0), q6u1, 1);

                // Load Q8 quants.
                let q8v0 =
                    _mm256_loadu_si256(y.as_ptr().add(q8_off0 + Q8_QS_OFFSET) as *const __m256i);
                let q8v1 =
                    _mm256_loadu_si256(y.as_ptr().add(q8_off1 + Q8_QS_OFFSET) as *const __m256i);
                let q8v = _mm512_inserti64x4(_mm512_castsi256_si512(q8v0), q8v1, 1);

                // Offset-32: maddubs(q6_u, q8) - maddubs(32, q8) in 512-bit.
                let p16 = _mm512_sub_epi16(
                    _mm512_maddubs_epi16(q6u, q8v),
                    _mm512_maddubs_epi16(m32s_512, q8v),
                );

                // Sub-block scales: build a 512-bit scale vector.
                // Lower 256: q8b0 scales, upper 256: q8b1 scales.
                let sc0_a = q6[Q6_SC_OFFSET + q8b0 * 2] as i8 as i16;
                let sc0_b = q6[Q6_SC_OFFSET + q8b0 * 2 + 1] as i8 as i16;
                let sc1_a = q6[Q6_SC_OFFSET + q8b1 * 2] as i8 as i16;
                let sc1_b = q6[Q6_SC_OFFSET + q8b1 * 2 + 1] as i8 as i16;

                let sc_vec = _mm512_set_epi16(
                    sc1_b, sc1_b, sc1_b, sc1_b, sc1_b, sc1_b, sc1_b, sc1_b, sc1_a, sc1_a, sc1_a,
                    sc1_a, sc1_a, sc1_a, sc1_a, sc1_a, sc0_b, sc0_b, sc0_b, sc0_b, sc0_b, sc0_b,
                    sc0_b, sc0_b, sc0_a, sc0_a, sc0_a, sc0_a, sc0_a, sc0_a, sc0_a, sc0_a,
                );
                let p32 = _mm512_madd_epi16(sc_vec, p16);

                // Extract per-half 8 x i32, convert to f32, FMA-accumulate.
                let p32_lo = _mm512_castsi512_si256(p32);
                let p32_hi = _mm512_extracti64x4_epi64(p32, 1);

                let vd0 = _mm256_set1_ps(d6 * q8d0);
                let vd1 = _mm256_set1_ps(d6 * q8d1);
                acc = _mm256_fmadd_ps(vd0, _mm256_cvtepi32_ps(p32_lo), acc);
                acc = _mm256_fmadd_ps(vd1, _mm256_cvtepi32_ps(p32_hi), acc);
            }
        }

        hsum_f32x8(acc)
    }
}

/// Unpack 32 contiguous 6-bit values from a Q6_K block into a YMM register.
///
/// Identical to the AVX2 implementation — operates at 256-bit granularity
/// since each sub-block is exactly 32 elements.
#[inline(always)]
unsafe fn unpack_q6x32(q6: &[u8], elem_base: usize) -> __m256i {
    unsafe {
        let ql_off = Q6_QL_OFFSET + elem_base / 2;
        let qh_off = Q6_QH_OFFSET + elem_base / 4;

        // ql: 16 packed bytes -> 32 sequential 4-bit values via interleave.
        let ql_raw = _mm_loadu_si128(q6.as_ptr().add(ql_off) as *const __m128i);
        let m4_128 = _mm_set1_epi8(0x0F);
        let ql_even = _mm_and_si128(ql_raw, m4_128);
        let ql_odd = _mm_and_si128(_mm_srli_epi16(ql_raw, 4), m4_128);

        let ql_lo = _mm_unpacklo_epi8(ql_even, ql_odd);
        let ql_hi = _mm_unpackhi_epi8(ql_even, ql_odd);
        let ql_vec = _mm256_insertf128_si256(_mm256_castsi128_si256(ql_lo), ql_hi, 1);

        // qh: 8 bytes -> 32 x 2-bit values shifted to [5:4].
        let qh_ptr = q6.as_ptr().add(qh_off);
        let mut qh_bytes = [0u8; 32];
        for k in 0..8 {
            let b = *qh_ptr.add(k);
            qh_bytes[k * 4] = (b & 3) << 4;
            qh_bytes[k * 4 + 1] = ((b >> 2) & 3) << 4;
            qh_bytes[k * 4 + 2] = ((b >> 4) & 3) << 4;
            qh_bytes[k * 4 + 3] = ((b >> 6) & 3) << 4;
        }
        let qh_vec = _mm256_loadu_si256(qh_bytes.as_ptr() as *const __m256i);

        _mm256_or_si256(ql_vec, qh_vec)
    }
}

// ── GEMM: Q x Q8 -> F32 (K-tiled) ───────────────────────────────────

/// Max super-blocks per row (Q4/Q6: 256-elem blocks, supports k <= 16384).
const MAX_NB_256: usize = 64;
/// Max Q8 blocks per row (32-elem blocks, supports k <= 16384).
const MAX_NB_32: usize = 512;

pub(crate) fn gemm_q8_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q8_BLOCK_SIZE, 0);
    let nb = k / Q8_BLOCK_SIZE;
    let row_bytes = nb * Q8_BLOCK_BYTES;

    // Small-N fast path: vec_dot per (i,j) avoids pre-decode overhead.
    if n <= 4 {
        for i in 0..m {
            let w_row = &w[i * row_bytes..][..row_bytes];
            for j in 0..n {
                let x_row = &x[j * row_bytes..][..row_bytes];
                c[i * n + j] += vec_dot_q8_q8(w_row, x_row, k);
            }
        }
        return;
    }

    debug_assert!(nb <= MAX_NB_32);

    unsafe {
        for i in 0..m {
            let w_row = w.as_ptr().add(i * row_bytes);

            // Pre-decode weight scales for all K-blocks.
            let mut wd = [0.0f32; MAX_NB_32];
            for ib in 0..nb {
                wd[ib] = read_f16_hw(
                    core::slice::from_raw_parts(w_row.add(ib * Q8_BLOCK_BYTES), Q8_BLOCK_BYTES),
                    0,
                );
            }

            for j in 0..n {
                let x_row = x.as_ptr().add(j * row_bytes);
                let mut acc = _mm512_setzero_ps();

                let pairs = nb / 2;
                let leftover = nb % 2;

                for ip in 0..pairs {
                    let ib = ip * 2;
                    let wb0 = w_row.add(ib * Q8_BLOCK_BYTES);
                    let wb1 = w_row.add((ib + 1) * Q8_BLOCK_BYTES);
                    let xb0 = x_row.add(ib * Q8_BLOCK_BYTES);
                    let xb1 = x_row.add((ib + 1) * Q8_BLOCK_BYTES);

                    let xd0 = {
                        let bits = u16::from_le_bytes([*xb0, *xb0.add(1)]);
                        f16_to_f32_hw(bits)
                    };
                    let xd1 = {
                        let bits = u16::from_le_bytes([*xb1, *xb1.add(1)]);
                        f16_to_f32_hw(bits)
                    };

                    // Load 2 blocks into ZMM.
                    let qw_lo = _mm256_loadu_si256(wb0.add(Q8_QS_OFFSET) as *const __m256i);
                    let qw_hi = _mm256_loadu_si256(wb1.add(Q8_QS_OFFSET) as *const __m256i);
                    let qw = _mm512_inserti64x4(_mm512_castsi256_si512(qw_lo), qw_hi, 1);

                    let qx_lo = _mm256_loadu_si256(xb0.add(Q8_QS_OFFSET) as *const __m256i);
                    let qx_hi = _mm256_loadu_si256(xb1.add(Q8_QS_OFFSET) as *const __m256i);
                    let qx = _mm512_inserti64x4(_mm512_castsi256_si512(qx_lo), qx_hi, 1);

                    let sum32 = signed_dot_512(qw, qx);

                    // Per-block scale: need to split.
                    let lo = _mm512_castsi512_si256(sum32);
                    let hi = _mm512_extracti64x4_epi64(sum32, 1);
                    let s0 = hsum_i32x8(lo) as f32;
                    let s1 = hsum_i32x8(hi) as f32;

                    let d0 = wd[ib] * xd0;
                    let d1 = wd[ib + 1] * xd1;

                    acc = _mm512_add_ps(acc, _mm512_set1_ps(d0 * s0 + d1 * s1));
                }

                if leftover != 0 {
                    let ib = pairs * 2;
                    let wb = w_row.add(ib * Q8_BLOCK_BYTES);
                    let xb = x_row.add(ib * Q8_BLOCK_BYTES);
                    let xd = {
                        let bits = u16::from_le_bytes([*xb, *xb.add(1)]);
                        f16_to_f32_hw(bits)
                    };

                    let qw = _mm256_loadu_si256(wb.add(Q8_QS_OFFSET) as *const __m256i);
                    let qx = _mm256_loadu_si256(xb.add(Q8_QS_OFFSET) as *const __m256i);

                    let lo = _mm256_and_si256(qw, _mm256_set1_epi8(127));
                    let hi = _mm256_and_si256(qw, _mm256_set1_epi8(-128));
                    let positive =
                        _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(lo, qx));
                    let negative =
                        _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(hi, qx));
                    let exact = _mm256_sub_epi32(positive, negative);
                    let sum32 = exact;
                    let s = hsum_i32x8(sum32) as f32;
                    acc = _mm512_add_ps(acc, _mm512_set1_ps(wd[ib] * xd * s));
                }

                c[i * n + j] += _mm512_reduce_add_ps(acc);
            }
        }
    }
}

#[cfg(feature = "alloc")]
pub(crate) fn gemm_q4_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q4_BLOCK_SIZE, 0);
    let nb = k / Q4_BLOCK_SIZE;
    let w_row_bytes = nb * Q4_BLOCK_BYTES;
    let x_row_bytes = nb * 8 * Q8_BLOCK_BYTES;

    // Small-N fast path.
    if n <= 4 {
        for i in 0..m {
            let w_row = &w[i * w_row_bytes..][..w_row_bytes];
            for j in 0..n {
                let x_row = &x[j * x_row_bytes..][..x_row_bytes];
                c[i * n + j] += vec_dot_q4_q8(w_row, x_row, k);
            }
        }
        return;
    }

    debug_assert!(nb <= MAX_NB_256);

    let nb8 = nb * 8;

    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let ones_16 = _mm256_set1_epi16(1);
        let ones_u8 = _mm256_set1_epi8(1);

        // Pre-compute (q8_scale x byte_sum) for every Q8 block.
        let mut q8d_bsum = vec![0.0f32; n * nb8];
        for j in 0..n {
            for idx in 0..nb8 {
                let q8_ptr = x.as_ptr().add(j * x_row_bytes + idx * Q8_BLOCK_BYTES);
                let q8d = f16_to_f32_hw(u16::from_le_bytes([*q8_ptr, *q8_ptr.add(1)]));
                let q8v = _mm256_loadu_si256(q8_ptr.add(Q8_QS_OFFSET) as *const __m256i);
                let bsum = hsum_i32x8(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(ones_u8, q8v),
                    ones_16,
                ));
                q8d_bsum[j * nb8 + idx] = q8d * bsum as f32;
            }
        }

        for i in 0..m {
            let w_row = w.as_ptr().add(i * w_row_bytes);

            // Pre-decode weight headers.
            let mut d_sc = [[0.0f32; 8]; MAX_NB_256];
            let mut dm_mn = [[0.0f32; 8]; MAX_NB_256];
            for ib in 0..nb {
                let q4 =
                    core::slice::from_raw_parts(w_row.add(ib * Q4_BLOCK_BYTES), Q4_BLOCK_BYTES);
                let d = read_f16_hw(q4, Q4_D_OFFSET);
                let dmin = read_f16_hw(q4, Q4_DMIN_OFFSET);
                let (sc, mn) = unpack_q4_scales(&q4[Q4_SC_OFFSET..Q4_SC_OFFSET + 12]);
                for sb in 0..8 {
                    d_sc[ib][sb] = d * sc[sb] as f32;
                    dm_mn[ib][sb] = dmin * mn[sb] as f32;
                }
            }

            for ib in 0..nb {
                let q4_qs = w_row.add(ib * Q4_BLOCK_BYTES + Q4_QS_OFFSET);

                // Pre-extract interleaved nibble pairs for all 4 sub-block pairs.
                let mut first_32 = [_mm256_setzero_si256(); 4];
                let mut second_32 = [_mm256_setzero_si256(); 4];
                for jj in 0..4 {
                    let q4bits = _mm256_loadu_si256(q4_qs.add(jj * 32) as *const __m256i);
                    let q4_even = _mm256_and_si256(q4bits, m4);
                    let q4_odd = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                    let inter_lo = _mm256_unpacklo_epi8(q4_even, q4_odd);
                    let inter_hi = _mm256_unpackhi_epi8(q4_even, q4_odd);
                    first_32[jj] = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x20);
                    second_32[jj] = _mm256_permute2x128_si256(inter_lo, inter_hi, 0x31);
                }

                for j in 0..n {
                    let q8_base = x.as_ptr().add(j * x_row_bytes + ib * 8 * Q8_BLOCK_BYTES);
                    let mut block_acc = _mm256_setzero_ps();

                    for jj in 0..4 {
                        let sb_lo = jj * 2;
                        let sb_hi = jj * 2 + 1;

                        let q8_lo = q8_base.add(sb_lo * Q8_BLOCK_BYTES);
                        let q8_hi = q8_base.add(sb_hi * Q8_BLOCK_BYTES);

                        let q8d_lo = {
                            let bits = u16::from_le_bytes([*q8_lo, *q8_lo.add(1)]);
                            f16_to_f32_hw(bits)
                        };
                        let q8d_hi = {
                            let bits = u16::from_le_bytes([*q8_hi, *q8_hi.add(1)]);
                            f16_to_f32_hw(bits)
                        };

                        let q8v_lo = _mm256_loadu_si256(q8_lo.add(Q8_QS_OFFSET) as *const __m256i);
                        let q8v_hi = _mm256_loadu_si256(q8_hi.add(Q8_QS_OFFSET) as *const __m256i);

                        let p32_lo =
                            _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(first_32[jj], q8v_lo));
                        let p32_hi =
                            _mm256_madd_epi16(ones_16, _mm256_maddubs_epi16(second_32[jj], q8v_hi));

                        let vd_lo = _mm256_set1_ps(d_sc[ib][sb_lo] * q8d_lo);
                        let vd_hi = _mm256_set1_ps(d_sc[ib][sb_hi] * q8d_hi);
                        block_acc = _mm256_fmadd_ps(vd_lo, _mm256_cvtepi32_ps(p32_lo), block_acc);
                        block_acc = _mm256_fmadd_ps(vd_hi, _mm256_cvtepi32_ps(p32_hi), block_acc);
                    }

                    // Scalar min offset using pre-computed (q8d x bsum).
                    let db = &q8d_bsum[j * nb8 + ib * 8..];
                    let dm = &dm_mn[ib];
                    let mut min_val = 0.0f32;
                    for sb in 0..8 {
                        min_val += dm[sb] * db[sb];
                    }

                    c[i * n + j] += hsum_f32x8(block_acc) - min_val;
                }
            }
        }
    }
}

#[cfg(not(feature = "alloc"))]
pub(crate) fn gemm_q4_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    crate::arch::portable::qdot::gemm_q4_q8(w, x, c, m, n, k)
}

pub(crate) fn gemm_q6_q8(w: &[u8], x: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % Q6_BLOCK_SIZE, 0);
    let nb = k / Q6_BLOCK_SIZE;
    let w_row_bytes = nb * Q6_BLOCK_BYTES;
    let x_row_bytes_q6 = nb * 8 * Q8_BLOCK_BYTES;

    // Small-N fast path.
    if n <= 4 {
        for i in 0..m {
            let w_row = &w[i * w_row_bytes..][..w_row_bytes];
            for j in 0..n {
                let x_row = &x[j * x_row_bytes_q6..][..x_row_bytes_q6];
                c[i * n + j] += vec_dot_q6_q8(w_row, x_row, k);
            }
        }
        return;
    }

    debug_assert!(nb <= MAX_NB_256);

    unsafe {
        let m32s = _mm256_set1_epi8(32);

        for i in 0..m {
            let w_row = &w[i * w_row_bytes..];

            for ib in 0..nb {
                let q6 = &w_row[ib * Q6_BLOCK_BYTES..][..Q6_BLOCK_BYTES];
                let d6 = read_f16_hw(q6, Q6_D_OFFSET);

                // Pre-unpack all 8 x 32 Q6 unsigned values.
                let mut q6u = [_mm256_setzero_si256(); 8];
                let mut sc = [0i16; 16];
                for q8b in 0..8 {
                    q6u[q8b] = unpack_q6x32(q6, q8b * 32);
                    sc[q8b * 2] = q6[Q6_SC_OFFSET + q8b * 2] as i8 as i16;
                    sc[q8b * 2 + 1] = q6[Q6_SC_OFFSET + q8b * 2 + 1] as i8 as i16;
                }

                for j in 0..n {
                    let q8_seg = x.as_ptr().add(j * x_row_bytes_q6 + ib * 8 * Q8_BLOCK_BYTES);
                    let mut block_acc = _mm256_setzero_ps();

                    for q8b in 0..8 {
                        let q8_ptr = q8_seg.add(q8b * Q8_BLOCK_BYTES);
                        let q8d = {
                            let bits = u16::from_le_bytes([*q8_ptr, *q8_ptr.add(1)]);
                            f16_to_f32_hw(bits)
                        };

                        let q8v = _mm256_loadu_si256(q8_ptr.add(Q8_QS_OFFSET) as *const __m256i);

                        // Offset-32: maddubs(q6_u, q8) - maddubs(32, q8).
                        let p16 = _mm256_sub_epi16(
                            _mm256_maddubs_epi16(q6u[q8b], q8v),
                            _mm256_maddubs_epi16(m32s, q8v),
                        );

                        let sc_vec = _mm256_set_epi16(
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2 + 1],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                            sc[q8b * 2],
                        );
                        let p32 = _mm256_madd_epi16(sc_vec, p16);

                        let vd = _mm256_set1_ps(d6 * q8d);
                        block_acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32), block_acc);
                    }

                    c[i * n + j] += hsum_f32x8(block_acc);
                }
            }
        }
    }
}
