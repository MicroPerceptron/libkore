//! Scalar-to-scalar element conversion and F16/B16 bit manipulation.
//!
//! Every scalar dtype (F64, F32, F16, B16, I64, I32, I16, I8, U8) can
//! be cast to every other scalar dtype. Quantized casts are not handled
//! here — the backend returns `UnsupportedCast` for those.
//!
//! Conversion pipeline:
//!   1. Decode source bytes → `f64` or `i64` intermediate
//!   2. Cross-convert if needed (float↔int)
//!   3. Encode intermediate → target bytes

use axnn::DType;

// ── Public entry point ──────────────────────────────────────────────

/// Convert one scalar element from `src_dt` to `dst_dt`.
///
/// `src` must be `src_dt.block_byte_size()` bytes.
/// `dst` must be `dst_dt.block_byte_size()` bytes.
///
/// # Panics
///
/// Panics if either dtype is quantized. Callers must guard against
/// this before reaching here.
pub(crate) fn cast_element(src: &[u8], src_dt: DType, dst: &mut [u8], dst_dt: DType) {
    debug_assert!(!src_dt.is_quantized() && !dst_dt.is_quantized());

    if dst_dt.is_float() {
        // Target is float: go through f64 intermediate.
        let val = decode_to_f64(src, src_dt);
        encode_from_f64(val, dst, dst_dt);
    } else {
        // Target is integer: go through i64 intermediate.
        let val = decode_to_i64(src, src_dt);
        encode_from_i64(val, dst, dst_dt);
    }
}

// ── Decode: bytes → intermediate ────────────────────────────────────

/// Read one element as f64.
fn decode_to_f64(bytes: &[u8], dt: DType) -> f64 {
    match dt {
        DType::F64 => f64::from_le_bytes(bytes[..8].try_into().unwrap()),
        DType::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
        DType::F16 => f16_to_f32(u16::from_le_bytes(bytes[..2].try_into().unwrap())) as f64,
        DType::B16 => b16_to_f32(u16::from_le_bytes(bytes[..2].try_into().unwrap())) as f64,
        DType::B8 => e4m3_to_f32(bytes[0]) as f64,
        DType::F8 => e5m2_to_f32(bytes[0]) as f64,
        DType::I64 => i64::from_le_bytes(bytes[..8].try_into().unwrap()) as f64,
        DType::I32 => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
        DType::I16 => i16::from_le_bytes(bytes[..2].try_into().unwrap()) as f64,
        DType::I8 => (bytes[0] as i8) as f64,
        DType::U8 => bytes[0] as f64,
        _ => unreachable!("quantized dtypes should not reach cast_element"),
    }
}

/// Read one element as i64.
fn decode_to_i64(bytes: &[u8], dt: DType) -> i64 {
    match dt {
        DType::F64 => f64::from_le_bytes(bytes[..8].try_into().unwrap()) as i64,
        DType::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()) as i64,
        DType::F16 => f16_to_f32(u16::from_le_bytes(bytes[..2].try_into().unwrap())) as i64,
        DType::B16 => b16_to_f32(u16::from_le_bytes(bytes[..2].try_into().unwrap())) as i64,
        DType::B8 => e4m3_to_f32(bytes[0]) as i64,
        DType::F8 => e5m2_to_f32(bytes[0]) as i64,
        DType::I64 => i64::from_le_bytes(bytes[..8].try_into().unwrap()),
        DType::I32 => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as i64,
        DType::I16 => i16::from_le_bytes(bytes[..2].try_into().unwrap()) as i64,
        DType::I8 => (bytes[0] as i8) as i64,
        DType::U8 => bytes[0] as i64,
        _ => unreachable!("quantized dtypes should not reach cast_element"),
    }
}

// ── Encode: intermediate → bytes ────────────────────────────────────

/// Write one f64 value into the target format.
fn encode_from_f64(val: f64, dst: &mut [u8], dt: DType) {
    match dt {
        DType::F64 => dst[..8].copy_from_slice(&val.to_le_bytes()),
        DType::F32 => dst[..4].copy_from_slice(&(val as f32).to_le_bytes()),
        DType::F16 => {
            let bits = f32_to_f16(val as f32);
            dst[..2].copy_from_slice(&bits.to_le_bytes());
        }
        DType::B16 => {
            let bits = f32_to_b16(val as f32);
            dst[..2].copy_from_slice(&bits.to_le_bytes());
        }
        _ => unreachable!("encode_from_f64 called with non-float target"),
    }
}

/// Write one i64 value into the target format.
fn encode_from_i64(val: i64, dst: &mut [u8], dt: DType) {
    match dt {
        DType::I64 => dst[..8].copy_from_slice(&val.to_le_bytes()),
        DType::I32 => dst[..4].copy_from_slice(&(val as i32).to_le_bytes()),
        DType::I16 => dst[..2].copy_from_slice(&(val as i16).to_le_bytes()),
        DType::I8 => dst[0] = val as i8 as u8,
        DType::U8 => dst[0] = val as u8,
        _ => unreachable!("encode_from_i64 called with non-integer target"),
    }
}

// ── F16 (IEEE 754 half-precision) ───────────────────────────────────
//
// Layout: sign(1) + exponent(5) + mantissa(10)
// Exponent bias: 15   (f32 bias: 127)
// Denormals flushed to zero (matches GPU flush-to-zero behavior).

/// Decode F16 bits → f32.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    let f32_bits = if exp == 0 {
        // Zero or denormal → flush to signed zero.
        sign << 31
    } else if exp == 0x1F {
        // Inf or NaN.
        let f32_mant = mant << 13; // shift into f32 mantissa position
        (sign << 31) | (0xFF << 23) | f32_mant
    } else {
        // Normal: re-bias exponent (15 → 127).
        let f32_exp = exp + 127 - 15;
        (sign << 31) | (f32_exp << 23) | (mant << 13)
    };

    f32::from_bits(f32_bits)
}

/// Encode f32 → F16 bits.
pub(crate) fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // Inf or NaN.
        let f16_mant = if mant != 0 { 0x200 } else { 0 }; // preserve NaN-ness
        return ((sign << 15) | (0x1F << 10) | f16_mant) as u16;
    }

    // Re-bias: f32 bias 127 → f16 bias 15.
    let new_exp = exp - 127 + 15;

    if new_exp >= 0x1F {
        // Overflow → infinity.
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if new_exp <= 0 {
        // Underflow → flush to zero (no denormal support).
        return (sign << 15) as u16;
    }

    let f16_mant = mant >> 13; // truncate mantissa (23 → 10 bits)
    ((sign << 15) | ((new_exp as u32) << 10) | f16_mant) as u16
}

// ── F16 packed byte helpers ────────────────────────────────────────
//
// Convenience for domain kernels that operate on `&[u8]` packed f16.

/// Read one f16 from a 2-byte LE chunk, return as f32.
#[inline(always)]
pub(crate) fn read_f16(src: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([src[0], src[1]]))
}

/// Write one f32 as f16 into a 2-byte LE chunk.
#[inline(always)]
pub(crate) fn write_f16(dst: &mut [u8], val: f32) {
    let bytes = f32_to_f16(val).to_le_bytes();
    dst[0] = bytes[0];
    dst[1] = bytes[1];
}

// ── BF16 packed byte helpers ──────────────────────────────────────

/// Read one bf16 from a 2-byte LE chunk, return as f32.
#[inline(always)]
pub(crate) fn read_bf16(src: &[u8]) -> f32 {
    b16_to_f32(u16::from_le_bytes([src[0], src[1]]))
}

/// Write one f32 as bf16 into a 2-byte LE chunk.
#[inline(always)]
pub(crate) fn write_bf16(dst: &mut [u8], val: f32) {
    let bytes = f32_to_b16(val).to_le_bytes();
    dst[0] = bytes[0];
    dst[1] = bytes[1];
}

// ── B16 (bfloat16) ─────────────────────────────────────────────────
//
// Layout: sign(1) + exponent(8) + mantissa(7)
// Same exponent as f32, just truncated mantissa.
// This makes conversion trivial: top 16 bits of f32.

/// Decode B16 bits → f32.
pub(crate) fn b16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Encode f32 → B16 bits (truncate mantissa, no rounding).
///
/// NaN inputs are preserved as quiet NaN in BF16 (quiet bit forced).
pub(crate) fn f32_to_b16(val: f32) -> u16 {
    let bits = val.to_bits();
    // Signaling NaNs (exponent=0xFF, mantissa!=0, quiet bit clear)
    // would truncate to infinity — force the quiet bit to preserve NaN.
    if bits & 0x7FFF_FFFF > 0x7F80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    (bits >> 16) as u16
}

// ── F8 (OCP E4M3 / E5M2) ──────────────────────────────────────────
//
// One-byte floats used for weight quantization with an *external*
// per-tensor scale. Decode is exact — including subnormals, which carry
// real signal in a format this narrow (unlike F16, we do NOT flush to
// zero). These are the scalar references the SIMD `arch::*::f8` kernels
// are validated against.

/// Narrow `value` to the nearest finite encoding of an 8-bit float, given that format's exact
/// decoder. Ties keep the lower canonical bit pattern; NaN maps to the canonical NaN.
///
/// **Saturating, not NaN-on-overflow.** This encodes offline constants, where a value past the
/// format's range should become the largest representable one rather than poison a tensor. The
/// device-side TOSA `CAST` operator makes the opposite choice for E4M3 (virtio-accel ADR 0009),
/// because a saturated 448 produced there would be indistinguishable from a genuine one; a
/// constant encoder has no such ambiguity, since the caller can see the value it asked for.
///
/// Exhaustive over the 256 encodings rather than a closed-form narrowing: these are offline
/// constants, so clarity beats speed, and a search cannot be subtly wrong about rounding. This is
/// the scalar reference the same way the decoders above are.
fn nearest_fp8(value: f32, decode: fn(u8) -> f32) -> u8 {
    if value.is_nan() {
        return 0x7F;
    }
    // Search magnitudes and apply the sign separately. Comparing signed values would lose -0,
    // whose magnitude ties with +0 and would keep the lower pattern.
    let sign = if value.is_sign_negative() { 0x80 } else { 0x00 };
    let magnitude = value.abs();
    let (mut best, mut best_error) = (0u8, f32::INFINITY);
    let (mut largest, mut largest_value) = (0u8, 0.0f32);
    let mut bits = 0u8;
    loop {
        let candidate = decode(bits);
        if candidate.is_finite() {
            if candidate > largest_value {
                largest_value = candidate;
                largest = bits;
            }
            let error = (candidate - magnitude).abs();
            if error < best_error {
                best_error = error;
                best = bits;
            }
        }
        if bits == 0x7F {
            break;
        }
        bits += 1;
    }
    // Saturation must be explicit rather than falling out of the search: far past the format's
    // range every candidate's error rounds to the same float, so the nearest test would keep
    // whichever pattern it saw first -- zero among them.
    if magnitude >= largest_value {
        sign | largest
    } else {
        sign | best
    }
}

/// Narrow to E4M3 (`torch.float8_e4m3fn`), saturating at ±448. See [`nearest_fp8`].
pub(crate) fn f32_to_e4m3(value: f32) -> u8 {
    nearest_fp8(value, e4m3_to_f32)
}

/// Narrow to E5M2, saturating at the largest finite value rather than producing an infinity.
/// See [`nearest_fp8`].
pub(crate) fn f32_to_e5m2(value: f32) -> u8 {
    nearest_fp8(value, e5m2_to_f32)
}

/// Decode an E4M3 byte (`torch.float8_e4m3fn`: bias 7, no infinities,
/// `S.1111.111` = NaN) to f32. Exact on all 256 encodings.
pub(crate) const fn e4m3_to_f32(bits: u8) -> f32 {
    let s = ((bits >> 7) as u32) << 31;
    let e = (bits >> 3) & 0x0F; // 4-bit exponent
    let m = (bits & 0x07) as u32; // 3-bit mantissa
    let f32_bits = if e == 0x0F && m == 0x07 {
        s | 0x7FC0_0000 // the sole NaN
    } else if e == 0 {
        if m == 0 {
            s // ±0
        } else {
            // Subnormal: normalize the 3-bit mantissa into f32.
            let mut mant = m;
            let mut shift = 0u32;
            while mant & 0x08 == 0 {
                mant <<= 1;
                shift += 1;
            }
            mant &= 0x07;
            s | ((127 + 1 - 7 - shift) << 23) | (mant << 20)
        }
    } else {
        // Normal: re-bias exponent (7 → 127), place the 3-bit mantissa.
        s | (((e as u32) + 127 - 7) << 23) | (m << 20)
    };
    f32::from_bits(f32_bits)
}

/// Decode an E5M2 byte (bias 15, IEEE-style infinities/NaNs) to f32.
/// Exact on all 256 encodings; exponent width and bias match F16.
pub(crate) const fn e5m2_to_f32(bits: u8) -> f32 {
    let s = ((bits >> 7) as u32) << 31;
    let e = (bits >> 2) & 0x1F; // 5-bit exponent
    let m = (bits & 0x03) as u32; // 2-bit mantissa
    let f32_bits = if e == 0x1F {
        s | 0x7F80_0000 | (m << 21) // inf (m=0) / NaN (m≠0)
    } else if e == 0 {
        if m == 0 {
            s
        } else {
            let mut mant = m;
            let mut shift = 0u32;
            while mant & 0x04 == 0 {
                mant <<= 1;
                shift += 1;
            }
            mant &= 0x03;
            s | ((127 + 1 - 15 - shift) << 23) | (mant << 21)
        }
    } else {
        s | (((e as u32) + 127 - 15) << 23) | (m << 21)
    };
    f32::from_bits(f32_bits)
}

// ── Unsigned floats (unsigned-float UF8 / UF16) ────────────────────────
//
// These are storage codecs, not new `axnn::DType` variants.  That boundary is
// deliberate: the formats are useful for physically non-negative rendering
// fields, while the graph/device ABI has no truthful UF8/UF16 element type.
// Keeping the codecs here gives the scalar tails and the SIMD kernels one
// exact semantic reference without putting the sibling crate's LUT-backed
// arithmetic on a hot path.

/// Decode an unsigned E4M4 byte (bias 7, IEEE-style infinities/NaNs) to f32.
/// Exact on all 256 encodings, including subnormals.
pub(crate) const fn uf8e4m4_to_f32(bits: u8) -> f32 {
    let e = (bits >> 4) & 0x0F;
    let m = (bits & 0x0F) as u32;
    let f32_bits = if e == 0x0F {
        0x7F80_0000 | (m << 19)
    } else if e == 0 {
        if m == 0 {
            0
        } else {
            // m · 2^-10; exact because m ≤ 15 and the scale is a power
            // of two. This avoids a variable normalization loop on random-
            // access voxel/splat paths.
            return (m as f32) * 0.000_976_562_5;
        }
    } else {
        (((e as u32) + 127 - 7) << 23) | (m << 19)
    };
    f32::from_bits(f32_bits)
}

/// Relayout unsigned E4M4 as IEEE F16 bits, exactly and without using the
/// floating-point environment.
pub(crate) const fn uf8e4m4_to_f16_bits(bits: u8) -> u16 {
    let exp = (bits >> 4) & 0x0F;
    let mant = (bits & 0x0F) as u16;
    if exp == 0x0F {
        return 0x7C00 | (mant << 6);
    }
    if exp != 0 {
        return (((exp as u16) + 8) << 10) | (mant << 6);
    }
    if mant == 0 {
        return 0;
    }
    let top = (u16::BITS - 1 - mant.leading_zeros()) as u16;
    let f16_exp = top + 5;
    let f16_mant = (mant - (1u16 << top)) << (10 - top);
    (f16_exp << 10) | f16_mant
}

/// Decode an unsigned E5M3 byte (bias 15, IEEE-style infinities/NaNs) to f32.
/// Exact on all 256 encodings, including subnormals.
pub(crate) const fn uf8e5m3_to_f32(bits: u8) -> f32 {
    let e = (bits >> 3) & 0x1F;
    let m = (bits & 0x07) as u32;
    let f32_bits = if e == 0x1F {
        0x7F80_0000 | (m << 20)
    } else if e == 0 {
        if m == 0 {
            0
        } else {
            return (m as f32) * 0.000_007_629_394_5; // m · 2^-17, exact
        }
    } else {
        (((e as u32) + 127 - 15) << 23) | (m << 20)
    };
    f32::from_bits(f32_bits)
}

/// Decode unsigned E5M11 bits (bias 15) to f32.
/// Exact on all 65,536 encodings, including subnormals.
pub(crate) const fn uf16e5m11_to_f32(bits: u16) -> f32 {
    let e = (bits >> 11) & 0x1F;
    let m = (bits & 0x07FF) as u32;
    let f32_bits = if e == 0x1F {
        0x7F80_0000 | (m << 12)
    } else if e == 0 {
        if m == 0 {
            0
        } else {
            return (m as f32) * 0.000_000_029_802_322; // m · 2^-25, exact
        }
    } else {
        (((e as u32) + 127 - 15) << 23) | (m << 12)
    };
    f32::from_bits(f32_bits)
}

/// Encode a non-negative f32 in an unsigned binary-float layout using
/// round-to-nearest, ties-to-even. Negative inputs (including `-inf`) and
/// NaNs become the layout's canonical quiet NaN; positive overflow becomes
/// infinity. This is the `unsigned-float` crate's conversion contract.
fn f32_to_unsigned_rne(value: f32, exp_bits: u32, mantissa_bits: u32, bias: i32) -> u32 {
    let bits = value.to_bits();
    let sign = bits >> 31;
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let fraction = bits & 0x007F_FFFF;
    let max_exp = (1u32 << exp_bits) - 1;
    let quiet_nan = (max_exp << mantissa_bits) | (1u32 << (mantissa_bits - 1));

    if f32_exp == 0xFF {
        return if fraction == 0 && sign == 0 {
            max_exp << mantissa_bits
        } else {
            quiet_nan
        };
    }
    if bits << 1 == 0 {
        return 0;
    }
    if sign != 0 {
        return quiet_nan;
    }

    let (significand, exponent) = if f32_exp == 0 {
        (fraction as u64, 1 - 127 - 23)
    } else {
        (((1u32 << 23) | fraction) as u64, f32_exp - 127 - 23)
    };
    if significand == 0 {
        return 0;
    }

    let top = u64::BITS - 1 - significand.leading_zeros();
    let value_unbiased = top as i32 + exponent;
    let max_finite_unbiased = max_exp as i32 - 1 - bias;
    let min_normal_unbiased = 1 - bias;
    if value_unbiased > max_finite_unbiased {
        return max_exp << mantissa_bits;
    }

    if value_unbiased < min_normal_unbiased {
        let scale = min_normal_unbiased - mantissa_bits as i32;
        let rounded = round_shift_rne(significand, exponent - scale);
        if rounded == 0 {
            return 0;
        }
        if rounded >= (1u64 << mantissa_bits) {
            return 1u32 << mantissa_bits;
        }
        return rounded as u32;
    }

    let mut unbiased = value_unbiased;
    let mut rounded = round_shift_rne(significand, mantissa_bits as i32 - top as i32);
    if rounded == (1u64 << (mantissa_bits + 1)) {
        rounded >>= 1;
        unbiased += 1;
    }
    let biased = unbiased + bias;
    if biased >= max_exp as i32 {
        return max_exp << mantissa_bits;
    }
    ((biased as u32) << mantissa_bits) | (rounded as u32 - (1u32 << mantissa_bits))
}

#[inline(always)]
fn round_shift_rne(value: u64, shift: i32) -> u64 {
    if shift >= 0 {
        return value.checked_shl(shift as u32).unwrap_or(u64::MAX);
    }
    let right = (-shift) as u32;
    if right >= u64::BITS {
        return 0;
    }
    let quotient = value >> right;
    let remainder_mask = (1u64 << right) - 1;
    let remainder = value & remainder_mask;
    let halfway = 1u64 << (right - 1);
    if remainder > halfway || (remainder == halfway && quotient & 1 != 0) {
        quotient + 1
    } else {
        quotient
    }
}

#[inline(always)]
pub(crate) fn f32_to_uf8e4m4_rne(value: f32) -> u8 {
    f32_to_unsigned_rne(value, 4, 4, 7) as u8
}

#[inline(always)]
pub(crate) fn f32_to_uf8e5m3_rne(value: f32) -> u8 {
    f32_to_unsigned_rne(value, 5, 3, 15) as u8
}

#[inline(always)]
pub(crate) fn f32_to_uf16e5m11_rne(value: f32) -> u16 {
    f32_to_unsigned_rne(value, 5, 11, 15) as u16
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── F16 round-trip ─────────────────────────────────────────

    #[test]
    fn f16_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
        assert_eq!(f32_to_f16(0.0f32), 0x0000);
    }

    #[test]
    fn f16_one() {
        // 1.0 in F16: sign=0, exp=15 (0b01111), mant=0 → 0x3C00
        assert_eq!(f16_to_f32(0x3C00), 1.0f32);
        assert_eq!(f32_to_f16(1.0f32), 0x3C00);
    }

    #[test]
    fn f16_negative() {
        assert_eq!(f16_to_f32(0xBC00), -1.0f32);
        assert_eq!(f32_to_f16(-1.0f32), 0xBC00);
    }

    #[test]
    fn f16_inf() {
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
    }

    #[test]
    fn f16_nan() {
        assert!(f16_to_f32(0x7E00).is_nan());
        let bits = f32_to_f16(f32::NAN);
        assert!(f16_to_f32(bits).is_nan());
    }

    #[test]
    fn f16_overflow_to_inf() {
        // 100000.0 exceeds F16 range (~65504) → should become inf.
        assert!(f16_to_f32(f32_to_f16(100000.0f32)).is_infinite());
    }

    #[test]
    fn f16_underflow_to_zero() {
        // Very small value flushes to zero (no denormal support).
        assert_eq!(f16_to_f32(f32_to_f16(1.0e-9f32)), 0.0f32);
    }

    // ── B16 round-trip ─────────────────────────────────────────

    #[test]
    fn b16_one() {
        let bits = f32_to_b16(1.0f32);
        assert_eq!(b16_to_f32(bits), 1.0f32);
    }

    #[test]
    fn b16_negative() {
        let bits = f32_to_b16(-42.0f32);
        assert_eq!(b16_to_f32(bits), -42.0f32);
    }

    #[test]
    fn b16_precision_loss() {
        // B16 has 7-bit mantissa vs f32's 23-bit. Values that differ
        // only in the low mantissa bits should map to the same B16.
        let a = 1.001f32;
        let b = 1.002f32;
        assert_eq!(f32_to_b16(a), f32_to_b16(b));
    }

    // ── Element cast ───────────────────────────────────────────

    #[test]
    fn cast_f32_to_f64() {
        let src = 3.14f32.to_le_bytes();
        let mut dst = [0u8; 8];
        cast_element(&src, DType::F32, &mut dst, DType::F64);
        let result = f64::from_le_bytes(dst);
        assert!((result - 3.14f64).abs() < 1e-6);
    }

    #[test]
    fn cast_i32_to_f32() {
        let src = 42i32.to_le_bytes();
        let mut dst = [0u8; 4];
        cast_element(&src, DType::I32, &mut dst, DType::F32);
        let result = f32::from_le_bytes(dst);
        assert_eq!(result, 42.0f32);
    }

    #[test]
    fn cast_f32_to_i32_truncates() {
        let src = 3.99f32.to_le_bytes();
        let mut dst = [0u8; 4];
        cast_element(&src, DType::F32, &mut dst, DType::I32);
        let result = i32::from_le_bytes(dst);
        assert_eq!(result, 3); // truncation, not rounding
    }

    #[test]
    fn cast_i8_to_i64_sign_extends() {
        let src = [0xFE]; // -2 as i8
        let mut dst = [0u8; 8];
        cast_element(&src, DType::I8, &mut dst, DType::I64);
        let result = i64::from_le_bytes(dst);
        assert_eq!(result, -2);
    }

    #[test]
    fn cast_u8_to_i32() {
        let src = [255u8];
        let mut dst = [0u8; 4];
        cast_element(&src, DType::U8, &mut dst, DType::I32);
        let result = i32::from_le_bytes(dst);
        assert_eq!(result, 255);
    }

    #[test]
    fn unsigned_float_codecs_match_sibling_oracle_exhaustively() {
        use unsigned_float::{Uf8, Uf8E5M3, Uf16};

        for raw in 0u16..=u8::MAX as u16 {
            let raw = raw as u8;
            let want4 = Uf8::from_bits(raw).to_f32();
            let got4 = uf8e4m4_to_f32(raw);
            if want4.is_nan() {
                assert!(got4.is_nan(), "UF8 E4M4 decode 0x{raw:02X}");
            } else {
                assert_eq!(got4, want4, "UF8 E4M4 decode 0x{raw:02X}");
                assert_eq!(f32_to_uf8e4m4_rne(want4), raw, "UF8 E4M4 encode");
            }

            let want5 = Uf8E5M3::from_bits(raw).to_f32();
            let got5 = uf8e5m3_to_f32(raw);
            if want5.is_nan() {
                assert!(got5.is_nan(), "UF8 E5M3 decode 0x{raw:02X}");
            } else {
                assert_eq!(got5, want5, "UF8 E5M3 decode 0x{raw:02X}");
                assert_eq!(f32_to_uf8e5m3_rne(want5), raw, "UF8 E5M3 encode");
            }
        }

        for raw in 0u32..=u16::MAX as u32 {
            let raw = raw as u16;
            let want = Uf16::from_bits(raw).to_f32();
            let got = uf16e5m11_to_f32(raw);
            if want.is_nan() {
                assert!(got.is_nan(), "UF16 E5M11 decode 0x{raw:04X}");
            } else {
                assert_eq!(got, want, "UF16 E5M11 decode 0x{raw:04X}");
                assert_eq!(f32_to_uf16e5m11_rne(want), raw, "UF16 E5M11 encode");
            }
        }
    }

    #[test]
    fn unsigned_float_encoder_matches_sibling_on_boundaries_and_ties() {
        use unsigned_float::{Uf8, Uf8E5M3, Uf16};

        let values = [
            -f32::INFINITY,
            -1.0,
            -0.0,
            0.0,
            f32::from_bits(1),
            2.0f32.powi(-26),
            2.0f32.powi(-25),
            1.0 + 1.0 / 32.0,
            1.0 + 3.0 / 32.0,
            248.0,
            248.0 + 4.0,
            65_520.0,
            f32::INFINITY,
            f32::NAN,
        ];
        for value in values {
            assert_eq!(
                f32_to_uf8e4m4_rne(value),
                Uf8::from_f32(value).to_bits(),
                "UF8 E4M4 {value:?}"
            );
            assert_eq!(
                f32_to_uf8e5m3_rne(value),
                Uf8E5M3::from_f32(value).to_bits(),
                "UF8 E5M3 {value:?}"
            );
            assert_eq!(
                f32_to_uf16e5m11_rne(value),
                Uf16::from_f32(value).to_bits(),
                "UF16 E5M11 {value:?}"
            );
        }
    }

    #[test]
    fn unsigned_float_rne_midpoints_choose_even_neighbor() {
        fn check_u8(max: u8, decode: fn(u8) -> f32, encode: fn(f32) -> u8) {
            for low in 0..max {
                let a = decode(low);
                let b = decode(low + 1);
                let midpoint = a + (b - a) * 0.5;
                let below = f32::from_bits(midpoint.to_bits() - 1);
                let above = f32::from_bits(midpoint.to_bits() + 1);
                assert_eq!(encode(below), low, "below midpoint at 0x{low:02X}");
                assert_eq!(encode(above), low + 1, "above midpoint at 0x{low:02X}");
                let even = if low & 1 == 0 { low } else { low + 1 };
                assert_eq!(encode(midpoint), even, "tie at 0x{low:02X}");
            }
        }

        check_u8(0xEF, uf8e4m4_to_f32, f32_to_uf8e4m4_rne);
        check_u8(0xF7, uf8e5m3_to_f32, f32_to_uf8e5m3_rne);

        for low in 0u16..0xF7FF {
            let a = uf16e5m11_to_f32(low);
            let b = uf16e5m11_to_f32(low + 1);
            let midpoint = a + (b - a) * 0.5;
            let below = f32::from_bits(midpoint.to_bits() - 1);
            let above = f32::from_bits(midpoint.to_bits() + 1);
            assert_eq!(f32_to_uf16e5m11_rne(below), low);
            assert_eq!(f32_to_uf16e5m11_rne(above), low + 1);
            let even = if low & 1 == 0 { low } else { low + 1 };
            assert_eq!(f32_to_uf16e5m11_rne(midpoint), even);
        }

        assert_eq!(f32_to_uf8e4m4_rne(252.0), 0xF0);
        assert_eq!(f32_to_uf8e5m3_rne(63_488.0), 0xF8);
        assert_eq!(f32_to_uf16e5m11_rne(65_528.0), 0xF800);
    }
}

#[cfg(test)]
mod fp8_narrowing_tests {
    use super::{e4m3_to_f32, e5m2_to_f32, f32_to_e4m3, f32_to_e5m2};

    /// Every finite encoding survives a decode/encode round trip, so the narrowing agrees with
    /// the decoders it is the inverse of -- both zeros included, since the sign is carried
    /// separately from the magnitude search.
    #[test]
    fn narrowing_inverts_the_decoders_on_every_finite_encoding() {
        for (name, decode, encode) in [
            (
                "e4m3",
                e4m3_to_f32 as fn(u8) -> f32,
                f32_to_e4m3 as fn(f32) -> u8,
            ),
            (
                "e5m2",
                e5m2_to_f32 as fn(u8) -> f32,
                f32_to_e5m2 as fn(f32) -> u8,
            ),
        ] {
            for bits in 0..=u8::MAX {
                let value = decode(bits);
                if !value.is_finite() {
                    continue;
                }
                assert_eq!(encode(value), bits, "{name}: 0x{bits:02x} -> {value}");
            }
        }
    }

    /// Overflow saturates rather than becoming NaN or infinity, which is the whole point of this
    /// encoder being separate from the device-side `CAST` policy.
    #[test]
    fn overflow_saturates_and_nan_is_canonical() {
        assert_eq!(f32_to_e4m3(1.0e30), 0x7E, "e4m3 max finite is 448");
        assert_eq!(f32_to_e4m3(-1.0e30), 0xFE);
        assert_eq!(e4m3_to_f32(f32_to_e4m3(1.0e30)), 448.0);
        assert!(e5m2_to_f32(f32_to_e5m2(1.0e30)).is_finite());
        assert!(e4m3_to_f32(f32_to_e4m3(f32::NAN)).is_nan());
        assert!(e5m2_to_f32(f32_to_e5m2(f32::NAN)).is_nan());
    }
}
