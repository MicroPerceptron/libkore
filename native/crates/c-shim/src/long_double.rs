//! `long double`'s two conversions, `strtold` and `ldexpl`, as bytes.
//!
//! The type is the C compiler's, never Rust's: x87 double-extended (a 64-bit
//! significand with an explicit integer bit) on x86_64, IEEE binary128 on
//! aarch64 and riscv64. Rust has no type for the first and no ABI promise
//! that its `f128` crosses `extern "C"` as C's `long double` for the second,
//! so the consumer's C layer owns the type: it passes its value's bytes and
//! `LDBL_MANT_DIG`, and this side computes on them. Nothing here names an
//! architecture — the format is the C side's word, checked.
//!
//! The arithmetic is LLVM's APFloat ([`rustc_apfloat`]), the code clang
//! folds `long double` literals with: decimal conversion is correctly
//! rounded for any length of input, and scaling rounds once, subnormals and
//! overflow included. The subject grammar is [`crate::number`]'s, so
//! `strtold` reads exactly the sequence `strtod` does.

use rustc_apfloat::ieee::{Quad, X87DoubleExtended};
use rustc_apfloat::{Float, Round, Status};

use crate::number::{FloatSubject, float_subject};

/// `sizeof(long double)` on every kore C ABI: x87's ten bytes padded to
/// sixteen, or binary128's sixteen. The C side asserts its own type agrees.
pub const BYTES: usize = 16;

/// A `long double` format, by the `LDBL_MANT_DIG` that names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// x87 double-extended (`LDBL_MANT_DIG == 64`).
    X87,
    /// IEEE binary128 (`LDBL_MANT_DIG == 113`).
    Quad,
}

impl Format {
    /// The format `LDBL_MANT_DIG` names, or `None` for one kore has no ABI
    /// with.
    pub const fn from_mant_dig(mant_dig: i32) -> Option<Self> {
        match mant_dig {
            64 => Some(Self::X87),
            113 => Some(Self::Quad),
            _ => None,
        }
    }
}

/// A `long double` value as its little-endian bytes: the significant bits
/// low (x87's ten, binary128's sixteen), any padding zero.
pub const fn to_bytes(bits: u128) -> [u8; BYTES] {
    bits.to_le_bytes()
}

/// The bits of a `long double`'s bytes. x87's six padding bytes are
/// whatever the C side's storage held, so they are cleared here.
pub const fn from_bytes(format: Format, bytes: [u8; BYTES]) -> u128 {
    let bits = u128::from_le_bytes(bytes);
    match format {
        Format::X87 => bits & ((1u128 << 80) - 1),
        Format::Quad => bits,
    }
}

/// What a `strtold` parse found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parsed {
    /// The value, as [`to_bytes`] lays it out.
    pub bits: u128,
    /// Overflow to infinity, or a nonzero number rounded to zero: `ERANGE`.
    pub range: bool,
    /// Bytes consumed; zero when there was no number.
    pub end: usize,
}

/// Parse a `long double` from the start of `text`.
pub fn parse(format: Format, text: &[u8]) -> Parsed {
    match format {
        Format::X87 => parse_as::<X87DoubleExtended>(text),
        Format::Quad => parse_as::<Quad>(text),
    }
}

fn parse_as<F: Float>(text: &[u8]) -> Parsed {
    let (value, range, end) = match float_subject(text) {
        FloatSubject::None => (F::ZERO, false, 0),
        FloatSubject::Special { nan, negative, end } => {
            let value = if nan { F::NAN } else { F::INFINITY };
            (if negative { -value } else { value }, false, end)
        }
        FloatSubject::Decimal {
            negative,
            digits,
            nonzero,
            end,
        } => {
            // The subject is ASCII digits, a `.` and an exponent — always
            // UTF-8, and always inside APFloat's decimal grammar (a subject
            // has at least one digit, and an exponent only with digits).
            let converted = core::str::from_utf8(digits)
                .ok()
                .and_then(|digits| F::from_str_r(digits, Round::NearestTiesToEven).ok());
            let Some(converted) = converted else {
                // Unreachable by the grammar; refuse the conversion rather
                // than invent a value.
                return Parsed {
                    bits: F::ZERO.to_bits(),
                    range: false,
                    end: 0,
                };
            };
            let value = if negative {
                -converted.value
            } else {
                converted.value
            };
            let range =
                converted.status.intersects(Status::OVERFLOW) || (value.is_zero() && nonzero);
            (value, range, end)
        }
    };
    Parsed {
        bits: value.to_bits(),
        range,
        end,
    }
}

/// `ldexpl`: `bits × 2^exponent`, rounded once to nearest-even.
pub fn scale(format: Format, bits: u128, exponent: i32) -> u128 {
    match format {
        Format::X87 => scale_as::<X87DoubleExtended>(bits, exponent),
        Format::Quad => scale_as::<Quad>(bits, exponent),
    }
}

fn scale_as<F: Float>(bits: u128, exponent: i32) -> u128 {
    F::from_bits(bits)
        .scalbn_r(exponent, Round::NearestTiesToEven)
        .to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::vec::Vec;

    // The host's own C library is the reference: on an x86_64 host its
    // `long double` is x87 and glibc's `strtold` rounds correctly, and
    // `strtof128` is binary128.
    unsafe extern "C" {
        fn strtold(text: *const core::ffi::c_char, end: *mut *mut core::ffi::c_char) -> X87Bytes;
        fn ldexpl(value: X87Bytes, exponent: i32) -> X87Bytes;
    }

    /// `long double` as the host ABI moves it: Rust has no x87 type, so
    /// these tests read the host's answer through its bytes.
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct X87Bytes([u8; 16]);

    fn host(text: &str) -> (u128, usize) {
        let c = CString::new(text).unwrap();
        let mut end = core::ptr::null_mut();
        // SAFETY: a NUL-terminated string and a valid end pointer.
        let value = unsafe { host_strtold(c.as_ptr(), &mut end) };
        let consumed = end as usize - c.as_ptr() as usize;
        (from_bytes(Format::X87, value), consumed)
    }

    /// glibc's `strtold` returns in `st(0)`; this trampoline stores it.
    /// The destination rides `r12`, callee-saved, so the call keeps it.
    unsafe fn host_strtold(
        text: *const core::ffi::c_char,
        end: *mut *mut core::ffi::c_char,
    ) -> [u8; 16] {
        let mut out = [0u8; 16];
        // SAFETY: the caller's pointers; `fstp` stores the ten bytes and
        // leaves the x87 stack empty, as the ABI requires.
        unsafe {
            core::arch::asm!(
                "call {f}",
                "fstp tbyte ptr [r12]",
                f = sym strtold,
                in("r12") out.as_mut_ptr(),
                in("rdi") text,
                in("rsi") end,
                clobber_abi("C"),
            );
        }
        out
    }

    fn ours(text: &str) -> (u128, usize) {
        let parsed = parse(Format::X87, text.as_bytes());
        (parsed.bits, parsed.end)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x87_parses_as_the_host_does() {
        let mut corpus: Vec<std::string::String> = [
            "0",
            "1",
            "0.1",
            "1e4932",
            "1e4933",
            "1e-4951",
            "3.6e-4951",
            "3.7e-4951",
            "1.18973149535723176502e4932",
            "1.18973149535723176508e4932",
            "3.36210314311209350626e-4932",
            "6.95e-4950",
            "0.30000000000000000001",
            "123456789012345678901234567890",
            "  -2.5e3x",
            ".5",
            "5.",
            "1e+",
            "inf",
            "-INFINITY",
            "nan",
            "18446744073709551615",
            "18446744073709551616.5",
            "9.999999999999999999999e999",
            "2.2250738585072011e-308",
            "1.00000000000000000005421010862427522170037264004349708557128906250",
            "1.0000000000000000000542101086242752217003726400434970855712890625",
            "1.00000000000000000016263032587282567",
            "4.9406564584124654e-324",
        ]
        .iter()
        .map(|s| std::string::String::from(*s))
        .collect();
        // A deterministic spread of digit strings and exponents.
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..4000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let digits = 1 + (seed % 40) as usize;
            let mut s = std::string::String::new();
            let mut x = seed;
            for i in 0..digits {
                if i == digits / 2 && seed & 1 == 0 {
                    s.push('.');
                }
                s.push((b'0' + (x % 10) as u8) as char);
                x = x / 10 ^ x.rotate_left(29);
            }
            let exp = ((seed >> 20) % 9900) as i64 - 4950;
            s.push_str(&std::format!("e{exp}"));
            corpus.push(s);
        }
        for text in &corpus {
            assert_eq!(ours(text), host(text), "{text}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x87_scales_as_the_host_does() {
        let values = [
            "1",
            "0.1",
            "3.0e-4950",
            "1.18973149535723176502e4932",
            "-7.25",
            "inf",
            "0",
        ];
        for value in values {
            let (bits, _) = host(value);
            for exponent in [
                -20000, -16445, -16444, -16383, -100, -1, 0, 1, 100, 16383, 16384, 20000,
            ] {
                let x = X87Bytes(to_bytes(bits));
                // SAFETY: glibc's `ldexpl` on an x87 value passed in memory.
                let expected = unsafe { host_ldexpl(x, exponent) };
                assert_eq!(
                    scale(Format::X87, bits, exponent),
                    from_bytes(Format::X87, expected),
                    "{value} {exponent}"
                );
            }
        }
    }

    /// `ldexpl` takes its `long double` on the stack and returns in `st(0)`.
    unsafe fn host_ldexpl(value: X87Bytes, exponent: i32) -> [u8; 16] {
        let mut out = [0u8; 16];
        // SAFETY: SysV passes an x87 argument in memory, in order, aligned
        // 16 (an asm block without `nostack` enters call-aligned); the
        // destination rides callee-saved `r12` across the call.
        unsafe {
            core::arch::asm!(
                "sub rsp, 16",
                "movdqu xmm0, [{value}]",
                "movdqu [rsp], xmm0",
                "call {f}",
                "add rsp, 16",
                "fstp tbyte ptr [r12]",
                f = sym ldexpl,
                value = in(reg) value.0.as_ptr(),
                in("r12") out.as_mut_ptr(),
                in("edi") exponent,
                clobber_abi("C"),
            );
        }
        out
    }

    #[test]
    fn quad_parses_round_numbers_exactly() {
        // binary128 of 1.0: biased exponent 0x3fff, zero fraction.
        assert_eq!(parse(Format::Quad, b"1").bits, 0x3fff_u128 << 112);
        assert_eq!(parse(Format::Quad, b"-2").bits, (0xc000_u128) << 112);
        // 0.1 is the nearest binary128, rounded up in its last bit.
        assert_eq!(
            parse(Format::Quad, b"0.1").bits,
            0x3ffb_9999_9999_9999_9999_9999_9999_999a
        );
        let huge = parse(Format::Quad, b"1e5000");
        assert!(huge.range && huge.bits == 0x7fff_u128 << 112);
        let tiny = parse(Format::Quad, b"1e-5000");
        assert!(tiny.range && tiny.bits == 0);
        assert_eq!(
            scale(Format::Quad, parse(Format::Quad, b"1").bits, 3),
            parse(Format::Quad, b"8").bits
        );
    }

    #[test]
    fn formats_are_named_by_mant_dig() {
        assert_eq!(Format::from_mant_dig(64), Some(Format::X87));
        assert_eq!(Format::from_mant_dig(113), Some(Format::Quad));
        assert_eq!(Format::from_mant_dig(53), None);
        // x87's padding never leaks into the value.
        assert_eq!(from_bytes(Format::X87, [0xff; 16]), (1u128 << 80) - 1);
    }
}
