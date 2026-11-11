//! The float conversions against the host C library's own. glibc's
//! `snprintf` is the reference for `double`, through every flag, width and
//! precision. For the two `long double` formats it is glibc's `strfroml`
//! (x87, on an x86_64 host) and `strfromf128` (binary128), reached through
//! trampolines: Rust has no x87 type to pass, nor a binary128 one that
//! crosses into C. Where musl and glibc disagree — `%a` of a subnormal, of an
//! x87 value, and x87's invalid encodings — the cases are musl's spellings,
//! written out.

extern crate std;

use core::ffi::{c_char, c_int};
use std::ffi::CString;
use std::string::String;
use std::vec::Vec;

use super::super::{Args, Length, Sink, format};
use super::{DOUBLE_LIMBS, Decimal, Float, LONG_DOUBLE_LIMBS, window};

/// The one float argument a test format converts.
enum Value {
    Double(f64),
    Long(Float),
}

struct One(Option<Value>);

impl Args for One {
    fn signed(&mut self, _: Length) -> i64 {
        unreachable!("a float test formats one float")
    }
    fn unsigned(&mut self, _: Length) -> u64 {
        unreachable!("a float test formats one float")
    }
    fn int(&mut self) -> c_int {
        unreachable!("a float test formats one float")
    }
    fn pointer(&mut self) -> usize {
        unreachable!("a float test formats one float")
    }
    fn double(&mut self) -> f64 {
        match self.0.take() {
            Some(Value::Double(value)) => value,
            _ => panic!("the format read a double it was not given"),
        }
    }
    fn long_double(&mut self) -> Float {
        match self.0.take() {
            Some(Value::Long(value)) => value,
            _ => panic!("the format read a long double it was not given"),
        }
    }
}

struct Collect(Vec<u8>);

impl Sink for Collect {
    fn put(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
}

fn ours(spec: &str, value: Value) -> String {
    let spec = CString::new(spec).unwrap();
    let mut args = One(Some(value));
    let mut sink = Collect(Vec::new());
    // SAFETY: NUL-terminated, and the one conversion matches the argument.
    let count = unsafe { format(&mut sink, spec.as_ptr().cast(), &mut args) };
    assert!(args.0.is_none(), "the format did not read its argument");
    assert_eq!(count, sink.0.len(), "the count is the output's length");
    String::from_utf8(sink.0).unwrap()
}

/// Room for the widest `%f` of a `long double` with a long precision.
const HOST_BUFFER: usize = 1 << 14;

unsafe extern "C" {
    fn snprintf(buffer: *mut c_char, size: usize, format: *const c_char, ...) -> c_int;
}

fn host_double(spec: &str, value: f64) -> String {
    let spec = CString::new(spec).unwrap();
    let mut buffer = std::vec![0u8; HOST_BUFFER];
    // SAFETY: a buffer of its stated size and a matching double.
    let written = unsafe {
        snprintf(
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            spec.as_ptr(),
            value,
        )
    };
    let written = usize::try_from(written).unwrap();
    assert!(written < buffer.len());
    String::from_utf8(buffer[..written].to_vec()).unwrap()
}

/// xorshift64*: fixed seeds, so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// Values the conversions turn on: zeros and ones, exact ties at every
/// place, the powers of ten either side of a rounding carry, the subnormal
/// and normal edges, the largest finite value, and the non-finite ones.
fn edge_doubles() -> Vec<f64> {
    let mut values = std::vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        1.5,
        2.5,
        3.5,
        -2.5,
        0.125,
        0.375,
        0.625,
        0.05,
        0.15,
        0.25,
        0.35,
        9.5,
        10.5,
        1e-7,
        1e-5,
        1e-4,
        0.000099999995,
        9.9999995,
        99999.95,
        999999.5,
        123456.0,
        1234567.0,
        1e15,
        1e16,
        1e17,
        1e21,
        1e22,
        1e23,
        1e100,
        9007199254740992.0,
        9007199254740994.0,
        1.0 / 3.0,
        2.0 / 3.0,
        0.1,
        0.2,
        0.3,
        f64::EPSILON,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(3),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
    ];
    for exponent in -30..30 {
        values.push(10f64.powi(exponent));
        values.push(5.0 * 10f64.powi(exponent));
    }
    values
}

/// Random bit patterns (every exponent, so mostly huge and tiny values)
/// and random values of everyday size.
fn random_doubles(count: usize) -> Vec<f64> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut values = Vec::with_capacity(2 * count);
    for _ in 0..count {
        values.push(f64::from_bits(rng.next()));
        let everyday = (rng.next() & 0x800f_ffff_ffff_ffff) | ((1023 - 40 + rng.next() % 80) << 52);
        values.push(f64::from_bits(everyday));
    }
    values
}

const DOUBLE_SPECS: &[&str] = &[
    "%e",
    "%.0e",
    "%.1e",
    "%.3e",
    "%.16e",
    "%.17e",
    "%.40e",
    "%E",
    "%#.0e",
    "%+e",
    "% .2e",
    "[%15.3e]",
    "[%-15.3e]",
    "%015.3e",
    "%f",
    "%.0f",
    "%.1f",
    "%.3f",
    "%.20f",
    "%.1100f",
    "%F",
    "%#.0f",
    "%+08.2f",
    "[%-12.4f]",
    "%g",
    "%.0g",
    "%.1g",
    "%.2g",
    "%.10g",
    "%.17g",
    "%.40g",
    "%G",
    "%+g",
    "%012g",
    "[%-12g]",
    "%a",
    "%.0a",
    "%.1a",
    "%.3a",
    "%.13a",
    "%.20a",
    "%A",
    "%#.0a",
    "%+a",
    "%020a",
    "[%-20a]",
];

/// `%a` is one conversion musl and glibc spell differently for a `double`:
/// glibc keeps a subnormal's leading `0`.
fn hex_differs(spec: &str, value: f64) -> bool {
    spec.contains(['a', 'A']) && value != 0.0 && value.abs() < f64::MIN_POSITIVE
}

/// `%#g` is the other: when the rounding carries the exponent up to the
/// precision, glibc drops the trailing zeros `#` keeps (musl keeps them, as
/// C 7.21.6.1p8 says). The reference is built from glibc's `%e` and `%#f` or
/// `%#e`, as that paragraph defines `%g`.
const ALTERNATE_GENERAL: &[(&str, Option<i64>)] =
    &[("%#g", None), ("%#.3g", Some(3)), ("%#.0g", Some(0))];

fn host_alternate_general(precision: Option<i64>, value: f64) -> String {
    if !value.is_finite() {
        return host_double("%#g", value);
    }
    let significant = match precision {
        Some(0) => 1,
        Some(precision) => precision,
        None => 6,
    };
    let scientific = host_double(&std::format!("%.{}e", significant - 1), value);
    let exponent: i64 = scientific.rsplit_once('e').unwrap().1.parse().unwrap();
    if significant > exponent && exponent >= -4 {
        host_double(&std::format!("%#.{}f", significant - 1 - exponent), value)
    } else {
        host_double(&std::format!("%#.{}e", significant - 1), value)
    }
}

#[test]
fn doubles_print_as_the_host_prints_them() {
    let mut values = edge_doubles();
    values.extend(random_doubles(1500));
    for &value in &values {
        for &spec in DOUBLE_SPECS {
            if hex_differs(spec, value) {
                continue;
            }
            assert_eq!(
                ours(spec, Value::Double(value)),
                host_double(spec, value),
                "{spec} of {value:e} ({:#018x})",
                value.to_bits()
            );
        }
        for &(spec, precision) in ALTERNATE_GENERAL {
            assert_eq!(
                ours(spec, Value::Double(value)),
                host_alternate_general(precision, value),
                "{spec} of {value:e} ({:#018x})",
                value.to_bits()
            );
        }
    }
}

#[test]
fn an_alternate_general_keeps_its_zeros_through_a_carry_as_musl_does() {
    let cases: &[(&str, f64, &str)] = &[
        ("%#g", 999999.5, "1.00000e+06"),
        ("%#.3g", 999.5, "1.00e+03"),
        ("%#.2g", 99.5, "1.0e+02"),
        ("%#.1g", 9.5, "1.e+01"),
        ("%#.0g", 9.5, "1.e+01"),
        ("%#.6g", 0.0009999995, "0.00100000"),
    ];
    for &(spec, value, expected) in cases {
        assert_eq!(
            ours(spec, Value::Double(value)),
            expected,
            "{spec} of {value:e}"
        );
    }
}

#[test]
fn a_subnormal_double_in_hex_is_normalized_as_musl_spells_it() {
    let cases: &[(&str, u64, &str)] = &[
        ("%a", 1, "0x1p-1074"),
        ("%a", 3, "0x1.8p-1073"),
        ("%.0a", 3, "0x2p-1073"),
        ("%a", 0x000f_ffff_ffff_ffff, "0x1.ffffffffffffep-1023"),
        ("%.1A", 0x0008_0000_0000_0001, "0X1.0P-1023"),
    ];
    for &(spec, bits, expected) in cases {
        assert_eq!(
            ours(spec, Value::Double(f64::from_bits(bits))),
            expected,
            "{spec} of {bits:#x}"
        );
    }
}

#[test]
fn every_format_holds_its_widest_expansion() {
    // The widest integer part, built whole by doubling.
    let max = Decimal::<DOUBLE_LIMBS>::new((1 << 53) - 1, 971, 0);
    assert_eq!(max.leading(), 308);
    // The longest fraction a window can reach, whatever the precision.
    let least = Decimal::<DOUBLE_LIMBS>::new(1, -1074, window(-1_000_000, -1074));
    assert_eq!(least.leading(), -324);
    assert_eq!(
        (-339..=-324)
            .rev()
            .map(|p| least.digit(p))
            .collect::<Vec<_>>(),
        [4, 9, 4, 0, 6, 5, 6, 4, 5, 8, 4, 1, 2, 4, 6, 5]
    );
    assert!(!least.sticky, "nothing of the expansion was let go");
    assert_eq!(least.lowest(), Some(-1074));
    let quad_max = Decimal::<LONG_DOUBLE_LIMBS>::new((1 << 113) - 1, 16271, 0);
    assert_eq!(quad_max.leading(), 4932);
    let quad_least = Decimal::<LONG_DOUBLE_LIMBS>::new(1, -16494, window(-1_000_000, -16494));
    assert_eq!(quad_least.leading(), -4966);
    assert_eq!(quad_least.lowest(), Some(-16494));
    let x87_least = Decimal::<LONG_DOUBLE_LIMBS>::new(1, -16445, window(-1_000_000, -16445));
    assert_eq!(x87_least.lowest(), Some(-16445));
}

#[test]
fn a_huge_precision_prints_its_zeros_without_storing_them() {
    let text = ours("%.100000f", Value::Double(0.5));
    assert_eq!(text.len(), 2 + 100_000);
    assert!(text.starts_with("0.5000"));
    assert!(text[3..].bytes().all(|b| b == b'0'));
}

/// The `(precision, conversion)` pairs `strfrom*` accepts, which name no
/// flag or width: those are the engine's own, which the `double` sweep has
/// covered.
const WIDE_SPECS: &[(Option<u32>, char)] = &[
    (None, 'e'),
    (Some(0), 'e'),
    (Some(1), 'e'),
    (Some(5), 'e'),
    (Some(18), 'e'),
    (Some(25), 'e'),
    (Some(36), 'e'),
    (Some(60), 'E'),
    (None, 'f'),
    (Some(0), 'f'),
    (Some(3), 'f'),
    (Some(30), 'f'),
    (Some(60), 'F'),
    (None, 'g'),
    (Some(0), 'g'),
    (Some(1), 'g'),
    (Some(5), 'g'),
    (Some(18), 'g'),
    (Some(21), 'g'),
    (Some(36), 'G'),
    (Some(50), 'g'),
];

fn wide_spec(precision: Option<u32>, conversion: char, long: bool) -> String {
    let length = if long { "L" } else { "" };
    match precision {
        Some(precision) => std::format!("%.{precision}{length}{conversion}"),
        None => std::format!("%{length}{conversion}"),
    }
}

/// Random `long double` encodings of `format`: every exponent, every
/// everyday one, and the subnormals; each a valid encoding.
fn random_wide(rng: &mut Rng, x87: bool) -> u128 {
    let sign = u128::from(rng.next() >> 63) << if x87 { 79 } else { 127 };
    let roll = rng.next() % 8;
    let biased: u128 = match roll {
        0 => 0,
        1..=3 => u128::from(16383 - 200 + rng.next() % 400),
        _ => u128::from(1 + rng.next() % 0x7ffd),
    };
    let fraction = (u128::from(rng.next()) << 64) | u128::from(rng.next());
    if x87 {
        let integer = u128::from(biased != 0) << 63;
        sign | biased << 64 | integer | (fraction & ((1 << 63) - 1))
    } else {
        sign | biased << 112 | (fraction & ((1 << 112) - 1))
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64_host {
    use super::*;

    unsafe extern "C" {
        fn strfroml();
        fn strfromf128();
    }

    /// glibc's `strfroml(buffer, size, format, value)`: the x87 value rides
    /// the stack, as System V passes every `long double`.
    fn host_x87(spec: &str, bits: u128) -> String {
        let spec = CString::new(spec).unwrap();
        let mut buffer = std::vec![0u8; HOST_BUFFER];
        let value = bits.to_le_bytes();
        let written: i32;
        // SAFETY: the stack stays 16-aligned for the call (`asm!` enters
        // aligned and the frame is sixteen bytes), the arguments are live,
        // and every caller-saved register is declared clobbered.
        unsafe {
            core::arch::asm!(
                "sub rsp, 16",
                "mov rax, qword ptr [r10]",
                "mov qword ptr [rsp], rax",
                "mov rax, qword ptr [r10 + 8]",
                "mov qword ptr [rsp + 8], rax",
                "call {f}",
                "add rsp, 16",
                f = sym strfroml,
                in("r10") value.as_ptr(),
                in("rdi") buffer.as_mut_ptr(),
                in("rsi") buffer.len(),
                in("rdx") spec.as_ptr(),
                lateout("eax") written,
                clobber_abi("C"),
            );
        }
        let written = usize::try_from(written).unwrap();
        assert!(written < buffer.len());
        String::from_utf8(buffer[..written].to_vec()).unwrap()
    }

    /// glibc's `strfromf128(buffer, size, format, value)`: `_Float128`
    /// rides `xmm0`.
    fn host_quad(spec: &str, bits: u128) -> String {
        let spec = CString::new(spec).unwrap();
        let mut buffer = std::vec![0u8; HOST_BUFFER];
        let value = bits.to_le_bytes();
        let written: i32;
        // SAFETY: as `host_x87`; the value is loaded into `xmm0` from live
        // bytes, and `xmm0` is caller-saved.
        unsafe {
            core::arch::asm!(
                "movdqu xmm0, xmmword ptr [r10]",
                "call {f}",
                f = sym strfromf128,
                in("r10") value.as_ptr(),
                in("rdi") buffer.as_mut_ptr(),
                in("rsi") buffer.len(),
                in("rdx") spec.as_ptr(),
                lateout("eax") written,
                clobber_abi("C"),
            );
        }
        let written = usize::try_from(written).unwrap();
        assert!(written < buffer.len());
        String::from_utf8(buffer[..written].to_vec()).unwrap()
    }

    fn edge_wide(x87: bool) -> Vec<u128> {
        if x87 {
            std::vec![
                0,
                1 << 79,
                0x3fff_8000_0000_0000_0000,
                0x3ffb_cccc_cccc_cccc_cccd,
                0x4000_a000_0000_0000_0000,
                0x7ffe_ffff_ffff_ffff_ffff,
                0x0001_8000_0000_0000_0000,
                0x0000_0000_0000_0000_0001,
                0x0000_7fff_ffff_ffff_ffff,
                0x7fff_8000_0000_0000_0000,
                0xffff_8000_0000_0000_0000,
                0x7fff_c000_0000_0000_0000,
            ]
        } else {
            std::vec![
                0,
                1 << 127,
                0x3fff_0000_0000_0000_0000_0000_0000_0000,
                0x3ffb_9999_9999_9999_9999_9999_9999_999a,
                0x7ffe_ffff_ffff_ffff_ffff_ffff_ffff_ffff,
                0x0001_0000_0000_0000_0000_0000_0000_0000,
                0x0000_0000_0000_0000_0000_0000_0000_0001,
                0x7fff_0000_0000_0000_0000_0000_0000_0000,
                0xffff_0000_0000_0000_0000_0000_0000_0000,
                0x7fff_8000_0000_0000_0000_0000_0000_0000,
            ]
        }
    }

    fn sweep(x87: bool) {
        let mut rng = Rng(if x87 {
            0x0123_4567_89ab_cdef
        } else {
            0xfedc_ba98_7654_3210
        });
        let mut values = edge_wide(x87);
        values.extend((0..400).map(|_| random_wide(&mut rng, x87)));
        for &bits in &values {
            let value = if x87 {
                Float::from_x87(bits)
            } else {
                Float::from_quad(bits)
            };
            for &(precision, conversion) in WIDE_SPECS {
                let host = if x87 {
                    host_x87(&wide_spec(precision, conversion, false), bits)
                } else {
                    host_quad(&wide_spec(precision, conversion, false), bits)
                };
                let spec = wide_spec(precision, conversion, true);
                assert_eq!(
                    ours(&spec, Value::Long(value)),
                    host,
                    "{spec} of {bits:#034x}"
                );
            }
            // glibc and musl agree on `%a` of a normal binary128 value only.
            let subnormal = (bits >> 112) & 0x7fff == 0 && bits << 1 != 0;
            if !x87 && !subnormal {
                for spec in ["%a", "%.0a", "%.5a", "%.28a", "%.40A"] {
                    let long = spec.replace('a', "La").replace('A', "LA");
                    assert_eq!(
                        ours(&long, Value::Long(value)),
                        host_quad(spec, bits),
                        "{long} of {bits:#034x}"
                    );
                }
            }
        }
    }

    #[test]
    fn x87_long_doubles_print_as_the_host_prints_them() {
        sweep(true);
    }

    #[test]
    fn binary128_long_doubles_print_as_the_host_prints_them() {
        sweep(false);
    }
}

#[test]
fn an_x87_value_in_hex_is_normalized_as_musl_spells_it() {
    let cases: &[(u128, &str)] = &[
        (0x3fff_8000_0000_0000_0000, "0x1p+0"),
        (0x3ffb_cccc_cccc_cccc_cccd, "0x1.999999999999999ap-4"),
        (0x7ffe_ffff_ffff_ffff_ffff, "0x1.fffffffffffffffep+16383"),
        (0x0000_0000_0000_0000_0001, "0x1p-16445"),
        (0xbfff_c000_0000_0000_0000, "-0x1.8p+0"),
    ];
    for &(bits, expected) in cases {
        assert_eq!(
            ours("%La", Value::Long(Float::from_x87(bits))),
            expected,
            "{bits:#x}"
        );
    }
}

#[test]
fn x87_encodings_no_fpu_produces_are_read_as_musl_reads_them() {
    let spell = |bits: u128| ours("%Lg", Value::Long(Float::from_x87(bits)));
    // An unnormal: a clear integer bit under a nonzero exponent.
    assert_eq!(spell(0x3fff_4000_0000_0000_0000), "nan");
    // A pseudo-NaN and a pseudo-infinity: the top exponent, integer bit clear.
    assert_eq!(spell(0x7fff_4000_0000_0000_0000), "nan");
    assert_eq!(spell(0xffff_0000_0000_0000_0000), "-inf");
    // A pseudo-denormal is the value its bits spell: the least normal.
    let least_normal = ours(
        "%.30Le",
        Value::Long(Float::from_x87(0x0001_8000_0000_0000_0000)),
    );
    assert_eq!(
        ours(
            "%.30Le",
            Value::Long(Float::from_x87(0x0000_8000_0000_0000_0000))
        ),
        least_normal
    );
    // The padding above the 80 bits is not the value's.
    assert_eq!(
        ours(
            "%Lg",
            Value::Long(Float::from_x87(0xdead_0000_3fff_8000_0000_0000_0000))
        ),
        "1"
    );
}
