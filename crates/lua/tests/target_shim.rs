//! Host harness for the exact freestanding numeric shim used by the target.

#[path = "../src/numeric.rs"]
mod numeric;

use core::ffi::{c_char, c_double, c_int};
use std::{ffi::CStr, ptr};

unsafe extern "C" {
    fn snprintf(dst: *mut c_char, cap: usize, format: *const c_char, ...) -> c_int;
    #[cfg(kore_lua_c_engine)]
    fn kore_lua_host_smoke() -> c_int;
}

fn target_format(value: f64, precision: usize) -> String {
    let mut bytes = [0xa5u8; 80];
    // SAFETY: `bytes` is a writable destination of the supplied capacity.
    let len = unsafe {
        numeric::kore_lua_format_number(bytes.as_mut_ptr().cast(), 64, value as c_double, precision)
    };
    assert!(len < 64, "formatter returned an unstored length {len}");
    assert_eq!(bytes[len], 0, "formatter omitted its terminator");
    assert!(bytes[64..].iter().all(|&byte| byte == 0xa5));
    std::str::from_utf8(&bytes[..len]).unwrap().to_owned()
}

fn libc_format(value: f64, precision: usize) -> String {
    let mut bytes = [0 as c_char; 64];
    // SAFETY: the format has the matching promoted `int`, `double` arguments
    // and `bytes` is a writable C destination.
    let len = unsafe {
        snprintf(
            bytes.as_mut_ptr(),
            bytes.len(),
            b"%.*g\0".as_ptr().cast(),
            precision as c_int,
            value as c_double,
        )
    };
    assert!(len >= 0 && (len as usize) < bytes.len());
    // SAFETY: successful snprintf always terminates this sufficiently large
    // destination.
    unsafe { CStr::from_ptr(bytes.as_ptr()) }
        .to_str()
        .unwrap()
        .to_owned()
}

fn target_parse(text: &str) -> (f64, usize) {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    let mut end = ptr::null_mut();
    // SAFETY: `bytes` is NUL-terminated and `end` is writable.
    let value = unsafe { numeric::kore_lua_strtod(bytes.as_ptr().cast(), &mut end) as f64 };
    let offset = unsafe { end.offset_from(bytes.as_ptr().cast_mut().cast::<c_char>()) };
    assert!(offset >= 0);
    (value, offset as usize)
}

fn same_lua_number(left: f64, right: f64) -> bool {
    left == right || (left.is_nan() && right.is_nan())
}

fn assert_format_matches_libc(value: f64) {
    for precision in [15, 17] {
        let target = target_format(value, precision);
        let libc = libc_format(value, precision);
        assert_eq!(
            target,
            libc,
            "format mismatch for bits={:016x}, precision={precision}",
            value.to_bits()
        );
    }

    let first = target_format(value, 15);
    let (check, consumed) = target_parse(&first);
    if value.is_finite() {
        assert_eq!(consumed, first.len());
    }
    let target_two_pass = if check != value {
        target_format(value, 17)
    } else {
        first
    };

    let libc_first = libc_format(value, 15);
    let (check, consumed) = target_parse(&libc_first);
    if value.is_finite() {
        assert_eq!(consumed, libc_first.len());
    }
    let libc_two_pass = if check != value {
        libc_format(value, 17)
    } else {
        libc_first
    };
    assert_eq!(
        target_two_pass,
        libc_two_pass,
        "two-pass mismatch for bits={:016x}",
        value.to_bits()
    );
}

fn main() {
    let corpus = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        42.5,
        1e-4,
        1e-5,
        1e14,
        1e15,
        1e16,
        1e100,
        1e308,
        f64::from_bits(1),
        f64::from_bits(2),
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x3fefffffffffffff),
        f64::from_bits(0x3ff0000000000001),
        999_999_999_999_999.9,
        1_234_567_890_123_456.0,
        12_345_678_901_234_568.0,
        f64::from_bits(0xc366ae2c20764753),
        f64::from_bits(0xce6336ec93771f1d),
    ];
    for value in corpus {
        assert_format_matches_libc(value);
    }

    assert_eq!(target_format(f64::NAN, 15), "nan");
    assert_eq!(target_format(f64::INFINITY, 15), "inf");
    assert_eq!(target_format(f64::NEG_INFINITY, 15), "-inf");
    assert_eq!(target_format(-0.0, 15), "-0");
    assert_eq!(target_format(1e-5, 15), "1e-05");
    assert_eq!(target_format(1e15, 15), "1e+15");
    assert_eq!(target_format(1e100, 15), "1e+100");
    assert_eq!(target_format(1e308, 15), "1e+308");
    assert_eq!(
        target_format(f64::from_bits(1), 15),
        "4.94065645841247e-324"
    );
    let transition = 1_234_567_890_123_456.0;
    let first = target_format(transition, 15);
    assert_eq!(first, "1.23456789012346e+15");
    assert_ne!(target_parse(&first).0, transition);
    assert_eq!(target_format(transition, 17), "1234567890123456");

    /* A fixed-seed corpus catches rounding and notation threshold drift. */
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..100_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = f64::from_bits(state);
        if !value.is_nan() {
            assert_format_matches_libc(value);
        }
    }

    /* Precision 17 must round-trip every finite binary64 through our parser. */
    state = 0x1319_8a2e_0370_7344;
    for _ in 0..100_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = f64::from_bits(state);
        if value.is_finite() {
            let text = target_format(value, 17);
            let (parsed, consumed) = target_parse(&text);
            assert_eq!(consumed, text.len());
            assert!(
                same_lua_number(parsed, value),
                "round-trip mismatch: bits={:016x}, text={text}, parsed={:016x}",
                value.to_bits(),
                parsed.to_bits()
            );
        }
    }

    /* The FFI length is always initialized length, even under truncation. */
    let mut short = [0xa5u8; 8];
    // SAFETY: each nonzero capacity describes a prefix of `short`.
    unsafe {
        assert_eq!(
            numeric::kore_lua_format_number(ptr::null_mut(), 0, 1e100, 15),
            0
        );
        assert_eq!(
            numeric::kore_lua_format_number(short.as_mut_ptr().cast(), 0, 1e100, 15),
            0
        );
        assert!(short.iter().all(|&byte| byte == 0xa5));
        assert_eq!(
            numeric::kore_lua_format_number(short.as_mut_ptr().cast(), 1, 1e100, 15),
            0
        );
        assert_eq!(short[0], 0);
        short.fill(0xa5);
        assert_eq!(
            numeric::kore_lua_format_number(short.as_mut_ptr().cast(), 5, 1e100, 15),
            4
        );
    }
    assert_eq!(&short[..5], b"1e+1\0");
    assert!(short[5..].iter().all(|&byte| byte == 0xa5));

    /* Public tonumber strings are not limited to the lexer token ceiling. */
    let mut long_number = " ".repeat(400);
    long_number.push_str("-1.25e2tail");
    let (parsed, consumed) = target_parse(&long_number);
    assert_eq!(parsed, -125.0);
    assert_eq!(consumed, 407);

    let parse_cases: [(&str, f64, usize); 6] = [
        ("  +12.5tail", 12.5, 7),
        ("\x0b1.5tail", 1.5, 4),
        (".5x", 0.5, 2),
        ("1e+tail", 1.0, 1),
        ("-0", -0.0, 2),
        ("not-a-number", 0.0, 0),
    ];
    for (text, expected, consumed) in parse_cases {
        let (parsed, actual_consumed) = target_parse(text);
        assert_eq!(
            parsed.to_bits(),
            expected.to_bits(),
            "parse mismatch: {text}"
        );
        assert_eq!(actual_consumed, consumed, "end mismatch: {text}");
    }

    /* Link the C engine against these exact exports in the build-script run. */
    #[cfg(kore_lua_c_engine)]
    // SAFETY: the linked host harness takes no arguments and owns its VM.
    assert_eq!(unsafe { kore_lua_host_smoke() }, 0);
}
