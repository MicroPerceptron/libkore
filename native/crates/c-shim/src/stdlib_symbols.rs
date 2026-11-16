//! The `<stdlib.h>` and `<string.h>` routines that are pure computation:
//! number parsing, `qsort`, `strerror`.
//!
//! The parsers set `errno` as ISO C requires, so they sit with it behind the
//! `runtime` feature; `qsort` and `strerror` are pure but travel with them.

use core::ffi::{
    c_char, c_double, c_float, c_int, c_long, c_longlong, c_ulong, c_ulonglong, c_void,
};

use crate::errno::{self, EINVAL, ERANGE};
use crate::number;

/// The bytes before `text`'s NUL.
///
/// # Safety
/// `text` is NUL-terminated.
unsafe fn bytes<'a>(text: *const c_char) -> &'a [u8] {
    // SAFETY: the caller's contract.
    unsafe { core::ffi::CStr::from_ptr(text) }.to_bytes()
}

/// Store `text + end` through `end_out`, as the parsers' `endptr`.
///
/// # Safety
/// `end_out` is null or writable; `end` lies within `text`'s string.
unsafe fn report_end(text: *const c_char, end: usize, end_out: *mut *mut c_char) {
    if !end_out.is_null() {
        // SAFETY: the caller's contract.
        unsafe { *end_out = text.add(end).cast_mut() };
    }
}

/// The shared front of the integer parsers: parse, report the end, and set
/// `EINVAL` for a base C does not define.
///
/// # Safety
/// C's `strtol` contract.
unsafe fn parse(
    text: *const c_char,
    end_out: *mut *mut c_char,
    base: c_int,
) -> Option<number::Integer> {
    // SAFETY: the caller's contract.
    let subject = unsafe { bytes(text) };
    let parsed = u32::try_from(base)
        .ok()
        .and_then(|base| number::integer(subject, base));
    let end = parsed.map_or(0, |parsed| parsed.end);
    // SAFETY: the end lies within the string.
    unsafe { report_end(text, end, end_out) };
    if parsed.is_none() {
        errno::set(EINVAL);
    }
    parsed
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strtoll(
    text: *const c_char,
    end: *mut *mut c_char,
    base: c_int,
) -> c_longlong {
    // SAFETY: C's contract.
    let Some(parsed) = (unsafe { parse(text, end, base) }) else {
        return 0;
    };
    let (value, clamped) = parsed.signed(c_longlong::BITS);
    if clamped {
        errno::set(ERANGE);
    }
    value
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strtol(text: *const c_char, end: *mut *mut c_char, base: c_int) -> c_long {
    // SAFETY: C's contract.
    let Some(parsed) = (unsafe { parse(text, end, base) }) else {
        return 0;
    };
    let (value, clamped) = parsed.signed(c_long::BITS);
    if clamped {
        errno::set(ERANGE);
    }
    value as c_long
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strtoull(
    text: *const c_char,
    end: *mut *mut c_char,
    base: c_int,
) -> c_ulonglong {
    // SAFETY: C's contract.
    let Some(parsed) = (unsafe { parse(text, end, base) }) else {
        return 0;
    };
    let (value, clamped) = parsed.unsigned(c_ulonglong::BITS);
    if clamped {
        errno::set(ERANGE);
    }
    value
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strtoul(text: *const c_char, end: *mut *mut c_char, base: c_int) -> c_ulong {
    // SAFETY: C's contract.
    let Some(parsed) = (unsafe { parse(text, end, base) }) else {
        return 0;
    };
    let (value, clamped) = parsed.unsigned(c_ulong::BITS);
    if clamped {
        errno::set(ERANGE);
    }
    value as c_ulong
}

/// `atoi`/`atol`: `strtol` in base 10, as glibc defines them — out-of-range
/// input is undefined in C; here it is `strtol`'s clamp, truncated.
#[unsafe(no_mangle)]
unsafe extern "C" fn atoi(text: *const c_char) -> c_int {
    // SAFETY: C's contract.
    unsafe { strtol(text, core::ptr::null_mut(), 10) as c_int }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn atol(text: *const c_char) -> c_long {
    // SAFETY: C's contract.
    unsafe { strtol(text, core::ptr::null_mut(), 10) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strtod(text: *const c_char, end: *mut *mut c_char) -> c_double {
    // SAFETY: C's contract.
    let parsed = number::double(unsafe { bytes(text) });
    unsafe { report_end(text, parsed.end, end) };
    if parsed.range {
        errno::set(ERANGE);
    }
    parsed.value
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strtof(text: *const c_char, end: *mut *mut c_char) -> c_float {
    // SAFETY: C's contract.
    let parsed = number::single(unsafe { bytes(text) });
    unsafe { report_end(text, parsed.end, end) };
    if parsed.range {
        errno::set(ERANGE);
    }
    parsed.value
}

/// A C comparator.
type Compare = unsafe extern "C" fn(*const c_void, *const c_void) -> c_int;

#[unsafe(no_mangle)]
unsafe extern "C" fn qsort(base: *mut c_void, count: usize, size: usize, compare: Compare) {
    // SAFETY: C's contract: `count * size` bytes at `base`, and a comparator
    // that reads only the elements it is handed.
    unsafe {
        crate::sort::heapsort(base.cast(), count, size, &mut |a, b| {
            compare(a.cast(), b.cast()) > 0
        })
    };
}

#[unsafe(no_mangle)]
extern "C" fn strerror(code: c_int) -> *mut c_char {
    errno::describe(code).as_ptr().cast_mut()
}

/// `strtold`'s byte-level half (`long_double`): parse `text` as the
/// `long double` `mant_dig` names, store its [`long_double::BYTES`] bytes at
/// `out`, and the end at `*end`. Answers the `errno` the C wrapper sets —
/// `ERANGE` as `strtod` would, `EINVAL` for a format or buffer this crate
/// has no ABI for (the value is then `+0` and nothing is consumed) — or `0`.
///
/// # Safety
/// C's `strtold` contract for `text` and `end`; `out` is writable for
/// `out_len` bytes.
#[cfg(feature = "long-double")]
#[unsafe(no_mangle)]
unsafe extern "C" fn __kore_strtold_bytes(
    text: *const c_char,
    end: *mut *mut c_char,
    mant_dig: c_int,
    out: *mut u8,
    out_len: usize,
) -> c_int {
    use crate::long_double;
    let Some(format) =
        long_double::Format::from_mant_dig(mant_dig).filter(|_| out_len == long_double::BYTES)
    else {
        // SAFETY: C's contract.
        unsafe { report_end(text, 0, end) };
        if !out.is_null() {
            // SAFETY: `out_len` writable bytes.
            unsafe { core::ptr::write_bytes(out, 0, out_len) };
        }
        return EINVAL;
    };
    // SAFETY: C's contract.
    let parsed = long_double::parse(format, unsafe { bytes(text) });
    unsafe {
        report_end(text, parsed.end, end);
        out.cast::<[u8; long_double::BYTES]>()
            .write_unaligned(long_double::to_bytes(parsed.bits));
    }
    if parsed.range { ERANGE } else { 0 }
}

/// `ldexpl`'s byte-level half: replace the `long double` at `value` with
/// itself times `2^exponent`, rounded once. `EINVAL` (value untouched) for a
/// format or buffer this crate has no ABI for, else `0`.
///
/// # Safety
/// `value` is readable and writable for `len` bytes.
#[cfg(feature = "long-double")]
#[unsafe(no_mangle)]
unsafe extern "C" fn __kore_ldexpl_bytes(
    value: *mut u8,
    len: usize,
    mant_dig: c_int,
    exponent: c_int,
) -> c_int {
    use crate::long_double;
    let Some(format) =
        long_double::Format::from_mant_dig(mant_dig).filter(|_| len == long_double::BYTES)
    else {
        return EINVAL;
    };
    let slot = value.cast::<[u8; long_double::BYTES]>();
    // SAFETY: the caller's `len` bytes.
    let bits = long_double::from_bytes(format, unsafe { slot.read_unaligned() });
    let scaled = long_double::scale(format, bits, exponent);
    // SAFETY: as above.
    unsafe { slot.write_unaligned(long_double::to_bytes(scaled)) };
    0
}
