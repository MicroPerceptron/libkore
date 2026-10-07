//! Freestanding decimal parsing and correctly rounded `%g` formatting.

use core::{
    ffi::{c_char, c_double},
    fmt::{self, Write},
    ptr, slice,
    str::FromStr,
};

/// Largest output produced by Rust's `LowerExp` formatter for an `f64`.
///
/// The proven implementation currently needs at most 24 bytes. Keep slack so
/// this adapter remains safe if that private spelling changes in Rust core.
const RAW_NUMBER_BYTES: usize = 32;
const LUA_NUMBER_BYTES: usize = 64;

/// ISO C locale whitespace. Rust's ASCII predicate deliberately omits VT
/// (`0x0b`), while Lua's lexer, patterns, and `strtod` contract include it.
pub(crate) const fn c_locale_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

struct Buffer<'a> {
    bytes: &'a mut [u8],
    len: usize,
}

impl Buffer<'_> {
    fn push(&mut self, byte: u8) -> bool {
        let Some(slot) = self.bytes.get_mut(self.len) else {
            return false;
        };
        *slot = byte;
        self.len += 1;
        true
    }

    fn extend(&mut self, bytes: &[u8]) -> bool {
        let Some(end) = self.len.checked_add(bytes.len()) else {
            return false;
        };
        let Some(dst) = self.bytes.get_mut(self.len..end) else {
            return false;
        };
        dst.copy_from_slice(bytes);
        self.len = end;
        true
    }

    fn repeat(&mut self, byte: u8, count: usize) -> bool {
        for _ in 0..count {
            if !self.push(byte) {
                return false;
            }
        }
        true
    }
}

impl Write for Buffer<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.extend(text.as_bytes()).then_some(()).ok_or(fmt::Error)
    }
}

/// Parse the decimal grammar Lua asks of `strtod`, retaining C's end pointer.
///
/// # Safety
///
/// `text` must point to a readable NUL-terminated C string. If non-null,
/// `end` must point to writable storage for one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_lua_strtod(text: *const c_char, end: *mut *mut c_char) -> c_double {
    if text.is_null() {
        return 0.0;
    }
    let mut len = 0usize;
    // SAFETY: the C contract above guarantees a readable string through its
    // terminator. Unlike a token-only ceiling, this also preserves public
    // `tonumber` behavior for long, valid Lua strings.
    while unsafe { *text.add(len) } != 0 {
        len += 1;
    }
    // SAFETY: the scan above established this live prefix.
    let bytes = unsafe { slice::from_raw_parts(text.cast::<u8>(), len) };
    let mut at = 0;
    while bytes.get(at).is_some_and(|byte| c_locale_whitespace(*byte)) {
        at += 1;
    }
    let number_start = at;
    if bytes
        .get(at)
        .is_some_and(|byte| matches!(*byte, b'+' | b'-'))
    {
        at += 1;
    }
    let mut digits = 0usize;
    while bytes.get(at).is_some_and(u8::is_ascii_digit) {
        at += 1;
        digits += 1;
    }
    if bytes.get(at) == Some(&b'.') {
        at += 1;
        while bytes.get(at).is_some_and(u8::is_ascii_digit) {
            at += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        if !end.is_null() {
            // SAFETY: guaranteed by the caller contract.
            unsafe { *end = text.cast_mut() };
        }
        return 0.0;
    }
    let mantissa_end = at;
    if bytes
        .get(at)
        .is_some_and(|byte| matches!(*byte, b'e' | b'E'))
    {
        let exponent = at;
        at += 1;
        if bytes
            .get(at)
            .is_some_and(|byte| matches!(*byte, b'+' | b'-'))
        {
            at += 1;
        }
        let exponent_start = at;
        while bytes.get(at).is_some_and(u8::is_ascii_digit) {
            at += 1;
        }
        if at == exponent_start {
            at = exponent;
        }
    }
    let parsed_end = at.max(mantissa_end);
    let Ok(number) = core::str::from_utf8(&bytes[number_start..parsed_end]) else {
        if !end.is_null() {
            // SAFETY: guaranteed by the caller contract.
            unsafe { *end = text.cast_mut() };
        }
        return 0.0;
    };
    let Ok(value) = f64::from_str(number) else {
        if !end.is_null() {
            // SAFETY: guaranteed by the caller contract.
            unsafe { *end = text.cast_mut() };
        }
        return 0.0;
    };
    if !end.is_null() {
        // SAFETY: `parsed_end <= len`, so this remains within the caller's
        // live C string (or one past its non-NUL prefix at the terminator).
        unsafe { *end = text.add(parsed_end).cast_mut() };
    }
    value
}

fn parse_exponent(bytes: &[u8]) -> Option<i32> {
    let (&first, rest) = bytes.split_first()?;
    let (negative, digits) = match first {
        b'-' => (true, rest),
        b'+' => (false, rest),
        _ => (false, bytes),
    };
    if digits.is_empty() {
        return None;
    }
    let mut value = 0i32;
    for &digit in digits {
        if !digit.is_ascii_digit() {
            return None;
        }
        value = value
            .checked_mul(10)?
            .checked_add(i32::from(digit - b'0'))?;
    }
    Some(if negative { -value } else { value })
}

fn write_exponent(out: &mut Buffer<'_>, exponent: i32) -> bool {
    if !out.push(b'e') || !out.push(if exponent < 0 { b'-' } else { b'+' }) {
        return false;
    }
    let magnitude = exponent.unsigned_abs();
    if magnitude >= 100 {
        out.push(b'0' + (magnitude / 100) as u8)
            && out.push(b'0' + ((magnitude / 10) % 10) as u8)
            && out.push(b'0' + (magnitude % 10) as u8)
    } else {
        out.push(b'0' + (magnitude / 10) as u8) && out.push(b'0' + (magnitude % 10) as u8)
    }
}

/// Normalize Rust's correctly rounded scientific representation to `%g`.
///
/// Rust core supplies the difficult, proven binary-to-decimal conversion. This
/// adapter only moves the decimal point and spells an exponent with the C/Lua
/// sign and minimum-two-digit convention.
fn format_number(value: f64, precision: usize, dst: &mut [u8; LUA_NUMBER_BYTES]) -> Option<usize> {
    let precision = precision.clamp(1, 17);
    let fractional_digits = precision - 1;
    let mut raw = [0u8; RAW_NUMBER_BYTES];
    // Scoped so the writer's borrow of `raw` ends before the shared read
    // below, which is what the block is for — `Buffer` has no `Drop`, so a
    // `drop()` call here would be a move that says nothing.
    let raw_len = {
        let mut raw_out = Buffer {
            bytes: &mut raw,
            len: 0,
        };
        write!(&mut raw_out, "{value:.fractional_digits$e}").ok()?;
        raw_out.len
    };
    let raw = &raw[..raw_len];

    let special: Option<&[u8]> = match raw {
        b"NaN" => Some(b"nan"),
        b"inf" => Some(b"inf"),
        b"-inf" => Some(b"-inf"),
        _ => None,
    };
    if let Some(special) = special {
        dst[..special.len()].copy_from_slice(special);
        return Some(special.len());
    }

    let (negative, unsigned) = raw
        .strip_prefix(b"-")
        .map_or((false, raw), |unsigned| (true, unsigned));
    let e_at = unsigned.iter().position(|&byte| byte == b'e')?;
    let exponent = parse_exponent(&unsigned[e_at + 1..])?;
    let mut digits = [0u8; 17];
    let mut digit_count = 0usize;
    for &byte in &unsigned[..e_at] {
        if byte == b'.' {
            continue;
        }
        if !byte.is_ascii_digit() || digit_count == digits.len() {
            return None;
        }
        digits[digit_count] = byte;
        digit_count += 1;
    }
    if digit_count == 0 {
        return None;
    }
    while digit_count > 1 && digits[digit_count - 1] == b'0' {
        digit_count -= 1;
    }

    let scientific = exponent < -4 || exponent >= precision as i32;
    let mut out = Buffer { bytes: dst, len: 0 };
    if negative && !out.push(b'-') {
        return None;
    }
    if scientific {
        if !out.push(digits[0])
            || (digit_count > 1 && (!out.push(b'.') || !out.extend(&digits[1..digit_count])))
            || !write_exponent(&mut out, exponent)
        {
            return None;
        }
    } else if exponent < 0 {
        if !out.extend(b"0.")
            || !out.repeat(b'0', (-exponent - 1) as usize)
            || !out.extend(&digits[..digit_count])
        {
            return None;
        }
    } else {
        let integer_digits = exponent as usize + 1;
        if integer_digits >= digit_count {
            if !out.extend(&digits[..digit_count])
                || !out.repeat(b'0', integer_digits - digit_count)
            {
                return None;
            }
        } else if !out.extend(&digits[..integer_digits])
            || !out.push(b'.')
            || !out.extend(&digits[integer_digits..digit_count])
        {
            return None;
        }
    }
    Some(out.len)
}

/// Write Lua's core number representation and return bytes actually stored.
///
/// This is intentionally not `snprintf`: a short destination truncates safely
/// and the returned length never exceeds `cap - 1`. Lua's core buffer is large
/// enough for every result, while the stricter contract prevents callers from
/// treating an unstored would-write length as initialized memory.
///
/// # Safety
///
/// When `cap != 0`, `dst` must point to `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_lua_format_number(
    dst: *mut c_char,
    cap: usize,
    value: c_double,
    precision: usize,
) -> usize {
    if dst.is_null() || cap == 0 {
        return 0;
    }
    let mut formatted = [0u8; LUA_NUMBER_BYTES];
    let Some(len) = format_number(value, precision, &mut formatted) else {
        // SAFETY: the caller supplied at least one writable byte.
        unsafe { dst.write(0) };
        return 0;
    };
    let stored = len.min(cap - 1);
    // SAFETY: the caller supplied `cap` writable bytes and `stored < cap`.
    unsafe {
        ptr::copy_nonoverlapping(formatted.as_ptr(), dst.cast::<u8>(), stored);
        dst.add(stored).write(0);
    }
    stored
}
