//! The `<wctype.h>` classes and case maps, answered from Rust's Unicode
//! tables: a grammar's scanner asks whether a code point is a letter or a
//! space, and the answer should not depend on a locale kore does not have.

use core::ffi::c_int;

/// C's `wint_t`: an unsigned 32-bit code point (or `WEOF`).
type WInt = u32;

fn class(code: WInt, test: impl Fn(char) -> bool) -> c_int {
    c_int::from(char::from_u32(code).is_some_and(test))
}

#[unsafe(no_mangle)]
extern "C" fn iswalnum(code: WInt) -> c_int {
    class(code, char::is_alphanumeric)
}

#[unsafe(no_mangle)]
extern "C" fn iswalpha(code: WInt) -> c_int {
    class(code, char::is_alphabetic)
}

#[unsafe(no_mangle)]
extern "C" fn iswblank(code: WInt) -> c_int {
    class(code, |c| c == ' ' || c == '\t' || (c.is_whitespace() && !matches!(c, '\n'..='\r' | '\u{85}' | '\u{2028}' | '\u{2029}')))
}

#[unsafe(no_mangle)]
extern "C" fn iswdigit(code: WInt) -> c_int {
    class(code, |c| c.is_ascii_digit())
}

#[unsafe(no_mangle)]
extern "C" fn iswlower(code: WInt) -> c_int {
    class(code, char::is_lowercase)
}

#[unsafe(no_mangle)]
extern "C" fn iswupper(code: WInt) -> c_int {
    class(code, char::is_uppercase)
}

#[unsafe(no_mangle)]
extern "C" fn iswspace(code: WInt) -> c_int {
    class(code, char::is_whitespace)
}

#[unsafe(no_mangle)]
extern "C" fn iswxdigit(code: WInt) -> c_int {
    class(code, |c| c.is_ascii_hexdigit())
}

/// Printable, not a space, not a letter or digit: ISO's definition.
#[unsafe(no_mangle)]
extern "C" fn iswpunct(code: WInt) -> c_int {
    class(code, |c| !c.is_whitespace() && !c.is_alphanumeric() && !c.is_control())
}

/// The one-code-point case maps; a mapping that expands to several code
/// points (`ß` → `SS`) leaves the code point as it is, as C's does.
fn single(mut mapped: impl Iterator<Item = char>, code: WInt) -> WInt {
    match (mapped.next(), mapped.next()) {
        (Some(c), None) => c as WInt,
        _ => code,
    }
}

#[unsafe(no_mangle)]
extern "C" fn towlower(code: WInt) -> WInt {
    char::from_u32(code).map_or(code, |c| single(c.to_lowercase(), code))
}

#[unsafe(no_mangle)]
extern "C" fn towupper(code: WInt) -> WInt {
    char::from_u32(code).map_or(code, |c| single(c.to_uppercase(), code))
}
