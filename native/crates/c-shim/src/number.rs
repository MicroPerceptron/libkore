//! The parsing behind `strtol`, `strtoul`, `strtod` and their siblings: ISO C
//! 7.22.1's grammar over the bytes before a string's NUL, pure so the host
//! suite drives the code the target runs.
//!
//! Integers follow the standard exactly: leading C-locale whitespace, an
//! optional sign, base `0` choosing 16 for `0x`, 8 for a leading `0` and 10
//! otherwise, overflow reported rather than wrapped, and an end offset that
//! is zero when no digit was found.
//!
//! Decimal floats are rounded correctly — the digits go to core's parser,
//! which rounds any length of input exactly — and `inf`, `infinity` and
//! `nan[(chars)]` are recognized in any case. **Hexadecimal floats are not
//! parsed**: `0x1p3` reads as `0` and ends at the `x`, which is C89's
//! reading. No consumer needs more; TCC, the one that compiles C, parses
//! hexadecimal float literals itself.
//!
//! The subject grammar is public ([`float_subject`]) so a parser for a type
//! core has no parser for — TCC's `long double` — reads exactly the sequence
//! `strtod` reads.

/// ISO C locale whitespace (`isspace`): space, `\t \n \v \f \r`.
pub const fn c_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// What an integer parse found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Integer {
    /// The value's magnitude, saturated at `u64::MAX` when `overflow`.
    pub magnitude: u64,
    /// A `-` preceded the digits.
    pub negative: bool,
    /// The magnitude did not fit 64 bits.
    pub overflow: bool,
    /// Bytes consumed through the last digit; zero when there was no number.
    pub end: usize,
}

/// Parse an integer in `base` (0 or 2..=36) from the start of `text`.
/// `None` for any other base — C's `EINVAL`, with no conversion.
pub fn integer(text: &[u8], base: u32) -> Option<Integer> {
    if base == 1 || base > 36 {
        return None;
    }
    let mut at = 0;
    while text.get(at).copied().is_some_and(c_space) {
        at += 1;
    }
    let negative = text.get(at) == Some(&b'-');
    if matches!(text.get(at), Some(b'+' | b'-')) {
        at += 1;
    }
    let digit = |byte: u8, base: u32| char::from(byte).to_digit(base);
    let hex_prefix = matches!(text.get(at..at + 2), Some([b'0', b'x' | b'X']))
        && text
            .get(at + 2)
            .is_some_and(|&byte| digit(byte, 16).is_some());
    let base = match base {
        0 if hex_prefix => 16,
        0 if text.get(at) == Some(&b'0') => 8,
        0 => 10,
        base => base,
    };
    // `0x` is skipped only when a hexadecimal digit follows it; otherwise
    // the `0` is the number and parsing ends at the `x`.
    if base == 16 && hex_prefix {
        at += 2;
    }
    let first = at;
    let mut magnitude = 0u64;
    let mut overflow = false;
    while let Some(value) = text.get(at).and_then(|&byte| digit(byte, base)) {
        match magnitude
            .checked_mul(u64::from(base))
            .and_then(|scaled| scaled.checked_add(u64::from(value)))
        {
            Some(next) => magnitude = next,
            None => {
                overflow = true;
                magnitude = u64::MAX;
            }
        }
        at += 1;
    }
    let end = if at == first { 0 } else { at };
    Some(Integer {
        magnitude,
        negative,
        overflow,
        end,
    })
}

impl Integer {
    /// The value as a signed integer no wider than `bits`, and whether it
    /// was clamped (C's `ERANGE`).
    pub fn signed(self, bits: u32) -> (i64, bool) {
        let max = (1u64 << (bits - 1)) - 1;
        if self.negative {
            if self.overflow || self.magnitude > max + 1 {
                (-(max as i64) - 1, true)
            } else {
                (0i64.wrapping_sub(self.magnitude as i64), false)
            }
        } else if self.overflow || self.magnitude > max {
            (max as i64, true)
        } else {
            (self.magnitude as i64, false)
        }
    }

    /// The value as an unsigned integer no wider than `bits`, and whether it
    /// was clamped. A `-` negates in the unsigned type, as C says.
    pub fn unsigned(self, bits: u32) -> (u64, bool) {
        let max = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        if self.overflow || self.magnitude > max {
            (max, true)
        } else if self.negative {
            (self.magnitude.wrapping_neg() & max, false)
        } else {
            (self.magnitude, false)
        }
    }
}

/// What a float parse found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Float<T> {
    pub value: T,
    /// Overflow to infinity, or a nonzero number rounded to zero: `ERANGE`.
    pub range: bool,
    /// Bytes consumed; zero when there was no number.
    pub end: usize,
}

/// The floating subject sequence at the start of `text` (ISO C 7.22.1.3),
/// leading whitespace skipped: what `strtod` and its siblings convert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatSubject<'t> {
    /// No subject: nothing is converted, and the end pointer is the input.
    None,
    /// `inf`, `infinity` or `nan[(n-char-sequence)]`, in any case.
    Special {
        nan: bool,
        negative: bool,
        end: usize,
    },
    /// A decimal number. `digits` is the sign-free subject — digits, an
    /// optional `.`, an optional exponent — in the grammar a
    /// correctly-rounding parser takes; `nonzero` says a mantissa digit was
    /// not `0`, so a zero result is an underflow rather than a zero.
    Decimal {
        negative: bool,
        digits: &'t [u8],
        nonzero: bool,
        end: usize,
    },
}

/// Scan the floating subject at the start of `text`.
pub fn float_subject(text: &[u8]) -> FloatSubject<'_> {
    if let Some((nan, negative, end)) = special(text) {
        return FloatSubject::Special { nan, negative, end };
    }
    match decimal_subject(text) {
        Some((negative, digits, nonzero, end)) => FloatSubject::Decimal {
            negative,
            digits,
            nonzero,
            end,
        },
        None => FloatSubject::None,
    }
}

/// The decimal subject sequence at the start of `text`: its sign, the slice
/// core's parser takes, whether any mantissa digit was nonzero, and the end.
fn decimal_subject(text: &[u8]) -> Option<(bool, &[u8], bool, usize)> {
    let mut at = 0;
    while text.get(at).copied().is_some_and(c_space) {
        at += 1;
    }
    let negative = text.get(at) == Some(&b'-');
    if matches!(text.get(at), Some(b'+' | b'-')) {
        at += 1;
    }
    let start = at;
    let mut digits = 0usize;
    let mut nonzero = false;
    while let Some(&byte) = text.get(at).filter(|byte| byte.is_ascii_digit()) {
        nonzero |= byte != b'0';
        digits += 1;
        at += 1;
    }
    if text.get(at) == Some(&b'.') {
        at += 1;
        while let Some(&byte) = text.get(at).filter(|byte| byte.is_ascii_digit()) {
            nonzero |= byte != b'0';
            digits += 1;
            at += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    // An exponent counts only when at least one digit follows it.
    if matches!(text.get(at), Some(b'e' | b'E')) {
        let mut exponent = at + 1;
        if matches!(text.get(exponent), Some(b'+' | b'-')) {
            exponent += 1;
        }
        if text.get(exponent).is_some_and(u8::is_ascii_digit) {
            at = exponent;
            while text.get(at).is_some_and(u8::is_ascii_digit) {
                at += 1;
            }
        }
    }
    Some((negative, &text[start..at], nonzero, at))
}

/// `inf`, `infinity` or `nan[(n-char-sequence)]`, case-insensitively, at the
/// start of `text`: `(is_nan, negative, end)`.
fn special(text: &[u8]) -> Option<(bool, bool, usize)> {
    let mut at = 0;
    while text.get(at).copied().is_some_and(c_space) {
        at += 1;
    }
    let negative = text.get(at) == Some(&b'-');
    if matches!(text.get(at), Some(b'+' | b'-')) {
        at += 1;
    }
    let word = |at: usize, word: &[u8]| {
        text.get(at..at + word.len())
            .is_some_and(|got| got.eq_ignore_ascii_case(word))
    };
    if word(at, b"infinity") {
        return Some((false, negative, at + 8));
    }
    if word(at, b"inf") {
        return Some((false, negative, at + 3));
    }
    if word(at, b"nan") {
        let mut end = at + 3;
        // `nan(` followed by alphanumerics and `_`, then `)`.
        if text.get(end) == Some(&b'(') {
            let mut close = end + 1;
            while text
                .get(close)
                .is_some_and(|&byte| byte.is_ascii_alphanumeric() || byte == b'_')
            {
                close += 1;
            }
            if text.get(close) == Some(&b')') {
                end = close + 1;
            }
        }
        return Some((true, negative, end));
    }
    None
}

macro_rules! float_parser {
    ($name:ident, $ty:ty) => {
        /// Parse a float from the start of `text`.
        pub fn $name(text: &[u8]) -> Float<$ty> {
            let (negative, subject, nonzero, end) = match float_subject(text) {
                FloatSubject::Special { nan, negative, end } => {
                    let value = if nan { <$ty>::NAN } else { <$ty>::INFINITY };
                    return Float {
                        value: if negative { -value } else { value },
                        range: false,
                        end,
                    };
                }
                FloatSubject::None => {
                    return Float {
                        value: 0.0,
                        range: false,
                        end: 0,
                    };
                }
                FloatSubject::Decimal {
                    negative,
                    digits,
                    nonzero,
                    end,
                } => (negative, digits, nonzero, end),
            };
            // The subject is ASCII digits, `.` and an exponent: always UTF-8,
            // and always in core's grammar (a lone `.` was refused above).
            let value: $ty = core::str::from_utf8(subject)
                .ok()
                .and_then(|digits| digits.parse().ok())
                .unwrap_or(0.0);
            let range = value.is_infinite() || (value == 0.0 && nonzero);
            Float {
                value: if negative { -value } else { value },
                range,
                end,
            }
        }
    };
}

float_parser!(double, f64);
float_parser!(single, f32);

#[cfg(test)]
mod tests {
    use super::*;

    fn int(text: &str, base: u32) -> Integer {
        integer(text.as_bytes(), base).expect("a valid base")
    }

    #[test]
    fn integers_follow_the_c_grammar() {
        assert_eq!(
            int("  42abc", 10),
            Integer {
                magnitude: 42,
                negative: false,
                overflow: false,
                end: 4
            }
        );
        assert_eq!(int("-17", 10).signed(64), (-17, false));
        assert_eq!(int("+9", 0).signed(64), (9, false));
        assert_eq!(int("0x1F", 0).magnitude, 31);
        assert_eq!(int("0x1F", 16).magnitude, 31);
        assert_eq!(int("017", 0).magnitude, 15);
        assert_eq!(int("z", 36).magnitude, 35);
        // `0x` without a hex digit after it: the `0` is the number.
        assert_eq!(
            int("0xg", 0),
            Integer {
                magnitude: 0,
                negative: false,
                overflow: false,
                end: 1
            }
        );
        assert_eq!(int("0x", 16).end, 1);
        // No digits: no conversion, end zero.
        assert_eq!(int("  -", 10).end, 0);
        assert_eq!(int("", 10).end, 0);
        assert!(integer(b"1", 1).is_none());
        assert!(integer(b"1", 37).is_none());
    }

    #[test]
    fn integer_overflow_clamps_and_reports() {
        assert_eq!(int("9223372036854775807", 10).signed(64), (i64::MAX, false));
        assert_eq!(int("9223372036854775808", 10).signed(64), (i64::MAX, true));
        assert_eq!(
            int("-9223372036854775808", 10).signed(64),
            (i64::MIN, false)
        );
        assert_eq!(int("-9223372036854775809", 10).signed(64), (i64::MIN, true));
        assert_eq!(
            int("99999999999999999999999", 10).signed(64),
            (i64::MAX, true)
        );
        assert_eq!(int("2147483648", 10).signed(32), (i32::MAX as i64, true));
        assert_eq!(
            int("18446744073709551615", 10).unsigned(64),
            (u64::MAX, false)
        );
        assert_eq!(
            int("18446744073709551616", 10).unsigned(64),
            (u64::MAX, true)
        );
        // A minus sign negates in the unsigned type.
        assert_eq!(int("-1", 10).unsigned(64), (u64::MAX, false));
        assert_eq!(int("-1", 10).unsigned(32), (u32::MAX as u64, false));
        // Overflowed digits are still consumed.
        assert_eq!(int("99999999999999999999999x", 10).end, 23);
    }

    #[test]
    fn decimal_floats_round_correctly() {
        assert_eq!(
            double(b"1.5"),
            Float {
                value: 1.5,
                range: false,
                end: 3
            }
        );
        assert_eq!(double(b"  -2.5e3x").value, -2500.0);
        assert_eq!(double(b"  -2.5e3x").end, 8);
        assert_eq!(double(b".5").value, 0.5);
        assert_eq!(double(b"5.").value, 5.0);
        assert_eq!(double(b"5.").end, 2);
        // An exponent with no digits is not part of the number.
        assert_eq!(double(b"1e+").end, 1);
        assert_eq!(double(b"0.1").value, 0.1);
        assert_eq!(single(b"0.1").value, 0.1f32);
        assert_eq!(double(b"").end, 0);
        assert_eq!(double(b".").end, 0);
        assert_eq!(double(b"-").end, 0);
    }

    #[test]
    fn float_range_and_specials() {
        let huge = double(b"1e400");
        assert!(huge.value.is_infinite() && huge.range);
        let tiny = double(b"1e-400");
        assert!(tiny.value == 0.0 && tiny.range);
        let zero = double(b"0.000");
        assert!(zero.value == 0.0 && !zero.range);
        assert!(single(b"1e39").range);
        assert_eq!(double(b"INF").value, f64::INFINITY);
        assert_eq!(double(b"-Infinity").value, f64::NEG_INFINITY);
        assert_eq!(double(b"-Infinity").end, 9);
        assert!(double(b"nan(0x1)").value.is_nan());
        assert_eq!(double(b"nan(abc)").end, 8);
        assert_eq!(double(b"nan(").end, 3);
        // Hexadecimal floats read as C89 does: the `0`, ending at the `x`.
        assert_eq!(
            double(b"0x1p3"),
            Float {
                value: 0.0,
                range: false,
                end: 1
            }
        );
    }

    #[test]
    fn the_subject_is_the_parsers_own() {
        assert_eq!(
            float_subject(b" -1.25e3x"),
            FloatSubject::Decimal {
                negative: true,
                digits: b"1.25e3",
                nonzero: true,
                end: 8
            }
        );
        assert_eq!(
            float_subject(b"0.000"),
            FloatSubject::Decimal {
                negative: false,
                digits: b"0.000",
                nonzero: false,
                end: 5
            }
        );
        assert_eq!(
            float_subject(b"-INF"),
            FloatSubject::Special {
                nan: false,
                negative: true,
                end: 4
            }
        );
        assert_eq!(
            float_subject(b"nan(7)"),
            FloatSubject::Special {
                nan: true,
                negative: false,
                end: 6
            }
        );
        assert_eq!(float_subject(b" ."), FloatSubject::None);
        assert_eq!(float_subject(b"+"), FloatSubject::None);
    }
}
