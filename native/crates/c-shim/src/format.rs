//! The `printf` conversion engine behind every formatted write in this crate.
//!
//! One engine serves `snprintf` into a caller's buffer and `fprintf` into a
//! stream, so the two cannot disagree about what a conversion means. It is
//! generic over where arguments come from ([`Args`]: a C `va_list` on the
//! target, a scripted list in host tests) and where bytes go ([`Sink`]), and
//! it is pure: the host suite drives exactly the code the target runs.
//!
//! ## What it formats
//!
//! ISO C's integer, character, string and pointer conversions in full —
//! `d i o u x X c s p %`, the `- + space # 0` flags, field width and
//! precision (both also as `*`), and the `hh h l ll j z t` length modifiers.
//! A precision bounds a `%s` read, so `%.*s` over a counted, unterminated
//! string never reads past its count. `%p` is spelled `0x` and the address
//! in lowercase hexadecimal (the spelling is implementation-defined; this is
//! the one this crate has always used).
//!
//! The float conversions `e E f F g G a A` too, of a `double` and (with `L`)
//! of a `long double` in either of kore's formats, every flag, width and
//! precision honoured and every digit exact ([`float`]): the C programs
//! libkore.so runs print through this engine.
//!
//! ## What it deliberately does not
//!
//! - **`%n`** stores nothing: its pointer is consumed and ignored. A format
//!   that writes through an argument is the classic format-string attack,
//!   and no consumer needs one.
//! - **Wide characters**: an ASCII `%lc` is written as its byte and anything
//!   else as `?`; `%ls` consumes its pointer and is spelled back.
//!
//! An unknown conversion is spelled back and consumes nothing — its type
//! cannot be known, and guessing would misalign every argument after it.

use core::ffi::c_int;
use core::marker::PhantomData;

mod float;

pub use float::Float;

/// The C type a length modifier names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Length {
    /// `hh`: `char`, promoted to `int` in a variadic call.
    Char,
    /// `h`: `short`, promoted to `int`.
    Short,
    /// No modifier: `int`.
    Int,
    /// `l`: `long`.
    Long,
    /// `ll` (and BSD's `q`): `long long`.
    LongLong,
    /// `j`: `intmax_t`.
    Max,
    /// `z`: `size_t`.
    Size,
    /// `t`: `ptrdiff_t`.
    Diff,
    /// `L`: `long double` beside a float conversion; `long long` beside an
    /// integer one, as glibc reads it.
    LongDouble,
}

/// Where each conversion's argument comes from.
///
/// Every method consumes exactly one argument of the C type it names, as a
/// variadic callee must.
pub trait Args {
    /// A signed integer of `length`'s type, sign-extended.
    fn signed(&mut self, length: Length) -> i64;
    /// An unsigned integer of `length`'s type, zero-extended.
    fn unsigned(&mut self, length: Length) -> u64;
    /// An `int`: a `%c` argument, or a `*` width or precision.
    fn int(&mut self) -> c_int;
    /// A pointer, as an address: `%p`, `%n`, `%ls`, and `%s`'s string.
    fn pointer(&mut self) -> usize;
    /// A `double` (a `float` arrives promoted to one).
    fn double(&mut self) -> f64;
    /// A `long double`, read in the C ABI's own format. Rust has no type for
    /// it, so the source decodes it.
    fn long_double(&mut self) -> Float;
}

/// Where formatted bytes go. The engine counts them; a sink only stores.
pub trait Sink {
    /// Accept the next run of output bytes.
    fn put(&mut self, bytes: &[u8]);
}

/// Format `format` into `sink`, drawing arguments from `args`, and return
/// how many bytes the complete output has — the count `snprintf` returns
/// even when its buffer is shorter.
///
/// # Safety
/// `format` is NUL-terminated, and its conversions match the arguments
/// `args` yields.
pub unsafe fn format(sink: &mut impl Sink, format: *const u8, args: &mut impl Args) -> usize {
    let format = Format {
        base: format,
        _bytes: PhantomData,
    };
    let mut out = Out { sink, count: 0 };
    let mut at = 0usize;
    loop {
        // The literal run up to the next conversion, written whole.
        let start = at;
        // SAFETY (every `format.at` / `format.slice` below): reads stop at
        // the terminator, and `at` only advances past bytes already read as
        // non-NUL.
        while !matches!(unsafe { format.at(at) }, 0 | b'%') {
            at += 1;
        }
        out.put(unsafe { format.slice(start, at) });
        if unsafe { format.at(at) } == 0 {
            return out.count;
        }
        let spec_start = at;
        at += 1;

        let mut spec = Spec::default();
        loop {
            match unsafe { format.at(at) } {
                b'-' => spec.left = true,
                b'+' => spec.plus = true,
                b' ' => spec.space = true,
                b'#' => spec.alternate = true,
                b'0' => spec.zero = true,
                _ => break,
            }
            at += 1;
        }
        if unsafe { format.at(at) } == b'*' {
            at += 1;
            let width = args.int();
            // A negative `*` width is a `-` flag and a positive width.
            spec.left |= width < 0;
            spec.width = (width.unsigned_abs() as usize).min(MAX_FIELD);
        } else {
            spec.width = unsafe { format.decimal(&mut at) };
        }
        if unsafe { format.at(at) } == b'.' {
            at += 1;
            if unsafe { format.at(at) } == b'*' {
                at += 1;
                // A negative precision is taken as if it were omitted.
                spec.precision = usize::try_from(args.int()).ok();
            } else {
                spec.precision = Some(unsafe { format.decimal(&mut at) });
            }
        }
        let (length, skip) = match unsafe { [format.at(at), format.at(at + 1)] } {
            [b'h', b'h'] => (Length::Char, 2),
            [b'h', _] => (Length::Short, 1),
            [b'l', b'l'] => (Length::LongLong, 2),
            [b'l', _] => (Length::Long, 1),
            [b'q', _] => (Length::LongLong, 1),
            [b'j', _] => (Length::Max, 1),
            [b'z', _] => (Length::Size, 1),
            [b't', _] => (Length::Diff, 1),
            [b'L', _] => (Length::LongDouble, 1),
            _ => (Length::Int, 0),
        };
        at += skip;
        let conversion = unsafe { format.at(at) };
        if conversion == 0 {
            // A specification cut off by the terminator: spell it back.
            out.put(unsafe { format.slice(spec_start, at) });
            return out.count;
        }
        at += 1;
        // The specification's own bytes, for the conversions spelled back.
        let spelled = unsafe { format.slice(spec_start, at) };

        match conversion {
            b'%' => out.put(b"%"),
            b'd' | b'i' => {
                let value = args.signed(length);
                integer(
                    &mut out,
                    spec,
                    value.unsigned_abs(),
                    value < 0,
                    Radix::Signed,
                );
            }
            b'u' => integer(&mut out, spec, args.unsigned(length), false, Radix::Decimal),
            b'o' => integer(&mut out, spec, args.unsigned(length), false, Radix::Octal),
            b'x' => integer(&mut out, spec, args.unsigned(length), false, Radix::Hex),
            b'X' => integer(
                &mut out,
                spec,
                args.unsigned(length),
                false,
                Radix::HexUpper,
            ),
            b'c' => {
                let value = args.int();
                let byte = if length == Length::Long {
                    // `wint_t`: only an ASCII code point has a byte here.
                    u8::try_from(value)
                        .ok()
                        .filter(u8::is_ascii)
                        .unwrap_or(b'?')
                } else {
                    value as u8
                };
                text(&mut out, spec, &[byte]);
            }
            b's' if length == Length::Long => {
                args.pointer();
                out.put(spelled);
            }
            b's' => {
                let address = args.pointer() as *const u8;
                if address.is_null() {
                    text(&mut out, spec, b"(null)");
                } else {
                    // SAFETY: C promises a string readable to its NUL, or to
                    // the precision when one bounds the read.
                    let bytes = unsafe { c_string(address, spec.precision) };
                    text(&mut out, spec, bytes);
                }
            }
            b'p' => {
                let mut digits = [0u8; 22];
                let digits = render(args.pointer() as u64, Radix::Hex, &mut digits);
                let mut buffer = [0u8; 18];
                buffer[..2].copy_from_slice(b"0x");
                buffer[2..2 + digits.len()].copy_from_slice(digits);
                text(&mut out, spec, &buffer[..2 + digits.len()]);
            }
            b'n' => {
                args.pointer();
            }
            b'e' | b'E' | b'f' | b'F' | b'g' | b'G' | b'a' | b'A' => {
                if length == Length::LongDouble {
                    let value = args.long_double();
                    float::convert::<{ float::LONG_DOUBLE_LIMBS }, _>(
                        &mut out, spec, conversion, value,
                    );
                } else {
                    let value = Float::from_f64(args.double());
                    float::convert::<{ float::DOUBLE_LIMBS }, _>(&mut out, spec, conversion, value);
                }
            }
            _ => out.put(spelled),
        }
    }
}

/// A NUL-terminated C format string.
struct Format<'f> {
    base: *const u8,
    _bytes: PhantomData<&'f [u8]>,
}

impl<'f> Format<'f> {
    /// # Safety
    /// `at` lies at or before the terminator.
    unsafe fn at(&self, at: usize) -> u8 {
        // SAFETY: the caller's contract.
        unsafe { *self.base.add(at) }
    }

    /// The bytes in `start..end`.
    ///
    /// # Safety
    /// The range was read and lies before the terminator.
    unsafe fn slice(&self, start: usize, end: usize) -> &'f [u8] {
        // SAFETY: the caller's contract.
        unsafe { core::slice::from_raw_parts(self.base.add(start), end - start) }
    }

    /// A run of decimal digits as a field width or precision, saturating at
    /// [`MAX_FIELD`].
    ///
    /// # Safety
    /// `at` lies at or before the terminator.
    unsafe fn decimal(&self, at: &mut usize) -> usize {
        let mut value = 0usize;
        // SAFETY: the caller's contract; a digit is never the terminator.
        while let digit @ b'0'..=b'9' = unsafe { self.at(*at) } {
            value = value
                .saturating_mul(10)
                .saturating_add(usize::from(digit - b'0'))
                .min(MAX_FIELD);
            *at += 1;
        }
        value
    }
}

/// A `*` or decimal width or precision is an `int`; this is the largest.
const MAX_FIELD: usize = c_int::MAX as usize;

/// Output padding comes from these, a run at a time.
const SPACES: [u8; 32] = [b' '; 32];
const ZEROS: [u8; 32] = [b'0'; 32];

/// The engine's counting front: every byte a sink sees passes through here.
struct Out<'s, S: Sink> {
    sink: &'s mut S,
    count: usize,
}

impl<S: Sink> Out<'_, S> {
    fn put(&mut self, bytes: &[u8]) {
        if !bytes.is_empty() {
            self.sink.put(bytes);
            self.count = self.count.saturating_add(bytes.len());
        }
    }

    fn pad(&mut self, fill: &[u8; 32], mut count: usize) {
        while count > 0 {
            let run = count.min(fill.len());
            self.put(&fill[..run]);
            count -= run;
        }
    }
}

/// One parsed conversion specification.
#[derive(Clone, Copy, Debug, Default)]
struct Spec {
    left: bool,
    plus: bool,
    space: bool,
    alternate: bool,
    zero: bool,
    width: usize,
    precision: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Radix {
    Octal,
    /// `%d`/`%i`: decimal, and the only radix the `+` and space flags mark.
    Signed,
    /// `%u`.
    Decimal,
    Hex,
    HexUpper,
}

/// The bytes of the C string at `address`, stopping at its NUL or, when
/// `limit` is given, after `limit` bytes — whichever comes first.
///
/// # Safety
/// `address` is readable up to its NUL or `limit`.
unsafe fn c_string<'s>(address: *const u8, limit: Option<usize>) -> &'s [u8] {
    let limit = limit.unwrap_or(usize::MAX);
    let mut len = 0usize;
    // SAFETY: the caller's contract bounds every read.
    while len < limit && unsafe { *address.add(len) } != 0 {
        len += 1;
    }
    unsafe { core::slice::from_raw_parts(address, len) }
}

/// `value`'s digits, written into the tail of `buffer` (22 bytes hold a u64
/// in octal).
fn render(mut value: u64, radix: Radix, buffer: &mut [u8; 22]) -> &[u8] {
    let (base, digits): (u64, &[u8; 16]) = match radix {
        Radix::Octal => (8, b"0123456789abcdef"),
        Radix::Signed | Radix::Decimal => (10, b"0123456789abcdef"),
        Radix::Hex => (16, b"0123456789abcdef"),
        Radix::HexUpper => (16, b"0123456789ABCDEF"),
    };
    let mut at = buffer.len();
    loop {
        at -= 1;
        buffer[at] = digits[(value % base) as usize];
        value /= base;
        if value == 0 {
            break;
        }
    }
    &buffer[at..]
}

/// An integer conversion: sign or prefix, the precision's leading zeros,
/// digits, and the field's padding, placed as ISO C 7.21.6.1 places them.
fn integer(out: &mut Out<'_, impl Sink>, spec: Spec, magnitude: u64, negative: bool, radix: Radix) {
    let mut buffer = [0u8; 22];
    let mut digits = render(magnitude, radix, &mut buffer);
    // A zero precision prints no digits for a zero value.
    if magnitude == 0 && spec.precision == Some(0) {
        digits = &[];
    }
    let prefix: &[u8] = match radix {
        _ if negative => b"-",
        Radix::Signed if spec.plus => b"+",
        Radix::Signed if spec.space => b" ",
        Radix::Hex if spec.alternate && magnitude != 0 => b"0x",
        Radix::HexUpper if spec.alternate && magnitude != 0 => b"0X",
        _ => b"",
    };
    let mut leading = spec
        .precision
        .map_or(0, |precision| precision.saturating_sub(digits.len()));
    // `#` with `o` raises the precision just enough for a leading zero.
    if spec.alternate && radix == Radix::Octal && leading == 0 && digits.first() != Some(&b'0') {
        leading = 1;
    }
    let fill = spec
        .width
        .saturating_sub(prefix.len() + leading + digits.len());
    // `0` pads between the prefix and the digits, unless a precision was
    // given or the field is left-justified.
    let zero_fill = spec.zero && !spec.left && spec.precision.is_none();
    if !spec.left && !zero_fill {
        out.pad(&SPACES, fill);
    }
    out.put(prefix);
    if zero_fill {
        out.pad(&ZEROS, fill);
    }
    out.pad(&ZEROS, leading);
    out.put(digits);
    if spec.left {
        out.pad(&SPACES, fill);
    }
}

/// A text conversion (`%c`, `%s`, `%p`): the bytes and the field's padding.
/// The `0` flag means nothing here, as in C.
fn text(out: &mut Out<'_, impl Sink>, spec: Spec, bytes: &[u8]) {
    let fill = spec.width.saturating_sub(bytes.len());
    if !spec.left {
        out.pad(&SPACES, fill);
    }
    out.put(bytes);
    if spec.left {
        out.pad(&SPACES, fill);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::{collections::VecDeque, string::String, vec::Vec};

    /// One scripted argument.
    #[derive(Clone, Copy, Debug)]
    enum Arg {
        Int(i64),
        Uint(u64),
        Ptr(usize),
        Double(f64),
        LongDouble(Float),
    }

    /// A NUL-terminated (or, for `%.*s`, counted) string argument.
    fn text(bytes: &'static [u8]) -> Arg {
        Arg::Ptr(bytes.as_ptr() as usize)
    }

    struct Script {
        args: VecDeque<Arg>,
        /// Each consumption, so a test can prove alignment.
        taken: Vec<&'static str>,
    }

    impl Script {
        fn take(&mut self, kind: &'static str) -> Arg {
            self.taken.push(kind);
            self.args
                .pop_front()
                .expect("the format consumed more arguments than given")
        }
    }

    impl Args for Script {
        fn signed(&mut self, length: Length) -> i64 {
            let Arg::Int(value) = self.take("signed") else {
                panic!("a signed conversion met another type")
            };
            // The truncation a C callee applies to a promoted argument.
            match length {
                Length::Char => value as i8 as i64,
                Length::Short => value as i16 as i64,
                Length::Int => value as i32 as i64,
                _ => value,
            }
        }
        fn unsigned(&mut self, length: Length) -> u64 {
            let value = match self.take("unsigned") {
                Arg::Uint(value) => value,
                Arg::Int(value) => value as u64,
                other => panic!("an unsigned conversion met {other:?}"),
            };
            match length {
                Length::Char => value as u8 as u64,
                Length::Short => value as u16 as u64,
                Length::Int => value as u32 as u64,
                _ => value,
            }
        }
        fn int(&mut self) -> c_int {
            let Arg::Int(value) = self.take("int") else {
                panic!("an int met another type")
            };
            value as c_int
        }
        fn pointer(&mut self) -> usize {
            let Arg::Ptr(value) = self.take("pointer") else {
                panic!("a pointer met another type")
            };
            value
        }
        fn double(&mut self) -> f64 {
            let Arg::Double(value) = self.take("double") else {
                panic!("a double met another type")
            };
            value
        }
        fn long_double(&mut self) -> Float {
            let Arg::LongDouble(value) = self.take("long double") else {
                panic!("a long double met another type")
            };
            value
        }
    }

    struct Collect(Vec<u8>);

    impl Sink for Collect {
        fn put(&mut self, bytes: &[u8]) {
            self.0.extend_from_slice(bytes);
        }
    }

    /// Format a NUL-terminated `format`; return the text, count and script.
    fn run(format: &[u8], args: &[Arg]) -> (String, usize, Vec<&'static str>) {
        assert_eq!(format.last(), Some(&0), "test formats are NUL-terminated");
        let mut script = Script {
            args: args.iter().copied().collect(),
            taken: Vec::new(),
        };
        let mut sink = Collect(Vec::new());
        // SAFETY: the format is NUL-terminated and the script matches it.
        let count = unsafe { super::format(&mut sink, format.as_ptr(), &mut script) };
        assert!(
            script.args.is_empty(),
            "the format left arguments unconsumed"
        );
        (String::from_utf8(sink.0).unwrap(), count, script.taken)
    }

    fn check(format: &[u8], args: &[Arg], expected: &str) {
        let (out, count, _) = run(format, args);
        assert_eq!(
            out,
            expected,
            "format {:?}",
            String::from_utf8_lossy(format)
        );
        assert_eq!(count, expected.len());
    }

    // Integer, character and string expectations match the host C library's
    // printf for the same format and arguments.

    #[test]
    fn literal_text_and_percent() {
        check(b"plain\0", &[], "plain");
        check(b"100%% sure\0", &[], "100% sure");
        check(b"\0", &[], "");
    }

    #[test]
    fn signed_decimal_with_flags_width_and_precision() {
        check(b"%d\0", &[Arg::Int(42)], "42");
        check(b"%d\0", &[Arg::Int(-42)], "-42");
        check(b"%i\0", &[Arg::Int(0)], "0");
        check(b"%5d|\0", &[Arg::Int(42)], "   42|");
        check(b"%-5d|\0", &[Arg::Int(42)], "42   |");
        check(b"%05d\0", &[Arg::Int(-42)], "-0042");
        check(b"%+d %+d\0", &[Arg::Int(7), Arg::Int(-7)], "+7 -7");
        check(b"% d % d\0", &[Arg::Int(7), Arg::Int(-7)], " 7 -7");
        check(b"%.3d\0", &[Arg::Int(7)], "007");
        check(b"%6.3d|\0", &[Arg::Int(-7)], "  -007|");
        check(b"%06.3d|\0", &[Arg::Int(7)], "   007|");
        check(b"%.0d|\0", &[Arg::Int(0)], "|");
        check(b"%-2d|\0", &[Arg::Int(3)], "3 |");
        check(b"%d\0", &[Arg::Int(i32::MIN as i64)], "-2147483648");
        check(b"%lld\0", &[Arg::Int(i64::MIN)], "-9223372036854775808");
        check(b"%ld\0", &[Arg::Int(1 << 40)], "1099511627776");
        check(b"%hhd\0", &[Arg::Int(300)], "44");
        check(b"%hd\0", &[Arg::Int(70000)], "4464");
    }

    #[test]
    fn unsigned_octal_and_hexadecimal() {
        check(b"%u\0", &[Arg::Uint(4294967295)], "4294967295");
        check(
            b"%x %X\0",
            &[Arg::Uint(0xbeef), Arg::Uint(0xbeef)],
            "beef BEEF",
        );
        check(b"%08x\0", &[Arg::Uint(0x1a)], "0000001a");
        check(b"%#x %#X\0", &[Arg::Uint(255), Arg::Uint(255)], "0xff 0XFF");
        check(b"%#x\0", &[Arg::Uint(0)], "0");
        check(b"%#010x\0", &[Arg::Uint(255)], "0x000000ff");
        check(
            b"%o %#o %#o\0",
            &[Arg::Uint(8), Arg::Uint(8), Arg::Uint(0)],
            "10 010 0",
        );
        check(b"%#.0o\0", &[Arg::Uint(0)], "0");
        check(b"%.8x\0", &[Arg::Uint(0x1234)], "00001234");
        check(b"%3X|\0", &[Arg::Uint(0xa)], "  A|");
        check(b"%8.2X|\0", &[Arg::Uint(0x5)], "      05|");
        check(b"%lx\0", &[Arg::Uint(0xdead_beef_cafe)], "deadbeefcafe");
        check(b"%08llx\0", &[Arg::Uint(0x12)], "00000012");
        check(b"%llu\0", &[Arg::Uint(u64::MAX)], "18446744073709551615");
        check(
            b"%zu %zx\0",
            &[Arg::Uint(4096), Arg::Uint(4096)],
            "4096 1000",
        );
        check(b"%hhx\0", &[Arg::Uint(0x1ff)], "ff");
        // `+` and space belong to signed conversions only.
        check(b"%+u % u\0", &[Arg::Uint(5), Arg::Uint(5)], "5 5");
    }

    #[test]
    fn characters_strings_and_pointers() {
        check(b"[%c]\0", &[Arg::Int(b'A' as i64)], "[A]");
        check(
            b"[%3c][%-3c]\0",
            &[Arg::Int(b'x' as i64), Arg::Int(b'y' as i64)],
            "[  x][y  ]",
        );
        check(b"%s\0", &[text(b"hello\0")], "hello");
        check(
            b"[%8s][%-8s]\0",
            &[text(b"ab\0"), text(b"cd\0")],
            "[      ab][cd      ]",
        );
        check(b"%.2s\0", &[text(b"hello\0")], "he");
        check(b"%s\0", &[Arg::Ptr(0)], "(null)");
        check(
            b"%lc %lc\0",
            &[Arg::Int(b'q' as i64), Arg::Int(0x263a)],
            "q ?",
        );
        // `%p` is this crate's own spelling, not the host's.
        check(b"%p\0", &[Arg::Ptr(0x1000)], "0x1000");
        check(b"%p\0", &[Arg::Ptr(0)], "0x0");
        check(b"[%10p]\0", &[Arg::Ptr(0xabc)], "[     0xabc]");
    }

    #[test]
    fn a_precision_bounds_the_read_of_an_unterminated_string() {
        // No NUL in these bytes: `%.*s` must stop at its count.
        check(b"[%.*s]\0", &[Arg::Int(3), text(b"abcd")], "[abc]");
        check(b"[%.4s]\0", &[text(b"abcd")], "[abcd]");
    }

    #[test]
    fn star_width_and_precision() {
        check(b"[%*d]\0", &[Arg::Int(5), Arg::Int(42)], "[   42]");
        check(b"[%*d]\0", &[Arg::Int(-5), Arg::Int(42)], "[42   ]");
        check(b"[%*s]\0", &[Arg::Int(4), text(b"x\0")], "[   x]");
        check(b"[%.*s]\0", &[Arg::Int(2), text(b"xyz\0")], "[xy]");
        // A negative precision is as if it were absent.
        check(b"[%.*s]\0", &[Arg::Int(-1), text(b"xyz\0")], "[xyz]");
        check(b"[%.*d]\0", &[Arg::Int(4), Arg::Int(7)], "[0007]");
    }

    #[test]
    fn floats_are_formatted_and_the_arguments_after_them_line_up() {
        let (out, _, taken) = run(b"%0.3f s, %u lines\0", &[Arg::Double(1.5), Arg::Uint(9)]);
        assert_eq!(out, "1.500 s, 9 lines");
        assert_eq!(taken, ["double", "unsigned"]);
        // `L` reads a long double, in the source's own format.
        let x87_two_and_a_half = Float::from_x87(0x4000_a000_0000_0000_0000);
        let (out, _, taken) = run(
            b"a %Lf then %d\0",
            &[Arg::LongDouble(x87_two_and_a_half), Arg::Int(7)],
        );
        assert_eq!(out, "a 2.500000 then 7");
        assert_eq!(taken, ["long double", "signed"]);
        check(
            b"[%8.2e|%-8g|%+a]\0",
            &[Arg::Double(-0.0), Arg::Double(1e-5), Arg::Double(0.5)],
            "[-0.00e+00|1e-05   |+0x1p-1]",
        );
    }

    #[test]
    fn n_stores_nothing_and_unknown_conversions_consume_nothing() {
        let (out, _, taken) = run(b"ab%ncd\0", &[Arg::Ptr(0x10)]);
        assert_eq!(out, "abcd");
        assert_eq!(taken, ["pointer"]);
        let (out, _, taken) = run(b"(%rip) %d\0", &[Arg::Int(1)]);
        assert_eq!(out, "(%rip) 1");
        assert_eq!(taken, ["signed"]);
    }

    #[test]
    fn a_specification_cut_off_by_the_terminator_is_spelled_back() {
        check(b"tail %-05\0", &[], "tail %-05");
        check(b"tail %\0", &[], "tail %");
    }

    #[test]
    fn a_field_width_saturates_at_int_max() {
        let digits = b"99999999999999999999x\0";
        let format = Format {
            base: digits.as_ptr(),
            _bytes: PhantomData,
        };
        let mut at = 0;
        // SAFETY: NUL-terminated; the scan stops at the `x`.
        assert_eq!(unsafe { format.decimal(&mut at) }, MAX_FIELD);
        assert_eq!(at, 20);
    }
}
