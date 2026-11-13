//! The float conversions of the `printf` engine ([`super::format`]): `%e %E
//! %f %F %g %G %a %A`, of a `double` and of a `long double` in either format
//! kore's C ABIs give it — x87 double-extended on x86_64, IEEE binary128 on
//! aarch64 and riscv64.
//!
//! ## Every digit is the value's
//!
//! A finite float is `mantissa × 2^exponent` with an integer mantissa, so its
//! decimal expansion is finite, and a conversion prints that expansion
//! rounded once, to nearest with ties to even, at the last digit it keeps:
//! the correctly rounded conversion C asks for in the default rounding mode
//! (C11 7.21.6.1p13, F.5), and what glibc and musl print. A [`Decimal`] holds
//! the expansion in base-10⁹ limbs, built from the mantissa by doubling 29
//! bits or halving 9 bits at a time (musl's scheme, kept in integers), so no
//! digit is estimated.
//!
//! Halving moves digits only toward the low end: a limb below the last digit
//! a conversion can keep never changes one above it. Such limbs are folded
//! into one sticky bit as they appear, which is all a round to nearest asks
//! of them, so a conversion stores at most the digits it prints and two limbs
//! more, and never more than the expansion has: a `%.1000000f` of a `double`
//! costs what its 1074 binary places give. Doubling carries upward, so an
//! integer part is built whole: at most 4933 digits, for a `long double`.
//!
//! ## Spellings
//!
//! An infinity is `inf` and a NaN `nan`, upper case under `E F G A`, each
//! signed as any value is: a NaN's sign bit is its sign, as glibc and musl
//! print it. `%a` normalizes every nonzero value to a leading `1`, subnormals
//! included, as musl does (glibc spells a subnormal `double` with a leading
//! `0`, and an x87 value with its explicit integer bit); with a precision it
//! rounds to nearest-even, which can carry the leading digit to `2`.

use core::cmp::Ordering;

use super::{Out, SPACES, Sink, Spec, ZEROS};

/// A float as a conversion reads it: its sign bit, and what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Float {
    negative: bool,
    class: Class,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// `mantissa × 2^exponent`; a zero mantissa is a zero of that sign.
    Finite {
        mantissa: u128,
        exponent: i32,
    },
    Infinite,
    Nan,
}

/// IEEE binary64: 52 stored fraction bits under an 11-bit exponent, the sign
/// on top.
const F64_FRACTION: u32 = 52;
const F64_BIAS: i32 = 1023;
const F64_EXPONENT_MASK: u64 = 0x7ff;
const F64_SIGN: u32 = u64::BITS - 1;

/// x87 double-extended: a 64-bit significand whose integer bit is stored,
/// under a 15-bit exponent and the sign, in the low 80 bits.
const X87_SIGNIFICAND: u32 = u64::BITS;
const X87_FRACTION: u32 = X87_SIGNIFICAND - 1;
const X87_BIAS: i32 = 16383;
const X87_SIGN: u32 = X87_SIGNIFICAND + 15;

/// IEEE binary128: 112 stored fraction bits under a 15-bit exponent, the
/// sign on top.
const QUAD_FRACTION: u32 = 112;
const QUAD_BIAS: i32 = 16383;
const QUAD_SIGN: u32 = u128::BITS - 1;

/// The 15-bit exponent field both `long double` formats share.
const WIDE_EXPONENT_MASK: u128 = 0x7fff;

impl Float {
    /// A `double`.
    pub fn from_f64(value: f64) -> Self {
        let bits = value.to_bits();
        let biased = (bits >> F64_FRACTION) & F64_EXPONENT_MASK;
        let fraction = u128::from(bits & ((1 << F64_FRACTION) - 1));
        let class = match biased {
            F64_EXPONENT_MASK if fraction == 0 => Class::Infinite,
            F64_EXPONENT_MASK => Class::Nan,
            // Subnormal: the least exponent, and no implicit bit.
            0 => Class::Finite {
                mantissa: fraction,
                exponent: 1 - F64_BIAS - F64_FRACTION as i32,
            },
            _ => Class::Finite {
                mantissa: fraction | 1 << F64_FRACTION,
                exponent: biased as i32 - F64_BIAS - F64_FRACTION as i32,
            },
        };
        Self {
            negative: bits >> F64_SIGN != 0,
            class,
        }
    }

    /// An x87 double-extended `long double`: its 80 bits, low-aligned (any
    /// padding above them is ignored).
    ///
    /// The encodings no x87 since the 387 produces are read as musl reads
    /// them: an unnormal (a nonzero exponent and a clear integer bit) is a
    /// NaN, a pseudo-infinity (the top exponent, a clear integer bit and a
    /// zero fraction) an infinity, and a pseudo-denormal (a zero exponent and
    /// a set integer bit) the value its bits spell.
    pub fn from_x87(bits: u128) -> Self {
        let negative = (bits >> X87_SIGN) & 1 != 0;
        let biased = (bits >> X87_SIGNIFICAND) & WIDE_EXPONENT_MASK;
        let significand = bits as u64;
        let integer = significand >> X87_FRACTION != 0;
        let class = match biased {
            WIDE_EXPONENT_MASK if significand << 1 == 0 => Class::Infinite,
            WIDE_EXPONENT_MASK => Class::Nan,
            0 => Class::Finite {
                mantissa: u128::from(significand),
                exponent: 1 - X87_BIAS - X87_FRACTION as i32,
            },
            _ if !integer => Class::Nan,
            _ => Class::Finite {
                mantissa: u128::from(significand),
                exponent: biased as i32 - X87_BIAS - X87_FRACTION as i32,
            },
        };
        Self { negative, class }
    }

    /// An IEEE binary128 `long double`.
    pub fn from_quad(bits: u128) -> Self {
        let biased = (bits >> QUAD_FRACTION) & WIDE_EXPONENT_MASK;
        let fraction = bits & ((1 << QUAD_FRACTION) - 1);
        let class = match biased {
            WIDE_EXPONENT_MASK if fraction == 0 => Class::Infinite,
            WIDE_EXPONENT_MASK => Class::Nan,
            0 => Class::Finite {
                mantissa: fraction,
                exponent: 1 - QUAD_BIAS - QUAD_FRACTION as i32,
            },
            _ => Class::Finite {
                mantissa: fraction | 1 << QUAD_FRACTION,
                exponent: biased as i32 - QUAD_BIAS - QUAD_FRACTION as i32,
            },
        };
        Self {
            negative: bits >> QUAD_SIGN != 0,
            class,
        }
    }
}

/// The precision `e`, `f` and `g` take when none is given (C11 7.21.6.1p8).
const DEFAULT_PRECISION: i64 = 6;

/// One base-10⁹ limb's radix.
const BASE: u32 = 1_000_000_000;

/// Decimal digits per limb.
const LIMB_DIGITS: i64 = 9;

/// `POW10[k]` is `10^k`, up to a limb's radix.
const POW10: [u32; 10] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
];

/// Bits one doubling pass shifts in: a limb below 10⁹ times 2²⁹, plus the
/// carry below 2²⁹ + 1, stays under 2⁶³, and the carry out fits one limb.
const DOUBLE_STEP: u32 = 29;

/// Bits one halving pass shifts out. 10⁹ = 2⁹·5⁹, so a limb's remainder
/// below 2⁹ passes to the next limb as exactly `remainder × 10⁹ / 2⁹`, and
/// a pass adds at most one limb below the last.
const HALVE_STEP: u32 = 9;

/// Limbs kept free in front of an expansion: rounding at the digit just
/// above it materializes one limb, and a carry out of the top one more.
const HEADROOM: usize = 2;

/// Decimal digits an integer of `bits` bits can have: `⌊bits·log₁₀2⌋ + 1`,
/// by 30103/100000, which exceeds log₁₀2.
const fn decimal_digits(bits: u32) -> usize {
    bits as usize * 30103 / 100_000 + 1
}

/// Limbs a mantissa occupies before it is scaled: any `u128`.
const MANTISSA_LIMBS: usize = decimal_digits(u128::BITS).div_ceil(LIMB_DIGITS as usize);

/// The limbs a format's widest expansion needs: its largest integer part
/// (`mantissa_bits + max_exponent` bits, which doubling builds whole), or its
/// longest fraction (one digit per binary place below the point, and the
/// window's one limb more) beside the integer part a fraction can have; and
/// the rounding headroom.
const fn capacity(mantissa_bits: u32, max_exponent: i32, min_exponent: i32) -> usize {
    let whole = decimal_digits(mantissa_bits + max_exponent as u32).div_ceil(LIMB_DIGITS as usize);
    let fraction = (min_exponent.unsigned_abs() as usize).div_ceil(LIMB_DIGITS as usize)
        + 1
        + decimal_digits(mantissa_bits).div_ceil(LIMB_DIGITS as usize);
    (if whole > fraction { whole } else { fraction }) + HEADROOM
}

const fn max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

/// A `double`'s expansion: 125 limbs.
pub(super) const DOUBLE_LIMBS: usize = capacity(
    F64_FRACTION + 1,
    (2 * F64_BIAS) - F64_BIAS - F64_FRACTION as i32,
    1 - F64_BIAS - F64_FRACTION as i32,
);

/// A `long double`'s expansion, in either format: 1840 limbs.
pub(super) const LONG_DOUBLE_LIMBS: usize = max(
    capacity(
        X87_FRACTION + 1,
        (2 * X87_BIAS) - X87_BIAS - X87_FRACTION as i32,
        1 - X87_BIAS - X87_FRACTION as i32,
    ),
    capacity(
        QUAD_FRACTION + 1,
        (2 * QUAD_BIAS) - QUAD_BIAS - QUAD_FRACTION as i32,
        1 - QUAD_BIAS - QUAD_FRACTION as i32,
    ),
);

/// The decimal expansion of a finite nonzero value, most significant limb
/// first: `limbs[start..start + len]`, the first of which is `10^(9·top)`'s.
/// Digits below the last stored limb are zero, or — when `sticky` is set —
/// include a nonzero one the window let go.
struct Decimal<const N: usize> {
    limbs: [u32; N],
    start: usize,
    len: usize,
    top: i32,
    sticky: bool,
}

impl<const N: usize> Decimal<N> {
    /// `mantissa × 2^exponent` (`mantissa` nonzero), exact in every limb whose
    /// base-10⁹ exponent is at least `cut`.
    fn new(mantissa: u128, exponent: i32, cut: i64) -> Self {
        let mut low_first = [0u32; MANTISSA_LIMBS];
        let mut count = 0;
        let mut rest = mantissa;
        while rest != 0 {
            low_first[count] = (rest % u128::from(BASE)) as u32;
            rest /= u128::from(BASE);
            count += 1;
        }
        // Doubling grows toward the front and halving toward the back.
        let start = if exponent > 0 { N - count } else { HEADROOM };
        let mut decimal = Self {
            limbs: [0; N],
            start,
            len: count,
            top: count as i32 - 1,
            sticky: false,
        };
        for (slot, &limb) in decimal.limbs[start..start + count]
            .iter_mut()
            .zip(low_first[..count].iter().rev())
        {
            *slot = limb;
        }
        match exponent.cmp(&0) {
            Ordering::Greater => decimal.double(exponent.unsigned_abs()),
            Ordering::Less => decimal.halve(exponent.unsigned_abs(), cut),
            Ordering::Equal => {}
        }
        decimal
    }

    fn double(&mut self, mut shift: u32) {
        while shift > 0 {
            let step = shift.min(DOUBLE_STEP);
            let mut carry = 0u64;
            for limb in self.limbs[self.start..self.start + self.len]
                .iter_mut()
                .rev()
            {
                let wide = (u64::from(*limb) << step) + carry;
                *limb = (wide % u64::from(BASE)) as u32;
                carry = wide / u64::from(BASE);
            }
            if carry != 0 {
                self.start -= 1;
                self.len += 1;
                self.top += 1;
                self.limbs[self.start] = carry as u32;
            }
            shift -= step;
        }
    }

    fn halve(&mut self, mut shift: u32, cut: i64) {
        self.fold_below(cut);
        while shift > 0 && self.len > 0 {
            let step = shift.min(HALVE_STEP);
            let mask = (1u32 << step) - 1;
            let share = BASE >> step;
            let mut carry = 0u32;
            for limb in &mut self.limbs[self.start..self.start + self.len] {
                let rest = *limb & mask;
                *limb = (*limb >> step) + carry;
                carry = share * rest;
            }
            // Only the leading limb can empty in a pass: the one after it
            // gains the leading limb's whole remainder.
            if self.limbs[self.start] == 0 {
                self.start += 1;
                self.len -= 1;
                self.top -= 1;
            }
            if carry != 0 {
                if i64::from(self.top) - self.len as i64 >= cut {
                    self.limbs[self.start + self.len] = carry;
                    self.len += 1;
                } else {
                    self.sticky = true;
                }
            }
            shift -= step;
        }
    }

    /// Fold every limb below the base-10⁹ exponent `cut` into `sticky`.
    fn fold_below(&mut self, cut: i64) {
        let keep = (i64::from(self.top) - cut + 1).clamp(0, self.len as i64) as usize;
        let end = self.start + self.len;
        if self.limbs[self.start + keep..end]
            .iter()
            .any(|&limb| limb != 0)
        {
            self.sticky = true;
        }
        self.len = keep;
    }

    fn is_zero(&self) -> bool {
        self.len == 0
    }

    /// The position of the leading digit: `p` for the digit worth `10^p`.
    /// The expansion is nonzero.
    fn leading(&self) -> i64 {
        let first = self.limbs[self.start];
        let digits = POW10
            .iter()
            .position(|&power| first < power)
            .unwrap_or(POW10.len());
        LIMB_DIGITS * i64::from(self.top) + digits as i64 - 1
    }

    /// The digit at `position`.
    #[cfg(test)]
    fn digit(&self, position: i64) -> u32 {
        let back = i64::from(self.top) - position.div_euclid(LIMB_DIGITS);
        if !(0..self.len as i64).contains(&back) {
            return 0;
        }
        let limb = self.limbs[self.start + back as usize];
        limb / POW10[position.rem_euclid(LIMB_DIGITS) as usize] % 10
    }

    /// Keep the digits at `position` and above, rounded to nearest, ties to
    /// even, on everything below.
    fn round(&mut self, position: i64) {
        let below_everything = self.is_zero() || position - 1 > self.leading();
        let limb = position.div_euclid(LIMB_DIGITS);
        let lowest = i64::from(self.top) - self.len as i64 + 1;
        if below_everything {
            // Under a tenth of the last kept digit's unit: zero. An empty
            // expansion here had every digit folded below its window, which
            // lies under the digit after `position`.
            self.len = 0;
            self.sticky = false;
            return;
        }
        if limb < lowest {
            // Every stored digit is kept, and what lies below them is under a
            // tenth of the last kept digit's unit (the window reaches below
            // `position`'s limb whenever anything was folded).
            self.sticky = false;
            return;
        }
        if limb > i64::from(self.top) {
            // `position - 1` is the top digit of the leading limb, so the
            // kept digit is the first of the limb above: materialize it.
            self.start -= 1;
            self.len += 1;
            self.top += 1;
            self.limbs[self.start] = 0;
        }
        let at = self.start + (i64::from(self.top) - limb) as usize;
        let end = self.start + self.len;
        let k = position.rem_euclid(LIMB_DIGITS) as usize;
        let (round_digit, rest_nonzero) = if k > 0 {
            let under = self.limbs[at] % POW10[k];
            (
                under / POW10[k - 1],
                under % POW10[k - 1] != 0 || self.limbs[at + 1..end].iter().any(|&l| l != 0),
            )
        } else if at + 1 < end {
            let next = self.limbs[at + 1];
            let top_digit = POW10[LIMB_DIGITS as usize - 1];
            (
                next / top_digit,
                next % top_digit != 0 || self.limbs[at + 2..end].iter().any(|&l| l != 0),
            )
        } else {
            (0, false)
        };
        let rest_nonzero = rest_nonzero || self.sticky;
        self.limbs[at] -= self.limbs[at] % POW10[k];
        self.len = at + 1 - self.start;
        self.sticky = false;
        let odd = self.limbs[at] / POW10[k] % 2 == 1;
        if round_digit > 5 || (round_digit == 5 && (rest_nonzero || odd)) {
            let mut carry_at = at;
            self.limbs[carry_at] += POW10[k];
            while self.limbs[carry_at] >= BASE {
                self.limbs[carry_at] -= BASE;
                if carry_at == self.start {
                    self.start -= 1;
                    self.len += 1;
                    self.top += 1;
                    self.limbs[self.start] = 0;
                }
                carry_at -= 1;
                self.limbs[carry_at] += 1;
            }
        }
        while self.len > 0 && self.limbs[self.start] == 0 {
            self.start += 1;
            self.len -= 1;
            self.top -= 1;
        }
    }

    /// The position of the lowest nonzero stored digit.
    fn lowest(&self) -> Option<i64> {
        let stored = &self.limbs[self.start..self.start + self.len];
        let back = stored.iter().rposition(|&limb| limb != 0)?;
        let limb = stored[back];
        let zeros = POW10[1..]
            .iter()
            .take_while(|&&power| limb % power == 0)
            .count();
        Some(LIMB_DIGITS * (i64::from(self.top) - back as i64) + zeros as i64)
    }

    /// Write the digits at positions `from` down to `to`, inclusive.
    fn write<S: Sink>(&self, out: &mut Out<'_, S>, from: i64, to: i64) {
        if from < to {
            return;
        }
        let mut position = from;
        let (high, low) = if self.is_zero() {
            (i64::MIN, i64::MAX)
        } else {
            (
                LIMB_DIGITS * i64::from(self.top) + LIMB_DIGITS - 1,
                LIMB_DIGITS * (i64::from(self.top) - self.len as i64 + 1),
            )
        };
        if position > high {
            let zeros = position - high.max(to - 1);
            out.pad(&ZEROS, zeros as usize);
            position -= zeros;
        }
        while position >= to && position >= low {
            let limb = position.div_euclid(LIMB_DIGITS);
            let at = self.start + (i64::from(self.top) - limb) as usize;
            let mut text = [b'0'; LIMB_DIGITS as usize];
            let mut value = self.limbs[at];
            for byte in text.iter_mut().rev() {
                *byte = b'0' + (value % 10) as u8;
                value /= 10;
            }
            let hi = position.rem_euclid(LIMB_DIGITS) as usize;
            let lo = if to.div_euclid(LIMB_DIGITS) == limb {
                to.rem_euclid(LIMB_DIGITS) as usize
            } else {
                0
            };
            let last = LIMB_DIGITS as usize - 1;
            out.put(&text[last - hi..=last - lo]);
            position -= (hi - lo + 1) as i64;
        }
        if position >= to {
            out.pad(&ZEROS, (position - to + 1) as usize);
        }
    }
}

/// Write the digits of `digits` (all zeros for a zero) from `from` down to
/// `to`.
fn write_digits<const N: usize, S: Sink>(
    out: &mut Out<'_, S>,
    digits: Option<&Decimal<N>>,
    from: i64,
    to: i64,
) {
    match digits {
        Some(digits) => digits.write(out, from, to),
        None if from >= to => out.pad(&ZEROS, (from - to + 1) as usize),
        None => {}
    }
}

/// The lowest base-10⁹ limb a rounding at `position` must read exactly: the
/// one holding the digit after it, and one more. Nothing below the
/// expansion's last digit is ever needed: a fraction of `-exponent` binary
/// places has exactly that many decimal ones.
fn window(position: i64, exponent: i32) -> i64 {
    let cut = (position - 1).div_euclid(LIMB_DIGITS) - 1;
    cut.max(i64::from(exponent).div_euclid(LIMB_DIGITS) - 1)
}

/// log₁₀2 = 0.30102999566398…, truncated to eleven places: below the truth,
/// by less than 10⁻¹¹.
const LOG10_2_NUMERATOR: i64 = 30_102_999_566;
const LOG10_2_DENOMINATOR: i64 = 100_000_000_000;

/// A position at or below that of `mantissa × 2^exponent`'s leading digit.
/// The value is at least `2^power` for `power = bits − 1 + exponent`, whose
/// leading digit sits at `⌊power·log₁₀2⌋`; the truncated logarithm moves
/// that floor by at most one over every exponent a `long double` has, and
/// one digit of slack absorbs it.
fn leading_floor(mantissa: u128, exponent: i32) -> i64 {
    let power = i64::from(u128::BITS - mantissa.leading_zeros()) - 1 + i64::from(exponent);
    (power * LOG10_2_NUMERATOR).div_euclid(LOG10_2_DENOMINATOR) - 1
}

/// `spec`'s precision for `e`, `f` and `g`.
fn precision(spec: Spec) -> i64 {
    spec.precision
        .map_or(DEFAULT_PRECISION, |precision| precision as i64)
}

/// Format `value` by `conversion` (`e E f F g G a A`) under `spec`, its
/// expansion held in at most `N` limbs. Never inlined into the engine: a
/// `long double`'s limbs are 7 KiB of stack, which only its conversions pay.
#[inline(never)]
pub(super) fn convert<const N: usize, S: Sink>(
    out: &mut Out<'_, S>,
    spec: Spec,
    conversion: u8,
    value: Float,
) {
    let upper = conversion.is_ascii_uppercase();
    let sign: &[u8] = if value.negative {
        b"-"
    } else if spec.plus {
        b"+"
    } else if spec.space {
        b" "
    } else {
        b""
    };
    let (mantissa, exponent) = match value.class {
        Class::Infinite => return special(out, spec, sign, if upper { b"INF" } else { b"inf" }),
        Class::Nan => return special(out, spec, sign, if upper { b"NAN" } else { b"nan" }),
        Class::Finite { mantissa, exponent } => (mantissa, exponent),
    };
    match conversion.to_ascii_lowercase() {
        b'a' => hexadecimal(out, spec, sign, upper, mantissa, exponent),
        b'f' => fixed::<N, S>(out, spec, sign, mantissa, exponent),
        b'e' => scientific::<N, S>(out, spec, sign, upper, mantissa, exponent),
        _ => general::<N, S>(out, spec, sign, upper, mantissa, exponent),
    }
}

/// Lay out a number of `body` bytes after `sign` and `prefix` in `spec`'s
/// field: spaces before it, or after it under `-`, or zeros between the
/// prefix and the body under `0` (which `-` overrides).
fn field<S: Sink>(
    out: &mut Out<'_, S>,
    spec: Spec,
    sign: &[u8],
    prefix: &[u8],
    body: usize,
    write: impl FnOnce(&mut Out<'_, S>),
) {
    let fill = spec.width.saturating_sub(sign.len() + prefix.len() + body);
    let zeros = spec.zero && !spec.left;
    if !spec.left && !zeros {
        out.pad(&SPACES, fill);
    }
    out.put(sign);
    out.put(prefix);
    if zeros {
        out.pad(&ZEROS, fill);
    }
    write(out);
    if spec.left {
        out.pad(&SPACES, fill);
    }
}

/// An infinity or a NaN: a word, which the `0` flag never pads.
fn special<S: Sink>(out: &mut Out<'_, S>, spec: Spec, sign: &[u8], word: &[u8; 3]) {
    let spec = Spec {
        zero: false,
        ..spec
    };
    field(out, spec, sign, b"", word.len(), |out| out.put(word));
}

/// `%f`: the digits down to the precision's last.
fn fixed<const N: usize, S: Sink>(
    out: &mut Out<'_, S>,
    spec: Spec,
    sign: &[u8],
    mantissa: u128,
    exponent: i32,
) {
    let precision = precision(spec);
    let digits = (mantissa != 0).then(|| {
        let mut digits = Decimal::<N>::new(mantissa, exponent, window(-precision, exponent));
        digits.round(-precision);
        digits
    });
    fixed_layout(out, spec, sign, digits.as_ref(), precision);
}

/// `%e`: one digit, the precision's after the point, and the exponent.
fn scientific<const N: usize, S: Sink>(
    out: &mut Out<'_, S>,
    spec: Spec,
    sign: &[u8],
    upper: bool,
    mantissa: u128,
    exponent: i32,
) {
    let precision = precision(spec);
    let (digits, leading) = significant::<N>(mantissa, exponent, precision);
    exponent_layout(out, spec, sign, upper, digits.as_ref(), leading, precision);
}

/// `%g`: `%e`'s rounding to the precision's significant digits, laid out as
/// `%f` when the exponent is at least −4 and below the precision, without
/// trailing fraction zeros unless `#` keeps them (C11 7.21.6.1p8).
fn general<const N: usize, S: Sink>(
    out: &mut Out<'_, S>,
    spec: Spec,
    sign: &[u8],
    upper: bool,
    mantissa: u128,
    exponent: i32,
) {
    let significant_digits = match spec.precision {
        Some(0) => 1,
        Some(precision) => precision as i64,
        None => DEFAULT_PRECISION,
    };
    let (digits, leading) = significant::<N>(mantissa, exponent, significant_digits - 1);
    let lowest = digits.as_ref().and_then(Decimal::lowest);
    if significant_digits > leading && leading >= -4 {
        let precision = significant_digits - 1 - leading;
        let fraction = if spec.alternate {
            precision
        } else {
            lowest.map_or(0, |low| (-low).clamp(0, precision))
        };
        fixed_layout(out, spec, sign, digits.as_ref(), fraction);
    } else {
        let precision = significant_digits - 1;
        let fraction = if spec.alternate {
            precision
        } else {
            lowest.map_or(0, |low| (leading - low).clamp(0, precision))
        };
        exponent_layout(out, spec, sign, upper, digits.as_ref(), leading, fraction);
    }
}

/// The expansion rounded to `precision` digits after its leading one, and
/// that leading digit's position (`0` for a zero). The window reaches from a
/// bound below the leading digit, so the expansion is never empty.
fn significant<const N: usize>(
    mantissa: u128,
    exponent: i32,
    precision: i64,
) -> (Option<Decimal<N>>, i64) {
    if mantissa == 0 {
        return (None, 0);
    }
    let floor = leading_floor(mantissa, exponent);
    let mut digits = Decimal::<N>::new(mantissa, exponent, window(floor - precision, exponent));
    let leading = digits.leading();
    digits.round(leading - precision);
    // A carry can open a new leading digit: 9.96 at one place is 1.0e1.
    let leading = digits.leading();
    (Some(digits), leading)
}

/// `[-]d…d[.d…d]`: the integer digits (one `0` when there are none), then the
/// point and `fraction` digits when there are any or `#` asks for the point.
fn fixed_layout<const N: usize, S: Sink>(
    out: &mut Out<'_, S>,
    spec: Spec,
    sign: &[u8],
    digits: Option<&Decimal<N>>,
    fraction: i64,
) {
    let leading = digits
        .filter(|digits| !digits.is_zero())
        .map(Decimal::leading);
    let whole = leading.map_or(1, |leading| leading.max(0) + 1);
    let point = fraction > 0 || spec.alternate;
    let body = whole as usize + usize::from(point) + fraction as usize;
    field(out, spec, sign, b"", body, |out| {
        write_digits(out, digits, whole - 1, 0);
        if point {
            out.put(b".");
        }
        write_digits(out, digits, -1, -fraction);
    });
}

/// `[-]d[.d…d]e±dd`: the leading digit, then the point and `fraction` digits
/// when there are any or `#` asks for the point, then the exponent in at
/// least two digits.
fn exponent_layout<const N: usize, S: Sink>(
    out: &mut Out<'_, S>,
    spec: Spec,
    sign: &[u8],
    upper: bool,
    digits: Option<&Decimal<N>>,
    leading: i64,
    fraction: i64,
) {
    let point = fraction > 0 || spec.alternate;
    let mut text = [0u8; EXPONENT_TEXT];
    let power = exponent_text(&mut text, if upper { b'E' } else { b'e' }, leading, 2);
    let body = 1 + usize::from(point) + fraction as usize + power.len();
    field(out, spec, sign, b"", body, |out| {
        write_digits(out, digits, leading, leading);
        if point {
            out.put(b".");
        }
        write_digits(out, digits, leading - 1, leading - fraction);
        out.put(power);
    });
}

/// The letter, the sign and up to six digits: a binary exponent of a
/// `long double` subnormal (−16494) is the widest.
const EXPONENT_TEXT: usize = 8;

/// `letter`, a sign, and `value`'s decimal digits, at least `min_digits`.
fn exponent_text(
    text: &mut [u8; EXPONENT_TEXT],
    letter: u8,
    value: i64,
    min_digits: usize,
) -> &[u8] {
    let mut digits = [b'0'; EXPONENT_TEXT - 2];
    let mut at = digits.len();
    let mut rest = value.unsigned_abs();
    while rest != 0 || digits.len() - at < min_digits {
        at -= 1;
        digits[at] = b'0' + (rest % 10) as u8;
        rest /= 10;
    }
    let count = digits.len() - at;
    text[0] = letter;
    text[1] = if value < 0 { b'-' } else { b'+' };
    text[2..2 + count].copy_from_slice(&digits[at..]);
    &text[..2 + count]
}

/// `%a`: `0x1.h…hp±d`, the value normalized to a leading `1` and every
/// fraction digit it has — or, with a precision, that many, rounded to
/// nearest-even.
fn hexadecimal<S: Sink>(
    out: &mut Out<'_, S>,
    spec: Spec,
    sign: &[u8],
    upper: bool,
    mantissa: u128,
    exponent: i32,
) {
    let alphabet: &[u8; 16] = if upper {
        b"0123456789ABCDEF"
    } else {
        b"0123456789abcdef"
    };
    let prefix: &[u8] = if upper { b"0X" } else { b"0x" };
    let (mut lead, mut fraction, mut count, power) = if mantissa == 0 {
        (0u8, 0u128, 0usize, 0i64)
    } else {
        // The bits below the leading one, padded out to whole hex digits,
        // without the trailing zero digits a precision did not ask for.
        let tail = u128::BITS - 1 - mantissa.leading_zeros();
        let digits = tail.div_ceil(4);
        let mut fraction = (mantissa ^ (1 << tail)) << (4 * digits - tail);
        let mut count = digits as usize;
        while count > 0 && fraction & 0xf == 0 {
            fraction >>= 4;
            count -= 1;
        }
        (1u8, fraction, count, i64::from(exponent) + i64::from(tail))
    };
    let mut zeros = 0usize;
    if let Some(precision) = spec.precision {
        if precision < count {
            let dropped = 4 * (count - precision) as u32;
            let half = 1u128 << (dropped - 1);
            let rest = fraction & ((1u128 << dropped) - 1);
            fraction >>= dropped;
            let odd = if precision == 0 {
                lead & 1 == 1
            } else {
                fraction & 1 == 1
            };
            if rest > half || (rest == half && odd) {
                fraction += 1;
                if precision == 0 || fraction >> (4 * precision) != 0 {
                    lead += 1;
                    fraction = 0;
                }
            }
            count = precision;
        } else {
            zeros = precision - count;
        }
    }
    let point = count + zeros > 0 || spec.alternate;
    let mut text = [0u8; EXPONENT_TEXT];
    let power = exponent_text(&mut text, if upper { b'P' } else { b'p' }, power, 1);
    let body = 1 + usize::from(point) + count + zeros + power.len();
    field(out, spec, sign, prefix, body, |out| {
        out.put(&[alphabet[usize::from(lead)]]);
        if point {
            out.put(b".");
        }
        let mut digits = [0u8; HEX_FRACTION_DIGITS];
        for (i, digit) in digits[..count].iter_mut().enumerate() {
            *digit = alphabet[((fraction >> (4 * (count - 1 - i))) & 0xf) as usize];
        }
        out.put(&digits[..count]);
        out.pad(&ZEROS, zeros);
        out.put(power);
    });
}

/// Hex digits below the leading one any mantissa can have: a `u128`'s.
const HEX_FRACTION_DIGITS: usize = (u128::BITS as usize - 1).div_ceil(4);

#[cfg(test)]
mod tests;
