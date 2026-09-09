//! The two text helpers every command needs and none should re-spell: a
//! bounded whitespace splitter over one command line, and a bounded line
//! buffer for composing an answer out of pieces.

/// A whitespace splitter over one completed command line.
///
/// Borrowed, never copied: a command line is a `&[u8]` the caller already
/// holds bounded, so there is nothing here to size.
pub struct Args<'a> {
    rest: &'a [u8],
}

impl<'a> Args<'a> {
    pub fn new(line: &'a [u8]) -> Self {
        Self { rest: line }
    }

    /// The next space-separated word, or `None` at the end of the line.
    pub fn next(&mut self) -> Option<&'a [u8]> {
        let start = self.skip_spaces();
        if start >= self.rest.len() {
            self.rest = &self.rest[self.rest.len()..];
            return None;
        }
        let mut end = start;
        while end < self.rest.len() && self.rest[end] != b' ' {
            end += 1;
        }
        let word = &self.rest[start..end];
        self.rest = &self.rest[end..];
        Some(word)
    }

    /// Everything left on the line, leading spaces trimmed — what `write`
    /// takes as its payload, so a written line may contain spaces.
    pub fn rest(&self) -> &'a [u8] {
        &self.rest[self.skip_spaces()..]
    }

    fn skip_spaces(&self) -> usize {
        let mut at = 0;
        while at < self.rest.len() && self.rest[at] == b' ' {
            at += 1;
        }
        at
    }
}

/// Render `v` in decimal into `buf`'s prefix; returns the digit count.
pub fn dec(mut v: u64, buf: &mut [u8; 20]) -> usize {
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    let n = buf.len() - i;
    buf.copy_within(i.., 0);
    n
}

/// A bounded line under composition. Pushes past `N` are dropped, never
/// grown: the answer rides one bounded output page, and a line that does
/// not fit is truncated here rather than overflowing there.
pub struct LineBuf<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> Default for LineBuf<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> LineBuf<N> {
    pub const fn new() -> Self {
        Self {
            buf: [0; N],
            len: 0,
        }
    }

    pub fn push(&mut self, src: &[u8]) {
        let n = src.len().min(N - self.len);
        self.buf[self.len..self.len + n].copy_from_slice(&src[..n]);
        self.len += n;
    }

    pub fn push_u64(&mut self, value: u64) {
        let mut digits = [0u8; 20];
        let n = dec(value, &mut digits);
        self.push(&digits[..n]);
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_split_on_spaces_and_keep_the_rest_verbatim() {
        let mut args = Args::new(b"  write  a/b  hello world ");
        assert_eq!(args.next(), Some(&b"write"[..]));
        assert_eq!(args.next(), Some(&b"a/b"[..]));
        assert_eq!(args.rest(), b"hello world ");
        assert_eq!(args.next(), Some(&b"hello"[..]));
        assert_eq!(args.next(), Some(&b"world"[..]));
        assert_eq!(args.next(), None);
        assert_eq!(args.rest(), b"");
    }

    #[test]
    fn a_line_buffer_truncates_rather_than_grows() {
        let mut line = LineBuf::<8>::new();
        line.push(b"abcdef");
        line.push_u64(1234);
        assert_eq!(line.bytes(), b"abcdef12");
        assert_eq!(line.len(), 8);
    }

    #[test]
    fn decimal_rendering_covers_zero_and_the_widest_value() {
        let mut buf = [0u8; 20];
        let n = dec(0, &mut buf);
        assert_eq!(&buf[..n], b"0");
        let n = dec(u64::MAX, &mut buf);
        assert_eq!(&buf[..n], b"18446744073709551615");
    }
}
