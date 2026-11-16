//! The two text helpers every command needs and none should re-spell: a
//! bounded whitespace splitter over one command line (with `"quoted"` words,
//! so a name with spaces can be named), and a bounded line buffer for
//! composing an answer out of pieces.

/// A whitespace splitter over one completed command line.
///
/// Borrowed, never copied: a command line is a `&[u8]` the caller already
/// holds bounded, so there is nothing here to size. Copying the splitter
/// copies only its position, so a verb can look ahead (count its operands)
/// without consuming them.
#[derive(Clone, Copy, Debug)]
pub struct Args<'a> {
    rest: &'a [u8],
}

impl<'a> Args<'a> {
    pub fn new(line: &'a [u8]) -> Self {
        Self { rest: line }
    }

    /// The next word, or `None` at the end of the line. A word is a run of
    /// non-spaces, or — opened by `"` — everything up to the next `"`,
    /// spaces included and quotes excluded, so `inspect "Untitled Folder"`
    /// names the folder Files makes. A quote left open runs to the end of
    /// the line. Borrowed either way: a quoted word is a span of the line.
    pub fn next(&mut self) -> Option<&'a [u8]> {
        let start = self.skip_spaces();
        if start >= self.rest.len() {
            self.rest = &self.rest[self.rest.len()..];
            return None;
        }
        if self.rest[start] == b'"' {
            let open = start + 1;
            let close = self.rest[open..]
                .iter()
                .position(|&byte| byte == b'"')
                .map_or(self.rest.len(), |at| open + at);
            let word = &self.rest[open..close];
            self.rest = &self.rest[(close + 1).min(self.rest.len())..];
            return Some(word);
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
    fn a_quoted_word_keeps_its_spaces_and_drops_its_quotes() {
        let mut args = Args::new(b"remove \"Untitled Folder\" next \"\" \"open to the end");
        assert_eq!(args.next(), Some(&b"remove"[..]));
        assert_eq!(args.next(), Some(&b"Untitled Folder"[..]));
        assert_eq!(args.next(), Some(&b"next"[..]));
        assert_eq!(args.next(), Some(&b""[..]), "an empty quote is an empty word");
        assert_eq!(args.next(), Some(&b"open to the end"[..]));
        assert_eq!(args.next(), None);
        // A quote inside a word is part of it: only a word's first byte opens.
        let mut args = Args::new(b"a\"b c");
        assert_eq!(args.next(), Some(&b"a\"b"[..]));
        assert_eq!(args.next(), Some(&b"c"[..]));
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
