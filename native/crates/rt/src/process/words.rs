//! **The words of an argument tail.**  A launcher hands a program its
//! arguments as one opaque byte string ([`ProcessIo::args`]); a program that
//! wants an argument vector splits it here, the way a POSIX shell splits a
//! command line into words — with quoting, without expansion:
//!
//! - unquoted whitespace separates words;
//! - `'…'` is literal to the next `'`;
//! - `"…"` is literal to the next unescaped `"`, where `\"` and `\\` stand for
//!   `"` and `\` and every other backslash is itself;
//! - an unquoted `\` takes the next byte literally (`\ ` is a space inside a
//!   word);
//! - quoted and unquoted pieces that touch form one word (`a"b c"d` is
//!   `ab cd`), and `""` or `''` alone is an empty word.
//!
//! So `-c 'print("hello world")'` passes `print("hello world")` whole. Bytes
//! at or above 0x80 never equal a quote, a backslash or ASCII whitespace, so
//! UTF-8 passes through unchanged. Splitting allocates nothing: [`words`]
//! validates each word as it finds it, and [`Word::bytes`] decodes it lazily.
//!
//! [`ProcessIo::args`]: kore_abi::process::ProcessIo::args

use core::fmt;

/// Why an argument tail does not split into words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordsError {
    /// The quote (`'` or `"`) opened at byte `at` is never closed.
    Unterminated { quote: u8, at: usize },
    /// The tail ends in a backslash with nothing left to escape.
    TrailingBackslash,
}

impl fmt::Display for WordsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Unterminated { quote, at } => {
                write!(
                    f,
                    "the {} opened at byte {at} is never closed",
                    char::from(quote)
                )
            }
            Self::TrailingBackslash => f.write_str("the arguments end in a lone backslash"),
        }
    }
}

/// The words of `line`, in order ([`Words`]).
pub fn words(line: &[u8]) -> Words<'_> {
    Words { line, at: 0 }
}

/// An iterator over the words of an argument tail. The first malformed word
/// is reported once and ends the iteration.
#[derive(Clone, Debug)]
pub struct Words<'a> {
    line: &'a [u8],
    at: usize,
}

impl<'a> Iterator for Words<'a> {
    type Item = Result<Word<'a>, WordsError>;

    fn next(&mut self) -> Option<Self::Item> {
        let rest = self.line.get(self.at..)?;
        let start = self.at + rest.iter().take_while(|b| b.is_ascii_whitespace()).count();
        if start == self.line.len() {
            self.at = start;
            return None;
        }
        match word_end(self.line, start) {
            Ok(end) => {
                self.at = end;
                Some(Ok(Word {
                    raw: &self.line[start..end],
                }))
            }
            Err(error) => {
                self.at = self.line.len();
                Some(Err(error))
            }
        }
    }
}

impl core::iter::FusedIterator for Words<'_> {}

/// Where the word that starts at `start` ends: the first unquoted whitespace,
/// or the end of the line.
fn word_end(line: &[u8], start: usize) -> Result<usize, WordsError> {
    let mut i = start;
    while let Some(&byte) = line.get(i) {
        match byte {
            b if b.is_ascii_whitespace() => return Ok(i),
            b'\\' => {
                if i + 1 == line.len() {
                    return Err(WordsError::TrailingBackslash);
                }
                i += 2;
            }
            b'\'' => {
                let close = line[i + 1..].iter().position(|&b| b == b'\'').ok_or(
                    WordsError::Unterminated {
                        quote: b'\'',
                        at: i,
                    },
                )?;
                i += close + 2;
            }
            b'"' => {
                let mut j = i + 1;
                loop {
                    match line.get(j) {
                        None => return Err(WordsError::Unterminated { quote: b'"', at: i }),
                        Some(b'"') => break,
                        Some(b'\\') if matches!(line.get(j + 1), Some(b'"' | b'\\')) => j += 2,
                        Some(_) => j += 1,
                    }
                }
                i = j + 1;
            }
            _ => i += 1,
        }
    }
    Ok(i)
}

/// One word of an argument tail, as written (quotes and escapes included);
/// [`bytes`](Self::bytes) is what it stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Word<'a> {
    raw: &'a [u8],
}

impl<'a> Word<'a> {
    /// The word as written on the line.
    pub fn raw(&self) -> &'a [u8] {
        self.raw
    }

    /// The word's bytes with its quoting removed.
    pub fn bytes(&self) -> Unquoted<'a> {
        Unquoted {
            raw: self.raw,
            at: 0,
            quote: None,
        }
    }
}

/// The bytes a [`Word`] stands for, decoded as they are read.
#[derive(Clone, Debug)]
pub struct Unquoted<'a> {
    raw: &'a [u8],
    at: usize,
    /// The quote currently open, if any.
    quote: Option<u8>,
}

impl Iterator for Unquoted<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        loop {
            let byte = *self.raw.get(self.at)?;
            self.at += 1;
            match (self.quote, byte) {
                (None, b'\'' | b'"') => self.quote = Some(byte),
                (Some(open), b) if b == open => self.quote = None,
                (None, b'\\') => {
                    let escaped = *self.raw.get(self.at)?;
                    self.at += 1;
                    return Some(escaped);
                }
                (Some(b'"'), b'\\') if matches!(self.raw.get(self.at), Some(b'"' | b'\\')) => {
                    let escaped = self.raw[self.at];
                    self.at += 1;
                    return Some(escaped);
                }
                (_, b) => return Some(b),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    /// Every word's decoded bytes, or the first error.
    fn split(line: &str) -> Result<Vec<Vec<u8>>, WordsError> {
        words(line.as_bytes())
            .map(|word| word.map(|word| word.bytes().collect()))
            .collect()
    }

    fn ok(line: &str) -> Vec<Vec<u8>> {
        split(line).unwrap_or_else(|error| panic!("{line:?}: {error}"))
    }

    fn words_of(items: &[&str]) -> Vec<Vec<u8>> {
        items.iter().map(|item| item.as_bytes().to_vec()).collect()
    }

    #[test]
    fn unquoted_whitespace_separates_words() {
        assert_eq!(ok("a bc  d\te\n"), words_of(&["a", "bc", "d", "e"]));
        assert_eq!(
            ok("   lead and trail   "),
            words_of(&["lead", "and", "trail"])
        );
        assert!(ok("").is_empty());
        assert!(ok(" \t \n").is_empty());
    }

    #[test]
    fn single_quotes_are_literal() {
        assert_eq!(
            ok(r#"-c 'print("hello world")'"#),
            words_of(&["-c", r#"print("hello world")"#])
        );
        assert_eq!(ok(r"'a\b \' x"), words_of(&[r"a\b \", "x"]));
    }

    #[test]
    fn double_quotes_escape_only_quote_and_backslash() {
        assert_eq!(ok(r#""a b""#), words_of(&["a b"]));
        assert_eq!(ok(r#""say \"hi\"""#), words_of(&[r#"say "hi""#]));
        assert_eq!(ok(r#""c:\\dir""#), words_of(&[r"c:\dir"]));
        assert_eq!(
            ok(r#""keep \n and \' as written""#),
            words_of(&[r"keep \n and \' as written"])
        );
        assert_eq!(ok(r#""it's""#), words_of(&["it's"]));
    }

    #[test]
    fn an_unquoted_backslash_takes_the_next_byte() {
        assert_eq!(ok(r"my\ file.txt next"), words_of(&["my file.txt", "next"]));
        assert_eq!(ok(r#"\"quoted\" \\"#), words_of(&[r#""quoted""#, r"\"]));
    }

    #[test]
    fn touching_pieces_form_one_word() {
        assert_eq!(ok(r#"a"b c"'d e'f g"#), words_of(&["ab cd ef", "g"]));
        assert_eq!(ok(r#"--name="two words""#), words_of(&["--name=two words"]));
    }

    #[test]
    fn empty_quotes_are_empty_words() {
        assert_eq!(ok(r#"a "" b ''"#), words_of(&["a", "", "b", ""]));
    }

    #[test]
    fn utf8_passes_through() {
        assert_eq!(ok("\"héllo wörld\" ü"), words_of(&["héllo wörld", "ü"]));
    }

    #[test]
    fn a_malformed_tail_is_refused_and_ends_the_words() {
        assert_eq!(
            split("ok 'open"),
            Err(WordsError::Unterminated {
                quote: b'\'',
                at: 3
            })
        );
        assert_eq!(
            split(r#"ok "open \""#),
            Err(WordsError::Unterminated { quote: b'"', at: 3 })
        );
        assert_eq!(split(r"ok \"), Err(WordsError::TrailingBackslash));
        let mut iter = words(b"a 'b c");
        assert!(matches!(iter.next(), Some(Ok(_))));
        assert!(matches!(iter.next(), Some(Err(_))));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn a_word_keeps_its_raw_spelling() {
        let mut iter = words(br#"x "a b"c"#);
        assert_eq!(iter.next().unwrap().unwrap().raw(), b"x");
        assert_eq!(iter.next().unwrap().unwrap().raw(), br#""a b"c"#);
    }
}
