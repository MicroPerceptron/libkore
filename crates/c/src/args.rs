//! **A C program's arguments**: `main`'s `argc` and `argv`.
//!
//! A launcher hands a program its command line on the process page: the name
//! it resolved the program by ([`ProcessIo::command`]) and the opaque tail
//! after it ([`ProcessIo::args`]). C wants a vector. The host reads the page
//! once, on thread 0 before the main thread exists, splits the tail the way
//! every kore program that wants words does ([`words`]: a POSIX shell's
//! quoting, without expansion), and keeps the vector for the life of the
//! program. The C library's start asks for it through
//! [`__kore_program_arguments`] and passes it to `main`.
//!
//! `argv[0]` is the command name, or the empty string when the launcher gave
//! none (C11 §5.1.2.2.1p2). A launch with no page at all has no arguments,
//! not even a name: `argc` is zero, which the same clause permits. The
//! strings are the program's to change, as that clause requires, and
//! `argv[argc]` is null.
//!
//! A tail that does not split, an unclosed quote or a lone trailing
//! backslash, is the launcher's line being malformed, not something `main`
//! could be handed: the host says so on `stderr` and ends the program before
//! it runs, as `wasm` and `tcc` answer the same line.
//!
//! [`ProcessIo::command`]: kore_std::abi::process::ProcessIo::command
//! [`ProcessIo::args`]: kore_std::abi::process::ProcessIo::args
//! [`words`]: kore_std::process::words

extern crate alloc;

use alloc::vec::Vec;

use kore_std::process::words::{self, WordsError};

/// An argument vector's storage: every word NUL-terminated, back to back,
/// and where each begins.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Vector {
    bytes: Vec<u8>,
    starts: Vec<usize>,
}

impl Vector {
    /// `command`, then the words of `tail`.
    ///
    /// A NUL inside a word would end the C string early and hand `main` a
    /// different argument than the line spelled, so such a line is no
    /// vector either.
    pub(crate) fn of(command: &[u8], tail: &[u8]) -> Result<Self, Malformed> {
        let mut vector = Self {
            bytes: Vec::new(),
            starts: Vec::new(),
        };
        vector.push(command.iter().copied())?;
        for word in words::words(tail) {
            vector.push(word.map_err(Malformed::Words)?.bytes())?;
        }
        Ok(vector)
    }

    fn push(&mut self, word: impl Iterator<Item = u8>) -> Result<(), Malformed> {
        self.starts.push(self.bytes.len());
        for byte in word {
            if byte == 0 {
                return Err(Malformed::Nul {
                    word: self.starts.len() - 1,
                });
            }
            self.bytes.push(byte);
        }
        self.bytes.push(0);
        Ok(())
    }

    /// Words held: `argc`.
    pub(crate) fn len(&self) -> usize {
        self.starts.len()
    }

    /// Word `index`, without its terminator.
    #[cfg(test)]
    fn word(&self, index: usize) -> &[u8] {
        let start = self.starts[index];
        let end = self.bytes[start..]
            .iter()
            .position(|byte| *byte == 0)
            .unwrap();
        &self.bytes[start..start + end]
    }
}

/// Why a command line is no argument vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Malformed {
    /// The tail does not split into words.
    Words(WordsError),
    /// Word `word` (0 is the command name) holds a NUL byte.
    Nul { word: usize },
}

impl core::fmt::Display for Malformed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Words(error) => error.fmt(f),
            Self::Nul { word } => write!(f, "argument {word} holds a NUL byte"),
        }
    }
}

#[cfg(target_os = "none")]
pub(crate) use metal::capture;

#[cfg(target_os = "none")]
mod metal {
    use super::alloc::boxed::Box;
    use super::alloc::vec::Vec;
    use super::{Malformed, Vector};
    use core::ffi::{c_char, c_int};
    use core::sync::atomic::{AtomicI32, AtomicPtr, Ordering};
    use kore_std::runtime::entry::Rt;

    /// `argc`, and the vector `argv` points at: `argc` pointers and a null.
    /// Published once by [`capture`] on thread 0 before the main thread is
    /// founded, whose founding orders these stores before `main`'s loads.
    static ARGC: AtomicI32 = AtomicI32::new(0);
    static ARGV: AtomicPtr<*mut c_char> = AtomicPtr::new(core::ptr::null_mut());

    /// Read the launcher's page and keep the program's argument vector.
    /// Thread 0, once, before any other thread exists. A launch that carried
    /// no page leaves the program without arguments.
    pub(crate) fn capture(rt: &Rt) -> Result<(), Malformed> {
        let Some(page) = kore_std::process::map(rt) else {
            return Ok(());
        };
        // A page that does not validate is no command line: no arguments,
        // as for a launch without one.
        let (Ok(command), Ok(tail)) = (page.command(), page.args()) else {
            return Ok(());
        };
        let Vector { bytes, starts } = Vector::of(command, tail)?;
        // The program's for its whole life, and its to change: never freed.
        let bytes: &'static mut [u8] = Box::leak(bytes.into_boxed_slice());
        let base = bytes.as_mut_ptr();
        let mut pointers: Vec<*mut c_char> = starts
            .iter()
            // SAFETY: each start is an offset inside `bytes`.
            .map(|start| unsafe { base.add(*start) }.cast::<c_char>())
            .collect();
        pointers.push(core::ptr::null_mut());
        let argc = c_int::try_from(starts.len()).unwrap_or(c_int::MAX);
        let pointers: &'static mut [*mut c_char] = Box::leak(pointers.into_boxed_slice());
        ARGV.store(pointers.as_mut_ptr(), Ordering::Release);
        ARGC.store(argc, Ordering::Release);
        Ok(())
    }

    /// The program's argument vector: answers `argc` and stores `argv`
    /// through `argv`, `argv[argc]` null. The C library's start calls it on
    /// the main thread before `main`. A program whose launch carried no
    /// command line gets zero and a vector holding only the null.
    ///
    /// # Safety
    /// `argv` is null or names one writable `char **`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __kore_program_arguments(argv: *mut *mut *mut c_char) -> c_int {
        /// The vector of a program without arguments: its null.
        struct Empty(core::cell::UnsafeCell<[*mut c_char; 1]>);
        // SAFETY: never written: C may not store through `argv[argc]`'s
        // slot, and there is no other.
        unsafe impl Sync for Empty {}
        static EMPTY: Empty = Empty(core::cell::UnsafeCell::new([core::ptr::null_mut()]));

        let held = ARGV.load(Ordering::Acquire);
        let (count, vector) = if held.is_null() {
            (0, EMPTY.0.get().cast::<*mut c_char>())
        } else {
            (ARGC.load(Ordering::Acquire), held)
        };
        if !argv.is_null() {
            // SAFETY: the caller's contract.
            unsafe { argv.write(vector) };
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(command: &[u8], tail: &[u8]) -> Vec<Vec<u8>> {
        let vector = Vector::of(command, tail).unwrap();
        (0..vector.len())
            .map(|index| vector.word(index).to_vec())
            .collect()
    }

    #[test]
    fn the_command_is_the_first_word_and_the_tail_follows_as_a_shell_splits_it() {
        assert_eq!(vector(b"answer", b""), [b"answer".to_vec()]);
        assert_eq!(
            vector(b"copy", b"  a.txt   b.txt "),
            [b"copy".to_vec(), b"a.txt".to_vec(), b"b.txt".to_vec()]
        );
        assert_eq!(
            vector(b"say", br#"-c 'two words' "a \"quote\"" back\ slash """#),
            [
                b"say".to_vec(),
                b"-c".to_vec(),
                b"two words".to_vec(),
                br#"a "quote""#.to_vec(),
                b"back slash".to_vec(),
                b"".to_vec(),
            ]
        );
        // A launcher that names nothing still gives `argv[0]`, empty.
        assert_eq!(vector(b"", b"x"), [b"".to_vec(), b"x".to_vec()]);
    }

    #[test]
    fn every_word_is_its_own_terminated_string() {
        let vector = Vector::of(b"p", b"ab c").unwrap();
        assert_eq!(vector.bytes, b"p\0ab\0c\0");
        assert_eq!(vector.starts, [0, 2, 5]);
        assert_eq!(vector.len(), 3);
    }

    #[test]
    fn a_line_that_does_not_split_is_no_vector() {
        assert_eq!(
            Vector::of(b"p", b"'open"),
            Err(Malformed::Words(WordsError::Unterminated {
                quote: b'\'',
                at: 0
            }))
        );
        assert_eq!(
            Vector::of(b"p", b"a \\"),
            Err(Malformed::Words(WordsError::TrailingBackslash))
        );
        // A NUL would end the C string early: `a\0b` is not the word `a`.
        assert_eq!(Vector::of(b"p", b"x a\0b"), Err(Malformed::Nul { word: 2 }));
        assert_eq!(Vector::of(b"p\0q", b""), Err(Malformed::Nul { word: 0 }));
        let mut text = alloc::string::String::new();
        core::fmt::Write::write_fmt(&mut text, format_args!("{}", Malformed::Nul { word: 2 }))
            .unwrap();
        assert_eq!(text, "argument 2 holds a NUL byte");
    }
}
