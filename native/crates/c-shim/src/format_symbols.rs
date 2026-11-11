//! The `<stdio.h>` writes: the `printf` family over [`crate::format`], and
//! the character and block writes over [`crate::stdio`]'s streams.
//!
//! `snprintf` returns what ISO says — the length the complete output would
//! have had — writing at most `n - 1` bytes and a NUL. A count past `INT_MAX`
//! returns `-1`, C's `EOVERFLOW` case. The stream writes return `EOF` when
//! their stream's write fails, which on `stdout`/`stderr` means no
//! [`StandardSink`](crate::stdio::StandardSink) was installed.

use core::ffi::{VaList, c_char, c_int, c_long, c_longlong, c_uint, c_ulong, c_ulonglong, c_void};

use crate::format::{Args, Length, Sink, format};
use crate::stdio::{self, File, stdout};

const EOF: c_int = -1;

/// A C `va_list` as the engine's argument source.
struct VaArgs<'a, 'f>(&'a mut VaList<'f>);

impl Args for VaArgs<'_, '_> {
    fn signed(&mut self, length: Length) -> i64 {
        // SAFETY (each `next_arg`): the format's conversion names this C
        // type, the variadic caller's half of the contract.
        unsafe {
            match length {
                // `char` and `short` arrive promoted to `int`.
                Length::Char => i64::from(self.0.next_arg::<c_int>() as i8),
                Length::Short => i64::from(self.0.next_arg::<c_int>() as i16),
                Length::Int => i64::from(self.0.next_arg::<c_int>()),
                Length::Long => self.0.next_arg::<c_long>(),
                // glibc reads `%Ld` as `long long`; so does this.
                Length::LongLong | Length::Max | Length::LongDouble => {
                    self.0.next_arg::<c_longlong>()
                }
                Length::Size | Length::Diff => self.0.next_arg::<isize>() as i64,
            }
        }
    }

    fn unsigned(&mut self, length: Length) -> u64 {
        // SAFETY: as `signed`.
        unsafe {
            match length {
                Length::Char => u64::from(self.0.next_arg::<c_uint>() as u8),
                Length::Short => u64::from(self.0.next_arg::<c_uint>() as u16),
                Length::Int => u64::from(self.0.next_arg::<c_uint>()),
                Length::Long => self.0.next_arg::<c_ulong>(),
                Length::LongLong | Length::Max | Length::LongDouble => {
                    self.0.next_arg::<c_ulonglong>()
                }
                Length::Size | Length::Diff => self.0.next_arg::<usize>() as u64,
            }
        }
    }

    fn int(&mut self) -> c_int {
        // SAFETY: as `signed`.
        unsafe { self.0.next_arg::<c_int>() }
    }

    fn pointer(&mut self) -> usize {
        // SAFETY: as `signed`.
        unsafe { self.0.next_arg::<*const c_void>() as usize }
    }

    fn double(&mut self) -> f64 {
        // SAFETY: as `signed`.
        unsafe { self.0.next_arg::<f64>() }
    }

    fn long_double(&mut self) -> crate::format::Float {
        // SAFETY: as `signed`; the conversion's `L` names a `long double`.
        unsafe { crate::va_long_double::next(self.0) }
    }
}

/// A C buffer that keeps counting past its end, as `snprintf` must.
struct Bounded {
    target: *mut u8,
    /// Bytes that fit before the terminator.
    room: usize,
    stored: usize,
}

impl Sink for Bounded {
    fn put(&mut self, bytes: &[u8]) {
        let take = bytes.len().min(self.room - self.stored);
        // SAFETY: `stored + take <= room`, inside the caller's buffer.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.target.add(self.stored), take)
        };
        self.stored += take;
    }
}

/// `vsnprintf` with an explicit room: `None` is `vsprintf`'s unbounded
/// buffer, whose size the C caller vouches for.
///
/// # Safety
/// C's `vsnprintf` contract.
unsafe fn bounded(
    target: *mut c_char,
    len: Option<usize>,
    format_: *const c_char,
    args: &mut VaList,
) -> c_int {
    let room = match len {
        Some(0) => 0,
        Some(len) => len - 1,
        None => usize::MAX,
    };
    let mut out = Bounded {
        target: target.cast(),
        room,
        stored: 0,
    };
    // SAFETY: the caller's contract.
    let count = unsafe { format(&mut out, format_.cast(), &mut VaArgs(args)) };
    if len != Some(0) {
        // SAFETY: `stored <= room`, which left one byte for the NUL.
        unsafe { *out.target.add(out.stored) = 0 };
    }
    c_int::try_from(count).unwrap_or(-1)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn vsnprintf(
    target: *mut c_char,
    len: usize,
    format: *const c_char,
    mut args: VaList,
) -> c_int {
    // SAFETY: the caller's contract is vsnprintf's.
    unsafe { bounded(target, Some(len), format, &mut args) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn snprintf(
    target: *mut c_char,
    len: usize,
    format: *const c_char,
    args: ...
) -> c_int {
    // SAFETY: the caller's contract is vsnprintf's.
    unsafe { vsnprintf(target, len, format, args) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn vsprintf(
    target: *mut c_char,
    format: *const c_char,
    mut args: VaList,
) -> c_int {
    // SAFETY: the caller vouches that the buffer holds the output.
    unsafe { bounded(target, None, format, &mut args) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn sprintf(target: *mut c_char, format: *const c_char, args: ...) -> c_int {
    // SAFETY: as `vsprintf`.
    unsafe { vsprintf(target, format, args) }
}

/// A stream write, batched: the engine's runs collect here and reach the
/// stream a buffer at a time.
struct Streamed {
    file: *mut File,
    buffer: [u8; 256],
    held: usize,
    failed: bool,
}

impl Streamed {
    fn drain(&mut self) {
        // SAFETY: the stream is live for the whole call.
        if self.held > 0 && !unsafe { stdio::write(self.file, &self.buffer[..self.held]) } {
            self.failed = true;
        }
        self.held = 0;
    }
}

impl Sink for Streamed {
    fn put(&mut self, bytes: &[u8]) {
        if self.held + bytes.len() > self.buffer.len() {
            self.drain();
        }
        if bytes.len() > self.buffer.len() {
            // A run bigger than the buffer goes straight through.
            // SAFETY: as `drain`.
            if !unsafe { stdio::write(self.file, bytes) } {
                self.failed = true;
            }
            return;
        }
        self.buffer[self.held..self.held + bytes.len()].copy_from_slice(bytes);
        self.held += bytes.len();
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn vfprintf(file: *mut File, format_: *const c_char, mut args: VaList) -> c_int {
    // One call, one lock: every batch reaches the stream before another
    // thread's call does.
    // SAFETY: C's vfprintf contract: a live stream and a matching format.
    unsafe {
        stdio::locked(file, || {
            let mut out = Streamed {
                file,
                buffer: [0; 256],
                held: 0,
                failed: false,
            };
            let count = format(&mut out, format_.cast(), &mut VaArgs(&mut args));
            out.drain();
            if out.failed {
                return EOF;
            }
            c_int::try_from(count).unwrap_or(-1)
        })
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn fprintf(file: *mut File, format: *const c_char, args: ...) -> c_int {
    // SAFETY: as `vfprintf`.
    unsafe { vfprintf(file, format, args) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn vprintf(format: *const c_char, args: VaList) -> c_int {
    // SAFETY: as `vfprintf`, on `stdout`.
    unsafe { vfprintf(stdout.file(), format, args) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn printf(format: *const c_char, args: ...) -> c_int {
    // SAFETY: as `vfprintf`, on `stdout`.
    unsafe { vfprintf(stdout.file(), format, args) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn fputc(byte: c_int, file: *mut File) -> c_int {
    let byte = byte as u8;
    // SAFETY: C's contract: a live stream.
    if unsafe { stdio::write(file, &[byte]) } {
        c_int::from(byte)
    } else {
        EOF
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn putc(byte: c_int, file: *mut File) -> c_int {
    // SAFETY: as `fputc`.
    unsafe { fputc(byte, file) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn putchar(byte: c_int) -> c_int {
    // SAFETY: `stdout` is live.
    unsafe { fputc(byte, stdout.file()) }
}

/// The bytes of a C string, without its NUL.
///
/// # Safety
/// `text` is NUL-terminated.
unsafe fn c_str<'a>(text: *const c_char) -> &'a [u8] {
    // SAFETY: the caller's contract.
    unsafe { core::ffi::CStr::from_ptr(text) }.to_bytes()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn fputs(text: *const c_char, file: *mut File) -> c_int {
    // SAFETY: C's contract: a NUL-terminated string and a live stream.
    if unsafe { stdio::write(file, c_str(text)) } {
        0
    } else {
        EOF
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn puts(text: *const c_char) -> c_int {
    let file = stdout.file();
    // The line and its newline under one lock, as one call.
    // SAFETY: as `fputs`, on `stdout`.
    let written = unsafe {
        stdio::locked(file, || {
            stdio::write(file, c_str(text)) && stdio::write(file, b"\n")
        })
    };
    if written { 0 } else { EOF }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn fwrite(
    source: *const c_void,
    size: usize,
    count: usize,
    file: *mut File,
) -> usize {
    let Some(total) = size.checked_mul(count) else {
        return 0;
    };
    if total == 0 {
        return 0;
    }
    // SAFETY: C promises `size * count` readable bytes and a live stream.
    let bytes = unsafe { core::slice::from_raw_parts(source.cast::<u8>(), total) };
    if unsafe { stdio::write(file, bytes) } {
        count
    } else {
        0
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn fflush(file: *mut File) -> c_int {
    if file.is_null() {
        // "Every output stream": the standard pair holds nothing back, and a
        // consumer's streams flush on their own close.
        return 0;
    }
    // SAFETY: C's contract: a live stream.
    if unsafe { stdio::flush(file) } {
        0
    } else {
        EOF
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn fclose(file: *mut File) -> c_int {
    // SAFETY: C's contract: a live stream, not used again.
    if unsafe { stdio::close(file) } {
        0
    } else {
        EOF
    }
}
