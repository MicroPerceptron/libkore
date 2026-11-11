//! `errno`: one per thread, as C requires, and the codes this crate and its
//! consumers set.
//!
//! The word is the calling thread's runtime-block slot
//! ([`kore_rt::thread::c_errno`]), not an ELF thread-local: a TLS segment in
//! every image that carries C would buy each of them a TCB migration, a
//! template every spawn must place, and `kore-late`'s refusal of a host with
//! TLS — all for one `int`.
//!
//! The numbers are musl's `<errno.h>` — the headers every kore C build
//! compiles against (`zig cc -target <arch>-linux-musl`), which use the
//! same values on x86_64, aarch64 and riscv64. A consumer's C code compares
//! `errno` against those headers' macros, so these must agree with them.

use core::ffi::{CStr, c_int};

pub const EPERM: c_int = 1;
pub const ENOENT: c_int = 2;
pub const EIO: c_int = 5;
pub const EBADF: c_int = 9;
pub const EAGAIN: c_int = 11;
pub const ENOMEM: c_int = 12;
pub const EACCES: c_int = 13;
pub const EBUSY: c_int = 16;
pub const EEXIST: c_int = 17;
pub const ENOTDIR: c_int = 20;
pub const EISDIR: c_int = 21;
pub const EINVAL: c_int = 22;
pub const ENFILE: c_int = 23;
pub const EMFILE: c_int = 24;
pub const EFBIG: c_int = 27;
pub const ENOSPC: c_int = 28;
pub const ESPIPE: c_int = 29;
pub const EROFS: c_int = 30;
pub const ERANGE: c_int = 34;
pub const ENAMETOOLONG: c_int = 36;
pub const ENOSYS: c_int = 38;
pub const ENOTEMPTY: c_int = 39;
pub const ELOOP: c_int = 40;
pub const EOVERFLOW: c_int = 75;

/// Set this thread's `errno`.
pub fn set(code: c_int) {
    // SAFETY: the calling thread's own word, live for its life and touched
    // by no other thread (`kore_rt::thread::c_errno`'s contract).
    unsafe { kore_rt::thread::c_errno().write(code) };
}

/// This thread's `errno`.
pub fn get() -> c_int {
    // SAFETY: as `set`.
    unsafe { kore_rt::thread::c_errno().read() }
}

#[unsafe(no_mangle)]
extern "C" fn __errno_location() -> *mut c_int {
    kore_rt::thread::c_errno()
}

/// The message `strerror` answers for `code`.
pub fn describe(code: c_int) -> &'static CStr {
    match code {
        0 => c"No error information",
        EPERM => c"Operation not permitted",
        ENOENT => c"No such file or directory",
        EIO => c"I/O error",
        EBADF => c"Bad file descriptor",
        EAGAIN => c"Resource temporarily unavailable",
        ENOMEM => c"Out of memory",
        EACCES => c"Permission denied",
        EBUSY => c"Resource busy",
        EEXIST => c"File exists",
        ENOTDIR => c"Not a directory",
        EISDIR => c"Is a directory",
        EINVAL => c"Invalid argument",
        ENFILE => c"Too many open files in system",
        EMFILE => c"No file descriptors available",
        EFBIG => c"File too large",
        ENOSPC => c"No space left on device",
        ESPIPE => c"Invalid seek",
        EROFS => c"Read-only file system",
        ERANGE => c"Result not representable",
        ENAMETOOLONG => c"Filename too long",
        ENOSYS => c"Function not implemented",
        ENOTEMPTY => c"Directory not empty",
        ELOOP => c"Symbolic link loop",
        EOVERFLOW => c"Value too large for data type",
        _ => c"Unknown error",
    }
}
