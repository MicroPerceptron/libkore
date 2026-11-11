//! The POSIX names over [`crate::fd`] and [`crate::mmap`]: `open`, `close`,
//! `mmap` and `munmap`, as `<fcntl.h>`, `<unistd.h>` and `<sys/mman.h>`
//! declare them. Each sets `errno` and answers `-1` (`MAP_FAILED` for `mmap`)
//! when it refuses.
//!
//! Exported on Kore alone, like the rest of the C library: on a host these
//! names are the host libc's.

use core::ffi::{CStr, c_char, c_int, c_void};

use kore_c_shim::errno::set as set_errno;

use crate::fd::{self, Kind, NAMES, TABLE, errno};
use crate::mmap::{self, Request};

/// `mmap`'s refusal: `MAP_FAILED`, `(void *)-1`.
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

fn refused(code: c_int) -> c_int {
    set_errno(code);
    -1
}

/// `open(path, flags)`: a descriptor for the capability `path` names
/// ([`fd::open`]). The mode a variadic `open` may carry belongs to `O_CREAT`,
/// which no name here takes, so it is never read.
///
/// # Safety
/// `path` is null or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn open(path: *const c_char, flags: c_int, _mode: ...) -> c_int {
    if path.is_null() {
        return refused(errno::EFAULT);
    }
    // SAFETY: the caller's contract: a NUL-terminated string.
    let path = unsafe { CStr::from_ptr(path) }.to_bytes();
    fd::open(path, flags).unwrap_or_else(refused)
}

/// `close(fd)`: forget the descriptor. The capability it named stays the
/// program's, and mappings made through it stand.
#[unsafe(no_mangle)]
pub extern "C" fn close(fd: c_int) -> c_int {
    match TABLE.release(fd) {
        Ok(()) => 0,
        Err(code) => refused(code),
    }
}

/// `mmap(addr, len, prot, flags, fd, offset)` ([`mmap`]'s module docs).
/// `addr` is a hint POSIX lets an implementation ignore, and it is.
///
/// `EBADF`: `fd` is not open. `EACCES`: it was not opened for reading, or a
/// writable mapping was asked of one not opened for writing. `ENXIO`: the
/// calling thread does not hold the capability. Otherwise as
/// [`Request::of`] and [`mmap::map`] refuse.
#[unsafe(no_mangle)]
pub extern "C" fn mmap(
    _addr: *mut c_void,
    len: usize,
    prot: c_int,
    flags: c_int,
    fd: c_int,
    offset: i64,
) -> *mut c_void {
    let mapped = TABLE.get(fd).and_then(|(row, access)| {
        let request = Request::of(len, prot, flags, offset)?;
        if !access.read || (request.write && !access.write) {
            return Err(errno::EACCES);
        }
        let name = &NAMES[row];
        let held = fd::holding(name).ok_or(errno::ENXIO)?;
        match name.kind {
            Kind::Memory => mmap::map(held, request),
        }
    });
    match mapped {
        Ok(base) => base.cast(),
        Err(code) => {
            set_errno(code);
            MAP_FAILED
        }
    }
}

/// `munmap(addr, len)`: unmap the program's mapping at `addr`, whole
/// ([`mmap::unmap`]).
#[unsafe(no_mangle)]
pub extern "C" fn munmap(addr: *mut c_void, len: usize) -> c_int {
    match mmap::unmap(addr.addr(), len) {
        Ok(()) => 0,
        Err(code) => refused(code),
    }
}
