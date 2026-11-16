//! The POSIX names over [`crate::fd`], [`crate::mmap`] and [`crate::device`]:
//! `open`, `close`, `read`, `write`, `poll`, `mmap` and `munmap`, as
//! `<fcntl.h>`, `<unistd.h>`, `<poll.h>` and `<sys/mman.h>` declare them. Each
//! sets `errno` and answers `-1` (`MAP_FAILED` for `mmap`) when it refuses.
//!
//! Exported on Kore alone, like the rest of the C library: on a host these
//! names are the host libc's.

use core::ffi::{CStr, c_char, c_int, c_short, c_ulong, c_void};

use kore_c_shim::errno::set as set_errno;

use crate::device;
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
/// program's, and mappings made through it stand. A device's descriptor
/// gives the device up as well ([`device`]).
#[unsafe(no_mangle)]
pub extern "C" fn close(fd: c_int) -> c_int {
    let device = matches!(
        TABLE.get(fd),
        Ok((row, _)) if matches!(NAMES[row].kind, Kind::Device { .. })
    );
    match TABLE.release(fd) {
        Ok(()) => {
            if device {
                device::close();
            }
            0
        }
        Err(code) => refused(code),
    }
}

/// The numbers of `stdout` and `stderr`, which no `open` hands out.
const STDOUT: c_int = 1;
const STDERR: c_int = 2;

/// `read(fd, buf, count)`: on a device's descriptor, wait for an interrupt
/// and answer the device's interrupt count, one 32-bit word, as a UIO device
/// does ([`device::read`]).
///
/// `EBADF`: `fd` is not open for reading. `EINVAL`: `count` is not four, or
/// the descriptor names something that is not read (the memory supply).
/// `EFAULT`: `buf` is null. Otherwise as [`device::read`] refuses.
///
/// # Safety
/// `buf` is null or valid for `count` bytes of writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize {
    let answer = TABLE.get(fd).and_then(|(row, access)| {
        if !access.read {
            return Err(errno::EBADF);
        }
        match NAMES[row].kind {
            Kind::Memory => Err(errno::EINVAL),
            Kind::Device { .. } => {
                if count != device::WORD {
                    return Err(errno::EINVAL);
                }
                if buf.is_null() {
                    return Err(errno::EFAULT);
                }
                device::read()
            }
        }
    });
    match answer {
        Ok(interrupts) => {
            // SAFETY: the caller's contract: four writable bytes, of any
            // alignment.
            unsafe { buf.cast::<u32>().write_unaligned(interrupts) };
            device::WORD as isize
        }
        Err(code) => refused(code) as isize,
    }
}

/// `write(fd, buf, count)`: on descriptor 1 or 2, `count` bytes to `stdout`
/// or `stderr`, in order with what `<stdio.h>` writes; on a device's
/// descriptor, one 32-bit word rings the doorbell ([`device::ring`]). The
/// word is not read.
///
/// `EBADF`: `fd` is not open for writing, or the launch carried no stream.
/// `EINVAL`: a device was written anything but four bytes, or the descriptor
/// names something that is not written (the memory supply). `EFAULT`: `buf`
/// is null and `count` is not zero. Otherwise as [`device::ring`] refuses.
///
/// # Safety
/// `buf` is null or valid for `count` bytes of reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn write(fd: c_int, buf: *const c_void, count: usize) -> isize {
    if fd == STDOUT || fd == STDERR {
        if count == 0 {
            return 0;
        }
        if buf.is_null() {
            return refused(errno::EFAULT) as isize;
        }
        // POSIX leaves a count past `SSIZE_MAX` to the implementation: this
        // one writes what the return value can report.
        let count = count.min(isize::MAX as usize);
        // SAFETY: the caller's contract: `count` readable bytes.
        let bytes = unsafe { core::slice::from_raw_parts(buf.cast::<u8>(), count) };
        return if crate::stdio::write_standard(fd == STDERR, bytes) {
            count as isize
        } else {
            refused(errno::EBADF) as isize
        };
    }
    let answer = TABLE.get(fd).and_then(|(row, access)| {
        if !access.write {
            return Err(errno::EBADF);
        }
        match NAMES[row].kind {
            Kind::Memory => Err(errno::EINVAL),
            Kind::Device { .. } => {
                if count != device::WORD {
                    return Err(errno::EINVAL);
                }
                if buf.is_null() {
                    return Err(errno::EFAULT);
                }
                device::ring()
            }
        }
    });
    match answer {
        Ok(()) => device::WORD as isize,
        Err(code) => refused(code) as isize,
    }
}

/// `<poll.h>`'s `struct pollfd`, musl's layout.
#[repr(C)]
pub struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}

/// `<poll.h>`'s events, musl's values.
const POLLIN: c_short = 0x001;
const POLLOUT: c_short = 0x004;
const POLLERR: c_short = 0x008;
const POLLNVAL: c_short = 0x020;

/// What each entry can report now, and whether one of them asks for a
/// device's interrupt and does not have it: the one thing a `poll` here can
/// wait for.
///
/// # Safety
/// `fds` is valid for `nfds` entries.
unsafe fn poll_pass(fds: *mut PollFd, nfds: usize) -> (c_int, bool) {
    let mut ready = 0;
    let mut waits = false;
    for at in 0..nfds {
        // SAFETY: the caller's contract; one entry is borrowed at a time.
        let entry = unsafe { &mut *fds.add(at) };
        entry.revents = 0;
        if entry.fd < 0 {
            // POSIX: a negative descriptor is skipped.
            continue;
        }
        entry.revents = match TABLE.get(entry.fd) {
            // A standard stream takes a write whenever it is asked.
            Err(_) if entry.fd == STDOUT || entry.fd == STDERR => entry.events & POLLOUT,
            Err(_) => POLLNVAL,
            Ok((row, access)) => match NAMES[row].kind {
                // Mapping a supply never waits.
                Kind::Memory => entry.events & (POLLIN | POLLOUT),
                Kind::Device { .. } => {
                    // The doorbell never waits.
                    let mut events = if access.write {
                        entry.events & POLLOUT
                    } else {
                        0
                    };
                    if access.read && entry.events & POLLIN != 0 {
                        match device::readable() {
                            Some(true) => events |= POLLIN,
                            Some(false) => waits = true,
                            None => events |= POLLNVAL,
                        }
                    }
                    events
                }
            },
        };
        if entry.revents != 0 {
            ready += 1;
        }
    }
    (ready, waits)
}

/// `poll(fds, nfds, timeout)`: which descriptors are ready, waiting up to
/// `timeout` milliseconds (forever when negative) for a device's interrupt.
///
/// `POLLIN` on a device's descriptor is an interrupt waiting to be `read`;
/// `POLLOUT` is always ready, there and on the standard output streams; a
/// memory supply's descriptor is always ready; a number that is not open
/// reports `POLLNVAL`. A device that closes during the wait reports
/// `POLLNVAL`, and one this thread cannot wait on `POLLERR`.
///
/// `EFAULT`: `fds` is null and `nfds` is not zero. `EINVAL`: more entries
/// than there are descriptors, or a wait with no timeout that nothing in the
/// set can end. A wait with a timeout and nothing to wait for sleeps it out
/// and answers zero.
///
/// # Safety
/// `fds` is null or valid for `nfds` entries of reads and writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int {
    let Some(nfds) = usize::try_from(nfds)
        .ok()
        .filter(|&nfds| nfds <= fd::CAPACITY + fd::FIRST as usize)
    else {
        return refused(errno::EINVAL);
    };
    if nfds != 0 && fds.is_null() {
        return refused(errno::EFAULT);
    }
    // SAFETY: the caller's contract.
    let (ready, waits) = unsafe { poll_pass(fds, nfds) };
    if ready != 0 || timeout == 0 {
        return ready;
    }
    let deadline = match u64::try_from(timeout) {
        Ok(milliseconds) => {
            // The calibrated counter, or the kernel's own stamp on a target
            // without one: a timeout is honoured wherever a thread can park.
            let Some(now) = kore_std::runtime::clock::deadline_base_nanos() else {
                return refused(errno::EINVAL);
            };
            Some(now.saturating_add(milliseconds.saturating_mul(1_000_000)))
        }
        Err(_) => None,
    };
    if !waits {
        return match deadline {
            // Nothing here can become ready: the timeout is the whole wait.
            Some(at) => {
                // A deadline park of this thread; a thread that cannot park
                // answers at once, which the caller's own loop absorbs.
                let _ = kore_std::runtime::cap::verbs::sleep_until(at);
                0
            }
            None => refused(errno::EINVAL),
        };
    }
    match device::wait_readable(deadline) {
        // An interrupt, or the timeout: either way the entries say which.
        // SAFETY: the caller's contract.
        Ok(_) => unsafe { poll_pass(fds, nfds) }.0,
        Err(code) => {
            // The device closed, or cannot be waited on: say so on its
            // entries, which is what ended the wait.
            let event = if code == errno::EBADF {
                POLLNVAL
            } else {
                POLLERR
            };
            let mut ready = 0;
            for at in 0..nfds {
                // SAFETY: the caller's contract.
                let entry = unsafe { &mut *fds.add(at) };
                let device = matches!(
                    TABLE.get(entry.fd),
                    Ok((row, _)) if matches!(NAMES[row].kind, Kind::Device { .. })
                );
                if device && entry.events & POLLIN != 0 {
                    entry.revents |= event;
                }
                if entry.revents != 0 {
                    ready += 1;
                }
            }
            ready
        }
    }
}

/// `mmap(addr, len, prot, flags, fd, offset)`: of the memory supply
/// ([`mmap`]'s module docs), or of a device, whose offset says which of its
/// maps is meant ([`device::Map`]). `addr` is a hint POSIX lets an
/// implementation ignore, and it is.
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
        let name = &NAMES[row];
        match name.kind {
            Kind::Memory => {
                let request = Request::of(len, prot, flags, offset)?;
                if !access.read || (request.write && !access.write) {
                    return Err(errno::EACCES);
                }
                let held = fd::holding(name).ok_or(errno::ENXIO)?;
                mmap::map(held, request)
            }
            Kind::Device { .. } => {
                let map = device::Map::of(len, prot, flags, offset)?;
                if !access.read || (map.writes(prot) && !access.write) {
                    return Err(errno::EACCES);
                }
                // The calling thread's own share, as for any descriptor.
                fd::holding(name).ok_or(errno::ENXIO)?;
                device::map(map)
            }
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
/// ([`mmap::unmap`], or [`device::unmap`] for a device's).
#[unsafe(no_mangle)]
pub extern "C" fn munmap(addr: *mut c_void, len: usize) -> c_int {
    let unmapped = device::unmap(addr.addr(), len).unwrap_or_else(|| mmap::unmap(addr.addr(), len));
    match unmapped {
        Ok(()) => 0,
        Err(code) => refused(code),
    }
}
