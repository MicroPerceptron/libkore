//! Blocking IPv4 stream and datagram entries in the C program's descriptor table.
//! The program host owns each flow and serializes requests from `main`.

use core::ffi::{c_int, c_void};
use core::mem::size_of;

use kore_c_shim::errno::set as set_errno;
use kore_std::driver::net::{IpProto, Ipv4Addr};

use crate::fd::{TABLE, errno};

const AF_INET: c_int = 2;
const SOCK_STREAM: c_int = 1;
const SOCK_DGRAM: c_int = 2;
const IPPROTO_TCP: c_int = 6;
const IPPROTO_UDP: c_int = 17;

pub(crate) type Endpoint = (Ipv4Addr, u16);

/// The populated fields of `sockaddr_in`. Callers may leave `sin_zero`
/// uninitialized.
#[repr(C)]
struct SockaddrInPrefix {
    family: u16,
    port: [u8; 2],
    address: [u8; 4],
}

const SOCKADDR_IN_BYTES: u32 = 16;
const _: () = assert!(size_of::<SockaddrInPrefix>() == 8);

#[derive(Clone, Copy)]
pub(crate) enum Request {
    Open(IpProto),
    Connect(c_int, Endpoint),
    Close(c_int),
    Send(c_int, *const u8, usize),
    Recv(c_int, *mut u8, usize),
}

fn result(value: Result<usize, c_int>) -> isize {
    match value {
        Ok(n) => n as isize,
        Err(code) => {
            set_errno(code);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn socket(domain: c_int, kind: c_int, protocol: c_int) -> c_int {
    let protocol = match (domain, kind, protocol) {
        (AF_INET, SOCK_STREAM, 0 | IPPROTO_TCP) => Ok(IpProto::TCP),
        (AF_INET, SOCK_DGRAM, 0 | IPPROTO_UDP) => Ok(IpProto::UDP),
        (AF_INET, SOCK_STREAM | SOCK_DGRAM, _) => Err(errno::EPROTONOSUPPORT),
        (AF_INET, _, _) => Err(errno::ENOTSUP),
        _ => Err(errno::EAFNOSUPPORT),
    };
    result(protocol.and_then(|protocol| {
        // SAFETY: this request carries no borrowed buffers.
        unsafe { crate::program::socket(Request::Open(protocol)) }
    })) as c_int
}

/// # Safety
/// `address` is null or points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn connect(fd: c_int, address: *const c_void, len: u32) -> c_int {
    let connected = TABLE.require_socket(fd).and_then(|()| {
        if address.is_null() {
            return Err(errno::EFAULT);
        }
        if len < SOCKADDR_IN_BYTES {
            return Err(errno::EINVAL);
        }
        // SAFETY: the caller supplies the object bytes. Its alignment is
        // immaterial and all field bit patterns are valid.
        let address = unsafe { address.cast::<SockaddrInPrefix>().read_unaligned() };
        if c_int::from(address.family) != AF_INET {
            return Err(errno::EAFNOSUPPORT);
        }
        let peer = (Ipv4Addr(address.address), u16::from_be_bytes(address.port));
        if peer.1 == 0 || peer.0.is_unspecified() || peer.0.is_multicast() || peer.0.is_broadcast()
        {
            return Err(errno::EINVAL);
        }
        // SAFETY: the endpoint is passed by value.
        unsafe { crate::program::socket(Request::Connect(fd, peer)) }
    });
    result(connected) as c_int
}

pub(crate) fn close(fd: c_int) -> c_int {
    // SAFETY: this request carries no borrowed buffers.
    result(unsafe { crate::program::socket(Request::Close(fd)) }) as c_int
}

fn transfer(request: Request, fd: c_int, null: bool, len: usize, flags: c_int) -> isize {
    result(TABLE.require_socket(fd).and_then(|()| {
        if flags != 0 {
            return Err(errno::ENOTSUP);
        }
        if len > isize::MAX as usize {
            return Err(errno::EINVAL);
        }
        if null && len != 0 {
            return Err(errno::EFAULT);
        }
        // SAFETY: the caller of send/recv keeps its buffer valid over this
        // synchronous call. An empty buffer is handled on the host side.
        unsafe { crate::program::socket(request) }
    }))
}

/// # Safety
/// A nonempty buffer has `len` readable bytes for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn send(fd: c_int, buf: *const c_void, len: usize, flags: c_int) -> isize {
    transfer(
        Request::Send(fd, buf.cast(), len),
        fd,
        buf.is_null(),
        len,
        flags,
    )
}

/// # Safety
/// A nonempty buffer has `len` exclusively writable bytes for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn recv(fd: c_int, buf: *mut c_void, len: usize, flags: c_int) -> isize {
    transfer(
        Request::Recv(fd, buf.cast(), len),
        fd,
        buf.is_null(),
        len,
        flags,
    )
}
