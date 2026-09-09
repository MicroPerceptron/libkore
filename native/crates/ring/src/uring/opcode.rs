//! Typed operation builders — `opcode::Read::new(...).build().user_data(x)`,
//! the io-uring idiom. Buffers are registered ([`types::Buf`]); there are no
//! raw-pointer variants by design (I5).

use super::{Entry, types};
use kore_abi::uring::{IoSqeFlags, IoUringSqe, op};

fn sqe(opcode: u8, flags: u8, fd: i32, buf_index: u16, len: u32, off: u64) -> IoUringSqe {
    IoUringSqe {
        opcode,
        flags,
        ioprio: 0,
        fd,
        off,
        addr: 0,
        len,
        op_flags: 0,
        user_data: 0,
        buf_index,
        personality: 0,
        splice_fd_in: 0,
        addr3: 0,
        addr3_pad: 0,
    }
}

/// A no-op — completes immediately. The porting/liveness smoke test.
#[derive(Clone, Copy, Debug, Default)]
pub struct Nop;

impl Nop {
    pub fn new() -> Self {
        Nop
    }
    pub fn build(self) -> Entry {
        Entry(sqe(op::NOP, 0, 0, 0, 0, 0))
    }
}

/// Read `len` bytes from registered file `file` (at file offset
/// [`offset`](Read::offset)) into registered buffer `buf` — the
/// `READ_FIXED` shape (fixed file + fixed buffer).
#[derive(Clone, Copy, Debug)]
pub struct Read {
    file: types::Fixed,
    buf: types::Buf,
    len: u32,
    offset: u64,
}

impl Read {
    pub fn new(file: types::Fixed, buf: types::Buf, len: u32) -> Self {
        Self {
            file,
            buf,
            len,
            offset: 0,
        }
    }
    /// File offset to read from (default 0).
    #[must_use]
    pub fn offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }
    pub fn build(self) -> Entry {
        Entry(sqe(
            op::READ_FIXED,
            IoSqeFlags::FIXED_FILE.bits(),
            self.file.0 as i32,
            self.buf.0 as u16,
            self.len,
            self.offset,
        ))
    }
}

/// Write `len` bytes from registered buffer `buf` to registered file
/// `file` (at file offset [`offset`](Write::offset)) — the `WRITE_FIXED`
/// shape.
#[derive(Clone, Copy, Debug)]
pub struct Write {
    file: types::Fixed,
    buf: types::Buf,
    len: u32,
    offset: u64,
}

impl Write {
    pub fn new(file: types::Fixed, buf: types::Buf, len: u32) -> Self {
        Self {
            file,
            buf,
            len,
            offset: 0,
        }
    }
    #[must_use]
    pub fn offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }
    pub fn build(self) -> Entry {
        Entry(sqe(
            op::WRITE_FIXED,
            IoSqeFlags::FIXED_FILE.bits(),
            self.file.0 as i32,
            self.buf.0 as u16,
            self.len,
            self.offset,
        ))
    }
}

/// Copy `len` bytes from registered buffer `src` (at [`src_offset`]) to
/// registered buffer `dst` (at [`dst_offset`]) — the device-to-device
/// transfer primitive over two shared windows ([`op::PEER_COPY`], a kore
/// extension for the `MmioRegion` peer leaf: an NVMe controller-memory-buffer,
/// an accelerator aperture).
///
/// Both operands are **registered buffers** (indices, never host addresses —
/// I5) and no capability rides the ring: the windows are lease-arranged and
/// named here by index. The wire field reuse is pinned by [`op::PEER_COPY`]'s
/// contract: `buf_index` = source, `splice_fd_in` = destination, `off` =
/// source offset, `addr` = destination offset. Registration already proved the
/// asymmetric access: the source is READ-admitted with completion routing;
/// the destination admits WRITE (possibly through the posted-write-only peer
/// proof). The SQE cannot weaken or manufacture either proof.
///
/// [`src_offset`]: PeerCopy::src_offset
/// [`dst_offset`]: PeerCopy::dst_offset
#[derive(Clone, Copy, Debug)]
pub struct PeerCopy {
    src: types::Buf,
    dst: types::Buf,
    len: u32,
    src_offset: u64,
    dst_offset: u64,
}

impl PeerCopy {
    pub fn new(src: types::Buf, dst: types::Buf, len: u32) -> Self {
        Self {
            src,
            dst,
            len,
            src_offset: 0,
            dst_offset: 0,
        }
    }
    /// Byte offset within the source buffer (default 0).
    #[must_use]
    pub fn src_offset(mut self, offset: u64) -> Self {
        self.src_offset = offset;
        self
    }
    /// Byte offset within the destination buffer (default 0).
    #[must_use]
    pub fn dst_offset(mut self, offset: u64) -> Self {
        self.dst_offset = offset;
        self
    }
    pub fn build(self) -> Entry {
        // Direct construction: the shared `sqe()` helper zeroes `addr` and
        // `splice_fd_in`, which this op needs for the destination operand.
        // The field reuse matches `op::PEER_COPY`'s pinned contract exactly.
        Entry(IoUringSqe {
            opcode: op::PEER_COPY,
            flags: 0,
            ioprio: 0,
            fd: 0,
            off: self.src_offset,
            addr: self.dst_offset,
            len: self.len,
            op_flags: 0,
            user_data: 0,
            buf_index: u16::try_from(self.src.0).expect("source buffer index must fit in u16"),
            personality: 0,
            splice_fd_in: i32::try_from(self.dst.0)
                .expect("destination buffer index must fit in i32"),
            addr3: 0,
            addr3_pad: 0,
        })
    }
}
