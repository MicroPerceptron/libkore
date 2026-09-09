//! Reference types for a submission's operands — kore names resources by
//! registered index, never by host pointer.

/// A registered file/device: an index into the cap arena's fixed-file
/// table (the cap arena IS the registration table). The kore analog of
/// `io_uring`'s fixed-file
/// index — kore has no POSIX fd to pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fixed(pub u32);

/// A registered buffer: an index into the cap arena's fixed-buffer table.
/// Operations read/write *this* buffer, so no host address rides the ring.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Buf(pub u32);
