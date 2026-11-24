//! What a transfer **is**, and the narrow trait a protocol driver holds.
//!
//! # Why a write-then-read pair, and not a message list
//!
//! The general I2C abstraction is a list of messages with per-message
//! direction and restart flags. This seam models exactly one shape —
//! *optional write, optional restart, optional read, one stop* — because
//! that is the entire vocabulary HID-over-I2C speaks (fetch a descriptor,
//! send a command, drain a report) and it is what the DesignWare command
//! stream encodes natively.
//!
//! Modelling more would mean either a heap-allocated list (this crate is
//! heapless) or a fixed-capacity array whose cap is a lie the first time
//! something needs one more. Modelling exactly this means every valid
//! [`Xfer`] is representable and every representable `Xfer` is valid — no
//! unreachable arms, no cap to outgrow. A device that genuinely needs
//! chained restarts gets a widened `Xfer` and one new impl arm, at the
//! point where a real caller demands it.

use crate::error::RequestError;

/// The largest single transfer the seam admits, in bytes (write + read).
///
/// Not a hardware limit — a *framing* one. HID-over-I2C's own maximum
/// input length is a 16-bit field, but this seam queues a transfer as one
/// command sequence and pumps it against a FIFO; an unbounded transfer
/// would mean an unbounded time with the bus held, on a path whose entire
/// discipline is bounded waits. 4 KiB is far above the largest report
/// descriptor seen in practice and far below anything that could hold the
/// bus long enough to matter.
pub const MAX_XFER_BYTES: usize = 4096;

/// The highest lawful 7-bit target address.
pub const MAX_ADDR: u8 = 0x7F;

/// One I2C transaction: write `write`, then — if `read_len` is non-zero —
/// a repeated START and `read_len` bytes read, then one STOP.
///
/// The restart is implied rather than a flag, because a write-then-read
/// with a STOP between the phases is a different transaction that loses
/// the bus in the middle, and no consumer of this seam wants it. Making it
/// unrepresentable is cheaper than documenting not to ask for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xfer<'a> {
    /// The 7-bit target address.
    pub addr: u8,
    /// Bytes to write. Empty for a pure read.
    pub write: &'a [u8],
    /// Bytes to read after the repeated START. Zero for a pure write.
    pub read_len: usize,
}

impl<'a> Xfer<'a> {
    /// A pure write.
    pub const fn write(addr: u8, bytes: &'a [u8]) -> Self {
        Self {
            addr,
            write: bytes,
            read_len: 0,
        }
    }

    /// A pure read.
    pub const fn read(addr: u8, len: usize) -> Self {
        Self {
            addr,
            write: &[],
            read_len: len,
        }
    }

    /// Write, repeated START, read — the register-fetch shape.
    pub const fn write_read(addr: u8, bytes: &'a [u8], len: usize) -> Self {
        Self {
            addr,
            write: bytes,
            read_len: len,
        }
    }

    /// Total bytes this transfer moves in both directions.
    pub const fn len(&self) -> usize {
        self.write.len() + self.read_len
    }

    /// Whether the transfer moves no bytes at all — always a caller bug,
    /// never a lawful "address probe" (an address probe is a zero-length
    /// *write*, which this seam does not model because no consumer needs
    /// it and admitting it would make the empty case ambiguous).
    pub const fn is_empty(&self) -> bool {
        self.write.is_empty() && self.read_len == 0
    }

    /// Validate the request before any register is touched.
    ///
    /// Called by the engine on every transfer rather than trusted from
    /// the constructors, because `Xfer`'s fields are public: a caller may
    /// build one by struct literal and skip them entirely. A validation
    /// that only the constructors perform is one a struct literal walks
    /// straight past.
    pub fn validate(&self, read_buf: usize) -> Result<(), RequestError> {
        if self.addr > MAX_ADDR {
            return Err(RequestError::AddressTooWide(self.addr as u16));
        }
        if self.is_empty() {
            return Err(RequestError::Empty);
        }
        if self.len() > MAX_XFER_BYTES {
            return Err(RequestError::TooLong { bytes: self.len() });
        }
        if read_buf < self.read_len {
            return Err(RequestError::ReadBufferTooSmall {
                need: self.read_len,
                have: read_buf,
            });
        }
        Ok(())
    }
}

/// The **narrow bus trait a protocol driver holds** — the whole of what
/// `kore-i2c-hid` knows about I2C.
///
/// Two methods, no controller, no registers, no pacing. A protocol driver
/// written against this cannot accidentally depend on DesignWare, which is
/// what makes the same HID implementation serve Intel, AMD, and Qualcomm
/// unchanged.
///
/// `embedded-hal`'s `I2c` was considered and declined: adopting it would
/// pull a trait hierarchy the tree has no other consumer for, and put a
/// foreign error type between a NACK and the witness line that has to
/// carry it (#405).
pub trait I2cBus {
    /// How this bus reports failure. Left associated so the DesignWare
    /// engine can surface its full abort detail while a test double
    /// surfaces something trivial.
    type Error: core::fmt::Debug;

    /// Write `bytes` to `addr`.
    fn write(&mut self, addr: u8, bytes: &[u8]) -> Result<(), Self::Error>;

    /// Write `bytes` to `addr`, repeated START, then read into `read`.
    /// Returns the number of bytes read.
    fn write_read(&mut self, addr: u8, bytes: &[u8], read: &mut [u8])
    -> Result<usize, Self::Error>;

    /// Read into `read` with no preceding write.
    ///
    /// Defaulted onto [`write_read`](Self::write_read) with an empty write
    /// because every controller that can do one can do the other, and a
    /// separate required method would be a second thing each impl could
    /// get subtly wrong.
    fn read(&mut self, addr: u8, read: &mut [u8]) -> Result<usize, Self::Error> {
        self.write_read(addr, &[], read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_refuses_every_malformed_request() {
        let mut buf = [0u8; 8];
        assert_eq!(
            Xfer::write(0x80, &[1]).validate(buf.len()),
            Err(RequestError::AddressTooWide(0x80))
        );
        assert_eq!(Xfer::write(0x2c, &[]).validate(0), Err(RequestError::Empty));
        assert_eq!(
            Xfer::read(0x2c, 9).validate(buf.len()),
            Err(RequestError::ReadBufferTooSmall { need: 9, have: 8 })
        );
        let big = [0u8; 8];
        assert_eq!(
            Xfer::write_read(0x2c, &big, MAX_XFER_BYTES).validate(usize::MAX),
            Err(RequestError::TooLong {
                bytes: MAX_XFER_BYTES + 8
            })
        );
        // The boundary itself is lawful, not off by one.
        assert_eq!(
            Xfer::read(MAX_ADDR, MAX_XFER_BYTES).validate(usize::MAX),
            Ok(())
        );
        buf[0] = 0;
        assert_eq!(
            Xfer::write_read(0x2c, &[1, 2], 8).validate(buf.len()),
            Ok(())
        );
    }

    #[test]
    fn the_three_constructors_agree_with_the_fields() {
        assert_eq!(Xfer::write(0x2c, &[9]).read_len, 0);
        assert_eq!(Xfer::read(0x2c, 4).write, &[] as &[u8]);
        let wr = Xfer::write_read(0x2c, &[1, 2], 4);
        assert_eq!((wr.len(), wr.is_empty()), (6, false));
        assert!(Xfer::write(0x2c, &[]).is_empty());
    }
}
