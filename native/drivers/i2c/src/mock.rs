//! Test doubles shared by every layer above the seam.
//!
//! Two things live here, and they are deliberately at different levels:
//!
//! - [`I2cTarget`] — a **device** on the bus. A controller mock (the
//!   DesignWare crate's register file) drives one of these, so a protocol
//!   test can exercise the real command framing all the way down.
//! - [`MockBus`] — an [`I2cBus`] that talks to an `I2cTarget` directly,
//!   with no controller at all. A protocol driver's own tests use this to
//!   assert protocol behaviour without also re-testing FIFO mechanics.
//!
//! Behind a feature so no image can link them.

use crate::bus::I2cBus;

/// How a target answered one transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetReply {
    /// Addressed, and `n` bytes were placed in the read buffer.
    Acked(usize),
    /// Nobody answered the address.
    NackAddr,
    /// The address was acked, then the target NACKed the write byte at
    /// index `n` — the case that distinguishes "no device" from "device
    /// rejected this command", which a driver must not conflate.
    NackData(usize),
}

/// A device on the bus.
pub trait I2cTarget {
    /// Handle one write-then-read transaction. `out` is `read_len` long.
    fn transact(&mut self, addr: u8, write: &[u8], out: &mut [u8]) -> TargetReply;
}

/// How a [`MockBus`] transaction failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MockError {
    /// Nobody answered the address.
    NackAddr,
    /// The target rejected the write byte at this index.
    NackData(usize),
}

/// An [`I2cBus`] straight onto an [`I2cTarget`].
pub struct MockBus<T> {
    target: T,
    /// Every transaction, in order: `(addr, write bytes, read length)`.
    pub log: alloc::vec::Vec<(u8, alloc::vec::Vec<u8>, usize)>,
}

impl<T: I2cTarget> MockBus<T> {
    /// Wrap a target.
    pub fn new(target: T) -> Self {
        Self {
            target,
            log: alloc::vec::Vec::new(),
        }
    }

    /// The target, for assertions about what it saw.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// The target, mutably, for staging the next reply.
    pub fn target_mut(&mut self) -> &mut T {
        &mut self.target
    }
}

impl<T: I2cTarget> I2cBus for MockBus<T> {
    type Error = MockError;

    fn write(&mut self, addr: u8, bytes: &[u8]) -> Result<(), Self::Error> {
        self.write_read(addr, bytes, &mut []).map(|_| ())
    }

    fn write_read(
        &mut self,
        addr: u8,
        bytes: &[u8],
        read: &mut [u8],
    ) -> Result<usize, Self::Error> {
        self.log.push((addr, bytes.to_vec(), read.len()));
        match self.target.transact(addr, bytes, read) {
            TargetReply::Acked(n) => Ok(n),
            TargetReply::NackAddr => Err(MockError::NackAddr),
            TargetReply::NackData(i) => Err(MockError::NackData(i)),
        }
    }
}
