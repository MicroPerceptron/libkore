//! **The device model's contract**: what a device with no hardware is, to
//! the transport that carries a driver to it.
//!
//! A model answers commands. A driver sends one command as one chain of
//! descriptors, the bytes it gives the device to read followed by the bytes
//! it gives the device to write, and the model reads a request from the
//! first and writes a response into the second. That is the whole exchange,
//! and it is all a model sees of a driver: it is handed lengths and byte
//! ports, never a device address, a descriptor or a ring.
//!
//! Everything else a driver meets is the transport's
//! ([`Transport`](crate::transport::Transport)): the register block, the
//! status it steps through, the queue, the doorbell and the interrupt. A
//! model is written once and knows none of it.

pub use virtio_accel_transport::{ChainRegion, ReadableBytes, WritableBytes};

/// A reset could not return the model to its first state. The transport
/// stops serving for good, and the host drops the instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unrecoverable;

/// One device model.
pub trait VirtualDevice {
    /// What the driver reads from [`DEVICE_ID`](crate::regs::DEVICE_ID) to
    /// tell which model it has been given.
    const DEVICE_ID: u32;

    /// The largest queue the model takes: a power of two, and at most
    /// [`MAX_QUEUE_SIZE`](crate::ring::MAX_QUEUE_SIZE).
    const MAX_QUEUE_SIZE: u16;

    /// The longest chain one command may be, in descriptors: at least two
    /// (a request and a response) and at most the queue's size.
    const MAX_CHAIN_DESCRIPTORS: u16;

    /// The device configuration a driver reads at
    /// [`CONFIG`](crate::regs::CONFIG) before its first command. Fixed for
    /// the life of the model.
    fn config(&self) -> &[u8];

    /// One command. `regions` is the chain's layout, `request` its readable
    /// bytes and `response` its writable bytes, each as one run however the
    /// driver split it.
    ///
    /// Returns the number of response bytes written, from the start of
    /// `response`: what the driver is told the device used. Zero says the
    /// chain could not carry a response at all, and nothing was written.
    fn command(
        &mut self,
        regions: &[ChainRegion],
        request: &dyn ReadableBytes,
        response: &mut dyn WritableBytes,
    ) -> u32;

    /// The model's own turn, taken after each command: work a command
    /// admitted and answered runs here. Hardware does this by itself; a
    /// model does it when the transport says so. Answers whether any ran.
    fn drive(&mut self) -> bool;

    /// Whether the model can no longer answer truthfully. The transport
    /// then stops taking commands and tells the driver to reset the device.
    fn needs_reset(&self) -> bool;

    /// Forget everything a driver made and return to the first state. The
    /// transport has stopped taking and completing chains before it calls
    /// this, and nothing the driver held from before names anything after.
    fn reset(&mut self) -> Result<(), Unrecoverable>;
}
