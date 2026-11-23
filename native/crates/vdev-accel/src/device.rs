//! **The device**: the protocol's command engine over the CPU backend.
//!
//! A virtio-accel device has one job per driver request: take a request
//! frame, act on it, and write a response frame. The protocol crate ships the
//! engine that does this for any backend (`virtio_accel_device`'s
//! `CommandProcessor`): it decodes and validates the frame, keeps the object
//! graph (contexts, buffers, programs, queues, events) and refuses a request
//! before the backend sees it. This module is that engine instantiated over
//! [`SpmdBackend`] with the FP8 registry, which is the whole device.
//!
//! Three things a transport needs, and nothing about any transport:
//!
//! - [`AccelDevice::CONFIG`], the configuration a driver reads before it
//!   sends a frame: the protocol version and the largest frames and chains
//!   this device takes;
//! - [`AccelDevice::command`], one request in and one response out;
//! - [`AccelDevice::drive`], the device's own turn. A submission is admitted
//!   and answered at once, as the protocol requires, and its work runs here.
//!   On hardware this is the silicon's; for this device it is a call the
//!   host makes between commands, so a driver that submits and then polls
//!   sees the event complete.
//!
//! Those are the device model's contract in `kore-vdev`
//! ([`VirtualDevice`]), which is how a driver reaches this device: through
//! that crate's register block and queue, with nothing here knowing of
//! either.
//!
//! Two of the protocol's rules that a driver writer meets first: a buffer
//! bound to a submission is the event's until the driver destroys the event,
//! so results are read after `DestroyEvent`, not after the poll that saw it
//! complete; and a reset gives every object a new namespace, so an id a
//! driver kept from before the reset names nothing afterwards.

use kore_svc_accel_spmd::SpmdBackend;
use kore_vdev::device::{ReadableBytes, Unrecoverable, VirtualDevice, WritableBytes};
use virtio_accel_core::{ByteSink, ByteSource, TransportByteSink, TransportByteSource};
use virtio_accel_device::{
    ChainRegion, CommandOutcome, CommandProcessError, CommandProcessor, CommandProcessorInitError,
    DeviceHealth, ObjectNamespace, ResetDisposition, ResourcePolicy,
};
use virtio_accel_proto::{
    BASELINE_COMMAND_QUEUES, Le16, Le32, PROTOCOL_MAJOR, PROTOCOL_MINOR, WireConfig,
};
use zerocopy::IntoBytes;

use crate::fp8;

/// Descriptors one command chain may have. A request is at most a header, a
/// fixed payload and one run of data, and a response a header and one run:
/// sixteen leaves a driver room to split either without making a chain a
/// place to hide unbounded work.
pub const MAX_CHAIN_DESCRIPTORS: u16 = 16;

/// The largest request or response frame, header included.
///
/// A buffer transfer rides inside a frame, so this bounds one transfer, not
/// a buffer: a driver moves a larger buffer in pieces. 64 KiB is sixteen
/// pages of a driver's shared memory per frame in flight.
pub const MAX_FRAME_BYTES: u32 = 64 * 1024;

/// Buffer backing the device will hold for a driver at once: 2 MiB, two
/// million E4M3 values. The backing is the host's heap, so this is what one
/// driver can make the device's host allocate, and a host that serves
/// several drivers from one heap sizes that heap by it.
pub const MAX_BUFFER_BACKING_BYTES: u64 = 2 << 20;

/// Resident program bytes the device will hold. A program here is a
/// registered kernel's handle, a few bytes; 64 KiB admits every program the
/// backend's own count limit allows many times over.
pub const MAX_PROGRAM_RESIDENT_BYTES: u64 = 64 * 1024;

/// What a driver reads from the register block's device id: `"accl"`. The
/// protocol assigns itself no Virtio device id, so the name is kore's.
pub const DEVICE_ID: u32 = u32::from_le_bytes(*b"accl");

/// The virtual accelerator.
pub struct AccelDevice {
    processor: CommandProcessor<SpmdBackend>,
    /// The namespace the device's objects live in. Each reset takes the
    /// next, which is what makes an id from before a reset name nothing
    /// after it.
    namespace: ObjectNamespace,
}

impl AccelDevice {
    /// What a driver reads before its first frame.
    pub const CONFIG: WireConfig = WireConfig {
        protocol_major: Le16::new(PROTOCOL_MAJOR),
        protocol_minor: Le16::new(PROTOCOL_MINOR),
        command_queue_count: Le16::new(BASELINE_COMMAND_QUEUES),
        max_chain_descriptors: Le16::new(MAX_CHAIN_DESCRIPTORS),
        max_request_bytes: Le32::new(MAX_FRAME_BYTES),
        max_response_bytes: Le32::new(MAX_FRAME_BYTES),
    };

    /// A device in its first state.
    pub fn new() -> Result<Self, CommandProcessorInitError> {
        const FIRST: ObjectNamespace = ObjectNamespace::new(1).unwrap();
        Self::in_namespace(FIRST)
    }

    fn in_namespace(namespace: ObjectNamespace) -> Result<Self, CommandProcessorInitError> {
        const POLICY: ResourcePolicy =
            ResourcePolicy::new(MAX_BUFFER_BACKING_BYTES, MAX_PROGRAM_RESIDENT_BYTES).unwrap();
        Ok(Self {
            processor: CommandProcessor::new(
                SpmdBackend::new(fp8::KERNELS),
                &Self::CONFIG,
                namespace,
                POLICY,
            )?,
            namespace,
        })
    }

    /// One command chain: `request` is its device-readable bytes, `response`
    /// its device-writable bytes and `regions` their layout in the chain.
    ///
    /// The outcome says how many response bytes were written, or that the
    /// chain cannot carry a response at all. An `Err` is the transport's own
    /// memory failing under the engine, not a protocol error: those are
    /// responses.
    pub fn command(
        &mut self,
        regions: &[ChainRegion],
        request: &dyn ByteSource,
        response: &mut dyn ByteSink,
    ) -> Result<CommandOutcome, CommandProcessError> {
        self.processor.process(regions, request, response)
    }

    /// Run the work submissions admitted, oldest first. Answers whether any
    /// ran, which is when an event a driver holds may have become terminal.
    pub fn drive(&mut self) -> bool {
        let mut ran = false;
        // One admitted launch is in flight at a time, so this loop runs it
        // and ends: the backend refuses a second admission until then.
        while self.processor.accelerator().run_pending() {
            ran = true;
        }
        ran
    }

    /// Whether the device still serves, or needs the reset a transport
    /// offers its driver.
    pub fn health(&self) -> DeviceHealth {
        self.processor.health()
    }
}

impl core::fmt::Debug for AccelDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The engine's object graph is its own business, and a driver's.
        f.debug_struct("AccelDevice")
            .field("namespace", &self.namespace.get())
            .field("health", &self.health())
            .finish_non_exhaustive()
    }
}

impl VirtualDevice for AccelDevice {
    const DEVICE_ID: u32 = DEVICE_ID;
    /// The largest ring there is: four of the longest chains, or thirty-two
    /// commands of two descriptors, in flight at once.
    const MAX_QUEUE_SIZE: u16 = kore_vdev::ring::MAX_QUEUE_SIZE;
    const MAX_CHAIN_DESCRIPTORS: u16 = MAX_CHAIN_DESCRIPTORS;

    fn config(&self) -> &[u8] {
        const CONFIG: &WireConfig = &AccelDevice::CONFIG;
        CONFIG.as_bytes()
    }

    fn command(
        &mut self,
        regions: &[ChainRegion],
        request: &dyn ReadableBytes,
        response: &mut dyn WritableBytes,
    ) -> u32 {
        // The engine's byte ports over the transport's, with no copy
        // between: a frame is read from and written to the driver's memory.
        let request = TransportByteSource::new(request);
        let mut response = TransportByteSink::new(response);
        match AccelDevice::command(self, regions, &request, &mut response) {
            Ok(CommandOutcome::Response { used, .. }) => used,
            // A frame with no usable response, or the chain's memory failing
            // under the engine. Either way nothing was written the driver
            // may read; if the failure came after the engine had acted, the
            // engine has marked itself, and `needs_reset` says so.
            Ok(CommandOutcome::Unusable(_)) | Err(_) => 0,
        }
    }

    fn drive(&mut self) -> bool {
        AccelDevice::drive(self)
    }

    fn needs_reset(&self) -> bool {
        self.health() != DeviceHealth::Running
    }

    fn reset(&mut self) -> Result<(), Unrecoverable> {
        // The namespace is sixteen bits and never reused: a device reset
        // sixty-five thousand times is not brought back, because an id from
        // its first life would name an object of its last.
        let next = self
            .namespace
            .get()
            .checked_add(1)
            .and_then(ObjectNamespace::new)
            .ok_or(Unrecoverable)?;
        let report = self.processor.reset(next).map_err(|_| Unrecoverable)?;
        match report.disposition {
            ResetDisposition::BackendReusable => self.namespace = next,
            // The backend could not account for something it held. The
            // protocol forbids calling it again: it is dropped whole, and a
            // new engine over a new backend takes its place.
            ResetDisposition::BackendDiscardRequired => {
                *self = Self::in_namespace(next).map_err(|_| Unrecoverable)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
