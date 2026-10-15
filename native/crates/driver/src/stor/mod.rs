//! The **storage transport** engine (`res/STORAGE_TRANSPORT.md` §2).
//!
//! Three layers of the five live here:
//!
//! - **L0, the queue set** ([`QueueSet`]): one worker's slice of a
//!   controller — one completion queue and [`QueueSet::SQS`] submission
//!   queues. [`Single`] adapts every existing [`DirectBlockDevice`] as a
//!   one-queue set; the NVMe multi-queue set lives in `kore-nvme`.
//! - **L1, the engine** ([`engine::StorEngine`]): pops client submissions
//!   from each ring's class channels, validates them against the ring's
//!   grants and targets, splits them into device commands, keeps every class
//!   inside its budget, and posts exactly one completion per submission. It
//!   never waits.
//! - The **authority** half of L3 ([`grant`]): what a grant is and how a
//!   byte range inside it resolves to device blocks.
//!
//! Everything is generic and fixed-size; a production row instantiates the
//! engine once with its queue set and class policy. Nothing is `dyn`, and
//! nothing allocates.

pub mod engine;
pub mod grant;
pub mod lanes;
pub mod timeline;

use core::sync::atomic::AtomicU32;

use crate::block::{
    BlockBuffer, BlockError, BlockGeometry, BlockHealth, DeviceCommand, DeviceCompletion,
    DirectBlockDevice, Liveness,
};

/// An engine-minted command tag: the device echoes it back as the
/// completion's cookie, and the engine indexes its command table by it.
/// Opaque to the queue set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tag(pub u32);

impl Tag {
    /// The cookie a device carries for this tag.
    pub const fn cookie(self) -> u64 {
        self.0 as u64
    }
}

/// **L0** — one worker's slice of a storage controller: a completion queue
/// and `SQS` submission queues, driven only by the one worker that owns it.
///
/// Implementations execute only when called; they spawn nothing and take no
/// lock. `stage` never rings a doorbell; `publish` rings one per queue.
pub trait QueueSet {
    /// Device-visible caller memory a command reads into or writes from.
    type Target;
    /// Submission queues in this set.
    const SQS: usize;

    /// The namespace's geometry.
    fn geometry(&self) -> BlockGeometry;
    /// Bytes a target spans.
    fn target_bytes(target: &Self::Target) -> u64;
    /// Allocate `bytes` of device-visible memory the set's owner funds —
    /// its own bounce and metadata buffers. Client memory never comes from
    /// here; it is adopted.
    fn allocate(&mut self, bytes: usize) -> Result<Self::Target, BlockError>;
    /// The largest target [`allocate`](Self::allocate) will make.
    fn max_target_bytes(&self) -> usize;
    /// Block, bounded, until the device posts a completion, and hold it for
    /// the next [`reap`](Self::reap). `false` when nothing arrived within
    /// the set's own wait budget (or nothing was in flight). A set with no
    /// way to sleep may spin; it must still return.
    fn park(&mut self) -> bool {
        false
    }
    /// Commands submission queue `sq` can still accept.
    fn room(&self, sq: usize) -> usize;
    /// Stage one command on `sq`, to be published by [`publish`](Self::publish).
    fn stage(
        &mut self,
        sq: usize,
        command: DeviceCommand,
        target: Option<&mut Self::Target>,
        tag: Tag,
    ) -> Result<(), BlockError>;
    /// Make every command staged on `sq` visible to the device.
    fn publish(&mut self, sq: usize);
    /// Drain up to `out.len()` completions, in the order the device posted
    /// them. Each carries the staging [`Tag`]'s cookie.
    fn reap(&mut self, out: &mut [DeviceCompletion]) -> usize;
    /// Apply completion-side data to `target` — a copied transport finishes
    /// a read here; DMA has already landed and uses the default.
    fn finish(&mut self, _target: Option<&mut Self::Target>) -> Result<(), BlockError> {
        Ok(())
    }
    /// Out-of-band liveness, never a command ([`DirectBlockDevice::liveness`]).
    fn liveness(&mut self) -> Liveness {
        Liveness::Unknown
    }
    /// Physical health on the infrequent control plane
    /// ([`DirectBlockDevice::health`]).
    fn health(&mut self) -> Result<BlockHealth, BlockError> {
        Err(BlockError::Unsupported)
    }
    /// Stop the device once nothing is in flight.
    fn quiesce(&mut self) -> Result<(), BlockError> {
        Ok(())
    }
    /// Monotonic nanoseconds, when the set's task can read a clock.
    fn clock_ns(&self) -> Option<u64> {
        None
    }
    /// Ask the device to abandon the command staged on `sq` as `tag` — the
    /// deadline's escalation ([`DirectBlockDevice::abort`]'s contract: the
    /// command's own completion still follows). `false`, the default, when
    /// the set has no abort.
    fn abort(&mut self, _sq: usize, _tag: Tag) -> bool {
        false
    }
    /// Reset the device and bring every queue back
    /// ([`DirectBlockDevice::reset`]'s contract): `true` means every staged
    /// command is gone — none completes — and its target is free. `false`,
    /// the default, when the set cannot.
    fn reset(&mut self) -> bool {
        false
    }
    /// The word a monitored wait can watch for the next completion.
    fn completion_word(&self) -> Option<&AtomicU32> {
        None
    }
}

/// Every [`DirectBlockDevice`] as a one-queue [`QueueSet`], at most `DEPTH`
/// commands in flight.
///
/// This is how the existing devices — the memory double, a clamped region,
/// today's NVMe backend — serve the engine without being rewritten.
pub struct Single<D: DirectBlockDevice, const DEPTH: usize> {
    device: D,
    outstanding: usize,
    /// A completion `park` took from the device's wait, returned first by
    /// the next `reap`.
    parked: Option<DeviceCompletion>,
}

impl<D: DirectBlockDevice, const DEPTH: usize> Single<D, DEPTH> {
    /// Wrap `device`.
    pub const fn new(device: D) -> Self {
        Self {
            device,
            outstanding: 0,
            parked: None,
        }
    }

    /// The wrapped device.
    pub fn device(&self) -> &D {
        &self.device
    }

    /// The wrapped device, mutably — a test's fault knobs, a control step.
    pub fn device_mut(&mut self) -> &mut D {
        &mut self.device
    }

    /// Commands staged and not yet reaped.
    pub const fn outstanding(&self) -> usize {
        self.outstanding
    }

    /// Unwrap, once nothing is outstanding.
    pub fn into_device(self) -> Result<D, Self> {
        if self.outstanding == 0 {
            Ok(self.device)
        } else {
            Err(self)
        }
    }
}

impl<D: DirectBlockDevice, const DEPTH: usize> QueueSet for Single<D, DEPTH> {
    type Target = D::Buffer;
    const SQS: usize = 1;

    fn geometry(&self) -> BlockGeometry {
        self.device.geometry()
    }

    fn target_bytes(target: &Self::Target) -> u64 {
        target.len() as u64
    }

    fn allocate(&mut self, bytes: usize) -> Result<Self::Target, BlockError> {
        self.device.allocate_buffer(bytes)
    }

    fn max_target_bytes(&self) -> usize {
        self.device.max_buffer_bytes()
    }

    fn park(&mut self) -> bool {
        if self.parked.is_some() {
            return true;
        }
        if self.outstanding == 0 {
            return false;
        }
        self.parked = self.device.wait();
        self.parked.is_some()
    }

    fn room(&self, _sq: usize) -> usize {
        DEPTH.saturating_sub(self.outstanding)
    }

    fn stage(
        &mut self,
        _sq: usize,
        command: DeviceCommand,
        target: Option<&mut Self::Target>,
        tag: Tag,
    ) -> Result<(), BlockError> {
        if self.outstanding >= DEPTH {
            return Err(BlockError::Backpressure);
        }
        self.device.stage(command, target, tag.cookie())?;
        self.outstanding += 1;
        Ok(())
    }

    fn publish(&mut self, _sq: usize) {
        self.device.publish();
    }

    fn reap(&mut self, out: &mut [DeviceCompletion]) -> usize {
        let mut count = 0;
        if let Some(first) = out.first_mut()
            && let Some(parked) = self.parked.take()
        {
            *first = parked;
            count = 1;
        }
        while count < out.len() {
            let Some(completion) = self.device.poll() else {
                break;
            };
            out[count] = completion;
            count += 1;
        }
        self.outstanding = self.outstanding.saturating_sub(count);
        count
    }

    fn finish(&mut self, target: Option<&mut Self::Target>) -> Result<(), BlockError> {
        self.device.finish(target)
    }

    fn liveness(&mut self) -> Liveness {
        self.device.liveness()
    }

    fn health(&mut self) -> Result<BlockHealth, BlockError> {
        self.device.health()
    }

    fn quiesce(&mut self) -> Result<(), BlockError> {
        if self.outstanding != 0 || self.parked.is_some() {
            return Err(BlockError::Draining);
        }
        self.device.quiesce()
    }

    fn clock_ns(&self) -> Option<u64> {
        self.device.clock_ns()
    }

    fn abort(&mut self, _sq: usize, tag: Tag) -> bool {
        self.device.abort(tag.cookie())
    }

    fn reset(&mut self) -> bool {
        self.device.reset()
    }
}
