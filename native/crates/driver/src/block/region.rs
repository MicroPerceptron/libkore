//! A bounded sub-extent view over one direct backend.
//!
//! [`RegionDevice`] is the **designated-region clamp**: constructed whole it
//! is the identity view, and narrowed onto a designated extent every
//! submitted LBA is rebased into that extent and range-checked against it.
//! Code above the wrapper cannot name a block outside the designation — the
//! *shape* protects foreign data, not a convention.  A caller that was never
//! granted a region designation therefore cannot write anywhere at all
//! outside the view it was built with.
//!
//! The wrapper is deliberately dumb about where designations come from: a
//! GPT partition, a fixed platform carve, or a test's arbitrary window all
//! narrow the same way.
//!
//! A view can also be sealed [`read_only`](RegionDevice::read_only): a
//! designation that grants read authority only (a pack image) then cannot
//! reach a write or discard through any caller above it — the refusal lives
//! in the clamp's own rebase, not in a channel flag a caller could skip.

use super::{
    BlockError, BlockGeometry, BlockHealth, DeviceCommand, DeviceCompletion, DirectBlockDevice,
};

/// One backend viewed through a bounded block extent.
pub struct RegionDevice<D: DirectBlockDevice> {
    device: D,
    first_lba: u64,
    blocks: u64,
    /// Sealed read-only: writes and discards are refused at rebase.
    read_only: bool,
}

impl<D: DirectBlockDevice> RegionDevice<D> {
    /// The identity view: the whole backend.
    pub fn whole(device: D) -> Result<Self, BlockError> {
        let geometry = device.geometry().validate()?;
        Ok(Self {
            device,
            first_lba: 0,
            blocks: geometry.block_count,
            read_only: false,
        })
    }

    /// Narrow the identity view onto `[first_lba, first_lba + blocks)`.
    ///
    /// Refused unless the view is currently whole (a narrowed view must
    /// never widen or slide) and the extent fits the backend exactly.  A
    /// refused narrow returns the intact view to its owner.
    pub fn narrow(self, first_lba: u64, blocks: u64) -> Result<Self, (Self, BlockError)> {
        let device_blocks = self.device.geometry().block_count;
        if self.first_lba != 0 || self.blocks != device_blocks {
            return Err((self, BlockError::Geometry));
        }
        let Some(end) = first_lba.checked_add(blocks) else {
            return Err((self, BlockError::Range));
        };
        if blocks == 0 || end > device_blocks {
            return Err((self, BlockError::Range));
        }
        Ok(Self {
            device: self.device,
            first_lba,
            blocks,
            read_only: self.read_only,
        })
    }

    /// Seal the view read-only.  One-way: nothing unseals a view, and
    /// [`narrow`](Self::narrow) keeps the seal, so read authority can never
    /// widen into write authority by re-deriving the clamp.
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// The view's extent as `(first_lba, blocks)` in backend coordinates.
    pub fn region(&self) -> (u64, u64) {
        (self.first_lba, self.blocks)
    }

    /// Whether this view is the whole backend.
    pub fn is_whole(&self) -> bool {
        self.first_lba == 0 && self.blocks == self.device.geometry().block_count
    }

    /// Borrow the wrapped backend.  The region bounds are private, so inner
    /// access can query or drive the backend but never move the view.
    pub fn inner(&self) -> &D {
        &self.device
    }

    /// Mutably borrow the wrapped backend.
    pub fn inner_mut(&mut self) -> &mut D {
        &mut self.device
    }

    /// Recover the wrapped backend, discarding the view.  The next view
    /// over it starts [`whole`](Self::whole) again — a narrowed authority
    /// never transfers implicitly.
    pub fn into_inner(self) -> D {
        self.device
    }

    /// Rebase one view-relative extent into backend coordinates, refusing
    /// anything that does not fit the region entirely.
    fn rebase(&self, lba: u64, blocks: u32) -> Result<u64, BlockError> {
        let end = lba.checked_add(blocks as u64).ok_or(BlockError::Range)?;
        if end > self.blocks {
            return Err(BlockError::Range);
        }
        self.first_lba.checked_add(lba).ok_or(BlockError::Range)
    }

    fn rebase_command(&self, command: DeviceCommand) -> Result<DeviceCommand, BlockError> {
        if self.read_only
            && matches!(
                command,
                DeviceCommand::Write { .. } | DeviceCommand::Discard { .. }
            )
        {
            return Err(BlockError::ReadOnly);
        }
        Ok(match command {
            DeviceCommand::Read {
                lba,
                blocks,
                buffer_offset,
            } => DeviceCommand::Read {
                lba: self.rebase(lba, blocks)?,
                blocks,
                buffer_offset,
            },
            DeviceCommand::Write {
                lba,
                blocks,
                buffer_offset,
            } => DeviceCommand::Write {
                lba: self.rebase(lba, blocks)?,
                blocks,
                buffer_offset,
            },
            DeviceCommand::Discard { lba, blocks } => DeviceCommand::Discard {
                lba: self.rebase(lba, blocks)?,
                blocks,
            },
            DeviceCommand::Flush => DeviceCommand::Flush,
        })
    }
}

impl<D: DirectBlockDevice> DirectBlockDevice for RegionDevice<D> {
    type Buffer = D::Buffer;

    /// A clamp changes addresses, not how the queue advances.
    const INDEPENDENT_PROGRESS: bool = D::INDEPENDENT_PROGRESS;

    fn namespace_lba(&self) -> u64 {
        self.device.namespace_lba().saturating_add(self.first_lba)
    }

    fn abort(&mut self, cookie: u64) -> bool {
        self.device.abort(cookie)
    }

    fn geometry(&self) -> BlockGeometry {
        let inner = self.device.geometry();
        BlockGeometry {
            block_count: self.blocks,
            read_only: inner.read_only || self.read_only,
            ..inner
        }
    }

    fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError> {
        self.device.allocate_buffer(bytes)
    }

    fn max_buffer_bytes(&self) -> usize {
        self.device.max_buffer_bytes()
    }

    fn health(&mut self) -> Result<BlockHealth, BlockError> {
        self.device.health()
    }

    // A clamp narrows addresses; it has nothing to say about whether the
    // backend beneath it is alive.
    fn liveness(&mut self) -> crate::block::Liveness {
        self.device.liveness()
    }

    // Nor about what time it is: the clock is the backend's, and a clamp
    // that swallowed it would turn a timed channel into an unmeasured one
    // (`status fs` read "backend publishes no clock" on metal for exactly
    // this reason — dagfs narrows its lane onto the designated extent).
    fn clock_ns(&self) -> Option<u64> {
        self.device.clock_ns()
    }

    fn submit(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        let command = self.rebase_command(command)?;
        self.device.submit(command, buffer, cookie)
    }

    fn stage(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        let command = self.rebase_command(command)?;
        self.device.stage(command, buffer, cookie)
    }

    fn publish(&mut self) {
        self.device.publish();
    }

    fn poll(&mut self) -> Option<DeviceCompletion> {
        self.device.poll()
    }

    fn poll_batch(&mut self, out: &mut [Option<DeviceCompletion>]) -> usize {
        self.device.poll_batch(out)
    }

    fn quiesce(&mut self) -> Result<(), BlockError> {
        self.device.quiesce()
    }

    async fn quiesce_async(&mut self) -> Result<(), BlockError> {
        self.device.quiesce_async().await
    }

    // Forwarded so a backend's own delivery park is never shadowed by the
    // trait's spin default.
    fn wait(&mut self) -> Option<DeviceCompletion> {
        self.device.wait()
    }

    // Forwarded because the trait's defaults are *no-ops*, and a clamp that
    // silently answered for its backend would be worse than one that could
    // not be built: `finish` is where a copied transport moves a completed
    // read into the caller's buffer, so a swallowed `finish` hands back a
    // stale buffer and calls it success, and a swallowed `progress` never
    // makes the correlated step such a transport needs.  A region clamp
    // narrows *addresses*; it has no business narrowing the data path.
    fn finish(&mut self, buffer: Option<&mut Self::Buffer>) -> Result<(), BlockError> {
        self.device.finish(buffer)
    }

    async fn progress(&mut self) -> Result<(), BlockError> {
        self.device.progress().await
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::block::memory::MemoryBlockDevice;
    use crate::block::{BlockBuffer, BlockChannel, CompletionMode};

    // 4096 disk bytes over 512-byte blocks = an 8-block backend.
    type Device = MemoryBlockDevice<4096, 512>;

    fn device() -> Device {
        Device::new(512, 8).unwrap()
    }

    /// The clamp forwards the backend's clock, so a channel over a narrowed
    /// region still books latency — the shape dagfs runs on metal.
    #[test]
    fn a_narrowed_region_still_times_completions() {
        let mut backend = device();
        backend.set_clock(500, 2_000);
        let region = RegionDevice::whole(backend)
            .unwrap()
            .narrow(2, 4)
            .map_err(|(_, error)| error)
            .unwrap();
        assert_eq!(region.clock_ns(), Some(500));
        let mut channel: BlockChannel<_, 2, 4> =
            BlockChannel::new(region, CompletionMode::Poll).unwrap();
        let id = channel.submit_flush().unwrap();
        assert_eq!(channel.poll().unwrap().request, id);
        let books = channel.stats().latency;
        assert_eq!(
            (books.samples, books.sum_ns, books.max_ns),
            (1, 2_000, 2_000)
        );
        assert!(books.snapshot(channel.stats().max_outstanding).measured());
    }

    /// A clamp names where its block zero sits in the namespace, and a
    /// clamp of a clamp adds both offsets — the coordinates a transport
    /// grant is written in.
    #[test]
    fn nested_regions_report_their_namespace_origin() {
        let whole = RegionDevice::whole(device()).unwrap();
        assert_eq!(whole.namespace_lba(), 0);
        let outer = whole.narrow(2, 6).map_err(|(_, error)| error).unwrap();
        assert_eq!(outer.namespace_lba(), 2);
        let inner = RegionDevice::whole(outer)
            .unwrap()
            .narrow(1, 3)
            .map_err(|(_, error)| error)
            .unwrap();
        assert_eq!(inner.namespace_lba(), 3);
        let channel: BlockChannel<_, 2, 4> = BlockChannel::new(inner, CompletionMode::Poll).unwrap();
        assert_eq!(channel.namespace_lba(), 3);
    }

    #[test]
    fn whole_view_is_the_identity() {
        let region = RegionDevice::whole(device()).unwrap();
        assert!(region.is_whole());
        assert_eq!(region.region(), (0, 8));
        assert_eq!(region.geometry().block_count, 8);
    }

    #[test]
    fn narrow_bounds_geometry_and_rebases_io() {
        let mut whole = RegionDevice::whole(device()).unwrap();
        // Mark backend LBA 2 through the whole view, then find that same
        // block at region LBA 0 through a [2, +4) view.
        let mut buffer = whole.allocate_buffer(512).unwrap();
        buffer.bytes_mut()[..4].copy_from_slice(&[0xa5; 4]);
        whole
            .submit(
                DeviceCommand::Write {
                    lba: 2,
                    blocks: 1,
                    buffer_offset: 0,
                },
                Some(&mut buffer),
                1,
            )
            .unwrap();
        whole.poll().unwrap().result.unwrap();

        let mut region = whole.narrow(2, 4).map_err(|(_, error)| error).unwrap();
        assert!(!region.is_whole());
        assert_eq!(region.geometry().block_count, 4);
        let mut back = region.allocate_buffer(512).unwrap();
        region
            .submit(
                DeviceCommand::Read {
                    lba: 0,
                    blocks: 1,
                    buffer_offset: 0,
                },
                Some(&mut back),
                2,
            )
            .unwrap();
        region.poll().unwrap().result.unwrap();
        assert_eq!(&back.bytes()[..4], &[0xa5; 4]);
    }

    #[test]
    fn io_cannot_leave_the_region() {
        let mut region = RegionDevice::whole(device())
            .unwrap()
            .narrow(2, 4)
            .map_err(|(_, error)| error)
            .unwrap();
        let mut buffer = region.allocate_buffer(512).unwrap();
        // One past the region's last block, and a run crossing its end.
        for (lba, blocks) in [(4, 1), (3, 2), (u64::MAX, 1)] {
            assert_eq!(
                region.submit(
                    DeviceCommand::Write {
                        lba,
                        blocks,
                        buffer_offset: 0,
                    },
                    Some(&mut buffer),
                    3,
                ),
                Err(BlockError::Range)
            );
        }
    }

    #[test]
    fn narrow_refuses_slide_widen_and_overflow() {
        let region = RegionDevice::whole(device()).unwrap();
        let (region, error) = region.narrow(4, 8).err().unwrap();
        assert_eq!(error, BlockError::Range);
        let (region, error) = region.narrow(0, 0).err().unwrap();
        assert_eq!(error, BlockError::Range);
        let (region, error) = region.narrow(u64::MAX, 2).err().unwrap();
        assert_eq!(error, BlockError::Range);
        // A narrowed view must never narrow again (slide) or widen.
        let region = region.narrow(2, 4).map_err(|(_, error)| error).unwrap();
        let (_, error) = region.narrow(2, 2).err().unwrap();
        assert_eq!(error, BlockError::Geometry);
    }

    /// A clamp narrows addresses, not the data path.  A copied transport
    /// finishes a read inside `finish`; if the clamp answered with the
    /// trait's no-op default instead of forwarding, the caller would get a
    /// stale buffer reported as success.
    #[test]
    fn a_region_forwards_the_completion_side_of_a_copied_transport() {
        /// A transport that, like a real remote endpoint, only lands read
        /// data when the channel calls `finish`.
        struct Copied {
            inner: Device,
            landed: u8,
        }
        impl DirectBlockDevice for Copied {
            type Buffer = <Device as DirectBlockDevice>::Buffer;
            fn geometry(&self) -> BlockGeometry {
                self.inner.geometry()
            }
            fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError> {
                self.inner.allocate_buffer(bytes)
            }
            fn submit(
                &mut self,
                command: DeviceCommand,
                buffer: Option<&mut Self::Buffer>,
                cookie: u64,
            ) -> Result<(), BlockError> {
                // Record the rebased target so the test can prove the clamp
                // still applied, then complete without touching the buffer.
                if let DeviceCommand::Read { lba, .. } = command {
                    self.landed = 0xd0 | (lba as u8);
                }
                self.inner.submit(command, buffer, cookie)
            }
            fn poll(&mut self) -> Option<DeviceCompletion> {
                self.inner.poll()
            }
            fn quiesce(&mut self) -> Result<(), BlockError> {
                self.inner.quiesce()
            }
            fn finish(&mut self, buffer: Option<&mut Self::Buffer>) -> Result<(), BlockError> {
                let buffer = buffer.ok_or(BlockError::UnknownBuffer)?;
                buffer.bytes_mut().fill(self.landed);
                Ok(())
            }
        }

        let device = Copied {
            inner: device(),
            landed: 0,
        };
        let region = RegionDevice::whole(device)
            .unwrap()
            .narrow(2, 4)
            .map_err(|(_, error)| error)
            .unwrap();
        let mut channel: BlockChannel<_, 2, 4> =
            BlockChannel::new(region, CompletionMode::Poll).unwrap();
        let buffer = channel.register_buffer(512).unwrap();
        let request = channel.submit_read(1, 1, buffer, 0).unwrap();
        let completion = channel.wait().unwrap();
        assert_eq!(completion.request, request);
        completion.result.unwrap();
        let mut back = [0u8; 512];
        channel.read_buffer(buffer, 0, &mut back).unwrap();
        // Region LBA 1 rebased to backend LBA 3, and `finish` ran: without
        // the forward the buffer would still be all zero.
        assert_eq!(back, [0xd3; 512]);
    }

    #[test]
    fn a_channel_over_a_region_stays_inside_it() {
        let region = RegionDevice::whole(device())
            .unwrap()
            .narrow(2, 4)
            .map_err(|(_, error)| error)
            .unwrap();
        let mut channel: BlockChannel<_, 2, 4> =
            BlockChannel::new(region, CompletionMode::Poll).unwrap();
        assert_eq!(channel.geometry().block_count, 4);
        let buffer = channel.register_buffer(512).unwrap();
        channel.write_buffer(buffer, 0, &[7; 512]).unwrap();
        let id = channel.submit_write(3, 1, buffer, 0).unwrap();
        assert_eq!(channel.wait().unwrap().request, id);
        assert!(channel.submit_write(4, 1, buffer, 0).is_err());
    }

    /// A sealed view refuses writes and discards in the clamp itself — a
    /// device call below a channel that forgot its own read-only state still
    /// cannot reach the blocks — while reads rebase exactly as before, and a
    /// later narrow keeps the seal.
    #[test]
    fn a_read_only_view_refuses_writes_below_the_channel() {
        let mut region = RegionDevice::whole(device())
            .unwrap()
            .read_only()
            .narrow(2, 4)
            .map_err(|(_, error)| error)
            .unwrap();
        assert!(region.geometry().read_only);
        let mut buffer = region.allocate_buffer(512).unwrap();
        let write = DeviceCommand::Write {
            lba: 0,
            blocks: 1,
            buffer_offset: 0,
        };
        assert_eq!(
            region.submit(write, Some(&mut buffer), 1),
            Err(BlockError::ReadOnly)
        );
        assert_eq!(
            region.submit(DeviceCommand::Discard { lba: 0, blocks: 1 }, None, 2),
            Err(BlockError::ReadOnly)
        );
        let read = DeviceCommand::Read {
            lba: 1,
            blocks: 1,
            buffer_offset: 0,
        };
        region.submit(read, Some(&mut buffer), 3).unwrap();

        let mut channel: BlockChannel<_, 2, 4> =
            BlockChannel::new(region, CompletionMode::Poll).unwrap();
        let registered = channel.register_buffer(512).unwrap();
        assert_eq!(
            channel.submit_write(0, 1, registered, 0),
            Err(BlockError::ReadOnly)
        );
    }
}
