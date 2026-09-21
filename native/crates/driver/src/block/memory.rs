//! Deterministic memory-backed direct device used by conformance suites.
//! It implements the real asynchronous submit/poll contract: submissions
//! complete into a bounded queue, and tests may hold or fault completions.

use super::{
    BlockBuffer, BlockError, BlockGeometry, BlockIoHints, DeviceCommand, DeviceCompletion,
    DirectBlockDevice,
};

const COMPLETIONS: usize = 64;

pub struct MemoryBuffer<const BYTES: usize> {
    bytes: [u8; BYTES],
    len: usize,
}

impl<const BYTES: usize> BlockBuffer for MemoryBuffer<BYTES> {
    fn len(&self) -> usize {
        self.len
    }
    fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes[..self.len]
    }
}

pub struct MemoryBlockDevice<const DISK_BYTES: usize, const BUFFER_BYTES: usize> {
    geometry: BlockGeometry,
    disk: [u8; DISK_BYTES],
    completions: [Option<DeviceCompletion>; COMPLETIONS],
    head: usize,
    tail: usize,
    queued: usize,
    hold: bool,
    /// Empty waits still to run out before a held completion is released —
    /// the stand-in for a device that answers late but does answer.
    hold_for: Option<u32>,
    fault: Option<BlockError>,
    fault_after: usize,
    flushes: u64,
    submissions: u64,
    publishes: u64,
    liveness: crate::block::Liveness,
    /// The clock this double reports through
    /// [`DirectBlockDevice::clock_ns`] — `None` (the default) is a backend
    /// with no time source, so nothing is measured; a test that wants
    /// latency books sets it and gives it a per-poll step.
    clock_ns: Option<u64>,
    /// Nanoseconds the clock advances on every `poll` — time passing while
    /// a command is in flight, so a completion is observed later than its
    /// submission.
    clock_step_ns: u64,
}

impl<const DISK_BYTES: usize, const BUFFER_BYTES: usize>
    MemoryBlockDevice<DISK_BYTES, BUFFER_BYTES>
{
    pub fn new(block_size: u32, max_transfer_blocks: u32) -> Result<Self, BlockError> {
        if block_size == 0 || DISK_BYTES == 0 || DISK_BYTES % block_size as usize != 0 {
            return Err(BlockError::Geometry);
        }
        let geometry = BlockGeometry {
            block_size,
            block_count: (DISK_BYTES / block_size as usize) as u64,
            max_transfer_blocks,
            atomic_write_blocks: 1,
            flush: true,
            discard: true,
            read_only: false,
            io_hints: BlockIoHints::NONE,
        }
        .validate()?;
        Ok(Self {
            geometry,
            disk: [0; DISK_BYTES],
            completions: [None; COMPLETIONS],
            head: 0,
            tail: 0,
            queued: 0,
            hold: false,
            hold_for: None,
            fault: None,
            fault_after: 0,
            flushes: 0,
            submissions: 0,
            publishes: 0,
            // A double has no register to read; say so rather than vouch.
            liveness: crate::block::Liveness::Unknown,
            clock_ns: None,
            clock_step_ns: 0,
        })
    }

    /// Give the double a clock reading, so the channel above books latency:
    /// `now_ns` at once, advancing by `step_ns` on every poll — the stand-in
    /// for time passing between a submission and its completion.  Set on
    /// the control plane before anything is outstanding.
    pub fn set_clock(&mut self, now_ns: u64, step_ns: u64) {
        self.clock_ns = Some(now_ns);
        self.clock_step_ns = step_ns;
    }

    pub fn inject(&mut self, error: BlockError) {
        self.inject_after(0, error);
    }
    pub fn set_read_only(&mut self, read_only: bool) {
        self.geometry.read_only = read_only;
    }
    /// Override optional performance guidance for geometry-sensitive tests.
    pub fn with_io_hints(mut self, io_hints: BlockIoHints) -> Self {
        self.geometry.io_hints = io_hints;
        self
    }
    pub fn inject_completion(
        &mut self,
        cookie: u64,
        result: Result<(), BlockError>,
    ) -> Result<(), BlockError> {
        self.queue(DeviceCompletion { cookie, result })
    }
    /// Fail the submission after `successful_submissions` more commands.
    pub fn inject_after(&mut self, successful_submissions: usize, error: BlockError) {
        self.fault = Some(error);
        self.fault_after = successful_submissions;
    }
    pub fn hold_completions(&mut self, hold: bool) {
        self.hold = hold;
        self.hold_for = None;
    }
    /// Hold completions through the next `waits` empty waits, then release
    /// them: a device that is slow for that many budgets and then completes.
    pub fn hold_completions_for(&mut self, waits: u32) {
        self.hold = true;
        self.hold_for = Some(waits);
    }
    /// Drive the out-of-band liveness answer, standing in for a controller
    /// status register.
    pub fn set_liveness(&mut self, liveness: crate::block::Liveness) {
        self.liveness = liveness;
    }
    pub const fn flushes(&self) -> u64 {
        self.flushes
    }
    pub const fn submissions(&self) -> u64 {
        self.submissions
    }
    pub const fn publishes(&self) -> u64 {
        self.publishes
    }
    pub fn corrupt_byte(&mut self, offset: usize, xor: u8) -> Result<(), BlockError> {
        let byte = self.disk.get_mut(offset).ok_or(BlockError::Range)?;
        *byte ^= xor;
        Ok(())
    }

    fn queue(&mut self, completion: DeviceCompletion) -> Result<(), BlockError> {
        if self.queued == COMPLETIONS {
            return Err(BlockError::Backpressure);
        }
        self.completions[self.tail] = Some(completion);
        self.tail = (self.tail + 1) % COMPLETIONS;
        self.queued += 1;
        Ok(())
    }
}

impl<const DISK_BYTES: usize, const BUFFER_BYTES: usize> DirectBlockDevice
    for MemoryBlockDevice<DISK_BYTES, BUFFER_BYTES>
{
    type Buffer = MemoryBuffer<BUFFER_BYTES>;

    /// Completions are queued by `submit` itself, so there is no correlated
    /// step to await — this double stands in for a hardware queue.
    const INDEPENDENT_PROGRESS: bool = true;

    fn liveness(&mut self) -> crate::block::Liveness {
        self.liveness
    }

    /// The trait's polled wait, with the late-device double folded in: a
    /// hold armed for `n` waits runs `n` budgets out empty, then lifts
    /// itself so the queued completions flow on the wait after.
    fn wait(&mut self) -> Option<DeviceCompletion> {
        if let Some(left) = self.hold_for {
            if left == 0 {
                self.hold = false;
                self.hold_for = None;
            } else {
                self.hold_for = Some(left - 1);
                return None;
            }
        }
        use crate::block::CompletionPacer;
        let mut pacer = crate::block::SpinPacer::default();
        loop {
            if let Some(completion) = self.poll() {
                return Some(completion);
            }
            if !pacer.pace() {
                return self.poll();
            }
        }
    }

    fn clock_ns(&self) -> Option<u64> {
        self.clock_ns
    }

    fn geometry(&self) -> BlockGeometry {
        self.geometry
    }

    fn max_buffer_bytes(&self) -> usize {
        BUFFER_BYTES
    }

    fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError> {
        if bytes == 0 || bytes > BUFFER_BYTES {
            return Err(BlockError::TooLarge);
        }
        Ok(MemoryBuffer {
            bytes: [0; BUFFER_BYTES],
            len: bytes,
        })
    }

    fn submit(
        &mut self,
        command: DeviceCommand,
        mut buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        if self.queued == COMPLETIONS {
            return Err(BlockError::Backpressure);
        }
        self.submissions = self.submissions.saturating_add(1);
        let fault = if self.fault.is_some() && self.fault_after == 0 {
            self.fault.take()
        } else {
            if self.fault.is_some() {
                self.fault_after -= 1;
            }
            None
        };
        let result = if let Some(error) = fault {
            Err(error)
        } else {
            let block = self.geometry.block_size as usize;
            match command {
                DeviceCommand::Read {
                    lba,
                    blocks,
                    buffer_offset,
                } => {
                    let bytes = blocks as usize * block;
                    let disk_at = lba as usize * block;
                    let buf = buffer.as_deref_mut().ok_or(BlockError::UnknownBuffer)?;
                    buf.bytes_mut()[buffer_offset..buffer_offset + bytes]
                        .copy_from_slice(&self.disk[disk_at..disk_at + bytes]);
                    Ok(())
                }
                DeviceCommand::Write {
                    lba,
                    blocks,
                    buffer_offset,
                } => {
                    let bytes = blocks as usize * block;
                    let disk_at = lba as usize * block;
                    let buf = buffer.as_deref().ok_or(BlockError::UnknownBuffer)?;
                    self.disk[disk_at..disk_at + bytes]
                        .copy_from_slice(&buf.bytes()[buffer_offset..buffer_offset + bytes]);
                    Ok(())
                }
                DeviceCommand::Flush => {
                    self.flushes = self.flushes.saturating_add(1);
                    Ok(())
                }
                DeviceCommand::Discard { lba, blocks } => {
                    let bytes = blocks as usize * block;
                    let disk_at = lba as usize * block;
                    self.disk[disk_at..disk_at + bytes].fill(0);
                    Ok(())
                }
            }
        };
        self.queue(DeviceCompletion { cookie, result })
    }

    fn poll(&mut self) -> Option<DeviceCompletion> {
        if self.hold || self.queued == 0 {
            return None;
        }
        // Time passes while a command is in flight: the clock the channel
        // reads at completion is a step past the one it read at submission.
        self.clock_ns = self
            .clock_ns
            .map(|now| now.saturating_add(self.clock_step_ns));
        let completion = self.completions[self.head].take();
        self.head = (self.head + 1) % COMPLETIONS;
        self.queued -= 1;
        completion
    }

    fn publish(&mut self) {
        self.publishes = self.publishes.saturating_add(1);
    }

    fn quiesce(&mut self) -> Result<(), BlockError> {
        if self.queued == 0 {
            Ok(())
        } else {
            Err(BlockError::Draining)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockChannel, ChannelState, CompletionMode, LatencyBooks, RequestId};

    type Device = MemoryBlockDevice<8192, 4096>;
    type Channel = BlockChannel<Device, 2, 4>;

    fn channel() -> Channel {
        BlockChannel::new(Device::new(512, 4).unwrap(), CompletionMode::Poll).unwrap()
    }

    #[test]
    fn direct_read_write_flush_and_discard_round_trip() {
        let mut channel = channel();
        let buffer = channel.register_buffer(2048).unwrap();
        channel.write_buffer(buffer, 0, &[0xa5; 1024]).unwrap();
        let write = channel.submit_write(2, 2, buffer, 0).unwrap();
        assert_eq!(channel.poll().unwrap().request, write);
        let flush = channel.submit_flush().unwrap();
        assert_eq!(channel.poll().unwrap().request, flush);

        channel.write_buffer(buffer, 0, &[0; 1024]).unwrap();
        let read = channel.submit_read(2, 2, buffer, 0).unwrap();
        assert_eq!(channel.poll().unwrap().request, read);
        let mut bytes = [0; 1024];
        channel.read_buffer(buffer, 0, &mut bytes).unwrap();
        assert_eq!(bytes, [0xa5; 1024]);

        let discard = channel.submit_discard(2, 2).unwrap();
        assert_eq!(channel.poll().unwrap().request, discard);
        let read = channel.submit_read(2, 2, buffer, 0).unwrap();
        assert_eq!(channel.poll().unwrap().request, read);
        channel.read_buffer(buffer, 0, &mut bytes).unwrap();
        assert_eq!(bytes, [0; 1024]);
    }

    #[test]
    fn a_batch_publishes_once_and_reaps_a_dense_completion_prefix() {
        let mut channel = channel();
        let a = channel.register_buffer(512).unwrap();
        let b = channel.register_buffer(512).unwrap();
        let (first, second) = {
            let mut batch = channel.batch();
            let first = batch.stage_read(0, 1, a, 0).unwrap();
            let second = batch.stage_read(1, 1, b, 0).unwrap();
            batch.publish();
            (first, second)
        };
        let completions = channel.poll_batch::<4>();
        assert_eq!(completions[0].unwrap().request, first);
        assert_eq!(completions[1].unwrap().request, second);
        assert!(completions[2..].iter().all(Option::is_none));
        assert_eq!(channel.outstanding(), 0);
        assert_eq!(channel.control(|device| device.publishes()), Ok(1));
    }

    #[test]
    fn bounds_alignment_busy_backpressure_cancellation_and_revoke_fail_closed() {
        let mut channel = channel();
        let a = channel.register_buffer(2048).unwrap();
        assert_eq!(channel.submit_read(0, 1, a, 1), Err(BlockError::Alignment));
        assert_eq!(channel.submit_read(16, 1, a, 0), Err(BlockError::Range));
        assert_eq!(channel.submit_read(0, 5, a, 0), Err(BlockError::TooLarge));

        let request = channel.submit_read(0, 1, a, 0).unwrap();
        assert_eq!(channel.submit_read(1, 1, a, 0), Err(BlockError::BufferBusy));
        assert_eq!(channel.unregister_buffer(a), Err(BlockError::BufferBusy));
        channel.cancel(request).unwrap();
        assert_eq!(channel.poll().unwrap().result, Err(BlockError::Cancelled));

        assert_eq!(channel.quiesce(), Ok(()));
        assert_eq!(channel.state(), ChannelState::Revoked);
        assert_eq!(channel.submit_flush(), Err(BlockError::Revoked));
    }

    /// Latency is booked only when the backend publishes a clock, at the
    /// depth each command was submitted behind, into the shared base-2
    /// histogram.
    #[test]
    fn latency_is_measured_at_depth_only_when_the_backend_has_a_clock() {
        // No clock: nothing measured, and that is a posture, not a zero.
        let mut blind = channel();
        let id = blind.submit_flush().unwrap();
        assert_eq!(blind.poll().unwrap().request, id);
        assert_eq!(blind.stats().latency, LatencyBooks::EMPTY);
        assert!(!blind.stats().latency.snapshot(1).measured());

        // A clock that reads 1 µs now and moves 3 µs per poll: three
        // commands submitted back to back complete 3, 6 and 9 µs later.
        let mut channel = channel();
        channel
            .control(|device| device.set_clock(1_000, 3_000))
            .unwrap();
        let mut ids = [RequestId(0); 3];
        for id in &mut ids {
            *id = channel.submit_flush().unwrap();
        }
        for expected in ids {
            assert_eq!(channel.poll().unwrap().request, expected);
        }
        let books = channel.stats().latency;
        assert_eq!(books.samples, 3);
        assert_eq!(books.sum_ns, 18_000);
        assert_eq!(books.max_ns, 9_000);
        // Submitted behind 0, 1 and 2 commands respectively.
        assert_eq!(books.depth_sum, 3);
        type L = kore_abi::monitor::FilesystemLatencySnapshot;
        // 3 µs and 6 µs share the <8 µs bucket; 9 µs is the next one up.
        assert_eq!(books.buckets[L::bucket_index(3_000)], 2);
        assert_eq!(books.buckets[L::bucket_index(9_000)], 1);
        assert_eq!(books.buckets.iter().map(|b| u64::from(*b)).sum::<u64>(), 3);
        let snapshot = books.snapshot(channel.stats().max_outstanding);
        assert_eq!(snapshot.max_outstanding(), 3);
        assert_eq!(snapshot.mean_ns(), Some(6_000));
        assert_eq!(snapshot.mean_depth_milli(), Some(1_000));
        assert_eq!(snapshot.percentile_upper_ns(500), Some(8_192));
        assert_eq!(snapshot.percentile_upper_ns(990), Some(32_768));
    }

    #[test]
    fn stale_buffer_and_late_completion_never_alias_new_authority() {
        let mut channel = channel();
        let old = channel.register_buffer(512).unwrap();
        channel.unregister_buffer(old).unwrap();
        let fresh = channel.register_buffer(512).unwrap();
        assert_ne!(old.to_bits(), fresh.to_bits());
        assert_eq!(
            channel.write_buffer(old, 0, b"x"),
            Err(BlockError::UnknownBuffer)
        );
        channel
            .control(|device| device.inject_completion(999, Ok(())).unwrap())
            .unwrap();
        assert_eq!(
            channel.poll().unwrap().result,
            Err(BlockError::StaleCompletion)
        );
        assert_eq!(channel.late_completions(), 1);
    }

    #[test]
    fn queue_wrap_stats_irq_pacing_read_only_loss_and_draining_are_typed() {
        let mut channel = channel();
        let mut ids = [RequestId(0); 4];
        for id in &mut ids {
            *id = channel.submit_flush().unwrap();
        }
        assert_eq!(channel.submit_flush(), Err(BlockError::Backpressure));
        for expected in ids {
            assert_eq!(channel.poll().unwrap().request, expected);
        }
        for _ in 0..20 {
            let id = channel.submit_flush().unwrap();
            assert_eq!(channel.poll().unwrap().request, id);
        }
        assert_eq!(channel.stats().max_outstanding, 4);
        assert_eq!(channel.stats().backpressure, 1);

        let live = channel.submit_flush().unwrap();
        assert_eq!(channel.quiesce(), Err(BlockError::Draining));
        assert_eq!(channel.poll().unwrap().request, live);
        assert_eq!(channel.quiesce(), Ok(()));

        let mut device = Device::new(512, 4).unwrap();
        device.set_read_only(true);
        let mut read_only: Channel = BlockChannel::new(device, CompletionMode::Interrupt).unwrap();
        let buffer = read_only.register_buffer(512).unwrap();
        assert_eq!(
            read_only.submit_write(0, 1, buffer, 0),
            Err(BlockError::ReadOnly)
        );
        // An empty poll against an idle queue is not a wait: nothing is
        // outstanding, so the pacing seam is never reached. Counting it as
        // one is what made a busy-spinning machine read as healthy.
        assert!(read_only.poll().is_none());
        assert_eq!(read_only.stats().poll_calls, 1);
        assert_eq!(read_only.stats().device_waits, 0);
        assert!(read_only.wait().is_none());
        assert_eq!(read_only.stats().device_waits, 0);
        read_only.mark_lost();
        assert_eq!(
            read_only.submit_read(0, 1, buffer, 0),
            Err(BlockError::Lost)
        );
    }
}
