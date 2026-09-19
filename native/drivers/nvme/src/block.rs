//! Hardware-neutral direct block-channel adapter over one NVMe qpair.
//!
//! This is the only layer where [`kore_driver::block`] vocabulary meets NVMe
//! namespace commands, PRPs, IOVAs, and completion status.  Consumers above
//! it own a `BlockChannel<NvmeBlockDevice<...>>` and cannot name those device
//! details.

use core::mem::size_of;

use kore_driver::block::{
    BlockBuffer, BlockError, BlockGeometry, BlockHealth, BlockIoHints, CompletionPacer,
    CountedPacer, DeviceCommand, DeviceCompletion, DirectBlockDevice,
};
use kore_driver::seam::{Dma, DmaPool, Regs};

use crate::{
    Controller, Error, Namespace, Qpair,
    commands::DsmRange,
    namespace::MAX_NLB_BLOCKS,
    prp::{ListPage, max_single_list_transfer},
};

/// Identify Controller ONCS bit 2 (NVM Command Set: Dataset Management).
const ONCS_DATASET_MANAGEMENT: u16 = 1 << 2;

/// One registered DMA buffer plus an optional private PRP-list page. Buffers
/// spanning at most two controller pages fit directly in PRP1/PRP2; larger
/// buffers keep a private list so a later submission cannot rewrite entries
/// the device may still be consuming.
pub struct NvmeBuffer<B: Dma> {
    data: B,
    list: Option<B>,
    len: usize,
}

impl<B: Dma> BlockBuffer for NvmeBuffer<B> {
    fn len(&self) -> usize {
        self.len
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: `len` was checked against the owned DMA allocation.
        unsafe { core::slice::from_raw_parts(self.data.cpu(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above; the channel prevents CPU access while busy.
        unsafe { core::slice::from_raw_parts_mut(self.data.cpu(), self.len) }
    }
}

/// One passive NVMe data queue. It owns the controller parent so the qpair's
/// BAR/DMA lifetime cannot outlive the authority it derives from.
pub struct NvmeBlockDevice<R: Regs, P: DmaPool> {
    controller: Controller<R, P>,
    namespace: Namespace,
    qpair: Qpair<P::Buf, P::Buf>,
    geometry: BlockGeometry,
    dsm: P::Buf,
    /// Reused SMART/Health DMA buffer; status probes never grow the DMA pool.
    health: P::Buf,
    dsm_cookie: Option<u64>,
    shutdown_polls: u32,
    /// Caller-selected relaxation for SMART and lifecycle command waits.
    /// The driver core cannot choose a runtime address-monitor implementation.
    lifecycle_pause: fn(),
    quiesced: bool,
}

impl<R: Regs, P: DmaPool> NvmeBlockDevice<R, P> {
    /// Finish identify/queue provisioning and bind one direct channel.
    pub fn provision(
        controller: Controller<R, P>,
        qid: u16,
        depth: u16,
        vector: Option<u16>,
        max_polls: u32,
        pause: fn(),
    ) -> Result<Self, BlockError> {
        let mut pacer = CountedPacer::new(max_polls, pause);
        Self::provision_paced(controller, qid, depth, vector, &mut pacer, pause, max_polls)
    }

    /// Finish controller provisioning under one caller-owned pacing budget.
    /// This is the production boot path: all admin commands share the same
    /// absolute deadline instead of each receiving a fresh poll count.
    pub fn provision_paced(
        mut controller: Controller<R, P>,
        qid: u16,
        depth: u16,
        vector: Option<u16>,
        pacer: &mut dyn CompletionPacer,
        lifecycle_pause: fn(),
        shutdown_polls: u32,
    ) -> Result<Self, BlockError> {
        let info = controller.identify_paced(pacer).map_err(map_submit_error)?;
        let namespace = controller.namespace();
        let wanted = qid.max(1);
        let (sq, cq) = controller
            .set_num_queues_paced(wanted, pacer)
            .map_err(map_submit_error)?;
        if sq < wanted || cq < wanted {
            return Err(BlockError::Backpressure);
        }
        let qpair = controller
            .create_io_qpair_paced(qid, depth, vector, pacer)
            .map_err(map_submit_error)?;
        let dsm = controller
            .pool()
            .alloc(size_of::<DsmRange>())
            .ok_or(BlockError::Backpressure)?;
        let health = controller
            .pool()
            .alloc(size_of::<crate::commands::SmartLog>())
            .ok_or(BlockError::Backpressure)?;

        let prp_ceiling = max_single_list_transfer(controller.page());
        if info.block_size == 0 || info.block_count == 0 {
            return Err(BlockError::Geometry);
        }
        let mdts = if info.mdts_bytes == 0 {
            prp_ceiling
        } else {
            prp_ceiling.min(info.mdts_bytes)
        };
        let max_transfer_blocks =
            (mdts / info.block_size as u64).min(u64::from(MAX_NLB_BLOCKS)) as u32;
        let geometry = BlockGeometry {
            block_size: info.block_size,
            block_count: info.block_count,
            max_transfer_blocks,
            // The core does not yet resolve namespace NAWUN's controller
            // fallback and boundary rules; one logical block is therefore
            // the only honest guaranteed atomic unit.
            atomic_write_blocks: 1,
            flush: true,
            discard: info.oncs & ONCS_DATASET_MANAGEMENT != 0,
            read_only: false,
            io_hints: BlockIoHints {
                preferred_write_blocks: info.preferred_write_blocks,
                write_alignment_blocks: info.write_alignment_blocks,
                optimal_write_blocks: info.optimal_write_blocks,
                optimal_boundary_blocks: info.optimal_boundary_blocks,
                supported_block_sizes: info.supported_block_sizes,
            },
        }
        .validate()?;

        Ok(Self {
            controller,
            namespace,
            qpair,
            geometry,
            dsm,
            health,
            dsm_cookie: None,
            shutdown_polls,
            lifecycle_pause,
            quiesced: false,
        })
    }

    pub fn qpair_stats(&self) -> crate::QpairStats {
        self.qpair.stats()
    }

    /// **The word a completion's arrival stores to** — this channel's data
    /// queue, forwarded from [`Qpair::completion_word`].
    ///
    /// For a caller composing an address-monitor tier under its completion
    /// wait: snapshot the word, sleep while it is unchanged, then poll. The
    /// wait itself is not offered here, because the mechanism lives in
    /// `kore-rt` and this crate's core must not depend on it — see
    /// [`Qpair::completion_word`] for why the split falls exactly there.
    ///
    /// `None` on a ring base this crate cannot prove `u32`-aligned; the caller
    /// keeps its ordinary pacing.
    pub fn completion_word(&self) -> Option<&core::sync::atomic::AtomicU32> {
        self.qpair.completion_word()
    }

    /// Return the capability-backed register and DMA seams after the qpair
    /// has been quiesced. The common driver host uses this to retire the
    /// ordinary BAR/IOMMU/region descendants in dependency order.
    pub fn into_seam(self) -> (R, P) {
        let Self {
            controller,
            qpair,
            dsm,
            health,
            ..
        } = self;
        drop(qpair);
        drop(dsm);
        drop(health);
        controller.into_seam()
    }

    fn list_page<'a>(buffer: &'a mut NvmeBuffer<P::Buf>) -> Option<ListPage<'a>> {
        buffer.list.as_mut().map(|list| {
            let count = list.len() / size_of::<u64>();
            // SAFETY: Dma allocations are page-aligned and `count` covers
            // only the owned list allocation.
            let entries =
                unsafe { core::slice::from_raw_parts_mut(list.cpu().cast::<u64>(), count) };
            ListPage {
                iova: list.iova(),
                entries,
            }
        })
    }

    fn build_sqe(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut NvmeBuffer<P::Buf>>,
        cookie: u64,
    ) -> Result<crate::commands::NvmeSqe, BlockError> {
        Ok(match command {
            DeviceCommand::Read {
                lba,
                blocks,
                buffer_offset,
            } => {
                let buffer = buffer.ok_or(BlockError::UnknownBuffer)?;
                let iova = kore_abi::device::IovaAddr(
                    buffer
                        .data
                        .iova()
                        .0
                        .checked_add(buffer_offset as u64)
                        .ok_or(BlockError::Range)?,
                );
                let list = Self::list_page(buffer);
                self.namespace
                    .read(lba, blocks, iova, list)
                    .map_err(map_submit_error)?
            }
            DeviceCommand::Write {
                lba,
                blocks,
                buffer_offset,
            } => {
                let buffer = buffer.ok_or(BlockError::UnknownBuffer)?;
                let iova = kore_abi::device::IovaAddr(
                    buffer
                        .data
                        .iova()
                        .0
                        .checked_add(buffer_offset as u64)
                        .ok_or(BlockError::Range)?,
                );
                let list = Self::list_page(buffer);
                self.namespace
                    .write(lba, blocks, iova, list)
                    .map_err(map_submit_error)?
            }
            DeviceCommand::Flush => self.namespace.flush(),
            DeviceCommand::Discard { lba, blocks } => {
                if self.dsm_cookie.is_some() {
                    return Err(BlockError::Backpressure);
                }
                // SAFETY: `dsm` owns at least one complete DsmRange.
                unsafe {
                    self.dsm
                        .cpu()
                        .cast::<DsmRange>()
                        .write(DsmRange::new(lba, blocks))
                };
                self.dsm_cookie = Some(cookie);
                self.namespace
                    .deallocate(self.dsm.iova(), 1)
                    .map_err(map_submit_error)?
            }
        })
    }

    fn device_completion(&mut self, completion: crate::Completion) -> DeviceCompletion {
        if self.dsm_cookie == Some(completion.cookie) {
            self.dsm_cookie = None;
        }
        DeviceCompletion {
            cookie: completion.cookie,
            result: if completion.is_success() {
                Ok(())
            } else {
                Err(map_status(completion.cqe.status_code_type()))
            },
        }
    }
}

impl<R: Regs, P: DmaPool> DirectBlockDevice for NvmeBlockDevice<R, P> {
    type Buffer = NvmeBuffer<P::Buf>;

    /// A submitted command advances in the controller's own queues; there is
    /// no correlated reply for this side to await, which is why `progress`
    /// stays the ready default.  Sharing one controller between several
    /// designated regions relies on exactly that.
    const INDEPENDENT_PROGRESS: bool = true;

    /// One `CSTS` read — the controller's own verdict on itself.
    ///
    /// This is why a late completion need not be treated as a dead device:
    /// NVMe publishes `CSTS.CFS` (Controller Fatal Status) and `CSTS.RDY` in
    /// a BAR register, so liveness is answerable *without* the command path
    /// that is in doubt. `health()` cannot serve here — fetching a SMART log
    /// is itself a command, and would be exactly as stuck.
    ///
    /// The driver's bring-up waits already use `CSTS.CFS` as their fault
    /// predicate; this is the same predicate on the data path.
    fn liveness(&mut self) -> kore_driver::block::Liveness {
        use kore_driver::block::Liveness;
        let csts = self.controller.regs().read32(crate::regs::REG_CSTS);
        // An all-ones read is a vanished device (a PCIe link down reads as
        // 0xFFFF_FFFF), not a controller reporting every bit set.
        if csts == u32::MAX {
            return Liveness::Failed;
        }
        if csts & crate::regs::CSTS_CFS != 0 {
            return Liveness::Failed;
        }
        if csts & crate::regs::CSTS_RDY == 0 {
            // Ready dropped out from under a provisioned queue: the
            // controller reset or is resetting, and nothing outstanding will
            // complete.
            return Liveness::Failed;
        }
        Liveness::Healthy
    }

    fn geometry(&self) -> BlockGeometry {
        self.geometry
    }

    fn max_buffer_bytes(&self) -> usize {
        usize::try_from(
            u64::from(self.geometry.max_transfer_blocks)
                .saturating_mul(u64::from(self.geometry.block_size)),
        )
        .unwrap_or(usize::MAX)
    }

    fn health(&mut self) -> Result<BlockHealth, BlockError> {
        let log = self
            .controller
            .smart_into(&mut self.health, self.shutdown_polls, self.lifecycle_pause)
            .map_err(map_submit_error)?;
        Ok(BlockHealth {
            critical_warning: log.critical_warning,
            temperature_celsius: i16::try_from(log.temp_celsius()).unwrap_or(i16::MAX),
            available_spare_percent: log.avail_spare,
            spare_threshold_percent: log.spare_threshold,
            percentage_used: log.percentage_used,
            power_cycles: crate::commands::SmartLog::counter(&log.power_cycles),
            power_on_hours: crate::commands::SmartLog::counter(&log.power_on_hours),
            unsafe_shutdowns: crate::commands::SmartLog::counter(&log.unsafe_shutdowns),
            media_errors: crate::commands::SmartLog::counter(&log.media_errors),
        })
    }

    fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError> {
        if bytes == 0
            || bytes as u64
                > self.geometry.max_transfer_blocks as u64 * self.geometry.block_size as u64
        {
            return Err(BlockError::TooLarge);
        }
        let data = self
            .controller
            .pool()
            .alloc_aligned(bytes, self.controller.page().bytes() as usize)
            .ok_or(BlockError::Backpressure)?;
        if data.len() < bytes {
            return Err(BlockError::Integrity);
        }
        let page_bytes = self.controller.page().bytes() as usize;
        let list = if bytes > page_bytes.saturating_mul(2) {
            Some(
                self.controller
                    .pool()
                    .alloc_aligned(page_bytes, page_bytes)
                    .ok_or(BlockError::Backpressure)?,
            )
        } else {
            None
        };
        Ok(NvmeBuffer {
            data,
            list,
            len: bytes,
        })
    }

    fn submit(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        let sqe = self.build_sqe(command, buffer, cookie)?;
        if let Err(error) = self.qpair.submit_one(sqe, cookie, self.controller.regs()) {
            if self.dsm_cookie == Some(cookie) {
                self.dsm_cookie = None;
            }
            return Err(match error {
                crate::QpairError::Full => BlockError::Backpressure,
                crate::QpairError::BadDepth => BlockError::Geometry,
            });
        }
        Ok(())
    }

    fn stage(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        let sqe = self.build_sqe(command, buffer, cookie)?;
        if let Err(error) = self.qpair.submit(sqe, cookie) {
            if self.dsm_cookie == Some(cookie) {
                self.dsm_cookie = None;
            }
            return Err(map_qpair_error(error));
        }
        Ok(())
    }

    fn publish(&mut self) {
        self.qpair.ring(self.controller.regs());
    }

    fn poll(&mut self) -> Option<DeviceCompletion> {
        let completion = self.qpair.poll(self.controller.regs())?;
        Some(self.device_completion(completion))
    }

    fn poll_batch(&mut self, out: &mut [Option<DeviceCompletion>]) -> usize {
        const MAX_BATCH: usize = kore_driver::block::DEFAULT_OUTSTANDING;
        let limit = out.len().min(MAX_BATCH);
        let mut raw = [None; MAX_BATCH];
        let count = self
            .qpair
            .poll_batch(self.controller.regs(), &mut raw[..limit]);
        for index in 0..count {
            out[index] = raw[index]
                .take()
                .map(|completion| self.device_completion(completion));
        }
        count
    }

    fn quiesce(&mut self) -> Result<(), BlockError> {
        if self.quiesced {
            return Ok(());
        }
        let stats = self.qpair.stats();
        if stats.submitted != stats.completed {
            return Err(BlockError::Draining);
        }
        self.controller
            .shutdown(self.shutdown_polls, self.lifecycle_pause)
            .map_err(map_submit_error)?;
        self.quiesced = true;
        Ok(())
    }
}

fn map_qpair_error(error: crate::QpairError) -> BlockError {
    match error {
        crate::QpairError::Full => BlockError::Backpressure,
        crate::QpairError::BadDepth => BlockError::Geometry,
    }
}

fn map_submit_error(error: Error) -> BlockError {
    match error {
        Error::DmaAlloc | Error::Qpair(crate::QpairError::Full) => BlockError::Backpressure,
        Error::NotReady | Error::Timeout => BlockError::Transport,
        Error::Status { sct, .. } => map_status(sct),
        Error::Qpair(crate::QpairError::BadDepth) | Error::Geometry => BlockError::Geometry,
        Error::Prp(crate::prp::PrpError::ListNeeded | crate::prp::PrpError::TooLarge)
        | Error::ExceedsMdts => BlockError::TooLarge,
        Error::Prp(crate::prp::PrpError::NotEnoughPages) => BlockError::Integrity,
    }
}

fn map_status(status_code_type: u8) -> BlockError {
    match status_code_type {
        0 => BlockError::Transport,
        2 => BlockError::Media,
        3 => BlockError::Integrity,
        _ => BlockError::Transport,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use core::cell::Cell;

    use crate::mock::{MOCK_BLOCK_SIZE, MockBuf, MockNvmeDevice, MockPool};
    use kore_driver::block::{BlockChannel, CompletionMode};

    /// Production [`CapPool`](kore_driver::host::CapPool) geometry over the
    /// mock's host-pointer DMA buffers: one 4 KiB-rounded bump cursor with a
    /// fixed limit and stronger alignment charged as real skipped capacity.
    struct BudgetPool {
        inner: MockPool,
        cursor: Rc<Cell<usize>>,
        limit: usize,
    }

    impl BudgetPool {
        fn new(limit: usize) -> (Self, Rc<Cell<usize>>) {
            let cursor = Rc::new(Cell::new(0));
            (
                Self {
                    inner: MockPool,
                    cursor: Rc::clone(&cursor),
                    limit,
                },
                cursor,
            )
        }

        fn allocate(&self, bytes: usize, alignment: usize) -> Option<MockBuf> {
            let alignment = alignment.max(4096);
            if !alignment.is_power_of_two() {
                return None;
            }
            let size = bytes.max(1).checked_add(4095)? & !4095;
            let mask = alignment - 1;
            let start = self.cursor.get().checked_add(mask)? & !mask;
            let end = start.checked_add(size)?;
            if end > self.limit {
                return None;
            }
            let buffer = self.inner.alloc_aligned(bytes, alignment)?;
            self.cursor.set(end);
            Some(buffer)
        }
    }

    impl DmaPool for BudgetPool {
        type Buf = MockBuf;

        fn alloc(&self, bytes: usize) -> Option<Self::Buf> {
            self.allocate(bytes, 4096)
        }

        fn alloc_aligned(&self, bytes: usize, alignment: usize) -> Option<Self::Buf> {
            self.allocate(bytes, alignment)
        }
    }

    #[test]
    fn hardware_neutral_channel_drives_the_real_nvme_core() {
        let dev = MockNvmeDevice::new();
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let mut backend = NvmeBlockDevice::provision(ctrl, 1, 8, Some(0), 16, || {}).unwrap();
        assert!(backend.geometry().discard);
        let health = backend.health().expect("typed SMART health sample");
        assert_eq!(health.temperature_celsius, 36);
        assert_eq!(health.percentage_used, 3);
        assert_eq!(health.power_cycles, 7);
        assert_eq!(health.media_errors, 0);
        // A second sample reuses the backend's dedicated DMA buffer.
        assert_eq!(backend.health().unwrap(), health);
        let mut channel: BlockChannel<_, 3, 8> =
            BlockChannel::new(backend, CompletionMode::Interrupt).unwrap();
        assert_eq!(channel.mode(), CompletionMode::Interrupt);
        let a = channel.register_buffer(MOCK_BLOCK_SIZE as usize).unwrap();
        let b = channel.register_buffer(MOCK_BLOCK_SIZE as usize).unwrap();

        // A failed later stage cannot leave the accepted prefix invisible:
        // dropping the affine batch publishes exactly that prefix.
        let before = dev.io_sq_doorbells();
        let staged = {
            let mut batch = channel.batch();
            let staged = batch.stage_read(0, 1, a, 0).unwrap();
            assert_eq!(batch.stage_read(1, 1, a, 0), Err(BlockError::BufferBusy));
            staged
        };
        assert_eq!(dev.io_sq_doorbells() - before, 1);
        assert_eq!(channel.poll().unwrap().request, staged);

        channel
            .write_buffer(a, 0, &[0x5a; MOCK_BLOCK_SIZE as usize])
            .unwrap();

        // Two distinct buffers permit two outstanding commands on one owned
        // qpair; cookies return to the right request.
        let before = dev.io_sq_doorbells();
        let (write, read) = {
            let mut batch = channel.batch();
            let write = batch.stage_write(20, 1, a, 0).unwrap();
            let read = batch.stage_read(3, 1, b, 0).unwrap();
            batch.publish();
            (write, read)
        };
        assert_eq!(dev.io_sq_doorbells() - before, 1, "one SQ doorbell");
        assert_eq!(channel.outstanding(), 2);
        let cq_before = dev.io_cq_doorbells();
        let [Some(c1), Some(c2)] = channel.poll_batch::<2>() else {
            panic!("both mock completions are ready")
        };
        assert_eq!(dev.io_cq_doorbells() - cq_before, 1, "one CQ doorbell");
        assert_eq!([c1.request, c2.request], [write, read]);
        assert!(c1.result.is_ok() && c2.result.is_ok());
        assert!(dev.disk_snapshot(20, 1).iter().all(|b| *b == 0x5a));

        let flush = channel.submit_flush().unwrap();
        assert_eq!(channel.poll().unwrap().request, flush);
        let discard = channel.submit_discard(20, 1).unwrap();
        assert_eq!(channel.poll().unwrap().request, discard);
        assert!(dev.disk_snapshot(20, 1).iter().all(|b| *b == 0));
        channel.quiesce().unwrap();
    }

    #[test]
    fn larger_controller_page_aligns_data_and_prp_list_buffers() {
        let dev = MockNvmeDevice::new().with_mps(1).unwrap();
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let mut backend = NvmeBlockDevice::provision(ctrl, 1, 8, None, 16, || {}).unwrap();
        let page_bytes = 2 * MOCK_BLOCK_SIZE as usize;
        let mut buffer = backend.allocate_buffer(2 * page_bytes).unwrap();
        assert_eq!(buffer.data.iova().0 & 0x1fff, 0);
        assert!(buffer.list.is_none(), "two pages fit directly in PRP1/PRP2");
        let listed = backend.allocate_buffer(2 * page_bytes + 1).unwrap();
        assert_eq!(
            listed.list.as_ref().unwrap().iova().0 & 0x1fff,
            0,
            "larger buffers own an aligned private PRP list"
        );

        backend
            .submit(
                DeviceCommand::Read {
                    lba: 3,
                    blocks: 4,
                    buffer_offset: 0,
                },
                Some(&mut buffer),
                7,
            )
            .unwrap();
        let completion = backend.poll().unwrap();
        assert_eq!(completion.cookie, 7);
        assert_eq!(completion.result, Ok(()));
        assert_eq!(buffer.bytes(), dev.disk_snapshot(3, 4));
    }

    #[test]
    fn mps_256k_and_four_fs_buffers_fit_the_exact_2mib_budget() {
        const DMA_BUDGET: usize = 2 * 1024 * 1024;
        const FS_BUFFER: usize = 256 * 1024;

        let dev = MockNvmeDevice::new().with_mps(6).unwrap();
        let (pool, cursor) = BudgetPool::new(DMA_BUDGET);
        let ctrl = Controller::bring_up(dev, pool, 16, || {}).unwrap();
        let mut backend = NvmeBlockDevice::provision(ctrl, 1, 32, None, 16, || {}).unwrap();
        let buffers = [
            backend.allocate_buffer(FS_BUFFER).unwrap(),
            backend.allocate_buffer(FS_BUFFER).unwrap(),
            backend.allocate_buffer(FS_BUFFER).unwrap(),
            backend.allocate_buffer(FS_BUFFER).unwrap(),
        ];

        assert!(buffers.iter().all(|buffer| buffer.list.is_none()));
        assert_eq!(cursor.get(), DMA_BUDGET, "the fourth buffer ends at 2 MiB");
    }
}
