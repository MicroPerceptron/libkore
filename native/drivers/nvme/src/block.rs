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
    commands::{DsmRange, HmbDescriptor},
    controller::{DeviceInfo, HMB_MAX_CHUNKS},
    namespace::MAX_NLB_BLOCKS,
    prp::{ListPage, MemPage, max_single_list_transfer},
};

/// Identify Controller ONCS bit 2 (NVM Command Set: Dataset Management).
const ONCS_DATASET_MANAGEMENT: u16 = 1 << 2;

/// PRP-list pages shared by commands into **adopted** buffers — the
/// caller-funded streaming targets ([`NvmeBlockDevice::adopt_buffer`]).
///
/// A registered buffer keeps a private list page because it has at most one
/// command in flight; an adopted target is multi-GiB and takes many commands
/// at once, each needing its own list for as long as the device may read it.
/// So adopted transfers borrow a page from this pool at submission and return
/// it at their completion: the pool is sized by *queue occupancy*, never by
/// target size, which is what keeps a 16 GiB target from costing 32 MiB of
/// this row's DMA pool in list pages.  One hundred and twenty-eight covers a
/// ring streaming adopted commands at full depth on one queue (the
/// transport engine's class budgets bound the depth), and exhaustion is
/// ordinary backpressure the engine retries.
pub const ADOPTED_LIST_PAGES: usize = 128;

/// Polls an Abort's admin completion may take before the abort counts as
/// refused — bounded, because it runs on the engine's deadline scan.
const ABORT_POLLS: u32 = 1 << 20;

/// One registered DMA buffer plus an optional private PRP-list page. Buffers
/// spanning at most two controller pages fit directly in PRP1/PRP2; larger
/// buffers keep a private list so a later submission cannot rewrite entries
/// the device may still be consuming.
pub struct NvmeBuffer<B: Dma> {
    data: B,
    list: Option<B>,
    len: usize,
}

impl<B: Dma> NvmeBuffer<B> {
    /// Wrap an allocation that was never adopted, so a refused adoption can
    /// still travel the one release path. It carries no list page and must
    /// not be submitted against.
    pub fn from_data(data: B) -> Self {
        let len = data.len();
        Self {
            data,
            list: None,
            len,
        }
    }

    /// Recover the DMA allocation from an idle buffer — how an adopted
    /// target's device mapping returns to the owner that must release it
    /// quiescently.  A private list page, if any, is dropped with the
    /// wrapper: it is this row's own pool allocation.
    pub fn into_data(self) -> B {
        self.data
    }
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

/// Each submission queue's arbitration priority, by index: urgent, then
/// high, then medium (with the three-class policy: Demand, Commit, Stream).
/// Honoured only under weighted round robin
/// ([`Controller::weighted_arbitration`]); plain round robin ignores it, and
/// a pair's one queue is urgent exactly as every queue was before
/// priorities existed.
const PRIORITIES: [crate::commands::SqPriority; 4] = [
    crate::commands::SqPriority::Urgent,
    crate::commands::SqPriority::High,
    crate::commands::SqPriority::Medium,
    crate::commands::SqPriority::Low,
];

/// One passive NVMe data queue. It owns the controller parent so the qpair's
/// BAR/DMA lifetime cannot outlive the authority it derives from.
pub struct NvmeBlockDevice<R: Regs, P: DmaPool, const SQS: usize = 1> {
    controller: Controller<R, P>,
    namespace: Namespace,
    qpair: Qpair<P::Buf, P::Buf, SQS>,
    geometry: BlockGeometry,
    dsm: P::Buf,
    /// Reused SMART/Health DMA buffer; status probes never grow the DMA pool.
    health: P::Buf,
    dsm_cookie: Option<u64>,
    /// The shared list-page pool for adopted targets, allocated on the first
    /// adoption (a row that never adopts pays nothing), and the cookie each
    /// page is lent to while its command is in flight.
    adopted_lists: [Option<P::Buf>; ADOPTED_LIST_PAGES],
    adopted_owners: [Option<u64>; ADOPTED_LIST_PAGES],
    shutdown_polls: u32,
    /// Caller-selected relaxation for SMART and lifecycle command waits.
    /// The driver core cannot choose a runtime address-monitor implementation.
    lifecycle_pause: fn(),
    quiesced: bool,
    /// What a controller reset must replay to bring the I/O queues back:
    /// the `(submission, completion)` queue counts asked for (Number of
    /// Queues does not survive a reset) and the queue group's vector.
    queue_request: (u16, u16),
    vector: Option<u16>,
    /// Controller resets this device has been through.
    resets: u32,
}

/// How much host memory to offer the controller for its mapping tables: the
/// least of what it asks for (Identify HMPRE) and the consumer's allowance,
/// rounded down to whole controller pages — and nothing at all when the
/// controller uses none, or when what could be offered is below the floors
/// it advertises (a buffer it may refuse is not worth the round trip).
pub fn host_memory_buffer_bytes(info: &DeviceInfo, allowance: u64, page: MemPage) -> u64 {
    if info.hmb_preferred_bytes == 0 || allowance == 0 {
        return 0;
    }
    let page = page.bytes() as u64;
    let bytes = info.hmb_preferred_bytes.min(allowance) / page * page;
    if bytes == 0 || bytes < info.hmb_min_bytes || bytes < info.hmb_min_descriptor_bytes {
        return 0;
    }
    bytes
}

impl<R: Regs, P: DmaPool> NvmeBlockDevice<R, P, 1> {
    /// Finish identify/queue provisioning and bind one direct channel.
    /// `hmb_max_bytes` is the most host memory the controller may be given
    /// for its mapping tables (0 offers none).
    pub fn provision(
        controller: Controller<R, P>,
        qid: u16,
        depth: u16,
        vector: Option<u16>,
        hmb_max_bytes: u64,
        max_polls: u32,
        pause: fn(),
    ) -> Result<Self, BlockError> {
        let mut pacer = CountedPacer::new(max_polls, pause);
        Self::provision_paced(
            controller,
            qid,
            depth,
            vector,
            hmb_max_bytes,
            &mut pacer,
            pause,
            max_polls,
        )
    }

    /// Finish controller provisioning under one caller-owned pacing budget.
    /// This is the production boot path: all admin commands share the same
    /// absolute deadline instead of each receiving a fresh poll count.
    #[allow(clippy::too_many_arguments)]
    pub fn provision_paced(
        controller: Controller<R, P>,
        qid: u16,
        depth: u16,
        vector: Option<u16>,
        hmb_max_bytes: u64,
        pacer: &mut dyn CompletionPacer,
        lifecycle_pause: fn(),
        shutdown_polls: u32,
    ) -> Result<Self, BlockError> {
        Self::provision_queues_paced(
            controller,
            qid,
            depth,
            vector,
            hmb_max_bytes,
            pacer,
            lifecycle_pause,
            shutdown_polls,
        )
    }
}

impl<R: Regs, P: DmaPool, const SQS: usize> NvmeBlockDevice<R, P, SQS> {
    /// [`provision_paced`](NvmeBlockDevice::provision_paced) for a device of
    /// `SQS` submission queues over one completion queue (ids from `qid`),
    /// each `depth` deep — one per service class.
    #[allow(clippy::too_many_arguments)]
    pub fn provision_queues_paced(
        mut controller: Controller<R, P>,
        qid: u16,
        depth: u16,
        vector: Option<u16>,
        hmb_max_bytes: u64,
        pacer: &mut dyn CompletionPacer,
        lifecycle_pause: fn(),
        shutdown_polls: u32,
    ) -> Result<Self, BlockError> {
        let info = controller.identify_paced(pacer).map_err(map_submit_error)?;
        let namespace = controller.namespace();
        let wanted = qid.max(1);
        // A pair asks for `qid` queues of each kind, as it always did. A
        // group asks for one submission queue per class beside its one
        // completion queue, and runs with however many it is granted (at
        // least one) — fewer queues fold classes together, never refuse.
        let mut queue_request = (wanted, wanted);
        let live = if SQS == 1 {
            let (sq, cq) = controller
                .set_num_queues_paced(wanted, pacer)
                .map_err(map_submit_error)?;
            if sq < wanted || cq < wanted {
                return Err(BlockError::Backpressure);
            }
            1
        } else {
            let want_sqs = u16::try_from(SQS)
                .ok()
                .and_then(|n| n.checked_add(wanted - 1))
                .ok_or(BlockError::Geometry)?;
            queue_request = (want_sqs, wanted);
            let (sq, cq) = controller
                .set_queue_counts_paced(want_sqs, wanted, pacer)
                .map_err(map_submit_error)?;
            if cq < wanted || sq < wanted {
                return Err(BlockError::Backpressure);
            }
            usize::from(sq - (wanted - 1)).min(SQS)
        };
        // The host memory buffer, before the I/O queues exist (the order
        // Linux keeps).  One page-aligned allocation is one descriptor: the
        // pool's IOVA window is contiguous by construction, whatever the
        // frames beneath it.  A controller that uses none, an allowance
        // below its minimum, a pool that cannot carry it, or a refusal all
        // leave the device running WITHOUT one — a slower filesystem, never
        // a missing one — and the enable takes its own counted budget so a
        // controller that is slow to take 64 MiB cannot spend the shared
        // bring-up deadline the queue creation below still needs.  The
        // buffer is disabled again by `Controller::shutdown`, which the
        // quiesce hook reaches while bus mastering is still open; the
        // revoke path that skips quiesce is contained by the host closing
        // bus mastering (or binding the requester to Abort) before the
        // frames are reaped, so the controller can never reach memory it
        // no longer owns.
        let hmb_bytes = host_memory_buffer_bytes(&info, hmb_max_bytes, controller.page());
        if hmb_bytes != 0 {
            let page = controller.page().bytes() as usize;
            let list = controller
                .pool()
                .alloc(size_of::<HmbDescriptor>() * HMB_MAX_CHUNKS);
            let chunk = controller.pool().alloc_aligned(hmb_bytes as usize, page);
            if let (Some(list), Some(chunk)) = (list, chunk) {
                let mut chunks = [const { None }; HMB_MAX_CHUNKS];
                chunks[0] = Some(chunk);
                let mut hmb_pacer = CountedPacer::new(shutdown_polls, lifecycle_pause);
                let _ = controller.enable_host_memory_buffer(list, chunks, &mut hmb_pacer);
            }
        }
        let qpair = controller
            .create_io_queue_group_paced::<SQS>(
                qid,
                qid,
                live,
                depth,
                core::array::from_fn(|index| PRIORITIES[index.min(PRIORITIES.len() - 1)]),
                vector,
                pacer,
            )
            .map_err(map_submit_error)?;
        // An arbitration burst of one command per queue per round keeps a
        // deep bulk queue from taking a whole burst ahead of a demand read.
        // Under weighted round robin the urgent queue is served first and
        // the weights split the rest: high four turns to medium's one, so
        // durability writes outpace bulk streams. Best effort: a controller
        // that refuses the feature keeps its default burst and weights.
        if SQS > 1 {
            let _ = controller.set_arbitration_paced(
                crate::commands::Arbitration {
                    burst: crate::commands::ArbitrationBurst::Pow2(0),
                    high: 3,
                    medium: 0,
                    low: 0,
                },
                pacer,
            );
        }
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
            adopted_lists: [const { None }; ADOPTED_LIST_PAGES],
            adopted_owners: [None; ADOPTED_LIST_PAGES],
            shutdown_polls,
            lifecycle_pause,
            quiesced: false,
            queue_request,
            vector,
            resets: 0,
        })
    }

    /// **Reset the controller** and bring the I/O queues back over the same
    /// memory (`res/STORAGE_TRANSPORT.md` §7): the escalation for a command
    /// the device will not give up — an Abort it refused. Every command in
    /// flight is lost with the reset; nothing the controller held survives.
    /// Namespace, geometry, buffers and adopted targets do. Answers how many
    /// commands were lost; an `Err` leaves the device unusable, as any
    /// failed bring-up does.
    pub fn reset_paced(&mut self, pacer: &mut dyn CompletionPacer) -> Result<u32, BlockError> {
        self.controller.reset_paced(pacer).map_err(map_submit_error)?;
        let lost = self.qpair.reset();
        // Number of Queues is volatile across a reset: ask again, and settle
        // for no fewer than the group already runs.
        let (sqs, cqs) = self.queue_request;
        let (granted_sq, granted_cq) = if SQS == 1 {
            self.controller.set_num_queues_paced(cqs, pacer)
        } else {
            self.controller.set_queue_counts_paced(sqs, cqs, pacer)
        }
        .map_err(map_submit_error)?;
        if granted_cq < cqs || usize::from(granted_sq) < self.qpair.live_sqs() {
            return Err(BlockError::Backpressure);
        }
        self.controller
            .recreate_io_queue_group_paced(
                &self.qpair,
                core::array::from_fn(|index| PRIORITIES[index.min(PRIORITIES.len() - 1)]),
                self.vector,
                pacer,
            )
            .map_err(map_submit_error)?;
        if SQS > 1 {
            let _ = self.controller.set_arbitration_paced(
                crate::commands::Arbitration {
                    burst: crate::commands::ArbitrationBurst::Pow2(0),
                    high: 3,
                    medium: 0,
                    low: 0,
                },
                pacer,
            );
        }
        // Every command that borrowed a list page is gone with the reset.
        self.adopted_owners = [None; ADOPTED_LIST_PAGES];
        self.dsm_cookie = None;
        self.resets += 1;
        Ok(lost)
    }

    /// Controller resets this device has been through.
    pub fn resets(&self) -> u32 {
        self.resets
    }

    /// Wrap a DMA allocation the **caller** funded — a device mapping of a
    /// client's lease, not this row's pool — as a buffer a channel can adopt
    /// as a streaming target (`BlockChannel::adopt_target`).
    ///
    /// It carries no private list page: its commands borrow one from the
    /// shared [`ADOPTED_LIST_PAGES`] pool for exactly their flight, so any
    /// number of commands may target one adopted buffer at once.  The pool is
    /// allocated here on first use; a refusal returns the allocation, whose
    /// owner must still release it.
    ///
    /// The allocation must be page-aligned (every DMA window this tree hands
    /// out is) and a whole number of logical blocks long.
    pub fn adopt_buffer(
        &mut self,
        data: P::Buf,
    ) -> Result<NvmeBuffer<P::Buf>, (P::Buf, BlockError)> {
        let page = self.controller.page().bytes() as usize;
        let len = data.len();
        if len == 0
            || len % self.geometry.block_size as usize != 0
            || data.iova().0 % page as u64 != 0
        {
            return Err((data, BlockError::Alignment));
        }
        if self.adopted_lists[0].is_none() {
            for slot in &mut self.adopted_lists {
                match self.controller.pool().alloc_aligned(page, page) {
                    Some(list) => *slot = Some(list),
                    None => return Err((data, BlockError::Backpressure)),
                }
            }
        }
        Ok(NvmeBuffer {
            data,
            list: None,
            len,
        })
    }

    /// Return an adopted buffer's allocation to the pool it came from for
    /// quiescent release — for an import, the step that lets the owner's
    /// DMA host retire its device mapping.
    ///
    /// # Safety
    /// No command may be in flight against `buffer` and the device must hold
    /// no other reference to it — the [`DmaPool::release_quiesced`] contract.
    /// A channel hands an adopted target back only when nothing is in flight
    /// against it, which is the first half; the caller proves the rest.
    pub unsafe fn release_adopted(
        &self,
        buffer: NvmeBuffer<P::Buf>,
    ) -> Result<(), kore_driver::seam::DmaReleaseFailure<P::Buf>> {
        // SAFETY: forwarded to the caller above.
        unsafe { self.controller.pool().release_quiesced(buffer.into_data()) }
    }

    /// Lend a free shared list page to `cookie`'s command.
    fn lend_adopted_list(&mut self, cookie: u64) -> Option<usize> {
        let index = self
            .adopted_owners
            .iter()
            .zip(self.adopted_lists.iter())
            .position(|(owner, list)| owner.is_none() && list.is_some())?;
        self.adopted_owners[index] = Some(cookie);
        Some(index)
    }

    /// Return the page lent to `cookie`, if any — at its completion, or when
    /// its submission was refused before the device saw it.
    fn return_adopted_list(&mut self, cookie: u64) {
        if let Some(owner) = self
            .adopted_owners
            .iter_mut()
            .find(|owner| **owner == Some(cookie))
        {
            *owner = None;
        }
    }

    /// Shared list pages currently lent to in-flight commands.
    pub fn adopted_lists_in_flight(&self) -> usize {
        self.adopted_owners.iter().flatten().count()
    }

    pub fn qpair_stats(&self) -> crate::QpairStats {
        self.qpair.stats()
    }

    /// Whether the controller arbitrates its queues by weighted round robin
    /// with urgent priority (`CC.AMS`), rather than plain round robin.
    pub fn weighted_arbitration(&self) -> bool {
        self.controller.weighted_arbitration()
    }

    /// The controller's raw `CSTS`, read from its BAR without a command —
    /// the same register [`liveness`](DirectBlockDevice::liveness) judges.
    pub fn controller_status(&self) -> u32 {
        self.controller.regs().read32(crate::regs::REG_CSTS)
    }

    /// The data queue's submission side: `(software tail, the head the
    /// controller last reported, doorbell writes)`.  Tail ahead of head with
    /// nothing completing is the shape of commands the controller was never
    /// told about, or never fetched.
    pub fn submission_probe(&self) -> (u16, u16, u64) {
        (
            self.qpair.sq_tail(),
            self.qpair.last_sq_head(),
            self.qpair.doorbells(),
        )
    }

    /// The raw submission entry `back` slots behind the tail (0 = the last
    /// staged) — see [`Qpair::sq_entry_raw`](crate::Qpair::sq_entry_raw).
    pub fn submission_entry(&self, back: u16) -> Option<[u8; 64]> {
        self.qpair.sq_entry_raw(back)
    }

    /// Host memory the controller holds for its mapping tables, in bytes
    /// (0 = none granted, or already disabled).
    pub fn host_memory_buffer_bytes(&self) -> u64 {
        self.controller.info().hmb_bytes
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
            adopted_lists,
            ..
        } = self;
        drop(qpair);
        drop(dsm);
        drop(health);
        drop(adopted_lists);
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

    /// Build one data transfer.  A registered buffer uses its private list
    /// page; an adopted buffer (no private page) first tries PRP1/PRP2 only
    /// and borrows a shared page when the span needs a list.  A borrowed
    /// page stays lent to `cookie` until [`return_adopted_list`] — the
    /// caller returns it on a refused submission, completion returns it
    /// otherwise.
    fn build_transfer(
        &mut self,
        write: bool,
        lba: u64,
        blocks: u32,
        buffer: &mut NvmeBuffer<P::Buf>,
        buffer_offset: usize,
        cookie: u64,
    ) -> Result<crate::commands::NvmeSqe, BlockError> {
        let end = buffer_offset
            .checked_add(self.namespace.bytes(blocks) as usize)
            .ok_or(BlockError::Range)?;
        if end > buffer.len {
            return Err(BlockError::Range);
        }
        let iova = kore_abi::device::IovaAddr(
            buffer
                .data
                .iova()
                .0
                .checked_add(buffer_offset as u64)
                .ok_or(BlockError::Range)?,
        );
        let build = |namespace: &Namespace, list: Option<ListPage<'_>>| {
            if write {
                namespace.write(lba, blocks, iova, list)
            } else {
                namespace.read(lba, blocks, iova, list)
            }
        };
        if buffer.list.is_some() {
            let list = Self::list_page(buffer);
            return build(&self.namespace, list).map_err(map_submit_error);
        }
        match build(&self.namespace, None) {
            Err(Error::Prp(crate::prp::PrpError::ListNeeded)) => {}
            other => return other.map_err(map_submit_error),
        }
        let index = self
            .lend_adopted_list(cookie)
            .ok_or(BlockError::Backpressure)?;
        let page = self.adopted_lists[index]
            .as_mut()
            .expect("lent page exists");
        let count = page.len() / size_of::<u64>();
        // SAFETY: the page is a page-aligned pool allocation lent to this
        // command alone; no other command can hold it until it is returned.
        let entries = unsafe { core::slice::from_raw_parts_mut(page.cpu().cast::<u64>(), count) };
        let list = ListPage {
            iova: page.iova(),
            entries,
        };
        let built = build(&self.namespace, Some(list)).map_err(map_submit_error);
        if built.is_err() {
            self.adopted_owners[index] = None;
        }
        built
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
                self.build_transfer(false, lba, blocks, buffer, buffer_offset, cookie)?
            }
            DeviceCommand::Write {
                lba,
                blocks,
                buffer_offset,
            } => {
                let buffer = buffer.ok_or(BlockError::UnknownBuffer)?;
                self.build_transfer(true, lba, blocks, buffer, buffer_offset, cookie)?
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
        // NVMe completes a command only after its data transfer, so the
        // device has finished reading the list page by now.
        self.return_adopted_list(completion.cookie);
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

impl<R: Regs, P: DmaPool, const SQS: usize> NvmeBlockDevice<R, P, SQS> {
    /// Submission queues this device runs (`1..=SQS`; fewer than `SQS` when
    /// the controller granted fewer).
    pub fn live_queues(&self) -> usize {
        self.qpair.live_sqs()
    }

    /// Commands submission queue `sq` can still take.
    pub fn room_on(&self, sq: usize) -> usize {
        usize::from(self.qpair.room_on(sq))
    }

    /// Stage one command on submission queue `sq` (not yet visible to the
    /// device — [`publish_on`](Self::publish_on)). The class-aware twin of
    /// [`DirectBlockDevice::stage`], which uses queue 0.
    pub fn stage_on(
        &mut self,
        sq: usize,
        command: DeviceCommand,
        buffer: Option<&mut NvmeBuffer<P::Buf>>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        let sqe = self.build_sqe(command, buffer, cookie)?;
        if let Err(error) = self.qpair.submit_on(sq, sqe, cookie) {
            if self.dsm_cookie == Some(cookie) {
                self.dsm_cookie = None;
            }
            self.return_adopted_list(cookie);
            return Err(map_qpair_error(error));
        }
        Ok(())
    }

    /// Abort the command `cookie` staged on queue `sq` (NVMe Abort, 08h).
    /// `true` when the controller agreed: the command's own completion then
    /// arrives as *Command Abort Requested*. The admin wait is bounded.
    pub fn abort_on(&mut self, sq: usize, cookie: u64) -> bool {
        let Some((qid, cid)) = self.qpair.locate(sq, cookie) else {
            return false;
        };
        let mut pacer = kore_driver::block::CountedPacer::new(ABORT_POLLS, core::hint::spin_loop);
        matches!(
            self.controller.abort_paced(qid, cid, &mut pacer),
            Ok(crate::controller::AbortOutcome::Aborted)
        )
    }

    /// Ring submission queue `sq`'s doorbell for everything staged on it.
    pub fn publish_on(&mut self, sq: usize) {
        self.qpair.ring_on(sq, self.controller.regs());
    }
}

/// The device as the storage transport engine drives it: `SQS` submission
/// queues over one completion queue, a class per queue
/// ([`kore_driver::stor::engine::ThreeClass`]). Named by path, never
/// imported, so the device's [`DirectBlockDevice`] methods keep their
/// unambiguous names everywhere else.
impl<R: Regs, P: DmaPool, const SQS: usize> kore_driver::stor::QueueSet
    for NvmeBlockDevice<R, P, SQS>
{
    type Target = NvmeBuffer<P::Buf>;
    const SQS: usize = SQS;

    fn geometry(&self) -> BlockGeometry {
        self.geometry
    }

    fn target_bytes(target: &Self::Target) -> u64 {
        target.len() as u64
    }

    fn allocate(&mut self, bytes: usize) -> Result<Self::Target, BlockError> {
        DirectBlockDevice::allocate_buffer(self, bytes)
    }

    fn max_target_bytes(&self) -> usize {
        DirectBlockDevice::max_buffer_bytes(self)
    }

    fn room(&self, sq: usize) -> usize {
        self.room_on(sq)
    }

    fn stage(
        &mut self,
        sq: usize,
        command: DeviceCommand,
        target: Option<&mut Self::Target>,
        tag: kore_driver::stor::Tag,
    ) -> Result<(), BlockError> {
        self.stage_on(sq, command, target, tag.cookie())
    }

    fn abort(&mut self, sq: usize, tag: kore_driver::stor::Tag) -> bool {
        self.abort_on(sq, tag.cookie())
    }

    fn reset(&mut self) -> bool {
        kore_driver::block::DirectBlockDevice::reset(self)
    }

    fn publish(&mut self, sq: usize) {
        self.publish_on(sq);
    }

    fn reap(&mut self, out: &mut [DeviceCompletion]) -> usize {
        let mut count = 0;
        while count < out.len() {
            let Some(completion) = DirectBlockDevice::poll(self) else {
                break;
            };
            out[count] = completion;
            count += 1;
        }
        count
    }

    fn liveness(&mut self) -> kore_driver::block::Liveness {
        DirectBlockDevice::liveness(self)
    }

    fn health(&mut self) -> Result<kore_driver::block::BlockHealth, BlockError> {
        DirectBlockDevice::health(self)
    }

    fn quiesce(&mut self) -> Result<(), BlockError> {
        DirectBlockDevice::quiesce(self)
    }

    fn completion_word(&self) -> Option<&core::sync::atomic::AtomicU32> {
        self.qpair.completion_word()
    }
}

impl<R: Regs, P: DmaPool, const SQS: usize> DirectBlockDevice for NvmeBlockDevice<R, P, SQS> {
    type Buffer = NvmeBuffer<P::Buf>;

    /// A submitted command advances in the controller's own queues; there is
    /// no correlated reply for this side to await, which is why `progress`
    /// stays the ready default.  Sharing one controller between several
    /// designated regions relies on exactly that.
    const INDEPENDENT_PROGRESS: bool = true;

    fn abort(&mut self, cookie: u64) -> bool {
        self.abort_on(0, cookie)
    }

    fn reset(&mut self) -> bool {
        let mut pacer = kore_driver::block::CountedPacer::new(self.shutdown_polls, self.lifecycle_pause);
        self.reset_paced(&mut pacer).is_ok()
    }

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
            self.return_adopted_list(cookie);
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
            self.return_adopted_list(cookie);
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

    /// The storage transport engine over a three-queue group: a Demand
    /// read, a Stream read and a Commit flush each ride their own hardware
    /// submission queue (qids 1, 2, 3) and complete through the one shared
    /// completion queue, the read bytes landing in the ring's target.
    #[test]
    fn the_engine_puts_each_class_on_its_own_hardware_queue() {
        extern crate std;
        use kore_abi::stor::Class;
        use kore_driver::stor::engine::{StorEngine, StorHost, TargetAccess, ThreeClass};
        use kore_driver::stor::grant::{ClassSet, ExtentRun, Grant, Rights};
        use kore_ring::stor::{Op, Request, StorClient, WINDOW_BYTES, split};

        struct Quiet;
        impl StorHost for Quiet {
            fn kick(&mut self, _: kore_driver::stor::engine::RingId) {}
            fn drained(&mut self, _: kore_driver::stor::engine::RingId, _: kore_abi::stor::GrantId) {}
            fn departed(&mut self, _: kore_driver::stor::engine::RingId) {}
            fn expired(&mut self, _: kore_driver::stor::engine::Expiry) {}
        }

        let dev = MockNvmeDevice::new();
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let mut pacer = kore_driver::block::CountedPacer::new(16, || {});
        let backend = NvmeBlockDevice::<_, _, 3>::provision_queues_paced(
            ctrl, 1, 8, None, 0, &mut pacer, || {}, 16,
        )
        .unwrap();
        let layout = std::alloc::Layout::from_size_align(WINDOW_BYTES, 4096).unwrap();
        // SAFETY: non-zero size; leaked for a 'static window.
        let window: &'static mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(std::alloc::alloc_zeroed(layout), WINDOW_BYTES)
        };
        let (client, service) = split(window).unwrap();
        let mut engine: std::boxed::Box<StorEngine<'static, _, ThreeClass, 1, 1, 2, 16, 16>> =
            std::boxed::Box::new(StorEngine::new(backend).ok().expect("three queues"));
        let ring = engine.attach(service, ClassSet::ALL).ok().unwrap();
        let grant = engine
            .install(
                ring,
                Grant::new(
                    &[ExtentRun { lba: 0, blocks: 16 }],
                    Rights::READ | Rights::FLUSH,
                    ClassSet::ALL,
                )
                .unwrap(),
            )
            .unwrap();
        let target = {
            let queues = engine.queues_mut();
            kore_driver::stor::QueueSet::allocate(queues, 2 * MOCK_BLOCK_SIZE as usize).unwrap()
        };
        let t = engine.adopt_target(ring, target, TargetAccess::ReadInto).ok().unwrap();
        let mut client = StorClient::new(client);
        let request = |op, offset, target_offset, user_data| Request {
            op,
            grant,
            offset,
            len: if op == Op::Flush { 0 } else { MOCK_BLOCK_SIZE },
            target: t,
            target_offset,
            user_data,
            timeline: 0,
            deadline_ns: 0,
        };
        client
            .submit(Class::Demand, &request(Op::Read, 0, 0, 1))
            .unwrap();
        client
            .submit(
                Class::Stream,
                &request(Op::Read, u64::from(MOCK_BLOCK_SIZE), u64::from(MOCK_BLOCK_SIZE), 2),
            )
            .unwrap();
        client.submit(Class::Commit, &request(Op::Flush, 0, 0, 3)).unwrap();
        client.submit(Class::Commit, &request(Op::Flush, 0, 0, 4)).unwrap();
        for class in Class::ALL {
            client.publish(class);
        }
        engine.advance(&mut Quiet);
        engine.advance(&mut Quiet);
        let mut done = alloc::vec::Vec::new();
        while let Some(completion) = client.reap() {
            done.push((completion.user_data, completion.result));
        }
        done.sort_unstable_by_key(|(user, _)| *user);
        assert_eq!(
            done,
            [
                (1, Ok(MOCK_BLOCK_SIZE)),
                (2, Ok(MOCK_BLOCK_SIZE)),
                (3, Ok(0)),
                (4, Ok(0))
            ]
        );
        // Each class on its own hardware queue, in priority order: Demand
        // (urgent), Commit (high), Stream (medium).
        assert_eq!(dev.sq_consumed(1), Some(1), "Demand on SQ 1");
        assert_eq!(dev.sq_consumed(2), Some(2), "both Commit flushes on SQ 2");
        assert_eq!(dev.sq_consumed(3), Some(1), "Stream on SQ 3");
        let target = engine.release_target(ring, t).unwrap().into_inner();
        assert_eq!(&target.bytes()[..MOCK_BLOCK_SIZE as usize], &dev.disk_snapshot(0, 1)[..]);
        assert_eq!(&target.bytes()[MOCK_BLOCK_SIZE as usize..], &dev.disk_snapshot(1, 1)[..]);
    }

    /// A three-queue group: one submission queue per class, all bound to
    /// one completion queue; commands staged on each queue complete through
    /// that one queue, each with its own cookie and its own bytes.
    /// A command the device holds on one queue of a group is aborted by
    /// its cookie: the Abort names that queue and the command's id, and the
    /// command's own completion comes back failed. A cookie nothing holds
    /// is not aborted.
    #[test]
    fn a_held_command_on_a_group_queue_is_aborted_by_its_cookie() {
        let dev = MockNvmeDevice::new();
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let mut pacer = kore_driver::block::CountedPacer::new(16, || {});
        let mut backend = NvmeBlockDevice::<_, _, 3>::provision_queues_paced(
            ctrl, 1, 8, None, 0, &mut pacer, || {}, 16,
        )
        .unwrap();
        let mut buffer = backend.allocate_buffer(MOCK_BLOCK_SIZE as usize).unwrap();
        dev.swallow_completions(1);
        backend
            .stage_on(
                1,
                DeviceCommand::Read {
                    lba: 2,
                    blocks: 1,
                    buffer_offset: 0,
                },
                Some(&mut buffer),
                42,
            )
            .unwrap();
        backend.publish_on(1);
        assert!(backend.poll().is_none(), "the device holds it");

        assert!(!backend.abort_on(1, 43), "nothing carries that cookie");
        assert!(backend.abort_on(1, 42));
        let aborts = dev.aborts();
        assert_eq!(aborts.len(), 1);
        assert_eq!((aborts[0].0, aborts[0].2), (2, true), "queue 1 of the group is qid 2");
        let completion = backend.poll().expect("the victim posts its own completion");
        assert_eq!(completion.cookie, 42);
        assert!(completion.result.is_err(), "failed, not delivered");
        assert!(backend.poll().is_none());
    }

    /// A reset drops the command the device held, and brings the whole
    /// queue group back over the same memory: every submission queue bound
    /// to its completion queue again, and the next read served.
    #[test]
    fn a_reset_loses_what_the_device_held_and_brings_every_queue_back() {
        let dev = MockNvmeDevice::new();
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let mut pacer = kore_driver::block::CountedPacer::new(16, || {});
        let mut backend = NvmeBlockDevice::<_, _, 3>::provision_queues_paced(
            ctrl, 1, 8, None, 0, &mut pacer, || {}, 16,
        )
        .unwrap();
        let mut buffer = backend.allocate_buffer(MOCK_BLOCK_SIZE as usize).unwrap();
        dev.swallow_completions(1);
        backend
            .stage_on(
                1,
                DeviceCommand::Read {
                    lba: 2,
                    blocks: 1,
                    buffer_offset: 0,
                },
                Some(&mut buffer),
                42,
            )
            .unwrap();
        backend.publish_on(1);
        assert!(backend.poll().is_none(), "the device holds it");

        assert_eq!(backend.reset_paced(&mut pacer), Ok(1), "one command lost");
        assert_eq!(dev.resets(), 1);
        assert_eq!(backend.resets(), 1);
        assert!(backend.poll().is_none(), "the held completion never posts");
        for qid in 1..=3 {
            assert_eq!(dev.sq_binding(qid), Some(1), "SQ {qid} is bound to CQ 1 again");
        }
        backend
            .stage_on(
                2,
                DeviceCommand::Read {
                    lba: 3,
                    blocks: 1,
                    buffer_offset: 0,
                },
                Some(&mut buffer),
                43,
            )
            .unwrap();
        backend.publish_on(2);
        let completion = backend.poll().expect("the recreated queue serves");
        assert_eq!(completion.cookie, 43);
        assert!(completion.result.is_ok());
        assert_eq!(
            &buffer.bytes()[..MOCK_BLOCK_SIZE as usize],
            &dev.disk_snapshot(3, 1)[..]
        );
    }

    #[test]
    fn a_three_queue_group_shares_one_completion_queue() {
        let dev = MockNvmeDevice::new();
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let mut pacer = kore_driver::block::CountedPacer::new(16, || {});
        let mut backend = NvmeBlockDevice::<_, _, 3>::provision_queues_paced(
            ctrl, 1, 8, None, 0, &mut pacer, || {}, 16,
        )
        .unwrap();
        assert_eq!(backend.live_queues(), 3);
        for qid in 1..=3 {
            assert_eq!(dev.sq_binding(qid), Some(1), "SQ {qid} posts to CQ 1");
        }
        let mut buffers: alloc::vec::Vec<_> = (0..3)
            .map(|_| backend.allocate_buffer(MOCK_BLOCK_SIZE as usize).unwrap())
            .collect();
        for (sq, buffer) in buffers.iter_mut().enumerate() {
            backend
                .stage_on(
                    sq,
                    DeviceCommand::Read {
                        lba: sq as u64,
                        blocks: 1,
                        buffer_offset: 0,
                    },
                    Some(buffer),
                    100 + sq as u64,
                )
                .unwrap();
            backend.publish_on(sq);
        }
        let mut cookies = alloc::vec::Vec::new();
        while let Some(completion) = backend.poll() {
            assert_eq!(completion.result, Ok(()));
            cookies.push(completion.cookie);
        }
        cookies.sort_unstable();
        assert_eq!(cookies, [100, 101, 102]);
        for (sq, buffer) in buffers.iter().enumerate() {
            assert_eq!(buffer.bytes(), &dev.disk_snapshot(sq as u64, 1)[..]);
        }
        assert_eq!(backend.qpair_stats().stray, 0);
    }

    #[test]
    fn hardware_neutral_channel_drives_the_real_nvme_core() {
        let dev = MockNvmeDevice::new();
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let mut backend = NvmeBlockDevice::provision(ctrl, 1, 8, Some(0), 0, 16, || {}).unwrap();
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
        let mut backend = NvmeBlockDevice::provision(ctrl, 1, 8, None, 0, 16, || {}).unwrap();
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

    /// A caller-funded buffer adopted as a streaming target takes several
    /// list-sized reads in flight at once: each borrows a shared list page for
    /// exactly its flight, every byte lands where the disk says, and the pool
    /// is whole again once the queue drains.
    #[test]
    fn an_adopted_target_streams_list_sized_reads_concurrently() {
        let dev = MockNvmeDevice::new();
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let mut backend = NvmeBlockDevice::provision(ctrl, 1, 8, None, 0, 16, || {}).unwrap();
        // 64 blocks of "client" memory: a mapping this row did not allocate
        // from its channel budget.
        let funded = dev
            .pool()
            .alloc_aligned(64 * MOCK_BLOCK_SIZE as usize, 4096)
            .unwrap();
        let target = backend
            .adopt_buffer(funded)
            .map_err(|(_, error)| error)
            .unwrap();
        assert!(target.list.is_none(), "no private list page");
        let mut channel: BlockChannel<_, 2, 8> =
            BlockChannel::new(backend, CompletionMode::Poll).unwrap();
        let id = channel
            .adopt_target(target)
            .map_err(|(_, error)| error)
            .unwrap();

        // Three 8-block (eight-page) reads need a PRP list each.
        let reads = [(40u64, 0usize), (8, 16), (20, 48)];
        let mut ids = [kore_driver::block::RequestId(0); 3];
        {
            let mut batch = channel.batch();
            for ((lba, block), slot) in reads.iter().zip(ids.iter_mut()) {
                *slot = batch
                    .stage_read(*lba, 8, id, block * MOCK_BLOCK_SIZE as usize)
                    .unwrap();
            }
            batch.publish();
        }
        assert_eq!(channel.buffer_inflight(id), Some(3));
        assert_eq!(
            channel.control(|d| d.adopted_lists_in_flight()),
            Err(BlockError::BufferBusy),
            "control is refused while data is in flight"
        );
        let mut done = 0;
        while let Some(completion) = channel.poll() {
            assert!(ids.contains(&completion.request));
            completion.result.unwrap();
            done += 1;
        }
        assert_eq!(done, 3);
        assert_eq!(channel.control(|d| d.adopted_lists_in_flight()), Ok(0));
        // The channel never reads a target through the CPU; its owner does,
        // once the mapping has come back.
        let buffer = channel.release_buffer(id).unwrap();
        for (lba, block) in reads {
            let at = block * MOCK_BLOCK_SIZE as usize;
            let landed = &kore_driver::block::BlockBuffer::bytes(&buffer)
                [at..at + 8 * MOCK_BLOCK_SIZE as usize];
            assert_eq!(landed, &dev.disk_snapshot(lba, 8)[..], "lba {lba}");
        }
        // The mapping returns to its owner for quiescent release.
        let data = buffer.into_data();
        assert_eq!(data.len(), 64 * MOCK_BLOCK_SIZE as usize);
    }

    /// A DRAM-less controller is handed the least of its appetite and the
    /// consumer's allowance, as one descriptor; a controller that uses none,
    /// or an allowance under its floor, gets nothing and the mount proceeds.
    #[test]
    fn a_dram_less_controller_is_given_a_host_memory_buffer_of_one_descriptor() {
        // 64 KiB preferred, 16 KiB minimum, four descriptors at most.
        let dev = MockNvmeDevice::new();
        dev.set_hmb(16, 4, 4);
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let backend = NvmeBlockDevice::provision(ctrl, 1, 8, None, 1 << 20, 16, || {}).unwrap();
        assert_eq!(
            dev.hmb(),
            Some((16, 1)),
            "64 KiB = 16 pages, one descriptor"
        );
        assert_eq!(backend.host_memory_buffer_bytes(), 64 * 1024);

        // The allowance caps it, in whole pages.
        let dev = MockNvmeDevice::new();
        dev.set_hmb(16, 4, 4);
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let backend =
            NvmeBlockDevice::provision(ctrl, 1, 8, None, 32 * 1024 + 100, 16, || {}).unwrap();
        assert_eq!(dev.hmb(), Some((8, 1)));
        assert_eq!(backend.host_memory_buffer_bytes(), 32 * 1024);

        // Below the controller's minimum: not offered.
        let dev = MockNvmeDevice::new();
        dev.set_hmb(16, 4, 4);
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let backend = NvmeBlockDevice::provision(ctrl, 1, 8, None, 8 * 1024, 16, || {}).unwrap();
        assert_eq!(dev.hmb(), None);
        assert_eq!(backend.host_memory_buffer_bytes(), 0);

        // A controller with no appetite gets none whatever the allowance.
        let dev = MockNvmeDevice::new();
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let backend = NvmeBlockDevice::provision(ctrl, 1, 8, None, 1 << 30, 16, || {}).unwrap();
        assert_eq!(dev.hmb(), None);
        assert_eq!(backend.host_memory_buffer_bytes(), 0);
    }

    /// The buffer is disabled at the controller before the orderly
    /// shutdown, through the same quiesce the filesystem takes.
    #[test]
    fn quiesce_disables_the_host_memory_buffer_before_shutdown() {
        let dev = MockNvmeDevice::new();
        dev.set_hmb(16, 4, 4);
        let ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        let mut backend = NvmeBlockDevice::provision(ctrl, 1, 8, None, 1 << 20, 16, || {}).unwrap();
        assert!(dev.hmb().is_some());
        backend.quiesce().unwrap();
        assert_eq!(dev.hmb(), None);
        assert_eq!(dev.hmb_disables(), 1);
        assert_eq!(backend.host_memory_buffer_bytes(), 0);
    }

    /// A pool that cannot carry the buffer leaves the controller without
    /// one and the filesystem's own budget untouched: provisioning still
    /// succeeds.
    #[test]
    fn a_pool_too_small_for_the_buffer_still_provisions_the_device() {
        let dev = MockNvmeDevice::new();
        dev.set_hmb(16_384, 1_024, 8); // 64 MiB preferred, 4 MiB minimum
        let (pool, _cursor) = BudgetPool::new(2 << 20);
        let ctrl = Controller::bring_up(dev.clone(), pool, 16, || {}).unwrap();
        let backend = NvmeBlockDevice::provision(ctrl, 1, 8, None, 64 << 20, 16, || {}).unwrap();
        assert_eq!(
            dev.hmb(),
            None,
            "2 MiB cannot carry 4 MiB, so none was offered"
        );
        assert_eq!(backend.host_memory_buffer_bytes(), 0);
    }

    #[test]
    fn mps_256k_and_four_fs_buffers_fit_the_exact_2mib_budget() {
        const DMA_BUDGET: usize = 2 * 1024 * 1024;
        const FS_BUFFER: usize = 256 * 1024;

        let dev = MockNvmeDevice::new().with_mps(6).unwrap();
        let (pool, cursor) = BudgetPool::new(DMA_BUDGET);
        let ctrl = Controller::bring_up(dev, pool, 16, || {}).unwrap();
        let mut backend = NvmeBlockDevice::provision(ctrl, 1, 32, None, 0, 16, || {}).unwrap();
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
