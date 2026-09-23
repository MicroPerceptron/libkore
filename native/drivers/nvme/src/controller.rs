//! The **controller** — `spdk_nvme_ctrlr`: owns the admin qpair and the
//! bring-up sequence, and answers Identify.  Generic over the platform
//! [`seam`](crate::seam); the boot-test provides a kernel seam, the
//! [`mock`](crate::mock) a host one.
//!
//! Bring-up (NVMe base spec §3.1.1 — the controller-initialization flow):
//! read `CAP`, disable and wait for `!CSTS.RDY`, size + allocate the admin
//! SQ/CQ, program `AQA`/`ASQ`/`ACQ`, then enable and wait for `CSTS.RDY`.
//! Preparation and activation are separate type states so a capability host
//! can publish the complete DMA graph before the device is allowed to master
//! it. Then Identify Controller + Namespace fill [`DeviceInfo`].

use crate::commands::{HmbDescriptor, IdentifyController, IdentifyNamespace, NvmeSqe};
use crate::prp::MemPage;
use crate::qpair::{Completion, Qpair};
use crate::regs::{
    CC_EN, CC_SHN_MASK, CC_SHN_NORMAL, CSTS_CFS, CSTS_RDY, CSTS_SHST_COMPLETE, CSTS_SHST_MASK,
    Capabilities, REG_ACQ, REG_AQA, REG_ASQ, REG_CAP, REG_CC, REG_CSTS, REG_VS, cc_enable,
};
use crate::seam::{Dma, DmaPool, Regs};
use crate::{Error, Result};
use kore_driver::block::{CompletionPacer, CountedPacer};

use core::fmt;
use core::mem::size_of;
use core::sync::atomic::{Ordering, fence};

/// Admin queue depth (spec allows up to 4096; 32 is the conventional
/// bring-up size and covers every setup command outstanding at once).
pub const ADMIN_DEPTH: u16 = 32;

/// Identify-derived controller + namespace facts the driver acts on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Controller version (`VS`).
    pub version: u32,
    /// Maximum Data Transfer Size in bytes (0 = unlimited).
    pub mdts_bytes: u64,
    /// Number of namespaces.
    pub num_namespaces: u32,
    /// Optional NVM Command Support (Identify Controller ONCS).
    pub oncs: u16,
    /// Namespace 1's logical block size in bytes.
    pub block_size: u32,
    /// Namespace 1's size in logical blocks.
    pub block_count: u64,
    /// Preferred namespace write granularity, in logical blocks.
    pub preferred_write_blocks: u32,
    /// Preferred namespace write alignment, in logical blocks.
    pub write_alignment_blocks: u32,
    /// Optimal namespace write command size, in logical blocks.
    pub optimal_write_blocks: u32,
    /// Optimal I/O boundary, in logical blocks (0 = unspecified).
    pub optimal_boundary_blocks: u32,
    /// Every metadata-free LBA data size the namespace offers (bit `k` =
    /// `512 << k` bytes) — the active one is `block_size`.
    pub supported_block_sizes: u32,
    /// Host Memory Buffer the controller would like, in bytes (Identify
    /// HMPRE; 0 = it uses none).  A DRAM-less controller keeps its mapping
    /// tables here and pages them from NAND without one.
    pub hmb_preferred_bytes: u64,
    /// The smallest Host Memory Buffer the controller accepts (HMMIN).
    pub hmb_min_bytes: u64,
    /// The smallest chunk one descriptor may name (HMMINDS; 0 = no floor).
    pub hmb_min_descriptor_bytes: u64,
    /// Descriptors the controller accepts at most (HMMAXD; 0 = no limit).
    pub hmb_max_descriptors: u16,
    /// Host Memory Buffer currently enabled, in bytes (0 = none).
    pub hmb_bytes: u64,
}

/// Identify reports the Host Memory Buffer sizes in **fixed 4 KiB units**
/// — not `CC.MPS`, which is the unit Set Features takes them in.  The
/// conversion to pages happens once, at the command builder.
const fn hmb_units_to_bytes(units: u32) -> u64 {
    (units as u64) << 12
}

/// Chunks a host memory buffer may be built from — 64 MiB in the 4 MiB
/// pieces a capability host carves from a grown DMA window, which is more
/// than any consumer-class HMMAXD this driver has met.
pub const HMB_MAX_CHUNKS: usize = 16;

/// A host memory buffer the controller is using: the descriptor table and
/// the chunks it names.  Retained by the controller for exactly as long as
/// the feature is enabled — the device reads and writes these frames at
/// any moment, so they outlive every command and are released only after
/// the controller acknowledged the disable.
pub struct HostMemoryBuffer<B> {
    list: B,
    chunks: [Option<B>; HMB_MAX_CHUNKS],
    bytes: u64,
}

impl<B> HostMemoryBuffer<B> {
    /// Bytes the controller was given.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Chunks the buffer is made of.
    pub fn chunks(&self) -> usize {
        self.chunks.iter().flatten().count()
    }

    /// The descriptor table's storage.
    pub fn list(&self) -> &B {
        &self.list
    }
}

/// Decode Identify Controller MDTS in the units fixed by the base spec:
/// `2^MDTS` times the controller's minimum memory-page size, not a fixed
/// 4 KiB and not the currently configured page. Zero means no limit.
fn mdts_bytes(mdts: u8, mpsmin: u8) -> Result<u64> {
    if mdts == 0 {
        return Ok(0);
    }
    let exponent = 12u32
        .checked_add(u32::from(mpsmin))
        .and_then(|value| value.checked_add(u32::from(mdts)))
        .ok_or(Error::Geometry)?;
    1u64.checked_shl(exponent).ok_or(Error::Geometry)
}

/// A bound NVMe controller.
pub struct Controller<R: Regs, D: DmaPool> {
    regs: R,
    pool: D,
    caps: Capabilities,
    /// The **configured** memory page (`CC.MPS = MPSMIN` — the smallest
    /// page the controller accepts; never a hardcoded 4 KiB).  All PRP
    /// math derives from this.
    page: MemPage,
    admin: Qpair<D::Buf, D::Buf>,
    info: DeviceInfo,
    /// The host memory buffer in use, if one was enabled.
    hmb: Option<HostMemoryBuffer<D::Buf>>,
}

/// A disabled NVMe controller whose admin queues have been allocated and
/// programmed, but which has not been allowed to fetch from them.
///
/// This affine type is the boundary a capability-backed host uses to publish
/// the pool's IOMMU window: [`Controller::prepare`] establishes every device
/// pointer while `CC.EN == 0`, and [`start`](Self::start) is the only path that
/// consumes that prepared graph and enables the controller.
pub struct PreparedController<R: Regs, D: DmaPool> {
    regs: R,
    pool: D,
    caps: Capabilities,
    page: MemPage,
    admin: Qpair<D::Buf, D::Buf>,
    info: DeviceInfo,
}

/// A refused transition from prepared to running.
///
/// The controller seams and every admin-queue handle ride back with the error.
/// This is load-bearing for capability hosts: they can restore the seams to
/// their common resource owner, and only then disable bus mastering and retire
/// the mappings. The value deliberately cannot become a `PreparedController`
/// again: the best-effort rollback may not have observed `!RDY`, so that type
/// state would claim more than the hardware proved. Dropping this value is the
/// legacy behavior for hosts whose pool was already device-visible.
#[must_use = "a refused start still owns the prepared controller seams"]
pub struct StartError<R: Regs, D: DmaPool> {
    error: Error,
    // Driver-internal DMA children drop before either outer platform seam.
    admin: Qpair<D::Buf, D::Buf>,
    regs: R,
    pool: D,
}

impl<R: Regs, D: DmaPool> fmt::Debug for StartError<R, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StartError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<R: Regs, D: DmaPool> StartError<R, D> {
    /// The hardware refusal, without consuming the retained seams.
    pub fn error(&self) -> Error {
        self.error
    }

    /// Recover both platform seams beside the hardware refusal.
    pub fn into_seam(self) -> (Error, R, D) {
        let Self {
            error,
            admin,
            regs,
            pool,
        } = self;
        drop(admin);
        (error, regs, pool)
    }

    /// Discard the prepared seams and retain only the old scalar error.
    /// Convenience bring-up paths use this when their DMA was visible before
    /// preparation and no outer capability owner needs reconstruction.
    pub fn into_error(self) -> Error {
        self.error
    }
}

impl<R: Regs, D: DmaPool> PreparedController<R, D> {
    /// The controller capabilities decoded during preparation.
    pub fn caps(&self) -> &Capabilities {
        &self.caps
    }

    /// The configured memory page that will be written to `CC.MPS` at start.
    pub fn page(&self) -> MemPage {
        self.page
    }

    /// Release the platform seams without ever enabling the controller.
    pub fn into_seam(self) -> (R, D) {
        let Self { regs, pool, .. } = self;
        (regs, pool)
    }

    /// Enable the prepared controller after its DMA graph has become visible
    /// to the device, then wait for `CSTS.RDY`.
    ///
    /// If readiness is refused, this method clears `CC.EN` again and makes a
    /// bounded best-effort wait for `!RDY`, then returns the still-owned
    /// prepared state in [`StartError`]. The capability host still owns the
    /// outer containment policy (notably disabling PCI bus mastering) when a
    /// device ignores that stop.
    pub fn start(
        self,
        max_polls: u32,
        pause: impl FnMut(),
    ) -> core::result::Result<Controller<R, D>, StartError<R, D>> {
        let mut pacer = CountedPacer::new(max_polls, pause);
        self.start_paced(&mut pacer)
    }

    /// Elapsed-time/composed-pacer sibling of [`start`](Self::start).
    pub fn start_paced(
        self,
        pacer: &mut dyn CompletionPacer,
    ) -> core::result::Result<Controller<R, D>, StartError<R, D>> {
        let Self {
            regs,
            pool,
            caps,
            page,
            admin,
            info,
        } = self;

        regs.write32(REG_CC, cc_enable(caps.mpsmin));
        if let Err(error) =
            Controller::<R, D>::wait_csts_paced(&regs, |csts| csts & CSTS_RDY != 0, pacer)
        {
            let cc = regs.read32(REG_CC);
            regs.write32(REG_CC, cc & !(CC_EN | CC_SHN_MASK));
            let _ = Controller::<R, D>::wait_csts_paced(&regs, |csts| csts & CSTS_RDY == 0, pacer);
            return Err(StartError {
                error,
                admin,
                regs,
                pool,
            });
        }

        Ok(Controller {
            regs,
            pool,
            caps,
            page,
            admin,
            info,
            hmb: None,
        })
    }
}

impl<R: Regs, D: DmaPool> Controller<R, D> {
    /// **Bring the controller up** over the seam. Convenience composition of
    /// [`prepare`](Self::prepare) and [`PreparedController::start`] for hosts
    /// whose DMA pool is already device-visible (the mock and legacy paths).
    ///
    /// `max_polls`/`pause` bound the two `CSTS` waits (disable-ack and
    /// ready): on hardware `pause` is a real inter-poll delay; the host
    /// mock completes synchronously, so a no-op `pause` and a small budget
    /// suffice.
    pub fn bring_up(regs: R, pool: D, max_polls: u32, mut pause: impl FnMut()) -> Result<Self> {
        Self::prepare(regs, pool, max_polls, &mut pause)?
            .start(max_polls, pause)
            .map_err(StartError::into_error)
    }

    /// Disable the controller and wait for `!CSTS.RDY`, then allocate and
    /// program its complete admin-queue DMA graph without setting `CC.EN`.
    ///
    /// A capability host can prepare against a CPU-visible pool, publish that
    /// pool into the device's IOMMU domain, and only then consume the returned
    /// value with [`PreparedController::start`].
    pub fn prepare(
        regs: R,
        pool: D,
        max_polls: u32,
        pause: impl FnMut(),
    ) -> Result<PreparedController<R, D>> {
        let mut pacer = CountedPacer::new(max_polls, pause);
        Self::prepare_paced(regs, pool, &mut pacer)
    }

    /// Prepare the controller using a host-composed completion pacer.
    pub fn prepare_paced(
        regs: R,
        pool: D,
        pacer: &mut dyn CompletionPacer,
    ) -> Result<PreparedController<R, D>> {
        let caps = Capabilities::from_cap(regs.read64(REG_CAP));
        let version = regs.read32(REG_VS);
        // Configure the smallest page the controller accepts — correct
        // even when MPSMIN > 0 (a hardcoded 4 KiB would be illegal there).
        let page = MemPage::from_mps(caps.mpsmin);

        // Disable first: no queue buffer is allocated or programmed until the
        // controller has acknowledged that it can no longer fetch DMA.
        let cc = regs.read32(REG_CC);
        regs.write32(REG_CC, cc & !(CC_EN | CC_SHN_MASK));
        Self::wait_csts_paced(&regs, |csts| csts & CSTS_RDY == 0, pacer)?;

        // Size the admin queue within the controller's MQES. `max_queue_depth`
        // saturates: `max_queue_entries() as u16` wraps a maximal MQES to zero
        // and refuses the most capable controllers.
        let depth = ADMIN_DEPTH.min(caps.max_queue_depth());
        let sq = pool
            .alloc_aligned(depth as usize * size_of::<NvmeSqe>(), page.bytes() as usize)
            .ok_or(Error::DmaAlloc)?;
        let cq = pool
            .alloc_aligned(
                depth as usize * size_of::<crate::commands::NvmeCqe>(),
                page.bytes() as usize,
            )
            .ok_or(Error::DmaAlloc)?;
        let admin = Qpair::new(0, sq, cq, depth, &caps)?;

        // Program the admin queue: attributes (0-based sizes) + base IOVAs
        // (typed to the wire edge — the raw u64 exists only here).
        let aqa = ((admin.depth() as u32 - 1) << 16) | (admin.depth() as u32 - 1);
        regs.write32(REG_AQA, aqa);
        regs.write64(REG_ASQ, admin.sq_iova().0);
        regs.write64(REG_ACQ, admin.cq_iova().0);

        Ok(PreparedController {
            regs,
            pool,
            caps,
            page,
            admin,
            info: DeviceInfo {
                version,
                ..DeviceInfo::default()
            },
        })
    }

    /// Poll `CSTS` until `done` holds, failing on `CSTS.CFS` (fatal) or
    /// timeout.
    ///
    /// The loop is [`kore_driver::wait::handshake`], shared with every other
    /// driver's bring-up waits; `CSTS.CFS` is its fault predicate, which is
    /// where the abort-early behaviour this controller has always had now
    /// lives for everyone.
    fn wait_csts(
        regs: &R,
        done: impl Fn(u32) -> bool,
        max_polls: u32,
        pause: &mut impl FnMut(),
    ) -> Result<()> {
        kore_driver::wait::handshake(
            regs,
            // Still a probe count, and still named as one: this controller's
            // bring-up budget is caller-supplied all the way up its public
            // API, so clocking it is an API change to `bring_up`/`identify`/
            // `smart` rather than a local one. `Budget::spins` keeps today's
            // behaviour exactly and makes the remaining count greppable.
            kore_driver::Budget::spins(max_polls),
            pause,
            |r: &R| done(r.read32(REG_CSTS)),
            |r: &R| r.read32(REG_CSTS) & CSTS_CFS != 0,
        )
        // Both failures are `NotReady` to this caller, as before — the
        // distinction is preserved in the `Ack` for a future caller that wants
        // to tell "dead controller" from "slow controller".
        .ok_or(Error::NotReady, Error::NotReady)
    }

    fn wait_csts_paced(
        regs: &R,
        done: impl Fn(u32) -> bool,
        pacer: &mut dyn CompletionPacer,
    ) -> Result<()> {
        kore_driver::wait::handshake_paced(
            regs,
            pacer,
            |r: &R| done(r.read32(REG_CSTS)),
            |r: &R| r.read32(REG_CSTS) & CSTS_CFS != 0,
        )
        .ok_or(Error::NotReady, Error::NotReady)
    }

    /// The controller capabilities.
    pub fn caps(&self) -> &Capabilities {
        &self.caps
    }

    /// The configured memory page (`CC.MPS`) all PRP math derives from.
    pub fn page(&self) -> MemPage {
        self.page
    }

    /// The identify-derived device info (populated by [`identify`](Self::identify)).
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Borrow the register seam (doorbells, MSI-X table — the boot-test /
    /// I/O paths).
    pub fn regs(&self) -> &R {
        &self.regs
    }

    /// Borrow the DMA pool (I/O buffers, PRP pages).
    pub fn pool(&self) -> &D {
        &self.pool
    }

    /// Run one admin command to completion, mapping a bad status to
    /// [`Error::Status`].
    fn admin_wait(
        &mut self,
        cmd: NvmeSqe,
        max_polls: u32,
        pause: impl FnMut(),
    ) -> Result<Completion> {
        let mut pacer = CountedPacer::new(max_polls, pause);
        self.admin_wait_paced(cmd, &mut pacer)
    }

    fn admin_wait_paced(
        &mut self,
        cmd: NvmeSqe,
        pacer: &mut dyn CompletionPacer,
    ) -> Result<Completion> {
        let c = self
            .admin
            .submit_and_wait_paced(cmd, &self.regs, pacer)
            .ok_or(Error::Timeout)?;
        if !c.is_success() {
            return Err(Error::Status {
                sct: c.cqe.status_code_type(),
                sc: c.cqe.status_code(),
            });
        }
        Ok(c)
    }

    /// **Identify** the controller + namespace 1, filling [`DeviceInfo`].
    pub fn identify(&mut self, max_polls: u32, mut pause: impl FnMut()) -> Result<DeviceInfo> {
        let mut pacer = CountedPacer::new(max_polls, &mut pause);
        self.identify_paced(&mut pacer)
    }

    /// Identify through a caller-composed elapsed-time/counting pacer.
    pub fn identify_paced(&mut self, pacer: &mut dyn CompletionPacer) -> Result<DeviceInfo> {
        // Identify Controller into a 4 KiB DMA buffer.
        let buf = self.pool.alloc(4096).ok_or(Error::DmaAlloc)?;
        self.admin_wait_paced(NvmeSqe::identify_controller(buf.iova()), pacer)?;
        // SAFETY: the device wrote 4096 valid bytes of IdentifyController
        // into the buffer we own for this read.
        let idc: IdentifyController =
            unsafe { (buf.cpu() as *const IdentifyController).read_volatile() };
        self.info.mdts_bytes = mdts_bytes(idc.mdts, self.caps.mpsmin)?;
        self.info.num_namespaces = idc.nn;
        self.info.oncs = idc.oncs;
        self.info.hmb_preferred_bytes = hmb_units_to_bytes(idc.hmpre);
        self.info.hmb_min_bytes = hmb_units_to_bytes(idc.hmmin);
        self.info.hmb_min_descriptor_bytes = hmb_units_to_bytes(idc.hmminds);
        self.info.hmb_max_descriptors = idc.hmmaxd;

        // Identify Namespace 1 into the same buffer (reused).
        self.admin_wait_paced(NvmeSqe::identify_namespace(1, buf.iova()), pacer)?;
        // SAFETY: as above, IdentifyNamespace.
        let idn: IdentifyNamespace =
            unsafe { (buf.cpu() as *const IdentifyNamespace).read_volatile() };
        self.info.block_size = idn.block_size();
        self.info.block_count = idn.nsze;
        if idn.metadata_size() != 0 {
            // This core has no metadata pointer/data-protection transport.
            // Accepting an LBA format with metadata would make every PRP
            // transfer shorter than the namespace command describes.
            return Err(Error::Geometry);
        }
        let (granularity, alignment, optimal, boundary) = idn.io_performance();
        self.info.preferred_write_blocks = granularity;
        self.info.write_alignment_blocks = alignment;
        self.info.optimal_write_blocks = optimal;
        self.info.optimal_boundary_blocks = boundary;
        self.info.supported_block_sizes = idn.supported_block_sizes();

        Ok(self.info)
    }

    /// Namespace 1 as an addressable block store (valid after
    /// [`identify`](Self::identify)) — carries the configured page (PRP
    /// math matches `CC.MPS`) and the controller's MDTS (oversized
    /// transfers refuse before reaching the device).
    pub fn namespace(&self) -> crate::Namespace {
        crate::Namespace::new(
            1,
            self.info.block_size,
            self.info.block_count,
            self.page,
            self.info.mdts_bytes,
        )
    }

    /// Fetch the **SMART / Health** log (temperature, wear, counters) —
    /// the driver's health row.
    pub fn smart(
        &mut self,
        max_polls: u32,
        pause: impl FnMut(),
    ) -> Result<crate::commands::SmartLog> {
        let mut buf = self.pool.alloc(512).ok_or(Error::DmaAlloc)?;
        self.smart_into(&mut buf, max_polls, pause)
    }

    /// Fetch SMART / Health into caller-owned DMA storage. Long-lived block
    /// devices use this form so repeated status probes reuse one allocation
    /// instead of consuming their capability-backed bump pool.
    pub fn smart_into(
        &mut self,
        buf: &mut D::Buf,
        max_polls: u32,
        mut pause: impl FnMut(),
    ) -> Result<crate::commands::SmartLog> {
        if buf.len() < 512 {
            return Err(Error::Geometry);
        }
        self.admin_wait(NvmeSqe::get_log_smart(buf.iova()), max_polls, &mut pause)?;
        // SAFETY: the device wrote 512 valid bytes of SmartLog.
        Ok(unsafe { (buf.cpu() as *const crate::commands::SmartLog).read_volatile() })
    }

    /// **Create an I/O queue pair** `qid` of `depth` entries, optionally
    /// interrupt-driven (`msix_vector`).  Allocates the SQ/CQ, then issues
    /// Create-CQ **then** Create-SQ on the admin queue (order matters — the
    /// SQ binds to an existing CQ).  Returns the ready qpair.
    pub fn create_io_qpair(
        &mut self,
        qid: u16,
        depth: u16,
        msix_vector: Option<u16>,
        max_polls: u32,
        pause: impl FnMut(),
    ) -> Result<Qpair<D::Buf, D::Buf>> {
        let mut pacer = CountedPacer::new(max_polls, pause);
        self.create_io_qpair_paced(qid, depth, msix_vector, &mut pacer)
    }

    /// Create one I/O queue pair through a composed pacer.
    pub fn create_io_qpair_paced(
        &mut self,
        qid: u16,
        depth: u16,
        msix_vector: Option<u16>,
        pacer: &mut dyn CompletionPacer,
    ) -> Result<Qpair<D::Buf, D::Buf>> {
        // Saturating, for the reason given at the admin queue above: the I/O
        // queue carries the identical wrap.
        let depth = depth.min(self.caps.max_queue_depth());
        let sq = self
            .pool
            .alloc_aligned(
                depth as usize * size_of::<NvmeSqe>(),
                self.page.bytes() as usize,
            )
            .ok_or(Error::DmaAlloc)?;
        let cq = self
            .pool
            .alloc_aligned(
                depth as usize * size_of::<crate::commands::NvmeCqe>(),
                self.page.bytes() as usize,
            )
            .ok_or(Error::DmaAlloc)?;
        let qp = Qpair::new(qid, sq, cq, depth, &self.caps)?;

        let (iv, ien) = msix_vector.map_or((0, false), |v| (v, true));
        self.admin_wait_paced(
            NvmeSqe::create_io_cq(qid, depth, qp.cq_iova(), iv, ien),
            pacer,
        )?;
        self.admin_wait_paced(NvmeSqe::create_io_sq(qid, depth, qp.sq_iova(), qid), pacer)?;
        Ok(qp)
    }

    /// Tell the controller how many I/O queues to allocate (Set Features
    /// 0x07) — called once before creating any.  Returns the granted
    /// `(sq, cq)` counts.
    pub fn set_num_queues(
        &mut self,
        want: u16,
        max_polls: u32,
        pause: impl FnMut(),
    ) -> Result<(u16, u16)> {
        let mut pacer = CountedPacer::new(max_polls, pause);
        self.set_num_queues_paced(want, &mut pacer)
    }

    /// **Enable a Host Memory Buffer** made of `chunks` (device-visible,
    /// `CC.MPS`-aligned, each a whole number of pages) described in `list`
    /// (16 bytes per chunk, 16-byte aligned).  The controller retains both
    /// until [`disable_host_memory_buffer`](Self::disable_host_memory_buffer)
    /// succeeds — which [`shutdown`](Self::shutdown) attempts first — because
    /// the device may read or write these frames at any moment after this
    /// returns `Ok`.  Returns the bytes the controller was given.
    ///
    /// The sizing decision is the caller's: it has Identify's
    /// [`hmb_preferred_bytes`](DeviceInfo::hmb_preferred_bytes) and the
    /// floors, and it owns the memory.  What this checks is the spec's
    /// shape — at least one chunk, no more than HMMAXD, each chunk
    /// page-aligned and page-sized, none below HMMINDS, the whole not below
    /// HMMIN — and refuses `Geometry` rather than hand the device a table it
    /// would reject or, worse, misread.
    pub fn enable_host_memory_buffer(
        &mut self,
        list: D::Buf,
        chunks: [Option<D::Buf>; HMB_MAX_CHUNKS],
        pacer: &mut dyn CompletionPacer,
    ) -> Result<u64> {
        if self.hmb.is_some() {
            return Err(Error::Geometry);
        }
        let page = self.page.bytes() as u64;
        let count = chunks.iter().flatten().count();
        let max = usize::from(self.info.hmb_max_descriptors);
        if count == 0 || (max != 0 && count > max) {
            return Err(Error::Geometry);
        }
        if list.len() < count * size_of::<HmbDescriptor>() || list.iova().0 & 0xF != 0 {
            return Err(Error::Geometry);
        }
        let mut total = 0u64;
        for (index, chunk) in chunks.iter().flatten().enumerate() {
            let bytes = chunk.len() as u64;
            let aligned = chunk.iova().0 % page == 0 && bytes % page == 0;
            if bytes == 0 || !aligned || bytes < self.info.hmb_min_descriptor_bytes {
                return Err(Error::Geometry);
            }
            let pages = u32::try_from(bytes / page).map_err(|_| Error::Geometry)?;
            total = total.checked_add(bytes).ok_or(Error::Geometry)?;
            let descriptor = HmbDescriptor::new(chunk.iova(), pages);
            // SAFETY: `list` is a DMA buffer this task owns for its whole
            // life, and the bound above proved room for `count` entries;
            // volatile because the device reads the table after submit.
            unsafe {
                (list.cpu() as *mut HmbDescriptor)
                    .add(index)
                    .write_volatile(descriptor);
            }
        }
        if total < self.info.hmb_min_bytes {
            return Err(Error::Geometry);
        }
        let hsize = u32::try_from(total / page).map_err(|_| Error::Geometry)?;
        // The table is complete before the doorbell publishes it.
        fence(Ordering::Release);
        let entries = count as u32;
        self.admin_wait_paced(
            NvmeSqe::enable_host_memory_buffer(hsize, list.iova(), entries),
            pacer,
        )?;
        self.info.hmb_bytes = total;
        self.hmb = Some(HostMemoryBuffer {
            list,
            chunks,
            bytes: total,
        });
        Ok(total)
    }

    /// The host memory buffer in use, if any.
    pub fn host_memory_buffer(&self) -> Option<&HostMemoryBuffer<D::Buf>> {
        self.hmb.as_ref()
    }

    /// **Disable the Host Memory Buffer**, releasing its memory only once
    /// the controller acknowledged: a refused or timed-out disable keeps the
    /// buffer retained, because the device may still be writing it.  A
    /// no-op when none is enabled.
    pub fn disable_host_memory_buffer(
        &mut self,
        max_polls: u32,
        mut pause: impl FnMut(),
    ) -> Result<()> {
        if self.hmb.is_none() {
            return Ok(());
        }
        self.admin_wait(NvmeSqe::disable_host_memory_buffer(), max_polls, &mut pause)?;
        self.info.hmb_bytes = 0;
        self.hmb = None;
        Ok(())
    }

    /// Negotiate queue counts through a composed pacer.
    pub fn set_num_queues_paced(
        &mut self,
        want: u16,
        pacer: &mut dyn CompletionPacer,
    ) -> Result<(u16, u16)> {
        let c = self.admin_wait_paced(
            NvmeSqe::set_num_queues(want.saturating_sub(1), want.saturating_sub(1)),
            pacer,
        )?;
        let nsq = (c.cqe.result & 0xFFFF) as u16 + 1;
        let ncq = ((c.cqe.result >> 16) & 0xFFFF) as u16 + 1;
        Ok((nsq, ncq))
    }

    /// **Orderly shutdown** (base spec §3.1.5): notify normal shutdown, wait
    /// for the controller to report shutdown complete, then disable and wait
    /// for `!RDY` so the next owner's bring-up starts from a clean state.
    ///
    /// This is a driver's quiesce hook: after it returns `Ok`, the
    /// controller has flushed and stopped all DMA, so the derived BAR/IOVA
    /// authority above it can be retired without racing an in-flight
    /// transfer.  A failure means the device did not acknowledge within the
    /// poll budget (or faulted); the caller's containment still runs — a
    /// half-quiesced controller is exactly why capability retirement is
    /// unconditional.
    pub fn shutdown(&mut self, max_polls: u32, mut pause: impl FnMut()) -> Result<()> {
        // The host memory buffer goes first, while the controller can still
        // take an admin command.  A refused disable is not fatal to the
        // shutdown: once `CSTS.SHST` reports complete the controller masters
        // nothing, so the frames are safe — and they stay retained in
        // `self.hmb` until `into_seam` regardless, which is what makes a
        // silent device harmless rather than a use-after-release.
        let _ = self.disable_host_memory_buffer(max_polls, &mut pause);
        let cc = self.regs.read32(REG_CC);
        self.regs
            .write32(REG_CC, (cc & !CC_SHN_MASK) | CC_SHN_NORMAL);
        Self::wait_csts(
            &self.regs,
            |csts| csts & CSTS_SHST_MASK == CSTS_SHST_COMPLETE,
            max_polls,
            &mut pause,
        )?;
        let cc = self.regs.read32(REG_CC);
        self.regs.write32(REG_CC, cc & !(CC_EN | CC_SHN_MASK));
        Self::wait_csts(
            &self.regs,
            |csts| csts & CSTS_RDY == 0,
            max_polls,
            &mut pause,
        )
    }

    /// Release the platform seam — the unbind half of the driver-host
    /// containment contract ([`kore_driver::host::DeviceBinding::unbind`]
    /// must reconstruct the resource set it was bound from).  Drops the
    /// admin qpair (its ring buffers are pool slices with no lifecycle of
    /// their own) and returns the register window and DMA pool.
    ///
    /// Call after [`shutdown`](Self::shutdown): a live controller still owns
    /// the admin queues this drops.
    pub fn into_seam(self) -> (R, D) {
        let Self { regs, pool, .. } = self;
        (regs, pool)
    }
}

#[cfg(test)]
mod tests {
    use super::mdts_bytes;
    use crate::Error;

    #[test]
    fn mdts_uses_cap_mpsmin_units() {
        assert_eq!(mdts_bytes(0, 9), Ok(0));
        assert_eq!(mdts_bytes(5, 0), Ok(128 * 1024));
        assert_eq!(mdts_bytes(5, 1), Ok(256 * 1024));
    }

    #[test]
    fn hostile_mdts_shift_refuses() {
        assert_eq!(mdts_bytes(u8::MAX, 0), Err(Error::Geometry));
    }

    use super::{Controller, HMB_MAX_CHUNKS};
    use crate::mock::{MockBuf, MockNvmeDevice};
    use crate::seam::DmaPool;
    use kore_driver::block::CountedPacer;

    /// A live mock controller, identified, advertising a DRAM-less
    /// appetite: `preferred`/`min` in 4 KiB units, at most `max` chunks.
    fn dram_less(
        preferred: u32,
        min: u32,
        max: u16,
    ) -> (
        MockNvmeDevice,
        Controller<MockNvmeDevice, crate::mock::MockPool>,
    ) {
        let dev = MockNvmeDevice::new();
        dev.set_hmb(preferred, min, max);
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        (dev, ctrl)
    }

    fn chunks(
        ctrl: &Controller<MockNvmeDevice, crate::mock::MockPool>,
        sizes: &[usize],
    ) -> [Option<MockBuf>; HMB_MAX_CHUNKS] {
        let mut out = [const { None }; HMB_MAX_CHUNKS];
        for (slot, bytes) in sizes.iter().enumerate() {
            out[slot] = Some(ctrl.pool().alloc_aligned(*bytes, 4096).unwrap());
        }
        out
    }

    #[test]
    fn identify_decodes_the_host_memory_buffer_appetite_in_bytes() {
        let (_dev, ctrl) = dram_less(16_384, 1_024, 8);
        let info = ctrl.info();
        assert_eq!(
            info.hmb_preferred_bytes,
            64 << 20,
            "HMPRE is in 4 KiB units"
        );
        assert_eq!(info.hmb_min_bytes, 4 << 20);
        assert_eq!(info.hmb_max_descriptors, 8);
        assert_eq!(info.hmb_bytes, 0, "nothing enabled yet");
    }

    #[test]
    fn a_host_memory_buffer_is_enabled_from_page_chunks_and_disabled_first_at_shutdown() {
        // 16 KiB preferred, 8 KiB minimum, four descriptors at most.
        let (dev, mut ctrl) = dram_less(4, 2, 4);
        let table = chunks(&ctrl, &[8192, 8192]);
        let list = ctrl.pool().alloc(4096).unwrap();
        let mut pacer = CountedPacer::new(16, || {});
        assert_eq!(
            ctrl.enable_host_memory_buffer(list, table, &mut pacer),
            Ok(16_384)
        );
        // HSIZE is in the configured page (4 KiB here): 4 pages, 2 entries.
        assert_eq!(dev.hmb(), Some((4, 2)));
        let hmb = ctrl.host_memory_buffer().expect("retained while enabled");
        assert_eq!(hmb.bytes(), 16_384);
        assert_eq!(hmb.chunks(), 2);
        assert_eq!(ctrl.info().hmb_bytes, 16_384);
        // A second enable while one stands is refused, not stacked.
        let again = chunks(&ctrl, &[8192, 8192]);
        let list2 = ctrl.pool().alloc(4096).unwrap();
        assert_eq!(
            ctrl.enable_host_memory_buffer(list2, again, &mut pacer),
            Err(Error::Geometry)
        );
        // Shutdown disables the buffer before CC.SHN, and releases it.
        ctrl.shutdown(16, || {}).unwrap();
        assert_eq!(dev.hmb(), None);
        assert_eq!(dev.hmb_disables(), 1);
        assert!(ctrl.host_memory_buffer().is_none());
        assert_eq!(ctrl.info().hmb_bytes, 0);
    }

    #[test]
    fn a_table_the_controller_would_reject_is_refused_before_it_is_sent() {
        // Too many descriptors for HMMAXD.
        let (dev, mut ctrl) = dram_less(4, 1, 1);
        let table = chunks(&ctrl, &[8192, 8192]);
        let list = ctrl.pool().alloc(4096).unwrap();
        let mut pacer = CountedPacer::new(16, || {});
        assert_eq!(
            ctrl.enable_host_memory_buffer(list, table, &mut pacer),
            Err(Error::Geometry)
        );
        assert_eq!(dev.hmb(), None, "nothing reached the device");

        // Below the controller's minimum.
        let (dev, mut ctrl) = dram_less(8, 4, 0);
        let table = chunks(&ctrl, &[4096]);
        let list = ctrl.pool().alloc(4096).unwrap();
        assert_eq!(
            ctrl.enable_host_memory_buffer(list, table, &mut pacer),
            Err(Error::Geometry)
        );
        assert_eq!(dev.hmb(), None);

        // A chunk that is not a whole number of the CONFIGURED page: an
        // 8 KiB-page controller handed a 4 KiB chunk (the pool cannot
        // produce a non-4 KiB length, so the page is what moves).
        let dev = MockNvmeDevice::new().with_mps(1).unwrap();
        dev.set_hmb(4, 0, 0);
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        assert_eq!(ctrl.page().bytes(), 8192);
        let mut table = [const { None }; HMB_MAX_CHUNKS];
        table[0] = Some(ctrl.pool().alloc_aligned(4096, 8192).unwrap());
        let list = ctrl.pool().alloc(4096).unwrap();
        assert_eq!(
            ctrl.enable_host_memory_buffer(list, table, &mut pacer),
            Err(Error::Geometry)
        );
        assert_eq!(dev.hmb(), None);
    }

    #[test]
    fn a_controller_without_a_buffer_shuts_down_without_a_disable() {
        let (dev, mut ctrl) = dram_less(0, 0, 0);
        ctrl.shutdown(16, || {}).unwrap();
        assert_eq!(dev.hmb_disables(), 0);
    }
}
