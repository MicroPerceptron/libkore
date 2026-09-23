//! **NvmeSwap** — the disk swap backend: `SystemService<Request =
//! SwapRequest>` over the kore-nvme core, the lessor loop's disk tier (the
//! claim made when the vocabulary was designed, now cashed: the loop
//! takes this backend **without changing a line**).
//!
//! Target performance is structural, not bolted on—but the target is not yet
//! a production claim:
//!
//! - **Page bytes transit a bounce page this driver owns.**  They reach
//!   the service through the client's staging window (a plain CPU
//!   mapping), and there is no edge from a mapped view to a
//!   device-visible address in this driver's domain, so `Out` copies
//!   staging → [`bounce`](NvmeSwap::bounce) → DMA and `In` reverses it.
//!   One 4 KiB memcpy per page, and every byte the controller touches
//!   lies in frames this driver published to it at intake — the honest
//!   alternative to casting a CPU pointer into an IOVA.
//! - **O(1) slots.**  A slot is `blocks_per_page` LBAs; allocation is a
//!   free-stack pop (else a watermark bump), release is a push, and
//!   stored-ness is one bit in a packed bitmap — no scans anywhere.
//! - **Batched, deferred discards.**  `Discard` retires the slot
//!   immediately but queues the DSM range; one Dataset Management command
//!   deallocates up to [`DSM_BATCH`] ranges when the table fills (or on
//!   [`flush_discards`](NvmeSwap::flush_discards)).  The table page is
//!   allocated once at bind and reused.
//! - **No flush on the swap path.**  Swap pages need no crash durability
//!   (a lost swap page is lost RAM either way), so `Out` never pays for
//!   `Flush`; write-back caching in the device stays fully effective.
//! - **Single-PRP transfers.**  A 4 KiB page at the controller's 4 KiB+
//!   MPS is always PRP1-only — no list allocation on the hot path.
//!
//! **The completion wait is a seam** ([`CompletionPacer`]): the ops poll
//! the completion queue and pace between empty polls however the caller
//! chose — the bounded spin ([`SpinPacer`]: host tests, and the metal
//! fallback when no interrupt route was realized) or a real park on the
//! task's held MSI-X route (`kore-svc-swap-nvme`'s pacer, the production
//! posture).  Unit tests complete against the `cfg(test)` controller
//! double at the spin posture; test completion is evidence of the core's
//! logic, never of interrupt delivery — that proof lives in the service's
//! readiness probe and the boot gate's route witness.

use alloc::vec::Vec;

use kore_abi::device::IovaAddr;
use kore_abi::svc::{SwapRepWire, SwapReqWire};
use kore_abi::task::{Bootstrap, Profile, RoleSlot, role};
use kore_rt::service::{BindError, Bundle, Serve, SystemService};
use kore_rt::svc::SvcProtocol;
use kore_svc_swap::{SwapRefusal, SwapRequest, SwapResponse};

use crate::commands::DsmRange;
use crate::seam::{Dma, DmaPool, Regs};
use crate::{Controller, Namespace, Qpair};

/// The swap page (the lessor's unit — matches `kore-svc-mm`'s `PAGE_BYTES`).
pub const PAGE_BYTES: u64 = 4096;

/// LBAs per swap page for a namespace of `block_size`-byte blocks — or
/// `None` when the format cannot carry whole pages at all: a zero size
/// (identify's own overflow sentinel), one past [`PAGE_BYTES`], or one
/// that is not a power of two.
///
/// Identify data is **device-controlled input**, so the shape is
/// validated, never assumed: dividing by an unvalidated block size is how
/// a hostile or broken descriptor (LBADS 0, or 13+) becomes a panic in
/// the swap row.  `None` here means this backend must do no LBA
/// arithmetic on the namespace — the class binding refuses bring-up on
/// it, and a generic bind over it yields a zero-capacity store.
pub fn blocks_per_page(block_size: u32) -> Option<u32> {
    (block_size.is_power_of_two() && (block_size as u64) <= PAGE_BYTES)
        .then(|| (PAGE_BYTES / block_size as u64) as u32)
}

/// Word `i` of the self-test's probe pattern for `seed` — the splitmix64
/// finalizer over `seed + i`, so every word differs and the whole page is
/// a function of the boot-unique seed (see [`NvmeSwap::self_test`] for
/// why fixed bytes would be a false-pass hazard).
fn probe_word(seed: u64, i: usize) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Discard ranges accumulated before one DSM command deallocates them
/// all (the table page holds up to 256; 64 amortizes well while keeping
/// reclaim latency bounded).
pub const DSM_BATCH: usize = 64;

/// The completion-wait seam and its polled posture, hoisted to the
/// hardware-neutral block crate so every channel consumer — this swap face,
/// the filesystem's direct provider — waits through one contract.  The
/// re-export preserves this module's original paths; the semantics
/// (including the pace-`false` → one final poll discipline) are unchanged.
pub use kore_driver::block::{CompletionPacer, MAX_POLLS, SpinPacer};

/// The backend's visible books.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NvmeSwapStats {
    /// Pages currently stored on disk.
    pub stored_pages: u32,
    /// Slot capacity (the swap partition size, in pages).
    pub capacity_pages: u32,
    /// Pages written out / restored in / discarded, lifetime.
    pub outs: u64,
    /// Pages restored.
    pub ins: u64,
    /// Pages discarded.
    pub discards: u64,
    /// DSM commands issued (each covers up to [`DSM_BATCH`] ranges).
    pub dsm_commands: u64,
    /// Ops refused [`SwapRefusal::Backend`] at the device layer — a failed
    /// status, a completion that never came ([`timeouts`](Self::timeouts)
    /// counts those root events), or a submit the wedge blocked.
    pub io_errors: u64,
    /// Commands that never completed inside their wait budget.  Each one
    /// wedges the datapath until its late completion is reaped — see
    /// [`NvmeSwap::io`].
    pub timeouts: u64,
}

/// The endowment: a brought-up controller + one I/O qpair + the slot budget.
///
/// This constructor-shaped generic bundle is host-testable but cannot resolve
/// itself from a production bootstrap.  The concrete class adapter is
/// `kore_svc_swap_nvme::NvmeSwapBinding` (`services/swap-nvme`): it consumes
/// the realized device resources, brings up the controller, creates the
/// qpair, and retains the resulting affine authority tree.
/// **The designated swap region** — the block range this backend owns.
///
/// It exists because the alternative destroyed a disk.  This store used to
/// address the namespace **raw**: slot `n` meant LBA `n × blocks_per_page`,
/// so slot 0 was LBA 0 — the protective MBR and primary GPT header — and
/// the whole store was "every block on the device".  On QEMU that was an
/// empty scratch image and nothing complained; on a real machine the first
/// write took the partition table with it.
///
/// A region must therefore be **given**, never assumed, and
/// [`base_lba`](Self::base_lba) is validated non-zero: a swap store that
/// can address LBA 0 is not a bug to be careful around, it is a shape that
/// must not be representable.  The production row binds with `None` — a
/// zero-capacity store that refuses every write — and earns a region only
/// by reading the machine's partition table
/// ([`NvmeSwap::discover_region`]): the blocks a human or an installer
/// designated with the kore type GUID, and no others.  An unprovisioned
/// disk leaves the store empty for the row's whole life, which is the
/// only honest posture for a driver that cannot prove which blocks are
/// its.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapRegion {
    /// First LBA of the region — **never zero**.
    base_lba: u64,
    /// Region length in blocks.
    blocks: u64,
}

impl SwapRegion {
    /// A region at `base_lba` spanning `blocks` — `None` for a zero base
    /// (LBA 0 is never swap), an empty span, or arithmetic that would wrap
    /// past the end of the device.
    pub fn new(base_lba: u64, blocks: u64) -> Option<Self> {
        if base_lba == 0 || blocks == 0 || base_lba.checked_add(blocks).is_none() {
            return None;
        }
        Some(Self { base_lba, blocks })
    }

    /// First LBA of the region.
    pub fn base_lba(&self) -> u64 {
        self.base_lba
    }

    /// Region length in blocks.
    pub fn blocks(&self) -> u64 {
        self.blocks
    }
}

pub struct NvmeSwapEndowment<R: Regs, D: DmaPool> {
    /// The controller (owns the seam + admin qpair).
    pub ctrl: Controller<R, D>,
    /// The I/O qpair swap traffic rides.
    pub qp: Qpair<D::Buf, D::Buf>,
    /// **The block range this store owns**, or `None` when nothing has
    /// designated one — then the store binds at zero capacity and every
    /// `Out` refuses [`SwapRefusal::StoreFull`], which is exactly true: a
    /// store with no region has no room.  See [`SwapRegion`].
    pub region: Option<SwapRegion>,
    /// Swap capacity ceiling in pages (clamped to the region).
    pub capacity_pages: u32,
}

impl<R: Regs, D: DmaPool> Bundle for NvmeSwapEndowment<R, D> {
    /// The driver profile: **the DMA pool and nothing else**
    /// (`res/DEVICE_SERVICE.md` §10).  No `role::DEVICE`: a driver row
    /// *claims* its controller from the device broker at startup, so the
    /// founder never chooses an NVMe function — exactly the input row's
    /// posture.  (Swap adds no extra role — the store capacity is a
    /// bind-time parameter, not a founded cap.)
    const PROFILE: Profile = Profile::new("nvme-swap", &[RoleSlot::at(role::DMA_POOL)]);

    /// Fail-closed for the **generic** endowment — deliberately, and not for
    /// lack of the intake.  The device-cap seam now exists
    /// (`kore_driver::host::DeviceEndowment<B>`, `res/DEVICE_SERVICE.md` §8): a userland
    /// swap task resolves it to `CapRegs`/`CapPool`, then `Controller::bring_up`
    /// over that seam.  But `NvmeSwapEndowment<R, D>` is *generic* over the
    /// seam (the mock uses `MockNvmeDevice`/`MockPool`; the metal task uses
    /// `CapRegs`/`CapPool`), so this impl cannot pick a concrete `R`/`D` — the
    /// concrete `NvmeSwap<CapRegs, CapPool>` task does the resolve + bring-up,
    /// and that bring-up is an **on-metal path** (real MMIO behind the BAR; a
    /// byte-backed mock BAR does not drive an NVMe controller).  Host tests
    /// build the endowment and [`SystemService::bind`] it directly.
    fn resolve(_boot: &Bootstrap<'_>) -> Result<Self, BindError> {
        Err(BindError::MissingRole {
            role: role::DEVICE,
            index: 0,
        })
    }
}

/// The disk swap service.  See the module docs for the performance
/// contract.
pub struct NvmeSwap<R: Regs, D: DmaPool> {
    ctrl: Controller<R, D>,
    qp: Qpair<D::Buf, D::Buf>,
    ns: Namespace,
    /// LBAs per page-slot.
    blocks_per_page: u32,
    /// First LBA of the designated region — every slot is relative to it,
    /// so no slot can ever address LBA 0 (see [`SwapRegion`]).  Zero only
    /// when `capacity == 0`, where no I/O is reachable at all.
    base_lba: u64,
    /// Slot capacity — zero until a region is discovered.
    capacity: u32,
    /// The bind-time capacity ceiling a discovered region is clamped to
    /// (bounding this store's own bookkeeping, not the disk).
    capacity_ceiling: u32,
    /// Slots ever touched (allocation watermark — slots below it are on
    /// the free stack or stored).
    watermark: u32,
    /// Retired slots ready for reuse (O(1) pop).
    free: Vec<u32>,
    /// Stored-ness, one bit per slot (u64-packed).
    stored: Vec<u64>,
    /// Deferred deallocate ranges + the reused DMA table page.
    pending_discards: Vec<DsmRange>,
    dsm_table: D::Buf,
    /// The reused transfer page — see [`bounce`](Self::bounce).
    bounce: D::Buf,
    /// The cid of a command that timed out and is still owed a completion
    /// — **the wedge**.  While set, no new command may be submitted: the
    /// device may yet DMA the bounce page the stale command's PRP names,
    /// so reusing it would hand a client bytes the device is still
    /// writing (or write a torn page to disk).  Cleared when the late
    /// completion is finally reaped ([`io`](Self::io)'s entry drain).
    wedged: Option<u16>,
    stats: NvmeSwapStats,
}

impl<R: Regs, D: DmaPool> NvmeSwap<R, D> {
    #[inline]
    /// The device LBA of slot `slot` — **relative to the designated
    /// region**, so slot 0 is the region's first block and never LBA 0.
    fn lba(&self, slot: u32) -> u64 {
        self.base_lba + slot as u64 * self.blocks_per_page as u64
    }

    #[inline]
    fn is_stored(&self, slot: u32) -> bool {
        (slot as usize) < self.stored.len() * 64
            && self.stored[slot as usize / 64] & (1 << (slot % 64)) != 0
    }

    #[inline]
    fn set_stored(&mut self, slot: u32, on: bool) {
        let (i, b) = (slot as usize / 64, 1u64 << (slot % 64));
        if on {
            self.stored[i] |= b;
        } else {
            self.stored[i] &= !b;
        }
    }

    /// Submit one command and drive its completion: poll the queue, and
    /// between empty polls wait however the [`CompletionPacer`] says —
    /// a spin hint (the polled posture) or a real park on the held MSI-X
    /// route.  On a spent pacer, one final drain before refusing: a
    /// completion that landed without a delivered interrupt is still a
    /// completion.
    ///
    /// **A completion answers only its own command.**  Every wait matches
    /// on the cid the submit stamped (the admin path's `submit_and_wait`
    /// guard, which this path used to lack): after a timeout, the timed-out
    /// command's completion still arrives *eventually*, and without the
    /// match the next op would claim it — returning before its own
    /// transfer finished, which for a restore means handing the client
    /// whatever the bounce page held before the device wrote it.
    ///
    /// **A timeout wedges the datapath.**  The unfinished command's PRP
    /// names the bounce page (or the DSM table), and the device may DMA it
    /// at any moment until the completion is reaped — so while one is
    /// outstanding, no new command is submitted and every op refuses
    /// [`SwapRefusal::Backend`].  The wedge clears itself: each call first
    /// drains the queue, and reaping the late completion (the device is
    /// done with its DMA by then — NVMe completes *after* the transfer)
    /// reopens the path.  A device that never completes leaves the row
    /// serving honest refusals, which is what a hung disk is.
    fn io(
        &mut self,
        cmd: crate::commands::NvmeSqe,
        cookie: u64,
        pacer: &mut dyn CompletionPacer,
    ) -> Result<(), SwapRefusal> {
        self.io_paced(cmd, cookie, pacer, false)
    }

    /// [`io`](Self::io) with the completion wait's ORDER chosen by the
    /// caller.  `park_first = false` is the hot path: poll before pacing,
    /// because a served op must never yield the CPU when its answer is
    /// already in the queue.  `park_first = true` is the **delivery
    /// witness's** order — one pace *before* the first poll.
    ///
    /// It exists because the witness raced its own evidence: under a
    /// synchronous device model (QEMU TCG runs the NVMe bottom-half on
    /// the vCPU's own thread) the completion can be posted before the
    /// submitting instruction stream reaches its first poll, so the
    /// poll-first loop returns without ever parking and the route's fire
    /// count stays zero — [`note::NO_DELIVERY`] reported against a
    /// machine whose delivery works.  KVM's separate iothread usually
    /// hides the window; any code-size change re-rolls it (CI caught it,
    /// local runs did not).  The kernel LATCHES a fire that arrives
    /// before its park (`runtime/irq`: `fire` on an empty waiter slot
    /// records `pending`; `register` consumes it and completes
    /// immediately), so parking first is correct in both orders: the
    /// park resolves `Fired` whether the interrupt beat it or not, and
    /// `fired == 0` after a park-first probe genuinely means
    /// undelivered.
    fn io_paced(
        &mut self,
        cmd: crate::commands::NvmeSqe,
        cookie: u64,
        pacer: &mut dyn CompletionPacer,
        park_first: bool,
    ) -> Result<(), SwapRefusal> {
        // Reap late completions first — the only lawful wedge release.
        while let Some(c) = self.qp.poll(self.ctrl.regs()) {
            if self.wedged == Some(c.cid) {
                self.wedged = None;
            }
        }
        if self.wedged.is_some() {
            // The timed-out command is still the device's: its DMA may yet
            // touch the bounce page, so nothing new may be staged there.
            self.stats.io_errors += 1;
            return Err(SwapRefusal::Backend);
        }
        let cid = self
            .qp
            .submit_one(cmd, cookie, self.ctrl.regs())
            .map_err(|_| SwapRefusal::Backend)?;
        if park_first {
            // The witness order (see above).  A spent budget here is not
            // a refusal yet — the loop's final polls still run.
            let _ = pacer.pace();
        }
        loop {
            match self.qp.poll(self.ctrl.regs()) {
                // A predecessor's stray completion is never this op's
                // answer — drain it and poll again at once.
                Some(c) if c.cid != cid => continue,
                Some(c) if c.is_success() => return Ok(()),
                Some(_) => {
                    self.stats.io_errors += 1;
                    return Err(SwapRefusal::Backend);
                }
                None => {
                    if !pacer.pace() {
                        break;
                    }
                }
            }
        }
        while let Some(c) = self.qp.poll(self.ctrl.regs()) {
            if c.cid != cid {
                continue;
            }
            if c.is_success() {
                return Ok(());
            }
            self.stats.io_errors += 1;
            return Err(SwapRefusal::Backend);
        }
        // Never completed: the command — and the bounce page its PRP
        // names — stays the device's until the late completion is reaped.
        self.wedged = Some(cid);
        self.stats.timeouts += 1;
        self.stats.io_errors += 1;
        Err(SwapRefusal::Backend)
    }

    /// The **bounce buffer** — one page of this driver's own DMA pool,
    /// allocated at bind and reused for every transfer.
    ///
    /// Why a copy and not DMA straight from the tenant's frames: a page's
    /// bytes reach this service through the client's staging window (a
    /// plain CPU mapping), and there is no edge from a mapped view to a
    /// device-visible address in this driver's IOMMU domain — the
    /// `Derive × kind = IovaRegion` leaf takes a `MemLease` lessor
    /// position, which a receiver of a view does not hold, and minting a
    /// sub-lease to fake one is barred on purpose (ckG2b-1: a sub is
    /// never registered for reachability, so a revoke could not clear the
    /// device mapping).
    ///
    /// So the page transits memory this driver *does* own end to end: the
    /// pool window was carved, mapped, and published to the device at
    /// intake, so `iova()` names the very frames `cpu()` writes.  One
    /// 4 KiB memcpy per page — the same copy the compressing sibling pays
    /// anyway, and the honest alternative to a cast that would have the
    /// controller DMA to whatever physical address a CPU pointer happens
    /// to equal.
    fn bounce(&self) -> (*mut u8, IovaAddr) {
        (self.bounce.cpu(), self.bounce.iova())
    }

    fn page_out_from(
        &mut self,
        page: &[u8],
        pacer: &mut dyn CompletionPacer,
    ) -> Result<u32, SwapRefusal> {
        if page.len() != PAGE_BYTES as usize {
            return Err(SwapRefusal::BadPage);
        }
        // Stage the bytes in the driver's own device-visible frames.
        let (cpu, iova) = self.bounce();
        // SAFETY: `bounce` is a whole pool-allocated page (>= PAGE_BYTES,
        // page-aligned) this driver owns exclusively, and `page` is a
        // distinct staging borrow — the regions cannot overlap.
        unsafe { core::ptr::copy_nonoverlapping(page.as_ptr(), cpu, PAGE_BYTES as usize) };
        // O(1) slot: free-stack pop, else watermark bump.
        let token = match self.free.pop() {
            Some(t) => t,
            None if self.watermark < self.capacity => {
                let t = self.watermark;
                self.watermark += 1;
                t
            }
            None => return Err(SwapRefusal::StoreFull),
        };
        // The controller reads the bounce frames — memory this driver
        // published to the device at intake.
        let cmd = self
            .ns
            .write(self.lba(token), self.blocks_per_page, iova, None)
            .map_err(|_| SwapRefusal::Backend)?;
        match self.io(cmd, token as u64, pacer) {
            Ok(()) => {
                self.set_stored(token, true);
                self.stats.stored_pages += 1;
                self.stats.outs += 1;
                Ok(token)
            }
            Err(e) => {
                // The slot was never stored — hand it straight back.
                self.free.push(token);
                Err(e)
            }
        }
    }

    fn page_in_into(
        &mut self,
        token: u32,
        page: &mut [u8],
        pacer: &mut dyn CompletionPacer,
    ) -> Result<(), SwapRefusal> {
        if !self.is_stored(token) {
            return Err(SwapRefusal::BadToken);
        }
        if page.len() != PAGE_BYTES as usize {
            return Err(SwapRefusal::BadDst);
        }
        let (cpu, iova) = self.bounce();
        // The controller writes the bounce frames; the bytes reach the
        // client through the staging copy below.
        let cmd = self
            .ns
            .read(self.lba(token), self.blocks_per_page, iova, None)
            .map_err(|_| SwapRefusal::Backend)?;
        self.io(cmd, token as u64, pacer)?;
        // SAFETY: as `page_out` — a driver-owned whole page, and `page`
        // is a distinct exclusive staging borrow, so no overlap.
        unsafe { core::ptr::copy_nonoverlapping(cpu, page.as_mut_ptr(), PAGE_BYTES as usize) };
        // Restore consumes the stored copy (the disk bytes are stale the
        // moment the slot is reusable — the next Out overwrites, so no
        // DSM is owed here).
        self.set_stored(token, false);
        self.free.push(token);
        self.stats.stored_pages -= 1;
        self.stats.ins += 1;
        Ok(())
    }

    fn discard(&mut self, token: u32, pacer: &mut dyn CompletionPacer) -> Result<(), SwapRefusal> {
        if !self.is_stored(token) {
            return Err(SwapRefusal::BadToken);
        }
        self.set_stored(token, false);
        self.free.push(token);
        self.stats.stored_pages -= 1;
        self.stats.discards += 1;
        // Deferred DSM: queue the range; one command reclaims the batch.
        self.pending_discards
            .push(DsmRange::new(self.lba(token), self.blocks_per_page));
        if self.pending_discards.len() >= DSM_BATCH {
            self.flush_discards_paced(pacer);
        }
        Ok(())
    }

    /// Issue the batched Dataset Management deallocate for every pending
    /// discard (public: the host loop flushes on idle) — the spin-paced
    /// form; the serve loop's paced sibling is
    /// [`flush_discards_paced`](Self::flush_discards_paced).
    pub fn flush_discards(&mut self) {
        self.flush_discards_paced(&mut SpinPacer::default());
    }

    /// [`flush_discards`](Self::flush_discards) over a caller-chosen
    /// pacer.  A DSM failure is
    /// deliberately non-fatal — deallocate is advisory (the blocks are
    /// already logically free); the ranges are dropped either way and the
    /// error stays visible in the books.
    pub fn flush_discards_paced(&mut self, pacer: &mut dyn CompletionPacer) {
        let n = self.pending_discards.len();
        if n == 0 {
            return;
        }
        debug_assert!(n <= 256, "a DSM table page holds 256 ranges");
        // Write the range table into the reused DMA page.
        // SAFETY: dsm_table is a CPU-visible page ≥ n × 16 bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.pending_discards.as_ptr(),
                self.dsm_table.cpu() as *mut DsmRange,
                n,
            );
        }
        if let Ok(cmd) = self.ns.deallocate(self.dsm_table.iova(), n as u16) {
            if self.io(cmd, u64::MAX, pacer).is_ok() {
                self.stats.dsm_commands += 1;
            }
        }
        self.pending_discards.clear();
    }

    /// Serve one **byte-moving** swap op over the client's staging window
    /// and a caller-chosen pacer — the form the production serve loop
    /// drives: the completion wait inside is a park on the held MSI-X
    /// route when the caller's pacer carries one.
    ///
    /// `staging` is the window the row adopted at
    /// [`SwapRequest::Attach`]; the row owns that mapping (it is
    /// capability-shaped and this core is not), so it arrives here as
    /// plain bytes.  `None` — or a page index outside the window —
    /// refuses [`SwapRefusal::NoStaging`]: the index is client data, and
    /// the window's length is the only authority on it.
    ///
    /// `Attach` itself is **not** handled here.  Adopting a transferred
    /// capability is the row's business (`kore-svc-swap-nvme`), and this
    /// core holds no session authority at all.
    pub fn handle(
        &mut self,
        req: SwapRequest,
        staging: Option<&mut [u8]>,
        pacer: &mut dyn CompletionPacer,
    ) -> SwapResponse {
        match req {
            SwapRequest::Attach { .. } => SwapResponse::Refused(SwapRefusal::NoStaging),
            SwapRequest::Out { page } => {
                let Some(bytes) = staging
                    .as_deref()
                    .and_then(|s| kore_svc_swap::staging_page(s, page))
                else {
                    return SwapResponse::Refused(SwapRefusal::NoStaging);
                };
                // The staging borrow ends with this copy (page_out stages
                // into the bounce buffer immediately), so the driver's own
                // `&mut self` and the window never alias.
                match self.page_out_from(bytes, pacer) {
                    Ok(token) => SwapResponse::Stored { token },
                    Err(r) => SwapResponse::Refused(r),
                }
            }
            SwapRequest::In { token, page } => {
                let mut staging = staging;
                let Some(bytes) = staging
                    .as_deref_mut()
                    .and_then(|s| kore_svc_swap::staging_page_mut(s, page))
                else {
                    return SwapResponse::Refused(SwapRefusal::NoStaging);
                };
                match self.page_in_into(token, bytes, pacer) {
                    Ok(()) => SwapResponse::Restored,
                    Err(r) => SwapResponse::Refused(r),
                }
            }
            SwapRequest::Discard { token } => match self.discard(token, pacer) {
                Ok(()) => SwapResponse::Discarded,
                Err(r) => SwapResponse::Refused(r),
            },
            // The books are the ROW's business, like `Attach`: the core
            // knows its own stats but not the route posture or the
            // staging window, and half an answer would read as whole.
            SwapRequest::Status => SwapResponse::Refused(SwapRefusal::Backend),
        }
    }

    /// **The I/O-path self-test**, driven through `pacer` — a
    /// write/read/verify round-trip when the store owns blocks to write,
    /// a bounded read when it owns none.
    ///
    /// The write leg once ran unconditionally, and with no region concept
    /// "slot 0" was LBA 0 — every boot overwrote the machine's protective
    /// MBR and primary GPT header, and QEMU's scratch image made it
    /// invisible.  The line that survives: **a driver may write only
    /// inside a region something designated as its** ([`SwapRegion`]).
    /// So:
    ///
    /// - **No region** (or slot 0 holds a stored page): one read of the
    ///   first addressable block, bytes discarded unread — proves the
    ///   queue and, route-parked, the MSI-X delivery, and changes
    ///   nothing.
    /// - **A designated region with slot 0 free**: write a `seed`-derived
    ///   pattern to the region's first slot, clobber the bounce page (a
    ///   read that silently no-ops must not pass), read the slot back,
    ///   and compare — the complete datapath, bounce page → device write
    ///   → device read → bytes intact, before the row ever beacons.
    ///
    /// `seed` must be **boot-unique** (the row passes its kernel-anchored
    /// probe deadline): the pattern lands at the same LBA every boot, so
    /// a fixed pattern would let a device whose writes silently fail
    /// round-trip *last* boot's bytes and pass.  Slot 0's blocks stay
    /// scratch to the store afterwards — nothing marks them stored, and
    /// the first real `Out` to the slot overwrites them.
    pub fn self_test(
        &mut self,
        pacer: &mut dyn CompletionPacer,
        seed: u64,
    ) -> Result<(), SwapRefusal> {
        let (cpu, iova) = self.bounce();
        if self.capacity == 0 || self.is_stored(0) {
            let read = self
                .ns
                .read(self.lba(0), self.blocks_per_page, iova, None)
                .map_err(|_| SwapRefusal::Backend)?;
            // Park-first: this read IS the delivery witness (see
            // `io_paced`) — a completion that beats the first poll must
            // not zero the fire count.
            return self.io_paced(read, u64::MAX - 1, pacer, true);
        }
        // The write leg — lawful exactly here: slot 0 lies inside the
        // designated region and holds no stored page.
        let words = PAGE_BYTES as usize / 8;
        // SAFETY: the bounce page is a whole pool-allocated, page-aligned
        // (hence u64-aligned) page this driver owns exclusively.
        let bounce = unsafe { core::slice::from_raw_parts_mut(cpu.cast::<u64>(), words) };
        for (i, w) in bounce.iter_mut().enumerate() {
            *w = probe_word(seed, i);
        }
        let write = self
            .ns
            .write(self.lba(0), self.blocks_per_page, iova, None)
            .map_err(|_| SwapRefusal::Backend)?;
        // Park-first on the sequence's FIRST command — the delivery
        // witness (see `io_paced`); the read-back keeps the hot order.
        self.io_paced(write, u64::MAX - 1, pacer, true)?;
        bounce.fill(!seed);
        let read = self
            .ns
            .read(self.lba(0), self.blocks_per_page, iova, None)
            .map_err(|_| SwapRefusal::Backend)?;
        self.io(read, u64::MAX - 1, pacer)?;
        // SAFETY: as above — the read's completion was reaped, so the
        // device is done writing the page.
        let bounce = unsafe { core::slice::from_raw_parts(cpu.cast::<u64>(), words) };
        for (i, w) in bounce.iter().enumerate() {
            if *w != probe_word(seed, i) {
                self.stats.io_errors += 1;
                return Err(SwapRefusal::Backend);
            }
        }
        Ok(())
    }

    /// **Discover this store's region from the machine's partition
    /// table** — read-only, and the only thing that ever gives this store
    /// blocks to write.
    ///
    /// Reads the primary GPT header and its entry array through the
    /// driver's own bounce page, then looks for the partition whose type
    /// is [`KORE_SWAP_TYPE`](kore_driver::gpt::KORE_SWAP_TYPE).  On a
    /// match the store is provisioned in place: base LBA, capacity, and
    /// the stored-ness bitmap all come from what the table says, so the
    /// blocks this store can address are exactly the blocks a human (or an
    /// installer) designated.
    ///
    /// Every failure — no table, a damaged one, no kore partition — leaves
    /// the store at **zero capacity**, which refuses every write.  The
    /// error is returned for the row's readiness note; nothing about a
    /// failed discovery is fatal, and nothing about it writes.
    pub fn discover_region(
        &mut self,
        pacer: &mut dyn CompletionPacer,
    ) -> Result<(), kore_driver::gpt::GptError> {
        use kore_driver::gpt;

        // A namespace whose LBA format cannot carry pages admits no LBA
        // arithmetic at all (`blocks_per_page` refused it at bind) —
        // refuse before the first division by its block size.
        if self.blocks_per_page == 0 {
            return Err(gpt::GptError::BadGeometry);
        }
        // Provisioned already: the books (stored bitmap, free stack,
        // watermark) are live, and re-provisioning would reset them under
        // stored pages while `stats.stored_pages` still counts them.  One
        // designation per binding; a second read of the table is a no-op.
        if self.capacity != 0 {
            return Ok(());
        }
        let bs = self.ns.block_size as usize;
        let per_read = (self.bounce.len() / bs).max(1) as u32;

        // 1 · the primary header, at LBA 1.
        self.read_into_bounce(gpt::PRIMARY_HEADER_LBA, 1, pacer)
            .map_err(|_| gpt::GptError::NotGpt)?;
        // SAFETY: the bounce page is driver-owned and at least one block;
        // the device just wrote `bs` bytes of it.
        let block = unsafe { core::slice::from_raw_parts(self.bounce.cpu(), bs) };
        let geom = gpt::parse_header(block)?;

        // 2 · the entry array, in bounce-sized chunks.  The whole array
        //     must be checksummed together, so it is staged into a fixed
        //     buffer sized by the module's own ceiling — no allocation on
        //     a path that runs before the store exists.
        let bytes = geom.entry_array_bytes();
        if bytes > gpt::MAX_ENTRY_ARRAY_BYTES {
            return Err(gpt::GptError::BadGeometry);
        }
        let mut entries = [0u8; gpt::MAX_ENTRY_ARRAY_BYTES];
        let mut done = 0usize;
        let mut lba = geom.entry_lba;
        while done < bytes {
            let want = ((bytes - done).div_ceil(bs) as u32).min(per_read);
            self.read_into_bounce(lba, want, pacer)
                .map_err(|_| gpt::GptError::BadGeometry)?;
            let got = (want as usize * bs).min(bytes - done);
            // SAFETY: as above — `got` bytes the device just wrote into
            // the driver-owned bounce page.
            let chunk = unsafe { core::slice::from_raw_parts(self.bounce.cpu(), got) };
            entries[done..done + got].copy_from_slice(chunk);
            done += got;
            lba += want as u64;
        }

        let extent = gpt::find_partition(&entries[..bytes], &geom, gpt::KORE_SWAP_TYPE)?;
        // The table is the authority, but the device is the last word: a
        // partition running past the namespace is a designation bug.
        if extent.end_lba >= self.ns.block_count {
            return Err(gpt::GptError::BadGeometry);
        }
        let Some(region) = SwapRegion::new(extent.start_lba, extent.blocks()) else {
            return Err(gpt::GptError::BadGeometry);
        };
        self.provision(region);
        Ok(())
    }

    /// One bounded read into the bounce page.
    fn read_into_bounce(
        &mut self,
        lba: u64,
        blocks: u32,
        pacer: &mut dyn CompletionPacer,
    ) -> Result<(), SwapRefusal> {
        let (_, iova) = self.bounce();
        let cmd = self
            .ns
            .read(lba, blocks, iova, None)
            .map_err(|_| SwapRefusal::Backend)?;
        self.io(cmd, u64::MAX - 2, pacer)
    }

    /// Install a discovered region: base LBA, capacity, and the
    /// bookkeeping sized to it.  Private because a region may only come
    /// from [`discover_region`](Self::discover_region) — the store never
    /// designates its own blocks, provisions twice, or provisions over an
    /// unusable LBA format (both guarded at the discovery door).
    fn provision(&mut self, region: SwapRegion) {
        debug_assert!(self.blocks_per_page > 0, "guarded at discover_region");
        debug_assert_eq!(self.capacity, 0, "one designation per binding");
        let region_pages = (region.blocks() / self.blocks_per_page.max(1) as u64) as u32;
        self.base_lba = region.base_lba();
        self.capacity = self.capacity_ceiling.min(region_pages);
        self.watermark = 0;
        self.free.clear();
        self.stored = alloc::vec![0u64; (self.capacity as usize).div_ceil(64)];
    }

    /// Whether deferred discard ranges await their DSM — the serve loop's
    /// idle-flush predicate (see [`flush_discards`](Self::flush_discards)).
    pub fn has_pending_discards(&self) -> bool {
        !self.pending_discards.is_empty()
    }

    /// The books.
    pub fn stats(&self) -> NvmeSwapStats {
        NvmeSwapStats {
            capacity_pages: self.capacity,
            ..self.stats
        }
    }

    /// The qpair's books (submitted / completed / errors), used by the
    /// service row to compose its typed [`SwapStatus`](kore_svc_swap::SwapStatus).
    pub fn qp_stats(&self) -> crate::QpairStats {
        self.qp.stats()
    }

    /// **Quiesce the device** — flush the deferred discards, then run the
    /// controller's orderly shutdown.  The class binding's quiesce hook
    /// ([`kore_driver::host::DeviceBinding::quiesce`]): after `Ok`, the
    /// controller has stopped all DMA, so the authority tree above the seam
    /// can be retired without racing an in-flight transfer.
    pub fn quiesce(&mut self, max_polls: u32, mut pause: impl FnMut()) -> crate::Result<()> {
        self.flush_discards();
        self.ctrl.shutdown(max_polls, &mut pause)
    }

    /// Release the platform seam — the unbind half of the containment
    /// contract.  Drops the I/O qpair and the DSM table page (pool slices
    /// with no lifecycle of their own) and passes through the controller's
    /// own [`into_seam`](crate::Controller::into_seam).
    pub fn into_seam(self) -> (R, D) {
        let Self { ctrl, qp, .. } = self;
        drop(qp);
        ctrl.into_seam()
    }
}

/// The **wire** half — the swap vocabulary this backend speaks, cited
/// from the vocabulary's own crate exactly as `kore-svc-swap-nvme`'s
/// production composition does: *one wire form, N backend services*.
/// Every codec delegates rather than re-encoding, so the zram tier, the
/// production disk row, and this driver-core face answer the same rings
/// byte-for-byte — which is what lets a pager client take any of them
/// without changing a line.
///
/// A swap backend can mint no authority
/// ([`REPLY_GRANTS`](SvcProtocol::REPLY_GRANTS) `= 0`): it holds pages.
impl<R: Regs, D: DmaPool> SvcProtocol for NvmeSwap<R, D> {
    type Request = SwapRequest;
    type Response = SwapResponse;
    type ReqWire = SwapReqWire;
    type RepWire = SwapRepWire;

    const REPLY_GRANTS: usize = 0;

    fn encode_req(req: &SwapRequest) -> SwapReqWire {
        kore_svc_swap::wire::encode_req(req)
    }
    fn decode_req(wire: &SwapReqWire) -> Option<SwapRequest> {
        kore_svc_swap::wire::decode_req(wire)
    }
    fn encode_rep(rep: &SwapResponse, grants: u8) -> SwapRepWire {
        kore_svc_swap::wire::encode_rep(rep, grants)
    }
    fn decode_rep(wire: &SwapRepWire) -> Option<(SwapResponse, u8)> {
        kore_svc_swap::wire::decode_rep(wire)
    }
}

impl<R: Regs, D: DmaPool> SystemService for NvmeSwap<R, D> {
    const NAME: &'static str = "nvme-swap";
    type Endowment = NvmeSwapEndowment<R, D>;

    fn bind(e: NvmeSwapEndowment<R, D>) -> Self {
        let ns = e.ctrl.namespace();
        // Identify data is device input — an LBA format that cannot carry
        // whole pages binds a store that must do no LBA arithmetic at all
        // (`blocks_per_page` 0 ⇒ capacity 0 below, and discovery refuses
        // before its first division).  The production class binding
        // refuses such a device outright; this guard is for the generic
        // path, where `bind` cannot fail.
        let blocks_per_page = blocks_per_page(ns.block_size).unwrap_or(0);
        // Capacity comes from the DESIGNATED REGION, never from the
        // namespace: "every block on the device" is what cost a real
        // machine its partition table.  No region ⇒ no capacity ⇒ every
        // write refuses, and `base_lba` stays 0 only because no I/O can
        // reach it.
        let (base_lba, capacity) = match e.region {
            Some(region) if blocks_per_page > 0 => {
                // Whole pages inside the region, clamped to the ceiling and
                // to what the namespace actually holds.
                let region_pages = (region.blocks() / blocks_per_page as u64) as u32;
                let fits = region
                    .base_lba()
                    .checked_add(region.blocks())
                    .is_some_and(|end| end <= ns.block_count);
                if fits {
                    (region.base_lba(), e.capacity_pages.min(region_pages))
                } else {
                    // A region past the end of the device is a designation
                    // bug; bind with no store rather than a truncated one.
                    (0, 0)
                }
            }
            _ => (0, 0),
        };
        let dsm_table = e
            .ctrl
            .pool()
            .alloc(4096)
            .expect("bind-time DSM table page (the zram bind-allocates posture)");
        let bounce = e
            .ctrl
            .pool()
            .alloc(PAGE_BYTES as usize)
            .expect("bind-time bounce page (the zram bind-allocates posture)");
        Self {
            ctrl: e.ctrl,
            qp: e.qp,
            ns,
            blocks_per_page,
            base_lba,
            capacity,
            capacity_ceiling: e.capacity_pages,
            watermark: 0,
            free: Vec::new(),
            stored: alloc::vec![0u64; (capacity as usize).div_ceil(64)],
            pending_discards: Vec::with_capacity(DSM_BATCH),
            dsm_table,
            bounce,
            wedged: None,
            stats: NvmeSwapStats::default(),
        }
    }

    /// Serve one swap op — [`handle`](Self::handle) at the spin-paced
    /// posture (the host tests' path; production drives `handle` with the
    /// route-park pacer).  Never calls [`Serve::grant`] — a swap backend
    /// holds pages, it mints nothing (the zram contract, verbatim).
    fn serve(&mut self, req: SwapRequest, sv: &mut Serve<'_, SwapResponse>) {
        // No staging at this altitude: the window is a capability the ROW
        // adopts (`kore-svc-swap-nvme`'s `DiskSwap`), and this core holds
        // no session authority.  Every byte-moving verb therefore refuses
        // here — the production entry drives `handle` with the row's
        // adopted window instead.  Kept rather than dropped so the type
        // still carries its name, endowment, and books.
        let resp = self.handle(req, None, &mut SpinPacer::default());
        sv.reply(resp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockNvmeDevice;
    use kore_svc_swap::staging_page_mut;

    /// The tests' designated region — non-zero on purpose (the whole
    /// point), and inside the mock's 64-block backing disk.
    const TEST_REGION_LBA: u64 = 8;
    const TEST_REGION_BLOCKS: u64 = 48;

    /// A bound swap face over the host mock, plus a staging window.
    fn bound(pages: usize) -> (NvmeSwap<MockNvmeDevice, crate::mock::MockPool>, Vec<u8>) {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).expect("bring-up");
        ctrl.identify(16, || {}).expect("identify");
        ctrl.set_num_queues(1, 16, || {}).expect("set queues");
        let qp = ctrl.create_io_qpair(1, 32, None, 16, || {}).expect("qpair");
        let svc = NvmeSwap::bind(NvmeSwapEndowment {
            ctrl,
            qp,
            // A region well clear of LBA 0 — the conventional first-partition
            // start, so the tests exercise the offset rather than the
            // shape that destroyed a disk.
            region: SwapRegion::new(TEST_REGION_LBA, TEST_REGION_BLOCKS),
            capacity_pages: 64,
        });
        (svc, alloc::vec![0u8; pages * PAGE_BYTES as usize])
    }

    /// **The datapath, end to end**: a page staged by the client goes out
    /// to the device through the bounce buffer and comes back byte-identical.
    /// This is what `view_iova`'s typed refusal could never do — the store
    /// and restore now move real bytes over real (mock) DMA.
    #[test]
    fn a_page_round_trips_through_the_bounce_buffer() {
        let (mut svc, mut staging) = bound(2);
        let page: Vec<u8> = (0..PAGE_BYTES as usize).map(|i| (i * 7) as u8).collect();
        staging_page_mut(&mut staging, 1)
            .unwrap()
            .copy_from_slice(&page);

        let stored = svc.handle(
            SwapRequest::Out { page: 1 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        );
        let SwapResponse::Stored { token } = stored else {
            panic!("page-out stores: {stored:?}");
        };
        assert_eq!(svc.stats().outs, 1);

        // Clobber staging: a restore that silently no-ops must not pass.
        staging_page_mut(&mut staging, 1).unwrap().fill(0xAA);
        assert_eq!(
            svc.handle(
                SwapRequest::In { token, page: 1 },
                Some(&mut staging),
                &mut SpinPacer::default()
            ),
            SwapResponse::Restored
        );
        assert_eq!(
            kore_svc_swap::staging_page(&staging, 1).unwrap(),
            &page[..],
            "the restored bytes are the stored bytes, through the device"
        );
        assert_eq!(svc.stats().ins, 1);
        assert_eq!(svc.stats().stored_pages, 0, "restore consumed the copy");
        assert_eq!(svc.stats().io_errors, 0);
    }

    /// Two pages stored back to back keep their own bytes — the bounce
    /// buffer is reused, so a second transfer must not serve the first
    /// one's contents.
    #[test]
    fn the_reused_bounce_page_does_not_leak_between_transfers() {
        let (mut svc, mut staging) = bound(2);
        staging_page_mut(&mut staging, 0).unwrap().fill(0x11);
        staging_page_mut(&mut staging, 1).unwrap().fill(0x22);

        let SwapResponse::Stored { token: a } = svc.handle(
            SwapRequest::Out { page: 0 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        ) else {
            panic!("first stores");
        };
        let SwapResponse::Stored { token: b } = svc.handle(
            SwapRequest::Out { page: 1 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        ) else {
            panic!("second stores");
        };
        assert_ne!(a, b, "distinct slots");

        staging.fill(0);
        svc.handle(
            SwapRequest::In { token: a, page: 0 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        );
        svc.handle(
            SwapRequest::In { token: b, page: 1 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        );
        assert!(
            kore_svc_swap::staging_page(&staging, 0)
                .unwrap()
                .iter()
                .all(|&x| x == 0x11)
        );
        assert!(
            kore_svc_swap::staging_page(&staging, 1)
                .unwrap()
                .iter()
                .all(|&x| x == 0x22)
        );
    }

    /// Every byte-moving verb refuses without staging, and refuses an
    /// index outside the window — the client's page index is data.
    #[test]
    fn missing_or_out_of_bounds_staging_is_a_refusal() {
        let (mut svc, mut staging) = bound(1);
        assert_eq!(
            svc.handle(
                SwapRequest::Out { page: 0 },
                None,
                &mut SpinPacer::default()
            ),
            SwapResponse::Refused(SwapRefusal::NoStaging),
            "no window"
        );
        assert_eq!(
            svc.handle(
                SwapRequest::Out { page: 1 },
                Some(&mut staging),
                &mut SpinPacer::default()
            ),
            SwapResponse::Refused(SwapRefusal::NoStaging),
            "one past the last page"
        );
        // Attach is the row's business, never the core's.
        assert_eq!(
            svc.handle(
                SwapRequest::Attach {
                    announced: kore_abi::session::SlotRef::from_wire(3)
                },
                Some(&mut staging),
                &mut SpinPacer::default()
            ),
            SwapResponse::Refused(SwapRefusal::NoStaging),
        );
    }

    /// A discarded page is unrestorable, and the DSM batch is deferred —
    /// the store's accounting stays honest across the staging rewrite.
    #[test]
    fn discard_retires_the_token_and_defers_the_dsm() {
        let (mut svc, mut staging) = bound(1);
        staging_page_mut(&mut staging, 0).unwrap().fill(0x5A);
        let SwapResponse::Stored { token } = svc.handle(
            SwapRequest::Out { page: 0 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        ) else {
            panic!("stores");
        };
        assert_eq!(
            svc.handle(
                SwapRequest::Discard { token },
                Some(&mut staging),
                &mut SpinPacer::default()
            ),
            SwapResponse::Discarded
        );
        assert_eq!(svc.stats().discards, 1);
        assert_eq!(svc.stats().dsm_commands, 0, "one discard defers its DSM");
        assert!(
            svc.has_pending_discards(),
            "the deferred range is visible to the idle-flush predicate"
        );
        assert_eq!(
            svc.handle(
                SwapRequest::In { token, page: 0 },
                Some(&mut staging),
                &mut SpinPacer::default()
            ),
            SwapResponse::Refused(SwapRefusal::BadToken),
            "a discarded token restores nothing"
        );
        svc.flush_discards();
        assert_eq!(svc.stats().dsm_commands, 1, "the flush issues the batch");
        assert!(!svc.has_pending_discards(), "the flush drained the table");
    }
}

/// The completion-attribution + wedge discipline: a timed-out command's
/// late completion must never be claimed by a successor, and the bounce
/// page stays quarantined until the device is provably done with it.
#[cfg(test)]
mod completion_tests {
    use super::*;
    use crate::mock::MockNvmeDevice;
    use kore_svc_swap::{staging_page, staging_page_mut};

    fn rig() -> (
        NvmeSwap<MockNvmeDevice, crate::mock::MockPool>,
        MockNvmeDevice,
    ) {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).expect("bring-up");
        ctrl.identify(16, || {}).expect("identify");
        ctrl.set_num_queues(1, 16, || {}).expect("set queues");
        let qp = ctrl.create_io_qpair(1, 32, None, 16, || {}).expect("qpair");
        let svc = NvmeSwap::bind(NvmeSwapEndowment {
            ctrl,
            qp,
            region: SwapRegion::new(8, 48),
            capacity_pages: 64,
        });
        (svc, dev)
    }

    /// **The regression: a successor claiming its predecessor's
    /// completion.**  A command whose completion the device withholds
    /// refuses at the pacer's edge and wedges the datapath; the next op
    /// refuses WITHOUT submitting (the bounce page is still the
    /// device's); reaping the late completion reopens the path, and the
    /// bytes that then round-trip are the successor's own.
    #[test]
    fn a_timed_out_command_wedges_until_its_completion_is_reaped() {
        let (mut svc, dev) = rig();
        let mut staging = alloc::vec![0u8; 2 * PAGE_BYTES as usize];
        staging_page_mut(&mut staging, 0).unwrap().fill(0x11);
        staging_page_mut(&mut staging, 1).unwrap().fill(0x22);

        dev.swallow_completions(1);
        assert_eq!(
            svc.handle(
                SwapRequest::Out { page: 0 },
                Some(&mut staging),
                &mut SpinPacer::new(4),
            ),
            SwapResponse::Refused(SwapRefusal::Backend),
            "no completion inside the budget"
        );
        assert_eq!(svc.stats().timeouts, 1);
        assert_eq!(svc.qp_stats().submitted, 1);

        // Wedged: the withheld command's PRP still names the bounce page,
        // so the next op must refuse BEFORE anything reaches the device —
        // without the cid match it would have claimed the stale
        // completion as its own instead.
        assert_eq!(
            svc.handle(
                SwapRequest::Out { page: 1 },
                Some(&mut staging),
                &mut SpinPacer::default(),
            ),
            SwapResponse::Refused(SwapRefusal::Backend),
            "wedged ops refuse"
        );
        assert_eq!(
            svc.qp_stats().submitted,
            1,
            "nothing new reached the device while wedged"
        );

        // The late completion lands; the next op reaps it, unwedges, and
        // serves correctly.
        dev.release_swallowed();
        let SwapResponse::Stored { token } = svc.handle(
            SwapRequest::Out { page: 1 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        ) else {
            panic!("the post-release op stores");
        };
        assert_eq!(svc.qp_stats().submitted, 2);
        staging_page_mut(&mut staging, 1).unwrap().fill(0);
        assert_eq!(
            svc.handle(
                SwapRequest::In { token, page: 1 },
                Some(&mut staging),
                &mut SpinPacer::default(),
            ),
            SwapResponse::Restored
        );
        assert!(
            staging_page(&staging, 1)
                .unwrap()
                .iter()
                .all(|&b| b == 0x22),
            "the restored bytes are the successor's, not the timed-out op's"
        );
    }

    /// The books tell the story: a timeout counts once under `timeouts`
    /// (and as an io_error), the backend remains wedged, and the late
    /// completion reopens it.
    #[test]
    fn timeouts_are_counted_and_the_wedge_is_visible() {
        let (mut svc, dev) = rig();
        let mut staging = alloc::vec![0u8; PAGE_BYTES as usize];

        dev.swallow_completions(1);
        svc.handle(
            SwapRequest::Out { page: 0 },
            Some(&mut staging),
            &mut SpinPacer::new(4),
        );
        assert_eq!(svc.stats().timeouts, 1);
        assert_eq!(svc.stats().io_errors, 1);

        assert!(svc.wedged.is_some(), "the timed-out command remains owed");

        dev.release_swallowed();
        svc.handle(
            SwapRequest::Out { page: 0 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        );
        assert!(svc.wedged.is_none(), "reaping the completion unwedges");
    }
}

/// Namespace geometry is device-controlled input: every shape identify
/// can report must decode and bind without panicking, and only formats
/// that carry whole pages may reach LBA arithmetic.
#[cfg(test)]
mod geometry_tests {
    use super::*;
    use crate::mock::MockNvmeDevice;

    #[test]
    fn blocks_per_page_validates_device_input() {
        assert_eq!(blocks_per_page(4096), Some(1));
        assert_eq!(blocks_per_page(512), Some(8));
        assert_eq!(
            blocks_per_page(1),
            Some(4096),
            "absurd but arithmetically safe"
        );
        assert_eq!(blocks_per_page(0), None, "identify's own overflow sentinel");
        assert_eq!(blocks_per_page(8192), None, "cannot carry a 4 KiB page");
        assert_eq!(blocks_per_page(3072), None, "not a power of two");
    }

    /// **The regression: `bind` divided by an unvalidated block size.**
    /// An 8 KiB LBA format binds a store with no capacity — even over a
    /// designated region — and discovery refuses before its first
    /// division; nothing panics anywhere.
    #[test]
    fn an_unusable_lba_format_binds_empty_and_never_panics() {
        let dev = MockNvmeDevice::new();
        dev.set_lbads(13); // 8 KiB blocks
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        assert_eq!(ctrl.info().block_size, 8192);
        ctrl.set_num_queues(1, 16, || {}).unwrap();
        let qp = ctrl.create_io_qpair(1, 32, None, 16, || {}).unwrap();
        let mut svc = NvmeSwap::bind(NvmeSwapEndowment {
            ctrl,
            qp,
            region: SwapRegion::new(2048, 4096),
            capacity_pages: 64,
        });
        assert_eq!(svc.stats().capacity_pages, 0, "no page math ⇒ no store");

        let mut staging = alloc::vec![0u8; PAGE_BYTES as usize];
        assert_eq!(
            svc.handle(
                SwapRequest::Out { page: 0 },
                Some(&mut staging),
                &mut SpinPacer::default(),
            ),
            SwapResponse::Refused(SwapRefusal::StoreFull)
        );
        assert_eq!(
            svc.discover_region(&mut SpinPacer::default()),
            Err(kore_driver::gpt::GptError::BadGeometry),
            "discovery refuses before dividing by the block size"
        );
    }

    /// An LBADS past the u32 width decodes to the 0 sentinel — never a
    /// shift panic — and 0 fails the page-math validation like every
    /// other unusable size.
    #[test]
    fn an_absurd_lbads_decodes_to_the_zero_sentinel() {
        let dev = MockNvmeDevice::new();
        dev.set_lbads(40);
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        assert_eq!(ctrl.info().block_size, 0, "the sentinel, not a panic");
        ctrl.set_num_queues(1, 16, || {}).unwrap();
        let qp = ctrl.create_io_qpair(1, 32, None, 16, || {}).unwrap();
        let svc = NvmeSwap::bind(NvmeSwapEndowment {
            ctrl,
            qp,
            region: SwapRegion::new(2048, 4096),
            capacity_pages: 64,
        });
        assert_eq!(svc.stats().capacity_pages, 0);
    }
}

/// GPT discovery driven over the device: the mock disk carries a real
/// table (built with the same crates the parser trusts), and the store
/// reads it through its own bounce page — the chunked entry-array path
/// the QEMU gate exercises, now host-covered.
#[cfg(test)]
mod discovery_tests {
    use super::*;
    use crate::mock::{MOCK_BLOCK_SIZE, MOCK_DISK_BLOCKS, MockNvmeDevice};
    use gpt_disk_types::{
        Crc32, GptHeader, GptHeaderRevision, GptHeaderSignature, GptPartitionAttributes,
        GptPartitionEntry, GptPartitionName, GptPartitionType, LbaLe, U32Le, guid,
    };
    use kore_driver::gpt::{GptError, KORE_SWAP_TYPE};
    use kore_svc_swap::{staging_page, staging_page_mut};

    const ENTRY_COUNT: u32 = 128;
    const ENTRY_SIZE: u32 = 128;

    fn crc32(bytes: &[u8]) -> u32 {
        const ALGO: crc::Crc<u32> = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);
        ALGO.checksum(bytes)
    }

    /// Write a valid one-partition table into the mock disk: header at
    /// LBA 1, the 16 KiB entry array at LBA 2 — four of the mock's 4 KiB
    /// blocks, which is exactly what makes discovery's chunked
    /// entry-array read real (the bounce page holds one block per read).
    fn provision_mock_disk(dev: &MockNvmeDevice, start: u64, end: u64) {
        let mut entries = [0u8; (ENTRY_COUNT * ENTRY_SIZE) as usize];
        let entry = GptPartitionEntry {
            partition_type_guid: GptPartitionType(KORE_SWAP_TYPE),
            unique_partition_guid: guid!("11111111-2222-3333-4444-555555555555"),
            starting_lba: LbaLe::from_u64(start),
            ending_lba: LbaLe::from_u64(end),
            attributes: GptPartitionAttributes::default(),
            name: GptPartitionName::default(),
        };
        entries[..core::mem::size_of::<GptPartitionEntry>()]
            .copy_from_slice(bytemuck::bytes_of(&entry));

        let mut header = GptHeader {
            signature: GptHeaderSignature::EFI_COMPATIBLE_PARTITION_TABLE_HEADER,
            revision: GptHeaderRevision::VERSION_1_0,
            header_size: U32Le::from_u32(92),
            header_crc32: Crc32(U32Le::from_u32(0)),
            reserved: U32Le::from_u32(0),
            my_lba: LbaLe::from_u64(1),
            alternate_lba: LbaLe::from_u64(MOCK_DISK_BLOCKS as u64 - 1),
            first_usable_lba: LbaLe::from_u64(6),
            last_usable_lba: LbaLe::from_u64(MOCK_DISK_BLOCKS as u64 - 2),
            disk_guid: guid!("99999999-8888-7777-6666-555555555555"),
            partition_entry_lba: LbaLe::from_u64(2),
            number_of_partition_entries: U32Le::from_u32(ENTRY_COUNT),
            size_of_partition_entry: U32Le::from_u32(ENTRY_SIZE),
            partition_entry_array_crc32: Crc32(U32Le::from_u32(crc32(&entries))),
        };
        let crc = crc32(&bytemuck::bytes_of(&header)[..92]);
        header.header_crc32 = Crc32(U32Le::from_u32(crc));

        let mut block = [0u8; MOCK_BLOCK_SIZE as usize];
        block[..core::mem::size_of::<GptHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
        dev.disk_write(1, &block);
        dev.disk_write(2, &entries);
    }

    fn rig() -> (
        NvmeSwap<MockNvmeDevice, crate::mock::MockPool>,
        MockNvmeDevice,
    ) {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).expect("bring-up");
        ctrl.identify(16, || {}).expect("identify");
        ctrl.set_num_queues(1, 16, || {}).expect("set queues");
        let qp = ctrl.create_io_qpair(1, 32, None, 16, || {}).expect("qpair");
        let svc = NvmeSwap::bind(NvmeSwapEndowment {
            ctrl,
            qp,
            region: None,
            capacity_pages: 64,
        });
        (svc, dev)
    }

    /// **Discovery end to end over the device**: the store provisions
    /// itself to exactly the designated extent, and a page round-trips
    /// inside it.
    #[test]
    fn discovery_provisions_the_store_from_the_disk_table() {
        let (mut svc, dev) = rig();
        provision_mock_disk(&dev, 8, 55);
        assert_eq!(svc.stats().capacity_pages, 0, "binds empty");

        svc.discover_region(&mut SpinPacer::default())
            .expect("the table designates a region");
        assert_eq!(
            svc.stats().capacity_pages,
            48,
            "LBAs 8..=55 = 48 blocks = 48 pages at 4 KiB blocks"
        );

        let mut staging = alloc::vec![0u8; PAGE_BYTES as usize];
        staging_page_mut(&mut staging, 0).unwrap().fill(0x77);
        let SwapResponse::Stored { token } = svc.handle(
            SwapRequest::Out { page: 0 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        ) else {
            panic!("a page stores inside the discovered region");
        };
        assert!(
            dev.disk_snapshot(8, 1).iter().all(|&b| b == 0x77),
            "slot 0 is the REGION's first block"
        );
        staging_page_mut(&mut staging, 0).unwrap().fill(0);
        assert_eq!(
            svc.handle(
                SwapRequest::In { token, page: 0 },
                Some(&mut staging),
                &mut SpinPacer::default(),
            ),
            SwapResponse::Restored
        );
        assert!(
            staging_page(&staging, 0)
                .unwrap()
                .iter()
                .all(|&b| b == 0x77)
        );
    }

    /// A second discovery is a no-op once provisioned: the books —
    /// stored bitmap, free stack, counters — survive it untouched.
    #[test]
    fn rediscovery_never_resets_a_provisioned_store() {
        let (mut svc, dev) = rig();
        provision_mock_disk(&dev, 8, 55);
        svc.discover_region(&mut SpinPacer::default()).unwrap();

        let mut staging = alloc::vec![0u8; PAGE_BYTES as usize];
        staging_page_mut(&mut staging, 0).unwrap().fill(0x3C);
        let SwapResponse::Stored { token } = svc.handle(
            SwapRequest::Out { page: 0 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        ) else {
            panic!("stores");
        };
        assert_eq!(svc.stats().stored_pages, 1);

        svc.discover_region(&mut SpinPacer::default())
            .expect("rediscovery is an idempotent Ok");
        assert_eq!(svc.stats().stored_pages, 1, "the books survived");
        staging_page_mut(&mut staging, 0).unwrap().fill(0);
        assert_eq!(
            svc.handle(
                SwapRequest::In { token, page: 0 },
                Some(&mut staging),
                &mut SpinPacer::default(),
            ),
            SwapResponse::Restored,
            "the stored page is still restorable"
        );
    }

    /// The mock's seeded (garbage) disk carries no signature: discovery
    /// says `NotGpt` and the store stays empty — the QEMU gate's rung 9.
    #[test]
    fn an_unpartitioned_disk_designates_nothing() {
        let (mut svc, _dev) = rig();
        assert_eq!(
            svc.discover_region(&mut SpinPacer::default()),
            Err(GptError::NotGpt)
        );
        assert_eq!(svc.stats().capacity_pages, 0);
    }
}

/// The readiness probe: write/read/verify inside a designated region,
/// read-only everywhere else.
#[cfg(test)]
mod probe_tests {
    use super::*;
    use crate::mock::MockNvmeDevice;
    use kore_svc_swap::{staging_page, staging_page_mut};

    const REGION_LBA: u64 = 8;

    fn rig() -> (
        NvmeSwap<MockNvmeDevice, crate::mock::MockPool>,
        MockNvmeDevice,
    ) {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).expect("bring-up");
        ctrl.identify(16, || {}).expect("identify");
        ctrl.set_num_queues(1, 16, || {}).expect("set queues");
        let qp = ctrl.create_io_qpair(1, 32, None, 16, || {}).expect("qpair");
        let svc = NvmeSwap::bind(NvmeSwapEndowment {
            ctrl,
            qp,
            region: SwapRegion::new(REGION_LBA, 48),
            capacity_pages: 64,
        });
        (svc, dev)
    }

    /// With a region the probe WRITES — and only inside the region: the
    /// pattern lands at the region's first block, round-trips, and every
    /// block outside stays byte-identical.
    #[test]
    fn the_probe_writes_only_inside_the_region_and_round_trips() {
        let (mut svc, dev) = rig();
        let outside_before = dev.disk_snapshot(0, REGION_LBA as u32);
        let inside_before = dev.disk_snapshot(REGION_LBA, 1);

        svc.self_test(&mut SpinPacer::default(), 0xC0FF_EE00)
            .expect("the write/read/verify probe passes");

        assert_eq!(
            dev.disk_snapshot(0, REGION_LBA as u32),
            outside_before,
            "nothing outside the region changed — LBA 0 included"
        );
        assert_ne!(
            dev.disk_snapshot(REGION_LBA, 1),
            inside_before,
            "the pattern reached the region's first block"
        );
    }

    /// The pattern is a function of the seed: two probes leave different
    /// bytes at the same LBA, which is what keeps a boot from passing on
    /// the PREVIOUS boot's pattern when the device silently drops writes.
    #[test]
    fn the_probe_pattern_is_seed_unique() {
        let (mut svc, dev) = rig();
        svc.self_test(&mut SpinPacer::default(), 1).unwrap();
        let first_boot = dev.disk_snapshot(REGION_LBA, 1);
        svc.self_test(&mut SpinPacer::default(), 2).unwrap();
        assert_ne!(
            dev.disk_snapshot(REGION_LBA, 1),
            first_boot,
            "a different seed writes different bytes"
        );
    }

    /// A stored page in slot 0 downgrades the probe to the read leg — a
    /// probe must never clobber a live page, whenever it runs.
    #[test]
    fn the_probe_never_overwrites_a_stored_page() {
        let (mut svc, dev) = rig();
        let mut staging = alloc::vec![0u8; PAGE_BYTES as usize];
        staging_page_mut(&mut staging, 0).unwrap().fill(0x99);
        let SwapResponse::Stored { token } = svc.handle(
            SwapRequest::Out { page: 0 },
            Some(&mut staging),
            &mut SpinPacer::default(),
        ) else {
            panic!("stores into slot 0");
        };

        let before = dev.disk_snapshot(REGION_LBA, 1);
        svc.self_test(&mut SpinPacer::default(), 0xABCD)
            .expect("the read-leg probe still passes");
        assert_eq!(
            dev.disk_snapshot(REGION_LBA, 1),
            before,
            "the stored page's blocks are untouched"
        );

        staging_page_mut(&mut staging, 0).unwrap().fill(0);
        assert_eq!(
            svc.handle(
                SwapRequest::In { token, page: 0 },
                Some(&mut staging),
                &mut SpinPacer::default(),
            ),
            SwapResponse::Restored
        );
        assert!(
            staging_page(&staging, 0)
                .unwrap()
                .iter()
                .all(|&b| b == 0x99)
        );
    }
}

#[cfg(test)]
mod region_tests {
    use super::*;
    use crate::mock::MockNvmeDevice;

    /// **The regression that cost a real disk its partition table.**
    ///
    /// A store must never address LBA 0.  With a region designated, slot 0
    /// is the region's first block; with none, the store has no capacity
    /// and no slot exists to address anything at all.
    #[test]
    fn no_slot_ever_addresses_lba_zero() {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        ctrl.set_num_queues(1, 16, || {}).unwrap();
        let qp = ctrl.create_io_qpair(1, 32, None, 16, || {}).unwrap();
        let svc = NvmeSwap::bind(NvmeSwapEndowment {
            ctrl,
            qp,
            region: SwapRegion::new(2048, 4096),
            capacity_pages: 64,
        });
        assert_eq!(svc.lba(0), 2048, "slot 0 is the REGION's first block");
        assert!(svc.stats().capacity_pages > 0);
        for slot in 0..svc.stats().capacity_pages {
            assert!(svc.lba(slot) >= 2048, "every slot sits inside the region");
        }
    }

    /// With no region designated the store binds EMPTY: zero capacity, so
    /// no write is reachable and the self-test stays a **read** — the
    /// posture of every machine whose disk designates nothing for kore.
    #[test]
    fn an_undesignated_store_has_no_capacity_and_writes_nothing() {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        ctrl.set_num_queues(1, 16, || {}).unwrap();
        let qp = ctrl.create_io_qpair(1, 32, None, 16, || {}).unwrap();
        let mut svc = NvmeSwap::bind(NvmeSwapEndowment {
            ctrl,
            qp,
            region: None,
            capacity_pages: 64,
        });
        assert_eq!(svc.stats().capacity_pages, 0, "no region ⇒ no store");

        // The WHOLE disk, not just LBA 0: an undesignated store owns no
        // block anywhere, so the self-test may change none of them.
        let before = dev.disk_snapshot(0, crate::mock::MOCK_DISK_BLOCKS);
        let mut staging = alloc::vec![0u8; PAGE_BYTES as usize];
        assert_eq!(
            svc.handle(
                SwapRequest::Out { page: 0 },
                Some(&mut staging),
                &mut SpinPacer::default()
            ),
            SwapResponse::Refused(SwapRefusal::StoreFull),
            "an undesignated store refuses every write"
        );
        svc.self_test(&mut SpinPacer::default(), 0xB00B_5EED)
            .expect("the read-only self-test still proves the queue");
        assert_eq!(
            dev.disk_snapshot(0, crate::mock::MOCK_DISK_BLOCKS),
            before,
            "no block changed — the whole point"
        );
    }

    /// A region may not start at LBA 0, be empty, or wrap — the shape is
    /// refused at construction, not guarded against at every use.
    #[test]
    fn a_region_at_lba_zero_is_unrepresentable() {
        assert!(SwapRegion::new(0, 4096).is_none(), "LBA 0 is never swap");
        assert!(SwapRegion::new(2048, 0).is_none(), "an empty region");
        assert!(SwapRegion::new(u64::MAX, 2).is_none(), "a wrapping region");
        assert!(SwapRegion::new(2048, 4096).is_some());
    }

    /// A region declared past the end of the namespace binds empty rather
    /// than silently truncating into blocks it was never given.
    #[test]
    fn a_region_past_the_device_end_binds_empty() {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        ctrl.set_num_queues(1, 16, || {}).unwrap();
        let qp = ctrl.create_io_qpair(1, 32, None, 16, || {}).unwrap();
        let svc = NvmeSwap::bind(NvmeSwapEndowment {
            ctrl,
            qp,
            region: SwapRegion::new(crate::mock::MOCK_BLOCK_COUNT, 4096),
            capacity_pages: 64,
        });
        assert_eq!(svc.stats().capacity_pages, 0);
    }
}
