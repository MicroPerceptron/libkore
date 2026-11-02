//! Production filesystem composition over a provisioned direct NVMe channel.
//!
//! Two production postures land at this one entry seam:
//!
//! - **Completion pacing.**  The class declaration asks for one optional
//!   MSI-family message; when the platform realizes it, the I/O completion
//!   queue signals that message and every completion wait parks the task on the
//!   held route ([`kore_driver::host::RoutePacer`]) with a deadline read
//!   from the founder-published clock page.  A machine that realizes no
//!   route still bounds every wait by that same clock deadline
//!   ([`kore_driver::host::DeadlinePacer`]) — it polls instead of parking,
//!   but what it is bounded BY is elapsed time either way.  Only a machine
//!   whose clock page cannot be read at all falls back to the spin budget,
//!   because there is then no time source to bound against.
//! - **The designated regions.**  A whole disk is only written when it is
//!   provably kore's (recognized roots) or provably nobody's (all-blank
//!   roots).  A disk carrying foreign metadata is consulted read-only for a
//!   conservative user partition ([`kore_driver::gpt::KORE_FS_TYPE`]) and a
//!   rebuildable system partition ([`kore_driver::gpt::KORE_SYSTEM_TYPE`]).
//!   Foreign or stale-era bytes remain evidence on the user side; only the
//!   measured archive era authorizes replacement inside the system clamp.
//!   Two affine [`EngineLane`]s over one [`EngineBackend`] — each lane a ring
//!   on the storage transport engine (`res/STORAGE_TRANSPORT.md`) — feed
//!   independent engines, each mounted behind an exact [`RegionDevice`]
//!   clamp. The frontend tags
//!   engine-local authority words and exposes one unchanged filesystem row.
//!   No user designation still means the bounded RAM posture; no system
//!   designation keeps the compatibility store on that user engine.
//! - **Client rings.** A session may attach its own ring on the same engine
//!   (`res/STORAGE_TRANSPORT.md` §4, slice S4): a window it funds and lends,
//!   into which the row installs a read grant over the
//!   [`kore_driver::gpt::KORE_PACK_TYPE`] partition, so that immutable image
//!   streams into client-funded memory at queue depth with no call per read.
//!   Absent a designation the row mounts exactly as before and a pack grant
//!   answers `Unsupported`.

use alloc::boxed::Box;
use alloc::vec::Vec;

#[cfg(all(target_arch = "x86_64", target_os = "none"))]
use core::sync::atomic::Ordering;
use kore_driver::block::{
    BlockChannel, BlockError, CompletionMode, DeviceCommand, DeviceCompletion, DirectBlockDevice,
    RegionDevice,
};
use kore_driver::block::refresh::{self, Bucket, RegionRefresh};
use kore_driver::stor::engine::ThreeClass;
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
use kore_driver::stor::engine::TargetAccess;
use kore_driver::stor::engine::RingId;
use kore_abi::stor::GrantId;
use kore_driver::stor::grant::{ClassSet, ExtentRun, Grant, Rights};
use kore_driver::stor::lanes::{EngineBackend, EngineLane, LaneClasses};
use kore_driver::block::{CompletionPacer, SpinPacer};
use kore_driver::gpt;
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
use kore_driver::host::RoutePacer;
use kore_driver::host::{
    BoundDevice, CapBuf, CapPool, CapRegs, ClassQuiesce, DeadlinePacer, DeviceBinding,
    DeviceResources, DmaActive, DmaPrepared,
};
use kore_driver::host::QuietRetire;
use kore_driver::seam::{Quiet, Regs};
use kore_nvme::{
    PreparedController,
    block::{NvmeBlockDevice, NvmeBuffer},
    prp::MemPage,
};
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
use kore_rt::hint;
use kore_rt::service::{PeerClose, ReturnMode, Serve, StartFailure, SystemService};
use kore_rt::svc::SvcProtocol;
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
use kore_rt::svc::{ClientEndpoints, GrantPin, SvcClient, client_rings};
use kore_rt::svc::{PeerAdmissionRefusal, PeerId, PeerMapPlan};
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
use kore_svc_block::{BlockClient, BlockRefusal, BlockService, ProvisionRequest, RingBlock};


use crate::persistence::{PersistentFs, VolumeProbe, bootstrap_data_start};
use crate::{
    FsEndowment, FsError, FsRequest, FsResponse, FsService, ProviderError, RamService, SessionId,
    persistence::{CommitStage, PersistenceError, PersistenceState, commit_engine},
    provider::BlockProvider,
    router::{self, RoutedRequest, Store},
    window::DataWindow,
};

/// Boot-critical durable-storage budget. `CAP.TO` is the controller's
/// worst-case legal transition time, not permission for an optional service
/// to hold the first shell for up to 127.5 seconds. We honor the advertised
/// value up to this policy ceiling, then degrade to the volatile filesystem.
const BRING_UP_BUDGET_NS: u64 = 1_000_000_000;
const BRING_UP_FALLBACK_POLLS: u32 = kore_driver::block::MAX_POLLS;

enum BringUpPacer {
    Deadline(DeadlinePacer),
    Spin(SpinPacer),
}

impl CompletionPacer for BringUpPacer {
    fn pace(&mut self) -> bool {
        match self {
            Self::Deadline(pacer) => pacer.pace(),
            Self::Spin(pacer) => pacer.pace(),
        }
    }
}

fn bring_up_deadline(caps: kore_nvme::Capabilities) -> Option<u64> {
    let now = kore_rt::clock::uptime().ok()?.as_nanos();
    Some(now.saturating_add(bring_up_budget_ns(caps)))
}

const fn bring_up_budget_ns(caps: kore_nvme::Capabilities) -> u64 {
    let advertised = caps.ready_timeout_ns();
    if advertised == 0 {
        BRING_UP_BUDGET_NS
    } else if advertised < BRING_UP_BUDGET_NS {
        advertised
    } else {
        BRING_UP_BUDGET_NS
    }
}

fn bring_up_pacer(deadline_ns: Option<u64>) -> BringUpPacer {
    match deadline_ns {
        Some(deadline_ns) => BringUpPacer::Deadline(DeadlinePacer::new(deadline_ns)),
        None => BringUpPacer::Spin(SpinPacer::new(BRING_UP_FALLBACK_POLLS)),
    }
}
/// The one I/O queue's depth, shared by every ring. The engine's
/// [`ENGINE_COMMANDS`] fit with room (client rings stream at full depth while
/// the filesystem lanes keep theirs), and the controller clamps it to its own
/// `CAP.MQES`. The ring costs 40 KiB of the pool at this depth.
const IO_QDEPTH: u16 = 512;
/// Submission queues the filesystem's controller runs: one per service class
/// (Demand, Stream, Commit — `res/STORAGE_TRANSPORT.md` §3), all bound to one
/// completion queue on the one interrupt vector the binding realizes.
const FS_QUEUES: usize = 3;
/// Each class queue's depth: past the largest class budget (Stream's 128
/// commands) with room for the ring's one reserved slot. The completion
/// queue holds all three (480 entries), below the single queue's 512.
const FS_SQ_DEPTH: u16 = 160;
const _: () = assert!(FS_QUEUES * FS_SQ_DEPTH as usize <= IO_QDEPTH as usize);
/// The filesystem's controller: a three-queue group.
type FsNvme = NvmeBlockDevice<CapRegs, CapPool, FS_QUEUES>;
/// Capability-backed DMA bytes requested from the block provisioner — a
/// mirror of the block row's `DMA_POOL@1` (the founder's LARGE class), which
/// is subleased whole; the request is an intent, not a size.
const FS_DMA_POOL_BYTES: usize = 0x800_0000;
/// The filesystem's OWN share of that pool: the queue/control graph plus the
/// four cumulatively allocated buffers of a whole-disk -> region rebuild.
/// Held at the ordinary 2 MiB class the row used to be founded with, so the
/// page ceiling below does not move with the pool; everything above it is
/// the controller's host memory buffer.
const FS_IO_DMA_BYTES: usize = 0x20_0000;
/// Largest controller page whose complete queue/control graph plus the four
/// cumulatively allocated buffers of a whole-disk -> region rebuild fits the
/// filesystem's 2 MiB share. At 512 KiB, the four page-aligned queue starts
/// consume so much padding that even the first filesystem buffer cannot be
/// admitted.
const MAX_FS_NVME_PAGE_BYTES: u64 = (FS_IO_DMA_BYTES / 8) as u64;
/// Most host memory the controller is offered for its mapping tables
/// (Identify HMPRE decides how much of it is taken; 0 there takes none).
/// 64 MiB is what a DRAM-less PCIe 4 consumer part asks for and what Linux
/// grants it by default; the pool leaves room for it beside the
/// filesystem's own share.
const FS_HMB_MAX_BYTES: u64 = 0x400_0000;
const _: () = assert!(
    FS_IO_DMA_BYTES as u64 + FS_HMB_MAX_BYTES <= FS_DMA_POOL_BYTES as u64,
    "the filesystem's share and the host memory buffer must fit the founded pool"
);

const fn fs_nvme_page_supported(page: MemPage) -> bool {
    page.bytes() <= MAX_FS_NVME_PAGE_BYTES
}

/// Completion-wait budget for one route-parked block operation, in
/// nanoseconds — one **round** of the wait, not the device timeout.  It is
/// the interrupt-loss recovery cadence: on expiry the wait's final poll
/// catches a completion whose interrupt never arrived, at a cost of one
/// second rather than the whole stall ceiling.  A healthy controller that
/// has simply not finished is granted further rounds by the provider
/// (`provider::block::STALL_BUDGETS`, 30 s in all — Linux's I/O timeout),
/// because a DRAM-less consumer drive under garbage collection is slow for
/// seconds and was being refused, and the filesystem operation on top of it
/// failed, on a budget written for a device that answers in microseconds.
/// Read fresh per wait from the founder-published clock page, so it needs
/// no executor anchor.
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
const OP_BUDGET_NS: u64 = 1_000_000_000;

/// **How long one monitored turn may hold this core** — the nested fast tier
/// under the route-less wait, sized to nothing like its container.
///
/// The route-less posture has no interrupt and therefore nothing that could
/// ever deliver a wake, so its wait must find the completion itself. It used
/// to do that by spinning until [`OP_BUDGET_NS`]. An address monitor
/// ([`kore_rt::hint::wait_on`] over the CQ dword the controller's DMA stores
/// to) finds it strictly better on both axes that matter: the core idles in
/// `C0.1` instead of retiring `PAUSE`, and the wake is the completion's own
/// store rather than the next poll.
///
/// **The sizing is the whole design, and it is a ratio, not a duration.**
/// `UMWAIT` parks a core *without yielding it to the scheduler*, so a
/// monitored wait sized like the device budget would hold a core for a full
/// second and starve every runnable task on it — a nested bound sized like the
/// thing it nests inside, which is a defect this tree has already measured at
/// a 2.3x gate regression elsewhere. It is exactly
/// [`kore_driver::host::PROBE_PAUSE`], the cadence at which the outer elapsed
/// deadline pacer rechecks its clock. A shorter turn would add wake/re-arm
/// work without tightening that outer observation; a longer turn would make
/// the deadline coarser and hold this core past the shared probe policy.
///
/// A healthy NVMe command completes in tens of microseconds, so a healthy wait
/// spends a handful of these turns rather than one. That is the better shape,
/// not a compromise: every turn is the same `C0.1` idle, so the power posture
/// is unchanged, and each one re-reads the clock and so honours the outer
/// deadline more tightly than a single long sleep could.
///
/// [`kore_rt::hint::wait_on`]: ../../kore_rt/hint/fn.wait_on.html
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
const MONITOR_CAP: core::time::Duration = core::time::Duration::from_micros(4);
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
const BLOCK_CLIENT_INDEX: u16 = <DurableService as SvcProtocol>::SERVE_PAIRS as u16;

pub struct NvmeFsBinding;

pub struct NvmeFsPrepared {
    controller: PreparedController<CapRegs, CapPool>,
    vector: Option<u16>,
    pacer: BringUpPacer,
}

pub struct NvmeFsBound {
    backend: Option<FsNvme>,
    resources: DeviceResources,
}

impl DeviceBinding for NvmeFsBinding {
    type Pci = kore_svc_block::service::NvmeIoPci;
    type Config = ();
    type Prepared = NvmeFsPrepared;
    type Bound = NvmeFsBound;
    type Error = BlockError;

    fn prepare(
        resources: &mut DeviceResources<DmaPrepared>,
        (): Self::Config,
    ) -> Result<Self::Prepared, Self::Error> {
        // Read before the BAR moves into the controller: whether the intake
        // realized the declared vector decides the CQ's interrupt posture.
        let vector = resources.route().is_some().then_some(0);
        let regs = resources.take_bar(0).ok_or(BlockError::Transport)?;
        let pool = resources.take_pool().ok_or(BlockError::Backpressure)?;
        let caps = kore_nvme::Capabilities::from_cap(regs.read64(kore_nvme::regs::REG_CAP));
        let mut pacer = bring_up_pacer(bring_up_deadline(caps));
        let controller = kore_nvme::Controller::prepare_paced(regs, pool, &mut pacer)
            .map_err(|_| BlockError::Transport)?;
        if !fs_nvme_page_supported(controller.page()) {
            let (regs, pool) = controller.into_seam();
            resources.restore_bar(0, regs);
            resources.restore_pool(pool);
            return Err(BlockError::Geometry);
        }
        Ok(NvmeFsPrepared {
            controller,
            vector,
            pacer,
        })
    }

    fn start(
        prepared: Self::Prepared,
        mut resources: DeviceResources<DmaActive>,
    ) -> Result<Self::Bound, Self::Error> {
        let NvmeFsPrepared {
            controller,
            vector,
            mut pacer,
        } = prepared;
        // Weighted round robin where the controller offers it: the class
        // queues' priorities (Demand urgent, Commit high, Stream medium)
        // then govern what the controller fetches next.
        let controller = match controller.with_weighted_arbitration().start_paced(&mut pacer) {
            Ok(controller) => controller,
            Err(failure) => {
                let (_error, regs, pool) = failure.into_seam();
                resources.restore_bar(0, regs);
                resources.restore_pool(pool);
                return Err(BlockError::Transport);
            }
        };
        // The CQ signals MSI-X table entry 0 — the same entry the realized
        // route remapped — iff the route exists; a route-less bind creates
        // a silent CQ and the completion wait stays the bounded spin.
        let backend = FsNvme::provision_queues_paced(
            controller,
            1,
            FS_SQ_DEPTH,
            vector,
            FS_HMB_MAX_BYTES,
            &mut pacer,
            kore_driver::host::relax,
            BRING_UP_FALLBACK_POLLS,
        )?;
        Ok(NvmeFsBound {
            backend: Some(backend),
            resources,
        })
    }

    fn quiesce(bound: &mut Self::Bound) -> Result<ClassQuiesce, Self::Error> {
        bound.backend.as_mut().ok_or(BlockError::Lost)?.quiesce()?;
        Ok(ClassQuiesce::Stopped)
    }

    fn unbind(mut bound: Self::Bound) -> DeviceResources<DmaActive> {
        let backend = bound
            .backend
            .take()
            .expect("NVMe filesystem binding lost backend");
        let (regs, pool) = backend.into_seam();
        bound.resources.restore_bar(0, regs);
        bound.resources.restore_pool(pool);
        bound.resources
    }
}

/// The direct backend value owned by one filesystem worker. The common host
/// retains all ordinary capability children and performs ordered containment.
pub struct DirectNvme {
    /// `None` only after the terminal teardown (#809 §L) consumed it; every
    /// lane has retired by then, so no I/O path can observe the absence.
    bound: Option<BoundDevice<NvmeFsBinding>>,
    /// Route parks issued by the completion wait (the wait's denominator).
    route_parks: u64,
    /// Waits that took the bounded spin because the clock could supply no
    /// deadline at all — the only posture left with no time source. Counted
    /// because it is otherwise invisible: such a machine bounds its disk
    /// waits by retired instructions and still reports `route=held`.
    spin_waits: u64,
    /// Parks resolved as real route fires — the MSI-X-landed witness
    /// (`0` with growing parks = a delivery-dead machine running on the
    /// final-poll fallback).
    route_fires: u64,
    /// Waits bounded by a clock deadline with no route to park on — the
    /// route-less posture on a machine whose clock *is* readable. Counted
    /// apart from `spin_waits` because the two are no longer the same
    /// fallback: this one is bounded by elapsed time, that one only by
    /// retired instructions.
    deadline_waits: u64,
    /// Route-less turns that actually slept the core on the CQ line — the
    /// monitored fast tier's numerator.
    ///
    /// The one fact that cannot be inferred from anything else on the box. No
    /// emulator implements a user-mode monitor (QEMU's TCG decodes neither
    /// `UMWAIT` nor `MWAITX`), so under the boot gates this is *always* zero
    /// while `monitor_turns` climbs — and a metal run that also reported zero
    /// would look identical to a green gate. Without this counter, "the fast
    /// path is live on this machine" and "the fast path silently degraded to a
    /// spin" are the same observation.
    monitor_sleeps: u64,
    /// Route-less turns that reached the monitored tier at all — its
    /// denominator, counted whether or not the machine could sleep.
    ///
    /// `sleeps / turns` is the ratio a metal A/B is read by; `turns` alone is
    /// how many times a route-less wait went round, which is the latency
    /// signal the old spin loop never exposed.
    monitor_turns: u64,
    /// Targets whose stage-1 release was refused. They stay owned and
    /// unreachable instead of poisoning unrelated targets or the filesystem
    /// lanes. Terminal containment retires their import owners.
    quarantined_targets: Vec<CapBuf>,
    /// Commands staged through the queue-set face and not yet reaped — so
    /// a park never sleeps on a device holding nothing.
    in_flight: usize,
    /// The completion a park took from the route wait, reaped first.
    parked: Option<DeviceCompletion>,
    /// What imported-window retirements still owe.
    retirements: RetirementBooks,
}

/// What this controller's imported-window retirements still owe
/// (`res/LIFECYCLE_HARDENING.md` §6).
///
/// A released window that has not retired yet holds its own IOVA range and
/// owner, and no command names it: nothing a new import uses. So only a
/// failed requester-wide retirement, which leaves the latch unproven, refuses
/// new client windows. A deferred window is retried: live (its engine's
/// `Quiet` proof still stands) before the next adoption, and whole at every
/// idle pass. One stuck window used to refuse every later client until a
/// reboot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RetirementBooks {
    /// Released windows the host still owns.
    deferred: usize,
    /// The idle-boundary retirement failed: the requester's latch is
    /// unproven until a later idle pass succeeds.
    failed: bool,
}

impl RetirementBooks {
    /// Whether an idle pass has anything to do.
    fn owed(&self) -> bool {
        self.failed || self.deferred != 0
    }

    /// An idle-boundary pass: it retires every released window and replays
    /// the latch, so its outcome replaces the failure.
    fn settle_idle(&mut self, healthy: bool, deferred: usize) {
        self.failed = !healthy;
        self.deferred = deferred;
    }

    /// A live pass retires only proven windows and never touches the latch:
    /// it cannot clear a failure, only add one.
    fn settle_live(&mut self, healthy: bool, deferred: usize) {
        self.failed |= !healthy;
        self.deferred = deferred;
    }

    /// The refusal a new client window meets, if any.
    fn admission(&self) -> Result<(), ProviderError> {
        if self.failed {
            Err(ProviderError::Retiring)
        } else {
            Ok(())
        }
    }
}

impl DirectNvme {
    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    fn claim(opened: kore_svc_block::OpenedChannel) -> Result<Self, BlockError> {
        let bound = BoundDevice::<NvmeFsBinding>::claim(opened.device, opened.pool)
            .map_err(|_| BlockError::Transport)?;
        Ok(Self {
            bound: Some(bound),
            route_parks: 0,
            spin_waits: 0,
            route_fires: 0,
            deadline_waits: 0,
            monitor_sleeps: 0,
            monitor_turns: 0,
            quarantined_targets: Vec::new(),
            in_flight: 0,
            parked: None,
            retirements: RetirementBooks::default(),
        })
    }

    /// Whether the intake realized the optional MSI-X route — the
    /// completion-pacing and `enable_irq_parks` decider.
    pub fn has_route(&self) -> bool {
        self.bound
            .as_ref()
            .is_some_and(|bound| bound.parts().resources.route().is_some())
    }

    /// The live binding.  Unreachable after the terminal teardown: it runs
    /// only once every lane has retired, and a lane is the only I/O door.
    fn bound(&self) -> &BoundDevice<NvmeFsBinding> {
        self.bound
            .as_ref()
            .expect("terminal teardown consumed the binding under a live lane")
    }

    fn bound_mut(&mut self) -> &mut BoundDevice<NvmeFsBinding> {
        self.bound
            .as_mut()
            .expect("terminal teardown consumed the binding under a live lane")
    }

    /// The terminal teardown (#809 §L): class stop (NVMe `CC.SHN` already
    /// completed by the last lane's quiesce), the kernel's fabric proof, then
    /// containment.  `None` if already torn down.
    fn teardown(&mut self) -> Option<kore_driver::host::BindingTeardown<BlockError>> {
        self.bound.take().map(BoundDevice::teardown)
    }

    /// Host memory the controller holds for its mapping tables, in bytes
    /// (0 = none granted).
    pub fn host_memory_buffer_bytes(&self) -> u64 {
        self.backend().host_memory_buffer_bytes()
    }

    /// The completion wait's books: `(parks, fires, deadline waits, spin
    /// waits, monitored sleeps, monitored turns)`.
    pub fn route_stats(&self) -> (u64, u64, u64, u64, u64, u64) {
        (
            self.route_parks,
            self.route_fires,
            self.deadline_waits,
            self.spin_waits,
            self.monitor_sleeps,
            self.monitor_turns,
        )
    }

    fn backend(&self) -> &FsNvme {
        self.bound()
            .parts()
            .backend
            .as_ref()
            .expect("live binding has backend")
    }

    /// Map one client-lent fabric route into this controller's DMA domain
    /// and wrap it as an adoptable ring target (`import_dma`, the path the
    /// accelerator rows already take for client leases).
    ///
    /// Reached through [`EngineBackend::control`]: an import maps new memory
    /// into the domain and touches nothing a command in flight uses. The
    /// controller only ever *writes* a target, but `import_dma`'s
    /// `device_write` is exactly that admission.
    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    /// A refusal carries a diagnostic code beside its error: an
    /// [`IntakeError`] code for a refused import, [`ADOPT_REFUSED`] with the
    /// block error for a refused adoption.
    ///
    /// [`IntakeError`]: kore_driver::host::IntakeError
    fn adopt_target(
        &mut self,
        owner: kore_rt::cap::ReturnedCapability,
    ) -> Result<NvmeBuffer<CapBuf>, (FsError, u32)> {
        // Only a failed requester-wide retirement refuses the import; the
        // retry that clears it needs the controller idle
        // ([`Self::retry_retirement`]).
        if let Err(cause) = self.retirements.admission() {
            return Err((
                FsError::ProviderFailure(cause),
                ADOPT_REFUSED | (BlockError::Backpressure as u32) << 8,
            ));
        }
        // A route that is not a read-write device route is refused here and
        // dropped: the filesystem mints no reply grants, and the client still
        // holds the parent it scoped this child from.
        let import = kore_driver::host::DmaImport::claim(
            owner,
            kore_abi::session::FabricAccess::READ | kore_abi::session::FabricAccess::WRITE,
        )
            .map_err(|_| (FsError::AccessDenied, CLAIM_REFUSED))?;
        let parts = self.bound_mut().parts_mut();
        let data = match import {
            // A lent lease takes the accelerators' import: mapped here too,
            // so it costs this service window, but it is the path the Xe and
            // NPU clients prove on metal — the control for a route that
            // never completes.
            lease @ kore_driver::host::DmaImport::Lease(_) => {
                parts.resources.import_dma(lease, true)
            }
            // The lane only ever points the controller at a target, so
            // nothing of it is mapped into this service: a multi-GiB model
            // arena must not cost the filesystem row address space or page
            // tables.
            route => parts.resources.import_dma_device(route, true),
        }
        .map_err(|error| {
            (
                FsError::ProviderFailure(ProviderError::Transport),
                error.diagnostic_code() as u32,
            )
        })?;
        let backend = parts.backend.as_mut().expect("live binding has backend");
        match backend.adopt_buffer(data) {
            Ok(buffer) => Ok(buffer),
            Err((data, error)) => {
                // Nothing was submitted against it; hand it straight back.
                let buffer = NvmeBuffer::from_data(data);
                self.release_target(buffer);
                Err((
                    FsError::ProviderFailure(crate::provider::map_block(error)),
                    ADOPT_REFUSED | (error as u32) << 8,
                ))
            }
        }
    }

    /// Return a target the controller refused before any command could name
    /// it: the allocation goes back to the pool, and its device mapping
    /// retires at the next idle pass. Reached with lanes possibly busy, so it
    /// never claims the requester idle itself.
    fn release_target(&mut self, buffer: NvmeBuffer<CapBuf>) {
        let parts = self.bound_mut().parts_mut();
        let backend = parts.backend.as_ref().expect("live binding has backend");
        // SAFETY: the channel handed this target back only with nothing in
        // flight against it, and `control_idle` admitted this call only with
        // no command in flight on any lane, so the controller holds no
        // reference.
        let released = unsafe { backend.release_adopted(buffer) };
        if let Err(kore_driver::seam::DmaReleaseFailure::Refused(data)) = released {
            // This target failed its own stage-1 retirement invariant. Keep
            // its exact owner unreachable; other imported mappings and the
            // filesystem's established windows remain valid.
            if self.quarantined_targets.try_reserve(1).is_ok() {
                self.quarantined_targets.push(data);
            } else {
                core::mem::forget(data);
            }
            return;
        }
        self.retirements.deferred = parts.resources.deferred_retirements();
    }

    /// Retire one target the engine handed back with its proof, while the
    /// controller runs on: only the imported window's device mapping goes,
    /// the latch stays open for every other window
    /// (`res/STORAGE_TRANSPORT.md` §11). A live retirement left unfinished
    /// falls to the idle boundary ([`retry_retirement`](Self::retry_retirement)).
    /// `true` once the device mapping is gone.
    fn retire_quiet(&mut self, target: Quiet<NvmeBuffer<CapBuf>>) -> bool {
        let parts = self.bound_mut().parts_mut();
        match parts.resources.retire_quiet(target.map(NvmeBuffer::into_data)) {
            QuietRetire::Retired => true,
            QuietRetire::Deferred => {
                self.retirements.deferred = parts.resources.deferred_retirements();
                false
            }
            QuietRetire::Refused(data) => {
                // Not a live import of this binding: keep its owner
                // unreachable, as a refused stage-1 release does.
                if self.quarantined_targets.try_reserve(1).is_ok() {
                    self.quarantined_targets.push(data);
                } else {
                    core::mem::forget(data);
                }
                false
            }
        }
    }

    /// Retry every incomplete retirement. Reached only through
    /// [`EngineBackend::control_idle`] — an ordinary release needs the whole
    /// requester quiescent.
    #[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
    fn retry_retirement(&mut self) {
        if !self.retirements.owed() {
            return;
        }
        let resources = &self.bound().parts().resources;
        let healthy = resources.resume_retired_dma(true);
        let deferred = resources.deferred_retirements();
        self.retirements.settle_idle(healthy, deferred);
    }

    /// Retry the live retirements alone, with lanes running: each window
    /// still carries its engine's proof.
    #[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
    fn retry_live_retirement(&mut self) {
        if self.retirements.deferred == 0 {
            return;
        }
        let resources = &self.bound().parts().resources;
        let healthy = resources.resume_retired_dma(false);
        let deferred = resources.deferred_retirements();
        self.retirements.settle_live(healthy, deferred);
    }

    fn retirement_owed(&self) -> bool {
        self.retirements.owed()
    }

    fn backend_mut(&mut self) -> &mut FsNvme {
        self.bound_mut()
            .parts_mut()
            .backend
            .as_mut()
            .expect("live binding has backend")
    }
}

/// The controller as the transport engine drives it: the group's class
/// queues, with this wrapper's route-aware wait as the park. Named by path,
/// never imported, so `DirectNvme`'s device methods keep their names.
impl kore_driver::stor::QueueSet for DirectNvme {
    type Target = NvmeBuffer<CapBuf>;
    const SQS: usize = FS_QUEUES;

    fn geometry(&self) -> kore_driver::block::BlockGeometry {
        DirectBlockDevice::geometry(self)
    }

    fn target_bytes(target: &Self::Target) -> u64 {
        kore_driver::block::BlockBuffer::len(target) as u64
    }

    fn allocate(&mut self, bytes: usize) -> Result<Self::Target, BlockError> {
        DirectBlockDevice::allocate_buffer(self, bytes)
    }

    fn max_target_bytes(&self) -> usize {
        DirectBlockDevice::max_buffer_bytes(self)
    }

    fn park(&mut self) -> bool {
        if self.parked.is_some() {
            return true;
        }
        if self.in_flight == 0 {
            return false;
        }
        self.parked = DirectBlockDevice::wait(self);
        self.parked.is_some()
    }

    fn room(&self, sq: usize) -> usize {
        self.backend().room_on(sq)
    }

    fn stage(
        &mut self,
        sq: usize,
        command: DeviceCommand,
        target: Option<&mut Self::Target>,
        tag: kore_driver::stor::Tag,
    ) -> Result<(), BlockError> {
        self.backend_mut().stage_on(sq, command, target, tag.cookie())?;
        self.in_flight += 1;
        Ok(())
    }

    fn abort(&mut self, sq: usize, tag: kore_driver::stor::Tag) -> bool {
        self.backend_mut().abort_on(sq, tag.cookie())
    }

    fn reset(&mut self) -> bool {
        if !kore_driver::block::DirectBlockDevice::reset(self.backend_mut()) {
            return false;
        }
        // Nothing staged survives: no completion is owed, none is parked.
        self.in_flight = 0;
        self.parked = None;
        true
    }

    fn publish(&mut self, sq: usize) {
        self.backend_mut().publish_on(sq);
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
            let Some(completion) = DirectBlockDevice::poll(self) else {
                break;
            };
            out[count] = completion;
            count += 1;
        }
        self.in_flight = self.in_flight.saturating_sub(count);
        count
    }

    fn liveness(&mut self) -> kore_driver::block::Liveness {
        DirectBlockDevice::liveness(self)
    }

    fn health(&mut self) -> Result<kore_driver::block::BlockHealth, BlockError> {
        DirectBlockDevice::health(self)
    }

    fn quiesce(&mut self) -> Result<(), BlockError> {
        if self.in_flight != 0 || self.parked.is_some() {
            return Err(BlockError::Draining);
        }
        DirectBlockDevice::quiesce(self)
    }

    fn clock_ns(&self) -> Option<u64> {
        DirectBlockDevice::clock_ns(self)
    }
}

impl DirectBlockDevice for DirectNvme {
    type Buffer = NvmeBuffer<CapBuf>;

    /// The controller's queues advance on their own — inherited from the
    /// backend this wrapper only adds route-aware completion pacing to.  It
    /// is what lets one binding be shared by the two designated regions.
    const INDEPENDENT_PROGRESS: bool =
        <FsNvme as DirectBlockDevice>::INDEPENDENT_PROGRESS;

    fn geometry(&self) -> kore_driver::block::BlockGeometry {
        self.backend().geometry()
    }

    fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError> {
        self.backend_mut().allocate_buffer(bytes)
    }

    fn max_buffer_bytes(&self) -> usize {
        self.backend().max_buffer_bytes()
    }

    fn health(&mut self) -> Result<kore_driver::block::BlockHealth, BlockError> {
        self.backend_mut().health()
    }

    fn liveness(&mut self) -> kore_driver::block::Liveness {
        self.backend_mut().liveness()
    }

    /// The founder-published clock page, so the channel above books every
    /// completion's latency at the depth it was driven — `res/RESIDENCY.md`
    /// §8's "NVMe latency at queue depth", measured on this box.  An
    /// uncalibrated page reads as no clock and measures nothing.
    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    fn clock_ns(&self) -> Option<u64> {
        kore_rt::clock::uptime().ok().map(|now| now.as_nanos())
    }

    fn submit(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        self.backend_mut().submit(command, buffer, cookie)
    }

    fn stage(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        self.backend_mut().stage(command, buffer, cookie)
    }

    fn publish(&mut self) {
        self.backend_mut().publish();
    }

    fn poll(&mut self) -> Option<DeviceCompletion> {
        self.backend_mut().poll()
    }

    fn poll_batch(&mut self, out: &mut [Option<DeviceCompletion>]) -> usize {
        self.backend_mut().poll_batch(out)
    }

    fn quiesce(&mut self) -> Result<(), BlockError> {
        self.backend_mut().quiesce()
    }

    /// The production completion wait: park on the held MSI-X route with a
    /// deadline read fresh from the founder-published clock page, so the
    /// CPU is yielded while the command is in flight.  Route-less bindings
    /// — and an uncalibrated clock page — keep the trait's bounded-spin
    /// posture; either way the final poll catches a completion that landed
    /// without a delivered interrupt.
    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    fn wait(&mut self) -> Option<DeviceCompletion> {
        // A field borrow, not `bound_mut()`: the wait books below write other
        // fields of `self` while `parts` is live.
        let parts = self
            .bound
            .as_mut()
            .expect("terminal teardown consumed the binding under a live lane")
            .parts_mut();
        let backend = parts.backend.as_mut().expect("live binding has backend");
        let route = parts.resources.route();
        let deadline_ns = kore_rt::clock::uptime()
            .ok()
            .map(|now| now.as_nanos().saturating_add(OP_BUDGET_NS));
        let route = match (route, deadline_ns) {
            (Some(route), Some(deadline_ns)) => Some((route, deadline_ns)),
            // No route, but the clock answers: this wait is still bounded by
            // ELAPSED TIME. It used to take the spin budget, which bounds
            // retired instructions instead — so a slow device could exhaust
            // it in a duration nobody chose, and the provider treats an
            // expired wait as evidence to declare the channel lost. A
            // route-less machine is the degraded posture; it is the worst
            // place to answer a slow disk by declaring it dead.
            (None, Some(deadline_ns)) => {
                self.deadline_waits = self.deadline_waits.saturating_add(1);
                let mut pacer = DeadlinePacer::new(deadline_ns);
                let mut sleeps = 0u64;
                let mut turns = 0u64;
                // **The monitored fast tier, nested inside the clocked bound.**
                // This posture has no route, so nothing exists that could ever
                // deliver a wake — but a completion IS a store: the controller
                // DMAs the CQE and flips its phase bit. So this core can sleep
                // on that very word and be woken by the silicon, needing no
                // interrupt and no waker. Per turn, capped at `MONITOR_CAP`
                // (microseconds) and NOT at `deadline_ns` (one second): a
                // monitored sleep does not yield the core, so a turn sized like
                // the container would starve every runnable task on it.
                let completion = loop {
                    if let Some(completion) = backend.poll() {
                        break Some(completion);
                    }
                    turns += 1;
                    let slept = match backend.completion_word() {
                        // `Relaxed` is sufficient and is not a shortcut: this
                        // value is a comparison token for `wait_on`, never a
                        // synchronization edge. Reading it stale costs one
                        // turn slept against an event that already landed;
                        // reading it fresh costs one turn not slept. The edge
                        // that publishes the CQE's other fifteen bytes is the
                        // acquire fence inside `CompRing::poll`, which the
                        // poll at the top of this loop still takes.
                        Some(word) => {
                            let stale = word.load(Ordering::Relaxed);
                            hint::wait_on(word, stale, MONITOR_CAP) == hint::WaitEnd::Monitored
                        }
                        // No monitor expressible on this ring; the outer tier
                        // is unchanged and still correct.
                        None => false,
                    };
                    // A turn that genuinely slept has already bought its wait,
                    // and bought a better one — a store ends it immediately
                    // where a spin batch only notices at its end. Spinning
                    // again would pay twice and hand back the power the sleep
                    // just saved. Either way the bound is checked on the same
                    // counter, so an expired wait still means elapsed time.
                    let within = if slept {
                        sleeps += 1;
                        pacer.pace_after_wait()
                    } else {
                        pacer.pace()
                    };
                    if !within {
                        break backend.poll();
                    }
                };
                self.monitor_sleeps = self.monitor_sleeps.saturating_add(sleeps);
                self.monitor_turns = self.monitor_turns.saturating_add(turns);
                return completion;
            }
            // No clock at all — no time source exists to bound against, so
            // the spin budget is the honest last resort rather than a choice.
            // Counted, because a machine waiting on an instruction count is a
            // fact that appears nowhere else.
            (_, None) => {
                self.spin_waits = self.spin_waits.saturating_add(1);
                let mut pacer = SpinPacer::default();
                loop {
                    if let Some(completion) = backend.poll() {
                        return Some(completion);
                    }
                    if !pacer.pace() {
                        return backend.poll();
                    }
                }
            }
        };
        let (route, deadline_ns) = route.expect("both arms above return");
        let mut pacer = RoutePacer::new(route, deadline_ns);
        let completion = loop {
            if let Some(completion) = backend.poll() {
                break Some(completion);
            }
            if !pacer.pace() {
                break backend.poll();
            }
        };
        self.route_parks = self.route_parks.saturating_add(pacer.parks());
        self.route_fires = self.route_fires.saturating_add(pacer.fired());
        completion
    }
}

/// The user and system lanes over the one controller. Every other reader
/// of the disk — the pack region's streams among them — is a client ring.
const FS_LANES: usize = 2;
/// The pack region's maintenance lane (`res/PACK_STREAMING.md` §7), after
/// the filesystem lanes: the row's own scan and content-preserving refresh
/// of the pack, clamped to its extent. Claimed only when a session first
/// asks for maintenance, so a row that never maintains its pack is the row
/// it was before — and an unclaimed lane never holds the controller up.
const PACK_LANE: usize = FS_LANES;
/// Every lane the row drives itself.
const ROW_LANES: usize = FS_LANES + 1;
/// Commands one lane keeps in flight, and the depth one class of client
/// rings is budgeted. A Nemotron token's routed rows are thousands of 4 KiB
/// reads, and a DRAM-less NVMe reaches its random-read rate only past a
/// hundred or so in flight: at 32 the first chat on the Lunar Lake box
/// streamed them at 280 MB/s against a region that does 5 GB/s
/// sequentially. The driver's adopted list-page pool is sized to match, so
/// full-depth streaming of large records is not throttled by list pages.
const LANE_INFLIGHT: usize = 128;
/// Commands the transport engine keeps on the controller at once: both
/// filesystem lanes' full depth and as much again for the client rings,
/// below the queue's own so the controller never refuses a staged command.
/// The classes' budgets, not this table, share it out; the pack
/// maintenance lane rides the Stream budget beside the client rings, and
/// holds at most a handful of commands (`kore_driver::block::refresh::SLOTS`
/// chunks and a flush).
const ENGINE_COMMANDS: usize = (FS_LANES + 1) * LANE_INFLIGHT;
const _: () = assert!(ENGINE_COMMANDS < IO_QDEPTH as usize);
/// Submissions the engine holds open across every ring.
const ENGINE_OPS: usize = ENGINE_COMMANDS;
/// Targets one ring can hold: a lane's block-channel buffers, or a
/// client's destinations (a model arena and its expert-cache chunks).
const RING_TARGETS: usize = 64;
/// The controller as the transport engine drives it (`res/STORAGE_TRANSPORT.md`
/// §3, slice S2): one submission queue per class over one completion queue
/// and one interrupt vector, every lane a ring on the engine.
type NvmeQueues = DirectNvme;
/// Client rings the engine serves beside the lanes (`res/STORAGE_TRANSPORT.md`
/// §9, slice S4): one per filesystem session that attaches one.
const STOR_CLIENTS: usize = 4;
/// Ring slots: the lanes first, then the client rings.
const ENGINE_RINGS: usize = ROW_LANES + STOR_CLIENTS;
/// Grants one ring holds: a lane's whole-namespace grant, or a client's
/// pack grant — with room for the file grants reads move onto (S5).
const ENGINE_GRANTS: usize = 4;
/// The engine the backend borrows, placed on the heap in the start path.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
type NvmeEngine = kore_driver::stor::engine::StorEngine<
    'static,
    NvmeQueues,
    ThreeClass,
    ENGINE_RINGS,
    ENGINE_GRANTS,
    RING_TARGETS,
    ENGINE_OPS,
    ENGINE_COMMANDS,
>;
type SharedNvme = EngineBackend<
    'static,
    NvmeQueues,
    ThreeClass,
    ROW_LANES,
    ENGINE_RINGS,
    ENGINE_GRANTS,
    RING_TARGETS,
    ENGINE_OPS,
    ENGINE_COMMANDS,
>;
// The backend is built in the start path, inside the serve future on a
// main stack the row nearly fills; its engine tables (over 100 KiB) live on
// the heap, placed by `StorEngine::init_in`, so the backend itself must stay
// small. Growing it past this is a stack-budget change, not a refactor.
const _: () = assert!(core::mem::size_of::<SharedNvme>() <= 4096);

/// Each lane's class map: the filesystem lanes read on Demand and write on
/// Commit, so their metadata and reads fill ahead of client rings' bulk
/// Stream reads, which stay inside the Stream budget. Pack maintenance is
/// bulk background work: everything it issues rides Stream, behind every
/// read a thread is blocked on.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
const LANE_CLASSES: [LaneClasses; ROW_LANES] = [
    LaneClasses::FILESYSTEM,
    LaneClasses::FILESYSTEM,
    LaneClasses::STREAM,
];

/// One lane's transport window: page-aligned, owned for the row's lifetime
/// (the engine and the lane's client both borrow it `'static`, as the lanes
/// borrow the backend).
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
fn lane_window() -> Option<&'static mut [u8]> {
    let bytes = kore_ring::stor::WINDOW_BYTES;
    let layout = core::alloc::Layout::from_size_align(bytes, 4096).ok()?;
    // SAFETY: the layout is non-zero; the allocation is leaked on purpose —
    // it lives as long as the row, like the backend that borrows it.
    let base = unsafe { alloc::alloc::alloc_zeroed(layout) };
    let base = core::ptr::NonNull::new(base)?;
    // SAFETY: freshly allocated, `bytes` long, exclusively this caller's.
    Some(unsafe { core::slice::from_raw_parts_mut(base.as_ptr(), bytes) })
}

/// Diagnostic classes a refused target import carries beside its error,
/// above the [`IntakeError`](kore_driver::host::IntakeError) range (1..=11),
/// with the `BlockError` in the next byte where one applies: the lent
/// capability was not a memory import, or the backend refused to adopt the
/// imported window.
const CLAIM_REFUSED: u32 = 0xc1;
const ADOPT_REFUSED: u32 = 0xc2;

type SharedLane = EngineLane<
    'static,
    'static,
    NvmeQueues,
    ThreeClass,
    ROW_LANES,
    ENGINE_RINGS,
    ENGINE_GRANTS,
    RING_TARGETS,
    ENGINE_OPS,
    ENGINE_COMMANDS,
>;
type Provider = BlockProvider<RegionDevice<SharedLane>, 8, 32>;

/// Buckets the pack region is judged and refreshed in. A 40 GiB pack makes
/// 40 MiB buckets and a scan of 16 384 probes, about two seconds of idle
/// device time; the report is one 8-byte record per bucket, 8 KiB beside
/// a 64 KiB data window. A finer table costs scan time linearly.
const PACK_BUCKETS: usize = 1024;

/// The pack region's scan-and-refresh job, on [`PACK_LANE`] clamped to the
/// pack's extent.
type PackJob = RegionRefresh<'static, RegionDevice<SharedLane>>;

/// Where the pack maintenance lane stands.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
enum PackMaintenance {
    /// No session has asked yet: the lane is unclaimed.
    Unclaimed,
    Live(Box<PackJob>),
    /// The lane was claimed and could not carry a job (an extent the clamp
    /// refused, no buffer memory), or it was retired at quiesce: every ask
    /// answers `Unsupported` for the rest of this boot.
    Unavailable,
}

/// One engine and its independent persistence generation. The user side may
/// deliberately be volatile; a system region is installed only after its
/// exactly-clamped volume has recovered at, or been rebuilt for, the measured
/// archive era.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
enum MountedEngine {
    Durable {
        fs: Box<FsService<Provider>>,
        mount: PersistenceState,
        commits: u64,
    },
    Volatile(Box<RamService>),
}

#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
impl MountedEngine {
    fn configure_peer_map(&mut self, map: PeerMapPlan) {
        match self {
            Self::Durable { fs, .. } => fs.configure_peer_map(map),
            Self::Volatile(fs) => fs.configure_peer_map(map),
        }
    }

    fn establish_admin_namespace(&mut self) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.establish_admin_namespace(),
            Self::Volatile(fs) => fs.establish_admin_namespace(),
        }
    }

    fn establish_admin_source(&mut self) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.establish_admin_source(),
            Self::Volatile(fs) => fs.establish_admin_source(),
        }
    }

    fn establish_admin_apps(&mut self) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.establish_admin_apps(),
            Self::Volatile(fs) => fs.establish_admin_apps(),
        }
    }

    fn establish_admin_libs(&mut self) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.establish_admin_libs(),
            Self::Volatile(fs) => fs.establish_admin_libs(),
        }
    }

    fn establish_admin_share(&mut self) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.establish_admin_share(),
            Self::Volatile(fs) => fs.establish_admin_share(),
        }
    }

    fn establish_admin_pkgs(&mut self) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.establish_admin_pkgs(),
            Self::Volatile(fs) => fs.establish_admin_pkgs(),
        }
    }

    fn establish_admin_workspace(&mut self) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.establish_admin_workspace(),
            Self::Volatile(fs) => fs.establish_admin_workspace(),
        }
    }

    fn peer_admitted(&mut self, result: Result<PeerId, PeerAdmissionRefusal>) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.peer_admitted(result),
            Self::Volatile(fs) => fs.peer_admitted(result),
        }
    }

    fn mirror_peer_admitted(&mut self, peer: PeerId) -> Result<(), PeerAdmissionRefusal> {
        match self {
            Self::Durable { fs, .. } => fs.mirror_peer_admitted(peer),
            Self::Volatile(fs) => fs.mirror_peer_admitted(peer),
        }
    }

    fn peer_closed(&mut self, via: u16, reason: PeerClose) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.peer_closed(via, reason),
            Self::Volatile(fs) => fs.peer_closed(via, reason),
        }
    }

    fn session_for_via(&self, via: u16) -> Option<SessionId> {
        match self {
            Self::Durable { fs, .. } => fs.session_for_via(via),
            Self::Volatile(fs) => fs.session_for_via(via),
        }
    }

    fn ensure_dynamic_session(&mut self, session: SessionId) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.ensure_dynamic_session(session),
            Self::Volatile(fs) => fs.ensure_dynamic_session(session),
        }
    }

    fn install_apps_view(&mut self, session: SessionId) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.install_apps_view(session),
            Self::Volatile(fs) => fs.install_apps_view(session),
        }
    }

    fn install_libs_view(&mut self, session: SessionId) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.install_libs_view(session),
            Self::Volatile(fs) => fs.install_libs_view(session),
        }
    }

    fn install_share_view(&mut self, session: SessionId) -> bool {
        match self {
            Self::Durable { fs, .. } => fs.install_share_view(session),
            Self::Volatile(fs) => fs.install_share_view(session),
        }
    }

    fn mapped_window_span(&self, session: SessionId) -> Option<crate::service::MappedWindowSpan> {
        match self {
            Self::Durable { fs, .. } => fs.mapped_window_span(session),
            Self::Volatile(fs) => fs.mapped_window_span(session),
        }
    }

    async fn serve_request_async(
        &mut self,
        request: FsRequest,
        sv: &mut Serve<'_, FsResponse>,
    ) -> FsResponse {
        let session = self.session_for_via(sv.via());
        let response = match self {
            Self::Durable { fs, .. } => fs.serve_request_async(request, sv).await,
            Self::Volatile(fs) => fs.serve_request_async(request, sv).await,
        };
        self.commit_after(session, &request, response).await
    }

    async fn dispatch_async(
        &mut self,
        session: SessionId,
        request: &FsRequest,
        window: Option<&mut DataWindow<'_>>,
    ) -> FsResponse {
        let response = match self {
            Self::Durable { fs, .. } => fs.dispatch_async(session, request, window).await,
            Self::Volatile(fs) => fs.dispatch_async(session, request, window).await,
        };
        self.commit_after(Some(session), request, response).await
    }

    async fn commit_after(
        &mut self,
        session: Option<SessionId>,
        request: &FsRequest,
        mut response: FsResponse,
    ) -> FsResponse {
        let Self::Durable { fs, mount, commits } = self else {
            return response;
        };
        let deferring = session.is_some_and(|session| fs.is_deferring(session));
        let closing = matches!(request, FsRequest::DeferCommits { hold: false })
            && !matches!(response, FsResponse::Refused(_));
        if (DurableService::needs_commit(request, &response) && !deferring) || closing {
            if let Err(error) =
                commit_engine(fs.engine(), mount, SessionId::new(0), None::<CommitStage>).await
            {
                response = FsResponse::Refused(DurableService::persistence_error(error));
            } else {
                *commits = commits.saturating_add(1);
            }
        }
        response
    }

    fn degraded(&self) -> bool {
        matches!(
            self,
            Self::Durable { mount, .. }
                if mount.status().mode == crate::persistence::MountMode::DegradedReadOnly
        )
    }

    fn durable(&self) -> bool {
        matches!(self, Self::Durable { .. })
    }
}

/// One client-visible filesystem row over a conservative user engine and an
/// optional rebuildable system engine. The router is the only layer that sees
/// both; each engine retains its original session and authority model.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
struct MountedFs {
    user: MountedEngine,
    system: Option<MountedEngine>,
    /// A GPT system designation suppresses the legacy user-volume fallback
    /// even if its engine could not be made current. Otherwise a failed era
    /// repair would redirect seeding into user-owned bytes.
    system_designated: bool,
    /// Independent logical-region lifecycle facts. They deliberately do not
    /// live on the shared block backend: one controller can host a refused
    /// user region beside a rebuilt system region (or the reverse).
    user_lifecycle: RegionLifecycle,
    system_lifecycle: RegionLifecycle,
    /// Stable post-seed posture restored after exec closes its catalogue
    /// verification/copy window.
    system_seed_resume: RegionLifecycle,
}

impl MountedFs {
    fn set_system_seed_active(&mut self, via: u16, active: bool) -> Result<(), FsError> {
        // Authorize the fixed pair itself, not a derived session number: pair
        // one is exec's founder-installed administration capability, while
        // pair zero is the legacy boot client and recyclable pairs are mapped
        // dynamically.
        if via != crate::service::contract::ADMIN_VIA {
            return Err(FsError::AccessDenied);
        }
        if !self.system_designated || self.system.is_none() {
            return Ok(());
        }
        if active {
            if self.system_lifecycle != RegionLifecycle::Reseeding {
                self.system_seed_resume = self.system_lifecycle;
                self.system_lifecycle = RegionLifecycle::Reseeding;
            }
        } else if self.system_lifecycle == RegionLifecycle::Reseeding {
            self.system_lifecycle = self.system_seed_resume;
        }
        Ok(())
    }

    fn configure_peer_map(&mut self, map: PeerMapPlan) {
        self.user.configure_peer_map(map);
        if let Some(system) = self.system.as_mut() {
            system.configure_peer_map(map);
        }
    }

    fn establish_boot_policy(&mut self) -> Result<(), u64> {
        if !self.user.establish_admin_namespace() {
            return Err(19);
        }
        if !self.user.establish_admin_workspace() {
            return Err(24);
        }
        match self.system.as_mut() {
            Some(system) => {
                if !system.establish_admin_namespace() || !system.establish_admin_source() {
                    return Err(19);
                }
                if !system.establish_admin_apps() {
                    return Err(25);
                }
                if !system.establish_admin_libs() {
                    return Err(26);
                }
                if !system.establish_admin_pkgs() {
                    return Err(27);
                }
                if !system.establish_admin_share() {
                    return Err(31);
                }
            }
            None if !self.system_designated => {
                if !self.user.establish_admin_source() {
                    return Err(19);
                }
                if !self.user.establish_admin_apps() {
                    return Err(25);
                }
                if !self.user.establish_admin_libs() {
                    return Err(26);
                }
                if !self.user.establish_admin_pkgs() {
                    return Err(27);
                }
                if !self.user.establish_admin_share() {
                    return Err(31);
                }
            }
            // A designated system extent exists but is unavailable. Publish
            // neither system grant: exec observes an unavailable store and a
            // later boot can retry the bounded repair. The user workspace is
            // already live and remains the only mutable namespace.
            None => {}
        }
        Ok(())
    }

    fn has_system(&self) -> bool {
        self.system.is_some()
    }

    fn durable(&self) -> bool {
        self.user.durable() || self.system.as_ref().is_some_and(MountedEngine::durable)
    }

    fn degraded(&self) -> bool {
        self.user.degraded() || self.system.as_ref().is_some_and(MountedEngine::degraded)
    }

    fn peer_admitted(&mut self, result: Result<PeerId, PeerAdmissionRefusal>) -> bool {
        let Ok(peer) = result else {
            return self.user.peer_admitted(result);
        };
        if let Some(system) = self.system.as_mut()
            && let Err(refusal) = system.mirror_peer_admitted(peer)
        {
            // Publish the secondary's refusal through the primary engine's
            // one admission completion slot — the secondary owns no
            // control-plane reply — so a system store whose heap is full
            // reads as the full table it is.
            let _ = self.user.peer_admitted(Err(refusal));
            return false;
        }
        let accepted = self.user.peer_admitted(Ok(peer));
        if !accepted && let Some(system) = self.system.as_mut() {
            let _ = system.peer_closed(peer.arrival(), PeerClose::Died);
        }
        accepted
    }

    fn peer_closed(&mut self, via: u16, reason: PeerClose) -> bool {
        let user = self.user.peer_closed(via, reason);
        let system = self
            .system
            .as_mut()
            .is_none_or(|system| system.peer_closed(via, reason));
        user && system
    }

    fn window_for<'a>(span: crate::service::MappedWindowSpan) -> DataWindow<'a> {
        // SAFETY: the primary frontend retains the affine returned-capability
        // owner for `span`; `ServeHost` serializes this row, and every window
        // produced here is consumed before another is made. The fabricated
        // lifetime cannot escape this private function's immediate caller.
        DataWindow::new(unsafe { core::slice::from_raw_parts_mut(span.base.as_ptr(), span.len) })
    }

    fn session(&self, via: u16) -> Result<SessionId, FsResponse> {
        self.user
            .session_for_via(via)
            .ok_or(FsResponse::Refused(FsError::AccessDenied))
    }

    async fn serve_request_async(
        &mut self,
        request: FsRequest,
        sv: &mut Serve<'_, FsResponse>,
    ) -> FsResponse {
        let routed = match router::route(request, self.has_system()) {
            Ok(routed) => routed,
            Err(error) => return FsResponse::Refused(error),
        };
        match routed {
            RoutedRequest::One(Store::User, request) => {
                let applied_peer = match request {
                    FsRequest::ApplyView { peer, .. } => Some(peer),
                    _ => None,
                };
                let response = self.user.serve_request_async(request, sv).await;
                // The user view is the terminal's structural root. The
                // secondary engine mirrors only the explicit read-only apps
                // projection; failure is the established best-effort posture
                // (the terminal still boots without a store view).
                if !matches!(response, FsResponse::Refused(_))
                    && let (Some(peer), Some(system)) = (applied_peer, self.system.as_mut())
                {
                    let _ = system.install_apps_view(SessionId::new(peer.slot() as u16));
                    let _ = system.install_libs_view(SessionId::new(peer.slot() as u16));
                    let _ = system.install_share_view(SessionId::new(peer.slot() as u16));
                }
                router::tag_response(Store::User, response)
            }
            RoutedRequest::One(Store::System, request) => {
                let session = match self
                    .system
                    .as_ref()
                    .and_then(|system| system.session_for_via(sv.via()))
                {
                    Some(session) => session,
                    None => return FsResponse::Refused(FsError::AccessDenied),
                };
                let host_control = matches!(
                    request,
                    FsRequest::BindView { .. } | FsRequest::GrantView { .. }
                );
                let span = request
                    .needs_window()
                    .then(|| self.user.mapped_window_span(session))
                    .flatten();
                let Some(system) = self.system.as_mut() else {
                    return FsResponse::Refused(FsError::InvalidHandle);
                };
                let response = if host_control {
                    system.serve_request_async(request, sv).await
                } else {
                    if !system.ensure_dynamic_session(session) {
                        return FsResponse::Refused(FsError::OutOfObjects);
                    }
                    let mut window = span.map(Self::window_for);
                    system
                        .dispatch_async(session, &request, window.as_mut())
                        .await
                };
                router::tag_response(Store::System, response)
            }
            RoutedRequest::Both(request @ FsRequest::AttachWindow { .. }) => {
                let session = match self.session(sv.via()) {
                    Ok(session) => session,
                    Err(response) => return response,
                };
                let response = self.user.serve_request_async(request, sv).await;
                if matches!(response, FsResponse::Refused(_)) {
                    return response;
                }
                if let Some(system) = self.system.as_mut() {
                    if !system.ensure_dynamic_session(session) {
                        return FsResponse::Refused(FsError::OutOfObjects);
                    }
                    let mirrored = system.dispatch_async(session, &request, None).await;
                    if matches!(mirrored, FsResponse::Refused(_)) {
                        return mirrored;
                    }
                    // One window serves both regions, so one limit must hold
                    // for both: the smaller.
                    if let (
                        FsResponse::WindowAttached { max_transfer: user },
                        FsResponse::WindowAttached {
                            max_transfer: system,
                        },
                    ) = (response, mirrored)
                    {
                        return FsResponse::WindowAttached {
                            max_transfer: user.min(system),
                        };
                    }
                }
                response
            }
            RoutedRequest::Both(request @ FsRequest::DeferCommits { .. }) => {
                let session = match self.session(sv.via()) {
                    Ok(session) => session,
                    Err(response) => return response,
                };
                if !self.user.ensure_dynamic_session(session) {
                    return FsResponse::Refused(FsError::OutOfObjects);
                }
                let user = self.user.dispatch_async(session, &request, None).await;
                if matches!(user, FsResponse::Refused(_)) {
                    return user;
                }
                if let Some(system) = self.system.as_mut() {
                    if !system.ensure_dynamic_session(session) {
                        return FsResponse::Refused(FsError::OutOfObjects);
                    }
                    let system = system.dispatch_async(session, &request, None).await;
                    if matches!(system, FsResponse::Refused(_)) {
                        return system;
                    }
                }
                FsResponse::Done
            }
            RoutedRequest::ListGrants { window_offset } => {
                let session = match self.session(sv.via()) {
                    Ok(session) => session,
                    Err(response) => return response,
                };
                if !self.user.ensure_dynamic_session(session) {
                    return FsResponse::Refused(FsError::OutOfObjects);
                }
                let Some(span) = self.user.mapped_window_span(session) else {
                    return FsResponse::Refused(FsError::NoWindow);
                };
                let request = FsRequest::ListGrants { window_offset };
                let mut window = Self::window_for(span);
                let user = self
                    .user
                    .dispatch_async(session, &request, Some(&mut window))
                    .await;
                let FsResponse::GrantsListed {
                    entries: user_entries,
                } = user
                else {
                    return user;
                };
                if let Err(error) = router::tag_grant_entries(
                    &mut window,
                    window_offset,
                    0,
                    user_entries as usize,
                    Store::User,
                ) {
                    return FsResponse::Refused(error);
                }
                let Some(system) = self.system.as_mut() else {
                    return FsResponse::GrantsListed {
                        entries: user_entries,
                    };
                };
                if !system.ensure_dynamic_session(session) {
                    return FsResponse::Refused(FsError::OutOfObjects);
                }
                let system_offset = match router::grant_append_offset(window_offset, user_entries) {
                    Ok(offset) => offset,
                    Err(error) => return FsResponse::Refused(error),
                };
                let request = FsRequest::ListGrants {
                    window_offset: system_offset,
                };
                let response = system
                    .dispatch_async(session, &request, Some(&mut window))
                    .await;
                let FsResponse::GrantsListed {
                    entries: system_entries,
                } = response
                else {
                    return response;
                };
                if let Err(error) = router::tag_grant_entries(
                    &mut window,
                    system_offset,
                    0,
                    system_entries as usize,
                    Store::System,
                ) {
                    return FsResponse::Refused(error);
                }
                match user_entries.checked_add(system_entries) {
                    Some(entries) => FsResponse::GrantsListed { entries },
                    None => FsResponse::Refused(FsError::Range),
                }
            }
            RoutedRequest::SystemStatus { window_offset } => {
                let session = match self.session(sv.via()) {
                    Ok(session) => session,
                    Err(response) => return response,
                };
                if !self.user.ensure_dynamic_session(session) {
                    return FsResponse::Refused(FsError::OutOfObjects);
                }
                let Some(span) = self.user.mapped_window_span(session) else {
                    return FsResponse::Refused(FsError::NoWindow);
                };
                let request = FsRequest::SystemStatus { window_offset };
                let mut window = Self::window_for(span);
                let response = self
                    .user
                    .dispatch_async(session, &request, Some(&mut window))
                    .await;
                if !matches!(response, FsResponse::SystemStatusReported { .. }) {
                    return response;
                }
                let user = match router::read_status(&window, window_offset) {
                    Ok(status) => status,
                    Err(error) => return FsResponse::Refused(error),
                };
                let Some(system) = self.system.as_mut() else {
                    return response;
                };
                if !system.ensure_dynamic_session(session) {
                    return FsResponse::Refused(FsError::OutOfObjects);
                }
                let response = system
                    .dispatch_async(session, &request, Some(&mut window))
                    .await;
                if !matches!(response, FsResponse::SystemStatusReported { .. }) {
                    return response;
                }
                let system = match router::read_status(&window, window_offset) {
                    Ok(status) => status,
                    Err(error) => return FsResponse::Refused(error),
                };
                let folded = router::fold_status(user, system);
                match window.put_system_status(window_offset, &folded) {
                    Ok(()) => response,
                    Err(error) => FsResponse::Refused(error),
                }
            }
            RoutedRequest::SystemSeed { active } => {
                if let Err(error) = self.set_system_seed_active(sv.via(), active) {
                    return FsResponse::Refused(error);
                }
                FsResponse::Done
            }
            RoutedRequest::Both(_) => FsResponse::Refused(FsError::Unsupported),
            // `DurableService::serve_async` answers the ring verbs before the
            // engines are consulted; reaching here means this row has none.
            RoutedRequest::Stor(_) => FsResponse::Refused(FsError::Unsupported),
        }
    }
}

/// The production filesystem row. `None` exists only during the bounded
/// pre-readiness acquisition of its generic block endpoint.
pub struct DurableService {
    fs: Option<MountedFs>,
    /// The one physical backend both lanes borrow — remembered at mount so
    /// the terminal path (#809 §L) can reach the binding after both lanes
    /// have retired.  `None` in the volatile posture.
    shared: Option<&'static SharedNvme>,
    peer_map: Option<PeerMapPlan>,
    /// The measured image-archive digest founder stamped as immutable boot
    /// data. Only the already-clamped system locator may interpret it as
    /// destructive rebuild policy.
    archive_era: [u8; crate::persistence::ERA_BYTES],
    route_held: bool,
    /// The readiness note ([`note`]) — `0` until initialize decides.
    note: u64,
    /// The pack region, for the read grants client rings are given.
    pack_region: Option<gpt::Extent>,
    /// The pack region's maintenance (`res/PACK_STREAMING.md` §7).
    pack: PackMaintenance,
    /// Client rings, at most one per session.
    rings: [Option<ClientRing>; STOR_CLIENTS],
}

/// The durable row's readiness-note map — the swap row's discipline: a
/// note-less ready line is the fully healthy posture (durable mount, MSI-X
/// route held), and every degradation names itself on the beacon, because
/// on a serial-less machine that line is the only place the fact appears.
pub mod note {
    /// Volatile: the whole disk is foreign and its GPT designates no kore
    /// filesystem partition — the undesignated-machine posture.
    pub const VOLATILE_UNDESIGNATED: u64 = 1;
    /// Volatile: a designated partition exists but was refused — foreign
    /// bytes inside it, an impossible extent, or too small to mount.
    pub const VOLATILE_REFUSED_REGION: u64 = 2;
    /// Durable, but no MSI-X route was realized: the polled posture.
    pub const DURABLE_POLLED: u64 = 3;
    /// Durable, but recovery used the fallback root: degraded read-only.
    pub const DURABLE_DEGRADED: u64 = 4;
    /// The designated system store could not be made current. The user engine
    /// remains mounted; system grants are not published.
    pub const SYSTEM_UNAVAILABLE: u64 = 5;
    /// The designated system store was rebuilt for the measured archive era.
    /// Seeding remains an explicit exec transaction after publication.
    pub const SYSTEM_REBUILT: u64 = 6;
}

/// One logical region's durable lifecycle, separate from the shared physical
/// controller counters and from the row's scalar readiness note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegionLifecycle {
    Unknown,
    Undesignated,
    Refused,
    Reformatting,
    Reseeding,
    Unavailable,
    Rebuilt,
    Current,
}

impl RegionLifecycle {
    const fn wire(self) -> u16 {
        use kore_abi::svc::FsSystemStatusWire;
        match self {
            Self::Unknown => FsSystemStatusWire::LIFECYCLE_UNKNOWN,
            Self::Undesignated => FsSystemStatusWire::LIFECYCLE_UNDESIGNATED,
            Self::Refused => FsSystemStatusWire::LIFECYCLE_REFUSED,
            Self::Reformatting => FsSystemStatusWire::LIFECYCLE_REFORMATTING,
            Self::Reseeding => FsSystemStatusWire::LIFECYCLE_RESEEDING,
            Self::Unavailable => FsSystemStatusWire::LIFECYCLE_UNAVAILABLE,
            Self::Rebuilt => FsSystemStatusWire::LIFECYCLE_REBUILT,
            Self::Current => FsSystemStatusWire::LIFECYCLE_CURRENT,
        }
    }
}

/// Where the media classification landed: a provider to mount at the
/// stated posture, or the deliberate RAM fallback after exact containment
/// (carrying the readiness note that names why).
///
/// Public, with [`DurableService::locate`] and
/// [`DurableService::whole_medium_provider`], so that a host tool working on
/// an offline disk image (`tools/kore-vol`) reaches exactly the region, and
/// takes exactly the posture, the next boot will: one decision procedure,
/// not a second copy of it.
pub enum Located<D: DirectBlockDevice, const B: usize, const O: usize> {
    /// A kore volume: recover it.
    Recover(BlockProvider<RegionDevice<D>, B, O>),
    /// Blank roots: format the region, stamped with `era`.
    Format {
        provider: BlockProvider<RegionDevice<D>, B, O>,
        era: [u8; crate::persistence::ERA_BYTES],
    },
    /// Reachable only from [`DurableService::locate_system`], after the
    /// `KORE_SYSTEM_TYPE` extent is beneath the provider's clamp.
    Reformat {
        provider: BlockProvider<RegionDevice<D>, B, O>,
        era: [u8; crate::persistence::ERA_BYTES],
    },
    /// No durable region: the RAM posture, with the [`note`] that says why.
    Volatile(u64),
}

impl DurableService {
    /// Consult a foreign disk's GPT — read-only — for the partition whose
    /// type GUID designates the kore filesystem.  Every failure (no table,
    /// damaged table, no match, an entry array outside the provider's
    /// guarded metadata window) answers `None`: without an exact
    /// designation there is no write authority anywhere on this disk.
    async fn discover_region<D: DirectBlockDevice, const B: usize, const O: usize>(
        provider: &BlockProvider<RegionDevice<D>, B, O>,
        type_guid: gpt::Guid,
    ) -> Option<gpt::Extent> {
        let geometry = provider.geometry();
        let block = geometry.block_size as usize;
        if block > 4096 {
            return None;
        }
        let mut header = [0u8; 4096];
        provider
            .read_metadata(gpt::PRIMARY_HEADER_LBA, &mut header[..block])
            .await
            .ok()?;
        let table = gpt::parse_header(&header[..block]).ok()?;
        let array_bytes = table.entry_array_bytes();
        let blocks = array_bytes.div_ceil(block);
        let end = table.entry_lba.checked_add(blocks as u64)?;
        if end > crate::provider::GPT_DISCOVERY_END_LBA {
            return None;
        }
        let len = blocks * block;
        let mut entries = [0u8; gpt::MAX_ENTRY_ARRAY_BYTES];
        provider
            .read_metadata(table.entry_lba, &mut entries[..len])
            .await
            .ok()?;
        // The type GUID is a parameter, not a constant, because issue #626
        // discovers TWO kore regions on one disk: the user region
        // ([`gpt::KORE_FS_TYPE`]) and the self-healing system store
        // ([`gpt::KORE_SYSTEM_TYPE`]). GPT partitions never overlap, so the two
        // extents this returns for the two type GUIDs are disjoint by
        // construction — the foundation of the destructive-ops containment the
        // system formatter relies on.
        gpt::find_partition(&entries[..len], &table, type_guid).ok()
    }

    /// The provider over a whole medium, as the row composes it before it
    /// classifies anything: the device behind an identity [`RegionDevice`]
    /// (which [`Self::locate`] may narrow onto a designated partition), its
    /// channel, and the current profile's bootstrap data start. Detail 14
    /// is a device or channel the composition refuses, 15 a geometry with no
    /// room for the metadata reservation.
    pub fn whole_medium_provider<D: DirectBlockDevice, const B: usize, const O: usize>(
        device: D,
        mode: CompletionMode,
    ) -> Result<BlockProvider<RegionDevice<D>, B, O>, StartFailure> {
        let device = RegionDevice::whole(device).map_err(|_| StartFailure::new(14))?;
        let channel = BlockChannel::new(device, mode).map_err(|_| StartFailure::new(14))?;
        let bootstrap = bootstrap_data_start(channel.geometry().block_size)
            .map_err(|_| StartFailure::new(15))?;
        BlockProvider::new_with_data_start(channel, bootstrap).map_err(|_| StartFailure::new(15))
    }

    /// Classify the provisioned medium and land on exactly one mount
    /// posture.  A whole disk that is kore's or blank keeps the identity
    /// view; a foreign disk is consulted for its designated partition and,
    /// when one exists, the composition narrows onto it behind the
    /// [`RegionDevice`] clamp and re-probes inside.  Designation admits
    /// the region; only recognized or blank roots inside it authorize user
    /// writes — stale foreign bytes inside a user partition still land in
    /// the RAM posture rather than being overwritten. The separate system
    /// locator owns its explicit archive-era rebuild policy, so this never
    /// answers [`Located::Reformat`].
    pub async fn locate<D: DirectBlockDevice, const B: usize, const O: usize>(
        provider: BlockProvider<RegionDevice<D>, B, O>,
        mode: CompletionMode,
    ) -> Result<Located<D, B, O>, StartFailure> {
        let probe = PersistentFs::probe(&provider)
            .await
            .map_err(|_| StartFailure::new(16))?;
        let provider = match probe {
            VolumeProbe::Kore => return Ok(Located::Recover(provider)),
            VolumeProbe::Blank => {
                return Ok(Located::Format {
                    provider,
                    era: [0; crate::persistence::ERA_BYTES],
                });
            }
            VolumeProbe::Foreign => provider,
        };
        let Some(extent) = Self::discover_region(&provider, gpt::KORE_FS_TYPE).await else {
            // A normal metal boot disk begins with GPT/MBR metadata and
            // designates nothing for kore. Close the direct DMA channel
            // child-before-parent, then keep the service available through
            // the identical bounded RAM composition until an installer
            // provisions a designated partition.
            provider.retire().await.map_err(|_| StartFailure::new(20))?;
            return Ok(Located::Volatile(note::VOLATILE_UNDESIGNATED));
        };
        // Rebuild the composition narrowed onto the designation. The
        // channel and the bound backend ride through intact; only the
        // boot-local buffers and the whole-disk view retire.
        let channel = provider.into_channel().map_err(|_| StartFailure::new(22))?;
        let device = match channel.into_device() {
            Ok(device) => device,
            Err(_) => return Err(StartFailure::new(22)),
        };
        let device = match device.narrow(extent.start_lba, extent.blocks()) {
            Ok(device) => device,
            // A designated partition the clamp refuses (impossible extent)
            // conveys no authority; the drop path contains the binding.
            Err(_) => return Ok(Located::Volatile(note::VOLATILE_REFUSED_REGION)),
        };
        let channel = BlockChannel::new(device, mode).map_err(|_| StartFailure::new(22))?;
        let bootstrap = match bootstrap_data_start(channel.geometry().block_size) {
            Ok(bootstrap) => bootstrap,
            Err(_) => return Ok(Located::Volatile(note::VOLATILE_REFUSED_REGION)),
        };
        let provider = match BlockProvider::new_with_data_start(channel, bootstrap) {
            Ok(provider) => provider,
            // Too small for the metadata reservation: a provisioning fault
            // that conveys no formatting authority.
            Err(_) => return Ok(Located::Volatile(note::VOLATILE_REFUSED_REGION)),
        };
        let probe = PersistentFs::probe(&provider)
            .await
            .map_err(|_| StartFailure::new(23))?;
        match probe {
            VolumeProbe::Kore => Ok(Located::Recover(provider)),
            VolumeProbe::Blank => Ok(Located::Format {
                provider,
                era: [0; crate::persistence::ERA_BYTES],
            }),
            VolumeProbe::Foreign => {
                provider.retire().await.map_err(|_| StartFailure::new(20))?;
                Ok(Located::Volatile(note::VOLATILE_REFUSED_REGION))
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    async fn mount_located(
        located: Located<SharedLane, 8, 32>,
        session: SessionId,
    ) -> Result<(MountedEngine, Option<u64>, RegionLifecycle), StartFailure> {
        match located {
            Located::Recover(provider) => {
                // Detail 17 keeps its low byte; the witness above it names
                // the recovery stage and its cause, so a photographed
                // refusal says why the workspace did not mount.
                let volume = PersistentFs::recover_witnessed(provider, session)
                    .await
                    .map_err(|failure| StartFailure::new(17 | (u64::from(failure.witness()) << 8)))?;
                let (engine, mount) = volume.into_parts();
                let mut fs = Box::new(FsService::new(engine));
                fs.bind_session(session);
                Ok((
                    MountedEngine::Durable {
                        fs,
                        mount,
                        commits: 0,
                    },
                    None,
                    RegionLifecycle::Current,
                ))
            }
            Located::Format { provider, era } => {
                let lifecycle = if era.iter().all(|byte| *byte == 0) {
                    RegionLifecycle::Current
                } else {
                    RegionLifecycle::Rebuilt
                };
                let volume = PersistentFs::format_with_era(provider, session, era)
                    .await
                    .map_err(|_| StartFailure::new(18))?;
                let (engine, mount) = volume.into_parts();
                let mut fs = Box::new(FsService::new(engine));
                fs.bind_session(session);
                Ok((
                    MountedEngine::Durable {
                        fs,
                        mount,
                        commits: 0,
                    },
                    None,
                    lifecycle,
                ))
            }
            Located::Reformat { provider, era } => {
                let volume = PersistentFs::reformat_with_era(provider, session, era)
                    .await
                    .map_err(|_| StartFailure::new(30))?;
                let (engine, mount) = volume.into_parts();
                let mut fs = Box::new(FsService::new(engine));
                fs.bind_session(session);
                Ok((
                    MountedEngine::Durable {
                        fs,
                        mount,
                        commits: 0,
                    },
                    Some(note::SYSTEM_REBUILT),
                    RegionLifecycle::Rebuilt,
                ))
            }
            Located::Volatile(note) => {
                let lifecycle = match note {
                    note::VOLATILE_UNDESIGNATED => RegionLifecycle::Undesignated,
                    note::VOLATILE_REFUSED_REGION => RegionLifecycle::Refused,
                    _ => RegionLifecycle::Unknown,
                };
                Ok((
                    MountedEngine::Volatile(Box::new(<RamService as SystemService>::bind(
                        FsEndowment::UNSTAMPED,
                    ))),
                    Some(note),
                    lifecycle,
                ))
            }
        }
    }

    /// Build lane one directly inside the authenticated system extent and
    /// apply the archive-era policy there, never at the whole-disk or user
    /// provider layers.
    ///
    /// All usable roots must be current before recovery is admitted. That is
    /// deliberately stronger than checking only the newest root: ordinary
    /// recovery may select the fallback if the newest image is corrupt.
    async fn locate_system<D: DirectBlockDevice>(
        lane: D,
        extent: gpt::Extent,
        mode: CompletionMode,
        archive_era: [u8; crate::persistence::ERA_BYTES],
    ) -> Result<Option<Located<D, 8, 32>>, StartFailure> {
        let whole = RegionDevice::whole(lane).map_err(|_| StartFailure::new(26))?;
        let device = match whole.narrow(extent.start_lba, extent.blocks()) {
            Ok(device) => device,
            Err((whole, _)) => {
                // A rejected GPT extent must not strand a live shared lane.
                // Retiring this one lane leaves the user mount untouched and
                // quiesces the controller only if it was the last owner.
                let mut lane = whole.into_inner();
                let _ = lane.quiesce();
                return Err(StartFailure::new(26));
            }
        };
        let channel = BlockChannel::new(device, mode).map_err(|_| StartFailure::new(26))?;
        let bootstrap = bootstrap_data_start(channel.geometry().block_size)
            .map_err(|_| StartFailure::new(26))?;
        let provider = BlockProvider::new_with_data_start(channel, bootstrap)
            .map_err(|_| StartFailure::new(26))?;
        // A TBD manifest digest is absence of policy, not an era. Never turn
        // it into authority to destroy a designated region.
        if archive_era.iter().all(|byte| *byte == 0) {
            provider.retire().await.map_err(|_| StartFailure::new(28))?;
            return Ok(None);
        }
        let probe = match PersistentFs::probe(&provider).await {
            Ok(probe) => probe,
            Err(_) => {
                // Probe still borrows the provider, so contain the sibling
                // lane explicitly before degrading to the user-only mount.
                provider.retire().await.map_err(|_| StartFailure::new(28))?;
                return Err(StartFailure::new(27));
            }
        };
        match probe {
            VolumeProbe::Kore => {
                let current = match PersistentFs::roots_match_era(&provider, archive_era).await {
                    Ok(current) => current,
                    Err(_) => {
                        provider.retire().await.map_err(|_| StartFailure::new(28))?;
                        return Err(StartFailure::new(29));
                    }
                };
                Ok(Some(if current {
                    Located::Recover(provider)
                } else {
                    Located::Reformat {
                        provider,
                        era: archive_era,
                    }
                }))
            }
            VolumeProbe::Blank => Ok(Some(Located::Format {
                provider,
                era: archive_era,
            })),
            VolumeProbe::Foreign => Ok(Some(Located::Reformat {
                provider,
                era: archive_era,
            })),
        }
    }

    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    async fn initialize(
        &mut self,
        opened: kore_svc_block::OpenedChannel,
    ) -> Result<(), StartFailure> {
        let backend = DirectNvme::claim(opened).map_err(|_| StartFailure::new(13))?;
        let realized_route = backend.has_route();
        let mode = if realized_route {
            CompletionMode::Interrupt
        } else {
            CompletionMode::Poll
        };
        // The service heap owns the one physical backend for the row's whole
        // lifetime. Both affine lanes borrow it, so their provider types prove
        // the controller outlives either mount without `Rc` or a global.
        let windows = [lane_window(), lane_window(), lane_window()];
        let [Some(user_window), Some(system_window), Some(pack_window)] = windows else {
            return Err(StartFailure::new(14));
        };
        // The engine's tables (over 100 KiB) are written straight into
        // their heap slot: this runs inside the serve future on a main
        // stack the row already nearly fills, so an engine-sized temporary
        // here is a stack overflow waiting for the next table to grow.
        let place: &'static mut core::mem::MaybeUninit<NvmeEngine> =
            Box::leak(Box::<NvmeEngine>::new_uninit());
        let engine = NvmeEngine::init_in(place, backend).map_err(|_| StartFailure::new(14))?;
        let shared: &'static SharedNvme = Box::leak(Box::new(
            SharedNvme::new(engine, [user_window, system_window, pack_window], LANE_CLASSES)
                .map_err(|_| StartFailure::new(14))?,
        ));
        self.shared = Some(shared);
        let user_lane = shared.claim(0).map_err(|_| StartFailure::new(14))?;
        let provider = Self::whole_medium_provider(user_lane, mode)?;
        let session = SessionId::new(0);
        // Read every designation while lane zero still has the whole-disk
        // view. Claim the other lanes before `locate` may retire lane zero:
        // the last live lane is what quiesces the physical controller.
        let foreign = PersistentFs::probe(&provider).await == Ok(VolumeProbe::Foreign);
        let system_extent = if foreign {
            Self::discover_region(&provider, gpt::KORE_SYSTEM_TYPE).await
        } else {
            None
        };
        let system_lane = match system_extent {
            Some(_) => Some(shared.claim(1).map_err(|_| StartFailure::new(26))?),
            None => None,
        };
        // The pack region is optional in every posture: without one the
        // filesystem mounts exactly as before and a pack grant answers
        // `Unsupported`.
        let pack_extent = if foreign {
            Self::discover_region(&provider, gpt::KORE_PACK_TYPE).await
        } else {
            None
        };
        let system_designated = system_extent.is_some();
        let user_located = Self::locate(provider, mode).await?;
        let (user, user_note, user_lifecycle) = Self::mount_located(user_located, session).await?;
        if let Some(note) = user_note {
            self.note = note;
        }
        let mut system_lifecycle = if system_designated {
            RegionLifecycle::Unavailable
        } else {
            RegionLifecycle::Undesignated
        };
        let mut system_seed_resume = system_lifecycle;
        let system = match (system_lane, system_extent) {
            (Some(lane), Some(extent)) => {
                match Self::locate_system(lane, extent, mode, self.archive_era).await {
                    Ok(Some(located)) => match Self::mount_located(located, session).await {
                        Ok((mounted, mount_note, lifecycle)) => {
                            system_seed_resume = lifecycle;
                            // A rebuild has made the volume transactionally
                            // sound, but its measured catalogue is not current
                            // until exec verifies/copies every member. Enter
                            // that phase before status can observe the mount.
                            system_lifecycle = if lifecycle == RegionLifecycle::Rebuilt {
                                RegionLifecycle::Reseeding
                            } else {
                                lifecycle
                            };
                            if let Some(note) = mount_note {
                                self.note = note;
                            }
                            Some(mounted)
                        }
                        Err(_) => {
                            // A failed rebuild/mount never retracts the user
                            // engine already mounted above and never publishes
                            // system grants. The next boot retries from the
                            // flushed dual-root transaction boundary.
                            self.note = note::SYSTEM_UNAVAILABLE;
                            system_lifecycle = RegionLifecycle::Unavailable;
                            None
                        }
                    },
                    Ok(None) => {
                        self.note = note::SYSTEM_UNAVAILABLE;
                        system_lifecycle = RegionLifecycle::Unavailable;
                        None
                    }
                    Err(_) => {
                        self.note = note::SYSTEM_UNAVAILABLE;
                        system_lifecycle = RegionLifecycle::Refused;
                        None
                    }
                }
            }
            _ => None,
        };
        self.pack_region = pack_extent;
        let mut mounted = MountedFs {
            user,
            system,
            system_designated,
            user_lifecycle,
            system_lifecycle,
            system_seed_resume,
        };
        mounted.establish_boot_policy().map_err(StartFailure::new)?;
        if let Some(map) = self.peer_map {
            mounted.configure_peer_map(map);
        }
        self.route_held = realized_route && mounted.durable();
        self.note = if matches!(self.note, note::SYSTEM_UNAVAILABLE | note::SYSTEM_REBUILT) {
            // System-era posture is the one fact the scalar readiness note
            // cannot reconstruct from the mounted engines. Route and degraded
            // state remain present in the structured status record.
            self.note
        } else if mounted.degraded() {
            note::DURABLE_DEGRADED
        } else if mounted.durable() && !self.route_held {
            note::DURABLE_POLLED
        } else {
            self.note
        };
        self.fs = Some(mounted);
        Ok(())
    }

    /// The readiness note computed by [`initialize`](Self::initialize) —
    /// `0` is the fully healthy durable posture; see [`note`].
    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    fn readiness_note(&self) -> u64 {
        self.note
    }

    /// Whether the mounted composition holds its realized MSI-X route —
    /// the entry's `enable_irq_parks` decider.  The RAM posture (and the
    /// pre-initialize window) holds none: its route died with the retired
    /// channel, and arming the executor's idle parks on a dead route would
    /// wedge every later park.
    #[cfg(all(target_arch = "x86_64", target_os = "none"))]
    fn holds_route(&self) -> bool {
        self.route_held
    }

    fn refresh_engine_status(
        mounted: &mut MountedEngine,
        note: u64,
        route_held: bool,
        include_shared_waits: bool,
        user_lifecycle: RegionLifecycle,
        system_lifecycle: RegionLifecycle,
    ) {
        match mounted {
            MountedEngine::Durable { fs, mount, commits } => {
                let provider = fs.engine_ref().provider();
                let volume = provider.geometry();
                let occupied_blocks = provider.occupied_blocks();
                let channel_lost =
                    provider.channel_state() == kore_driver::block::ChannelState::Lost;
                let device = provider
                    .control(|region| {
                        let shared = region.inner().backend();
                        let physical = shared.geometry();
                        shared.control(|queues| {
                            let device = queues;
                            let route = device.route_stats();
                            let health = device.health().ok();
                            let hmb = device.host_memory_buffer_bytes();
                            (route, physical, health, hmb)
                        })
                    })
                    .ok()
                    .and_then(Result::ok);
                let (
                    mut route_parks,
                    mut route_fires,
                    mut deadline_waits,
                    mut spin_waits,
                    mut monitor_sleeps,
                    mut monitor_turns,
                ) = device.map(|(route, _, _, _)| route).unwrap_or_default();
                let host_memory_buffer = device.is_some_and(|(_, _, _, hmb)| hmb != 0);
                // These counters live on the one physical backend, not on a
                // lane. Report them through exactly one engine before the two
                // status records are folded, or every wait would double.
                if !include_shared_waits {
                    route_parks = 0;
                    route_fires = 0;
                    deadline_waits = 0;
                    spin_waits = 0;
                    monitor_sleeps = 0;
                    monitor_turns = 0;
                }
                let storage = match (device, occupied_blocks) {
                    (Some((_, physical, health, _)), Some(occupied_blocks))
                        if physical.block_size == volume.block_size =>
                    {
                        let block_bytes = u64::from(volume.block_size);
                        let storage = kore_abi::monitor::FilesystemStorageSnapshot::new(
                            volume.block_size,
                            physical.bytes(),
                            volume.bytes(),
                            occupied_blocks.saturating_mul(block_bytes),
                            u64::from(volume.max_transfer_blocks).saturating_mul(block_bytes),
                            volume.read_only
                                || mount.status().mode
                                    == crate::persistence::MountMode::DegradedReadOnly,
                            volume.flush,
                            volume.discard,
                        )
                        .with_layout(
                            volume.io_hints.supported_block_sizes,
                            u64::from(volume.io_hints.preferred_write_blocks)
                                .saturating_mul(block_bytes),
                            u64::from(volume.io_hints.optimal_write_blocks)
                                .saturating_mul(block_bytes),
                        );
                        health.map_or(storage, |health| {
                            storage.with_health(kore_abi::monitor::FilesystemHealthSnapshot::new(
                                health.critical_warning,
                                health.temperature_celsius,
                                health.available_spare_percent,
                                health.spare_threshold_percent,
                                health.percentage_used,
                                health.power_cycles,
                                health.power_on_hours,
                                health.unsafe_shutdowns,
                                health.media_errors,
                            ))
                        })
                    }
                    _ => kore_abi::monitor::FilesystemStorageSnapshot::EMPTY,
                };
                let backend = crate::service::FsBackendStatus {
                    route_parks,
                    route_fires,
                    deadline_waits,
                    spin_waits,
                    monitor_sleeps,
                    monitor_turns,
                    durable: true,
                    degraded_read_only: mount.status().mode
                        == crate::persistence::MountMode::DegradedReadOnly,
                    route_held,
                    channel_lost,
                    host_memory_buffer,
                    note,
                    user_lifecycle: user_lifecycle.wire(),
                    system_lifecycle: system_lifecycle.wire(),
                    commits: *commits,
                    provider: fs.engine_ref().provider().stats(),
                    channel: fs.engine_ref().provider().channel_stats(),
                    storage,
                };
                fs.set_backend_status(backend);
            }
            MountedEngine::Volatile(fs) => {
                fs.set_backend_status(crate::service::FsBackendStatus {
                    note,
                    user_lifecycle: user_lifecycle.wire(),
                    system_lifecycle: system_lifecycle.wire(),
                    ..Default::default()
                });
            }
        }
    }

    /// Freeze each backend's independent counters immediately before the
    /// router asks the engines for their status records.
    fn refresh_backend_status(&mut self) {
        let note = self.note;
        let route_held = self.route_held;
        let Some(mounted) = self.fs.as_mut() else {
            return;
        };
        let user_owns_shared_waits = mounted.user.durable();
        Self::refresh_engine_status(
            &mut mounted.user,
            note,
            route_held,
            user_owns_shared_waits,
            mounted.user_lifecycle,
            mounted.system_lifecycle,
        );
        if let Some(system) = mounted.system.as_mut() {
            Self::refresh_engine_status(
                system,
                0,
                route_held,
                !user_owns_shared_waits,
                RegionLifecycle::Unknown,
                RegionLifecycle::Unknown,
            );
        }
    }

    fn needs_commit(request: &FsRequest, response: &FsResponse) -> bool {
        if matches!(response, FsResponse::Refused(_)) {
            return false;
        }
        matches!(
            request,
            FsRequest::Create { .. }
                | FsRequest::WriteAt { .. }
                | FsRequest::CommitReplace { .. }
                | FsRequest::Append { .. }
                | FsRequest::Resize { .. }
                | FsRequest::Rename { .. }
                | FsRequest::Relocate { .. }
                | FsRequest::Link { .. }
                | FsRequest::Unlink { .. }
                | FsRequest::DetachEdge { .. }
                | FsRequest::Sync { .. }
                | FsRequest::CommitDerived { .. }
        )
    }

    fn persistence_error(error: PersistenceError) -> FsError {
        match error {
            PersistenceError::Provider(error) => error.into(),
            PersistenceError::Filesystem(error) => error,
            PersistenceError::Unformatted
            | PersistenceError::Incompatible
            | PersistenceError::Corrupt
            | PersistenceError::Bounds
            | PersistenceError::Interrupted(_) => {
                FsError::ProviderFailure(ProviderError::Integrity)
            }
        }
    }
}

/// Retire a target the engine gave back with its proof, at once and with
/// the controller running ([`DirectNvme::retire_quiet`]). `true` once its
/// device mapping is gone.
fn retire_target(shared: &SharedNvme, target: Quiet<NvmeBuffer<CapBuf>>) -> bool {
    shared.control(|queues| queues.retire_quiet(target)).unwrap_or(false)
}


/// One session's storage ring (`res/STORAGE_TRANSPORT.md` §9, slice S4): a
/// window the client funded and lent, attached to the engine beside the
/// lanes. The client submits by writing its own window; the pair carries
/// only control.
struct ClientRing {
    session: SessionId,
    ring: RingId,
    /// The lent view's owner. The engine borrows the mapped window for
    /// `'static`, so this is dropped only after the engine retired the ring
    /// and handed the window back ([`EngineBackend::retire_client`]).
    _window: kore_rt::cap::ReturnedCapability,
    /// Target slots adopted and not yet released.
    targets: u64,
    /// Version grants: the lease (store-tagged wire bits) each pins, and its
    /// grant. A lease with a grant here is released only once the grant
    /// has drained ([`DurableService::settle_version_grant`]).
    versions: [Option<(u64, GrantId)>; VERSION_GRANTS],
}

/// Version grants one ring may hold at once: every grant slot but the one a
/// pack grant takes.
const VERSION_GRANTS: usize = ENGINE_GRANTS - 1;

/// A version's whole blocks as one read grant (the extent is contiguous;
/// runs split only at the run length's width).
fn version_grant(blocks: crate::provider::DeviceBlocks) -> Result<Grant, FsError> {
    let mut runs = [ExtentRun::default(); kore_driver::stor::grant::MAX_RUNS];
    let mut count = 0;
    let mut lba = blocks.lba;
    let end = blocks.lba.checked_add(blocks.blocks).ok_or(FsError::Range)?;
    while lba < end {
        let run = runs.get_mut(count).ok_or(FsError::Range)?;
        let span = (end - lba).min(u64::from(u32::MAX)) as u32;
        *run = ExtentRun { lba, blocks: span };
        count += 1;
        lba += u64::from(span);
    }
    Grant::new(&runs[..count], Rights::READ, ClassSet::DATA).map_err(|_| FsError::Range)
}

const _: () = assert!(RING_TARGETS <= 64, "a ring's targets are a u64 mask");

/// The pack region as one read grant: every block, in as many runs as the
/// run length's width needs. Demand for reads a thread waits on, Stream for
/// read-ahead — never Commit.
fn pack_grant(extent: gpt::Extent) -> Result<Grant, FsError> {
    let mut runs = [ExtentRun::default(); kore_driver::stor::grant::MAX_RUNS];
    let mut count = 0;
    let mut lba = extent.start_lba;
    let end = extent.start_lba.saturating_add(extent.blocks());
    while lba < end {
        let run = runs.get_mut(count).ok_or(FsError::Range)?;
        let blocks = (end - lba).min(u64::from(u32::MAX)) as u32;
        *run = ExtentRun { lba, blocks };
        count += 1;
        lba += u64::from(blocks);
    }
    Grant::new(&runs[..count], Rights::READ, ClassSet::DATA).map_err(|_| FsError::Range)
}

/// Take down one client ring: stop its popping, hand back every target
/// nothing uses, retire it once drained, then drop the window's owner.
/// `false` while commands are still in flight: ask again after a later wake.
fn retire_ring(shared: &SharedNvme, slot: &mut Option<ClientRing>) -> bool {
    let Some(entry) = slot.as_mut() else {
        return true;
    };
    if shared.detach_client(entry.ring).is_err() {
        return false;
    }
    let mut adopted = entry.targets;
    while adopted != 0 {
        let index = adopted.trailing_zeros() as u16;
        adopted &= adopted - 1;
        if let Ok(target) = shared.release_client(entry.ring, index) {
            entry.targets &= !(1u64 << index);
            let _ = retire_target(shared, target);
        }
    }
    if entry.targets != 0 || shared.retire_client(entry.ring).is_err() {
        return false;
    }
    // The engine no longer references the window: the mapping may go.
    *slot = None;
    true
}

/// Map a lent ring window and attach it. The window is the client's shared
/// memory; the engine reads it only through the channel protocols.
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
fn attach_ring(
    shared: &SharedNvme,
    owner: kore_rt::cap::ReturnedCapability,
) -> Result<(RingId, kore_rt::cap::ReturnedCapability), FsError> {
    use kore_rt::cap::verbs;
    let slot = owner.slot().ok_or(FsError::NoWindow)?;
    let view = verbs::take_view(slot).ok_or(FsError::NoWindow)?;
    // A sink mapping (`res/STORAGE_TRANSPORT.md` §12, option B): should the
    // client die with its ring attached, the engine's next pass reads a sink
    // page — bytes the channel protocols already decode safely — instead of
    // faulting on a hole, until `peer_closed` retires the ring.
    let mapped = verbs::map_view_sink(view).ok_or(FsError::NoWindow)?;
    if mapped.len() < kore_ring::stor::WINDOW_BYTES {
        return Err(FsError::Range);
    }
    // SAFETY: `map_view` installed `[base, +len)` in this task's space over
    // the lent view, page-aligned; `owner` keeps it mapped and is dropped
    // only after the engine retires the ring (`retire_ring`). The client
    // writes the same bytes concurrently: the engine touches them only
    // through the channel protocols (atomics, masked indices, records that
    // decode from any bytes), never through this slice directly.
    let window: &'static mut [u8] = unsafe {
        core::slice::from_raw_parts_mut(mapped.cast::<u8>().as_ptr(), kore_ring::stor::WINDOW_BYTES)
    };
    let ring = shared.attach_client(window, ClassSet::DATA).map_err(|error| match error {
        // `service` refuses a window the client never formatted.
        BlockError::Geometry => FsError::Range,
        BlockError::Backpressure => FsError::OutOfHandles,
        other => FsError::ProviderFailure(crate::provider::map_block(other)),
    })?;
    Ok((ring, owner))
}

/// Host builds have no capability arena to map from.
#[cfg(not(all(target_arch = "x86_64", target_os = "none")))]
fn attach_ring(
    _shared: &SharedNvme,
    _owner: kore_rt::cap::ReturnedCapability,
) -> Result<(RingId, kore_rt::cap::ReturnedCapability), FsError> {
    Err(FsError::Unsupported)
}

/// Import a lent device route and adopt it as a target of `ring`, reads
/// landing in it.
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
fn attach_ring_target(
    shared: &SharedNvme,
    ring: RingId,
    owner: kore_rt::cap::ReturnedCapability,
) -> Result<(u16, u64), FsError> {
    // Settle what earlier clients' windows owe before admitting this one:
    // the whole pass if the controller is idle, else the live retries.
    if shared.control_idle(|queues| queues.retry_retirement()).is_err() {
        let _ = shared.control(|queues| queues.retry_live_retirement());
    }
    let target = shared
        .control(|queues| queues.adopt_target(owner))
        .map_err(|error| FsError::ProviderFailure(crate::provider::map_block(error)))?
        .map_err(|(error, _)| error)?;
    let bytes = kore_driver::block::BlockBuffer::len(&target) as u64;
    match shared.adopt_client(ring, target, TargetAccess::ReadInto) {
        Ok(slot) => Ok((slot, bytes)),
        // The mapping retires before the refusal answers, so the lender may
        // retire its route; one that outlived it answers `Busy`, as a
        // release does.
        Err((target, error)) => Err(if !retire_target(shared, target) {
            FsError::Busy
        } else {
            match error {
                BlockError::Backpressure => FsError::OutOfHandles,
                other => FsError::ProviderFailure(crate::provider::map_block(other)),
            }
        }),
    }
}

/// Host builds have no capability arena to import from.
#[cfg(not(all(target_arch = "x86_64", target_os = "none")))]
fn attach_ring_target(
    _shared: &SharedNvme,
    _ring: RingId,
    _owner: kore_rt::cap::ReturnedCapability,
) -> Result<(u16, u64), FsError> {
    Err(FsError::Unsupported)
}

/// Claim [`PACK_LANE`] and build the pack's maintenance job over it,
/// clamped to `extent`. Every refusal retires the claimed lane, so a failed
/// claim never holds the controller up at quiesce.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
fn claim_pack(shared: &'static SharedNvme, extent: gpt::Extent) -> PackMaintenance {
    let Ok(lane) = shared.claim(PACK_LANE) else {
        return PackMaintenance::Unavailable;
    };
    // The lane's geometry is the one the backend validated when it was
    // built, so the identity view cannot be refused.
    let Ok(whole) = RegionDevice::whole(lane) else {
        return PackMaintenance::Unavailable;
    };
    let device = match whole.narrow(extent.start_lba, extent.blocks()) {
        Ok(device) => device,
        Err((whole, _)) => {
            let _ = whole.into_inner().quiesce();
            return PackMaintenance::Unavailable;
        }
    };
    // The bucket books live as long as the row, as the lanes' windows do.
    let buckets: &'static mut [Bucket] =
        Box::leak(alloc::vec![Bucket::default(); PACK_BUCKETS].into_boxed_slice());
    match RegionRefresh::new(device, buckets) {
        Ok(job) => PackMaintenance::Live(Box::new(job)),
        Err((device, _)) => {
            let _ = device.into_inner().quiesce();
            PackMaintenance::Unavailable
        }
    }
}

/// Stop `pack`'s job, wait out what the device holds, and retire its lane.
/// `false` when a command would not come back: the lane stays claimed and
/// the sweep is partial.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
fn retire_pack(pack: PackMaintenance) -> bool {
    let PackMaintenance::Live(mut job) = pack else {
        return true;
    };
    if !job.drain() {
        // Leaked with its lane: the device may still write into its
        // buffers, which must therefore never be reused.
        Box::leak(job);
        return false;
    }
    match (*job).into_device() {
        Ok(device) => device.into_inner().quiesce().is_ok(),
        Err(job) => {
            Box::leak(Box::new(job));
            false
        }
    }
}

/// Land `job`'s report in `window` at `window_offset`: the summary, then
/// as many bucket records as fit. Answers the bytes landed.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
fn write_pack_report(job: &PackJob, window: &mut DataWindow<'_>, window_offset: u32) -> Result<u32, FsError> {
    use kore_abi::svc::{PackBucketWire, PackMaintenanceWire as Wire};
    use zerocopy::IntoBytes;
    use zerocopy::byteorder::little_endian::{U16, U32, U64};
    let short = |n: u32| U16::new(u16::try_from(n).unwrap_or(u16::MAX));
    let room = window
        .len()
        .saturating_sub(window_offset as usize)
        .saturating_sub(Wire::BYTES)
        / PackBucketWire::BYTES;
    let buckets = job.buckets();
    let entries = buckets.len().min(room);
    let shape = job.shape();
    let scan = job.scan_summary();
    let books = job.refresh_summary();
    let (outcome, fault) = match job.outcome() {
        refresh::Outcome::None => (Wire::OUTCOME_NONE, 0),
        refresh::Outcome::Scanned => (Wire::OUTCOME_SCANNED, 0),
        refresh::Outcome::Refreshed => (Wire::OUTCOME_REFRESHED, 0),
        refresh::Outcome::Cancelled => (Wire::OUTCOME_CANCELLED, 0),
        refresh::Outcome::Faulted(error) => (Wire::OUTCOME_FAULTED, (error as u8).saturating_add(1)),
    };
    let summary = Wire {
        phase: match job.phase() {
            refresh::Phase::Idle => Wire::PHASE_IDLE,
            refresh::Phase::Scanning => Wire::PHASE_SCANNING,
            refresh::Phase::Refreshing => Wire::PHASE_REFRESHING,
            refresh::Phase::Draining => Wire::PHASE_DRAINING,
        },
        outcome,
        fault,
        flags: if scan.complete { Wire::FLAG_SCAN_COMPLETE } else { 0 },
        buckets: short(shape.buckets),
        entries: short(entries as u32),
        probes: short(refresh::PROBES as u32),
        stale_ratio: short(refresh::STALE_RATIO as u32),
        in_flight: short(job.in_flight()),
        passes: short(refresh::PASSES),
        bucket_bytes: U64::new(shape.bucket_bytes()),
        region_bytes: U64::new(shape.region_blocks * u64::from(shape.block_bytes)),
        chunk_bytes: U32::new(shape.chunk_bytes()),
        scans: U32::new(scan.scans),
        scanned: short(scan.scanned),
        mapped: short(scan.mapped),
        stale: short(scan.stale),
        planned: short(books.buckets_planned),
        done: short(books.buckets_done),
        skipped: short(books.buckets_skipped),
        lifetime: short(books.buckets_lifetime),
        pass: short(scan.pass),
        failed_probes: U32::new(scan.failed_probes),
        p25_ns: U32::new(scan.p25_ns),
        p50_ns: U32::new(scan.p50_ns),
        p90_ns: U32::new(scan.p90_ns),
        bytes: U64::new(books.bytes),
        zero_chunks: U32::new(books.zero_chunks),
        read_errors: U32::new(books.read_errors),
        unstable_chunks: U32::new(books.unstable_chunks),
        write_retries: U32::new(books.write_retries),
        mismatches: U32::new(books.mismatches),
    };
    let len = Wire::BYTES + entries * PackBucketWire::BYTES;
    let len = u32::try_from(len).map_err(|_| FsError::Range)?;
    let out = window.destination(window_offset, len)?;
    let (head, records) = out.split_at_mut(Wire::BYTES);
    head.copy_from_slice(summary.as_bytes());
    for (bucket, record) in buckets.iter().zip(records.chunks_exact_mut(PackBucketWire::BYTES)) {
        let flag = |set: bool, bit: u8| if set { bit } else { 0 };
        let wire = PackBucketWire {
            median_ns: U32::new(bucket.median_ns),
            mapped_probes: bucket.mapped_probes,
            failed_probes: bucket.failed_probes,
            flags: flag(bucket.stale, PackBucketWire::FLAG_STALE)
                | flag(bucket.refreshed, PackBucketWire::FLAG_REFRESHED)
                | flag(bucket.incomplete, PackBucketWire::FLAG_INCOMPLETE)
                | flag(bucket.mapped(), PackBucketWire::FLAG_MAPPED),
            reserved: 0,
        };
        record.copy_from_slice(wire.as_bytes());
    }
    Ok(len)
}

impl DurableService {
    /// The pack's maintenance job, claimed on the first ask. `Unsupported`
    /// without a pack region, without a running controller (a volatile
    /// posture retired every filesystem lane), or after a failed claim.
    fn pack_job(&mut self) -> Result<&mut PackJob, FsError> {
        if matches!(self.pack, PackMaintenance::Unclaimed) {
            let (Some(shared), Some(extent)) = (self.shared, self.pack_region) else {
                return Err(FsError::Unsupported);
            };
            if !(0..FS_LANES).any(|lane| shared.is_live(lane)) {
                return Err(FsError::Unsupported);
            }
            self.pack = claim_pack(shared, extent);
        }
        match &mut self.pack {
            PackMaintenance::Live(job) => Ok(job),
            _ => Err(FsError::Unsupported),
        }
    }

    /// Serve one pack maintenance ask (`res/PACK_STREAMING.md` §7): start
    /// or stop the job, give it its first pass, and land the report in the
    /// asking session's window.
    ///
    /// No authority beyond an attached session: the job only rewrites the
    /// pack's own bytes back unchanged, and at most once per bucket per
    /// boot — less endurance than any session with a writable home already
    /// spends.
    fn serve_maintain(
        &mut self,
        action: kore_dagfs::MaintainAction,
        window_offset: u32,
        via: u16,
    ) -> FsResponse {
        use kore_dagfs::MaintainAction;
        let Some(fs) = self.fs.as_ref() else {
            return FsResponse::Refused(FsError::Unsupported);
        };
        let Some(session) = fs.user.session_for_via(via) else {
            return FsResponse::Refused(FsError::AccessDenied);
        };
        let Some(span) = fs.user.mapped_window_span(session) else {
            return FsResponse::Refused(FsError::NoWindow);
        };
        let job = match self.pack_job() {
            Ok(job) => job,
            Err(error) => return FsResponse::Refused(error),
        };
        let started = match action {
            MaintainAction::Status => Ok(()),
            MaintainAction::Scan => job.scan(),
            MaintainAction::RefreshStale => job.refresh_stale(),
            MaintainAction::RefreshRange { first, count } => job.refresh_range(first, count),
            MaintainAction::Cancel => {
                job.cancel();
                Ok(())
            }
        };
        if let Err(refusal) = started {
            return FsResponse::Refused(match refusal {
                refresh::Refusal::Busy => FsError::Busy,
                refresh::Refusal::Range => FsError::Range,
                refresh::Refusal::NoClock => FsError::Unsupported,
                // Nothing to refresh by until a scan has judged the
                // buckets: the health index is not built yet.
                refresh::Refusal::NotScanned => FsError::IndexIncomplete,
            });
        }
        self.advance_stor();
        let PackMaintenance::Live(job) = &self.pack else {
            return FsResponse::Refused(FsError::Unsupported);
        };
        let mut window = MountedFs::window_for(span);
        match write_pack_report(job, &mut window, window_offset) {
            Ok(len) => FsResponse::StorMaintained { len },
            Err(error) => FsResponse::Refused(error),
        }
    }

    /// One engine pass for the client rings on every wake of the row — each
    /// request, and the controller's interrupt through
    /// [`SystemService::poll_background`]. When the engine goes idle it
    /// parks every ring, so a client that publishes afterwards owes a
    /// `StorKick`; while commands are in flight their completion is the
    /// wake, and a client's publish costs nothing.
    fn advance_stor(&mut self) {
        let Some(shared) = self.shared else {
            return;
        };
        let maintaining = matches!(&self.pack, PackMaintenance::Live(job) if job.phase() != refresh::Phase::Idle);
        if maintaining || self.rings.iter().any(Option::is_some) {
            let _ = shared.pump();
        }
        if maintaining && let PackMaintenance::Live(job) = &mut self.pack {
            // Probes time one read alone on the device: only when nothing
            // else is on it. Every other step rides the Stream budget.
            let idle = shared.commands_in_flight() == Some(0);
            job.advance(idle);
        }
        // What earlier windows' retirements owe settles the whole-latch way
        // on the first pass that finds the controller idle.
        if shared
            .inspect(|queues| queues.retirement_owed())
            .unwrap_or(false)
        {
            let _ = shared.control_idle(|queues| queues.retry_retirement());
        }
    }

    /// Before a version lease is released, revoke the ring grant it pins.
    /// `false` while a read under the grant is still in flight — the
    /// release is refused `Busy` and the client asks again; the lease (and
    /// with it the blocks) outlives every command that reads them.
    fn settle_version_grant(&mut self, via: u16, lease: kore_dagfs::VersionLease) -> bool {
        let (Some(shared), Some(fs)) = (self.shared, self.fs.as_ref()) else {
            return true;
        };
        let Some(session) = fs.user.session_for_via(via) else {
            return true;
        };
        let wire = lease.to_bits();
        for entry in self.rings.iter_mut().flatten() {
            if entry.session != session {
                continue;
            }
            let ring = entry.ring;
            for slot in entry.versions.iter_mut() {
                let Some((bits, grant)) = *slot else { continue };
                if bits != wire {
                    continue;
                }
                match shared.revoke(ring, grant) {
                    Ok(kore_driver::stor::engine::Settle::Later) => return false,
                    // Revoked now, or already drained and freed (a stale id).
                    Ok(_) | Err(_) => *slot = None,
                }
            }
        }
        true
    }

    /// Serve one storage-ring verb (`res/STORAGE_TRANSPORT.md` §9, S4).
    fn serve_stor(&mut self, request: FsRequest, sv: &mut Serve<'_, FsResponse>) -> FsResponse {
        // A capability lent to a refused request is dropped — revoked back
        // to the client's parent, never kept.
        let lent = match request {
            FsRequest::StorAttach { announced } | FsRequest::StorTarget { announced } => sv
                .returned_input(announced)
                .and_then(|mut input| input.take()),
            _ => None,
        };
        let (Some(shared), Some(fs)) = (self.shared, self.fs.as_ref()) else {
            return FsResponse::Refused(FsError::Unsupported);
        };
        let Some(session) = fs.user.session_for_via(sv.via()) else {
            return FsResponse::Refused(FsError::AccessDenied);
        };
        let index = self
            .rings
            .iter()
            .position(|ring| ring.as_ref().is_some_and(|ring| ring.session == session));
        let ring = index.and_then(|index| self.rings[index].as_ref().map(|ring| ring.ring));
        match request {
            FsRequest::StorAttach { .. } => {
                let Some(owner) = lent else {
                    return FsResponse::Refused(FsError::NoWindow);
                };
                if index.is_some() {
                    return FsResponse::Refused(FsError::AlreadyExists);
                }
                let Some(free) = self.rings.iter().position(Option::is_none) else {
                    return FsResponse::Refused(FsError::OutOfHandles);
                };
                match attach_ring(shared, owner) {
                    Ok((ring, owner)) => {
                        self.rings[free] = Some(ClientRing {
                            session,
                            ring,
                            _window: owner,
                            targets: 0,
                            versions: [None; VERSION_GRANTS],
                        });
                        let geometry = shared.geometry();
                        let max_transfer = u64::from(geometry.max_transfer_blocks)
                            * u64::from(geometry.block_size);
                        FsResponse::StorAttached {
                            block_size: geometry.block_size,
                            max_transfer: u32::try_from(max_transfer).unwrap_or(u32::MAX),
                        }
                    }
                    Err(error) => FsResponse::Refused(error),
                }
            }
            FsRequest::StorGrantPack => {
                let (Some(ring), Some(extent)) = (ring, self.pack_region) else {
                    return FsResponse::Refused(if ring.is_none() {
                        FsError::NoWindow
                    } else {
                        FsError::Unsupported
                    });
                };
                let bytes = extent.blocks() * u64::from(shared.geometry().block_size);
                match pack_grant(extent).and_then(|grant| {
                    shared.install(ring, grant).map_err(|_| FsError::OutOfHandles)
                }) {
                    Ok(grant) => FsResponse::StorGranted {
                        grant: grant.fd() as u32,
                        bytes,
                    },
                    Err(error) => FsResponse::Refused(error),
                }
            }
            FsRequest::StorGrantVersion { lease } => {
                let (Some(index), Some(ring)) = (index, ring) else {
                    return FsResponse::Refused(FsError::NoWindow);
                };
                let Some(free) = self.rings[index]
                    .as_ref()
                    .and_then(|entry| entry.versions.iter().position(Option::is_none))
                else {
                    return FsResponse::Refused(FsError::OutOfHandles);
                };
                let wire = lease.to_bits();
                if self.rings[index]
                    .as_ref()
                    .is_some_and(|entry| entry.versions.iter().flatten().any(|(bits, _)| *bits == wire))
                {
                    return FsResponse::Refused(FsError::AlreadyExists);
                }
                let (store, lease) = crate::router::split_lease(lease);
                let engine = match store {
                    crate::router::Store::User => Some(&fs.user),
                    crate::router::Store::System => fs.system.as_ref(),
                };
                let blocks = match engine {
                    Some(MountedEngine::Durable { fs: engine, .. }) => engine.lent_blocks(session, lease),
                    _ => Err(FsError::Unsupported),
                };
                let blocks = match blocks {
                    Ok(Some(blocks)) if blocks.blocks != 0 => blocks,
                    // Nothing whole to grant: the version reads through its
                    // lease alone.
                    Ok(_) => return FsResponse::Refused(FsError::Range),
                    Err(error) => return FsResponse::Refused(error),
                };
                let bytes = blocks.blocks * u64::from(blocks.block_size);
                match version_grant(blocks).and_then(|grant| {
                    shared.install(ring, grant).map_err(|_| FsError::OutOfHandles)
                }) {
                    Ok(grant) => {
                        if let Some(entry) = self.rings[index].as_mut() {
                            entry.versions[free] = Some((wire, grant));
                        }
                        FsResponse::StorGranted {
                            grant: grant.fd() as u32,
                            bytes,
                        }
                    }
                    Err(error) => FsResponse::Refused(error),
                }
            }
            FsRequest::StorTarget { .. } => {
                let (Some(index), Some(ring), Some(owner)) = (index, ring, lent) else {
                    return FsResponse::Refused(FsError::NoWindow);
                };
                match attach_ring_target(shared, ring, owner) {
                    Ok((slot, bytes)) => {
                        if let Some(entry) = self.rings[index].as_mut() {
                            entry.targets |= 1u64 << slot;
                        }
                        FsResponse::StorTargetAdopted { slot, bytes }
                    }
                    Err(error) => FsResponse::Refused(error),
                }
            }
            FsRequest::StorRelease { slot } => {
                let (Some(index), Some(ring)) = (index, ring) else {
                    return FsResponse::Refused(FsError::NoWindow);
                };
                match shared.release_client(ring, slot) {
                    Ok(target) => {
                        if let Some(entry) = self.rings[index].as_mut() {
                            entry.targets &= !(1u64 << slot);
                        }
                        // `Done` promises the lender its route has no device
                        // child left here, so it may retire it. A mapping
                        // that outlives the release answers `Busy`: the
                        // slot is gone, and the lender must hold the route.
                        if retire_target(shared, target) {
                            FsResponse::Done
                        } else {
                            FsResponse::Refused(FsError::Busy)
                        }
                    }
                    Err(BlockError::BufferBusy) => FsResponse::Refused(FsError::Busy),
                    Err(_) => FsResponse::Refused(FsError::InvalidHandle),
                }
            }
            FsRequest::StorInspect => {
                let narrow = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
                match shared.stats() {
                    Some((stats, _)) => FsResponse::StorInspected {
                        submissions: stats.submissions,
                        completions: stats.completions,
                        in_flight: narrow(shared.commands_in_flight().unwrap_or(0) as u64),
                        expired: narrow(stats.expired),
                        stale: narrow(stats.stale_completions),
                        timeline_refusals: narrow(stats.timeline_refusals),
                        stage_refusals: narrow(stats.stage_refusals),
                        last_stage_error: stats
                            .last_stage_error
                            .map_or(0, |error| (error as u8).saturating_add(1)),
                        aborts: narrow(stats.aborts),
                        aborts_refused: u8::try_from(stats.aborts_refused).unwrap_or(u8::MAX),
                        weighted: shared
                            .inspect(|queues| queues.backend().weighted_arbitration())
                            .unwrap_or(false),
                        rings: u8::try_from(
                            FS_LANES
                                + usize::from(matches!(self.pack, PackMaintenance::Live(_)))
                                + self.rings.iter().flatten().count(),
                        )
                        .unwrap_or(u8::MAX),
                    },
                    None => FsResponse::Refused(FsError::Busy),
                }
            }
            FsRequest::StorKick => {
                // The wake itself: `serve_async` pumped before dispatching,
                // and the reply leaves the ring advanced.
                FsResponse::Done
            }
            FsRequest::StorDetach => {
                let Some(index) = index else {
                    return FsResponse::Done;
                };
                if retire_ring(shared, &mut self.rings[index]) {
                    FsResponse::Done
                } else {
                    FsResponse::Refused(FsError::Busy)
                }
            }
            _ => FsResponse::Refused(FsError::Unsupported),
        }
    }
}

impl SystemService for DurableService {
    const NAME: &'static str = "fs";
    /// An I/O worker: parked on completions for most of its life, its work
    /// done in microseconds between parks.  Keep it off the performance
    /// cores; a completion wait cannot tell the difference.
    const CORE_PREFERENCE: kore_abi::task::CorePreference =
        kore_abi::task::CorePreference::Efficient;
    type Endowment = FsEndowment;
    const RETURN_MODE: ReturnMode = ReturnMode::RetainedAny;

    /// See [`crate::FS_DEVICE_CLASSES`] — declared once, cited by both impls.
    const DEVICE_CLASSES: &'static [kore_rt::service::DeviceClass] = crate::FS_DEVICE_CLASSES;

    /// The terminal sequence (#809 §L, L.8 item 2): the row holds two lanes
    /// and the client rings over one controller, and the controller's own
    /// quiesce (`CC.SHN`) runs only when the **last** live lane retires — so
    /// this is an ordered sequence, not one call. The client rings go first,
    /// releasing the client memory mapped into the controller while the
    /// filesystem lanes still keep it running.  Retire the user and system engines' providers
    /// (unregister their DMA buffers, quiesce their channels; the last one
    /// takes the controller through orderly shutdown), then reach the binding
    /// through the shared backend for the host's class → fabric →
    /// containment teardown.  A lane that could not drain leaves `partial`
    /// set: that is a different state from skipped, and the containment
    /// sweep after the drain knows which.
    async fn quiesce_devices(&mut self, _deadline_ns: u64) -> kore_rt::service::QuiesceSweep {
        let mut sweep = kore_rt::service::QuiesceSweep::NONE;
        let Some(shared) = self.shared.take() else {
            // Volatile posture: nothing was ever bound.
            return sweep;
        };
        // Client rings first: their targets are client memory mapped into
        // this controller. A ring still draining leaves the sweep partial.
        for slot in self.rings.iter_mut() {
            if !retire_ring(shared, slot) {
                sweep.partial = true;
            }
        }
        // Then the pack maintenance lane, while the filesystem lanes still
        // keep the controller running: its job stops issuing, the device
        // returns what it holds, and the lane retires without being the
        // last one.
        if !retire_pack(core::mem::replace(&mut self.pack, PackMaintenance::Unavailable)) {
            sweep.partial = true;
        }
        if let Some(mounted) = self.fs.take() {
            let MountedFs { user, system, .. } = mounted;
            for engine in [Some(user), system].into_iter().flatten() {
                if let MountedEngine::Durable { fs, .. } = engine
                    && (*fs).into_engine().into_provider().retire().await.is_err()
                {
                    sweep.partial = true;
                }
            }
        }
        match shared.control_idle(|queues| {
            let direct = queues;
            direct.retry_retirement();
            direct.teardown()
        }) {
            Ok(Some(outcome)) => outcome.record(&mut sweep),
            // Already consumed — a second ask after the first completed.
            Ok(None) => {}
            // A lane still has work outstanding: the controller cannot be
            // stopped under it.  Partial, and the kernel-side containment
            // after the drain owns the function.
            Err(_) => sweep.partial = true,
        }
        sweep
    }

    /// The faces grow with the programs that read files, as the RAM
    /// composition's do ([`RamService`]): both engines'
    /// sessions are allocated per slot from the same heap.
    fn peer_arena(&self) -> Option<&'static (dyn core::alloc::Allocator + Sync)> {
        Some(crate::service_alloc())
    }

    fn bind(endowment: FsEndowment) -> Self {
        Self {
            fs: None,
            shared: None,
            peer_map: None,
            archive_era: endowment.archive_era(),
            route_held: false,
            note: 0,
            pack_region: None,
            pack: PackMaintenance::Unclaimed,
            rings: core::array::from_fn(|_| None),
        }
    }

    fn start_on_metal(&mut self, rt: &kore_rt::entry::Rt) -> Result<(), StartFailure> {
        let map = crate::service::contract::peer_map(rt).map_err(|_| StartFailure::new(21))?;
        self.peer_map = Some(map);
        if let Some(fs) = self.fs.as_mut() {
            fs.configure_peer_map(map);
        }
        Ok(())
    }

    fn serve(&mut self, request: FsRequest, sv: &mut Serve<'_, FsResponse>) {
        let Some(mounted) = self.fs.as_mut() else {
            sv.reply(FsResponse::Refused(FsError::ProviderFailure(
                ProviderError::Lost,
            )));
            return;
        };
        let response = if mounted.durable() {
            // A durable request must retain the executor waker across its
            // dependent block-service transaction. The metal host calls
            // `serve_async`; fail closed if a synchronous harness attempts to
            // bypass that progress seam.
            FsResponse::Refused(FsError::ProviderFailure(ProviderError::Lost))
        } else {
            crate::block_on(mounted.serve_request_async(request, sv))
        };
        sv.reply(response);
    }

    async fn serve_async(&mut self, request: FsRequest, sv: &mut Serve<'_, FsResponse>) {
        // Advance the client rings without waiting on the controller;
        // ordinary filesystem traffic remains dispatchable while they run.
        self.advance_stor();
        if let FsRequest::StorMaintain {
            action,
            window_offset,
        } = request
        {
            let response = self.serve_maintain(action, window_offset, sv.via());
            sv.reply(response);
            return;
        }
        if matches!(
            request,
            FsRequest::StorAttach { .. }
                | FsRequest::StorGrantPack
                | FsRequest::StorGrantVersion { .. }
                | FsRequest::StorTarget { .. }
                | FsRequest::StorRelease { .. }
                | FsRequest::StorKick
                | FsRequest::StorInspect
                | FsRequest::StorDetach
        ) {
            let response = self.serve_stor(request, sv);
            // A detach or release may have freed work the next pass parks.
            self.advance_stor();
            sv.reply(response);
            return;
        }
        // A version whose blocks a ring grant names outlives every read
        // under that grant: the release waits for the grant to drain.
        if let FsRequest::ReleaseVersion { lease } = request
            && !self.settle_version_grant(sv.via(), lease)
        {
            sv.reply(FsResponse::Refused(FsError::Busy));
            return;
        }
        // A status answer must read as one moment: fold the mount's current
        // posture and counters into the frontend immediately before the
        // request is served, and only then — every other verb pays nothing.
        if matches!(request, FsRequest::SystemStatus { .. }) {
            self.refresh_backend_status();
        }
        let Some(mounted) = self.fs.as_mut() else {
            sv.reply(FsResponse::Refused(FsError::ProviderFailure(
                ProviderError::Lost,
            )));
            return;
        };
        let response = mounted.serve_request_async(request, sv).await;
        sv.reply(response);
    }

    fn peer_admitted(&mut self, result: Result<PeerId, PeerAdmissionRefusal>) -> bool {
        match self.fs.as_mut() {
            Some(fs) => fs.peer_admitted(result),
            None => result.is_err(),
        }
    }

    fn poll_background(&mut self) {
        self.advance_stor();
    }

    fn peer_closed(&mut self, via: u16, reason: PeerClose) -> bool {
        // A departed session's ring goes first: its targets are client
        // memory mapped into the controller, and a command still in flight
        // owns its destination, so the peer stays closing until the ring
        // drained. The host retries this hook after later wakes, each of
        // which pumps the engine.
        if let (Some(fs), Some(shared)) = (self.fs.as_ref(), self.shared)
            && let Some(session) = fs.user.session_for_via(via)
            && let Some(slot) = self
                .rings
                .iter_mut()
                .find(|ring| ring.as_ref().is_some_and(|ring| ring.session == session))
            && !retire_ring(shared, slot)
        {
            return false;
        }
        if let Some(fs) = self.fs.as_mut() {
            fs.peer_closed(via, reason)
        } else {
            true
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "none"))]
async fn open_block_channel(
    transport: &kore_rt::transport::TrapTransport,
    boot: &kore_abi::task::Bootstrap<'_>,
) -> Result<kore_svc_block::OpenedChannel, StartFailure> {
    use kore_abi::task::role;
    use kore_rt::service::MAX_GRANTS;

    let kick = boot
        .handle(role::SVC_SEND, BLOCK_CLIENT_INDEX)
        .ok_or(StartFailure::new(1))?;
    let ring = boot
        .handle(role::SVC_RING, BLOCK_CLIENT_INDEX)
        .ok_or(StartFailure::new(2))?;
    let leased = kore_rt::cap::verbs::take_view(ring).ok_or(StartFailure::new(3))?;
    let window = kore_rt::cap::verbs::map_view(leased).ok_or(StartFailure::new(4))?;
    // SAFETY: this task is the sole client-side attacher; block owns the
    // serving half and both use only the typed SPSC contract.
    let window: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
    let rings = client_rings::<BlockService>(
        window,
        BlockService::REQUEST_CAPACITY,
        BlockService::RESPONSE_CAPACITY,
    )
    .map_err(|_| StartFailure::new(5))?;
    let mut grants = [const { None }; MAX_GRANTS];
    grants[0] = Some(GrantPin::retained(
        boot.handle(role::SVC_GRANT, 0)
            .ok_or(StartFailure::new(6))?,
    ));
    grants[1] = Some(GrantPin::retained(
        boot.handle(role::SVC_GRANT, 1)
            .ok_or(StartFailure::new(7))?,
    ));
    let mut client = RingBlock(SvcClient::new(
        transport,
        rings,
        ClientEndpoints { kick, grants },
    ));
    client
        .open(ProvisionRequest {
            writable: true,
            // The declared intent; whether the platform realizes the
            // optional route is discovered at claim time, and a route-less
            // machine serves the identical channel at the polled posture.
            mode: CompletionMode::Interrupt,
            requested_queue_depth: IO_QDEPTH,
            requested_buffer_bytes: FS_DMA_POOL_BYTES,
        })
        .await
        .map_err(|refusal| {
            StartFailure::new(match refusal {
                BlockRefusal::Unavailable => 9,
                BlockRefusal::Policy => 10,
                BlockRefusal::Exhausted => 11,
                BlockRefusal::Lost => 12,
            })
        })
}

/// Run block acquisition and both filesystem serving pairs on the executor
/// owned by `#[kore_rt::main]`.  The task has one doorbell; startup and serving
/// are phases of one root future, not independently driven runtimes.
///
/// # Bring-up detail map
///
/// A founder report of `rung 4` is the runtime's service-start rung; its detail
/// decodes here. Details 1–7 cover the block client endpoints/rings/grants, 8
/// is retired, 9–12 are block provisioning refusals (unavailable, policy,
/// exhausted, lost), 13 is direct NVMe claim, 14 whole-view/channel
/// construction, 15 provider construction, 16 whole-disk media probe, 17
/// kore-volume recovery (its higher bytes, when present, are the
/// recovery witness: stage, error kind, cause — `RecoverFailure::witness`,
/// so `0x__1104` reads stage, kind and cause from the digits above `11`),
/// 18 blank formatting, 19 administration-root
/// publication, 20 foreign-media channel containment, 21 recyclable-peer
/// mapping geometry, 22 user-region channel rebuild, 23 the probe inside the
/// user region, 24 durable-workspace publication, 25 application-store
/// publication, 26 system-region channel construction, 27 the probe inside
/// the system region, 28 system-lane containment or missing-era retirement,
/// 29 archive-era root admission, 30 bounded system reformat, and 31
/// shared-data (`share/`) publication. (`establish_boot_policy` also reports
/// 26 and 27 for library and package publication, colliding with the
/// system-region rungs above; 31 is appended fresh so the data directory
/// does not add a third collision.)
/// Renumber nothing; append new details so a photographed metal failure
/// remains decodable.
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
pub async fn run(rt: kore_rt::entry::Rt) -> ! {
    use kore_rt::entry::{
        RUNG_SERVICE_ENDOWMENT, RUNG_SERVICE_SERVE_SETUP, RUNG_SERVICE_START, refuse_with,
    };
    use kore_rt::service::{Bundle, SystemService};

    let Ok(endowment) = FsEndowment::resolve(rt.bootstrap()) else {
        refuse_with(&rt, RUNG_SERVICE_ENDOWMENT);
    };
    let mut service = DurableService::bind(endowment);
    if let Err(failure) = service.start_on_metal(&rt) {
        refuse_with(&rt, failure.payload(RUNG_SERVICE_START));
    }
    let transport = kore_rt::entry::transport();
    let opened = match open_block_channel(transport, rt.bootstrap()).await {
        Ok(opened) => opened,
        Err(failure) => refuse_with(&rt, failure.payload(RUNG_SERVICE_START)),
    };
    if let Err(failure) = service.initialize(opened).await {
        refuse_with(&rt, failure.payload(RUNG_SERVICE_START));
    }
    // The steady park arms the held MSI-X route beside the doorbell (the
    // kernel's combination is lawful for a route-holding task and refused
    // otherwise) — which is also what earns the boot gate's
    // `-remapped-msi-irq` witness at the beacon.  Enabled only once the
    // mounted composition provably retains the route: the RAM fallback
    // retired its channel, and its dead route must never join the parks.
    if service.holds_route() {
        kore_rt::executor::enable_irq_parks().await;
    }
    // `serve_forever` owns the complete bounded filesystem engine. Pin its
    // long-lived future in this task's named heap supply so the outer root
    // future carries only the box pointer after startup; inlining the full
    // serve state beside initialization exceeds the fixed service stack.
    let note = service.readiness_note();
    let serving = Box::pin(kore_rt::entry::serve_forever(&rt, service, None, note));
    serving.await;
    refuse_with(&rt, RUNG_SERVICE_SERVE_SETUP)
}

impl SvcProtocol for DurableService {
    type Request = <crate::FsProtocol as SvcProtocol>::Request;
    type Response = <crate::FsProtocol as SvcProtocol>::Response;
    type ReqWire = <crate::FsProtocol as SvcProtocol>::ReqWire;
    type RepWire = <crate::FsProtocol as SvcProtocol>::RepWire;

    const REPLY_GRANTS: usize = <crate::FsProtocol as SvcProtocol>::REPLY_GRANTS;
    const SERVE_PAIRS: usize = <crate::FsProtocol as SvcProtocol>::SERVE_PAIRS;

    fn reply_matches(request: &Self::ReqWire, response: &FsResponse, grants: u8) -> bool {
        <crate::FsProtocol as SvcProtocol>::reply_matches(request, response, grants)
    }

    fn encode_req(request: &FsRequest) -> Self::ReqWire {
        <crate::FsProtocol as SvcProtocol>::encode_req(request)
    }
    fn decode_req(wire: &Self::ReqWire) -> Option<FsRequest> {
        <crate::FsProtocol as SvcProtocol>::decode_req(wire)
    }
    fn encode_rep(response: &FsResponse, grants: u8) -> Self::RepWire {
        <crate::FsProtocol as SvcProtocol>::encode_rep(response, grants)
    }
    fn decode_rep(wire: &Self::RepWire) -> Option<(FsResponse, u8)> {
        <crate::FsProtocol as SvcProtocol>::decode_rep(wire)
    }
    fn decode_rep_for(request: &Self::ReqWire, reply: &Self::RepWire) -> Option<(FsResponse, u8)> {
        <crate::FsProtocol as SvcProtocol>::decode_rep_for(request, reply)
    }
}

// The boot manifest cites `FsProtocol`; the durable entry must remain the same
// service face even though its provider startup is different.
const _: () = {
    // `run` constructs this value on the founded service stack before moving
    // it into the named heap. Keep the provider-neutral fallback bounded well
    // below the 4 MiB founder stack rather than letting a future engine-growth
    // change turn foreign-media containment into a stack fault.
    assert!(core::mem::size_of::<DurableService>() < 512 * 1024);
    assert!(
        <DurableService as SvcProtocol>::REPLY_GRANTS
            == <crate::FsProtocol as SvcProtocol>::REPLY_GRANTS
    );
    assert!(
        <DurableService as SvcProtocol>::SERVE_PAIRS
            == <crate::FsProtocol as SvcProtocol>::SERVE_PAIRS
    );
};

#[cfg(test)]
mod retirement_tests {
    use super::{ProviderError, RetirementBooks};

    /// A window still waiting to retire refuses no one; only a failed
    /// requester-wide retirement does, and only until an idle pass succeeds.
    #[test]
    fn only_a_failed_requester_retirement_refuses_new_windows() {
        let mut books = RetirementBooks::default();
        assert!(!books.owed());
        // A live retirement deferred: owed, but admission stays open.
        books.deferred = 1;
        assert!(books.owed());
        assert_eq!(books.admission(), Ok(()));
        // The live retry still refused: nothing failed.
        books.settle_live(true, 1);
        assert_eq!(books.admission(), Ok(()));
        // The idle pass fails: the latch is unproven, windows are refused.
        books.settle_idle(false, 1);
        assert_eq!(books.admission(), Err(ProviderError::Retiring));
        // A live pass cannot prove the latch again, even when it retires.
        books.settle_live(true, 0);
        assert_eq!(books.admission(), Err(ProviderError::Retiring));
        assert!(books.owed(), "the failure still needs an idle pass");
        // The next successful idle pass clears it.
        books.settle_idle(true, 0);
        assert_eq!(books.admission(), Ok(()));
        assert!(!books.owed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::VolumeProbe;
    use kore_driver::stor::Single;
    use kore_driver::stor::engine::{OneQueue, StorEngine};
    use crate::{Name, ObjectKind, persistence::PersistentFs};
    use kore_driver::block::memory::MemoryBlockDevice;
    use kore_driver::gpt::{KORE_FS_TYPE, KORE_SYSTEM_TYPE, synthetic};

    // A 640-block (512-byte) disk; the GPT designation is large enough for
    // the current dual 128 KiB metadata reservation plus a payload arena,
    // without making the inline cold-boot fixture dominate the test stack.
    type Device = MemoryBlockDevice<327680, 8192>;
    type TestProvider = BlockProvider<RegionDevice<Device>, 3, 8>;
    type Volume = PersistentFs<RegionDevice<Device>, 3, 8>;

    // Slice 4's two-region witness needs a larger disk, but keeping that
    // by-value backend out of every pre-existing fixture is load-bearing in
    // unoptimized tests: async state retains several device-shaped variants.
    type DualDevice = MemoryBlockDevice<786432, 8192>;

    /// The transport engine over one memory disk, `L` lanes — production's
    /// composition (`initialize`) with the device a test can take back.
    type TestQueues<D> = Single<D, 32>;
    type TestEngine<D, const L: usize> =
        StorEngine<'static, TestQueues<D>, OneQueue, L, 1, 16, 64, 32>;
    type TestBackend<D, const L: usize> =
        EngineBackend<'static, TestQueues<D>, OneQueue, L, L, 1, 16, 64, 32>;

    /// `device` behind a fresh engine and `L` filesystem lanes; the engine
    /// is placed in its heap slot as `initialize` places it.
    fn engine_backend<D: DirectBlockDevice, const L: usize>(device: D) -> TestBackend<D, L>
    where
        D::Buffer: kore_driver::block::BlockBuffer,
    {
        let place = Box::leak(Box::<TestEngine<D, L>>::new_uninit());
        let engine = TestEngine::<D, L>::init_in(place, Single::new(device))
            .ok()
            .expect("the one-queue policy fits");
        let windows = core::array::from_fn(|_| lane_window().expect("a lane window"));
        EngineBackend::new(engine, windows, [LaneClasses::FILESYSTEM; L])
            .ok()
            .expect("a fresh engine takes its lanes")
    }

    /// Take the disk back from a backend whose lanes are all idle — the
    /// "power off" between two boots of one disk.
    fn engine_device<D: DirectBlockDevice, const L: usize>(backend: TestBackend<D, L>) -> D
    where
        D::Buffer: kore_driver::block::BlockBuffer,
    {
        let engine = backend.into_engine().ok().expect("every lane idle");
        // SAFETY: the engine lives in a leaked slot nothing else reaches;
        // reading it out moves it, and the slot is never read or dropped
        // again.
        let engine = unsafe { core::ptr::read(engine) };
        engine
            .into_queues()
            .ok()
            .expect("no ring left attached")
            .into_device()
            .ok()
            .expect("nothing outstanding")
    }
    type DualProvider = BlockProvider<RegionDevice<DualDevice>, 3, 8>;
    type SystemProvider = BlockProvider<RegionDevice<DualDevice>, 8, 32>;
    type SystemVolume = PersistentFs<RegionDevice<DualDevice>, 8, 32>;

    const PART_START: u64 = 80;
    const PART_END: u64 = 630;
    const SYSTEM_PART_START: u64 = 700;
    const SYSTEM_PART_END: u64 = 1500;
    const TEST_ERA: [u8; crate::persistence::ERA_BYTES] = [0x6a; crate::persistence::ERA_BYTES];

    #[test]
    fn fs_nvme_page_ceiling_matches_the_bounded_dma_budget() {
        assert!(fs_nvme_page_supported(MemPage::from_mps(6)));
        assert!(!fs_nvme_page_supported(MemPage::from_mps(7)));
    }

    // DetachEdge mutates the persisted namespace exactly like Unlink (it can
    // even reclaim a populated subtree Unlink refuses), so a bare detach must
    // reach commit_after or a crash resurrects the detached edge on reboot.
    #[test]
    fn detach_edge_is_a_committing_verb() {
        let request = FsRequest::DetachEdge {
            parent: crate::authority::Handle::from_bits(0),
            name: Name::new("tool").unwrap(),
        };
        assert!(DurableService::needs_commit(&request, &FsResponse::Done));
        assert!(!DurableService::needs_commit(
            &request,
            &FsResponse::Refused(FsError::NotFound)
        ));
    }

    #[test]
    fn nvme_bring_up_budget_honors_cap_to_but_cannot_hold_the_shell() {
        let missing = kore_nvme::Capabilities::from_cap(0);
        let half_second = kore_nvme::Capabilities::from_cap(1u64 << 24);
        let fifteen_seconds = kore_nvme::Capabilities::from_cap(30u64 << 24);

        assert_eq!(bring_up_budget_ns(missing), BRING_UP_BUDGET_NS);
        assert_eq!(bring_up_budget_ns(half_second), 500_000_000);
        assert_eq!(bring_up_budget_ns(fifteen_seconds), BRING_UP_BUDGET_NS);
    }

    fn whole_provider(device: Device) -> TestProvider {
        DurableService::whole_medium_provider(device, CompletionMode::Poll).unwrap()
    }

    /// Take the raw backend out of a provider for a cold-boot simulation.
    fn eject(provider: TestProvider) -> Device {
        provider
            .into_channel()
            .unwrap()
            .into_device()
            .map_err(|(_, error)| error)
            .unwrap()
            .into_inner()
    }

    /// A disk that is foreign as a whole — protective MBR at LBA 0, a GPT
    /// at LBA 1 — and designates `[PART_START, PART_END]` for kore.
    fn designated_disk() -> Device {
        let mut mbr = [0u8; 512];
        mbr[510] = 0x55;
        mbr[511] = 0xaa;
        let (header, entries) = synthetic::table(KORE_FS_TYPE, PART_START, PART_END, PART_END);
        let provider = whole_provider(Device::new(512, 16).unwrap());
        crate::block_on(provider.write_metadata(0, &mbr)).unwrap();
        crate::block_on(provider.write_metadata(1, &header)).unwrap();
        crate::block_on(provider.write_metadata(2, &entries)).unwrap();
        eject(provider)
    }

    fn dual_whole_provider(device: DualDevice) -> DualProvider {
        DurableService::whole_medium_provider(device, CompletionMode::Poll).unwrap()
    }

    fn dual_eject(provider: DualProvider) -> DualDevice {
        provider
            .into_channel()
            .unwrap()
            .into_device()
            .map_err(|(_, error)| error)
            .unwrap()
            .into_inner()
    }

    fn system_eject(provider: SystemProvider) -> DualDevice {
        provider
            .into_channel()
            .unwrap()
            .into_device()
            .map_err(|(_, error)| error)
            .unwrap()
            .into_inner()
    }

    fn format_system_era(
        device: DualDevice,
        extent: gpt::Extent,
        era: [u8; crate::persistence::ERA_BYTES],
    ) -> DualDevice {
        let session = SessionId::new(0);
        let Some(Located::Format {
            provider,
            era: selected,
        }) = crate::block_on(DurableService::locate_system(
            device,
            extent,
            CompletionMode::Poll,
            era,
        ))
        .unwrap()
        else {
            panic!("a blank system extent must format");
        };
        assert_eq!(selected, era);
        let mut volume =
            crate::block_on(SystemVolume::format_with_era(provider, session, selected)).unwrap();
        let root = volume
            .engine_mut()
            .open_root(session, crate::Rights::ALL)
            .unwrap();
        crate::block_on(volume.engine_mut().create(
            session,
            root.handle,
            &name("stale-system-name"),
            ObjectKind::File,
        ))
        .unwrap();
        crate::block_on(volume.commit(session)).unwrap();
        let (engine, _) = volume.into_parts();
        system_eject(engine.into_provider())
    }

    fn write_system_sentinels(device: DualDevice) -> DualDevice {
        let mut channel: BlockChannel<DualDevice, 2, 4> =
            BlockChannel::new(device, CompletionMode::Poll).unwrap();
        let buffer = channel.register_buffer(512).unwrap();
        for (lba, byte) in [(SYSTEM_PART_START - 1, 0xa5), (SYSTEM_PART_END + 1, 0x5a)] {
            channel.fill_buffer(buffer, byte).unwrap();
            let request = channel.submit_write(lba, 1, buffer, 0).unwrap();
            assert_eq!(channel.wait().unwrap().request, request);
        }
        channel.unregister_buffer(buffer).unwrap();
        channel.into_device().map_err(|(_, error)| error).unwrap()
    }

    fn admit_matching_system_era(
        device: DualDevice,
        extent: gpt::Extent,
        era: [u8; crate::persistence::ERA_BYTES],
    ) -> DualDevice {
        let Some(Located::Recover(provider)) = crate::block_on(DurableService::locate_system(
            device,
            extent,
            CompletionMode::Poll,
            era,
        ))
        .unwrap() else {
            panic!("matching v3 roots must recover without reseeding policy");
        };
        system_eject(provider)
    }

    fn reformat_system_era(
        device: DualDevice,
        extent: gpt::Extent,
        era: [u8; crate::persistence::ERA_BYTES],
    ) -> DualDevice {
        let Some(Located::Reformat {
            provider,
            era: selected,
        }) = crate::block_on(DurableService::locate_system(
            device,
            extent,
            CompletionMode::Poll,
            era,
        ))
        .unwrap()
        else {
            panic!("an old system era must request a bounded reformat");
        };
        assert_eq!(selected, era);
        let volume = crate::block_on(SystemVolume::reformat_with_era(
            provider,
            SessionId::new(0),
            selected,
        ))
        .unwrap();
        assert_eq!(volume.status().era, era);
        let (engine, _) = volume.into_parts();
        system_eject(engine.into_provider())
    }

    fn assert_system_sentinels(device: DualDevice) {
        let mut channel: BlockChannel<DualDevice, 2, 4> =
            BlockChannel::new(device, CompletionMode::Poll).unwrap();
        let buffer = channel.register_buffer(512).unwrap();
        let mut observed = [0u8; 512];
        for (lba, byte) in [(SYSTEM_PART_START - 1, 0xa5), (SYSTEM_PART_END + 1, 0x5a)] {
            let request = channel.submit_read(lba, 1, buffer, 0).unwrap();
            assert_eq!(channel.wait().unwrap().request, request);
            channel.read_buffer(buffer, 0, &mut observed).unwrap();
            assert!(observed.iter().all(|seen| *seen == byte));
        }
    }

    fn format_old_system_device(
        extent: gpt::Extent,
        era: [u8; crate::persistence::ERA_BYTES],
    ) -> Device {
        let shared = engine_backend::<Device, 1>(Device::new(512, 16).unwrap());
        let lane = shared.claim(0).unwrap();
        let Some(Located::Format {
            provider,
            era: selected,
        }) = crate::block_on(DurableService::locate_system(
            lane,
            extent,
            CompletionMode::Poll,
            era,
        ))
        .unwrap()
        else {
            panic!("blank system media must request format");
        };
        let volume = crate::block_on(PersistentFs::format_with_era(
            provider,
            SessionId::new(0),
            selected,
        ))
        .unwrap();
        drop(volume);
        engine_device(shared)
    }

    fn reformat_system_device(
        device: Device,
        extent: gpt::Extent,
        era: [u8; crate::persistence::ERA_BYTES],
        cut_after: Option<usize>,
    ) -> (Device, bool, usize) {
        let shared = engine_backend::<Device, 1>(device);
        let lane = shared.claim(0).unwrap();
        let Some(Located::Reformat {
            provider,
            era: selected,
        }) = crate::block_on(DurableService::locate_system(
            lane,
            extent,
            CompletionMode::Poll,
            era,
        ))
        .unwrap()
        else {
            panic!("stale system media must request reformat");
        };
        let before = shared.control(|queues| queues.device().submissions()).unwrap();
        if let Some(commands) = cut_after {
            shared
                .control(|queues| queues.device_mut().inject_after(commands, BlockError::Media))
                .unwrap();
        }
        let rebuilt = crate::block_on(PersistentFs::reformat_with_era(
            provider,
            SessionId::new(0),
            selected,
        ));
        let succeeded = rebuilt.is_ok();
        drop(rebuilt);
        let commands = usize::try_from(
            shared.control(|queues| queues.device().submissions()).unwrap() - before,
        )
        .unwrap();
        (engine_device(shared), succeeded, commands)
    }

    fn assert_reboot_never_admits_a_stale_era(
        device: Device,
        extent: gpt::Extent,
        era: [u8; crate::persistence::ERA_BYTES],
    ) {
        let shared = engine_backend::<Device, 1>(device);
        let lane = shared.claim(0).unwrap();
        let located = crate::block_on(DurableService::locate_system(
            lane,
            extent,
            CompletionMode::Poll,
            era,
        ))
        .expect("a consumed one-shot cut must leave readable media")
        .expect("the nonzero archive era keeps system policy active");
        match &located {
            Located::Recover(provider) => assert!(
                crate::block_on(PersistentFs::roots_match_era(provider, era)).unwrap(),
                "ordinary recovery may only admit the requested era"
            ),
            Located::Format { .. } | Located::Reformat { .. } => {}
            Located::Volatile(_) => panic!("system policy never fabricates a RAM recovery"),
        }
        drop(located);
        drop(shared);
    }

    fn dual_designated_disk() -> DualDevice {
        let mut mbr = [0u8; 512];
        mbr[510] = 0x55;
        mbr[511] = 0xaa;
        let (header, entries) = synthetic::table_pair(
            (KORE_FS_TYPE, PART_START, PART_END),
            Some((KORE_SYSTEM_TYPE, SYSTEM_PART_START, SYSTEM_PART_END)),
            SYSTEM_PART_END,
        );
        let provider = dual_whole_provider(DualDevice::new(512, 16).unwrap());
        crate::block_on(provider.write_metadata(0, &mbr)).unwrap();
        crate::block_on(provider.write_metadata(1, &header)).unwrap();
        crate::block_on(provider.write_metadata(2, &entries)).unwrap();
        dual_eject(provider)
    }

    fn name(value: &str) -> Name {
        Name::new(value).unwrap()
    }

    fn on_large_stack(f: impl FnOnce() + Send + 'static) {
        // The region-discovery cold-boot arc holds the whole-disk and narrowed
        // provider/engine shapes across a locate boundary. The raised inline
        // directory arena makes this the larger of the two durability
        // fixtures even though publication images themselves are heap-backed.
        std::thread::Builder::new()
            .name("durable-region-test".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap()
    }

    fn on_system_rebuild_stack(f: impl FnOnce() + Send + 'static) {
        // This witness deliberately carries the 768 KiB whole-disk double by
        // value through four independent cold-boot futures. Production uses a
        // small shared-lane reference, but the test must retain the backing to
        // inspect bytes outside the clamp after every provider is consumed.
        // Keep that debug-only ownership shape off the runner's normal stack.
        std::thread::Builder::new()
            .name("system-era-rebuild-test".into())
            .stack_size(256 * 1024 * 1024)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap()
    }

    #[test]
    fn a_disk_designating_nothing_lands_volatile() {
        on_large_stack(|| {
            // Foreign roots, no GPT: the exact metal boot-disk shape the
            // RAM fallback exists for.
            let provider = whole_provider(Device::new(512, 16).unwrap());
            let mut junk = [0u8; 512];
            junk[..8].copy_from_slice(b"NTFS    ");
            crate::block_on(provider.write_metadata(0, &junk)).unwrap();
            let located =
                crate::block_on(DurableService::locate(provider, CompletionMode::Poll)).unwrap();
            assert!(matches!(
                located,
                Located::Volatile(note::VOLATILE_UNDESIGNATED)
            ));
        });
    }

    #[test]
    fn an_unavailable_designated_system_store_never_falls_back_into_user_bytes() {
        let user = <RamService as SystemService>::bind(FsEndowment::UNSTAMPED);
        let mut mounted = MountedFs {
            user: MountedEngine::Volatile(Box::new(user)),
            system: None,
            system_designated: true,
            user_lifecycle: RegionLifecycle::Current,
            system_lifecycle: RegionLifecycle::Unavailable,
            system_seed_resume: RegionLifecycle::Unavailable,
        };
        assert!(mounted.establish_boot_policy().is_ok());
        let MountedEngine::Volatile(user) = &mounted.user else {
            unreachable!()
        };
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let grants = user.engine_ref().grants(admin).unwrap();
        assert!(grants.iter().all(|grant| {
            grant.display_name != Some(name(crate::SYSTEM_SOURCE_GRANT))
                && grant.display_name != Some(name(crate::SYSTEM_APPS_GRANT))
        }));
    }

    #[test]
    fn system_seed_lifecycle_is_idempotent_and_restores_its_mount_outcome() {
        let user = <RamService as SystemService>::bind(FsEndowment::UNSTAMPED);
        let system = <RamService as SystemService>::bind(FsEndowment::UNSTAMPED);
        let mut mounted = MountedFs {
            user: MountedEngine::Volatile(Box::new(user)),
            system: Some(MountedEngine::Volatile(Box::new(system))),
            system_designated: true,
            user_lifecycle: RegionLifecycle::Current,
            // A rebuilt mount enters this posture before status becomes
            // reachable; exec's first `active` publication is idempotent.
            system_lifecycle: RegionLifecycle::Reseeding,
            system_seed_resume: RegionLifecycle::Rebuilt,
        };

        assert_eq!(
            mounted.set_system_seed_active(0, true),
            Err(FsError::AccessDenied)
        );
        assert_eq!(mounted.system_lifecycle, RegionLifecycle::Reseeding);
        mounted
            .set_system_seed_active(crate::service::contract::ADMIN_VIA, true)
            .unwrap();
        assert_eq!(mounted.system_lifecycle, RegionLifecycle::Reseeding);
        mounted
            .set_system_seed_active(crate::service::contract::ADMIN_VIA, true)
            .unwrap();
        assert_eq!(mounted.system_seed_resume, RegionLifecycle::Rebuilt);
        mounted
            .set_system_seed_active(crate::service::contract::ADMIN_VIA, false)
            .unwrap();
        assert_eq!(mounted.system_lifecycle, RegionLifecycle::Rebuilt);

        // The reformatting vocabulary is a real bounded phase even though a
        // service cannot answer clients until initialization has completed.
        assert_eq!(
            RegionLifecycle::Reformatting.wire(),
            kore_abi::svc::FsSystemStatusWire::LIFECYCLE_REFORMATTING
        );
    }

    #[test]
    fn an_app_only_secondary_peer_completes_and_releases_its_session() {
        use kore_rt::svc::PeerId;

        let mut user = <RamService as SystemService>::bind(FsEndowment::UNSTAMPED);
        let mut system = <RamService as SystemService>::bind(FsEndowment::UNSTAMPED);
        assert!(user.establish_admin_source());
        assert!(system.establish_admin_source());
        assert!(system.establish_admin_apps());
        user.register_system_invocation_view(0xD1, crate::Rights::ALL)
            .unwrap();

        let mut mounted = MountedFs {
            user: MountedEngine::Volatile(Box::new(user)),
            system: Some(MountedEngine::Volatile(Box::new(system))),
            system_designated: true,
            user_lifecycle: RegionLifecycle::Current,
            system_lifecycle: RegionLifecycle::Rebuilt,
            system_seed_resume: RegionLifecycle::Rebuilt,
        };
        let first = PeerId::from_arrival(0x102).unwrap();
        assert!(mounted.peer_admitted(Ok(first)));

        let MountedEngine::Volatile(user) = &mut mounted.user else {
            unreachable!()
        };
        user.apply_invocation_view(0xD1, first, crate::Rights::ALL)
            .unwrap();
        let MountedEngine::Volatile(system) = mounted.system.as_mut().unwrap() else {
            unreachable!()
        };
        let session = SessionId::new(first.slot() as u16);
        assert!(system.install_apps_view(session));
        let grants = system.engine_ref().grants(session).unwrap();
        assert!(
            grants
                .iter()
                .all(|grant| grant.rights == crate::Rights::READ_ONLY),
            "the secondary session exposes no authority wider than the app view"
        );
        let apps = grants
            .into_iter()
            .find(|grant| grant.display_name == Some(name(crate::SYSTEM_APPS_VIEW)))
            .expect("the split system engine publishes the canonical apps view");
        assert_eq!(apps.rights, crate::Rights::READ_ONLY);

        assert!(
            mounted.peer_closed(first.arrival(), PeerClose::Orderly),
            "both engines must leave the peer cell reusable"
        );
        let replacement = PeerId::from_arrival(0x202).unwrap();
        assert_eq!(replacement.slot(), first.slot());
        assert!(
            mounted.peer_admitted(Ok(replacement)),
            "the exact bounded cell is available to its next generation"
        );
    }

    #[test]
    fn dual_designations_mount_two_disjoint_engines_over_one_backend() {
        on_large_stack(|| {
            let shared = engine_backend::<DualDevice, 2>(dual_designated_disk());
            let user_lane = shared.claim(0).unwrap();
            let user_device = RegionDevice::whole(user_lane).unwrap();
            let user_channel = BlockChannel::new(user_device, CompletionMode::Poll).unwrap();
            let bootstrap = bootstrap_data_start(user_channel.geometry().block_size).unwrap();
            let user_provider =
                BlockProvider::<_, 8, 32>::new_with_data_start(user_channel, bootstrap).unwrap();

            let user_extent = crate::block_on(DurableService::discover_region(
                &user_provider,
                KORE_FS_TYPE,
            ))
            .unwrap();
            let system_extent = crate::block_on(DurableService::discover_region(
                &user_provider,
                KORE_SYSTEM_TYPE,
            ))
            .unwrap();
            assert!(user_extent.end_lba < system_extent.start_lba);

            // Claim the sibling before the user locator may retire its lane;
            // this is the production lifecycle ordering, not test ceremony.
            let system_lane = shared.claim(1).unwrap();
            let Located::Format {
                provider: user_provider,
                ..
            } = crate::block_on(DurableService::locate(user_provider, CompletionMode::Poll))
                .unwrap()
            else {
                panic!("blank user designation formats");
            };
            let Some(Located::Format {
                provider: system_provider,
                era,
            }) = crate::block_on(DurableService::locate_system(
                system_lane,
                system_extent,
                CompletionMode::Poll,
                TEST_ERA,
            ))
            .unwrap()
            else {
                panic!("blank system designation formats");
            };
            assert_eq!(era, TEST_ERA);
            assert_eq!(user_provider.geometry().block_count, user_extent.blocks());
            assert_eq!(
                system_provider.geometry().block_count,
                system_extent.blocks()
            );

            let session = SessionId::new(0);
            let user = crate::block_on(PersistentFs::format(user_provider, session)).unwrap();
            let system = crate::block_on(PersistentFs::format(system_provider, session)).unwrap();
            let (user_engine, _) = user.into_parts();
            let (system_engine, _) = system.into_parts();
            let mut user = FsService::new(user_engine);
            let mut system = FsService::new(system_engine);
            user.bind_session(session);
            system.bind_session(session);
            assert!(user.establish_admin_namespace());
            assert!(user.establish_admin_workspace());
            assert!(system.establish_admin_namespace());
            assert!(system.establish_admin_source());
            assert!(system.establish_admin_apps());

            let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
            assert!(
                user.engine_ref()
                    .grants(admin)
                    .unwrap()
                    .iter()
                    .all(|grant| grant.display_name != Some(name(crate::SYSTEM_APPS_GRANT)))
            );
            assert!(
                system
                    .engine_ref()
                    .grants(admin)
                    .unwrap()
                    .iter()
                    .any(|grant| grant.display_name == Some(name(crate::SYSTEM_APPS_GRANT)))
            );
            assert_eq!(shared.stats().map(|(stats, _)| stats.stale_completions), Some(0));
        });
    }

    #[test]
    fn changed_system_era_reformats_inside_both_adjacent_sentinels() {
        on_system_rebuild_stack(|| {
            const OLD_ERA: [u8; crate::persistence::ERA_BYTES] =
                [0x71; crate::persistence::ERA_BYTES];
            const NEW_ERA: [u8; crate::persistence::ERA_BYTES] =
                [0x72; crate::persistence::ERA_BYTES];
            let extent = gpt::Extent {
                start_lba: SYSTEM_PART_START,
                end_lba: SYSTEM_PART_END,
            };
            // Write one entire sentinel block on each side through the raw
            // whole-disk channel. The narrowed system provider cannot name
            // either LBA, so these are an adversarial witness of the clamp,
            // not merely an assertion about offset arithmetic.
            let device = format_system_era(dual_designated_disk(), extent, OLD_ERA);
            let device = write_system_sentinels(device);
            let device = admit_matching_system_era(device, extent, OLD_ERA);
            let device = reformat_system_era(device, extent, NEW_ERA);
            assert_system_sentinels(device);
        });
    }

    #[test]
    fn every_reformat_command_cut_reboots_without_admitting_the_stale_era() {
        on_large_stack(|| {
            const OLD_ERA: [u8; crate::persistence::ERA_BYTES] =
                [0x81; crate::persistence::ERA_BYTES];
            const NEW_ERA: [u8; crate::persistence::ERA_BYTES] =
                [0x82; crate::persistence::ERA_BYTES];
            let extent = gpt::Extent {
                start_lba: PART_START,
                end_lba: PART_END,
            };

            // Measure the exact command count rather than hand-maintaining a
            // guessed cut matrix beside provider batching. Then replay every
            // boundary against a fresh old-era medium.
            let old = format_old_system_device(extent, OLD_ERA);
            let (current, succeeded, command_count) =
                reformat_system_device(old, extent, NEW_ERA, None);
            assert!(succeeded);
            assert!(
                command_count > 2,
                "the witness includes clear, flush, and publish"
            );
            drop(current);

            for cut_after in 0..command_count {
                let old = format_old_system_device(extent, OLD_ERA);
                let (interrupted, succeeded, _) =
                    reformat_system_device(old, extent, NEW_ERA, Some(cut_after));
                assert!(!succeeded, "cut {cut_after} must interrupt its command");
                assert_reboot_never_admits_a_stale_era(interrupted, extent, NEW_ERA);
            }
        });
    }

    #[test]
    fn a_designated_but_foreign_region_is_never_overwritten() {
        on_large_stack(|| {
            // Stale bytes at the partition's first block: designation
            // admits the region, but old data is evidence, not consent.
            // The fixture writes through a raw channel — the provider's
            // own metadata window deliberately cannot reach the partition.
            let mut channel: BlockChannel<Device, 2, 4> =
                BlockChannel::new(designated_disk(), CompletionMode::Poll).unwrap();
            let buffer = channel.register_buffer(512).unwrap();
            let mut stale = [0u8; 512];
            stale[..4].copy_from_slice(b"ext4");
            channel.write_buffer(buffer, 0, &stale).unwrap();
            let request = channel.submit_write(PART_START, 1, buffer, 0).unwrap();
            assert_eq!(channel.wait().unwrap().request, request);
            channel.unregister_buffer(buffer).unwrap();
            let device = channel.into_device().map_err(|(_, error)| error).unwrap();

            let located = crate::block_on(DurableService::locate(
                whole_provider(device),
                CompletionMode::Poll,
            ))
            .unwrap();
            assert!(matches!(
                located,
                Located::Volatile(note::VOLATILE_REFUSED_REGION)
            ));
        });
    }

    #[test]
    fn a_dynamic_peers_writes_land_in_the_persisted_tree_and_survive_recovery() {
        use kore_rt::svc::PeerId;

        on_large_stack(|| {
            let session = SessionId::new(0);
            let device = designated_disk();
            let Located::Format { provider, .. } = crate::block_on(DurableService::locate(
                whole_provider(device),
                CompletionMode::Poll,
            ))
            .unwrap() else {
                panic!("a blank designated region must land on Format");
            };
            let volume = crate::block_on(Volume::format(provider, session)).unwrap();
            let (engine, mut mount) = volume.into_parts();
            let mut fs = FsService::new(engine);
            fs.bind_session(session);
            assert!(fs.establish_admin_source());

            // Admit a dynamic peer exactly the way the serving host does.
            // Admission parks it in the attach transition; its structural
            // root arrives only through the endpoint-authenticated view
            // application — a delegation of the persisted system root, so
            // what it creates is a fact of the persisted tree.
            let peer = PeerId::from_arrival(0x106).unwrap();
            assert!(fs.peer_admitted(Ok(peer)));
            fs.register_system_invocation_view(0xD1, crate::Rights::ALL)
                .unwrap();
            fs.apply_invocation_view(0xD1, peer, crate::Rights::ALL)
                .unwrap();
            let peer_session = SessionId::new(peer.slot() as u16);
            let root = fs
                .engine()
                .open_root(peer_session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(fs.engine().create(
                peer_session,
                root.handle,
                &name("data"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(fs.engine().write_at(
                peer_session,
                file.handle,
                0,
                b"peer-durable",
                crate::ANY_REVISION,
            ))
            .unwrap();
            // The durable entry's own commit: session zero's namespace.
            crate::block_on(commit_engine(fs.engine(), &mut mount, session, None)).unwrap();

            // Peer teardown must retire only the delegation, never the tree
            // it was a view of.
            assert!(fs.peer_closed(peer.arrival(), PeerClose::Orderly));

            // Cold boot: rediscover, recover, and read the peer's bytes back
            // through a fresh peer generation.
            let device = eject(fs.into_engine().into_provider());
            let Located::Recover(provider) = crate::block_on(DurableService::locate(
                whole_provider(device),
                CompletionMode::Poll,
            ))
            .unwrap() else {
                panic!("a committed designated region must land on Recover");
            };
            let volume = crate::block_on(Volume::recover(provider, session)).unwrap();
            let (engine, _) = volume.into_parts();
            let mut fs = FsService::new(engine);
            fs.bind_session(session);
            assert!(fs.establish_admin_source());
            let reborn = PeerId::from_arrival(0x206).unwrap();
            assert!(fs.peer_admitted(Ok(reborn)));
            fs.register_system_invocation_view(0xD2, crate::Rights::ALL)
                .unwrap();
            fs.apply_invocation_view(0xD2, reborn, crate::Rights::ALL)
                .unwrap();
            let peer_session = SessionId::new(reborn.slot() as u16);
            let root = fs
                .engine()
                .open_root(peer_session, crate::Rights::ALL)
                .unwrap();
            let opened = fs
                .engine()
                .open_at(peer_session, root.handle, &name("data"), crate::Rights::ALL)
                .unwrap();
            let mut back = [0u8; 12];
            let read = crate::block_on(fs.engine().read(peer_session, opened.handle, 0, &mut back))
                .unwrap();
            assert_eq!(&back[..read], b"peer-durable");
        });
    }

    #[test]
    fn a_designated_region_formats_persists_and_recovers_across_cold_boots() {
        on_large_stack(|| {
            let session = SessionId::new(0);
            let device = designated_disk();

            // First boot: whole disk is foreign, the designation admits
            // the blank region, and the mount formats inside it.
            let located = crate::block_on(DurableService::locate(
                whole_provider(device),
                CompletionMode::Poll,
            ))
            .unwrap();
            let Located::Format { provider, .. } = located else {
                panic!("a blank designated region must land on Format");
            };
            assert_eq!(provider.geometry().block_count, PART_END - PART_START + 1);

            let mut volume = crate::block_on(Volume::format(provider, session)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(session, crate::Rights::ALL)
                .unwrap();
            let file = crate::block_on(volume.engine_mut().create(
                session,
                root.handle,
                &name("data"),
                ObjectKind::File,
            ))
            .unwrap();
            crate::block_on(volume.engine_mut().write_at(
                session,
                file.handle,
                0,
                b"region-durable",
                crate::ANY_REVISION,
            ))
            .unwrap();
            crate::block_on(volume.commit(session)).unwrap();

            // Cold boot: rediscover from the whole disk, land on Recover,
            // and read the exact bytes back from inside the region.
            let (engine, _) = volume.into_parts();
            let device = eject(engine.into_provider());
            let located = crate::block_on(DurableService::locate(
                whole_provider(device),
                CompletionMode::Poll,
            ))
            .unwrap();
            let Located::Recover(provider) = located else {
                panic!("a formatted designated region must land on Recover");
            };
            let recovered = SessionId::new(1);
            let mut volume = crate::block_on(Volume::recover(provider, recovered)).unwrap();
            let root = volume
                .engine_mut()
                .open_root(recovered, crate::Rights::ALL)
                .unwrap();
            let opened = volume
                .engine_mut()
                .open_at(recovered, root.handle, &name("data"), crate::Rights::ALL)
                .unwrap();
            let mut back = [0u8; 14];
            let read = crate::block_on(volume.engine_mut().read(
                recovered,
                opened.handle,
                0,
                &mut back,
            ))
            .unwrap();
            assert_eq!(&back[..read], b"region-durable");

            // The partition table itself was never rewritten: the whole
            // disk still probes foreign with an intact designation.
            let (engine, _) = volume.into_parts();
            let device = eject(engine.into_provider());
            let provider = whole_provider(device);
            assert_eq!(
                crate::block_on(PersistentFs::probe(&provider)),
                Ok(VolumeProbe::Foreign),
                "the region mount must never touch the whole-disk roots"
            );
            assert!(
                crate::block_on(DurableService::discover_region(
                    &provider,
                    gpt::KORE_FS_TYPE
                ))
                .is_some()
            );
        });
    }
}
