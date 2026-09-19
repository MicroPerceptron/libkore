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
//!   Two affine [`Lane`]s over one [`SharedBackend`] feed independent engines,
//!   each mounted behind an exact [`RegionDevice`] clamp. The frontend tags
//!   engine-local authority words and exposes one unchanged filesystem row.
//!   No user designation still means the bounded RAM posture; no system
//!   designation keeps the compatibility store on that user engine.

use alloc::boxed::Box;

#[cfg(all(target_arch = "x86_64", target_os = "none"))]
use core::sync::atomic::Ordering;
use kore_driver::block::{
    BlockChannel, BlockError, CompletionMode, DeviceCommand, DeviceCompletion, DirectBlockDevice,
    Lane, RegionDevice, SharedBackend,
};
use kore_driver::block::{CompletionPacer, SpinPacer};
use kore_driver::gpt;
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
use kore_driver::host::RoutePacer;
use kore_driver::host::{
    BoundDevice, CapBuf, CapPool, CapRegs, ClassQuiesce, DeadlinePacer, DeviceBinding,
    DeviceResources, DmaActive, DmaPrepared,
};
use kore_driver::seam::Regs;
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
const IO_QDEPTH: u16 = 32;
/// Capability-backed DMA bytes requested from the block provisioner.
const FS_DMA_POOL_BYTES: usize = 0x20_0000;
/// Largest controller page whose complete queue/control graph plus the four
/// cumulatively allocated buffers of a whole-disk -> region rebuild fits the
/// founded 2 MiB DMA pool. At 512 KiB, the four page-aligned queue starts
/// consume so much padding that even the first filesystem buffer cannot be
/// admitted.
const MAX_FS_NVME_PAGE_BYTES: u64 = (FS_DMA_POOL_BYTES / 8) as u64;

const fn fs_nvme_page_supported(page: MemPage) -> bool {
    page.bytes() <= MAX_FS_NVME_PAGE_BYTES
}

/// Completion-wait budget for one route-parked block operation, in
/// nanoseconds — the failure edge, not pacing (`kore-svc-swap-nvme`'s
/// reasoning): a healthy device completes in microseconds, and on expiry
/// the wait's final poll still catches an undelivered completion before
/// the channel is declared lost.  Read fresh per wait from the
/// founder-published clock page, so it needs no executor anchor.
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
/// First declared client in this service's boot fragment. Declared roles are
/// separate from ordinary serving pairs and the existing block provisioner.
pub const WORKSPACE_CLIENT_INDEX: u16 = kore_abi::task::SERVICE_CLIENT_INDEX_BASE;

pub struct NvmeFsBinding;

pub struct NvmeFsPrepared {
    controller: PreparedController<CapRegs, CapPool>,
    vector: Option<u16>,
    pacer: BringUpPacer,
}

pub struct NvmeFsBound {
    backend: Option<NvmeBlockDevice<CapRegs, CapPool>>,
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
        let controller = match controller.start_paced(&mut pacer) {
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
        let backend = NvmeBlockDevice::provision_paced(
            controller,
            1,
            IO_QDEPTH,
            vector,
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

    fn backend(&self) -> &NvmeBlockDevice<CapRegs, CapPool> {
        self.bound()
            .parts()
            .backend
            .as_ref()
            .expect("live binding has backend")
    }

    fn backend_mut(&mut self) -> &mut NvmeBlockDevice<CapRegs, CapPool> {
        self.bound_mut()
            .parts_mut()
            .backend
            .as_mut()
            .expect("live binding has backend")
    }
}

impl DirectBlockDevice for DirectNvme {
    type Buffer = NvmeBuffer<CapBuf>;

    /// The controller's queues advance on their own — inherited from the
    /// backend this wrapper only adds route-aware completion pacing to.  It
    /// is what lets one binding be shared by the two designated regions.
    const INDEPENDENT_PROGRESS: bool =
        <NvmeBlockDevice<CapRegs, CapPool> as DirectBlockDevice>::INDEPENDENT_PROGRESS;

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

const FS_LANES: usize = 2;
const LANE_INFLIGHT: usize = 32;
type SharedNvme = SharedBackend<DirectNvme, FS_LANES, LANE_INFLIGHT>;
type SharedLane = Lane<'static, DirectNvme, FS_LANES, LANE_INFLIGHT>;
type Provider = BlockProvider<RegionDevice<SharedLane>, 8, 32>;

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

    fn mirror_peer_admitted(&mut self, peer: PeerId) -> bool {
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

    async fn serve_workspace_request_async(
        &mut self,
        workspace: &mut crate::live_workspace::WorkspaceMount<LiveChannel>,
        request: FsRequest,
        sv: &mut Serve<'_, FsResponse>,
    ) -> FsResponse {
        let Some(session) = self.session_for_via(sv.via()) else {
            return FsResponse::Refused(FsError::AccessDenied);
        };
        let intercepted = match self {
            Self::Durable { fs, .. } => workspace.before_request(fs, session, &request).await,
            Self::Volatile(fs) => workspace.before_request(fs, session, &request).await,
        };
        // Import/refresh publishes topology during a read-shaped request.
        // It must reach this store's native persistence generation even
        // though the ordinary mutation classifier would not commit a read.
        if workspace.topology_dirty() {
            if let Self::Durable { fs, mount, commits } = self {
                if let Err(error) =
                    commit_engine(fs.engine(), mount, SessionId::new(0), None::<CommitStage>).await
                {
                    return FsResponse::Refused(DurableService::persistence_error(error));
                }
                *commits = commits.saturating_add(1);
            }
            workspace.topology_committed();
        }
        match intercepted {
            Ok(Some(response)) => response,
            Err(error) => FsResponse::Refused(error),
            Ok(None) => self.serve_request_async(request, sv).await,
        }
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

type LiveChannel =
    kore_svc_hostlink::RingHostLink<'static, 'static, kore_rt::transport::TrapTransport>;

/// One client-visible filesystem row over a conservative user engine and an
/// optional rebuildable system engine. The router is the only layer that sees
/// both; each engine retains its original session and authority model.
#[cfg_attr(not(all(target_arch = "x86_64", target_os = "none")), allow(dead_code))]
struct MountedFs {
    workspace: crate::live_workspace::WorkspaceMount<LiveChannel>,
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
            && !system.mirror_peer_admitted(peer)
        {
            // Publish a refusal through the primary engine's one admission
            // completion slot. The secondary owns no control-plane reply.
            let _ = self
                .user
                .peer_admitted(Err(PeerAdmissionRefusal::PolicyUnavailable));
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
                let response = self
                    .user
                    .serve_workspace_request_async(&mut self.workspace, request, sv)
                    .await;
                // The user view is the terminal's structural root. The
                // secondary engine mirrors only the explicit read-only apps
                // projection; failure is the established best-effort posture
                // (the terminal still boots without a store view).
                if !matches!(response, FsResponse::Refused(_))
                    && let (Some(peer), Some(system)) = (applied_peer, self.system.as_mut())
                {
                    let _ = system.install_apps_view(SessionId::new(peer.slot() as u16));
                    let _ = system.install_libs_view(SessionId::new(peer.slot() as u16));
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
enum Located<D: DirectBlockDevice, const B: usize, const O: usize> {
    Recover(BlockProvider<RegionDevice<D>, B, O>),
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

    /// Classify the provisioned medium and land on exactly one mount
    /// posture.  A whole disk that is kore's or blank keeps the identity
    /// view; a foreign disk is consulted for its designated partition and,
    /// when one exists, the composition narrows onto it behind the
    /// [`RegionDevice`] clamp and re-probes inside.  Designation admits
    /// the region; only recognized or blank roots inside it authorize user
    /// writes — stale foreign bytes inside a user partition still land in
    /// the RAM posture rather than being overwritten. The separate system
    /// locator owns its explicit archive-era rebuild policy.
    async fn locate<D: DirectBlockDevice, const B: usize, const O: usize>(
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
                let volume = PersistentFs::recover(provider, session)
                    .await
                    .map_err(|_| StartFailure::new(17))?;
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
        let shared: &'static SharedNvme = Box::leak(Box::new(
            SharedNvme::new(backend).map_err(|_| StartFailure::new(14))?,
        ));
        self.shared = Some(shared);
        let user_lane = shared.claim(0).map_err(|_| StartFailure::new(14))?;
        let device = RegionDevice::whole(user_lane).map_err(|_| StartFailure::new(14))?;
        let channel = BlockChannel::new(device, mode).map_err(|_| StartFailure::new(14))?;
        let bootstrap = bootstrap_data_start(channel.geometry().block_size)
            .map_err(|_| StartFailure::new(15))?;
        let provider = BlockProvider::new_with_data_start(channel, bootstrap)
            .map_err(|_| StartFailure::new(15))?;
        let session = SessionId::new(0);
        // Read both designations while lane zero still has the whole-disk
        // view. Claim lane one before `locate` may retire lane zero: the last
        // live lane is what quiesces the physical controller.
        let system_extent = if PersistentFs::probe(&provider).await == Ok(VolumeProbe::Foreign) {
            Self::discover_region(&provider, gpt::KORE_SYSTEM_TYPE).await
        } else {
            None
        };
        let system_lane = match system_extent {
            Some(_) => Some(shared.claim(1).map_err(|_| StartFailure::new(26))?),
            None => None,
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
        let mut mounted = MountedFs {
            workspace: crate::live_workspace::WorkspaceMount::empty(),
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
                        shared.control(|device| {
                            let route = device.route_stats();
                            let health = device.health().ok();
                            (route, physical, health)
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
                ) = device.map(|(route, _, _)| route).unwrap_or_default();
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
                    (Some((_, physical, health)), Some(occupied_blocks))
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

    /// The terminal sequence (#809 §L, L.8 item 2): the row holds *two*
    /// lanes over one controller, and the controller's own quiesce (`CC.SHN`)
    /// runs only when the **last** live lane retires — so this is an ordered
    /// sequence, not one call.  Retire the user and system engines' providers
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
        match shared.control(DirectNvme::teardown) {
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

    fn bind(endowment: FsEndowment) -> Self {
        Self {
            fs: None,
            shared: None,
            peer_map: None,
            archive_era: endowment.archive_era(),
            route_held: false,
            note: 0,
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

    fn peer_closed(&mut self, via: u16, reason: PeerClose) -> bool {
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

#[cfg(all(target_arch = "x86_64", target_os = "none"))]
fn open_workspace_channel(
    transport: &'static kore_rt::transport::TrapTransport,
    boot: &kore_abi::task::Bootstrap<'_>,
) -> Result<LiveChannel, StartFailure> {
    use kore_abi::task::role;
    use kore_svc_hostlink::{HostLinkWire, RingHostLink};

    let kick = boot
        .handle(role::SVC_SEND, WORKSPACE_CLIENT_INDEX)
        .ok_or(StartFailure::new(31))?;
    let ring = boot
        .handle(role::SVC_RING, WORKSPACE_CLIENT_INDEX)
        .ok_or(StartFailure::new(32))?;
    let leased = kore_rt::cap::verbs::take_view(ring).ok_or(StartFailure::new(33))?;
    let window = kore_rt::cap::verbs::map_view(leased).ok_or(StartFailure::new(34))?;
    // SAFETY: the founded leaf and its mapping remain held for the task's
    // lifetime, like the block client above. This is the only client-side
    // attacher for HostLink's importer pair; neither broker nor agents receive
    // this view. HostLink owns its separately derived serving sibling.
    let window: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
    let rings = client_rings::<HostLinkWire>(
        window,
        HostLinkWire::REQUEST_CAPACITY,
        HostLinkWire::RESPONSE_CAPACITY,
    )
    .map_err(|_| StartFailure::new(35))?;
    Ok(RingHostLink::new(
        transport,
        rings,
        ClientEndpoints {
            kick,
            grants: [const { None }; kore_rt::service::MAX_GRANTS],
        },
    ))
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
/// kore-volume recovery, 18 blank formatting, 19 administration-root
/// publication, 20 foreign-media channel containment, 21 recyclable-peer
/// mapping geometry, 22 user-region channel rebuild, 23 the probe inside the
/// user region, 24 durable-workspace publication, 25 application-store
/// publication, 26 system-region channel construction, 27 the probe inside
/// the system region, 28 system-lane containment or missing-era retirement,
/// 29 archive-era root admission, and 30 bounded system reformat. Details
/// 31–35 identify missing workspace kick/ring, invalid view, failed mapping,
/// or malformed rings; 36 means the user mount was absent at attachment.
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
    let workspace = match open_workspace_channel(transport, rt.bootstrap()) {
        Ok(channel) => crate::live_workspace::WorkspaceMount::new(channel),
        Err(failure) => refuse_with(&rt, failure.payload(RUNG_SERVICE_START)),
    };
    let Some(mounted) = service.fs.as_mut() else {
        refuse_with(&rt, StartFailure::new(36).payload(RUNG_SERVICE_START));
    };
    // A missing capability is a founding failure; a disconnected host is a
    // retryable source condition. Requests attach/refresh through this sole
    // affine owner after user/system routing, so system seeding stays local.
    mounted.workspace = workspace;
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
mod tests {
    use super::*;
    use crate::persistence::VolumeProbe;
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
        let device = RegionDevice::whole(device).unwrap();
        let channel = BlockChannel::new(device, CompletionMode::Poll).unwrap();
        let bootstrap = bootstrap_data_start(channel.geometry().block_size).unwrap();
        BlockProvider::new_with_data_start(channel, bootstrap).unwrap()
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
        let device = RegionDevice::whole(device).unwrap();
        let channel = BlockChannel::new(device, CompletionMode::Poll).unwrap();
        let bootstrap = bootstrap_data_start(channel.geometry().block_size).unwrap();
        BlockProvider::new_with_data_start(channel, bootstrap).unwrap()
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
        type Shared = SharedBackend<Device, 1, 8>;
        let shared = Shared::new(Device::new(512, 16).unwrap()).unwrap();
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
        shared.into_device()
    }

    fn reformat_system_device(
        device: Device,
        extent: gpt::Extent,
        era: [u8; crate::persistence::ERA_BYTES],
        cut_after: Option<usize>,
    ) -> (Device, bool, usize) {
        type Shared = SharedBackend<Device, 1, 8>;
        let shared = Shared::new(device).unwrap();
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
        let before = shared.control(|device| device.submissions()).unwrap();
        if let Some(commands) = cut_after {
            shared
                .control(|device| device.inject_after(commands, BlockError::Media))
                .unwrap();
        }
        let rebuilt = crate::block_on(PersistentFs::reformat_with_era(
            provider,
            SessionId::new(0),
            selected,
        ));
        let succeeded = rebuilt.is_ok();
        drop(rebuilt);
        let commands =
            usize::try_from(shared.control(|device| device.submissions()).unwrap() - before)
                .unwrap();
        (shared.into_device(), succeeded, commands)
    }

    fn assert_reboot_never_admits_a_stale_era(
        device: Device,
        extent: gpt::Extent,
        era: [u8; crate::persistence::ERA_BYTES],
    ) {
        type Shared = SharedBackend<Device, 1, 8>;
        let shared = Shared::new(device).unwrap();
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
            workspace: crate::live_workspace::WorkspaceMount::empty(),
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
            workspace: crate::live_workspace::WorkspaceMount::empty(),
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
            workspace: crate::live_workspace::WorkspaceMount::empty(),
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
            type Shared = SharedBackend<DualDevice, 2, 8>;
            let shared = Shared::new(dual_designated_disk()).unwrap();
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
            assert_eq!(shared.stats().stray, 0);
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
