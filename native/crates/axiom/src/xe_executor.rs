//! Runtime-owned Xe execution of prepared Axiom instructions.
//!
//! The caller owns the accelerator face, loaded programs and immutable weight
//! supply. This owner retains scratch custody and the existing buffer ledger;
//! no graph borrow or output pointer survives `advance`.

use core::cell::Cell;

use ax_exec::{ChainedInput, KernelProgress, NativeRef, PreparedKernelExecutor};
use axnn::lowering::{KernelAccuracy, MoeParams};
use axnn_cpu::CpuKernelPlan;
use kore_abi::session::{FabricAccess, FabricRouteScopeReq};
use kore_rt::{
    cap::{FabricRouteToken, StashedLease, StashedMemory, verbs},
    chan::{NoticeRx, RecvError},
    executor::{WakeReason, clock_ns, next_wake_on_for},
    svc::{CallError, RetainedReturnCallError},
    transport::{Status, TrapTransport},
};
use kore_svc_accel::{
    client::{
        BufferExecution, BufferExecutionActivity, BufferExecutionError, ExecutionBuffer,
        ExecutionError, ExecutionLedger, ExecutionOutcome, ExecutionProgress, ImportTarget,
        OrderedExecution, OrderedLaunch, ProgramReuse, RingAccel, stage_ordered_launches,
    },
    contract::BufferUsage,
    service::{AccelEventState, AccelRefusal, AccelRequest, AccelResponse},
};

use crate::xe_batch::{self, Member, MemberKey, MemberState, Placement, WINDOW, Window};
use crate::xe_graph::{KERNEL_BUFFER_COUNT, Kernel, KernelLaunch, MAX_PORTS, Port};
use crate::xe_moe::{self, WavePlan, WaveRefusal};
use kore_xe_kernels::MoeWave;

// Fabric routes and owned heap leases have the page geometry of the runtime
// memory interface. Tensor interiors stay in binding offsets.
const PAGE: usize = 4096;
const PIPELINE_DEPTH: usize = kore_abi::svc::MAX_ACCEL_PIPELINE_DEPTH;

fn scratch_slot(
    capacities: &[u64; KERNEL_BUFFER_COUNT],
    available: &[bool; KERNEL_BUFFER_COUNT],
    required: u64,
) -> Option<usize> {
    capacities
        .iter()
        .enumerate()
        .filter(|(index, bytes)| available[*index] && **bytes >= required)
        .min_by_key(|(_, bytes)| **bytes)
        .map(|(index, _)| index)
}

/// A program already loaded on the caller's context. Selection compares the
/// sealed package identity, not a model node number or a claimed GPU vendor.
/// The executor's activity as one word for a watchdog: bit 63 inside a
/// call into the service, bits 56..59 that call (1 submit, 2 poll, 3
/// retire), bits 48..55 the activity (0 idle, 1 prepare, 2 scratch select,
/// 3 scratch map, 4 scratch retire, 5 recover map, 6 recover retire, 7
/// import, 8 execute, 9 export, 10 finished), bits 24..47 the instruction
/// (all ones for none) and bits 0..23 the kernels completed.
pub fn observed_activity(
    inside: bool,
    request: u8,
    activity: Activity,
    completed: u64,
    instruction: Option<usize>,
) -> u64 {
    let code: u64 = match activity {
        Activity::Idle => 0,
        Activity::Prepare => 1,
        Activity::ScratchSelect { .. } => 2,
        Activity::ScratchMap { .. } => 3,
        Activity::ScratchRetire { .. } => 4,
        Activity::RecoverMap { .. } => 5,
        Activity::RecoverRetire => 6,
        Activity::Wire(BufferExecutionActivity::Import { .. }) => 7,
        Activity::Wire(BufferExecutionActivity::Execute) => 8,
        Activity::Wire(BufferExecutionActivity::Export { .. }) => 9,
        Activity::Wire(BufferExecutionActivity::Finished) => 10,
    };
    (u64::from(inside) << 63)
        | (u64::from(request & 0xf) << 56)
        | (code << 48)
        | ((instruction.map_or(0xff_ffff, |i| i as u64) & 0xff_ffff) << 24)
        | (completed & 0xff_ffff)
}

/// Store an [`observed_activity`] word to `observer`, when there is one.
fn publish_activity(
    observer: Option<&core::sync::atomic::AtomicU64>,
    inside: bool,
    request: u8,
    activity: Activity,
    completed: u64,
    instruction: Option<usize>,
) {
    if let Some(observer) = observer {
        observer.store(
            observed_activity(inside, request, activity, completed, instruction),
            core::sync::atomic::Ordering::Release,
        );
    }
}

/// The words of an [`observed_activity`] word, for a report.
pub fn describe_observed_activity(word: u64) -> alloc::string::String {
    let inside = word >> 63 != 0;
    let request = match (word >> 56) & 0xf {
        1 => "submit",
        2 => "poll",
        3 => "retire",
        _ => "no call",
    };
    let activity = match (word >> 48) & 0xff {
        0 => "idle",
        1 => "prepare",
        2 => "scratch select",
        3 => "scratch map",
        4 => "scratch retire",
        5 => "recover map",
        6 => "recover retire",
        7 => "import",
        8 => "execute",
        9 => "export",
        10 => "finished",
        _ => "?",
    };
    let instruction = (word >> 24) & 0xff_ffff;
    alloc::format!(
        "{} {} ({activity}) i{} k{}",
        if inside { "in" } else { "after" },
        request,
        if instruction == 0xff_ffff { alloc::string::String::from("-") } else { alloc::format!("{instruction}") },
        word & 0xff_ffff
    )
}

pub struct LoadedKernel {
    pub artifact: kore_xe_kernels::XeKernelArtifact,
    pub program: u64,
}

/// Bounded observation policy supplied by the runtime. Completion notices are
/// wake hints; only event retirement plus export permits CPU access.
#[derive(Clone, Copy)]
pub struct WaitPolicy {
    pub job_ns: u64,
    pub call_ns: u64,
    pub completion_ns: u64,
    pub operation_ns: u64,
}

/// Host wall-time observations, including IPC/scheduling delays. Missing or
/// regressing clocks count as unclocked observations, never zero-time work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WallTime {
    pub ns: u64,
    pub calls: u64,
    pub unclocked: u64,
}

impl WallTime {
    fn record(&mut self, start: Option<u64>, end: Option<u64>) {
        self.calls = self.calls.saturating_add(1);
        if let Some(elapsed) = start.zip(end).and_then(|(a, b)| b.checked_sub(a)) {
            self.ns = self.ns.saturating_add(elapsed);
        } else {
            self.unclocked = self.unclocked.saturating_add(1);
        }
    }

    fn since(self, before: Self) -> Self {
        Self {
            ns: self.ns.saturating_sub(before.ns),
            calls: self.calls.saturating_sub(before.calls),
            unclocked: self.unclocked.saturating_sub(before.unclocked),
        }
    }
}

/// Opt-in observations for completed host calls, including returned errors.
/// Categories do not overlap. In-flight/cancelled futures that never return
/// are absent; execute includes host publication/observation/retirement and
/// must not be interpreted as GPU kernel time. No per-job logging or storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Profile {
    pub prepare: WallTime,
    pub import: WallTime,
    pub execute: WallTime,
    pub export: WallTime,
    pub recover: WallTime,
    pub notice: WallTime,
    /// The batched lane's `execute` calls by the request each carried: a
    /// job's submission, a poll (Pending, or the completion), and its
    /// retirement (the event's destroy). Each is one wire round trip.
    pub submit: WallTime,
    pub poll: WallTime,
    pub retire: WallTime,
    /// CPU wall time copying completed recurrent successor states from scratch.
    /// Excludes GPU waits, ordinary outputs, direct targets and prefix commit.
    pub state_copy: WallTime,
    /// Completion notices the service actually delivered, out of
    /// `notice.calls` waits: zero means every wait ran out its timeout.
    pub notices: u64,
    /// A gated sequence's host half ([`GateService`]): the serve calls that
    /// answered a signal and the host time they took, and the turns the
    /// core slept on the signal word between them.
    pub gate: WallTime,
    pub gate_serves: u64,
    pub monitor_turns: u64,
    /// How the accel client's reply waits ended, read from the client when
    /// the profile is (cumulative from the client's construction, so a
    /// `since` delta is the span's): replies already there, woken by the
    /// service's reply kick, or first seen after a 1 ms re-kick deadline.
    pub replies: kore_rt::svc::ReplyWaitStats,
    /// Ordered submissions of the batched executor, and the members
    /// (instructions and MoE wave launches) they carried.
    pub sequences: u64,
    /// Admissions acknowledged before the preceding event was retired.
    pub queued_sequences: u64,
    pub members: u64,
    /// Actual host copy-back bytes, and bytes whose host publication was
    /// avoided after all consumers were admitted. These are traffic counts,
    /// not elapsed time or estimates of physical DRAM transactions.
    pub copied_output_bytes: u64,
    pub copied_state_bytes: u64,
    /// Successor-state bytes written directly into retained decoder storage.
    pub direct_state_bytes: u64,
    pub avoided_output_bytes: u64,
    /// MoE expert waves the batched lane admitted and their routes, and the
    /// waves it declined, counted by reason in [`WAVE_DECLINES`] order; a
    /// declined wave, and every later one of its instruction, runs on the
    /// CPU.
    pub waves: u64,
    pub wave_routes: u64,
    pub wave_declined: [u64; WAVE_DECLINES.len()],
    /// Routed-expert instructions (`advance_moe_routed`) the lane would not
    /// plan on the device, by reason in [`WAVE_DECLINES`] order. None is a
    /// fallback to the CPU: the engine then routes the layer on the host
    /// and offers its waves as above.
    pub routed_declined: [u64; WAVE_DECLINES.len()],
    /// Members (instructions offered to the window) the batched lane
    /// declined, by reason in [`MEMBER_DECLINES`] order. A final decline
    /// (nothing admitted ahead of the offer) runs the instruction on the
    /// CPU, or under strict placement fails the invocation.
    pub member_declined: [u64; MEMBER_DECLINES.len()],
}

/// Why an offered expert wave stayed on the CPU, as named in
/// [`Profile::wave_declined`].
pub const WAVE_DECLINES: [&str; 7] = [
    "program", "accuracy", "geometry", "resident", "kernel", "window", "scratch",
];

/// Why an offered member stayed on the CPU, as named in
/// [`Profile::member_declined`]: no package for the op or shape; a package
/// not loaded; a small op offered alone (it joins only as a follower); an
/// operand view the package cannot read; no room in the window (retried
/// after a drain); no room in the scratch arena, or overlapping ranges.
pub const MEMBER_DECLINES: [&str; 6] = [
    "package", "program", "solo", "operands", "window", "scratch",
];

#[derive(Clone, Copy, Debug)]
enum MemberDecline {
    Package,
    Program,
    Solo,
    Operands,
    Window,
    Scratch,
}

impl MemberDecline {
    const fn index(self) -> usize {
        match self {
            Self::Package => 0,
            Self::Program => 1,
            Self::Solo => 2,
            Self::Operands => 3,
            Self::Window => 4,
            Self::Scratch => 5,
        }
    }

    /// A decline the executor could have avoided: the package is loaded and
    /// the op is eligible, but this lane would not place it.
    const fn placement(self) -> bool {
        matches!(self, Self::Operands | Self::Scratch)
    }
}

#[derive(Clone, Copy, Debug)]
enum WaveDecline {
    /// A wave package is not loaded (unsealed, or refused by this GPU).
    Program,
    /// The instruction asks for more fidelity than the FP16 DPAS packages.
    Accuracy,
    Plan(WaveRefusal),
    /// A launch's geometry does not fit its package.
    Kernel,
    /// The admission window lacks room for the wave's launches.
    Window,
    /// The scratch arena lacks room for the wave's region.
    Scratch,
}

impl WaveDecline {
    const fn index(self) -> usize {
        match self {
            Self::Program => 0,
            Self::Accuracy => 1,
            Self::Plan(WaveRefusal::Geometry) => 2,
            Self::Plan(WaveRefusal::NotResident) => 3,
            Self::Kernel => 4,
            Self::Window => 5,
            Self::Scratch => 6,
        }
    }
}

impl Profile {
    pub fn since(self, before: Self) -> Self {
        Self {
            prepare: self.prepare.since(before.prepare),
            import: self.import.since(before.import),
            execute: self.execute.since(before.execute),
            export: self.export.since(before.export),
            recover: self.recover.since(before.recover),
            notice: self.notice.since(before.notice),
            submit: self.submit.since(before.submit),
            poll: self.poll.since(before.poll),
            retire: self.retire.since(before.retire),
            state_copy: self.state_copy.since(before.state_copy),
            notices: self.notices.saturating_sub(before.notices),
            gate: self.gate.since(before.gate),
            gate_serves: self.gate_serves.saturating_sub(before.gate_serves),
            monitor_turns: self.monitor_turns.saturating_sub(before.monitor_turns),
            replies: self.replies.since(before.replies),
            sequences: self.sequences.saturating_sub(before.sequences),
            queued_sequences: self
                .queued_sequences
                .saturating_sub(before.queued_sequences),
            members: self.members.saturating_sub(before.members),
            copied_output_bytes: self
                .copied_output_bytes
                .saturating_sub(before.copied_output_bytes),
            copied_state_bytes: self
                .copied_state_bytes
                .saturating_sub(before.copied_state_bytes),
            direct_state_bytes: self
                .direct_state_bytes
                .saturating_sub(before.direct_state_bytes),
            avoided_output_bytes: self
                .avoided_output_bytes
                .saturating_sub(before.avoided_output_bytes),
            waves: self.waves.saturating_sub(before.waves),
            wave_routes: self.wave_routes.saturating_sub(before.wave_routes),
            wave_declined: core::array::from_fn(|i| {
                self.wave_declined[i].saturating_sub(before.wave_declined[i])
            }),
            routed_declined: core::array::from_fn(|i| {
                self.routed_declined[i].saturating_sub(before.routed_declined[i])
            }),
            member_declined: core::array::from_fn(|i| {
                self.member_declined[i].saturating_sub(before.member_declined[i])
            }),
        }
    }
}

/// Immutable bytes backed by an independently retained fabric parent. Only
/// wholly contained, page-rounded READ scopes may be lent without a copy.
pub struct SharedInputs<'a> {
    backing: InputBacking<'a>,
    bytes: &'a [u8],
    scoped: &'a [ScopedInput<'a>],
}

#[derive(Clone, Copy)]
enum InputBacking<'a> {
    Local,
    Route(&'a FabricRouteToken),
    Resident(u64),
}

impl<'a> SharedInputs<'a> {
    /// No persistent weight mapping: all admitted inputs use owned scratch.
    pub const fn local() -> Self {
        Self {
            backing: InputBacking::Local,
            bytes: &[],
            scoped: &[],
        }
    }
    /// # Safety
    /// `bytes` must start at offset zero of `parent`, and its backing mapping,
    /// registration and supply must remain alive until all device work and
    /// imports have retired, including when a graph future or this owner is
    /// abandoned. The caller's containment guard must quarantine that supply
    /// on abandonment. Neither CPU nor another device may mutate these bytes.
    pub unsafe fn new(parent: &'a FabricRouteToken, bytes: &'a [u8]) -> Option<Self> {
        if bytes.is_empty() || bytes.as_ptr() as usize % PAGE != 0 {
            return None;
        }
        Some(Self {
            backing: InputBacking::Route(parent),
            bytes,
            scoped: &[],
        })
    }

    /// Bind interior offsets of an archive already imported on this context.
    /// Projections reuse that import without per-instruction page-table changes.
    ///
    /// # Safety
    /// `buffer` must name offset zero of `bytes` on the supplied face/context,
    /// with READ authority and at least `bytes.len()` mapped bytes. Its owner
    /// must retain the import, mapping, supply and immutable contents through
    /// quiescence or process containment, including abandoned continuations.
    pub unsafe fn resident(buffer: u64, bytes: &'a [u8]) -> Option<Self> {
        if buffer == 0 || bytes.is_empty() || bytes.as_ptr() as usize % PAGE != 0 {
            return None;
        }
        Some(Self {
            backing: InputBacking::Resident(buffer),
            bytes,
            scoped: &[],
        })
    }

    /// Add independently retained regions whose offered input ranges stay
    /// stable until retirement. Unlike an immutable archive, each region gets
    /// a fresh READ scope/import on use, publishing writes from the latest load.
    pub fn with_scoped(mut self, regions: &'a [ScopedInput<'a>]) -> Self {
        self.scoped = regions;
        self
    }

    /// The resident import holding `address`: `(buffer, offset, bytes)` of
    /// the whole region it imports.
    fn resident_region(&self, address: usize) -> Option<(u64, u64, u64)> {
        if let InputBacking::Resident(buffer) = self.backing {
            let base = self.bytes.as_ptr() as usize;
            if (base..base + self.bytes.len()).contains(&address) {
                return Some((buffer, (address - base) as u64, self.bytes.len() as u64));
            }
        }
        self.scoped.iter().find_map(|region| {
            let buffer = region.resident?;
            (region.base..region.base + region.bytes)
                .contains(&address)
                .then(|| (buffer, (address - region.base) as u64, region.bytes as u64))
        })
    }

    fn resolve(
        &self,
        input: &[u8],
        required: usize,
    ) -> Option<(InputBacking<'a>, (u64, u64, u64))> {
        if let Some(span) = shared_span(
            self.bytes.as_ptr() as usize,
            self.bytes.len(),
            input.as_ptr() as usize,
            input.len(),
            required,
        ) {
            return Some((self.backing, span));
        }
        self.scoped.iter().find_map(|region| {
            shared_span(
                region.base,
                region.bytes,
                input.as_ptr() as usize,
                input.len(),
                required,
            )
            .map(|span| {
                let backing = region
                    .resident
                    .map_or(InputBacking::Route(region.parent), InputBacking::Resident);
                (backing, span)
            })
        })
    }

    /// The buffers this view binds in place, by handle and size, for a
    /// failure report (no addresses).
    fn describe(&self, out: &mut impl core::fmt::Write) {
        match self.backing {
            InputBacking::Resident(buffer) => {
                let _ = write!(out, "archive {buffer}/{}", self.bytes.len());
            }
            InputBacking::Local => {
                let _ = write!(out, "local");
            }
            InputBacking::Route(_) => {
                let _ = write!(out, "route/{}", self.bytes.len());
            }
        }
        for region in self.scoped {
            match region.resident {
                Some(buffer) => {
                    let _ = write!(out, " scoped {buffer}/{}", region.bytes);
                }
                None => {
                    let _ = write!(out, " scoped route/{}", region.bytes);
                }
            }
        }
    }

    /// The bytes of a buffer this view binds in place: the archive's, or a
    /// scoped region's imported over its whole extent.
    fn buffer_bytes(&self, buffer: u64) -> Option<u64> {
        if let InputBacking::Resident(id) = self.backing
            && id == buffer
        {
            return Some(self.bytes.len() as u64);
        }
        self.scoped
            .iter()
            .find(|region| region.resident == Some(buffer))
            .map(|region| region.bytes as u64)
    }
}

/// Registration metadata for a runtime-owned, potentially mutable cache.
/// No Rust slice to the whole cache is retained across its future writes.
pub struct ScopedInput<'a> {
    parent: &'a FabricRouteToken,
    base: usize,
    bytes: usize,
    /// A buffer this face imported once over the whole region: offered
    /// ranges bind as interior offsets, with no per-use scope, import,
    /// export or page-table change.
    resident: Option<u64>,
}

impl<'a> ScopedInput<'a> {
    /// # Safety
    /// `base` must identify offset zero of `parent` with `bytes` live mapped
    /// bytes. The runtime owns this mapping, registration and supply until all
    /// imported scopes retire, including cancellation and abandonment. Every
    /// input range offered to this executor must remain unchanged until its
    /// event and READ scope retire. Cache eviction and incoming DMA must obey
    /// that pin. On failure the owner quarantines the parent until containment.
    /// A call-scoped source slice alone does not establish these guarantees.
    pub unsafe fn new(parent: &'a FabricRouteToken, base: *const u8, bytes: usize) -> Option<Self> {
        if base.is_null()
            || base as usize % PAGE != 0
            || bytes == 0
            || bytes % PAGE != 0
            || (base as usize).checked_add(bytes).is_none()
        {
            return None;
        }
        Some(Self {
            parent,
            base: base as usize,
            bytes,
            resident: None,
        })
    }

    /// The region imported once, whole, as `buffer`.
    ///
    /// # Safety
    /// Everything [`Self::new`] requires, and: `buffer` names offset zero of
    /// the whole region on this executor's face/context with READ authority
    /// and at least `bytes` mapped bytes, and its owner retains the import
    /// through quiescence or process containment. The per-use pin keeps its
    /// meaning with event retirement as the edge: an offered range stays
    /// unchanged while any event that binds it is live. Only DMA (not CPU
    /// stores) may rewrite the region while it is imported, since the
    /// import publishes CPU writes once.
    pub unsafe fn resident(
        parent: &'a FabricRouteToken,
        base: *const u8,
        bytes: usize,
        buffer: u64,
    ) -> Option<Self> {
        if buffer == 0 {
            return None;
        }
        // SAFETY: forwarded; the caller upholds `new`'s contract.
        let mut region = unsafe { Self::new(parent, base, bytes) }?;
        region.resident = Some(buffer);
        Some(region)
    }
}

/// Decoder-owned storage imported once read-write. Unlike scratch, the CPU
/// owns its contents between invocations; only recurrent target ranges are
/// bound here, never arbitrary call-scoped activation inputs.
pub struct ResidentState<'a> {
    buffer: u64,
    base: usize,
    bytes: usize,
    _region: core::marker::PhantomData<&'a FabricRouteToken>,
}

impl ResidentState<'_> {
    /// # Safety
    /// `buffer` names offset zero of this mapped region on the executor's
    /// face/context, imported READ|WRITE with MUTABLE_STATE. The owner retains
    /// the import, mapping, registration and supply through quiescence or
    /// process containment, even if the graph future is abandoned. Every
    /// offered target stays allocated at the same address and is exclusively
    /// device-accessible until its admission retires: no CPU reads, writes,
    /// references, reset, reuse or deallocation may overlap that interval.
    /// After cancellation, drain before reusing the decoder, or quarantine
    /// the entire arena. A call-scoped target borrow is not this proof.
    pub unsafe fn new(buffer: u64, base: *mut u8, bytes: usize) -> Option<Self> {
        if buffer == 0 || base.is_null() || bytes == 0 {
            return None;
        }
        shared_span(base as usize, bytes, base as usize, bytes, bytes)?;
        Some(Self {
            buffer,
            base: base as usize,
            bytes,
            _region: core::marker::PhantomData,
        })
    }

    /// Resolve only whole, isolated cache lines. Publication/acquisition on
    /// noncoherent devices must not touch a neighboring CPU allocation.
    fn resolve(&self, target: axnn_cpu::StateTargetRef<'_>) -> Option<Placement> {
        let start = target.as_mut_ptr() as usize;
        let bytes = target.len().checked_mul(core::mem::size_of::<f32>())?;
        let alignment = xe_batch::ALIGN as usize;
        if bytes == 0 || start % alignment != 0 || bytes % alignment != 0 {
            return None;
        }
        let (page, _, interior) = shared_span(self.base, self.bytes, start, bytes, bytes)?;
        Some(Placement::Resident {
            buffer: self.buffer,
            offset: page + interior,
        })
    }
}

fn recurrent_target_matches(
    admitted: Option<(Placement, u64)>,
    target: Option<axnn_cpu::StateTargetRef<'_>>,
    region: Option<&ResidentState<'_>>,
) -> bool {
    match (admitted, target) {
        (None, None) => true,
        (Some((placement, bytes)), Some(target)) => {
            target
                .len()
                .checked_mul(core::mem::size_of::<f32>())
                .map(|n| n as u64)
                == Some(bytes)
                && match placement {
                    Placement::Scratch { .. } => true,
                    Placement::Resident { .. } => {
                        region.and_then(|region| region.resolve(target)) == Some(placement)
                    }
                }
        }
        _ => false,
    }
}

/// Writable scratch imported once through a lent route, kept mapped by this
/// process: every port stages into and reads from a fixed slot of it, with
/// no per-job import, export or page-table change.
///
/// Visibility is the provider's: an aliased writable import republishes each
/// bound range at submit and invalidates the written ranges at completion,
/// so CPU writes use fresh, cache-line-disjoint scratch, and CPU reads follow
/// the exact producing event's completion. Regions are not reused until all
/// admitted consumers retire, including across queued submissions.
pub struct ResidentScratch<'a> {
    buffer: u64,
    base: *mut u8,
    bytes: usize,
    _region: core::marker::PhantomData<&'a mut [u8]>,
}

impl<'a> ResidentScratch<'a> {
    /// # Safety
    /// `buffer` names offset zero of `[base, base + bytes)`, imported on this
    /// executor's face/context READ|WRITE (program input and output) through
    /// a route this process lent, so its own mapping stays valid. The
    /// mapping, import and supply live through quiescence or process
    /// containment. Nothing but the executor accesses the region.
    pub unsafe fn new(buffer: u64, base: *mut u8, bytes: usize) -> Option<Self> {
        if buffer == 0
            || base.is_null()
            || base as usize % PAGE != 0
            || bytes == 0
            || bytes % PAGE != 0
            || (base as usize).checked_add(bytes).is_none()
        {
            return None;
        }
        Some(Self {
            buffer,
            base,
            bytes,
            _region: core::marker::PhantomData,
        })
    }
}

/// The resident scratch carved into one fixed, page-aligned slot per
/// capacity, and which slot each port holds for the live job.
struct ResidentSlots<'a> {
    scratch: ResidentScratch<'a>,
    offsets: [u64; KERNEL_BUFFER_COUNT],
    taken: [bool; KERNEL_BUFFER_COUNT],
    port: [usize; KERNEL_BUFFER_COUNT],
}

impl<'a> ResidentSlots<'a> {
    fn carve(
        scratch: ResidentScratch<'a>,
        capacities: &[u64; KERNEL_BUFFER_COUNT],
    ) -> Option<Self> {
        let mut offsets = [0; KERNEL_BUFFER_COUNT];
        let mut end = 0u64;
        for (offset, &capacity) in offsets.iter_mut().zip(capacities) {
            *offset = end;
            end = end.checked_add(capacity)?;
        }
        (end <= scratch.bytes as u64).then_some(Self {
            scratch,
            offsets,
            taken: [false; KERNEL_BUFFER_COUNT],
            port: [0; KERNEL_BUFFER_COUNT],
        })
    }
}

/// The tail of the attached stage an ordered sequence table is written to:
/// its offset in the stage, and this process's writable view of it.
pub struct SequenceStage<'a> {
    offset: u64,
    base: *mut u8,
    bytes: usize,
    _region: core::marker::PhantomData<&'a mut [u8]>,
}

impl<'a> SequenceStage<'a> {
    /// # Safety
    /// `[base, base + bytes)` is this process's writable mapping of the
    /// attached stage's `[offset, offset + bytes)`, retained while the
    /// executor lives. Only the executor writes it, and only while no
    /// submission naming it is outstanding.
    pub unsafe fn new(offset: u64, base: *mut u8, bytes: usize) -> Option<Self> {
        (!base.is_null() && bytes != 0 && offset.checked_add(bytes as u64).is_some()).then_some(
            Self {
                offset,
                base,
                bytes,
                _region: core::marker::PhantomData,
            },
        )
    }
}

/// F32 ones at the top of batched scratch: the gate of the Mamba group RMS
/// for up to eight tokens (`8 * 4096` values), written once.
const ONES_BYTES: u64 = 8 * 4096 * 4;
/// The binding an unused wave chunk slot takes: a page of the ones run, a
/// different one per slot (the provider refuses overlapping bindings).
const UNUSED_CHUNK_BYTES: u64 = 4096;
/// The wave index of a routed-expert instruction the device routes itself
/// (`advance_moe_routed`): it is the instruction's only wave.
const ROUTED_WAVE: usize = usize::MAX;
/// The wave index of a resident KV append or attention read
/// (`advance_kv`): one member, or an attention read and its reduction.
const KV_WAVE: usize = usize::MAX - 1;
/// Causal prefix tokens per attention split: a workgroup walks at most this
/// many (about 0.5 ms on Xe2) before the reduction merges the splits, and
/// a decode row's few workgroups fill the device through its splits.
const KV_SPLIT_TOKENS: usize = 1024;
/// Splits an attention read takes at most.
const KV_MAX_SPLITS: usize = 32;
/// Scratch an attention read's partials may take: rows x splits x
/// `KV_PARTIAL_BYTES`. Many rows fill the device by themselves, so a
/// prefill chunk reads with few splits or one.
const KV_PARTIALS_BUDGET: usize = 8 << 20;
/// One row's partial for one split: its maximum, its sum and 128 columns.
const KV_PARTIAL_BYTES: usize = 130 * 4;

/// A resident KV cache the batched lane binds in place: the identity axiom
/// carries (`ResidentKvCache::allocation`, under the executor's owner) and
/// the imported buffer that holds its `bytes` from `offset`.
#[derive(Clone, Copy, Debug)]
pub struct ResidentKv {
    pub allocation: core::num::NonZeroU64,
    pub buffer: u64,
    pub offset: u64,
    pub bytes: u64,
    pub params: axnn::lowering::KvCacheParams,
}
const _: () = assert!(xe_moe::CHUNK_SLOTS as u64 * UNUSED_CHUNK_BYTES <= ONES_BYTES);

/// One kernel of a pre-lowered, pre-bound "frozen" decode sequence
/// ([`Executor::submit_frozen`]): the caller lowered it once and already
/// resolved every port's placement — the `Parameter` control block staged into
/// its own slot, the inputs and the written output into the resident scratch
/// (or an already-imported archive/KV buffer). The executor only encodes the
/// placements into the sequence table and drives the one ordered submission;
/// it allocates and stages nothing, so the frozen working buffer keeps the
/// offsets its producer fixed.
#[derive(Clone, Copy, Debug)]
pub struct FrozenLaunch {
    pub kernel: Kernel,
    /// Resolved placement for each of `kernel.ports()`, in slot order. Every
    /// port must already name an imported buffer: `Placement::Scratch` binds
    /// the resident scratch, `Placement::Resident` an archive, KV or
    /// recurrent-state import (a frozen decode's state slots, source and
    /// target, which it ping-pongs itself). The `Port::Parameter` slot must
    /// hold the kernel's control block. A `Port::Ones` resolves to the
    /// executor's ones run regardless of this entry.
    pub ports: [Placement; MAX_PORTS],
    /// This kernel reads a producer's write in the same submission, so the
    /// device must finish earlier writes first. The first kernel of a
    /// submission has no predecessor inside it; its flag is ignored.
    pub barrier: bool,
    /// Hold this kernel until a host word reaches a value (a gate,
    /// `res/QWEN_EXPERT_STREAMING.md` §3).
    pub gate: Option<FrozenSync>,
    /// Write a host word once this kernel retires (a signal).
    pub signal: Option<FrozenSync>,
}

/// A frozen launch's gate or signal word: where it lies (the resident scratch
/// or an import, like a port) and the value a gate waits for or a signal
/// writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrozenSync {
    pub word: Placement,
    pub value: u64,
}

/// The host half of a gated frozen sequence (`res/QWEN_EXPERT_STREAMING.md`
/// §3), lent to [`Executor::submit_frozen_gated`]: what the batch's gates
/// wait on. The batch's signals advance one word of a polled import; the
/// service answers each — the host's work for that layer, then the gate's
/// release — in order.
pub trait GateService {
    /// The low half of the word the batch's signals advance: the executor
    /// sleeps its core on this line while a gated window flies, and the
    /// batch's write to it is the wake.
    fn signal_word(&self) -> &core::sync::atomic::AtomicU32;
    /// Answer every signal that arrived, in order; whether any was. A fault
    /// (a storage read that failed) is the service's to record and the
    /// caller's to report once the run retires: the service still releases
    /// the gate, so the batch drains and the queue stays live. Async because
    /// the host's work may call a service (a storage ring's kick); the
    /// executor runs nothing else meanwhile.
    #[allow(async_fn_in_trait, reason = "one executor thread drives it; no Send bound to name")]
    async fn serve(&mut self) -> bool;
}

/// The gate service of a sequence with no gates.
enum NoGates {}

impl GateService for NoGates {
    fn signal_word(&self) -> &core::sync::atomic::AtomicU32 {
        match *self {}
    }

    async fn serve(&mut self) -> bool {
        match *self {}
    }
}

/// The longest turn a gated wait sleeps on the signal word: a bound on how
/// late it notices a completion notice (which writes elsewhere) and its
/// slice's end, never on a signal's wake, which is the write itself.
/// Doorbell parks take turns of the same length where no address monitor
/// exists, so there a signal waits up to one turn.
const GATE_TURN_NS: u64 = 50_000;

/// Whether two of a member's bindings would overlap: the Xe provider
/// refuses such a launch outright (read/read included), which would fail
/// the whole submission. `ones` is where the ones run sits in scratch.
fn placements_overlap(kernel: &Kernel, ports: &[Placement], ones: u64) -> bool {
    let range = |slot: usize| {
        let port = kernel.ports()[slot];
        let (buffer, offset) = match (port, ports[slot]) {
            (Port::Ones(_), _) => (None, ones),
            (_, Placement::Resident { buffer, offset }) => (Some(buffer), offset),
            (_, Placement::Scratch { offset }) => (None, offset),
        };
        (
            buffer,
            offset,
            offset.saturating_add(kernel.port_bytes(port)),
        )
    };
    let count = kernel.ports().len();
    (0..count).any(|a| {
        let (buffer_a, start_a, end_a) = range(a);
        (a + 1..count).any(|b| {
            let (buffer_b, start_b, end_b) = range(b);
            buffer_a == buffer_b && start_a < end_b && start_b < end_a
        })
    })
}

/// A batched executor's state: the admission window over resident scratch
/// (its arena below the ones run), the stage the sequence table goes to,
/// and a bounded set of independently retained submissions.
struct Batched<'a> {
    window: Window,
    scratch: ResidentScratch<'a>,
    ones: u64,
    stage: SequenceStage<'a>,
    flights: [Option<BatchFlight>; PIPELINE_DEPTH],
    flight_count: usize,
    /// A timed-out IPC retains correlation; resume its ledger before touching
    /// any other ledger, even when another event has completed meanwhile.
    request_owner: Option<usize>,
    /// Overlap the host with the device: a wait returns once its launched
    /// flight is on the device. Off by default: on Gigatron the extra
    /// flights, and CPU and GPU sharing the package's power limit, made it a
    /// net loss.
    overlap: bool,
    pipeline_depth: usize,
    /// Staged members one submission carries at most. Below the window it is
    /// a diagnostic: at 1 every member is its own job, so no dependency
    /// relies on an in-batch barrier.
    sequence_limit: usize,
}

struct BatchFlight {
    ledger: ExecutionLedger<core::iter::Once<OrderedExecution>>,
    members: usize,
    program: u64,
    submitted: Option<u64>,
    deadline: Option<u64>,
}

/// The loaded program of `artifact`, by package content
/// ([`XeKernelArtifact::same`](kore_xe_kernels::XeKernelArtifact::same)):
/// the app's copy and the executor's may sit at different addresses.
fn loaded_program(
    programs: &[LoadedKernel],
    artifact: &kore_xe_kernels::XeKernelArtifact,
) -> Option<u64> {
    programs
        .iter()
        .find(|p| p.artifact.same(*artifact))
        .map(|p| p.program)
}

/// Encode one frozen launch into its ordered-sequence member: every port
/// bound from the caller's resolved placement, against `scratch_buffer` (a
/// `Port::Ones` resolves to the ones run). The program is already resolved;
/// `barrier` is the member's ordering edge to the launches before it.
fn frozen_ordered_launch(
    scratch_buffer: u64,
    ones: u64,
    queue: u64,
    timeout_ns: u64,
    program: u64,
    launch: &FrozenLaunch,
    barrier: bool,
) -> Result<OrderedLaunch, BufferExecutionError> {
    let kernel = launch.kernel;
    let ports = launch.ports;
    let mut slot = 0usize;
    let bindings = kernel
        .port_bindings(|port| {
            let placement = ports[slot];
            slot += 1;
            match (port, placement) {
                (Port::Ones(_), _) => (scratch_buffer, ones),
                (_, Placement::Resident { buffer, offset }) => (buffer, offset),
                (_, Placement::Scratch { offset }) => (scratch_buffer, offset),
            }
        })
        .map_err(|_| BufferExecutionError::Refused(AccelRefusal::InvalidArgument))?;
    let word = |sync: Option<FrozenSync>| {
        sync.map(|sync| {
            let (buffer, offset) = match sync.word {
                Placement::Resident { buffer, offset } => (buffer, offset),
                Placement::Scratch { offset } => (scratch_buffer, offset),
            };
            kore_svc_accel::service::AccelSync {
                buffer,
                offset,
                value: sync.value,
            }
        })
    };
    Ok(OrderedLaunch {
        queue,
        program,
        bindings,
        timeout_ns,
        launch: kernel.launch(),
        barrier,
        gate: word(launch.gate),
        signal: word(launch.signal),
    })
}

fn shared_span(
    base: usize,
    bytes: usize,
    start: usize,
    logical: usize,
    required: usize,
) -> Option<(u64, u64, u64)> {
    if base % PAGE != 0 || logical == 0 || required < logical {
        return None;
    }
    let supply_end = base.checked_add(bytes)?;
    if start.checked_add(required)? > supply_end {
        return None;
    }
    let offset = start.checked_sub(base)?;
    let end = offset.checked_add(required)?;
    let rounded = end.checked_add(PAGE - 1)? / PAGE * PAGE;
    if rounded > bytes {
        return None;
    }
    let page = offset / PAGE * PAGE;
    Some((page as u64, (rounded - page) as u64, (offset - page) as u64))
}

struct Active {
    instruction: usize,
    kernel: Kernel,
    execution: BufferExecution<KERNEL_BUFFER_COUNT, KernelLaunch>,
    deadline: Option<u64>,
    returned: [Option<StashedMemory>; KERNEL_BUFFER_COUNT],
    output_bytes: u64,
    /// The loaded program, which keys its completion-time estimate.
    program: u64,
    /// When the device first owed this job's completion (the submit reply).
    submitted: Option<u64>,
}

/// Programs whose completion time is tracked, keyed by a flight's first
/// member: every projection kind at every width, the wave up kernels, the
/// conv and the scan can lead a flight. An unpaced program waits at the
/// floor and polls its job to completion.
const PACED_PROGRAMS: usize = 48;
/// The longest paced notice wait: a sanity bound, far above any projection.
const PACE_CEILING_NS: u64 = 250_000_000;

/// Notices a poll may find already delivered: one per completed job, so at
/// most the pipeline's flights, with room for a straggler from a retired one.
const STALE_NOTICES: usize = 2 * PIPELINE_DEPTH + 1;

/// Consume the completion notices already delivered, at most
/// [`STALE_NOTICES`]: they announce completions the poll about to run
/// observes itself. Left there, a stale notice ends the next wait at once
/// while the channel stays full, so the service drops the notice that wait
/// is for (184 of 858 on Panther Lake, 2026-10-03) and the job is seen only
/// when its paced slice runs out. The poll is the truth and a notice only a
/// wake: one that lands after this drain is fresh, and one that landed
/// before it announced a completion the poll sees. `false` reports a notice
/// that is not a completion's, a protocol violation the caller refuses.
fn drain_stale_notices(completion: &mut NoticeRx<'static, TrapTransport>) -> bool {
    for _ in 0..STALE_NOTICES {
        match completion.try_recv() {
            Ok(Some(notice)) if notice.cookie == kore_svc_accel::COOKIE_ACCEL_DONE => {}
            Ok(Some(_)) => return false,
            // Nothing pending (a post now stands) or a transient refusal:
            // the wait that follows deals with either.
            Ok(None) | Err(_) => break,
        }
    }
    // A drain that consumed its whole bound left the post down across the
    // poll that follows; stand it, as `wait_notice_inner` does.
    let _ = completion.restand();
    true
}

/// Per-program estimate of submit-to-completion time: an exponential moving
/// average (weight 1/4 on each sample). The first notice wait of a job sleeps
/// about three quarters of it, so a job the service cannot announce is polled
/// a few times near its end instead of every slice from its start; a
/// delivered notice still ends any wait at once.
#[derive(Clone, Copy)]
struct Pace {
    programs: [(u64, u64); PACED_PROGRAMS],
}

impl Default for Pace {
    fn default() -> Self {
        Self {
            programs: [(0, 0); PACED_PROGRAMS],
        }
    }
}

impl Pace {
    fn expected(&self, program: u64) -> Option<u64> {
        self.programs
            .iter()
            .find(|(id, _)| *id == program && program != 0)
            .map(|(_, ns)| *ns)
    }

    fn record(&mut self, program: u64, ns: u64) {
        if program == 0 {
            return;
        }
        if let Some(entry) = self.programs.iter_mut().find(|(id, _)| *id == program) {
            entry.1 = entry.1 - entry.1 / 4 + ns / 4;
        } else if let Some(entry) = self.programs.iter_mut().find(|(id, _)| *id == 0) {
            *entry = (program, ns);
        }
    }

    /// The notice wait for a job `elapsed` into its run: the remainder of
    /// its expected time, then a quarter of it per retry, never below
    /// `floor`. A delivered notice ends any wait at once, and the Xe row
    /// sends one per job (`bd6d9e0c5` dropped 15 of 2,350), so a timeout
    /// only bounds how late a job whose notice was lost is observed. Each
    /// timed-out wait costs a poll: three quarters and then the floor made
    /// 3.9 waits and 5.9 calls a job.
    fn slice(&self, program: u64, elapsed: u64, floor: u64, ceiling: u64) -> u64 {
        self.expected(program)
            .map(|expected| expected.saturating_sub(elapsed).max(expected / 4))
            .unwrap_or(0)
            .clamp(floor, ceiling.max(floor))
    }
}

/// Last local operation; byte counts and binding indices carry no authority.
#[derive(Clone, Copy, Debug)]
pub enum Activity {
    Idle,
    Prepare,
    ScratchSelect { port: usize, required: u64 },
    ScratchMap { port: usize, capacity: u64 },
    ScratchRetire { port: usize },
    RecoverMap { bytes: u64 },
    RecoverRetire,
    Wire(BufferExecutionActivity),
}

#[derive(Clone, Copy, Debug)]
struct Attempt {
    instruction: usize,
    program: u64,
    inputs: [u64; 3],
    output: u64,
}

/// A binding the lane refused before submitting it: past the end of the
/// buffer it names, which the service would refuse as `OutOfBounds` without
/// saying which member or slot.
#[derive(Clone, Copy, Debug)]
struct BindingFault {
    member: MemberKey,
    slot: usize,
    buffer: u64,
    offset: u64,
    len: u64,
    limit: u64,
}

/// Copy-only execution observation. This does not carry buffer access,
/// retirement authority, or a guarantee that an in-flight call has completed.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    pub completed: u64,
    pub instruction: Option<usize>,
    pub activity: Activity,
    pub pending: bool,
    pub profile: Option<Profile>,
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            completed: 0,
            instruction: None,
            activity: Activity::Idle,
            pending: false,
            profile: None,
        }
    }
}

/// One serial prepared-instruction lane. Scratch allocations are returned to
/// this owner after each event, so repeated graphs do not carve more heap.
/// Weight routes are affine READ children of the caller's retained supply.
/// Drop does not imply device cancellation; call `cancel_step` to drain an
/// abandoned invocation before reusing its face or releasing its parent.
pub struct Executor<'a, 'w, 't> {
    client: &'a mut RingAccel<'w, 't, TrapTransport>,
    completion: &'a mut NoticeRx<'static, TrapTransport>,
    context: u64,
    queue: u64,
    programs: &'a [LoadedKernel],
    inputs: SharedInputs<'a>,
    recurrent_state: Option<ResidentState<'a>>,
    scratch: [Option<StashedMemory>; KERNEL_BUFFER_COUNT],
    capacities: [u64; KERNEL_BUFFER_COUNT],
    // Placement follows each scratch allocation through binding permutations.
    // Exact view retirement removes PTEs; the kernel MAP cursor never rewinds.
    scratch_windows: [Option<kore_abi::task::layout::Window>; KERNEL_BUFFER_COUNT],
    /// Resident scratch, when the caller imported one: the leases above are
    /// then absent and every port stages into a fixed slot of it.
    resident: Option<ResidentSlots<'a>>,
    /// Batched execution, when constructed with `new_batched`: the per-port
    /// single-job paths then decline everything.
    batch: Option<Batched<'a>>,
    /// The resident KV caches this lane may bind in place, under the
    /// owner epoch axiom's identities carry; `None` declines every
    /// resident KV operation.
    kv: Option<(core::num::NonZeroU64, &'a [ResidentKv])>,
    /// Where the executor publishes its activity for a watchdog on another
    /// thread ([`Self::with_activity_observer`]): see [`observed_activity`].
    observer: Option<&'a core::sync::atomic::AtomicU64>,
    pace: Pace,
    policy: WaitPolicy,
    active: Option<Active>,
    failure: Option<BufferExecutionError>,
    completed: u64,
    waves_completed: u64,
    wave_routes_completed: u64,
    profile: Option<Profile>,
    attempt: Option<Attempt>,
    /// The binding a submission was refused for before it left the host.
    binding_fault: Option<BindingFault>,
    activity: Activity,
    progress: Option<&'a Cell<Progress>>,
    /// Decline every dense projection (see [`Self::decline_dense`]).
    dense_declined: bool,
    /// Fail the invocation instead of letting a member the lane would not
    /// place run on the CPU (see [`Self::with_strict_placement`]).
    strict: bool,
}

impl<'a, 'w, 't> Executor<'a, 'w, 't> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &'a mut RingAccel<'w, 't, TrapTransport>,
        completion: &'a mut NoticeRx<'static, TrapTransport>,
        context: u64,
        queue: u64,
        programs: &'a [LoadedKernel],
        inputs: SharedInputs<'a>,
        scratch: [StashedLease; KERNEL_BUFFER_COUNT],
        capacities: [u64; KERNEL_BUFFER_COUNT],
        policy: WaitPolicy,
    ) -> Result<Self, AccelRefusal> {
        Self::build(
            client,
            completion,
            context,
            queue,
            programs,
            inputs,
            scratch.map(|lease| Some(lease.into())),
            None,
            capacities,
            policy,
        )
    }

    /// An executor over resident scratch: one slot per capacity, carved in
    /// order at page-aligned offsets, staged into and read in place.
    #[allow(clippy::too_many_arguments)]
    pub fn new_resident(
        client: &'a mut RingAccel<'w, 't, TrapTransport>,
        completion: &'a mut NoticeRx<'static, TrapTransport>,
        context: u64,
        queue: u64,
        programs: &'a [LoadedKernel],
        inputs: SharedInputs<'a>,
        scratch: ResidentScratch<'a>,
        capacities: [u64; KERNEL_BUFFER_COUNT],
        policy: WaitPolicy,
    ) -> Result<Self, AccelRefusal> {
        let slots =
            ResidentSlots::carve(scratch, &capacities).ok_or(AccelRefusal::InvalidArgument)?;
        Self::build(
            client,
            completion,
            context,
            queue,
            programs,
            inputs,
            core::array::from_fn(|_| None),
            Some(slots),
            capacities,
            policy,
        )
    }

    /// An executor that admits a window of prepared instructions (and a wave
    /// of expert batches) and submits each window as one ordered sequence
    /// (`crate::xe_batch`). All of `scratch` but its top `ONES_BYTES` is the
    /// window's arena; the top holds F32 ones, written once here. Sequence
    /// tables are written to `stage`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_batched(
        client: &'a mut RingAccel<'w, 't, TrapTransport>,
        completion: &'a mut NoticeRx<'static, TrapTransport>,
        context: u64,
        queue: u64,
        programs: &'a [LoadedKernel],
        inputs: SharedInputs<'a>,
        scratch: ResidentScratch<'a>,
        stage: SequenceStage<'a>,
        policy: WaitPolicy,
    ) -> Result<Self, AccelRefusal> {
        let ones = (scratch.bytes as u64)
            .checked_sub(ONES_BYTES)
            .filter(|&ones| ones >= ONES_BYTES)
            .ok_or(AccelRefusal::InvalidArgument)?;
        // SAFETY: `ResidentScratch::new`'s contract makes this executor the
        // region's sole accessor, no event is live yet, and the ones run lies
        // wholly inside it (checked above).
        unsafe {
            let ones = scratch.base.add(ones as usize).cast::<f32>();
            for index in 0..(ONES_BYTES / 4) as usize {
                ones.add(index).write(1.0);
            }
        }
        let mut executor = Self::build(
            client,
            completion,
            context,
            queue,
            programs,
            inputs,
            core::array::from_fn(|_| None),
            None,
            [PAGE as u64; KERNEL_BUFFER_COUNT],
            policy,
        )?;
        executor.batch = Some(Batched {
            window: Window::new(ones),
            scratch,
            ones,
            stage,
            flights: core::array::from_fn(|_| None),
            flight_count: 0,
            request_owner: None,
            overlap: false,
            pipeline_depth: PIPELINE_DEPTH,
            sequence_limit: WINDOW,
        });
        Ok(executor)
    }

    /// Bound queued jobs. One preserves the serialized comparison; the
    /// default ping-pong depth permits the next ready job before retirement.
    pub fn with_pipeline_depth(mut self, depth: usize) -> Self {
        if let Some(batch) = self.batch.as_mut() {
            batch.pipeline_depth = depth.clamp(1, PIPELINE_DEPTH);
        }
        self
    }

    /// Carry at most `limit` members (1 ..= the window) per submission (see
    /// `Batched::sequence_limit`).
    pub fn with_sequence_limit(mut self, limit: usize) -> Self {
        if let Some(batch) = self.batch.as_mut() {
            batch.sequence_limit = limit.clamp(1, WINDOW);
        }
        self
    }

    /// Let the batched lane overlap the host with the device (see
    /// `Batched::overlap`).
    pub fn with_overlap(mut self, overlap: bool) -> Self {
        if let Some(batch) = self.batch.as_mut() {
            batch.overlap = overlap;
        }
        self
    }

    /// Use independently retained decoder storage for recurrent outputs.
    /// Targets outside it or without cache-line isolation keep copy-back.
    pub fn with_recurrent_state(mut self, state: ResidentState<'a>) -> Self {
        self.recurrent_state = Some(state);
        self
    }

    /// Strict placement: a projection, recurrence or expert wave whose
    /// package is loaded but which this lane would not place on its final
    /// offer (nothing admitted ahead of it, so the engine would run it on
    /// the CPU) fails the invocation with the reason
    /// (`BufferExecutionError::Declined`), instead of a silent fallback that
    /// keeps the chat running at a fraction of its speed. Small ops offered
    /// alone, ops without a package and unloaded packages still run natively
    /// by design.
    pub fn with_strict_placement(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Bind the resident KV caches `caches` in place for the appends and
    /// attention reads of a decoder built over them
    /// (`axmodel::Decoder::new_with_resident_kv_in`): each identified to
    /// axiom by `owner` and its `allocation`. Each cache's buffer must stay
    /// imported, read-write, for the executor's life.
    /// Publish the executor's activity to `observer` on every change, packed
    /// by [`observed_activity`]: a watchdog on another thread reads it when
    /// this thread has not moved for a while, so a stall names the lane's
    /// state (inside which call, after how many kernels, on which
    /// instruction) rather than nothing.
    pub fn with_activity_observer(mut self, observer: &'a core::sync::atomic::AtomicU64) -> Self {
        self.observer = Some(observer);
        self.publish_activity(false, 0);
        self
    }

    fn set_activity(&mut self, activity: Activity) {
        self.activity = activity;
        self.publish_activity(false, 0);
    }

    /// Store the packed word; `inside` with `request` (1 submit, 2 poll, 3
    /// retire) brackets a call into the service.
    fn publish_activity(&self, inside: bool, request: u8) {
        let instruction = self
            .active
            .as_ref()
            .map(|active| active.instruction)
            .or_else(|| self.batch.as_ref().and_then(|batch| Self::window_instruction(&batch.window)));
        publish_activity(self.observer, inside, request, self.activity, self.completed, instruction);
    }

    /// The instruction of the window's last member, for the observer.
    fn window_instruction(window: &Window) -> Option<usize> {
        window.members().last().map(|member| match member.key {
            MemberKey::Instruction(i) | MemberKey::Wave { instruction: i, .. } => i,
        })
    }

    pub fn with_resident_kv(mut self, owner: core::num::NonZeroU64, caches: &'a [ResidentKv]) -> Self {
        self.kv = Some((owner, caches));
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        client: &'a mut RingAccel<'w, 't, TrapTransport>,
        completion: &'a mut NoticeRx<'static, TrapTransport>,
        context: u64,
        queue: u64,
        programs: &'a [LoadedKernel],
        inputs: SharedInputs<'a>,
        scratch: [Option<StashedMemory>; KERNEL_BUFFER_COUNT],
        resident: Option<ResidentSlots<'a>>,
        capacities: [u64; KERNEL_BUFFER_COUNT],
        policy: WaitPolicy,
    ) -> Result<Self, AccelRefusal> {
        if context == 0
            || queue == 0
            || programs.iter().any(|p| p.program == 0)
            || capacities
                .iter()
                .any(|&bytes| bytes == 0 || bytes % PAGE as u64 != 0)
            || policy.job_ns == 0
            || policy.call_ns == 0
            || policy.completion_ns == 0
            || policy.operation_ns < policy.job_ns
        {
            return Err(AccelRefusal::InvalidArgument);
        }
        Ok(Self {
            client,
            completion,
            context,
            queue,
            programs,
            inputs,
            recurrent_state: None,
            scratch,
            capacities,
            scratch_windows: [None; KERNEL_BUFFER_COUNT],
            resident,
            batch: None,
            pace: Pace::default(),
            policy,
            active: None,
            failure: None,
            completed: 0,
            waves_completed: 0,
            wave_routes_completed: 0,
            profile: None,
            attempt: None,
            binding_fault: None,
            activity: Activity::Idle,
            progress: None,
            dense_declined: false,
            strict: false,
            kv: None,
            observer: None,
        })
    }

    pub fn failure(&self) -> Option<&BufferExecutionError> {
        self.failure.as_ref()
    }
    /// Observe this lane from the caller's cooperative task without borrowing
    /// its active graph future or installing another timer. Like NPU draft
    /// profiling, the caller owns the observation cell for the lane's lifetime.
    pub fn observe_progress(&mut self, progress: &'a Cell<Progress>) {
        self.progress = Some(progress);
        self.publish_progress();
    }

    fn progress_snapshot(&self) -> Progress {
        Progress {
            completed: self.completed,
            instruction: self.attempt.map(|attempt| attempt.instruction),
            activity: self.activity,
            pending: self.active.is_some()
                || self.batch.as_ref().is_some_and(|b| b.flight_count != 0),
            profile: self.profile,
        }
    }

    fn publish_progress(&self) {
        if let Some(progress) = self.progress {
            progress.set(self.progress_snapshot());
        }
    }

    /// Capture before cancellation drains the active ledger. No mapped
    /// addresses, tensor contents or capability grants appear in this report.
    pub fn write_failure_report(&self, out: &mut impl core::fmt::Write) {
        let _ = writeln!(
            out,
            "GPU attempt: {:?}; local: {:?}",
            self.attempt, self.activity
        );
        let refusal = self
            .active
            .as_ref()
            .and_then(|a| a.execution.failure_activity());
        let _ = writeln!(out, "GPU refused activity: {refusal:?}");
        let available = self.scratch.each_ref().map(Option::is_some);
        let _ = writeln!(
            out,
            "GPU scratch capacities: {:?}; locally held: {available:?}",
            self.capacities
        );
        if let Some(fault) = self.binding_fault {
            let _ = writeln!(out, "GPU binding past its buffer: {fault:?}");
        }
        let _ = write!(out, "GPU resident buffers (handle/bytes): ");
        self.inputs.describe(out);
        let _ = writeln!(out);
        // The window as the failure found it: every member with its state,
        // launch and bindings (buffer handles and offsets, no addresses), so
        // a refusal the service does not explain can be read off the host.
        if let Some(batch) = self.batch.as_ref() {
            let _ = writeln!(
                out,
                "GPU window: arena {} of {} bytes; scratch buffer {}; members (state key program launch; slot:buffer+offset/len):",
                batch.window.arena.used(),
                batch.ones,
                batch.scratch.buffer,
            );
            for member in batch.window.members() {
                let _ = write!(
                    out,
                    "  {:?} {:?} p{} {:?}",
                    member.state,
                    member.key,
                    member.program,
                    member.kernel.launch()
                );
                let mut place = xe_batch::place(member, batch.scratch.buffer, batch.ones);
                for (slot, &port) in member.kernel.ports().iter().enumerate() {
                    let (buffer, offset) = place(port);
                    let _ = write!(
                        out,
                        " {slot}:{buffer}+{offset}/{}",
                        member.kernel.port_bytes(port)
                    );
                }
                let _ = writeln!(out);
            }
        }
    }
    /// Enable observations from this point. Repeated calls preserve counters.
    pub fn enable_profiling(&mut self) {
        self.profile.get_or_insert_with(Profile::default);
    }
    pub fn profile(&self) -> Option<Profile> {
        self.profile.map(|profile| Profile {
            replies: self.client.client.reply_waits(),
            ..profile
        })
    }
    fn profile_start(&self) -> Option<u64> {
        self.profile
            .as_ref()
            .and_then(|_| kore_rt::clock::uptime().ok().map(|t| t.as_nanos()))
    }
    fn profile_record(&mut self, start: Option<u64>, sample: fn(&mut Profile) -> &mut WallTime) {
        if let Some(profile) = self.profile.as_mut() {
            sample(profile).record(start, kore_rt::clock::uptime().ok().map(|t| t.as_nanos()));
        }
        self.publish_progress();
    }
    /// One execute call, in `execute` and in the request kind's own sample.
    fn profile_record_execute(
        &mut self,
        start: Option<u64>,
        request: fn(&mut Profile) -> &mut WallTime,
    ) {
        if let Some(profile) = self.profile.as_mut() {
            let end = kore_rt::clock::uptime().ok().map(|t| t.as_nanos());
            profile.execute.record(start, end);
            request(profile).record(start, end);
        }
        self.publish_progress();
    }
    pub fn completed(&self) -> u64 {
        self.completed
    }
    /// Retired MoE expert waves; their launches are in `completed`.
    pub fn waves_completed(&self) -> u64 {
        self.waves_completed
    }
    /// Expert routes (token, expert pairs) in retired waves.
    pub fn wave_routes_completed(&self) -> u64 {
        self.wave_routes_completed
    }
    pub fn progress(&self) -> Option<ExecutionProgress> {
        self.active
            .as_ref()
            .map(|active| active.execution.progress())
    }
    pub fn quiescent(&self) -> bool {
        self.active.is_none()
            && self
                .batch
                .as_ref()
                .is_none_or(|b| b.flight_count == 0 && b.window.is_empty())
    }

    /// Enable a fresh per-turn device measurement, or freeze its accumulated
    /// report. Only an idle execution lane may change instrumentation. The
    /// bounded control call retains its exact correlation on cancellation;
    /// retrying the same request resumes it rather than resetting twice.
    pub async fn configure_profiling(&mut self, enabled: bool) -> Result<(), BufferExecutionError> {
        if !self.quiescent() {
            return Err(Self::local(AccelRefusal::Busy));
        }
        if !enabled {
            // A cancelled enable still owns its exact request. Settle only
            // that request before disabling; never publish a second reset or
            // consume another operation's reply. A pending disable is left
            // intact by Outstanding and resumed below.
            let enable = AccelRequest::ConfigureProfiling {
                ctx: self.context,
                enabled: true,
            };
            match self
                .client
                .client
                .resume_exact_bounded(&enable, self.policy.call_ns)
                .await
            {
                Ok(reply) => self.profiling_reply(true, reply)?,
                Err(CallError::NoOutstanding | CallError::Outstanding) => {}
                Err(error) => {
                    return Err(BufferExecutionError::Execution(
                        ExecutionError::Infrastructure(error),
                    ));
                }
            }
        }
        let request = AccelRequest::ConfigureProfiling {
            ctx: self.context,
            enabled,
        };
        let client = &mut self.client.client;
        let reply = match client
            .resume_exact_bounded(&request, self.policy.call_ns)
            .await
        {
            Err(CallError::NoOutstanding) => {
                client.call_bounded(&request, self.policy.call_ns).await
            }
            reply => reply,
        }
        .map_err(|error| BufferExecutionError::Execution(ExecutionError::Infrastructure(error)))?;
        self.profiling_reply(enabled, reply)
    }

    fn profiling_reply(
        &mut self,
        enabled: bool,
        reply: kore_rt::svc::CallReply<AccelResponse>,
    ) -> Result<(), BufferExecutionError> {
        if reply.grants != 0 {
            self.client.client.poison_retained(reply.retained);
            return Err(Self::local(AccelRefusal::Internal));
        }
        match reply.response {
            AccelResponse::ProfilingConfigured {
                ctx,
                enabled: actual,
            } if ctx == self.context && actual == enabled => Ok(()),
            AccelResponse::Refused(refusal) => Err(Self::local(refusal)),
            _ => Err(Self::local(AccelRefusal::Internal)),
        }
    }

    /// The Xe row's telemetry text (clock, completions, completion
    /// wakes), read only while no job is live so it never displaces the
    /// ledger's own calls. Returns the bytes written; zero when busy.
    pub async fn read_telemetry(&mut self, out: &mut [u8]) -> usize {
        if !self.quiescent() {
            return 0;
        }
        self.client
            .read_telemetry_report(out, self.policy.call_ns)
            .await
    }

    /// A status-tool read with a per-page observation bound and Stop check.
    pub async fn read_telemetry_checked(
        &mut self,
        out: &mut [u8],
        budget_ns: u64,
        should_stop: impl FnMut() -> bool,
    ) -> (usize, bool) {
        if !self.quiescent() {
            return (0, false);
        }
        self.client
            .read_telemetry_report_checked(out, budget_ns, should_stop)
            .await
    }

    /// Whether a bounded telemetry read left a control reply outstanding.
    /// A caller must not resume generation on a late status reply.
    pub fn control_idle(&self) -> bool {
        self.client.client.is_idle()
    }

    /// Return scratch custody only after the retained ledger is fully drained.
    /// An unretired owner is returned intact for continued cancellation.
    pub fn release(self) -> Result<[Option<StashedMemory>; KERNEL_BUFFER_COUNT], Self> {
        if !self.quiescent() {
            return Err(self);
        }
        Ok(self.scratch)
    }

    fn fail(&mut self, error: BufferExecutionError) -> ax_exec::Error {
        self.failure = Some(error);
        ax_exec::Error::InvocationFinished
    }

    fn local(refusal: AccelRefusal) -> BufferExecutionError {
        BufferExecutionError::Refused(refusal)
    }

    fn observe(
        &mut self,
        instruction: usize,
        output: &mut [f32],
    ) -> Result<Option<KernelProgress>, ax_exec::Error> {
        if let Some(active) = &self.active {
            if active.instruction != instruction {
                return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
            }
            return match active.execution.progress() {
                ExecutionProgress::Finished(ExecutionOutcome::Retired(
                    AccelEventState::Complete,
                )) => {
                    self.recover(Some(output)).map_err(|e| self.fail(e))?;
                    self.completed += 1;
                    self.set_activity(Activity::Idle);
                    self.publish_progress();
                    Ok(Some(KernelProgress::Complete))
                }
                ExecutionProgress::Pending => Ok(Some(KernelProgress::Pending)),
                ExecutionProgress::Finished(ExecutionOutcome::Refused(refusal))
                | ExecutionProgress::Finished(ExecutionOutcome::Retired(
                    AccelEventState::Failed(refusal),
                )) => Err(self.fail(Self::local(refusal))),
                ExecutionProgress::Finished(
                    ExecutionOutcome::NotSubmitted
                    | ExecutionOutcome::Retired(AccelEventState::Cancelled),
                ) => Err(ax_exec::Error::InvocationFinished),
                ExecutionProgress::Finished(ExecutionOutcome::Retired(
                    AccelEventState::Pending,
                )) => Err(self.fail(Self::local(AccelRefusal::Internal))),
            };
        }
        Ok(None)
    }

    fn program(&self, kernel: &Kernel) -> Option<u64> {
        if self.dense_declined && kernel.is_dense() {
            return None;
        }
        loaded_program(self.programs, &kernel.artifact())
    }

    /// The bytes of a loaded program's package: an upper bound on the ISA
    /// the sequence encoder places in the provider's command window.
    /// What the provider stages for `program` in a sequence: one package
    /// of its family ([`XeKernelArtifact::staged_bytes`]).
    fn package_bytes(&self, program: u64) -> u64 {
        self.programs
            .iter()
            .find(|p| p.program == program)
            .map_or(0, |p| p.artifact.staged_bytes() as u64)
    }

    /// Leave every dense projection on the CPU from now on: their loaded
    /// programs read as absent, so each offer declines. For a dense self-test
    /// that failed on this device.
    pub fn decline_dense(&mut self) {
        self.dense_declined = true;
    }

    fn can_prepare(&self, kernel: &Kernel, inputs: &[Option<NativeRef<'_>>; 3]) -> bool {
        let mut available = core::array::from_fn(|index| {
            self.resident
                .as_ref()
                .map_or(self.scratch[index].is_some(), |slots| !slots.taken[index])
        });
        for (index, &required) in kernel.input_bytes().iter().enumerate() {
            let Some(input) = inputs[index] else {
                return false;
            };
            if self
                .inputs
                .resolve(input.as_bytes(), required as usize)
                .is_none()
            {
                let Some(slot) = scratch_slot(&self.capacities, &available, required) else {
                    return false;
                };
                available[slot] = false;
            }
        }
        for bytes in [kernel.control_bytes().len() as u64, kernel.output_bytes()] {
            let Some(slot) = scratch_slot(&self.capacities, &available, bytes) else {
                return false;
            };
            available[slot] = false;
        }
        true
    }

    /// Reuse only this executor's reserved placement. Callers close the exact
    /// affine view before transferring its lease, and remap only after export
    /// returns custody. The kernel refuses any still-occupied range.
    fn map_scratch(
        memory: &StashedMemory,
        window: &mut Option<kore_abi::task::layout::Window>,
    ) -> Result<kore_rt::cap::MappedLeaseView, BufferExecutionError> {
        let lease = memory
            .slot()
            .ok_or_else(|| Self::local(AccelRefusal::InvalidArgument))?;
        let view = match *window {
            Some(span) => verbs::map_owned_lease_at_rw(lease, span.base(), span.len()),
            None => verbs::map_owned_lease_rw(lease),
        }
        .ok_or_else(|| Self::local(AccelRefusal::ResourceLimit))?;
        if window.is_none() {
            *window = Some(kore_abi::task::layout::Window::new(
                view.as_ptr() as u64,
                view.len() as u64,
            ));
        }
        Ok(view)
    }

    /// Stage port `index` and name its buffer and binding offset.
    fn prepare_scratch(
        &mut self,
        index: usize,
        contents: &[u8],
        required: u64,
        usage: BufferUsage,
    ) -> Result<(ExecutionBuffer, u64), BufferExecutionError> {
        // Reuse the smallest sufficient allocation. Resident weight ports
        // leave their scratch available, so a vocabulary-sized output never
        // forces every small projection to map, clear and flush that extent.
        self.set_activity(Activity::ScratchSelect {
            port: index,
            required,
        });
        if let Some(slots) = self.resident.as_mut() {
            let available = core::array::from_fn(|slot| !slots.taken[slot]);
            let slot = scratch_slot(&self.capacities, &available, required)
                .ok_or_else(|| Self::local(AccelRefusal::ResourceLimit))?;
            slots.taken[slot] = true;
            slots.port[index] = slot;
            let offset = slots.offsets[slot];
            let required = usize::try_from(required)
                .map_err(|_| Self::local(AccelRefusal::InvalidArgument))?;
            // SAFETY: `carve` proved every slot lies inside the region, and
            // `required` fits its slot. No event is live while a job is
            // prepared, so the device touches no scratch byte, and this
            // executor is the region's sole CPU accessor.
            let bytes = unsafe {
                core::slice::from_raw_parts_mut(slots.scratch.base.add(offset as usize), required)
            };
            if contents.len() > required {
                return Err(Self::local(AccelRefusal::InvalidArgument));
            }
            bytes[..contents.len()].copy_from_slice(contents);
            // Only what the kernel reads is cleared: the bound range, never
            // the slot's whole capacity. An output needs none; the kernel
            // writes every byte recovery reads back.
            if !usage.contains(BufferUsage::PROGRAM_OUTPUT) {
                bytes[contents.len()..].fill(0);
            }
            return Ok((ExecutionBuffer::Resident(slots.scratch.buffer), offset));
        }
        let available = core::array::from_fn(|index| self.scratch[index].is_some());
        let slot = scratch_slot(&self.capacities, &available, required)
            .ok_or_else(|| Self::local(AccelRefusal::ResourceLimit))?;
        self.scratch.swap(index, slot);
        self.capacities.swap(index, slot);
        self.scratch_windows.swap(index, slot);
        let memory = self.scratch[index]
            .take()
            .ok_or_else(|| Self::local(AccelRefusal::InvalidArgument))?;
        self.set_activity(Activity::ScratchMap {
            port: index,
            capacity: self.capacities[index],
        });
        let mut view = Self::map_scratch(&memory, &mut self.scratch_windows[index])?;
        let capacity = usize::try_from(self.capacities[index])
            .map_err(|_| Self::local(AccelRefusal::InvalidArgument))?;
        if view.len() < capacity || required > capacity as u64 {
            return Err(Self::local(AccelRefusal::InvalidArgument));
        }
        // SAFETY: the exclusively owned scratch lease has not been transferred.
        // No graph pointer escapes this copy. Zero padding covers packed words.
        unsafe {
            let bytes = core::slice::from_raw_parts_mut(view.as_mut_ptr(), capacity);
            if contents.len() as u64 > required {
                return Err(Self::local(AccelRefusal::InvalidArgument));
            }
            bytes[contents.len()..].fill(0);
            bytes[..contents.len()].copy_from_slice(contents);
        }
        self.set_activity(Activity::ScratchRetire { port: index });
        if !view.try_retire_outcome().is_closed() {
            return Err(Self::local(AccelRefusal::Busy));
        }
        Ok((
            ExecutionBuffer::Transfer {
                target: ImportTarget {
                    context: self.context,
                    bytes: self.capacities[index],
                    alignment: PAGE as u64,
                    usage: usage.bits(),
                },
                source: memory,
            },
            0,
        ))
    }

    fn prepare(
        &mut self,
        instruction: usize,
        kernel: Kernel,
        program: u64,
        inputs: &[Option<NativeRef<'_>>; 3],
    ) -> Result<(), BufferExecutionError> {
        let start = self.profile_start();
        self.attempt = Some(Attempt {
            instruction,
            program,
            inputs: core::array::from_fn(|i| kernel.input_bytes().get(i).copied().unwrap_or(0)),
            output: kernel.output_bytes(),
        });
        self.set_activity(Activity::Prepare);
        let result = self.prepare_inner(instruction, kernel, program, inputs);
        self.profile_record(start, |p| &mut p.prepare);
        result
    }

    fn prepare_inner(
        &mut self,
        instruction: usize,
        kernel: Kernel,
        program: u64,
        inputs: &[Option<NativeRef<'_>>; 3],
    ) -> Result<(), BufferExecutionError> {
        let output_bytes = kernel.output_bytes();
        let count = kernel.input_bytes().len();
        let mut buffers = core::array::from_fn(|_| ExecutionBuffer::Unused);
        let mut offsets = [0; KERNEL_BUFFER_COUNT];
        // Resident reads bound so far. The provider refuses two bindings of
        // one launch over the same bytes (reads included), so a read that
        // repeats one (x * x) is staged instead.
        let mut bound: [Option<(u64, u64, u64)>; 3] = [None; 3];
        for index in 0..count {
            let bytes = inputs[index]
                .ok_or_else(|| Self::local(AccelRefusal::InvalidArgument))?
                .as_bytes();
            let required = usize::try_from(kernel.input_bytes()[index])
                .map_err(|_| Self::local(AccelRefusal::InvalidArgument))?;
            let resolved =
                self.inputs
                    .resolve(bytes, required)
                    .filter(|&(backing, (offset, _, interior))| {
                        let InputBacking::Resident(buffer) = backing else {
                            return true;
                        };
                        let start = offset + interior;
                        let end = start + required as u64;
                        !bound
                            .iter()
                            .flatten()
                            .any(|&(other, a, b)| other == buffer && a < end && start < b)
                    });
            buffers[index] = if let Some((backing, (offset, length, interior))) = resolved {
                match backing {
                    InputBacking::Local => return Err(Self::local(AccelRefusal::InvalidArgument)),
                    InputBacking::Resident(buffer) => {
                        offsets[index] = offset + interior;
                        bound[index] =
                            Some((buffer, offsets[index], offsets[index] + required as u64));
                        ExecutionBuffer::Resident(buffer)
                    }
                    InputBacking::Route(parent) => {
                        let route = parent
                            .scope(&FabricRouteScopeReq::new(
                                offset,
                                length,
                                FabricAccess::READ,
                            ))
                            .map_err(|_| Self::local(AccelRefusal::Denied))?;
                        offsets[index] = interior;
                        ExecutionBuffer::Transfer {
                            target: ImportTarget {
                                context: self.context,
                                bytes: length,
                                alignment: PAGE as u64,
                                usage: BufferUsage::PROGRAM_INPUT.bits(),
                            },
                            source: StashedMemory::Route(route),
                        }
                    }
                }
            } else {
                let (buffer, offset) = self.prepare_scratch(
                    index,
                    bytes,
                    kernel.input_bytes()[index],
                    BufferUsage::PROGRAM_INPUT,
                )?;
                offsets[index] = offset;
                buffer
            };
        }
        let (parameter, offset) = self.prepare_scratch(
            count,
            kernel.control_bytes(),
            kernel.control_bytes().len() as u64,
            BufferUsage::PROGRAM_INPUT,
        )?;
        buffers[count] = parameter;
        offsets[count] = offset;
        let (output, offset) = self.prepare_scratch(
            count + 1,
            &[],
            kernel.output_bytes(),
            BufferUsage::PROGRAM_OUTPUT,
        )?;
        buffers[count + 1] = output;
        offsets[count + 1] = offset;
        let bind = kernel
            .launch_builder(self.queue, program, offsets, self.policy.job_ns)
            .map_err(|_| Self::local(AccelRefusal::InvalidArgument))?;
        let execution = self
            .client
            .buffer_execution(buffers, bind)
            .map_err(|_| Self::local(AccelRefusal::Busy))?;
        self.active = Some(Active {
            instruction,
            kernel,
            execution,
            deadline: None,
            returned: core::array::from_fn(|_| None),
            output_bytes,
            program,
            submitted: None,
        });
        Ok(())
    }

    fn recover(&mut self, output: Option<&mut [f32]>) -> Result<(), BufferExecutionError> {
        let start = self.profile_start();
        let result = self.recover_inner(output);
        self.profile_record(start, |p| &mut p.recover);
        result
    }

    fn recover_inner(
        &mut self,
        mut output: Option<&mut [f32]>,
    ) -> Result<(), BufferExecutionError> {
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| Self::local(AccelRefusal::InvalidArgument))?;
        let last = active.kernel.input_bytes().len() + 1;
        for index in 0..=last {
            if active.returned[index].is_none() {
                active.returned[index] = active.execution.take_buffer(index);
            }
            if let Some(memory) = active.returned[index].as_mut() {
                if index == last {
                    if let Some(output) = output.as_deref_mut() {
                        {
                            self.activity = Activity::RecoverMap {
                            bytes: active.output_bytes,
                        };
                            publish_activity(self.observer, false, 0, self.activity, self.completed, Some(active.instruction));
                        }
                        let mut view = Self::map_scratch(memory, &mut self.scratch_windows[index])?;
                        let bytes = core::mem::size_of_val(output);
                        if bytes != active.output_bytes as usize
                            || view.len() < active.kernel.output_bytes() as usize
                        {
                            return Err(Self::local(AccelRefusal::InvalidArgument));
                        }
                        // SAFETY: retirement/export returned sole scratch custody;
                        // the graph output is a distinct call-scoped allocation.
                        unsafe {
                            core::ptr::copy_nonoverlapping(
                                view.as_mut_ptr().cast::<f32>(),
                                output.as_mut_ptr(),
                                output.len(),
                            );
                        }
                        if let Some(profile) = self.profile.as_mut() {
                            profile.copied_output_bytes =
                                profile.copied_output_bytes.saturating_add(bytes as u64);
                        }
                        {
                            self.activity = Activity::RecoverRetire;
                            publish_activity(self.observer, false, 0, self.activity, self.completed, Some(active.instruction));
                        }
                        if !view.try_retire_outcome().is_closed() {
                            return Err(Self::local(AccelRefusal::Busy));
                        }
                    }
                }
                match memory {
                    StashedMemory::Lease(_) => self.scratch[index] = active.returned[index].take(),
                    StashedMemory::Route(_) => {
                        if !memory.try_retire_outcome().is_closed() {
                            return Err(Self::local(AccelRefusal::Busy));
                        }
                        active.returned[index] = None;
                    }
                }
            }
        }
        if let Some(slots) = self.resident.as_mut() {
            if let Some(output) = output.as_deref_mut() {
                {
                            self.activity = Activity::RecoverMap {
                    bytes: active.output_bytes,
                };
                            publish_activity(self.observer, false, 0, self.activity, self.completed, Some(active.instruction));
                        }
                let slot = slots.port[last];
                if core::mem::size_of_val(output) != active.output_bytes as usize
                    || active.output_bytes > self.capacities[slot]
                {
                    return Err(Self::local(AccelRefusal::InvalidArgument));
                }
                // SAFETY: the event retired and the provider completed the
                // device-to-CPU visibility handoff, so these loads observe the
                // device's result; the slot holds `output_bytes` in bounds and
                // the graph output is a distinct call-scoped allocation.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        slots
                            .scratch
                            .base
                            .add(slots.offsets[slot] as usize)
                            .cast::<f32>(),
                        output.as_mut_ptr(),
                        output.len(),
                    );
                }
                if let Some(profile) = self.profile.as_mut() {
                    profile.copied_output_bytes = profile
                        .copied_output_bytes
                        .saturating_add(active.output_bytes);
                }
            }
            slots.taken = [false; KERNEL_BUFFER_COUNT];
        }
        self.active = None;
        Ok(())
    }

    /// Advance the same retained ledger after an error/cancelled graph. A false
    /// result leaves all ownership live; the caller bounds its cleanup phase
    /// and quarantines parents if it cannot obtain true.
    pub async fn cancel_step(&mut self) -> Result<bool, BufferExecutionError> {
        if let Some(batch) = self.batch.as_mut() {
            for flight in batch.flights.iter_mut().flatten() {
                flight.ledger.request_cancel();
            }
            if batch.flight_count != 0 {
                let index = batch.request_owner.unwrap_or(0);
                batch.request_owner = Some(index);
                let flight = batch.flights[index].as_mut().expect("retained flight");
                let result = flight
                    .ledger
                    .advance_bounded(self.client, self.policy.call_ns)
                    .await;
                batch.request_owner = (!self.client.execution_request_idle()).then_some(index);
                match result.map_err(BufferExecutionError::Execution)? {
                    ExecutionProgress::Finished(_) => {
                        batch.flights[index] = None;
                        batch.flights[index..].rotate_left(1);
                        batch.flight_count -= 1;
                    }
                    ExecutionProgress::Pending => {
                        if flight.ledger.awaiting_device_completion()
                            && batch.request_owner.is_none()
                        {
                            self.wait_notice(self.policy.completion_ns).await?;
                        }
                        return Ok(false);
                    }
                }
                if batch.flight_count != 0 {
                    return Ok(false);
                }
            }
            // Every ledger retired (or never submitted). Only now may shared
            // scratch and staged metadata be reclaimed together.
            batch.window.complete_submitted();
            batch.window.discard_staged();
            batch.request_owner = None;
            return Ok(true);
        }
        let Some(active) = self.active.as_mut() else {
            return Ok(true);
        };
        active.execution.request_cancel();
        match active
            .execution
            .advance_bounded(self.client, self.policy.call_ns)
            .await?
        {
            ExecutionProgress::Finished(_) => {
                self.recover(None)?;
                Ok(true)
            }
            ExecutionProgress::Pending => {
                if active.execution.awaiting_device_completion() {
                    self.wait_notice(self.policy.completion_ns).await?;
                }
                Ok(false)
            }
        }
    }

    async fn wait_notice(&mut self, slice_ns: u64) -> Result<(), BufferExecutionError> {
        let start = self.profile_start();
        let result = self.wait_notice_inner(slice_ns).await;
        self.profile_record(start, |p| &mut p.notice);
        result
    }

    async fn wait_notice_inner(&mut self, slice_ns: u64) -> Result<(), BufferExecutionError> {
        let deadline = clock_ns().await.saturating_add(slice_ns);
        let notice = loop {
            match self.completion.try_recv() {
                Ok(Some(notice)) => break Some(notice),
                Ok(None) => {
                    let Some(left) = deadline
                        .checked_sub(clock_ns().await)
                        .filter(|&left| left > 0)
                    else {
                        break None;
                    };
                    let post = self
                        .completion
                        .standing_post()
                        .ok_or_else(|| Self::local(AccelRefusal::InvalidArgument))?;
                    match next_wake_on_for(post, left).await {
                        WakeReason::Delivery => {
                            break self
                                .completion
                                .settle_delivery()
                                .map_err(Self::notice_error)?;
                        }
                        WakeReason::Deadline => break None,
                        // Another source woke the executor: most often the
                        // drain a fused service call schedules when it re-posts
                        // the shared doorbell. That is not this flight's
                        // notice. Ending the wait on it made every wait return
                        // at once, and the executor polled the service ~40
                        // times a window (Panther Lake, 2026-10-04), keeping
                        // the CPU busy and taking package power from the GPU.
                        // Park again for what is left of the slice.
                        WakeReason::Doorbell | WakeReason::Irq | WakeReason::Word => {}
                    }
                }
                Err(RecvError::Refused(Status::WouldBlock)) => {
                    kore_rt::sleep_for(slice_ns).await;
                    break None;
                }
                Err(error) => return Err(Self::notice_error(error)),
            }
        };
        match notice {
            Some(notice) => self.took_notice(notice),
            None => Ok(()),
        }
    }

    /// A completion notice ended a wait: the poll that follows is the truth.
    fn took_notice(&mut self, notice: kore_abi::session::Notice) -> Result<(), BufferExecutionError> {
        if notice.cookie != kore_svc_accel::COOKIE_ACCEL_DONE {
            return Err(Self::local(AccelRefusal::InvalidArgument));
        }
        // The poll that follows is a service round trip, which is exactly
        // when the next flight's completion is announced. With the post
        // down the service's send gets `WouldBlock` and the notice is
        // dropped for good (165 of 2360 jobs on Panther Lake, 2026-10-04),
        // so that job is seen only when its paced wait runs out. This
        // channel carries completion notices alone. A refused re-post
        // leaves the next wait to retry.
        let _ = self.completion.restand();
        if let Some(profile) = self.profile.as_mut() {
            profile.notices = profile.notices.saturating_add(1);
        }
        Ok(())
    }

    /// Answer the gates' signals that arrived, timing the calls that
    /// answered one.
    async fn serve_gates<G: GateService>(&mut self, gates: &mut G) -> bool {
        let start = self.profile_start();
        let served = gates.serve().await;
        if served {
            self.profile_record(start, |p| &mut p.gate);
            if let Some(profile) = self.profile.as_mut() {
                profile.gate_serves = profile.gate_serves.saturating_add(1);
            }
        }
        served
    }

    /// [`Self::wait_notice`] for a gated sequence: the device may stop at a
    /// gate while the window flies, so the wait answers its signals as they
    /// arrive. The core sleeps on the signal word's line (`UMWAIT`), which
    /// the batch's write to it wakes, so a gate waits on the host's work
    /// alone; between turns a completion notice ends the wait as it ends the
    /// plain one, and the slice bounds it by the live counter (the
    /// executor's own clock refreshes only on its parks). Without an address
    /// monitor or a readable counter each turn is a doorbell park instead:
    /// a notice still ends it at once, a signal waits up to one turn.
    async fn wait_gated<G: GateService>(&mut self, slice_ns: u64, gates: &mut G) -> Result<(), BufferExecutionError> {
        let start = self.profile_start();
        let result = self.wait_gated_inner(slice_ns, gates).await;
        self.profile_record(start, |p| &mut p.notice);
        result
    }

    async fn wait_gated_inner<G: GateService>(
        &mut self,
        slice_ns: u64,
        gates: &mut G,
    ) -> Result<(), BufferExecutionError> {
        let counter = || kore_rt::clock::uptime().ok().map(|time| time.as_nanos());
        let deadline = counter()
            .filter(|_| kore_rt::hint::available())
            .map(|now| now.saturating_add(slice_ns));
        // The parked form's bound: turns of a slice.
        let mut turns = slice_ns.div_ceil(GATE_TURN_NS).max(1);
        loop {
            // Read before serving, so a signal that lands while the service
            // works moves the word past `seen` and the sleep below returns.
            let seen = gates.signal_word().load(core::sync::atomic::Ordering::Acquire);
            if self.serve_gates(gates).await {
                continue;
            }
            match self.completion.try_recv() {
                Ok(Some(notice)) => return self.took_notice(notice),
                // A post stands, or could not stand now: the turn bounds it.
                Ok(None) | Err(RecvError::Refused(Status::WouldBlock)) => {}
                Err(error) => return Err(Self::notice_error(error)),
            }
            match deadline {
                Some(deadline) => {
                    let Some(left) = counter()
                        .and_then(|now| deadline.checked_sub(now))
                        .filter(|&left| left > 0)
                    else {
                        return Ok(());
                    };
                    let turn = core::time::Duration::from_nanos(left.min(GATE_TURN_NS));
                    kore_rt::hint::wait_on(gates.signal_word(), seen, turn);
                    if let Some(profile) = self.profile.as_mut() {
                        profile.monitor_turns = profile.monitor_turns.saturating_add(1);
                    }
                }
                None => {
                    if turns == 0 {
                        return Ok(());
                    }
                    turns -= 1;
                    let Some(post) = self.completion.standing_post() else {
                        kore_rt::sleep_for(GATE_TURN_NS).await;
                        continue;
                    };
                    if let WakeReason::Delivery = next_wake_on_for(post, GATE_TURN_NS).await
                        && let Some(notice) = self.completion.settle_delivery().map_err(Self::notice_error)?
                    {
                        return self.took_notice(notice);
                    }
                }
            }
        }
    }

    fn notice_error(error: RecvError) -> BufferExecutionError {
        BufferExecutionError::Execution(ExecutionError::Infrastructure(match error {
            RecvError::Refused(status) => CallError::Refused(status),
            RecvError::Transport => CallError::Transport,
        }))
    }
}

// The batched lane (`new_batched`): admission, retirement, submission.
impl Executor<'_, '_, '_> {
    /// Place `bytes` and its immutable control `trailer`, zero-padded to
    /// `required`, in fresh arena scratch. Most operands have no trailer.
    fn batched_stage(
        batch: &mut Batched<'_>,
        bytes: &[u8],
        trailer: &[u8],
        required: u64,
    ) -> Option<u64> {
        let written = bytes.len().checked_add(trailer.len())?;
        if written as u64 > required {
            return None;
        }
        let offset = batch.window.arena.allocate(required)?;
        // SAFETY: the arena spans `[0, ones)` of the scratch mapping, and this
        // region is fresh: no admitted member's port names it, so neither the
        // device nor the host reads it until the member that owns it is
        // submitted. Regions never share a cache line (64-byte placement), so
        // this store cannot dirty a line a live submission writes.
        unsafe {
            let destination = batch.scratch.base.add(offset as usize);
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
            core::ptr::copy_nonoverlapping(
                trailer.as_ptr(),
                destination.add(bytes.len()),
                trailer.len(),
            );
            core::ptr::write_bytes(destination.add(written), 0, required as usize - written);
        }
        Some(offset)
    }

    /// Where a member's input operand lives, and whether its producer is
    /// still staged (the member then needs a barrier): a pending producer's
    /// output in place, the resident import host bytes lie in, or a fresh
    /// staged copy of them. `None` when the arena cannot place it.
    fn operand_placement(
        batch: &mut Batched<'_>,
        shared: &SharedInputs<'_>,
        input: ChainedInput<'_>,
        required: u64,
    ) -> Option<(Placement, bool)> {
        Some(match input {
            ChainedInput::Pending {
                instruction,
                offset,
                elements,
            } => batch.window.chained(instruction, offset, elements).ok()?,
            ChainedInput::Host(value) => {
                let bytes = value.as_bytes();
                match shared.resolve(bytes, usize::try_from(required).ok()?) {
                    Some((InputBacking::Resident(buffer), (offset, _, interior))) => (
                        Placement::Resident {
                            buffer,
                            offset: offset + interior,
                        },
                        false,
                    ),
                    _ => (
                        Placement::Scratch {
                            offset: Self::batched_stage(batch, bytes, &[], required)?,
                        },
                        false,
                    ),
                }
            }
        })
    }

    /// Answer a retry of `key`: Pending until its submission completed, then
    /// its output copied to `output` and the member retired (Complete).
    /// `None` when `key` is not admitted.
    fn batched_retire(
        &mut self,
        key: MemberKey,
        output: &mut [f32],
    ) -> Result<Option<KernelProgress>, ax_exec::Error> {
        self.batched_retire_into(key, output, None)
    }

    /// [`Self::batched_retire`], also publishing recurrent successor states.
    /// Retained targets are already final; scratch targets need copy-back.
    fn batched_retire_into(
        &mut self,
        key: MemberKey,
        output: &mut [f32],
        target: Option<axnn_cpu::StateTargetRef<'_>>,
    ) -> Result<Option<KernelProgress>, ax_exec::Error> {
        let batch = self.batch.as_mut().expect("batched lane");
        let Some((index, member)) = batch.window.find(key) else {
            return Ok(None);
        };
        let member = *member;
        if member.state != MemberState::Done {
            return Ok(Some(KernelProgress::Pending));
        }
        let target_fits =
            recurrent_target_matches(member.recurrent, target, self.recurrent_state.as_ref());
        if index != 0
            || core::mem::size_of_val(output) as u64 != member.output_bytes
            || !target_fits
        {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        // SAFETY: the member's submission retired and the provider completed
        // the device-to-CPU visibility handoff, so these loads observe the
        // device's result. Scratch regions hold `output_bytes` and any staged
        // states; graph output and target are distinct exclusive resources.
        // A direct target is already written/acquired in place; no CPU
        // reference to its contents was created on a pending retry.
        unsafe {
            core::ptr::copy_nonoverlapping(
                batch.scratch.base.add(member.output as usize).cast::<f32>(),
                output.as_mut_ptr(),
                output.len(),
            );
            if let (Some((Placement::Scratch { offset }, _)), Some(target)) =
                (member.recurrent, target)
            {
                // Observe only this CPU copy, after completion/acquisition and
                // the ordinary output copy. Direct targets do not enter here,
                // so they record no copy calls rather than zero-duration ones.
                let start = self
                    .profile
                    .as_ref()
                    .and_then(|_| kore_rt::clock::uptime().ok().map(|time| time.as_nanos()));
                core::ptr::copy_nonoverlapping(
                    batch.scratch.base.add(offset as usize).cast::<f32>(),
                    target.as_mut_ptr(),
                    target.len(),
                );
                if let Some(profile) = self.profile.as_mut() {
                    let end = kore_rt::clock::uptime().ok().map(|time| time.as_nanos());
                    profile.state_copy.record(start, end);
                }
            }
        }
        batch.window.retire_front(key);
        if let Some(profile) = self.profile.as_mut() {
            profile.copied_output_bytes = profile
                .copied_output_bytes
                .saturating_add(member.output_bytes);
            match member.recurrent {
                Some((Placement::Scratch { .. }, bytes)) => {
                    profile.copied_state_bytes = profile.copied_state_bytes.saturating_add(bytes);
                }
                Some((Placement::Resident { .. }, bytes)) => {
                    profile.direct_state_bytes = profile.direct_state_bytes.saturating_add(bytes);
                }
                None => {}
            }
        }
        self.completed += 1;
        self.set_activity(Activity::Idle);
        self.publish_progress();
        Ok(Some(KernelProgress::Complete))
    }

    /// Admit a pure instruction into the window, or decline it (nothing
    /// staged, nothing changed) so the engine drains or runs it natively.
    fn batched_chained(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        let key = MemberKey::Instruction(instruction);
        if let Some(progress) = self.batched_retire(key, output)? {
            return Ok(progress);
        }
        // A norm, gather or element-wise map costs the CPU microseconds and
        // a submission of its own a full device round trip, so it joins the
        // device only as a follower: beside members already in the window,
        // or reading a producer that is still pending. A graph whose
        // projections are not loaded (a one-token prompt tail, a model this
        // catalog does not know) then keeps its small ops on the CPU.
        let projection = matches!(
            plan.semantic,
            axnn::lowering::SemanticOp::MatMulTransposed
                | axnn::lowering::SemanticOp::MatMulNvfp4Transposed
        );
        let follows = self
            .batch
            .as_ref()
            .is_some_and(|batch| batch.window.members().next().is_some())
            || inputs
                .iter()
                .flatten()
                .any(|input| matches!(input, ChainedInput::Pending { .. }));
        if !projection && !follows {
            return self.member_declined(MemberDecline::Solo, false);
        }
        // The engine offers again after a drain; a decline with nothing
        // admitted ahead is final, and the instruction then runs on the CPU.
        let strict_final = self.strict && !follows;
        let Some(kernel) = Kernel::from_plan(plan, minimum) else {
            return self.member_declined(MemberDecline::Package, false);
        };
        if !kernel.inputs_aligned(inputs.iter().map(|input| match input {
            Some(ChainedInput::Host(value)) => Some(value.as_bytes()),
            _ => None,
        })) {
            return self.member_declined(MemberDecline::Operands, strict_final);
        }
        let Some(program) = self.program(&kernel) else {
            return self.member_declined(MemberDecline::Program, false);
        };
        if core::mem::size_of_val(output) as u64 != kernel.output_bytes() {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        self.batched_operands_fit(&kernel, plan, inputs)?;
        self.batched_admit(key, kernel, program, inputs, None, None, strict_final)
    }

    /// Count a declined member and decline it: nothing was staged. Under
    /// strict placement a final decline the lane could have avoided
    /// (`MemberDecline::placement`) fails the invocation instead.
    fn member_declined(
        &mut self,
        why: MemberDecline,
        strict_final: bool,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if let Some(profile) = self.profile.as_mut() {
            profile.member_declined[why.index()] += 1;
        }
        if strict_final && why.placement() {
            return Err(self.fail(BufferExecutionError::Declined(MEMBER_DECLINES[why.index()])));
        }
        Ok(KernelProgress::Unsupported)
    }

    /// Admit a recurrent instruction (a Mamba convolution or scan) into the
    /// window with its state, or decline it: the state before the run is
    /// staged beside the operands, including the package's control trailer.
    /// Successor states bind independently retained decoder storage when
    /// available, otherwise scratch that retirement copies to the target.
    fn batched_recurrent(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        state: ax_exec::RecurrentState<'_>,
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        let key = MemberKey::Instruction(instruction);
        let ax_exec::RecurrentState {
            source,
            target,
            tree,
        } = state;
        if let Some(progress) = self.batched_retire_into(key, output, Some(target))? {
            return Ok(progress);
        }
        // A recurrence is worth a launch of its own (its CPU kernel costs
        // milliseconds a call), so unlike the small ops it is admitted even
        // with nothing to follow. A decline with nothing admitted ahead of
        // it is final: the engine then runs it on the CPU.
        let strict_final = self.strict
            && !self
                .batch
                .as_ref()
                .is_some_and(|batch| batch.window.members().next().is_some())
            && !inputs
                .iter()
                .flatten()
                .any(|input| matches!(input, ChainedInput::Pending { .. }));
        // The target holds one state slot, or one per token of the run.
        let slots = match (source.len(), target.len()) {
            (0, _) | (_, 0) => return self.member_declined(MemberDecline::Package, false),
            (state, states) if states.is_multiple_of(state) => states / state,
            _ => return Err(self.fail(Self::local(AccelRefusal::InvalidArgument))),
        };
        let Some(kernel) = Kernel::from_plan(plan, minimum)
            .and_then(|kernel| kernel.with_state_slots(u32::try_from(slots).ok()?))
            .and_then(|kernel| match tree {
                Some(tree) => kernel.with_token_tree(tree),
                None => Some(kernel),
            })
        else {
            return self.member_declined(MemberDecline::Package, false);
        };
        if !kernel.inputs_aligned(inputs.iter().map(|input| match input {
            Some(ChainedInput::Host(value)) => Some(value.as_bytes()),
            _ => None,
        })) {
            return self.member_declined(MemberDecline::Operands, strict_final);
        }
        let Some(program) = self.program(&kernel) else {
            return self.member_declined(MemberDecline::Program, false);
        };
        let Some((source_bytes, target_bytes)) = kernel.state_bytes() else {
            return self.member_declined(MemberDecline::Package, false);
        };
        if core::mem::size_of_val(output) as u64 != kernel.output_bytes()
            || core::mem::size_of_val(source) as u64 != source_bytes
            || target
                .len()
                .checked_mul(core::mem::size_of::<f32>())
                .map(|n| n as u64)
                != Some(target_bytes)
        {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        self.batched_operands_fit(&kernel, plan, inputs)?;
        let placement = self
            .recurrent_state
            .as_ref()
            .and_then(|state| state.resolve(target));
        let source = NativeRef::F32(source);
        self.batched_admit(
            key,
            kernel,
            program,
            inputs,
            None,
            Some((source.as_bytes(), placement)),
            strict_final,
        )
    }

    /// Place `kernel`'s ports for a new member `key` and admit it: pending
    /// operands bind their producer's output region, host operands a
    /// resident import in place or fresh arena scratch, the accumulator a
    /// fresh region filled with `base`, a recurrent package's state before
    /// the run a fresh region filled with `state` and its successor states
    /// a retained target or fresh region read back at retirement. Declines leave the window
    /// and the arena as they were, and count by reason: a full window is
    /// retried after a drain, while a placement the arena refuses on a
    /// final offer (`strict_final`) fails under strict placement.
    #[allow(clippy::too_many_arguments)]
    fn batched_admit(
        &mut self,
        key: MemberKey,
        kernel: Kernel,
        program: u64,
        inputs: &[Option<ChainedInput<'_>>; 3],
        base: Option<&[u8]>,
        state: Option<(&[u8], Option<Placement>)>,
        strict_final: bool,
    ) -> Result<KernelProgress, ax_exec::Error> {
        let start = self.profile_start();
        let package_bytes = self.package_bytes(program);
        let shared = &self.inputs;
        let batch = self.batch.as_mut().expect("batched lane");
        // No slot, or the member's package would take the sequence past the
        // provider's command window: a drain, then the offer again.
        if batch.window.is_full() || !batch.window.fits(program, package_bytes) {
            return self.member_declined(MemberDecline::Window, false);
        }
        let mark = batch.window.arena.mark();
        let mut placed =
            || -> Option<([Placement; MAX_PORTS], bool, u64, Option<(Placement, u64)>)> {
                let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
                let (mut reads_staged, mut output_offset, mut recurrent) = (false, None, None);
                for (slot, &port) in kernel.ports().iter().enumerate() {
                    let required = kernel.port_bytes(port);
                    ports[slot] = match port {
                        Port::Input(index) => {
                            let input = inputs[usize::from(index)]?;
                            let (placement, staged) =
                                Self::operand_placement(batch, shared, input, required)?;
                            reads_staged |= staged;
                            placement
                        }
                        Port::Parameter => Placement::Scratch {
                            offset: Self::batched_stage(batch, kernel.control_bytes(), &[], required)?,
                        },
                        Port::Ones(bytes) => {
                            if bytes > ONES_BYTES {
                                return None;
                            }
                            Placement::Scratch { offset: batch.ones }
                        }
                        Port::Output => {
                            let offset = batch.window.arena.allocate(required)?;
                            output_offset = Some(offset);
                            Placement::Scratch { offset }
                        }
                        Port::Accumulator => {
                            let base = base.filter(|base| base.len() as u64 == required)?;
                            let offset = Self::batched_stage(batch, base, &[], required)?;
                            output_offset = Some(offset);
                            Placement::Scratch { offset }
                        }
                        Port::StateSource(bytes) => {
                            let (state, _) =
                                state.filter(|(state, _)| state.len() as u64 == bytes)?;
                            Placement::Scratch {
                                offset: Self::batched_stage(
                                    batch,
                                    state,
                                    kernel.control_bytes(),
                                    required,
                                )?,
                            }
                        }
                        // The decoder's own storage when the offer names an
                        // exact target there, else fresh scratch the host
                        // copies the successor states out of.
                        Port::StateTarget(_) => {
                            let placement = match state?.1 {
                                Some(placement) => placement,
                                None => Placement::Scratch {
                                    offset: batch.window.arena.allocate(required)?,
                                },
                            };
                            recurrent = Some((placement, required));
                            placement
                        }
                        // Only a MoE wave binds chunks and a region
                        // (`batched_wave`), only a KV member a cache and
                        // partials (`batched_kv`).
                        Port::Chunk { .. }
                        | Port::Wave(_)
                        | Port::KvCache { .. }
                        | Port::Partials(_)
                        | Port::Discard(_) => return None,
                    };
                }
                Some((ports, reads_staged, output_offset?, recurrent))
            };
        let Some((ports, reads_staged, output_offset, recurrent)) = placed() else {
            batch.window.arena.rewind(mark);
            return self.member_declined(MemberDecline::Scratch, strict_final);
        };
        // Two operands naming the same bytes (x * x, or one resident view
        // read twice) would make the provider refuse the whole sequence.
        if placements_overlap(&kernel, &ports, batch.ones) {
            batch.window.arena.rewind(mark);
            return self.member_declined(MemberDecline::Scratch, strict_final);
        }
        let member = Member {
            key,
            kernel,
            program,
            ports,
            output: output_offset,
            output_bytes: kernel.output_bytes(),
            recurrent,
            barrier: false,
            state: MemberState::Staged,
        };
        if batch
            .window
            .admit(member, reads_staged, package_bytes)
            .is_err()
        {
            batch.window.arena.rewind(mark);
            return self.member_declined(MemberDecline::Window, false);
        }
        let instruction = match key {
            MemberKey::Instruction(instruction) | MemberKey::Wave { instruction, .. } => {
                instruction
            }
        };
        self.attempt = Some(Attempt {
            instruction,
            program,
            inputs: core::array::from_fn(|i| kernel.input_bytes().get(i).copied().unwrap_or(0)),
            output: kernel.output_bytes(),
        });
        self.set_activity(Activity::Prepare);
        self.profile_record(start, |p| &mut p.prepare);
        Ok(KernelProgress::Pending)
    }

    /// Count a declined wave (profiling only) and decline it: nothing was
    /// staged, and the wave runs on the CPU.
    fn wave_declined(&mut self, why: WaveDecline) -> Result<KernelProgress, ax_exec::Error> {
        if let Some(profile) = self.profile.as_mut() {
            profile.wave_declined[why.index()] += 1;
        }
        // Every wave decline is final (the instruction's remaining waves run
        // on the CPU). Under strict placement, one the lane could have
        // avoided fails the invocation; an unsealed package or a fidelity
        // above the packages' is by design.
        if self.strict && !matches!(why, WaveDecline::Program | WaveDecline::Accuracy) {
            return Err(self.fail(BufferExecutionError::Declined(WAVE_DECLINES[why.index()])));
        }
        Ok(KernelProgress::Unsupported)
    }

    /// A routed-expert instruction the lane will not plan on the device:
    /// counted, and never final, even under strict placement. The engine
    /// then routes the layer on the host and offers its waves through
    /// `advance_moe_experts`, where a decline is the CPU fallback that
    /// strict placement refuses. A streaming expert cache (Gigatron, or a
    /// launch that could not hold every block) declines here on every layer.
    fn routed_declined(&mut self, why: WaveDecline) -> Result<KernelProgress, ax_exec::Error> {
        if let Some(profile) = self.profile.as_mut() {
            profile.routed_declined[why.index()] += 1;
        }
        Ok(KernelProgress::Unsupported)
    }

    /// Answer a retry of wave `(instruction, wave)`: Pending until its
    /// submission completed, then optionally publish OUT and retire the wave's
    /// members, oldest first. Omit publication only when Axiom has admitted all
    /// remaining output consumers on the device. `None` means not admitted.
    fn wave_retire(
        &mut self,
        instruction: usize,
        wave: usize,
        routes: usize,
        output: Option<&mut [f32]>,
    ) -> Result<Option<KernelProgress>, ax_exec::Error> {
        let batch = self.batch.as_mut().expect("batched lane");
        let ours = |key: MemberKey| matches!(key, MemberKey::Wave { instruction: i, wave: w, .. } if (i, w) == (instruction, wave));
        let (mut parts, mut done, mut out) = (0u8, true, None);
        for member in batch.window.members().filter(|m| ours(m.key)) {
            parts += 1;
            done &= member.state == MemberState::Done;
            if member.output_bytes != 0 {
                out = Some((member.output, member.output_bytes));
            }
        }
        if parts == 0 {
            return Ok(None);
        }
        if !done {
            return Ok(Some(KernelProgress::Pending));
        }
        let front = batch.window.members().next().map(|m| m.key);
        let Some((offset, out_bytes)) = out.filter(|&(_, bytes)| {
            output
                .as_ref()
                .is_none_or(|out| bytes == size_of_val(&**out) as u64)
        }) else {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        };
        if front
            != Some(MemberKey::Wave {
                instruction,
                wave,
                part: 0,
            })
        {
            return Err(self.fail(Self::local(AccelRefusal::Internal)));
        }
        // SAFETY: the wave's submission retired and the provider completed
        // the device-to-CPU visibility handoff, so these loads observe the
        // combine's result.
        // OUT lies in the arena and holds `bytes`; the graph output is a
        // distinct call-scoped allocation.
        if let Some(output) = output {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    batch.scratch.base.add(offset as usize).cast::<f32>(),
                    output.as_mut_ptr(),
                    output.len(),
                );
            }
            if let Some(profile) = self.profile.as_mut() {
                profile.copied_output_bytes = profile.copied_output_bytes.saturating_add(out_bytes);
            }
        } else if let Some(profile) = self.profile.as_mut() {
            profile.avoided_output_bytes = profile.avoided_output_bytes.saturating_add(out_bytes);
        }
        for part in 0..parts {
            batch.window.retire_front(MemberKey::Wave {
                instruction,
                wave,
                part,
            });
        }
        self.completed += u64::from(parts);
        self.waves_completed += 1;
        self.wave_routes_completed += routes as u64;
        self.set_activity(Activity::Idle);
        self.publish_progress();
        Ok(Some(KernelProgress::Complete))
    }

    /// Admit one expert wave: its region (table, hidden rows, the output so
    /// far) staged into scratch, each chunk set's grouped up and down
    /// launches over the expert-cache imports its blocks live in, then one
    /// combine. Declines leave nothing staged.
    #[allow(clippy::too_many_arguments)]
    fn batched_wave(
        &mut self,
        instruction: usize,
        wave: usize,
        hidden: &[f32],
        routes: &[axnn_cpu::MoeBatchRow],
        experts: &[axnn_cpu::ExpertWeights<'_>],
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if let Some(progress) = self.wave_retire(instruction, wave, routes.len(), Some(output))? {
            return Ok(progress);
        }
        if minimum != KernelAccuracy::Approximate {
            return self.wave_declined(WaveDecline::Accuracy);
        }
        let programs = MoeWave::ALL.map(|part| {
            kore_xe_kernels::moe_wave(part)
                .and_then(|artifact| loaded_program(self.programs, &artifact))
        });
        let [Some(up), Some(down), Some(combine)] = programs else {
            return self.wave_declined(WaveDecline::Program);
        };
        let width = xe_moe::MOE_HIDDEN;
        if !hidden.len().is_multiple_of(width) || output.len() != hidden.len() {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        let tokens = hidden.len() / width;
        let start = self.profile_start();
        let shared = &self.inputs;
        let plan = match WavePlan::new(routes, experts, tokens, |address| {
            shared.resident_region(address)
        }) {
            Ok(plan) => plan,
            Err(why) => return self.wave_declined(WaveDecline::Plan(why)),
        };
        // Every launch first, so admission cannot stop halfway: each set's
        // up, each set's down, the combine.
        let sets = plan.sets();
        let mut launches = [None; 2 * xe_moe::MAX_SETS + 1];
        for (index, launch) in launches[..2 * sets + 1].iter_mut().enumerate() {
            let (part, program, set) = match index {
                i if i < sets => (MoeWave::Up, up, i),
                i if i < 2 * sets => (MoeWave::Down, down, i - sets),
                _ => (MoeWave::Combine, combine, 0),
            };
            let rows = if part == MoeWave::Combine {
                tokens
            } else {
                plan.tiles(set)
            };
            let header = WavePlan::header_offset(set);
            let chunks = plan.chunks(set);
            let Some(kernel) = Kernel::moe_wave(
                part,
                rows,
                plan.bytes() - header,
                chunks.map(|chunk| chunk.map_or(UNUSED_CHUNK_BYTES, |(_, bytes)| bytes)),
            ) else {
                return self.wave_declined(WaveDecline::Kernel);
            };
            *launch = Some((kernel, program, header, chunks, part));
        }
        // The wave's packages (up, down and combine, however many sets) and
        // its members against the provider's command window.
        let package_bytes =
            [up, down, combine].map(|program| (program, self.package_bytes(program)));
        let bytes_of = |program: u64| {
            package_bytes
                .iter()
                .find(|(p, _)| *p == program)
                .map_or(0, |(_, bytes)| *bytes)
        };
        let mut wave_members = [(0u64, 0u64); 2 * xe_moe::MAX_SETS + 1];
        for (entry, launch) in wave_members.iter_mut().zip(launches[..2 * sets + 1].iter()) {
            let program = launch.expect("built above").1;
            *entry = (program, bytes_of(program));
        }
        let batch = self.batch.as_mut().expect("batched lane");
        if batch.window.len() + 2 * sets + 1 > WINDOW
            || !batch.window.fits_all(&wave_members[..2 * sets + 1])
        {
            return self.wave_declined(WaveDecline::Window);
        }
        let mark = batch.window.arena.mark();
        let Some(region) = batch.window.arena.allocate(plan.bytes()) else {
            return self.wave_declined(WaveDecline::Scratch);
        };
        // SAFETY: as in `batched_stage`: a fresh arena region no admitted
        // member names, 64-byte aligned, holding the plan's words.
        let words = unsafe {
            core::slice::from_raw_parts_mut(
                batch.scratch.base.add(region as usize).cast::<u32>(),
                plan.bytes() as usize / 4,
            )
        };
        if plan.fill(words, routes, hidden, output).is_err() {
            batch.window.arena.rewind(mark);
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        let (out, out_bytes) = plan.out();
        for (index, launch) in launches[..2 * sets + 1].iter().enumerate() {
            let (kernel, program, header, chunks, part) = launch.expect("built above");
            let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
            let wave_slot = if part == MoeWave::Combine {
                0
            } else {
                for (slot, chunk) in chunks.iter().enumerate() {
                    ports[slot] = match *chunk {
                        Some((buffer, _)) => Placement::Resident { buffer, offset: 0 },
                        // The provider refuses overlapping bindings, so an
                        // unused slot takes a page of the ones run of its
                        // own; no group names it.
                        None => Placement::Scratch {
                            offset: batch.ones + slot as u64 * UNUSED_CHUNK_BYTES,
                        },
                    };
                }
                xe_moe::CHUNK_SLOTS
            };
            ports[wave_slot] = Placement::Scratch {
                offset: region + header,
            };
            let combine = part == MoeWave::Combine;
            let member = Member {
                key: MemberKey::Wave {
                    instruction,
                    wave,
                    part: index as u8,
                },
                kernel,
                program,
                ports,
                output: if combine { region + out } else { region },
                output_bytes: if combine { out_bytes } else { 0 },
                recurrent: None,
                barrier: false,
                state: MemberState::Staged,
            };
            // The first down reads what every up wrote, and the combine
            // what every down wrote: each waits for all earlier launches.
            let reads_staged = index == sets || combine;
            batch
                .window
                .admit(member, reads_staged, bytes_of(program))
                .expect("the window had room for the whole wave");
        }
        if let Some(profile) = self.profile.as_mut() {
            profile.waves += 1;
            profile.wave_routes += routes.len() as u64;
        }
        self.set_activity(Activity::Prepare);
        self.profile_record(start, |p| &mut p.prepare);
        Ok(KernelProgress::Pending)
    }

    /// Answer a retry of a resident KV operation: Pending until its
    /// submission completed, then the attention read's rows copied to
    /// `output`, or the append's token published (the CPU kernel's zero),
    /// and the members retired. `None` when nothing of it is admitted.
    fn kv_retire(
        &mut self,
        instruction: usize,
        output: &mut ax_exec::KvOutput<'_>,
    ) -> Result<Option<KernelProgress>, ax_exec::Error> {
        let batch = self.batch.as_mut().expect("batched lane");
        let ours = |key: MemberKey| matches!(key, MemberKey::Wave { instruction: i, wave: w, .. } if (i, w) == (instruction, KV_WAVE));
        let (mut parts, mut done, mut out) = (0u8, true, None);
        for member in batch.window.members().filter(|m| ours(m.key)) {
            parts += 1;
            done &= member.state == MemberState::Done;
            if member.output_bytes != 0 {
                out = Some((member.output, member.output_bytes));
            }
        }
        if parts == 0 {
            return Ok(None);
        }
        if !done {
            return Ok(Some(KernelProgress::Pending));
        }
        if batch.window.members().next().map(|m| m.key)
            != Some(MemberKey::Wave {
                instruction,
                wave: KV_WAVE,
                part: 0,
            })
        {
            return Err(self.fail(Self::local(AccelRefusal::Internal)));
        }
        match output {
            ax_exec::KvOutput::Appended(token) => {
                if out.is_some() {
                    return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
                }
                token.fill(0);
            }
            ax_exec::KvOutput::Attention(rows) => {
                let Some((offset, _)) = out.filter(|&(_, bytes)| bytes == size_of_val(*rows) as u64)
                else {
                    return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
                };
                // SAFETY: the submission retired and the provider completed
                // the device-to-CPU handoff; the region lies in the arena
                // and holds these bytes, and `rows` is a distinct host slice.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        batch.scratch.base.add(offset as usize).cast::<f32>(),
                        rows.as_mut_ptr(),
                        rows.len(),
                    );
                }
                if let Some(profile) = self.profile.as_mut() {
                    profile.copied_output_bytes = profile
                        .copied_output_bytes
                        .saturating_add(size_of_val(*rows) as u64);
                }
            }
        }
        let batch = self.batch.as_mut().expect("batched lane");
        for part in 0..parts {
            batch.window.retire_front(MemberKey::Wave {
                instruction,
                wave: KV_WAVE,
                part,
            });
        }
        Ok(Some(KernelProgress::Complete))
    }

    /// Admit a resident KV append or attention read as the wave
    /// `(instruction, KV_WAVE)`: the cache bound in place from the
    /// executor's table, the keys, values or queries where their producers
    /// put them (a pending admission's output, a resident import, or
    /// staged), the control block staged, and after an attention read over
    /// a long prefix its reduction. A resident operation has no CPU
    /// fallback: a decline with admissions ahead of it drains the window and
    /// is offered again; with nothing ahead, it fails the invocation.
    fn batched_kv(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        cache: ax_exec::ResidentKvCache,
        inputs: &[Option<ChainedInput<'_>>; 3],
        output_len: usize,
    ) -> Result<KernelProgress, ax_exec::Error> {
        use axnn::lowering::KernelParams;
        let Some((owner, table)) = self.kv else {
            return Ok(KernelProgress::Unsupported);
        };
        let Some(entry) = table
            .iter()
            .find(|entry| cache.owner() == owner && entry.allocation == cache.allocation())
            .copied()
        else {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        };
        if entry.params != cache.params() {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        // Rows come from the F32 operand (host or pending), positions from
        // the graph's I32 input, which is always on the host.
        let rows = |input: Option<ChainedInput<'_>>, per_row: usize| match input {
            Some(ChainedInput::Host(NativeRef::F32(values))) => Some(values.len() / per_row),
            Some(ChainedInput::Pending { elements, .. }) => Some(elements / per_row),
            _ => None,
        };
        let position = |input: Option<ChainedInput<'_>>| match input {
            Some(ChainedInput::Host(NativeRef::I32(positions))) => {
                u32::try_from(*positions.first()?).ok()
            }
            _ => None,
        };
        let head = (entry.params.kv_heads * entry.params.head_dim) as usize;
        // The members in order, and whether each waits on what the member
        // before it (or a pending admission) wrote.
        let mut members: [Option<(Kernel, bool)>; 2] = [None, None];
        match plan.params() {
            KernelParams::KvAppend(params) => {
                let (Some(first), Some(tokens)) = (position(inputs[0]), rows(inputs[1], head)) else {
                    return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
                };
                if rows(inputs[2], head) != Some(tokens) || output_len != 1 {
                    return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
                }
                let Some(kernel) = Kernel::kv_append(tokens, first, params) else {
                    return self.kv_declined("kv append kernel");
                };
                members[0] = Some((kernel, false));
            }
            KernelParams::AttentionDecode(params) => {
                let query = (params.query_heads * params.cache.head_dim) as usize;
                let (Some(first), Some(tokens)) = (position(inputs[1]), rows(inputs[2], query)) else {
                    return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
                };
                if output_len != tokens * query {
                    return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
                }
                // The causal prefix, in splits of KV_SPLIT_TOKENS within the
                // partials budget; a read after this window's append waits
                // for it.
                let prefix = first as usize + tokens;
                let rows = tokens * params.query_heads as usize;
                let budget = (KV_PARTIALS_BUDGET / (rows * KV_PARTIAL_BYTES)).max(1);
                let splits = (prefix / KV_SPLIT_TOKENS).clamp(1, KV_MAX_SPLITS).min(budget) as u32;
                let after_append = matches!(inputs[0], Some(ChainedInput::Pending { .. }));
                let Some(read) = Kernel::kv_attention(tokens, first, params, splits) else {
                    return self.kv_declined("kv attention kernel");
                };
                members[0] = Some((read, after_append));
                if splits > 1 {
                    let Some(reduce) = Kernel::kv_reduce(tokens, first, params, splits) else {
                        return self.kv_declined("kv reduce kernel");
                    };
                    members[1] = Some((reduce, true));
                }
            }
            _ => return Ok(KernelProgress::Unsupported),
        }
        let count = members.iter().flatten().count();
        let mut programs = [0u64; 2];
        for (slot, (kernel, _)) in members.iter().flatten().enumerate() {
            let Some(program) = self.program(kernel) else {
                return self.kv_declined("kv program");
            };
            programs[slot] = program;
        }
        let package_bytes: [(u64, u64); 2] =
            core::array::from_fn(|slot| (programs[slot], self.package_bytes(programs[slot])));
        let start = self.profile_start();
        let shared = &self.inputs;
        let batch = self.batch.as_mut().expect("batched lane");
        if batch.window.len() + count > WINDOW || !batch.window.fits_all(&package_bytes[..count]) {
            return self.kv_declined("kv window");
        }
        let mark = batch.window.arena.mark();
        // Fresh regions the members share: the partials the read writes
        // and its reduction reads, and the output the last member writes.
        let output_bytes = members.iter().flatten().last().map_or(0, |(k, _)| k.output_bytes());
        let mut staged = || -> Option<([[Placement; MAX_PORTS]; 2], [bool; 2], u64, u64)> {
            let output = batch.window.arena.allocate(output_bytes.max(4))?;
            let mut partials = None;
            let mut ports = [[Placement::Scratch { offset: 0 }; MAX_PORTS]; 2];
            let mut reads_staged = [false; 2];
            for (slot, (kernel, _)) in members.iter().flatten().enumerate() {
                for (index, &port) in kernel.ports().iter().enumerate() {
                    let required = kernel.port_bytes(port);
                    ports[slot][index] = match port {
                        Port::Input(input) => {
                            let input = inputs[usize::from(input)]?;
                            let (placement, staged) =
                                Self::operand_placement(batch, shared, input, required)?;
                            reads_staged[slot] |= staged;
                            placement
                        }
                        Port::KvCache { bytes, .. } => {
                            if bytes > entry.bytes {
                                return None;
                            }
                            Placement::Resident {
                                buffer: entry.buffer,
                                offset: entry.offset,
                            }
                        }
                        Port::Parameter => Placement::Scratch {
                            offset: Self::batched_stage(
                                batch,
                                kernel.control_bytes(),
                                &[],
                                required,
                            )?,
                        },
                        Port::Partials(bytes) => {
                            let offset = match partials {
                                Some(offset) => offset,
                                None => batch.window.arena.allocate(bytes)?,
                            };
                            partials = Some(offset);
                            Placement::Scratch { offset }
                        }
                        Port::Output => Placement::Scratch { offset: output },
                        _ => return None,
                    };
                }
                if placements_overlap(kernel, &ports[slot], batch.ones) {
                    return None;
                }
            }
            Some((ports, reads_staged, output, output_bytes))
        };
        let Some((ports, reads_staged, output, output_bytes)) = staged() else {
            batch.window.arena.rewind(mark);
            return self.kv_declined("kv scratch");
        };
        for (slot, (kernel, waits)) in members.iter().flatten().enumerate() {
            let last = slot + 1 == count;
            let member = Member {
                key: MemberKey::Wave {
                    instruction,
                    wave: KV_WAVE,
                    part: slot as u8,
                },
                kernel: *kernel,
                program: programs[slot],
                ports: ports[slot],
                output,
                output_bytes: if last { output_bytes } else { 0 },
                recurrent: None,
                barrier: *waits,
                state: MemberState::Staged,
            };
            batch
                .window
                .admit(member, reads_staged[slot] || *waits, package_bytes[slot].1)
                .expect("the window had room for the whole KV wave");
        }
        self.set_activity(Activity::Prepare);
        self.profile_record(start, |p| &mut p.prepare);
        Ok(KernelProgress::Pending)
    }

    /// A KV member the lane could not admit: behind admissions, a drain
    /// and another offer; with nothing ahead, the invocation fails, as a
    /// resident cache has no CPU fallback.
    fn kv_declined(&mut self, why: &'static str) -> Result<KernelProgress, ax_exec::Error> {
        let ahead = self
            .batch
            .as_ref()
            .is_some_and(|batch| batch.window.members().next().is_some());
        if ahead {
            return Ok(KernelProgress::Unsupported);
        }
        Err(self.fail(BufferExecutionError::Declined(why)))
    }

    /// Admit a layer's whole routed-expert instruction for the device to
    /// route itself: the wave plan (`xe-moe-plan.comp`) reading the
    /// router's scores and the hidden rows where they are (a pending
    /// producer's output in place), then the grouped up and down over the
    /// layer's resident imports and the combine, as one wave
    /// `(instruction, ROUTED_WAVE)` whose combine writes the instruction's
    /// output. Only a layer whose every expert the store holds in place for
    /// good, in at most five imports, is admitted; declines leave nothing
    /// staged, and the engine then routes the layer on the host.
    fn batched_routed(
        &mut self,
        instruction: usize,
        params: MoeParams,
        inputs: &[Option<ChainedInput<'_>>; 3],
        store: axnn_cpu::ExpertStoreRef<'_>,
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<Result<KernelProgress, WaveDecline>, ax_exec::Error> {
        if minimum != KernelAccuracy::Approximate {
            return Ok(Err(WaveDecline::Accuracy));
        }
        if params.experts as usize != xe_moe::ROUTED_EXPERTS
            || params.top_k as usize != xe_moe::ROUTED_TOP_K
        {
            return Ok(Err(WaveDecline::Plan(WaveRefusal::Geometry)));
        }
        let wave_programs = MoeWave::ALL.map(|part| {
            kore_xe_kernels::moe_wave(part)
                .and_then(|artifact| loaded_program(self.programs, &artifact))
        });
        let plan_program = kore_xe_kernels::moe_plan()
            .and_then(|artifact| loaded_program(self.programs, &artifact));
        let (Some(planner), [Some(up), Some(down), Some(combine)]) = (plan_program, wave_programs)
        else {
            return Ok(Err(WaveDecline::Program));
        };
        let width = xe_moe::MOE_HIDDEN;
        let (Some(hidden), Some(scores), Some(ChainedInput::Host(NativeRef::F32(bias)))) =
            (inputs[0], inputs[1], inputs[2])
        else {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        };
        if output.is_empty() || !output.len().is_multiple_of(width) {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        let tokens = output.len() / width;
        let start = self.profile_start();
        let mut experts = [axnn_cpu::ExpertWeights::EMPTY; xe_moe::ROUTED_EXPERTS];
        if !store.resident(&mut experts) {
            return Ok(Err(WaveDecline::Plan(WaveRefusal::NotResident)));
        }
        let shared = &self.inputs;
        let routed = match xe_moe::RoutedPlan::new(&experts, tokens, |address| {
            shared.resident_region(address)
        }) {
            Ok(routed) => routed,
            Err(why) => return Ok(Err(WaveDecline::Plan(why))),
        };
        let chunks = routed.chunks();
        let chunk_bytes = chunks.map(|chunk| chunk.map_or(UNUSED_CHUNK_BYTES, |(_, bytes)| bytes));
        let region_bytes = routed.bytes();
        let kernels = (
            Kernel::moe_plan(tokens, region_bytes),
            Kernel::moe_wave(MoeWave::Up, routed.max_tiles(), region_bytes, chunk_bytes),
            Kernel::moe_wave(MoeWave::Down, routed.max_tiles(), region_bytes, chunk_bytes),
            Kernel::moe_wave(MoeWave::Combine, tokens, region_bytes, chunk_bytes),
        );
        let (Some(plan_kernel), Some(up_kernel), Some(down_kernel), Some(combine_kernel)) = kernels
        else {
            return Ok(Err(WaveDecline::Kernel));
        };
        let members = [
            (plan_kernel, planner),
            (up_kernel, up),
            (down_kernel, down),
            (combine_kernel, combine),
        ];
        let package_bytes = members.map(|(_, program)| (program, self.package_bytes(program)));
        let shared = &self.inputs;
        let batch = self.batch.as_mut().expect("batched lane");
        if batch.window.len() + members.len() > WINDOW || !batch.window.fits_all(&package_bytes) {
            return Ok(Err(WaveDecline::Window));
        }
        let mark = batch.window.arena.mark();
        let mut staged = || -> Option<(u64, [Placement; MAX_PORTS], bool)> {
            let region = batch.window.arena.allocate(region_bytes)?;
            // SAFETY: as in `batched_stage`: a fresh arena region no admitted
            // member names, 64-byte aligned, holding the plan's words.
            let words = unsafe {
                core::slice::from_raw_parts_mut(
                    batch.scratch.base.add(region as usize).cast::<u32>(),
                    region_bytes as usize / 4,
                )
            };
            routed
                .fill(words, bias, params.routed_scale.to_f32(), params.normalize)
                .ok()?;
            let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
            let mut reads_staged = false;
            for (slot, &port) in plan_kernel.ports().iter().enumerate() {
                ports[slot] = match port {
                    Port::Input(index) => {
                        let input = if index == 0 { hidden } else { scores };
                        let required = plan_kernel.port_bytes(port);
                        let (placement, staged) =
                            Self::operand_placement(batch, shared, input, required)?;
                        reads_staged |= staged;
                        placement
                    }
                    Port::Wave(_) => Placement::Scratch { offset: region },
                    _ => return None,
                };
            }
            Some((region, ports, reads_staged))
        };
        let Some((region, plan_ports, plan_reads_staged)) = staged() else {
            batch.window.arena.rewind(mark);
            return Ok(Err(WaveDecline::Scratch));
        };
        if placements_overlap(&plan_kernel, &plan_ports, batch.ones) {
            batch.window.arena.rewind(mark);
            return Ok(Err(WaveDecline::Scratch));
        }
        let (out, out_bytes) = routed.out();
        for (part, &(kernel, program)) in members.iter().enumerate() {
            let ports = match part {
                0 => plan_ports,
                3 => {
                    let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
                    ports[0] = Placement::Scratch { offset: region };
                    ports
                }
                _ => {
                    let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
                    for (slot, chunk) in chunks.iter().enumerate() {
                        ports[slot] = match *chunk {
                            Some((buffer, _)) => Placement::Resident { buffer, offset: 0 },
                            // As in `batched_wave`: an unused slot takes a
                            // page of the ones run of its own.
                            None => Placement::Scratch {
                                offset: batch.ones + slot as u64 * UNUSED_CHUNK_BYTES,
                            },
                        };
                    }
                    ports[xe_moe::CHUNK_SLOTS] = Placement::Scratch { offset: region };
                    ports
                }
            };
            let combine = part == 3;
            let member = Member {
                key: MemberKey::Wave {
                    instruction,
                    wave: ROUTED_WAVE,
                    part: part as u8,
                },
                kernel,
                program,
                ports,
                output: if combine { region + out } else { region },
                output_bytes: if combine { out_bytes } else { 0 },
                recurrent: None,
                barrier: false,
                state: MemberState::Staged,
            };
            // The plan waits for staged producers of its operands; each later
            // launch reads what the one before it wrote.
            let reads_staged = if part == 0 { plan_reads_staged } else { true };
            batch
                .window
                .admit(member, reads_staged, package_bytes[part].1)
                .expect("the window had room for the whole routed wave");
        }
        if let Some(profile) = self.profile.as_mut() {
            profile.waves += 1;
            profile.wave_routes += (tokens * xe_moe::ROUTED_TOP_K) as u64;
        }
        self.set_activity(Activity::Prepare);
        self.profile_record(start, |p| &mut p.prepare);
        Ok(Ok(KernelProgress::Pending))
    }

    /// Run `kernel` by itself, outside any graph: one member over host
    /// operands (staged, or bound in place when they lie in a resident
    /// import), submitted as its own job and retired into `output`. For an
    /// on-device self-test before a graph runs; nothing else may be admitted.
    pub async fn run_alone(
        &mut self,
        kernel: Kernel,
        inputs: [Option<NativeRef<'_>>; 3],
        output: &mut [f32],
    ) -> Result<(), &'static str> {
        const KEY: MemberKey = MemberKey::Instruction(usize::MAX);
        if self.failure.is_some() {
            return Err("the executor already failed");
        }
        if self
            .batch
            .as_ref()
            .is_none_or(|batch| !batch.window.is_empty())
        {
            return Err("needs an empty batched lane");
        }
        let program = self.program(&kernel).ok_or("its program is not loaded")?;
        let chained = inputs.map(|input| input.map(ChainedInput::Host));
        match self.batched_admit(KEY, kernel, program, &chained, None, None, false) {
            Ok(KernelProgress::Pending) => {}
            Ok(_) => return Err("the batched lane declined it"),
            Err(_) => return Err("admission failed"),
        }
        // Held to completion: under overlap a wait would return with the
        // job on the device and nothing to retire.
        self.batched_wait_held(true)
            .await
            .map_err(|_| "the job failed")?;
        match self.batched_retire(KEY, output) {
            Ok(Some(KernelProgress::Complete)) => Ok(()),
            _ => Err("its result did not retire"),
        }
    }

    /// Run one resident KV operation by itself on the batched lane, for a
    /// self-test: admitted as its own window, held to completion, retired
    /// into `output`. Host operands only; the lane must be empty.
    pub async fn run_kv_alone(
        &mut self,
        plan: &CpuKernelPlan,
        cache: ax_exec::ResidentKvCache,
        inputs: [Option<NativeRef<'_>>; 3],
        mut output: ax_exec::KvOutput<'_>,
    ) -> Result<(), &'static str> {
        const INSTRUCTION: usize = usize::MAX - 1;
        if self.failure.is_some() {
            return Err("the executor already failed");
        }
        if self
            .batch
            .as_ref()
            .is_none_or(|batch| !batch.window.is_empty())
        {
            return Err("needs an empty batched lane");
        }
        if self.kv.is_none() {
            return Err("no resident caches");
        }
        let chained = inputs.map(|input| input.map(ChainedInput::Host));
        let output_len = match &output {
            ax_exec::KvOutput::Appended(token) => token.len(),
            ax_exec::KvOutput::Attention(rows) => rows.len(),
        };
        match self.batched_kv(INSTRUCTION, plan, cache, &chained, output_len) {
            Ok(KernelProgress::Pending) => {}
            Ok(_) => return Err("the batched lane declined it"),
            Err(_) => return Err("admission failed"),
        }
        self.batched_wait_held(true)
            .await
            .map_err(|_| "the job failed")?;
        match self.kv_retire(INSTRUCTION, &mut output) {
            Ok(Some(KernelProgress::Complete)) => Ok(()),
            _ => Err("its result did not retire"),
        }
    }

    /// Check a pure or fused offer's operands against its plan: host
    /// operands exactly the plan's bytes, pending ones F32 of that length.
    fn batched_operands_fit(
        &mut self,
        kernel: &Kernel,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
    ) -> Result<(), ax_exec::Error> {
        for (index, input) in inputs[..kernel.input_bytes().len()].iter().enumerate() {
            let (Some(input), Some(desc)) = (input, plan.input(index)) else {
                return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
            };
            let fits = match input {
                ChainedInput::Host(value) => {
                    value.dtype() == desc.dtype
                        && Some(value.as_bytes().len()) == desc.checked_storage_bytes()
                }
                ChainedInput::Pending { elements, .. } => {
                    desc.dtype == axnn::DType::F32
                        && elements.checked_mul(4) == desc.checked_storage_bytes()
                }
            };
            if !fits {
                return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
            }
        }
        Ok(())
    }

    /// Admit a fused run as one member keyed by its last instruction: the
    /// run's projection package with its epilogue, an accumulate run's base
    /// staged into the accumulator. A base a pending admission produces is
    /// declined (it would need a device copy first).
    #[allow(clippy::too_many_arguments)]
    fn batched_fused(
        &mut self,
        last: usize,
        run: ax_exec::FusedRun,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        base: Option<ChainedInput<'_>>,
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        let key = MemberKey::Instruction(last);
        if let Some(progress) = self.batched_retire(key, output)? {
            return Ok(progress);
        }
        let Some(kernel) = Kernel::fused_run(plan, run, minimum) else {
            return Ok(KernelProgress::Unsupported);
        };
        let Some(program) = self.program(&kernel) else {
            return Ok(KernelProgress::Unsupported);
        };
        if core::mem::size_of_val(output) as u64 != kernel.output_bytes() {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        if matches!(run, ax_exec::FusedRun::RmsNorm { .. }) {
            // The first two ports belong to the binary plan. The third is
            // the fused tail's weight, whose captured geometry is checked
            // separately because it is not an operand of that first plan.
            let binary = Kernel::from_plan(plan, minimum)
                .ok_or_else(|| self.fail(Self::local(AccelRefusal::InvalidArgument)))?;
            self.batched_operands_fit(&binary, plan, inputs)?;
            let fits = match base {
                Some(ChainedInput::Host(NativeRef::F32(weight))) => {
                    core::mem::size_of_val(weight) as u64 == kernel.input_bytes()[2]
                }
                Some(ChainedInput::Pending { elements, .. }) => {
                    elements.checked_mul(4).map(|n| n as u64) == Some(kernel.input_bytes()[2])
                }
                _ => false,
            };
            if !fits {
                return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
            }
            let mut operands = *inputs;
            operands[2] = base;
            return self.batched_admit(key, kernel, program, &operands, None, None, false);
        }
        self.batched_operands_fit(&kernel, plan, inputs)?;
        let base = match (run, base) {
            (ax_exec::FusedRun::SquaredRelu, None) => None,
            (ax_exec::FusedRun::Accumulate, Some(ChainedInput::Host(NativeRef::F32(base)))) => {
                Some(NativeRef::F32(base).as_bytes())
            }
            (ax_exec::FusedRun::Accumulate, Some(_)) => return Ok(KernelProgress::Unsupported),
            _ => return Err(self.fail(Self::local(AccelRefusal::InvalidArgument))),
        };
        self.batched_admit(key, kernel, program, inputs, base, None, false)
    }

    /// Write the staged members' table to the stage and open one ordered
    /// submission for all of them.
    fn batched_submit(&mut self) -> Result<(), BufferExecutionError> {
        let queue = self.queue;
        let timeout_ns = self.policy.job_ns;
        let batch = self.batch.as_mut().expect("batched lane");
        if batch.flight_count == PIPELINE_DEPTH || !self.client.execution_request_idle() {
            return Err(Self::local(AccelRefusal::Busy));
        }
        let mut launches: [Option<OrderedLaunch>; WINDOW] = [None; WINDOW];
        let mut count = 0;
        let mut fault = None;
        for member in batch.window.staged().take(batch.sequence_limit) {
            let bindings = member
                .kernel
                .port_bindings(xe_batch::place(member, batch.scratch.buffer, batch.ones))
                .map_err(|_| Self::local(AccelRefusal::InvalidArgument))?;
            // Every range against the buffer it names, as the service checks
            // them: its refusal (`OutOfBounds`) names no member or slot.
            for (slot, binding) in bindings.iter().enumerate() {
                let limit = if binding.buffer == batch.scratch.buffer {
                    Some(batch.scratch.bytes as u64)
                } else if let Some(state) = self.recurrent_state.as_ref()
                    && binding.buffer == state.buffer
                {
                    Some(state.bytes as u64)
                } else {
                    self.inputs.buffer_bytes(binding.buffer)
                };
                if let Some(limit) = limit
                    && binding.offset.saturating_add(binding.len) > limit
                    && fault.is_none()
                {
                    fault = Some(BindingFault {
                        member: member.key,
                        slot,
                        buffer: binding.buffer,
                        offset: binding.offset,
                        len: binding.len,
                        limit,
                    });
                }
            }
            launches[count] = Some(OrderedLaunch {
                queue,
                program: member.program,
                bindings,
                timeout_ns,
                launch: member.kernel.launch(),
                // The first member of a submission follows nothing in it.
                barrier: member.barrier && count > 0,
                gate: None,
                signal: None,
            });
            count += 1;
        }
        if let Some(fault) = fault {
            self.binding_fault = Some(fault);
            return Err(Self::local(AccelRefusal::OutOfBounds));
        }
        let first = launches[0].ok_or_else(|| Self::local(AccelRefusal::InvalidArgument))?;
        let dense: [OrderedLaunch; WINDOW] = core::array::from_fn(|i| launches[i].unwrap_or(first));
        // SAFETY: `SequenceStage::new`'s contract: this executor alone writes
        // the region. Earlier admission replies have consumed their tables;
        // no outstanding request can still read this staging span.
        let staging =
            unsafe { core::slice::from_raw_parts_mut(batch.stage.base, batch.stage.bytes) };
        let mut groups = [None; WINDOW];
        let tables = stage_ordered_launches(
            &dense[..count],
            staging,
            batch.stage.offset,
            &mut groups,
            ProgramReuse::Share,
        )
        .map_err(Self::local)?;
        let table = groups[0]
            .filter(|_| tables == 1)
            .ok_or_else(|| Self::local(AccelRefusal::InvalidArgument))?;
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        let ledger = self
            .client
            .execution_ledger_sequence(core::iter::once(table))
            .ok_or_else(|| Self::local(AccelRefusal::Busy))?;
        let batch = self.batch.as_mut().expect("batched lane");
        let members = batch.window.submit_first(batch.sequence_limit);
        batch.flights[batch.flight_count] = Some(BatchFlight {
            ledger,
            members,
            program: first.program,
            submitted: None,
            deadline: None,
        });
        batch.flight_count += 1;
        if let Some(profile) = self.profile.as_mut() {
            profile.sequences = profile.sequences.saturating_add(1);
            profile.members = profile.members.saturating_add(count as u64);
        }
        Ok(())
    }

    /// Queue ready work before observing an older event. The device orders
    /// dependencies on its one ring; host publication still retires oldest
    /// first. A full pipeline, a real host dependency, or a held self-test
    /// drives completion instead of admitting an unbounded backlog.
    async fn batched_wait(&mut self) -> Result<(), ax_exec::Error> {
        self.batched_wait_held(false).await
    }

    async fn batched_wait_held(&mut self, hold: bool) -> Result<(), ax_exec::Error> {
        self.drive::<NoGates>(hold, None, None).await
    }

    /// The flights' one driver. The graph lane (`frozen` none) submits the
    /// window's staged members and returns to its lookahead once the device
    /// holds a flight or the oldest retires. A frozen sequence submits its
    /// next window whenever a flight slot is free, so the device always has
    /// the following window queued behind the one it runs (the ring orders
    /// them), and returns once its last window retired. A gated one answers
    /// its signals before every service call and through every wait.
    async fn drive<G: GateService>(
        &mut self,
        hold: bool,
        frozen: Option<&[FrozenLaunch]>,
        mut gates: Option<&mut G>,
    ) -> Result<(), ax_exec::Error> {
        // The frozen sequence's first launch not yet submitted.
        let mut next = 0;
        loop {
            if let Some(gates) = gates.as_deref_mut() {
                while self.serve_gates(gates).await {}
            }
            let now = clock_ns().await;
            let batch = self.batch.as_ref().expect("batched lane");
            let staged = match frozen {
                None => batch.window.staged().next().is_some(),
                Some(launches) => next < launches.len(),
            };
            let can_submit = batch.request_owner.is_none()
                && staged
                && batch.flight_count < batch.pipeline_depth;
            let launched = can_submit;
            let index = if can_submit {
                match frozen {
                    None => self.batched_submit().map_err(|e| self.fail(e))?,
                    Some(launches) => {
                        let end = self.frozen_cut(launches, next)?;
                        self.frozen_submit(&launches[next..end])
                            .map_err(|e| self.fail(e))?;
                        next = end;
                    }
                }
                self.batch.as_ref().expect("batched lane").flight_count - 1
            } else {
                let batch = self.batch.as_ref().expect("batched lane");
                if batch.flight_count == 0 {
                    return Ok(());
                }
                batch.request_owner.unwrap_or(0)
            };
            let started = self.profile_start();
            let (observer, activity, completed) = (self.observer, self.activity, self.completed);
            let batch = self.batch.as_mut().expect("batched lane");
            let instruction = Self::window_instruction(&batch.window);
            let flight = batch.flights[index].as_mut().expect("retained flight");
            let deadline = *flight
                .deadline
                .get_or_insert(now.saturating_add(self.policy.operation_ns));
            if now >= deadline {
                flight.ledger.request_cancel();
                return Err(self.fail(BufferExecutionError::Execution(
                    ExecutionError::Infrastructure(CallError::Timeout),
                )));
            }
            let submitting = flight.ledger.event().is_none();
            let request: fn(&mut Profile) -> &mut WallTime = if submitting {
                |p| &mut p.submit
            } else if flight.ledger.awaiting_device_completion() {
                |p| &mut p.poll
            } else {
                |p| &mut p.retire
            };
            let kind: u8 = if submitting {
                1
            } else if flight.ledger.awaiting_device_completion() {
                2
            } else {
                3
            };
            if kind == 2 && !drain_stale_notices(self.completion) {
                return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
            }
            // Install correlation ownership before an await can be cancelled.
            batch.request_owner = Some(index);
            publish_activity(observer, true, kind, activity, completed, instruction);
            let result = flight
                .ledger
                .advance_queued_bounded(self.client, self.policy.call_ns)
                .await;
            publish_activity(observer, false, kind, activity, completed, instruction);
            self.profile_record_execute(started, request);
            let idle = self.client.execution_request_idle();
            let batch = self.batch.as_mut().expect("batched lane");
            let retrying_admission = matches!(result, Ok(ExecutionProgress::Pending))
                && batch.flights[index]
                    .as_ref()
                    .expect("retained flight")
                    .ledger
                    .event()
                    .is_none();
            batch.request_owner = (!idle || retrying_admission).then_some(index);
            let flight = batch.flights[index].as_mut().expect("retained flight");
            let awaiting = flight.ledger.awaiting_device_completion();
            if awaiting && flight.submitted.is_none() {
                flight.submitted = Some(now);
                if index > 0
                    && let Some(profile) = self.profile.as_mut()
                {
                    profile.queued_sequences = profile.queued_sequences.saturating_add(1);
                }
            }
            match result {
                Ok(ExecutionProgress::Finished(ExecutionOutcome::Retired(
                    AccelEventState::Complete,
                ))) => {
                    // Only the oldest event is polled/retired; later ledgers
                    // are advanced solely to obtain their admission replies.
                    if index != 0 || !idle {
                        return Err(self.fail(Self::local(AccelRefusal::Internal)));
                    }
                    let flight = batch.flights[0].take().expect("retired flight");
                    batch.window.complete_first(flight.members);
                    batch.flights.rotate_left(1);
                    batch.flight_count -= 1;
                    if let Some(submitted) = flight.submitted {
                        let done = clock_ns().await;
                        self.pace
                            .record(flight.program, done.saturating_sub(submitted));
                    }
                    self.set_activity(Activity::Idle);
                    self.publish_progress();
                    if frozen.is_none() {
                        return Ok(());
                    }
                }
                Ok(ExecutionProgress::Finished(
                    ExecutionOutcome::Refused(refusal)
                    | ExecutionOutcome::Retired(AccelEventState::Failed(refusal)),
                )) => {
                    return Err(self.fail(Self::local(refusal)));
                }
                Ok(ExecutionProgress::Finished(_)) => {
                    return Err(self.fail(Self::local(AccelRefusal::Internal)));
                }
                Ok(ExecutionProgress::Pending)
                | Err(ExecutionError::Infrastructure(CallError::Timeout)) => {
                    if awaiting
                        && idle
                        && !hold
                        && (batch.pipeline_depth > 1 || batch.overlap)
                        && (launched || submitting || index > 0)
                    {
                        // Admission is acknowledged. Return to graph lookahead
                        // while this job executes; staging may now be reused.
                        // A frozen sequence stages its next window instead.
                        self.set_activity(Activity::Idle);
                        if frozen.is_none() {
                            return Ok(());
                        }
                        continue;
                    }
                    let slice = if retrying_admission && idle {
                        // Busy is confirmed non-admission. Keep the staging
                        // owner, but park between bounded retries.
                        Some(self.policy.completion_ns)
                    } else if awaiting && idle {
                        let (program, submitted) = (flight.program, flight.submitted);
                        let elapsed = now.saturating_sub(submitted.unwrap_or(now));
                        Some(self.pace.slice(
                            program,
                            elapsed,
                            self.policy.completion_ns,
                            PACE_CEILING_NS,
                        ))
                    } else {
                        None
                    };
                    if let Some(slice) = slice {
                        let waited = match gates.as_deref_mut() {
                            Some(gates) => self.wait_gated(slice, gates).await,
                            None => self.wait_notice(slice).await,
                        };
                        waited.map_err(|e| self.fail(e))?;
                    }
                }
                Err(error) => return Err(self.fail(BufferExecutionError::Execution(error))),
            }
        }
    }

    /// Submit a pre-lowered, pre-bound "frozen" sequence as ordered work: the
    /// caller resolved every kernel's placements already, so this lane encodes
    /// them into the batched sequence machinery (one table per window, one
    /// completion fence per table). The frozen working buffer lives in the
    /// same resident scratch `new_batched` was built over; this method
    /// consumes none of its window arena and writes nothing but the sequence
    /// tables.
    ///
    /// A window holds at most [`Batched::sequence_limit`] kernels and ends
    /// sooner where the next kernel's package would take it past the
    /// provider's command window (each distinct program's ISA is staged once
    /// per window, as the batched lane charges it). Up to the pipeline depth
    /// of windows fly at once: the next one is admitted while the device runs
    /// the one before it, ordered behind it on the one ring, and the stage is
    /// rewritten only once the previous admission was acknowledged. Returns
    /// once the last window retired. The executor must be quiet: no graph
    /// lane work admitted or flying.
    pub async fn submit_frozen(
        &mut self,
        launches: &[FrozenLaunch],
    ) -> Result<(), ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        match self.batch.as_ref() {
            Some(batch) if batch.flight_count == 0 && batch.window.is_empty() => {}
            _ => return Err(self.fail(Self::local(AccelRefusal::InvalidArgument))),
        }
        if launches.is_empty() {
            return Ok(());
        }
        self.drive::<NoGates>(false, Some(launches), None).await
    }

    /// [`Self::submit_frozen`] for a sequence whose launches carry gates and
    /// signals ([`FrozenLaunch::gate`], [`FrozenLaunch::signal`]): `gates`
    /// answers the signals while the windows fly, so the device waits at a
    /// gate only for the host's work for it. Returns once the last window
    /// retired; every signal the run made has been answered by then. The
    /// sync words' values are the caller's: it resets them between runs,
    /// while no window flies.
    pub async fn submit_frozen_gated<G: GateService>(
        &mut self,
        launches: &[FrozenLaunch],
        gates: &mut G,
    ) -> Result<(), ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        match self.batch.as_ref() {
            Some(batch) if batch.flight_count == 0 && batch.window.is_empty() => {}
            _ => return Err(self.fail(Self::local(AccelRefusal::InvalidArgument))),
        }
        if launches.is_empty() {
            return Ok(());
        }
        self.drive(false, Some(launches), Some(&mut *gates)).await?;
        // A signal the last window wrote after its final poll.
        while self.serve_gates(gates).await {}
        Ok(())
    }

    /// The end of the frozen window that starts at `start`: as many launches
    /// as the sequence limit and the provider's command window admit.
    fn frozen_cut(
        &mut self,
        launches: &[FrozenLaunch],
        start: usize,
    ) -> Result<usize, ax_exec::Error> {
        let limit = self.batch.as_ref().map_or(1, |batch| batch.sequence_limit);
        let mut budget = crate::xe_batch::Window::new(0);
        let mut end = start;
        // A window's gates and signals ride after its entries, one table
        // of them: it ends before they would pass the table's ceiling.
        let mut syncs = 0;
        while end < launches.len() && end - start < limit {
            let Some(program) = self.program(&launches[end].kernel) else {
                return Err(self.fail(Self::local(AccelRefusal::Unsupported)));
            };
            let bytes = self.package_bytes(program);
            let records = usize::from(launches[end].gate.is_some()) + usize::from(launches[end].signal.is_some());
            if !budget.fits(program, bytes) || syncs + records > kore_abi::svc::MAX_ACCEL_SEQUENCE_SYNCS {
                break;
            }
            budget.charge(program, bytes);
            syncs += records;
            end += 1;
        }
        if end == start {
            // One kernel alone exceeds the provider's command window.
            return Err(self.fail(Self::local(AccelRefusal::ResourceLimit)));
        }
        Ok(end)
    }

    /// Encode one window of frozen launches into the sequence table and open
    /// its ordered submission in the next flight slot.
    fn frozen_submit(&mut self, window: &[FrozenLaunch]) -> Result<(), BufferExecutionError> {
        let queue = self.queue;
        let timeout_ns = self.policy.job_ns;
        let (scratch_buffer, scratch_bytes, ones) = {
            let batch = self.batch.as_ref().expect("batched lane");
            if batch.flight_count == PIPELINE_DEPTH || !self.client.execution_request_idle() {
                return Err(Self::local(AccelRefusal::Busy));
            }
            (batch.scratch.buffer, batch.scratch.bytes as u64, batch.ones)
        };
        let mut launches: [Option<OrderedLaunch>; WINDOW] = [None; WINDOW];
        let mut count = 0;
        let mut fault = None;
        for (index, launch) in window.iter().enumerate() {
            let kernel = launch.kernel;
            let program = self
                .program(&kernel)
                .ok_or_else(|| Self::local(AccelRefusal::Unsupported))?;
            let member = frozen_ordered_launch(
                scratch_buffer,
                ones,
                queue,
                timeout_ns,
                program,
                launch,
                launch.barrier && index > 0,
            )?;
            // Every range against the buffer it names, as `batched_submit`
            // checks them: the provider's `OutOfBounds` names no member.
            for binding in member.bindings.iter() {
                let limit = if binding.buffer == scratch_buffer {
                    Some(scratch_bytes)
                } else {
                    self.inputs.buffer_bytes(binding.buffer)
                };
                if let Some(limit) = limit
                    && binding.offset.saturating_add(binding.len) > limit
                    && fault.is_none()
                {
                    fault = Some(BindingFault {
                        // The frozen sequence has no instruction identity; the
                        // window index stands in for the failure report.
                        member: MemberKey::Instruction(index),
                        slot: binding.slot as usize,
                        buffer: binding.buffer,
                        offset: binding.offset,
                        len: binding.len,
                        limit,
                    });
                }
            }
            launches[count] = Some(member);
            count += 1;
        }
        if let Some(fault) = fault {
            self.binding_fault = Some(fault);
            return Err(Self::local(AccelRefusal::OutOfBounds));
        }
        let first = launches[0].ok_or_else(|| Self::local(AccelRefusal::InvalidArgument))?;
        let dense: [OrderedLaunch; WINDOW] = core::array::from_fn(|i| launches[i].unwrap_or(first));
        let batch = self.batch.as_mut().expect("batched lane");
        // SAFETY: as in `batched_submit`: `SequenceStage::new`'s contract makes
        // this executor the region's sole writer. Earlier admission replies
        // have consumed their tables (the wire is idle); no outstanding
        // request can still read this staging span.
        let staging =
            unsafe { core::slice::from_raw_parts_mut(batch.stage.base, batch.stage.bytes) };
        let mut groups = [None; WINDOW];
        let tables = stage_ordered_launches(
            &dense[..count],
            staging,
            batch.stage.offset,
            &mut groups,
            ProgramReuse::Share,
        )
        .map_err(Self::local)?;
        let table = groups[0]
            .filter(|_| tables == 1)
            .ok_or_else(|| Self::local(AccelRefusal::InvalidArgument))?;
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        let ledger = self
            .client
            .execution_ledger_sequence(core::iter::once(table))
            .ok_or_else(|| Self::local(AccelRefusal::Busy))?;
        let batch = self.batch.as_mut().expect("batched lane");
        // No window members ride along: retirement completes none.
        batch.flights[batch.flight_count] = Some(BatchFlight {
            ledger,
            members: 0,
            program: first.program,
            submitted: None,
            deadline: None,
        });
        batch.flight_count += 1;
        if let Some(profile) = self.profile.as_mut() {
            profile.sequences = profile.sequences.saturating_add(1);
            profile.members = profile.members.saturating_add(count as u64);
        }
        Ok(())
    }
}

impl PreparedKernelExecutor for Executor<'_, '_, '_> {
    fn retire_unobserved(&mut self, instruction: usize) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        let Some(batch) = self.batch.as_mut() else {
            return Ok(KernelProgress::Unsupported);
        };
        let key = MemberKey::Instruction(instruction);
        let Some((index, member)) = batch.window.find(key) else {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        };
        let member = *member;
        // Resource outputs must still be published to the state owner.
        if member.recurrent.is_some() {
            return Ok(KernelProgress::Unsupported);
        }
        if index != 0 {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        if member.state != MemberState::Done {
            return Ok(KernelProgress::Pending);
        }
        // Every downstream binding already owns its scratch offset. Removing
        // this member does not reuse that region: the arena resets only after
        // the whole window (including those consumers) has retired.
        batch.window.retire_front(key);
        if let Some(profile) = self.profile.as_mut() {
            profile.avoided_output_bytes = profile
                .avoided_output_bytes
                .saturating_add(member.output_bytes);
        }
        self.completed += 1;
        self.set_activity(Activity::Idle);
        self.publish_progress();
        Ok(KernelProgress::Complete)
    }

    fn admission_window(&self) -> usize {
        if self.batch.is_some() { WINDOW } else { 1 }
    }

    fn retire_moe_unobserved(
        &mut self,
        instruction: usize,
        wave: usize,
        routes: usize,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        if self.batch.is_none() {
            return Ok(KernelProgress::Unsupported);
        }
        self.wave_retire(instruction, wave, routes, None)?
            .ok_or(ax_exec::Error::InvocationFinished)
    }

    fn moe_output_pending(&self, instruction: usize, wave: usize) -> bool {
        self.batch.as_ref().is_some_and(|batch| {
            batch.window.members().any(|member| {
                matches!(member.key, MemberKey::Wave { instruction: i, wave: w, .. }
                    if i == instruction && w == wave)
                    && member.output_bytes != 0
            })
        })
    }

    fn moe_wave_capacity(
        &self,
        params: MoeParams,
        tokens: usize,
        routes: &[axnn_cpu::MoeBatchRow],
    ) -> usize {
        let Some(batch) = self.batch.as_ref() else {
            return axnn_cpu::DEFAULT_WAVE_EXPERTS;
        };
        if params.hidden as usize != xe_moe::MOE_HIDDEN
            || params.intermediate as usize != xe_moe::MOE_INTERMEDIATE
        {
            return axnn_cpu::DEFAULT_WAVE_EXPERTS;
        }
        let programs = MoeWave::ALL.map(|part| {
            kore_xe_kernels::moe_wave(part)
                .and_then(|artifact| loaded_program(self.programs, &artifact))
        });
        let [Some(up), Some(down), Some(combine)] = programs else {
            return axnn_cpu::DEFAULT_WAVE_EXPERTS;
        };
        let packages = [up, down, combine].map(|p| (p, self.package_bytes(p)));
        // Exact expert placements become known only after the source load.
        // Counting every resident region is a safe upper bound on chunk sets;
        // no model- or device-specific expert count enters this policy.
        let imports = usize::from(matches!(self.inputs.backing, InputBacking::Resident(_)))
            + self
                .inputs
                .scoped
                .iter()
                .filter(|r| r.resident.is_some())
                .count();
        WavePlan::capacity(routes, tokens, imports, |sets, bytes| {
            let count = 2 * sets + 1;
            if batch.window.len() + count > WINDOW {
                return false;
            }
            let mut members = [packages[0]; WINDOW];
            members[sets..2 * sets].fill(packages[1]);
            members[2 * sets] = packages[2];
            // Probe a copy: sizing cannot consume scratch or retain authority.
            let mut arena = batch.window.arena;
            batch.window.fits_all(&members[..count]) && arena.allocate(bytes).is_some()
        })
    }

    fn advance_moe_experts(
        &mut self,
        instruction: usize,
        wave: usize,
        plan: &CpuKernelPlan,
        params: MoeParams,
        hidden: &[f32],
        routes: &[axnn_cpu::MoeBatchRow],
        experts: &[axnn_cpu::ExpertWeights<'_>],
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        // Waves run only in the batched lane; the single-job lane leaves
        // them, and experts of another shape, to the CPU.
        if self.batch.is_none() || !wave_plan_matches(plan, params) {
            return Ok(KernelProgress::Unsupported);
        }
        self.batched_wave(instruction, wave, hidden, routes, experts, output, minimum)
    }

    fn advance_moe_routed(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        store: axnn_cpu::ExpertStoreRef<'_>,
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        let axnn::lowering::KernelParams::MoeExperts(params) = plan.params() else {
            return Ok(KernelProgress::Unsupported);
        };
        if self.batch.is_none() || !wave_plan_matches(plan, params) {
            return Ok(KernelProgress::Unsupported);
        }
        let routes = output.len() / xe_moe::MOE_HIDDEN * xe_moe::ROUTED_TOP_K;
        // A routed admission's output is always published: the engine reads
        // it on the host (no device consumer is admitted behind it here).
        if let Some(progress) = self.wave_retire(instruction, ROUTED_WAVE, routes, Some(output))? {
            return Ok(progress);
        }
        // A decline is a value here, never the strict-placement failure a
        // final wave decline is: the engine then routes the layer itself.
        match self.batched_routed(instruction, params, inputs, store, output, minimum)? {
            Ok(progress) => Ok(progress),
            Err(why) => self.routed_declined(why),
        }
    }

    fn advance_chained(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        if self.batch.is_some() {
            return self.batched_chained(instruction, plan, inputs, output, minimum);
        }
        let mut host = [None; 3];
        for (host, input) in host.iter_mut().zip(inputs) {
            match input {
                None => {}
                Some(ChainedInput::Host(value)) => *host = Some(*value),
                Some(ChainedInput::Pending { .. }) => return Ok(KernelProgress::Unsupported),
            }
        }
        self.advance(instruction, plan, &host, output, minimum)
    }

    fn advance_kv(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        cache: ax_exec::ResidentKvCache,
        inputs: &[Option<NativeRef<'_>>; 3],
        output: ax_exec::KvOutput<'_>,
        _: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        if self.batch.is_none() || self.kv.is_none() {
            return Ok(KernelProgress::Unsupported);
        }
        let mut output = output;
        if let Some(progress) = self.kv_retire(instruction, &mut output)? {
            return Ok(progress);
        }
        let output_len = match &output {
            ax_exec::KvOutput::Appended(token) => token.len(),
            ax_exec::KvOutput::Attention(rows) => rows.len(),
        };
        let chained = inputs.map(|input| input.map(ChainedInput::Host));
        self.batched_kv(instruction, plan, cache, &chained, output_len)
    }

    fn advance_kv_chained(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        cache: ax_exec::ResidentKvCache,
        inputs: &[Option<ChainedInput<'_>>; 3],
        output: ax_exec::KvOutput<'_>,
        _: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        if self.batch.is_none() || self.kv.is_none() {
            return Ok(KernelProgress::Unsupported);
        }
        let mut output = output;
        if let Some(progress) = self.kv_retire(instruction, &mut output)? {
            return Ok(progress);
        }
        let output_len = match &output {
            ax_exec::KvOutput::Appended(token) => token.len(),
            ax_exec::KvOutput::Attention(rows) => rows.len(),
        };
        self.batched_kv(instruction, plan, cache, inputs, output_len)
    }

    fn advance_recurrent(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        state: ax_exec::RecurrentState<'_>,
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        // Only the batched lane carries a state pair; the serial lane's
        // per-port path binds inputs, parameter, output.
        if self.batch.is_none() {
            return Ok(KernelProgress::Unsupported);
        }
        self.batched_recurrent(instruction, plan, inputs, state, output, minimum)
    }

    fn advance_fused(
        &mut self,
        last: usize,
        run: ax_exec::FusedRun,
        plan: &CpuKernelPlan,
        inputs: &[Option<ChainedInput<'_>>; 3],
        base: Option<ChainedInput<'_>>,
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        if self.batch.is_none() {
            return Ok(KernelProgress::Unsupported);
        }
        self.batched_fused(last, run, plan, inputs, base, output, minimum)
    }

    fn begin(&mut self) -> Result<(), ax_exec::Error> {
        if self.active.is_some() || self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        if let Some(batch) = self.batch.as_mut() {
            // Members an abandoned invocation staged, or finished without
            // publishing, hold no device access; one in flight still does.
            if batch.flight_count != 0 || !batch.window.discard_staged() {
                return Err(ax_exec::Error::InvocationFinished);
            }
            batch.request_owner = None;
        }
        Ok(())
    }

    fn advance(
        &mut self,
        instruction: usize,
        plan: &CpuKernelPlan,
        inputs: &[Option<NativeRef<'_>>; 3],
        output: &mut [f32],
        minimum: KernelAccuracy,
    ) -> Result<KernelProgress, ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        if self.batch.is_some() {
            return Ok(KernelProgress::Unsupported);
        }
        if let Some(progress) = self.observe(instruction, output)? {
            return Ok(progress);
        }
        let Some(kernel) = Kernel::from_plan(plan, minimum).filter(|kernel| {
            kernel.legacy_layout()
                && kernel.inputs_aligned(inputs.iter().map(|input| input.map(|r| r.as_bytes())))
        }) else {
            return Ok(KernelProgress::Unsupported);
        };
        let Some(program) = self.program(&kernel) else {
            return Ok(KernelProgress::Unsupported);
        };
        let count = kernel.input_bytes().len();
        if core::mem::size_of_val(output) as u64 != kernel.output_bytes() {
            return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
        }
        for (index, input) in inputs[..count].iter().enumerate() {
            let Some(input) = input else {
                return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
            };
            let Some(desc) = plan.input(index) else {
                return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
            };
            if input.dtype() != desc.dtype
                || Some(input.as_bytes().len()) != desc.checked_storage_bytes()
            {
                return Err(self.fail(Self::local(AccelRefusal::InvalidArgument)));
            }
        }
        if !self.can_prepare(&kernel, inputs) {
            return Ok(KernelProgress::Unsupported);
        }
        self.prepare(instruction, kernel, program, inputs)
            .map_err(|e| self.fail(e))?;
        Ok(KernelProgress::Pending)
    }

    async fn wait(&mut self) -> Result<(), ax_exec::Error> {
        if self.failure.is_some() {
            return Err(ax_exec::Error::InvocationFinished);
        }
        if self.batch.is_some() {
            return self.batched_wait().await;
        }
        let now = clock_ns().await;
        let started = self.profile_start();
        let Some(active) = self.active.as_mut() else {
            return Err(ax_exec::Error::InvocationFinished);
        };
        let deadline = *active
            .deadline
            .get_or_insert(now.saturating_add(self.policy.operation_ns));
        if now >= deadline {
            active.execution.request_cancel();
            return Err(self.fail(BufferExecutionError::Execution(
                ExecutionError::Infrastructure(CallError::Timeout),
            )));
        }
        let was_awaiting = active.execution.awaiting_device_completion();
        let mut activity = BufferExecutionActivity::Finished;
        let observation = self.progress;
        let snapshot = Progress {
            completed: self.completed,
            instruction: Some(active.instruction),
            activity: self.activity,
            pending: true,
            profile: self.profile,
        };
        let result = active
            .execution
            .advance_bounded_observed(self.client, self.policy.call_ns, |a| {
                activity = a;
                if let Some(observation) = observation {
                    observation.set(Progress {
                        activity: Activity::Wire(a),
                        ..snapshot
                    });
                }
            })
            .await;
        let awaiting_completion = active.execution.awaiting_device_completion();
        if awaiting_completion && active.submitted.is_none() {
            // The submit reply just made completion owed: the job's start.
            active.submitted = Some(now);
        }
        let (program, submitted) = (active.program, active.submitted);
        if was_awaiting
            && !awaiting_completion
            && let Some(submitted) = submitted
        {
            // This advance observed the completion: one sample for the pace.
            let done = clock_ns().await;
            self.pace.record(program, done.saturating_sub(submitted));
        }
        self.set_activity(Activity::Wire(activity));
        match activity {
            BufferExecutionActivity::Import { .. } => {
                self.profile_record(started, |p| &mut p.import)
            }
            BufferExecutionActivity::Execute => self.profile_record(started, |p| &mut p.execute),
            BufferExecutionActivity::Export { .. } => {
                self.profile_record(started, |p| &mut p.export)
            }
            BufferExecutionActivity::Finished => {}
        }
        match result {
            Ok(ExecutionProgress::Finished(_)) => Ok(()),
            Ok(ExecutionProgress::Pending)
            | Err(BufferExecutionError::Execution(ExecutionError::Infrastructure(
                CallError::Timeout,
            )))
            | Err(BufferExecutionError::Import(
                RetainedReturnCallError::NotMoved(CallError::Timeout)
                | RetainedReturnCallError::AfterTransfer(CallError::Timeout),
            ))
            | Err(BufferExecutionError::Export(CallError::Timeout)) => {
                if awaiting_completion {
                    let elapsed = now.saturating_sub(submitted.unwrap_or(now));
                    let slice = self.pace.slice(
                        program,
                        elapsed,
                        self.policy.completion_ns,
                        PACE_CEILING_NS,
                    );
                    self.wait_notice(slice).await.map_err(|e| self.fail(e))?;
                }
                Ok(())
            }
            Err(error) => Err(self.fail(error)),
        }
    }
}

/// A MoE instruction whose experts the sealed wave packages compute:
/// NVFP4 blocks of their shape, squared ReLU.
fn wave_plan_matches(plan: &CpuKernelPlan, params: MoeParams) -> bool {
    params.is_valid()
        && params.weights == axnn::DType::U8
        && params.activation == axnn::lowering::ExpertActivation::SquaredRelu
        && params.hidden as usize == xe_moe::MOE_HIDDEN
        && params.intermediate as usize == xe_moe::MOE_INTERMEDIATE
        && plan.semantic == axnn::lowering::SemanticOp::MoeExperts
        && plan.params() == axnn::lowering::KernelParams::MoeExperts(params)
}

#[cfg(test)]
mod tests {
    use super::{
        KERNEL_BUFFER_COUNT, PIPELINE_DEPTH, ResidentScratch, ResidentSlots, scratch_slot,
        shared_span,
    };

    #[test]
    fn recurrent_targets_are_bounded_and_retry_identity_is_exact() {
        use super::{Placement, ResidentState, recurrent_target_matches};
        use axnn_cpu::StateTargetRef;
        #[repr(align(4096))]
        struct State([f32; 2048]);
        let mut storage = State([0.0; 2048]);
        let base = storage.0.as_mut_ptr().cast::<u8>();
        // Geometry fixture only; no device uses this synthetic buffer ID.
        let region = unsafe { ResidentState::new(7, base, 4096) }.unwrap();
        let first = StateTargetRef::new(&mut storage.0[..16]);
        let placement = Placement::Resident {
            buffer: 7,
            offset: 0,
        };
        assert_eq!(region.resolve(first), Some(placement));
        assert!(recurrent_target_matches(
            Some((placement, 64)),
            Some(first),
            Some(&region)
        ));
        assert!(!recurrent_target_matches(
            Some((placement, 128)),
            Some(first),
            Some(&region)
        ));
        assert!(!recurrent_target_matches(
            Some((placement, 64)),
            Some(first),
            None
        ));
        assert!(!recurrent_target_matches(
            Some((
                Placement::Resident {
                    buffer: 8,
                    offset: 0
                },
                64
            )),
            Some(first),
            Some(&region)
        ));
        let other = StateTargetRef::new(&mut storage.0[16..32]);
        assert!(!recurrent_target_matches(
            Some((placement, 64)),
            Some(other),
            Some(&region)
        ));
        assert_eq!(
            region.resolve(StateTargetRef::new(&mut storage.0[1..17])),
            None
        );
        assert_eq!(
            region.resolve(StateTargetRef::new(&mut storage.0[..15])),
            None
        );
        assert_eq!(
            region.resolve(StateTargetRef::new(&mut storage.0[1008..1040])),
            None
        );
        assert_eq!(
            region.resolve(StateTargetRef::new(&mut storage.0[1024..1040])),
            None
        );
        assert_eq!(
            region.resolve(StateTargetRef::new(&mut storage.0[..0])),
            None
        );
        assert!(unsafe { ResidentState::new(0, base, 4096) }.is_none());
        assert!(unsafe { ResidentState::new(7, base, 4095) }.is_none());
        assert!(unsafe { ResidentState::new(7, base.wrapping_add(4), 4096) }.is_none());
        assert!(unsafe { ResidentState::new(7, core::ptr::null_mut(), 4096) }.is_none());
        assert!(unsafe { ResidentState::new(7, base, usize::MAX) }.is_none());
    }

    /// Exercise production admission/retirement with synthetic device writes.
    /// This proves placement and ownership bookkeeping, not hardware execution.
    #[test]
    fn recurrent_direct_and_staged_targets_retire_and_cancel_without_early_publication() {
        use super::*;
        use axnn::lowering::{
            CausalConv1dParams, EpilogueActivation, KernelRequest, ResourceKey, StateBinding,
        };
        use axnn::{DType, Shape, TensorDesc};
        use axnn_cpu::StateTargetRef;
        use kore_abi::session::SlotRef;
        use kore_rt::svc::{ClientEndpoints, REP_CAP, REQ_CAP, SvcClient};
        use kore_svc_accel::wire::AccelWire;
        // Sealed convolution geometry; scratch holds its source + output and
        // fallback target below the executor's ones region.
        const CHANNELS: usize = 6144;
        const STATE: usize = CHANNELS * 3;
        const SCRATCH: usize = 384 * 1024;
        const RING: usize = kore_rt::svc::window_bytes::<AccelWire>(REQ_CAP, REP_CAP);
        #[repr(align(4096))]
        struct Scratch([u8; SCRATCH]);
        #[repr(align(4096))]
        struct States([f32; STATE]);
        #[repr(align(4096))]
        struct Inputs([f32; CHANNELS * 6]);
        #[repr(align(64))]
        struct Ring([u8; RING]);
        let mut scratch = Scratch([0; SCRATCH]);
        let mut states = States([0.0; STATE]);
        let inputs = Inputs([0.0; CHANNELS * 6]);
        let source = [0.0; STATE];
        let mut output = [0.0; CHANNELS];
        let mut stage = [0u8; 4096];
        let mut ring = Ring([0; RING]);
        let (rings, mut server) =
            kore_rt::svc::split_window::<AccelWire>(&mut ring.0, REQ_CAP, REP_CAP).unwrap();
        let transport = &TrapTransport::new();
        let mut client = RingAccel {
            client: SvcClient::new(
                transport,
                rings,
                ClientEndpoints {
                    kick: SlotRef::from_wire(1),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            ),
            return_tx: SlotRef::from_wire(2),
            return_land: SlotRef::from_wire(3),
        };
        const TEST_TRANSPORT: TrapTransport = TrapTransport::new();
        let mut completion = NoticeRx::new(&TEST_TRANSPORT, SlotRef::from_wire(4));
        let binding = StateBinding {
            source: ResourceKey {
                group: 1,
                binding: 0,
            },
            target: ResourceKey {
                group: 1,
                binding: 1,
            },
        };
        let params = CausalConv1dParams::new_activated(
            binding,
            CHANNELS as u32,
            4,
            EpilogueActivation::Silu,
        )
        .unwrap();
        let desc = |shape| TensorDesc::try_new(Shape::from(shape), DType::F32).unwrap();
        let request = KernelRequest::causal_conv1d(
            &desc([1, CHANNELS]),
            &desc([CHANNELS, 4]),
            &TensorDesc::try_new(Shape::from([CHANNELS]), DType::F32).unwrap(),
            params,
        )
        .unwrap();
        let plan =
            axnn_cpu::plan_cpu_kernel(&request, axnn_cpu::CpuCapabilities::compiled()).unwrap();
        let kernel = Kernel::from_plan(&plan, KernelAccuracy::Approximate).unwrap();
        let programs = [LoadedKernel {
            artifact: kernel.artifact(),
            program: 1,
        }];
        let state_base = states.0.as_mut_ptr().cast::<u8>();
        // SAFETY: all fixture buffers stay allocated until this executor is
        // dropped, and synthetic writes below have the sole access while pending.
        let (scratch, inputs_region, retained, stage) = unsafe {
            (
                ResidentScratch::new(1, scratch.0.as_mut_ptr(), SCRATCH).unwrap(),
                SharedInputs::resident(2, NativeRef::F32(&inputs.0).as_bytes()).unwrap(),
                ResidentState::new(3, state_base, core::mem::size_of_val(&states.0)).unwrap(),
                SequenceStage::new(0, stage.as_mut_ptr(), stage.len()).unwrap(),
            )
        };
        let mut executor = Executor::new_batched(
            &mut client,
            &mut completion,
            1,
            1,
            &programs,
            inputs_region,
            scratch,
            stage,
            WaitPolicy {
                job_ns: 1,
                call_ns: 1,
                completion_ns: 1,
                operation_ns: 1,
            },
        )
        .unwrap()
        .with_recurrent_state(retained);
        executor.enable_profiling();
        let operands = [
            Some(ChainedInput::Host(NativeRef::F32(&inputs.0[..CHANNELS]))),
            Some(ChainedInput::Host(NativeRef::F32(
                &inputs.0[CHANNELS..CHANNELS * 5],
            ))),
            Some(ChainedInput::Host(NativeRef::F32(
                &inputs.0[CHANNELS * 5..],
            ))),
        ];
        let target = StateTargetRef::new(&mut states.0);
        for direct in [true, false] {
            // No admission is live here. Distinguish each result from the
            // previous pass so a missing fallback copy cannot look correct.
            unsafe { target.values() }.fill(-1.0);
            if !direct {
                executor.recurrent_state = None;
            }
            assert_eq!(
                executor
                    .batched_recurrent(
                        0,
                        &plan,
                        &operands,
                        ax_exec::RecurrentState {
                            source: &source,
                            target,
                            tree: None,
                        },
                        &mut output,
                        KernelAccuracy::Approximate
                    )
                    .unwrap(),
                KernelProgress::Pending
            );
            let batch = executor.batch.as_mut().unwrap();
            let member = *batch.window.find(MemberKey::Instruction(0)).unwrap().1;
            let (placement, bytes) = member.recurrent.unwrap();
            assert_eq!(bytes, (STATE * 4) as u64);
            let bindings = member
                .kernel
                .port_bindings(xe_batch::place(&member, batch.scratch.buffer, batch.ones))
                .unwrap();
            assert_eq!(
                bindings.iter().nth(4).unwrap().buffer,
                if direct { 3 } else { 1 }
            );
            batch.window.submit_staged();
            assert!(
                !batch.window.discard_staged(),
                "in-flight targets stay owned"
            );
            // No actual job is sent by this fixture. Write exactly the target
            // selected by production placement, as a completed device would.
            let destination = match placement {
                Placement::Resident {
                    buffer: 3,
                    offset: 0,
                } => target.as_mut_ptr(),
                Placement::Scratch { offset } => unsafe {
                    batch.scratch.base.add(offset as usize).cast::<f32>()
                },
                _ => panic!("unexpected recurrent placement"),
            };
            unsafe {
                core::slice::from_raw_parts_mut(destination, STATE).fill(7.0);
            }
            assert_eq!(
                executor
                    .batched_retire_into(MemberKey::Instruction(0), &mut output, Some(target))
                    .unwrap(),
                Some(KernelProgress::Pending)
            );
            if !direct {
                assert!(
                    unsafe { target.values() }
                        .iter()
                        .all(|&value| value == -1.0)
                );
            }
            assert_eq!(executor.completed(), u64::from(!direct));
            assert_eq!(executor.profile().unwrap().state_copy.calls, 0);
            executor.batch.as_mut().unwrap().window.complete_submitted();
            assert_eq!(
                executor
                    .batched_retire_into(MemberKey::Instruction(0), &mut output, Some(target))
                    .unwrap(),
                Some(KernelProgress::Complete)
            );
            assert!(unsafe { target.values() }.iter().all(|&value| value == 7.0));
            assert_eq!(
                executor.profile().unwrap().state_copy.calls,
                u64::from(!direct)
            );
        }
        let profile = executor.profile().unwrap();
        assert_eq!(profile.direct_state_bytes, (STATE * 4) as u64);
        assert_eq!(profile.copied_state_bytes, (STATE * 4) as u64);
        // An abandoned, never-submitted state admission can be discarded
        // without changing the target or counting it as a completed write.
        executor.recurrent_state = unsafe { ResidentState::new(3, state_base, STATE * 4) };
        assert_eq!(
            executor
                .batched_recurrent(
                    1,
                    &plan,
                    &operands,
                    ax_exec::RecurrentState {
                        source: &source,
                        target,
                        tree: None,
                    },
                    &mut output,
                    KernelAccuracy::Approximate
                )
                .unwrap(),
            KernelProgress::Pending
        );
        assert!(executor.batch.as_mut().unwrap().window.discard_staged());
        assert!(unsafe { target.values() }.iter().all(|&value| value == 7.0));
        assert_eq!(
            executor.profile().unwrap().direct_state_bytes,
            (STATE * 4) as u64
        );
        drop(executor);
        assert!(
            server.req.pop().is_none(),
            "fixture must never submit a real GPU request"
        );
    }

    /// Production MoE admission plus a real dependent map: the output binding
    /// names the combine's tensor, and skipping copy-back preserves its scratch
    /// until that consumer retires. Device writes are simulated, no IPC mocked.
    #[test]
    fn expert_output_chains_into_gpu_consumer_without_host_publication() {
        use super::*;
        use alloc::vec;
        use axnn::lowering::KernelRequest;
        use axnn::{BinaryOp, DType, Shape, TensorDesc};
        use kore_abi::session::SlotRef;
        use kore_rt::svc::{ClientEndpoints, REP_CAP, REQ_CAP, SvcClient};
        use kore_svc_accel::wire::AccelWire;
        #[repr(align(4096))]
        #[derive(Clone)]
        struct Page([u8; 4096]);
        #[repr(align(4096))]
        struct Scratch([u8; 384 * 1024]);
        const RING: usize = kore_rt::svc::window_bytes::<AccelWire>(REQ_CAP, REP_CAP);
        #[repr(align(64))]
        struct Ring([u8; RING]);
        let pages = vec![Page([0; 4096]); xe_moe::SECTIONS.used.div_ceil(4096)];
        // SAFETY: contiguous initialized pages, borrowed immutably throughout
        // the fixture. No live device exists in this test.
        let archive =
            unsafe { core::slice::from_raw_parts(pages[0].0.as_ptr(), pages.len() * 4096) };
        let codes = xe_moe::MOE_INTERMEDIATE * xe_moe::MOE_HIDDEN / 2;
        let scales = codes / 8;
        let experts = [axnn_cpu::ExpertWeights {
            up: &archive[..codes],
            down: &archive[xe_moe::SECTIONS.down_codes..][..codes],
            up_block_scales: &archive[xe_moe::SECTIONS.up_scales..][..scales],
            down_block_scales: &archive[xe_moe::SECTIONS.down_scales..][..scales],
            up_scale: 1.0,
            down_scale: 1.0,
        }];
        let width = xe_moe::MOE_HIDDEN;
        let desc = TensorDesc::try_new(Shape::from([width]), DType::F32).unwrap();
        let request = KernelRequest::binary(BinaryOp::Add, &desc, &desc).unwrap();
        let plan =
            axnn_cpu::plan_cpu_kernel(&request, axnn_cpu::CpuCapabilities::compiled()).unwrap();
        let kernel = Kernel::from_plan(&plan, KernelAccuracy::Approximate).unwrap();
        let mut programs: alloc::vec::Vec<_> = MoeWave::ALL
            .into_iter()
            .enumerate()
            .map(|(i, part)| LoadedKernel {
                artifact: kore_xe_kernels::moe_wave(part).unwrap(),
                program: 1 + i as u64,
            })
            .collect();
        programs.push(LoadedKernel {
            artifact: kernel.artifact(),
            program: 4,
        });
        let hidden = vec![0.0; width];
        let mut host_output = vec![-9.0; width];
        let mut follower_output = vec![-7.0; width];
        let routes = [axnn_cpu::MoeBatchRow::new(0, 0, 1.0)];
        let mut scratch = Scratch([0; 384 * 1024]);
        let mut table = [0u8; 4096];
        let mut ring = Ring([0; RING]);
        let (rings, _server) =
            kore_rt::svc::split_window::<AccelWire>(&mut ring.0, REQ_CAP, REP_CAP).unwrap();
        let transport = TrapTransport::new();
        let mut client = RingAccel {
            client: SvcClient::new(
                &transport,
                rings,
                ClientEndpoints {
                    kick: SlotRef::from_wire(1),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            ),
            return_tx: SlotRef::from_wire(2),
            return_land: SlotRef::from_wire(3),
        };
        const TEST_TRANSPORT: TrapTransport = TrapTransport::new();
        let mut completion = NoticeRx::new(&TEST_TRANSPORT, SlotRef::from_wire(4));
        // SAFETY: fixture storage stays allocated until the executor is dropped;
        // the test alone simulates device writes into its retained scratch.
        let mut executor = unsafe {
            Executor::new_batched(
                &mut client,
                &mut completion,
                1,
                1,
                &programs,
                SharedInputs::resident(2, archive).unwrap(),
                ResidentScratch::new(1, scratch.0.as_mut_ptr(), scratch.0.len()).unwrap(),
                SequenceStage::new(0, table.as_mut_ptr(), table.len()).unwrap(),
                WaitPolicy {
                    job_ns: 1,
                    call_ns: 1,
                    completion_ns: 1,
                    operation_ns: 1,
                },
            )
        }
        .unwrap();
        executor.enable_profiling();
        assert_eq!(
            executor
                .batched_wave(
                    0,
                    0,
                    &hidden,
                    &routes,
                    &experts,
                    &mut host_output,
                    KernelAccuracy::Approximate
                )
                .unwrap(),
            KernelProgress::Pending
        );
        assert!(executor.moe_output_pending(0, 0));
        assert!(!executor.moe_output_pending(0, 1));
        let inputs = [
            Some(ChainedInput::Pending {
                instruction: 0,
                offset: 0,
                elements: width,
            }),
            Some(ChainedInput::Host(NativeRef::F32(&hidden))),
            None,
        ];
        assert_eq!(
            executor
                .advance_chained(
                    1,
                    &plan,
                    &inputs,
                    &mut follower_output,
                    KernelAccuracy::Approximate
                )
                .unwrap(),
            KernelProgress::Pending
        );
        let batch = executor.batch.as_mut().unwrap();
        assert_eq!(batch.window.len(), 4);
        let producer = *batch.window.producer(0).unwrap();
        let follower = *batch.window.producer(1).unwrap();
        assert_eq!(
            follower.ports[0],
            Placement::Scratch {
                offset: producer.output
            }
        );
        assert!(follower.barrier);
        assert_eq!(batch.window.submit_first(3), 3);
        assert_eq!(batch.window.submit_first(1), 1);
        assert_eq!(
            executor.retire_moe_unobserved(0, 0, 1).unwrap(),
            KernelProgress::Pending
        );
        let batch = executor.batch.as_mut().unwrap();
        // SAFETY: no actual GPU accesses this fixture. Complete only the
        // producer first, leaving the dependent consumer's storage retained.
        unsafe {
            core::slice::from_raw_parts_mut(
                batch
                    .scratch
                    .base
                    .add(follower.output as usize)
                    .cast::<f32>(),
                width,
            )
            .fill(2.0);
        }
        batch.window.complete_first(3);
        assert_eq!(
            executor.retire_moe_unobserved(0, 0, 1).unwrap(),
            KernelProgress::Complete
        );
        assert!(host_output.iter().all(|&value| value == -9.0));
        assert_eq!(
            executor.profile().unwrap().avoided_output_bytes,
            (width * 4) as u64
        );
        assert_eq!(
            executor
                .advance_chained(
                    1,
                    &plan,
                    &inputs,
                    &mut follower_output,
                    KernelAccuracy::Approximate
                )
                .unwrap(),
            KernelProgress::Pending
        );
        executor.batch.as_mut().unwrap().window.complete_first(1);
        assert_eq!(
            executor
                .advance_chained(
                    1,
                    &plan,
                    &inputs,
                    &mut follower_output,
                    KernelAccuracy::Approximate
                )
                .unwrap(),
            KernelProgress::Complete
        );
        assert!(follower_output.iter().all(|&value| value == 2.0));
        assert_eq!(executor.batch.as_ref().unwrap().window.arena.used(), 0);
    }

    /// The first wave boot failed its whole submission: unused chunk slots
    /// repeated chunk 0, and the provider refuses overlapping bindings.
    #[test]
    fn a_wave_with_unused_chunk_slots_binds_no_range_twice() {
        use super::{
            Kernel, MAX_PORTS, MoeWave, Placement, UNUSED_CHUNK_BYTES, placements_overlap, xe_moe,
        };
        let ones = 1 << 20;
        let chunks = [Some((7, 512 << 20)), None, None, None, None];
        let kernel = Kernel::moe_wave(
            MoeWave::Up,
            4,
            4096,
            chunks.map(|chunk| chunk.map_or(UNUSED_CHUNK_BYTES, |(_, bytes)| bytes)),
        )
        .unwrap();
        let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
        for (slot, chunk) in chunks.iter().enumerate() {
            ports[slot] = match *chunk {
                Some((buffer, _)) => Placement::Resident { buffer, offset: 0 },
                None => Placement::Scratch {
                    offset: ones + slot as u64 * UNUSED_CHUNK_BYTES,
                },
            };
        }
        ports[xe_moe::CHUNK_SLOTS] = Placement::Scratch { offset: 2 << 20 };
        assert!(!placements_overlap(&kernel, &ports, ones));
        let mut repeated = ports;
        repeated[1] = repeated[0];
        assert!(placements_overlap(&kernel, &repeated, ones));
        // The same offsets in different buffers do not overlap.
        let mut other = ports;
        other[1] = Placement::Resident {
            buffer: 8,
            offset: 0,
        };
        assert!(!placements_overlap(&kernel, &other, ones));
    }

    #[test]
    fn wall_observations_distinguish_zero_elapsed_from_unavailable_clocks() {
        let mut profile = super::Profile::default();
        profile.import.record(Some(10), Some(20));
        let before = profile;
        profile.import.record(Some(20), Some(20));
        profile.import.record(None, Some(30));
        profile.import.record(Some(30), None);
        profile.import.record(Some(40), Some(39));
        profile.execute.record(Some(50), Some(75));
        let delta = profile.since(before);
        assert_eq!(
            delta.import,
            super::WallTime {
                ns: 0,
                calls: 4,
                unclocked: 3
            }
        );
        assert_eq!(
            delta.execute,
            super::WallTime {
                ns: 25,
                calls: 1,
                unclocked: 0
            }
        );
        assert_eq!(delta.notice, super::WallTime::default());
    }

    #[test]
    fn state_copy_wall_time_deltas_keep_unclocked_calls() {
        let mut profile = super::Profile::default();
        profile.state_copy.record(Some(10), Some(17));
        let before = profile;
        profile.state_copy.record(Some(100), Some(135));
        profile.state_copy.record(None, Some(150));
        let delta = profile.since(before);
        assert_eq!(
            delta.state_copy,
            super::WallTime {
                ns: 35,
                calls: 2,
                unclocked: 1,
            }
        );
        assert_eq!(delta.retire, super::WallTime::default());
        assert_eq!(delta.execute, super::WallTime::default());
        assert_eq!(before.since(profile).state_copy, super::WallTime::default());
    }

    #[test]
    fn resident_inputs_leave_large_scratch_available_for_the_head() {
        let capacities = [128 << 10, 192 << 10, 4096, 4096, 4 << 20];
        let mut available = [true; 5];
        // Activation, scale parameter, hidden output: preserve the large head
        // allocation while consuming distinct available leases.
        for (required, expected) in [(86 << 10, 0), (4, 2), (86 << 10, 1)] {
            let slot = scratch_slot(&capacities, &available, required).unwrap();
            assert_eq!(slot, expected);
            available[slot] = false;
        }
        assert_eq!(scratch_slot(&capacities, &available, 4 << 20), Some(4));
        available[4] = false;
        assert_eq!(scratch_slot(&capacities, &available, 86 << 10), None);
    }

    /// The first wait sleeps three quarters of the expected time, less what
    /// already elapsed, never under the floor or over the ceiling; unknown
    /// programs wait the floor; the estimate moves a quarter per sample.
    #[test]
    fn pacing_waits_the_expected_time_then_a_quarter_per_retry() {
        let mut pace = super::Pace::default();
        assert_eq!(pace.slice(7, 0, 1_000, 100_000), 1_000);
        pace.record(7, 40_000);
        assert_eq!(pace.expected(7), Some(40_000));
        assert_eq!(pace.slice(7, 0, 1_000, 100_000), 40_000);
        assert_eq!(pace.slice(7, 29_500, 1_000, 100_000), 10_500);
        assert_eq!(pace.slice(7, 39_500, 1_000, 100_000), 10_000);
        assert_eq!(pace.slice(7, 45_000, 1_000, 100_000), 10_000);
        assert_eq!(pace.slice(7, 0, 1_000, 20_000), 20_000);
        pace.record(7, 80_000);
        assert_eq!(pace.expected(7), Some(50_000));
        pace.record(0, 5);
        assert_eq!(pace.expected(0), None, "program zero is never paced");
    }

    #[repr(align(4096))]
    struct Region([u8; 6 * 4096]);

    /// State/control staging must not overwrite adjacent arena allocations;
    /// a short binding refuses before consuming scratch or writing bytes.
    #[test]
    fn batched_staging_keeps_control_after_payload_with_bounded_padding() {
        use super::{Batched, Executor, SequenceStage, WINDOW, Window};
        let mut region = Region([0xa5; 6 * 4096]);
        let mut table = [0u8; 64];
        // SAFETY: these local mappings outlive the batch, which has no GPU
        // submission. Only batched_stage writes the scratch during its life.
        let (scratch, stage) = unsafe {
            (
                ResidentScratch::new(7, region.0.as_mut_ptr(), region.0.len()).unwrap(),
                SequenceStage::new(0, table.as_mut_ptr(), table.len()).unwrap(),
            )
        };
        let mut batch = Batched {
            window: Window::new(4096),
            scratch,
            ones: 4096,
            stage,
            flights: core::array::from_fn(|_| None),
            flight_count: 0,
            request_owner: None,
            overlap: false,
            pipeline_depth: PIPELINE_DEPTH,
            sequence_limit: WINDOW,
        };
        let payload = [0x3c; 96];
        let shape = [0, 2, 0, 0, 1, 0, 0, 0]; // 512 tokens, one state
        let bytes = payload.len() + shape.len();
        assert_eq!(
            Executor::batched_stage(&mut batch, &payload, &shape, bytes as u64 - 1),
            None
        );
        assert_eq!(
            Executor::batched_stage(&mut batch, &payload, &shape, bytes as u64 + 16),
            Some(0)
        );
        let next = Executor::batched_stage(&mut batch, &[0x5a; 4], &[], 4).unwrap() as usize;
        drop(batch);
        assert_eq!(&region.0[..payload.len()], &payload);
        assert_eq!(&region.0[payload.len()..bytes], &shape);
        assert!(region.0[bytes..bytes + 16].iter().all(|&b| b == 0));
        assert!(region.0[bytes + 16..next].iter().all(|&b| b == 0xa5));
        assert_eq!(&region.0[next..next + 4], &[0x5a; 4]);
        assert!(region.0[next + 4..].iter().all(|&b| b == 0xa5));
    }

    /// Resident scratch carves one slot per capacity, in order, at page
    /// offsets, and refuses a region the capacities do not fit.
    #[test]
    fn resident_scratch_carves_a_slot_per_capacity_in_order() {
        let mut region = Region([0; 6 * 4096]);
        let base = region.0.as_mut_ptr();
        let capacities = [4096, 8192, 4096, 4096, 4096];
        // SAFETY: construction only records the geometry.
        let scratch = unsafe { ResidentScratch::new(7, base, 6 * 4096) }.unwrap();
        let slots = ResidentSlots::carve(scratch, &capacities).unwrap();
        assert_eq!(slots.offsets, [0, 4096, 12288, 16384, 20480]);
        assert_eq!(slots.taken, [false; KERNEL_BUFFER_COUNT]);
        // SAFETY: as above.
        let short = unsafe { ResidentScratch::new(7, base, 5 * 4096) }.unwrap();
        assert!(ResidentSlots::carve(short, &capacities).is_none());
    }

    #[test]
    fn resident_scratch_refuses_unnamed_or_unaligned_regions() {
        let mut region = Region([0; 6 * 4096]);
        let base = region.0.as_mut_ptr();
        // SAFETY: every call only validates the geometry.
        unsafe {
            assert!(ResidentScratch::new(0, base, 4096).is_none());
            assert!(ResidentScratch::new(7, core::ptr::null_mut(), 4096).is_none());
            assert!(ResidentScratch::new(7, base.add(64), 4096).is_none());
            assert!(ResidentScratch::new(7, base, 4000).is_none());
            assert!(ResidentScratch::new(7, base, 0).is_none());
            assert!(ResidentScratch::new(7, base, 4096).is_some());
        }
    }

    #[test]
    fn read_scopes_keep_interiors_and_refuse_unowned_page_edges() {
        assert_eq!(
            shared_span(0x10000, 0x3000, 0x10390, 0x1a0, 0x1a0),
            Some((0, 4096, 0x390))
        );
        assert_eq!(
            shared_span(0x10000, 0x3000, 0x10ffe, 2, 4),
            Some((0, 8192, 0xffe))
        );
        assert_eq!(shared_span(0x10000, 0x2fff, 0x12ffe, 1, 1), None);
        assert_eq!(shared_span(0x10000, 0x3000, 0xffff, 1, 1), None);
        assert_eq!(shared_span(0x10000, usize::MAX, usize::MAX, 1, 8), None);
        assert_eq!(shared_span(0x10000, 0x3000, 0x10390, 8, 4), None);
    }

    /// A frozen submission encodes its pre-bound kernels into exactly one
    /// ordered sequence table — one submit, one completion — carrying each
    /// kernel's program, launch and caller-resolved bindings in order.
    #[test]
    fn frozen_submission_encodes_one_ordered_sequence_with_one_completion() {
        use super::{frozen_ordered_launch, FrozenLaunch, MAX_PORTS, Placement, ProgramReuse, stage_ordered_launches};
        use super::{Kernel, OrderedExecution};
        use alloc::vec::Vec;
        use axnn::lowering::{KernelAccuracy, KernelRequest};
        use axnn::{BinaryOp, Shape, TensorDesc};

        let add = |elements: usize| {
            let desc = TensorDesc::try_new(Shape::from([elements]), axnn::DType::F32).unwrap();
            let request = KernelRequest::binary(BinaryOp::Add, &desc, &desc).unwrap();
            Kernel::select(&request, KernelAccuracy::Approximate).unwrap()
        };
        // Two 8-element adds; each binds input0, input1, its staged control
        // block, then its output, all in the same scratch buffer at disjoint
        // 64-byte-grouped offsets.
        let launch_at = |offset: u64, kernel: Kernel| {
            let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
            ports[0] = Placement::Scratch { offset };
            ports[1] = Placement::Scratch { offset: offset + 64 };
            ports[2] = Placement::Scratch { offset: offset + 128 };
            ports[3] = Placement::Scratch { offset: offset + 192 };
            FrozenLaunch {
                kernel,
                ports,
                barrier: true,
                gate: None,
                signal: None,
            }
        };
        let first = frozen_ordered_launch(9, 1 << 20, 5, 1_000, 7, &launch_at(0, add(8)), false)
            .unwrap();
        let second = frozen_ordered_launch(
            9,
            1 << 20,
            5,
            1_000,
            7,
            &launch_at(256, add(8)),
            true,
        )
        .unwrap();
        assert_eq!(first.program, 7);
        assert!(!first.barrier, "the first member follows nothing in the table");
        assert!(second.barrier, "the second member waits on the first's writes");
        let slots: Vec<_> = first
            .bindings
            .iter()
            .map(|b| (b.buffer, b.offset, b.len, b.slot))
            .collect();
        assert_eq!(
            slots,
            [
                (9, 0, 32, 0),
                (9, 64, 32, 1),
                (9, 128, 8, 2), // the staged control block
                (9, 192, 32, 3),
            ]
        );
        // One table for both members: the encoder splits only at the entry
        // ceiling or a queue/timeout change, neither of which applies.
        let mut staging = [0u8; 2048];
        let mut groups = [None; 2];
        let tables = stage_ordered_launches(
            &[first, second],
            &mut staging,
            0,
            &mut groups,
            ProgramReuse::Share,
        )
        .unwrap();
        assert_eq!(tables, 1, "one table, so one completion fence");
        let table: OrderedExecution = groups[0].expect("the single table");
        assert_eq!(table.count, 2, "both kernels ride the one submission");
        assert_eq!(
            table.len as usize,
            2 * core::mem::size_of::<kore_abi::svc::AccelSequenceEntryWire>(),
        );
        assert!(groups[1].is_none());
    }

    /// Every lane's window takes the protocol's whole table, and a full
    /// window stages as one table: one submit, one completion.
    #[test]
    fn a_window_carries_the_whole_protocol_table() {
        use super::{
            FrozenLaunch, MAX_PORTS, Placement, ProgramReuse, WINDOW, frozen_ordered_launch,
            stage_ordered_launches,
        };
        use axnn::lowering::{KernelAccuracy, KernelRequest};
        use axnn::{BinaryOp, Shape, TensorDesc};

        assert_eq!(WINDOW, kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES);

        let desc = TensorDesc::try_new(Shape::from([8]), axnn::DType::F32).unwrap();
        let request = KernelRequest::binary(BinaryOp::Add, &desc, &desc).unwrap();
        let kernel = super::Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
        let members: alloc::vec::Vec<_> = (0..WINDOW as u64)
            .map(|index| {
                let base = index * 256;
                let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
                for (slot, port) in ports.iter_mut().take(4).enumerate() {
                    *port = Placement::Scratch {
                        offset: base + slot as u64 * 64,
                    };
                }
                let launch = FrozenLaunch {
                    kernel,
                    ports,
                    barrier: true,
                    gate: None,
                    signal: None,
                };
                frozen_ordered_launch(9, 1 << 20, 5, 1_000, 7, &launch, index > 0).unwrap()
            })
            .collect();
        let mut staging =
            [0u8; WINDOW * core::mem::size_of::<kore_abi::svc::AccelSequenceEntryWire>()];
        let mut groups = [None; WINDOW];
        let tables =
            stage_ordered_launches(&members, &mut staging, 0, &mut groups, ProgramReuse::Share)
                .unwrap();
        assert_eq!(tables, 1);
        assert_eq!(groups[0].unwrap().count as usize, WINDOW);
    }

    /// The frozen resolution maps an archive-bound weight input to its named
    /// resident buffer and a scratch activation (as well as the staged control
    /// block and the written output) to the scratch buffer: the encoder binds
    /// each placement's own buffer, copying nothing and staging nothing.
    #[test]
    fn frozen_resolution_binds_weights_resident_and_activations_scratch() {
        use super::{frozen_ordered_launch, FrozenLaunch, Kernel, MAX_PORTS, Placement, Port};
        use axnn::lowering::{KernelAccuracy, KernelRequest};
        use axnn::{BinaryOp, Shape, TensorDesc};

        // Two 8-element F32 operands. Input 0 stands for a resident weight, a
        // projection's operand taken straight from the imported archive; input
        // 1 for a scratch activation staged in the caller's working buffer.
        let desc = TensorDesc::try_new(Shape::from([8]), axnn::DType::F32).unwrap();
        let kernel = Kernel::select(
            &KernelRequest::binary(BinaryOp::Add, &desc, &desc).unwrap(),
            KernelAccuracy::Approximate,
        )
        .unwrap();
        assert_eq!(
            kernel.ports(),
            &[
                Port::Input(0),
                Port::Input(1),
                Port::Parameter,
                Port::Output
            ][..]
        );
        let archive = 0x7f00_0000_0011;
        let mut ports = [Placement::Scratch { offset: 0 }; MAX_PORTS];
        ports[0] = Placement::Resident {
            buffer: archive,
            offset: 0x1000,
        };
        ports[1] = Placement::Scratch { offset: 64 };
        ports[2] = Placement::Scratch { offset: 128 };
        ports[3] = Placement::Scratch { offset: 192 };
        let launch = FrozenLaunch {
            kernel,
            ports,
            barrier: true,
            gate: None,
            signal: None,
        };
        let member = frozen_ordered_launch(9, 1 << 20, 5, 1_000, 7, &launch, false).unwrap();
        let bindings: alloc::vec::Vec<(u64, u64)> = member
            .bindings
            .iter()
            .map(|binding| (binding.buffer, binding.offset))
            .collect();
        assert_eq!(
            &bindings[..],
            &[(archive, 0x1000), (9, 64), (9, 128), (9, 192)][..],
            "the weight binds the archive, everything else the scratch buffer"
        );
    }
}
