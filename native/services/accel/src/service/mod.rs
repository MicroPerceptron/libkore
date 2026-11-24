//! The accel vocabulary, the backend-generic service host, and its
//! capability arms (staging attach, completion-signal mint).
//!
//! [`AccelService<A>`] is **one serve body over one seam**: every verb
//! funnels through [`AccelService::step`] into the hosted engine
//! ([`state`]), and a backend swap is an endowment change, never a
//! protocol change (`res/USERLAND.md` §5.9 criterion 5).  The concrete
//! rows — `AccelService<IvpuBackend>` first — each add one
//! `SystemService` impl beside their endowment; the serve body is this
//! module's and only this module's.

mod staging;
pub use staging::PEER_RING_BYTES;
mod peers;
mod state;

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::session::{Notice, SlotRef};
#[allow(deprecated)]
use kore_driver::accel::Dispatch;
use kore_driver::accel::{LaunchShape, LossEvidence};
use kore_driver::accel::NativeAccelerator;
use kore_rt::cap::{CHANNEL_ENDPOINT_RIGHTS, ChanSendToken, ChannelRoot, ChannelSpaceToken, verbs};
use kore_rt::chan::ChanTx;
use kore_rt::service::{GrantArg, PeerClose, ReturnMode, Serve};
use kore_rt::transport::Transport;
#[cfg(test)]
use virtio_accel_core::Accelerator;
use virtio_accel_core::{AccessMode, DeviceInfo, DeviceLimits};

use crate::COOKIE_ACCEL_DONE;
use state::{Engine, PAIRS};

/// Program bindings one submission can carry on the wire
/// ([`kore_abi::svc::MAX_ACCEL_BINDINGS`], re-cited so protocol users
/// need not reach into the ABI crate).
pub const MAX_ACCEL_BINDINGS: usize = kore_abi::svc::MAX_ACCEL_BINDINGS;

/// Staging attaches one pair may spend — a receiver's map burns an
/// irreversible span of this task's map-window cursor, so re-attaches
/// are counted and bounded rather than free. Admission spends the attempt
/// before mapping; a failed map may already have consumed cursor space.
pub const MAX_ATTACHES: u32 = 4;

/// The return leg carries both staging leaves (`Attach`) and owned memory
/// leases (`ProvisionMemory`). A species-filtered post cannot receive both:
/// the kernel would reject an owned supply before its request reaches the
/// service. Each consuming arm checks the authenticated species and rights
/// before taking ownership; the host retains normal refusal cleanup.
/// Cited by every backend row so the rendezvous contract stays identical.
pub const ACCEL_RETURN_MODE: ReturnMode = ReturnMode::RetainedAny;

/// Compatibility namespace for existing packed NPU-witness clients.
/// The generic host forwards the backend's domain rather than assuming this one.
pub const LOSS_EVIDENCE_DOMAIN: u32 = u32::from_le_bytes(*b"NPUL");

/// The rights a staging-window transfer must carry (and the ones a
/// client attenuates its view to): read *and* write, both sides.
pub const STAGING_RIGHTS: Rights = Rights::ACCESS.union(Rights::MODIFY);

/// An owned provisioning supply must permit CPU/device mapping and retirement.
pub const PROVISION_RIGHTS: Rights = STAGING_RIGHTS.union(Rights::DERIVE).union(Rights::REVOKE);

/// The request kinds [`AccelService::step`] times, as the report names them.
const REQUEST_KINDS: [&str; 4] = ["submit", "poll", "destroy", "other"];
/// The report prints microseconds; the clock counts nanoseconds.
const NS_PER_US: u64 = 1_000;

/// Which [`REQUEST_KINDS`] entry `req` is counted under.
fn request_kind(req: &AccelRequest) -> usize {
    match req {
        AccelRequest::Submit { .. }
        | AccelRequest::SubmitLaunch { .. }
        | AccelRequest::SubmitSequence { .. } => 0,
        AccelRequest::PollEvent { .. } => 1,
        AccelRequest::DestroyEvent { .. } => 2,
        _ => 3,
    }
}

/// One request kind's service time: calls, their sum and the longest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ServiceTime {
    calls: u64,
    ns: u64,
    longest_ns: u64,
}

impl ServiceTime {
    const ZERO: Self = Self {
        calls: 0,
        ns: 0,
        longest_ns: 0,
    };

    fn record(&mut self, ns: u64) {
        self.calls = self.calls.saturating_add(1);
        self.ns = self.ns.saturating_add(ns);
        self.longest_ns = self.longest_ns.max(ns);
    }

    fn average_ns(&self) -> u64 {
        self.ns.checked_div(self.calls).unwrap_or(0)
    }
}

/// One program binding of a [`AccelRequest::Submit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccelBinding {
    /// The bound buffer's object id.
    pub buffer: u64,
    /// Byte offset into the buffer.
    pub offset: u64,
    /// Byte length (non-zero).
    pub len: u64,
    /// The program's binding slot number (artifact-defined meaning).
    pub slot: u32,
    /// The access the program takes — gated host-side against the
    /// buffer's declared usage (Wire ABI §4.4).
    pub access: AccessMode,
}

/// The bounded binding list of one submission: `1..=`
/// [`MAX_ACCEL_BINDINGS`] entries, order-preserving (binding order is
/// not semantic to the device, but the wire keeps it stable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccelBindings {
    entries: [Option<AccelBinding>; MAX_ACCEL_BINDINGS],
}

/// One decoded member of a staged ordered submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccelSequenceEntry {
    pub program: u64,
    pub bindings: AccelBindings,
    /// The member's own launch, admitted as a single
    /// [`AccelRequest::SubmitLaunch`] of the same program would be.
    pub launch: LaunchShape,
    /// Start only after every earlier member's writes are complete and
    /// visible (`kore_driver::accel::OrderedExecution::barrier`).
    pub barrier: bool,
    /// Wait for a host word first (`OrderedExecution::gate`).
    pub gate: Option<AccelSync>,
    /// Write a host word on retirement (`OrderedExecution::signal`).
    pub signal: Option<AccelSync>,
}

/// A gate's or signal's word ([`AccelSequenceEntry::gate`]): the quadword at
/// `offset` in buffer object `buffer`, and the value a gate waits for or a
/// signal writes (`kore_abi::svc::AccelSequenceSyncWire`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccelSync {
    pub buffer: u64,
    pub offset: u64,
    pub value: u64,
}

impl AccelBindings {
    /// A bounded list — `None` when empty or past the wire bound.
    pub fn new(bindings: &[AccelBinding]) -> Option<Self> {
        if bindings.is_empty() || bindings.len() > MAX_ACCEL_BINDINGS {
            return None;
        }
        let mut entries = [None; MAX_ACCEL_BINDINGS];
        for (dst, binding) in entries.iter_mut().zip(bindings) {
            *dst = Some(*binding);
        }
        Some(Self { entries })
    }

    /// The live bindings, in submission order.
    pub fn iter(&self) -> impl Iterator<Item = &AccelBinding> + '_ {
        self.entries.iter().flatten()
    }

    /// How many bindings are live.
    pub fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }

    /// Whether the list is empty (unreachable for a constructed value —
    /// [`new`](Self::new) refuses empties — but the codec's raw path
    /// asks).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The accel request vocabulary — protocol 1.0's object model over the
/// kore rings (one enum, one intake; see the crate docs for the two
/// planes).  Object ids are opaque `u64`s minted by the service;
/// `staging`/`len` pairs address the attached staging window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(deprecated)]
pub enum AccelRequest {
    /// Adopt the staging window transferred over the return channel
    /// (the swap `Attach` shape verbatim); `announced` is the
    /// founder-published return-landing correlation value and carries no
    /// authority itself.
    Attach {
        /// The return-landing correlation value naming the transfer.
        announced: SlotRef,
    },
    /// Transfer a parent lease or bounded fabric route for an exact,
    /// reclaimable staging mapping. The sender may retain a separate CPU
    /// mapping when lending a route. Staging bytes must remain unchanged
    /// from request publication through its reply (including uncertain replies).
    AttachOwned { announced: SlotRef },
    /// Stop using staging and retire its exact mapping and parent. Retry a
    /// Busy reply without touching the old client view or attaching another.
    DetachStaging,
    /// Founder-only authenticated peer bundle component.
    AdmitPeer {
        part: kore_rt::svc::PeerBundlePart,
        announced: SlotRef,
    },
    /// Consume the outer host's actual admission result.
    TakePeerAdmission,
    /// Close this recyclable arrival; never names another peer.
    ClosePeer,
    /// Administration-pair request to retire one exact peer generation.
    RetirePeer { peer: kore_rt::svc::PeerId },
    /// Fence retirement before the launcher reclaims the local backing.
    ConfirmRetired { peer: kore_rt::svc::PeerId },
    /// The device's identity, capabilities, and limits.
    Query,
    /// Create one context (`flags`: protocol 1.0 reserves every
    /// non-empty set).
    CreateContext {
        /// `ContextFlags` bits.
        flags: u32,
    },
    /// Destroy an empty context.
    DestroyContext {
        /// The context id.
        ctx: u64,
    },
    /// Import one whole owned host-DRAM lease as a Shared buffer.
    ImportBuffer {
        ctx: u64,
        bytes: u64,
        align: u64,
        usage: u32,
        announced: SlotRef,
    },
    /// Allocate one provider-owned buffer inside `ctx`.
    CreateBuffer {
        /// The owning context.
        ctx: u64,
        /// Requested bytes (non-zero).
        bytes: u64,
        /// Requested alignment (non-zero power of two).
        align: u64,
        /// Memory domain: 1 host, 2 device, 3 shared.
        domain: u8,
        /// `BufferUsage` bits (non-empty).
        usage: u32,
    },
    /// Transfer an owned supply to fund additional budgets before the first
    /// context request. Unsupported providers refuse this additive operation.
    ProvisionMemory {
        /// Correlated retained-capability landing.
        announced: SlotRef,
        /// Additional client-buffer backing budget.
        buffer_bytes: u64,
        /// Additional resident-program budget.
        program_bytes: u64,
    },
    /// Budget this pair against an existing provider-owned pool. The provider
    /// must refuse when the retained pool cannot cover all live pair budgets;
    /// this request conveys no new backing.
    ReserveMemory {
        buffer_bytes: u64,
        program_bytes: u64,
    },
    /// Release an unreferenced buffer.
    /// Detach an imported buffer and return its allocation as one retained grant.
    ExportBuffer { buffer: u64 },
    FreeBuffer {
        /// The buffer id.
        buffer: u64,
    },
    /// Move `len` bytes from staging `stage` into the buffer at
    /// `offset`.
    WriteBuffer {
        /// The buffer id.
        buffer: u64,
        /// Byte offset into the buffer.
        offset: u64,
        /// Byte offset into the staging window.
        stage: u64,
        /// Byte length (non-zero).
        len: u64,
    },
    /// Move `len` bytes from the buffer at `offset` into staging
    /// `stage`.
    ReadBuffer {
        /// The buffer id.
        buffer: u64,
        /// Byte offset into the buffer.
        offset: u64,
        /// Byte offset into the staging window.
        stage: u64,
        /// Byte length (non-zero).
        len: u64,
    },
    /// Read immutable failure text after loss; independent of device/staging.
    ReadFailure { offset: u64 },
    /// Read the provider's operational telemetry (clock, completions,
    /// completion wakes) as bounded text pages. Offset zero snapshots it;
    /// later offsets page through that snapshot. Answered with the same
    /// [`AccelResponse::Failure`] text page as [`Self::ReadFailure`].
    ReadTelemetry { offset: u64 },
    /// Opt into device timing while this owned context has no retained events.
    /// Enabling resets the interval; disabling preserves its final counters.
    ConfigureProfiling { ctx: u64, enabled: bool },
    /// Copy the newest `len` bytes at most of the provider's firmware log
    /// into the staging window at `stage`. Diagnostic: what the device's
    /// own firmware said about the jobs it ran, for a client that has no
    /// other console.
    ReadLog {
        /// Byte offset into the staging window.
        stage: u64,
        /// Capacity at `stage` (non-zero).
        len: u64,
    },
    /// Create a resident program from the artifact bytes staged at
    /// `stage`.
    LoadProgram {
        /// The owning context.
        ctx: u64,
        /// Byte offset of the artifact in the staging window.
        stage: u64,
        /// Artifact byte length (non-zero).
        len: u64,
        /// The provider-storage budget the resident program may retain.
        resident_bytes: u64,
        /// The artifact format tag (non-zero; [`crate::FORMAT_VPUX_ELF`]…).
        format: u32,
        /// The format's opaque target-identity words.
        target: [u32; 12],
    },
    /// Release an unreferenced resident program.
    UnloadProgram {
        /// The program id.
        program: u64,
    },
    /// Create one accelerator execution queue inside `ctx` (`flags`:
    /// protocol 1.0 reserves every non-empty set).
    CreateQueue {
        /// The owning context.
        ctx: u64,
        /// `QueueFlags` bits.
        flags: u32,
    },
    /// Release an unreferenced execution queue.
    DestroyQueue {
        /// The queue id.
        queue: u64,
    },
    /// Admit one deployed `OP_SUBMIT` execution with its exact historical
    /// `(grid, group, SIMD)` contract.
    ///
    /// Compatibility-only: [`Dispatch::scalar`] is the artifact-defined
    /// sentinel. Every non-scalar value reaches a backend unchanged and is
    /// rejected unless that backend can honor all axes and prove the requested
    /// SIMD width against the compiled program.
    #[deprecated(note = "legacy OP_SUBMIT compatibility only; use SubmitLaunch")]
    Submit {
        /// The execution queue.
        queue: u64,
        /// The resident program.
        program: u64,
        /// Relative timeout in ns from admission; zero = infinite.
        timeout_ns: u64,
        /// The buffer bindings.
        bindings: AccelBindings,
        /// Exact deployed grid, group, and client-requested SIMD width.
        dispatch: Dispatch,
    },
    /// Admit one execution through the portable launch vocabulary. The reply
    /// carries the event id at the acceptance boundary, and completion rides
    /// the signal channel ([`AccelRequest::Signal`]) with
    /// [`AccelRequest::PollEvent`] as the re-readable truth.
    SubmitLaunch {
        /// The execution queue.
        queue: u64,
        /// The resident program.
        program: u64,
        /// Relative timeout in ns from admission; zero = infinite.
        timeout_ns: u64,
        /// The buffer bindings.
        bindings: AccelBindings,
        /// Whether the artifact defines its launch shape or the client
        /// explicitly supplies portable workgroup geometry. Backends must
        /// reject unsupported explicit geometry; it is never advisory.
        launch: LaunchShape,
    },
    /// Admit a dependency-ordered staged table as one provider event.
    SubmitSequence {
        queue: u64,
        stage: u64,
        len: u64,
        count: u32,
        timeout_ns: u64,
    },
    /// Observe an event's state without blocking.
    PollEvent {
        /// The event id.
        event: u64,
    },
    /// Attempt to cancel a pending event (requires the cancellation
    /// capability; `Busy` means completion won — poll).
    CancelEvent {
        /// The event id.
        event: u64,
    },
    /// Release a terminal event and its invocation resources.
    DestroyEvent {
        /// The event id.
        event: u64,
    },
    /// Mint this pair's completion channel; the recv end rides the
    /// reply's one retained grant.  Re-signalling revokes the previous
    /// channel (T6 closes the old recv wherever it traveled) and mints
    /// afresh.
    Signal,
}

impl AccelRequest {
    /// Whether this request maintains a client's private service face.
    ///
    /// Face lifecycle belongs to the service identity and remains reachable
    /// while its hardware epoch is idle, lost, or terminal. Backend wrappers
    /// use this classification before applying device-readiness gates.
    pub const fn is_peer_control(self) -> bool {
        matches!(
            self,
            Self::AdmitPeer { .. }
                | Self::TakePeerAdmission
                | Self::RetirePeer { .. }
                | Self::ConfirmRetired { .. }
                | Self::ClosePeer
        )
    }
}

/// The accel reply vocabulary. Completion channels and import hand-back
/// leases travel as retained grants; reply data alone conveys no authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelResponse {
    /// Bounded text page. Offset is echoed; unused bytes are zero.
    Failure {
        offset: u64,
        len: u8,
        bytes: [u8; kore_abi::svc::AccelRepWire::FAILURE_CHUNK_BYTES],
    },
    /// Pre-publication import refusal with the original lease on grant lane 0.
    ImportReturned { refusal: AccelRefusal },
    /// Device and CPU mappings closed; one retained lease grant carries custody.
    BufferExported { buffer: u64 },
    /// Component retained; this does not claim a completed admission.
    PeerAccepted,
    /// Host-authored installed peer identity.
    PeerAdmitted { peer: kore_rt::svc::PeerId },
    /// Closing requested; slot reuse still waits for domain cleanup.
    PeerClosing,
    /// The exact peer generation is absent and its serving cell is reusable.
    PeerRetired,
    /// The owned staging mapping and parent are both retired.
    Detached,
    /// The staging window is live; `bytes` is its mapped length (read
    /// off the mapping, never trusted from the wire).
    Attached {
        /// Window byte length.
        bytes: u64,
    },
    /// The device's validated metadata.
    Info(AccelInfo),
    /// A fresh context.
    ContextMade {
        /// Its id.
        ctx: u64,
    },
    /// The device owner retained and activated additional supplied memory.
    Provisioned {
        /// Actual supply length, at least the sum of the requested increments.
        bytes: u64,
    },
    /// The provider accepted this context's exact profiling setting.
    ProfilingConfigured { ctx: u64, enabled: bool },
    /// A fresh buffer.
    BufferMade {
        /// Its id.
        buffer: u64,
    },
    /// A fresh resident program.
    ProgramMade {
        /// Its id.
        program: u64,
    },
    /// A fresh execution queue.
    QueueMade {
        /// Its id.
        queue: u64,
    },
    /// The subject was released; `subject` echoes it (the uncorrelated
    /// -pair echo `reply_matches` checks).
    Released {
        /// The released object's id.
        subject: u64,
    },
    /// The staged bytes are in the buffer.
    Wrote {
        /// The buffer id, echoed.
        buffer: u64,
    },
    /// The buffer bytes are in the staging window.
    Read {
        /// The buffer id, echoed.
        buffer: u64,
    },
    /// The firmware log tail is in the staging window.
    Log {
        /// How many bytes were written there (zero: no log retained).
        bytes: u64,
    },
    /// Admitted; completion will signal and [`AccelResponse::Event`]
    /// remains the truth.
    Submitted {
        /// The submission's event id.
        event: u64,
    },
    /// Acceptance is uncertain: the event **owns the submission's
    /// resources** until terminal (protocol 1.0's acceptance boundary);
    /// `refusal` says what went wrong.
    SubmitIndeterminate {
        /// The event that owns the resources.
        event: u64,
        /// The admission error.
        refusal: AccelRefusal,
    },
    /// An event's observed state.
    Event {
        /// The event id, echoed.
        event: u64,
        /// The state observed.
        state: AccelEventState,
    },
    /// Cancellation won; the event is terminal.
    Cancelled {
        /// The event id, echoed.
        event: u64,
    },
    /// The completion channel's recv end rides this reply's retained
    /// grant.
    Signaled,
    /// Refused — nothing created, nothing lost, no authority moved.
    Refused(AccelRefusal),
}

/// The device metadata [`AccelRequest::Query`] answers — protocol 1.0's
/// discovery payload, lifted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccelInfo {
    /// The device instance uuid.
    pub uuid: [u8; 16],
    /// The accelerator class (`contract::AcceleratorClass` wire value:
    /// 1 NPU, 2 GPU, 3 DSP…).
    pub class: u16,
    /// Vendor id.
    pub vendor: u32,
    /// Device id.
    pub device: u32,
    /// `contract::Capabilities` bits.
    pub capabilities: u64,
    /// The device's resource limits (the contract's own struct — the
    /// eight words every backend advertises).
    pub limits: DeviceLimits,
}

impl AccelInfo {
    fn from_native<A: NativeAccelerator>(info: DeviceInfo) -> Self {
        let mut native = Self::from_contract(info);
        if A::EXPLICIT_WORKGROUPS {
            native.capabilities |= kore_driver::accel::EXPLICIT_WORKGROUPS;
        }
        native
    }

    /// Lift the contract's discovery struct.
    pub fn from_contract(info: DeviceInfo) -> Self {
        Self {
            uuid: info.identity.uuid,
            class: info.identity.class.get(),
            vendor: info.identity.vendor_id,
            device: info.identity.device_id,
            capabilities: info.capabilities.bits(),
            limits: info.limits,
        }
    }
}

/// An event's observed state ([`AccelResponse::Event`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelEventState {
    /// Not yet terminal.
    Pending,
    /// Terminal: the submission executed.
    Complete,
    /// Terminal: the submission failed.
    Failed(AccelRefusal),
    /// Terminal: cancellation won.
    Cancelled,
}

/// Why the accel service refused — the wire vocabulary
/// ([`kore_abi::svc::AccelRepWire`]'s `REFUSE_*` codes, typed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelRefusal {
    /// The backend cannot perform the operation.
    Unsupported,
    /// Individually valid pieces that cannot compose.
    Incompatible,
    /// A malformed argument.
    InvalidArgument,
    /// A range escapes its buffer or the staging window.
    OutOfBounds,
    /// The object is busy (live children, in-flight work, a racing
    /// completion).
    Busy,
    /// The provider could not allocate.
    OutOfMemory,
    /// A device limit or retained-byte budget is exhausted.
    ResourceLimit,
    /// The submission's deadline expired.
    DeadlineExpired,
    /// The complete device epoch is lost: every object in it is gone and its
    /// ids never resolve again. The row answers this until it has contained
    /// the device and hosts a fresh epoch under a new namespace
    /// ([`AccelService::refound`]), or for good when it cannot.
    DeviceLost,
    /// A binding's access exceeds the buffer's declared usage.
    PermissionDenied,
    /// A provider- or engine-internal fault.
    Internal,
    /// Opaque provider-owned namespace and code, preserved without reinterpretation.
    External { domain: u32, code: i64 },
    /// The id names no live object (never issued, stale, wrong kind, or
    /// another epoch).
    BadObject,
    /// No staging window is attached, or the range lies outside it.
    NoStaging,
    /// The object belongs to another endpoint, or the verb is not this
    /// endpoint's.
    Denied,
    /// The completion-channel mint refused; polling remains lawful.
    NoChannel,
    /// The row hosts no live engine (degraded founding).
    NotReady,
}

/// One pair's session state: its staging window, its attach spend, and
/// its completion-signal lane.
struct PairSession {
    /// Exact host-stamped arrival; slot decoding never attenuates ownership.
    arrival: Option<u16>,
    staging: Option<&'static mut [u8]>,
    owned_staging: Option<staging::OwnedStaging>,
    attaches: u32,
    signal: Option<SignalLane>,
}

/// Decode placement only. Object ownership always retains the full arrival.
/// Encode a peer arrival from the (slot, generation) pair it means.
///
/// `PeerId`'s slot field width is not this crate's to know — it widened from
/// two bits to a byte in #575, which silently reinterpreted every hardcoded
/// arrival literal (slot 1/generation 1 became slot 5/generation 0, which
/// `from_arrival` refuses). Tests state the identity they mean and let
/// [`kore_rt::svc::PeerId::SLOT_CAPACITY`] place it.
#[cfg(test)]
pub(crate) const fn arrival_of(slot: u16, generation: u16) -> u16 {
    generation * kore_rt::svc::PeerId::SLOT_CAPACITY as u16 + slot
}

pub(crate) fn pair_index(via: u16) -> usize {
    kore_rt::svc::PeerId::from_arrival(via).map_or(usize::from(via), |peer| peer.slot())
}

impl PairSession {
    const EMPTY: Self = Self {
        arrival: None,
        staging: None,
        owned_staging: None,
        attaches: 0,
        signal: None,
    };
}

/// A minted completion channel: the **root** is the T6 revocation pivot
/// (retiring it closes the granted recv wherever it traveled), the
/// **send** is the service's signal lane.  The mint's own recv is
/// retired at mint time — the client's recv is a fresh derive staged in
/// the grant session (the image `FaceBank` discipline).
struct SignalLane {
    root: ChannelRoot,
    send: ChanSendToken,
}

impl SignalLane {
    /// Revoke the lane: root first (the revocation), then the send's
    /// bookkeeping retire (its record died with the root; a refusal
    /// there changes nothing the root did not already settle).
    fn retire(self) -> Result<(), Self> {
        let Self { root, send } = self;
        match root.try_retire() {
            Ok(_) => {
                let _ = send.try_retire();
                Ok(())
            }
            Err(root) => Err(Self { root, send }),
        }
    }
}

/// One hosted device epoch. `Lost` is deliberately distinct from both idle
/// and terminal: it still owns the complete backend/state graph, but no client
/// or device-plane accessor can reach it while the hardware row takes the
/// affine discard handoff.
enum HostEpoch<A: NativeAccelerator> {
    /// No validated device was hosted.
    Idle,
    /// The only client- and device-plane reachable state.
    Live(Engine<A>),
    /// A poisoned namespace awaiting its owning row's quiesce/containment.
    Lost(Engine<A>),
    /// The lost graph left this protocol host for owner-controlled retention
    /// or discard. Identity stays queryable; other operations answer DeviceLost.
    Terminal(DeviceInfo),
}

/// The affine handoff of a complete discard-only backend epoch.
///
/// A hardware row uses [`Self::quiesce_and_retain`] while external retirement
/// evidence is pending, or [`Self::quiesce_and_discard`] once stopping or
/// permanent quarantine makes graph destruction safe. This is
/// intentionally not a reset surface: the old object namespace can never
/// become live again. A row that can prove its device quiet uses
/// [`Self::discard_quiesced`] instead, whose [`ContainedEpoch`] is what
/// [`AccelService::refound`] takes to host a fresh epoch under a new
/// namespace (`res/LIFECYCLE_HARDENING.md` §5).
#[must_use = "a lost accelerator epoch must be quiesced and discarded"]
pub struct LostEngineEpoch<A: NativeAccelerator> {
    engine: Engine<A>,
}

/// The owner's word that a lost epoch is over: its device was proved quiet by
/// the row's own witness and its graph was discarded. Minted only by
/// [`LostEngineEpoch::discard_quiesced`] and consumed by
/// [`AccelService::refound`]; it names the epoch it ends, so it cannot
/// refound any other.
#[must_use = "a contained epoch is the only way to host a fresh one"]
#[derive(Debug)]
pub struct ContainedEpoch {
    namespace: u16,
}

/// A stopped/discard-only graph and its resource owner awaiting external
/// retirement evidence. Dropping this value deliberately retains both: an
/// interrupted recovery must not run mapping or backing destructors. Storage
/// remains charged to the terminal task until explicit discard or task reap.
/// This owns no reset/re-admission path and exposes no backend handles.
#[must_use = "retain the lost graph until retirement is authorized"]
pub struct RetainedLostEpoch<A: NativeAccelerator, O> {
    parts: core::mem::ManuallyDrop<(Engine<A>, O)>,
}

impl<A: NativeAccelerator, O> RetainedLostEpoch<A, O> {
    /// Explicitly discard the passive graph, then return its resource owner.
    /// The caller must first establish quiescence and the authority needed for
    /// the graph's mapping/backing retirement. A fabric report alone does not
    /// establish that every destructor can safely retire its owned resources.
    pub fn discard(self) -> O {
        let (engine, owner) = core::mem::ManuallyDrop::into_inner(self.parts);
        drop(engine);
        owner
    }
}

impl<A: NativeAccelerator> LostEngineEpoch<A> {
    /// Stop the hardware while retaining the complete graph and resource owner
    /// across an asynchronous retirement exchange. Install retained ownership
    /// before calling the class hook, so even unwinding the hook cannot drop
    /// backing whose stop/retirement evidence was never returned.
    pub fn quiesce_and_retain<O, R>(
        self,
        owner: O,
        quiesce: impl FnOnce(&mut A, &mut O) -> R,
    ) -> (RetainedLostEpoch<A, O>, R) {
        let mut retained = RetainedLostEpoch {
            parts: core::mem::ManuallyDrop::new((self.engine, owner)),
        };
        let (engine, owner) = &mut *retained.parts;
        let result = quiesce(engine.backend_mut(), owner);
        (retained, result)
    }

    /// Run the hardware owner's quiesce/containment step, then deterministically
    /// discard every accepted event, its dependent handles, and the backend
    /// instance. Before returning, the callback must either prove device fetch
    /// stopped or ensure any still-reachable backing cannot be recycled before
    /// task reap. Its result reports hardware truth; it never changes this
    /// epoch into a reusable one.
    pub fn quiesce_and_discard<R>(self, quiesce: impl FnOnce(&mut A) -> R) -> R {
        let (retained, result) = self.quiesce_and_retain((), |backend, _| quiesce(backend));
        retained.discard();
        result
    }

    /// Stop the device and discard the graph only if the owner's callback
    /// returns its hardware witness `P` that nothing of this epoch can still
    /// reach memory (an engine or function reset, every device mapping
    /// retired). Without one the complete graph comes back retained, for
    /// the owner to keep until task reap exactly as
    /// [`Self::quiesce_and_retain`] would.
    pub fn discard_quiesced<P>(
        self,
        quiesce: impl FnOnce(&mut A) -> Option<P>,
    ) -> Result<(ContainedEpoch, P), RetainedLostEpoch<A, ()>> {
        let namespace = self.engine.namespace();
        let (retained, proof) = self.quiesce_and_retain((), |backend, _| quiesce(backend));
        match proof {
            Some(proof) => {
                retained.discard();
                Ok((ContainedEpoch { namespace }, proof))
            }
            None => Err(retained),
        }
    }
}

/// Retained diagnostic budget per service epoch. Enough for one native
/// submission's six bindings plus program/queue metadata; never model bytes.
/// Truncation cannot affect containment or admission.
pub const FAILURE_REPORT_BYTES: usize = 2048;
/// Bounded diagnostic snapshot: the existing device report plus nine kernel
/// families, timestamp validity and host-gap counters. This has its own bound
/// so growing optional measurements does not grow retained failure evidence.
pub const TELEMETRY_REPORT_BYTES: usize = 4096;

/// The accelerator service host — one backend behind the class protocol.
///
/// Constructed by a backend row's `bind` through [`Self::host`]; a
/// backend whose metadata refuses validation (or a machine without the
/// device) still constructs, hosts no engine, and answers every verb
/// [`AccelRefusal::NotReady`] — the degraded-but-serving posture.
pub struct AccelService<A: NativeAccelerator> {
    epoch: HostEpoch<A>,
    /// The rendezvous mint grant signal channels are rooted in
    /// (`role::CHANNEL_SPACE`); `None` hosts a poll-only service.
    channels: Option<ChannelSpaceToken>,
    sessions: [PairSession; PAIRS],
    peers: peers::Peers,
    staging_window: Option<kore_abi::task::layout::Window>,
    /// The lost epoch's device witness, kept past the affine handoff that
    /// moves its engine out (`take_lost_epoch`), so a client arriving after
    /// containment still learns the fault and not merely the loss.
    terminal_evidence: Option<LossEvidence>,
    /// The namespace of the epoch `take_lost_epoch` handed out: the only one
    /// a [`ContainedEpoch`] may end.
    terminal_namespace: Option<u16>,
    failure_report: heapless::String<FAILURE_REPORT_BYTES>,
    /// The last telemetry snapshot [`AccelRequest::ReadTelemetry`] pages.
    telemetry_report: heapless::String<TELEMETRY_REPORT_BYTES>,
    served: u64,
    refused: u64,
    dropped_notices: u64,
    /// This service's own time per request kind ([`REQUEST_KINDS`]): from
    /// [`Self::step`]'s entry to its reply, so the wait in the ring, the
    /// IPC and the client's wake are outside it.
    service_time: [ServiceTime; REQUEST_KINDS.len()],
    /// Which pair's `Signal` the in-flight grant session belongs to —
    /// consumed by [`Self::grants_unwound`] when the host drops the
    /// session whole.
    last_signal_via: Option<u16>,
}

impl<A: NativeAccelerator> AccelService<A> {
    pub fn peer_admitted(
        &mut self,
        result: Result<kore_rt::svc::PeerId, kore_rt::svc::PeerAdmissionRefusal>,
    ) -> bool {
        if !self.peers.pending() {
            return false;
        }
        // A terminal provider remains an identity service: a newly launched
        // client must be able to attach a private face, query the retained
        // identity, and receive DeviceLost from data-plane requests. Making
        // peer admission depend on a live engine instead refuses the process
        // launch itself, so a client can never reach its CPU fallback.
        // The session table folds a recyclable peer onto its slot
        // (`pair_index`), the same cells the fixed administration pairs
        // occupy by index. The runtime only hands a recyclable peer a cell
        // whose fixed pair is attached, but that invariant lives in another
        // crate: a row that serves an admin pair without attaching it once
        // seated an application peer in the exec cell. Refuse the collision
        // here, where the custody is, instead of trusting the allocator.
        let accepted = result.is_ok_and(|peer| {
            peer.slot() >= crate::wire::ACCEL_SERVE_PAIRS && self.claim_arrival(peer.arrival())
        });
        self.peers.admitted(result, accepted)
    }

    /// Native creation callback under the same arrival, metadata, policy and
    /// cleanup checks as ordinary allocation. A callback is invoked only after
    /// admission checks; post-creation refusal retains a table cleanup owner.
    pub fn import_buffer_using<F>(
        &mut self,
        via: u16,
        ctx: u64,
        bytes: u64,
        align: u64,
        usage: u32,
        create: F,
    ) -> AccelResponse
    where
        F: FnOnce(
            &A,
            &A::Context,
            virtio_accel_core::BufferDesc,
        ) -> Result<
            virtio_accel_core::AllocatedBuffer<A::Buffer>,
            virtio_accel_core::BackendError,
        >,
    {
        if !self.claim_arrival(via) {
            return AccelResponse::Refused(AccelRefusal::Denied);
        }
        if let Err(error) = self.poll_backend_health() {
            return self.with_loss_evidence(AccelResponse::Refused(error));
        }
        let result = match &mut self.epoch {
            HostEpoch::Live(engine) => {
                engine.create_buffer_using(via, ctx, bytes, align, 3, usage, create)
            }
            HostEpoch::Idle => Err(AccelRefusal::NotReady),
            HostEpoch::Lost(_) | HostEpoch::Terminal(_) => Err(AccelRefusal::DeviceLost),
        };
        self.seal_lost_epoch();
        self.with_loss_evidence(result.unwrap_or_else(AccelResponse::Refused))
    }

    /// Release through the same ownership/state engine, retaining backend
    /// allocation custody in the native caller instead of recycling it.
    pub fn free_buffer_using<F>(&mut self, via: u16, buffer: u64, release: F) -> AccelResponse
    where
        F: FnOnce(&A, A::Buffer) -> Result<(), virtio_accel_core::ReleaseFailure<A::Buffer>>,
    {
        if !self.claim_arrival(via) {
            return AccelResponse::Refused(AccelRefusal::Denied);
        }
        if let Err(error) = self.poll_backend_health() {
            return self.with_loss_evidence(AccelResponse::Refused(error));
        }
        let result = match &mut self.epoch {
            HostEpoch::Live(engine) => engine.free_buffer_using(via, buffer, release),
            HostEpoch::Idle => Err(AccelRefusal::NotReady),
            HostEpoch::Lost(_) | HostEpoch::Terminal(_) => Err(AccelRefusal::DeviceLost),
        };
        self.seal_lost_epoch();
        self.with_loss_evidence(result.unwrap_or_else(AccelResponse::Refused))
    }

    fn claim_arrival(&mut self, via: u16) -> bool {
        let Some(session) = self.sessions.get_mut(pair_index(via)) else {
            return false;
        };
        match session.arrival {
            Some(arrival) => arrival == via,
            None => {
                session.arrival = Some(via);
                true
            }
        }
    }

    /// Host `backend` behind the protocol.
    ///
    /// `namespace` is this device epoch's nonzero object-id namespace;
    /// the two budgets bound what all clients together may keep resident
    /// (buffer backing / program storage).  Metadata that refuses
    /// validation hosts no engine — the service still serves, refusing
    /// [`AccelRefusal::NotReady`], so a row founds on every machine and
    /// its status line says why it is idle.
    pub fn host(
        backend: A,
        channels: Option<ChannelSpaceToken>,
        namespace: u16,
        max_buffer_backing: u64,
        max_program_resident: u64,
    ) -> Self {
        let epoch = match Engine::host(backend, namespace, max_buffer_backing, max_program_resident)
        {
            Ok(engine) => HostEpoch::Live(engine),
            Err(_) => HostEpoch::Idle,
        };
        Self {
            epoch,
            channels,
            sessions: [const { PairSession::EMPTY }; PAIRS],
            peers: peers::Peers::new(),
            staging_window: None,
            terminal_evidence: None,
            terminal_namespace: None,
            failure_report: heapless::String::new(),
            telemetry_report: heapless::String::new(),
            served: 0,
            refused: 0,
            dropped_notices: 0,
            service_time: [ServiceTime::ZERO; REQUEST_KINDS.len()],
            last_signal_via: None,
        }
    }

    /// Reserve the founder's transient geometry for this host's exact
    /// staging cells. Configure before any owned attachment is admitted.
    pub fn configure_staging(&mut self, window: kore_abi::task::layout::Window) -> bool {
        if self.staging_window.is_some()
            || self
                .sessions
                .iter()
                .any(|pair| pair.owned_staging.is_some())
            || staging::cell(window, (PAIRS - 1) as u16).is_none()
        {
            return false;
        }
        self.peers.map = Some(staging::peer_map(window));
        self.staging_window = Some(window);
        true
    }

    /// Host **nothing** — the claim-less machine's posture: every verb
    /// answers [`AccelRefusal::NotReady`], the status line reads
    /// `down`, and the row still founds and serves (the degraded-service
    /// doctrine).  What a backend row constructs when its device claim
    /// came back empty-handed.
    pub fn idle(channels: Option<ChannelSpaceToken>) -> Self {
        Self {
            epoch: HostEpoch::Idle,
            channels,
            sessions: [const { PairSession::EMPTY }; PAIRS],
            peers: peers::Peers::new(),
            staging_window: None,
            terminal_evidence: None,
            terminal_namespace: None,
            failure_report: heapless::String::new(),
            telemetry_report: heapless::String::new(),
            served: 0,
            refused: 0,
            dropped_notices: 0,
            service_time: [ServiceTime::ZERO; REQUEST_KINDS.len()],
            last_signal_via: None,
        }
    }

    /// Whether this protocol host has no engine or retained terminal identity.
    /// A lost engine is not idle: its owner must still take and contain it.
    pub fn is_idle(&self) -> bool {
        matches!(&self.epoch, HostEpoch::Idle)
    }

    /// Whether a validated engine is live behind the protocol.
    pub fn hosted(&self) -> bool {
        matches!(&self.epoch, HostEpoch::Live(_))
    }

    /// Whether this row must contain its device before it serves again. It
    /// holds until [`Self::refound`] hosts a fresh epoch (or for good): no
    /// request, backend accessor, or completion sweep can reach the poisoned
    /// namespace again.
    pub fn restart_required(&self) -> bool {
        matches!(&self.epoch, HostEpoch::Lost(_) | HostEpoch::Terminal(_))
    }

    /// The refusal a sealed epoch answers data-plane requests with. Plain
    /// `DeviceLost` when the provider retained no device witness; otherwise
    /// the same loss carrying the provider's domain and full-width code, so
    /// a client with no other surface
    /// (the aftershock HUD) can name the fault, not merely the loss.
    pub fn lost_refusal(&self) -> AccelRefusal {
        let evidence = match &self.epoch {
            // A live engine that has just reported its loss is sealed by the
            // caller's next boundary; its witness is already retained.
            HostEpoch::Live(engine) | HostEpoch::Lost(engine) => engine.backend().loss_evidence(),
            HostEpoch::Terminal(_) => self.terminal_evidence,
            HostEpoch::Idle => None,
        };
        match evidence {
            None => AccelRefusal::DeviceLost,
            Some(LossEvidence { domain, code }) => AccelRefusal::External { domain, code },
        }
    }

    /// Rewrite a bare `DeviceLost` in any reply shape into the witnessed
    /// form [`lost_refusal`](Self::lost_refusal) produces. Every reply
    /// leaves through here, so the *first* observation of the loss — the
    /// engine's own `backend_err`, or the pre-request health probe — names
    /// the fault exactly as later, sealed-epoch refusals do.
    fn with_loss_evidence(&self, response: AccelResponse) -> AccelResponse {
        match response {
            AccelResponse::Refused(AccelRefusal::DeviceLost) => {
                AccelResponse::Refused(self.lost_refusal())
            }
            AccelResponse::Event {
                event,
                state: AccelEventState::Failed(AccelRefusal::DeviceLost),
            } => AccelResponse::Event {
                event,
                state: AccelEventState::Failed(self.lost_refusal()),
            },
            AccelResponse::SubmitIndeterminate {
                event,
                refusal: AccelRefusal::DeviceLost,
            } => AccelResponse::SubmitIndeterminate {
                event,
                refusal: self.lost_refusal(),
            },
            other => other,
        }
    }

    /// Test seam: the witnessed refusal a sealed epoch would answer with.
    #[cfg(test)]
    pub(crate) fn witnessed_refusal_for_test(&self) -> AccelRefusal {
        self.lost_refusal()
    }

    /// Move the complete poisoned graph to its hardware owner exactly once.
    /// The service becomes a terminal identity-only face before returning, so
    /// stale object IDs cannot race or be reused while containment runs.
    pub fn take_lost_epoch(&mut self) -> Option<LostEngineEpoch<A>> {
        let prior = core::mem::replace(&mut self.epoch, HostEpoch::Idle);
        match prior {
            HostEpoch::Lost(engine) => {
                let info = engine.info();
                self.terminal_evidence = engine.backend().loss_evidence();
                self.terminal_namespace = Some(engine.namespace());
                self.epoch = HostEpoch::Terminal(info);
                Some(LostEngineEpoch { engine })
            }
            other => {
                self.epoch = other;
                None
            }
        }
    }

    /// **Host a fresh device epoch in place of a contained one**
    /// (`res/LIFECYCLE_HARDENING.md` §5). The row reset its device, proved
    /// the old epoch quiet ([`LostEngineEpoch::discard_quiesced`]) and built
    /// `backend` over the reset hardware.
    ///
    /// The new epoch takes the next namespace, so every id a client still
    /// holds from the old one answers `BadObject`, never a new object. Every
    /// pair's funding starts empty (it lived in the old engine), and the
    /// completion lanes the seal retired stay retired; connected peers keep
    /// their faces and staging and provision, signal and create afresh.
    ///
    /// Refused, handing `backend` back, unless this host is terminal with
    /// exactly `contained`'s epoch, or when the namespace space is spent (the
    /// row then needs refounding) or `backend` fails validation.
    pub fn refound(
        &mut self,
        backend: A,
        contained: ContainedEpoch,
        max_buffer_backing: u64,
        max_program_resident: u64,
    ) -> Result<(), A> {
        if !matches!(self.epoch, HostEpoch::Terminal(_))
            || self.terminal_namespace != Some(contained.namespace)
        {
            return Err(backend);
        }
        // Namespaces only ever grow within one service instance, so no
        // earlier epoch's ids can alias a later one's.
        let Some(namespace) = contained.namespace.checked_add(1) else {
            return Err(backend);
        };
        let engine = Engine::host(backend, namespace, max_buffer_backing, max_program_resident)?;
        self.epoch = HostEpoch::Live(engine);
        self.terminal_evidence = None;
        self.terminal_namespace = None;
        self.failure_report.clear();
        self.last_signal_via = None;
        Ok(())
    }

    /// The hosted backend, mutably — the device-plane door for the row
    /// that owns hardware duties the protocol host cannot (interrupt
    /// acknowledgement, mailbox drains).  The host keeps the object
    /// tables; the row keeps the silicon.
    pub fn backend_mut(&mut self) -> Option<&mut A> {
        match &mut self.epoch {
            HostEpoch::Live(engine) => Some(engine.backend_mut()),
            HostEpoch::Idle | HostEpoch::Lost(_) | HostEpoch::Terminal(_) => None,
        }
    }

    /// The hosted backend, shared — the read-only sibling (the row's
    /// typed `next_deadline` hook reads the device's earliest enforced
    /// event deadline through this).
    pub fn backend(&self) -> Option<&A> {
        match &self.epoch {
            HostEpoch::Live(engine) => Some(engine.backend()),
            HostEpoch::Idle | HostEpoch::Lost(_) | HostEpoch::Terminal(_) => None,
        }
    }

    /// Fund one pair's byte budgets with capacity published by the owning
    /// driver — any time, repeatedly, per pair. The founded policy is the
    /// device-wide ceiling every live pair's funding must fit inside; a
    /// pair's funding and spend die with it (`close_peer`), so a relaunched
    /// client provisions afresh instead of inheriting a closed epoch.
    /// The publisher runs only after fallible policy bookkeeping succeeds.
    pub fn provision_with(
        &mut self,
        via: u16,
        buffer_bytes: u64,
        program_bytes: u64,
        publish: impl FnOnce(u64) -> Result<(), AccelRefusal>,
    ) -> Result<(), AccelRefusal> {
        self.poll_backend_health()?;
        let result = match &mut self.epoch {
            HostEpoch::Live(engine) => engine.provision(via, buffer_bytes, program_bytes, publish),
            HostEpoch::Idle => Err(AccelRefusal::NotReady),
            HostEpoch::Lost(_) | HostEpoch::Terminal(_) => Err(AccelRefusal::DeviceLost),
        };
        self.seal_lost_epoch();
        result
    }

    /// One bounded provider-wide health sweep. Hardware rows call this before
    /// draining requests after a device wake; [`Self::step`] repeats it at
    /// each request boundary so a fault raised by an earlier submission in the
    /// same drain cannot admit later work. A `DeviceLost` edge is sealed before
    /// this method returns, making the poisoned backend unreachable.
    pub fn poll_backend_health(&mut self) -> Result<(), AccelRefusal> {
        let outcome = match &mut self.epoch {
            HostEpoch::Live(engine) => engine.poll_health(),
            HostEpoch::Idle => Ok(()),
            HostEpoch::Lost(_) | HostEpoch::Terminal(_) => Err(AccelRefusal::DeviceLost),
        };
        self.seal_lost_epoch();
        outcome
    }

    /// Seal a newly poisoned live engine behind the affine owner handoff.
    /// Called at every provider-calling boundary before control returns to the
    /// row; only the inert failure-metadata hook runs during the transition.
    fn seal_lost_epoch(&mut self) {
        if !matches!(&self.epoch, HostEpoch::Live(engine) if engine.is_lost()) {
            return;
        }
        self.seal();
    }

    /// **Retire a healthy live epoch** on the row's terminal path (#809 §L):
    /// the same Live → Lost handoff `seal_lost_epoch` performs for a
    /// poisoned engine, taken deliberately for a machine going down so the
    /// row's existing lost-epoch containment (`take_lost_epoch` →
    /// `quiesce_and_discard`) becomes its shutdown sequence.  `true` when an
    /// engine moved; `false` when there was nothing live to retire.
    pub fn retire_live_epoch(&mut self) -> bool {
        if !matches!(&self.epoch, HostEpoch::Live(_)) {
            return false;
        }
        self.seal();
        true
    }

    /// The Live → Lost handoff: move the engine behind the affine owner
    /// handoff and surrender every session's access to it. The sole provider
    /// hook formats retained metadata; no hardware access or allocation.
    fn seal(&mut self) {
        let prior = core::mem::replace(&mut self.epoch, HostEpoch::Idle);
        let HostEpoch::Live(engine) = prior else {
            unreachable!("the lost predicate only matched a live epoch")
        };
        // Snapshot only inert provider metadata. Once the owner contains the
        // device, no diagnostic request may call back into the backend.
        engine.backend().failure_report(&mut self.failure_report);
        self.epoch = HostEpoch::Lost(engine);
        self.last_signal_via = None;
        for session in &mut self.sessions {
            // A staging mapping is task-lifetime authority and has no unmap
            // verb, but the terminal epoch must surrender its Rust access to
            // it immediately. Task withdrawal later removes the mapping.
            session.staging = None;
            // Completion lanes cannot carry useful news after DeviceLost.
            // Retire them explicitly: capability Drop is not a revoke. A
            // retryable tombstone remains owned for the task-reap fallback.
            if let Some(lane) = session.signal.take()
                && let Err(lane) = lane.retire()
            {
                session.signal = Some(lane);
            }
        }
    }

    #[cfg(test)]
    fn live_engine(&self) -> Option<&Engine<A>> {
        match &self.epoch {
            HostEpoch::Live(engine) => Some(engine),
            _ => None,
        }
    }

    #[cfg(test)]
    fn live_engine_mut(&mut self) -> Option<&mut Engine<A>> {
        match &mut self.epoch {
            HostEpoch::Live(engine) => Some(engine),
            _ => None,
        }
    }

    /// **The serve body** — every backend instantiation's
    /// `SystemService::serve` delegates here unchanged.
    pub fn step(&mut self, req: AccelRequest, sv: &mut Serve<'_, AccelResponse>) {
        let kind = request_kind(&req);
        let started = kore_rt::clock::uptime().ok().map(|now| now.as_nanos());
        self.serve_request(req, sv);
        let ended = kore_rt::clock::uptime().ok().map(|now| now.as_nanos());
        if let Some((start, end)) = started.zip(ended) {
            self.service_time[kind].record(end.saturating_sub(start));
        }
    }

    fn serve_request(&mut self, req: AccelRequest, sv: &mut Serve<'_, AccelResponse>) {
        self.served = self.served.saturating_add(1);
        if !self.claim_arrival(sv.via()) {
            sv.reply(AccelResponse::Refused(AccelRefusal::Denied));
            return;
        }
        self.last_signal_via = None;
        if req.is_peer_control() {
            // Peer lifecycle belongs to the service identity, not to the
            // hardware epoch. Lost/terminal hosts continue admitting faces;
            // their ordinary operations fail closed with DeviceLost below.
            let reply = if let AccelRequest::RetirePeer { peer } = req {
                let exact = self
                    .sessions
                    .get(peer.slot())
                    .is_some_and(|session| session.arrival == Some(peer.arrival()));
                if crate::wire::is_admin_pair(sv.via()) && exact {
                    sv.retire_peer(peer, PeerClose::Orderly);
                    AccelResponse::PeerClosing
                } else {
                    AccelResponse::Refused(AccelRefusal::Denied)
                }
            } else if let AccelRequest::ConfirmRetired { peer } = req {
                let exact = self
                    .sessions
                    .get(peer.slot())
                    .is_some_and(|session| session.arrival == Some(peer.arrival()));
                if crate::wire::is_admin_pair(sv.via()) && !exact {
                    AccelResponse::PeerRetired
                } else {
                    AccelResponse::Refused(AccelRefusal::Busy)
                }
            } else {
                self.peers.handle(req, sv).expect("control request")
            };
            sv.reply(reply);
            return;
        }
        // Poll before every request, not merely once per wake: an earlier
        // submission in this same ring drain may have raised a provider-wide
        // fault. The default backend probe inlines to `Ok(())`; hardware with
        // an asynchronous fault source pays one bounded control-plane probe.
        let health_refusal = self.poll_backend_health().err();
        let via = sv.via();
        let response = if matches!(&req, AccelRequest::DetachStaging) {
            self.detach_staging(via)
        } else if matches!(
            &req,
            AccelRequest::ReadFailure { .. } | AccelRequest::ReadTelemetry { .. }
        ) {
            // Read-only text: neither needs a healthy device to answer.
            self.answer(via, req)
        } else if matches!(&req, AccelRequest::Query) {
            // Immutable identity remains queryable after the provider-wide
            // edge has sealed and even after its affine graph was discarded.
            self.answer(via, AccelRequest::Query)
        } else if let Some(refusal) = health_refusal {
            // A transient health-probe error does not poison the namespace,
            // but neither does it prove this admission safe. DeviceLost has
            // already sealed above and reaches this same correlated refusal.
            AccelResponse::Refused(refusal)
        } else {
            match req {
                AccelRequest::Attach { announced } => self.attach(via, announced, sv),
                AccelRequest::AttachOwned { announced } => self.attach_owned(via, announced, sv),
                AccelRequest::Signal => self.signal(via, sv),
                other => self.answer(via, other),
            }
        };
        if matches!(response, AccelResponse::Refused(_)) {
            self.refused = self.refused.saturating_add(1);
        }
        let response = self.with_loss_evidence(response);
        sv.reply(response);
    }

    /// The `Serve`-independent verbs — separated so ownership stamping
    /// is testable per pair without a transport.
    fn answer(&mut self, via: u16, req: AccelRequest) -> AccelResponse {
        if !self.claim_arrival(via) {
            return AccelResponse::Refused(AccelRefusal::Denied);
        }
        if let AccelRequest::ReadFailure { offset } = req {
            if !self.restart_required() {
                return AccelResponse::Refused(AccelRefusal::NotReady);
            }
            let mut bytes = [0; kore_abi::svc::AccelRepWire::FAILURE_CHUNK_BYTES];
            let start = match usize::try_from(offset) {
                Ok(start) if start <= self.failure_report.len() => start,
                _ => return AccelResponse::Refused(AccelRefusal::OutOfBounds),
            };
            let tail = &self.failure_report.as_bytes()[start..];
            let len = tail.len().min(bytes.len());
            bytes[..len].copy_from_slice(&tail[..len]);
            return AccelResponse::Failure { offset, len: len as u8, bytes };
        }
        if let AccelRequest::ReadTelemetry { offset } = req {
            if offset == 0 {
                use core::fmt::Write;
                let Self {
                    epoch,
                    telemetry_report,
                    dropped_notices,
                    service_time,
                    ..
                } = self;
                telemetry_report.clear();
                match epoch {
                    HostEpoch::Live(engine) => {
                        engine.backend().telemetry_report(telemetry_report);
                    }
                    HostEpoch::Idle | HostEpoch::Lost(_) | HostEpoch::Terminal(_) => {
                        let _ = telemetry_report.push_str("no live epoch\n");
                    }
                }
                let _ = writeln!(telemetry_report, "host: dropped notices {dropped_notices}");
                // One row per kind (the report splits on ';').
                let _ = telemetry_report.push_str("host: service time");
                for (name, time) in REQUEST_KINDS.iter().zip(service_time.iter()) {
                    let _ = write!(
                        telemetry_report,
                        "; {name} {} calls, {} us avg, {} us longest",
                        time.calls,
                        time.average_ns() / NS_PER_US,
                        time.longest_ns / NS_PER_US,
                    );
                }
                let _ = telemetry_report.push('\n');
            }
            let mut bytes = [0; kore_abi::svc::AccelRepWire::FAILURE_CHUNK_BYTES];
            let start = match usize::try_from(offset) {
                Ok(start) if start <= self.telemetry_report.len() => start,
                _ => return AccelResponse::Refused(AccelRefusal::OutOfBounds),
            };
            let tail = &self.telemetry_report.as_bytes()[start..];
            let len = tail.len().min(bytes.len());
            bytes[..len].copy_from_slice(&tail[..len]);
            return AccelResponse::Failure {
                offset,
                len: len as u8,
                bytes,
            };
        }
        if let AccelRequest::Query = req {
            return match &self.epoch {
                HostEpoch::Live(engine) | HostEpoch::Lost(engine) => {
                    AccelResponse::Info(AccelInfo::from_native::<A>(engine.info()))
                }
                HostEpoch::Terminal(info) => {
                    AccelResponse::Info(AccelInfo::from_native::<A>(*info))
                }
                HostEpoch::Idle => AccelResponse::Refused(AccelRefusal::NotReady),
            };
        }
        if self.restart_required() {
            return AccelResponse::Refused(self.lost_refusal());
        }
        let Self {
            epoch, sessions, ..
        } = self;
        let HostEpoch::Live(engine) = epoch else {
            return AccelResponse::Refused(AccelRefusal::NotReady);
        };
        let outcome = match req {
            AccelRequest::ReadFailure { .. }
            | AccelRequest::ReadTelemetry { .. }
            | AccelRequest::ExportBuffer { .. }
            | AccelRequest::ProvisionMemory { .. }
            | AccelRequest::ReserveMemory { .. }
            | AccelRequest::ImportBuffer { .. }
            | AccelRequest::AdmitPeer { .. }
            | AccelRequest::TakePeerAdmission
            | AccelRequest::RetirePeer { .. }
            | AccelRequest::ConfirmRetired { .. }
            | AccelRequest::ClosePeer => Err(AccelRefusal::Unsupported),
            AccelRequest::CreateContext { flags } => engine.create_context(via, flags),
            AccelRequest::DestroyContext { ctx } => engine.destroy_context(via, ctx),
            AccelRequest::ConfigureProfiling { ctx, enabled } => {
                engine.configure_profiling(via, ctx, enabled)
            }
            AccelRequest::CreateBuffer {
                ctx,
                bytes,
                align,
                domain,
                usage,
            } => engine.create_buffer(via, ctx, bytes, align, domain, usage),
            AccelRequest::FreeBuffer { buffer } => engine.free_buffer(via, buffer),
            AccelRequest::WriteBuffer {
                buffer,
                offset,
                stage,
                len,
            } => stage_slice(sessions, via, stage, len)
                .and_then(|data| engine.write_buffer(via, buffer, offset, data)),
            AccelRequest::ReadBuffer {
                buffer,
                offset,
                stage,
                len,
            } => stage_slice_mut(sessions, via, stage, len)
                .and_then(|out| engine.read_buffer(via, buffer, offset, out)),
            AccelRequest::ReadLog { stage, len } => stage_slice_mut(sessions, via, stage, len)
                .and_then(|out| engine.read_firmware_log(out)),
            AccelRequest::LoadProgram {
                ctx,
                stage,
                len,
                resident_bytes,
                format,
                target,
            } => stage_slice(sessions, via, stage, len).and_then(|artifact| {
                engine.load_program(via, ctx, artifact, resident_bytes, format, target)
            }),
            AccelRequest::UnloadProgram { program } => engine.unload_program(via, program),
            AccelRequest::CreateQueue { ctx, flags } => engine.create_queue(via, ctx, flags),
            AccelRequest::DestroyQueue { queue } => engine.destroy_queue(via, queue),
            #[allow(deprecated)]
            AccelRequest::Submit {
                queue,
                program,
                timeout_ns,
                bindings,
                dispatch,
            } if dispatch.is_scalar() => engine.submit(
                via,
                queue,
                program,
                timeout_ns,
                &bindings,
                LaunchShape::ArtifactDefined,
            ),
            #[allow(deprecated)]
            AccelRequest::Submit {
                queue,
                program,
                timeout_ns,
                bindings,
                dispatch,
            } => engine.submit_legacy(via, queue, program, timeout_ns, &bindings, dispatch),
            AccelRequest::SubmitLaunch {
                queue,
                program,
                timeout_ns,
                bindings,
                launch,
            } => engine.submit(via, queue, program, timeout_ns, &bindings, launch),
            AccelRequest::SubmitSequence {
                queue,
                stage,
                len,
                count,
                timeout_ns,
            } => stage_slice(sessions, via, stage, len).and_then(|bytes| {
                let count = usize::try_from(count).map_err(|_| AccelRefusal::InvalidArgument)?;
                let entries = crate::wire::decode_sequence(bytes, count)?;
                engine.submit_sequence(via, queue, &entries, timeout_ns)
            }),
            AccelRequest::PollEvent { event } => engine.poll_event(via, event),
            AccelRequest::CancelEvent { event } => engine.cancel_event(via, event),
            AccelRequest::DestroyEvent { event } => engine.destroy_event(via, event),
            AccelRequest::Attach { .. }
            | AccelRequest::AttachOwned { .. }
            | AccelRequest::DetachStaging
            | AccelRequest::Signal
            | AccelRequest::Query => {
                unreachable!("handled before the engine dispatch")
            }
        };
        let response = outcome.unwrap_or_else(AccelResponse::Refused);
        self.seal_lost_epoch();
        response
    }

    /// The `Attach` arm: adopt the transferred view, map it, and hand
    /// the mapping to [`Self::attach_staging`].
    fn attach(
        &mut self,
        via: u16,
        announced: SlotRef,
        sv: &mut Serve<'_, AccelResponse>,
    ) -> AccelResponse {
        match self.attach_staging_with(via, || adopt_staging(sv, announced)) {
            Ok(bytes) => AccelResponse::Attached { bytes },
            Err(refusal) => AccelResponse::Refused(refusal),
        }
    }

    /// **Adopt already mapped staging bytes**, used by host fixtures. The
    /// metal `Attach` path shares the admission implementation but supplies a
    /// deferred mapper, so it can refuse before consuming mapping space.
    ///
    /// A re-attach replaces the window. The metal path admits and spends an
    /// attempt before invoking the mapper, including failures: a refused map
    /// may already have advanced the finite, irreversible mapping cursor.
    pub fn attach_staging(
        &mut self,
        via: u16,
        window: &'static mut [u8],
    ) -> Result<u64, AccelRefusal> {
        self.attach_staging_with(via, || Some(window))
    }

    /// Keep admission and mapping ordered in the same path on metal and host.
    /// Refusal must not take or forget the returned capability: the serve
    /// host still owns its ordinary returned-input cleanup until `map` runs.
    fn attach_staging_with(
        &mut self,
        via: u16,
        map: impl FnOnce() -> Option<&'static mut [u8]>,
    ) -> Result<u64, AccelRefusal> {
        if !self.claim_arrival(via) {
            return Err(AccelRefusal::Denied);
        }
        let session = self
            .sessions
            .get_mut(pair_index(via))
            .ok_or(AccelRefusal::Denied)?;
        if session.owned_staging.is_some() {
            return Err(AccelRefusal::Busy);
        }
        if session.attaches >= MAX_ATTACHES {
            return Err(AccelRefusal::ResourceLimit);
        }
        session.attaches += 1;
        let window = map()
            .filter(|window| !window.is_empty())
            .ok_or(AccelRefusal::NoStaging)?;
        let bytes = window.len() as u64;
        session.staging = Some(window);
        Ok(bytes)
    }

    fn attach_owned(
        &mut self,
        via: u16,
        announced: SlotRef,
        sv: &mut Serve<'_, AccelResponse>,
    ) -> AccelResponse {
        let result = (|| {
            let session = self
                .sessions
                .get_mut(pair_index(via))
                .ok_or(AccelRefusal::Denied)?;
            if session.attaches != 0 || session.staging.is_some() || session.owned_staging.is_some()
            {
                return Err(AccelRefusal::Busy);
            }
            let cell = self
                .staging_window
                .and_then(|window| staging::cell(window, pair_index(via) as u16))
                .ok_or(AccelRefusal::Unsupported)?;
            let mut input = sv
                .returned_input(announced)
                .ok_or(AccelRefusal::NoStaging)?;
            let capability = input.capability().ok_or(AccelRefusal::NoStaging)?;
            let slot = capability.slot().ok_or(AccelRefusal::NoStaging)?;
            let bytes = if capability.is_lease_with(PROVISION_RIGHTS) {
                verbs::lease_geometry(slot).map(|(_, bytes)| bytes)
            } else if capability.is_kind_with(ResourceKind::MemoryFabricRoute, PROVISION_RIGHTS) {
                capability.describe_fabric_route().ok().map(|info| info.bytes)
            } else {
                return Err(AccelRefusal::Denied);
            }.ok_or(AccelRefusal::NoStaging)?;
            if bytes == 0 || bytes & 4095 != 0 || bytes > cell.len() {
                return Err(AccelRefusal::ResourceLimit);
            }
            usize::try_from(bytes).map_err(|_| AccelRefusal::ResourceLimit)?;
            let parent = input
                .take()
                .ok_or(AccelRefusal::NoStaging)?
                .try_into_memory(PROVISION_RIGHTS)
                .map_err(|_| AccelRefusal::Denied)?;
            let owned = staging::OwnedStaging::map(parent, cell.base(), bytes);
            let mapped = owned.window().is_some();
            // Retain custody even when mapping failed: parent retirement must
            // prove this cell reusable before another attachment can enter.
            session.owned_staging = Some(owned);
            if !mapped {
                return Err(AccelRefusal::NoStaging);
            }
            // Borrow bytes only while processing a request. In particular, a
            // shared route must not install a permanent exclusive Rust slice.
            Ok(bytes)
        })();
        match result {
            Ok(bytes) => AccelResponse::Attached { bytes },
            Err(reason) => AccelResponse::Refused(reason),
        }
    }

    /// Runtime-authenticated peer retirement. False retains the closing cell;
    /// no generation may reuse it until graph, staging and signal custody close.
    pub fn close_peer(&mut self, via: u16) -> bool {
        self.close_peer_using(via, |backend, buffer| backend.free_buffer(buffer))
    }

    /// Close the same peer graph with native buffer retirement. Imported
    /// backing must pass through its host-owned DMA release path, just as
    /// explicit export does. Rejected releases retain their object for retry.
    pub fn close_peer_using<F>(&mut self, via: u16, release: F) -> bool
    where
        F: FnMut(&A, A::Buffer) -> Result<(), virtio_accel_core::ReleaseFailure<A::Buffer>>,
    {
        if !self.claim_arrival(via) {
            return false;
        }
        let retired = match &mut self.epoch {
            HostEpoch::Live(engine) => engine.retire_peer_using(via, release).is_ok(),
            // A sealed epoch's graph, this peer's objects included, is
            // unreachable from every request and belongs whole to the row's
            // containment: its device custody is the graph's, never the
            // session's. What the session itself holds (staging, the signal
            // lane) is closed below exactly as for a live epoch. Refusing here
            // stranded the cell for good, so each launch after a loss spent a
            // pair until exec could admit no face at all.
            HostEpoch::Lost(_) | HostEpoch::Terminal(_) | HostEpoch::Idle => true,
        };
        self.seal_lost_epoch();
        if !retired {
            return false;
        }
        if !matches!(self.detach_staging(via), AccelResponse::Detached) {
            return false;
        }
        let session = &mut self.sessions[pair_index(via)];
        if let Some(lane) = session.signal.take() {
            if let Err(lane) = lane.retire() {
                session.signal = Some(lane);
                return false;
            }
        }
        *session = PairSession::EMPTY;
        if self.last_signal_via == Some(via) {
            self.last_signal_via = None;
        }
        true
    }

    fn detach_staging(&mut self, via: u16) -> AccelResponse {
        if !self.claim_arrival(via) {
            return AccelResponse::Refused(AccelRefusal::Denied);
        }
        let Some(session) = self.sessions.get_mut(pair_index(via)) else {
            return AccelResponse::Refused(AccelRefusal::Denied);
        };
        let Some(owned) = session.owned_staging.as_mut() else {
            return if session.attaches != 0 {
                // Losing Rust access at DeviceLost did not retire legacy MAP
                // mappings; never acknowledge those as detached afterwards.
                AccelResponse::Refused(AccelRefusal::Unsupported)
            } else {
                AccelResponse::Detached
            };
        };
        session.staging = None;
        if !owned.retire() {
            return AccelResponse::Refused(AccelRefusal::Busy);
        }
        session.owned_staging = None;
        AccelResponse::Detached
    }

    /// The `Signal` arm: revoke any standing lane, mint afresh off the
    /// endowed channel space, retain root + send, and stage the fresh
    /// recv derive in this reply's grant session.
    fn signal(&mut self, via: u16, sv: &mut Serve<'_, AccelResponse>) -> AccelResponse {
        let Some(session) = self.sessions.get_mut(pair_index(via)) else {
            return AccelResponse::Refused(AccelRefusal::Denied);
        };
        let Some(channels) = self.channels.as_ref() else {
            return AccelResponse::Refused(AccelRefusal::NoChannel);
        };
        // Replace semantics: the old channel dies first, so exactly one
        // live recv per pair ever exists (T6 reaps the granted one).
        if let Some(lane) = session.signal.take() {
            if let Err(lane) = lane.retire() {
                session.signal = Some(lane);
                return AccelResponse::Refused(AccelRefusal::Busy);
            }
        }
        match verbs::mint_channel(channels) {
            Ok(pair) => {
                // The mint's own recv retires at once — the client's is
                // a fresh derive staged below, unwound with the session
                // if delivery refuses (I3).
                if pair.recv.try_retire().is_err() {
                    let _ = SignalLane {
                        root: pair.root,
                        send: pair.send,
                    }
                    .retire();
                    return AccelResponse::Refused(AccelRefusal::Internal);
                }
                let root_slot = pair.root.slot();
                session.signal = Some(SignalLane {
                    root: pair.root,
                    send: pair.send,
                });
                self.last_signal_via = Some(via);
                sv.grant(root_slot, CHANNEL_ENDPOINT_RIGHTS, GrantArg::ChanRecv);
                AccelResponse::Signaled
            }
            Err(_) => AccelResponse::Refused(AccelRefusal::NoChannel),
        }
    }

    /// The `grants_dropped` repair, shared by every instantiation: the
    /// staged recv never reached the client, so the promise is unwound —
    /// the fresh lane is revoked and the reply rewritten to a refusal.
    pub fn grants_unwound(&mut self, response: &mut AccelResponse, emitted: usize) {
        if emitted == 0 {
            return;
        }
        if let Some(via) = self.last_signal_via.take()
            && let Some(session) = self.sessions.get_mut(pair_index(via))
            && let Some(lane) = session.signal.take()
            && let Err(lane) = lane.retire()
        {
            // A stuck retire leaves a retryable tombstone; the next
            // `Signal` retires it again before minting.
            session.signal = Some(lane);
        }
        *response = AccelResponse::Refused(AccelRefusal::Denied);
    }

    /// **The completion pump** — one bounded sweep, then at most one
    /// [`COOKIE_ACCEL_DONE`] notice per pair with news (the
    /// per-drained-batch doctrine: the poll is the truth, the notice
    /// only wakes).  A `WouldBlock`ed notice is dropped and counted —
    /// the client's own poll still observes every terminal event.
    ///
    /// Driven by the row's run loop after each serve drain; the ivpu
    /// row's JOB_DONE relay worker is what makes the loop wake promptly.
    pub async fn pump_completions<T: Transport>(&mut self, transport: &T) -> u32 {
        let Some(due) = self.completion_sweep() else {
            return 0;
        };
        let mut sent = 0u32;
        for (via, count) in due.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            let Some(lane) = self.sessions[via].signal.as_ref() else {
                continue;
            };
            let tx = ChanTx::new(transport, lane.send.slot());
            match tx.send(Self::done_notice(*count)).await {
                Ok(()) => sent += 1,
                Err(_) => {
                    self.dropped_notices = self.dropped_notices.saturating_add(1);
                }
            }
        }
        sent
    }

    /// [`pump_completions`](Self::pump_completions) committed synchronously:
    /// the same sweep and the same one-notice-per-pair doorbell, for a row
    /// that reaps device completions inside its bounded, never-waiting
    /// `poll_background` hook (a route wake finds no ring work, so the
    /// notice must leave from the hook or the client sleeps out its own
    /// timeout). A `WouldBlock`ed notice is dropped and counted exactly as
    /// the async pump drops it.
    pub fn pump_completions_now<T: Transport>(&mut self, transport: &T) -> u32 {
        let Some(due) = self.completion_sweep() else {
            return 0;
        };
        let mut sent = 0u32;
        for (via, count) in due.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            let Some(lane) = self.sessions[via].signal.as_ref() else {
                continue;
            };
            let tx = ChanTx::new(transport, lane.send.slot());
            match tx.send_now(Self::done_notice(*count)) {
                Ok(()) => sent += 1,
                Err(_) => {
                    self.dropped_notices = self.dropped_notices.saturating_add(1);
                }
            }
        }
        sent
    }

    /// One bounded completion sweep: every live event not yet notified is
    /// polled, and the per-pair counts of newly terminal events come back.
    /// `None` when no live epoch remains or the sweep itself sealed one.
    fn completion_sweep(&mut self) -> Option<[u32; PAIRS]> {
        let due = match &mut self.epoch {
            HostEpoch::Live(engine) => engine.due_by_pair(),
            HostEpoch::Idle | HostEpoch::Lost(_) | HostEpoch::Terminal(_) => return None,
        };
        self.seal_lost_epoch();
        if self.restart_required() {
            return None;
        }
        Some(due)
    }

    const fn done_notice(count: u32) -> Notice {
        Notice {
            cookie: COOKIE_ACCEL_DONE,
            payload: count as u64,
        }
    }
}

/// Resolve `[offset, offset+len)` of pair `via`'s staging window, read
/// side.  The bounds check every byte-moving verb goes through: offsets
/// are client data, and the mapped window's length is the only authority
/// on what they may name.
fn stage_slice<'s>(
    sessions: &'s [PairSession; PAIRS],
    via: u16,
    offset: u64,
    len: u64,
) -> Result<&'s [u8], AccelRefusal> {
    let session = sessions.get(pair_index(via)).ok_or(AccelRefusal::Denied)?;
    if session.arrival != Some(via) {
        return Err(AccelRefusal::Denied);
    }
    let window = match session.owned_staging.as_ref() {
        Some(owned) => {
            let (base, len) = owned.window().ok_or(AccelRefusal::NoStaging)?;
            // SAFETY: the live exact view covers len bytes. The request/reply
            // protocol reserves staging until the service has consumed it.
            unsafe { core::slice::from_raw_parts(base, len) }
        }
        None => session.staging.as_deref().ok_or(AccelRefusal::NoStaging)?,
    };
    if len == 0 {
        return Err(AccelRefusal::InvalidArgument);
    }
    let end = offset.checked_add(len).ok_or(AccelRefusal::OutOfBounds)?;
    if end > window.len() as u64 {
        return Err(AccelRefusal::OutOfBounds);
    }
    Ok(&window[offset as usize..end as usize])
}

/// [`stage_slice`]'s mutable twin (the `ReadBuffer` destination).
fn stage_slice_mut<'s>(
    sessions: &'s mut [PairSession; PAIRS],
    via: u16,
    offset: u64,
    len: u64,
) -> Result<&'s mut [u8], AccelRefusal> {
    let session = sessions
        .get_mut(pair_index(via))
        .ok_or(AccelRefusal::Denied)?;
    if session.arrival != Some(via) {
        return Err(AccelRefusal::Denied);
    }
    let window = match session.owned_staging.as_mut() {
        Some(owned) => {
            let (base, len) = owned.window().ok_or(AccelRefusal::NoStaging)?;
            // SAFETY: this request owns the pair's staging use until its
            // reply; detach cannot retire the mapping during this borrow.
            unsafe { core::slice::from_raw_parts_mut(base, len) }
        }
        None => session.staging.as_deref_mut().ok_or(AccelRefusal::NoStaging)?,
    };
    if len == 0 {
        return Err(AccelRefusal::InvalidArgument);
    }
    let end = offset.checked_add(len).ok_or(AccelRefusal::OutOfBounds)?;
    if end > window.len() as u64 {
        return Err(AccelRefusal::OutOfBounds);
    }
    Ok(&mut window[offset as usize..end as usize])
}

/// **Adopt the staging window a request transferred** — the metal half
/// of [`AccelRequest::Attach`], byte-for-byte the swap service's
/// discipline: take the returned view, map it once, keep the mapping for
/// the service's life (a receiver has no unmap verb, so the owner is
/// deliberately forgotten rather than dropped — dropping it would retire
/// the record and leave the mapping pointing at reclaimed frames).
///
/// `None` on any refusal — wrong correlation, nothing delivered, a
/// non-view record, or a mapping the kernel declined.
fn adopt_staging<R>(sv: &mut Serve<'_, R>, announced: SlotRef) -> Option<&'static mut [u8]> {
    let mut input = sv.returned_input(announced)?;
    if !input
        .capability()?
        .is_kind_with(ResourceKind::LeasedRegion, STAGING_RIGHTS)
    {
        return None;
    }
    let owner = input.take()?;
    let slot = owner.slot()?;
    let view = verbs::take_view(slot)?;
    let window = verbs::map_view(view)?;
    core::mem::forget(owner);
    // SAFETY: `map_view` installed `[base, +len)` R/W-USER in THIS
    // task's own space over the client's staging backing, page-aligned
    // and live for the task's whole life (nothing unmaps it — a receiver
    // has no unmap verb, and the owner above is retained).  The client
    // writes bytes there and this service reads/writes them under the
    // request/reply protocol — the same single-owner-at-a-time
    // discipline the pair rings rely on.
    Some(unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) })
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::rc::Rc;
    use alloc::vec;
    use alloc::vec::Vec;

    use core::cell::{Cell, RefCell};

    use kore_abi::task::{Bootstrap, Profile};
    use kore_driver::accel::Workgroups;
    use kore_rt::service::{BindError, Bundle, SystemService, step};
    use kore_rt::svc::SvcProtocol;
    use virtio_accel_core::{
        AllocatedBuffer, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferInfo,
        BufferUsage, ContextDesc, EventState, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
    };
    use virtio_accel_mock::reference;
    type MockAccelerator = kore_driver::accel::ArtifactBackend<virtio_accel_mock::MockAccelerator>;

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum DroppedResource {
        Context,
        Buffer,
        Program,
        Queue,
        Event,
    }

    #[derive(Clone)]
    struct DropAudit {
        resources: Rc<RefCell<Vec<(DroppedResource, bool)>>>,
        quiesced: Rc<Cell<bool>>,
        backend_dropped: Rc<Cell<bool>>,
        buffer_release_attempts: Rc<Cell<u32>>,
    }

    impl DropAudit {
        fn new() -> Self {
            Self {
                resources: Rc::new(RefCell::new(Vec::new())),
                quiesced: Rc::new(Cell::new(false)),
                backend_dropped: Rc::new(Cell::new(false)),
                buffer_release_attempts: Rc::new(Cell::new(0)),
            }
        }
    }

    struct Audited<T> {
        inner: T,
        kind: DroppedResource,
        audit: DropAudit,
    }

    impl<T> Audited<T> {
        fn new(inner: T, kind: DroppedResource, audit: &DropAudit) -> Self {
            Self {
                inner,
                kind,
                audit: audit.clone(),
            }
        }
    }

    impl<T> Drop for Audited<T> {
        fn drop(&mut self) {
            assert!(
                !self.audit.backend_dropped.get(),
                "a provider handle outlived the backend that created it",
            );
            self.audit
                .resources
                .borrow_mut()
                .push((self.kind, self.audit.quiesced.get()));
        }
    }

    /// A backend whose first submission crosses the strongest acceptance
    /// boundary: the event exists, but the complete provider outcome is lost.
    /// Its affine handles audit that the owner quiesced before discarding them.
    #[derive(Clone, Copy)]
    enum DeviceLossPoint {
        HealthLost,
        HealthBusy,
        Submit,
        Poll,
        TerminalEvent,
        TerminalProvider,
        Healthy,
    }

    struct DeviceLostBackend {
        inner: MockAccelerator,
        audit: DropAudit,
        loss_point: DeviceLossPoint,
        pad_buffer_backing: bool,
        misdescribe_buffer: bool,
        reject_buffer_cleanup: bool,
        /// The device witness retained for its loss. None (the default)
        /// keeps every bare-DeviceLost expectation.
        witness: Cell<Option<LossEvidence>>,
        /// Advertise Xe's object limits instead of the mock's: one context
        /// holding one live event, so the device has one event slot.
        one_event_slot: bool,
    }

    impl DeviceLostBackend {
        fn new(audit: DropAudit, loss_point: DeviceLossPoint) -> Self {
            Self {
                witness: Cell::new(None),
                one_event_slot: false,
                inner: MockAccelerator::default(),
                audit,
                loss_point,
                pad_buffer_backing: false,
                misdescribe_buffer: false,
                reject_buffer_cleanup: false,
            }
        }

        /// A provider that returns a real allocation whose backing charge
        /// exceeds the host budget, then rejects release of the unpublished
        /// handle. The host must quarantine this graph rather than leak an
        /// unreachable child and continue serving.
        fn cleanup_rejecting(audit: DropAudit) -> Self {
            Self {
                witness: Cell::new(None),
                one_event_slot: false,
                inner: MockAccelerator::default(),
                audit,
                loss_point: DeviceLossPoint::Poll,
                pad_buffer_backing: true,
                misdescribe_buffer: false,
                reject_buffer_cleanup: true,
            }
        }

        /// Return a handle with provider metadata that does not answer the
        /// requested descriptor. DeviceState must own it before rejecting so
        /// cleanup crosses `free_buffer` rather than Rust `Drop` directly.
        fn misdescribing(audit: DropAudit) -> Self {
            Self {
                witness: Cell::new(None),
                one_event_slot: false,
                inner: MockAccelerator::default(),
                audit,
                loss_point: DeviceLossPoint::Poll,
                pad_buffer_backing: false,
                misdescribe_buffer: true,
                reject_buffer_cleanup: false,
            }
        }
    }

    impl Drop for DeviceLostBackend {
        fn drop(&mut self) {
            self.audit.backend_dropped.set(true);
        }
    }

    impl NativeAccelerator for DeviceLostBackend {
        fn configure_profiling(
            &self,
            _context: &Self::Context,
            _enabled: bool,
        ) -> Result<(), BackendError> {
            self.poll_health()
        }

        fn failure_report(&self, out: &mut dyn core::fmt::Write) {
            assert!(!self.audit.backend_dropped.get());
            let _ = out.write_str("retained test failure");
        }

        fn loss_evidence(&self) -> Option<LossEvidence> {
            self.witness.get()
        }

        fn poll_health(&self) -> Result<(), BackendError> {
            match self.loss_point {
                DeviceLossPoint::HealthLost => Err(BackendError::DeviceLost),
                DeviceLossPoint::HealthBusy => Err(BackendError::Busy),
                DeviceLossPoint::Submit
                | DeviceLossPoint::Poll
                | DeviceLossPoint::TerminalEvent
                | DeviceLossPoint::TerminalProvider
                | DeviceLossPoint::Healthy => Ok(()),
            }
        }
    }

    impl Accelerator for DeviceLostBackend {
        type Context = Audited<<MockAccelerator as Accelerator>::Context>;
        type Buffer = Audited<<MockAccelerator as Accelerator>::Buffer>;
        type Program = Audited<<MockAccelerator as Accelerator>::Program>;
        type Queue = Audited<<MockAccelerator as Accelerator>::Queue>;
        type Event = Audited<<MockAccelerator as Accelerator>::Event>;

        fn device_info(&self) -> Result<DeviceInfo, BackendError> {
            let mut info = self.inner.device_info()?;
            if self.one_event_slot {
                info.limits.max_contexts = 1;
                info.limits.max_events_per_context = 1;
            }
            Ok(info)
        }

        fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
            self.inner
                .create_context(desc)
                .map(|inner| Audited::new(inner, DroppedResource::Context, &self.audit))
        }

        fn destroy_context(
            &self,
            context: Self::Context,
        ) -> Result<(), ReleaseFailure<Self::Context>> {
            drop(context);
            Ok(())
        }

        fn allocate_buffer(
            &self,
            context: &Self::Context,
            desc: BufferDesc,
        ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
            let (buffer, info) = self
                .inner
                .allocate_buffer(&context.inner, desc)?
                .into_parts();
            let info = if self.misdescribe_buffer {
                let actual_desc = BufferDesc::new(
                    desc.bytes(),
                    desc.alignment(),
                    desc.domain,
                    desc.usage | BufferUsage::TRANSFER_SOURCE,
                )?;
                BufferInfo::new(
                    actual_desc,
                    info.allocation_bytes(),
                    info.alignment(),
                    info.properties(),
                )?
            } else if self.pad_buffer_backing {
                BufferInfo::new(
                    info.desc(),
                    info.allocation_bytes()
                        .checked_add(64)
                        .ok_or(BackendError::OutOfBounds)?,
                    info.alignment(),
                    info.properties(),
                )?
            } else {
                info
            };
            Ok(AllocatedBuffer::new(
                Audited::new(buffer, DroppedResource::Buffer, &self.audit),
                info,
            ))
        }

        fn write_buffer(
            &self,
            buffer: &mut Self::Buffer,
            offset: u64,
            data: &dyn virtio_accel_core::ByteSource,
        ) -> Result<(), BackendError> {
            self.inner.write_buffer(&mut buffer.inner, offset, data)
        }

        fn read_buffer(
            &self,
            buffer: &Self::Buffer,
            offset: u64,
            data: &mut dyn virtio_accel_core::ByteSink,
        ) -> Result<(), BackendError> {
            self.inner.read_buffer(&buffer.inner, offset, data)
        }

        fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
            self.audit
                .buffer_release_attempts
                .set(self.audit.buffer_release_attempts.get().saturating_add(1));
            if self.reject_buffer_cleanup {
                return Err(ReleaseFailure::Rejected {
                    error: BackendError::Busy,
                    resource: buffer,
                });
            }
            drop(buffer);
            Ok(())
        }

        fn load_program(
            &self,
            context: &Self::Context,
            artifact: ArtifactRef<'_>,
        ) -> Result<Self::Program, BackendError> {
            self.inner
                .load_program(&context.inner, artifact)
                .map(|inner| Audited::new(inner, DroppedResource::Program, &self.audit))
        }

        fn unload_program(
            &self,
            program: Self::Program,
        ) -> Result<(), ReleaseFailure<Self::Program>> {
            drop(program);
            Ok(())
        }

        fn create_queue(
            &self,
            context: &Self::Context,
            desc: QueueDesc,
        ) -> Result<Self::Queue, BackendError> {
            self.inner
                .create_queue(&context.inner, desc)
                .map(|inner| Audited::new(inner, DroppedResource::Queue, &self.audit))
        }

        fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
            drop(queue);
            Ok(())
        }

        fn submit(
            &self,
            queue: &Self::Queue,
            program: &Self::Program,
            bindings: &[BindingRef<'_, Self::Buffer>],
            timeout: Timeout,
        ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
            let native: Vec<_> = bindings
                .iter()
                .map(|binding| BindingRef {
                    slot: binding.slot,
                    buffer: &binding.buffer.inner,
                    range: binding.range,
                    access: binding.access,
                })
                .collect();
            let event = self
                .inner
                .submit(&queue.inner, &program.inner, &native, timeout)
                .map_err(|failure| match failure {
                    SubmitFailure::Rejected(error) => SubmitFailure::Rejected(error),
                    SubmitFailure::Indeterminate { error, event: _ } => {
                        SubmitFailure::Rejected(error)
                    }
                })?;
            let event = Audited::new(event, DroppedResource::Event, &self.audit);
            match self.loss_point {
                DeviceLossPoint::Submit => Err(SubmitFailure::Indeterminate {
                    error: BackendError::DeviceLost,
                    event,
                }),
                DeviceLossPoint::HealthLost
                | DeviceLossPoint::HealthBusy
                | DeviceLossPoint::Poll
                | DeviceLossPoint::TerminalEvent
                | DeviceLossPoint::TerminalProvider
                | DeviceLossPoint::Healthy => Ok(event),
            }
        }

        fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
            match self.loss_point {
                DeviceLossPoint::HealthLost
                | DeviceLossPoint::HealthBusy
                | DeviceLossPoint::Submit
                | DeviceLossPoint::Healthy => self.inner.poll_event(&event.inner),
                DeviceLossPoint::Poll => Err(BackendError::DeviceLost),
                DeviceLossPoint::TerminalEvent => Ok(EventState::Failed(BackendError::DeviceLost)),
                DeviceLossPoint::TerminalProvider => {
                    Ok(EventState::Failed(BackendError::External {
                        domain: 0x5550_5649,
                        code: 0xfedc_ba98,
                    }))
                }
            }
        }

        fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
            self.inner.cancel_event(&event.inner)
        }

        fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
            drop(event);
            Ok(())
        }
    }

    /// A backend that records explicit workgroup geometry, delegating every
    /// other operation to the reference mock. It overrides the default
    /// rejection so a test can prove that a client-built launch survives the
    /// wire and engine byte-exact.
    struct RecordingBackend {
        inner: MockAccelerator,
        last_workgroups: Cell<Option<Workgroups>>,
        /// Each member of the most recent ordered submission: its launch,
        /// barrier edge, and binding count, in table order.
        last_sequence: RefCell<Vec<(LaunchShape, bool, usize)>>,
    }

    impl RecordingBackend {
        fn new() -> Self {
            Self {
                inner: MockAccelerator::default(),
                last_workgroups: Cell::new(None),
                last_sequence: RefCell::new(Vec::new()),
            }
        }

        /// The geometry of the most recent submission the engine handed the
        /// backend, or `None` before any.
        fn recorded(&self) -> Option<Workgroups> {
            self.last_workgroups.get()
        }
    }

    impl NativeAccelerator for RecordingBackend {
        const EXPLICIT_WORKGROUPS: bool = true;
        fn submit_with_launch(
            &self,
            queue: &Self::Queue,
            program: &Self::Program,
            bindings: &[BindingRef<'_, Self::Buffer>],
            launch: LaunchShape,
            timeout: Timeout,
        ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
            if let LaunchShape::Workgroups(workgroups) = launch {
                self.last_workgroups.set(Some(workgroups));
            }
            self.inner.submit(queue, program, bindings, timeout)
        }

        /// Record the table exactly as the engine lowered it, then run its
        /// first member on the reference mock so the event is real.
        fn submit_ordered(
            &self,
            queue: &Self::Queue,
            executions: &[kore_driver::accel::OrderedExecution<'_, Self::Program, Self::Buffer>],
            timeout: Timeout,
        ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
            *self.last_sequence.borrow_mut() = executions
                .iter()
                .map(|member| (member.launch, member.barrier, member.bindings.len()))
                .collect();
            let first = &executions[0];
            self.inner
                .submit(queue, first.program, first.bindings, timeout)
        }
    }

    impl Accelerator for RecordingBackend {
        type Context = <MockAccelerator as Accelerator>::Context;
        type Buffer = <MockAccelerator as Accelerator>::Buffer;
        type Program = <MockAccelerator as Accelerator>::Program;
        type Queue = <MockAccelerator as Accelerator>::Queue;
        type Event = <MockAccelerator as Accelerator>::Event;

        fn device_info(&self) -> Result<DeviceInfo, BackendError> {
            self.inner.device_info()
        }
        fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
            self.inner.create_context(desc)
        }
        fn destroy_context(
            &self,
            context: Self::Context,
        ) -> Result<(), ReleaseFailure<Self::Context>> {
            self.inner.destroy_context(context)
        }
        fn allocate_buffer(
            &self,
            context: &Self::Context,
            desc: BufferDesc,
        ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
            self.inner.allocate_buffer(context, desc)
        }
        fn write_buffer(
            &self,
            buffer: &mut Self::Buffer,
            offset: u64,
            data: &dyn virtio_accel_core::ByteSource,
        ) -> Result<(), BackendError> {
            self.inner.write_buffer(buffer, offset, data)
        }
        fn read_buffer(
            &self,
            buffer: &Self::Buffer,
            offset: u64,
            data: &mut dyn virtio_accel_core::ByteSink,
        ) -> Result<(), BackendError> {
            self.inner.read_buffer(buffer, offset, data)
        }
        fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
            self.inner.free_buffer(buffer)
        }
        fn load_program(
            &self,
            context: &Self::Context,
            artifact: ArtifactRef<'_>,
        ) -> Result<Self::Program, BackendError> {
            self.inner.load_program(context, artifact)
        }
        fn unload_program(
            &self,
            program: Self::Program,
        ) -> Result<(), ReleaseFailure<Self::Program>> {
            self.inner.unload_program(program)
        }
        fn create_queue(
            &self,
            context: &Self::Context,
            desc: QueueDesc,
        ) -> Result<Self::Queue, BackendError> {
            self.inner.create_queue(context, desc)
        }
        fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
            self.inner.destroy_queue(queue)
        }
        fn submit(
            &self,
            queue: &Self::Queue,
            program: &Self::Program,
            bindings: &[BindingRef<'_, Self::Buffer>],
            timeout: Timeout,
        ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
            self.inner.submit(queue, program, bindings, timeout)
        }

        fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
            self.inner.poll_event(event)
        }
        fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
            self.inner.cancel_event(event)
        }
        fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
            self.inner.destroy_event(event)
        }
    }

    #[test]
    fn native_workgroup_query_is_separate_from_upstream_metadata_and_survives_retirement() {
        let backend = RecordingBackend::new();
        let upstream = backend.device_info().unwrap();
        let bit = kore_driver::accel::EXPLICIT_WORKGROUPS;
        assert_eq!(upstream.capabilities.bits() & bit, 0);
        assert_eq!(upstream.validate(), Ok(()));
        let explicit = LaunchShape::Workgroups(Workgroups::new([1; 3], [1; 3]).unwrap());
        assert_eq!(backend.validate_launch_shape(explicit), Ok(()));
        let plain = MockAccelerator::default();
        assert_eq!(
            plain.validate_launch_shape(explicit),
            Err(BackendError::Unsupported)
        );
        assert_eq!(
            plain.validate_launch_shape(LaunchShape::ArtifactDefined),
            Ok(())
        );
        let mut svc = AccelService::host(backend, None, 1, 1 << 32, 1 << 32);
        let before = svc.answer(0, AccelRequest::Query);
        let AccelResponse::Info(info) = before else {
            panic!("native query");
        };
        assert_eq!(info.capabilities, upstream.capabilities.bits() | bit);
        svc.retire_live_epoch();
        assert_eq!(svc.answer(0, AccelRequest::Query), before);
        drop(svc.take_lost_epoch().unwrap()); // no hardware or admitted work
        assert_eq!(svc.answer(0, AccelRequest::Query), before);
    }

    /// **The consumer path delivers the geometry.** [`Workgroups`] built the
    /// way a client builds it — through [`crate::dispatch`], not by hand —
    /// survives the wire codec (`encode_req`/`decode_req`) *and* the engine
    /// and reaches the backend's `submit_with_launch` byte-identical.
    #[test]
    fn explicit_workgroups_reach_the_backend_through_the_wire() {
        use crate::wire::{decode_req, encode_req};

        let mut svc = AccelService::host(RecordingBackend::new(), None, 1, 1 << 32, 1 << 32);
        assert!(
            svc.hosted(),
            "the recording wrapper inherits the mock's metadata"
        );
        assert_eq!(
            svc.backend().unwrap().recorded(),
            None,
            "nothing submitted yet"
        );

        // Stand up the object graph up to a queue over a lawful host staging
        // window (an ordinary leaked buffer, the swap tests' shape).
        let window = Box::leak(vec![0u8; 4096].into_boxed_slice());
        svc.attach_staging(0, window).expect("attaches");
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(svc.answer(
            0,
            AccelRequest::CreateBuffer {
                ctx,
                bytes: 64,
                align: 16,
                domain: 3, // shared
                usage: BufferUsage::MUTABLE_STATE.bits(),
            },
        ));
        let artifact = reference::ReferenceArtifact::xor(7, 0x5a);
        let bytes = artifact.as_bytes();
        svc.sessions[0].staging.as_deref_mut().unwrap()[..bytes.len()].copy_from_slice(bytes);
        let program = expect_id(svc.answer(
            0,
            AccelRequest::LoadProgram {
                ctx,
                stage: 0,
                len: bytes.len() as u64,
                resident_bytes: reference::RESIDENT_BYTES,
                format: reference::ARTIFACT_FORMAT.get(),
                target: reference::TARGET_IDENTITY.0,
            },
        ));
        let queue = expect_id(svc.answer(0, AccelRequest::CreateQueue { ctx, flags: 0 }));

        // The consumer's geometry — lowered through the bridge, a non-trivial
        // grid so a dropped or defaulted dispatch cannot pass unnoticed.
        let workgroups = crate::dispatch::for_grid([16, 8, 2], [64, 1, 1]).unwrap();
        let bindings = AccelBindings::new(&[AccelBinding {
            buffer,
            offset: 0,
            len: 8,
            slot: 7,
            access: AccessMode::ReadWrite,
        }])
        .unwrap();

        // Round-trip the whole submission through the wire codec first (as a
        // client would over the ring), then serve the decoded form — so the
        // geometry is proven across the encode/decode seam AND the engine.
        let request = AccelRequest::SubmitLaunch {
            queue,
            program,
            timeout_ns: 0,
            bindings,
            launch: LaunchShape::Workgroups(workgroups),
        };
        let decoded = decode_req(&encode_req(&request)).expect("the submit round-trips the wire");
        assert_eq!(decoded, request, "the wire preserves the whole submission");
        let _event = expect_id(svc.answer(0, decoded));

        // The backend saw exactly the portable geometry the consumer
        // expressed — group count and logical local size, unmodified end to
        // end. SIMD width remains provider-owned.
        assert_eq!(svc.backend().unwrap().recorded(), Some(workgroups));
        let seen = svc.backend().unwrap().recorded().unwrap();
        assert_eq!(seen.group_count(), [16, 8, 2]);
        assert_eq!(seen.local_size(), [64, 1, 1]);
    }

    /// **An ordered table reaches the backend member by member.** Entries
    /// staged by the client packer survive the staging codec and the engine
    /// with each member's own launch shape and barrier edge, repeated
    /// programs share one table, and a reserved flag bit or a backend
    /// without explicit workgroups refuses before any event exists.
    #[test]
    fn staged_sequence_members_reach_the_backend_with_launch_and_barrier() {
        use crate::client::{OrderedLaunch, ProgramReuse, stage_ordered_launches};
        const ENTRY: usize = core::mem::size_of::<kore_abi::svc::AccelSequenceEntryWire>();

        let mut svc = AccelService::host(RecordingBackend::new(), None, 1, 1 << 32, 1 << 32);
        stage(&mut svc, 4 * 4096);
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(svc.answer(
            0,
            AccelRequest::CreateBuffer {
                ctx,
                bytes: 64,
                align: 16,
                domain: 3, // shared
                usage: BufferUsage::MUTABLE_STATE.bits(),
            },
        ));
        let artifact = reference::ReferenceArtifact::xor(7, 0x5a);
        let bytes = artifact.as_bytes();
        write_staging(&mut svc, 0, bytes);
        let program = expect_id(svc.answer(
            0,
            AccelRequest::LoadProgram {
                ctx,
                stage: 0,
                len: bytes.len() as u64,
                resident_bytes: reference::RESIDENT_BYTES,
                format: reference::ARTIFACT_FORMAT.get(),
                target: reference::TARGET_IDENTITY.0,
            },
        ));
        let queue = expect_id(svc.answer(0, AccelRequest::CreateQueue { ctx, flags: 0 }));
        let bindings = AccelBindings::new(&[AccelBinding {
            buffer,
            offset: 0,
            len: 8,
            slot: 7,
            access: AccessMode::ReadWrite,
        }])
        .unwrap();
        let explicit = LaunchShape::Workgroups(Workgroups::new([16, 8, 2], [64, 1, 1]).unwrap());
        let members = [
            (explicit, true),
            (explicit, false),
            (LaunchShape::ArtifactDefined, true),
        ]
        .map(|(launch, barrier)| OrderedLaunch {
            queue,
            program,
            bindings,
            timeout_ns: 0,
            launch,
            barrier,
            gate: None,
            signal: None,
        });

        let mut table = [0u8; 3 * ENTRY];
        let mut groups = [None; 3];
        assert_eq!(
            stage_ordered_launches(&members, &mut table, 8192, &mut groups, ProgramReuse::Share),
            Ok(1)
        );
        write_staging(&mut svc, 8192, &table);
        let group = groups[0].unwrap();
        let submit = AccelRequest::SubmitSequence {
            queue,
            stage: group.stage,
            len: group.len,
            count: group.count,
            timeout_ns: 0,
        };
        let _event = expect_id(svc.answer(0, submit));
        assert_eq!(
            *svc.backend().unwrap().last_sequence.borrow(),
            [
                (explicit, true, 1),
                (explicit, false, 1),
                (LaunchShape::ArtifactDefined, true, 1)
            ]
        );

        // A reserved flag bit in any member refuses the whole table at decode.
        svc.backend().unwrap().last_sequence.borrow_mut().clear();
        let mut poisoned = table;
        poisoned[ENTRY + 12] |= 0x02;
        write_staging(&mut svc, 8192, &poisoned);
        assert_eq!(
            svc.answer(0, submit),
            AccelResponse::Refused(AccelRefusal::InvalidArgument)
        );
        assert!(svc.backend().unwrap().last_sequence.borrow().is_empty());

        // A backend without the explicit-workgroup extension refuses the
        // table at the single-launch gate, whatever its other members say.
        let mut plain = AccelService::host(MockAccelerator::default(), None, 1, 1 << 32, 1 << 32);
        stage(&mut plain, 4 * 4096);
        let ctx = expect_id(plain.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let queue = expect_id(plain.answer(0, AccelRequest::CreateQueue { ctx, flags: 0 }));
        write_staging(&mut plain, 8192, &table);
        assert_eq!(
            plain.answer(
                0,
                AccelRequest::SubmitSequence {
                    queue,
                    stage: group.stage,
                    len: group.len,
                    count: group.count,
                    timeout_ns: 0,
                }
            ),
            AccelResponse::Refused(AccelRefusal::Unsupported)
        );
    }

    /// The host-proof endowment: nothing — the mock needs no hardware,
    /// and the machinery under proof (serve body, protocol consts,
    /// summarize) does not care.  A real row's endowment holds its
    /// device claim, DMA pool, and `role::CHANNEL_SPACE` grant.
    /// (`pub` only because the trait impl's associated type leaks the
    /// name; the tests module itself is `cfg(test)`-private.)
    pub struct MockEndowment;

    impl Bundle for MockEndowment {
        const PROFILE: Profile = Profile::new("accel-mock", &[]);
        fn resolve(_: &Bootstrap<'_>) -> Result<Self, BindError> {
            Ok(MockEndowment)
        }
    }

    /// The **host proof**: `AccelService<A>` satisfies `SystemService`
    /// exactly as a hardware row will — one `bind` building the host,
    /// one `serve` delegating to the shared step, the shared return
    /// mode.  This instantiation is test-only because the mock is; the
    /// impl shape is the ivpu row's to copy.
    impl SystemService for AccelService<MockAccelerator> {
        const NAME: &'static str = "accel-mock";
        const RETURN_MODE: ReturnMode = ACCEL_RETURN_MODE;
        type Endowment = MockEndowment;

        fn bind(_: MockEndowment) -> Self {
            AccelService::host(MockAccelerator::default(), None, 1, 1 << 32, 1 << 32)
        }

        fn serve(&mut self, req: AccelRequest, sv: &mut Serve<'_, AccelResponse>) {
            self.step(req, sv);
        }

        fn grants_dropped(&mut self, response: &mut AccelResponse, emitted: usize) {
            self.grants_unwound(response, emitted);
        }
    }

    /// Exercise the same public serve boundary with a backend whose
    /// provider-wide health edge is controllable. Keeping this as a concrete
    /// test instantiation proves the generic host, not a test-only shortcut
    /// around [`AccelService::step`].
    impl SystemService for AccelService<DeviceLostBackend> {
        const NAME: &'static str = "accel-health-audit";
        const RETURN_MODE: ReturnMode = ACCEL_RETURN_MODE;
        type Endowment = MockEndowment;

        fn bind(_: MockEndowment) -> Self {
            AccelService::host(
                DeviceLostBackend::new(DropAudit::new(), DeviceLossPoint::HealthLost),
                None,
                1,
                1 << 32,
                1 << 32,
            )
        }

        fn serve(&mut self, req: AccelRequest, sv: &mut Serve<'_, AccelResponse>) {
            self.step(req, sv);
        }

        fn grants_dropped(&mut self, response: &mut AccelResponse, emitted: usize) {
            self.grants_unwound(response, emitted);
        }
    }

    fn hosted() -> AccelService<MockAccelerator> {
        let svc = AccelService::host(MockAccelerator::default(), None, 1, 1 << 32, 1 << 32);
        assert!(svc.hosted(), "the mock's metadata validates");
        svc
    }

    #[test]
    fn lost_and_terminal_hosts_never_classify_as_idle_initialization() {
        let idle = AccelService::<MockAccelerator>::idle(None);
        assert!(idle.is_idle());
        let mut svc = hosted();
        assert!(!svc.is_idle());
        svc.retire_live_epoch();
        assert!(!svc.hosted());
        assert!(
            !svc.is_idle(),
            "loss still owns the engine requiring containment"
        );
        let lost = svc
            .take_lost_epoch()
            .expect("lost engine remains reachable");
        assert!(
            !svc.is_idle(),
            "terminal identity must not enter initialization recovery"
        );
        drop(lost); // the mock has no hardware or admitted work
    }

    #[test]
    fn return_rendezvous_admits_both_protocol_species() {
        // A typed leaf-only post rejects the client's owned supply before
        // ProvisionMemory can run. Check the deployed service policy against
        // both actual capability movements, not just its Attach predecessor.
        let mode = <AccelService<MockAccelerator> as SystemService>::RETURN_MODE;
        for (kind, rights) in [
            (ResourceKind::LeasedRegion, STAGING_RIGHTS),
            (ResourceKind::MemLease, PROVISION_RIGHTS),
            (ResourceKind::MemoryFabricRoute, PROVISION_RIGHTS),
        ] {
            let admitted = match mode {
                ReturnMode::RetainedAny => true,
                ReturnMode::Retained {
                    kind: expected,
                    minimum_rights,
                } => expected == kind && rights.contains(minimum_rights),
                _ => false, // walking/fixed modes do not retain these owners
            };
            assert!(admitted, "the return post must admit {kind:?}");
        }
    }

    #[test]
    fn owned_staging_requires_geometry_and_detach_survives_device_loss() {
        let mut svc = hosted();
        let attach = AccelRequest::AttachOwned {
            announced: SlotRef::from_wire(9),
        };
        assert_eq!(
            step(&mut svc, attach).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::Unsupported))
        );
        assert!(svc.configure_staging(kore_abi::task::layout::Window::new(8 << 30, 64 << 30)));
        // With geometry present, a request without a real transferred parent
        // still cannot manufacture staging authority.
        assert_eq!(
            step(&mut svc, attach).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::NoStaging))
        );
        assert_eq!(
            step(
                &mut svc,
                AccelRequest::Attach {
                    announced: SlotRef::from_wire(9)
                }
            )
            .into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::NoStaging))
        );
        assert_eq!(
            step(&mut svc, attach).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::Busy))
        );
        stage(&mut svc, 4096);
        assert_eq!(
            step(&mut svc, AccelRequest::DetachStaging).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::Unsupported))
        );
        assert!(svc.sessions[0].staging.is_some());
        svc.retire_live_epoch();
        assert_eq!(
            step(&mut svc, AccelRequest::DetachStaging).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::Unsupported))
        );
        let mut svc = hosted();
        svc.retire_live_epoch();
        assert_eq!(
            step(&mut svc, AccelRequest::DetachStaging).into_reply(),
            Some(AccelResponse::Detached)
        );
        assert_eq!(
            step(&mut svc, AccelRequest::CreateContext { flags: 0 }).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::DeviceLost))
        );
    }

    #[test]
    fn generic_host_refuses_unsupported_memory_provisioning() {
        let mut svc = hosted();
        assert_eq!(
            svc.answer(
                0,
                AccelRequest::ProvisionMemory {
                    announced: SlotRef::from_wire(17),
                    buffer_bytes: 4096,
                    program_bytes: 0,
                }
            ),
            AccelResponse::Refused(AccelRefusal::Unsupported)
        );
    }

    /// The founded policy is the ceiling; a pair funds inside it whenever it
    /// likes — before, between, and after object lifecycles — and its budget
    /// dies with the pair, so the same slot's next generation provisions
    /// afresh instead of inheriting a closed epoch (the aftershock-relaunch
    /// regression: one `CreateContext` used to close provisioning forever).
    #[test]
    fn provisioning_is_per_pair_and_reopens_with_the_pair() {
        let mut svc = AccelService::host(MockAccelerator::default(), None, 1, 16384, 16384);
        let published = Cell::new(false);
        svc.provision_with(0, 8192, 8192, |required| {
            assert_eq!(required, 16384);
            published.set(true);
            Ok(())
        })
        .unwrap();
        assert!(published.get());
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(svc.answer(
            0,
            AccelRequest::CreateBuffer {
                ctx,
                bytes: 4096,
                align: 4096,
                domain: 3,
                usage: BufferUsage::TRANSFER_DESTINATION.bits(),
            },
        ));
        // Funding after the first context is ordinary now.
        svc.provision_with(0, 4096, 0, |_| Ok(())).unwrap();
        // The device-wide ceiling still refuses over-funding honestly.
        assert_eq!(
            svc.provision_with(0, 8192, 0, |_| panic!("over-ceiling published")),
            Err(AccelRefusal::ResourceLimit)
        );
        assert_eq!(
            svc.answer(0, AccelRequest::FreeBuffer { buffer }),
            AccelResponse::Released { subject: buffer }
        );
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyContext { ctx }),
            AccelResponse::Released { subject: ctx }
        );
        // The pair's departure returns its funding: the slot's next
        // generation may fund the full ceiling again.
        assert!(svc.close_peer(0));
        svc.provision_with(0, 16384, 16384, |required| {
            assert_eq!(required, 32768);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn existing_pool_admits_only_live_budgets_covered_by_retained_capacity() {
        let mut svc = AccelService::host(MockAccelerator::default(), None, 1, 64 << 20, 64 << 20);
        let retained_capacity = 32 << 20;
        // Init's stable 32 MiB supply funds only one page of its own budget;
        // the provider retains the full extent for subsequent pair requests.
        svc.provision_with(0, 4096, 0, |required| {
            assert_eq!(required, 4096);
            Ok(())
        })
        .unwrap();
        let reserve = |svc: &mut AccelService<MockAccelerator>, via, buffers, programs| {
            svc.provision_with(via, buffers, programs, |required| {
                (required <= retained_capacity)
                    .then_some(())
                    .ok_or(AccelRefusal::ResourceLimit)
            })
        };
        reserve(&mut svc, 1, 4096, 5 << 20).unwrap();
        assert_eq!(
            reserve(&mut svc, 2, 0, 30 << 20),
            Err(AccelRefusal::ResourceLimit)
        );
        // A refused reservation committed no pair budget.
        reserve(&mut svc, 2, 4096, 4096).unwrap();
    }

    /// A funded pair is bounded by its own funding even while the aggregate
    /// ceiling still has room, and a committed release credits the exact
    /// charge back.
    #[test]
    fn funded_pair_spends_only_its_funding_and_releases_credit_back() {
        let mut svc = AccelService::host(MockAccelerator::default(), None, 1, 16384, 16384);
        svc.provision_with(0, 4096, 0, |_| Ok(())).unwrap();
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let create = |svc: &mut AccelService<MockAccelerator>| {
            svc.answer(
                0,
                AccelRequest::CreateBuffer {
                    ctx,
                    bytes: 4096,
                    align: 4096,
                    domain: 3,
                    usage: BufferUsage::TRANSFER_DESTINATION.bits(),
                },
            )
        };
        let buffer = expect_id(create(&mut svc));
        assert_eq!(
            create(&mut svc),
            AccelResponse::Refused(AccelRefusal::ResourceLimit)
        );
        assert_eq!(
            svc.answer(0, AccelRequest::FreeBuffer { buffer }),
            AccelResponse::Released { subject: buffer }
        );
        expect_id(create(&mut svc));
    }

    #[test]
    fn provisioning_rejects_invalid_budgets_and_seals_publication_loss() {
        let mut svc = hosted();
        assert_eq!(
            svc.provision_with(0, u64::MAX, 1, |_| panic!("invalid budget published")),
            Err(AccelRefusal::InvalidArgument)
        );
        assert_eq!(
            svc.provision_with(0, 4096, 0, |_| Err(AccelRefusal::DeviceLost)),
            Err(AccelRefusal::DeviceLost)
        );
        assert!(svc.restart_required());
        assert_eq!(
            svc.provision_with(0, 4096, 0, |_| panic!("lost epoch published")),
            Err(AccelRefusal::DeviceLost)
        );
    }

    fn expect_id(response: AccelResponse) -> u64 {
        match response {
            AccelResponse::ContextMade { ctx: id }
            | AccelResponse::BufferMade { buffer: id }
            | AccelResponse::ProgramMade { program: id }
            | AccelResponse::QueueMade { queue: id }
            | AccelResponse::Submitted { event: id } => id,
            other => panic!("expected a fresh id, got {other:?}"),
        }
    }

    /// Give the service a staging window the way host tests lawfully can
    /// — an ordinary leaked buffer (the swap tests' shape).
    fn stage<A: NativeAccelerator>(svc: &mut AccelService<A>, bytes: usize) {
        let window = Box::leak(vec![0u8; bytes].into_boxed_slice());
        svc.attach_staging(0, window).expect("attaches");
    }

    fn write_staging<A: NativeAccelerator>(svc: &mut AccelService<A>, offset: usize, bytes: &[u8]) {
        let window = svc.sessions[0].staging.as_deref_mut().unwrap();
        window[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    fn read_staging<A: NativeAccelerator>(
        svc: &AccelService<A>,
        offset: usize,
        len: usize,
    ) -> &[u8] {
        &svc.sessions[0].staging.as_deref().unwrap()[offset..offset + len]
    }

    /// The whole object model, end to end through the serve step: the
    /// conformance target's xor program runs on the mock behind the
    /// protocol — context, buffer, staged write, program load, queue,
    /// submit, harness completion, poll, staged read-back, and the full
    /// release ladder.
    #[test]
    #[allow(deprecated)]
    fn the_reference_program_runs_end_to_end_through_the_protocol() {
        let mut svc = hosted();
        stage(&mut svc, 4096);

        let ctx = expect_id(
            step(&mut svc, AccelRequest::CreateContext { flags: 0 })
                .into_reply()
                .unwrap(),
        );
        let buffer = expect_id(
            step(
                &mut svc,
                AccelRequest::CreateBuffer {
                    ctx,
                    bytes: 64,
                    align: 16,
                    domain: 3, // shared
                    usage: (BufferUsage::TRANSFER_SOURCE
                        | BufferUsage::TRANSFER_DESTINATION
                        | BufferUsage::MUTABLE_STATE)
                        .bits(),
                },
            )
            .into_reply()
            .unwrap(),
        );

        // Stage the input bytes and write them into the buffer.
        let input = [0x00, 0x11, 0x7f, 0x80, 0xa5, 0xff, 0x3c, 0xc3];
        write_staging(&mut svc, 0, &input);
        assert_eq!(
            step(
                &mut svc,
                AccelRequest::WriteBuffer {
                    buffer,
                    offset: 0,
                    stage: 0,
                    len: 8
                }
            )
            .into_reply()
            .unwrap(),
            AccelResponse::Wrote { buffer },
        );

        // Stage the reference artifact (xor 0x5a over slot 7) and load it.
        let artifact = reference::ReferenceArtifact::xor(7, 0x5a);
        let bytes = artifact.as_bytes();
        write_staging(&mut svc, 256, bytes);
        let program = expect_id(
            step(
                &mut svc,
                AccelRequest::LoadProgram {
                    ctx,
                    stage: 256,
                    len: bytes.len() as u64,
                    resident_bytes: reference::RESIDENT_BYTES,
                    format: reference::ARTIFACT_FORMAT.get(),
                    target: reference::TARGET_IDENTITY.0,
                },
            )
            .into_reply()
            .unwrap(),
        );

        let queue = expect_id(
            step(&mut svc, AccelRequest::CreateQueue { ctx, flags: 0 })
                .into_reply()
                .unwrap(),
        );

        let bindings = AccelBindings::new(&[AccelBinding {
            buffer,
            offset: 0,
            len: 8,
            slot: 7,
            access: AccessMode::ReadWrite,
        }])
        .unwrap();
        assert_eq!(
            step(
                &mut svc,
                AccelRequest::SubmitLaunch {
                    queue,
                    program,
                    timeout_ns: 0,
                    bindings,
                    launch: LaunchShape::Workgroups(
                        Workgroups::new([8, 1, 1], [1, 1, 1]).unwrap(),
                    ),
                },
            )
            .into_reply()
            .unwrap(),
            AccelResponse::Refused(AccelRefusal::Unsupported),
            "a backend that did not advertise explicit workgroups must reject them",
        );
        assert_eq!(
            step(
                &mut svc,
                AccelRequest::Submit {
                    queue,
                    program,
                    timeout_ns: 0,
                    bindings,
                    dispatch: Dispatch::new([8, 1, 1], [1, 1, 1], 8).unwrap(),
                },
            )
            .into_reply()
            .unwrap(),
            AccelResponse::Refused(AccelRefusal::Unsupported),
            "a default backend must reject non-scalar OP13 instead of dropping its geometry",
        );
        let event = expect_id(
            step(
                &mut svc,
                AccelRequest::SubmitLaunch {
                    queue,
                    program,
                    timeout_ns: 0,
                    bindings,
                    launch: LaunchShape::ArtifactDefined,
                },
            )
            .into_reply()
            .unwrap(),
        );

        // Profiling cannot reset an interval while its event remains owned.
        assert_eq!(
            svc.answer(0, AccelRequest::ConfigureProfiling { ctx, enabled: true }),
            AccelResponse::Refused(AccelRefusal::Busy)
        );

        // The event owns its bindings until it is destroyed. Transfers must
        // not race that ownership, even though the provider's handle methods
        // themselves only take ordinary Rust borrows.
        assert_eq!(
            step(
                &mut svc,
                AccelRequest::WriteBuffer {
                    buffer,
                    offset: 0,
                    stage: 0,
                    len: 1
                }
            )
            .into_reply()
            .unwrap(),
            AccelResponse::Refused(AccelRefusal::Busy),
        );
        assert_eq!(
            step(
                &mut svc,
                AccelRequest::ReadBuffer {
                    buffer,
                    offset: 0,
                    stage: 1024,
                    len: 1
                }
            )
            .into_reply()
            .unwrap(),
            AccelResponse::Refused(AccelRefusal::Busy),
        );

        // Pending until the harness completes it (poll_event never
        // drives an executor).
        assert_eq!(
            step(&mut svc, AccelRequest::PollEvent { event })
                .into_reply()
                .unwrap(),
            AccelResponse::Event {
                event,
                state: AccelEventState::Pending
            },
        );
        {
            let engine = svc.live_engine().unwrap();
            engine
                .backend()
                .complete(engine.event_handle(event).unwrap())
                .unwrap();
        }
        assert_eq!(
            step(&mut svc, AccelRequest::PollEvent { event })
                .into_reply()
                .unwrap(),
            AccelResponse::Event {
                event,
                state: AccelEventState::Complete
            },
        );

        assert_eq!(
            svc.answer(
                0,
                AccelRequest::ConfigureProfiling {
                    ctx,
                    enabled: false
                }
            ),
            AccelResponse::Refused(AccelRefusal::Busy)
        );

        // Terminality makes execution complete, but DeviceState releases its
        // binding references only when the event is destroyed.
        assert_eq!(
            step(
                &mut svc,
                AccelRequest::ReadBuffer {
                    buffer,
                    offset: 0,
                    stage: 1024,
                    len: 8
                }
            )
            .into_reply()
            .unwrap(),
            AccelResponse::Refused(AccelRefusal::Busy),
        );
        assert_eq!(
            step(&mut svc, AccelRequest::DestroyEvent { event })
                .into_reply()
                .unwrap(),
            AccelResponse::Released { subject: event },
        );

        // Read the buffer back through staging: the program xored it.
        assert_eq!(
            step(
                &mut svc,
                AccelRequest::ReadBuffer {
                    buffer,
                    offset: 0,
                    stage: 1024,
                    len: 8
                }
            )
            .into_reply()
            .unwrap(),
            AccelResponse::Read { buffer },
        );
        let expected: alloc::vec::Vec<u8> = input.iter().map(|b| b ^ 0x5a).collect();
        assert_eq!(read_staging(&svc, 1024, 8), &expected[..]);

        // The release ladder, child-first.
        for (req, subject) in [
            (AccelRequest::DestroyQueue { queue }, queue),
            (AccelRequest::UnloadProgram { program }, program),
            (AccelRequest::FreeBuffer { buffer }, buffer),
            (AccelRequest::DestroyContext { ctx }, ctx),
        ] {
            assert_eq!(
                step(&mut svc, req).into_reply().unwrap(),
                AccelResponse::Released { subject },
            );
        }

        // Every id is now stale — the generational discipline.
        assert_eq!(
            step(&mut svc, AccelRequest::PollEvent { event })
                .into_reply()
                .unwrap(),
            AccelResponse::Refused(AccelRefusal::BadObject),
        );
    }

    /// A v0.3.2 multi-binding submission retains both distinct buffers in the
    /// kore object graph, dispatches by slot rather than slice order, and makes
    /// the completed copy visible only after the event releases its children.
    #[test]
    #[allow(deprecated)]
    fn multi_binding_copy_retains_source_and_destination_until_event_release() {
        use crate::wire::{decode_req, encode_req};

        let mut svc = hosted();
        stage(&mut svc, 4096);
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let source = expect_id(
            svc.answer(
                0,
                AccelRequest::CreateBuffer {
                    ctx,
                    bytes: 64,
                    align: 16,
                    domain: 3,
                    usage: (BufferUsage::TRANSFER_SOURCE
                        | BufferUsage::TRANSFER_DESTINATION
                        | BufferUsage::PROGRAM_INPUT)
                        .bits(),
                },
            ),
        );
        let destination = expect_id(
            svc.answer(
                0,
                AccelRequest::CreateBuffer {
                    ctx,
                    bytes: 64,
                    align: 16,
                    domain: 3,
                    usage: (BufferUsage::TRANSFER_SOURCE
                        | BufferUsage::TRANSFER_DESTINATION
                        | BufferUsage::PROGRAM_OUTPUT)
                        .bits(),
                },
            ),
        );

        let input = [0x00, 0x11, 0x7f, 0x80, 0xa5, 0xff, 0x3c, 0xc3];
        let destination_before = [0x55; 8];
        for (buffer, stage_offset, bytes) in [
            (source, 0_u64, input.as_slice()),
            (destination, 32_u64, destination_before.as_slice()),
        ] {
            write_staging(&mut svc, stage_offset as usize, bytes);
            assert_eq!(
                svc.answer(
                    0,
                    AccelRequest::WriteBuffer {
                        buffer,
                        offset: 0,
                        stage: stage_offset,
                        len: bytes.len() as u64,
                    },
                ),
                AccelResponse::Wrote { buffer },
            );
        }

        let artifact = reference::ReferenceArtifact::copy(2, 7).unwrap();
        write_staging(&mut svc, 256, artifact.as_bytes());
        let program = expect_id(svc.answer(
            0,
            AccelRequest::LoadProgram {
                ctx,
                stage: 256,
                len: artifact.as_bytes().len() as u64,
                resident_bytes: reference::RESIDENT_BYTES,
                format: reference::ARTIFACT_FORMAT.get(),
                target: reference::TARGET_IDENTITY.0,
            },
        ));
        let queue = expect_id(svc.answer(0, AccelRequest::CreateQueue { ctx, flags: 0 }));

        // Deliberately present slot 7 before slot 2. Binding order is not
        // semantic, and each descriptor must remain paired with its own buffer.
        let bindings = AccelBindings::new(&[
            AccelBinding {
                buffer: destination,
                offset: 0,
                len: input.len() as u64,
                slot: 7,
                access: AccessMode::Write,
            },
            AccelBinding {
                buffer: source,
                offset: 0,
                len: input.len() as u64,
                slot: 2,
                access: AccessMode::Read,
            },
        ])
        .unwrap();
        let request = AccelRequest::SubmitLaunch {
            queue,
            program,
            timeout_ns: 0,
            bindings,
            launch: LaunchShape::ArtifactDefined,
        };
        let decoded = decode_req(&encode_req(&request)).expect("two bindings round-trip the wire");
        assert_eq!(
            decoded,
            AccelRequest::Submit {
                queue,
                program,
                timeout_ns: 0,
                bindings,
                dispatch: Dispatch::scalar(),
            },
            "artifact-defined launch canonicalizes to the deployed OP13 sentinel",
        );
        let event = expect_id(svc.answer(0, decoded));

        // DeviceState retains every binding, not just the writable/primary one.
        for buffer in [source, destination] {
            assert_eq!(
                svc.answer(
                    0,
                    AccelRequest::ReadBuffer {
                        buffer,
                        offset: 0,
                        stage: 2048,
                        len: 1,
                    },
                ),
                AccelResponse::Refused(AccelRefusal::Busy),
            );
        }

        {
            let engine = svc.live_engine().unwrap();
            engine
                .backend()
                .complete(engine.event_handle(event).unwrap())
                .unwrap();
        }
        assert_eq!(
            svc.answer(0, AccelRequest::PollEvent { event }),
            AccelResponse::Event {
                event,
                state: AccelEventState::Complete,
            },
        );

        // Terminal work still owns both buffers until DestroyEvent commits the
        // two-phase release in the device state.
        for buffer in [source, destination] {
            assert_eq!(
                svc.answer(
                    0,
                    AccelRequest::ReadBuffer {
                        buffer,
                        offset: 0,
                        stage: 2048,
                        len: 1,
                    },
                ),
                AccelResponse::Refused(AccelRefusal::Busy),
            );
        }
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyEvent { event }),
            AccelResponse::Released { subject: event },
        );

        for (buffer, stage_offset) in [(source, 512_u64), (destination, 1024_u64)] {
            assert_eq!(
                svc.answer(
                    0,
                    AccelRequest::ReadBuffer {
                        buffer,
                        offset: 0,
                        stage: stage_offset,
                        len: input.len() as u64,
                    },
                ),
                AccelResponse::Read { buffer },
            );
            assert_eq!(
                read_staging(&svc, stage_offset as usize, input.len()),
                input.as_slice(),
            );
        }

        for (request, subject) in [
            (AccelRequest::DestroyQueue { queue }, queue),
            (AccelRequest::UnloadProgram { program }, program),
            (AccelRequest::FreeBuffer { buffer: source }, source),
            (
                AccelRequest::FreeBuffer {
                    buffer: destination,
                },
                destination,
            ),
            (AccelRequest::DestroyContext { ctx }, ctx),
        ] {
            assert_eq!(svc.answer(0, request), AccelResponse::Released { subject },);
        }
    }

    fn accepted_loss_graph(
        loss_point: DeviceLossPoint,
    ) -> (
        AccelService<DeviceLostBackend>,
        DropAudit,
        AccelResponse,
        u64,
    ) {
        accepted_loss_graph_at(loss_point, 0)
    }

    fn accepted_loss_graph_at(
        loss_point: DeviceLossPoint,
        via: u16,
    ) -> (
        AccelService<DeviceLostBackend>,
        DropAudit,
        AccelResponse,
        u64,
    ) {
        let audit = DropAudit::new();
        let mut svc = AccelService::host(
            DeviceLostBackend::new(audit.clone(), loss_point),
            None,
            7,
            1 << 32,
            1 << 32,
        );
        svc.attach_staging(via, Box::leak(vec![0u8; 4096].into_boxed_slice()))
            .unwrap();
        let ctx = expect_id(svc.answer(via, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(
            svc.answer(
                via,
                AccelRequest::CreateBuffer {
                    ctx,
                    bytes: 64,
                    align: 16,
                    domain: 3,
                    usage: (BufferUsage::TRANSFER_SOURCE
                        | BufferUsage::TRANSFER_DESTINATION
                        | BufferUsage::PROGRAM_INPUT
                        | BufferUsage::PROGRAM_OUTPUT
                        | BufferUsage::MUTABLE_STATE)
                        .bits(),
                },
            ),
        );
        let artifact = reference::ReferenceArtifact::xor(7, 0x5a);
        svc.sessions[pair_index(via)]
            .staging
            .as_deref_mut()
            .unwrap()[..artifact.as_bytes().len()]
            .copy_from_slice(artifact.as_bytes());
        let program = expect_id(svc.answer(
            via,
            AccelRequest::LoadProgram {
                ctx,
                stage: 0,
                len: artifact.as_bytes().len() as u64,
                resident_bytes: reference::RESIDENT_BYTES,
                format: reference::ARTIFACT_FORMAT.get(),
                target: reference::TARGET_IDENTITY.0,
            },
        ));
        let queue = expect_id(svc.answer(via, AccelRequest::CreateQueue { ctx, flags: 0 }));
        let bindings = AccelBindings::new(&[AccelBinding {
            buffer,
            offset: 0,
            len: 8,
            slot: 7,
            access: AccessMode::ReadWrite,
        }])
        .unwrap();
        let response = svc.answer(
            via,
            AccelRequest::SubmitLaunch {
                queue,
                program,
                timeout_ns: 0,
                bindings,
                launch: LaunchShape::ArtifactDefined,
            },
        );
        (svc, audit, response, buffer)
    }

    /// A provider that keeps one live event (Xe) serves far past the 8192
    /// generations one id slot lasts. Sized to the provider's concurrency,
    /// the table refused the 8193rd submission with `ResourceLimit`, which is
    /// how Gigatron's serial lane ended a chat mid-decode.
    #[test]
    fn a_single_event_provider_outlives_one_slots_id_generations() {
        let audit = DropAudit::new();
        let mut backend = DeviceLostBackend::new(audit, DeviceLossPoint::TerminalProvider);
        backend.one_event_slot = true;
        let mut svc = AccelService::host(backend, None, 7, 1 << 32, 1 << 32);
        assert!(svc.hosted());
        svc.attach_staging(0, Box::leak(vec![0u8; 4096].into_boxed_slice()))
            .unwrap();
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(svc.answer(
            0,
            AccelRequest::CreateBuffer {
                ctx,
                bytes: 64,
                align: 16,
                domain: 3,
                usage: (BufferUsage::PROGRAM_INPUT
                    | BufferUsage::PROGRAM_OUTPUT
                    | BufferUsage::MUTABLE_STATE)
                    .bits(),
            },
        ));
        let artifact = reference::ReferenceArtifact::xor(7, 0x5a);
        svc.sessions[0].staging.as_deref_mut().unwrap()[..artifact.as_bytes().len()]
            .copy_from_slice(artifact.as_bytes());
        let program = expect_id(svc.answer(
            0,
            AccelRequest::LoadProgram {
                ctx,
                stage: 0,
                len: artifact.as_bytes().len() as u64,
                resident_bytes: reference::RESIDENT_BYTES,
                format: reference::ARTIFACT_FORMAT.get(),
                target: reference::TARGET_IDENTITY.0,
            },
        ));
        let queue = expect_id(svc.answer(0, AccelRequest::CreateQueue { ctx, flags: 0 }));
        let bindings = AccelBindings::new(&[AccelBinding {
            buffer,
            offset: 0,
            len: 8,
            slot: 7,
            access: AccessMode::ReadWrite,
        }])
        .unwrap();
        // Two slots' worth of generations and then some.
        for _ in 0..(2 << 13) + 3 {
            let event = expect_id(svc.answer(
                0,
                AccelRequest::SubmitLaunch {
                    queue,
                    program,
                    timeout_ns: 0,
                    bindings,
                    launch: LaunchShape::ArtifactDefined,
                },
            ));
            assert!(matches!(
                svc.answer(0, AccelRequest::PollEvent { event }),
                AccelResponse::Event {
                    state: AccelEventState::Failed(_),
                    ..
                }
            ));
            assert_eq!(
                svc.answer(0, AccelRequest::DestroyEvent { event }),
                AccelResponse::Released { subject: event }
            );
        }
        // Clients are still told the provider's own concurrency.
        assert_eq!(
            svc.live_engine().unwrap().info().limits.max_events_per_context,
            1
        );
    }

    /// A provider-wide fatal edge is observed before the requested resource
    /// boundary. The complete epoch stays affine until the row quiesces it;
    /// Query remains the terminal identity face both before and after discard.
    #[test]
    fn provider_health_loss_seals_before_admission_and_keeps_query_available() {
        let audit = DropAudit::new();
        let mut svc = AccelService::host(
            DeviceLostBackend::new(audit.clone(), DeviceLossPoint::HealthLost),
            None,
            7,
            1 << 32,
            1 << 32,
        );
        assert!(svc.hosted(), "the health edge has not been polled yet");

        assert!(matches!(
            step(&mut svc, AccelRequest::Query).into_reply(),
            Some(AccelResponse::Info(_)),
        ));
        assert!(svc.restart_required(), "Query's preflight sealed the epoch");
        assert!(svc.backend().is_none(), "the lost provider is unreachable");
        assert!(
            audit.resources.borrow().is_empty(),
            "health loss admitted no provider resource",
        );
        assert_eq!(
            step(&mut svc, AccelRequest::CreateContext { flags: 0 }).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::DeviceLost)),
        );

        svc.take_lost_epoch()
            .expect("the row owns the complete lost epoch")
            .quiesce_and_discard(|_| audit.quiesced.set(true));
        assert!(audit.backend_dropped.get());
        assert!(matches!(
            step(&mut svc, AccelRequest::Query).into_reply(),
            Some(AccelResponse::Info(_)),
        ));
        assert_eq!(
            step(&mut svc, AccelRequest::CreateContext { flags: 0 }).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::DeviceLost)),
            "the terminal identity face never reopens admission",
        );
    }

    #[test]
    fn interrupted_retained_quiescence_never_drops_graph_or_resource_owner() {
        struct Owner<'a>(&'a core::cell::Cell<u32>);
        impl Drop for Owner<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        for panic_in_stop in [false, true] {
            let audit = DropAudit::new();
            let dropped = core::cell::Cell::new(0);
            let mut svc = AccelService::host(
                DeviceLostBackend::new(audit.clone(), DeviceLossPoint::HealthLost),
                None,
                7,
                1 << 32,
                1 << 32,
            );
            let _ = step(&mut svc, AccelRequest::Query);
            let lost = svc.take_lost_epoch().unwrap();
            let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let (retained, ()) = lost.quiesce_and_retain(Owner(&dropped), |_, _| {
                    if panic_in_stop {
                        panic!("interrupted containment");
                    }
                    audit.quiesced.set(true);
                });
                drop(retained); // cancellation/owner withdrawal without a retirement receipt
            }));
            assert_eq!(stopped.is_err(), panic_in_stop);
            assert!(!audit.backend_dropped.get());
            assert!(audit.resources.borrow().is_empty());
            assert_eq!(
                dropped.get(),
                0,
                "unproved retirement retains the resource owner"
            );
            assert!(matches!(
                step(&mut svc, AccelRequest::Query).into_reply(),
                Some(AccelResponse::Info(_))
            ));
        }
    }

    /// A retryable health-probe error does not poison the provider, but it also
    /// cannot be discarded as proof of safety. Refuse this admission and leave
    /// the live epoch available for a later successful health observation.
    #[test]
    fn transient_health_error_refuses_without_admitting_or_poisoning() {
        let audit = DropAudit::new();
        let mut svc = AccelService::host(
            DeviceLostBackend::new(audit.clone(), DeviceLossPoint::HealthBusy),
            None,
            7,
            1 << 32,
            1 << 32,
        );

        assert_eq!(
            step(&mut svc, AccelRequest::CreateContext { flags: 0 }).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::Busy)),
        );
        assert!(
            svc.hosted(),
            "a retryable health error did not lose the epoch"
        );
        assert!(!svc.restart_required());
        assert!(
            audit.resources.borrow().is_empty(),
            "the refused request never crossed create_context",
        );
        assert!(
            matches!(
                step(&mut svc, AccelRequest::Query).into_reply(),
                Some(AccelResponse::Info(_)),
            ),
            "identity observation does not require device admission",
        );
    }

    /// DeviceLost after provider acceptance is a whole-epoch discard, not an
    /// event that can remain forever in the ordinary live tables. The service
    /// seals the namespace before returning, and its affine hardware handoff
    /// drops the event plus every dependency only after the owner quiesces.
    #[test]
    fn device_lost_submission_quiesces_and_discards_the_complete_epoch() {
        let (mut svc, audit, response, buffer) = accepted_loss_graph(DeviceLossPoint::Submit);
        let event = match response {
            AccelResponse::SubmitIndeterminate {
                event,
                refusal: AccelRefusal::DeviceLost,
            } => event,
            other => panic!("lost submit must expose its accepted event boundary: {other:?}"),
        };

        assert!(
            !svc.hosted(),
            "the poisoned provider is no longer reachable"
        );
        assert!(svc.restart_required());
        assert!(svc.backend().is_none());
        assert_eq!(
            svc.answer(0, AccelRequest::PollEvent { event }),
            AccelResponse::Refused(AccelRefusal::DeviceLost),
            "the accepted event id cannot re-enter the poisoned namespace",
        );
        assert_eq!(
            svc.answer(
                0,
                AccelRequest::ReadBuffer {
                    buffer,
                    offset: 0,
                    stage: 0,
                    len: 1,
                },
            ),
            AccelResponse::Refused(AccelRefusal::DeviceLost),
            "an event dependency cannot be reached through a stale handle",
        );
        assert!(audit.resources.borrow().is_empty());

        let lost = svc
            .take_lost_epoch()
            .expect("the hardware owner receives the lost epoch exactly once");
        assert!(svc.take_lost_epoch().is_none());
        let (retained, marker) = lost.quiesce_and_retain(0xfeed_u64, |_, owner| {
            assert_eq!(*owner, 0xfeed);
            assert!(audit.resources.borrow().is_empty());
            audit.quiesced.set(true);
            0xacce1_u64
        });
        assert_eq!(marker, 0xacce1_u64);
        assert!(
            !audit.backend_dropped.get(),
            "class stop alone must retain the backend mappings"
        );
        assert!(
            audit.resources.borrow().is_empty(),
            "every event/dependency remains owned during the broker exchange"
        );
        assert!(matches!(
            svc.answer(0, AccelRequest::Query),
            AccelResponse::Info(_)
        ));
        assert_eq!(
            svc.answer(0, AccelRequest::CreateContext { flags: 0 }),
            AccelResponse::Refused(AccelRefusal::DeviceLost)
        );
        assert_eq!(
            retained.discard(),
            0xfeed,
            "explicit discard returns the resource owner after graph drop"
        );
        assert!(audit.backend_dropped.get());

        let dropped = audit.resources.borrow();
        assert_eq!(dropped.len(), 5, "event and all four dependencies died");
        for kind in [
            DroppedResource::Context,
            DroppedResource::Buffer,
            DroppedResource::Program,
            DroppedResource::Queue,
            DroppedResource::Event,
        ] {
            assert!(
                dropped.contains(&(kind, true)),
                "{kind:?} was discarded only after quiesce",
            );
        }
        drop(dropped);

        assert!(matches!(
            svc.answer(0, AccelRequest::Query),
            AccelResponse::Info(_)
        ));
        assert_eq!(
            svc.answer(0, AccelRequest::CreateContext { flags: 0 }),
            AccelResponse::Refused(AccelRefusal::DeviceLost),
            "terminal rows cannot mint ids from a poisoned epoch",
        );
        assert!(svc.restart_required(), "the restart demand is sticky");
    }

    /// A provider allocation is already owned when the host discovers its
    /// actual backing exceeds policy. If provider cleanup rejects, the hidden
    /// object cannot be published and no client can retry its release. The
    /// whole epoch is therefore quarantined for owner-ordered discard instead
    /// of leaking an unreachable context child while continuing to serve.
    #[test]
    fn rejected_cleanup_of_an_unpublished_buffer_quarantines_the_epoch() {
        let audit = DropAudit::new();
        let mut svc = AccelService::host(
            DeviceLostBackend::cleanup_rejecting(audit.clone()),
            None,
            7,
            64,
            1 << 20,
        );
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));

        assert_eq!(
            svc.answer(
                0,
                AccelRequest::CreateBuffer {
                    ctx,
                    bytes: 64,
                    align: 1,
                    domain: 3,
                    usage: BufferUsage::MUTABLE_STATE.bits(),
                },
            ),
            AccelResponse::Refused(AccelRefusal::DeviceLost),
        );
        assert!(svc.restart_required());
        assert!(svc.backend().is_none());
        assert!(
            audit.resources.borrow().is_empty(),
            "the hidden buffer stays owned until hardware quiesce",
        );

        svc.take_lost_epoch()
            .expect("the hidden allocation makes the whole graph discard-only")
            .quiesce_and_discard(|_| audit.quiesced.set(true));
        assert!(audit.backend_dropped.get());
        let dropped = audit.resources.borrow();
        assert_eq!(dropped.len(), 2, "context plus unpublished buffer");
        assert!(dropped.iter().all(|(_, quiesced)| *quiesced));
    }

    /// Returned allocation metadata is validated only after DeviceState owns
    /// the provider handle. An off-descriptor allocation must cross the
    /// backend's explicit release boundary before the client sees refusal;
    /// directly dropping the temporary would bypass provider bookkeeping.
    #[test]
    fn off_descriptor_allocation_is_released_through_the_provider_boundary() {
        let audit = DropAudit::new();
        let mut svc = AccelService::host(
            DeviceLostBackend::misdescribing(audit.clone()),
            None,
            7,
            1 << 20,
            1 << 20,
        );
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));

        assert_eq!(
            svc.answer(
                0,
                AccelRequest::CreateBuffer {
                    ctx,
                    bytes: 64,
                    align: 1,
                    domain: 3,
                    usage: BufferUsage::MUTABLE_STATE.bits(),
                },
            ),
            AccelResponse::Refused(AccelRefusal::Incompatible),
        );
        assert_eq!(audit.buffer_release_attempts.get(), 1);
        assert!(svc.hosted(), "successful cleanup keeps the epoch usable");
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyContext { ctx }),
            AccelResponse::Released { subject: ctx },
            "cleanup committed the hidden child before refusal",
        );
    }

    /// A completion poll can discover the same epoch edge after an ordinary
    /// successful admission. The pump must throw away every partial wake count,
    /// seal before touching the transport, and hand the complete graph to the
    /// owner for ordered quiesce/discard.
    #[test]
    fn completion_device_loss_seals_before_notice_and_releases_the_accepted_graph() {
        let (mut svc, audit, response, buffer) = accepted_loss_graph(DeviceLossPoint::Poll);
        let event = expect_id(response);
        assert!(svc.hosted(), "the accepted event starts in the live epoch");

        let transport = kore_rt::transport::TrapTransport::new();
        let sent = kore_rt::try_block_on(svc.pump_completions(&transport), 1)
            .expect("the loss sweep completes without transport I/O");
        assert_eq!(sent, 0, "a poisoned sweep publishes no partial wake batch");
        assert!(svc.restart_required());
        assert!(svc.backend().is_none());
        assert_eq!(
            svc.answer(0, AccelRequest::PollEvent { event }),
            AccelResponse::Refused(AccelRefusal::DeviceLost),
        );
        assert_eq!(
            svc.answer(
                0,
                AccelRequest::ReadBuffer {
                    buffer,
                    offset: 0,
                    stage: 0,
                    len: 1,
                },
            ),
            AccelResponse::Refused(AccelRefusal::DeviceLost),
        );
        assert!(audit.resources.borrow().is_empty());

        svc.take_lost_epoch()
            .expect("the completion edge owns the whole lost epoch")
            .quiesce_and_discard(|_| audit.quiesced.set(true));
        assert!(audit.backend_dropped.get());
        let dropped = audit.resources.borrow();
        assert_eq!(dropped.len(), 5, "event and all four dependencies died");
        assert!(
            dropped.iter().all(|(_, quiesced)| *quiesced),
            "no provider handle was released ahead of hardware quiesce",
        );
    }

    /// A provider may report the same epoch failure inside a terminal event
    /// state rather than as the `poll_event` call's outer error. That encoding
    /// A loss with a retained device witness reaches the client as the
    /// witnessed refusal at every surface: the first observation (the
    /// pre-request health probe, through `step`'s choke point), the sealed
    /// epoch's later refusals, and after the affine handoff moved the engine
    /// out (`terminal_evidence`). A witness-less loss stays bare.
    /// Telemetry is a fresh snapshot at offset zero, paged from that
    /// snapshot after it, and answers even when the device is lost.
    #[test]
    fn telemetry_text_snapshots_at_offset_zero_and_pages_after_it() {
        let mut svc = hosted();
        // Offset zero takes the snapshot; later offsets page through it
        // until an empty page ends it.
        let mut text = Vec::new();
        loop {
            let reply = step(
                &mut svc,
                AccelRequest::ReadTelemetry {
                    offset: text.len() as u64,
                },
            )
            .into_reply()
            .unwrap();
            let AccelResponse::Failure { offset, len, bytes } = reply else {
                panic!("{reply:?}")
            };
            assert_eq!(offset, text.len() as u64);
            if len == 0 {
                break;
            }
            text.extend_from_slice(&bytes[..usize::from(len)]);
        }
        // The service-time books count requests after they are answered, so
        // the snapshot the first read took shows none yet.
        let idle = "0 calls, 0 us avg, 0 us longest";
        assert_eq!(
            core::str::from_utf8(&text).unwrap(),
            alloc::format!(
                "host: dropped notices 0\nhost: service time; submit {idle}; poll {idle}; \
                 destroy {idle}; other {idle}\n"
            )
        );
        assert!(matches!(
            step(&mut svc, AccelRequest::ReadTelemetry { offset: u64::MAX }).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::OutOfBounds))
        ));
        let (mut lost, _audit, _, _) = accepted_loss_graph(DeviceLossPoint::HealthLost);
        let AccelResponse::Failure { len, bytes, .. } =
            step(&mut lost, AccelRequest::ReadTelemetry { offset: 0 })
                .into_reply()
                .unwrap()
        else {
            panic!("a lost epoch still answers telemetry")
        };
        assert!(bytes[..usize::from(len)].starts_with(b"no live epoch"));
    }

    #[test]
    fn failure_text_survives_containment_without_staging_or_backend_access() {
        let (mut svc, audit, _, _) = accepted_loss_graph(DeviceLossPoint::HealthLost);
        let request = AccelRequest::ReadFailure { offset: 0 };
        let first = step(&mut svc, request).into_reply().unwrap();
        let AccelResponse::Failure { offset, len, bytes } = first else { panic!("{first:?}") };
        assert_eq!(offset, 0);
        assert_eq!(&bytes[..usize::from(len)], b"retained test failure");
        assert!(bytes[usize::from(len)..].iter().all(|b| *b == 0));
        svc.take_lost_epoch().unwrap().quiesce_and_discard(|_| audit.quiesced.set(true));
        assert!(audit.backend_dropped.get());
        assert_eq!(step(&mut svc, request).into_reply(), Some(first));
        assert!(matches!(step(&mut svc, AccelRequest::ReadFailure { offset: u64::MAX }).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::OutOfBounds))));
        assert!(matches!(step(&mut svc, AccelRequest::ReadFailure { offset: u64::from(len) }).into_reply(),
            Some(AccelResponse::Failure { len: 0, .. })));
        assert!(matches!(step(&mut svc, AccelRequest::CreateContext { flags: 0 }).into_reply(),
            Some(AccelResponse::Refused(AccelRefusal::DeviceLost))));
    }

    #[test]
    fn a_retained_loss_witness_rides_every_device_lost_reply() {
        const WITNESS: u64 = 0x80da_1000_0123_450b;
        let (mut svc, _audit, _response, _buffer) =
            accepted_loss_graph(DeviceLossPoint::HealthLost);
        svc.backend().expect("live").witness.set(Some(LossEvidence {
            domain: LOSS_EVIDENCE_DOMAIN,
            code: WITNESS as i64,
        }));
        let expected = AccelRefusal::External {
            domain: LOSS_EVIDENCE_DOMAIN,
            code: WITNESS as i64,
        };
        // The very first observation — the health probe before a request.
        assert_eq!(
            step(&mut svc, AccelRequest::CreateContext { flags: 0 }).into_reply(),
            Some(AccelResponse::Refused(expected)),
        );
        assert!(svc.restart_required());
        assert_eq!(svc.witnessed_refusal_for_test(), expected);
        // Past the handoff, the engine is gone but the witness is not.
        let lost = svc.take_lost_epoch().expect("sealed");
        assert_eq!(svc.witnessed_refusal_for_test(), expected);
        assert_eq!(
            step(&mut svc, AccelRequest::CreateContext { flags: 0 }).into_reply(),
            Some(AccelResponse::Refused(expected)),
        );
        lost.quiesce_and_discard(|_| ());

        // Native imports bypass `step`; their pre-request health refusal must
        // carry the same device witness before the row consumes the response.
        let (mut native, _audit, _response, _buffer) =
            accepted_loss_graph(DeviceLossPoint::HealthLost);
        native.backend().expect("live").witness.set(Some(LossEvidence {
            domain: LOSS_EVIDENCE_DOMAIN,
            code: WITNESS as i64,
        }));
        assert_eq!(
            native.import_buffer_using(
                0,
                0,
                4096,
                4096,
                BufferUsage::PROGRAM_INPUT.bits(),
                |_, _, _| panic!("lost device must not consume a native import"),
            ),
            AccelResponse::Refused(expected),
        );
        assert!(native.restart_required());
        native
            .take_lost_epoch()
            .expect("native loss retains the complete epoch")
            .quiesce_and_discard(|_| ());
    }

    #[test]
    fn zero_and_full_width_loss_codes_keep_the_provider_domain_after_handoff() {
        for code in [0, i64::MIN, -1] {
            let (mut svc, _, _, _) = accepted_loss_graph(DeviceLossPoint::HealthLost);
            let evidence = LossEvidence { domain: u32::from_le_bytes(*b"TEST"), code };
            svc.backend().unwrap().witness.set(Some(evidence));
            let expected = AccelRefusal::External { domain: evidence.domain, code };
            assert_eq!(step(&mut svc, AccelRequest::CreateContext { flags: 0 }).into_reply(),
                Some(AccelResponse::Refused(expected)));
            let lost = svc.take_lost_epoch().unwrap();
            assert_eq!(svc.witnessed_refusal_for_test(), expected);
            lost.quiesce_and_discard(|_| ());
        }
    }

    /// must take the identical quiesce/discard path; it is not an ordinary
    /// per-submission failure that leaves the device usable.
    #[test]
    fn terminal_event_device_loss_also_seals_and_discards_the_epoch() {
        let (mut svc, audit, response, _buffer) =
            accepted_loss_graph(DeviceLossPoint::TerminalEvent);
        let event = expect_id(response);

        assert_eq!(
            svc.answer(0, AccelRequest::PollEvent { event }),
            AccelResponse::Event {
                event,
                state: AccelEventState::Failed(AccelRefusal::DeviceLost),
            },
            "the discovering poll still reports the accepted event's terminal truth",
        );
        assert!(svc.restart_required());
        assert!(svc.backend().is_none());
        assert!(audit.resources.borrow().is_empty());

        svc.take_lost_epoch()
            .expect("terminal DeviceLost owns the whole poisoned graph")
            .quiesce_and_discard(|_| audit.quiesced.set(true));
        assert!(audit.backend_dropped.get());
        assert!(
            audit
                .resources
                .borrow()
                .iter()
                .all(|(_, quiesced)| *quiesced),
            "terminal-payload loss cannot drop a provider handle before quiesce",
        );
    }

    #[test]
    fn recyclable_arrivals_keep_generation_ownership_across_slot_reuse() {
        let mut svc = hosted();
        // Same slot, successive generations — the slot reuse this test is about.
        let old = kore_rt::svc::PeerId::from_arrival(arrival_of(1, 1))
            .unwrap()
            .arrival();
        let new = kore_rt::svc::PeerId::from_arrival(arrival_of(1, 2))
            .unwrap()
            .arrival();
        assert_eq!(pair_index(old), pair_index(new));
        let ctx = expect_id(svc.answer(old, AccelRequest::CreateContext { flags: 0 }));
        assert_eq!(
            svc.answer(new, AccelRequest::Query),
            AccelResponse::Refused(AccelRefusal::Denied)
        );
        assert!(!svc.close_peer(new));
        assert!(svc.close_peer(old));
        let replacement = expect_id(svc.answer(new, AccelRequest::CreateContext { flags: 0 }));
        assert_ne!(ctx, replacement);
        assert!(!svc.close_peer(old));
        assert_eq!(
            svc.answer(old, AccelRequest::Query),
            AccelResponse::Refused(AccelRefusal::Denied)
        );
        let stage = Box::leak(vec![0x5au8; 4096].into_boxed_slice());
        svc.attach_staging(new, stage).unwrap();
        assert_eq!(stage_slice(&svc.sessions, new, 0, 1), Ok(&[0x5a][..]));
        assert_eq!(
            stage_slice(&svc.sessions, old, 0, 1),
            Err(AccelRefusal::Denied)
        );
        assert!(matches!(
            stage_slice_mut(&mut svc.sessions, old, 0, 1),
            Err(AccelRefusal::Denied)
        ));
    }

    #[test]
    fn export_release_callback_preserves_ownership_and_retry_checks() {
        let mut svc = AccelService::host(MockAccelerator::default(), None, 7, 1 << 24, 1 << 24);
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(svc.import_buffer_using(
            0,
            ctx,
            4096,
            4096,
            BufferUsage::PROGRAM_INPUT.bits(),
            |backend, context, desc| backend.allocate_buffer(context, desc),
        ));
        let called = core::cell::Cell::new(false);
        assert!(matches!(
            svc.free_buffer_using(1, buffer, |_, _| {
                called.set(true);
                panic!("foreign owner")
            }),
            AccelResponse::Refused(_)
        ));
        assert!(!called.get());
        assert_eq!(
            svc.free_buffer_using(0, buffer, |_, resource| {
                Err(ReleaseFailure::Rejected {
                    error: BackendError::Busy,
                    resource,
                })
            }),
            AccelResponse::Refused(AccelRefusal::Busy)
        );
        assert_eq!(
            svc.free_buffer_using(0, buffer, |backend, resource| backend.free_buffer(resource)),
            AccelResponse::Released { subject: buffer }
        );
        assert!(matches!(
            svc.free_buffer_using(0, buffer, |_, _| panic!("already removed")),
            AccelResponse::Refused(_)
        ));
    }

    #[test]
    fn import_creation_callback_cannot_bypass_context_ownership() {
        let mut svc = AccelService::host(MockAccelerator::default(), None, 7, 1 << 24, 1 << 24);
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let called = core::cell::Cell::new(false);
        let response = svc.import_buffer_using(
            1,
            ctx,
            4096,
            4096,
            BufferUsage::PROGRAM_INPUT.bits(),
            |backend, context, desc| {
                called.set(true);
                backend.allocate_buffer(context, desc)
            },
        );
        assert!(matches!(response, AccelResponse::Refused(_)));
        assert!(
            !called.get(),
            "foreign context must refuse before consuming import backing"
        );
        let buffer = expect_id(svc.import_buffer_using(
            0,
            ctx,
            4096,
            4096,
            BufferUsage::PROGRAM_INPUT.bits(),
            |backend, context, desc| {
                called.set(true);
                backend.allocate_buffer(context, desc)
            },
        ));
        assert!(called.get());
        assert!(
            matches!(svc.answer(0, AccelRequest::FreeBuffer { buffer }), AccelResponse::Released { subject } if subject == buffer)
        );
    }

    #[test]
    fn failed_outer_import_can_close_and_relaunch_into_the_same_live_epoch() {
        let mut svc = hosted();
        let old = arrival_of(1, 1);
        let new = arrival_of(1, 2);
        let ctx = expect_id(svc.answer(old, AccelRequest::CreateContext { flags: 0 }));
        let domain = u32::from_le_bytes(*b"DMAL");
        assert_eq!(
            svc.import_buffer_using(
                old,
                ctx,
                4096,
                4096,
                BufferUsage::PROGRAM_INPUT.bits(),
                |_, _, _| Err(BackendError::External { domain, code: 12 }),
            ),
            AccelResponse::Refused(AccelRefusal::External { domain, code: 12 }),
        );
        assert!(!svc.restart_required());
        assert!(svc.close_peer(old));
        let fresh = expect_id(svc.answer(new, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(svc.import_buffer_using(
            new,
            fresh,
            4096,
            4096,
            BufferUsage::PROGRAM_INPUT.bits(),
            |backend, context, desc| backend.allocate_buffer(context, desc),
        ));
        assert!(matches!(
            svc.answer(new, AccelRequest::FreeBuffer { buffer }),
            AccelResponse::Released { subject } if subject == buffer
        ));
        assert!(svc.close_peer(new));
    }

    #[test]
    fn recyclable_completion_counts_use_the_slot_without_erasing_ownership() {
        const VIA: u16 = 5;
        let (mut svc, _, submitted, _) =
            accepted_loss_graph_at(DeviceLossPoint::TerminalProvider, VIA);
        let event = expect_id(submitted);
        let HostEpoch::Live(engine) = &mut svc.epoch else {
            panic!("live")
        };
        // Both the vector's width and which pair the fixture lands on follow
        // `MAX_SERVE_PAIRS`; a literal array states neither.
        let mut due = [0; PAIRS];
        due[pair_index(VIA)] = 1;
        assert_eq!(engine.due_by_pair(), due);
        assert_eq!(engine.due_by_pair(), [0; PAIRS]);
        assert_eq!(
            svc.answer(1, AccelRequest::PollEvent { event }),
            AccelResponse::Refused(AccelRefusal::Denied)
        );
        assert!(matches!(
            svc.answer(5, AccelRequest::PollEvent { event }),
            AccelResponse::Event {
                state: AccelEventState::Failed(_),
                ..
            }
        ));
    }

    #[test]
    fn peer_close_retires_its_pending_graph_and_preserves_another_owner() {
        let (mut svc, audit, submitted, _) = accepted_loss_graph(DeviceLossPoint::Healthy);
        let event = expect_id(submitted);
        // The fixture stage is ordinary leaked host memory, not a kernel
        // mapping. Remove it to isolate object-graph teardown from the
        // separately tested owned/legacy staging retirement contracts.
        svc.sessions[0] = PairSession::EMPTY;
        let peer_ctx = expect_id(svc.answer(1, AccelRequest::CreateContext { flags: 0 }));
        assert_eq!(
            svc.answer(0, AccelRequest::PollEvent { event }),
            AccelResponse::Event {
                event,
                state: AccelEventState::Pending
            }
        );
        assert!(svc.close_peer(0));
        assert!(svc.hosted());
        assert_eq!(
            audit
                .resources
                .borrow()
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<Vec<_>>(),
            [
                DroppedResource::Event,
                DroppedResource::Queue,
                DroppedResource::Program,
                DroppedResource::Buffer,
                DroppedResource::Context
            ]
        );
        assert_eq!(
            svc.answer(0, AccelRequest::PollEvent { event }),
            AccelResponse::Refused(AccelRefusal::BadObject)
        );
        let next_ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        assert_ne!(next_ctx, peer_ctx);
        assert_eq!(
            svc.answer(1, AccelRequest::DestroyContext { ctx: peer_ctx }),
            AccelResponse::Released { subject: peer_ctx }
        );
        assert!(svc.close_peer(0));
        assert!(svc.close_peer(0));
    }

    #[test]
    fn peer_close_retries_a_refused_release_without_dropping_its_owner() {
        let (mut svc, audit, _, _) = accepted_loss_graph(DeviceLossPoint::Healthy);
        svc.sessions[0] = PairSession::EMPTY; // host-only fixture staging
        svc.backend_mut().unwrap().reject_buffer_cleanup = true;
        assert!(!svc.close_peer(0));
        assert!(svc.hosted());
        assert_eq!(
            audit
                .resources
                .borrow()
                .iter()
                .map(|(kind, _)| *kind)
                .collect::<Vec<_>>(),
            [
                DroppedResource::Event,
                DroppedResource::Queue,
                DroppedResource::Program
            ]
        );
        assert!(!svc.close_peer(0));
        assert_eq!(audit.resources.borrow().len(), 3);
        svc.backend_mut().unwrap().reject_buffer_cleanup = false;
        assert!(svc.close_peer(0));
        assert_eq!(audit.resources.borrow().len(), 5);
    }

    #[test]
    fn peer_close_native_release_preserves_rejected_import_until_retry() {
        let (mut svc, audit, _, _) = accepted_loss_graph(DeviceLossPoint::Healthy);
        svc.sessions[0] = PairSession::EMPTY;
        // Model an imported buffer which the ordinary allocator cannot free.
        svc.backend_mut().unwrap().reject_buffer_cleanup = true;
        assert!(!svc.close_peer(0));
        assert_eq!(audit.resources.borrow().len(), 3);
        let other = expect_id(svc.answer(1, AccelRequest::CreateContext { flags: 0 }));
        let mut calls = 0;
        assert!(!svc.close_peer_using(0, |_, resource| {
            calls += 1;
            Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource,
            })
        }));
        assert_eq!(calls, 1);
        assert_eq!(audit.resources.borrow().len(), 3);
        assert!(svc.close_peer_using(0, |_, resource| {
            calls += 1;
            drop(resource);
            Ok(())
        }));
        assert_eq!(calls, 2);
        assert_eq!(audit.resources.borrow().len(), 5);
        assert!(svc.close_peer_using(0, |_, _| panic!("retired buffer revisited")));
        assert_eq!(
            svc.answer(1, AccelRequest::DestroyContext { ctx: other }),
            AccelResponse::Released { subject: other }
        );
    }

    /// A dead client of a sealed epoch closes its own session custody, and
    /// only that: its objects stay whole in the lost graph for the row's
    /// containment. Refusing left the cell closing forever, so every launch
    /// after a loss spent a pair until no face could be admitted.
    #[test]
    fn peer_close_in_a_sealed_epoch_closes_the_session_and_keeps_the_graph() {
        let (mut svc, audit, _, _) = accepted_loss_graph(DeviceLossPoint::Poll);
        // This fixture's host-only stage has no capability to retire.
        svc.sessions[0] = PairSession::EMPTY;
        assert!(!svc.close_peer(0), "the live retire attempt seals first");
        assert!(svc.restart_required());
        assert!(svc.close_peer(0), "a sealed epoch lets the session close");
        assert!(
            audit.resources.borrow().is_empty(),
            "closing the session released nothing of the lost graph"
        );
        let epoch = svc.take_lost_epoch().unwrap();
        assert!(svc.close_peer(0), "terminal: the graph is the row's");
        assert!(audit.resources.borrow().is_empty());
        epoch.quiesce_and_discard(|_| audit.quiesced.set(true));
        assert!(svc.close_peer(0));
        assert!(svc.restart_required());
        let mut legacy = hosted();
        legacy.sessions[0].attaches = 1;
        assert!(
            !legacy.close_peer(0),
            "legacy task-lifetime mappings cannot acknowledge reuse"
        );
        assert_eq!(legacy.sessions[0].attaches, 1);
    }

    /// Lose an epoch holding one context, and contain it with or without the
    /// row's witness.
    fn lose_and_contain(
        svc: &mut AccelService<MockAccelerator>,
        proved: bool,
    ) -> (u64, Option<ContainedEpoch>) {
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        assert!(svc.retire_live_epoch());
        let lost = svc.take_lost_epoch().expect("sealed");
        match lost.discard_quiesced(|_| proved.then_some(())) {
            Ok((contained, ())) => (ctx, Some(contained)),
            Err(retained) => {
                // Unproved: the graph stays owned until task reap.
                core::mem::forget(retained);
                (ctx, None)
            }
        }
    }

    #[test]
    fn a_contained_epoch_refounds_under_a_new_namespace() {
        let mut svc = hosted();
        let (old, contained) = lose_and_contain(&mut svc, true);
        assert!(svc.restart_required());
        assert!(
            svc.refound(MockAccelerator::default(), contained.unwrap(), 1 << 32, 1 << 32)
                .is_ok()
        );
        assert!(svc.hosted());
        assert!(!svc.restart_required());
        // An id from the lost epoch never resolves in the fresh one.
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyContext { ctx: old }),
            AccelResponse::Refused(AccelRefusal::BadObject)
        );
        let fresh = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        assert_ne!(fresh, old);
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyContext { ctx: fresh }),
            AccelResponse::Released { subject: fresh }
        );
        // The fresh epoch can itself be lost and refounded again.
        let (_, again) = lose_and_contain(&mut svc, true);
        assert!(
            svc.refound(MockAccelerator::default(), again.unwrap(), 1 << 32, 1 << 32)
                .is_ok()
        );
    }

    #[test]
    fn refounding_needs_this_hosts_own_contained_epoch() {
        let foreign = || {
            let mut other =
                AccelService::host(MockAccelerator::default(), None, 9, 1 << 32, 1 << 32);
            lose_and_contain(&mut other, true).1.unwrap()
        };
        // A live host refounds nothing.
        let mut live = hosted();
        assert!(
            live.refound(MockAccelerator::default(), foreign(), 1 << 32, 1 << 32)
                .is_err()
        );
        assert!(live.hosted());
        // Another service's epoch cannot end this one; its own can.
        let mut svc = hosted();
        let (_, mine) = lose_and_contain(&mut svc, true);
        assert!(
            svc.refound(MockAccelerator::default(), foreign(), 1 << 32, 1 << 32)
                .is_err()
        );
        assert!(svc.restart_required());
        assert!(
            svc.refound(MockAccelerator::default(), mine.unwrap(), 1 << 32, 1 << 32)
                .is_ok()
        );
        // Without the row's witness there is no contained epoch at all.
        let mut unproved = hosted();
        let (_, none) = lose_and_contain(&mut unproved, false);
        assert!(none.is_none());
        assert!(unproved.restart_required());
    }

    #[test]
    fn provider_failure_crosses_service_and_wire_without_sealing_or_releasing() {
        let (mut svc, audit, submitted, _) = accepted_loss_graph(DeviceLossPoint::TerminalProvider);
        let event = expect_id(submitted);
        let expected = AccelResponse::Event {
            event,
            state: AccelEventState::Failed(AccelRefusal::External {
                domain: 0x5550_5649,
                code: 0xfedc_ba98,
            }),
        };
        for _ in 0..2 {
            let response = svc.answer(0, AccelRequest::PollEvent { event });
            assert_eq!(response, expected);
            let wire = crate::wire::encode_rep(&response, 0);
            assert_eq!(crate::wire::decode_rep(&wire), Some((expected, 0)));
        }
        assert!(svc.hosted());
        assert!(!svc.restart_required());
        assert!(
            audit.resources.borrow().is_empty(),
            "failure observation retains every handle"
        );
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyEvent { event }),
            AccelResponse::Released { subject: event }
        );
        assert_eq!(
            audit.resources.borrow().len(),
            1,
            "only explicit event release retires its handle"
        );
        assert!(svc.hosted());
    }

    /// The completion sweep must recognize terminal-payload DeviceLost before
    /// publishing any wake batch, just as it does an outer poll error.
    #[test]
    fn completion_sweep_discards_terminal_event_device_loss_without_notice() {
        let (mut svc, audit, response, _buffer) =
            accepted_loss_graph(DeviceLossPoint::TerminalEvent);
        let _event = expect_id(response);
        let transport = kore_rt::transport::TrapTransport::new();

        let sent = kore_rt::try_block_on(svc.pump_completions(&transport), 1)
            .expect("the loss sweep completes without transport I/O");
        assert_eq!(sent, 0);
        assert!(svc.restart_required());

        svc.take_lost_epoch()
            .expect("completion sweep sealed the poisoned epoch")
            .quiesce_and_discard(|_| audit.quiesced.set(true));
        assert!(audit.backend_dropped.get());
    }

    /// The synchronous pump a row calls from its never-waiting background
    /// hook runs the same sweep: a terminal-payload loss seals the epoch and
    /// publishes nothing, exactly as the async pump does.
    #[test]
    fn synchronous_completion_sweep_seals_device_loss_without_notice() {
        let (mut svc, audit, response, _buffer) =
            accepted_loss_graph(DeviceLossPoint::TerminalEvent);
        let _event = expect_id(response);
        let transport = kore_rt::transport::TrapTransport::new();

        assert_eq!(svc.pump_completions_now(&transport), 0);
        assert!(svc.restart_required());
        assert_eq!(
            svc.pump_completions_now(&transport),
            0,
            "a sealed epoch has nothing left to sweep"
        );

        svc.take_lost_epoch()
            .expect("the synchronous sweep sealed the poisoned epoch")
            .quiesce_and_discard(|_| audit.quiesced.set(true));
        assert!(audit.backend_dropped.get());
    }

    /// Destruction performs its own terminal poll. It must not call the
    /// provider's release path after that poll says the whole device was lost;
    /// the affine epoch owner quiesces and discards the graph instead.
    #[test]
    fn destroy_event_does_not_release_through_a_terminally_lost_provider() {
        let (mut svc, audit, response, _buffer) =
            accepted_loss_graph(DeviceLossPoint::TerminalEvent);
        let event = expect_id(response);

        assert_eq!(
            svc.answer(0, AccelRequest::DestroyEvent { event }),
            AccelResponse::Refused(AccelRefusal::DeviceLost),
        );
        assert!(svc.restart_required());
        assert!(audit.resources.borrow().is_empty());

        svc.take_lost_epoch()
            .expect("destroy's terminal poll sealed the complete epoch")
            .quiesce_and_discard(|_| audit.quiesced.set(true));
        assert!(audit.backend_dropped.get());
    }

    #[test]
    fn profiling_configuration_checks_context_ownership_and_provider_support() {
        let mut svc = hosted();
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let configure = AccelRequest::ConfigureProfiling { ctx, enabled: true };
        assert_eq!(
            svc.answer(1, configure),
            AccelResponse::Refused(AccelRefusal::Denied)
        );
        assert_eq!(
            svc.answer(0, configure),
            AccelResponse::Refused(AccelRefusal::Unsupported)
        );
        svc.answer(0, AccelRequest::DestroyContext { ctx });
        assert_eq!(
            svc.answer(0, configure),
            AccelResponse::Refused(AccelRefusal::BadObject)
        );
    }

    #[test]
    fn profiling_configuration_reaches_the_owned_provider_context() {
        let mut svc = AccelService::host(
            DeviceLostBackend::new(DropAudit::new(), DeviceLossPoint::Healthy),
            None,
            7,
            1 << 32,
            1 << 32,
        );
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        for enabled in [true, false] {
            assert_eq!(
                svc.answer(0, AccelRequest::ConfigureProfiling { ctx, enabled }),
                AccelResponse::ProfilingConfigured { ctx, enabled }
            );
        }
    }

    #[test]
    fn query_reports_the_backend_identity() {
        let mut svc = hosted();
        let AccelResponse::Info(info) = step(&mut svc, AccelRequest::Query).into_reply().unwrap()
        else {
            panic!("query answers info");
        };
        assert_eq!(info.class, 1, "the mock is an NPU");
        assert_eq!(info.uuid, *b"virtio-accelmock");
        assert!(info.limits.max_contexts >= 1);
    }

    /// Ownership is the pair stamp, never request data: an object
    /// created via pair 0 refuses every verb from pair 1.
    #[test]
    fn via_stamping_denies_the_other_pair() {
        let mut svc = hosted();
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        assert_eq!(
            svc.answer(1, AccelRequest::DestroyContext { ctx }),
            AccelResponse::Refused(AccelRefusal::Denied),
        );
        assert_eq!(
            svc.answer(1, AccelRequest::CreateQueue { ctx, flags: 0 }),
            AccelResponse::Refused(AccelRefusal::Denied),
        );
        // The owner still may.
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyContext { ctx }),
            AccelResponse::Released { subject: ctx },
        );
    }

    #[test]
    fn staging_is_mandatory_and_bounded() {
        let mut svc = hosted();
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(svc.answer(
            0,
            AccelRequest::CreateBuffer {
                ctx,
                bytes: 64,
                align: 16,
                domain: 1,
                usage: BufferUsage::TRANSFER_DESTINATION.bits(),
            },
        ));
        // No window attached: every byte-moving verb refuses.
        assert_eq!(
            svc.answer(
                0,
                AccelRequest::WriteBuffer {
                    buffer,
                    offset: 0,
                    stage: 0,
                    len: 8
                }
            ),
            AccelResponse::Refused(AccelRefusal::NoStaging),
        );
        stage(&mut svc, 4096);
        // A range escaping the window refuses out-of-bounds — offsets
        // are client data.
        assert_eq!(
            svc.answer(
                0,
                AccelRequest::WriteBuffer {
                    buffer,
                    offset: 0,
                    stage: 4090,
                    len: 8
                }
            ),
            AccelResponse::Refused(AccelRefusal::OutOfBounds),
        );
        assert_eq!(
            svc.answer(
                0,
                AccelRequest::WriteBuffer {
                    buffer,
                    offset: 0,
                    stage: u64::MAX,
                    len: 8
                }
            ),
            AccelResponse::Refused(AccelRefusal::OutOfBounds),
        );
    }

    #[test]
    fn attaches_are_counted_and_refused_past_the_bound() {
        let mut svc = hosted();
        for _ in 0..MAX_ATTACHES {
            let window = Box::leak(vec![0u8; 4096].into_boxed_slice());
            svc.attach_staging(0, window).expect("within the bound");
        }
        let window = Box::leak(vec![0u8; 4096].into_boxed_slice());
        assert_eq!(
            svc.attach_staging(0, window),
            Err(AccelRefusal::ResourceLimit),
            "the map cursor is finite; the books say who spent it",
        );
    }

    #[test]
    fn staging_refusals_precede_mapping_and_failed_attempts_remain_spent() {
        let mut svc = hosted();
        let original = Box::leak(vec![0x5au8; 4096].into_boxed_slice());
        svc.attach_staging(0, original).unwrap();
        assert_eq!(
            svc.attach_staging_with(PAIRS as u16, || panic!(
                "different generation mapped over static pair"
            )),
            Err(AccelRefusal::Denied),
        );
        for _ in 1..MAX_ATTACHES {
            assert_eq!(
                svc.attach_staging_with(0, || None),
                Err(AccelRefusal::NoStaging),
            );
        }
        assert_eq!(
            svc.attach_staging_with(0, || panic!("exhausted pair mapped")),
            Err(AccelRefusal::ResourceLimit),
        );
        assert_eq!(svc.sessions[0].staging.as_deref().unwrap()[0], 0x5a);
        assert_eq!(svc.sessions[0].attaches, MAX_ATTACHES);
        // Another client's budget remains independent of these failures.
        assert_eq!(svc.sessions[1].attaches, 0);
        let other = Box::leak(vec![0u8; 4096].into_boxed_slice());
        assert_eq!(svc.attach_staging(1, other), Ok(4096));
    }

    /// The Wire ABI §4.4 host gate: an access mode the buffer's usage
    /// does not permit refuses before the provider sees the submission.
    #[test]
    fn submit_rejects_access_beyond_declared_usage() {
        let mut svc = hosted();
        stage(&mut svc, 4096);
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(svc.answer(
            0,
            AccelRequest::CreateBuffer {
                ctx,
                bytes: 64,
                align: 16,
                domain: 3,
                usage: BufferUsage::PROGRAM_INPUT.bits(),
            },
        ));
        let artifact = reference::ReferenceArtifact::xor(7, 0x5a);
        let bytes = artifact.as_bytes();
        write_staging(&mut svc, 0, bytes);
        let program = expect_id(svc.answer(
            0,
            AccelRequest::LoadProgram {
                ctx,
                stage: 0,
                len: bytes.len() as u64,
                resident_bytes: reference::RESIDENT_BYTES,
                format: reference::ARTIFACT_FORMAT.get(),
                target: reference::TARGET_IDENTITY.0,
            },
        ));
        let queue = expect_id(svc.answer(0, AccelRequest::CreateQueue { ctx, flags: 0 }));
        let bindings = AccelBindings::new(&[AccelBinding {
            buffer,
            offset: 0,
            len: 8,
            slot: 7,
            access: AccessMode::Write, // input-only buffer: refused
        }])
        .unwrap();
        assert_eq!(
            svc.answer(
                0,
                AccelRequest::SubmitLaunch {
                    queue,
                    program,
                    timeout_ns: 0,
                    bindings,
                    launch: LaunchShape::ArtifactDefined,
                }
            ),
            AccelResponse::Refused(AccelRefusal::PermissionDenied),
        );
    }

    #[test]
    fn lifecycle_refusals_hold_the_object_graph() {
        let mut svc = hosted();
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let queue = expect_id(svc.answer(0, AccelRequest::CreateQueue { ctx, flags: 0 }));
        // A parent with live children is busy, never silently torn.
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyContext { ctx }),
            AccelResponse::Refused(AccelRefusal::Busy),
        );
        // A queue id is not a context id — kind tags refuse the cross.
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyContext { ctx: queue }),
            AccelResponse::Refused(AccelRefusal::BadObject),
        );
        // Zero is never an object.
        assert_eq!(
            svc.answer(0, AccelRequest::FreeBuffer { buffer: 0 }),
            AccelResponse::Refused(AccelRefusal::BadObject),
        );
        // Reserved flag sets are unsupported, not guessed at.
        assert_eq!(
            svc.answer(0, AccelRequest::CreateContext { flags: 1 }),
            AccelResponse::Refused(AccelRefusal::Unsupported),
        );
    }

    /// A pending event cannot be destroyed; a terminal one can, and its
    /// terminal edge is counted into the signal sweep exactly once.
    #[test]
    fn completion_sweep_counts_each_terminal_edge_once() {
        let mut svc = hosted();
        stage(&mut svc, 4096);
        let ctx = expect_id(svc.answer(0, AccelRequest::CreateContext { flags: 0 }));
        let buffer = expect_id(svc.answer(
            0,
            AccelRequest::CreateBuffer {
                ctx,
                bytes: 64,
                align: 16,
                domain: 3,
                usage: BufferUsage::MUTABLE_STATE.bits(),
            },
        ));
        let artifact = reference::ReferenceArtifact::xor(7, 0x5a);
        let bytes = artifact.as_bytes();
        write_staging(&mut svc, 0, bytes);
        let program = expect_id(svc.answer(
            0,
            AccelRequest::LoadProgram {
                ctx,
                stage: 0,
                len: bytes.len() as u64,
                resident_bytes: reference::RESIDENT_BYTES,
                format: reference::ARTIFACT_FORMAT.get(),
                target: reference::TARGET_IDENTITY.0,
            },
        ));
        let queue = expect_id(svc.answer(0, AccelRequest::CreateQueue { ctx, flags: 0 }));
        let bindings = AccelBindings::new(&[AccelBinding {
            buffer,
            offset: 0,
            len: 8,
            slot: 7,
            access: AccessMode::ReadWrite,
        }])
        .unwrap();
        let event = expect_id(svc.answer(
            0,
            AccelRequest::SubmitLaunch {
                queue,
                program,
                timeout_ns: 0,
                bindings,
                launch: LaunchShape::ArtifactDefined,
            },
        ));

        // Pending: not destroyable, not due.
        assert_eq!(
            svc.answer(0, AccelRequest::DestroyEvent { event }),
            AccelResponse::Refused(AccelRefusal::Busy),
        );
        assert_eq!(svc.live_engine_mut().unwrap().due_by_pair(), [0; PAIRS]);

        {
            let engine = svc.live_engine().unwrap();
            engine
                .backend()
                .complete(engine.event_handle(event).unwrap())
                .unwrap();
        }
        let mut due = [0; PAIRS];
        due[0] = 1;
        assert_eq!(svc.live_engine_mut().unwrap().due_by_pair(), due);
        // The edge was counted; it does not repeat.
        assert_eq!(svc.live_engine_mut().unwrap().due_by_pair(), [0; PAIRS]);

        assert_eq!(
            svc.answer(0, AccelRequest::DestroyEvent { event }),
            AccelResponse::Released { subject: event },
        );
    }

    /// Signalling without an endowed channel space refuses — polling
    /// remains lawful (the host-test posture, and a channel-less row's).
    #[test]
    fn signal_without_channel_space_refuses_no_channel() {
        let mut svc = hosted();
        let out = step(&mut svc, AccelRequest::Signal);
        assert_eq!(out.granted(), 0);
        assert_eq!(
            out.into_reply().unwrap(),
            AccelResponse::Refused(AccelRefusal::NoChannel),
        );
    }

    /// The grant-drop repair rewrites the promise into a refusal.
    #[test]
    fn a_dropped_grant_session_unwinds_to_a_refusal() {
        let mut svc = hosted();
        let mut response = AccelResponse::Signaled;
        svc.grants_unwound(&mut response, 1);
        assert_eq!(response, AccelResponse::Refused(AccelRefusal::Denied));
        // An empty session changes nothing.
        let mut response = AccelResponse::Info(AccelInfo::from_contract(
            MockAccelerator::default().device_info().unwrap(),
        ));
        svc.grants_unwound(&mut response, 0);
        assert!(matches!(response, AccelResponse::Info(_)));
    }

    /// An unhosted engine (the degraded founding posture) still serves —
    /// every verb answers `NotReady`.
    #[test]
    fn an_unhosted_row_serves_not_ready() {
        /// A backend whose own metadata refuses validation (zero limit).
        #[derive(Debug)]
        struct Broken;
        impl NativeAccelerator for Broken {}
        impl Accelerator for Broken {
            type Context = ();
            type Buffer = ();
            type Program = ();
            type Queue = ();
            type Event = ();
            fn device_info(&self) -> Result<DeviceInfo, virtio_accel_core::BackendError> {
                Err(virtio_accel_core::BackendError::DeviceLost)
            }
            fn create_context(
                &self,
                _: virtio_accel_core::ContextDesc,
            ) -> Result<(), virtio_accel_core::BackendError> {
                unreachable!()
            }
            fn destroy_context(&self, _: ()) -> Result<(), virtio_accel_core::ReleaseFailure<()>> {
                unreachable!()
            }
            fn allocate_buffer(
                &self,
                _: &(),
                _: virtio_accel_core::BufferDesc,
            ) -> Result<virtio_accel_core::AllocatedBuffer<()>, virtio_accel_core::BackendError>
            {
                unreachable!()
            }
            fn write_buffer(
                &self,
                _: &mut (),
                _: u64,
                _: &dyn virtio_accel_core::ByteSource,
            ) -> Result<(), virtio_accel_core::BackendError> {
                unreachable!()
            }
            fn read_buffer(
                &self,
                _: &(),
                _: u64,
                _: &mut dyn virtio_accel_core::ByteSink,
            ) -> Result<(), virtio_accel_core::BackendError> {
                unreachable!()
            }
            fn free_buffer(&self, _: ()) -> Result<(), virtio_accel_core::ReleaseFailure<()>> {
                unreachable!()
            }
            fn load_program(
                &self,
                _: &(),
                _: virtio_accel_core::ArtifactRef<'_>,
            ) -> Result<(), virtio_accel_core::BackendError> {
                unreachable!()
            }
            fn unload_program(&self, _: ()) -> Result<(), virtio_accel_core::ReleaseFailure<()>> {
                unreachable!()
            }
            fn create_queue(
                &self,
                _: &(),
                _: virtio_accel_core::QueueDesc,
            ) -> Result<(), virtio_accel_core::BackendError> {
                unreachable!()
            }
            fn destroy_queue(&self, _: ()) -> Result<(), virtio_accel_core::ReleaseFailure<()>> {
                unreachable!()
            }
            fn submit(
                &self,
                _: &(),
                _: &(),
                _: &[virtio_accel_core::BindingRef<'_, ()>],
                _: virtio_accel_core::Timeout,
            ) -> Result<(), virtio_accel_core::SubmitFailure<()>> {
                unreachable!()
            }
            fn poll_event(
                &self,
                _: &(),
            ) -> Result<virtio_accel_core::EventState, virtio_accel_core::BackendError>
            {
                unreachable!()
            }
            fn destroy_event(&self, _: ()) -> Result<(), virtio_accel_core::ReleaseFailure<()>> {
                unreachable!()
            }
        }

        let mut svc = AccelService::host(Broken, None, 1, 1 << 20, 1 << 20);
        assert!(!svc.hosted());
        assert_eq!(
            svc.answer(0, AccelRequest::Query),
            AccelResponse::Refused(AccelRefusal::NotReady),
        );
        assert_eq!(
            svc.answer(0, AccelRequest::CreateContext { flags: 0 }),
            AccelResponse::Refused(AccelRefusal::NotReady),
        );
    }

    /// The generic instantiation's `SystemService` shape holds — bind
    /// from the (empty) endowment and serve through the framework step.
    #[test]
    fn the_system_service_shape_binds_and_serves() {
        let mut svc = <AccelService<MockAccelerator> as SystemService>::bind(MockEndowment);
        assert!(svc.hosted());
        let out = step(&mut svc, AccelRequest::Query);
        assert!(matches!(out.into_reply().unwrap(), AccelResponse::Info(_)));
        // The wire consts hold across the client marker and the host.
        assert_eq!(
            <AccelService<MockAccelerator> as SvcProtocol>::REPLY_GRANTS,
            <crate::wire::AccelWire as SvcProtocol>::REPLY_GRANTS,
        );
        assert_eq!(
            <AccelService<MockAccelerator> as SvcProtocol>::SERVE_PAIRS,
            <crate::wire::AccelWire as SvcProtocol>::SERVE_PAIRS,
        );
    }
}
