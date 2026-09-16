//! # `service` — the unified shape of a bootstrapped system service
//!
//! Every kore userland service — a resource manager, a device driver, a
//! daemon — is the same animal: a **role bundle** (the typed capability
//! endowment `π(ρ)` published to it at spawn — KRAPH §5.2's partitioned
//! bootstrap, the design that rules out a super-init) plus a **serve loop**
//! over it.  [`SystemService`] is that shape as a trait: the framework owns
//! everything dangerous, and the author writes only what is genuinely
//! per-service — the endowment *type*, the request *type*, and one serve
//! step.
//!
//! The misuse surface is closed by construction, not convention:
//!
//! - **Holding too much** — [`Bundle::resolve`] is fail-closed and the
//!   endowment type is the whole reachable authority; there is no ambient
//!   cap table to reach past it.
//! - **Fabricating / widening** — authority leaves a serve step only through
//!   [`Serve::grant`], which can express nothing but an attenuation of a
//!   slot the service already holds (the real dispatch stages it as a
//!   `Derive` — Σ5/Σ6-attenuated, stamp-bound; a forged subject fails
//!   `check` at the kernel, not here).
//! - **Unbounded emission** — one step emits at most [`MAX_GRANTS`] grants:
//!   the `res/SYSCALL.md` D4 bounded-session discipline applied to serving.
//! - **Leaking on death** — the bundle is the task's holdings; kill = revoke
//!   the task root and T6 reaps the subtree.  A service writes no cleanup.
//!
//! **Tokens cross the boundary, never geometry.**  What a client receives
//! from a grant is a delivered capability record — witnessed userland-side
//! as an affine token ([`crate::cap`]) — exactly as `map` takes `(virt,
//! token)` where Linux takes `(virt, phys)`.  The *window* a grant carves
//! is the granting service's internal choice and rides the staged
//! invocation's argument ([`Grant::arg`]); a [`Response`] carries data
//! only.  Peer-to-peer sharing needs no service verb at all: attenuate
//! your own token, move it over rendezvous ([`crate::cap`] module docs).
//!
//! **Allocation-free by contract**: [`step`] and every [`Serve`] operation
//! allocate nothing — a service's serve step may run under an allocator
//! that is itself waiting on a grant, so the framework must never
//! allocate to serve.
//!
//! **The loop is the executor's** (`res/USERLAND.md` §5.5 — #72 was
//! retired by §3.5: there is no kernel ring engine, ever): the doorbell
//! mux wakes [`ServeHost`](crate::svc::ServeHost), which drains the
//! request ring and drives [`step`] per decoded request — metal-proven
//! for mm (ckG2a-3a).  The concrete services live one crate each under
//! `services/` (`kore-svc-{mm,swap,exec,image,input}`,
//! `res/USERLAND.md` §1.1).
//!
//! **Relation to the `tower::Service` RPC veneer (#206):** the two traits
//! own one plane each (#148's split).  `SystemService` is the **authority
//! join-point** — who you are (the bundle) and what you may emit (bounded
//! grants, lowered to *synchronous* session commits) — deliberately sync
//! and allocation-free.  `tower::Service` is the **data-plane** veneer in
//! the SDK (#95): `call → Future` over SQ/CQ, `poll_ready` = ring
//! backpressure, `Layer`s for timeout/retry/auth.  A ring host adapts
//! *outward* (`impl tower::Service for a host over S: SystemService` —
//! the inner future is immediately ready), never inward: no capability
//! rides a ring entry, no grant becomes middleware, and #206's executor
//! decision stays contained in the SDK because this core doesn't need one.
//!
//! [`Response`]: crate::svc::SvcProtocol::Response

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::device::DeviceSpecies;
use kore_abi::mem::DmaClass;
use kore_abi::session::SlotRef;
use kore_abi::task::{Bootstrap, CorePreference, Profile};

/// One declared worker thread of a service ([`SystemService::WORKERS`]).
///
/// `entry` **returns**: falling off the end of it ends the thread with
/// status 0 (`kore_rt::thread::exit`), and the runtime's trampoline is
/// what turns the one into the other.  It used to diverge, because there
/// was no exit verb and a worker with nothing left to do had to park
/// forever rather than return into an unwind that did not exist.  There
/// is an unwind now.
///
/// A worker that must stay resident still may: parking forever is a
/// legitimate steady state (a driver relay whose refusal is itself the
/// diagnostic its status row reports).  Such an entry is still declared
/// `fn()` — a function that never returns satisfies that signature, and
/// writing it `-> !` here would only stop it from being stored in this
/// field.
#[derive(Clone, Copy, Debug)]
pub struct Worker {
    /// Diagnostic name — appears in the spawn witness, so a refused
    /// worker names itself on a serial-less machine.
    pub name: &'static str,
    /// The worker's entry.  Runs `%fs`-anchored on its own stack with
    /// its own cap arena: the ordinary crt0 contract, per thread.
    pub entry: fn(),
}

/// **A worker's declared holdings**, collected by
/// [`SystemService::worker_endowment`] — a bounded, allocation-free
/// builder rather than a returned slice, because the entries name
/// *runtime* arena slots (a service's endowment resolves at bind) and so
/// cannot be `&'static` data the way [`Worker`] itself is.
///
/// [`push`](Self::push) refuses past the ceiling instead of truncating: a
/// worker silently founded without the third of its three capabilities
/// would fail later, somewhere else, for a reason that no longer names
/// this decision.
#[derive(Debug, Default)]
pub struct WorkerEndowment {
    caps: [Option<crate::thread::WorkerCap>; crate::thread::MAX_WORKER_CAPS],
    len: usize,
}

impl WorkerEndowment {
    /// An empty declaration.
    pub const fn new() -> Self {
        Self {
            caps: [const { None }; crate::thread::MAX_WORKER_CAPS],
            len: 0,
        }
    }

    /// Declare one capability; `false` once the bounded set is full.
    pub fn push(&mut self, cap: crate::thread::WorkerCap) -> bool {
        if self.len >= self.caps.len() {
            return false;
        }
        self.caps[self.len] = Some(cap);
        self.len += 1;
        true
    }

    /// How many capabilities were declared.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing was declared (the single-threaded default).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The declared set, in push order — what the runtime hands the
    /// spawn.
    pub fn caps(&self) -> impl Iterator<Item = crate::thread::WorkerCap> + '_ {
        self.caps[..self.len].iter().flatten().copied()
    }
}

/// A bounded, service-defined startup failure carried in the generic startup
/// NACK. The low byte remains the runtime's startup rung; `detail` occupies the
/// upper 56 bits so a service can distinguish allocation, authority, parsing,
/// and hardware failures without allocating an error string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartFailure {
    detail: u64,
}

/// How a service's return channel treats the capability that lands in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnMode {
    /// The founder-published fixed landing is consumed or revoked during the
    /// request. This is the ordinary release/presentation protocol.
    Fixed,
    /// The delivered capability becomes long-lived service authority. The
    /// host donates a runtime-owned landing per pair and transfer and makes
    /// its affine owner available through [`Serve::returned_input`].
    Retained {
        /// Exact resource kind the receiver will accept.
        kind: kore_abi::cap::ResourceKind,
        /// Rights the delivered record must contain after sender attenuation.
        minimum_rights: kore_abi::cap::Rights,
    },
    /// The delivered capability becomes long-lived service authority, but
    /// the protocol intentionally accepts more than one resource species.
    ///
    /// This is the recyclable-session acceptor shape: a complete peer is a
    /// memory lease plus channel endpoints, all conveyed through the same
    /// bounded return leg. The untyped rendezvous post does not trust the
    /// request's component tag; the eventual assembler and admission host
    /// independently inspect every kernel-authenticated capability record.
    RetainedAny,
    /// The service rotates the landing slot after each capability delivery.
    ///
    /// The serve host still owns the channel operation and its ordering: it
    /// asks [`SystemService::pending_return_post`] for the next landing and
    /// confirms a successful post through
    /// [`SystemService::return_post_stood`] before publishing the reply. This
    /// is the L4 reply-and-wait discipline for a service whose affine inputs
    /// must occupy distinct long-lived arena slots (exec's convey table),
    /// without giving the service transport access or a parallel serve loop.
    Walking,
}

impl ReturnMode {
    /// Whether the host must rotate allocator-owned return targets.
    pub const fn is_retained(self) -> bool {
        matches!(self, Self::Retained { .. } | Self::RetainedAny)
    }
}

impl StartFailure {
    /// No service-specific detail; preserves the historical startup rung.
    pub const UNSPECIFIED: Self = Self { detail: 0 };

    /// Construct from a service's bounded numeric vocabulary.
    pub const fn new(detail: u64) -> Self {
        Self {
            detail: detail & 0x00ff_ffff_ffff_ffff,
        }
    }

    /// Lift a legacy Boolean startup predicate without inventing detail.
    pub fn require(ready: bool) -> Result<(), Self> {
        ready.then_some(()).ok_or(Self::UNSPECIFIED)
    }

    /// Join this detail to the runtime-owned low-byte rung.
    pub const fn payload(self, rung: u64) -> u64 {
        (rung & 0xff) | (self.detail << 8)
    }
}

/// Why a recyclable serving peer was retired.
///
/// All variants enter the same ownership teardown.  The distinction is only
/// an observable service fact, so an orderly close cannot accidentally get a
/// more complete cleanup path than a dead or revoked peer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerClose {
    /// The peer completed the protocol's close handshake.
    Orderly,
    /// The host observed that the peer task died.
    Died,
    /// The owner cancelled the session or revoked its authority.
    Revoked,
    /// Protocol violation forced the host to retire the peer.
    Protocol,
}

/// The unified bootstrapped-service shape: a typed role bundle + a request
/// vocabulary + one serve step.  (The kernel-side sibling of this naming
/// discipline is `RuntimeModule::NAME`.)
///
/// ## The host half of a protocol
///
/// The request and reply vocabularies are **not** declared here — they
/// live on [`SvcProtocol`](crate::svc::SvcProtocol), this trait's
/// supertrait, beside the wire forms and codecs that carry them.  The
/// split is the difference between a **wire** and a **host**: a
/// protocol is a vocabulary plus its encoding plus its grant policy; a
/// service is a thing with a name, an endowment, and a serve step that
/// *speaks* one.  A client of the wire needs only the former, so only
/// the former may be required of it.
///
/// What remains here is exactly the host's own facts — [`NAME`](Self::NAME),
/// [`Endowment`](Self::Endowment), [`bind`](Self::bind),
/// [`DEVICES`](Self::DEVICES), [`WORKERS`](Self::WORKERS),
/// [`serve`](Self::serve) — and every one of them is meaningless to a
/// consumer that merely calls the service.  `Self::Request` and
/// `Self::Response` still resolve on a `SystemService` through the
/// supertrait, so a host reads exactly as it did before; what changed is
/// that a *client* no longer has to prove it could be a host.
pub trait SystemService: crate::svc::SvcProtocol + Sized {
    /// Service name — diagnostics and, later, the service directory
    /// ([`role::NAMESPACE`](kore_abi::task::role::NAMESPACE)).
    const NAME: &'static str;

    /// The typed role bundle this service binds at startup — the whole of
    /// what it can ever reach.
    type Endowment: Bundle;

    /// Whether return-channel capability deliveries outlive the request that
    /// receives them.
    ///
    /// The ordinary return leg donates the founder-published fixed landing
    /// slot and expects the serve step to consume or revoke its record.  A
    /// lessor such as the image service instead keeps the delivered lease as
    /// the parent of long-lived derived children.  Opting in makes
    /// [`ServeHost`](crate::svc::ServeHost) donate a runtime-reserved slot per
    /// transfer and expose its affine owner through [`Serve::take_returned`],
    /// so the fixed post cannot be pinned forever by the first registration.
    const RETURN_MODE: ReturnMode = ReturnMode::Fixed;

    /// Capture the resolved endowment — the only construction path, so a
    /// service with an unresolvable bundle never exists.  Every manifest
    /// row binds and serves on metal: there is no construction or serving
    /// opt-out — a service that cannot serve lawfully is completed or
    /// removed from the manifest, never gated (the deleted `_ON_METAL`
    /// discipline, 2026-07-27 ruling).
    fn bind(endowment: Self::Endowment) -> Self;

    /// Named metadata arena for a service that owns dynamically minted pairs.
    #[cfg(feature = "heap")]
    fn minted_pair_arena(&self) -> Option<&'static (dyn core::alloc::Allocator + Sync)> {
        None
    }

    /// **What this row's `role::DEVICE` endowments select at founding** —
    /// entry `n` is the selector for the row's `role::DEVICE` index `n`, so
    /// a driver that owns two controllers (the input service: an xHCI
    /// function *and* an i8042 keyboard controller) declares two entries and
    /// the founder needs no per-service edit: N devices are data.
    ///
    /// Cited per row by the manifest generator (the same single-source
    /// discipline as [`Bundle::PROFILE`]), never inferred from an image
    /// name.  Empty for non-driver rows.
    const DEVICES: &'static [DeviceRequest] = &[];

    /// **Who stops each device** (#809 §L.5): `DEVICE_CLASSES[n]` describes
    /// `DEVICES[n]`; entries past `DEVICES.len()` are bindings claimed at
    /// runtime.  Cited by the manifest generator; the founder asserts each
    /// class against its selector at build time.  A row with a DMA-capable
    /// selector and no class is a build error.
    const DEVICE_CLASSES: &'static [DeviceClass] = &[];

    /// **Which core class this service's threads would rather run on** — a
    /// scheduler preference, never a pin ([`CorePreference`]).
    ///
    /// An I/O-bound service (a filesystem worker, a block or swap driver)
    /// parks on completions for most of its life and does its work in a few
    /// microseconds between parks; declaring `Efficient` keeps it off the
    /// performance cores the compute-bound tasks need, and costs it nothing
    /// a completion wait would notice.  Cited by the manifest generator per
    /// row, like [`DEVICES`](Self::DEVICES); inert on a homogeneous machine.
    const CORE_PREFERENCE: CorePreference = CorePreference::Any;

    /// **The service's worker threads** — declared, not hand-spawned.
    ///
    /// Each entry becomes a real thread of this task (the unified task
    /// model, #407: its own arena, its own parks, the same address
    /// space), spawned by [`run_service`](crate::entry::run_service)
    /// after `start_on_metal` and before the serve loop.  A service that
    /// needs a second wait — a driver relay parking on an interrupt
    /// route while thread 0 serves doorbell-parked — declares it here
    /// and writes no spawn code at all; `#[kore_rt::task]` keeps its
    /// cooperative meaning *within* each worker.
    ///
    /// Declared rather than spawned so the runtime owns the mechanism
    /// uniformly: one place decides ordering (workers exist before the
    /// service advertises readiness, so a client's first request never
    /// races a half-built service), one place reports refusals, and a
    /// service body stays a description of its own topology.
    ///
    /// Empty by default — a single-threaded service declares nothing.
    const WORKERS: &'static [Worker] = &[];

    /// Admit only workers whose runtime resources exist. Indices remain those
    /// of `WORKERS`; skipping a vacant optional device never renumbers siblings.
    fn worker_enabled(&self, _worker: usize) -> bool {
        true
    }

    /// Rights on the optional send end into this service's own doorbell.
    ///
    /// Worker-bearing services get the established `ACCESS | DERIVE`
    /// self-kick by default: they keep it for their whole lifetime and derive
    /// attenuated children into their worker tasks.  A single-threaded
    /// service normally gets no self-kick at all.
    ///
    /// A service with another kernel-scheduled producer may opt in without
    /// pretending that producer is one of [`WORKERS`](Self::WORKERS).  It may
    /// also request `REVOKE` when those producers have independent lifetimes
    /// and their derived endpoints must be reaped before the service exits.
    const SELF_KICK_RIGHTS: Rights = if Self::WORKERS.is_empty() {
        Rights::empty()
    } else {
        crate::svc::SEND_RIGHTS
    };

    /// One structurally authenticated serving peer reached a terminal
    /// lifecycle edge.
    ///
    /// Recyclable serving hosts call this exactly once after removing the
    /// peer's ring/endpoints from their live fan-in and before the slot may be
    /// reused.  Services use it to discard state keyed by
    /// [`Serve::via`]. The reason is host-authored lifecycle state, never a
    /// caller-selected request field.
    ///
    /// Return `true` only after every service-owned reference to this exact
    /// arrival generation is gone. Returning `false` keeps the host's slot in
    /// its closing state and retries this hook from the ordinary serve loop;
    /// a replacement cannot be admitted into the cell in the meantime. This
    /// is the domain half of peer teardown's two-phase commit: host-owned
    /// mappings/endpoints close first, service-owned authority closes second.
    ///
    /// Fixed founding topologies need no hook and retain the no-op default.
    fn peer_closed(&mut self, _via: u16, _reason: PeerClose) -> bool {
        true
    }

    /// Attach a recyclable peer's protocol-specific tail after its canonical
    /// request/reply regions. The host retains the mapping owner. Any ring
    /// halves retained here MUST be released by `peer_detaching`, even when
    /// this method refuses after partially inspecting/attaching the tail.
    fn attach_peer_aux(&mut self, _peer: crate::svc::PeerId, _aux: &'static mut [u8]) -> bool {
        true
    }

    /// Release all borrowed auxiliary-plane references before the host
    /// unmaps a recyclable pair. This is not the capability-revocation hook:
    /// retryable domain custody still closes in `peer_closed` afterward.
    fn peer_detaching(&mut self, _peer: crate::svc::PeerId) {}

    /// Observe completion of an admission requested through
    /// [`Serve::admit_peer`]. The host, not the wire request, chooses the
    /// generational identity and reports only the authenticated result.
    fn peer_admitted(
        &mut self,
        _result: Result<crate::svc::PeerId, crate::svc::PeerAdmissionRefusal>,
    ) -> bool {
        true
    }

    /// Whether a ring-published request's authority-bearing transaction is
    /// ready to be consumed.
    ///
    /// A client necessarily writes its SPSC record before entering the
    /// kernel session that atomically moves capability custody and rings the
    /// service doorbell.  Services whose request is authorized by that same
    /// transaction use this hook to keep the record unobservable until all
    /// authenticated landings stand.  The host peeks but does not dequeue;
    /// malformed wire records bypass the hook and are drained fail-closed.
    fn request_ready(&mut self, _request: &Self::Request, _via: u16) -> bool {
        true
    }

    /// **What worker `n` is founded holding.**
    ///
    /// A worker thread is a task, so it starts with no capabilities of
    /// its own: sharing an address space shares memory, never authority.
    /// A worker that must *act* — park on a route, drive I/O ports —
    /// therefore names here the capabilities the runtime should derive
    /// into its arena at spawn, off subjects this service already holds.
    ///
    /// The service is the only thing that knows which of its authorities
    /// a given worker needs, so the declaration lives here; the
    /// *mechanism* stays entirely in the runtime, which is the split
    /// [`WORKERS`](Self::WORKERS) already draws.  A worker that only
    /// reads shared state declares nothing, which is the default. The runtime
    /// separately derives the process's `role::HEAP@0` supply into every
    /// declared service worker: allocator memory is process-wide, but the
    /// capability required to grow a worker's private allocator shard is a
    /// per-task holding and therefore cannot follow from the shared VAS.
    ///
    /// Each entry is a **derive**, not a move: the service keeps its
    /// subject, so revoking a worker's authority stays its lever.
    fn worker_endowment(&self, worker: usize, out: &mut WorkerEndowment) {
        let _ = (worker, out);
    }

    /// Perform a bounded, service-owned startup action after the typed
    /// endowment has been resolved and before readiness is advertised.
    ///
    /// The default is inert.  Exec uses this one hook to claim its fixed
    /// spawn-engine slot band before any wire traffic can race the
    /// reservation. An error is fail-closed and prevents
    /// both `COOKIE_READY` and `COOKIE_SERVING`; its bounded detail is carried
    /// in the founder NACK instead of being discarded behind a Boolean.
    fn start_on_metal(&mut self, _rt: &crate::entry::Rt) -> Result<(), StartFailure> {
        Ok(())
    }

    /// Advance a service-owned asynchronous readiness proof.
    ///
    /// Called only after the generic host has attached every serving ring and
    /// established its standing return posts, but before `COOKIE_SERVING` is
    /// published. `Ok(true)` means the service may advertise readiness;
    /// `Ok(false)` parks on the ordinary executor doorbell and calls this hook
    /// again after the next wake. An error follows the same rung-preserving
    /// NACK path as [`start_on_metal`](Self::start_on_metal).
    ///
    /// Most services finish startup synchronously and inherit the immediate
    /// default. A service whose readiness depends on another scheduled
    /// producer — a hypervisor waiting for its first vCPU exit, for example —
    /// uses this seam instead of inventing a private executor or boot path.
    fn advance_readiness(&mut self) -> Result<bool, StartFailure> {
        Ok(true)
    }

    /// The row's **non-fatal readiness note** — OR'd into the serving
    /// beacon's payload after [`advance_readiness`](Self::advance_readiness)
    /// settles, rendered by the founder as the ready line's `note: rung N
    /// (detail …)` suffix and kept in the boot ledger.  Same encoding as a
    /// refusal payload (`[7:0]` rung, `[63:8]` detail; `0` = nothing to
    /// say), so a reader never learns two conventions.  This is the seam
    /// for a service whose bad news is discovered DURING its asynchronous
    /// readiness proof — a fixed `serve_forever` note argument is stamped
    /// before that proof runs.  Degraded-but-serving is the point: a row
    /// whose optional hardware is absent reports here and keeps serving,
    /// instead of NACKing a machine that simply lacks the device.
    fn ready_note(&self) -> u64 {
        0
    }

    /// Attach the protocol-specific bytes after the canonical request/reply
    /// regions for fixed serving pair `pair`. The founder formatted these
    /// bytes through [`SvcProtocol::init_pair_window`](crate::svc::SvcProtocol::init_pair_window)
    /// before sharing the window. The default ignores the unused tail of an
    /// ordinary page-sized pair.
    fn attach_pair_aux(&mut self, pair: usize, aux: &'static mut [u8]) -> bool {
        let _ = (pair, aux);
        true
    }

    /// Earliest absolute monotonic deadline owned by this service. Ordinary
    /// request-only services keep the inert default.
    fn next_deadline(&self) -> Option<u64> {
        None
    }

    /// Process every internal deadline due at `now_ns`. The return value is a
    /// bitset of fixed serving pairs whose auxiliary event rings received at
    /// least one record; the host sends one droppable wake notice per set bit
    /// after the batch. Bytes move through rings, authority never does.
    fn drive_deadlines(&mut self, now_ns: u64) -> u16 {
        let _ = now_ns;
        0
    }

    /// **Terminal device quiesce** (#809 §L) — called once by the generic
    /// host when the kernel's shutdown sweep wakes the row, before
    /// `deadline_ns` (absolute monotonic nanoseconds).  The row reaches each
    /// `BoundDevice` it holds and runs `teardown`, which owns the order —
    /// class stop, kernel fabric proof, containment; this hook only reaches
    /// the bindings and folds their outcomes.  Async because a row may have
    /// to drain something first (the filesystem retires two lanes over one
    /// controller before the controller may stop).  The default is inert:
    /// a row with no `DeviceClass::Binding` citation has nothing to reach.
    async fn quiesce_devices(&mut self, deadline_ns: u64) -> QuiesceSweep {
        let _ = deadline_ns;
        QuiesceSweep::NONE
    }

    /// Select this service's generic host driver.  The default is the ordinary
    /// request-only loop.  A deadline-driven service overrides this one typed
    /// seam and returns [`ServeHost::run_deadline_driven`]; ordinary service
    /// futures then contain no timer-policy branch or duplicate host state.
    ///
    /// Selecting the driver through this typed hook matters: a service may
    /// itself be a large bounded value, and its executor stack must contain
    /// exactly one affine host owner rather than both host-loop variants.
    fn drive_host<T: crate::transport::Transport>(
        host: crate::svc::ServeHost<'_, '_, T, Self>,
    ) -> impl core::future::Future<Output = core::convert::Infallible> {
        async move {
            host.run().await;
        }
    }

    /// Answer one request.  Authority leaves only through `sv`.
    fn serve(&mut self, req: Self::Request, sv: &mut Serve<'_, Self::Response>);

    /// Answer one request while retaining executor progress.
    ///
    /// Ordinary services inherit the synchronous serve step. A service whose
    /// reply depends on another service (for example durable filesystem I/O)
    /// overrides this seam so it may await that dependency without polling it
    /// under a foreign or no-op waker. The host still owns the one request
    /// transaction and does not publish the reply until this future settles.
    async fn serve_async(&mut self, req: Self::Request, sv: &mut Serve<'_, Self::Response>) {
        self.serve(req, sv);
    }

    /// Observe a successfully committed non-empty grant batch before its
    /// reply is encoded. `via` is the same structural serving-pair identity
    /// supplied to `serve`. The transferred scratch names are already vacant.
    /// This hook may retain correlation through a bounded kernel observation;
    /// failure cannot undo delivery or change the effected grant count.
    fn grants_committed(&mut self, response: &Self::Response, emitted: usize, via: u16) {
        let _ = (response, emitted, via);
    }

    /// Repair service-owned state after a non-empty grant session was
    /// **dropped whole** by [`ServeHost`](crate::svc::ServeHost).
    ///
    /// `serve` runs before the transport commit, so a service may have to
    /// reserve policy state before it emits a grant.  If the client's pinned
    /// post is absent, or the staged session otherwise refuses, none of the
    /// `emitted` grants took effect and the host calls this hook exactly once
    /// before encoding the reply with an effected count of zero.  A service
    /// whose reply promises the grant must both undo that reservation and
    /// rewrite `response` to a grant-less refusal here.
    ///
    /// The default preserves the existing leak-accounting policy of services
    /// whose allocated object remains service-owned when delivery fails.  The
    /// hook is deliberately narrower than [`Serve`]: it can mutate local
    /// policy and response data, but cannot emit authority or touch the
    /// transport.  Like `serve`, it must be bounded and allocation-free.
    fn grants_dropped(&mut self, response: &mut Self::Response, emitted: usize) {
        let _ = (response, emitted);
    }

    /// The next service-selected return landing to post for
    /// [`ReturnMode::Walking`].
    ///
    /// `None` means the bounded landing table is currently saturated. The
    /// host calls this before readiness and after every served request, and
    /// never publishes that request's reply until a returned target has been
    /// posted (or the service reports saturation). Other return modes ignore
    /// this hook.
    fn pending_return_post(&self) -> Option<SlotRef> {
        None
    }

    /// Confirm that the target returned by [`pending_return_post`](Self::pending_return_post)
    /// now has a standing receive.
    ///
    /// The callback is deliberately acknowledgement-only: the host owns the
    /// channel commit, while the service advances its affine landing cursor
    /// only after `Ok`/`WouldBlock` proves that the receive stands.
    fn return_post_stood(&mut self) {}

    /// A **dynamically minted serving pair** awaiting attach (slice C —
    /// the multi-client seam): polled by
    /// [`ServeHost`](crate::svc::ServeHost) after each served request, so
    /// a pair a serve step minted (the image service's per-child loader
    /// faces) is drained by the same loop from its very next wake — one
    /// engine, instantiated wider, never a parallel path.  The
    /// `pending_repost`/`repost_stood` handshake shape, host-mediated:
    /// the service exposes the spec, the host attaches and calls
    /// [`pair_attached`](Self::pair_attached) with the outcome.
    ///
    /// The default is inert — an ordinary service mints no pairs.
    fn pending_pair_attach(&mut self) -> Option<MintedPairSpec> {
        None
    }

    /// The host's answer to [`pending_pair_attach`](Self::pending_pair_attach):
    /// `true` = the pair is attached and draining; `false` = the bank was
    /// full or the identity collided, and the service must unwind its
    /// mint (the spec is consumed either way — re-expose to retry).
    fn pair_attached(&mut self, via: u16, attached: bool) {
        let _ = (via, attached);
    }

    /// A minted pair to **detach** (slice C's retire leg), by the identity
    /// it was attached with: polled beside
    /// [`pending_pair_attach`](Self::pending_pair_attach); the host frees
    /// the bank slot and stops draining that ring.  The service revokes
    /// the pair's channel roots itself — detach is bookkeeping, revoke is
    /// authority.
    fn pending_pair_detach(&mut self) -> Option<u16> {
        None
    }
}

/// One dynamically minted serving pair, as the service hands it to the
/// host ([`SystemService::pending_pair_attach`]) — the mapped pair
/// window's raw parts, the pair's grant-send lanes, its structural
/// identity, and its (optional) reply kick.  Kickless is the loader-face
/// shape: the stub spin-polls, so a reply kick would be pure spurious
/// wake traffic.
#[derive(Debug)]
pub struct MintedPairSpec {
    window_base: core::ptr::NonNull<u8>,
    window_len: usize,
    /// Grant-channel send ends, `[0..REPLY_GRANTS]` populated.
    pub grant_tx: [Option<SlotRef>; MAX_GRANTS],
    /// The identity requests off this pair are stamped with
    /// ([`Serve::via`]).  Must not collide with the fixed pairs
    /// (`< MAX_SERVE_PAIRS`) or another live minted pair; the service
    /// owns uniqueness (keying it to its own slot table is the natural
    /// choice) and the host refuses a collision through
    /// [`pair_attached`](SystemService::pair_attached).
    pub via: u16,
    /// Reply kick toward the pair's client, if it has one.
    pub kick: Option<SlotRef>,
    /// Optional receiver-authored admission proof. The host will not dequeue
    /// this pair's request ring until the proof has landed in this
    /// service-owned slot.
    pub admission: Option<MintedAdmission>,
}

/// Structurally bound admission for a dynamically minted serving pair.
/// Neither the landing nor the expected value is request-controlled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MintedAdmission {
    pub land: SlotRef,
    pub kind: MintedAdmissionKind,
}

/// The authenticated record that admits one request from a minted pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MintedAdmissionKind {
    /// A reusable endpoint sends this fixed, bind-inert cookie on every call.
    Notice(u64),
    /// A one-shot endpoint transfers an affine capability into the landing.
    Capability,
}

impl MintedPairSpec {
    /// Wrap a minted pair's pieces for host attach.
    ///
    /// # Safety
    ///
    /// `window` must be a live mapping of the pair's ring window in THIS
    /// task's own space, at least `len` bytes, formatted for this
    /// service's protocol, outliving the serve host (the map-once-per-slot
    /// contract), and touched by nothing but this pair's client (through
    /// the SPSC protocol) and the host once attached.  The constructor is
    /// where the obligation pins: the minting service owns the mapping's
    /// lifecycle, so only it can warrant this.
    pub unsafe fn new(
        window: core::ptr::NonNull<u8>,
        len: usize,
        grant_tx: [Option<SlotRef>; MAX_GRANTS],
        via: u16,
        kick: Option<SlotRef>,
    ) -> Self {
        Self {
            window_base: window,
            window_len: len,
            grant_tx,
            via,
            kick,
            admission: None,
        }
    }

    /// Gate each dequeued request on a notice delivered through the bound
    /// channel peer. The receive post must already stand when this spec is
    /// exposed to the host.
    pub fn authenticated(mut self, land: SlotRef, cookie: u64) -> Self {
        self.admission = Some(MintedAdmission {
            land,
            kind: MintedAdmissionKind::Notice(cookie),
        });
        self
    }

    /// Gate the next request on an affine capability delivery.
    pub fn authenticated_capability(mut self, land: SlotRef) -> Self {
        self.admission = Some(MintedAdmission {
            land,
            kind: MintedAdmissionKind::Capability,
        });
        self
    }

    /// The window's raw parts, for the host's attach.
    pub(crate) fn window_raw(&self) -> (core::ptr::NonNull<u8>, usize) {
        (self.window_base, self.window_len)
    }
}

/// What a driver row's `role::DEVICE` endowment selects at founding.
///
/// Two kinds of device exist, and a selector names which it wants — a
/// driver cannot be handed the wrong species by accident:
///
/// - [`Pci`](Self::Pci) — a PCI function, matched on its class triple,
///   optional exact vendor/device identity, and the capabilities the
///   service's access pattern demands;
/// - [`Platform`](Self::Platform) — a firmware-described **non-PCI**
///   device (#417), matched on a compressed ACPI id the device answers
///   to in `_HID` **or** `_CID`: fixed I/O/MMIO resources, an optional
///   interrupt line, and—when requested—a firmware IOMMU binding;
/// - [`PlatformI2cHid`](Self::PlatformI2cHid) — a topology-selected I2C
///   controller with a firmware-owned MMIO window and no PCI/IOMMU identity.
/// - [`PciI2cHid`](Self::PciI2cHid) — an enumerated PCI controller with
///   initialized firmware-correlated HID children; BAR0, without DMA.
///
/// Cited per row by the manifest generator — the selector is the service
/// type's own declaration, never a founder guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceSelector {
    /// A PCI function (the original shape).
    Pci(PciSelector),
    /// A firmware-described non-PCI device, by compressed ACPI id
    /// (e.g. `PNP0303` for an i8042 keyboard controller — derive the
    /// value with `eisa_id`, never hand-encode it).  The device matches
    /// when it answers to the id in `_HID` **or any `_CID`** — real
    /// firmware routinely puts the class id only in the compatible list
    /// (the Yoga's keyboard is `_HID=MSFT0001`, `_CID=PNP0303`).
    Platform {
        /// The compressed id to match — against the id the boot scout
        /// matched the device by, its `_HID` or a `_CID`.
        hid: u32,
        /// Require the firmware to have named an interrupt line: a
        /// driver that must park on its device refuses a row whose
        /// `_CRS` gave none. Polling-capable classes set this false
        /// explicitly rather than receiving a fabricated line.
        irq: bool,
        /// Require a firmware requester identity with a live IOMMU binding.
        /// DMA-capable ACPI devices fail arbitration here rather than after
        /// their affine root and the caller's sole DMA pool have moved.
        dma: bool,
    },
    /// A firmware-correlated MMIO I2C controller with at least one HID
    /// child.  The role is topology-derived rather than encoded as the
    /// controller's ACPI id: common controller ids are eight-character
    /// strings and cannot fit the compressed EISA word.
    PlatformI2cHid {
        /// Require the controller's own completion line.  The HID child's
        /// GPIO data-ready route is a separate capability lane (#408).
        irq: bool,
    },
    /// An Intel Lunar Lake / Panther Lake-P LPSS I2C controller with initialized firmware-
    /// correlated HID children and a user-mappable BAR0. This initial profile
    /// admits Intel a850/a851/a878–a87b and e450/e451/e478–e47b; another wrapper requires an
    /// explicit profile extension before the driver may apply LPSS resets.
    /// No DMA authority is required.
    PciI2cHid {
        /// Require the controller's firmware-resolved INTx completion line,
        /// independent of any HID child's GPIO data-ready interrupt.
        irq: bool,
    },
}

/// One device a service row is founded with — entry `n` of
/// [`SystemService::DEVICES`] becomes the row's `role::DEVICE` endowment at
/// index `n`.
///
/// The pair is what makes N devices *data*: the selector says which
/// hardware, and [`required`](Self::required) says what a machine that has
/// none of it means for this row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceRequest {
    /// Which device this index selects.
    pub selector: DeviceSelector,
    /// Whether the row refuses to found when firmware describes no
    /// match.  `false` leaves the endowment index VACANT and the
    /// service's bundle resolves it as absent — the shape a *second*
    /// input source takes: a machine with no i8042 still founds the
    /// input service, it just has one fewer keyboard.
    pub required: bool,
}

/// **What owns each device a row touches** (#809 §L.5) — `DEVICES[n]` is
/// described by `DEVICE_CLASSES[n]`, and entries past `DEVICES.len()` name
/// bindings the row claims at runtime (the filesystem's NVMe, provisioned by
/// the block row).  Cited by the manifest generator exactly like `DEVICES`;
/// the founder asserts every class against its selector at build time, so
/// a DMA-capable function with nobody to stop it is a compile error, not a
/// shutdown surprise.
///
/// The variants carry no reasons as data — a `&'static str` per row would
/// be heap-free but would also be a second literal the predicate cannot
/// check.  The *class* is the fact; the reason is the variant's doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceClass {
    /// A `DeviceBinding` class: the row holds a `BoundDevice` and is in the
    /// terminal quiesce sweep.  `bars` is the class's own declaration
    /// (`kore_driver::host::device_class::<B>()` fills it), asserted equal to
    /// the selector's so the two cannot drift.
    Binding { bars: PciBarMask },
    /// A raw token whose function issues no DMA and arms no route (the
    /// polled 16550 the output and stream-input rows hold).  Nothing to
    /// stop; knowingly outside the sweep.
    NoDma,
    /// A token this row claims only to provision to another row, which
    /// cites the binding (block → fs).  The founder resolves `to` to a row
    /// whose classes carry a matching `Binding`.
    Delegated { to: &'static str },
    /// Held by a relay worker as a raw platform / port token outside the
    /// `DeviceBinding` seam (i8042, the I2C-HID relays).  No DMA; knowingly
    /// unenforced (#809 §L.6) until a platform rendezvous exists.
    RelayHeld,
}

/// What one row's terminal quiesce did (#809 §L) — the row's own record
/// for its witness line, folded from each binding's `BindingTeardown`.
/// Not a reply to the kernel: the kernel observes completion at the
/// device's DMA epoch.  Counts, not bitmasks, so a row with more bindings
/// than bits still reports truthfully.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuiesceSweep {
    /// Bindings torn down.
    pub devices: u8,
    /// Bindings whose class answered `Stopped`.
    pub class_stopped: u8,
    /// Bindings the kernel proved quiet (`FabricVerdict::Proven`).
    pub fabric_proven: u8,
    /// Bindings the kernel contained at the IOMMU (`FabricVerdict::Contained`).
    pub fabric_contained: u8,
    /// Bindings whose capability containment completed.
    pub contained: u8,
    /// The deadline cut a multi-step sequence short (a lane still draining):
    /// recorded as partial, never as skipped.
    pub partial: bool,
}

impl QuiesceSweep {
    /// Nothing torn down — the inert default every deviceless row answers.
    pub const NONE: Self = Self {
        devices: 0,
        class_stopped: 0,
        fabric_proven: 0,
        fabric_contained: 0,
        contained: 0,
        partial: false,
    };

    /// Fold one binding's outcome in.
    pub fn note(
        &mut self,
        class_stopped: bool,
        fabric: Option<crate::cap::FabricVerdict>,
        contained: bool,
    ) {
        self.devices = self.devices.saturating_add(1);
        if class_stopped {
            self.class_stopped = self.class_stopped.saturating_add(1);
        }
        match fabric {
            Some(crate::cap::FabricVerdict::Proven) => {
                self.fabric_proven = self.fabric_proven.saturating_add(1);
            }
            Some(crate::cap::FabricVerdict::Contained) => {
                self.fabric_contained = self.fabric_contained.saturating_add(1);
            }
            None => {}
        }
        if contained {
            self.contained = self.contained.saturating_add(1);
        }
    }
}

/// Optional exact PCI identity constraint for a [`PciSelector`].
///
/// `ANY` deliberately retains the original class-only selector encoding as
/// zero. An exact identity packs `vendor | device << 16` into the request
/// word that was previously required to be zero, so old peers refuse the new
/// vocabulary instead of misinterpreting it. The private representation keeps
/// `0000:0000` from collapsing an exact request back into `ANY`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciIdentitySelector(u32);

impl PciIdentitySelector {
    /// Match any vendor/device pair that satisfies the class and resource
    /// requirements. Existing class drivers use this explicitly.
    pub const ANY: Self = Self(0);

    /// Match one exact PCI vendor/device pair.
    ///
    /// `0000:0000` has no exact representation because zero is the frozen
    /// wildcard encoding. PCI endpoints do not lawfully enumerate with that
    /// pair; refusing it also prevents an accidental wildcard claim.
    pub const fn exact(vendor: u16, device: u16) -> Self {
        let packed = vendor as u32 | ((device as u32) << 16);
        assert!(packed != 0, "0000:0000 is the PCI identity wildcard");
        Self(packed)
    }

    /// Match every function of one vendor that satisfies the class and
    /// resource requirements — the shape a driver whose admission table is
    /// its own (`XeDeviceProfile::for_pci`, the ivpu identity rows) claims
    /// with, so the claim names *who may be admitted* and the driver names
    /// *who is*. Encoded as the vendor with device `0xFFFF`: no PCI function
    /// lawfully enumerates with an all-ones device id, so an old peer that
    /// reads this word as an exact pair matches nothing rather than
    /// misinterpreting it.
    pub const fn vendor(vendor: u16) -> Self {
        assert!(vendor != 0 && vendor != 0xFFFF, "not a lawful PCI vendor id");
        Self(vendor as u32 | (VENDOR_ONLY_DEVICE << 16))
    }

    /// Lift the complete 32-bit request-word vocabulary.
    pub const fn from_packed(packed: u32) -> Self {
        Self(packed)
    }

    /// The vendor a vendor-scoped selector admits, or `None` for [`ANY`](Self::ANY)
    /// and exact pairs.
    pub const fn vendor_only(self) -> Option<u16> {
        if self.0 != 0 && (self.0 >> 16) == VENDOR_ONLY_DEVICE {
            Some(self.0 as u16)
        } else {
            None
        }
    }

    /// Lower into the stable request-word representation.
    pub const fn packed(self) -> u32 {
        self.0
    }

    /// The exact pair, or `None` for [`ANY`](Self::ANY) and vendor-scoped
    /// selectors.
    pub const fn exact_pair(self) -> Option<(u16, u16)> {
        if self.0 == 0 || (self.0 >> 16) == VENDOR_ONLY_DEVICE {
            None
        } else {
            Some((self.0 as u16, (self.0 >> 16) as u16))
        }
    }

    /// Whether one enumerated identity satisfies this constraint.
    pub const fn matches(self, vendor: u16, device: u16) -> bool {
        if let Some(required_vendor) = self.vendor_only() {
            return vendor == required_vendor;
        }
        match self.exact_pair() {
            Some((required_vendor, required_device)) => {
                vendor == required_vendor && required_device == device
            }
            None => true,
        }
    }
}

/// The device half of a vendor-scoped [`PciIdentitySelector`]: the all-ones
/// id no function lawfully enumerates with.
const VENDOR_ONLY_DEVICE: u32 = 0xFFFF;

/// Exact user-mappable PCI BARs a device claimant requires.
///
/// Only BAR indices 0 through 5 are representable. The private field prevents
/// a selector from silently carrying an out-of-range bit; unions remain exact
/// and allocation-free in `const` service declarations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciBarMask(u8);

impl PciBarMask {
    pub const BAR0: Self = Self(1 << 0);
    pub const BAR1: Self = Self(1 << 1);
    pub const BAR2: Self = Self(1 << 2);
    pub const BAR3: Self = Self(1 << 3);
    pub const BAR4: Self = Self(1 << 4);
    pub const BAR5: Self = Self(1 << 5);
    const ALL: u8 = (1 << 6) - 1;

    /// Lift a non-empty, in-range wire mask.
    #[inline]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits != 0 && bits & !Self::ALL == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// The PCI half of a [`DeviceSelector`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciSelector {
    /// PCI base class.
    pub class: u8,
    /// PCI subclass.
    pub subclass: u8,
    /// PCI programming interface.
    pub progif: u8,
    /// Optional exact vendor/device filter. Accelerator providers must use an
    /// exact identity so a class match cannot seize incompatible hardware.
    pub identity: PciIdentitySelector,
    /// Exact BAR indices the class will map during intake. The broker proves
    /// all of them user-mappable before consuming the inventory row.
    pub bars: PciBarMask,
    /// Require a live IOMMU binding (the device performs DMA).
    pub dma: bool,
    /// Interrupt delivery the claimant can actually consume.
    pub interrupts: PciInterruptRequirement,
}

/// Interrupt-delivery fact a PCI claimant requires before the broker grants
/// the affine device capability.
///
/// Exact MSI-X table-entry geometry required at claim time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciMsixBlock {
    first_entry: u16,
    count: u16,
}

impl PciMsixBlock {
    /// Xe's shared root/GT/GuC/display dispatcher entry plus its distinct
    /// HWE/completion entry.
    pub const ENTRIES_0_1: Self = Self {
        first_entry: 0,
        count: 2,
    };

    /// Construct a block representable by the interrupt-establishment seam.
    #[inline]
    pub const fn new(first_entry: u16, count: u16) -> Option<Self> {
        if count != 0 && count.is_power_of_two() && first_entry.checked_add(count).is_some() {
            Some(Self { first_entry, count })
        } else {
            None
        }
    }

    #[inline]
    pub const fn first_entry(self) -> u16 {
        self.first_entry
    }

    #[inline]
    pub const fn count(self) -> u16 {
        self.count
    }
}

/// This is an enum rather than booleans because an MSI-X block is an exact,
/// stricter contract, not an independent feature. A queue whose protocol owns
/// entries 0 and 1 must never claim an MSI-only or one-entry function and
/// discover the mismatch after the broker ledger has become permanent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PciInterruptRequirement {
    /// The driver has a lawful polling posture and requires no route.
    #[default]
    None,
    /// Any eligible remapped MSI-family route; MSI-X is preferred and
    /// ordinary MSI is accepted.
    MsiFamily,
    /// One exact remapped MSI-X table-entry block; ordinary MSI and a smaller
    /// table are not substitutable.
    Msix(PciMsixBlock),
}

/// Sealing for [`BootClaim`]: only runtime-defined affine possession tokens
/// can opt into the derive's generic claim step. Composite endowments keep a
/// handwritten [`Bundle`] implementation rather than smuggling construction
/// policy through a user-implementable claim trait.
#[doc(hidden)]
pub(crate) mod boot_claim_sealed {
    pub trait Sealed {}
}

/// A token that can be possessed from one founder-published boot slot.
///
/// This is the type half of [`derive@crate::Bundle`]. The field attribute owns
/// the contextual `(role, index)`; the field type owns how a record of its
/// species is authenticated and possessed. Keeping those facts separate is
/// load-bearing: one supply-token species serves several roles, while every
/// role still has exactly one type-checked claim path.
pub trait BootClaim: boot_claim_sealed::Sealed + Sized {
    /// Authenticate and possess the capability at `slot`.
    fn claim_boot(slot: SlotRef) -> Option<Self>;
}

/// One non-endowment bootstrap handle a bundle captures as wiring.
///
/// These coordinates are intentionally separate from [`Profile`]: a profile
/// delegates authority, while wiring names an already-constructed channel or
/// topology edge. The declaration is still static and inspectable so founder
/// generators and readiness gates do not need a second handwritten role list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WiringSlot {
    /// Bootstrap role discriminant.
    pub role: u16,
    /// Instance within the role.
    pub index: u16,
    /// Whether readiness must refuse when the founder omitted this coordinate.
    pub required: bool,
}

impl WiringSlot {
    /// A coordinate required for this service shape.
    pub const fn required(role: u16, index: u16) -> Self {
        Self {
            role,
            index,
            required: true,
        }
    }

    /// An optional topology coordinate. Its absence endows nothing and does
    /// not make the service's authority profile weaker.
    pub const fn optional(role: u16, index: u16) -> Self {
        Self {
            role,
            index,
            required: false,
        }
    }
}

/// A typed role bundle, resolved from the task's bootstrap context
/// ([`kore_abi::task::Bootstrap`] — the role-tagged handle table).
///
/// Resolution is **possession-shaped**: the handle table only *locates* a
/// record the kernel already seeded into this task's arena; binding takes
/// possession of it (a one-time claim), and metadata — a region's length —
/// rides the claimed token, never a config string (capabilities do not
/// serialize; a string is copyable, and disclosure ≠ transfer).
///
/// Ordinary bundles should derive this trait with [`derive@crate::Bundle`].
/// Each field declares its contextual boot role while its token type supplies
/// the sealed possession operation, so the profile, claim path, and optional
/// topology captures stay one source of truth. Handwritten implementations
/// remain for composite resources whose resolution performs more than one
/// capability claim or constructs a derived runtime object.
pub trait Bundle: Sized {
    /// The **capability profile** ([`kore_abi::task::Profile`]) this bundle is
    /// founded from — the role set the exec manager stamps into the child's
    /// boot block, *the same list [`resolve`](Bundle::resolve) reads back*.
    /// Declaring it here makes the founding side and the resolving side one
    /// source of truth, so they cannot drift; a device driver's is
    /// `[DEVICE, DMA_POOL, IRQ_ROUTE]`, mm's is `[MEM_SUPPLY]`.
    const PROFILE: Profile;

    /// Non-endowment bootstrap handles captured by this bundle. The derive
    /// emits this from fields marked `capture`; handwritten composite bundles
    /// default to no captured wiring.
    const WIRING: &'static [WiringSlot] = &[];

    /// Resolve the bundle — **fail-closed**: any missing role or failed
    /// claim refuses the whole bundle, so a mis-endowed service never
    /// starts (the runtime face of `BootWF`'s publication discipline).
    fn resolve(boot: &Bootstrap<'_>) -> Result<Self, BindError>;
}

/// Why a bundle refused to resolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindError {
    /// A required role handle is absent from the bootstrap table.
    MissingRole {
        /// The [`kore_abi::task::role`] discriminant.
        role: u16,
        /// Instance index within the role.
        index: u16,
    },
    /// The handle names a slot whose record could not be claimed — nothing
    /// seeded there, or possession already taken.
    BadHandle {
        /// The [`kore_abi::task::role`] discriminant.
        role: u16,
        /// Instance index within the role.
        index: u16,
    },
    /// Present immutable startup data violates the consuming service's contract.
    InvalidDatum {
        /// The datum role whose value was refused.
        role: u16,
        /// First word of the invalid value.
        index: u16,
    },
    /// A required immutable startup datum is absent from the validated boot
    /// block. Data and capabilities have separate tables and therefore
    /// separate refusal states: absence here must never be reported as a
    /// missing authority.
    MissingDatum {
        /// The [`kore_abi::task::role`] discriminant.
        role: u16,
        /// Instance index within the role.
        index: u16,
    },
}

/// Resolve a service's bundle from its bootstrap context and bind it — the
/// framework's construction entry (a real `main` calls this over its seeded
/// region; tests hand it a
/// [`BootstrapBuilder`](crate::bootstrap::BootstrapBuilder) block).
pub fn start<S: SystemService>(boot: &Bootstrap<'_>) -> Result<S, BindError> {
    S::Endowment::resolve(boot).map(S::bind)
}

/// The argument a staged grant carries — the derive's *verb-specific*
/// parameter, chosen by the granting service and executed by the kernel.
/// This is where an extent's window lives: the geometry is the **granting
/// side's internal choice**, and the client sees only the delivered token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrantArg {
    /// Move the exact held record, preserving its rights and lineage. An
    /// affine runtime delivery should use Serve::grant_returned for custody.
    Exact,
    /// The capability was delivered by the service's terminal kernel
    /// operation itself (currently `Found`) through this pair's already
    /// posted grant channel. The generic host publishes the correlated grant
    /// count but must not derive or send a second capability.
    ///
    /// A service may use this only when the operation and the result send are
    /// in the same atomic session. It is not a substitute for ordinary
    /// service grants.
    Direct,
    /// No argument (a plain attenuation).
    None,
    /// Derive a same-resource child into grant scratch before transferring it.
    ///
    /// This is the retained-ancestor form of a plain attenuation.  It is used
    /// when the serving task must keep the source as an independently
    /// revocable root (for example a provisioner retaining a `PciDevice`
    /// parent while handing a driver its own device child).
    Attenuate {
        /// The self-kind selected by the kernel's `(op x kind)` seam.
        kind: ResourceKind,
    },
    /// A window into the subject region (a memory grant's carve).
    Window {
        /// Byte offset into the subject.
        offset: u64,
        /// Byte length.
        len: u64,
        /// DMA addressing class the backing must satisfy — a *placement*
        /// property the granting lessor honored (`kore_abi::mem::AllocProps`);
        /// sealed onto the delivered token.  Reachability (an IOVA window)
        /// is the device-domain holder's separate act, never the grant's.
        class: DmaClass,
    },
    /// Carve a lessor-position sublease over a memory window.
    ///
    /// Unlike [`Window`](Self::Window), which yields a leaf view, this keeps
    /// `DERIVE | REVOKE` meaningful in the receiver so it can construct its
    /// own CPU and device mappings.  The source lease remains with the
    /// provisioner as the independently revocable ancestor.
    Sublease {
        /// Byte offset within the source lease.
        offset: u64,
        /// Byte length of the sublease.
        len: u64,
    },
    /// One entry of the subject **device inventory** (`res/DEVICE_SERVICE.md`
    /// §10) — [`Window`](Self::Window)'s analogue for hardware: a window
    /// names which slice of a memory subject a grant carves, this names
    /// which function of the machine a device broker hands over.
    ///
    /// The species rides the arg rather than being inferred, because the
    /// two device kinds grant disjoint authorities and an index alone does
    /// not say which one the claimant asked for.
    Device {
        /// The inventory index to hand out.
        index: u32,
        /// Which child kind the claimant asked for.  An index naming the
        /// other species refuses rather than coercing.
        species: DeviceSpecies,
    },
    /// A fresh **root supply lease** minted off the subject region (the
    /// lessor's `role::MEM_REGION` endowment) — [`Window`](Self::Window)'s
    /// sibling one authority tier up: a window carves a view out of a
    /// lease the service already holds, this mints a *new sibling lease*
    /// from the region itself (`Derive × PhysRegion → MemLease`, the
    /// [`kore_abi::session::LeaseReq`] wire — the same op the kernel
    /// founder allocates task memory with).  The delivered token is a
    /// root, so the client can carve, share, and convey it onward at a
    /// spawn; its frames return to the **region bitmap** when its lineage
    /// dies (T6), never to the granting service's books.
    ///
    /// The backing arrives zeroed by the kernel's lease-derive scrub
    /// invariant (fresh and reused frames alike) — zeroing is never a
    /// grant-time choice.
    Lease {
        /// Size in bytes (non-zero; the kernel refuses `LEASE_EXEC`
        /// without an X-mint, which no lessor holds).
        size: u64,
        /// Backing alignment as a log2 byte count: `0` for the 4 KiB
        /// default, else one of [`kore_abi::session::LEASE_ALIGN_4K`] /
        /// [`LEASE_ALIGN_2M`](kore_abi::session::LEASE_ALIGN_2M) /
        /// [`LEASE_ALIGN_1G`](kore_abi::session::LEASE_ALIGN_1G).
        align_log2: u8,
        /// [`kore_abi::session::LeaseReq::flags`] — the granting
        /// service's per-lease attribute choice (e.g.
        /// [`LEASE_SEALED`](kore_abi::session::LEASE_SEALED)); `0` for
        /// the plain supply shape.
        flags: u8,
        /// Kernel-authenticated lifecycle receipt for MM supply quota. `None`
        /// preserves the ordinary frozen 16-byte lease request.
        release: Option<SupplyReleaseGrant>,
    },
    /// A **sub-lease carve** off the subject lease — the lessor position
    /// itself for `[offset, offset+len)`, delivered whole (slice C: a
    /// minted loader face's ring client sub, or an image acquisition's fresh
    /// child beneath the persistent segment revocation pivot). Nested carves
    /// are owner-lineage-scoped in the kernel. [`Window`](Self::Window)'s sibling
    /// one custody tier up: a window delivers sight, this delivers the
    /// carve authority.
    Sub {
        /// Byte offset into the subject lease.
        offset: u64,
        /// Byte length.
        len: u64,
    },
    /// A fresh **recv end** derived off the subject channel root (slice
    /// C: a minted face's grant-bank lanes — one per channel, staged in
    /// the grant session so a dropped reply unwinds them with it, I3).
    /// The root stays the granting service's revocation pivot; retiring
    /// the face revokes it and T6 closes every granted recv wherever it
    /// traveled.
    ChanRecv,
    /// A fresh send end derived from a retained channel root.
    ChanSend,
}

/// MM-owned identity attached to a committed root-supply grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupplyReleaseGrant {
    pub receipt: u64,
    pub pair: u16,
}

/// One authority emission: derive an attenuation of the **held** capability
/// at `subject` (a slot of this service's own bundle) carrying `rights`,
/// with the verb-specific [`GrantArg`] — the staged-invocation artifact the
/// real dispatch commits (`InvocationWire`'s `subject`/`requested`/`arg`
/// shape).  The delivered child reaches the client as a **token** via
/// rendezvous delivery, never as reply data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grant {
    /// The held slot the grant attenuates.
    pub subject: SlotRef,
    /// The (sub)set of rights the child carries.
    pub rights: Rights,
    /// The derive's verb-specific argument.
    pub arg: GrantArg,
}

/// Grants one serve step may emit — bounded like a session
/// (`res/SYSCALL.md` D4): per-request authority output is finite by
/// construction, never a stream.
pub const MAX_GRANTS: usize = kore_abi::image::MAX_SEGMENTS + 1;

/// What one serve step produced: at most [`MAX_GRANTS`] grants + at most
/// one reply.  Fixed-capacity — **never allocates** (see the module docs).
#[derive(Debug)]
pub struct Outcome<R> {
    grants: [Option<Grant>; MAX_GRANTS],
    retained_grants: [Option<crate::cap::ReturnedCapability>; MAX_GRANTS],
    granted: usize,
    reply: Option<R>,
    host_action: Option<HostAction>,
}

/// Lifecycle work requested by the synchronous service join-point and
/// executed by the outer asynchronous host after the reply has been staged.
#[derive(Debug)]
pub(crate) enum HostAction {
    AdmitPeer {
        bundle: crate::svc::RecyclablePeerBundle,
        map: crate::svc::PeerMapPlan,
    },
    ClosePeer(PeerClose),
    RetirePeer {
        peer: crate::svc::PeerId,
        reason: PeerClose,
    },
}

impl<R> Outcome<R> {
    const fn empty() -> Self {
        Self {
            grants: [None; MAX_GRANTS],
            retained_grants: [const { None }; MAX_GRANTS],
            granted: 0,
            reply: None,
            host_action: None,
        }
    }

    /// Complete only after the whole grant transaction commits successfully.
    pub(crate) fn release_transferred_grants(&mut self) {
        for owner in &mut self.retained_grants {
            if let Some(owner) = owner.take() {
                owner.release_after_proven_move();
            }
        }
    }

    /// The emitted grants, in emission order.
    pub fn grants(&self) -> impl Iterator<Item = Grant> + '_ {
        self.grants[..self.granted].iter().flatten().copied()
    }

    /// How many grants were emitted.
    pub fn granted(&self) -> usize {
        self.granted
    }

    /// The reply, by reference.
    pub fn reply(&self) -> Option<&R> {
        self.reply.as_ref()
    }

    /// The reply, mutably — used by the serve host's grant-drop lifecycle
    /// hook before the response is lowered to the wire.
    pub(crate) fn reply_mut(&mut self) -> Option<&mut R> {
        self.reply.as_mut()
    }

    /// Take the reply.
    pub fn into_reply(self) -> Option<R> {
        self.reply
    }

    pub(crate) fn take_host_action(&mut self) -> Option<HostAction> {
        self.host_action.take()
    }
}

/// The serve-step mediator: one scoped authority input plus two lawful output
/// effects. A service may take the capability this same request returned,
/// attenuate held authority ([`grant`](Serve::grant)), and answer
/// ([`reply`](Serve::reply)); it cannot touch the transport or reach arbitrary
/// arena slots.
pub struct Serve<'o, R> {
    out: &'o mut Outcome<R>,
    returned: &'o mut Option<crate::cap::ReturnedCapability>,
    advertised_return: Option<SlotRef>,
    /// The **arrival pair's structural identity** (`res/USERLAND.md` §5.8;
    /// `res/DEVICE_SERVICE.md` §10's "stamp requests by endpoint identity"):
    /// which serving pair this request came off, stamped by the host —
    /// never a wire field a client could forge.  `0` for a single-pair
    /// service, the pair index for a multi-pair one.
    via: u16,
    /// Grant-send endpoints belonging to the arrival pair. These are stamped
    /// by the serving host, never selected by request data. Services use this
    /// only when a terminal kernel operation (for example `Found`) must
    /// publish an exact capability atomically into that pair's already-posted
    /// result lane.
    direct_grants: [Option<SlotRef>; MAX_GRANTS],
}

/// Scoped view of one capability delivered through a retained return leg.
///
/// Merely inspecting it leaves ownership with the serve host. The service
/// calls [`take`](Self::take) only at the point it has established a durable
/// lifecycle owner; otherwise the host performs retryable cleanup after the
/// serve step.
pub struct ReturnedInput<'a> {
    returned: &'a mut Option<crate::cap::ReturnedCapability>,
}

impl ReturnedInput<'_> {
    /// Inspect the delivered capability without taking cleanup ownership.
    pub fn capability(&self) -> Option<&crate::cap::ReturnedCapability> {
        self.returned.as_ref()
    }

    /// Move the capability into service-owned persistent state.
    pub fn take(&mut self) -> Option<crate::cap::ReturnedCapability> {
        self.returned.take()
    }
}

impl<R> Serve<'_, R> {
    /// The arrival pair's structural identity — see the field's doc.  A
    /// multi-client service keys per-caller policy (a claim ledger's owner
    /// column, a focus check) on this, never on request data.
    pub fn via(&self) -> u16 {
        self.via
    }

    /// The authenticated arrival pair's exact grant-send endpoint at `lane`.
    ///
    /// This is deliberately narrower than exposing the pair or host: the
    /// caller can name only a protocol-declared result lane, and the endpoint
    /// is useful solely as a terminal session destination. Ordinary derived
    /// grants continue through [`grant`](Self::grant).
    pub fn direct_grant_endpoint(&self, lane: usize) -> Option<SlotRef> {
        self.direct_grants.get(lane).copied().flatten()
    }

    /// Borrow the capability delivered through a retained return channel when
    /// this request names the founder-published correlation value for that
    /// pair.
    ///
    /// The record itself lives in a runtime-private slot. A mismatched wire
    /// value receives no authority; leaving the input untaken returns it to
    /// the host's bounded retryable-cleanup state.
    pub fn returned_input(&mut self, announced: SlotRef) -> Option<ReturnedInput<'_>> {
        if self.advertised_return != Some(announced) || self.returned.is_none() {
            return None;
        }
        Some(ReturnedInput {
            returned: self.returned,
        })
    }

    /// Emit an attenuation of the held capability at `subject`, with the
    /// derive's verb-specific `arg`.
    ///
    /// # Panics
    /// Past [`MAX_GRANTS`] — emitting unbounded authority from one step is
    /// a service *bug* (the D4 discipline), not a runtime condition.
    pub fn grant(&mut self, subject: SlotRef, rights: Rights, arg: GrantArg) {
        assert!(
            self.out.granted < MAX_GRANTS,
            "serve step exceeded MAX_GRANTS (D4: bounded emission)"
        );
        self.out.grants[self.out.granted] = Some(Grant {
            subject,
            rights,
            arg,
        });
        self.out.granted += 1;
    }

    /// Retain one affine delivery through the exact reply-grant transaction.
    /// The host settles its local reservation only after a proven move; a
    /// failed grant keeps the owner in the outcome's bounded cleanup path.
    pub fn grant_returned(
        &mut self,
        owner: crate::cap::ReturnedCapability,
    ) -> Result<(), crate::cap::ReturnedCapability> {
        let Some(subject) = owner.slot() else {
            return Err(owner);
        };
        if self.out.granted >= MAX_GRANTS {
            return Err(owner);
        }
        let index = self.out.granted;
        self.grant(subject, Rights::empty(), GrantArg::Exact);
        self.out.retained_grants[index] = Some(owner);
        Ok(())
    }

    /// Announce one capability already delivered atomically by the terminal
    /// service operation through the corresponding grant lane.
    pub fn direct_grant(&mut self) {
        self.grant(SlotRef::from_wire(0), Rights::empty(), GrantArg::Direct);
    }

    /// Set the reply — one per step (a second write debug-asserts, then
    /// overwrites).
    pub fn reply(&mut self, r: R) {
        debug_assert!(self.out.reply.is_none(), "serve step replied twice");
        self.out.reply = Some(r);
    }

    /// Hand one fully assembled affine peer bundle to the outer host for
    /// authenticated mapping and recyclable attachment.
    pub fn admit_peer(
        &mut self,
        bundle: crate::svc::RecyclablePeerBundle,
        map: crate::svc::PeerMapPlan,
    ) {
        debug_assert!(
            self.out.host_action.is_none(),
            "serve step requested two host actions"
        );
        self.out.host_action = Some(HostAction::AdmitPeer { bundle, map });
    }

    /// Retire the recyclable peer on which this request arrived. Static
    /// founder pairs cannot be retired through this path.
    pub fn close_peer(&mut self, reason: PeerClose) {
        debug_assert!(
            self.out.host_action.is_none(),
            "serve step requested two host actions"
        );
        self.out.host_action = Some(HostAction::ClosePeer(reason));
    }

    /// Ask the outer host to retire one exact recyclable generation selected
    /// by an authenticated administration service.
    ///
    /// The ticket is routing data, not authority: only the service code that
    /// received the request on its fixed admin pair may call this method, and
    /// [`ServeHost`](crate::svc::ServeHost) independently checks that the
    /// named generation is still the live occupant before mutation.
    pub fn retire_peer(&mut self, peer: crate::svc::PeerId, reason: PeerClose) {
        debug_assert!(
            self.out.host_action.is_none(),
            "serve step requested two host actions"
        );
        self.out.host_action = Some(HostAction::RetirePeer { peer, reason });
    }
}

/// Drive one request through a bound service — the framework's loop *body*.
/// The loop itself is [`ServeHost`](crate::svc::ServeHost)'s serve task on
/// the executor (`res/USERLAND.md` §5.5), which calls `step` per decoded
/// request.
pub fn step<S: SystemService>(svc: &mut S, req: S::Request) -> Outcome<S::Response> {
    step_with_return(svc, req, None, None, 0, [None; MAX_GRANTS]).0
}

/// Drive one request while joining a retained return-channel delivery to the
/// same serve-step scope, stamped with the arrival pair's identity (`via`).
/// Kept crate-visible: only [`ServeHost`] witnesses the
/// channel delivery and may construct this authority-bearing context.
pub(crate) fn step_with_return<S: SystemService>(
    svc: &mut S,
    req: S::Request,
    returned: Option<crate::cap::ReturnedCapability>,
    advertised_return: Option<SlotRef>,
    via: u16,
    direct_grants: [Option<SlotRef>; MAX_GRANTS],
) -> (Outcome<S::Response>, Option<crate::cap::ReturnedCapability>) {
    let mut out = Outcome::empty();
    let mut returned = returned;
    {
        let mut serve = Serve {
            out: &mut out,
            returned: &mut returned,
            advertised_return,
            via,
            direct_grants,
        };
        svc.serve(req, &mut serve);
    }
    (out, returned)
}

/// Executor-driven counterpart of [`step_with_return`]. The returned
/// capability and response/grant custody remain scoped to one host request;
/// only dependency progress may suspend.
pub(crate) async fn step_with_return_async<S: SystemService>(
    svc: &mut S,
    req: S::Request,
    returned: Option<crate::cap::ReturnedCapability>,
    advertised_return: Option<SlotRef>,
    via: u16,
    direct_grants: [Option<SlotRef>; MAX_GRANTS],
) -> (Outcome<S::Response>, Option<crate::cap::ReturnedCapability>) {
    let mut out = Outcome::empty();
    let mut returned = returned;
    {
        let mut serve = Serve {
            out: &mut out,
            returned: &mut returned,
            advertised_return,
            via,
            direct_grants,
        };
        svc.serve_async(req, &mut serve).await;
    }
    (out, returned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::task;

    /// An endowment-less test bundle — the anti-model in production (every
    /// real service holds something), but the machinery under test doesn't
    /// care.
    struct NoCaps;
    impl Bundle for NoCaps {
        const PROFILE: Profile = Profile::new("nocaps", &[]);
        fn resolve(_: &task::Bootstrap<'_>) -> Result<Self, BindError> {
            Ok(NoCaps)
        }
    }

    /// The wire half of a framework-only double.
    ///
    /// These services exist to exercise [`step`] and the grant accounting —
    /// never a ring.  They still declare a protocol
    /// because a host *is* a speaker of one
    /// ([`SvcProtocol`](crate::svc::SvcProtocol)'s module docs): the
    /// vocabularies live on the wire trait, so there is nowhere else for
    /// a service's `Request`/`Response` to come from.
    ///
    /// The wire form is one `u64` per direction — the smallest lawful
    /// [`RingRecord`](kore_ring::spsc::RingRecord) that carries these
    /// vocabularies — and the codecs **round-trip for real**, with the
    /// reply's grant count in the top byte.  A double whose `decode_rep`
    /// answered a blanket `None` would satisfy the signature while
    /// making the fail-closed contract vacuous: every reply would look
    /// like a protocol break, so the first test to drive one of these
    /// through a ring would fail for a reason that had nothing to do
    /// with what it was testing.
    macro_rules! trivial_protocol {
        ($ty:ty, $req:ty, $rep:ty, $enc:expr, $dec:expr, $renc:expr, $rdec:expr) => {
            impl crate::svc::SvcProtocol for $ty {
                type Request = $req;
                type Response = $rep;
                type ReqWire = u64;
                type RepWire = u64;
                const REPLY_GRANTS: usize = 0;

                fn encode_req(req: &Self::Request) -> u64 {
                    #[allow(clippy::redundant_closure_call)]
                    ($enc)(req)
                }
                fn decode_req(w: &u64) -> Option<Self::Request> {
                    #[allow(clippy::redundant_closure_call)]
                    ($dec)(*w)
                }
                fn encode_rep(rep: &Self::Response, grants: u8) -> u64 {
                    #[allow(clippy::redundant_closure_call)]
                    let body: u64 = ($renc)(rep);
                    body | (u64::from(grants) << 56)
                }
                fn decode_rep(w: &u64) -> Option<(Self::Response, u8)> {
                    #[allow(clippy::redundant_closure_call)]
                    let rep = ($rdec)(*w & 0x00ff_ffff_ffff_ffff)?;
                    Some((rep, (*w >> 56) as u8))
                }
            }
        };
    }

    /// Replies with the request + 1 and grants once from a fixed held slot.
    struct Echo;
    trivial_protocol!(
        Echo,
        u32,
        u32,
        |r: &u32| u64::from(*r),
        |w: u64| u32::try_from(w).ok(),
        |r: &u32| u64::from(*r),
        |w: u64| u32::try_from(w).ok()
    );
    impl SystemService for Echo {
        const NAME: &'static str = "echo";
        type Endowment = NoCaps;
        fn bind(_: NoCaps) -> Self {
            Echo
        }
        fn serve(&mut self, req: u32, sv: &mut Serve<'_, u32>) {
            sv.grant(SlotRef::from_wire(7), Rights::ACCESS, GrantArg::None);
            sv.reply(req + 1);
        }
    }

    #[test]
    fn step_collects_the_grant_and_the_reply() {
        let mut e = Echo;
        let out = step(&mut e, 41);
        assert_eq!(out.granted(), 1);
        assert_eq!(
            out.grants().next(),
            Some(Grant {
                subject: SlotRef::from_wire(7),
                rights: Rights::ACCESS,
                arg: GrantArg::None
            })
        );
        assert_eq!(out.into_reply(), Some(42));
    }

    /// A retained input stays with the host until the service explicitly
    /// adopts it, and the wire value is only the pair-correlation check.
    struct ReturnOwner {
        kept: Option<crate::cap::ReturnedCapability>,
    }

    trivial_protocol!(
        ReturnOwner,
        (SlotRef, bool),
        bool,
        |r: &(SlotRef, bool)| u64::from(r.0.index()) | (u64::from(r.1) << 32),
        |w: u64| Some((SlotRef::from_wire(w as u32), w >> 32 & 1 != 0)),
        |r: &bool| u64::from(*r),
        |w: u64| match w {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    );

    impl SystemService for ReturnOwner {
        const NAME: &'static str = "return-owner";
        const RETURN_MODE: ReturnMode = ReturnMode::Retained {
            kind: kore_abi::cap::ResourceKind::MemLease,
            minimum_rights: kore_abi::cap::Rights::empty(),
        };
        type Endowment = NoCaps;

        fn bind(_: NoCaps) -> Self {
            Self { kept: None }
        }

        fn serve(&mut self, (announced, adopt): Self::Request, sv: &mut Serve<'_, bool>) {
            if let Some(mut input) = sv.returned_input(announced)
                && adopt
            {
                self.kept = input.take();
            }
            sv.reply(self.kept.is_some());
        }
    }

    #[test]
    fn retained_return_requires_correlation_and_explicit_adoption() {
        let correlation = SlotRef::from_wire(7);
        let private = SlotRef::from_wire(42);

        let mut owner = ReturnOwner { kept: None };
        let (out, returned) = step_with_return(
            &mut owner,
            (SlotRef::from_wire(8), true),
            Some(crate::cap::ReturnedCapability::test_live(private)),
            Some(correlation),
            0,
            [None; MAX_GRANTS],
        );
        assert_eq!(out.into_reply(), Some(false));
        assert!(
            returned.is_some(),
            "mismatched correlation keeps host custody"
        );
        drop(returned);

        let (out, returned) = step_with_return(
            &mut owner,
            (correlation, false),
            Some(crate::cap::ReturnedCapability::test_live(private)),
            Some(correlation),
            0,
            [None; MAX_GRANTS],
        );
        assert_eq!(out.into_reply(), Some(false));
        assert!(returned.is_some(), "inspection alone keeps host custody");
        drop(returned);

        let (out, returned) = step_with_return(
            &mut owner,
            (correlation, true),
            Some(crate::cap::ReturnedCapability::test_live(private)),
            Some(correlation),
            0,
            [None; MAX_GRANTS],
        );
        assert_eq!(out.into_reply(), Some(true));
        assert!(
            returned.is_none(),
            "adoption moves custody into the service"
        );
        assert_eq!(
            owner.kept.as_ref().and_then(|cap| cap.slot()),
            Some(private)
        );
    }

    #[test]
    fn startup_failure_keeps_the_runtime_rung_and_bounded_detail() {
        assert_eq!(StartFailure::UNSPECIFIED.payload(4), 4);
        assert_eq!(StartFailure::new(7).payload(4), 0x704);
        assert_eq!(
            StartFailure::new(u64::MAX).payload(4),
            0xffff_ffff_ffff_ff04
        );
    }

    #[test]
    fn pci_identity_selector_separates_wildcard_from_exact_pairs() {
        assert_eq!(PciIdentitySelector::ANY.packed(), 0);
        assert_eq!(PciIdentitySelector::ANY.exact_pair(), None);
        assert!(PciIdentitySelector::ANY.matches(0x10de, 0x2684));

        let dg2 = PciIdentitySelector::exact(0x8086, 0x56a0);
        assert_eq!(dg2.packed(), 0x56a0_8086);
        assert_eq!(dg2.exact_pair(), Some((0x8086, 0x56a0)));
        assert!(dg2.matches(0x8086, 0x56a0));
        assert!(!dg2.matches(0x8086, 0x56a1));
        assert!(!dg2.matches(0x10de, 0x56a0));

        let intel = PciIdentitySelector::vendor(0x8086);
        assert_eq!(intel.packed(), 0xFFFF_8086);
        assert_eq!(intel.exact_pair(), None);
        assert_eq!(intel.vendor_only(), Some(0x8086));
        assert!(intel.matches(0x8086, 0x64a0));
        assert!(intel.matches(0x8086, 0xb080));
        assert!(!intel.matches(0x10de, 0x64a0));
        // Round trip through the wire word keeps the vendor scope.
        assert_eq!(PciIdentitySelector::from_packed(intel.packed()), intel);
        assert_eq!(dg2.vendor_only(), None);
        assert_eq!(PciIdentitySelector::ANY.vendor_only(), None);
    }

    #[test]
    #[should_panic(expected = "PCI identity wildcard")]
    fn zero_exact_pci_identity_cannot_degrade_into_a_wildcard() {
        let _ = PciIdentitySelector::exact(0, 0);
    }

    /// A service emitting past [`MAX_GRANTS`] is a bug, stopped at the
    /// boundary — never a silent drop (authority accounting must be exact).
    struct Greedy;
    trivial_protocol!(
        Greedy,
        (),
        (),
        |_: &()| 0u64,
        |w: u64| (w == 0).then_some(()),
        |_: &()| 0u64,
        |w: u64| (w == 0).then_some(())
    );
    impl SystemService for Greedy {
        const NAME: &'static str = "greedy";
        type Endowment = NoCaps;
        fn bind(_: NoCaps) -> Self {
            Greedy
        }
        fn serve(&mut self, _: (), sv: &mut Serve<'_, ()>) {
            for i in 0..=MAX_GRANTS {
                sv.grant(SlotRef::from_wire(i as u32), Rights::ACCESS, GrantArg::None);
            }
        }
    }

    #[test]
    #[should_panic(expected = "MAX_GRANTS")]
    fn unbounded_emission_is_stopped() {
        let _ = step(&mut Greedy, ());
    }

    // (The `start()`-over-a-real-endowment proofs moved to kore-svc-mm's
    // service tests with the crate split — the framework crate no longer
    // reaches a concrete service.)
}
