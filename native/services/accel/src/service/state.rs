//! The hosted engine — [`DeviceState`] object lifecycle over one
//! [`Accelerator`] backend.
//!
//! Everything protocol-1.0-semantic lives here: id resolution
//! (kind-tagged, generational — a stale id never resolves), per-context
//! child accounting, the retained-byte budget, the host-side validation
//! gates the contract assigns to hosts (`DeviceInfo::validate_*`,
//! [`BindingRef::validate_for_submit`] — Wire ABI §4.4), the
//! **acceptance boundary** (an indeterminate submit still admits the
//! event that owns the submission's resources), and the **two-phase
//! release** (`begin → provider → commit/restore`; an indeterminate
//! release invalidates the id and poisons the device).
//!
//! What deliberately does *not* live here: the wire, the staging window,
//! the signal channels, and `Serve` — those are the service face's
//! ([`super`]).  This split keeps the engine a pure function of
//! `(backend, requests)`, which is exactly the shape the conformance
//! suite gates.

use alloc::vec::Vec;
use kore_driver::accel::{NativeAccelerator, OrderedExecution, SyncRef};

use heapless::Vec as InlineVec;
#[allow(deprecated)]
use kore_driver::accel::Dispatch;
use kore_driver::accel::LaunchShape;
use virtio_accel_core::{
    AccessMode, ArtifactFormat, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferRange,
    BufferUsage, ByteSink, ByteSource, ContextDesc, ContextFlags, DeviceInfo, DeviceLimits,
    EventState, MemoryDomain, QueueDesc, QueueFlags, ReleaseFailure, SubmitFailure, TargetIdentity,
    Timeout,
};
use virtio_accel_device::{
    BufferCreateOutcome, CreateError, DeviceState, DeviceStateError, ObjectId, ObjectNamespace,
    ResourcePolicy,
};

use super::{AccelBindings, AccelEventState, AccelRefusal, AccelResponse, AccelSequenceEntry};

/// Serving pairs the session table is sized for — the runtime's fixed
/// ceiling. Recyclable arrivals retain their generation for ownership; only
/// session placement and completion counts decode the physical slot.
pub(crate) const PAIRS: usize = kore_rt::svc::MAX_SERVE_PAIRS;

/// One submitted-but-not-yet-destroyed event, beside its owner pair and
/// whether its terminal edge was already counted into a signal batch.
/// Bounded by the device's own event limits; the engine refuses
/// submissions past them before this list can grow.
struct LiveEvent {
    id: u64,
    via: u16,
    notified: bool,
}

/// One pair's funded/used retained-byte budgets.  Buffer backing and
/// program residency are funded and charged separately, mirroring the two
/// budgets [`ResourcePolicy`] bounds device-wide.
#[derive(Clone, Copy)]
struct PairFunding {
    buffer_funded: u64,
    program_funded: u64,
    buffer_used: u64,
    program_used: u64,
}

impl PairFunding {
    const EMPTY: Self = Self {
        buffer_funded: 0,
        program_funded: 0,
        buffer_used: 0,
        program_used: 0,
    };
}

/// The two wire dialects at the provider seam. `Legacy` never aliases
/// portable [`LaunchShape`]: its SIMD width survives intact until a backend
/// proves it against the compiled program or rejects it.
#[allow(deprecated)]
#[derive(Clone, Copy)]
enum SubmissionLaunch {
    Portable(LaunchShape),
    Legacy(Dispatch),
}

/// The engine: one backend, its validated metadata, the object tables,
/// and the two service-side ledgers the tables do not carry — which pair
/// owns each context, and which events still owe a completion notice.
pub(crate) struct Engine<A: NativeAccelerator> {
    info: DeviceInfo,
    // Provider handles die before the backend instance that created them.
    // This order is load-bearing for discard-only epochs: Rust drops fields
    // in declaration order, and no handle may outlive its provider owner.
    state: DeviceState<A::Context, A::Buffer, A::Program, A::Queue, A::Event>,
    backend: A,
    /// Context ownership: `(context id, arrival pair)` — the
    /// `Serve::via` stamp, checked on every op that touches the context
    /// or anything inside it.  Bounded by `limits.max_contexts`.
    owners: Vec<(u64, u16)>,
    /// Events awaiting their one terminal notice.  Bounded by the
    /// aggregate event limit.
    events: Vec<LiveEvent>,
    /// Each pair's funded byte economy — the third service-side ledger the
    /// object tables do not carry.  [`DeviceState`]'s policy is the device's
    /// hard ceiling for everyone together; a pair spends only what its own
    /// `ProvisionMemory` supplies funded, its releases credit that budget
    /// back, and the whole entry dies with the pair (`retire_peer`), so a
    /// relaunched client provisions afresh instead of being refused.
    funding: [PairFunding; PAIRS],
    /// The device epoch is discard-only (an indeterminate release, a backend
    /// `DeviceLost`, or an unpublished allocation that could not be cleaned
    /// up). The service immediately moves this complete graph into its affine
    /// discard handoff; this namespace is never made live again. A contained
    /// device may host a fresh epoch under a new namespace
    /// ([`AccelService::refound`](super::AccelService::refound)).
    lost: bool,
    /// This epoch's id namespace: every object id it issues carries it, so
    /// no id outlives its epoch.
    namespace: u16,
}

impl<A: NativeAccelerator> Engine<A> {
    /// Retire only this arrival's graph, children before parents. A refusal
    /// leaves remaining owners in their existing tables for the next close pass.
    pub(crate) fn retire_peer_using<F>(
        &mut self,
        via: u16,
        mut release: F,
    ) -> Result<(), AccelRefusal>
    where
        F: FnMut(&A, A::Buffer) -> Result<(), virtio_accel_core::ReleaseFailure<A::Buffer>>,
    {
        self.live()?;
        while let Some(ctx) = self
            .owners
            .iter()
            .find(|(_, owner)| *owner == via)
            .map(|(ctx, _)| *ctx)
        {
            loop {
                let next = self.state.event_ids().find(|id| {
                    self.state
                        .event_record(*id)
                        .is_ok_and(|r| r.context_id().get() == ctx)
                });
                let Some(id) = next else {
                    break;
                };
                if matches!(
                    self.poll_event(via, id.get())?,
                    AccelResponse::Event {
                        state: AccelEventState::Pending,
                        ..
                    }
                ) {
                    match self.cancel_event(via, id.get()) {
                        Ok(_) | Err(AccelRefusal::Busy) => {}
                        Err(error) => return Err(error),
                    }
                }
                self.destroy_event(via, id.get())?;
            }
            loop {
                let next = self.state.queue_ids().find(|id| {
                    self.state
                        .queue_record(*id)
                        .is_ok_and(|r| r.context_id().get() == ctx)
                });
                let Some(id) = next else {
                    break;
                };
                self.destroy_queue(via, id.get())?;
            }
            loop {
                let next = self.state.program_ids().find(|id| {
                    self.state
                        .program_record(*id)
                        .is_ok_and(|r| r.context_id().get() == ctx)
                });
                let Some(id) = next else {
                    break;
                };
                self.unload_program(via, id.get())?;
            }
            loop {
                let next = self.state.buffer_ids().find(|id| {
                    self.state
                        .buffer_record(*id)
                        .is_ok_and(|r| r.context_id().get() == ctx)
                });
                let Some(id) = next else {
                    break;
                };
                self.free_buffer_using(via, id.get(), &mut release)?;
            }
            self.destroy_context(via, ctx)?;
        }
        // The pair's economy dies with its graph: a later generation in this
        // slot funds itself afresh instead of inheriting or being refused.
        if let Some(entry) = self.funding.get_mut(super::pair_index(via)) {
            *entry = PairFunding::EMPTY;
        }
        Ok(())
    }

    /// The hosted backend, mutably — the row's device-plane door (see
    /// `AccelService::backend_mut`).
    pub(crate) fn backend_mut(&mut self) -> &mut A {
        &mut self.backend
    }

    /// The hosted backend, shared (see `AccelService::backend`).
    pub(crate) fn backend(&self) -> &A {
        &self.backend
    }

    /// Host `backend` behind validated metadata — `None` when the
    /// backend's own identity refuses validation (the degraded-founding
    /// posture: the row still serves, every op answering `NotReady`).
    ///
    /// `namespace` is this device epoch's nonzero id-namespace;
    /// `max_buffer_backing` / `max_program_resident` are the service's
    /// retained-byte budgets (what all clients together may keep
    /// resident), clamped by the engine on every create.
    ///
    /// A refusal hands the backend back, so a refounding owner keeps custody
    /// of a device it could not host.
    pub(crate) fn host(
        backend: A,
        namespace: u16,
        max_buffer_backing: u64,
        max_program_resident: u64,
    ) -> Result<Self, A> {
        let Some((info, state)) = (|| {
            let info = backend.device_info().ok()?;
            info.validate().ok()?;
            let namespace = ObjectNamespace::new(namespace)?;
            let policy = ResourcePolicy::new(max_buffer_backing, max_program_resident)?;
            let state =
                DeviceState::new(namespace, event_id_lifetime(info.limits), policy).ok()?;
            Some((info, state))
        })() else {
            return Err(backend);
        };
        Ok(Self {
            info,
            state,
            backend,
            owners: Vec::new(),
            events: Vec::new(),
            funding: [PairFunding::EMPTY; PAIRS],
            lost: false,
            namespace,
        })
    }

    /// This epoch's id namespace.
    pub(crate) fn namespace(&self) -> u16 {
        self.namespace
    }

    /// Admit one pair's funded byte-budget increments — any time, any pair,
    /// repeatedly. The epoch policy is the immutable device-wide ceiling: the
    /// sum of every live pair's funding must stay inside it, so an honest
    /// `ResourceLimit` lands at provision time, not at some later create.
    /// Bookkeeping is prepared before the device owner publishes new capacity.
    pub(crate) fn provision(
        &mut self,
        via: u16,
        buffer_bytes: u64,
        program_bytes: u64,
        publish: impl FnOnce(u64) -> Result<(), AccelRefusal>,
    ) -> Result<(), AccelRefusal> {
        self.live()?;
        if buffer_bytes
            .checked_add(program_bytes)
            .is_none_or(|bytes| bytes == 0)
        {
            return Err(AccelRefusal::InvalidArgument);
        }
        let slot = super::pair_index(via);
        let entry = self.funding.get(slot).ok_or(AccelRefusal::Denied)?;
        let buffer_funded = entry
            .buffer_funded
            .checked_add(buffer_bytes)
            .ok_or(AccelRefusal::ResourceLimit)?;
        let program_funded = entry
            .program_funded
            .checked_add(program_bytes)
            .ok_or(AccelRefusal::ResourceLimit)?;
        let policy = self.state.resource_policy();
        let ceiling = |total: Option<u64>, max: u64| {
            total
                .filter(|&total| total <= max)
                .ok_or(AccelRefusal::ResourceLimit)
        };
        let total_buffer = ceiling(
            self.funding
                .iter()
                .try_fold(buffer_bytes, |sum, f| sum.checked_add(f.buffer_funded)),
            policy.max_buffer_backing_bytes(),
        )?;
        let total_program = ceiling(
            self.funding
                .iter()
                .try_fold(program_bytes, |sum, f| sum.checked_add(f.program_funded)),
            policy.max_program_resident_bytes(),
        )?;
        let required_capacity = total_buffer
            .checked_add(total_program)
            .ok_or(AccelRefusal::ResourceLimit)?;
        if let Err(error) = publish(required_capacity) {
            if error == AccelRefusal::DeviceLost {
                self.lost = true;
            }
            return Err(error);
        }
        let entry = &mut self.funding[slot];
        entry.buffer_funded = buffer_funded;
        entry.program_funded = program_funded;
        Ok(())
    }

    /// Refuse a create whose bytes exceed the pair's remaining funded budget.
    /// `select` picks the `(used, funded)` axis; the check runs before the
    /// provider allocates, so an over-budget request costs the device nothing.
    /// A pair that never funded this axis spends the commons instead —
    /// bounded, as ever, by [`DeviceState`]'s aggregate policy ceiling — so
    /// rows without a provisioning verb (and their existing clients) keep
    /// their pre-funding behaviour.
    fn charge(
        &mut self,
        via: u16,
        bytes: u64,
        select: impl Fn(&mut PairFunding) -> (&mut u64, u64),
    ) -> Result<(), AccelRefusal> {
        let entry = self
            .funding
            .get_mut(super::pair_index(via))
            .ok_or(AccelRefusal::Denied)?;
        let (used, funded) = select(entry);
        if funded == 0 {
            return Ok(());
        }
        let charged = used
            .checked_add(bytes)
            .filter(|&charged| charged <= funded)
            .ok_or(AccelRefusal::ResourceLimit)?;
        *used = charged;
        Ok(())
    }

    /// Return released bytes to the pair's budget.  Saturating: an epoch that
    /// lost exact accounting (a lost device discards the ledger with the
    /// graph) must never underflow into phantom budget.
    fn credit(&mut self, via: u16, bytes: u64, select: impl Fn(&mut PairFunding) -> &mut u64) {
        if let Some(entry) = self.funding.get_mut(super::pair_index(via)) {
            let used = select(entry);
            *used = used.saturating_sub(bytes);
        }
    }

    /// The validated, immutable device metadata (identity answers even
    /// on a lost device — the facts did not change, the silicon did).
    pub(crate) fn info(&self) -> DeviceInfo {
        self.info
    }

    /// Whether a provider outcome made this complete object namespace
    /// discard-only.  The service face consumes this edge and seals the
    /// engine away from every later client or device-plane call.
    pub(crate) fn is_lost(&self) -> bool {
        self.lost
    }

    fn live(&self) -> Result<(), AccelRefusal> {
        if self.lost {
            Err(AccelRefusal::DeviceLost)
        } else {
            Ok(())
        }
    }

    /// Observe a provider-wide asynchronous fault before crossing another
    /// resource boundary. Device loss poisons the complete object namespace;
    /// transient probe errors are returned without weakening that rule.
    pub(crate) fn poll_health(&mut self) -> Result<(), AccelRefusal> {
        self.live()?;
        let outcome = self.backend.poll_health();
        outcome.map_err(|error| self.backend_err(error))
    }

    /// A backend error, folded to the wire vocabulary — and a reported
    /// device loss poisons the engine.
    fn backend_err(&mut self, error: BackendError) -> AccelRefusal {
        let refusal = refuse_backend(error);
        if refusal == AccelRefusal::DeviceLost {
            self.lost = true;
        }
        refusal
    }

    fn obj(raw: u64) -> Result<ObjectId, AccelRefusal> {
        ObjectId::from_raw(raw).ok_or(AccelRefusal::BadObject)
    }

    /// The `Serve::via` ownership gate: the context (and so everything
    /// inside it) answers only to the pair that created it.
    fn owned(&self, context: ObjectId, via: u16) -> Result<(), AccelRefusal> {
        match self.owners.iter().find(|(id, _)| *id == context.get()) {
            Some((_, owner)) if *owner == via => Ok(()),
            Some(_) => Err(AccelRefusal::Denied),
            // A resolvable context with no owner row is an engine
            // invariant break, not a client mistake.
            None => Err(AccelRefusal::Internal),
        }
    }

    fn context_of_buffer(&self, id: ObjectId) -> Result<ObjectId, AccelRefusal> {
        Ok(self
            .state
            .buffer_record(id)
            .map_err(refuse_state)?
            .context_id())
    }

    /// Resolve a client-named context id against the table **before**
    /// the ownership gate — an unknown, stale, or cross-kind id is the
    /// client's [`AccelRefusal::BadObject`], never an engine fault.
    fn context_checked(&self, raw: u64) -> Result<ObjectId, AccelRefusal> {
        let id = Self::obj(raw)?;
        self.state.context_record(id).map_err(refuse_state)?;
        Ok(id)
    }

    /// A profiling reset cannot straddle any admitted event, even one whose
    /// terminal result has been observed but whose ownership is not retired.
    pub(crate) fn configure_profiling(
        &mut self,
        via: u16,
        ctx: u64,
        enabled: bool,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let id = self.context_checked(ctx)?;
        self.owned(id, via)?;
        let record = self.state.context_record(id).map_err(refuse_state)?;
        if record.children().events != 0 {
            return Err(AccelRefusal::Busy);
        }
        let context = record.resource().map_err(refuse_state)?;
        self.backend
            .configure_profiling(context, enabled)
            .map_err(|error| self.backend_err(error))?;
        Ok(AccelResponse::ProfilingConfigured { ctx, enabled })
    }

    // ── creates ─────────────────────────────────────────────────────

    pub(crate) fn create_context(
        &mut self,
        via: u16,
        flags: u32,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let flags = ContextFlags::from_bits(flags).ok_or(AccelRefusal::InvalidArgument)?;
        let desc = ContextDesc { flags };
        self.info
            .validate_context_desc(desc)
            .map_err(|e| self.backend_err(e))?;
        // `owners.push` is part of admitting the provider-owned context.
        // Reserve its ledger slot first so a successful provider call cannot
        // be followed by an infallible allocation/panic or an untracked id.
        reserve_ledger_entries(&mut self.owners, 1)?;
        let Self { state, backend, .. } = self;
        let id = state
            .create_context_with(|| backend.create_context(desc))
            .map_err(|e| self.refuse_create(e))?;
        self.owners.push((id.get(), via));
        Ok(AccelResponse::ContextMade { ctx: id.get() })
    }

    pub(crate) fn create_buffer(
        &mut self,
        via: u16,
        ctx: u64,
        bytes: u64,
        align: u64,
        domain: u8,
        usage: u32,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.create_buffer_using(
            via,
            ctx,
            bytes,
            align,
            domain,
            usage,
            |backend, context, desc| backend.allocate_buffer(context, desc),
        )
    }

    pub(crate) fn create_buffer_using<F>(
        &mut self,
        via: u16,
        ctx: u64,
        bytes: u64,
        align: u64,
        domain: u8,
        usage: u32,
        create: F,
    ) -> Result<AccelResponse, AccelRefusal>
    where
        F: FnOnce(
            &A,
            &A::Context,
            BufferDesc,
        ) -> Result<virtio_accel_core::AllocatedBuffer<A::Buffer>, BackendError>,
    {
        self.live()?;
        let ctx = self.context_checked(ctx)?;
        self.owned(ctx, via)?;
        let domain = MemoryDomain::try_from(domain).map_err(refuse_backend)?;
        let usage = BufferUsage::from_bits(usage).ok_or(AccelRefusal::InvalidArgument)?;
        let desc = BufferDesc::new(bytes, align, domain, usage).map_err(refuse_backend)?;
        self.info
            .validate_buffer_desc(desc)
            .map_err(|e| self.backend_err(e))?;
        // Funding first: an unfunded request must cost the provider nothing.
        // Every non-admitted path below credits this charge straight back.
        self.charge(via, desc.bytes(), |f| (&mut f.buffer_used, f.buffer_funded))?;
        let Self { state, backend, .. } = self;
        let outcome = state
            .create_buffer_with(ctx, desc, |context, desc| {
                // Return the owned handle and its facts together. DeviceState
                // performs the honesty/budget gate *after* it has installed a
                // cleanup owner; rejecting metadata here would drop a newly
                // allocated provider handle without calling `free_buffer`.
                Ok(create(backend, context, desc)?.into_parts())
            })
            .map_err(|e| self.refuse_create(e));
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                self.credit(via, desc.bytes(), |f| &mut f.buffer_used);
                return Err(error);
            }
        };
        match outcome {
            BufferCreateOutcome::Admitted(id) => Ok(AccelResponse::BufferMade { buffer: id.get() }),
            // Allocated but over budget / off-descriptor: release it back
            // through the provider boundary, then refuse.  The id was
            // never exposed.
            BufferCreateOutcome::CleanupRequired { id, error } => {
                self.credit(via, desc.bytes(), |f| &mut f.buffer_used);
                if self.release_buffer(id).is_err() {
                    // A rejected cleanup restored a live provider handle in
                    // the table, but its ID is deliberately unpublished. No
                    // client can own or retry that release. Continuing would
                    // leak retained budget and a context child forever, so
                    // quarantine the complete epoch for owner-ordered
                    // quiesce/discard instead.
                    self.lost = true;
                    return Err(AccelRefusal::DeviceLost);
                }
                Err(self.backend_err(error))
            }
        }
    }

    pub(crate) fn load_program(
        &mut self,
        via: u16,
        ctx: u64,
        artifact: &[u8],
        resident_bytes: u64,
        format: u32,
        target: [u32; 12],
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let ctx = self.context_checked(ctx)?;
        self.owned(ctx, via)?;
        let format = ArtifactFormat::new(format).ok_or(AccelRefusal::InvalidArgument)?;
        let payload = SliceSource(artifact);
        let artifact_ref = ArtifactRef {
            format,
            target: TargetIdentity(target),
            payload: &payload,
            resident_bytes,
        };
        self.charge(via, resident_bytes, |f| {
            (&mut f.program_used, f.program_funded)
        })?;
        let Self { state, backend, .. } = self;
        let id = state
            .create_program_with(ctx, artifact.len() as u64, resident_bytes, |context| {
                backend.load_program(context, artifact_ref)
            })
            .map_err(|e| self.refuse_create(e));
        match id {
            Ok(id) => Ok(AccelResponse::ProgramMade { program: id.get() }),
            Err(error) => {
                self.credit(via, resident_bytes, |f| &mut f.program_used);
                Err(error)
            }
        }
    }

    pub(crate) fn create_queue(
        &mut self,
        via: u16,
        ctx: u64,
        flags: u32,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let ctx = self.context_checked(ctx)?;
        self.owned(ctx, via)?;
        let flags = QueueFlags::from_bits(flags).ok_or(AccelRefusal::InvalidArgument)?;
        let desc = QueueDesc { flags };
        self.info
            .validate_queue_desc(desc)
            .map_err(|e| self.backend_err(e))?;
        let Self { state, backend, .. } = self;
        let id = state
            .create_queue_with(ctx, |context| backend.create_queue(context, desc))
            .map_err(|e| self.refuse_create(e))?;
        Ok(AccelResponse::QueueMade { queue: id.get() })
    }

    // ── transfers ───────────────────────────────────────────────────

    pub(crate) fn write_buffer(
        &mut self,
        via: u16,
        buffer: u64,
        offset: u64,
        data: &[u8],
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let id = Self::obj(buffer)?;
        self.owned(self.context_of_buffer(id)?, via)?;
        let record = self.state.buffer_record(id).map_err(refuse_state)?;
        let in_flight = record.in_flight();
        let desc = record.info().desc();
        if !desc.usage.contains(BufferUsage::TRANSFER_DESTINATION) {
            return Err(AccelRefusal::PermissionDenied);
        }
        let range = BufferRange::new(offset, data.len() as u64).map_err(refuse_backend)?;
        if range.end() > desc.bytes() {
            return Err(AccelRefusal::OutOfBounds);
        }
        if in_flight != 0 {
            return Err(AccelRefusal::Busy);
        }
        let Self { state, backend, .. } = self;
        let target = state
            .buffer_record_mut(id)
            .map_err(refuse_state)?
            .resource_mut()
            .map_err(refuse_state)?;
        backend
            .write_buffer(target, offset, &SliceSource(data))
            .map_err(|e| self.backend_err(e))?;
        Ok(AccelResponse::Wrote { buffer })
    }

    pub(crate) fn read_buffer(
        &mut self,
        via: u16,
        buffer: u64,
        offset: u64,
        out: &mut [u8],
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let id = Self::obj(buffer)?;
        self.owned(self.context_of_buffer(id)?, via)?;
        let record = self.state.buffer_record(id).map_err(refuse_state)?;
        let in_flight = record.in_flight();
        let desc = record.info().desc();
        if !desc.usage.contains(BufferUsage::TRANSFER_SOURCE) {
            return Err(AccelRefusal::PermissionDenied);
        }
        let range = BufferRange::new(offset, out.len() as u64).map_err(refuse_backend)?;
        if range.end() > desc.bytes() {
            return Err(AccelRefusal::OutOfBounds);
        }
        if in_flight != 0 {
            return Err(AccelRefusal::Busy);
        }
        let Self { state, backend, .. } = self;
        let source = state
            .buffer_record(id)
            .map_err(refuse_state)?
            .resource()
            .map_err(refuse_state)?;
        backend
            .read_buffer(source, offset, &mut SliceSink(out))
            .map_err(|e| self.backend_err(e))?;
        Ok(AccelResponse::Read { buffer })
    }

    /// The provider's firmware log tail into `out` — no object, no context:
    /// the log is device-wide and read-only, so any live pair may read it.
    pub(crate) fn read_firmware_log(
        &mut self,
        out: &mut [u8],
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let bytes = self.backend.firmware_log(out) as u64;
        Ok(AccelResponse::Log { bytes })
    }

    // ── submission ──────────────────────────────────────────────────

    pub(crate) fn submit(
        &mut self,
        via: u16,
        queue: u64,
        program: u64,
        timeout_ns: u64,
        bindings: &AccelBindings,
        launch: LaunchShape,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.submit_inner(
            via,
            queue,
            program,
            timeout_ns,
            bindings,
            SubmissionLaunch::Portable(launch),
        )
    }

    /// Route one deployed non-scalar OP13 request without reinterpreting it as
    /// portable workgroups.
    #[allow(deprecated)]
    pub(crate) fn submit_legacy(
        &mut self,
        via: u16,
        queue: u64,
        program: u64,
        timeout_ns: u64,
        bindings: &AccelBindings,
        dispatch: Dispatch,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.submit_inner(
            via,
            queue,
            program,
            timeout_ns,
            bindings,
            SubmissionLaunch::Legacy(dispatch),
        )
    }

    fn submit_inner(
        &mut self,
        via: u16,
        queue: u64,
        program: u64,
        timeout_ns: u64,
        bindings: &AccelBindings,
        launch: SubmissionLaunch,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let queue_id = Self::obj(queue)?;
        let program_id = Self::obj(program)?;
        let context = self
            .state
            .queue_record(queue_id)
            .map_err(refuse_state)?
            .context_id();
        self.owned(context, via)?;
        if let SubmissionLaunch::Portable(portable) = launch {
            self.backend
                .validate_launch_shape(portable)
                .map_err(refuse_backend)?;
        }

        // The wire bound is 6; the device may bind fewer.
        let max_bindings = self
            .info
            .limits
            .max_bindings_per_submission
            .min(super::MAX_ACCEL_BINDINGS as u32);

        // Resolve every binding's descriptor and range up front — the
        // per-binding facts `validate_for_submit` and the bounds gate
        // need, in wire order.
        let mut descs = InlineVec::<BufferDesc, { super::MAX_ACCEL_BINDINGS }>::new();
        let mut buffer_ids = Vec::new();
        if buffer_ids.try_reserve(bindings.len()).is_err() {
            return Err(AccelRefusal::OutOfMemory);
        }
        for binding in bindings.iter() {
            let id = Self::obj(binding.buffer)?;
            let desc = self
                .state
                .buffer_record(id)
                .map_err(refuse_state)?
                .info()
                .desc();
            let range = BufferRange::new(binding.offset, binding.len).map_err(refuse_backend)?;
            if range.end() > desc.bytes() {
                return Err(AccelRefusal::OutOfBounds);
            }
            descs.push(desc).map_err(|_| AccelRefusal::ResourceLimit)?;
            buffer_ids.push(id);
        }
        let timeout = Timeout::from_wire_ns(timeout_ns);

        // As with context ownership, the completion ledger is part of event
        // admission.  Once the provider returns an event (including an
        // indeterminate submit), recording it must be allocation-free.
        reserve_ledger_entries(&mut self.events, 1)?;
        let mut indeterminate = None;
        let Self { state, backend, .. } = self;
        let admitted = state.create_event_with(queue_id, program_id, buffer_ids, |resources| {
            // Rebuild the borrowed binding slice over the resolved
            // handles, still in wire order (`buffer_by_id` preserves
            // identity, so `descs[i]` stays the descriptor of
            // `refs[i]`'s buffer — the pairing `validate_for_submit`
            // requires).
            let mut refs =
                InlineVec::<BindingRef<'_, A::Buffer>, { super::MAX_ACCEL_BINDINGS }>::new();
            for binding in bindings.iter() {
                let id = ObjectId::from_raw(binding.buffer).ok_or(BackendError::InvalidArgument)?;
                let buffer = resources
                    .buffer_by_id(id)
                    .map_err(|_| BackendError::InvalidArgument)?;
                refs.push(BindingRef {
                    slot: binding.slot,
                    buffer,
                    range: BufferRange::new(binding.offset, binding.len)?,
                    access: binding.access,
                })
                .map_err(|_| BackendError::ResourceLimit)?;
            }
            // The host-side Wire ABI §4.4 gate: reject an access mode
            // the buffer's declared usage does not permit, before the
            // provider is invoked.
            BindingRef::validate_for_submit(refs.as_slice(), descs.as_slice(), max_bindings)?;
            #[allow(deprecated)]
            let submitted = match launch {
                SubmissionLaunch::Portable(portable) => backend.submit_with_launch(
                    resources.queue(),
                    resources.program(),
                    refs.as_slice(),
                    portable,
                    timeout,
                ),
                SubmissionLaunch::Legacy(dispatch) => backend.submit_dispatch(
                    resources.queue(),
                    resources.program(),
                    refs.as_slice(),
                    dispatch,
                    timeout,
                ),
            };
            match submitted {
                Ok(event) => Ok(event),
                Err(SubmitFailure::Rejected(error)) => Err(error),
                // The acceptance boundary: the event exists and owns the
                // submission's resources until it turns terminal — admit
                // it, and carry the error beside the id.
                Err(SubmitFailure::Indeterminate { error, event }) => {
                    indeterminate = Some(error);
                    Ok(event)
                }
            }
        });
        let event_id = match admitted {
            Ok(id) => id,
            Err(e) => return Err(self.refuse_create(e)),
        };
        self.events.push(LiveEvent {
            id: event_id.get(),
            via,
            notified: false,
        });
        Ok(match indeterminate {
            None => AccelResponse::Submitted {
                event: event_id.get(),
            },
            Some(error) => AccelResponse::SubmitIndeterminate {
                event: event_id.get(),
                refusal: self.backend_err(error),
            },
        })
    }

    /// Admit a dependency-ordered program table as one provider event. Each
    /// invocation is validated under the ordinary binding contract; the event
    /// retains every repeated program/buffer reference until retirement.
    pub(crate) fn submit_sequence(
        &mut self,
        via: u16,
        queue: u64,
        entries: &[AccelSequenceEntry],
        timeout_ns: u64,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        if entries.is_empty() || entries.len() > kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES {
            return Err(AccelRefusal::InvalidArgument);
        }
        let queue_id = Self::obj(queue)?;
        let context = self
            .state
            .queue_record(queue_id)
            .map_err(refuse_state)?
            .context_id();
        self.owned(context, via)?;
        let max_bindings = self
            .info
            .limits
            .max_bindings_per_submission
            .min(super::MAX_ACCEL_BINDINGS as u32);

        let mut program_ids = Vec::new();
        program_ids
            .try_reserve_exact(entries.len())
            .map_err(|_| AccelRefusal::OutOfMemory)?;
        // Every binding, and each member's gate and signal word: the event
        // retains their buffers until it retires.
        let mut buffer_ids = Vec::new();
        buffer_ids
            .try_reserve_exact(entries.len().saturating_mul(super::MAX_ACCEL_BINDINGS + 2))
            .map_err(|_| AccelRefusal::OutOfMemory)?;
        let mut descs = Vec::new();
        descs
            .try_reserve_exact(entries.len())
            .map_err(|_| AccelRefusal::OutOfMemory)?;
        for entry in entries {
            // Each member's geometry is the single-launch admission gate:
            // a backend without explicit workgroups refuses the whole table
            // here, before any object lookup or event reservation.
            self.backend
                .validate_launch_shape(entry.launch)
                .map_err(refuse_backend)?;
            let program_id = Self::obj(entry.program)?;
            program_ids.push(program_id);
            let mut invocation_descs =
                InlineVec::<BufferDesc, { super::MAX_ACCEL_BINDINGS }>::new();
            for binding in entry.bindings.iter() {
                let id = Self::obj(binding.buffer)?;
                let desc = self
                    .state
                    .buffer_record(id)
                    .map_err(refuse_state)?
                    .info()
                    .desc();
                let range =
                    BufferRange::new(binding.offset, binding.len).map_err(refuse_backend)?;
                if range.end() > desc.bytes() {
                    return Err(AccelRefusal::OutOfBounds);
                }
                invocation_descs
                    .push(desc)
                    .map_err(|_| AccelRefusal::ResourceLimit)?;
                buffer_ids.push(id);
            }
            descs.push(invocation_descs);
            // A gate's or signal's word: a quadword of a buffer of this
            // context, inside it and 8-aligned (`AccelSequenceSyncWire`).
            for sync in [entry.gate, entry.signal].into_iter().flatten() {
                let id = Self::obj(sync.buffer)?;
                let desc = self
                    .state
                    .buffer_record(id)
                    .map_err(refuse_state)?
                    .info()
                    .desc();
                if !sync.offset.is_multiple_of(8) {
                    return Err(AccelRefusal::InvalidArgument);
                }
                if sync.offset.checked_add(8).is_none_or(|end| end > desc.bytes()) {
                    return Err(AccelRefusal::OutOfBounds);
                }
                buffer_ids.push(id);
            }
        }

        reserve_ledger_entries(&mut self.events, 1)?;
        let timeout = Timeout::from_wire_ns(timeout_ns);
        let mut indeterminate = None;
        let Self { state, backend, .. } = self;
        let admitted =
            state.create_event_sequence_with(queue_id, program_ids, buffer_ids, |resources| {
                let mut invocation_refs = Vec::new();
                invocation_refs
                    .try_reserve_exact(entries.len())
                    .map_err(|_| BackendError::OutOfMemory)?;
                for (entry, invocation_descs) in entries.iter().zip(descs.iter()) {
                    let mut refs =
                        InlineVec::<BindingRef<'_, A::Buffer>, { super::MAX_ACCEL_BINDINGS }>::new(
                        );
                    for binding in entry.bindings.iter() {
                        let id = ObjectId::from_raw(binding.buffer)
                            .ok_or(BackendError::InvalidArgument)?;
                        let buffer = resources
                            .buffer_by_id(id)
                            .map_err(|_| BackendError::InvalidArgument)?;
                        refs.push(BindingRef {
                            slot: binding.slot,
                            buffer,
                            range: BufferRange::new(binding.offset, binding.len)?,
                            access: binding.access,
                        })
                        .map_err(|_| BackendError::ResourceLimit)?;
                    }
                    BindingRef::validate_for_submit(
                        refs.as_slice(),
                        invocation_descs.as_slice(),
                        max_bindings,
                    )?;
                    invocation_refs.push(refs);
                }
                let mut executions = Vec::new();
                executions
                    .try_reserve_exact(entries.len())
                    .map_err(|_| BackendError::OutOfMemory)?;
                for (index, (refs, entry)) in invocation_refs.iter().zip(entries).enumerate() {
                    let program = resources
                        .program(index)
                        .map_err(|_| BackendError::InvalidArgument)?;
                    let word = |sync: Option<super::AccelSync>| {
                        sync.map(|sync| {
                            let id = ObjectId::from_raw(sync.buffer).ok_or(BackendError::InvalidArgument)?;
                            let buffer = resources
                                .buffer_by_id(id)
                                .map_err(|_| BackendError::InvalidArgument)?;
                            Ok::<_, BackendError>(SyncRef {
                                buffer,
                                offset: sync.offset,
                                value: sync.value,
                            })
                        })
                        .transpose()
                    };
                    executions.push(OrderedExecution {
                        program,
                        bindings: refs.as_slice(),
                        launch: entry.launch,
                        barrier: entry.barrier,
                        gate: word(entry.gate)?,
                        signal: word(entry.signal)?,
                    });
                }
                match backend.submit_ordered(resources.queue(), &executions, timeout) {
                    Ok(event) => Ok(event),
                    Err(SubmitFailure::Rejected(error)) => Err(error),
                    Err(SubmitFailure::Indeterminate { error, event }) => {
                        indeterminate = Some(error);
                        Ok(event)
                    }
                }
            });
        let event_id = match admitted {
            Ok(id) => id,
            Err(error) => return Err(self.refuse_create(error)),
        };
        self.events.push(LiveEvent {
            id: event_id.get(),
            via,
            notified: false,
        });
        Ok(match indeterminate {
            None => AccelResponse::Submitted {
                event: event_id.get(),
            },
            Some(error) => AccelResponse::SubmitIndeterminate {
                event: event_id.get(),
                refusal: self.backend_err(error),
            },
        })
    }

    // ── events ──────────────────────────────────────────────────────

    pub(crate) fn poll_event(
        &mut self,
        via: u16,
        event: u64,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let id = Self::obj(event)?;
        let record = self.state.event_record(id).map_err(refuse_state)?;
        let context = record.context_id();
        self.owned(context, via)?;
        let handle = self
            .state
            .event_record(id)
            .map_err(refuse_state)?
            .resource()
            .map_err(refuse_state)?;
        let state = self.backend.poll_event(handle);
        let state = state.map_err(|e| self.backend_err(e))?;
        // `DeviceLost` is an epoch edge whether the provider reports it as
        // the poll call's error or as this invocation's terminal failure.
        // Treating the latter as an ordinary failed job would leave the
        // poisoned backend reachable for later DMA/control operations.
        if matches!(state, EventState::Failed(BackendError::DeviceLost)) {
            self.lost = true;
        }
        Ok(AccelResponse::Event {
            event,
            state: AccelEventState::from_contract(state),
        })
    }

    pub(crate) fn cancel_event(
        &mut self,
        via: u16,
        event: u64,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let id = Self::obj(event)?;
        let record = self.state.event_record(id).map_err(refuse_state)?;
        self.owned(record.context_id(), via)?;
        self.info
            .validate_event_cancellation()
            .map_err(|e| self.backend_err(e))?;
        let handle = self
            .state
            .event_record(id)
            .map_err(refuse_state)?
            .resource()
            .map_err(refuse_state)?;
        let outcome = self.backend.cancel_event(handle);
        // `Busy` = completion won the race — the caller polls; that is
        // a lawful refusal, not a poisoning.
        outcome.map_err(|e| self.backend_err(e))?;
        Ok(AccelResponse::Cancelled { event })
    }

    pub(crate) fn destroy_event(
        &mut self,
        via: u16,
        event: u64,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let id = Self::obj(event)?;
        let record = self.state.event_record(id).map_err(refuse_state)?;
        self.owned(record.context_id(), via)?;
        // Only a *terminal* event may be released — the contract forbids
        // destroying pending work.
        let handle = self
            .state
            .event_record(id)
            .map_err(refuse_state)?
            .resource()
            .map_err(refuse_state)?;
        let observed = self.backend.poll_event(handle);
        match observed.map_err(|e| self.backend_err(e))? {
            EventState::Pending => return Err(AccelRefusal::Busy),
            EventState::Failed(BackendError::DeviceLost) => {
                self.lost = true;
                return Err(AccelRefusal::DeviceLost);
            }
            _ => {}
        }
        let resource = self.state.begin_event_release(id).map_err(refuse_state)?;
        match self.backend.destroy_event(resource) {
            Ok(()) => {
                self.state.commit_event_release(id).map_err(refuse_state)?;
                self.events.retain(|e| e.id != event);
                Ok(AccelResponse::Released { subject: event })
            }
            Err(failure) => Err(self.release_failed(
                failure,
                |s, r| s.state.restore_event_release(id, r).map_err(|e| e.error),
                |s| {
                    let committed = s.state.commit_event_release(id);
                    s.events.retain(|e| e.id != event);
                    committed.map(|_| ())
                },
            )),
        }
    }

    // ── releases ────────────────────────────────────────────────────

    pub(crate) fn destroy_context(
        &mut self,
        via: u16,
        ctx: u64,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let id = self.context_checked(ctx)?;
        self.owned(id, via)?;
        let resource = self.state.begin_context_release(id).map_err(refuse_state)?;
        match self.backend.destroy_context(resource) {
            Ok(()) => {
                self.state
                    .commit_context_release(id)
                    .map_err(refuse_state)?;
                self.owners.retain(|(owner, _)| *owner != ctx);
                Ok(AccelResponse::Released { subject: ctx })
            }
            Err(failure) => Err(self.release_failed(
                failure,
                |s, r| s.state.restore_context_release(id, r).map_err(|e| e.error),
                |s| {
                    let committed = s.state.commit_context_release(id);
                    s.owners.retain(|(owner, _)| *owner != ctx);
                    committed.map(|_| ())
                },
            )),
        }
    }

    pub(crate) fn free_buffer(
        &mut self,
        via: u16,
        buffer: u64,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.free_buffer_using(via, buffer, |backend, resource| {
            backend.free_buffer(resource)
        })
    }

    pub(crate) fn free_buffer_using<F>(
        &mut self,
        via: u16,
        buffer: u64,
        release: F,
    ) -> Result<AccelResponse, AccelRefusal>
    where
        F: FnOnce(&A, A::Buffer) -> Result<(), virtio_accel_core::ReleaseFailure<A::Buffer>>,
    {
        self.live()?;
        let id = Self::obj(buffer)?;
        self.owned(self.context_of_buffer(id)?, via)?;
        let bytes = self
            .state
            .buffer_record(id)
            .map_err(refuse_state)?
            .info()
            .desc()
            .bytes();
        self.release_buffer_using(id, release)?;
        // The create's exact charge, returned only on a committed release
        // (the cleanup path frees an uncharged, never-published buffer
        // through `release_buffer` directly and credits nothing).
        self.credit(via, bytes, |f| &mut f.buffer_used);
        Ok(AccelResponse::Released { subject: buffer })
    }

    /// The buffer release engine — shared by `FreeBuffer` and the
    /// create-cleanup path (where the id was never exposed).
    fn release_buffer(&mut self, id: ObjectId) -> Result<(), AccelRefusal> {
        self.release_buffer_using(id, |backend, resource| backend.free_buffer(resource))
    }

    fn release_buffer_using<F>(&mut self, id: ObjectId, release: F) -> Result<(), AccelRefusal>
    where
        F: FnOnce(&A, A::Buffer) -> Result<(), virtio_accel_core::ReleaseFailure<A::Buffer>>,
    {
        let resource = self.state.begin_buffer_release(id).map_err(refuse_state)?;
        match release(&self.backend, resource) {
            Ok(()) => self.state.commit_buffer_release(id).map_err(refuse_state),
            Err(failure) => Err(self.release_failed(
                failure,
                |s, r| s.state.restore_buffer_release(id, r).map_err(|e| e.error),
                |s| s.state.commit_buffer_release(id),
            )),
        }
    }

    pub(crate) fn unload_program(
        &mut self,
        via: u16,
        program: u64,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let id = Self::obj(program)?;
        let context = self
            .state
            .program_record(id)
            .map_err(refuse_state)?
            .context_id();
        self.owned(context, via)?;
        let resident_bytes = self
            .state
            .program_record(id)
            .map_err(refuse_state)?
            .resident_bytes();
        let resource = self.state.begin_program_release(id).map_err(refuse_state)?;
        match self.backend.unload_program(resource) {
            Ok(()) => {
                self.state
                    .commit_program_release(id)
                    .map_err(refuse_state)?;
                self.credit(via, resident_bytes, |f| &mut f.program_used);
                Ok(AccelResponse::Released { subject: program })
            }
            Err(failure) => Err(self.release_failed(
                failure,
                |s, r| s.state.restore_program_release(id, r).map_err(|e| e.error),
                |s| s.state.commit_program_release(id),
            )),
        }
    }

    pub(crate) fn destroy_queue(
        &mut self,
        via: u16,
        queue: u64,
    ) -> Result<AccelResponse, AccelRefusal> {
        self.live()?;
        let id = Self::obj(queue)?;
        let context = self
            .state
            .queue_record(id)
            .map_err(refuse_state)?
            .context_id();
        self.owned(context, via)?;
        let resource = self.state.begin_queue_release(id).map_err(refuse_state)?;
        match self.backend.destroy_queue(resource) {
            Ok(()) => {
                self.state.commit_queue_release(id).map_err(refuse_state)?;
                Ok(AccelResponse::Released { subject: queue })
            }
            Err(failure) => Err(self.release_failed(
                failure,
                |s, r| s.state.restore_queue_release(id, r).map_err(|e| e.error),
                |s| s.state.commit_queue_release(id),
            )),
        }
    }

    /// The shared tail of every failed provider release: **rejected**
    /// restores the live handle for retry; **indeterminate** invalidates
    /// the id and poisons the complete device epoch. The service seals and
    /// discards that namespace; recovery, where a row supervisor exists, must
    /// found a new backend epoch rather than reuse these ids.
    fn release_failed<R>(
        &mut self,
        failure: ReleaseFailure<R>,
        restore: impl FnOnce(&mut Self, R) -> Result<(), DeviceStateError>,
        commit: impl FnOnce(&mut Self) -> Result<(), DeviceStateError>,
    ) -> AccelRefusal {
        match failure {
            ReleaseFailure::Rejected { error, resource } => match restore(self, resource) {
                Ok(()) => self.backend_err(error),
                // A restore refusing is an engine invariant break: the
                // record was `Releasing` moments ago.
                Err(_) => {
                    self.lost = true;
                    AccelRefusal::Internal
                }
            },
            ReleaseFailure::Indeterminate { error } => {
                let _ = commit(self);
                self.lost = true;
                self.backend_err(error)
            }
        }
    }

    // ── completion sweep ────────────────────────────────────────────

    /// One bounded completion sweep: poll every event that has not yet
    /// been counted into a signal batch; a newly terminal one is counted
    /// toward its pair and marked, so each terminal edge is noticed at
    /// most once.  Polling stays the client's re-readable truth — this
    /// sweep only decides who gets woken.
    pub(crate) fn due_by_pair(&mut self) -> [u32; PAIRS] {
        let mut due = [0u32; PAIRS];
        if self.lost {
            return due;
        }
        let Self {
            state,
            backend,
            events,
            ..
        } = self;
        let mut device_lost = false;
        for event in events.iter_mut().filter(|e| !e.notified) {
            let Some(id) = ObjectId::from_raw(event.id) else {
                event.notified = true;
                continue;
            };
            let Ok(handle) = state.event_record(id).and_then(|record| record.resource()) else {
                // The record went away (destroyed between sweeps) — no
                // notice owed.
                event.notified = true;
                continue;
            };
            match backend.poll_event(handle) {
                Ok(EventState::Pending) => {}
                Ok(EventState::Failed(BackendError::DeviceLost)) => {
                    device_lost = true;
                    break;
                }
                Ok(_) => {
                    event.notified = true;
                    if let Some(count) = due.get_mut(super::pair_index(event.via)) {
                        *count = count.saturating_add(1);
                    }
                }
                // Ordinary poll errors are retryable. DeviceLost is an
                // epoch edge: stop touching the provider immediately and
                // discard any partial wake batch from this sweep.
                Err(BackendError::DeviceLost) => {
                    device_lost = true;
                    break;
                }
                Err(_) => {}
            }
        }
        if device_lost {
            self.lost = true;
            return [0; PAIRS];
        }
        due
    }

    fn refuse_create(&mut self, error: CreateError<BackendError>) -> AccelRefusal {
        match error {
            CreateError::State(e) => refuse_state(e),
            CreateError::Provider(e) => self.backend_err(e),
        }
    }

    /// Test seam: the raw event handle, for harness-driven completion
    /// (the conformance hooks' `complete_event` shape).
    #[cfg(test)]
    pub(crate) fn event_handle(&self, event: u64) -> Option<&A::Event> {
        let id = ObjectId::from_raw(event)?;
        self.state.event_record(id).ok()?.resource().ok()
    }
}

/// Event-id slots the device state holds per event a provider can keep live.
///
/// `virtio-accel-device` retires an object slot after 2^13 generations, so a
/// stale id can never resolve again. Sized to a provider's concurrency (Xe
/// keeps one live event) the table would end the device epoch at its 8192nd
/// submission: Gigatron's serial lane was refused `ResourceLimit` at exactly
/// its 8193rd job. The table is therefore sized for a lifetime, not for
/// concurrency. Slots are created lazily, one per 8192 retired ids, so this
/// costs nothing until it is used. Bounding concurrency stays with the
/// provider (Xe bounds its retained command slots; the NPU and SPMD
/// providers claim their own slots at submit), and clients are still told the
/// provider's own limit (`info` is unchanged).
const EVENT_ID_SLOTS_PER_EVENT: u32 = 1 << 16;

/// The provider's limits with the event table widened for its lifetime (see
/// [`EVENT_ID_SLOTS_PER_EVENT`]), inside both products `DeviceState` checks:
/// events × bindings per context and events × contexts must stay in `u32`.
fn event_id_lifetime(limits: DeviceLimits) -> DeviceLimits {
    let per_context = u32::MAX / limits.max_bindings_per_submission.max(1);
    let aggregate = u32::MAX / limits.max_contexts.max(1);
    let max_events_per_context = limits
        .max_events_per_context
        .saturating_mul(EVENT_ID_SLOTS_PER_EVENT)
        .min(per_context)
        .min(aggregate)
        .max(limits.max_events_per_context);
    DeviceLimits {
        max_events_per_context,
        ..limits
    }
}

/// Reserve service-ledger capacity before crossing a provider acceptance
/// boundary.  A successful return guarantees the following `push` cannot
/// allocate while this engine remains exclusively borrowed.
fn reserve_ledger_entries<T>(ledger: &mut Vec<T>, additional: usize) -> Result<(), AccelRefusal> {
    ledger
        .try_reserve(additional)
        .map_err(|_| AccelRefusal::OutOfMemory)
}

#[cfg(test)]
mod ledger_tests {
    use super::*;

    #[test]
    fn capacity_overflow_refuses_out_of_memory_without_mutating_the_ledger() {
        let mut ledger = Vec::<u8>::new();

        assert_eq!(
            reserve_ledger_entries(&mut ledger, usize::MAX),
            Err(AccelRefusal::OutOfMemory),
        );
        assert!(ledger.is_empty());
    }
}

/// A borrowed staging slice as the contract's byte source (`[u8]`'s own
/// impl cannot ride `&dyn` — an unsized base type never coerces to a
/// trait object, so the reference wears this thin coat).
#[derive(Debug)]
struct SliceSource<'a>(&'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        ByteSource::read_at(self.0, offset, target)
    }
    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self.0)
    }
}

/// [`SliceSource`]'s mutable twin (the `ReadBuffer` destination).
#[derive(Debug)]
struct SliceSink<'a>(&'a mut [u8]);

impl ByteSink for SliceSink<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        ByteSink::write_at(self.0, offset, source)
    }
    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(self.0)
    }
}

/// Fold the provider vocabulary onto the wire's.
fn refuse_backend(error: BackendError) -> AccelRefusal {
    match error {
        BackendError::Unsupported => AccelRefusal::Unsupported,
        BackendError::Incompatible => AccelRefusal::Incompatible,
        BackendError::InvalidArgument => AccelRefusal::InvalidArgument,
        BackendError::OutOfBounds => AccelRefusal::OutOfBounds,
        BackendError::Busy => AccelRefusal::Busy,
        BackendError::OutOfMemory => AccelRefusal::OutOfMemory,
        BackendError::ResourceLimit => AccelRefusal::ResourceLimit,
        BackendError::DeadlineExpired => AccelRefusal::DeadlineExpired,
        BackendError::DeviceLost => AccelRefusal::DeviceLost,
        BackendError::PermissionDenied => AccelRefusal::PermissionDenied,
        BackendError::External { domain, code } => AccelRefusal::External { domain, code },
    }
}

/// Fold the engine-state vocabulary onto the wire's.
fn refuse_state(error: DeviceStateError) -> AccelRefusal {
    match error {
        DeviceStateError::InvalidArgument => AccelRefusal::InvalidArgument,
        DeviceStateError::InvalidObject | DeviceStateError::StaleObject => AccelRefusal::BadObject,
        DeviceStateError::ContextMismatch => AccelRefusal::Incompatible,
        DeviceStateError::Busy | DeviceStateError::Releasing => AccelRefusal::Busy,
        DeviceStateError::ResourceLimit | DeviceStateError::ReferenceCountOverflow => {
            AccelRefusal::ResourceLimit
        }
        DeviceStateError::OutOfMemory => AccelRefusal::OutOfMemory,
        DeviceStateError::InvalidTransition => AccelRefusal::Internal,
    }
}

impl AccelEventState {
    /// Lift the contract's event state (folding a failure's error onto
    /// the wire refusal vocabulary).
    pub(crate) fn from_contract(state: EventState) -> Self {
        match state {
            EventState::Pending => Self::Pending,
            EventState::Complete => Self::Complete,
            EventState::Failed(error) => Self::Failed(refuse_backend(error)),
            EventState::Cancelled => Self::Cancelled,
        }
    }
}

// `AccessMode` rides the vocabulary directly (protocol 1.0's own wire
// values); this pin keeps the codec honest about them.
const _: () = {
    assert!(AccessMode::Read as u8 == 1);
    assert!(AccessMode::Write as u8 == 2);
    assert!(AccessMode::ReadWrite as u8 == 3);
};
