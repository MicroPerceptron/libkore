//! The **SPMD backend** — `impl Accelerator` over CPU workers.
//!
//! The accel protocol declares one vocabulary for every backend, "NPU / GPU /
//! CPU-SIMD" (`crates/abi/src/svc.rs`). The NPU and GPU have device-side
//! implementations; this crate supplies the CPU backend, but not yet a
//! founded service row. It needs no device, no BAR, no DMA pool and no broker
//! claim — its compute resource is the machine's own cores, reached through
//! [`kore_rt::spmd`].
//!
//! # Why the compute is not inside `submit`
//!
//! [`Accelerator::submit`]'s contract is explicit: *"synchronous work is
//! limited to validation and admission; the call must not wait for execution
//! to finish."* That rules out running a kernel inline. The R9 resident region
//! now provides the right non-blocking engine shape, but this backend is not
//! yet founded as a service and therefore has no long-lived region owner or
//! event-descriptor storage to bind to it:
//!
//! - [`spmd::dispatch_slices`] is the reusable launch, and it **blocks until
//!   done** — it owns a single static anchor and re-stages it only after
//!   joining;
//! - [`spmd::found_resident`] can serve a stream of jobs, but its unsafe `env`
//!   contract requires each admitted descriptor and everything it borrows to
//!   remain owned until the completion generation advances. That owner belongs
//!   in the future founded CPU service row, not in a stack-local `submit`.
//!
//! The current unfounded backend therefore keeps the same conformant shape
//! real hardware uses:
//! **admit, then drive, then observe.** `submit` validates and parks the work
//! in an event slot; [`SpmdBackend::run_pending`] performs one admitted
//! launch; `poll_event` reports without blocking. On a device the driving is
//! the device's; here it is the owning service's own turn — which is a
//! scheduling fact about a CPU backend, not a loosening of the contract.
//!
//! Founding the R9 engine changes `run_pending` and the private event owner;
//! no protocol surface moves.
//!
//! # What holds the bound bytes
//!
//! An admitted event records raw spans of its bound buffers rather than
//! borrowing them: the trait requires the caller to keep "every referenced
//! handle alive until the returned event is terminal and destroyed", which is
//! exactly the lifetime a raw span needs and exactly what a Rust borrow cannot
//! express across an admission boundary. The refusals in [`SpmdBackend::submit`]
//! are what make the spans sound — in particular the aliasing check, without
//! which one buffer bound as both input and output would hand the region
//! overlapping `&` and `&mut`.
//!
//! [`Accelerator::submit`]: virtio_accel_core::Accelerator::submit
//! [`spmd::dispatch_slices`]: kore_rt::spmd::dispatch_slices
//! [`spmd::found_resident`]: kore_rt::spmd::found_resident

extern crate alloc;

use alloc::vec::Vec;
use core::cell::{Cell, RefCell, UnsafeCell};
use kore_driver::accel::NativeAccelerator;

#[allow(deprecated)]
use kore_driver::accel::Dispatch;
use kore_driver::accel::LaunchShape;
use kore_driver::accel::Workgroups;
use virtio_accel_core::{
    AcceleratorClass, AccessMode, AllocatedBuffer, ArtifactRef, BackendError, BindingRef,
    BufferDesc, BufferInfo, BufferProperties, Capabilities, ContextDesc, DeviceIdentity,
    DeviceInfo, DeviceLimits, EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure,
    Timeout,
};

use crate::kernel::{Kernel, TileCtx, kernel_id, lookup};

/// The artifact format this row serves: registered kernels, named by
/// [`TargetIdentity`](virtio_accel_core::TargetIdentity) and carrying a small,
/// fixed manifest.
///
/// A distinct format rather than a reinterpretation of someone else's, so a
/// client that hands this row an NPU ELF is refused by *format* rather than
/// misparsed.
pub const FORMAT_REGISTERED: u32 = u32::from_le_bytes(*b"KSPM");

/// Versioned fixed manifest for a registered-kernel artifact: magic, version,
/// and a reserved-zero word. Keeping it nonzero makes the row admissible
/// through `virtio-accel-device`, whose program accounting correctly refuses
/// zero-byte artifacts.
pub const REGISTERED_MANIFEST: [u8; 8] = [b'K', b'S', b'P', b'M', 1, 0, 0, 0];

/// Contexts this backend hosts at once.
const MAX_CONTEXTS: u32 = 1;
/// Buffers per context.
const MAX_BUFFERS: u32 = 256;
/// Resident programs per context — a lookup costs nothing, so this bounds
/// only bookkeeping.
const MAX_PROGRAMS: u32 = 32;
/// Execution queues per context.
const MAX_QUEUES: u32 = 4;
/// Admitted events in flight.
const MAX_EVENTS: usize = 32;
/// Inputs one kernel may bind. Two covers the built-ins with headroom, and a
/// fixed array keeps the admitted record allocation-free.
const MAX_INPUTS: usize = 4;

/// A host buffer.
///
/// [`UnsafeCell`] rather than a plain `Vec` because a submission's output is
/// written through a **shared** binding (`BindingRef` carries `&B`), exactly
/// as a device backend writes through a shared DMA mapping. The exclusivity
/// argument is [`SpmdBackend::submit`]'s: one admitted launch at a time, no
/// buffer bound twice in it, and the region's workers touching disjoint
/// element spans of the one output.
pub struct SpmdBuffer {
    bytes: UnsafeCell<Vec<u8>>,
    desc: BufferDesc,
}

impl SpmdBuffer {
    fn len(&self) -> usize {
        // SAFETY: the length is fixed at allocation and never changes; no
        // writer resizes the vector.
        unsafe { (*self.bytes.get()).len() }
    }

    fn base(&self) -> *mut u8 {
        // SAFETY: the allocation outlives every span derived from it — the
        // trait pins the handle until its event is terminal and destroyed.
        unsafe { (*self.bytes.get()).as_mut_ptr() }
    }

    /// Read access for `read_buffer` and for host-side assertions.
    fn as_slice(&self) -> &[u8] {
        // SAFETY: no launch is in flight when the adapter calls a transfer
        // method — `run_pending` is synchronous and the service is
        // single-threaded per task.
        unsafe { &*self.bytes.get() }
    }
}

/// A context. The CPU has no per-context device state; this exists so the
/// protocol's object graph is honest rather than collapsed.
pub struct SpmdContext(());

/// A resident program: one registered kernel.
pub struct SpmdProgram {
    kernel: &'static Kernel,
}

/// Resident bytes retained by a registered program handle. This is the
/// canonical nonzero charge a host should put in `ArtifactRef::resident_bytes`.
pub const REGISTERED_RESIDENT_BYTES: u64 = core::mem::size_of::<SpmdProgram>() as u64;

impl SpmdProgram {
    /// The kernel this program resolved to — its diagnostic name is what a
    /// status row prints.
    pub const fn name(&self) -> &'static str {
        self.kernel.name
    }
}

/// An execution queue. Ordering is per-queue and, with one admitted launch at
/// a time, trivially satisfied.
pub struct SpmdQueue(());

/// A submission token — an index into the backend's event table.
pub struct SpmdEvent {
    slot: usize,
}

/// One bound span, recorded at admission.
#[derive(Clone, Copy)]
struct Span {
    base: *mut u8,
    len: usize,
}

/// An admitted-but-unrun launch.
struct Admitted {
    kernel: &'static Kernel,
    inputs: [Span; MAX_INPUTS],
    n_inputs: usize,
    out: Span,
    elems: u32,
    launch: LaunchShape,
}

enum Slot {
    Free,
    Admitted(Admitted),
    Terminal(EventState),
}

/// The output span, split among the region's workers.
///
/// [`spmd::dispatch_slices`](kore_rt::spmd::dispatch_slices) hands its body a
/// `Fn(Range<u32>) + Sync`, so the body cannot capture a `&mut [u8]`. This
/// type is the disjointness argument made explicit and contained: it carries
/// the whole span once, and [`tile`](Self::tile) hands out exactly one
/// worker's elements. The block split `dispatch_slices` performs is what makes
/// those spans disjoint, so no two workers ever alias — and every kernel above
/// this line stays entirely safe.
struct TileOut {
    base: *mut u8,
    len: usize,
    elem: usize,
}

// SAFETY: the only access is `tile`, which is called once per worker with that
// worker's own half-open element range. `dispatch_slices` derives those ranges
// from a block split of a single grid, so they are pairwise disjoint; distinct
// disjoint sub-slices of one allocation may be held mutably at once.
unsafe impl Sync for TileOut {}

impl TileOut {
    /// This worker's bytes.
    ///
    /// # Safety
    /// `range` must be one worker's own block-split span for this launch, and
    /// no other caller may hold an overlapping span concurrently.
    unsafe fn tile(&self, range: core::ops::Range<u32>) -> &mut [u8] {
        let start = range.start as usize * self.elem;
        let end = range.end as usize * self.elem;
        debug_assert!(end <= self.len, "a tile span must stay inside the output");
        // SAFETY: the caller's contract gives exclusivity over this span, and
        // the bounds were validated at admission (`elems * elem <= out.len`).
        unsafe { core::slice::from_raw_parts_mut(self.base.add(start), end - start) }
    }
}

/// The CPU accelerator row.
pub struct SpmdBackend {
    registry: &'static [Kernel],
    ctx_live: Cell<bool>,
    buffers: Cell<u32>,
    programs: Cell<u32>,
    queues: Cell<u32>,
    events: RefCell<[Slot; MAX_EVENTS]>,
}

impl SpmdBackend {
    /// A backend serving `registry`.
    pub fn new(registry: &'static [Kernel]) -> Self {
        Self {
            registry,
            ctx_live: Cell::new(false),
            buffers: Cell::new(0),
            programs: Cell::new(0),
            queues: Cell::new(0),
            events: RefCell::new([const { Slot::Free }; MAX_EVENTS]),
        }
    }

    /// **Drive one admitted launch**, oldest slot first; `true` when one ran.
    ///
    /// The execution step `submit` is forbidden to perform (see the module
    /// docs). The owning service calls this on its own turn; a test calls it
    /// directly. Blocking here is lawful and deliberate — it is the CPU
    /// backend's compute, and the trait constrains `submit` and `poll_event`,
    /// not the service's own scheduling.
    ///
    /// Off metal — or when a metal region launch refuses because the shared
    /// dispatch anchor is busy or the machine is exhausted —
    /// [`spmd::dispatch_slices`](kore_rt::spmd::dispatch_slices) answers
    /// `None` and the work runs **serially on this thread**. That is not a
    /// degraded mode to be hidden: SPMD is the acceleration here, never the
    /// semantics, so the same bytes come out either way and the host suite
    /// exercises this backend path.
    pub fn run_pending(&self) -> bool {
        let admitted = {
            let mut slots = self.events.borrow_mut();
            let Some(index) = slots
                .iter()
                .position(|slot| matches!(slot, Slot::Admitted(_)))
            else {
                return false;
            };
            match core::mem::replace(&mut slots[index], Slot::Free) {
                Slot::Admitted(work) => (index, work),
                other => {
                    slots[index] = other;
                    return false;
                }
            }
        };
        let (index, work) = admitted;
        Self::execute(&work);
        self.events.borrow_mut()[index] = Slot::Terminal(EventState::Complete);
        true
    }

    /// Run one admitted launch over the region, or serially where there is
    /// none.
    fn execute(work: &Admitted) {
        let kernel = work.kernel;
        let elem = kernel.elem_bytes as usize;
        let inputs: [&[u8]; MAX_INPUTS] = core::array::from_fn(|i| {
            let span = work.inputs[i];
            if i < work.n_inputs {
                // SAFETY: recorded at admission from a live buffer the trait
                // pins until this event is terminal; read-only, and the
                // aliasing refusal proved it is not the output.
                unsafe { core::slice::from_raw_parts(span.base as *const u8, span.len) }
            } else {
                &[]
            }
        });
        let inputs = &inputs[..work.n_inputs];
        let out = TileOut {
            base: work.out.base,
            len: work.out.len,
            elem,
        };
        let total = work.elems;
        let launch = work.launch;

        let body = |range: core::ops::Range<u32>| {
            // SAFETY: `dispatch_slices` block-splits the grid, so this range
            // is this worker's alone and disjoint from every sibling's.
            let slice = unsafe { out.tile(range.clone()) };
            (kernel.run)(TileCtx {
                tiles: range,
                total,
                launch,
                inputs,
                out: slice,
            });
        };

        if kore_rt::spmd::dispatch_slices(total, &body).is_none() {
            // The region launch refused: same grid, one CPU worker, this
            // thread. The future founded resident row removes this shared-
            // anchor contention from steady-state submissions.
            body(0..total);
        }
    }

    fn claim_event(&self) -> Result<usize, BackendError> {
        let slots = self.events.borrow();
        if slots.iter().any(|slot| matches!(slot, Slot::Admitted(_))) {
            return Err(BackendError::Busy);
        }
        slots
            .iter()
            .position(|slot| matches!(slot, Slot::Free))
            .ok_or(BackendError::ResourceLimit)
    }

    /// **The admission** both submit paths share.
    ///
    /// `launch` is R8's exact carried policy. Artifact-defined execution
    /// derives its item count from the output; explicit workgroups are checked
    /// against it without discarding their six logical axes.
    fn admit(
        &self,
        program: &SpmdProgram,
        bindings: &[BindingRef<'_, SpmdBuffer>],
        timeout: Timeout,
        launch: LaunchShape,
    ) -> Result<SpmdEvent, SubmitFailure<SpmdEvent>> {
        let reject = |e| Err(SubmitFailure::Rejected(e));
        let kernel = program.kernel;
        let n_in = kernel.inputs as usize;

        // A deadline would have to be enforced against a launch this backend
        // does not yet interrupt, so it is refused rather than accepted and
        // ignored — the same posture the reserved wire fields take.
        if !matches!(timeout, Timeout::Infinite) {
            return reject(BackendError::Unsupported);
        }
        if n_in > MAX_INPUTS || bindings.len() != n_in + 1 {
            return reject(BackendError::InvalidArgument);
        }

        let mut inputs = [Span {
            base: core::ptr::null_mut(),
            len: 0,
        }; MAX_INPUTS];
        let mut out: Option<Span> = None;
        let mut seen = [false; MAX_INPUTS + 1];

        for binding in bindings {
            let slot = binding.slot as usize;
            if slot > n_in || seen[slot] {
                return reject(BackendError::InvalidArgument);
            }
            seen[slot] = true;

            let buffer = binding.buffer;
            let Ok(offset) = usize::try_from(binding.range.offset) else {
                return reject(BackendError::OutOfBounds);
            };
            let Ok(len) = usize::try_from(binding.range.bytes()) else {
                return reject(BackendError::OutOfBounds);
            };
            let end = match offset.checked_add(len) {
                Some(end) if end <= buffer.len() => end,
                _ => return reject(BackendError::OutOfBounds),
            };
            // Defense in depth: the host is required to have checked this, and
            // repeating it costs a comparison against a client-shaped mistake
            // that would otherwise write through a read-only declaration.
            if !buffer.desc.allows_access(binding.access) {
                return reject(BackendError::InvalidArgument);
            }
            let want = if slot == n_in {
                AccessMode::Write
            } else {
                AccessMode::Read
            };
            if binding.access != want {
                return reject(BackendError::InvalidArgument);
            }
            let span = Span {
                // SAFETY: `offset <= end <= buffer.len()`, so the offset stays
                // inside the one allocation this pointer is derived from.
                base: unsafe { buffer.base().add(offset) },
                len: end - offset,
            };
            if slot == n_in {
                out = Some(span);
            } else {
                inputs[slot] = span;
            }
        }

        let Some(out) = out else {
            return reject(BackendError::InvalidArgument);
        };

        // **The aliasing refusal** — the check the recorded raw spans rest on.
        // One buffer bound as both an input and the output would hand the
        // region overlapping shared and exclusive views of the same bytes, and
        // no amount of care inside the kernel could make that sound.
        for input in inputs.iter().take(n_in) {
            let lo = input.base as usize;
            let hi = lo + input.len;
            let olo = out.base as usize;
            let ohi = olo + out.len;
            if lo < ohi && olo < hi {
                return reject(BackendError::InvalidArgument);
            }
        }

        let elem = kernel.elem_bytes as usize;
        if elem == 0 || out.len % elem != 0 {
            return reject(BackendError::InvalidArgument);
        }
        let Ok(derived) = u32::try_from(out.len / elem) else {
            return reject(BackendError::OutOfBounds);
        };
        let named = match launch {
            LaunchShape::ArtifactDefined => None,
            LaunchShape::Workgroups(workgroups) => {
                let items = workgroups
                    .group_count()
                    .into_iter()
                    .chain(workgroups.local_size())
                    .try_fold(1u64, |acc, axis| acc.checked_mul(u64::from(axis)));
                let Some(items) = items.and_then(|items| u32::try_from(items).ok()) else {
                    return reject(BackendError::OutOfBounds);
                };
                Some(items)
            }
        };
        // A carried geometry must describe the launch the bound output can
        // actually hold. Silently clamping to the smaller of the two would run
        // a different launch than the client asked for and report success.
        let elems = match named {
            Some(named) if named != derived => return reject(BackendError::InvalidArgument),
            Some(named) => named,
            None => derived,
        };
        // Every input must cover the same grid: a short input would read past
        // its own span in the kernel's element indexing.
        for input in inputs.iter().take(n_in) {
            if input.len < out.len {
                return reject(BackendError::OutOfBounds);
            }
        }

        let slot = self.claim_event().map_err(SubmitFailure::Rejected)?;
        self.events.borrow_mut()[slot] = Slot::Admitted(Admitted {
            kernel,
            inputs,
            n_inputs: n_in,
            out,
            elems,
            launch,
        });
        Ok(SpmdEvent { slot })
    }
}

impl virtio_accel_core::Accelerator for SpmdBackend {
    type Context = SpmdContext;
    type Buffer = SpmdBuffer;
    type Program = SpmdProgram;
    type Queue = SpmdQueue;
    type Event = SpmdEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(DeviceInfo {
            identity: DeviceIdentity {
                uuid: *b"kore-spmd-cpu\0\0\0",
                // The protocol assigns NPU/GPU/DSP and no CPU value. `OTHER`
                // is the honest answer: inventing a discriminant here would
                // claim a number the house protocol has not allocated.
                class: AcceleratorClass::OTHER,
                vendor_id: 0,
                device_id: 0,
            },
            // Host DRAM is both host-visible and directly bindable; there is
            // no device-local domain and no cancellation path.
            capabilities: Capabilities::HOST_VISIBLE_MEMORY | Capabilities::SHARED_MEMORY,
            limits: DeviceLimits {
                max_contexts: MAX_CONTEXTS,
                max_buffers_per_context: MAX_BUFFERS,
                max_programs_per_context: MAX_PROGRAMS,
                max_queues_per_context: MAX_QUEUES,
                max_events_per_context: MAX_EVENTS as u32,
                max_bindings_per_submission: (MAX_INPUTS + 1) as u32,
                max_buffer_bytes: 1 << 31,
                max_artifact_bytes: REGISTERED_MANIFEST.len() as u64,
            },
        })
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        if !desc.flags.is_empty() {
            return Err(BackendError::Unsupported);
        }
        if self.ctx_live.replace(true) {
            return Err(BackendError::ResourceLimit);
        }
        Ok(SpmdContext(()))
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
        if self.buffers.get() != 0 || self.programs.get() != 0 || self.queues.get() != 0 {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: context,
            });
        }
        if self
            .events
            .borrow()
            .iter()
            .any(|slot| !matches!(slot, Slot::Free))
        {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: context,
            });
        }
        self.ctx_live.set(false);
        Ok(())
    }

    fn allocate_buffer(
        &self,
        _context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        match desc.domain {
            MemoryDomain::Host | MemoryDomain::Shared => {}
            MemoryDomain::Device => return Err(BackendError::Unsupported),
        }
        if self.buffers.get() >= MAX_BUFFERS {
            return Err(BackendError::ResourceLimit);
        }
        let bytes = usize::try_from(desc.bytes()).map_err(|_| BackendError::OutOfBounds)?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)
            .map_err(|_| BackendError::OutOfMemory)?;
        data.resize(bytes, 0);

        // The allocator's guarantee, stated rather than assumed: a `Vec<u8>`
        // is byte-aligned, so a stricter request is refused instead of being
        // silently under-served — `BufferInfo::new` would catch it, and
        // catching it here names the reason.
        let alignment = desc.alignment();
        if alignment > 1 && (data.as_ptr() as u64) % alignment != 0 {
            return Err(BackendError::Incompatible);
        }
        let info = BufferInfo::new(
            desc,
            desc.bytes(),
            alignment,
            BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING,
        )?;
        self.buffers.set(self.buffers.get() + 1);
        Ok(AllocatedBuffer::new(
            SpmdBuffer {
                bytes: UnsafeCell::new(data),
                desc,
            },
            info,
        ))
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn virtio_accel_core::ByteSource,
    ) -> Result<(), BackendError> {
        if !buffer
            .desc
            .usage
            .contains(virtio_accel_core::BufferUsage::TRANSFER_DESTINATION)
        {
            return Err(BackendError::PermissionDenied);
        }
        let at = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let len = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let end = at.checked_add(len).ok_or(BackendError::OutOfBounds)?;
        if end > buffer.len() {
            return Err(BackendError::OutOfBounds);
        }
        // SAFETY: `&mut Self::Buffer` is exclusive, and no launch is in flight
        // (`run_pending` is synchronous).
        let bytes = unsafe { &mut *buffer.bytes.get() };
        data.read_at(0, &mut bytes[at..end])
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn virtio_accel_core::ByteSink,
    ) -> Result<(), BackendError> {
        if !buffer
            .desc
            .usage
            .contains(virtio_accel_core::BufferUsage::TRANSFER_SOURCE)
        {
            return Err(BackendError::PermissionDenied);
        }
        let at = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let len = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let end = at.checked_add(len).ok_or(BackendError::OutOfBounds)?;
        if end > buffer.len() {
            return Err(BackendError::OutOfBounds);
        }
        data.write_at(0, &buffer.as_slice()[at..end])
    }

    fn free_buffer(&self, buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        drop(buffer);
        self.buffers.set(self.buffers.get().saturating_sub(1));
        Ok(())
    }

    fn load_program(
        &self,
        _context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        if artifact.format.get() != FORMAT_REGISTERED {
            return Err(BackendError::Unsupported);
        }
        if artifact.payload.len() != REGISTERED_MANIFEST.len() as u64 {
            return Err(BackendError::InvalidArgument);
        }
        let mut manifest = [0u8; REGISTERED_MANIFEST.len()];
        artifact.payload.read_at(0, &mut manifest)?;
        if manifest != REGISTERED_MANIFEST {
            return Err(BackendError::InvalidArgument);
        }
        if artifact.resident_bytes < REGISTERED_RESIDENT_BYTES {
            return Err(BackendError::ResourceLimit);
        }
        let Some(id) = kernel_id(artifact.target) else {
            return Err(BackendError::InvalidArgument);
        };
        let Some(kernel) = lookup(self.registry, id) else {
            return Err(BackendError::Incompatible);
        };
        if self.programs.get() >= MAX_PROGRAMS {
            return Err(BackendError::ResourceLimit);
        }
        self.programs.set(self.programs.get() + 1);
        Ok(SpmdProgram { kernel })
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        drop(program);
        self.programs.set(self.programs.get().saturating_sub(1));
        Ok(())
    }

    fn create_queue(
        &self,
        _context: &Self::Context,
        desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        if !desc.flags.is_empty() {
            return Err(BackendError::Unsupported);
        }
        if self.queues.get() >= MAX_QUEUES {
            return Err(BackendError::ResourceLimit);
        }
        self.queues.set(self.queues.get() + 1);
        Ok(SpmdQueue(()))
    }

    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        drop(queue);
        self.queues.set(self.queues.get().saturating_sub(1));
        Ok(())
    }

    fn submit(
        &self,
        _queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        self.admit(program, bindings, timeout, LaunchShape::ArtifactDefined)
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        match &self.events.borrow()[event.slot] {
            Slot::Admitted(_) => Ok(EventState::Pending),
            Slot::Terminal(state) => Ok(*state),
            // A destroyed or never-admitted slot: the adapter is holding an
            // event it already retired.
            Slot::Free => Err(BackendError::InvalidArgument),
        }
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        let mut slots = self.events.borrow_mut();
        if matches!(slots[event.slot], Slot::Admitted(_)) {
            drop(slots);
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: event,
            });
        }
        slots[event.slot] = Slot::Free;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{BUILTIN, IDENTITY_MAGIC, KERNEL_ADD_F32};
    use kore_driver::accel::LaunchShape;
    use kore_driver::accel::Workgroups;
    use std::vec;
    use virtio_accel_core::{
        Accelerator, ArtifactFormat, BufferRange, BufferUsage, ContextDesc, ContextFlags,
        QueueFlags, TargetIdentity,
    };

    const ELEMS: usize = 19; // two SIMD steps plus a tail
    const WIDE: usize = ELEMS * 4;
    const BYTES: u64 = WIDE as u64;
    const KERNEL_OBSERVE_LAUNCH: u32 = 0xfeed;

    /// Report the carried launch through the kernel's own output span:
    /// element `i` becomes the launch axis at `i % 6`. Each element's value
    /// depends only on the launch and its own index, so every tiling of the
    /// grid — SPMD workers or the serial fallback — writes the same bytes,
    /// and the report lands in the submitting test's buffer rather than
    /// process state a concurrently running test could clobber.
    fn observe_launch(ctx: TileCtx<'_>) {
        let TileCtx {
            tiles, launch, out, ..
        } = ctx;
        let LaunchShape::Workgroups(workgroups) = launch else {
            return;
        };
        let [gx, gy, gz] = workgroups.group_count();
        let [lx, ly, lz] = workgroups.local_size();
        let axes = [gx, gy, gz, lx, ly, lz];
        for (i, tile) in tiles.enumerate() {
            let value = axes[tile as usize % axes.len()] as f32;
            out[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    static OBSERVING_REGISTRY: &[Kernel] = &[Kernel {
        id: KERNEL_OBSERVE_LAUNCH,
        name: "observe.launch",
        inputs: 0,
        elem_bytes: 4,
        lanes: 1,
        run: observe_launch,
    }];

    /// The six axes [`observe_launch`] reported into `buffer`.
    fn observed_launch(b: &SpmdBackend, buffer: &SpmdBuffer) -> [u32; 6] {
        let floats = read_floats(b, buffer);
        core::array::from_fn(|i| floats[i] as u32)
    }

    /// `ByteSource`/`ByteSink` are implemented for sized arrays, so transfers
    /// go through one — a `&[u8]` cannot become a `&dyn ByteSource`.
    fn packed(values: &[f32]) -> [u8; WIDE] {
        let mut out = [0u8; WIDE];
        for (i, v) in values.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn backend() -> SpmdBackend {
        SpmdBackend::new(BUILTIN)
    }

    fn desc(usage: BufferUsage) -> BufferDesc {
        BufferDesc::new(BYTES, 1, MemoryDomain::Host, usage).expect("descriptor")
    }

    fn alloc(b: &SpmdBackend, ctx: &SpmdContext, usage: BufferUsage, values: &[f32]) -> SpmdBuffer {
        let transfer_usage =
            usage | BufferUsage::TRANSFER_SOURCE | BufferUsage::TRANSFER_DESTINATION;
        let mut allocated = b
            .allocate_buffer(ctx, desc(transfer_usage))
            .expect("allocated");
        b.write_buffer(allocated.buffer_mut(), 0, &packed(values))
            .expect("written");
        allocated.into_parts().0
    }

    fn program(b: &SpmdBackend, ctx: &SpmdContext, id: u32) -> SpmdProgram {
        let mut words = [0u32; 12];
        words[0] = IDENTITY_MAGIC;
        words[1] = id;
        b.load_program(
            ctx,
            ArtifactRef {
                format: ArtifactFormat::new(FORMAT_REGISTERED).expect("format"),
                target: TargetIdentity(words),
                payload: &REGISTERED_MANIFEST,
                resident_bytes: REGISTERED_RESIDENT_BYTES,
            },
        )
        .expect("loaded")
    }

    fn bind<'a>(
        slot: u32,
        buffer: &'a SpmdBuffer,
        access: AccessMode,
    ) -> BindingRef<'a, SpmdBuffer> {
        BindingRef {
            slot,
            buffer,
            range: BufferRange::new(0, BYTES).expect("range"),
            access,
        }
    }

    fn read_floats(b: &SpmdBackend, buffer: &SpmdBuffer) -> vec::Vec<f32> {
        let mut out = [0u8; WIDE];
        b.read_buffer(buffer, 0, &mut out).expect("read");
        out.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// The whole path on a machine with no region: context, buffers, program,
    /// admission, the drive step, and a terminal poll carrying the right
    /// bytes. This is the claim the row exists to make — that the accel
    /// protocol admits a non-device tenant — and it runs on any host.
    #[test]
    fn a_submission_runs_end_to_end_and_produces_the_right_bytes() {
        let b = backend();
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let queue = b
            .create_queue(
                &ctx,
                QueueDesc {
                    flags: QueueFlags::empty(),
                },
            )
            .expect("queue");

        let lhs: vec::Vec<f32> = (0..ELEMS).map(|i| i as f32).collect();
        let rhs: vec::Vec<f32> = (0..ELEMS).map(|i| (i * 3) as f32).collect();
        let a = alloc(&b, &ctx, BufferUsage::PROGRAM_INPUT, &lhs);
        let c = alloc(&b, &ctx, BufferUsage::PROGRAM_INPUT, &rhs);
        let out = alloc(&b, &ctx, BufferUsage::PROGRAM_OUTPUT, &vec![0.0; ELEMS]);
        let prog = program(&b, &ctx, KERNEL_ADD_F32);

        let event = b
            .submit(
                &queue,
                &prog,
                &[
                    bind(0, &a, AccessMode::Read),
                    bind(1, &c, AccessMode::Read),
                    bind(2, &out, AccessMode::Write),
                ],
                Timeout::Infinite,
            )
            .map_err(|_| "submit")
            .expect("admitted");

        assert!(
            matches!(b.poll_event(&event), Ok(EventState::Pending)),
            "admission must not have executed anything"
        );
        assert!(b.run_pending(), "one launch was admitted");
        assert!(!b.run_pending(), "and only one");
        assert!(matches!(b.poll_event(&event), Ok(EventState::Complete)));

        let got = read_floats(&b, &out);
        for i in 0..ELEMS {
            assert_eq!(got[i], lhs[i] + rhs[i], "element {i}");
        }
        b.destroy_event(event)
            .map_err(|_| "destroy")
            .expect("freed");
    }

    /// `load_program` refuses every shape that is not a registered kernel,
    /// and each refusal is distinguishable — a client can tell "wrong format"
    /// from "unknown kernel".
    #[test]
    fn load_program_refuses_everything_that_is_not_a_registered_kernel() {
        let b = backend();
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let good = TargetIdentity({
            let mut w = [0u32; 12];
            w[0] = IDENTITY_MAGIC;
            w[1] = KERNEL_ADD_F32;
            w
        });
        let fmt = ArtifactFormat::new(FORMAT_REGISTERED).expect("format");

        let other_format = ArtifactFormat::new(FORMAT_REGISTERED ^ 1).expect("format");
        assert!(matches!(
            b.load_program(
                &ctx,
                ArtifactRef {
                    format: other_format,
                    target: good,
                    payload: &REGISTERED_MANIFEST,
                    resident_bytes: REGISTERED_RESIDENT_BYTES
                }
            ),
            Err(BackendError::Unsupported)
        ));

        assert!(
            matches!(
                b.load_program(
                    &ctx,
                    ArtifactRef {
                        format: fmt,
                        target: good,
                        payload: b"bytes",
                        resident_bytes: REGISTERED_RESIDENT_BYTES
                    }
                ),
                Err(BackendError::InvalidArgument)
            ),
            "a malformed manifest must be refused, never silently ignored"
        );

        let mut wrong_version = REGISTERED_MANIFEST;
        wrong_version[4] = 2;
        assert!(matches!(
            b.load_program(
                &ctx,
                ArtifactRef {
                    format: fmt,
                    target: good,
                    payload: &wrong_version,
                    resident_bytes: REGISTERED_RESIDENT_BYTES
                }
            ),
            Err(BackendError::InvalidArgument)
        ));

        assert!(matches!(
            b.load_program(
                &ctx,
                ArtifactRef {
                    format: fmt,
                    target: good,
                    payload: &REGISTERED_MANIFEST,
                    resident_bytes: REGISTERED_RESIDENT_BYTES - 1
                }
            ),
            Err(BackendError::ResourceLimit)
        ));

        assert!(matches!(
            b.load_program(
                &ctx,
                ArtifactRef {
                    format: fmt,
                    target: TargetIdentity([0; 12]),
                    payload: &REGISTERED_MANIFEST,
                    resident_bytes: REGISTERED_RESIDENT_BYTES
                }
            ),
            Err(BackendError::InvalidArgument)
        ));

        let unknown = TargetIdentity({
            let mut w = [0u32; 12];
            w[0] = IDENTITY_MAGIC;
            w[1] = 0xdead;
            w
        });
        assert!(
            matches!(
                b.load_program(
                    &ctx,
                    ArtifactRef {
                        format: fmt,
                        target: unknown,
                        payload: &REGISTERED_MANIFEST,
                        resident_bytes: REGISTERED_RESIDENT_BYTES
                    }
                ),
                Err(BackendError::Incompatible)
            ),
            "a well-formed identity naming no kernel is Incompatible, not invalid"
        );
    }

    /// **The refusal the recorded raw spans rest on.** One buffer bound as
    /// both an input and the output would give the region overlapping shared
    /// and exclusive views of the same bytes.
    #[test]
    fn submit_refuses_a_buffer_bound_as_both_input_and_output() {
        let b = backend();
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let queue = b
            .create_queue(
                &ctx,
                QueueDesc {
                    flags: QueueFlags::empty(),
                },
            )
            .expect("queue");
        let shared = alloc(&b, &ctx, BufferUsage::MUTABLE_STATE, &vec![1.0; ELEMS]);
        let other = alloc(&b, &ctx, BufferUsage::PROGRAM_INPUT, &vec![2.0; ELEMS]);
        let prog = program(&b, &ctx, KERNEL_ADD_F32);

        let refused = b.submit(
            &queue,
            &prog,
            &[
                bind(0, &shared, AccessMode::Read),
                bind(1, &other, AccessMode::Read),
                bind(2, &shared, AccessMode::Write),
            ],
            Timeout::Infinite,
        );
        assert!(
            matches!(
                refused,
                Err(SubmitFailure::Rejected(BackendError::InvalidArgument))
            ),
            "aliasing the output with an input must refuse"
        );
    }

    /// Admission validates shape before it records anything: binding count,
    /// access direction, and a deadline this backend cannot enforce.
    #[test]
    fn submit_refuses_malformed_admissions() {
        let b = backend();
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let queue = b
            .create_queue(
                &ctx,
                QueueDesc {
                    flags: QueueFlags::empty(),
                },
            )
            .expect("queue");
        let a = alloc(&b, &ctx, BufferUsage::PROGRAM_INPUT, &vec![1.0; ELEMS]);
        let c = alloc(&b, &ctx, BufferUsage::PROGRAM_INPUT, &vec![2.0; ELEMS]);
        let out = alloc(&b, &ctx, BufferUsage::PROGRAM_OUTPUT, &vec![0.0; ELEMS]);
        let prog = program(&b, &ctx, KERNEL_ADD_F32);

        let short = b.submit(
            &queue,
            &prog,
            &[bind(0, &a, AccessMode::Read), bind(1, &c, AccessMode::Read)],
            Timeout::Infinite,
        );
        assert!(
            matches!(
                short,
                Err(SubmitFailure::Rejected(BackendError::InvalidArgument))
            ),
            "a missing output binding must refuse"
        );

        let wrong_access = b.submit(
            &queue,
            &prog,
            &[
                bind(0, &a, AccessMode::Read),
                bind(1, &c, AccessMode::Read),
                bind(2, &out, AccessMode::Read),
            ],
            Timeout::Infinite,
        );
        assert!(
            matches!(
                wrong_access,
                Err(SubmitFailure::Rejected(BackendError::InvalidArgument))
            ),
            "the output slot must be bound writable"
        );

        let deadline = b.submit(
            &queue,
            &prog,
            &[
                bind(0, &a, AccessMode::Read),
                bind(1, &c, AccessMode::Read),
                bind(2, &out, AccessMode::Write),
            ],
            Timeout::AfterNs(core::num::NonZeroU64::new(1_000).expect("nonzero")),
        );
        assert!(
            matches!(
                deadline,
                Err(SubmitFailure::Rejected(BackendError::Unsupported))
            ),
            "a deadline this backend cannot enforce is refused, not ignored"
        );
    }

    /// An admitted event owns its bound spans until it is terminal, so it
    /// cannot be retired out from under the launch it authorized.
    #[test]
    fn an_admitted_event_refuses_destruction_until_it_has_run() {
        let b = backend();
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let queue = b
            .create_queue(
                &ctx,
                QueueDesc {
                    flags: QueueFlags::empty(),
                },
            )
            .expect("queue");
        let a = alloc(&b, &ctx, BufferUsage::PROGRAM_INPUT, &vec![1.0; ELEMS]);
        let c = alloc(&b, &ctx, BufferUsage::PROGRAM_INPUT, &vec![2.0; ELEMS]);
        let out = alloc(&b, &ctx, BufferUsage::PROGRAM_OUTPUT, &vec![0.0; ELEMS]);
        let prog = program(&b, &ctx, KERNEL_ADD_F32);

        let event = b
            .submit(
                &queue,
                &prog,
                &[
                    bind(0, &a, AccessMode::Read),
                    bind(1, &c, AccessMode::Read),
                    bind(2, &out, AccessMode::Write),
                ],
                Timeout::Infinite,
            )
            .map_err(|_| "submit")
            .expect("admitted");

        let event = match b.destroy_event(event) {
            Err(ReleaseFailure::Rejected { error, resource }) => {
                assert!(matches!(error, BackendError::Busy));
                resource
            }
            _ => panic!("destroying an unrun event must be rejected, and hand it back"),
        };
        assert!(b.run_pending());
        b.destroy_event(event)
            .map_err(|_| "destroy")
            .expect("terminal events retire");
    }

    /// Raw admitted spans are sound only under the backend's global
    /// single-execution rule. A second admission refuses until the first has
    /// actually run; a merely terminal event does not consume that execution
    /// slot.
    #[test]
    fn only_one_execution_is_admitted_globally() {
        let b = backend();
        let (_ctx, queue, a, c, out, prog) = rig(&b);
        let bindings = || {
            [
                bind(0, &a, AccessMode::Read),
                bind(1, &c, AccessMode::Read),
                bind(2, &out, AccessMode::Write),
            ]
        };

        let first = b
            .submit(&queue, &prog, &bindings(), Timeout::Infinite)
            .map_err(|_| "first submit")
            .expect("first admission");
        assert!(matches!(
            b.submit(&queue, &prog, &bindings(), Timeout::Infinite),
            Err(SubmitFailure::Rejected(BackendError::Busy))
        ));

        assert!(b.run_pending());
        let second = b
            .submit(&queue, &prog, &bindings(), Timeout::Infinite)
            .map_err(|_| "second submit")
            .expect("terminal work no longer blocks admission");
        assert!(b.run_pending());
        b.destroy_event(first)
            .map_err(|_| "destroy first")
            .expect("first terminal event retires");
        b.destroy_event(second)
            .map_err(|_| "destroy second")
            .expect("second terminal event retires");
    }

    /// Fixture: a context, a queue, two inputs, an output, and a program.
    fn rig(
        b: &SpmdBackend,
    ) -> (
        SpmdContext,
        SpmdQueue,
        SpmdBuffer,
        SpmdBuffer,
        SpmdBuffer,
        SpmdProgram,
    ) {
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let queue = b
            .create_queue(
                &ctx,
                QueueDesc {
                    flags: QueueFlags::empty(),
                },
            )
            .expect("queue");
        let a = alloc(b, &ctx, BufferUsage::PROGRAM_INPUT, &vec![1.0; ELEMS]);
        let c = alloc(b, &ctx, BufferUsage::PROGRAM_INPUT, &vec![2.0; ELEMS]);
        let out = alloc(b, &ctx, BufferUsage::PROGRAM_OUTPUT, &vec![0.0; ELEMS]);
        let prog = program(b, &ctx, KERNEL_ADD_F32);
        (ctx, queue, a, c, out, prog)
    }

    /// **R8's path**: a geometry whose work-item count matches the bound
    /// output admits, and the launch computes the same result the
    /// geometry-free path does.
    #[test]
    fn a_carried_geometry_matching_the_output_admits_and_runs() {
        let b = backend();
        let (_ctx, queue, a, c, out, prog) = rig(&b);
        // 19 items as a 19×1×1 grid of 1×1×1 groups.
        let workgroups = Workgroups::new([ELEMS as u32, 1, 1], [1, 1, 1]).expect("geometry");

        let event = b
            .submit_with_launch(
                &queue,
                &prog,
                &[
                    bind(0, &a, AccessMode::Read),
                    bind(1, &c, AccessMode::Read),
                    bind(2, &out, AccessMode::Write),
                ],
                LaunchShape::Workgroups(workgroups),
                Timeout::Infinite,
            )
            .map_err(|_| "submit_with_launch")
            .expect("admitted");
        assert!(b.run_pending());
        assert!(matches!(b.poll_event(&event), Ok(EventState::Complete)));
        assert!(
            read_floats(&b, &out).iter().all(|&v| v == 3.0),
            "1.0 + 2.0 across the whole carried grid"
        );
    }

    /// Equal products are not equal launch shapes. Preserve all six logical
    /// axes through admission and the CPU worker scheduler into the registered
    /// kernel's `TileCtx`.
    #[test]
    fn equal_product_workgroup_shapes_reach_the_kernel_distinctly() {
        let b = SpmdBackend::new(OBSERVING_REGISTRY);
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let queue = b
            .create_queue(
                &ctx,
                QueueDesc {
                    flags: QueueFlags::empty(),
                },
            )
            .expect("queue");
        let out = alloc(&b, &ctx, BufferUsage::PROGRAM_OUTPUT, &vec![0.0; ELEMS]);
        let prog = program(&b, &ctx, KERNEL_OBSERVE_LAUNCH);

        let shapes = [
            Workgroups::new([ELEMS as u32, 1, 1], [1, 1, 1]).unwrap(),
            Workgroups::new([1, 1, 1], [ELEMS as u32, 1, 1]).unwrap(),
        ];
        let expected = [[ELEMS as u32, 1, 1, 1, 1, 1], [1, 1, 1, ELEMS as u32, 1, 1]];

        for (shape, expected) in shapes.into_iter().zip(expected) {
            let event = b
                .submit_with_launch(
                    &queue,
                    &prog,
                    &[bind(0, &out, AccessMode::Write)],
                    LaunchShape::Workgroups(shape),
                    Timeout::Infinite,
                )
                .map_err(|_| "submit_with_launch")
                .expect("same-sized explicit shape admits");
            assert!(b.run_pending());
            assert_eq!(observed_launch(&b, &out), expected);
            b.destroy_event(event)
                .map_err(|_| "destroy")
                .expect("terminal event retires");
        }
    }

    /// Deployed OP13 geometry stays exact to the CPU backend: equal products
    /// retain their distinct axes, while a client SIMD width that differs from
    /// the compiled registered kernel is refused before admission.
    #[test]
    #[allow(deprecated)]
    fn legacy_dispatch_requires_compiled_simd_and_preserves_axes() {
        let b = SpmdBackend::new(OBSERVING_REGISTRY);
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let queue = b
            .create_queue(
                &ctx,
                QueueDesc {
                    flags: QueueFlags::empty(),
                },
            )
            .expect("queue");
        let out = alloc(&b, &ctx, BufferUsage::PROGRAM_OUTPUT, &vec![0.0; ELEMS]);
        let prog = program(&b, &ctx, KERNEL_OBSERVE_LAUNCH);

        let exact = Dispatch::new([1, 1, 1], [ELEMS as u32, 1, 1], 1).unwrap();
        let event = b
            .submit_dispatch(
                &queue,
                &prog,
                &[bind(0, &out, AccessMode::Write)],
                exact,
                Timeout::Infinite,
            )
            .map_err(|_| "legacy submit")
            .expect("compiled width admits");
        assert!(b.run_pending());
        assert_eq!(observed_launch(&b, &out), [1, 1, 1, ELEMS as u32, 1, 1]);
        b.destroy_event(event)
            .map_err(|_| "destroy")
            .expect("terminal event retires");

        let wrong_simd = Dispatch::new([1, 1, 1], [ELEMS as u32, 1, 1], 2).unwrap();
        assert!(matches!(
            b.submit_dispatch(
                &queue,
                &prog,
                &[bind(0, &out, AccessMode::Write)],
                wrong_simd,
                Timeout::Infinite,
            ),
            Err(SubmitFailure::Rejected(BackendError::Incompatible))
        ));
    }

    /// A grid the bound output cannot hold is refused rather than clamped —
    /// clamping would run a different launch than the client asked for and
    /// report success.
    #[test]
    fn a_geometry_the_output_cannot_hold_is_refused() {
        let b = backend();
        let (_ctx, queue, a, c, out, prog) = rig(&b);
        let too_big = Workgroups::new([ELEMS as u32 + 1, 1, 1], [1, 1, 1]).expect("geometry");
        assert!(matches!(
            b.submit_with_launch(
                &queue,
                &prog,
                &[
                    bind(0, &a, AccessMode::Read),
                    bind(1, &c, AccessMode::Read),
                    bind(2, &out, AccessMode::Write),
                ],
                LaunchShape::Workgroups(too_big),
                Timeout::Infinite,
            ),
            Err(SubmitFailure::Rejected(BackendError::InvalidArgument))
        ));
    }

    /// Explicit one-item geometry is literal and cannot alias the separate
    /// artifact-defined mode.
    #[test]
    fn explicit_one_item_launch_does_not_mean_artifact_defined() {
        let b = backend();
        let (_ctx, queue, a, c, out, prog) = rig(&b);
        assert!(matches!(
            b.submit_with_launch(
                &queue,
                &prog,
                &[
                    bind(0, &a, AccessMode::Read),
                    bind(1, &c, AccessMode::Read),
                    bind(2, &out, AccessMode::Write),
                ],
                LaunchShape::Workgroups(Workgroups::new([1; 3], [1; 3]).unwrap()),
                Timeout::Infinite,
            ),
            Err(SubmitFailure::Rejected(BackendError::InvalidArgument))
        ));
    }

    /// A device-local request has no honest answer on a CPU row.
    #[test]
    fn device_local_memory_is_refused_rather_than_faked() {
        let b = backend();
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let device = BufferDesc::new(BYTES, 1, MemoryDomain::Device, BufferUsage::PROGRAM_INPUT)
            .expect("descriptor");
        assert!(matches!(
            b.allocate_buffer(&ctx, device),
            Err(BackendError::Unsupported)
        ));
    }

    /// Transfer permissions are a provider contract too, not merely a check
    /// supplied by the generic command service.  Keeping the check here makes
    /// direct providers and future non-virtio callers preserve the descriptor's
    /// authority boundary.
    #[test]
    fn transfers_require_declared_usage() {
        let b = backend();
        let ctx = b
            .create_context(ContextDesc {
                flags: ContextFlags::empty(),
            })
            .expect("context");
        let mut source_only = b
            .allocate_buffer(&ctx, desc(BufferUsage::TRANSFER_SOURCE))
            .expect("source buffer");
        let mut destination_only = b
            .allocate_buffer(&ctx, desc(BufferUsage::TRANSFER_DESTINATION))
            .expect("destination buffer");
        let bytes = [0x5au8; WIDE];
        let mut sink = [0u8; WIDE];

        assert_eq!(
            b.write_buffer(source_only.buffer_mut(), 0, &bytes),
            Err(BackendError::PermissionDenied)
        );
        assert_eq!(
            b.read_buffer(destination_only.buffer(), 0, &mut sink),
            Err(BackendError::PermissionDenied)
        );
        b.write_buffer(destination_only.buffer_mut(), 0, &bytes)
            .expect("write-enabled descriptor");
        b.read_buffer(source_only.buffer(), 0, &mut sink)
            .expect("read-enabled descriptor");
    }

    /// The row advertises its bounded nonzero manifest and a class the
    /// protocol has actually allocated.
    #[test]
    fn device_info_advertises_the_registered_manifest_bound() {
        let info = backend().device_info().expect("info");
        assert_eq!(
            info.validate(),
            Ok(()),
            "the generic host can admit this row"
        );
        assert_eq!(
            info.limits.max_artifact_bytes,
            REGISTERED_MANIFEST.len() as u64
        );
        assert!(REGISTERED_RESIDENT_BYTES > 0);
        assert!(<SpmdBackend as NativeAccelerator>::EXPLICIT_WORKGROUPS);
        assert_eq!(info.identity.class, AcceleratorClass::OTHER);
    }
}

impl NativeAccelerator for SpmdBackend {
    const EXPLICIT_WORKGROUPS: bool = true;
    /// Honor the deployed OP13 dialect without letting its client SIMD become
    /// the preferred portable contract. The width is accepted only when it is
    /// exactly the compiled registered kernel's lane width; only after that
    /// proof are the unchanged grid/group axes lowered to logical workgroups.
    #[allow(deprecated)]
    fn submit_dispatch(
        &self,
        _queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        dispatch: Dispatch,
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        if dispatch.is_scalar() {
            return self.admit(program, bindings, timeout, LaunchShape::ArtifactDefined);
        }
        if dispatch.simd_width() != program.kernel.lanes {
            return Err(SubmitFailure::Rejected(BackendError::Incompatible));
        }
        let workgroups =
            Workgroups::new(dispatch.grid(), dispatch.group()).map_err(SubmitFailure::Rejected)?;
        self.admit(
            program,
            bindings,
            timeout,
            LaunchShape::Workgroups(workgroups),
        )
    }
    fn submit_with_launch(
        &self,
        _queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        launch: LaunchShape,
        timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        self.admit(program, bindings, timeout, launch)
    }
}
