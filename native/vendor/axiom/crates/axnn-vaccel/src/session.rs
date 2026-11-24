//! Provider-generic dispatch: one engine over the `Accelerator` contract.
//!
//! `Session<P>` owns a context, a queue, and a shape-keyed program cache
//! for one provider; `matmul_bytes` drives the full artifact → buffers →
//! submit → readback cycle. `Hetero<C, A>` composes two sessions —
//! typically the axtosa CPU provider plus an NPU provider — and routes
//! each call by *capability first, size second*: an artifact the
//! accelerator rejects (`Unsupported`/`Incompatible`, e.g. F8 on every
//! current NPU) falls back to the CPU permanently for that dtype, and
//! small problems stay on the CPU where submission latency dominates.
//! Callers may explicitly enable host F8→FP16 expansion for sufficiently
//! large jobs when an accelerator has an FP16 compute tier but no F8 tier.
//!
//! Generic cached programs also expose a nonblocking `submit_many` / `poll_many`
//! path. The session owns invocation events and fixed provider buffers behind
//! opaque tokens, so an event loop can interleave independent graph programs
//! and feed observed completion back to `ax::AdmissionPlanner` without a host
//! worker thread or unsafe provider-handle lifetime.

use std::alloc::{Allocator, Global};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use virtio_accel_core::{
    Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
    ByteSink, ByteSource, ContextDesc, EventState, MemoryDomain, QueueDesc, SubmitFailure, Timeout,
};
use virtio_accel_tosa::Target;

use crate::artifact::{self, MatMulDType};

/// Default buffer alignment requested from providers (Core ML wants 16 KiB).
const DEFAULT_BUFFER_ALIGN: u64 = 16 * 1024;

/// Artifacts are small; ask providers to keep them fully resident.
const RESIDENT_BYTES: u64 = u64::MAX;

/// Checked host-side layout for one fixed-shape MATMUL submission.
///
/// This is intentionally computed before artifact loading or route selection:
/// malformed caller input must never populate a program cache, poison a
/// capability decision, or overflow while choosing a backend.
#[derive(Clone, Copy, Debug)]
struct MatMulLayout {
    a_elements: usize,
    b_elements: usize,
    a_bytes: usize,
    b_bytes: usize,
    output_bytes: usize,
}

fn matmul_layout(
    dt: MatMulDType,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MatMulLayout, BackendError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(BackendError::InvalidArgument);
    }
    let a_elements = m.checked_mul(k).ok_or(BackendError::InvalidArgument)?;
    let b_elements = k.checked_mul(n).ok_or(BackendError::InvalidArgument)?;
    let output_elements = m.checked_mul(n).ok_or(BackendError::InvalidArgument)?;
    let a_bytes = a_elements
        .checked_mul(dt.input_size())
        .ok_or(BackendError::InvalidArgument)?;
    let b_bytes = b_elements
        .checked_mul(dt.input_size())
        .ok_or(BackendError::InvalidArgument)?;
    let output_bytes = output_elements
        .checked_mul(dt.output_size())
        .ok_or(BackendError::InvalidArgument)?;
    Ok(MatMulLayout {
        a_elements,
        b_elements,
        a_bytes,
        b_bytes,
        output_bytes,
    })
}

fn matmul_flops(m: usize, k: usize, n: usize) -> Result<u64, BackendError> {
    u64::try_from(m)
        .ok()
        .and_then(|m| u64::try_from(k).ok().and_then(|k| m.checked_mul(k)))
        .and_then(|mk| u64::try_from(n).ok().and_then(|n| mk.checked_mul(n)))
        .and_then(|mkn| mkn.checked_mul(2))
        .ok_or(BackendError::InvalidArgument)
}

#[derive(Debug)]
struct SliceSource<'a>(&'a [u8]);

impl ByteSource for SliceSource<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, target: &mut [u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(target.len())
            .filter(|end| *end <= self.0.len())
            .ok_or(BackendError::OutOfBounds)?;
        target.copy_from_slice(&self.0[start..end]);
        Ok(())
    }

    fn as_contiguous(&self) -> Option<&[u8]> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct SliceSink<'a>(&'a mut [u8]);

impl ByteSink for SliceSink<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn write_at(&mut self, offset: u64, source: &[u8]) -> Result<(), BackendError> {
        let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
        let end = start
            .checked_add(source.len())
            .filter(|end| *end <= self.0.len())
            .ok_or(BackendError::OutOfBounds)?;
        self.0[start..end].copy_from_slice(source);
        Ok(())
    }

    fn as_contiguous_mut(&mut self) -> Option<&mut [u8]> {
        Some(self.0)
    }
}

/// Provider buffers specialized to one compiled program signature.
///
/// Programs are already cached by their shape key, so keeping their fixed-size
/// bindings alive avoids making allocation part of every submission. Host-side
/// output vectors remain owned by the caller and are necessarily created per
/// invocation.
struct IoBuffers<B> {
    inputs: Vec<B>,
    outputs: Vec<B>,
    input_bytes: Vec<usize>,
    output_bytes: Vec<usize>,
}

/// One recurrent program output that becomes an input of the next
/// invocation without leaving provider-owned storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedbackBinding {
    pub input: usize,
    pub output: usize,
}

impl FeedbackBinding {
    pub const fn new(input: usize, output: usize) -> Self {
        Self { input, output }
    }
}

/// Provider-slot recurrence topology and invocation count for one resident
/// execution.
#[derive(Clone, Copy, Debug)]
pub struct RepeatedExecution<'a> {
    pub repetitions: usize,
    pub feedback: &'a [FeedbackBinding],
}

/// Host-observed phase timings for one fixed-shape provider execution.
///
/// `submit` includes provider admission, device execution, event polling, and
/// event release. It excludes explicit host-to-device and device-to-host
/// transfers, which are measured independently as `upload` and `readback`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionTimings {
    pub preparation: Duration,
    pub upload: Duration,
    pub submit: Duration,
    pub readback: Duration,
}

impl ExecutionTimings {
    /// Complete host-observed latency across all four phases.
    pub fn total(self) -> Duration {
        self.preparation + self.upload + self.submit + self.readback
    }
}

/// Provider outputs paired with their host-observed phase timings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfiledExecution<T> {
    pub outputs: T,
    pub timings: ExecutionTimings,
}

impl<'a> RepeatedExecution<'a> {
    pub const fn new(repetitions: usize, feedback: &'a [FeedbackBinding]) -> Self {
        Self {
            repetitions,
            feedback,
        }
    }
}

/// Opaque session-owned provider invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Invocation(u64);

impl Invocation {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Submission error with an invocation token when provider acceptance was
/// indeterminate. That token must still be polled to a terminal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvocationSubmitError {
    pub error: BackendError,
    pub invocation: Option<Invocation>,
}

/// Nonblocking observation of one session-owned invocation.
#[derive(Debug)]
pub enum InvocationStatus<T> {
    Pending,
    Complete(T),
    Failed(BackendError),
    Cancelled,
}

struct PendingInvocation<E> {
    key: u64,
    event: E,
}

struct RepeatedIoBuffers<B> {
    io: IoBuffers<B>,
    feedback: Vec<FeedbackBinding>,
}

/// One provider, one context/queue, and a MATMUL program cache.
pub struct Session<P: Accelerator> {
    provider: P,
    target: Target,
    buffer_alignment: u64,
    context: Option<P::Context>,
    queue: Option<P::Queue>,
    programs: HashMap<(u8, usize, usize, usize), P::Program>,
    generic: HashMap<u64, P::Program>,
    program_io: HashMap<(u8, usize, usize, usize), IoBuffers<P::Buffer>>,
    generic_io: HashMap<u64, IoBuffers<P::Buffer>>,
    repeated_io: HashMap<u64, RepeatedIoBuffers<P::Buffer>>,
    next_invocation: u64,
    pending: HashMap<Invocation, PendingInvocation<P::Event>>,
    generic_in_flight: HashSet<u64>,
    /// Handles whose explicit cache eviction was rejected by the provider.
    /// They are never rebound; retaining ownership until session teardown is
    /// safer than dropping a still-live native handle or making a partially
    /// released fixed-I/O signature reusable.
    quarantined_buffers: Vec<P::Buffer>,
    quarantined_programs: Vec<P::Program>,
}

fn allocate_io<P: Accelerator>(
    provider: &P,
    context: &P::Context,
    buffer_alignment: u64,
    input_bytes: &[usize],
    output_bytes: &[usize],
) -> Result<IoBuffers<P::Buffer>, BackendError> {
    allocate_io_with_feedback(
        provider,
        context,
        buffer_alignment,
        input_bytes,
        output_bytes,
        &[],
    )
}

fn allocate_io_with_feedback<P: Accelerator>(
    provider: &P,
    context: &P::Context,
    buffer_alignment: u64,
    input_bytes: &[usize],
    output_bytes: &[usize],
    feedback: &[FeedbackBinding],
) -> Result<IoBuffers<P::Buffer>, BackendError> {
    validate_feedback(input_bytes, output_bytes, feedback)?;
    let mut inputs = Vec::with_capacity(input_bytes.len());
    let mut outputs = Vec::with_capacity(output_bytes.len());
    let allocated = (|| {
        for (slot, &bytes) in input_bytes.iter().enumerate() {
            let recurrent = feedback.iter().any(|binding| binding.input == slot);
            let usage = BufferUsage::TRANSFER_DESTINATION
                | BufferUsage::PROGRAM_INPUT
                | if recurrent {
                    BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT
                } else {
                    BufferUsage::empty()
                };
            let desc =
                BufferDesc::new(bytes as u64, buffer_alignment, MemoryDomain::Shared, usage)?;
            inputs.push(provider.allocate_buffer(context, desc)?.into_parts().0);
        }
        for (slot, &bytes) in output_bytes.iter().enumerate() {
            let recurrent = feedback.iter().any(|binding| binding.output == slot);
            let usage = BufferUsage::TRANSFER_SOURCE
                | BufferUsage::PROGRAM_OUTPUT
                | if recurrent {
                    BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT
                } else {
                    BufferUsage::empty()
                };
            let desc =
                BufferDesc::new(bytes as u64, buffer_alignment, MemoryDomain::Shared, usage)?;
            outputs.push(provider.allocate_buffer(context, desc)?.into_parts().0);
        }
        Ok(())
    })();

    if let Err(error) = allocated {
        for buffer in inputs {
            let _ = provider.free_buffer(buffer);
        }
        for buffer in outputs {
            let _ = provider.free_buffer(buffer);
        }
        return Err(error);
    }

    Ok(IoBuffers {
        inputs,
        outputs,
        input_bytes: input_bytes.to_vec(),
        output_bytes: output_bytes.to_vec(),
    })
}

fn validate_feedback(
    input_bytes: &[usize],
    output_bytes: &[usize],
    feedback: &[FeedbackBinding],
) -> Result<(), BackendError> {
    for (index, binding) in feedback.iter().enumerate() {
        let Some(&input_bytes) = input_bytes.get(binding.input) else {
            return Err(BackendError::InvalidArgument);
        };
        let Some(&output_bytes) = output_bytes.get(binding.output) else {
            return Err(BackendError::InvalidArgument);
        };
        if input_bytes != output_bytes
            || feedback[..index]
                .iter()
                .any(|prior| prior.input == binding.input || prior.output == binding.output)
        {
            return Err(BackendError::InvalidArgument);
        }
    }
    Ok(())
}

fn write_inputs<P: Accelerator>(
    provider: &P,
    io: &mut IoBuffers<P::Buffer>,
    inputs: &[&[u8]],
) -> Result<(), BackendError> {
    if inputs.len() != io.inputs.len()
        || inputs
            .iter()
            .zip(&io.input_bytes)
            .any(|(data, &bytes)| data.len() != bytes)
    {
        return Err(BackendError::InvalidArgument);
    }
    for (buffer, data) in io.inputs.iter_mut().zip(inputs) {
        provider.write_buffer(buffer, 0, &SliceSource(data))?;
    }
    Ok(())
}

fn submit_io_event<P: Accelerator>(
    provider: &P,
    queue: &P::Queue,
    program: &P::Program,
    io: &IoBuffers<P::Buffer>,
) -> Result<P::Event, SubmitFailure<P::Event>> {
    let range = |bytes: usize| BufferRange::new(0, bytes as u64);
    let mut bindings = Vec::with_capacity(io.inputs.len() + io.outputs.len());
    for (slot, (buffer, &bytes)) in io.inputs.iter().zip(&io.input_bytes).enumerate() {
        let range = range(bytes).map_err(SubmitFailure::Rejected)?;
        bindings.push(BindingRef {
            slot: slot as u32,
            buffer,
            range,
            access: AccessMode::Read,
        });
    }
    for (buffer, &bytes) in io.outputs.iter().zip(&io.output_bytes) {
        let range = range(bytes).map_err(SubmitFailure::Rejected)?;
        bindings.push(BindingRef {
            slot: bindings.len() as u32,
            buffer,
            range,
            access: AccessMode::Write,
        });
    }

    provider.submit(queue, program, &bindings, Timeout::Infinite)
}

fn wait_event<P: Accelerator>(provider: &P, event: P::Event) -> Result<(), BackendError> {
    let state = loop {
        match provider.poll_event(&event) {
            Ok(EventState::Pending) => std::thread::yield_now(),
            Ok(terminal) => break terminal,
            // Poll errors leave the event live and are retryable unless the
            // provider itself is gone. The synchronous compatibility path
            // therefore keeps ownership instead of destroying a potentially
            // pending invocation.
            Err(BackendError::DeviceLost) => return Err(BackendError::DeviceLost),
            Err(_) => std::thread::yield_now(),
        }
    };
    let mut event = event;
    loop {
        match provider.destroy_event(event) {
            Ok(()) => break,
            Err(virtio_accel_core::ReleaseFailure::Rejected { resource, .. }) => {
                event = resource;
                std::thread::yield_now();
            }
            Err(virtio_accel_core::ReleaseFailure::Indeterminate { error }) => {
                return Err(error);
            }
        }
    }
    match state {
        EventState::Complete => Ok(()),
        EventState::Failed(error) => Err(error),
        EventState::Cancelled => Err(BackendError::DeadlineExpired),
        EventState::Pending => Err(BackendError::DeviceLost),
    }
}

fn submit_io<P: Accelerator>(
    provider: &P,
    queue: &P::Queue,
    program: &P::Program,
    io: &IoBuffers<P::Buffer>,
) -> Result<(), BackendError> {
    match submit_io_event(provider, queue, program, io) {
        Ok(event) => wait_event(provider, event),
        Err(SubmitFailure::Rejected(error)) => Err(error),
        Err(SubmitFailure::Indeterminate { event, .. }) => {
            // The event, not the submission return, now owns the invocation.
            // Drive it to a terminal state before exposing a synchronous
            // result or allowing its bound buffers to be reused.
            wait_event(provider, event)
        }
    }
}

fn read_outputs<P: Accelerator, A: Allocator + Clone>(
    provider: &P,
    io: &IoBuffers<P::Buffer>,
    allocator: A,
) -> Result<Vec<Vec<u8, A>, A>, BackendError> {
    read_output_buffers(provider, &io.outputs, &io.output_bytes, allocator)
}

fn read_output_buffers<P: Accelerator, A: Allocator + Clone>(
    provider: &P,
    buffers: &[P::Buffer],
    sizes: &[usize],
    allocator: A,
) -> Result<Vec<Vec<u8, A>, A>, BackendError> {
    let mut outputs = Vec::new_in(allocator.clone());
    outputs
        .try_reserve_exact(buffers.len())
        .map_err(|_| BackendError::OutOfMemory)?;
    for (buffer, &bytes) in buffers.iter().zip(sizes) {
        let mut output = Vec::new_in(allocator.clone());
        output
            .try_reserve_exact(bytes)
            .map_err(|_| BackendError::OutOfMemory)?;
        output.resize(bytes, 0);
        let mut sink = SliceSink(&mut output);
        provider.read_buffer(buffer, 0, &mut sink)?;
        outputs.push(output);
    }
    Ok(outputs)
}

fn execute_io_in<P: Accelerator, A: Allocator + Clone>(
    provider: &P,
    queue: &P::Queue,
    program: &P::Program,
    io: &mut IoBuffers<P::Buffer>,
    inputs: &[&[u8]],
    allocator: A,
) -> Result<Vec<Vec<u8, A>, A>, BackendError> {
    write_inputs(provider, io, inputs)?;
    submit_io(provider, queue, program, io)?;
    read_outputs(provider, io, allocator)
}

fn release_io<P: Accelerator>(provider: &P, io: IoBuffers<P::Buffer>) {
    for buffer in io.inputs {
        let _ = provider.free_buffer(buffer);
    }
    for buffer in io.outputs {
        let _ = provider.free_buffer(buffer);
    }
}

fn release_io_for_eviction<P: Accelerator>(
    provider: &P,
    io: IoBuffers<P::Buffer>,
    quarantine: &mut Vec<P::Buffer>,
    first_error: &mut Option<BackendError>,
) {
    for buffer in io.inputs.into_iter().chain(io.outputs) {
        match provider.free_buffer(buffer) {
            Ok(()) => {}
            Err(virtio_accel_core::ReleaseFailure::Rejected { error, resource }) => {
                first_error.get_or_insert(error);
                quarantine.push(resource);
            }
            Err(virtio_accel_core::ReleaseFailure::Indeterminate { error }) => {
                first_error.get_or_insert(error);
            }
        }
    }
}

impl<P: Accelerator> Session<P> {
    /// Stand up a context and queue on `provider`. `target` is the
    /// provider's advertised TOSA target (`AxTosaAccelerator::target()`,
    /// `COREML_TOSA_TARGET`, …); artifacts are stamped with it.
    pub fn new(provider: P, target: Target) -> Result<Self, BackendError> {
        Self::new_with_buffer_alignment(provider, target, DEFAULT_BUFFER_ALIGN)
    }

    /// Stand up a session with an explicit provider buffer-alignment request.
    ///
    /// Core ML uses the 16 KiB default. Providers whose native allocator has a
    /// different proven alignment, such as XDNA/HRX's 4 KiB pages, select it
    /// explicitly rather than weakening another provider's contract.
    pub fn new_with_buffer_alignment(
        provider: P,
        target: Target,
        buffer_alignment: u64,
    ) -> Result<Self, BackendError> {
        if buffer_alignment == 0 || !buffer_alignment.is_power_of_two() {
            return Err(BackendError::InvalidArgument);
        }
        let context = provider.create_context(ContextDesc::default())?;
        let queue = provider.create_queue(&context, QueueDesc::default())?;
        Ok(Self {
            provider,
            target,
            buffer_alignment,
            context: Some(context),
            queue: Some(queue),
            programs: HashMap::new(),
            generic: HashMap::new(),
            program_io: HashMap::new(),
            generic_io: HashMap::new(),
            repeated_io: HashMap::new(),
            next_invocation: 0,
            pending: HashMap::new(),
            generic_in_flight: HashSet::new(),
            quarantined_buffers: Vec::new(),
            quarantined_programs: Vec::new(),
        })
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Exact TOSA target stamped into artifacts loaded by this session.
    pub const fn target(&self) -> Target {
        self.target
    }

    /// Alignment requested for every provider-owned I/O buffer.
    pub const fn buffer_alignment(&self) -> u64 {
        self.buffer_alignment
    }

    fn context(&self) -> &P::Context {
        self.context.as_ref().expect("context lives until drop")
    }

    /// Parse artifact bytes, stamp them with this session's target, and
    /// load them on the provider.
    fn load(&self, bytes: &[u8]) -> Result<P::Program, BackendError> {
        let model = virtio_accel_tosa::parse(bytes).map_err(|_| BackendError::InvalidArgument)?;
        let artifact = model
            .artifact_ref(self.target, RESIDENT_BYTES)
            .map_err(|_| BackendError::Incompatible)?;
        self.provider.load_program(self.context(), artifact)
    }

    /// Get or build+load an arbitrary TOSA program under a caller-chosen
    /// cache key. `build` runs only on a cache miss.
    pub fn ensure(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
    ) -> Result<(), BackendError> {
        if !self.generic.contains_key(&key) {
            let program = self.load(&build())?;
            self.generic.insert(key, program);
        }
        Ok(())
    }

    /// The provider program loaded under a generic cache key by
    /// [`ensure`](Self::ensure), for provider-specific inspection such as
    /// exporting its compiled form. Borrowed only; the session keeps ownership.
    pub fn generic_program(&self, key: u64) -> Option<&P::Program> {
        self.generic.get(&key)
    }

    /// Evict one generic fixed-shape program and all of its retained I/O.
    ///
    /// Graph executables use unique session keys, so their owner can bound a
    /// viewport/shape cache without keeping every resized framebuffer resident
    /// until the whole session is dropped. In-flight keys are refused. A
    /// provider-rejected release is quarantined and never reused; teardown
    /// receives one final best-effort release attempt.
    pub(crate) fn evict_generic(&mut self, key: u64) -> Result<(), BackendError> {
        if self.generic_in_flight.contains(&key) {
            return Err(BackendError::Busy);
        }

        let mut first_error = None;
        if let Some(io) = self.generic_io.remove(&key) {
            release_io_for_eviction(
                &self.provider,
                io,
                &mut self.quarantined_buffers,
                &mut first_error,
            );
        }
        if let Some(resident) = self.repeated_io.remove(&key) {
            release_io_for_eviction(
                &self.provider,
                resident.io,
                &mut self.quarantined_buffers,
                &mut first_error,
            );
        }
        if let Some(program) = self.generic.remove(&key) {
            match self.provider.unload_program(program) {
                Ok(()) => {}
                Err(virtio_accel_core::ReleaseFailure::Rejected { error, resource }) => {
                    first_error.get_or_insert(error);
                    self.quarantined_programs.push(resource);
                }
                Err(virtio_accel_core::ReleaseFailure::Indeterminate { error }) => {
                    first_error.get_or_insert(error);
                }
            }
        }

        first_error.map_or(Ok(()), Err)
    }

    fn prepare_generic_io(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        output_bytes: &[usize],
    ) -> Result<(), BackendError> {
        if self.generic_in_flight.contains(&key) {
            return Err(BackendError::Busy);
        }
        self.ensure(key, build)?;
        match self.generic_io.get(&key) {
            Some(io)
                if io.input_bytes.len() != inputs.len()
                    || io
                        .input_bytes
                        .iter()
                        .zip(inputs)
                        .any(|(&bytes, data)| bytes != data.len())
                    || io.output_bytes.as_slice() != output_bytes =>
            {
                Err(BackendError::InvalidArgument)
            }
            Some(_) => Ok(()),
            None => {
                // The signature becomes session-owned on the first run. Warm
                // dispatches compare against it in place and allocate no
                // temporary input-length vector.
                let mut input_bytes = Vec::with_capacity(inputs.len());
                input_bytes.extend(inputs.iter().map(|data| data.len()));
                let io = allocate_io(
                    &self.provider,
                    self.context(),
                    self.buffer_alignment,
                    &input_bytes,
                    output_bytes,
                )?;
                self.generic_io.insert(key, io);
                Ok(())
            }
        }
    }

    fn track_invocation(&mut self, key: u64, event: P::Event) -> Invocation {
        let invocation = loop {
            let invocation = Invocation(self.next_invocation);
            self.next_invocation = self.next_invocation.wrapping_add(1);
            if !self.pending.contains_key(&invocation) {
                break invocation;
            }
        };
        self.generic_in_flight.insert(key);
        self.pending
            .insert(invocation, PendingInvocation { key, event });
        invocation
    }

    /// Admit a cached generic program without waiting for completion.
    ///
    /// Inputs are copied to retained provider buffers before submission. The
    /// returned token owns no provider handle; the session keeps the event,
    /// program, and bound buffers alive until [`poll_many_in`](Self::poll_many_in)
    /// observes a terminal state. Reusing the same program key while its
    /// fixed I/O buffers are in flight returns `BackendError::Busy`.
    pub fn submit_many(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        output_bytes: &[usize],
    ) -> Result<Invocation, InvocationSubmitError> {
        self.prepare_generic_io(key, build, inputs, output_bytes)
            .map_err(|error| InvocationSubmitError {
                error,
                invocation: None,
            })?;
        {
            let io = self.generic_io.get_mut(&key).expect("I/O was ensured");
            write_inputs(&self.provider, io, inputs).map_err(|error| InvocationSubmitError {
                error,
                invocation: None,
            })?;
        }
        let submitted = {
            let queue = self.queue.as_ref().expect("queue lives until drop");
            let program = self.generic.get(&key).expect("program was ensured");
            let io = self.generic_io.get(&key).expect("I/O was ensured");
            submit_io_event(&self.provider, queue, program, io)
        };
        match submitted {
            Ok(event) => Ok(self.track_invocation(key, event)),
            Err(SubmitFailure::Rejected(error)) => Err(InvocationSubmitError {
                error,
                invocation: None,
            }),
            Err(SubmitFailure::Indeterminate { error, event }) => {
                let invocation = self.track_invocation(key, event);
                Err(InvocationSubmitError {
                    error,
                    invocation: Some(invocation),
                })
            }
        }
    }

    /// Poll one admitted invocation and materialize outputs only after its
    /// event becomes terminal. Pending polls allocate no output planes.
    #[allow(clippy::type_complexity)]
    pub fn poll_many_in<A: Allocator + Clone>(
        &mut self,
        invocation: Invocation,
        allocator: A,
    ) -> Result<InvocationStatus<Vec<Vec<u8, A>, A>>, BackendError> {
        self.poll_many_with(invocation, |provider, buffers, sizes| {
            read_output_buffers(provider, buffers, sizes, allocator)
        })
    }

    /// Poll into caller-owned byte sinks, including segmented/pitched outputs.
    /// Sizes are checked before polling or consuming the token. Pending,
    /// failed and cancelled events never write destinations. A readback error
    /// can leave partial output: publish only `Complete(())`. Sinks are never
    /// retained by the provider or session beyond this call.
    pub fn poll_many_into(
        &mut self,
        invocation: Invocation,
        outputs: &mut [&mut dyn ByteSink],
    ) -> Result<InvocationStatus<()>, BackendError> {
        let pending = self
            .pending
            .get(&invocation)
            .ok_or(BackendError::InvalidArgument)?;
        let io = self
            .generic_io
            .get(&pending.key)
            .ok_or(BackendError::InvalidArgument)?;
        if outputs.len() != io.outputs.len()
            || outputs
                .iter()
                .zip(&io.output_bytes)
                .any(|(output, &bytes)| output.len() != bytes as u64)
        {
            return Err(BackendError::InvalidArgument);
        }
        self.poll_many_with(invocation, |provider, buffers, _| {
            for (buffer, output) in buffers.iter().zip(outputs) {
                provider.read_buffer(buffer, 0, *output)?;
            }
            Ok(())
        })
    }

    pub(crate) fn poll_many_with<T>(
        &mut self,
        invocation: Invocation,
        read: impl FnOnce(&P, &[P::Buffer], &[usize]) -> Result<T, BackendError>,
    ) -> Result<InvocationStatus<T>, BackendError> {
        let state = {
            let pending = self
                .pending
                .get(&invocation)
                .ok_or(BackendError::InvalidArgument)?;
            self.provider.poll_event(&pending.event)?
        };
        if state == EventState::Pending {
            return Ok(InvocationStatus::Pending);
        }

        let PendingInvocation { key, event } = self
            .pending
            .remove(&invocation)
            .expect("polled invocation remained present");
        if let Err(failure) = self.provider.destroy_event(event) {
            let error = failure.error();
            match failure {
                virtio_accel_core::ReleaseFailure::Rejected { resource, .. } => {
                    self.pending.insert(
                        invocation,
                        PendingInvocation {
                            key,
                            event: resource,
                        },
                    );
                }
                virtio_accel_core::ReleaseFailure::Indeterminate { .. } => {
                    // The event handle is invalid, but its relationship to
                    // the bound buffers is unknown until device recovery.
                    // Keep this fixed-I/O key poisoned instead of permitting
                    // unsafe reuse.
                }
            }
            return Err(error);
        }
        self.generic_in_flight.remove(&key);
        match state {
            EventState::Complete => {
                let io = self.generic_io.get(&key).expect("in-flight I/O exists");
                read(&self.provider, &io.outputs, &io.output_bytes).map(InvocationStatus::Complete)
            }
            EventState::Failed(error) => Ok(InvocationStatus::Failed(error)),
            EventState::Cancelled => Ok(InvocationStatus::Cancelled),
            EventState::Pending => unreachable!(),
        }
    }

    /// Global-allocator form of [`poll_many_in`](Self::poll_many_in).
    pub fn poll_many(
        &mut self,
        invocation: Invocation,
    ) -> Result<InvocationStatus<Vec<Vec<u8>>>, BackendError> {
        self.poll_many_in(invocation, Global)
    }

    /// Request bounded nonblocking cancellation. The token remains live and
    /// must still be polled so the terminal event can be destroyed safely.
    pub fn cancel(&self, invocation: Invocation) -> Result<(), BackendError> {
        let pending = self
            .pending
            .get(&invocation)
            .ok_or(BackendError::InvalidArgument)?;
        self.provider.cancel_event(&pending.event)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Whether `invocation` still names a provider event owned by this
    /// session. Higher-level progress machines use this after a polling error
    /// to distinguish a retryable observation failure from an indeterminate
    /// event release that invalidated the token.
    pub fn is_pending(&self, invocation: Invocation) -> bool {
        self.pending.contains_key(&invocation)
    }

    /// Run the cached program `key` (loading via `build` on first use):
    /// `inputs` bind to slots `0..n` in block-input order, and the
    /// single block output binds to slot `n`, returned as raw bytes.
    pub fn run(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        out_bytes: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let mut outputs = self.run_many(key, build, inputs, &[out_bytes])?;
        Ok(outputs.pop().expect("one requested output"))
    }

    /// Multi-output form of [`Session::run`], used by scheduled `ax`
    /// partitions whose boundary exposes more than one live tensor.
    pub fn run_many(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        output_bytes: &[usize],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.run_many_in(key, build, inputs, output_bytes, Global)
    }

    /// Timed multi-output execution using the same retained programs and I/O
    /// buffers as [`run_many`](Self::run_many).
    ///
    /// This is an evidence surface, not a different execution route: cold
    /// preparation may allocate provider buffers, while warm preparation is a
    /// signature/cache check. Artifact construction and program loading can be
    /// timed separately by calling [`ensure`](Self::ensure) first.
    pub fn run_many_profiled(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        output_bytes: &[usize],
    ) -> Result<ProfiledExecution<Vec<Vec<u8>>>, BackendError> {
        self.run_many_profiled_in(key, build, inputs, output_bytes, Global)
    }

    /// Allocator-aware form of [`run_many_profiled`](Self::run_many_profiled).
    pub fn run_many_profiled_in<A: Allocator + Clone>(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        output_bytes: &[usize],
        allocator: A,
    ) -> Result<ProfiledExecution<Vec<Vec<u8, A>, A>>, BackendError> {
        let started = Instant::now();
        self.prepare_generic_io(key, build, inputs, output_bytes)?;
        let preparation = started.elapsed();

        let provider = &self.provider;
        let queue = self.queue.as_ref().expect("queue lives until drop");
        let program = self.generic.get(&key).expect("program was ensured");
        let io = self.generic_io.get_mut(&key).expect("I/O was ensured");

        let started = Instant::now();
        write_inputs(provider, io, inputs)?;
        let upload = started.elapsed();

        let started = Instant::now();
        submit_io(provider, queue, program, io)?;
        let submit = started.elapsed();

        let started = Instant::now();
        let outputs = read_outputs(provider, io, allocator)?;
        let readback = started.elapsed();

        Ok(ProfiledExecution {
            outputs,
            timings: ExecutionTimings {
                preparation,
                upload,
                submit,
                readback,
            },
        })
    }

    /// Allocator-aware form of [`Session::run_many`]. Provider buffers are
    /// retained by the session; only the caller-owned output collection and
    /// byte planes use `allocator`.
    pub fn run_many_in<A: Allocator + Clone>(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        output_bytes: &[usize],
        allocator: A,
    ) -> Result<Vec<Vec<u8, A>, A>, BackendError> {
        self.prepare_generic_io(key, build, inputs, output_bytes)?;

        let provider = &self.provider;
        let queue = self.queue.as_ref().expect("queue lives until drop");
        let program = self.generic.get(&key).expect("program was ensured");
        let io = self.generic_io.get_mut(&key).expect("I/O was ensured");
        execute_io_in(provider, queue, program, io, inputs, allocator)
    }

    /// Execute a fixed-shape program repeatedly while selected outputs feed
    /// the next invocation's inputs entirely through provider-owned buffers.
    ///
    /// Initial inputs are transferred once and final outputs are read once.
    /// Each feedback binding swaps the matching input/output buffer handles
    /// after a non-final invocation, avoiding host materialization of recurrent
    /// state between submissions. The first call fixes the cache key's
    /// signature and feedback topology.
    pub fn run_many_repeated_in<A: Allocator + Clone>(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        output_bytes: &[usize],
        repeated: RepeatedExecution<'_>,
        allocator: A,
    ) -> Result<Vec<Vec<u8, A>, A>, BackendError> {
        self.run_many_repeated_with(
            key,
            build,
            inputs,
            output_bytes,
            repeated,
            |provider, buffers, sizes| read_output_buffers(provider, buffers, sizes, allocator),
        )
    }

    pub(crate) fn run_many_repeated_with<T>(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        output_bytes: &[usize],
        repeated: RepeatedExecution<'_>,
        read: impl FnOnce(&P, &[P::Buffer], &[usize]) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        if repeated.repetitions == 0 {
            return Err(BackendError::InvalidArgument);
        }
        self.ensure(key, build)?;

        match self.repeated_io.get(&key) {
            Some(resident)
                if resident.io.input_bytes.len() != inputs.len()
                    || resident
                        .io
                        .input_bytes
                        .iter()
                        .zip(inputs)
                        .any(|(&bytes, data)| bytes != data.len())
                    || resident.io.output_bytes.as_slice() != output_bytes
                    || resident.feedback.as_slice() != repeated.feedback =>
            {
                return Err(BackendError::InvalidArgument);
            }
            Some(_) => {}
            None => {
                let mut input_bytes = Vec::with_capacity(inputs.len());
                input_bytes.extend(inputs.iter().map(|data| data.len()));
                let io = allocate_io_with_feedback(
                    &self.provider,
                    self.context(),
                    self.buffer_alignment,
                    &input_bytes,
                    output_bytes,
                    repeated.feedback,
                )?;
                self.repeated_io.insert(
                    key,
                    RepeatedIoBuffers {
                        io,
                        feedback: repeated.feedback.to_vec(),
                    },
                );
            }
        }

        let provider = &self.provider;
        let queue = self.queue.as_ref().expect("queue lives until drop");
        let program = self.generic.get(&key).expect("program was ensured");
        let resident = self.repeated_io.get_mut(&key).expect("I/O was ensured");
        write_inputs(provider, &mut resident.io, inputs)?;
        for invocation in 0..repeated.repetitions {
            submit_io(provider, queue, program, &resident.io)?;
            if invocation + 1 != repeated.repetitions {
                let (input_buffers, output_buffers) =
                    (&mut resident.io.inputs, &mut resident.io.outputs);
                for binding in &resident.feedback {
                    core::mem::swap(
                        &mut input_buffers[binding.input],
                        &mut output_buffers[binding.output],
                    );
                }
            }
        }
        read(provider, &resident.io.outputs, &resident.io.output_bytes)
    }

    /// Global-allocator form of [`Session::run_many_repeated_in`].
    pub fn run_many_repeated(
        &mut self,
        key: u64,
        build: impl FnOnce() -> Vec<u8>,
        inputs: &[&[u8]],
        output_bytes: &[usize],
        repetitions: usize,
        feedback: &[FeedbackBinding],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.run_many_repeated_in(
            key,
            build,
            inputs,
            output_bytes,
            RepeatedExecution::new(repetitions, feedback),
            Global,
        )
    }

    /// Get or build+load the MATMUL program for `(dt, m, k, n)`.
    fn program(
        &mut self,
        dt: MatMulDType,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<&P::Program, BackendError> {
        matmul_layout(dt, m, k, n)?;
        let key = (dt.tag(), m, k, n);
        if !self.programs.contains_key(&key) {
            let artifact =
                artifact::matmul(dt, m, k, n).map_err(|_| BackendError::InvalidArgument)?;
            let program = self.load(&artifact)?;
            self.programs.insert(key, program);
        }
        Ok(&self.programs[&key])
    }

    /// Build and load the MATMUL program for `(dt, m, k, n)` without
    /// running it — the capability probe `Hetero` routes on.
    pub fn ensure_program(
        &mut self,
        dt: MatMulDType,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        self.program(dt, m, k, n).map(|_| ())
    }

    /// Run `out[m, n] = a[m, k] × b[k, n]` on the provider. `a` and `b`
    /// are little-endian planes of `dt`; the result plane is `dt`'s
    /// output type (f32 for F32, f16 otherwise).
    pub fn matmul_bytes(
        &mut self,
        dt: MatMulDType,
        m: usize,
        k: usize,
        n: usize,
        a: &[u8],
        b: &[u8],
    ) -> Result<Vec<u8>, BackendError> {
        let layout = matmul_layout(dt, m, k, n)?;
        if a.len() != layout.a_bytes {
            return Err(BackendError::InvalidArgument);
        }
        if b.len() != layout.b_bytes {
            return Err(BackendError::InvalidArgument);
        }
        let key = (dt.tag(), m, k, n);
        self.program(dt, m, k, n)?;
        if !self.program_io.contains_key(&key) {
            let io = allocate_io(
                &self.provider,
                self.context(),
                self.buffer_alignment,
                &[a.len(), b.len()],
                &[layout.output_bytes],
            )?;
            self.program_io.insert(key, io);
        }

        let provider = &self.provider;
        let queue = self.queue.as_ref().expect("queue lives until drop");
        let program = self.programs.get(&key).expect("program was ensured");
        let io = self.program_io.get_mut(&key).expect("I/O was ensured");
        let mut outputs = execute_io_in(provider, queue, program, io, &[a, b], Global)?;
        Ok(outputs.pop().expect("one requested output"))
    }

    /// F32 convenience wrapper over [`Session::matmul_bytes`].
    pub fn matmul_f32(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, BackendError> {
        let as_bytes = |plane: &[f32]| {
            plane
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let out = self.matmul_bytes(MatMulDType::F32, m, k, n, &as_bytes(a), &as_bytes(b))?;
        Ok(out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect())
    }
}

impl<P: Accelerator> Drop for Session<P> {
    fn drop(&mut self) {
        // Best-effort teardown in child-before-parent order; release
        // failures at drop time are unrecoverable and intentionally
        // ignored.
        for (_, pending) in std::mem::take(&mut self.pending) {
            let _ = wait_event(&self.provider, pending.event);
        }
        self.generic_in_flight.clear();
        for (_, io) in self.program_io.drain() {
            release_io(&self.provider, io);
        }
        for (_, io) in self.generic_io.drain() {
            release_io(&self.provider, io);
        }
        for (_, resident) in self.repeated_io.drain() {
            release_io(&self.provider, resident.io);
        }
        for buffer in self.quarantined_buffers.drain(..) {
            let _ = self.provider.free_buffer(buffer);
        }
        for (_, program) in self.programs.drain() {
            let _ = self.provider.unload_program(program);
        }
        for (_, program) in self.generic.drain() {
            let _ = self.provider.unload_program(program);
        }
        for program in self.quarantined_programs.drain(..) {
            let _ = self.provider.unload_program(program);
        }
        if let Some(queue) = self.queue.take() {
            let _ = self.provider.destroy_queue(queue);
        }
        if let Some(context) = self.context.take() {
            let _ = self.provider.destroy_context(context);
        }
    }
}

/// Which session serviced a heterogeneous call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Cpu,
    Accel,
}

/// Capability- and size-aware router over a CPU session plus an
/// optional accelerator session.
pub struct Hetero<C: Accelerator, A: Accelerator> {
    pub cpu: Session<C>,
    pub accel: Option<Session<A>>,
    /// dtypes the accelerator has rejected (`Unsupported`/`Incompatible`).
    rejected: HashSet<u8>,
    /// Minimum `2·m·k·n` FLOP count worth the accelerator round trip.
    pub min_accel_flops: u64,
    /// Minimum FLOP count at which rejected F8 inputs may be expanded on
    /// the host and submitted as FP16. `u64::MAX` (the default) disables
    /// promotion, preserving native F8 execution and its exact semantics.
    pub min_fp8_promotion_flops: u64,
}

impl<C: Accelerator, A: Accelerator> Hetero<C, A> {
    pub fn new(cpu: Session<C>, accel: Option<Session<A>>) -> Self {
        Self {
            cpu,
            accel,
            rejected: HashSet::new(),
            min_accel_flops: 0,
            min_fp8_promotion_flops: u64::MAX,
        }
    }

    /// Route one MATMUL: try the accelerator when it is present, has not
    /// rejected this dtype, and the problem is large enough; otherwise
    /// (or on a capability rejection) run on the CPU session.
    ///
    /// Capability is judged at *program load*: any load failure short of
    /// resource exhaustion or device loss means "this provider cannot
    /// take this program" and pins the dtype to the CPU. Errors after a
    /// successful load are real faults and propagate. If F8 promotion is
    /// enabled, a rejected B8/F8 program is expanded losslessly into
    /// FP16 inputs and retried through the accelerator's FP16 tier.
    pub fn matmul_bytes(
        &mut self,
        dt: MatMulDType,
        m: usize,
        k: usize,
        n: usize,
        a: &[u8],
        b: &[u8],
    ) -> Result<(Route, Vec<u8>), BackendError> {
        let layout = matmul_layout(dt, m, k, n)?;
        if a.len() != layout.a_bytes || b.len() != layout.b_bytes {
            return Err(BackendError::InvalidArgument);
        }
        let flops = matmul_flops(m, k, n)?;
        if let Some(accel) = &mut self.accel
            && flops >= self.min_accel_flops
        {
            if !self.rejected.contains(&dt.tag()) {
                match accel.ensure_program(dt, m, k, n) {
                    Ok(()) => {
                        let out = accel.matmul_bytes(dt, m, k, n, a, b)?;
                        return Ok((Route::Accel, out));
                    }
                    Err(
                        error @ (BackendError::OutOfMemory
                        | BackendError::ResourceLimit
                        | BackendError::DeviceLost),
                    ) => return Err(error),
                    Err(_) => {
                        self.rejected.insert(dt.tag());
                    }
                }
            }

            if flops >= self.min_fp8_promotion_flops
                && matches!(dt, MatMulDType::B8 | MatMulDType::F8)
                && !self.rejected.contains(&MatMulDType::F16.tag())
            {
                match accel.ensure_program(MatMulDType::F16, m, k, n) {
                    Ok(()) => {
                        let a = expand_fp8_to_f16(dt, a, layout.a_elements)?;
                        let b = expand_fp8_to_f16(dt, b, layout.b_elements)?;
                        let out = accel.matmul_bytes(MatMulDType::F16, m, k, n, &a, &b)?;
                        return Ok((Route::Accel, out));
                    }
                    Err(
                        error @ (BackendError::OutOfMemory
                        | BackendError::ResourceLimit
                        | BackendError::DeviceLost),
                    ) => return Err(error),
                    Err(_) => {
                        self.rejected.insert(MatMulDType::F16.tag());
                    }
                }
            }
        }
        let out = self.cpu.matmul_bytes(dt, m, k, n, a, b)?;
        Ok((Route::Cpu, out))
    }

    /// Whether the accelerator rejected the native `dt` program. An F8
    /// dtype can still route through explicit FP16 promotion.
    pub fn is_rejected(&self, dt: MatMulDType) -> bool {
        self.rejected.contains(&dt.tag())
    }
}

fn expand_fp8_to_f16(
    dt: MatMulDType,
    src: &[u8],
    elements: usize,
) -> Result<Vec<u8>, BackendError> {
    if src.len() != elements {
        return Err(BackendError::InvalidArgument);
    }
    let bytes = elements
        .checked_mul(2)
        .ok_or(BackendError::InvalidArgument)?;
    let mut dst = vec![0u8; bytes];
    match dt {
        MatMulDType::B8 => axnn_cpu::fp8::dequant_e4m3_f16(src, &mut dst, elements),
        MatMulDType::F8 => axnn_cpu::fp8::dequant_e5m2_f16(src, &mut dst, elements),
        _ => return Err(BackendError::InvalidArgument),
    }
    Ok(dst)
}
