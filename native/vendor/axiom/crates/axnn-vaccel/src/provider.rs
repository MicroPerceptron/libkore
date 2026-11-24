//! `AxTosaAccelerator` — axiom's CPU kernels as a virtio-accel provider.
//!
//! Implements the full `virtio_accel_core::Accelerator` contract over the
//! `axtosa` executor: TOSA 1.0 artifacts load, validate, and run on the
//! axnn-cpu SIMD kernel set — including the FP8 tier every current NPU
//! provider rejects. Submission executes synchronously (the CPU *is* the
//! device), so events are terminal by the time `submit` returns.
//!
//! Buffers are plain host allocations behind `RefCell`: the trait routes all
//! content access through `write_buffer`/`read_buffer`/`submit`. Submission
//! holds scoped borrows of distinct bound allocations and executes directly
//! against them, so admission introduces no hidden full-range bounce copies.

use core::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use virtio_accel_core::{
    Accelerator, AllocatedBuffer, ArtifactRef, BackendError, BindingRef, BufferDesc, BufferInfo,
    BufferProperties, ByteSink, ByteSource, Capabilities, ContextDesc, DeviceIdentity, DeviceInfo,
    DeviceLimits, EventState, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
};
use virtio_accel_tosa::{
    CapabilityDescriptor, DType, DTypeCapability, ExtensionSet, GraphCapabilities, Level, Op,
    OperatorCapability, ProfileSet, RuntimeConditionSupport, Target, TosaCapabilityProvider,
    ValueRoles, Version,
};

/// `BackendError::External` domain for axtosa execution failures.
pub const AX_ERROR_DOMAIN: u32 = 0x4158_4e4e; // "AXNN"

const AX_TOSA_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::ALL,
    Level::Level8K,
    ExtensionSet::FP8E4M3
        .union(ExtensionSet::FP8E5M2)
        .union(ExtensionSet::INT4),
);

const AX_TOSA_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(DType::BOOL, ValueRoles::ALL),
    DTypeCapability::new(DType::INT8, ValueRoles::ALL),
    DTypeCapability::new(DType::INT16, ValueRoles::ALL),
    DTypeCapability::new(DType::INT32, ValueRoles::ALL),
    DTypeCapability::new(DType::FP16, ValueRoles::ALL),
    DTypeCapability::new(DType::FP32, ValueRoles::ALL),
    DTypeCapability::new(DType::BF16, ValueRoles::ALL),
    DTypeCapability::new(DType::FP8E4M3, ValueRoles::ALL),
    DTypeCapability::new(DType::FP8E5M2, ValueRoles::ALL),
];

const AX_TOSA_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::new(Op::CONST),
    OperatorCapability::new(Op::CONST_SHAPE),
    OperatorCapability::new(Op::IDENTITY),
    OperatorCapability::new(Op::CAST),
    OperatorCapability::new(Op::ADD),
    OperatorCapability::new(Op::SUB),
    OperatorCapability::new(Op::MUL),
    OperatorCapability::new(Op::POW),
    OperatorCapability::new(Op::MAXIMUM),
    OperatorCapability::new(Op::MINIMUM),
    OperatorCapability::new(Op::ABS),
    OperatorCapability::new(Op::CEIL),
    OperatorCapability::new(Op::COS),
    OperatorCapability::new(Op::EXP),
    OperatorCapability::new(Op::FLOOR),
    OperatorCapability::new(Op::LOG),
    OperatorCapability::new(Op::NEGATE),
    OperatorCapability::new(Op::RECIPROCAL),
    OperatorCapability::new(Op::RSQRT),
    OperatorCapability::new(Op::SIN),
    OperatorCapability::new(Op::EQUAL),
    OperatorCapability::new(Op::GREATER),
    OperatorCapability::new(Op::GREATER_EQUAL),
    OperatorCapability::new(Op::LOGICAL_AND),
    OperatorCapability::new(Op::LOGICAL_OR),
    OperatorCapability::new(Op::LOGICAL_XOR),
    OperatorCapability::new(Op::LOGICAL_NOT),
    OperatorCapability::new(Op::SELECT),
    OperatorCapability::new(Op::TANH),
    OperatorCapability::new(Op::SIGMOID),
    OperatorCapability::new(Op::CLAMP),
    OperatorCapability::new(Op::MATMUL),
    OperatorCapability::new(Op::MAX_POOL2D),
    OperatorCapability::new(Op::REDUCE_SUM),
    OperatorCapability::new(Op::REDUCE_MAX),
    OperatorCapability::new(Op::RESHAPE),
    OperatorCapability::new(Op::ARGMAX),
];

const AX_TOSA_CAPABILITIES: &[CapabilityDescriptor] = &[CapabilityDescriptor {
    target: AX_TOSA_TARGET,
    dtypes: AX_TOSA_DTYPES,
    operators: AX_TOSA_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
}];

/// CPU TOSA provider backed by the axtosa executor.
#[derive(Debug, Default)]
pub struct AxTosaAccelerator {
    live_contexts: AtomicU64,
    live_buffers: AtomicU64,
    live_programs: AtomicU64,
    buffer_allocations: AtomicU64,
    buffer_write_bytes: AtomicU64,
    buffer_read_bytes: AtomicU64,
}

impl AxTosaAccelerator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total provider buffers allocated since construction.
    ///
    /// This is useful for verifying that a [`crate::Session`] has reached
    /// its steady state instead of allocating on every submission.
    pub fn buffer_allocations(&self) -> u64 {
        self.buffer_allocations.load(Ordering::Relaxed)
    }

    /// Provider buffers that have not yet been explicitly released.
    pub fn live_buffers(&self) -> u64 {
        self.live_buffers.load(Ordering::Relaxed)
    }

    /// Resident TOSA programs that have not yet been explicitly unloaded.
    pub fn live_programs(&self) -> u64 {
        self.live_programs.load(Ordering::Relaxed)
    }

    /// Bytes explicitly transferred into provider buffers since construction.
    pub fn buffer_write_bytes(&self) -> u64 {
        self.buffer_write_bytes.load(Ordering::Relaxed)
    }

    /// Bytes explicitly transferred out of provider buffers since construction.
    pub fn buffer_read_bytes(&self) -> u64 {
        self.buffer_read_bytes.load(Ordering::Relaxed)
    }

    /// The device-neutral target this provider accepts: TOSA 1.0, all
    /// profiles, Level 8K, with the FP8 and INT4 extensions.
    pub fn target() -> Target {
        AX_TOSA_TARGET
    }
}

impl TosaCapabilityProvider for AxTosaAccelerator {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        AX_TOSA_CAPABILITIES
    }
}

#[derive(Debug)]
pub struct AxContext(());

#[derive(Debug)]
pub struct AxBuffer {
    data: RefCell<Vec<u8>>,
}

#[derive(Debug)]
pub struct AxProgram {
    compiled: axtosa::Program,
    /// Number of block input tensors; bindings `0..inputs` feed them in
    /// order and bindings `inputs..inputs + outputs` receive results.
    inputs: usize,
    outputs: usize,
}

#[derive(Debug)]
pub struct AxQueue(());

#[derive(Debug)]
pub struct AxEvent {
    state: EventState,
}

fn read_source(source: &dyn ByteSource) -> Result<Vec<u8>, BackendError> {
    let len = usize::try_from(source.len()).map_err(|_| BackendError::OutOfMemory)?;
    let mut bytes = vec![0u8; len];
    source.read_at(0, &mut bytes)?;
    Ok(bytes)
}

fn checked_range(offset: u64, len: usize, capacity: usize) -> Result<(usize, usize), BackendError> {
    let start = usize::try_from(offset).map_err(|_| BackendError::OutOfBounds)?;
    let end = start
        .checked_add(len)
        .filter(|end| *end <= capacity)
        .ok_or(BackendError::OutOfBounds)?;
    Ok((start, end))
}

impl Accelerator for AxTosaAccelerator {
    type Context = AxContext;
    type Buffer = AxBuffer;
    type Program = AxProgram;
    type Queue = AxQueue;
    type Event = AxEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(DeviceInfo {
            identity: DeviceIdentity {
                uuid: *b"axiom.axtosa.cpu",
                class: virtio_accel_core::AcceleratorClass::OTHER,
                vendor_id: 0x4d50, // "MP"
                device_id: 0x0001,
            },
            capabilities: Capabilities::HOST_VISIBLE_MEMORY,
            limits: DeviceLimits {
                max_contexts: u32::MAX,
                max_buffers_per_context: u32::MAX,
                max_programs_per_context: u32::MAX,
                max_queues_per_context: u32::MAX,
                max_events_per_context: u32::MAX,
                max_bindings_per_submission: 256,
                max_buffer_bytes: u64::MAX,
                max_artifact_bytes: u64::MAX,
            },
        })
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        if !desc.flags.is_empty() {
            return Err(BackendError::Unsupported);
        }
        self.live_contexts.fetch_add(1, Ordering::Relaxed);
        Ok(AxContext(()))
    }

    fn destroy_context(
        &self,
        _context: Self::Context,
    ) -> Result<(), ReleaseFailure<Self::Context>> {
        self.live_contexts.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn allocate_buffer(
        &self,
        _context: &Self::Context,
        desc: BufferDesc,
    ) -> Result<AllocatedBuffer<Self::Buffer>, BackendError> {
        let bytes = usize::try_from(desc.bytes()).map_err(|_| BackendError::OutOfMemory)?;
        let info = BufferInfo::new(
            desc,
            desc.bytes(),
            desc.alignment(),
            BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING,
        )?;
        let allocated = AllocatedBuffer::new(
            AxBuffer {
                data: RefCell::new(vec![0u8; bytes]),
            },
            info,
        );
        self.live_buffers.fetch_add(1, Ordering::Relaxed);
        self.buffer_allocations.fetch_add(1, Ordering::Relaxed);
        Ok(allocated)
    }

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset: u64,
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        let len = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let mut dst = buffer.data.borrow_mut();
        let (start, end) = checked_range(offset, len, dst.len())?;
        data.read_at(0, &mut dst[start..end])?;
        self.buffer_write_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
        Ok(())
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        let len = usize::try_from(data.len()).map_err(|_| BackendError::OutOfBounds)?;
        let src = buffer.data.borrow();
        let (start, end) = checked_range(offset, len, src.len())?;
        data.write_at(0, &src[start..end])?;
        self.buffer_read_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
        Ok(())
    }

    fn free_buffer(&self, _buffer: Self::Buffer) -> Result<(), ReleaseFailure<Self::Buffer>> {
        self.live_buffers.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn load_program(
        &self,
        _context: &Self::Context,
        artifact: ArtifactRef<'_>,
    ) -> Result<Self::Program, BackendError> {
        if artifact.format != virtio_accel_tosa::ARTIFACT_FORMAT {
            return Err(BackendError::Unsupported);
        }
        if artifact.target != Self::target().to_identity() {
            return Err(BackendError::Incompatible);
        }
        let bytes = read_source(artifact.payload)?;
        // Parse, semantically validate, and copy the immutable execution plan
        // once. Warm submissions never revisit the FlatBuffer.
        let compiled = axtosa::Program::compile(&bytes).map_err(|error| match error {
            axtosa::ExecError::Parse(_) => BackendError::InvalidArgument,
            axtosa::ExecError::Analysis(_) | axtosa::ExecError::Unsupported(_) => {
                BackendError::Incompatible
            }
            axtosa::ExecError::Shape(_) | axtosa::ExecError::Binding { .. } => {
                BackendError::Unsupported
            }
        })?;
        let inputs = compiled.input_count();
        let outputs = compiled.output_count();
        self.live_programs.fetch_add(1, Ordering::Relaxed);
        Ok(AxProgram {
            compiled,
            inputs,
            outputs,
        })
    }

    fn unload_program(&self, _program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
        self.live_programs.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    fn create_queue(
        &self,
        _context: &Self::Context,
        _desc: QueueDesc,
    ) -> Result<Self::Queue, BackendError> {
        Ok(AxQueue(()))
    }

    fn destroy_queue(&self, _queue: Self::Queue) -> Result<(), ReleaseFailure<Self::Queue>> {
        Ok(())
    }

    fn submit(
        &self,
        _queue: &Self::Queue,
        program: &Self::Program,
        bindings: &[BindingRef<'_, Self::Buffer>],
        _timeout: Timeout,
    ) -> Result<Self::Event, SubmitFailure<Self::Event>> {
        let rejected = |error| Err(SubmitFailure::Rejected(error));

        let binding_for = |slot: usize| {
            bindings
                .iter()
                .find(|binding| binding.slot as usize == slot)
        };

        // `RefCell` enforces each allocation's access mode locally. Reject an
        // aliased write up front so malformed bindings produce a contract
        // error rather than a dynamic-borrow panic. Repeated read bindings to
        // the same allocation remain lawful.
        for (left_index, left) in bindings.iter().enumerate() {
            for right in &bindings[left_index + 1..] {
                if core::ptr::eq(left.buffer, right.buffer)
                    && (left.access != virtio_accel_core::AccessMode::Read
                        || right.access != virtio_accel_core::AccessMode::Read)
                {
                    return rejected(BackendError::InvalidArgument);
                }
            }
        }

        let mut input_ranges = Vec::with_capacity(program.inputs);
        let mut input_guards = Vec::with_capacity(program.inputs);
        for slot in 0..program.inputs {
            let Some(binding) = binding_for(slot) else {
                return rejected(BackendError::InvalidArgument);
            };
            let data = binding.buffer.data.borrow();
            let Ok((start, end)) = checked_range(
                binding.range.offset,
                binding.range.bytes() as usize,
                data.len(),
            ) else {
                return rejected(BackendError::OutOfBounds);
            };
            input_ranges.push((start, end));
            input_guards.push(data);
        }

        let mut output_ranges = Vec::with_capacity(program.outputs);
        let mut output_guards = Vec::with_capacity(program.outputs);
        for index in 0..program.outputs {
            let Some(binding) = binding_for(program.inputs + index) else {
                return rejected(BackendError::InvalidArgument);
            };
            let data = binding.buffer.data.borrow_mut();
            let Ok((start, end)) = checked_range(
                binding.range.offset,
                binding.range.bytes() as usize,
                data.len(),
            ) else {
                return rejected(BackendError::OutOfBounds);
            };
            output_ranges.push((start, end));
            output_guards.push(data);
        }

        let input_refs: Vec<&[u8]> = input_guards
            .iter()
            .zip(&input_ranges)
            .map(|(data, &(start, end))| &data[start..end])
            .collect();
        let mut output_refs: Vec<&mut [u8]> = output_guards
            .iter_mut()
            .zip(&output_ranges)
            .map(|(data, &(start, end))| &mut data[start..end])
            .collect();
        if program
            .compiled
            .execute_into(&input_refs, &mut output_refs)
            .is_err()
        {
            return rejected(BackendError::External {
                domain: AX_ERROR_DOMAIN,
                code: 1,
            });
        }

        Ok(AxEvent {
            state: EventState::Complete,
        })
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        Ok(event.state)
    }

    fn destroy_event(&self, _event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        Ok(())
    }
}
