#![feature(allocator_api)]
#![feature(f16)]

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use ax::{
    AdmissionPlanner, AllocationError, CostModel, DENSE_IMPLEMENTATION, DType, DispatchLane,
    DispatchRequest, GraphBuf, GraphBuilder, NodeFlags, NodeSpec, OpCode, Partition,
    PlacementError, Schedule, TensorFlags, TensorId, TensorSpec,
};
use axnn_vaccel::{
    ACCEL_BACKEND, AxCompileError, AxDeviceModel, AxDispatchVariant, AxFeedback, AxHetero,
    AxInvocationStatus, AxPartitionPhase, AxSparseBsrPolicy, AxTosaAccelerator, CPU_BACKEND,
    CPU_BSR_B8_IMPLEMENTATION, CPU_BSR_F32_IMPLEMENTATION, CPU_KERNEL_IMPLEMENTATION, Session,
    admit_variants, artifact,
};
use virtio_accel_core::{
    Accelerator, AllocatedBuffer, ArtifactRef, BackendError, BindingRef, BufferDesc, ByteSink,
    ByteSource, ContextDesc, DeviceInfo, EventState, QueueDesc, ReleaseFailure, SubmitFailure,
    Timeout,
};
use virtio_accel_tosa::{
    CapabilityDescriptor, DType as TosaDType, DTypeCapability, ExtensionSet, GraphCapabilities,
    Level, Op as TosaOp, OperatorCapability, ProfileSet, RuntimeConditionSupport, Target,
    TosaCapabilityProvider, ValueRoles, Version,
};

const ADD_ONLY_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::NONE,
);
const ADD_ONLY_DTYPES: &[DTypeCapability] =
    &[DTypeCapability::new(TosaDType::FP32, ValueRoles::ALL)];
const ADD_ONLY_OPERATORS: &[OperatorCapability] = &[OperatorCapability::new(TosaOp::ADD)];
const ADD_ONLY_DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    target: ADD_ONLY_TARGET,
    dtypes: ADD_ONLY_DTYPES,
    operators: ADD_ONLY_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

const INTERMEDIATE_ADD_DTYPES: &[DTypeCapability] = &[DTypeCapability::new(
    TosaDType::FP32,
    ValueRoles::INTERMEDIATE,
)];
const INTERMEDIATE_ADD_DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    target: ADD_ONLY_TARGET,
    dtypes: INTERMEDIATE_ADD_DTYPES,
    operators: ADD_ONLY_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

const LOGICAL_BRIDGE_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(TosaDType::FP32, ValueRoles::ALL),
    DTypeCapability::new(TosaDType::BOOL, ValueRoles::INTERMEDIATE),
];
const LOGICAL_BRIDGE_OPERATORS: &[OperatorCapability] =
    &[OperatorCapability::new(TosaOp::LOGICAL_NOT)];
const LOGICAL_BRIDGE_DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    target: ADD_ONLY_TARGET,
    dtypes: LOGICAL_BRIDGE_DTYPES,
    operators: LOGICAL_BRIDGE_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

struct AddOnlyCapability;

impl TosaCapabilityProvider for AddOnlyCapability {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        &[ADD_ONLY_DESCRIPTOR]
    }
}

#[derive(Default)]
struct AllocationCounts {
    live: AtomicUsize,
    total: AtomicUsize,
    largest: AtomicUsize,
    large_byte_allocations: AtomicUsize,
}

#[derive(Clone)]
struct TrackingAllocator(Arc<AllocationCounts>);

// SAFETY: allocation and deallocation are delegated unchanged to `Global`;
// the wrapper only records successful ownership transitions for this test.
unsafe impl Allocator for TrackingAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let allocation = Global.allocate(layout)?;
        self.0.live.fetch_add(1, Ordering::Relaxed);
        self.0.total.fetch_add(1, Ordering::Relaxed);
        self.0.largest.fetch_max(layout.size(), Ordering::Relaxed);
        if layout.align() == 1 && layout.size() >= 4096 {
            self.0
                .large_byte_allocations
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(allocation)
    }

    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        self.0.live.fetch_sub(1, Ordering::Relaxed);
        // SAFETY: `pointer` and `layout` are the pair returned by the delegated
        // `Global` allocation or one of Allocator's default grow operations.
        unsafe { Global.deallocate(pointer, layout) };
    }
}

fn sessions() -> AxHetero<AxTosaAccelerator, AxTosaAccelerator> {
    let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    AxHetero::new(cpu, Some(accel))
}

/// Contract-preserving test adapter that makes a completed axtosa event
/// observable as pending once. Execution is still performed by the real
/// provider; only event visibility is delayed so caller-driven polling is
/// exercised deterministically.
struct DelayedAccelerator {
    inner: AxTosaAccelerator,
    indeterminate_submit: bool,
    release_rejections: u8,
    load_count: Cell<u8>,
    fail_load: Option<u8>,
    capabilities: Option<&'static [CapabilityDescriptor]>,
}

struct DelayedEvent {
    inner: <AxTosaAccelerator as Accelerator>::Event,
    pending_polls: Cell<u8>,
    release_rejections: u8,
}

impl DelayedAccelerator {
    fn new() -> Self {
        Self {
            inner: AxTosaAccelerator::new(),
            indeterminate_submit: false,
            release_rejections: 0,
            load_count: Cell::new(0),
            fail_load: None,
            capabilities: None,
        }
    }

    fn adversarial() -> Self {
        Self {
            inner: AxTosaAccelerator::new(),
            indeterminate_submit: true,
            release_rejections: 1,
            load_count: Cell::new(0),
            fail_load: None,
            capabilities: None,
        }
    }

    fn fail_on_load(load: u8) -> Self {
        Self {
            inner: AxTosaAccelerator::new(),
            indeterminate_submit: false,
            release_rejections: 0,
            load_count: Cell::new(0),
            fail_load: Some(load),
            capabilities: None,
        }
    }

    fn live_programs(&self) -> u64 {
        self.inner.live_programs()
    }
}

impl TosaCapabilityProvider for DelayedAccelerator {
    fn tosa_capabilities(&self) -> &'static [CapabilityDescriptor] {
        self.capabilities
            .unwrap_or_else(|| self.inner.tosa_capabilities())
    }
}

impl Accelerator for DelayedAccelerator {
    type Context = <AxTosaAccelerator as Accelerator>::Context;
    type Buffer = <AxTosaAccelerator as Accelerator>::Buffer;
    type Program = <AxTosaAccelerator as Accelerator>::Program;
    type Queue = <AxTosaAccelerator as Accelerator>::Queue;
    type Event = DelayedEvent;

    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        self.inner.device_info()
    }

    fn create_context(&self, desc: ContextDesc) -> Result<Self::Context, BackendError> {
        self.inner.create_context(desc)
    }

    fn destroy_context(&self, context: Self::Context) -> Result<(), ReleaseFailure<Self::Context>> {
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
        data: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        self.inner.write_buffer(buffer, offset, data)
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &mut dyn ByteSink,
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
        let next = self.load_count.get().saturating_add(1);
        self.load_count.set(next);
        if self.fail_load == Some(next) {
            return Err(BackendError::InvalidArgument);
        }
        self.inner.load_program(context, artifact)
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), ReleaseFailure<Self::Program>> {
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
        match self.inner.submit(queue, program, bindings, timeout) {
            Ok(inner) => {
                let event = DelayedEvent {
                    inner,
                    pending_polls: Cell::new(1),
                    release_rejections: self.release_rejections,
                };
                if self.indeterminate_submit {
                    Err(SubmitFailure::Indeterminate {
                        error: BackendError::Busy,
                        event,
                    })
                } else {
                    Ok(event)
                }
            }
            Err(SubmitFailure::Rejected(error)) => Err(SubmitFailure::Rejected(error)),
            Err(SubmitFailure::Indeterminate { error, event }) => {
                Err(SubmitFailure::Indeterminate {
                    error,
                    event: DelayedEvent {
                        inner: event,
                        pending_polls: Cell::new(1),
                        release_rejections: self.release_rejections,
                    },
                })
            }
        }
    }

    fn poll_event(&self, event: &Self::Event) -> Result<EventState, BackendError> {
        if event.pending_polls.get() != 0 {
            event.pending_polls.set(event.pending_polls.get() - 1);
            return Ok(EventState::Pending);
        }
        self.inner.poll_event(&event.inner)
    }

    fn cancel_event(&self, event: &Self::Event) -> Result<(), BackendError> {
        self.inner.cancel_event(&event.inner)
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), ReleaseFailure<Self::Event>> {
        if event.release_rejections != 0 {
            return Err(ReleaseFailure::Rejected {
                error: BackendError::Busy,
                resource: DelayedEvent {
                    inner: event.inner,
                    pending_polls: event.pending_polls,
                    release_rejections: event.release_rejections - 1,
                },
            });
        }
        let pending_polls = event.pending_polls.get();
        match self.inner.destroy_event(event.inner) {
            Ok(()) => Ok(()),
            Err(ReleaseFailure::Rejected { error, resource }) => Err(ReleaseFailure::Rejected {
                error,
                resource: DelayedEvent {
                    inner: resource,
                    pending_polls: Cell::new(pending_polls),
                    release_rejections: 0,
                },
            }),
            Err(ReleaseFailure::Indeterminate { error }) => {
                Err(ReleaseFailure::Indeterminate { error })
            }
        }
    }
}

fn f16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| (value as f16).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

fn i32_bytes(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_le_bytes).collect()
}

fn prepared_native_bytes<A: Allocator>(
    prepared: &axnn_vaccel::PreparedNative<A>,
    dtypes: &[DType],
) -> Vec<Vec<u8>> {
    assert_eq!(prepared.output_count(), dtypes.len());
    dtypes
        .iter()
        .enumerate()
        .map(|(index, dtype)| match dtype {
            DType::I32 => i32_bytes(prepared.output_i32(index).unwrap().iter().copied()),
            DType::F32 => f32_bytes(prepared.output_f32(index).unwrap().iter().copied()),
            DType::U8 => prepared.output_u8(index).unwrap().to_vec(),
            _ => panic!("unsupported native test dtype {dtype:?}"),
        })
        .collect()
}

#[test]
fn static_sparse_f32_matmul_is_packed_cached_and_executed_natively() {
    const INPUT: usize = 64;
    const OUTPUT: usize = 4;
    const ROWS: usize = 2;

    // TOSA/right-hand layout: B[input, output].  Only the first 32-wide
    // reduction tile of output zero is nonzero, so BSR keeps one of eight
    // candidate blocks while preserving the dense graph semantics.
    let mut weights = vec![0.0_f32; INPUT * OUTPUT];
    for input in 0..32 {
        weights[input * OUTPUT] = (input as f32 + 1.0) * 0.03125;
    }
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(
            DType::F32,
            TensorFlags::INPUT,
            &[1, ROWS as u32, INPUT as u32],
            &[],
        )
        .unwrap();
    let constants = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1, INPUT as u32, OUTPUT as u32],
            &f32_bytes(weights.iter().copied()),
        )
        .unwrap();
    let output = builder
        .tensor(
            DType::F32,
            TensorFlags::OUTPUT,
            &[1, ROWS as u32, OUTPUT as u32],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[input, constants],
            &[output],
            &[],
        )
        .unwrap();
    let graph = builder.finish().unwrap();
    let graph = graph.parse().unwrap();

    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        cpu_kernels: None,
        sparse_bsr: Some(AxSparseBsrPolicy {
            block_k: 32,
            max_block_density_ppm: 400_000,
            block_index_cost: 8,
        }),
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), &[CPU_BACKEND]);
    assert_eq!(
        executable.schedule().implementations(),
        &[CPU_BSR_F32_IMPLEMENTATION]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
    assert_eq!(
        executable.schedule().plan().partitions[0].implementation,
        CPU_BSR_F32_IMPLEMENTATION
    );

    let activations: Vec<f32> = (0..ROWS * INPUT)
        .map(|index| index as f32 * 0.015625 - 0.75)
        .collect();
    let input_bytes = f32_bytes(activations.iter().copied());
    let first = runtime
        .run_in(&executable, &[&input_bytes], Global)
        .unwrap();
    let second = runtime
        .run_in(&executable, &[&input_bytes], Global)
        .unwrap();
    assert_eq!(
        second, first,
        "the packed constant must be reusable across runs"
    );
    let actual: Vec<f32> = first[0]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    for row in 0..ROWS {
        let expected: f32 = (0..INPUT)
            .map(|column| activations[row * INPUT + column] * weights[column * OUTPUT])
            .sum();
        assert!((actual[row * OUTPUT] - expected).abs() < 1.0e-5);
        assert_eq!(&actual[row * OUTPUT + 1..row * OUTPUT + OUTPUT], &[0.0; 3]);
    }
}

#[test]
fn adjacent_sparse_f32_matmuls_compile_and_execute_as_one_region() {
    const WIDTH: usize = 64;
    let mut weights = vec![0.0_f32; WIDTH * WIDTH];
    for input in 0..32 {
        weights[input * WIDTH] = 1.0;
    }
    let weight_bytes = f32_bytes(weights.iter().copied());
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[1, 1, WIDTH as u32], &[])
        .unwrap();
    let first_weights = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1, WIDTH as u32, WIDTH as u32],
            &weight_bytes,
        )
        .unwrap();
    let second_weights = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1, WIDTH as u32, WIDTH as u32],
            &weight_bytes,
        )
        .unwrap();
    let middle = builder
        .tensor(
            DType::F32,
            TensorFlags::default(),
            &[1, 1, WIDTH as u32],
            &[],
        )
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 1, WIDTH as u32], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[input, first_weights],
            &[middle],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[middle, second_weights],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        cpu_kernels: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().implementations(),
        [CPU_BSR_F32_IMPLEMENTATION; 2]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
    assert_eq!(executable.schedule().plan().partitions[0].node_count, 2);

    let input = f32_bytes((1..=WIDTH).map(|value| value as f32));
    let outputs = runtime.run_in(&executable, &[&input], Global).unwrap();
    let actual: Vec<f32> = outputs[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(actual[0], (1..=32).sum::<usize>() as f32);
    assert!(actual[1..].iter().all(|&value| value == 0.0));
}

#[test]
fn static_sparse_b8_matmul_stays_packed_until_f16_output() {
    const INPUT: usize = 64;
    const OUTPUT: usize = 4;
    const ROWS: usize = 2;

    let mut weights = vec![0_u8; INPUT * OUTPUT];
    for input in 0..32 {
        weights[input * OUTPUT] = axnn_cpu::fp8::f32_to_e4m3_rne((input as f32 + 1.0) / 64.0);
    }
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(
            DType::B8,
            TensorFlags::INPUT,
            &[1, ROWS as u32, INPUT as u32],
            &[],
        )
        .unwrap();
    let constants = builder
        .tensor(
            DType::B8,
            TensorFlags::CONSTANT,
            &[1, INPUT as u32, OUTPUT as u32],
            &weights,
        )
        .unwrap();
    let output = builder
        .tensor(
            DType::F16,
            TensorFlags::OUTPUT,
            &[1, ROWS as u32, OUTPUT as u32],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[input, constants],
            &[output],
            &[],
        )
        .unwrap();
    let graph = builder.finish().unwrap();
    let graph = graph.parse().unwrap();

    let model = AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), &[CPU_BACKEND]);
    assert_eq!(
        executable.schedule().implementations(),
        &[CPU_BSR_B8_IMPLEMENTATION]
    );

    let activations: Vec<u8> = (0..ROWS * INPUT)
        .map(|index| axnn_cpu::fp8::f32_to_e4m3_rne(index as f32 / 64.0 - 0.5))
        .collect();
    let result = runtime
        .run_in(&executable, &[&activations], Global)
        .unwrap();
    for row in 0..ROWS {
        let expected: f32 = (0..INPUT)
            .map(|column| {
                axnn_cpu::fp8::e4m3_to_f32(activations[row * INPUT + column])
                    * axnn_cpu::fp8::e4m3_to_f32(weights[column * OUTPUT])
            })
            .sum();
        let actual = f16::from_bits(u16::from_le_bytes(
            result[0][(row * OUTPUT) * 2..(row * OUTPUT + 1) * 2]
                .try_into()
                .unwrap(),
        )) as f32;
        assert!((actual - (expected as f16) as f32).abs() < 1.0e-6);
        for column in 1..OUTPUT {
            assert_eq!(
                &result[0][(row * OUTPUT + column) * 2..(row * OUTPUT + column + 1) * 2],
                &[0, 0]
            );
        }
    }
}

#[test]
fn static_sparse_f32_matmul_keeps_a_constant_activation_internal() {
    const WIDTH: usize = 96;
    let activations: Vec<f32> = (0..WIDTH)
        .map(|index| index as f32 * 0.0625 - 0.75)
        .collect();
    let weights: Vec<f32> = (0..WIDTH)
        .map(|index| {
            (index < 32)
                .then_some((index as f32 + 1.0) * 0.03125)
                .unwrap_or(0.0)
        })
        .collect();
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1, 1, WIDTH as u32],
            &f32_bytes(activations.iter().copied()),
        )
        .unwrap();
    let constants = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1, WIDTH as u32, 1],
            &f32_bytes(weights.iter().copied()),
        )
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 1, 1], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[input, constants],
            &[output],
            &[],
        )
        .unwrap();
    let graph = builder.finish().unwrap();
    let graph = graph.parse().unwrap();

    let model = AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().implementations(),
        &[CPU_BSR_F32_IMPLEMENTATION]
    );
    assert!(executable.input_ids().is_empty());
    let result = runtime.run_in(&executable, &[], Global).unwrap();
    let actual = f32::from_le_bytes(result[0].as_slice().try_into().unwrap());
    let expected: f32 = activations
        .iter()
        .zip(&weights)
        .map(|(activation, weight)| activation * weight)
        .sum();
    assert!((actual - expected).abs() < 1.0e-5);
}

#[test]
fn repeated_graph_keeps_feedback_in_provider_buffers() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[1], &[])
        .unwrap();
    let one = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &1.0_f32.to_le_bytes(),
        )
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[1], &[])
        .unwrap();
    builder
        .node(
            OpCode::ADD,
            NodeFlags::default(),
            &[input, one],
            &[output],
            &[],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let graph = bytes.parse().unwrap();

    let mut runtime = sessions();
    let executable = runtime
        .compile_in(
            &graph,
            &AxDeviceModel {
                accelerator: false,
                ..AxDeviceModel::default()
            },
            Global,
        )
        .unwrap();
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
    let provider = runtime.cpu.provider();
    let writes_before = provider.buffer_write_bytes();
    let reads_before = provider.buffer_read_bytes();

    let initial = 0.0_f32.to_le_bytes();
    let outputs = runtime
        .run_repeated_in(
            &executable,
            &[&initial],
            4,
            &[AxFeedback::new(input, output)],
            Global,
        )
        .unwrap();

    assert_eq!(f32::from_le_bytes(outputs[0][..4].try_into().unwrap()), 4.0);
    assert_eq!(
        runtime.cpu.provider().buffer_write_bytes() - writes_before,
        4,
        "recurrent values were uploaded between submissions"
    );
    assert_eq!(
        runtime.cpu.provider().buffer_read_bytes() - reads_before,
        4,
        "recurrent values were read back between submissions"
    );

    // Automatic recurrence must retain the same provider-resident fast path,
    // without reading/writing feedback through the host each repetition.
    let writes_before = runtime.cpu.provider().buffer_write_bytes();
    let reads_before = runtime.cpu.provider().buffer_read_bytes();
    let automatic = runtime
        .run_repeated_auto_in(
            &executable,
            &[&initial],
            4,
            &[AxFeedback::new(input, output)],
            Global,
        )
        .unwrap();
    assert_eq!(automatic, outputs);
    assert_eq!(
        runtime.cpu.provider().buffer_write_bytes() - writes_before,
        4
    );
    assert_eq!(runtime.cpu.provider().buffer_read_bytes() - reads_before, 4);
}

#[test]
fn releasing_graph_evicts_provider_program_and_fixed_io() {
    let mut builder = GraphBuilder::new_in(Global);
    let one = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &1.0_f32.to_le_bytes(),
        )
        .unwrap();
    let two = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &2.0_f32.to_le_bytes(),
        )
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[1], &[])
        .unwrap();
    builder
        .node(
            OpCode::ADD,
            NodeFlags::default(),
            &[one, two],
            &[output],
            &[],
        )
        .unwrap();
    let graph = builder.finish().unwrap();

    let mut runtime = sessions();
    let executable = runtime
        .compile_in(
            &graph.parse().unwrap(),
            &AxDeviceModel {
                accelerator: false,
                ..AxDeviceModel::default()
            },
            Global,
        )
        .unwrap();
    assert_eq!(runtime.cpu.provider().live_programs(), 1);
    let outputs = runtime.run_in(&executable, &[], Global).unwrap();
    assert_eq!(outputs[0].len(), 4);
    assert_eq!(runtime.cpu.provider().live_buffers(), 1);

    runtime.release_executable(executable).unwrap();
    assert_eq!(runtime.cpu.provider().live_buffers(), 0);
    assert_eq!(runtime.cpu.provider().live_programs(), 0);
}

#[test]
fn compile_failure_rolls_back_every_ensured_provider_program() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let widened = builder
        .tensor(DType::F32, TensorFlags::default(), &[4], &[])
        .unwrap();
    let rounded = builder
        .tensor(DType::F32, TensorFlags::default(), &[4], &[])
        .unwrap();
    let narrowed = builder
        .tensor(DType::F16, TensorFlags::default(), &[4], &[])
        .unwrap();
    let widened_again = builder
        .tensor(DType::F32, TensorFlags::default(), &[4], &[])
        .unwrap();
    let rounded_again = builder
        .tensor(DType::F32, TensorFlags::default(), &[4], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    builder
        .node(
            OpCode::CAST,
            NodeFlags::default(),
            &[input],
            &[widened],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::ROUND,
            NodeFlags::default(),
            &[widened],
            &[rounded],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::CAST,
            NodeFlags::default(),
            &[rounded],
            &[narrowed],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::CAST,
            NodeFlags::default(),
            &[narrowed],
            &[widened_again],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::ROUND,
            NodeFlags::default(),
            &[widened_again],
            &[rounded_again],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::CAST,
            NodeFlags::default(),
            &[rounded_again],
            &[output],
            &[],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let graph = bytes.parse().unwrap();
    let cpu = Session::new(
        DelayedAccelerator::fail_on_load(3),
        AxTosaAccelerator::target(),
    )
    .unwrap();
    let mut runtime: AxHetero<DelayedAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);

    assert!(matches!(
        runtime.compile_in(
            &graph,
            &AxDeviceModel {
                accelerator: false,
                sparse_bsr: None,
                ..AxDeviceModel::default()
            },
            Global,
        ),
        Err(AxCompileError::Backend(BackendError::InvalidArgument))
    ));
    assert_eq!(runtime.cpu.provider().load_count.get(), 3);
    assert_eq!(
        runtime.cpu.provider().live_programs(),
        0,
        "earlier provider programs leaked after a later load failed"
    );
}

fn cheap_elementwise_chain(nodes: usize, width: u32) -> GraphBuf<Global> {
    let mut builder = GraphBuilder::new_in(Global);
    let mut value = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[width], &[])
        .unwrap();
    for index in 0..nodes {
        let flags = if index + 1 == nodes {
            TensorFlags::OUTPUT
        } else {
            TensorFlags::default()
        };
        let output = builder.tensor(DType::F32, flags, &[width], &[]).unwrap();
        builder
            .node(OpCode::ABS, NodeFlags::default(), &[value], &[output], &[])
            .unwrap();
        value = output;
    }
    builder.finish().unwrap()
}

#[test]
fn dense_cpu_launch_cost_is_greedy_per_node_but_region_amortized() {
    let bytes = cheap_elementwise_chain(2, 4);
    let graph = bytes.parse().unwrap();
    let node = graph.node(0).unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_provider_launch_cost: 100,
        sparse_bsr: None,
        cpu_kernels: None,
        ..AxDeviceModel::default()
    };

    assert_eq!(model.execute_cost(CPU_BACKEND, &graph, node), 104);
    assert_eq!(
        model.region_execute_cost_implementation(CPU_BACKEND, &graph, node, DENSE_IMPLEMENTATION,),
        4,
    );
    assert_eq!(
        model.partition_cost_implementation(CPU_BACKEND, DENSE_IMPLEMENTATION),
        100,
    );
}

#[test]
fn direct_ax_device_cost_model_exposes_automatic_native_candidates() {
    let bytes = cheap_elementwise_chain(2, 32);
    let graph = bytes.parse().unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

    schedule.place(&graph, &model).unwrap();

    assert_eq!(schedule.implementations(), [CPU_KERNEL_IMPLEMENTATION; 2]);
}

#[test]
fn calibrated_common_cost_scale_selects_between_provider_and_native_kernels() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[32], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[32], &[])
        .unwrap();
    builder
        .node(OpCode::EXP, NodeFlags::default(), &[input], &[output], &[])
        .unwrap();
    let bytes = builder.finish().unwrap();
    let graph = bytes.parse().unwrap();

    let provider_favored = AxDeviceModel {
        accelerator: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut provider_schedule = Schedule::analyze_in(&graph, Global).unwrap();
    provider_schedule.place(&graph, &provider_favored).unwrap();
    assert_eq!(provider_schedule.implementations(), [DENSE_IMPLEMENTATION]);

    let native_favored = AxDeviceModel {
        cpu_provider_work_scale: 4,
        ..provider_favored
    };
    let mut native_schedule = Schedule::analyze_in(&graph, Global).unwrap();
    native_schedule.place(&graph, &native_favored).unwrap();
    assert_eq!(
        native_schedule.implementations(),
        [CPU_KERNEL_IMPLEMENTATION]
    );
}

#[test]
fn direct_device_model_rejects_attributes_before_implementation_admission() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    builder
        .node(OpCode::ABS, NodeFlags::default(), &[input], &[output], &[1])
        .unwrap();
    let bytes = builder.finish().unwrap();
    let graph = bytes.parse().unwrap();
    let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();

    assert_eq!(
        schedule.place(
            &graph,
            &AxDeviceModel {
                accelerator: false,
                ..AxDeviceModel::default()
            },
        ),
        Err(AllocationError::Placement(PlacementError::NoBackend {
            node: 0,
        })),
    );
}

fn interactive_depth_field(width: u32, height: u32) -> GraphBuf<Global> {
    let mut builder = GraphBuilder::new_in(Global);
    let scalar = [1_u32, 1];
    let x_shape = [1_u32, width];
    let y_shape = [height, 1_u32];
    let field = [height, width];
    let x_bytes = f32_bytes((0..width).map(|coordinate| coordinate as f32 + 0.5));
    let y_bytes = f32_bytes((0..height).map(|coordinate| coordinate as f32 + 0.5));
    let x = builder
        .tensor(DType::F32, TensorFlags::CONSTANT, &x_shape, &x_bytes)
        .unwrap();
    let y = builder
        .tensor(DType::F32, TensorFlags::CONSTANT, &y_shape, &y_bytes)
        .unwrap();
    let center_x = builder
        .tensor(DType::F32, TensorFlags::INPUT, &scalar, &[])
        .unwrap();
    let center_y = builder
        .tensor(DType::F32, TensorFlags::INPUT, &scalar, &[])
        .unwrap();
    let one = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &scalar,
            &1.0_f32.to_le_bytes(),
        )
        .unwrap();
    let mut binary = |op, left, right, shape: &[u32]| {
        let output = builder
            .tensor(DType::F32, TensorFlags::default(), shape, &[])
            .unwrap();
        builder
            .node(op, NodeFlags::default(), &[left, right], &[output], &[])
            .unwrap();
        output
    };
    let dx = binary(OpCode::SUB, x, center_x, &x_shape);
    let dy = binary(OpCode::SUB, y, center_y, &y_shape);
    let dx2 = binary(OpCode::MUL, dx, dx, &x_shape);
    let dy2 = binary(OpCode::MUL, dy, dy, &y_shape);
    let distance = binary(OpCode::ADD, dx2, dy2, &field);
    let denominator = binary(OpCode::ADD, one, distance, &field);
    drop(binary);
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &field, &[])
        .unwrap();
    builder
        .node(
            OpCode::RECIPROCAL,
            NodeFlags::default(),
            &[denominator],
            &[output],
            &[],
        )
        .unwrap();
    builder.finish().unwrap()
}

fn graph_with_boolean_abi_component() -> GraphBuf<Global> {
    let mut builder = GraphBuilder::new_in(Global);
    let shape = [4_u32];
    let x = builder
        .tensor(DType::F32, TensorFlags::INPUT, &shape, &[])
        .unwrap();
    let active = builder
        .tensor(DType::U8, TensorFlags::INPUT, &shape, &[])
        .unwrap();
    let zero = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &0.0_f32.to_le_bytes(),
        )
        .unwrap();
    let magnitude = builder
        .tensor(DType::F32, TensorFlags::default(), &shape, &[])
        .unwrap();
    let squared = builder
        .tensor(DType::F32, TensorFlags::default(), &shape, &[])
        .unwrap();
    let positive = builder
        .tensor(DType::U8, TensorFlags::default(), &shape, &[])
        .unwrap();
    let live = builder
        .tensor(DType::U8, TensorFlags::default(), &shape, &[])
        .unwrap();
    let selected = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &shape, &[])
        .unwrap();
    let active_out = builder
        .tensor(DType::U8, TensorFlags::OUTPUT, &shape, &[])
        .unwrap();
    builder
        .node(OpCode::ABS, NodeFlags::default(), &[x], &[magnitude], &[])
        .unwrap();
    builder
        .node(
            OpCode::MUL,
            NodeFlags::default(),
            &[magnitude, magnitude],
            &[squared],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::GREATER,
            NodeFlags::default(),
            &[squared, zero],
            &[positive],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::LOGICAL_AND,
            NodeFlags::default(),
            &[active, positive],
            &[live],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::SELECT,
            NodeFlags::default(),
            &[live, squared, zero],
            &[selected],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::LOGICAL_OR,
            NodeFlags::default(),
            &[live, live],
            &[active_out],
            &[],
        )
        .unwrap();
    builder.finish().unwrap()
}

fn matmul_abs_matmul_chain(width: u32) -> GraphBuf<Global> {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[1, 1, width], &[])
        .unwrap();
    let weight_bytes = f16_bytes((0..width as usize * width as usize).map(|index| {
        if index % (width as usize + 1) == 0 {
            1.0
        } else {
            0.0
        }
    }));
    let left_weights = builder
        .tensor(
            DType::F16,
            TensorFlags::CONSTANT,
            &[1, width, width],
            &weight_bytes,
        )
        .unwrap();
    let right_weights = builder
        .tensor(
            DType::F16,
            TensorFlags::CONSTANT,
            &[1, width, width],
            &weight_bytes,
        )
        .unwrap();
    let left = builder
        .tensor(DType::F16, TensorFlags::default(), &[1, 1, width], &[])
        .unwrap();
    let middle = builder
        .tensor(DType::F16, TensorFlags::default(), &[1, 1, width], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 1, width], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[input, left_weights],
            &[left],
            &[],
        )
        .unwrap();
    builder
        .node(OpCode::ABS, NodeFlags::default(), &[left], &[middle], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[middle, right_weights],
            &[output],
            &[],
        )
        .unwrap();
    builder.finish().unwrap()
}

fn matmul_abs_with_shared_host_output(width: u32) -> GraphBuf<Global> {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[1, 1, width], &[])
        .unwrap();
    let weight_bytes = f16_bytes((0..width as usize * width as usize).map(|index| {
        if index % (width as usize + 1) == 0 {
            1.0
        } else {
            0.0
        }
    }));
    let weights = builder
        .tensor(
            DType::F16,
            TensorFlags::CONSTANT,
            &[1, width, width],
            &weight_bytes,
        )
        .unwrap();
    let shared = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 1, width], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 1, width], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[input, weights],
            &[shared],
            &[],
        )
        .unwrap();
    builder
        .node(OpCode::ABS, NodeFlags::default(), &[shared], &[output], &[])
        .unwrap();
    builder.finish().unwrap()
}

fn leftward_region_dependency() -> GraphBuf<Global> {
    const WIDTH: u32 = 8192;
    const MATMUL_WIDTH: u32 = 256;
    const TRAILING_NODES: usize = 4;

    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[WIDTH], &[])
        .unwrap();
    let first = builder
        .tensor(DType::F16, TensorFlags::default(), &[WIDTH], &[])
        .unwrap();
    builder
        .node(OpCode::ABS, NodeFlags::default(), &[input], &[first], &[])
        .unwrap();

    let matmul_input = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[1, 1, MATMUL_WIDTH], &[])
        .unwrap();
    let weights = f16_bytes(
        (0..MATMUL_WIDTH as usize * MATMUL_WIDTH as usize).map(|index| {
            if index % (MATMUL_WIDTH as usize + 1) == 0 {
                1.0
            } else {
                0.0
            }
        }),
    );
    let matmul_weights = builder
        .tensor(
            DType::F16,
            TensorFlags::CONSTANT,
            &[1, MATMUL_WIDTH, MATMUL_WIDTH],
            &weights,
        )
        .unwrap();
    let matmul_output = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 1, MATMUL_WIDTH], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[matmul_input, matmul_weights],
            &[matmul_output],
            &[],
        )
        .unwrap();

    let mut value = first;
    for index in 0..TRAILING_NODES {
        let flags = if index + 1 == TRAILING_NODES {
            TensorFlags::OUTPUT
        } else {
            TensorFlags::default()
        };
        let output = builder.tensor(DType::F16, flags, &[WIDTH], &[]).unwrap();
        builder
            .node(OpCode::ABS, NodeFlags::default(), &[value], &[output], &[])
            .unwrap();
        value = output;
    }
    builder.finish().unwrap()
}

fn interleaved_independent_backend_graph() -> GraphBuf<Global> {
    const WIDTH: u32 = 8;
    let mut builder = GraphBuilder::new_in(Global);
    let matrix_input = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[1, 1, WIDTH], &[])
        .unwrap();
    let weights = f16_bytes((0..WIDTH as usize * WIDTH as usize).map(|index| {
        if index % (WIDTH as usize + 1) == 0 {
            1.0
        } else {
            0.0
        }
    }));
    let matrix_weights = builder
        .tensor(
            DType::F16,
            TensorFlags::CONSTANT,
            &[1, WIDTH, WIDTH],
            &weights,
        )
        .unwrap();
    let narrow_left = builder
        .tensor(DType::B8, TensorFlags::INPUT, &[WIDTH], &[])
        .unwrap();
    let narrow_right = builder
        .tensor(DType::B8, TensorFlags::INPUT, &[WIDTH], &[])
        .unwrap();

    let matrix_left = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 1, WIDTH], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[matrix_input, matrix_weights],
            &[matrix_left],
            &[],
        )
        .unwrap();
    let narrow_left_out = builder
        .tensor(DType::B8, TensorFlags::OUTPUT, &[WIDTH], &[])
        .unwrap();
    builder
        .node(
            OpCode::IDENTITY,
            NodeFlags::default(),
            &[narrow_left],
            &[narrow_left_out],
            &[],
        )
        .unwrap();
    let matrix_right = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 1, WIDTH], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[matrix_input, matrix_weights],
            &[matrix_right],
            &[],
        )
        .unwrap();
    let narrow_right_out = builder
        .tensor(DType::B8, TensorFlags::OUTPUT, &[WIDTH], &[])
        .unwrap();
    builder
        .node(
            OpCode::IDENTITY,
            NodeFlags::default(),
            &[narrow_right],
            &[narrow_right_out],
            &[],
        )
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn region_placement_amortizes_one_launch_for_long_elementwise_chain() {
    // Every individual 8K ABS is far cheaper on the CPU than one accelerator
    // launch. Taken together, 96 nodes repay one realistic default launch plus
    // unique ingress and egress transfers.
    let bytes = cheap_elementwise_chain(96, 8192);
    let graph = bytes.parse().unwrap();
    let model = AxDeviceModel::default();
    let mut greedy = Schedule::analyze_in(&graph, Global).unwrap();
    greedy.place(&graph, &model).unwrap();
    assert!(
        greedy
            .placements()
            .iter()
            .all(|&backend| backend == CPU_BACKEND),
        "the node-at-a-time baseline must expose the launch-amortization case"
    );

    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();

    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|&backend| backend == ACCEL_BACKEND)
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
}

#[test]
fn accelerator_only_model_disables_cpu_admission_instead_of_biasing_costs() {
    let bytes = cheap_elementwise_chain(3, 16);
    let graph = bytes.parse().unwrap();
    let model = AxDeviceModel {
        cpu: false,
        accelerator_launch_cost: u64::MAX / 4,
        transfer_fixed_cost: u64::MAX / 4,
        ..AxDeviceModel::default()
    };
    let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();
    schedule.place(&graph, &model).unwrap();
    assert_eq!(schedule.placements(), [ACCEL_BACKEND; 3]);

    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), [ACCEL_BACKEND; 3]);
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
}

#[test]
fn placement_aware_topology_coalesces_interleaved_independent_backends() {
    let bytes = interleaved_independent_backend_graph();
    let graph = bytes.parse().unwrap();
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let mut greedy = Schedule::analyze_in(&graph, Global).unwrap();
    greedy.place(&graph, &model).unwrap();
    assert_eq!(
        greedy.placements(),
        [ACCEL_BACKEND, CPU_BACKEND, ACCEL_BACKEND, CPU_BACKEND]
    );
    assert_eq!(greedy.build_plan(&graph).unwrap().partitions.len(), 4);

    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().placements(),
        [ACCEL_BACKEND, CPU_BACKEND, ACCEL_BACKEND, CPU_BACKEND]
    );
    assert_eq!(executable.schedule().order(), [0, 2, 1, 3]);
    assert_eq!(executable.schedule().plan().partitions.len(), 2);

    let matrix_input = f16_bytes((1..=8).map(|value| value as f32));
    let narrow_left = [0x00_u8, 0x01, 0x38, 0x3c, 0xb8, 0xbc, 0x7e, 0xfe];
    let narrow_right = [0x10_u8, 0x20, 0x30, 0x40, 0x90, 0xa0, 0xb0, 0xc0];
    let outputs = runtime
        .run_in(
            &executable,
            &[&matrix_input, &narrow_left, &narrow_right],
            Global,
        )
        .unwrap();
    assert_eq!(outputs[0], matrix_input);
    assert_eq!(outputs[1], narrow_left);
    assert_eq!(outputs[2], outputs[0]);
    assert_eq!(outputs[3], narrow_right);
}

#[test]
fn coreml_profile_bridges_boolean_abi_and_accelerates_the_component() {
    let bytes = graph_with_boolean_abi_component();
    let graph = bytes.parse().unwrap();
    let mut runtime = sessions();
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::coreml()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), [ACCEL_BACKEND; 6],);
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    let x = f32_bytes([-2.0, -1.0, 2.0, 3.0]);
    let active = [1_u8, 1, 1, 0];
    let outputs = runtime.run_in(&executable, &[&x, &active], Global).unwrap();
    assert_eq!(outputs[0], f32_bytes([4.0, 1.0, 4.0, 0.0]));
    assert_eq!(outputs[1], [1_u8, 1, 1, 0]);
    // Nonidentity boolean ABI output bridges cannot bypass conversion even
    // when the entire graph is a terminal provider partition.
    let mut field = [0xa5; 16];
    let mut mask = [0xa5; 4];
    runtime
        .run_into_in(
            &executable,
            &[&x, &active],
            &mut [
                axnn_vaccel::AxOutput::packed(&mut field),
                axnn_vaccel::AxOutput::packed(&mut mask),
            ],
            Global,
        )
        .unwrap();
    assert_eq!(field.as_slice(), outputs[0]);
    assert_eq!(mask.as_slice(), outputs[1]);
}

#[test]
fn region_placement_credits_launch_removed_by_partition_fusion() {
    let bytes = matmul_abs_matmul_chain(64);
    let graph = bytes.parse().unwrap();
    let model = AxDeviceModel {
        accelerator_launch_cost: 1_000,
        accelerator_speedup: 12,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let mut greedy = Schedule::analyze_in(&graph, Global).unwrap();
    greedy.place(&graph, &model).unwrap();
    assert_eq!(
        greedy.placements(),
        &[ACCEL_BACKEND, CPU_BACKEND, ACCEL_BACKEND],
        "the middle node must expose the accelerator-partition fusion case"
    );

    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().placements(),
        &[ACCEL_BACKEND, ACCEL_BACKEND, ACCEL_BACKEND]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
}

#[test]
fn region_placement_keeps_expensive_external_boundaries_on_cpu() {
    let bytes = cheap_elementwise_chain(96, 8192);
    let graph = bytes.parse().unwrap();
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 12,
        transfer_fixed_cost: 1_000_000,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };

    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|&backend| backend == CPU_BACKEND)
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
}

#[test]
fn region_placement_revisits_left_regions_after_right_demand_moves() {
    let bytes = leftward_region_dependency();
    let graph = bytes.parse().unwrap();
    let model = AxDeviceModel {
        accelerator_launch_cost: 1_000,
        accelerator_speedup: 12,
        transfer_fixed_cost: 8_000,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let mut greedy = Schedule::analyze_in(&graph, Global).unwrap();
    greedy.place(&graph, &model).unwrap();
    assert_eq!(greedy.placements()[0], CPU_BACKEND);
    assert_eq!(greedy.placements()[1], ACCEL_BACKEND);
    assert!(
        greedy.placements()[2..]
            .iter()
            .all(|&backend| backend == CPU_BACKEND)
    );

    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|&backend| backend == ACCEL_BACKEND)
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
}

#[test]
fn fused_ax_graph_compiles_once_and_reuses_accelerator_program() {
    let (m, dims) = (16, [8_usize, 16, 3]);
    let weights = vec![vec![0.03125; 8 * 16], vec![0.0625; 16 * 3]];
    let tosa = artifact::mlp_f16(m, &dims, &weights).unwrap();
    let ax_bytes = axtosa::lower_ax_in(&tosa, Global).unwrap();
    let graph = ax_bytes.parse().unwrap();
    let input = f16_bytes((0..m * 8).map(|index| index as f32 / 128.0 - 0.5));
    let expected = axtosa::execute(&tosa, &[&input]).unwrap();

    let mut runtime = sessions();
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let counts = Arc::new(AllocationCounts::default());
    let allocator = TrackingAllocator(Arc::clone(&counts));
    let executable = runtime.compile_in(&graph, &model, allocator).unwrap();
    assert!(counts.live.load(Ordering::Relaxed) > 0);
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|backend| *backend == ACCEL_BACKEND)
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
    assert!(executable.aggregate_cost() > 0);

    let run_counts = Arc::new(AllocationCounts::default());
    let first = runtime
        .run_in(
            &executable,
            &[&input],
            TrackingAllocator(Arc::clone(&run_counts)),
        )
        .unwrap();
    let warm_allocations = runtime
        .accel
        .as_ref()
        .unwrap()
        .provider()
        .buffer_allocations();
    let second = runtime
        .run_in(
            &executable,
            &[&input],
            TrackingAllocator(Arc::clone(&run_counts)),
        )
        .unwrap();
    assert_eq!(first[0].as_slice(), expected[0].as_slice());
    assert_eq!(
        second[0].as_slice(),
        first[0].as_slice(),
        "warm run changed output"
    );
    assert_eq!(warm_allocations, 2, "one input + one output arena");
    assert_eq!(
        runtime
            .accel
            .as_ref()
            .unwrap()
            .provider()
            .buffer_allocations(),
        warm_allocations,
        "warm run allocated new provider buffers"
    );
    assert!(
        run_counts.total.load(Ordering::Relaxed) > run_counts.live.load(Ordering::Relaxed),
        "run-time scratch allocations should be released"
    );
    drop(first);
    drop(second);
    assert_eq!(
        run_counts.live.load(Ordering::Relaxed),
        0,
        "run outputs escaped their supplied allocator"
    );
    assert!(
        counts.live.load(Ordering::Relaxed) > 0,
        "dropping run storage released persistent executable storage"
    );
    drop(executable);
    assert_eq!(
        counts.live.load(Ordering::Relaxed),
        0,
        "executable allocations escaped their supplied allocator"
    );
}

#[test]
fn scheduled_accelerator_to_cpu_boundary_executes_correctly() {
    // The shared MATMUL result is both a graph output and the ABS input. It
    // must reach the host regardless of ABS placement, while accelerating ABS
    // would add a second host egress for the final output.
    let bytes = matmul_abs_with_shared_host_output(64);
    let graph = bytes.parse().unwrap();
    let values: Vec<f32> = (0..64).map(|index| index as f32 / 16.0 - 2.0).collect();
    let input = f16_bytes(values.iter().copied());
    let expected_abs = f16_bytes(values.iter().map(|value| value.abs()));

    let mut runtime = sessions();
    let model = AxDeviceModel {
        accelerator_launch_cost: 300,
        accelerator_speedup: 12,
        transfer_fixed_cost: 100,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().placements(),
        &[ACCEL_BACKEND, CPU_BACKEND]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 2);

    let output = runtime.run_in(&executable, &[&input], Global).unwrap();
    assert_eq!(output[0], input);
    assert_eq!(output[1], expected_abs);
}

#[test]
fn caller_driven_mixed_graph_advances_one_provider_transition_at_a_time() {
    let bytes = matmul_abs_with_shared_host_output(64);
    let graph = bytes.parse().unwrap();
    let values: Vec<f32> = (0..64).map(|index| index as f32 / 16.0 - 2.0).collect();
    let input = f16_bytes(values.iter().copied());
    let expected_abs = f16_bytes(values.iter().map(|value| value.abs()));

    let cpu = Session::new(DelayedAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(
        DelayedAccelerator::adversarial(),
        AxTosaAccelerator::target(),
    )
    .unwrap();
    let mut runtime = AxHetero::new(cpu, Some(accel));
    let model = AxDeviceModel {
        accelerator_launch_cost: 300,
        accelerator_speedup: 12,
        transfer_fixed_cost: 100,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().plan().partitions.len(), 2);
    assert!(
        AxDispatchVariant::new(&executable, 0, 1, 100).is_none(),
        "a mixed executable cannot be represented by one backend reservation"
    );

    let mut invocation = runtime
        .begin_owned_in(&executable, &[&input], Global)
        .unwrap();
    let submitted = match runtime.advance_in(&mut invocation).unwrap() {
        AxInvocationStatus::Progress(progress) => progress,
        other => panic!("first step must only submit the accelerator partition: {other:?}"),
    };
    assert_eq!(submitted.partition, 0);
    assert_eq!(submitted.backend, ACCEL_BACKEND);
    assert_eq!(submitted.phase, AxPartitionPhase::Submitted);
    assert_eq!(submitted.submit_error, Some(BackendError::Busy));
    assert_eq!(invocation.pending_backend(), Some(ACCEL_BACKEND));

    // The synchronous compatibility path shares the exact fixed provider
    // buffers and must not race their live nonblocking invocation.
    assert!(matches!(
        runtime.run_in(&executable, &[&input], Global),
        Err(AxCompileError::Backend(BackendError::Busy))
    ));

    let waiting = match runtime.advance_in(&mut invocation).unwrap() {
        AxInvocationStatus::Progress(progress) => progress,
        other => panic!("delayed provider must remain pending for one poll: {other:?}"),
    };
    assert_eq!(waiting.phase, AxPartitionPhase::Waiting);
    assert!(matches!(
        runtime.advance_in(&mut invocation),
        Err(AxCompileError::Backend(BackendError::Busy))
    ));
    assert_eq!(
        runtime.accel.as_ref().unwrap().pending_count(),
        1,
        "rejected event release must return the live token for retry"
    );
    let accelerated = match runtime.advance_in(&mut invocation).unwrap() {
        AxInvocationStatus::Progress(progress) => progress,
        other => panic!("terminal poll should commit only partition zero: {other:?}"),
    };
    assert_eq!(accelerated.phase, AxPartitionPhase::Complete);
    assert_eq!(invocation.partition_index(), 1);

    let submitted = match runtime.advance_in(&mut invocation).unwrap() {
        AxInvocationStatus::Progress(progress) => progress,
        other => panic!("next step must submit the CPU partition: {other:?}"),
    };
    assert_eq!(submitted.partition, 1);
    assert_eq!(submitted.backend, CPU_BACKEND);
    assert_eq!(submitted.phase, AxPartitionPhase::Submitted);
    let waiting = match runtime.advance_in(&mut invocation).unwrap() {
        AxInvocationStatus::Progress(progress) => progress,
        other => panic!("CPU provider event must remain pending for one poll: {other:?}"),
    };
    assert_eq!(waiting.phase, AxPartitionPhase::Waiting);
    let outputs = match runtime.advance_in(&mut invocation).unwrap() {
        AxInvocationStatus::Complete(outputs) => outputs,
        other => panic!("terminal CPU poll must complete the graph: {other:?}"),
    };
    assert_eq!(outputs[0], input);
    assert_eq!(outputs[1], expected_abs);
    assert!(invocation.is_finished());
    assert!(matches!(
        runtime.advance_in(&mut invocation),
        Err(AxCompileError::InvocationFinished)
    ));
}

#[test]
fn bound_provider_outputs_preflight_wait_and_retry_release_without_host_planes() {
    use axnn_vaccel::AxOutput;
    let mut builder = GraphBuilder::new_in(Global);
    let input_id = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[4, 64], &[])
        .unwrap();
    let absolute = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4, 64], &[])
        .unwrap();
    let copied = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4, 64], &[])
        .unwrap();
    builder
        .node(
            OpCode::IDENTITY,
            NodeFlags::default(),
            &[input_id],
            &[copied],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::ABS,
            NodeFlags::default(),
            &[copied],
            &[absolute],
            &[],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let input = f32_bytes((0..256).map(|i| i as f32 - 128.0));
    let abs = f32_bytes((0..256).map(|i| (i as f32 - 128.0).abs()));
    let cpu = Session::new(
        DelayedAccelerator::adversarial(),
        AxTosaAccelerator::target(),
    )
    .unwrap();
    let mut runtime: AxHetero<_, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let model = AxDeviceModel {
        accelerator: false,
        cpu_kernels: None,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let executable = runtime
        .compile_in(&bytes.parse().unwrap(), &model, Global)
        .unwrap();
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
    let counts = Arc::new(AllocationCounts::default());
    let mut invocation = runtime
        .begin_in(&executable, &[&input], TrackingAllocator(counts.clone()))
        .unwrap();
    counts.largest.store(0, Ordering::Relaxed);
    assert!(matches!(
        runtime.advance_into_in(&mut invocation, &mut []),
        Err(AxCompileError::Backend(BackendError::InvalidArgument))
    ));
    assert_eq!(runtime.cpu.pending_count(), 0);
    assert_eq!(invocation.partition_index(), 0);
    let mut a = vec![0xa5; 7 + 4 * 272];
    let mut b = vec![0xa5; 1024];
    let mut rejected = 0;
    let mut completed = false;
    for _ in 0..8 {
        let status = runtime.advance_into_in(
            &mut invocation,
            &mut [
                AxOutput::pitched(&mut a, 7, 256, 4, 272).unwrap(),
                AxOutput::packed(&mut b),
            ],
        );
        match status {
            Ok(AxInvocationStatus::Complete(())) => {
                completed = true;
                break;
            }
            Ok(AxInvocationStatus::Progress(_)) => {}
            Err(AxCompileError::Backend(BackendError::Busy)) => {
                rejected += 1;
                assert!(invocation.pending_backend().is_some());
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(
            a.iter().chain(&b).all(|&v| v == 0xa5),
            "pending/rejected event exposed pixels"
        );
        assert_eq!(runtime.cpu.provider().inner.buffer_read_bytes(), 0);
    }
    assert!(completed);
    assert_eq!(rejected, 1);
    let mut expected = vec![0xa5; a.len()];
    for (row, data) in abs.as_chunks::<256>().0.iter().enumerate() {
        expected[7 + row * 272..7 + row * 272 + 256].copy_from_slice(data);
    }
    assert_eq!(a, expected);
    assert_eq!(b, input);
    assert!(
        counts.largest.load(Ordering::Relaxed) < 1024,
        "direct readback allocated a host tensor plane"
    );
    assert_eq!(runtime.cpu.provider().inner.buffer_read_bytes(), 2048);
    assert_eq!(
        invocation.host_copy_stats(),
        axnn_vaccel::AxHostCopyStats::default()
    );
    assert!(invocation.is_finished());
    assert!(matches!(
        runtime.advance_into_in(
            &mut invocation,
            &mut [
                AxOutput::pitched(&mut a, 7, 256, 4, 272).unwrap(),
                AxOutput::packed(&mut b)
            ]
        ),
        Err(AxCompileError::InvocationFinished)
    ));
    drop(invocation);
    assert_eq!(counts.live.load(Ordering::Relaxed), 0);
    runtime.release_executable(executable).unwrap();
}

#[test]
fn bound_native_outputs_match_legacy_and_reject_size_before_execution() {
    use axnn_vaccel::AxOutput;
    let mut builder = GraphBuilder::new_in(Global);
    let input_id = builder
        .tensor(
            DType::I32,
            TensorFlags::INPUT | TensorFlags::OUTPUT,
            &[2, 5],
            &[],
        )
        .unwrap();
    let output_id = builder
        .tensor(DType::I32, TensorFlags::OUTPUT, &[2, 5], &[])
        .unwrap();
    builder
        .node(
            OpCode::IDENTITY,
            NodeFlags::default(),
            &[input_id],
            &[output_id],
            &[],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    let input: Vec<u8> = [i32::MIN, i32::MAX, -1, 0, 16_777_217]
        .repeat(2)
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    let mut runtime = sessions();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        ..AxDeviceModel::default()
    };
    let executable = runtime
        .compile_in(&bytes.parse().unwrap(), &model, Global)
        .unwrap();
    let reference = runtime.run_in(&executable, &[&input], Global).unwrap();
    let mut passthrough = [0xa5; 40];
    let mut backing = [0xa5; 70];
    let mut wrong = [0xa5; 39];
    let mut invocation = runtime.begin_in(&executable, &[&input], Global).unwrap();
    assert!(matches!(
        runtime.advance_into_in(
            &mut invocation,
            &mut [
                AxOutput::packed(&mut passthrough),
                AxOutput::packed(&mut wrong)
            ]
        ),
        Err(AxCompileError::Backend(BackendError::InvalidArgument))
    ));
    assert_eq!(invocation.partition_index(), 0);
    assert!(!invocation.is_finished());
    assert_eq!(passthrough, [0xa5; 40]);
    runtime
        .run_into_in(
            &executable,
            &[&input],
            &mut [
                AxOutput::packed(&mut passthrough),
                AxOutput::pitched(&mut backing, 3, 20, 2, 32).unwrap(),
            ],
            Global,
        )
        .unwrap();
    assert_eq!(passthrough.as_slice(), reference[0]);
    let mut expected = [0xa5; 70];
    expected[3..23].copy_from_slice(&reference[1][..20]);
    expected[35..55].copy_from_slice(&reference[1][20..]);
    assert_eq!(backing, expected);
}

#[test]
fn independently_compiled_graphs_can_share_a_provider_queue_in_flight() {
    let bytes = matmul_abs_with_shared_host_output(64);
    let graph = bytes.parse().unwrap();
    let values: Vec<f32> = (0..64).map(|index| index as f32 / 16.0 - 2.0).collect();
    let input = f16_bytes(values.iter().copied());
    let expected_abs = f16_bytes(values.iter().map(|value| value.abs()));
    let cpu = Session::new(DelayedAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(DelayedAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(accel));
    let model = AxDeviceModel {
        accelerator_launch_cost: 300,
        accelerator_speedup: 12,
        transfer_fixed_cost: 100,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let interactive = runtime.compile_in(&graph, &model, Global).unwrap();
    let inference = runtime.compile_in(&graph, &model, Global).unwrap();
    let mut frame = runtime.begin_in(&interactive, &[&input], Global).unwrap();
    let mut model_run = runtime.begin_in(&inference, &[&input], Global).unwrap();

    assert!(matches!(
        runtime.advance_in(&mut frame).unwrap(),
        AxInvocationStatus::Progress(progress)
            if progress.phase == AxPartitionPhase::Submitted
    ));
    assert!(matches!(
        runtime.advance_in(&mut model_run).unwrap(),
        AxInvocationStatus::Progress(progress)
            if progress.phase == AxPartitionPhase::Submitted
    ));
    assert_eq!(runtime.accel.as_ref().unwrap().pending_count(), 2);

    // A second frame of the same executable would alias its fixed buffers;
    // callers must select another compiled quality slot or wait for release.
    let mut aliased = runtime.begin_in(&interactive, &[&input], Global).unwrap();
    assert!(matches!(
        runtime.advance_in(&mut aliased),
        Err(AxCompileError::Backend(BackendError::Busy))
    ));
    drop(aliased);

    let mut frame_outputs = None;
    let mut model_outputs = None;
    for _ in 0..8 {
        if frame_outputs.is_none() {
            if let AxInvocationStatus::Complete(outputs) = runtime.advance_in(&mut frame).unwrap() {
                frame_outputs = Some(outputs);
            }
        }
        if model_outputs.is_none() {
            if let AxInvocationStatus::Complete(outputs) =
                runtime.advance_in(&mut model_run).unwrap()
            {
                model_outputs = Some(outputs);
            }
        }
        if frame_outputs.is_some() && model_outputs.is_some() {
            break;
        }
    }
    for outputs in [frame_outputs.unwrap(), model_outputs.unwrap()] {
        assert_eq!(outputs[0], input);
        assert_eq!(outputs[1], expected_abs);
    }
    assert_eq!(runtime.cpu.pending_count(), 0);
    assert_eq!(runtime.accel.as_ref().unwrap().pending_count(), 0);
}

#[test]
fn admission_runs_interactive_cpu_graph_while_inference_owns_accelerator() {
    let frame_bytes = interactive_depth_field(4, 4);
    let frame_graph = frame_bytes.parse().unwrap();

    let (batch, dims) = (4, [8_usize, 16, 3]);
    let weights = vec![vec![0.03125; 8 * 16], vec![0.0625; 16 * 3]];
    let inference_tosa = artifact::mlp_f16(batch, &dims, &weights).unwrap();
    let inference_bytes = axtosa::lower_ax_in(&inference_tosa, Global).unwrap();
    let inference_graph = inference_bytes.parse().unwrap();
    let inference_input = f16_bytes((0..batch * dims[0]).map(|index| index as f32 / 128.0 - 0.5));
    let expected_inference = axtosa::execute(&inference_tosa, &[&inference_input]).unwrap();

    let cpu = Session::new(DelayedAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let accel = Session::new(DelayedAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(cpu, Some(accel));
    let cpu_model = AxDeviceModel {
        accelerator: false,
        ..AxDeviceModel::default()
    };
    let accelerator_model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let frame_cpu = runtime
        .compile_in(&frame_graph, &cpu_model, Global)
        .unwrap();
    let frame_accel = runtime
        .compile_in(&frame_graph, &accelerator_model, Global)
        .unwrap();
    let inference_accel = runtime
        .compile_in(&inference_graph, &accelerator_model, Global)
        .unwrap();

    let frame_variants = [
        AxDispatchVariant::new(&frame_cpu, 0, 16, 100).unwrap(),
        AxDispatchVariant::new(&frame_accel, 1, 8, 100).unwrap(),
    ];
    let inference_variant = AxDispatchVariant::new(&inference_accel, 2, 60, 100).unwrap();
    assert_eq!(frame_variants[0].candidate().backend, CPU_BACKEND);
    assert_eq!(frame_variants[1].candidate().backend, ACCEL_BACKEND);
    assert_eq!(inference_variant.candidate().backend, ACCEL_BACKEND);

    let mut lanes = [
        DispatchLane::new(CPU_BACKEND, 0),
        DispatchLane::new(ACCEL_BACKEND, 0),
    ];
    let mut planner = AdmissionPlanner::new(&mut lanes).unwrap();
    let inference_choice = admit_variants(
        &mut planner,
        DispatchRequest::new(0, u64::MAX, 100),
        &[inference_variant],
    )
    .unwrap();
    let inference_decision = inference_choice.decision();
    assert_eq!(inference_decision.backend, ACCEL_BACKEND);
    assert_eq!(inference_decision.finish_at, 60);

    let frame_choice = admit_variants(
        &mut planner,
        DispatchRequest::new(0, 16, 100),
        &frame_variants,
    )
    .unwrap();
    let frame_decision = frame_choice.decision();
    assert_eq!(frame_decision.candidate_index, 0);
    assert_eq!(frame_decision.backend, CPU_BACKEND);
    assert_eq!(frame_decision.finish_at, 16);
    assert!(frame_decision.deadline_met);
    assert_eq!(planner.lanes()[1].ready_at, 60);

    let center_x = 1.5_f32.to_le_bytes();
    let center_y = 1.5_f32.to_le_bytes();
    let mut model_run = runtime
        .begin_in(inference_choice.executable(), &[&inference_input], Global)
        .unwrap();
    let mut frame = runtime
        .begin_in(frame_choice.executable(), &[&center_x, &center_y], Global)
        .unwrap();

    assert!(matches!(
        runtime.advance_in(&mut model_run).unwrap(),
        AxInvocationStatus::Progress(progress)
            if progress.backend == ACCEL_BACKEND
                && progress.phase == AxPartitionPhase::Submitted
    ));
    assert!(matches!(
        runtime.advance_in(&mut frame).unwrap(),
        AxInvocationStatus::Progress(progress)
            if progress.backend == CPU_BACKEND
                && progress.phase == AxPartitionPhase::Submitted
    ));
    assert_eq!(runtime.accel.as_ref().unwrap().pending_count(), 1);
    assert_eq!(runtime.cpu.pending_count(), 1);

    let mut frame_outputs = None;
    let mut inference_outputs = None;
    for _ in 0..8 {
        if frame_outputs.is_none() {
            if let AxInvocationStatus::Complete(outputs) = runtime.advance_in(&mut frame).unwrap() {
                frame_outputs = Some(outputs);
            }
        }
        if inference_outputs.is_none() {
            if let AxInvocationStatus::Complete(outputs) =
                runtime.advance_in(&mut model_run).unwrap()
            {
                inference_outputs = Some(outputs);
            }
        }
        if frame_outputs.is_some() && inference_outputs.is_some() {
            break;
        }
    }

    let frame_outputs = frame_outputs.unwrap();
    let frame_values: Vec<f32> = frame_outputs[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    for (index, actual) in frame_values.into_iter().enumerate() {
        let x = (index % 4) as f32 + 0.5 - 1.5;
        let y = (index / 4) as f32 + 0.5 - 1.5;
        let expected = 1.0 / (1.0 + x * x + y * y);
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "depth lane {index}: {actual} vs {expected}"
        );
    }
    assert_eq!(inference_outputs.unwrap()[0], expected_inference[0]);
    assert_eq!(runtime.cpu.pending_count(), 0);
    assert_eq!(runtime.accel.as_ref().unwrap().pending_count(), 0);

    planner.reconcile(frame_decision, 14).unwrap();
    planner.reconcile(inference_decision, 55).unwrap();
    assert_eq!(
        planner.lanes(),
        &[
            DispatchLane::new(CPU_BACKEND, 14),
            DispatchLane::new(ACCEL_BACKEND, 55),
        ]
    );
}

#[test]
fn fp8_graph_promotes_matmuls_but_keeps_narrowing_on_cpu() {
    let (m, dims) = (16, [4_usize, 8, 6, 3]);
    let weights: Vec<Vec<u8>> = dims
        .windows(2)
        .enumerate()
        .map(|(layer, shape)| {
            (0..shape[0] * shape[1])
                .map(|index| {
                    let value = ((index * 17 + layer * 13) % 41) as f32 / 82.0 - 0.25;
                    axnn_cpu::fp8::f32_to_e4m3_rne(value)
                })
                .collect()
        })
        .collect();
    let tosa = artifact::mlp_b8(m, &dims, &weights).unwrap();
    let ax_bytes = axtosa::lower_ax_in(&tosa, Global).unwrap();
    let graph = ax_bytes.parse().unwrap();
    let input: Vec<u8> = (0..m * dims[0])
        .map(|index| axnn_cpu::fp8::f32_to_e4m3_rne(index as f32 / 64.0 - 0.5))
        .collect();
    let expected = axtosa::execute(&tosa, &[&input]).unwrap();

    let mut runtime = sessions();
    let model = AxDeviceModel {
        fp8_promotion: true,
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().placements(),
        &[
            ACCEL_BACKEND,
            ACCEL_BACKEND,
            CPU_BACKEND,
            ACCEL_BACKEND,
            ACCEL_BACKEND,
            CPU_BACKEND,
            ACCEL_BACKEND,
            ACCEL_BACKEND,
        ]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 5);

    let first = runtime.run_in(&executable, &[&input], Global).unwrap();
    let warm_accel_allocations = runtime
        .accel
        .as_ref()
        .unwrap()
        .provider()
        .buffer_allocations();
    let second = runtime.run_in(&executable, &[&input], Global).unwrap();
    assert_eq!(first, expected);
    assert_eq!(second, first);
    assert_eq!(
        runtime
            .accel
            .as_ref()
            .unwrap()
            .provider()
            .buffer_allocations(),
        warm_accel_allocations,
        "promoted accelerator partitions reallocated warm buffers"
    );

    let mut invocation = runtime.begin_in(&executable, &[&input], Global).unwrap();
    let mut steps = 0;
    let stepped = loop {
        steps += 1;
        match runtime.advance_in(&mut invocation).unwrap() {
            AxInvocationStatus::Progress(_) if steps < 16 => {}
            AxInvocationStatus::Complete(outputs) => break outputs,
            AxInvocationStatus::Progress(progress) => {
                panic!("five-partition graph did not terminate: {progress:?}")
            }
            AxInvocationStatus::Cancelled(progress) => {
                panic!("five-partition graph was unexpectedly cancelled: {progress:?}")
            }
        }
    };
    assert_eq!(steps, 10, "each provider partition submits, then polls");
    assert_eq!(stepped, expected);
}

#[test]
fn geometric_mask_graph_runs_as_one_backend_neutral_partition() {
    // A small but representative ray-state expression:
    // sign(x) / sqrt(x*x + y*y + epsilon). It exercises the operators a
    // fixed-step geodesic graph needs for normalization and masked updates.
    let shape = [4_u32];
    let scalar = [1_u32];
    let epsilon = 0.25_f32.to_le_bytes();
    let zero = 0.0_f32.to_le_bytes();
    let tensors = [
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags::INPUT,
            dimensions: &shape,
            constant: &[],
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags::INPUT,
            dimensions: &shape,
            constant: &[],
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags::CONSTANT,
            dimensions: &scalar,
            constant: &epsilon,
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags::CONSTANT,
            dimensions: &scalar,
            constant: &zero,
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags(0),
            dimensions: &shape,
            constant: &[],
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags(0),
            dimensions: &shape,
            constant: &[],
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags(0),
            dimensions: &shape,
            constant: &[],
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags(0),
            dimensions: &shape,
            constant: &[],
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags(0),
            dimensions: &shape,
            constant: &[],
        },
        TensorSpec {
            dtype: DType::U8,
            flags: TensorFlags(0),
            dimensions: &shape,
            constant: &[],
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags(0),
            dimensions: &shape,
            constant: &[],
        },
        TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags::OUTPUT,
            dimensions: &shape,
            constant: &[],
        },
    ];
    let mul_x = [TensorId(0), TensorId(0)];
    let mul_y = [TensorId(1), TensorId(1)];
    let add_xy = [TensorId(4), TensorId(5)];
    let add_epsilon = [TensorId(6), TensorId(2)];
    let rsqrt = [TensorId(7)];
    let greater = [TensorId(0), TensorId(3)];
    let negate = [TensorId(8)];
    let select = [TensorId(9), TensorId(8), TensorId(10)];
    let out4 = [TensorId(4)];
    let out5 = [TensorId(5)];
    let out6 = [TensorId(6)];
    let out7 = [TensorId(7)];
    let out8 = [TensorId(8)];
    let out9 = [TensorId(9)];
    let out10 = [TensorId(10)];
    let out11 = [TensorId(11)];
    let node = |op, inputs, outputs| NodeSpec {
        op,
        flags: NodeFlags::default(),
        inputs,
        outputs,
        attributes: &[],
    };
    let nodes = [
        node(OpCode::MUL, &mul_x, &out4),
        node(OpCode::MUL, &mul_y, &out5),
        node(OpCode::ADD, &add_xy, &out6),
        node(OpCode::ADD, &add_epsilon, &out7),
        node(OpCode::RSQRT, &rsqrt, &out8),
        node(OpCode::GREATER, &greater, &out9),
        node(OpCode::NEGATE, &negate, &out10),
        node(OpCode::SELECT, &select, &out11),
    ];
    let bytes = GraphBuf::encode_in(&tensors, &nodes, Global).unwrap();
    let graph = bytes.parse().unwrap();

    let x = [-3.0_f32, -1.0, 1.0, 3.0];
    let y = [4.0_f32, 0.0, 0.0, 4.0];
    let input_x = f32_bytes(x);
    let input_y = f32_bytes(y);
    let analyzed = Schedule::analyze_in(&graph, Global).unwrap();
    let partition = Partition {
        backend: ACCEL_BACKEND,
        implementation: DENSE_IMPLEMENTATION,
        order_start: 0,
        node_count: graph.node_count() as u32,
    };
    let tosa = artifact::ax_partition_in(
        &graph,
        analyzed.order(),
        analyzed.producers(),
        partition,
        Global,
    )
    .unwrap();
    let direct = axtosa::execute(&tosa.bytes, &[&input_x, &input_y]).unwrap();
    let roundtrip = axtosa::lower_ax_in(&tosa.bytes, Global).unwrap();
    let roundtrip = roundtrip.parse().unwrap();
    assert_eq!(roundtrip.node_count(), graph.node_count());
    let mut original_ops: Vec<_> = (0..graph.node_count())
        .map(|index| graph.node(index).unwrap().op)
        .collect();
    let mut roundtrip_ops: Vec<_> = (0..roundtrip.node_count())
        .map(|index| roundtrip.node(index).unwrap().op)
        .collect();
    original_ops.sort_by_key(|op| op.0);
    roundtrip_ops.sort_by_key(|op| op.0);
    assert_eq!(roundtrip_ops, original_ops);
    let mut runtime = sessions();
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::coreml()
    };
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert!(
        executable
            .schedule()
            .placements()
            .iter()
            .all(|backend| *backend == ACCEL_BACKEND)
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    let output = runtime
        .run_in(&executable, &[&input_x, &input_y], Global)
        .unwrap();
    assert_eq!(output, direct);
    let actual: Vec<f32> = output[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    for (index, got) in actual.into_iter().enumerate() {
        let magnitude = (x[index] * x[index] + y[index] * y[index] + 0.25).sqrt();
        let expected = x[index].signum() / magnitude;
        assert!(
            (got - expected).abs() <= 1.0e-6,
            "lane {index}: {got} vs {expected}"
        );
    }

    // GREATER (5) and SELECT (7) share the internal Boolean mask. Core ML
    // may place both on either backend, but no cost calibration may split the
    // component and turn that mask into provider model I/O.
    for (launch, speedup, transfer) in [
        (0, 32, 0),
        (1_000, 12, 0),
        (250_000, 12, 50_000),
        (1_000_000, 1, 250_000),
    ] {
        let profile = AxDeviceModel {
            accelerator_launch_cost: launch,
            accelerator_speedup: speedup,
            transfer_fixed_cost: transfer,
            ..AxDeviceModel::coreml()
        };
        let executable = runtime.compile_in(&graph, &profile, Global).unwrap();
        assert_eq!(
            executable.schedule().placements()[5],
            executable.schedule().placements()[7],
            "Boolean component split for launch={launch}, speedup={speedup}, transfer={transfer}"
        );
    }
}

#[test]
fn wgsl_semantic_recipes_execute_as_one_accelerator_partition() {
    let shape = [4_u32];
    let scalar = [1_u32];
    let mut builder = GraphBuilder::new_in(Global);
    let x = builder
        .tensor(DType::F32, TensorFlags::INPUT, &shape, &[])
        .unwrap();
    let divisor = builder
        .tensor(DType::F32, TensorFlags::INPUT, &shape, &[])
        .unwrap();
    let lo = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &scalar,
            &0.75_f32.to_le_bytes(),
        )
        .unwrap();
    let hi = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &scalar,
            &1.75_f32.to_le_bytes(),
        )
        .unwrap();
    let scale = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &scalar,
            &0.5_f32.to_le_bytes(),
        )
        .unwrap();
    let bias = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &scalar,
            &0.25_f32.to_le_bytes(),
        )
        .unwrap();
    let mut intermediate = || {
        builder
            .tensor(DType::F32, TensorFlags::default(), &shape, &[])
            .unwrap()
    };
    let quotient = intermediate();
    let root = intermediate();
    let clamped = intermediate();
    let affine = intermediate();
    let exponent = intermediate();
    let logarithm = intermediate();
    drop(intermediate);
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &shape, &[])
        .unwrap();
    for (op, inputs, output) in [
        (OpCode::DIV, vec![x, divisor], quotient),
        (OpCode::SQRT, vec![quotient], root),
        (OpCode::CLAMP, vec![root, lo, hi], clamped),
        (OpCode::FMA, vec![clamped, scale, bias], affine),
        (OpCode::EXP2, vec![affine], exponent),
        (OpCode::LOG2, vec![exponent], logarithm),
        (OpCode::TAN, vec![logarithm], output),
    ] {
        builder
            .node(op, NodeFlags::default(), &inputs, &[output], &[])
            .unwrap();
    }
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();

    let capability = AxTosaAccelerator::new();
    let model = AxDeviceModel {
        cpu: false,
        accelerator_launch_cost: 0,
        accelerator_speedup: 32,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    }
    .with_accelerator_tosa_capability(&capability, AxTosaAccelerator::target());
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), [ACCEL_BACKEND; 7]);
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    let x_values = [0.25_f32, 1.0, 4.0, 9.0];
    let divisors = [0.5_f32, 2.0, 2.0, 3.0];
    let x_bytes = f32_bytes(x_values);
    let divisor_bytes = f32_bytes(divisors);
    let result = runtime
        .run_in(&executable, &[&x_bytes, &divisor_bytes], Global)
        .unwrap();
    let actual: Vec<f32> = result[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    for (lane, (&x, &divisor)) in x_values.iter().zip(&divisors).enumerate() {
        let value = (x / divisor).sqrt().clamp(0.75, 1.75) * 0.5 + 0.25;
        let expected = value.tan();
        assert!(
            (actual[lane] - expected).abs() <= 3.0e-4,
            "lane {lane}: {} vs {expected}",
            actual[lane]
        );
    }
}

#[test]
fn native_axnn_plan_executes_broadcast_chain_as_one_cpu_partition() {
    let mut builder = GraphBuilder::new_in(Global);
    let y = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[2, 1], &[])
        .unwrap();
    let x = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[1, 3], &[])
        .unwrap();
    let angle = builder
        .tensor(DType::F32, TensorFlags::default(), &[2, 3], &[])
        .unwrap();
    let rounded = builder
        .tensor(DType::F32, TensorFlags::default(), &[2, 3], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 3], &[])
        .unwrap();
    builder
        .node(OpCode::ATAN2, NodeFlags::default(), &[y, x], &[angle], &[])
        .unwrap();
    builder
        .node(
            OpCode::ROUND,
            NodeFlags::default(),
            &[angle],
            &[rounded],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::RECIPROCAL,
            NodeFlags::default(),
            &[rounded],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();

    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().implementations(),
        [CPU_KERNEL_IMPLEMENTATION; 3]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    // Offset both planes by one byte to prove native execution never casts an
    // arbitrary graph ABI pointer to `&[f32]`.
    let mut y_storage = vec![0x55];
    y_storage.extend(f32_bytes([-1.0, 1.0]));
    let mut x_storage = vec![0xaa];
    x_storage.extend(f32_bytes([1.0, -1.0, 0.5]));
    let result = runtime
        .run_in(&executable, &[&y_storage[1..], &x_storage[1..]], Global)
        .unwrap();
    let actual: Vec<f32> = result[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let expected: Vec<f32> = [-1.0_f32, 1.0]
        .into_iter()
        .flat_map(|y| [1.0_f32, -1.0, 0.5].map(move |x| 1.0 / (y.atan2(x).round_ties_even())))
        .collect();
    for (lane, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "native lane {lane}: {actual} vs {expected}"
        );
    }
}

#[test]
fn native_axnn_catalog_executes_tuned_matmul_without_provider_lowering() {
    let mut builder = GraphBuilder::new_in(Global);
    let lhs = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[2, 3], &[])
        .unwrap();
    let rhs = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[3, 4], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 4], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[lhs, rhs],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();

    let executable = runtime.compile_in(&graph, &model, Global).unwrap();

    assert_eq!(
        executable.schedule().implementations(),
        [CPU_KERNEL_IMPLEMENTATION]
    );
    let lhs = f32_bytes([1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = f32_bytes([
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    let result = runtime.run_in(&executable, &[&lhs, &rhs], Global).unwrap();
    let actual: Vec<f32> = result[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(actual, [38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]);
}

#[test]
fn native_axnn_plan_keeps_boolean_masks_typed_and_completes_inline() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let zero = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1],
            &0.0_f32.to_le_bytes(),
        )
        .unwrap();
    let mask = builder
        .tensor(DType::U8, TensorFlags::default(), &[4], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    builder
        .node(
            OpCode::GREATER,
            NodeFlags::default(),
            &[input, zero],
            &[mask],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::SELECT,
            NodeFlags::default(),
            &[mask, input, zero],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().implementations(),
        [CPU_KERNEL_IMPLEMENTATION; 2]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    let input = f32_bytes([-2.0, -0.0, 0.5, 3.0]);
    let mut invocation = runtime.begin_in(&executable, &[&input], Global).unwrap();
    let AxInvocationStatus::Complete(result) = runtime.advance_in(&mut invocation).unwrap() else {
        panic!("native partition must complete without a provider event");
    };
    let actual: Vec<f32> = result[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(actual, [0.0, 0.0, 0.5, 3.0]);

    assert!(matches!(
        runtime.run_repeated_in(&executable, &[&input], 2, &[], Global),
        Err(AxCompileError::ResidentRequiresSinglePartition)
    ));
}

#[test]
fn canonical_integer_words_and_casts_match_prepared_and_ordinary_execution() {
    use axnn_cpu::CpuCapabilities;
    use axnn_vaccel::AxCpuKernelPolicy;

    fn output(
        builder: &mut GraphBuilder<Global>,
        op: OpCode,
        inputs: &[TensorId],
        dtype: DType,
        dimensions: &[u32],
    ) -> TensorId {
        let result = builder
            .tensor(dtype, TensorFlags::OUTPUT, dimensions, &[])
            .unwrap();
        builder
            .node(op, NodeFlags::default(), inputs, &[result], &[])
            .unwrap();
        result
    }

    let mut builder = GraphBuilder::new_in(Global);
    let x = builder
        .tensor(DType::I32, TensorFlags::INPUT, &[3, 1], &[])
        .unwrap();
    let y = builder
        .tensor(DType::I32, TensorFlags::INPUT, &[1, 7], &[])
        .unwrap();
    let mask = builder
        .tensor(DType::U8, TensorFlags::INPUT, &[3, 7], &[])
        .unwrap();
    let floats = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[3, 7], &[])
        .unwrap();
    let operators = [
        OpCode::ADD,
        OpCode::SUB,
        OpCode::MUL,
        OpCode::DIV,
        OpCode::REMAINDER,
        OpCode::BITWISE_AND,
        OpCode::BITWISE_OR,
        OpCode::BITWISE_XOR,
        OpCode::SHIFT_LEFT,
        OpCode::SHIFT_RIGHT,
    ];
    for op in operators {
        output(&mut builder, op, &[x, y], DType::I32, &[3, 7]);
    }
    output(
        &mut builder,
        OpCode::SELECT,
        &[mask, x, y],
        DType::I32,
        &[3, 7],
    );
    output(&mut builder, OpCode::EQUAL, &[x, y], DType::U8, &[3, 7]);
    let promoted = output(&mut builder, OpCode::CAST, &[x], DType::F32, &[3, 1]);
    output(&mut builder, OpCode::CAST, &[promoted], DType::I32, &[3, 1]);
    output(&mut builder, OpCode::CAST, &[floats], DType::I32, &[3, 7]);
    output(&mut builder, OpCode::CAST, &[floats], DType::F32, &[3, 7]);
    output(&mut builder, OpCode::CAST, &[x], DType::I32, &[3, 1]);
    let serialized = builder.finish().unwrap().as_bytes().to_vec();
    let encoded = GraphBuf::copy_from_in(&serialized, Global).unwrap();
    let graph = encoded.parse().unwrap();
    let dtypes: Vec<_> = (0..graph.tensor_count())
        .map(|i| graph.tensor(TensorId(i as u32)).unwrap())
        .filter(|tensor| tensor.flags.contains(TensorFlags::OUTPUT))
        .map(|tensor| tensor.dtype)
        .collect();

    let floating_values = [
        f32::NEG_INFINITY,
        -2_147_483_648.0,
        -f32::MAX,
        -16_777_218.0,
        -3.9,
        -0.0,
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xff81_2345),
        -f32::from_bits(1),
        f32::from_bits(1),
        0.0,
        3.9,
        16_777_216.0,
        16_777_218.0,
        2_147_483_520.0,
        2_147_483_648.0,
        f32::INFINITY,
        f32::MAX,
        1.5,
        -1.5,
        255.9,
    ];
    let converted_values = [
        i32::MIN,
        i32::MIN,
        i32::MIN,
        -16_777_218,
        -3,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        16_777_216,
        16_777_218,
        2_147_483_520,
        i32::MAX,
        i32::MAX,
        i32::MAX,
        1,
        -1,
        255,
    ];
    for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
        let model = AxDeviceModel {
            accelerator: false,
            cpu_tosa: false,
            sparse_bsr: None,
            cpu_kernels: Some(AxCpuKernelPolicy {
                capabilities,
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut runtime = sessions();
        let executable = runtime.compile_in(&graph, &model, Global).unwrap();
        assert_eq!(executable.schedule().plan().partitions.len(), 1);
        let counts = Arc::new(AllocationCounts::default());
        let mut prepared = executable
            .prepare_native_in(TrackingAllocator(Arc::clone(&counts)))
            .unwrap();
        let allocations = counts.total.load(Ordering::Relaxed);
        for generation in 0..2 {
            let x_values = if generation == 0 {
                [i32::MIN, i32::MAX, 16_777_217]
            } else {
                [i32::MAX, i32::MIN + 1, -16_777_217]
            };
            let y_values = if generation == 0 {
                [-1, 0, 1, 31, 32, 33, 63]
            } else {
                [65, -33, -32, 16, i32::MIN, i32::MAX, -7]
            };
            let masks: Vec<_> = (0..21)
                .map(|i| [0, 1, 2, 128, 255][(i + generation) % 5])
                .collect();
            let float_values: Vec<_> = (0..21)
                .map(|i| floating_values[(i + generation * 7) % 21])
                .collect();
            let x_bytes = i32_bytes(x_values);
            let y_bytes = i32_bytes(y_values);
            let float_bytes = f32_bytes(float_values);
            let inputs: [&[u8]; 4] = [&x_bytes, &y_bytes, &masks, &float_bytes];
            let pairs: Vec<_> = x_values
                .into_iter()
                .flat_map(|a| y_values.into_iter().map(move |b| (a, b)))
                .collect();
            let mut expected: Vec<Vec<u8>> = operators
                .into_iter()
                .map(|op| {
                    i32_bytes(pairs.iter().map(|&(a, b)| match op {
                        OpCode::ADD => (i64::from(a) + i64::from(b)) as i32,
                        OpCode::SUB => (i64::from(a) - i64::from(b)) as i32,
                        OpCode::MUL => (i64::from(a) * i64::from(b)) as i32,
                        OpCode::DIV => {
                            if b == 0 {
                                a
                            } else {
                                (i64::from(a) / i64::from(b)) as i32
                            }
                        }
                        OpCode::REMAINDER => {
                            if b == 0 {
                                0
                            } else {
                                (i64::from(a) % i64::from(b)) as i32
                            }
                        }
                        OpCode::BITWISE_AND => a & b,
                        OpCode::BITWISE_OR => a | b,
                        OpCode::BITWISE_XOR => a ^ b,
                        OpCode::SHIFT_LEFT => {
                            ((u64::from(a as u32) << (b as u32 & 31)) & u64::from(u32::MAX)) as i32
                        }
                        OpCode::SHIFT_RIGHT => (i64::from(a) >> (b as u32 & 31)) as i32,
                        _ => unreachable!(),
                    }))
                })
                .collect();
            expected.push(i32_bytes(
                pairs
                    .iter()
                    .zip(&masks)
                    .map(|(&(a, b), &m)| if m == 0 { b } else { a }),
            ));
            expected.push(pairs.iter().map(|&(a, b)| u8::from(a == b)).collect());
            let promoted_values = if generation == 0 {
                [-2_147_483_648.0, 2_147_483_648.0, 16_777_216.0]
            } else {
                [2_147_483_648.0, -2_147_483_648.0, -16_777_216.0]
            };
            expected.push(f32_bytes(promoted_values));
            expected.push(i32_bytes(if generation == 0 {
                [i32::MIN, i32::MAX, 16_777_216]
            } else {
                [i32::MAX, i32::MIN, -16_777_216]
            }));
            expected.push(i32_bytes(
                (0..21).map(|i| converted_values[(i + generation * 7) % 21]),
            ));
            // Same-type CAST is a storage identity, including signed zero and
            // every NaN payload; it must not reuse the converted integer plane.
            expected.push(float_bytes.clone());
            expected.push(x_bytes.clone());
            let ordinary = runtime.run_in(&executable, &inputs, Global).unwrap();
            prepared.run(&inputs).unwrap();
            assert_eq!(
                ordinary, expected,
                "ordinary {capabilities:?} generation {generation}"
            );
            assert_eq!(
                prepared_native_bytes(&prepared, &dtypes),
                expected,
                "prepared {capabilities:?} generation {generation}"
            );
        }
        assert_eq!(
            counts.total.load(Ordering::Relaxed),
            allocations,
            "warm integer execution allocated"
        );
        drop(prepared);
        assert_eq!(counts.live.load(Ordering::Relaxed), 0);
        assert_eq!(runtime.cpu.provider().buffer_allocations(), 0);
        assert_eq!(runtime.cpu.provider().buffer_write_bytes(), 0);
    }
}

#[test]
fn wgsl_integer_hash_and_byte_loads_observe_each_borrow_without_losing_ids() {
    use ax::buffer::U8BufferParams;
    use ax::voxel::ResourceKey;
    use axnn_cpu::{BrickGridRef, CpuCapabilities};
    use axnn_vaccel::{AxCpuKernelPolicy, NativeResourceBinding, NativeResourceKind, U8BufferRef};
    use axshader::{CompileOptions, ShaderResource, compile_wgsl_with_resource_specs_in};

    const KEY: ResourceKey = ResourceKey {
        group: 2,
        binding: 7,
    };
    const BASE: i32 = 16_777_217;
    fn hash(mut value: u32) -> u32 {
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb352d);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846ca68b);
        value ^ (value >> 16)
    }
    let source = r#"
        enable ax_buffer_access;
        @group(2) @binding(7) var<storage, read> materials: AxU8Buffer;
        struct Result {
            @location(0) @interpolate(flat) id: i32,
            @location(1) @interpolate(flat) hash: i32,
            @location(2) @interpolate(flat) material: i32,
            @location(3) grain: f32,
        }
        fn hash(value: i32) -> i32 {
            let a = (value ^ ((value >> 16u) & 65535i)) * 2146121005i;
            let b = (a ^ ((a >> 15u) & 131071i)) * -2073254261i;
            return b ^ ((b >> 16u) & 65535i);
        }
        @fragment fn main(@builtin(position) p: vec4<f32>) -> Result {
            let index = i32(p.x);
            let id = 16777217i + index;
            let hashed = hash(id);
            let low = (hashed & 2147483647i) % 101i;
            let residue = select(low, (low + 34i) % 101i, hashed < 0i);
            let material = ax_load_u8(&materials, select(-1i, index, p.y >= 0.0));
            return Result(id, hashed, material, 0.90 + f32(residue) * 0.0015);
        }
    "#;
    let serialized = compile_wgsl_with_resource_specs_in(
        source,
        CompileOptions::for_viewport("main", 11, 2),
        &[ShaderResource::U8Buffer(
            U8BufferParams::new(KEY, 256).unwrap(),
        )],
        Global,
    )
    .unwrap()
    .graph()
    .as_bytes()
    .to_vec();
    let encoded = GraphBuf::copy_from_in(&serialized, Global).unwrap();
    let graph = encoded.parse().unwrap();
    let xs = [
        -1.0,
        0.0,
        1.0,
        2.0,
        127.0,
        128.0,
        254.0,
        255.0,
        256.0,
        1024.0,
        -2_147_483_648.0,
    ];
    let ys = [-1.0, 1.0];
    let x_bytes = f32_bytes(xs);
    let y_bytes = f32_bytes(ys);
    let inputs: [&[u8]; 2] = [&x_bytes, &y_bytes];
    let ids: Vec<_> = ys
        .into_iter()
        .flat_map(|_| xs.into_iter().map(|x| BASE.wrapping_add(x as i32)))
        .collect();
    let hashed: Vec<_> = ids.iter().map(|&id| hash(id as u32)).collect();
    let grain = f32_bytes(hashed.iter().map(|&h| {
        let scaled = (h % 101) as f32 * 0.0015;
        0.90 + scaled
    }));
    for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
        let model = AxDeviceModel {
            accelerator: false,
            cpu_tosa: false,
            sparse_bsr: None,
            cpu_kernels: Some(AxCpuKernelPolicy {
                capabilities,
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut runtime = sessions();
        let executable = runtime.compile_in(&graph, &model, Global).unwrap();
        assert_eq!(executable.schedule().plan().partitions.len(), 1);
        let counts = Arc::new(AllocationCounts::default());
        let mut prepared = executable
            .prepare_native_in(TrackingAllocator(Arc::clone(&counts)))
            .unwrap();
        let allocations = counts.total.load(Ordering::Relaxed);
        let mut bytes = [0_u8; 256];
        for generation in 0..3 {
            for (index, value) in bytes.iter_mut().enumerate() {
                *value = (index as u8).wrapping_add(generation * 73);
            }
            let resources = [NativeResourceBinding::u8_buffer(
                KEY,
                U8BufferRef::new(&bytes).unwrap(),
            )];
            let expected_materials = ys
                .into_iter()
                .flat_map(|y| xs.into_iter().map(move |x| (y, x)))
                .map(|(y, x)| {
                    if y < 0.0 {
                        0
                    } else {
                        usize::try_from(x as i32)
                            .ok()
                            .and_then(|i| bytes.get(i))
                            .copied()
                            .unwrap_or(0) as i32
                    }
                })
                .collect::<Vec<_>>();
            let expected = [
                i32_bytes(ids.iter().copied()),
                i32_bytes(hashed.iter().map(|&h| h as i32)),
                i32_bytes(expected_materials),
                grain.clone(),
            ];
            let ordinary = runtime
                .run_with_resources_in(&executable, &inputs, &resources, Global)
                .unwrap();
            prepared.run_with_resources(&inputs, &resources).unwrap();
            assert_eq!(
                ordinary, expected,
                "ordinary {capabilities:?} generation {generation}"
            );
            assert_eq!(
                prepared_native_bytes(&prepared, &[DType::I32, DType::I32, DType::I32, DType::F32]),
                expected,
                "prepared {capabilities:?} generation {generation}"
            );

            let brick = [0_u64];
            let wrong_kind = [NativeResourceBinding::brick_grid(
                KEY,
                BrickGridRef::new(&brick, [4; 3]).unwrap(),
            )];
            let wrong_length = [NativeResourceBinding::u8_buffer(
                KEY,
                U8BufferRef::new(&bytes[..255]).unwrap(),
            )];
            let duplicate = [resources[0], resources[0]];
            for (case, invalid) in [
                (0, &[][..]),
                (1, &wrong_kind[..]),
                (2, &wrong_length[..]),
                (3, &duplicate[..]),
            ] {
                let error = prepared.run_with_resources(&inputs, invalid).unwrap_err();
                assert!(match (case, error) {
                    (0, AxCompileError::MissingResource(key))
                    | (3, AxCompileError::DuplicateResource(key)) => key == KEY,
                    (
                        1,
                        AxCompileError::ResourceKind {
                            resource,
                            expected: NativeResourceKind::U8Buffer,
                            actual: NativeResourceKind::BrickGrid,
                        },
                    ) => resource == KEY,
                    (
                        2,
                        AxCompileError::ResourceLength {
                            resource,
                            expected: 256,
                            actual: 255,
                        },
                    ) => resource == KEY,
                    (_, error) => panic!("unexpected byte-buffer failure: {error:?}"),
                });
                assert!(prepared.output_i32(0).is_none());
                assert!(prepared.output_i32(2).is_none());
                assert!(prepared.output_f32(3).is_none());
                prepared.run_with_resources(&inputs, &resources).unwrap();
                assert_eq!(
                    prepared_native_bytes(
                        &prepared,
                        &[DType::I32, DType::I32, DType::I32, DType::F32]
                    ),
                    expected
                );
            }
        }
        assert_eq!(
            counts.total.load(Ordering::Relaxed),
            allocations,
            "warm resource execution or preflight allocated"
        );
        drop(prepared);
        assert_eq!(counts.live.load(Ordering::Relaxed), 0);
        assert_eq!(runtime.cpu.provider().buffer_allocations(), 0);
        assert_eq!(
            runtime.cpu.provider().buffer_write_bytes(),
            0,
            "byte resources must not upload through the tensor provider"
        );
    }
}

#[test]
fn wgsl_math_operator_corpus_executes_natively() {
    // Inputs are runtime axes, so Naga cannot constant-fold the operations
    // under test. The non-square field also exercises broadcast indexing and
    // non-vector-width tails. Keep undefined WGSL domains out of the corpus.
    type Case = (&'static str, &'static str, fn(f32, f32) -> [f32; 4], f32);
    let cases: &[Case] = &[
        (
            "arithmetic",
            "return vec4<f32>(-x, x + y, x - y, x * y);",
            |x, y| [-x, x + y, x - y, x * y],
            0.0,
        ),
        (
            "division and truncating remainder",
            "let d = select(-2.0, 2.0, y >= 0.0);\n\
             return vec4<f32>(1.0 / d, x / d, x % d, (-x) % d);",
            |x, y| {
                let d = if y >= 0.0 { 2.0 } else { -2.0 };
                [1.0 / d, x / d, x % d, (-x) % d]
            },
            0.0,
        ),
        (
            "rounding including ties to even",
            "return vec4<f32>(ceil(x), floor(x), round(x), trunc(x));",
            |x, _| [x.ceil(), x.floor(), x.round_ties_even(), x.trunc()],
            0.0,
        ),
        (
            "fraction sign and saturation",
            "return vec4<f32>(fract(x), sign(x), abs(x), saturate(x));",
            |x, _| {
                let sign = if x == 0.0 { 0.0 } else { x.signum() };
                [x - x.floor(), sign, x.abs(), x.clamp(0.0, 1.0)]
            },
            0.0,
        ),
        (
            "minimum maximum and clamp",
            "return vec4<f32>(min(x, y), max(x, y),\n\
                 clamp(x, -1.0, 1.0), clamp(y, -0.5, 0.5));",
            |x, y| [x.min(y), x.max(y), x.clamp(-1.0, 1.0), y.clamp(-0.5, 0.5)],
            0.0,
        ),
        (
            "trigonometric and hyperbolic",
            "return vec4<f32>(sin(x), cos(x), tan(x * 0.25), tanh(x));",
            |x, _| [x.sin(), x.cos(), (x * 0.25).tan(), x.tanh()],
            4.0e-6,
        ),
        (
            "inverse trigonometric including domain endpoints",
            "let angle_y = select(x, 0.125, x == 0.0);\n\
             return vec4<f32>(asin(x * 0.4), acos(x * 0.4),\n\
                 atan(x), atan2(angle_y, y + 0.25));",
            |x, y| {
                let angle_y = if x == 0.0 { 0.125 } else { x };
                [
                    (x * 0.4).asin(),
                    (x * 0.4).acos(),
                    x.atan(),
                    angle_y.atan2(y + 0.25),
                ]
            },
            4.0e-6,
        ),
        (
            "exponential and logarithmic",
            "let positive = abs(x) + 0.25;\n\
             return vec4<f32>(exp(x), exp2(x), log(positive), log2(positive));",
            |x, _| {
                let positive = x.abs() + 0.25;
                [x.exp(), x.exp2(), positive.ln(), positive.log2()]
            },
            4.0e-6,
        ),
        (
            "power square root and inverse square root",
            "let positive = abs(x) + 0.25;\n\
             return vec4<f32>(pow(positive, y), sqrt(positive),\n\
                 inverseSqrt(positive), sqrt(abs(x)));",
            |x, y| {
                let positive = x.abs() + 0.25;
                [
                    positive.powf(y),
                    positive.sqrt(),
                    1.0 / positive.sqrt(),
                    x.abs().sqrt(),
                ]
            },
            4.0e-6,
        ),
        (
            "interpolation and step boundaries",
            "return vec4<f32>(mix(x, y, 0.25), fma(x, y, 0.25),\n\
                 step(y, x), smoothstep(-1.0, 1.0, x));",
            |x, y| {
                let t = ((x + 1.0) / 2.0).clamp(0.0, 1.0);
                [
                    x * 0.75 + y * 0.25,
                    x.mul_add(y, 0.25),
                    if x >= y { 1.0 } else { 0.0 },
                    t * t * (3.0 - 2.0 * t),
                ]
            },
            0.0,
        ),
        (
            "radians and degrees",
            "return vec4<f32>(radians(x), degrees(x),\n\
                 radians(degrees(x)), degrees(radians(x)));",
            |x, _| [x.to_radians(), x.to_degrees(), x, x],
            4.0e-6,
        ),
        (
            "ordered comparisons and vector select",
            "return select(vec4<f32>(0.0), vec4<f32>(1.0),\n\
                 vec4<bool>((x < y), (x <= y), (x > y), (x >= y)));",
            |x, y| [x < y, x <= y, x > y, x >= y].map(|value| u8::from(value) as f32),
            0.0,
        ),
        (
            "equality and composed Boolean expressions",
            "let equal = x == y;\n\
             let unequal = x != y;\n\
             return vec4<f32>(select(0.0, 1.0, equal), select(0.0, 1.0, unequal),\n\
                 select(0.0, 1.0, !(x < y) && !(x > y)),\n\
                 select(0.0, 1.0, equal || x > y));",
            |x, y| [x == y, x != y, x == y, x >= y].map(|value| u8::from(value) as f32),
            0.0,
        ),
        (
            "vector arithmetic and scalar lane broadcast",
            "let v = vec4<f32>(x, y, x + y, x - y);\n\
             return (2.0 * v + vec4<f32>(1.0, -1.0, 0.5, -0.5)) / 2.0;",
            |x, y| [x + 0.5, y - 0.5, x + y + 0.25, x - y - 0.25],
            0.0,
        ),
        (
            "vector dot length and distance",
            "let a = vec3<f32>(x, y, 1.0);\n\
             let b = vec3<f32>(y, -x, 2.0);\n\
             return vec4<f32>(dot(a, b), length(a), distance(a, b),\n\
                 dot(vec2<f32>(x, y), vec2<f32>(2.0, 3.0)));",
            |x, y| {
                [
                    2.0,
                    (x * x + y * y + 1.0).sqrt(),
                    ((x - y) * (x - y) + (y + x) * (y + x) + 1.0).sqrt(),
                    2.0 * x + 3.0 * y,
                ]
            },
            4.0e-6,
        ),
        (
            "normalization and swizzle",
            "let direction = normalize(vec4<f32>(x, y, 1.0, 2.0));\n\
             return direction.wzyx;",
            |x, y| {
                let length = (x * x + y * y + 5.0).sqrt();
                [2.0 / length, 1.0 / length, y / length, x / length]
            },
            4.0e-6,
        ),
        (
            "constructors constant indexing and repeated swizzles",
            "let v = vec4<f32>(vec2<f32>(x, y), vec2<f32>(3.0, 4.0));\n\
             return vec4<f32>(v[1], v.xy.y + v.yz.x, v.zw);",
            |_, y| [y, 2.0 * y, 3.0, 4.0],
            0.0,
        ),
        (
            "vector builtins and scalar interpolation weight",
            "let v = vec4<f32>(x, y, x + y, x - y);\n\
             return mix(abs(v), vec4<f32>(2.0), 0.25);",
            |x, y| [x, y, x + y, x - y].map(|value| value.abs() * 0.75 + 0.5),
            0.0,
        ),
    ];
    let xs = [-2.5_f32, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5];
    let ys = [-1.5_f32, 0.0, 2.5];
    for &(name, body, reference, tolerance) in cases {
        let source = format!(
            "@fragment\n\
             fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{\n\
                 let x = p.x; let y = p.y;\n{body}\n}}"
        );
        let outputs = run_wgsl_native(name, &source, &xs, &ys);
        let expected = ys
            .iter()
            .flat_map(|&y| xs.iter().map(move |&x| reference(x, y)));
        assert_wgsl_outputs(name, &outputs, expected, tolerance);
    }
}

fn run_wgsl_native(name: &str, source: &str, xs: &[f32], ys: &[f32]) -> Vec<Vec<u8>> {
    let compiled = axshader::compile_wgsl_in(
        source,
        axshader::CompileOptions::for_viewport("main", xs.len() as u32, ys.len() as u32),
        Global,
    )
    .unwrap_or_else(|error| panic!("{name}: WGSL compilation failed: {error}"));
    let graph = compiled.graph().parse().unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime
        .compile_in(&graph, &model, Global)
        .unwrap_or_else(|error| panic!("{name}: native admission failed: {error:?}"));
    assert!(
        executable
            .schedule()
            .implementations()
            .iter()
            .all(|&implementation| implementation == CPU_KERNEL_IMPLEMENTATION),
        "{name}: the source corpus must execute native kernels, not a TOSA test provider"
    );
    let x_bytes = f32_bytes(xs.iter().copied());
    let y_bytes = f32_bytes(ys.iter().copied());
    runtime
        .run_in(&executable, &[&x_bytes, &y_bytes], Global)
        .unwrap_or_else(|error| panic!("{name}: native execution failed: {error:?}"))
}

fn assert_wgsl_outputs(
    name: &str,
    outputs: &[Vec<u8>],
    expected: impl IntoIterator<Item = [f32; 4]>,
    tolerance: f32,
) {
    let expected: Vec<_> = expected.into_iter().collect();
    assert_eq!(outputs.len(), 4, "{name}: RGBA output count");
    for (channel, output) in outputs.iter().enumerate() {
        assert_eq!(
            output.len(),
            expected.len() * 4,
            "{name}: channel {channel} size"
        );
        for (pixel, bytes) in output.as_chunks::<4>().0.iter().enumerate() {
            let actual = f32::from_le_bytes(*bytes);
            let expected = expected[pixel][channel];
            assert!(
                actual.is_finite()
                    && (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "{name}: pixel {pixel}, channel {channel}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn wgsl_atan_reduction_boundaries_obey_the_accuracy_contract() {
    let source = r#"
        @fragment
        fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(atan(p.x), atan2(p.x, p.y), atan2(p.y, p.x), 0.0);
        }
    "#;
    // The reciprocal reduction must not feed an atan polynomial intended for
    // small arguments with inputs near one. All atan2 inputs are finite and
    // normal, inside WGSL's bounded-accuracy domain.
    let xs = [-2.5_f32, -1.25, -1.01, -1.001, 1.001, 1.01, 1.25, 2.5];
    let ys = [0.75_f32, 1.0, 1.25];
    let outputs = run_wgsl_native("atan reduction boundaries", source, &xs, &ys);
    let expected: Vec<_> = ys
        .iter()
        .flat_map(|&y| xs.iter().map(move |&x| [x.atan(), x.atan2(y), y.atan2(x)]))
        .collect();
    // WGSL 15.7.4.1 permits 4096 ULP for f32 atan/atan2:
    // https://www.w3.org/TR/WGSL/#floating-point-accuracy
    assert_eq!(outputs.len(), 4);
    for (channel, output) in outputs[..3].iter().enumerate() {
        assert_eq!(output.len(), expected.len() * 4);
        for (pixel, bytes) in output.as_chunks::<4>().0.iter().enumerate() {
            let actual = f32::from_le_bytes(*bytes);
            let expected = expected[pixel][channel];
            assert!(
                actual.is_finite() && actual.to_bits().abs_diff(expected.to_bits()) <= 4096,
                "atan reduction: pixel {pixel}, channel {channel}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn wgsl_helpers_and_conditional_returns_execute_natively() {
    type Case = (&'static str, &'static str, fn(f32, f32) -> [f32; 4]);
    let cases: &[Case] = &[
        (
            "nested helpers and per-call argument isolation",
            r#"
                fn identity(value: f32) -> f32 { return value; }
                fn scale(value: f32) -> f32 { return identity(value) * 2.0 + 1.0; }
                fn swap(value: vec2<f32>) -> vec2<f32> { return value.yx; }
                @fragment
                fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                    let first = scale(p.x);
                    let second = scale(p.y);
                    let nested = scale(scale(p.x));
                    return vec4<f32>(swap(vec2<f32>(first, second)), nested,
                        identity(p.x - p.y));
                }
            "#,
            |x, y| [2.0 * y + 1.0, 2.0 * x + 1.0, 4.0 * x + 3.0, x - y],
        ),
        (
            "Boolean helpers and nested conditional shading",
            r#"
                fn outside(value: f32) -> bool {
                    if value < 0.0 { return true; }
                    return value > 1.0;
                }
                fn shade(value: f32) -> f32 {
                    if outside(value) {
                        if value < 0.0 { return 0.25; }
                        return 0.75;
                    } else {
                        return value;
                    }
                }
                @fragment
                fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                    return vec4<f32>(shade(p.x), shade(p.y),
                        select(p.x, p.y, outside(p.x) && !outside(p.y)), 1.0);
                }
            "#,
            |x, y| {
                let outside = |value: f32| !(0.0..=1.0).contains(&value);
                let shade = |value| {
                    if value < 0.0 {
                        0.25
                    } else if value > 1.0 {
                        0.75
                    } else {
                        value
                    }
                };
                [
                    shade(x),
                    shade(y),
                    if outside(x) && !outside(y) { y } else { x },
                    1.0,
                ]
            },
        ),
        (
            "early returns guard inactive square root and reciprocal domains",
            r#"
                fn inverse_length(value: f32) -> f32 {
                    if value <= 0.0 { return 0.0; }
                    return 1.0 / sqrt(value);
                }
                @fragment
                fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                    if p.x < 0.0 {
                        return vec4<f32>(0.0, inverse_length(p.y), 0.0, 1.0);
                    }
                    if p.y <= 0.0 {
                        return vec4<f32>(inverse_length(p.x), 0.0, 1.0, 1.0);
                    }
                    return vec4<f32>(inverse_length(p.x), inverse_length(p.y),
                        sqrt(p.x * p.y), 1.0);
                }
            "#,
            |x, y| {
                let inverse = |value: f32| {
                    if value <= 0.0 {
                        0.0
                    } else {
                        1.0 / value.sqrt()
                    }
                };
                if x < 0.0 {
                    [0.0, inverse(y), 0.0, 1.0]
                } else if y <= 0.0 {
                    [inverse(x), 0.0, 1.0, 1.0]
                } else {
                    [inverse(x), inverse(y), (x * y).sqrt(), 1.0]
                }
            },
        ),
    ];
    let xs = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let ys = [-1.0_f32, 0.0, 1.0];
    for &(name, source, reference) in cases {
        let outputs = run_wgsl_native(name, source, &xs, &ys);
        let expected = ys
            .iter()
            .flat_map(|&y| xs.iter().map(move |&x| reference(x, y)));
        assert_wgsl_outputs(name, &outputs, expected, 4.0e-6);
    }
}

#[test]
fn wgsl_uniform_and_field_broadcasts_refresh_between_native_runs() {
    // Declaration order deliberately differs from (group, binding) ABI order.
    // Even the unused uniform has a slot; fields remain full row-major planes.
    let source = r#"
        @group(1) @binding(3) var<uniform> gain: vec2<f32>;
        @group(0) @binding(4) var<uniform> bias: f32;
        @group(0) @binding(1) var<uniform> unused: f32;
        @fragment
        fn main(@builtin(position) p: vec4<f32>,
                @location(0) colour: vec2<f32>,
                @location(1) coverage: f32) -> @location(0) vec4<f32> {
            return vec4<f32>(colour * gain + vec2<f32>(bias), coverage * p.x, p.y);
        }
    "#;
    let xs = [-2.0_f32, -0.5, 1.0, 3.0, 5.0];
    let ys = [0.5_f32, 2.5];
    let pixels = xs.len() * ys.len();
    let red: Vec<_> = (0..pixels).map(|index| index as f32 * 0.25 - 1.0).collect();
    let green: Vec<_> = (0..pixels)
        .map(|index| (pixels - index) as f32 * 0.125)
        .collect();
    let coverage: Vec<_> = (0..pixels)
        .map(|index| if index % 2 == 0 { 0.5 } else { 1.5 })
        .collect();
    let compiled = axshader::compile_wgsl_in(
        source,
        axshader::CompileOptions::for_viewport("main", xs.len() as u32, ys.len() as u32),
        Global,
    )
    .unwrap();
    let graph = compiled.graph().parse().unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert!(
        executable
            .schedule()
            .implementations()
            .iter()
            .all(|&implementation| implementation == CPU_KERNEL_IMPLEMENTATION)
    );
    for [bias, gain_red, gain_green] in [[0.5, -0.75, 1.25], [-2.0, 2.0, -0.5]] {
        let inputs = [
            f32_bytes(xs),
            f32_bytes(ys),
            f32_bytes(red.iter().copied()),
            f32_bytes(green.iter().copied()),
            f32_bytes(coverage.iter().copied()),
            f32_bytes([-137.0]),
            f32_bytes([bias]),
            f32_bytes([gain_red]),
            f32_bytes([gain_green]),
        ];
        let inputs: Vec<_> = inputs.iter().map(Vec::as_slice).collect();
        let outputs = runtime.run_in(&executable, &inputs, Global).unwrap();
        let expected = (0..pixels).map(|pixel| {
            [
                red[pixel] * gain_red + bias,
                green[pixel] * gain_green + bias,
                coverage[pixel] * xs[pixel % xs.len()],
                ys[pixel / xs.len()],
            ]
        });
        assert_wgsl_outputs("uniform/field broadcast", &outputs, expected, 0.0);
    }
}

#[test]
fn wgsl_scalar_length_and_distance_execute_with_scalar_overloads() {
    let source = r#"
        @fragment
        fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
            let magnitude = length(position.x - 2.0);
            let separation = distance(position.y, 1.0);
            return vec4<f32>(magnitude, separation, 0.0, 1.0);
        }
    "#;
    let compiled = axshader::compile_wgsl_in(
        source,
        axshader::CompileOptions::for_viewport("main", 3, 2),
        Global,
    )
    .unwrap();
    let graph = compiled.graph().parse().unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();

    let x = f32_bytes([0.5, 2.0, 4.0]);
    let y = f32_bytes([0.0, 4.0]);
    let outputs = runtime.run_in(&executable, &[&x, &y], Global).unwrap();
    let decode = |channel: &[u8]| {
        channel
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>()
    };
    assert_eq!(decode(&outputs[0]), [1.5, 0.0, 2.0, 1.5, 0.0, 2.0]);
    assert_eq!(decode(&outputs[1]), [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]);
    assert_eq!(decode(&outputs[2]), [0.0; 6]);
    assert_eq!(decode(&outputs[3]), [1.0; 6]);
}

#[test]
fn wgsl_identity_and_boolean_graph_is_one_pure_native_partition() {
    let source = r#"
        @fragment
        fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
            let right = position.x > 1.0;
            let lower = position.y != 0.5;
            let left = position.x < 1.0;
            let lit = (right && !lower) || (left && lower);
            let shade = select(0.25, 0.75, lit);
            return vec4<f32>(shade, shade, shade, 1.0);
        }
    "#;
    let compiled = axshader::compile_wgsl_in(
        source,
        axshader::CompileOptions::for_viewport("main", 3, 2),
        Global,
    )
    .unwrap();
    let graph = compiled.graph().parse().unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert!(
        executable
            .schedule()
            .implementations()
            .iter()
            .all(|&implementation| implementation == CPU_KERNEL_IMPLEMENTATION)
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    let x = f32_bytes([0.5, 1.5, 2.5]);
    let y = f32_bytes([0.5, 1.5]);
    let outputs = runtime.run_in(&executable, &[&x, &y], Global).unwrap();
    assert_eq!(outputs.len(), 4);
    let expected = [0.25_f32, 0.75, 0.75, 0.75, 0.25, 0.25];
    for channel in &outputs[..3] {
        let actual: Vec<f32> = channel
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(actual, expected);
    }
    let alpha: Vec<f32> = outputs[3]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(alpha, [1.0; 6]);
}

#[test]
fn wgsl_fragment_record_reparsed_outputs_preserve_order_aliases_and_integer_literals() {
    use axnn_cpu::CpuCapabilities;
    use axnn_vaccel::AxCpuKernelPolicy;

    let source = r#"
        struct Result {
            @location(3) pair: vec2<f32>,
            @location(1) @interpolate(flat) large: i32,
            @location(0) coordinates: vec2<f32>,
            @location(2) @interpolate(flat) lower: i32,
            @location(4) @interpolate(flat) upper: i32,
            @location(5) scalar: f32,
        }
        @group(0) @binding(0) var<uniform> gain: f32;
        const LARGE: i32 = 16777217i;
        @fragment fn main(@builtin(position) p: vec4<f32>) -> Result {
            let sum_value = p.x + p.y + gain;
            let first = Result(vec2<f32>(sum_value, gain), LARGE, p.xy,
                -2147483648, 2147483647i, -0.5);
            let shared_result = first;
            let other = Result(vec2<f32>(99.0), 16777218i, vec2<f32>(99.0),
                1i, 2i, 99.0);
            let returned = shared_result;
            return returned;
        }
    "#;
    let serialized = {
        let compiled = axshader::compile_wgsl_in(
            source,
            axshader::CompileOptions::for_viewport("main", 3, 2),
            Global,
        )
        .unwrap();
        compiled.graph().as_bytes().to_vec()
    };
    let encoded = GraphBuf::copy_from_in(&serialized, Global).unwrap();
    let graph = encoded.parse().unwrap();
    let x = f32_bytes([0.5, 1.5, 2.5]);
    let y = f32_bytes([-1.0, 4.0]);
    let gain = f32_bytes([2.0]);
    let inputs: [&[u8]; 3] = [&x, &y, &gain];
    let integer_plane = |value: i32| (0..6).flat_map(|_| value.to_le_bytes()).collect::<Vec<_>>();
    let expected = [
        f32_bytes([0.5, 1.5, 2.5, 0.5, 1.5, 2.5]),
        f32_bytes([-1.0, -1.0, -1.0, 4.0, 4.0, 4.0]),
        integer_plane(16_777_217),
        integer_plane(i32::MIN),
        f32_bytes([1.5, 2.5, 3.5, 6.5, 7.5, 8.5]),
        f32_bytes([2.0; 6]),
        integer_plane(i32::MAX),
        f32_bytes([-0.5; 6]),
    ];
    for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
        let model = AxDeviceModel {
            accelerator: false,
            cpu_tosa: false,
            sparse_bsr: None,
            cpu_kernels: Some(AxCpuKernelPolicy {
                capabilities,
                ..Default::default()
            }),
            ..AxDeviceModel::default()
        };
        let mut runtime = sessions();
        let executable = runtime.compile_in(&graph, &model, Global).unwrap();
        let outputs = runtime.run_in(&executable, &inputs, Global).unwrap();
        assert_eq!(outputs.len(), expected.len());
        let mut prepared = executable.prepare_native_in(Global).unwrap();
        prepared.run(&inputs).unwrap();
        for (index, expected) in expected.iter().enumerate() {
            assert_eq!(
                outputs[index].as_slice(),
                expected,
                "{capabilities:?}: output {index}"
            );
            if matches!(index, 2 | 3 | 6) {
                let values = expected
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| i32::from_le_bytes(*bytes))
                    .collect::<Vec<_>>();
                assert_eq!(prepared.output_i32(index).unwrap(), values);
            } else {
                let values = expected
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes))
                    .collect::<Vec<_>>();
                assert_eq!(prepared.output_f32(index).unwrap(), values);
            }
        }
    }
}

#[test]
fn wgsl_fragment_record_returns_exact_hit_and_shared_direction_after_resource_edits() {
    use ax::voxel::{ResourceKey, VoxelBrickTraceParams};
    use axnn_cpu::{BrickGridRef, CpuCapabilities};
    use axnn_vaccel::{AxCpuKernelPolicy, NativeResourceBinding};

    const KEY: ResourceKey = ResourceKey {
        group: 2,
        binding: 7,
    };
    const DIMS: [u32; 3] = [4, 4, 1_048_580];
    let source = r#"
        enable ax_voxel_query;
        struct Result {
            @location(2) direction: vec3<f32>,
            @location(1) @interpolate(flat) id: i32,
            @location(0) distance: f32,
        }
        @group(2) @binding(7) var<storage, read> occupancy: array<vec2<u32>>;
        fn direction(x: f32) -> vec3<f32> {
            return vec3<f32>(1.0, x * 0.125, 0.0);
        }
        @fragment fn main(@builtin(position) p: vec4<f32>) -> Result {
            let ray = direction(p.x);
            let hit = ax_brick_trace(&occupancy, vec3<f32>(0.25, 0.25, 1048576.5),
                ray, 16.0);
            let result = Result(ray, hit.cell_id, hit.t);
            let shared_result = result;
            return shared_result;
        }
    "#;
    let serialized = {
        let compiled = axshader::compile_wgsl_with_resources_in(
            source,
            axshader::CompileOptions::for_viewport("main", 3, 2),
            &[VoxelBrickTraceParams::new(KEY, DIMS).unwrap()],
            Global,
        )
        .unwrap();
        compiled.graph().as_bytes().to_vec()
    };
    let encoded = GraphBuf::copy_from_in(&serialized, Global).unwrap();
    let graph = encoded.parse().unwrap();
    let queries: Vec<_> = (0..graph.node_count())
        .map(|index| graph.node(index).unwrap())
        .filter(|node| node.op == OpCode::VOXEL_BRICK_TRACE)
        .collect();
    assert_eq!(
        queries.len(),
        1,
        "hit member access must not duplicate traversal"
    );
    let provider_model = AxDeviceModel::default();
    for backend in [CPU_BACKEND, ACCEL_BACKEND] {
        assert!(!provider_model.supports_implementation(
            backend,
            &graph,
            queries[0],
            DENSE_IMPLEMENTATION,
        ));
    }
    let x = f32_bytes([0.0, 1.0, 2.0]);
    let y = f32_bytes([0.5, 1.5]);
    let inputs: [&[u8]; 2] = [&x, &y];
    let mut bricks = vec![0_u64; DIMS[2] as usize / 4];
    let brick = 1_048_576 / 4;
    for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
        let model = AxDeviceModel {
            accelerator: false,
            cpu_tosa: false,
            sparse_bsr: None,
            cpu_kernels: Some(AxCpuKernelPolicy {
                capabilities,
                ..Default::default()
            }),
            ..AxDeviceModel::default()
        };
        let mut runtime = sessions();
        let executable = runtime.compile_in(&graph, &model, Global).unwrap();
        let mut prepared = executable.prepare_native_in(Global).unwrap();
        for (word, distance, id) in [
            (1 << 1, 0.75_f32, 16_777_217_i32),
            (1 << 2, 1.75, 16_777_218),
            (0, f32::INFINITY, -1),
            (1 << 1, 0.75, 16_777_217),
        ] {
            bricks[brick] = word;
            let resources = [NativeResourceBinding::brick_grid(
                KEY,
                BrickGridRef::new(&bricks, DIMS).unwrap(),
            )];
            let expected = [
                f32_bytes([distance; 6]),
                (0..6).flat_map(|_| id.to_le_bytes()).collect::<Vec<_>>(),
                f32_bytes([1.0; 6]),
                f32_bytes([0.0, 0.125, 0.25, 0.0, 0.125, 0.25]),
                f32_bytes([0.0; 6]),
            ];
            let outputs = runtime
                .run_with_resources_in(&executable, &inputs, &resources, Global)
                .unwrap();
            assert_eq!(outputs.len(), expected.len());
            prepared.run_with_resources(&inputs, &resources).unwrap();
            for (index, expected) in expected.iter().enumerate() {
                assert_eq!(
                    outputs[index].as_slice(),
                    expected,
                    "{capabilities:?}: output {index}"
                );
                if index == 1 {
                    assert_eq!(prepared.output_i32(index).unwrap(), [id; 6]);
                } else {
                    let values = expected
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|bytes| f32::from_le_bytes(*bytes))
                        .collect::<Vec<_>>();
                    assert_eq!(prepared.output_f32(index).unwrap(), values);
                }
            }
        }
    }
}

#[test]
fn wgsl_voxel_reparsed_asset_preserves_large_ids_and_live_resource_edits() {
    use ax::voxel::{ResourceKey, VoxelBrickTraceParams};
    use axnn_cpu::{BrickGridRef, CpuCapabilities};
    use axnn_vaccel::{AxCpuKernelPolicy, NativeResourceBinding};

    const KEY: ResourceKey = ResourceKey {
        group: 2,
        binding: 7,
    };
    const DIMS: [u32; 3] = [4, 4, 1_048_580];
    let source = r#"
        enable ax_voxel_query;
        @group(2) @binding(7) var<storage, read> occupancy: array<vec2<u32>>;
        @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
            let hit = ax_brick_trace(&occupancy, vec3<f32>(0.25, 0.25, 1048576.5),
                vec3<f32>(1.0, 0.0, 0.0), 16.0);
            return vec4<f32>(
                select(0.0, 1.0, hit.cell_id == 16777217i),
                select(0.0, 1.0, hit.cell_id == 16777218i),
                select(0.0, 1.0, hit.cell_id < 0i),
                select(hit.t, 0.0, hit.cell_id <= 16777216i));
        }
    "#;
    // Drop the compiler and its interface before admission: the asset must
    // carry all resource/type metadata. The six-ray tile also has a packet tail.
    let serialized = {
        let compiled = axshader::compile_wgsl_with_resources_in(
            source,
            axshader::CompileOptions::for_viewport("main", 3, 2),
            &[VoxelBrickTraceParams::new(KEY, DIMS).unwrap()],
            Global,
        )
        .unwrap();
        compiled.graph().as_bytes().to_vec()
    };
    let encoded = GraphBuf::copy_from_in(&serialized, Global).unwrap();
    let graph = encoded.parse().unwrap();
    let query = (0..graph.node_count())
        .map(|index| graph.node(index).unwrap())
        .find(|node| node.op == OpCode::VOXEL_BRICK_TRACE)
        .unwrap();
    let provider_model = AxDeviceModel::default();
    for backend in [CPU_BACKEND, ACCEL_BACKEND] {
        assert!(
            !provider_model.supports_implementation(backend, &graph, query, DENSE_IMPLEMENTATION),
            "a borrowed voxel query must not be placed on a TOSA provider"
        );
    }

    let x = f32_bytes([0.5, 1.5, 2.5]);
    let y = f32_bytes([0.5, 1.5]);
    let inputs: [&[u8]; 2] = [&x, &y];
    let mut bricks = vec![0_u64; DIMS[2] as usize / 4];
    let brick = 1_048_576 / 4;
    for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
        let model = AxDeviceModel {
            accelerator: false,
            cpu_tosa: false,
            sparse_bsr: None,
            cpu_kernels: Some(AxCpuKernelPolicy {
                capabilities,
                ..Default::default()
            }),
            ..AxDeviceModel::default()
        };
        let mut runtime = sessions();
        let executable = runtime.compile_in(&graph, &model, Global).unwrap();
        assert!(
            executable
                .schedule()
                .implementations()
                .iter()
                .all(|&implementation| implementation == CPU_KERNEL_IMPLEMENTATION)
        );
        let mut prepared = executable.prepare_native_in(Global).unwrap();
        // Alpha's comparison with 2^24 catches a lossy I32->F32 path even if
        // both equality literals were rounded along with the queried ID.
        for (word, rgba) in [
            (1 << 1, [1.0, 0.0, 0.0, 0.75]),
            (1 << 2, [0.0, 1.0, 0.0, 1.75]),
            (0, [0.0, 0.0, 1.0, 0.0]),
            (1 << 1, [1.0, 0.0, 0.0, 0.75]),
        ] {
            bricks[brick] = word;
            let resources = [NativeResourceBinding::brick_grid(
                KEY,
                BrickGridRef::new(&bricks, DIMS).unwrap(),
            )];
            let outputs = runtime
                .run_with_resources_in(&executable, &inputs, &resources, Global)
                .unwrap();
            assert_wgsl_outputs("borrowed WGSL voxel query", &outputs, [rgba; 6], 0.0);
            prepared.run_with_resources(&inputs, &resources).unwrap();
            for (channel, value) in rgba.into_iter().enumerate() {
                assert_eq!(
                    prepared.output_f32(channel).unwrap(),
                    [value; 6],
                    "{capabilities:?}: edited word {word}, channel {channel}"
                );
            }
        }
    }
}

// Independent arithmetic decoding: the runtime uses the packed-bit UF16
// promotion kernel, whereas the reference evaluates the E5M11 value formula.
fn rgb_uf16_reference(word: u16) -> f32 {
    assert!(word <= 0xf7ff);
    let exponent = word >> 11;
    let fraction = word & 0x7ff;
    if exponent == 0 {
        f32::from(fraction) * 2.0_f32.powi(-25)
    } else {
        f32::from(2048 + fraction) * 2.0_f32.powi(i32::from(exponent) - 26)
    }
}

fn rgb_sample_reference(rgb: &[[f32; 3]], dims: [u32; 3], point: [f32; 3]) -> [f32; 3] {
    if point.iter().any(|value| value.is_nan()) {
        return [f32::NAN; 3];
    }
    let q: [f32; 3] = core::array::from_fn(|axis| point[axis].clamp(0.0, (dims[axis] - 1) as f32));
    let lo = q.map(|value| value.floor() as usize);
    let hi: [usize; 3] = core::array::from_fn(|axis| (lo[axis] + 1).min(dims[axis] as usize - 1));
    let fraction: [f32; 3] = core::array::from_fn(|axis| q[axis] - lo[axis] as f32);
    let mut result = [0.0_f32; 3];
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let upper = [x, y, z];
                let index: [usize; 3] =
                    core::array::from_fn(|axis| if upper[axis] == 0 { lo[axis] } else { hi[axis] });
                let weight: [f32; 3] = core::array::from_fn(|axis| {
                    if upper[axis] == 0 {
                        1.0 - fraction[axis]
                    } else {
                        fraction[axis]
                    }
                });
                let weight = (weight[0] * weight[1]) * weight[2];
                let value =
                    rgb[index[0] + dims[0] as usize * (index[1] + dims[1] as usize * index[2])];
                for channel in 0..3 {
                    result[channel] = result[channel] + value[channel] * weight;
                }
            }
        }
    }
    result
}

fn assert_rgb_plane(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (lane, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "lane {lane}: {actual} must be NaN");
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "lane {lane}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn wgsl_rgb_volume_reparse_preserves_storage_broadcast_edges_and_live_borrows() {
    use ax::sampling::{ResourceKey, VolumeRgbParams, VolumeRgbStorage};
    use axnn_cpu::{CpuCapabilities, RgbVolumeRef};
    use axnn_vaccel::{AxCpuKernelPolicy, NativeResourceBinding};
    use axshader::{CompileOptions, ShaderResource, compile_wgsl_with_resource_specs_in};

    const KEY: ResourceKey = ResourceKey {
        group: 1,
        binding: 4,
    };
    const SOURCE: &str = r#"
        enable ax_volume_sampling;
        @group(1) @binding(4) var<storage, read> light: AxRgbVolume;
        @group(0) @binding(2) var<uniform> depth: f32;
        struct Result { @location(0) rgb: vec3<f32>, }
        @fragment fn main(@builtin(position) p: vec4<f32>) -> Result {
            let rgb = ax_sample_volume_rgb(&light, vec3<f32>(p.x, p.y, depth));
            return Result(rgb);
        }
    "#;
    let x_values = [f32::NEG_INFINITY, 0.375, 1.5, f32::INFINITY, f32::NAN];
    let y_values = [0.25, 2.5];
    let x = f32_bytes(x_values);
    let y = f32_bytes(y_values);
    for dims in [[3, 2, 2], [1, 2, 1]] {
        for storage in [
            VolumeRgbStorage::F32,
            VolumeRgbStorage::Uf16E5M11,
            VolumeRgbStorage::Uf8E5M3,
        ] {
            let params = VolumeRgbParams::new(KEY, dims, storage).unwrap();
            let serialized = compile_wgsl_with_resource_specs_in(
                SOURCE,
                CompileOptions::for_viewport("main", 5, 2),
                &[ShaderResource::RgbVolume(params)],
                Global,
            )
            .unwrap()
            .graph()
            .as_bytes()
            .to_vec();
            let encoded = GraphBuf::copy_from_in(&serialized, Global).unwrap();
            let graph = encoded.parse().unwrap();
            let samples: Vec<_> = (0..graph.node_count())
                .map(|index| graph.node(index).unwrap())
                .filter(|node| node.op == OpCode::VOLUME_TRILINEAR_RGB)
                .collect();
            assert_eq!(samples.len(), 1);
            assert_eq!(samples[0].output_count(), 3);
            assert_eq!(
                ax::sampling::decode_volume_rgb_attributes(samples[0].attributes()),
                Some(params)
            );
            for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
                let mut runtime = sessions();
                let executable = runtime
                    .compile_in(
                        &graph,
                        &AxDeviceModel {
                            accelerator: false,
                            cpu_tosa: false,
                            sparse_bsr: None,
                            cpu_kernels: Some(AxCpuKernelPolicy {
                                capabilities,
                                ..Default::default()
                            }),
                            ..Default::default()
                        },
                        Global,
                    )
                    .unwrap();
                let mut prepared = executable.prepare_native_in(Global).unwrap();
                for generation in 0..2 {
                    let count = params.voxel_count().unwrap();
                    let rgb: Vec<[f32; 3]> = (0..count)
                        .map(|index| {
                            core::array::from_fn(|channel| {
                                (index * 19 + channel * 7 + generation * 11 + 1) as f32 / 16.0
                            })
                        })
                        .collect();
                    let pool = [0_u16, 1, 0x07ff, 0x0800, 0x7401, 0x7801, 0xf7ff];
                    let words: Vec<[u16; 3]> = (0..count)
                        .map(|index| {
                            core::array::from_fn(|channel| {
                                pool[(index * 3 + channel + generation) % pool.len()]
                            })
                        })
                        .collect();
                    let decoded: Vec<[f32; 3]> = words
                        .iter()
                        .map(|rgb| rgb.map(rgb_uf16_reference))
                        .collect();
                    let bytes: Vec<[u8; 3]> = rgb
                        .iter()
                        .map(|v| v.map(axnn_cpu::uf::f32_to_uf8e5m3_rne))
                        .collect();
                    let decoded8: Vec<[f32; 3]> = bytes
                        .iter()
                        .map(|v| v.map(axnn_cpu::uf::uf8e5m3_to_f32))
                        .collect();
                    let (view, reference) = match storage {
                        VolumeRgbStorage::F32 => (RgbVolumeRef::new_f32(&rgb, dims).unwrap(), &rgb),
                        VolumeRgbStorage::Uf16E5M11 => {
                            (RgbVolumeRef::new_uf16(&words, dims).unwrap(), &decoded)
                        }
                        VolumeRgbStorage::Uf8E5M3 => {
                            (RgbVolumeRef::new_uf8e5m3(&bytes, dims).unwrap(), &decoded8)
                        }
                    };
                    let resources = [NativeResourceBinding::rgb_volume(KEY, view)];
                    for depth in [0.625_f32, f32::NAN] {
                        let z = f32_bytes([depth]);
                        let inputs: [&[u8]; 3] = [&x, &y, &z];
                        let expected: Vec<[f32; 3]> = y_values
                            .iter()
                            .flat_map(|&y| {
                                x_values.iter().map(move |&x| {
                                    rgb_sample_reference(reference, dims, [x, y, depth])
                                })
                            })
                            .collect();
                        let outputs = runtime
                            .run_with_resources_in(&executable, &inputs, &resources, Global)
                            .unwrap();
                        prepared.run_with_resources(&inputs, &resources).unwrap();
                        assert_eq!(outputs.len(), 3);
                        for channel in 0..3 {
                            let expected: Vec<f32> =
                                expected.iter().map(|rgb| rgb[channel]).collect();
                            let actual: Vec<f32> = outputs[channel]
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .map(|bytes| f32::from_le_bytes(*bytes))
                                .collect();
                            assert_rgb_plane(&actual, &expected);
                            assert_rgb_plane(prepared.output_f32(channel).unwrap(), &expected);
                        }
                    }
                }
                assert_eq!(
                    runtime.cpu.provider().buffer_write_bytes(),
                    0,
                    "native sampling must not upload the volume"
                );
            }
        }
    }
}

#[test]
fn wgsl_voxel_hit_feeds_rgb_volume_without_losing_integer_ids() {
    use ax::sampling::{ResourceKey, VolumeRgbParams, VolumeRgbStorage};
    use ax::voxel::VoxelBrickTraceParams;
    use axnn_cpu::{BrickGridRef, CpuCapabilities, RgbVolumeRef};
    use axnn_vaccel::{AxCpuKernelPolicy, NativeResourceBinding};
    use axshader::{CompileOptions, ShaderResource, compile_wgsl_with_resource_specs_in};

    const GRID: ResourceKey = ResourceKey {
        group: 2,
        binding: 7,
    };
    const LIGHT: ResourceKey = ResourceKey {
        group: 1,
        binding: 4,
    };
    const GRID_DIMS: [u32; 3] = [4, 4, 1_048_580];
    const LIGHT_DIMS: [u32; 3] = [3, 2, 2];
    let source = r#"
        enable ax_voxel_query;
        enable ax_volume_sampling;
        @group(2) @binding(7) var<storage, read> occupancy: array<vec2<u32>>;
        @group(1) @binding(4) var<storage, read> light: AxRgbVolume;
        struct Result {
            @location(0) rgb: vec3<f32>,
            @location(1) @interpolate(flat) cell_id: i32,
            @location(2) distance: f32,
        }
        @fragment fn main(@builtin(position) p: vec4<f32>) -> Result {
            let hit = ax_brick_trace(&occupancy, vec3<f32>(0.25, 0.25, 1048576.5), vec3<f32>(1.0, p.x * 0.125, 0.0), 16.0);
            let rgb = ax_sample_volume_rgb(&light, vec3<f32>(hit.t, p.y, 0.5));
            return Result(rgb, hit.cell_id, hit.t);
        }
    "#;
    let resource_specs = [
        ShaderResource::RgbVolume(
            VolumeRgbParams::new(LIGHT, LIGHT_DIMS, VolumeRgbStorage::Uf16E5M11).unwrap(),
        ),
        ShaderResource::VoxelBrickGrid(VoxelBrickTraceParams::new(GRID, GRID_DIMS).unwrap()),
    ];
    let serialized = compile_wgsl_with_resource_specs_in(
        source,
        CompileOptions::for_viewport("main", 3, 2),
        &resource_specs,
        Global,
    )
    .unwrap()
    .graph()
    .as_bytes()
    .to_vec();
    let encoded = GraphBuf::copy_from_in(&serialized, Global).unwrap();
    let graph = encoded.parse().unwrap();
    for op in [OpCode::VOXEL_BRICK_TRACE, OpCode::VOLUME_TRILINEAR_RGB] {
        assert_eq!(
            (0..graph.node_count())
                .filter(|&index| graph.node(index).unwrap().op == op)
                .count(),
            1
        );
    }
    let x = f32_bytes([0.0, 1.0, 2.0]);
    let y_values = [-1.0, 0.625];
    let y = f32_bytes(y_values);
    let inputs: [&[u8]; 2] = [&x, &y];
    let mut bricks = vec![0_u64; GRID_DIMS[2] as usize / 4];
    for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
        let mut runtime = sessions();
        let executable = runtime
            .compile_in(
                &graph,
                &AxDeviceModel {
                    accelerator: false,
                    cpu_tosa: false,
                    sparse_bsr: None,
                    cpu_kernels: Some(AxCpuKernelPolicy {
                        capabilities,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                Global,
            )
            .unwrap();
        let mut prepared = executable.prepare_native_in(Global).unwrap();
        for (generation, (word, distance, id)) in [
            (1_u64 << 1, 0.75_f32, 16_777_217_i32),
            (1 << 2, 1.75, 16_777_218),
            (0, f32::INFINITY, -1),
        ]
        .into_iter()
        .enumerate()
        {
            bricks[1_048_576 / 4] = word;
            let words: Vec<[u16; 3]> = (0..12)
                .map(|index| {
                    core::array::from_fn(|channel| {
                        0x7401 + ((index * 3 + channel + generation * 17) * 11) as u16
                    })
                })
                .collect();
            let decoded: Vec<[f32; 3]> = words
                .iter()
                .map(|rgb| rgb.map(rgb_uf16_reference))
                .collect();
            let resources = [
                NativeResourceBinding::brick_grid(
                    GRID,
                    BrickGridRef::new(&bricks, GRID_DIMS).unwrap(),
                ),
                NativeResourceBinding::rgb_volume(
                    LIGHT,
                    RgbVolumeRef::new_uf16(&words, LIGHT_DIMS).unwrap(),
                ),
            ];
            let outputs = runtime
                .run_with_resources_in(&executable, &inputs, &resources, Global)
                .unwrap();
            prepared.run_with_resources(&inputs, &resources).unwrap();
            let expected: Vec<[f32; 3]> = y_values
                .iter()
                .flat_map(|&y| [rgb_sample_reference(&decoded, LIGHT_DIMS, [distance, y, 0.5]); 3])
                .collect();
            assert_eq!(outputs.len(), 5);
            for channel in 0..3 {
                let plane: Vec<f32> = expected.iter().map(|rgb| rgb[channel]).collect();
                assert_eq!(outputs[channel], f32_bytes(plane.iter().copied()));
                assert_rgb_plane(prepared.output_f32(channel).unwrap(), &plane);
            }
            assert_eq!(
                outputs[3],
                (0..6).flat_map(|_| id.to_le_bytes()).collect::<Vec<_>>()
            );
            assert_eq!(prepared.output_i32(3).unwrap(), [id; 6]);
            assert_eq!(outputs[4], f32_bytes([distance; 6]));
            assert_eq!(prepared.output_f32(4).unwrap(), [distance; 6]);
        }
    }
}

#[test]
fn wgsl_rgb_volume_rejects_unregistered_or_incompatible_resource_schemas() {
    use ax::sampling::{ResourceKey, VolumeRgbParams, VolumeRgbStorage};
    use axshader::{CompileOptions, ShaderResource, compile_wgsl_with_resource_specs_in};

    let params = VolumeRgbParams::new(
        ResourceKey {
            group: 1,
            binding: 4,
        },
        [2, 2, 2],
        VolumeRgbStorage::Uf16E5M11,
    )
    .unwrap();
    for declaration in [
        "@group(1) @binding(4) var<storage, read_write> light: AxRgbVolume;",
        "@group(1) @binding(4) var<storage, read> light: array<vec3<f32>>;",
        "@group(1) @binding(4) var<storage, read> light: array<vec2<u32>>;",
    ] {
        let source = format!(
            "enable ax_volume_sampling;\n{declaration}\n@fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{ return vec4<f32>(ax_sample_volume_rgb(&light, p.xyz), 1.0); }}"
        );
        assert!(
            compile_wgsl_with_resource_specs_in(
                &source,
                CompileOptions::for_viewport("main", 2, 2),
                &[ShaderResource::RgbVolume(params)],
                Global
            )
            .is_err()
        );
    }
    let source = "enable ax_volume_sampling;\n@group(1) @binding(4) var<storage, read> light: AxRgbVolume;\n@fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> { return vec4<f32>(ax_sample_volume_rgb(&light, p.xyz), 1.0); }";
    compile_wgsl_with_resource_specs_in(
        source,
        CompileOptions::for_viewport("main", 2, 2),
        &[ShaderResource::RgbVolume(params)],
        Global,
    )
    .unwrap();
    for resources in [
        vec![],
        vec![ShaderResource::RgbVolume(VolumeRgbParams {
            dims: [0; 3],
            ..params
        })],
        vec![ShaderResource::RgbVolume(params); 2],
    ] {
        assert!(
            compile_wgsl_with_resource_specs_in(
                source,
                CompileOptions::for_viewport("main", 2, 2),
                &resources,
                Global
            )
            .is_err()
        );
    }
    assert!(
        axshader::compile_wgsl_in(source, CompileOptions::for_viewport("main", 2, 2), Global)
            .is_err()
    );
}

#[test]
fn byte_resource_preflight_precedes_provider_side_effects_and_async_partitions() {
    use ax::buffer::{U8BufferParams, encode_u8_buffer_attributes};
    use ax::voxel::ResourceKey;
    use axnn_cpu::BrickGridRef;
    use axnn_vaccel::{NativeResourceBinding, NativeResourceKind, U8BufferRef};

    const KEY: ResourceKey = ResourceKey {
        group: 3,
        binding: 9,
    };
    fn cast(builder: &mut GraphBuilder<Global>, input: TensorId, dtype: DType) -> TensorId {
        let output = builder
            .tensor(dtype, TensorFlags::default(), &[1, 7], &[])
            .unwrap();
        builder
            .node(OpCode::CAST, NodeFlags::default(), &[input], &[output], &[])
            .unwrap();
        output
    }
    fn load_output(
        builder: &mut GraphBuilder<Global>,
        index: TensorId,
        params: U8BufferParams,
    ) -> TensorId {
        let output = builder
            .tensor(DType::I32, TensorFlags::OUTPUT, &[1, 7], &[])
            .unwrap();
        builder
            .node(
                OpCode::BUFFER_LOAD_U8,
                NodeFlags::READS_RESOURCE,
                &[index],
                &[output],
                &encode_u8_buffer_attributes(params).unwrap(),
            )
            .unwrap();
        output
    }
    let params = U8BufferParams::new(KEY, 256).unwrap();
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[1, 7], &[])
        .unwrap();
    let float_index = cast(&mut builder, input, DType::F32);
    let index = cast(&mut builder, float_index, DType::I32);
    let first = load_output(&mut builder, index, params);
    let widened = cast(&mut builder, first, DType::F32);
    let half = cast(&mut builder, widened, DType::F16);
    let roundtrip = cast(&mut builder, half, DType::F32);
    let second_index = cast(&mut builder, roundtrip, DType::I32);
    let second = load_output(&mut builder, second_index, params);
    let sum = builder
        .tensor(DType::I32, TensorFlags::OUTPUT, &[1, 7], &[])
        .unwrap();
    // The first byte result is an output and a later native operand across
    // both provider partitions; it cannot be reclaimed at the first CAST.
    builder
        .node(
            OpCode::ADD,
            NodeFlags::default(),
            &[first, second],
            &[sum],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let cpu = Session::new(DelayedAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<DelayedAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let executable = runtime
        .compile_in(
            &graph,
            &AxDeviceModel {
                accelerator: false,
                sparse_bsr: None,
                ..Default::default()
            },
            Global,
        )
        .unwrap();
    assert_eq!(
        executable
            .schedule()
            .plan()
            .partitions
            .iter()
            .map(|partition| partition.implementation)
            .collect::<Vec<_>>(),
        [
            DENSE_IMPLEMENTATION,
            CPU_KERNEL_IMPLEMENTATION,
            DENSE_IMPLEMENTATION,
            CPU_KERNEL_IMPLEMENTATION
        ]
    );
    let coordinates = [-1.0, 0.0, 1.0, 127.0, 128.0, 255.0, 256.0];
    let input = f16_bytes(coordinates);
    let inputs: [&[u8]; 1] = [&input];
    let bytes: [u8; 256] = core::array::from_fn(|i| 255 - i as u8);
    let resources = [NativeResourceBinding::u8_buffer(
        KEY,
        U8BufferRef::new(&bytes).unwrap(),
    )];
    let brick = [0_u64];
    let long = [0_u8; 257];
    let wrong_kind = [NativeResourceBinding::brick_grid(
        KEY,
        BrickGridRef::new(&brick, [4; 3]).unwrap(),
    )];
    let short = [NativeResourceBinding::u8_buffer(
        KEY,
        U8BufferRef::new(&bytes[..255]).unwrap(),
    )];
    let long = [NativeResourceBinding::u8_buffer(
        KEY,
        U8BufferRef::new(&long).unwrap(),
    )];
    let duplicate = [resources[0], resources[0]];
    let before = (
        runtime.cpu.provider().inner.buffer_allocations(),
        runtime.cpu.provider().inner.buffer_write_bytes(),
        runtime.cpu.provider().inner.buffer_read_bytes(),
    );
    let counts = Arc::new(AllocationCounts::default());
    for (case, bindings) in [
        (0, &[][..]),
        (1, &wrong_kind[..]),
        (2, &short[..]),
        (3, &long[..]),
        (4, &duplicate[..]),
    ] {
        for asynchronous in [false, true] {
            let allocator = TrackingAllocator(Arc::clone(&counts));
            let error = if asynchronous {
                match runtime.begin_with_resources_in(&executable, &inputs, bindings, allocator) {
                    Err(error) => error,
                    Ok(_) => panic!("invalid byte resource admitted before async execution"),
                }
            } else {
                runtime
                    .run_with_resources_in(&executable, &inputs, bindings, allocator)
                    .unwrap_err()
            };
            assert!(match (case, error) {
                (0, AxCompileError::MissingResource(key))
                | (4, AxCompileError::DuplicateResource(key)) => key == KEY,
                (
                    1,
                    AxCompileError::ResourceKind {
                        resource,
                        expected: NativeResourceKind::U8Buffer,
                        actual: NativeResourceKind::BrickGrid,
                    },
                ) => resource == KEY,
                (
                    2,
                    AxCompileError::ResourceLength {
                        resource,
                        expected: 256,
                        actual: 255,
                    },
                )
                | (
                    3,
                    AxCompileError::ResourceLength {
                        resource,
                        expected: 256,
                        actual: 257,
                    },
                ) => resource == KEY,
                (_, error) => panic!("unexpected resource-preflight failure {error:?}"),
            });
        }
    }
    assert_eq!(counts.total.load(Ordering::Relaxed), 0);
    assert_eq!(counts.live.load(Ordering::Relaxed), 0);
    assert_eq!(runtime.cpu.pending_count(), 0);
    assert_eq!(
        (
            runtime.cpu.provider().inner.buffer_allocations(),
            runtime.cpu.provider().inner.buffer_write_bytes(),
            runtime.cpu.provider().inner.buffer_read_bytes()
        ),
        before
    );
    let first = coordinates.map(|i| {
        usize::try_from(i as i32)
            .ok()
            .and_then(|index| bytes.get(index))
            .copied()
            .unwrap_or(0) as i32
    });
    let second = first.map(|index| i32::from(bytes[index as usize]));
    let expected = [
        i32_bytes(first),
        i32_bytes(second),
        i32_bytes(first.into_iter().zip(second).map(|(a, b)| a + b)),
    ];
    assert_eq!(
        runtime
            .run_with_resources_in(&executable, &inputs, &resources, Global)
            .unwrap(),
        expected
    );
    assert_eq!(
        runtime.cpu.provider().inner.buffer_write_bytes() - before.1,
        14 + 28,
        "only F16 index and F32 byte-result planes should upload"
    );
    assert_eq!(
        runtime.cpu.provider().inner.buffer_read_bytes() - before.2,
        28 + 28
    );
    let allocations = runtime.cpu.provider().inner.buffer_allocations();
    let mut invocation = runtime
        .begin_with_resources_in(&executable, &inputs, &resources, Global)
        .unwrap();
    let mut submitted = Vec::new();
    let mut waiting = 0;
    let mut result = None;
    for _ in 0..24 {
        match runtime.advance_in(&mut invocation).unwrap() {
            AxInvocationStatus::Progress(progress) => match progress.phase {
                AxPartitionPhase::Submitted => submitted.push(progress.partition),
                AxPartitionPhase::Waiting => waiting += 1,
                _ => {}
            },
            AxInvocationStatus::Complete(outputs) => {
                result = Some(outputs);
                break;
            }
            AxInvocationStatus::Cancelled(_) => panic!("no cancellation requested"),
        }
    }
    assert_eq!(submitted, [0, 2]);
    assert_eq!(waiting, 2);
    assert_eq!(
        result.expect("four partitions complete within the step budget"),
        expected
    );
    assert_eq!(
        runtime.cpu.provider().inner.buffer_allocations(),
        allocations
    );
    assert!(invocation.is_finished());
    assert_eq!(runtime.cpu.pending_count(), 0);
}

#[test]
fn rgb_volume_outputs_cross_provider_partitions_with_resource_preflight() {
    use ax::sampling::{
        ResourceKey, VolumeRgbParams, VolumeRgbStorage, encode_volume_rgb_attributes,
    };
    use axnn_cpu::{BrickGridRef, RgbVolumeRef};
    use axnn_vaccel::{NativeResourceBinding, NativeResourceKind};

    fn sample_outputs(
        builder: &mut GraphBuilder<Global>,
        coordinates: [TensorId; 3],
        params: VolumeRgbParams,
    ) -> [TensorId; 3] {
        let outputs = [(); 3].map(|_| {
            builder
                .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 3], &[])
                .unwrap()
        });
        builder
            .node(
                OpCode::VOLUME_TRILINEAR_RGB,
                NodeFlags::READS_RESOURCE,
                &coordinates,
                &outputs,
                &encode_volume_rgb_attributes(params).unwrap(),
            )
            .unwrap();
        outputs
    }

    const KEY: ResourceKey = ResourceKey {
        group: 3,
        binding: 8,
    };
    const DIMS: [u32; 3] = [2, 2, 2];
    let params = VolumeRgbParams::new(KEY, DIMS, VolumeRgbStorage::Uf16E5M11).unwrap();
    let mut builder = GraphBuilder::new_in(Global);
    let half_x = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[1, 3], &[])
        .unwrap();
    let x = builder
        .tensor(DType::F32, TensorFlags::default(), &[1, 3], &[])
        .unwrap();
    builder
        .node(OpCode::CAST, NodeFlags::default(), &[half_x], &[x], &[])
        .unwrap();
    let y = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1, 1],
            &0.375_f32.to_le_bytes(),
        )
        .unwrap();
    let z = builder
        .tensor(
            DType::F32,
            TensorFlags::CONSTANT,
            &[1, 1],
            &0.625_f32.to_le_bytes(),
        )
        .unwrap();
    let first = sample_outputs(&mut builder, [x, y, z], params);
    let half_r = builder
        .tensor(DType::F16, TensorFlags::default(), &[1, 3], &[])
        .unwrap();
    let roundtrip_r = builder
        .tensor(DType::F32, TensorFlags::default(), &[1, 3], &[])
        .unwrap();
    builder
        .node(
            OpCode::CAST,
            NodeFlags::default(),
            &[first[0]],
            &[half_r],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::CAST,
            NodeFlags::default(),
            &[half_r],
            &[roundtrip_r],
            &[],
        )
        .unwrap();
    // All three first results remain graph outputs. Green and blue also cross
    // the second provider region before a later native sampler consumes them.
    sample_outputs(&mut builder, [roundtrip_r, first[1], first[2]], params);
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let cpu = Session::new(DelayedAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime: AxHetero<DelayedAccelerator, AxTosaAccelerator> = AxHetero::new(cpu, None);
    let executable = runtime
        .compile_in(
            &graph,
            &AxDeviceModel {
                accelerator: false,
                sparse_bsr: None,
                ..Default::default()
            },
            Global,
        )
        .unwrap();
    assert_eq!(
        executable
            .schedule()
            .plan()
            .partitions
            .iter()
            .map(|partition| partition.implementation)
            .collect::<Vec<_>>(),
        [
            DENSE_IMPLEMENTATION,
            CPU_KERNEL_IMPLEMENTATION,
            DENSE_IMPLEMENTATION,
            CPU_KERNEL_IMPLEMENTATION
        ]
    );
    let coordinate_values = [-0.5_f32, 0.25, 1.5];
    let input = f16_bytes(coordinate_values);
    let inputs: [&[u8]; 1] = [&input];
    let words: [[u16; 3]; 8] = core::array::from_fn(|index| {
        [
            0x7001 + index as u16 * 13,
            0x7400 + index as u16 * 7,
            0x7000 + index as u16 * 17,
        ]
    });
    let decoded = words.map(|rgb| rgb.map(rgb_uf16_reference));
    let resources = [NativeResourceBinding::rgb_volume(
        KEY,
        RgbVolumeRef::new_uf16(&words, DIMS).unwrap(),
    )];
    let brick = [0_u64];
    let wrong_kind = [NativeResourceBinding::brick_grid(
        KEY,
        BrickGridRef::new(&brick, [4; 3]).unwrap(),
    )];
    let wrong_storage = [NativeResourceBinding::rgb_volume(
        KEY,
        RgbVolumeRef::new_f32(&decoded, DIMS).unwrap(),
    )];
    let before = (
        runtime.cpu.provider().inner.buffer_allocations(),
        runtime.cpu.provider().inner.buffer_write_bytes(),
        runtime.cpu.provider().inner.buffer_read_bytes(),
    );
    let counts = Arc::new(AllocationCounts::default());
    for (case, bindings) in [(0, &[][..]), (1, &wrong_kind[..]), (2, &wrong_storage[..])] {
        for asynchronous in [false, true] {
            let allocator = TrackingAllocator(Arc::clone(&counts));
            let error = if asynchronous {
                match runtime.begin_with_resources_in(&executable, &inputs, bindings, allocator) {
                    Err(error) => error,
                    Ok(_) => panic!("invalid resource was admitted before async execution"),
                }
            } else {
                runtime
                    .run_with_resources_in(&executable, &inputs, bindings, allocator)
                    .unwrap_err()
            };
            assert!(match (case, error) {
                (0, AxCompileError::MissingResource(resource)) => resource == KEY,
                (
                    1,
                    AxCompileError::ResourceKind {
                        resource,
                        expected: NativeResourceKind::RgbVolume,
                        actual: NativeResourceKind::BrickGrid,
                    },
                ) => resource == KEY,
                (
                    2,
                    AxCompileError::ResourceStorage {
                        resource,
                        expected: VolumeRgbStorage::Uf16E5M11,
                        actual: VolumeRgbStorage::F32,
                    },
                ) => resource == KEY,
                (_, error) => panic!("unexpected preflight failure: {error:?}"),
            });
        }
    }
    assert_eq!(
        counts.total.load(Ordering::Relaxed),
        0,
        "resource failure allocated invocation storage"
    );
    assert_eq!(counts.live.load(Ordering::Relaxed), 0);
    assert_eq!(runtime.cpu.pending_count(), 0);
    assert_eq!(
        (
            runtime.cpu.provider().inner.buffer_allocations(),
            runtime.cpu.provider().inner.buffer_write_bytes(),
            runtime.cpu.provider().inner.buffer_read_bytes()
        ),
        before
    );

    let first_rgb =
        coordinate_values.map(|x| rgb_sample_reference(&decoded, DIMS, [x, 0.375, 0.625]));
    assert!(
        first_rgb
            .iter()
            .any(|rgb| rgb[0].to_bits() != ((rgb[0] as f16) as f32).to_bits())
    );
    let second_rgb = first_rgb
        .map(|rgb| rgb_sample_reference(&decoded, DIMS, [(rgb[0] as f16) as f32, rgb[1], rgb[2]]));
    let expected: Vec<Vec<u8>> = [first_rgb, second_rgb]
        .iter()
        .flat_map(|rgb| {
            (0..3).map(move |channel| f32_bytes(rgb.iter().map(|pixel| pixel[channel])))
        })
        .collect();
    let synchronous = runtime
        .run_with_resources_in(&executable, &inputs, &resources, Global)
        .unwrap();
    assert_eq!(synchronous, expected);
    assert_eq!(
        runtime.cpu.provider().inner.buffer_write_bytes() - before.1,
        6 + 12,
        "only F16 coordinates and the F32 red plane should be uploaded"
    );
    assert_eq!(
        runtime.cpu.provider().inner.buffer_read_bytes() - before.2,
        12 + 12
    );

    let allocations = runtime.cpu.provider().inner.buffer_allocations();
    let mut invocation = runtime
        .begin_with_resources_in(&executable, &inputs, &resources, Global)
        .unwrap();
    let mut submitted = Vec::new();
    let mut pending = 0;
    let mut asynchronous = None;
    for _ in 0..24 {
        match runtime.advance_in(&mut invocation).unwrap() {
            AxInvocationStatus::Progress(progress) => match progress.phase {
                AxPartitionPhase::Submitted => submitted.push(progress.partition),
                AxPartitionPhase::Waiting => pending += 1,
                _ => {}
            },
            AxInvocationStatus::Complete(outputs) => {
                asynchronous = Some(outputs);
                break;
            }
            AxInvocationStatus::Cancelled(_) => panic!("no cancellation was requested"),
        }
    }
    assert_eq!(submitted, [0, 2]);
    assert_eq!(
        pending, 2,
        "each delayed provider event must be observed pending"
    );
    assert_eq!(
        asynchronous.expect("four partitions complete within the step budget"),
        expected
    );
    assert!(invocation.is_finished());
    assert_eq!(runtime.cpu.pending_count(), 0);
    assert_eq!(
        runtime.cpu.provider().inner.buffer_allocations(),
        allocations
    );
}

#[test]
fn native_axnn_partition_releases_supplied_compile_and_run_allocators() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let rounded = builder
        .tensor(DType::F32, TensorFlags::default(), &[4], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    builder
        .node(
            OpCode::ROUND,
            NodeFlags::default(),
            &[input],
            &[rounded],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::RECIPROCAL,
            NodeFlags::default(),
            &[rounded],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let model = AxDeviceModel {
        accelerator: false,
        cpu_tosa: false,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();

    let compile_counts = Arc::new(AllocationCounts::default());
    let executable = runtime
        .compile_in(
            &graph,
            &model,
            TrackingAllocator(Arc::clone(&compile_counts)),
        )
        .unwrap();
    assert_eq!(
        executable.schedule().implementations(),
        [CPU_KERNEL_IMPLEMENTATION; 2]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);
    assert!(
        compile_counts.live.load(Ordering::Relaxed) > 0,
        "native executable did not retain its supplied compile allocator"
    );

    let run_counts = Arc::new(AllocationCounts::default());
    let input = f32_bytes([0.75, 1.5, -2.5, 4.25]);
    let outputs = runtime
        .run_in(
            &executable,
            &[&input],
            TrackingAllocator(Arc::clone(&run_counts)),
        )
        .unwrap();
    let actual: Vec<f32> = outputs[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    assert_eq!(actual, [1.0, 0.5, -0.5, 0.25]);
    assert!(
        run_counts.total.load(Ordering::Relaxed) > 0,
        "native execution bypassed its supplied run allocator"
    );
    assert!(
        run_counts.live.load(Ordering::Relaxed) > 0,
        "returned outputs should remain owned by the run allocator"
    );

    drop(outputs);
    assert_eq!(
        run_counts.live.load(Ordering::Relaxed),
        0,
        "native temporaries or outputs escaped their supplied run allocator"
    );
    assert!(
        compile_counts.live.load(Ordering::Relaxed) > 0,
        "dropping run storage released persistent native executable storage"
    );

    runtime.release_executable(executable).unwrap();
    assert_eq!(
        compile_counts.live.load(Ordering::Relaxed),
        0,
        "native executable allocations escaped their supplied compile allocator"
    );
}

#[test]
fn composite_recipe_capability_requires_every_primitive() {
    let mut builder = GraphBuilder::new_in(Global);
    let left = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[4096], &[])
        .unwrap();
    let right = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[4096], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4096], &[])
        .unwrap();
    builder
        .node(
            OpCode::DIV,
            NodeFlags::default(),
            &[left, right],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 64,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    }
    .with_accelerator_tosa_capability(&AddOnlyCapability, ADD_ONLY_TARGET);
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), [CPU_BACKEND]);
}

fn assert_rejected_at_placement(graph: &ax::Graph<'_>) {
    let mut runtime = sessions();
    match runtime.compile_in(graph, &AxDeviceModel::default(), Global) {
        Err(AxCompileError::Ax(AllocationError::Placement(PlacementError::NoBackend {
            node: 0,
        }))) => {}
        _ => panic!("unsupported dtype role must be rejected during placement"),
    }
}

#[test]
fn bf16_compute_is_not_advertised_by_the_axtosa_device_model() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::B16, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let output = builder
        .tensor(DType::B16, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    builder
        .node(OpCode::ABS, NodeFlags::default(), &[input], &[output], &[])
        .unwrap();
    let bytes = builder.finish().unwrap();
    assert_rejected_at_placement(&bytes.parse().unwrap());
}

#[test]
fn bf16_matmul_requires_an_explicit_provider_capability() {
    let mut builder = GraphBuilder::new_in(Global);
    let left = builder
        .tensor(DType::B16, TensorFlags::INPUT, &[1, 32, 64], &[])
        .unwrap();
    let right = builder
        .tensor(DType::B16, TensorFlags::INPUT, &[1, 64, 32], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 32, 32], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[left, right],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    assert_rejected_at_placement(&encoded.parse().unwrap());
}

#[test]
fn boolean_select_values_are_not_advertised_by_the_axtosa_device_model() {
    let mut builder = GraphBuilder::new_in(Global);
    let mask = builder
        .tensor(DType::U8, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let then = builder
        .tensor(DType::U8, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let otherwise = builder
        .tensor(DType::U8, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let output = builder
        .tensor(DType::U8, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    builder
        .node(
            OpCode::SELECT,
            NodeFlags::default(),
            &[mask, then, otherwise],
            &[output],
            &[],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    assert_rejected_at_placement(&bytes.parse().unwrap());
}

#[test]
fn provider_capabilities_reject_unsupported_nodes_before_program_admission() {
    let mut builder = GraphBuilder::new_in(Global);
    let left = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[4096], &[])
        .unwrap();
    let right = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[4096], &[])
        .unwrap();
    let sum = builder
        .tensor(DType::F32, TensorFlags::default(), &[4096], &[])
        .unwrap();
    builder
        .node(
            OpCode::ADD,
            NodeFlags::default(),
            &[left, right],
            &[sum],
            &[],
        )
        .unwrap();
    let output = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[4096], &[])
        .unwrap();
    builder
        .node(OpCode::CAST, NodeFlags::default(), &[sum], &[output], &[])
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();

    let model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 64,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    }
    .with_accelerator_tosa_capability(&AddOnlyCapability, ADD_ONLY_TARGET);
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(
        executable.schedule().placements(),
        &[ACCEL_BACKEND, CPU_BACKEND],
        "ADD is admitted by the provider descriptor; CAST/F16 is held on CPU"
    );
}

#[test]
fn finalized_partition_roles_replan_intermediate_only_capability_to_cpu() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[64], &[])
        .unwrap();
    let rounded = builder
        .tensor(DType::F32, TensorFlags::default(), &[64], &[])
        .unwrap();
    let sum = builder
        .tensor(DType::F32, TensorFlags::default(), &[64], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[64], &[])
        .unwrap();
    builder
        .node(
            OpCode::ROUND,
            NodeFlags::default(),
            &[input],
            &[rounded],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::ADD,
            NodeFlags::default(),
            &[rounded, rounded],
            &[sum],
            &[],
        )
        .unwrap();
    builder
        .node(OpCode::ROUND, NodeFlags::default(), &[sum], &[output], &[])
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let model = AxDeviceModel {
        cpu_tosa: false,
        accelerator_tosa_capability: Some(INTERMEDIATE_ADD_DESCRIPTOR),
        cpu_kernel_work_scale: 1_000,
        accelerator_launch_cost: 0,
        accelerator_speedup: 64,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), [CPU_BACKEND; 3]);
    assert_eq!(
        runtime.accel.as_ref().unwrap().provider().live_programs(),
        0
    );
}

#[test]
fn finalized_boolean_bridge_requires_its_synthesized_operators() {
    let mut builder = GraphBuilder::new_in(Global);
    let left = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[64], &[])
        .unwrap();
    let right = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[64], &[])
        .unwrap();
    let mask = builder
        .tensor(DType::U8, TensorFlags::default(), &[64], &[])
        .unwrap();
    let inverted = builder
        .tensor(DType::U8, TensorFlags::default(), &[64], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[64], &[])
        .unwrap();
    builder
        .node(
            OpCode::GREATER,
            NodeFlags::default(),
            &[left, right],
            &[mask],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::LOGICAL_NOT,
            NodeFlags::default(),
            &[mask],
            &[inverted],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::SELECT,
            NodeFlags::default(),
            &[inverted, left, right],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let model = AxDeviceModel {
        cpu_tosa: false,
        f32_boolean_boundary_bridge: true,
        accelerator_boolean_boundaries: false,
        accelerator_tosa_capability: Some(LOGICAL_BRIDGE_DESCRIPTOR),
        cpu_kernel_work_scale: 1_000,
        accelerator_launch_cost: 0,
        accelerator_speedup: 64,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), [CPU_BACKEND; 3]);
    assert_eq!(
        runtime.accel.as_ref().unwrap().provider().live_programs(),
        0
    );
}

#[test]
fn missing_exact_provider_target_disables_accelerator_placement() {
    let unmatched = Target::new(
        Version::TOSA_1_0,
        ProfileSet::INTEGER,
        Level::Level8K,
        ExtensionSet::NONE,
    );
    let model =
        AxDeviceModel::default().with_accelerator_tosa_capability(&AddOnlyCapability, unmatched);
    assert!(!model.accelerator);
    assert!(model.accelerator_tosa_capability.is_none());
}

#[test]
fn unsupported_boolean_crossing_replans_accel_cpu_accel_component_to_cpu() {
    let mut builder = GraphBuilder::new_in(Global);
    let left = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[64], &[])
        .unwrap();
    let right = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[64], &[])
        .unwrap();
    let mask = builder
        .tensor(DType::U8, TensorFlags::default(), &[64], &[])
        .unwrap();
    let rounded = builder
        .tensor(DType::F32, TensorFlags::default(), &[64], &[])
        .unwrap();
    let output = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[64], &[])
        .unwrap();
    // GREATER and ROUND are independently ready in authored order. SELECT
    // depends on both, so a cheap accelerator initially produces A/C/A runs.
    builder
        .node(
            OpCode::GREATER,
            NodeFlags::default(),
            &[left, right],
            &[mask],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::ROUND,
            NodeFlags::default(),
            &[left],
            &[rounded],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::SELECT,
            NodeFlags::default(),
            &[mask, rounded, right],
            &[output],
            &[],
        )
        .unwrap();
    let encoded = builder.finish().unwrap();
    let graph = encoded.parse().unwrap();
    let model = AxDeviceModel {
        cpu_tosa: false,
        accelerator_boolean_boundaries: false,
        f32_boolean_boundary_bridge: false,
        cpu_kernel_work_scale: 1_000,
        accelerator_launch_cost: 0,
        accelerator_speedup: 64,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        sparse_bsr: None,
        ..AxDeviceModel::default()
    };
    let mut runtime = sessions();
    let executable = runtime.compile_in(&graph, &model, Global).unwrap();
    assert_eq!(executable.schedule().placements(), [CPU_BACKEND; 3]);
    assert_eq!(
        executable.schedule().implementations(),
        [CPU_KERNEL_IMPLEMENTATION; 3]
    );
    assert_eq!(executable.schedule().plan().partitions.len(), 1);

    let left = f32_bytes((0..64).map(|value| value as f32 - 20.25));
    let right = f32_bytes((0..64).map(|_| 0.0));
    let outputs = runtime
        .run_in(&executable, &[&left, &right], Global)
        .unwrap();
    let actual: Vec<f32> = outputs[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    for (index, &value) in actual.iter().enumerate() {
        let source = index as f32 - 20.25;
        let expected = if source > 0.0 { source.round() } else { 0.0 };
        assert_eq!(value, expected);
    }
}

#[test]
fn automatic_model_uses_provider_roles_and_rejects_missing_target_without_presets() {
    for capabilities in [
        &[LOGICAL_BRIDGE_DESCRIPTOR][..],
        &[ADD_ONLY_DESCRIPTOR][..],
        &[][..],
    ] {
        let mut provider = DelayedAccelerator::new();
        provider.capabilities = Some(capabilities);
        let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
        let accel = Session::new(provider, ADD_ONLY_TARGET).unwrap();
        let mut runtime = AxHetero::new(cpu, Some(accel));
        let model = runtime.automatic_device_model();
        assert!(model.prefer_native_cpu);
        assert_eq!(model.accelerator, !capabilities.is_empty());
        assert!(!model.accelerator_boolean_boundaries || capabilities.is_empty());
        assert_eq!(
            model.f32_boolean_boundary_bridge,
            capabilities
                .first()
                .is_some_and(|cap| cap.supports_dtype(TosaDType::BOOL, ValueRoles::INTERMEDIATE))
        );
        // A provider without all required bridge operators must not turn this
        // logical graph into an invalid accelerator program.
        let graph = graph_with_boolean_abi_component();
        let executable = runtime
            .compile_auto_in(&graph.parse().unwrap(), Global)
            .unwrap();
        assert_eq!(executable.single_backend(), Some(CPU_BACKEND));
        let x = f32_bytes([-2.0, -1.0, 2.0, 3.0]);
        assert_eq!(
            runtime
                .run_in(&executable, &[&x, &[1, 1, 1, 0]], Global)
                .unwrap()[1],
            [1, 1, 1, 0]
        );
    }
    let runtime = sessions();
    let model = runtime.automatic_device_model();
    assert!(model.accelerator_boolean_boundaries);
    assert!(!model.f32_boolean_boundary_bridge);
}

#[test]
fn automatic_dispatch_accelerates_large_supported_work_and_keeps_cpu_coverage_fallback() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F16, TensorFlags::INPUT, &[1, 64, 128], &[])
        .unwrap();
    let weights = f16_bytes((0..128 * 128).map(|i| f32::from(i / 128 == i % 128)));
    let weight = builder
        .tensor(DType::F16, TensorFlags::CONSTANT, &[1, 128, 128], &weights)
        .unwrap();
    let output = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 64, 128], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[input, weight],
            &[output],
            &[],
        )
        .unwrap();
    let graph = builder.finish().unwrap();
    let mut runtime = sessions();
    let executable = runtime
        .compile_auto_in(&graph.parse().unwrap(), Global)
        .unwrap();
    assert_eq!(executable.single_backend(), Some(ACCEL_BACKEND));
    let input = f16_bytes((0..64 * 128).map(|i| (i % 11) as f32 / 8.0));
    assert_eq!(
        runtime.run_in(&executable, &[&input], Global).unwrap()[0],
        input
    );
    assert!(
        runtime
            .accel
            .as_ref()
            .unwrap()
            .provider()
            .buffer_allocations()
            > 0
    );
    // F16 MATMUL has no native CPU-kernel plan; automatic preference must
    // preserve the CPU TOSA coverage route when no accelerator is present.
    let mut cpu: AxHetero<_, AxTosaAccelerator> = AxHetero::new(
        Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap(),
        None,
    );
    let executable = cpu
        .compile_auto_in(&graph.parse().unwrap(), Global)
        .unwrap();
    assert_eq!(executable.single_backend(), Some(CPU_BACKEND));
    assert_eq!(
        executable.schedule().implementations(),
        [DENSE_IMPLEMENTATION]
    );
    assert_eq!(
        cpu.run_in(&executable, &[&input], Global).unwrap()[0],
        input
    );
}

#[test]
fn automatic_native_fusion_keeps_values_consumed_by_other_partitions() {
    let mut builder = GraphBuilder::new_in(Global);
    let input = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[1, 16], &[])
        .unwrap();
    let product = builder
        .tensor(DType::F32, TensorFlags::default(), &[1, 16], &[])
        .unwrap();
    let sum = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 16], &[])
        .unwrap();
    let narrow = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 16], &[])
        .unwrap();
    builder
        .node(
            OpCode::MUL,
            NodeFlags::default(),
            &[input, input],
            &[product],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::ADD,
            NodeFlags::default(),
            &[product, input],
            &[sum],
            &[],
        )
        .unwrap();
    builder
        .node(
            OpCode::CAST,
            NodeFlags::default(),
            &[product],
            &[narrow],
            &[],
        )
        .unwrap();
    let graph = builder.finish().unwrap();
    let mut runtime: AxHetero<_, AxTosaAccelerator> = AxHetero::new(
        Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap(),
        None,
    );
    let executable = runtime
        .compile_auto_in(&graph.parse().unwrap(), Global)
        .unwrap();
    let stats = executable.native_plan_stats();
    assert_eq!(
        (
            stats.logical_nodes,
            stats.instructions,
            stats.mul_add_fusions
        ),
        (2, 2, 0)
    );
    assert!(
        executable
            .schedule()
            .implementations()
            .contains(&DENSE_IMPLEMENTATION)
    );
    let input = f32_bytes((0..16).map(|i| i as f32 / 4.0));
    let outputs = runtime.run_in(&executable, &[&input], Global).unwrap();
    assert_eq!(
        outputs[0],
        f32_bytes((0..16).map(|i| {
            let x = i as f32 / 4.0;
            x * x + x
        }))
    );
    assert_eq!(
        outputs[1],
        f16_bytes((0..16).map(|i| {
            let x = i as f32 / 4.0;
            x * x
        }))
    );

    let repeated = runtime
        .run_repeated_auto_in(
            &executable,
            &[&input],
            2,
            &[AxFeedback::new(executable.input_ids()[0], sum)],
            Global,
        )
        .unwrap();
    let after_first = (0..16).map(|i| {
        let x = i as f32 / 4.0;
        x * x + x
    });
    assert_eq!(
        repeated[0],
        f32_bytes(after_first.clone().map(|x| x * x + x))
    );
    assert_eq!(repeated[1], f16_bytes(after_first.map(|x| x * x)));
}

#[test]
fn automatic_recurrence_retains_native_backing_and_restores_invariant_passthroughs() {
    const LANES: usize = 4096;
    let mut builder = GraphBuilder::new_in(Global);
    let a = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[LANES as u32], &[])
        .unwrap();
    let b = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[LANES as u32], &[])
        .unwrap();
    let scale = builder
        .tensor(
            DType::F32,
            TensorFlags(TensorFlags::INPUT.0 | TensorFlags::OUTPUT.0),
            &[1],
            &[],
        )
        .unwrap();
    let one = builder
        .tensor(
            DType::F32,
            TensorFlags(TensorFlags::CONSTANT.0 | TensorFlags::OUTPUT.0),
            &[1],
            &1.0_f32.to_le_bytes(),
        )
        .unwrap();
    let product = builder
        .tensor(DType::F32, TensorFlags::default(), &[LANES as u32], &[])
        .unwrap();
    let a_out = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[LANES as u32], &[])
        .unwrap();
    let b_out = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[LANES as u32], &[])
        .unwrap();
    for (op, inputs, output) in [
        (OpCode::MUL, [b, scale], product),
        (OpCode::ADD, [product, one], a_out),
        (OpCode::ADD, [a, one], b_out),
    ] {
        builder
            .node(op, NodeFlags::default(), &inputs, &[output], &[])
            .unwrap();
    }
    let graph = builder.finish().unwrap();
    let mut runtime: AxHetero<_, AxTosaAccelerator> = AxHetero::new(
        Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap(),
        None,
    );
    let executable = runtime
        .compile_auto_in(&graph.parse().unwrap(), Global)
        .unwrap();
    assert_eq!(executable.native_plan_stats().mul_add_fusions, 1);
    let a_bytes = f32_bytes([0.0; LANES]);
    let b_bytes = f32_bytes([10.0; LANES]);
    let scale_bytes = 2.0_f32.to_le_bytes();
    let inputs = [&a_bytes[..], &b_bytes[..], &scale_bytes[..]];
    for invalid in [
        vec![AxFeedback::new(a, a_out), AxFeedback::new(a, b_out)],
        vec![AxFeedback::new(a, a_out), AxFeedback::new(b, a_out)],
        vec![AxFeedback::new(scale, a_out)],
        vec![AxFeedback::new(a, TensorId(999))],
    ] {
        assert!(matches!(
            runtime.run_repeated_auto_in(&executable, &inputs, 3, &invalid, Global),
            Err(AxCompileError::InvalidFeedback { .. })
        ));
    }
    assert!(matches!(
        runtime.run_repeated_auto_in(&executable, &inputs, 0, &[], Global),
        Err(AxCompileError::InvalidRepetitionCount)
    ));
    assert!(matches!(
        runtime.run_repeated_auto_in(&executable, &inputs[..2], 3, &[], Global),
        Err(AxCompileError::InvalidInputCount { .. })
    ));

    let counts = Arc::new(AllocationCounts::default());
    let outputs = runtime
        .run_repeated_auto_in(
            &executable,
            &inputs,
            3,
            &[AxFeedback::new(a, a_out), AxFeedback::new(b, b_out)],
            TrackingAllocator(counts.clone()),
        )
        .unwrap();
    assert_eq!(outputs[0].as_slice(), scale_bytes);
    assert_eq!(outputs[1].as_slice(), 1.0_f32.to_le_bytes());
    assert_eq!(outputs[2].as_slice(), f32_bytes([45.0; LANES]));
    assert_eq!(outputs[3].as_slice(), f32_bytes([4.0; LANES]));
    if cfg!(target_endian = "little") {
        assert_eq!(
            counts.large_byte_allocations.load(Ordering::Relaxed),
            2,
            "only final F32 outputs should become byte planes"
        );
    }
    assert_eq!(runtime.cpu.provider().buffer_allocations(), 0);
    drop(outputs);
    assert_eq!(counts.live.load(Ordering::Relaxed), 0);
}

#[test]
fn mixed_provider_types_preserve_indeterminate_events_and_stream_charges() {
    use axnn_vaccel::{AxStream, AxStreamProgress};
    let basic = || Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
    let mut runtime = AxHetero::new(basic(), Some(basic()));
    let distinct = Session::new(
        DelayedAccelerator::adversarial(),
        AxTosaAccelerator::target(),
    )
    .unwrap();
    assert_eq!(runtime.add_accelerator(distinct).ok(), Some(2));
    let mut builder = GraphBuilder::new_in(Global);
    let x = builder
        .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
        .unwrap();
    let y = builder
        .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
        .unwrap();
    builder
        .node(OpCode::ABS, NodeFlags::default(), &[x], &[y], &[])
        .unwrap();
    let bytes = builder.finish().unwrap();
    let mut model = runtime.automatic_device_set();
    model.cpu.cpu = false;
    model.accelerators[0].accelerator = false;
    let executable = runtime
        .compile_set_in(&bytes.parse().unwrap(), &model, Global)
        .unwrap();
    assert_eq!(executable.single_backend(), Some(2));
    let mut stream = AxStream::new([&executable], 16, Global).unwrap();
    stream
        .try_submit(&runtime, 0, vec![f32_bytes([-1., 2., -3., 4.])])
        .unwrap_or_else(|_| panic!("admission"));
    let mut indeterminate = false;
    let mut rejected_release = false;
    let mut complete = false;
    for _ in 0..16 {
        match stream.poll(&mut runtime) {
            AxStreamProgress::Progress { progress, .. } => {
                indeterminate |= progress.submit_error.is_some();
                assert_eq!(stream.input_bytes(), 16);
            }
            AxStreamProgress::Failed {
                error: AxCompileError::Backend(BackendError::Busy),
                retained: true,
                ..
            } => {
                rejected_release = true;
                assert_eq!(stream.input_bytes(), 16);
                assert_eq!(stream.active_count(), 1);
            }
            AxStreamProgress::Complete { outputs, .. } => {
                assert_eq!(outputs[0].as_bytes(), f32_bytes([1., 2., 3., 4.]));
                complete = true;
                break;
            }
            _ => panic!("unexpected state"),
        }
    }
    assert!(indeterminate && rejected_release && complete);
    assert_eq!(stream.input_bytes(), 0);
    drop(stream);
    runtime.release_executable(executable).unwrap();
}

const FP8_TIER_TARGET: Target = Target::new(
    Version::TOSA_1_0,
    ProfileSet::FLOATING_POINT,
    Level::Level8K,
    ExtensionSet::FP8E4M3.union(ExtensionSet::FP8E5M2),
);
const FP8_TIER_DTYPES: &[DTypeCapability] = &[
    DTypeCapability::new(TosaDType::FP8E4M3, ValueRoles::ALL),
    DTypeCapability::new(TosaDType::FP16, ValueRoles::ALL),
];
// CONST is required alongside MATMUL: axnn's MATMUL carries no zero points, so the lowering
// synthesizes zero constants for them and the descriptor has to admit emitting those.
const FP8_TIER_OPERATORS: &[OperatorCapability] = &[
    OperatorCapability::new(TosaOp::MATMUL),
    OperatorCapability::new(TosaOp::CONST),
];
const FP8_TIER_DESCRIPTOR: CapabilityDescriptor = CapabilityDescriptor {
    target: FP8_TIER_TARGET,
    dtypes: FP8_TIER_DTYPES,
    operators: FP8_TIER_OPERATORS,
    graph: GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    },
};

fn fp8_matmul_graph() -> GraphBuf<Global> {
    let mut builder = GraphBuilder::new_in(Global);
    let left = builder
        .tensor(DType::B8, TensorFlags::INPUT, &[1, 4, 4], &[])
        .unwrap();
    let right = builder
        .tensor(DType::B8, TensorFlags::INPUT, &[1, 4, 2], &[])
        .unwrap();
    let out = builder
        .tensor(DType::F16, TensorFlags::OUTPUT, &[1, 4, 2], &[])
        .unwrap();
    builder
        .node(
            OpCode::MATMUL,
            NodeFlags::default(),
            &[left, right],
            &[out],
            &[],
        )
        .unwrap();
    builder.finish().unwrap()
}

/// FP8 on an accelerator is a declared tier, never a default, exactly as the BF16 MATMUL row is.
/// The accelerator profile admits the `(FP8, FP8) -> FP16` row so a provider's descriptor can
/// decide, which means the absence of a descriptor has to be a refusal rather than a shrug: an
/// Intel NPU at arch 40XX rejects FP8 outright while arch 5010 accepts it, and the difference is
/// only visible in what the provider advertises.
#[test]
fn fp8_matmul_reaches_the_accelerator_only_when_a_provider_declares_it() {
    let bytes = fp8_matmul_graph();
    let graph = bytes.parse().unwrap();
    let undeclared_model = AxDeviceModel {
        accelerator_launch_cost: 0,
        accelerator_speedup: 64,
        transfer_fixed_cost: 0,
        transfer_byte_cost: 0,
        ..AxDeviceModel::default()
    };
    let mut undeclared = Schedule::analyze_in(&graph, Global).unwrap();
    undeclared.place(&graph, &undeclared_model).unwrap();
    assert_eq!(
        undeclared.placements(),
        [CPU_BACKEND],
        "FP8 went to an accelerator that never claimed the tier"
    );

    let declared_model = AxDeviceModel {
        accelerator_tosa_capability: Some(FP8_TIER_DESCRIPTOR),
        ..undeclared_model
    };
    let mut declared = Schedule::analyze_in(&graph, Global).unwrap();
    declared.place(&graph, &declared_model).unwrap();
    assert_eq!(
        declared.placements(),
        [ACCEL_BACKEND],
        "a declared FP8 tier did not receive the MATMUL"
    );
}
