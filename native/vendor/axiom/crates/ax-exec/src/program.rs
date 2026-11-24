use alloc::vec::Vec;
use core::alloc::Allocator;

use ax::{
    CostModel, DType, Graph, GraphBuf, Implementation, NodeView, Partition, PlacementReport,
    Schedule, TensorFlags, TensorId, TensorView,
};
use axnn::lowering::{KernelParams, MAX_TENSOR_RESULTS, SemanticOp};
use virtio_accel_core::ByteSink;

use crate::native::{
    self, NativeCpuPartition, NativeValue, PreparedNativeInstruction, PreparedNativeScheduleEntry,
    reserve, tensor_storage_bytes,
};
use crate::storage::{NativeRef, NativeSlot, RuntimeValue};
use crate::{
    AxCpuKernelPolicy, AxHostCopyStats, AxNativePlanStats, AxOutput, AxTensor,
    CPU_KERNEL_IMPLEMENTATION, Error, NativeResourceBinding, PlanError, PreparedNative,
};

/// One retained native graph. The same partition compiler and instruction
/// executor are used by the host heterogeneous adapter. This is a CPU lane,
/// not a replacement provider scheduler or a device artifact compiler.
pub struct NativeProgram<A: Allocator> {
    graph: GraphBuf<A>,
    schedule: Schedule<A>,
    placement: PlacementReport,
    native: NativeCpuPartition<A>,
    inputs: Vec<TensorId, A>,
    outputs: Vec<TensorId, A>,
}

struct Model(AxCpuKernelPolicy);
impl CostModel for Model {
    fn backend_count(&self) -> u16 {
        1
    }
    fn supports(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> bool {
        backend == 0
            && ax::node_kernel_request(graph, node)
                .and_then(|r| self.0.plan(&r))
                .is_some()
    }
    fn execute_cost(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u64 {
        if backend != 0 {
            return u64::MAX;
        }
        ax::node_kernel_request(graph, node)
            .and_then(|r| self.0.plan(&r))
            .map_or(u64::MAX, |p| p.estimated_cost)
    }
    fn transfer_cost(&self, _: TensorView<'_>, from: Option<u16>, to: u16) -> u64 {
        if to == 0 && from.is_none_or(|from| from == 0) {
            0
        } else {
            u64::MAX
        }
    }
    fn external_output_backend(&self) -> Option<u16> {
        Some(0)
    }
    fn implementation_count(&self, _: u16, _: &Graph<'_>, _: NodeView<'_>) -> u16 {
        CPU_KERNEL_IMPLEMENTATION.0 + 1
    }
    fn supports_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> bool {
        implementation == CPU_KERNEL_IMPLEMENTATION && self.supports(backend, graph, node)
    }
    fn execute_cost_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> u64 {
        if implementation == CPU_KERNEL_IMPLEMENTATION {
            self.execute_cost(backend, graph, node)
        } else {
            u64::MAX
        }
    }
}

impl<A: Allocator + Clone> NativeProgram<A> {
    /// Select exact native fusions without caller optimization switches.
    /// The compiled CPU capability contract is used; no runtime ISA probing.
    pub fn compile_auto_in(graph: &Graph<'_>, allocator: A) -> Result<Self, Error> {
        Self::compile_in(graph, AxCpuKernelPolicy::default(), true, allocator)
    }

    /// Explicit policy/reference path for a runtime that owns CPU admission.
    pub fn compile_in(
        graph: &Graph<'_>,
        policy: AxCpuKernelPolicy,
        fusions: bool,
        allocator: A,
    ) -> Result<Self, Error> {
        let graph = GraphBuf::copy_from_in(graph.as_bytes(), allocator.clone())?;
        Self::from_graph_in(graph, policy, fusions, allocator)
    }

    /// Adopt an already owned graph without duplicating its encoded constants.
    /// Pass `true` for the same automatic exact fusions as `compile_auto_in`.
    pub fn from_graph_in(
        graph: GraphBuf<A>,
        policy: AxCpuKernelPolicy,
        fusions: bool,
        allocator: A,
    ) -> Result<Self, Error> {
        let parsed = graph.parse().map_err(|_| PlanError::InvalidGraph)?;
        let mut schedule = Schedule::analyze_in(&parsed, allocator.clone())?;
        let placement = schedule.place_optimized(&parsed, &Model(policy))?;
        schedule
            .build_plan(&parsed)
            .map_err(ax::AllocationError::from)?;
        let partition = match schedule.plan().partitions {
            [partition] => *partition,
            [] if parsed.node_count() == 0 => Partition {
                backend: 0,
                implementation: CPU_KERNEL_IMPLEMENTATION,
                order_start: 0,
                node_count: 0,
            },
            _ => return Err(Error::NativeRequiresSinglePartition),
        };
        let mut native = native::compile_native_cpu_partition_in(
            &parsed,
            schedule.order(),
            schedule.producers(),
            partition,
            policy,
            fusions,
            allocator.clone(),
        )?;
        let mut inputs = Vec::new_in(allocator.clone());
        let mut outputs = Vec::new_in(allocator);
        for index in 0..parsed.tensor_count() {
            let id = TensorId(index as u32);
            let tensor = parsed.tensor(id).map_err(|_| PlanError::InvalidGraph)?;
            if tensor.flags.contains(TensorFlags::INPUT) {
                reserve(&mut inputs, 1)?;
                inputs.push(id);
            }
            if tensor.flags.contains(TensorFlags::OUTPUT) {
                reserve(&mut outputs, 1)?;
                outputs.push(id);
                // Preserve constant/input passthrough outputs, including ones
                // not consumed by any instruction in the native region.
                if !native.outputs.contains(&id) {
                    reserve(&mut native.outputs, 1)?;
                    native.outputs.push(id);
                    let slot = if let Some(slot) = native.tensors.iter().position(|&v| v == id) {
                        slot
                    } else {
                        reserve(&mut native.tensors, 1)?;
                        reserve(&mut native.takes, 1)?;
                        native.tensors.push(id);
                        native.takes.push(0);
                        native.tensors.len() - 1
                    };
                    native.takes[slot] = native.takes[slot]
                        .checked_add(1)
                        .ok_or(PlanError::InvalidPartition)?;
                    if tensor.flags.contains(TensorFlags::INPUT) && !native.inputs.contains(&id) {
                        reserve(&mut native.inputs, 1)?;
                        native.inputs.push(id);
                    }
                }
            }
        }
        Ok(Self {
            graph,
            schedule,
            placement,
            native,
            inputs,
            outputs,
        })
    }
}

impl<A: Allocator> NativeProgram<A> {
    pub fn graph(&self) -> Graph<'_> {
        self.graph.parse().expect("retained validated graph")
    }
    pub fn schedule(&self) -> &Schedule<A> {
        &self.schedule
    }
    pub fn as_bytes(&self) -> &[u8] {
        self.graph.as_bytes()
    }
    pub fn placement(&self) -> PlacementReport {
        self.placement
    }
    pub fn kernel_plans(&self) -> impl Iterator<Item = &axnn_cpu::CpuKernelPlan> {
        self.native.nodes.iter().map(|node| &node.plan)
    }
    pub fn input_ids(&self) -> &[TensorId] {
        &self.inputs
    }
    pub fn output_ids(&self) -> &[TensorId] {
        &self.outputs
    }
    pub fn native_plan_stats(&self) -> AxNativePlanStats {
        self.native.stats()
    }

    pub fn prepare_native_in<R: Allocator + Clone>(
        &self,
        allocator: R,
    ) -> Result<PreparedNative<R>, Error> {
        self.prepare(false, allocator)
    }

    /// Prepare an arena that owns no input storage: intermediates, outputs,
    /// constants and scratch are allocated once, and every run borrows its
    /// inputs in place through
    /// [`PreparedNative::run_with_native_inputs_and_resources`]. For graphs
    /// whose inputs are large and long-lived, such as mapped weights. The
    /// byte-ingress runs are refused.
    pub fn prepare_borrowing_inputs_in<R: Allocator + Clone>(
        &self,
        allocator: R,
    ) -> Result<PreparedNative<R>, Error> {
        self.prepare(true, allocator)
    }

    fn prepare<R: Allocator + Clone>(
        &self,
        borrow_inputs: bool,
        allocator: R,
    ) -> Result<PreparedNative<R>, Error> {
        native::prepare_native_ingress_in(
            &self.native,
            self.graph(),
            &self.inputs,
            &self.outputs,
            borrow_inputs,
            allocator,
        )
        .map(|inner| PreparedNative { inner })
    }

    pub fn begin_in<'a, R: Allocator + Clone>(
        &'a self,
        inputs: &[&'a [u8]],
        allocator: R,
    ) -> Result<NativeInvocation<'a, A, R>, Error> {
        self.begin_with_resources_in(inputs, &[], allocator)
    }

    pub fn begin_with_resources_in<'a, R: Allocator + Clone>(
        &'a self,
        inputs: &[&'a [u8]],
        resources: &'a [NativeResourceBinding<'a>],
        allocator: R,
    ) -> Result<NativeInvocation<'a, A, R>, Error> {
        if inputs.len() != self.inputs.len() {
            return Err(Error::InvalidInputCount {
                expected: self.inputs.len(),
                actual: inputs.len(),
            });
        }
        let graph = self.graph();
        for (&id, bytes) in self.inputs.iter().zip(inputs) {
            let tensor = graph.tensor(id).map_err(|_| Error::MissingTensor(id))?;
            let expected = tensor_storage_bytes(tensor).ok_or(Error::MissingTensor(id))?;
            if expected != bytes.len() {
                return Err(Error::InvalidInputSize {
                    tensor: id,
                    expected,
                    actual: bytes.len(),
                });
            }
        }
        NativeInvocation::from_partition(
            &self.native,
            graph,
            &self.outputs,
            resources,
            |id| self.inputs.iter().position(|&v| v == id).map(|i| inputs[i]),
            allocator,
        )
    }
}

impl<A: Allocator> NativeCpuPartition<A> {
    pub fn stats(&self) -> AxNativePlanStats {
        let mut stats = AxNativePlanStats {
            logical_nodes: self.nodes.len(),
            ..Default::default()
        };
        for entry in &self.instructions {
            match entry {
                PreparedNativeScheduleEntry::Elided => continue,
                PreparedNativeScheduleEntry::Kernel(_) => {}
                PreparedNativeScheduleEntry::MulAdd(_) => stats.mul_add_fusions += 1,
                PreparedNativeScheduleEntry::SubMulAdd(_) => stats.sub_mul_add_fusions += 1,
                PreparedNativeScheduleEntry::ScaleInterleave3(_) => {
                    stats.scale_interleave3_fusions += 1
                }
                PreparedNativeScheduleEntry::CastShiftOr3(_) => stats.cast_shift_or3_fusions += 1,
            }
            stats.instructions += 1;
        }
        stats
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeProgress {
    /// At most the requested number of whole instructions ran. A kernel itself
    /// is not preemptible; runtimes should submit suitably bounded tiles.
    Progress {
        instructions: usize,
    },
    Ready,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Running,
    Ready,
    Finished,
}

/// Caller-driven execution with no hidden wait, thread, queue, or clock.
/// Inputs/resources stay borrowed until this invocation is dropped. Output
/// mappings are only borrowed by the completion call and never retained.
pub struct NativeInvocation<'a, M: Allocator, R: Allocator + Clone> {
    native: &'a NativeCpuPartition<M>,
    graph: Graph<'a>,
    outputs: &'a [TensorId],
    resources: &'a [NativeResourceBinding<'a>],
    values: Vec<Option<NativeSlot<'a, R>>, R>,
    takes: Vec<u32, R>,
    cursor: usize,
    state: State,
    copies: AxHostCopyStats,
    allocator: R,
}

impl<'a, M: Allocator, R: Allocator + Clone> NativeInvocation<'a, M, R> {
    pub(crate) fn from_partition(
        native: &'a NativeCpuPartition<M>,
        graph: Graph<'a>,
        outputs: &'a [TensorId],
        resources: &'a [NativeResourceBinding<'a>],
        input: impl Fn(TensorId) -> Option<&'a [u8]>,
        allocator: R,
    ) -> Result<Self, Error> {
        native::validate_resource_keys(resources)?;
        native::validate_native_resources(&native.nodes, resources)?;
        let mut values = Vec::new_in(allocator.clone());
        reserve(&mut values, native.tensors.len())?;
        let mut copies = AxHostCopyStats::default();
        for &id in &native.tensors {
            let tensor = graph.tensor(id).map_err(|_| Error::MissingTensor(id))?;
            let source = if tensor.flags.contains(TensorFlags::CONSTANT) {
                Some(tensor.constant())
            } else if native.inputs.contains(&id) {
                Some(input(id).ok_or(Error::MissingTensor(id))?)
            } else {
                None
            };
            values.push(match source {
                Some(bytes) => Some(decode_source(
                    tensor,
                    bytes,
                    &mut copies,
                    allocator.clone(),
                )?),
                None => None,
            });
        }
        let mut takes = Vec::new_in(allocator.clone());
        reserve(&mut takes, native.takes.len())?;
        takes.extend_from_slice(&native.takes);
        Ok(Self {
            native,
            graph,
            outputs,
            resources,
            values,
            takes,
            cursor: 0,
            state: State::Running,
            copies,
            allocator,
        })
    }

    pub fn host_copy_stats(&self) -> AxHostCopyStats {
        self.copies
    }

    pub fn advance(&mut self, budget: usize) -> Result<NativeProgress, Error> {
        if self.state == State::Finished {
            return Err(Error::InvocationFinished);
        }
        if budget == 0 {
            return Err(Error::InvalidStepBudget);
        }
        if self.state == State::Ready {
            return Ok(NativeProgress::Ready);
        }
        let result = self.advance_inner(budget);
        if result.is_err() {
            self.state = State::Finished;
        }
        result
    }

    fn advance_inner(&mut self, budget: usize) -> Result<NativeProgress, Error> {
        let mut executed = 0;
        while self.cursor < self.native.instructions.len() && executed < budget {
            let entry = self.native.instructions[self.cursor];
            self.cursor += 1;
            let Some(node) = entry.instruction(&self.native.nodes) else {
                continue;
            };
            let mut outputs: [Option<NativeValue<R>>; MAX_TENSOR_RESULTS] =
                core::array::from_fn(|_| None);
            for (output, slot) in outputs.iter_mut().zip(node.outputs()) {
                let id = self.native.tensors[slot as usize];
                let tensor = self
                    .graph
                    .tensor(id)
                    .map_err(|_| Error::MissingTensor(id))?;
                let bytes = tensor_storage_bytes(tensor).ok_or(Error::MissingTensor(id))?;
                *output = Some(native::allocate_native_value_in(
                    tensor.dtype,
                    bytes / tensor.dtype.block_byte_size() * tensor.dtype.block_size(),
                    self.allocator.clone(),
                )?);
            }
            let mut voxel_ids = Vec::new_in(self.allocator.clone());
            if let PreparedNativeInstruction::Kernel(kernel) = &node
                && kernel.plan.semantic == SemanticOp::VoxelBrickTrace
            {
                let elements = kernel.plan.shape.num_elements();
                reserve(&mut voxel_ids, elements)?;
                voxel_ids.resize(elements, 0);
            }
            let mut kernel_scratch = Vec::new_in(self.allocator.clone());
            if let PreparedNativeInstruction::Kernel(kernel) = &node
                && let Some(elements) = match kernel.plan.params() {
                    KernelParams::AttentionDecode(params) => {
                        Some(axnn_cpu::attention_scratch_elements(params))
                    }
                    KernelParams::MoeExperts(params) => {
                        Some(axnn_cpu::moe_scratch_elements(params))
                    }
                    _ => None,
                }
            {
                reserve(&mut kernel_scratch, elements)?;
                kernel_scratch.resize(elements, 0.0);
            }
            native::invoke_native_instruction(
                &node,
                self.values.as_slice(),
                &mut outputs[..node.output_slots().len()],
                self.resources,
                &mut voxel_ids,
                &mut kernel_scratch,
                &axnn_cpu::Serial,
            )?;
            for (output, slot) in outputs.iter_mut().zip(node.outputs()) {
                // A dead output reaches its last use at birth. Retaining it
                // until invocation end makes unused branches consume an
                // unbounded sum of full tensor allocations.
                self.values[slot as usize] = if self.takes[slot as usize] == 0 {
                    None
                } else {
                    output.take().map(NativeSlot::Owned)
                };
            }
            for &input in node.inputs() {
                let remaining = self
                    .takes
                    .get_mut(input as usize)
                    .ok_or(PlanError::InvalidPartition)?;
                *remaining = remaining
                    .checked_sub(1)
                    .ok_or(PlanError::InvalidPartition)?;
                if *remaining == 0 {
                    self.values[input as usize] = None;
                }
            }
            executed += 1;
        }
        if self.cursor == self.native.instructions.len() {
            self.state = State::Ready;
            Ok(NativeProgress::Ready)
        } else {
            Ok(NativeProgress::Progress {
                instructions: executed,
            })
        }
    }

    /// Invalid sinks are rejected before any instruction advances. No output
    /// bytes are touched until all instructions have completed successfully.
    pub fn advance_into(
        &mut self,
        budget: usize,
        outputs: &mut [AxOutput<'_>],
    ) -> Result<NativeProgress, Error> {
        if outputs.len() != self.outputs.len() {
            return Err(Error::InvalidInputCount {
                expected: self.outputs.len(),
                actual: outputs.len(),
            });
        }
        for (&id, output) in self.outputs.iter().zip(outputs.iter()) {
            let tensor = self
                .graph
                .tensor(id)
                .map_err(|_| Error::MissingTensor(id))?;
            let expected = tensor_storage_bytes(tensor).ok_or(Error::MissingTensor(id))?;
            if output.len() != expected as u64 {
                return Err(Error::InvalidInputSize {
                    tensor: id,
                    expected,
                    actual: output.len() as usize,
                });
            }
        }
        let progress = self.advance(budget)?;
        if progress != NativeProgress::Ready {
            return Ok(progress);
        }
        let values = self.take_values_in()?;
        for (value, output) in values.iter().zip(outputs) {
            output.write_at(0, value.as_slice())?;
            self.copies.output_materialization_bytes = self
                .copies
                .output_materialization_bytes
                .saturating_add(value.as_slice().len() as u64);
        }
        Ok(NativeProgress::Ready)
    }

    /// Return retained backing to an in-process consumer. This does not grant
    /// another process access or authorize reuse while that consumer reads it.
    pub fn take_retained_in(&mut self) -> Result<Vec<AxTensor<'a, R>, R>, Error> {
        self.take_values_map_in(|dtype, value| Ok(AxTensor { dtype, value }))
    }

    pub(crate) fn take_values_in(&mut self) -> Result<Vec<RuntimeValue<'a, R>, R>, Error> {
        self.take_values_map_in(|_, value| Ok(value))
    }

    // Map ownership wrappers in the one output-metadata allocation. Both the
    // public retained API and the host adapter preserve the original payload.
    pub(crate) fn take_values_map_in<T>(
        &mut self,
        mut map: impl FnMut(DType, RuntimeValue<'a, R>) -> Result<T, Error>,
    ) -> Result<Vec<T, R>, Error> {
        if self.state != State::Ready {
            return Err(Error::InvocationFinished);
        }
        // Reserve before consuming the completed invocation, permitting an
        // allocation failure to leave its outputs available for retry.
        let mut outputs = Vec::new_in(self.allocator.clone());
        reserve(&mut outputs, self.outputs.len())?;
        for &id in self.outputs {
            let slot = self
                .native
                .tensors
                .iter()
                .position(|&v| v == id)
                .ok_or(Error::MissingTensor(id))?;
            if self.values[slot].is_none() {
                return Err(Error::MissingTensor(id));
            }
        }
        self.state = State::Finished;
        for &id in self.outputs {
            let slot = self
                .native
                .tensors
                .iter()
                .position(|&v| v == id)
                .ok_or(Error::MissingTensor(id))?;
            let value = self.values[slot].take().ok_or(Error::MissingTensor(id))?;
            let value = match value {
                NativeSlot::Borrowed(view) => RuntimeValue::Borrowed(view.as_bytes()),
                NativeSlot::Owned(value)
                    if cfg!(target_endian = "little") || matches!(value, NativeValue::U8(_)) =>
                {
                    RuntimeValue::Native(value)
                }
                NativeSlot::Owned(value) => {
                    let bytes = native::encode_native_value_in(value, self.allocator.clone())?;
                    self.copies.native_boundary_encode_bytes = self
                        .copies
                        .native_boundary_encode_bytes
                        .saturating_add(bytes.len() as u64);
                    RuntimeValue::Owned(bytes)
                }
            };
            let dtype = self
                .graph
                .tensor(id)
                .map_err(|_| Error::MissingTensor(id))?
                .dtype;
            outputs.push(map(dtype, value)?);
        }
        Ok(outputs)
    }

    /// Native instructions finish within `advance`; no device can still own
    /// the buffers between calls. Provider cancellation is a separate contract.
    pub fn cancel(&mut self) -> Result<(), Error> {
        if self.state == State::Finished {
            return Err(Error::InvocationFinished);
        }
        self.state = State::Finished;
        self.values.clear();
        Ok(())
    }
}

pub(crate) fn decode_source<'a, A: Allocator + Clone>(
    tensor: TensorView<'_>,
    source: &'a [u8],
    copies: &mut AxHostCopyStats,
    allocator: A,
) -> Result<NativeSlot<'a, A>, Error> {
    let expected = tensor_storage_bytes(tensor).ok_or(Error::MissingTensor(tensor.id))?;
    if source.len() != expected {
        return Err(Error::InvalidInputSize {
            tensor: tensor.id,
            expected,
            actual: source.len(),
        });
    }
    if let Some(view) = NativeRef::from_bytes(tensor.dtype, source) {
        return Ok(NativeSlot::Borrowed(view));
    }
    let mut value = native::allocate_native_value_in(
        tensor.dtype,
        expected / tensor.dtype.block_byte_size() * tensor.dtype.block_size(),
        allocator,
    )?;
    native::decode_prepared_native(&mut value, source);
    copies.native_ingress_bytes = copies
        .native_ingress_bytes
        .saturating_add(source.len() as u64);
    Ok(NativeSlot::Owned(value))
}
