//! Provider adapters and an executable for scheduled `.ax` graphs.
//!
//! Runtime-specific state lives here, above the `ax` crate. The scheduler
//! sees only integer backend IDs and costs; this module lowers each resulting
//! partition to native kernels or the existing TOSA `Accelerator` contract.
//! [`AxHetero::compile_auto_in`] derives capability and ABI policy from the active
//! provider; [`AxHetero::compile_in`] accepts an explicit model. Both enable exact
//! native instruction fusions by default. Executable storage comes from the
//! caller's compile allocator, while transient
//! tensors and outputs use the independent allocator passed to
//! [`AxHetero::run_in`]. This layer has no implicit `Global` policy.

use core::alloc::Allocator;
use core::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use ax::{
    AdmissionError, AdmissionPlanner, AllocationError, CostModel, DENSE_IMPLEMENTATION, DType,
    DispatchCandidate, DispatchDecision, DispatchRequest, Graph, GraphBuf, Implementation,
    NodeView, OpCode, Partition, PlacementReport, Schedule, TensorFlags, TensorId, TensorView,
};
#[cfg(test)]
use axnn::lowering::{KernelAccuracy, SemanticOp, VolumeRgbParams, VoxelBrickTraceParams};
use axnn::lowering::{ResourceKey, VolumeRgbStorage};
use axnn_cpu::CpuKernelPlan;
#[cfg(test)]
use axnn_cpu::elementwise::MulAddOrder;
#[cfg(test)]
use axnn_cpu::{BrickGridRef, CpuCapabilities, RgbVolumeRef};
use virtio_accel_core::{Accelerator, BackendError};
use virtio_accel_tosa::{
    CapabilityDescriptor, DType as TosaDType, Target, TosaCapabilityProvider, ValueRoles,
};

use crate::artifact::{
    self, AxArtifactError, AxBooleanBoundaryPolicy, AxInputTransform, AxLoweringPolicy,
    AxOutputTransform, AxPrecisionPolicy,
};
use crate::lowering::{self, TosaProfile};
use crate::session::{
    FeedbackBinding, Invocation as SessionInvocation, InvocationStatus as SessionInvocationStatus,
    RepeatedExecution, Session,
};

use crate::device_set::{AxDeviceSet, DevicePolicies};
use crate::session_erased::{GraphSession, OutputReader, Outputs};
mod stream;
pub use stream::{AxStream, AxStreamProgress, AxStreamSubmitError};

pub const CPU_BACKEND: u16 = 0;
pub const ACCEL_BACKEND: u16 = 1;
/// CPU implementation ID for a packed F32 block-sparse `MATMUL` whose
/// immutable right-hand operand is compiled into BSR storage.
pub const CPU_BSR_F32_IMPLEMENTATION: Implementation = Implementation(1);
/// CPU implementation ID for a packed B8 (OCP E4M3) block-sparse `MATMUL` with F32
/// accumulation and an F16 graph-visible result.
pub const CPU_BSR_B8_IMPLEMENTATION: Implementation = Implementation(2);
/// CPU implementation ID for a packed F8 (OCP E5M2) block-sparse `MATMUL` with F32
/// accumulation and an F16 graph-visible result.
pub const CPU_BSR_F8_IMPLEMENTATION: Implementation = Implementation(3);
/// CPU implementation route whose concrete scalar/SIMD/ISA kernel remains
/// private to an allocator-owned [`CpuKernelPlan`]. One stable scheduler ID
/// therefore admits every catalogued semantic operation without multiplying
/// implementation variants into the `.ax` format or the placement ABI.
pub use ax_exec::CPU_KERNEL_IMPLEMENTATION;

use ax_exec::native::{
    NativeCpuPartition, NativeValue, PreparedNativeScheduleEntry, allocate_native_value_in,
    encode_native_value_in, run_native_cpu_partition_in, validate_native_resources,
    validate_resource_keys,
};
#[cfg(test)]
use ax_exec::native::{
    NativeResourceRequirement, PreparedMulAddKernel, PreparedNativeInstruction,
    ensure_native_value_in, strict_canonical_f32_binary,
};
pub use ax_exec::{AxCpuKernelPolicy, NativeResourceBinding, NativeResourceKind};

/// Calibratable policy for the native CPU BSR implementation.
///
/// The threshold is expressed in block density rather than element density:
/// one nonzero in a block retains the whole vector tile, which is the cost
/// reality the kernel sees.  Models can disable this path by setting
/// [`AxDeviceModel::sparse_bsr`] to `None`.
#[derive(Clone, Debug)]
pub struct AxSparseBsrPolicy {
    /// Reduction-axis BSR tile width.  `32` aligns with Q8 blocks and the
    /// vector widths used by the portable F8 path.
    pub block_k: usize,
    /// Largest retained-block density admitted to the sparse candidate, in
    /// parts per million.  The conservative default requires at least 60%
    /// zero blocks.
    pub max_block_density_ppm: u32,
    /// Cost-model charge for one retained block's column lookup and loop
    /// control, measured in equivalent multiply-add work units.
    pub block_index_cost: u64,
}

impl Default for AxSparseBsrPolicy {
    fn default() -> Self {
        Self {
            block_k: axnn_cpu::sparse::DEFAULT_BLOCK_K,
            max_block_density_ppm: 400_000,
            block_index_cost: 8,
        }
    }
}

/// A graph output that becomes a graph input for the next resident
/// invocation. Both tensors retain their graph ABI types; provider-specific
/// boundary bridges are validated and kept inside the execution session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxFeedback {
    pub input: TensorId,
    pub output: TensorId,
}

impl AxFeedback {
    pub const fn new(input: TensorId, output: TensorId) -> Self {
        Self { input, output }
    }
}

/// Calibratable cost surface for one host-visible CPU plus an optional
/// accelerator. Cost units are deliberately abstract and comparable only
/// within one placement invocation.
#[derive(Clone, Debug)]
pub struct AxDeviceModel {
    /// Whether the portable CPU lane is eligible for placement. Validation
    /// callers may disable it to prove that every graph node is admitted by
    /// the accelerator rather than merely making acceleration inexpensive.
    pub cpu: bool,
    /// Admit the legacy CPU TOSA provider as implementation zero. Native
    /// `axnn` kernels and sparse CPU implementations remain independently
    /// selectable when this is disabled.
    pub cpu_tosa: bool,
    /// Prefer a directly executable native CPU kernel over a CPU TOSA bridge
    /// when both implement the node. Automatic compilation enables this;
    /// explicit models retain cost-based competition by default. Unsupported
    /// native operations still retain the CPU provider as a coverage fallback.
    pub prefer_native_cpu: bool,
    pub accelerator: bool,
    /// Whether the accelerator can expose Boolean tensors at a compiled
    /// partition boundary. Core ML's NeuralNetwork model I/O cannot, even
    /// though Boolean values are valid inside a model; OpenVINO can when used
    /// with the matching provider release.
    pub accelerator_boolean_boundaries: bool,
    /// Bridge Boolean partition boundaries through lossless FP32 0/1 tensors
    /// while retaining Boolean values inside the provider graph. This is the
    /// Core ML policy; it is unnecessary for providers with native Boolean
    /// model features.
    pub f32_boolean_boundary_bridge: bool,
    /// Allow F8 MATMUL regions on an FP16-only accelerator by widening
    /// boundary inputs and constants. Explicit F8 narrowing nodes remain on
    /// the CPU, preserving graph-visible precision.
    pub fp8_promotion: bool,
    /// Fixed submission and ABI-materialization cost for one dense CPU
    /// provider/TOSA partition. The final runtime should calibrate this in
    /// the same units as kernel costs; zero preserves the legacy policy.
    pub cpu_provider_launch_cost: u64,
    /// Multiplier that converts dense CPU-provider recipe work into the
    /// runtime's common placement-cost unit.
    pub cpu_provider_work_scale: u64,
    /// Fixed decode/materialization/dispatch cost for one native `axnn`
    /// kernel partition. Greedy placement charges it conservatively per node;
    /// aggregate refinement amortizes it once per implementation run.
    pub cpu_kernel_launch_cost: u64,
    /// Multiplier that converts `axnn` catalog estimates into the same common
    /// placement-cost unit used by provider and transfer calibration.
    pub cpu_kernel_work_scale: u64,
    pub accelerator_launch_cost: u64,
    pub accelerator_speedup: u64,
    pub transfer_fixed_cost: u64,
    pub transfer_byte_cost: u64,
    /// Optional native CPU BSR candidate for static F32 or F8 matrix weights.
    pub sparse_bsr: Option<AxSparseBsrPolicy>,
    /// Automatic native kernel lowering on the CPU residency domain. `None`
    /// retains the provider-only path; `Some` lets `axnn` select a concrete
    /// kernel for operations not profitably or faithfully expressed by TOSA.
    pub cpu_kernels: Option<AxCpuKernelPolicy>,
    /// Combine exact native instruction sequences during ordinary compilation,
    /// including native regions inside mixed schedules. Disable for reference
    /// execution or diagnosis; this never enables FMA contraction/reassociation.
    pub native_fusions: bool,
    /// Exact semantic admission descriptor advertised by the active TOSA
    /// provider. When present, placement intersects axiom's lowering support
    /// with the provider's operator and dtype roles before attempting program
    /// admission.
    pub accelerator_tosa_capability: Option<CapabilityDescriptor>,
}

impl Default for AxDeviceModel {
    fn default() -> Self {
        Self {
            cpu: true,
            cpu_tosa: true,
            prefer_native_cpu: false,
            accelerator: true,
            accelerator_boolean_boundaries: true,
            f32_boolean_boundary_bridge: false,
            fp8_promotion: false,
            cpu_provider_launch_cost: 0,
            cpu_provider_work_scale: 1,
            cpu_kernel_launch_cost: 0,
            cpu_kernel_work_scale: 1,
            accelerator_launch_cost: 250_000,
            accelerator_speedup: 12,
            transfer_fixed_cost: 50_000,
            transfer_byte_cost: 4,
            sparse_bsr: Some(AxSparseBsrPolicy::default()),
            cpu_kernels: Some(AxCpuKernelPolicy::default()),
            native_fusions: true,
            accelerator_tosa_capability: None,
        }
    }
}

impl AxDeviceModel {
    /// Core ML floating-point profile. Core ML cannot expose Boolean model
    /// features, so partition boundaries use exact FP32 0/1 bridges while the
    /// internal comparison, logical, and select operations remain Boolean.
    pub fn coreml() -> Self {
        Self {
            accelerator_boolean_boundaries: false,
            f32_boolean_boundary_bridge: true,
            ..Self::default()
        }
    }

    /// Attach the exact TOSA descriptor advertised for `target` by a provider.
    ///
    /// This is deliberately provider-reported rather than a backend-name
    /// switch: new `Accelerator` implementations can participate in placement
    /// by publishing the shared TOSA capability contract.
    pub fn with_accelerator_tosa_capability<P: TosaCapabilityProvider>(
        mut self,
        provider: &P,
        target: Target,
    ) -> Self {
        self.accelerator_tosa_capability = provider
            .tosa_capabilities()
            .iter()
            .copied()
            .find(|descriptor| descriptor.target == target);
        self.accelerator &= self.accelerator_tosa_capability.is_some();
        self
    }

    fn node_compute_cost(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u64 {
        let profile = if backend == ACCEL_BACKEND {
            TosaProfile::Accelerator
        } else {
            TosaProfile::Cpu
        };
        let recipe_ops = lowering::recipe_for_node(graph, node, profile)
            .map(|recipe| recipe.operator_count())
            .unwrap_or(1);
        let work = node_work(graph, node)
            .unwrap_or(u64::MAX / 2)
            .saturating_mul(recipe_ops);
        match backend {
            CPU_BACKEND => work.saturating_mul(self.cpu_provider_work_scale),
            ACCEL_BACKEND => work.div_ceil(self.accelerator_speedup.max(1)),
            _ => u64::MAX,
        }
    }

    fn cpu_tosa_admitted(&self, graph: &Graph<'_>, node: NodeView<'_>) -> bool {
        self.cpu_tosa
            && cpu_tosa_supports(graph, node)
            && (!self.prefer_native_cpu || self.cpu_kernel_plan(graph, node).is_none())
    }

    fn cpu_kernel_plan(&self, graph: &Graph<'_>, node: NodeView<'_>) -> Option<CpuKernelPlan> {
        self.cpu_kernels?
            .plan(&ax::node_kernel_request(graph, node)?)
    }

    fn cpu_kernel_cost(&self, graph: &Graph<'_>, node: NodeView<'_>) -> Option<u64> {
        self.cpu_kernel_plan(graph, node).map(|plan| {
            plan.estimated_cost
                .saturating_mul(self.cpu_kernel_work_scale)
        })
    }

    fn sparse_bsr_cost(
        &self,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> Option<u64> {
        let policy = self.sparse_bsr.as_ref()?;
        let info = sparse_bsr_matmul_info(graph, node)?;
        if info.precision.implementation() != implementation {
            return None;
        }
        let profile = info.precision.profile(&info, policy.block_k).ok()?;
        if profile.block_density_ppm() > policy.max_block_density_ppm {
            return None;
        }
        // `node_work` prices a dense MAC as a multiply plus an add. Keep
        // that unit when comparing variants so block density alone cannot
        // make an otherwise dense-faster kernel look artificially cheap.
        let per_activation = (profile.stored_values as u64)
            .saturating_mul(2)
            .saturating_add((profile.nonzero_blocks as u64).saturating_mul(policy.block_index_cost))
            .saturating_add(info.output_features as u64);
        Some(per_activation.saturating_mul(info.activation_rows as u64))
    }
}

struct AxPlacementModel<'a, A: Allocator> {
    model: &'a dyn DevicePolicies,
    targets: &'a [BooleanPlacement<A>],
}

impl<A: Allocator> CostModel for AxPlacementModel<'_, A> {
    fn backend_count(&self) -> u16 {
        self.model.backend_count()
    }

    fn supports(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> bool {
        if backend != CPU_BACKEND
            && self
                .targets
                .get(backend as usize - 1)
                .and_then(|target| target.accelerator_forbidden.get(node.index))
                .is_some_and(|blocked| *blocked != 0)
        {
            return false;
        }
        self.model.supports(backend, graph, node)
    }

    fn execute_cost(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u64 {
        self.model.execute_cost(backend, graph, node)
    }

    fn transfer_cost(&self, tensor: TensorView<'_>, source: Option<u16>, destination: u16) -> u64 {
        self.model.transfer_cost(tensor, source, destination)
    }

    fn partition_cost(&self, backend: u16) -> u64 {
        self.model.partition_cost(backend)
    }

    fn partition_cost_implementation(&self, backend: u16, implementation: Implementation) -> u64 {
        self.model
            .partition_cost_implementation(backend, implementation)
    }

    fn external_output_backend(&self) -> Option<u16> {
        self.model.external_output_backend()
    }

    fn implementation_count(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u16 {
        self.model.implementation_count(backend, graph, node)
    }

    fn supports_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> bool {
        if backend != CPU_BACKEND
            && self
                .targets
                .get(backend as usize - 1)
                .and_then(|target| target.accelerator_forbidden.get(node.index))
                .is_some_and(|blocked| *blocked != 0)
        {
            return false;
        }
        self.model
            .supports_implementation(backend, graph, node, implementation)
    }

    fn execute_cost_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> u64 {
        self.model
            .execute_cost_implementation(backend, graph, node, implementation)
    }

    fn region_execute_cost_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> u64 {
        self.model
            .region_execute_cost_implementation(backend, graph, node, implementation)
    }
}

impl CostModel for AxDeviceModel {
    fn backend_count(&self) -> u16 {
        if self.accelerator { 2 } else { 1 }
    }

    fn supports(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> bool {
        if !node.attributes().is_empty() {
            return backend == CPU_BACKEND
                && self.cpu
                && self.cpu_kernel_plan(graph, node).is_some();
        }
        match backend {
            CPU_BACKEND if self.cpu => {
                self.cpu_tosa_admitted(graph, node) || self.cpu_kernel_plan(graph, node).is_some()
            }
            ACCEL_BACKEND if self.accelerator => accel_supports(graph, node, self),
            _ => false,
        }
    }

    fn execute_cost(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u64 {
        match backend {
            CPU_BACKEND => self
                .cpu_provider_launch_cost
                .saturating_add(self.node_compute_cost(backend, graph, node)),
            ACCEL_BACKEND => self
                .accelerator_launch_cost
                .saturating_add(self.node_compute_cost(backend, graph, node)),
            _ => u64::MAX,
        }
    }

    fn transfer_cost(&self, tensor: TensorView<'_>, source: Option<u16>, destination: u16) -> u64 {
        if destination == CPU_BACKEND && source.is_none() {
            return 0;
        }
        self.transfer_storage_bytes(tensor, source, destination)
            .map_or(u64::MAX / 2, |bytes| {
                self.transfer_fixed_cost
                    .saturating_add((bytes as u64).saturating_mul(self.transfer_byte_cost))
            })
    }

    fn partition_cost(&self, backend: u16) -> u64 {
        if backend == ACCEL_BACKEND {
            self.accelerator_launch_cost
        } else {
            0
        }
    }

    fn partition_cost_implementation(&self, backend: u16, implementation: Implementation) -> u64 {
        match (backend, implementation) {
            (CPU_BACKEND, DENSE_IMPLEMENTATION) => self.cpu_provider_launch_cost,
            (CPU_BACKEND, CPU_KERNEL_IMPLEMENTATION) => self.cpu_kernel_launch_cost,
            (ACCEL_BACKEND, DENSE_IMPLEMENTATION) => self.accelerator_launch_cost,
            _ => 0,
        }
    }

    fn external_output_backend(&self) -> Option<u16> {
        Some(CPU_BACKEND)
    }

    fn implementation_count(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u16 {
        AxDeviceModel::implementation_count(self, backend, graph, node)
    }

    fn supports_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> bool {
        AxDeviceModel::supports_implementation(self, backend, graph, node, implementation)
    }

    fn execute_cost_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> u64 {
        AxDeviceModel::execute_cost_implementation(self, backend, graph, node, implementation)
    }

    fn region_execute_cost_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> u64 {
        AxDeviceModel::region_execute_cost_implementation(
            self,
            backend,
            graph,
            node,
            implementation,
        )
    }
}

impl AxDeviceModel {
    fn transfer_storage_bytes(
        &self,
        tensor: TensorView<'_>,
        source: Option<u16>,
        destination: u16,
    ) -> Option<usize> {
        let bytes = tensor_storage_bytes(tensor)?;
        let crosses_accelerator = source == Some(ACCEL_BACKEND) || destination == ACCEL_BACKEND;
        if self.f32_boolean_boundary_bridge
            && crosses_accelerator
            && tensor.dtype == DType::U8
            && !tensor.flags.contains(TensorFlags::CONSTANT)
        {
            bytes.checked_mul(4)
        } else {
            Some(bytes)
        }
    }

    fn implementation_count(&self, backend: u16, _graph: &Graph<'_>, _node: NodeView<'_>) -> u16 {
        if backend != CPU_BACKEND {
            return 1;
        }
        if self.cpu_kernels.is_some() {
            CPU_KERNEL_IMPLEMENTATION.0 + 1
        } else if self.sparse_bsr.is_some() {
            CPU_BSR_F8_IMPLEMENTATION.0 + 1
        } else {
            1
        }
    }

    fn supports_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> bool {
        if !node.attributes().is_empty() {
            return backend == CPU_BACKEND
                && implementation == CPU_KERNEL_IMPLEMENTATION
                && self.cpu
                && self.cpu_kernel_plan(graph, node).is_some();
        }
        match implementation {
            DENSE_IMPLEMENTATION => match backend {
                CPU_BACKEND => self.cpu && self.cpu_tosa_admitted(graph, node),
                ACCEL_BACKEND => self.accelerator && accel_supports(graph, node, self),
                _ => false,
            },
            CPU_BSR_F32_IMPLEMENTATION | CPU_BSR_B8_IMPLEMENTATION | CPU_BSR_F8_IMPLEMENTATION => {
                backend == CPU_BACKEND
                    && self.cpu
                    && self.sparse_bsr_cost(graph, node, implementation).is_some()
            }
            CPU_KERNEL_IMPLEMENTATION => {
                backend == CPU_BACKEND && self.cpu && self.cpu_kernel_plan(graph, node).is_some()
            }
            _ => false,
        }
    }

    fn execute_cost_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> u64 {
        match implementation {
            DENSE_IMPLEMENTATION => self.execute_cost(backend, graph, node),
            CPU_BSR_F32_IMPLEMENTATION | CPU_BSR_B8_IMPLEMENTATION | CPU_BSR_F8_IMPLEMENTATION
                if backend == CPU_BACKEND =>
            {
                self.sparse_bsr_cost(graph, node, implementation)
                    .unwrap_or(u64::MAX)
            }
            CPU_KERNEL_IMPLEMENTATION if backend == CPU_BACKEND => {
                self.cpu_kernel_cost(graph, node).map_or(u64::MAX, |cost| {
                    self.cpu_kernel_launch_cost.saturating_add(cost)
                })
            }
            _ => u64::MAX,
        }
    }

    fn region_execute_cost_implementation(
        &self,
        backend: u16,
        graph: &Graph<'_>,
        node: NodeView<'_>,
        implementation: Implementation,
    ) -> u64 {
        match implementation {
            DENSE_IMPLEMENTATION => self.node_compute_cost(backend, graph, node),
            CPU_BSR_F32_IMPLEMENTATION | CPU_BSR_B8_IMPLEMENTATION | CPU_BSR_F8_IMPLEMENTATION
                if backend == CPU_BACKEND =>
            {
                self.sparse_bsr_cost(graph, node, implementation)
                    .unwrap_or(u64::MAX)
            }
            CPU_KERNEL_IMPLEMENTATION if backend == CPU_BACKEND => {
                self.cpu_kernel_cost(graph, node).unwrap_or(u64::MAX)
            }
            _ => u64::MAX,
        }
    }
}

#[derive(Debug)]
pub enum AxCompileError {
    Ax(AllocationError),
    Artifact(AxArtifactError),
    Backend(BackendError),
    MissingAccelerator,
    WrongExecutor,
    ResidentRequiresSinglePartition,
    NativeRequiresSinglePartition,
    InvalidRepetitionCount,
    InvalidFeedback {
        input: TensorId,
        output: TensorId,
    },
    InvalidInputCount {
        expected: usize,
        actual: usize,
    },
    InvalidInputSize {
        tensor: TensorId,
        expected: usize,
        actual: usize,
    },
    InvalidBooleanBoundary {
        tensor: TensorId,
    },
    MissingResource(ResourceKey),
    /// Runtime-owned state has no CPU mirror and needs a resident executor.
    ResidentResourceRequiresExecutor(ResourceKey),
    DuplicateResource(ResourceKey),
    ResourceKind {
        resource: ResourceKey,
        expected: NativeResourceKind,
        actual: NativeResourceKind,
    },
    ResourceDimensions {
        resource: ResourceKey,
        expected: [u32; 3],
        actual: [u32; 3],
    },
    ResourceStorage {
        resource: ResourceKey,
        expected: VolumeRgbStorage,
        actual: VolumeRgbStorage,
    },
    ResourceLength {
        resource: ResourceKey,
        expected: usize,
        actual: usize,
    },
    KvCacheGeometry {
        resource: ResourceKey,
        expected: axnn::lowering::KvCacheParams,
        actual: axnn::lowering::KvCacheParams,
    },
    MissingTensor(TensorId),
    UnsupportedImplementation {
        backend: u16,
        implementation: Implementation,
    },
    InvalidSparsePartition,
    Cpu(axnn_cpu::CpuError),
    Sparse(axnn_cpu::sparse::SparseError),
    InvocationFinished,
}

impl From<AllocationError> for AxCompileError {
    fn from(value: AllocationError) -> Self {
        Self::Ax(value)
    }
}

impl From<AxArtifactError> for AxCompileError {
    fn from(value: AxArtifactError) -> Self {
        Self::Artifact(value)
    }
}

impl From<BackendError> for AxCompileError {
    fn from(value: BackendError) -> Self {
        Self::Backend(value)
    }
}

impl From<axnn_cpu::sparse::SparseError> for AxCompileError {
    fn from(value: axnn_cpu::sparse::SparseError) -> Self {
        Self::Sparse(value)
    }
}

impl From<axnn_cpu::CpuError> for AxCompileError {
    fn from(value: axnn_cpu::CpuError) -> Self {
        Self::Cpu(value)
    }
}

#[derive(Debug)]
struct ExecutablePartition<A: Allocator> {
    backend: u16,
    implementation: Implementation,
    key: u64,
    inputs: Vec<TensorId, A>,
    input_transforms: Vec<AxInputTransform, A>,
    outputs: Vec<TensorId, A>,
    output_transforms: Vec<AxOutputTransform, A>,
    output_bytes: Vec<usize, A>,
    kind: ExecutablePartitionKind<A>,
}

#[derive(Debug)]
enum ExecutablePartitionKind<A: Allocator> {
    Provider,
    NativeCpu(NativeCpuPartition<A>),
    SparseF32(Vec<PackedSparseF32<A>, A>),
    SparseFp8(Vec<PackedSparseFp8<A>, A>),
}

impl From<ax_exec::Error> for AxCompileError {
    fn from(error: ax_exec::Error) -> Self {
        use ax_exec::{Error as E, PlanError as P};
        match error {
            E::Ax(e) => Self::Ax(e),
            E::Plan(e) => Self::Artifact(match e {
                P::InvalidGraph => AxArtifactError::InvalidGraph,
                P::InvalidPartition => AxArtifactError::InvalidPartition,
                P::NoPartitionOutputs => AxArtifactError::NoPartitionOutputs,
            }),
            E::Backend(e) => Self::Backend(e),
            E::NativeRequiresSinglePartition => Self::NativeRequiresSinglePartition,
            E::InvalidInputCount { expected, actual } => {
                Self::InvalidInputCount { expected, actual }
            }
            E::InvalidInputSize {
                tensor,
                expected,
                actual,
            } => Self::InvalidInputSize {
                tensor,
                expected,
                actual,
            },
            E::MissingResource(key) => Self::MissingResource(key),
            E::ResidentResourceRequiresExecutor(key) => {
                Self::ResidentResourceRequiresExecutor(key)
            }
            E::DuplicateResource(key) => Self::DuplicateResource(key),
            E::ResourceKind {
                resource,
                expected,
                actual,
            } => Self::ResourceKind {
                resource,
                expected,
                actual,
            },
            E::ResourceDimensions {
                resource,
                expected,
                actual,
            } => Self::ResourceDimensions {
                resource,
                expected,
                actual,
            },
            E::ResourceStorage {
                resource,
                expected,
                actual,
            } => Self::ResourceStorage {
                resource,
                expected,
                actual,
            },
            E::ResourceLength {
                resource,
                expected,
                actual,
            } => Self::ResourceLength {
                resource,
                expected,
                actual,
            },
            E::KvCacheGeometry {
                resource,
                expected,
                actual,
            } => Self::KvCacheGeometry {
                resource,
                expected,
                actual,
            },
            E::MissingTensor(id) => Self::MissingTensor(id),
            E::UnsupportedImplementation {
                backend,
                implementation,
            } => Self::UnsupportedImplementation {
                backend,
                implementation,
            },
            E::Cpu(e) => Self::Cpu(e),
            E::InvocationFinished => Self::InvocationFinished,
            E::InvalidStepBudget => Self::Backend(BackendError::InvalidArgument),
        }
    }
}

fn validate_executable_resources<A: Allocator>(
    executable: &AxExecutable<A>,
    resources: &[NativeResourceBinding<'_>],
) -> Result<(), AxCompileError> {
    validate_resource_keys(resources)?;
    for partition in &executable.partitions {
        if let ExecutablePartitionKind::NativeCpu(native) = &partition.kind {
            validate_native_resources(&native.nodes, resources)?;
        }
    }
    Ok(())
}

/// One compiled BSR constant.  It belongs to the executable's persistent
/// allocator, so graph runs never rescan or repack the dense weight bytes.
#[derive(Debug)]
struct PackedSparseF32<A: Allocator> {
    input: TensorId,
    output: TensorId,
    input_constant: Option<Vec<u8, A>>,
    output_features: usize,
    input_features: usize,
    activation_rows: usize,
    block_k: usize,
    row_offsets: Vec<u32, A>,
    column_indices: Vec<u32, A>,
    values: Vec<f32, A>,
}

/// One compiled F8 BSR constant.  Both operands stay in their original F8
/// byte planes until the architecture-specific dot kernel widens a block.
#[derive(Debug)]
struct PackedSparseFp8<A: Allocator> {
    input: TensorId,
    output: TensorId,
    input_constant: Option<Vec<u8, A>>,
    precision: SparseBsrPrecision,
    output_features: usize,
    input_features: usize,
    activation_rows: usize,
    block_k: usize,
    row_offsets: Vec<u32, A>,
    column_indices: Vec<u32, A>,
    values: Vec<u8, A>,
}

impl<A: Allocator> ExecutablePartition<A> {
    fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }
}

mod storage;
use storage::RuntimeValue;
pub use storage::{AxOutput, AxTensor};
#[cfg(test)]
use storage::{NativeRef, NativeSlot};

/// Reusable native arena backed by the portable ax-exec implementation.
/// Retains the established host error and output API.
pub struct PreparedNative<A: Allocator> {
    inner: ax_exec::native::PreparedNative<A>,
}
impl<A: Allocator> PreparedNative<A> {
    pub fn run(&mut self, inputs: &[&[u8]]) -> Result<(), AxCompileError> {
        self.inner.run(inputs).map_err(Into::into)
    }
    pub fn run_with_resources(
        &mut self,
        inputs: &[&[u8]],
        resources: &[NativeResourceBinding<'_>],
    ) -> Result<(), AxCompileError> {
        self.inner
            .run_with_resources(inputs, resources)
            .map_err(Into::into)
    }
    pub fn output_f32(&self, index: usize) -> Option<&[f32]> {
        self.inner.output_f32(index)
    }
    pub fn output_i32(&self, index: usize) -> Option<&[i32]> {
        self.inner.output_i32(index)
    }
    pub fn output_u8(&self, index: usize) -> Option<&[u8]> {
        self.inner.output_u8(index)
    }
    pub fn input_count(&self) -> usize {
        self.inner.input_count()
    }
    pub fn output_count(&self) -> usize {
        self.inner.output_count()
    }
    pub fn instruction_count(&self) -> usize {
        self.inner.instruction_count()
    }
    pub fn mul_add_fusion_count(&self) -> usize {
        self.inner.mul_add_fusion_count()
    }
    pub fn simd_mul_add_fusion_count(&self) -> usize {
        self.inner.simd_mul_add_fusion_count()
    }
    pub fn scale_interleave3_fusion_count(&self) -> usize {
        self.inner.scale_interleave3_fusion_count()
    }
    pub fn simd_scale_interleave3_fusion_count(&self) -> usize {
        self.inner.simd_scale_interleave3_fusion_count()
    }
    pub fn cast_shift_or3_fusion_count(&self) -> usize {
        self.inner.cast_shift_or3_fusion_count()
    }
    pub fn buffer_count(&self) -> usize {
        self.inner.buffer_count()
    }
    pub fn buffer_bytes(&self) -> usize {
        self.inner.buffer_bytes()
    }
    pub fn logical_buffer_count(&self) -> usize {
        self.inner.logical_buffer_count()
    }
    pub fn logical_buffer_bytes(&self) -> usize {
        self.inner.logical_buffer_bytes()
    }
    pub fn scratch_bytes(&self) -> usize {
        self.inner.scratch_bytes()
    }
}
// Existing adversarial unit tests inspect/corrupt the same shared arena.
#[cfg(test)]
impl<A: Allocator> core::ops::Deref for PreparedNative<A> {
    type Target = ax_exec::native::PreparedNative<A>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
#[cfg(test)]
impl<A: Allocator> core::ops::DerefMut for PreparedNative<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Observable phase of one placed partition in a caller-driven graph run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxPartitionPhase {
    /// Provider admission returned an event. `submit_error` is populated when
    /// acceptance was indeterminate, but the event still owns the invocation.
    Submitted,
    /// A bounded provider poll observed that execution is still pending.
    Waiting,
    /// One partition reached a terminal state and its outputs were committed
    /// to the graph's transient value table.
    Complete,
}

/// Progress observation returned after one bounded graph-invocation step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxPartitionProgress {
    pub partition: usize,
    pub backend: u16,
    pub phase: AxPartitionPhase,
    /// Provider error accompanying an indeterminate submission. The live
    /// event remains session-owned and must still be polled.
    pub submit_error: Option<BackendError>,
}

/// Result of one nonblocking graph-invocation step.
#[derive(Debug)]
#[must_use = "graph invocation progress must be observed"]
pub enum AxInvocationStatus<T> {
    Progress(AxPartitionProgress),
    Complete(T),
    Cancelled(AxPartitionProgress),
}

/// Copy payloads caused by native host representation or output ownership.
/// Excludes provider transfers, semantic kernels (including IDENTITY), sparse
/// packing, precision bridges, constant-output snapshots, and `begin_owned_in`'s
/// explicit input snapshot. Later `AxTensor::into_bytes_in` calls are also outside
/// these invocation counters.
/// These are software byte counts, not memory-controller traffic measurements.
pub use ax_exec::AxHostCopyStats;

struct PendingPartition {
    partition: usize,
    backend: u16,
    invocation: SessionInvocation,
}

/// Caller-owned progress state for one scheduled graph invocation.
///
/// Graph input planes remain borrowed, while partition outputs and boundary
/// transforms use `R`. A live value must keep being passed to
/// [`AxHetero::advance_in`] until it completes or is cancelled. Dropping a
/// state with a pending partition abandons only the caller token: the session
/// still retains the event and buffers, and that program key remains busy
/// until session teardown.
#[must_use = "a live graph invocation must be advanced to a terminal state"]
pub struct AxInvocation<'executable, 'input, M: Allocator, R: Allocator + Clone> {
    owner: u64,
    executable: &'executable AxExecutable<M>,
    values: Vec<Option<RuntimeValue<'input, R>>, R>,
    resources: &'input [NativeResourceBinding<'input>],
    allocator: R,
    next_partition: usize,
    host_copies: AxHostCopyStats,
    pending: Option<PendingPartition>,
    finished: bool,
}

impl<M: Allocator, R: Allocator + Clone> AxInvocation<'_, '_, M, R> {
    pub fn host_copy_stats(&self) -> AxHostCopyStats {
        self.host_copies
    }

    /// Index of the partition that is pending or will be submitted next.
    pub fn partition_index(&self) -> usize {
        self.pending
            .as_ref()
            .map(|pending| pending.partition)
            .unwrap_or(self.next_partition)
    }

    pub fn pending_backend(&self) -> Option<u16> {
        self.pending.as_ref().map(|pending| pending.backend)
    }

    pub const fn is_finished(&self) -> bool {
        self.finished
    }
}

fn own_byte_outputs<R: Allocator + Clone>(
    outputs: Vec<Vec<u8, R>, R>,
) -> Result<Vec<RuntimeValue<'static, R>, R>, AxCompileError> {
    let mut values = Vec::new_in(outputs.allocator().clone());
    reserve(&mut values, outputs.len())?;
    for output in outputs {
        values.push(RuntimeValue::Owned(output));
    }
    Ok(values)
}

fn read_partition_outputs_in<P: Accelerator, M: Allocator, R: Allocator + Clone>(
    provider: &P,
    buffers: &[P::Buffer],
    sizes: &[usize],
    graph: &Graph<'_>,
    partition: &ExecutablePartition<M>,
    allocator: R,
) -> Result<Vec<RuntimeValue<'static, R>, R>, BackendError> {
    if buffers.len() != sizes.len() {
        return Err(BackendError::InvalidArgument);
    }
    read_partition_outputs_from_in(
        &Outputs {
            provider,
            buffers,
            sizes,
        },
        graph,
        partition,
        allocator,
    )
}
fn read_partition_outputs_from_in<M: Allocator, R: Allocator + Clone>(
    reader: &dyn OutputReader,
    graph: &Graph<'_>,
    partition: &ExecutablePartition<M>,
    allocator: R,
) -> Result<Vec<RuntimeValue<'static, R>, R>, BackendError> {
    let sizes = reader.sizes();
    if sizes.len() != partition.outputs.len() {
        return Err(BackendError::InvalidArgument);
    }
    let mut outputs = Vec::new_in(allocator.clone());
    outputs
        .try_reserve_exact(sizes.len())
        .map_err(|_| BackendError::OutOfMemory)?;
    for (index, &size) in sizes.iter().enumerate() {
        let dtype = graph
            .tensor(partition.outputs[index])
            .map_err(|_| BackendError::InvalidArgument)?
            .dtype;
        let mut value = if cfg!(target_endian = "little")
            && partition.output_transforms[index] == AxOutputTransform::Identity
            && matches!(dtype, DType::F32 | DType::I32)
            && size.is_multiple_of(4)
        {
            RuntimeValue::Native(
                allocate_native_value_in(dtype, size / 4, allocator.clone())
                    .map_err(|_| BackendError::OutOfMemory)?,
            )
        } else {
            let mut bytes = Vec::new_in(allocator.clone());
            bytes
                .try_reserve_exact(size)
                .map_err(|_| BackendError::OutOfMemory)?;
            bytes.resize(size, 0);
            RuntimeValue::Owned(bytes)
        };
        let destination = match &mut value {
            RuntimeValue::Native(value) => value.as_native_bytes_mut(),
            RuntimeValue::Owned(value) => value.as_mut_slice(),
            RuntimeValue::Borrowed(_) => unreachable!(),
        };
        reader.read(index, &mut AxOutput::packed(destination))?;
        outputs.push(value);
    }
    Ok(outputs)
}

fn commit_partition_outputs<'a, M: Allocator, R: Allocator + Clone>(
    partition: &ExecutablePartition<M>,
    outputs: Vec<RuntimeValue<'a, R>, R>,
    values: &mut [Option<RuntimeValue<'a, R>>],
) -> Result<(), AxCompileError> {
    if partition.outputs.len() != partition.output_transforms.len()
        || partition.outputs.len() != outputs.len()
    {
        return Err(AxArtifactError::InvalidPartition.into());
    }
    for ((&tensor, &transform), bytes) in partition
        .outputs
        .iter()
        .zip(&partition.output_transforms)
        .zip(outputs)
    {
        let value = if transform == AxOutputTransform::Identity {
            bytes
        } else {
            let RuntimeValue::Owned(bytes) = bytes else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            RuntimeValue::Owned(transform_output_in(transform, tensor, bytes)?)
        };
        values[tensor.0 as usize] = Some(value);
    }
    Ok(())
}

fn collect_graph_outputs_in<'a, M: Allocator, R: Allocator + Clone>(
    executable: &AxExecutable<M>,
    values: &mut [Option<RuntimeValue<'a, R>>],
    allocator: R,
) -> Result<Vec<AxTensor<'a, R>, R>, AxCompileError> {
    let graph = executable.graph();
    let mut outputs = Vec::new_in(allocator.clone());
    reserve(&mut outputs, executable.outputs.len())?;
    for &id in &executable.outputs {
        if let Some(value) = values[id.0 as usize].take() {
            outputs.push(AxTensor { value });
            continue;
        }
        let tensor = graph
            .tensor(id)
            .map_err(|_| AxCompileError::MissingTensor(id))?;
        if !tensor.flags.contains(TensorFlags::CONSTANT) {
            return Err(AxCompileError::MissingTensor(id));
        }
        outputs.push(AxTensor {
            value: RuntimeValue::Owned(copy_bytes_in(tensor.constant(), allocator.clone())?),
        });
    }
    Ok(outputs)
}

pub use ax_exec::AxNativePlanStats;

/// An analyzed, placed, provider-compiled graph tied to the `AxHetero`
/// instance that owns its provider programs.
pub struct AxExecutable<A: Allocator> {
    owner: u64,
    graph: GraphBuf<A>,
    schedule: Schedule<A>,
    placement_report: PlacementReport,
    /// Indexed tensor slots that can cross a partition or graph boundary.
    /// Partition-internal temporaries are provider-owned and need no host
    /// entry during execution.
    value_slots: usize,
    inputs: Vec<TensorId, A>,
    outputs: Vec<TensorId, A>,
    partitions: Vec<ExecutablePartition<A>, A>,
}

impl<A: Allocator> AxExecutable<A> {
    /// Selected native kernels in partition and instruction order. Combine
    /// implementation names, accuracy and scratch bytes with the placement
    /// report when explaining a decision or recording calibrated timings.
    pub fn kernel_plans(&self) -> impl Iterator<Item = &CpuKernelPlan> {
        self.partitions
            .iter()
            .filter_map(|p| match &p.kind {
                ExecutablePartitionKind::NativeCpu(native) => Some(native),
                _ => None,
            })
            .flat_map(|native| native.nodes.iter().map(|node| &node.plan))
    }

    pub fn native_plan_stats(&self) -> AxNativePlanStats {
        let mut stats = AxNativePlanStats::default();
        for partition in &self.partitions {
            if let ExecutablePartitionKind::NativeCpu(native) = &partition.kind {
                stats.logical_nodes += native.nodes.len();
                for entry in &native.instructions {
                    match entry {
                        PreparedNativeScheduleEntry::Elided => continue,
                        PreparedNativeScheduleEntry::Kernel(_) => {}
                        PreparedNativeScheduleEntry::MulAdd(_) => stats.mul_add_fusions += 1,
                        PreparedNativeScheduleEntry::SubMulAdd(_) => stats.sub_mul_add_fusions += 1,
                        PreparedNativeScheduleEntry::ScaleInterleave3(_) => {
                            stats.scale_interleave3_fusions += 1
                        }
                        PreparedNativeScheduleEntry::CastShiftOr3(_) => {
                            stats.cast_shift_or3_fusions += 1
                        }
                    }
                    stats.instructions += 1;
                }
            }
        }
        stats
    }

    pub fn graph(&self) -> Graph<'_> {
        self.graph.parse().expect("owned graph was validated")
    }

    pub fn schedule(&self) -> &Schedule<A> {
        &self.schedule
    }

    /// Complete placed-graph cost under the model used at compilation.
    ///
    /// The value includes aggregate partition launches, compute, unique
    /// transfers, and output egress. It is not wall time until the adapter
    /// calibrates that model's units, but it provides the structural seed for
    /// an [`ax::AdmissionPlanner`] dispatch duration.
    pub const fn aggregate_cost(&self) -> u64 {
        self.placement_report.aggregate_cost
    }

    /// Decisions made by the core placement optimizer before provider
    /// compilation. This makes launch-amortization and clustering observable
    /// without exposing adapter-specific scheduling internals.
    pub const fn placement_report(&self) -> PlacementReport {
        self.placement_report
    }

    pub fn input_ids(&self) -> &[TensorId] {
        &self.inputs
    }

    pub fn output_ids(&self) -> &[TensorId] {
        &self.outputs
    }

    /// Prepare reusable typed storage for exactly one native CPU partition.
    ///
    /// Admits F32/U8 elementwise operations, exact I32 identity/comparison,
    /// F32 MATMUL, component interleaving and typed voxel queries. Provider,
    /// sparse, multi-partition, ordered and mutating
    /// graphs are rejected explicitly. The returned object does not borrow
    /// this executable; all retained storage belongs to `allocator`.
    pub fn prepare_native_in<R: Allocator + Clone>(
        &self,
        allocator: R,
    ) -> Result<PreparedNative<R>, AxCompileError> {
        prepare_native_in(self, allocator)
    }

    /// Backend lane that owns every placed partition, or `None` when the
    /// executable is empty or requires a compound cross-backend reservation.
    pub fn single_backend(&self) -> Option<u16> {
        let partitions = self.schedule.plan().partitions;
        let backend = partitions.first()?.backend;
        partitions
            .iter()
            .all(|partition| partition.backend == backend)
            .then_some(backend)
    }
}

/// A calibrated single-lane dispatch candidate tied to its placed executable.
///
/// Construction derives the backend from the final partition plan, preventing
/// a caller from reserving an accelerator lane and then invoking CPU work (or
/// vice versa). Mixed-backend executables deliberately return `None`: they
/// need a future compound reservation rather than a misleading single lane.
#[derive(Clone, Copy)]
pub struct AxDispatchVariant<'a, A: Allocator> {
    executable: &'a AxExecutable<A>,
    candidate: DispatchCandidate,
}

impl<'a, A: Allocator> AxDispatchVariant<'a, A> {
    pub fn new(
        executable: &'a AxExecutable<A>,
        variant: u16,
        duration: u64,
        fidelity: u16,
    ) -> Option<Self> {
        Some(Self {
            executable,
            candidate: DispatchCandidate::new(
                variant,
                executable.single_backend()?,
                duration,
                fidelity,
            ),
        })
    }

    #[inline]
    pub const fn candidate(&self) -> DispatchCandidate {
        self.candidate
    }

    #[inline]
    pub const fn executable(&self) -> &'a AxExecutable<A> {
        self.executable
    }
}

/// An admitted capacity reservation bound to the exact placed executable.
///
/// This prevents a host adapter from selecting one candidate and accidentally
/// invoking a differently placed executable after indexing its own parallel
/// table. The choice borrows only the executable, so its temporary variant
/// array may remain on the caller's stack.
#[derive(Clone, Copy)]
pub struct AxDispatchChoice<'a, A: Allocator> {
    decision: DispatchDecision,
    executable: &'a AxExecutable<A>,
}

impl<'a, A: Allocator> AxDispatchChoice<'a, A> {
    #[inline]
    pub const fn decision(&self) -> DispatchDecision {
        self.decision
    }

    #[inline]
    pub const fn executable(&self) -> &'a AxExecutable<A> {
        self.executable
    }
}

/// Admit one of a fixed set of placed executable variants without allocating.
///
/// The caller owns the clock, lane snapshot, and provider events. Completion
/// should be returned through [`AdmissionPlanner::reconcile`]. Mixed-backend
/// executables cannot enter this single-lane set because
/// [`AxDispatchVariant::new`] rejects them until compound reservations exist.
pub fn admit_variants<'a, A: Allocator, const N: usize>(
    planner: &mut AdmissionPlanner<'_>,
    request: DispatchRequest,
    variants: &[AxDispatchVariant<'a, A>; N],
) -> Result<AxDispatchChoice<'a, A>, AdmissionError> {
    let candidates: [DispatchCandidate; N] =
        core::array::from_fn(|index| variants[index].candidate());
    let decision = planner.admit(request, &candidates)?;
    Ok(AxDispatchChoice {
        decision,
        executable: variants[decision.candidate_index].executable(),
    })
}

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// CPU plus independently owned accelerator sessions for compiled `.ax` schedules.
pub struct AxHetero<C: Accelerator, A: Accelerator> {
    pub cpu: Session<C>,
    pub accel: Option<Session<A>>,
    additional_accelerators: Vec<(Box<dyn GraphSession>, AxDeviceModel)>,
    owner: u64,
    next_program: u64,
}

impl<C: Accelerator, A: Accelerator + TosaCapabilityProvider> AxHetero<C, A> {
    /// Conservative default placement with the active provider's exact target
    /// and dtype roles. No backend-name switch or CoreML-specific preset is
    /// needed. Costs are the common defaults, not measured host calibration.
    /// Callers may adjust the returned model and pass it to `compile_in`.
    pub fn automatic_device_model(&self) -> AxDeviceModel {
        let default = AxDeviceModel {
            prefer_native_cpu: true,
            ..AxDeviceModel::default()
        };
        let Some(accel) = &self.accel else {
            return AxDeviceModel {
                accelerator: false,
                ..default
            };
        };
        Self::automatic_accelerator_model(default, accel)
    }

    fn automatic_accelerator_model<P: Accelerator + TosaCapabilityProvider>(
        default: AxDeviceModel,
        accel: &Session<P>,
    ) -> AxDeviceModel {
        let mut model = default.with_accelerator_tosa_capability(accel.provider(), accel.target());
        if let Some(capability) = model.accelerator_tosa_capability {
            model.accelerator_boolean_boundaries = capability
                .supports_dtype(TosaDType::BOOL, ValueRoles::INPUT)
                && capability.supports_dtype(TosaDType::BOOL, ValueRoles::OUTPUT);
            model.f32_boolean_boundary_bridge = !model.accelerator_boolean_boundaries
                && capability.supports_dtype(TosaDType::BOOL, ValueRoles::INTERMEDIATE)
                && capability.supports_dtype(TosaDType::FP32, ValueRoles::ALL);
        }
        model
    }

    /// Derive each target's capability policy independently. Default costs
    /// describe the executor's host-staged byte path, including both legs of
    /// an accelerator-to-accelerator transfer. They are not calibration data.
    pub fn automatic_device_set(&self) -> AxDeviceSet {
        let cpu = self.automatic_device_model();
        let accelerators: Vec<_> = self
            .accel
            .iter()
            .map(|session| Self::automatic_accelerator_model(AxDeviceModel::default(), session))
            .chain(
                self.additional_accelerators
                    .iter()
                    .map(|(_, model)| model.clone()),
            )
            .collect();
        let mut routes = Vec::new();
        let count = accelerators.len() as u16 + 1;
        for destination in 0..count {
            for source in core::iter::once(None).chain((0..count).map(Some)) {
                if source == Some(destination) {
                    continue;
                }
                let mut fixed = 0u64;
                let mut per_byte = 0u64;
                for backend in source.into_iter().chain(core::iter::once(destination)) {
                    if let Some(model) = backend
                        .checked_sub(1)
                        .and_then(|i| accelerators.get(i as usize))
                    {
                        fixed = fixed.saturating_add(model.transfer_fixed_cost);
                        per_byte = per_byte.saturating_add(model.transfer_byte_cost);
                    }
                }
                routes.push(crate::AxTransferCost {
                    source,
                    destination,
                    fixed,
                    per_byte,
                });
            }
        }
        AxDeviceSet {
            cpu,
            accelerators,
            routes,
        }
    }

    /// Recommended compile entry point: derive provider capability/ABI policy,
    /// then select placement and exact native optimizations automatically.
    /// Explicit cost, precision and CPU masks remain available through `compile_in`.
    pub fn compile_auto_in<M: Allocator + Clone>(
        &mut self,
        source: &Graph<'_>,
        allocator: M,
    ) -> Result<AxExecutable<M>, AxCompileError> {
        self.compile_set_in(source, &self.automatic_device_set(), allocator)
    }
}

impl<C: Accelerator, A: Accelerator> AxHetero<C, A> {
    pub fn new(cpu: Session<C>, accel: Option<Session<A>>) -> Self {
        Self {
            cpu,
            accel,
            additional_accelerators: Vec::new(),
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            next_program: 0,
        }
    }

    /// Add a concrete provider session before compiling a multi-target graph.
    /// IDs are stable for this executor's lifetime. The first accelerator must
    /// exist; no ID is reused and each session retains its own event namespace.
    pub fn add_accelerator<P: Accelerator + TosaCapabilityProvider + 'static>(
        &mut self,
        session: Session<P>,
    ) -> Result<u16, Session<P>> {
        if self.accel.is_none() || self.additional_accelerators.len() >= u16::MAX as usize - 2 {
            return Err(session);
        }
        if self.additional_accelerators.try_reserve(1).is_err() {
            return Err(session);
        }
        let id = self.additional_accelerators.len() as u16 + 2;
        let model =
            AxHetero::<C, P>::automatic_accelerator_model(AxDeviceModel::default(), &session);
        let storage = match Box::<Session<P>>::try_new_uninit() {
            Ok(storage) => storage,
            Err(_) => return Err(session),
        };
        let session = Box::write(storage, session);
        self.additional_accelerators.push((session, model));
        Ok(id)
    }
    fn accelerator(&self, backend: u16) -> Option<&(dyn GraphSession + '_)> {
        if backend == 1 {
            self.accel.as_ref().map(|s| s as &dyn GraphSession)
        } else {
            self.additional_accelerators
                .get(backend.checked_sub(2)? as usize)
                .map(|s| s.0.as_ref())
        }
    }
    fn accelerator_mut(&mut self, backend: u16) -> Option<&mut (dyn GraphSession + '_)> {
        if backend == 1 {
            self.accel.as_mut().map(|s| s as &mut dyn GraphSession)
        } else {
            match self
                .additional_accelerators
                .get_mut(backend.checked_sub(2)? as usize)
            {
                Some(s) => Some(s.0.as_mut()),
                None => None,
            }
        }
    }
    /// Compile over every supplied target using the same canonical partition
    /// admission, rollback, native lowering and resumable execution paths.
    pub fn compile_set_in<M: Allocator + Clone>(
        &mut self,
        source: &Graph<'_>,
        model: &AxDeviceSet,
        allocator: M,
    ) -> Result<AxExecutable<M>, AxCompileError> {
        if model.backend_count() == 0
            || (1..model.backend_count()).any(|b| self.accelerator(b).is_none())
        {
            return Err(AxCompileError::MissingAccelerator);
        }
        self.compile_model_in(source, model, allocator)
    }

    /// Analyze, place, partition, lower, and provider-compile a canonical
    /// graph using an explicit model and storage owned by `allocator`.
    /// Compilation happens once; execution reuses the selected native instruction
    /// schedule and provider programs. Prefer `compile_auto_in` unless overriding
    /// cost, capability or precision policy.
    pub fn compile_in<M: Allocator + Clone>(
        &mut self,
        source: &Graph<'_>,
        model: &AxDeviceModel,
        allocator: M,
    ) -> Result<AxExecutable<M>, AxCompileError> {
        let mut active = model.clone();
        active.accelerator &= self.accel.is_some();
        self.compile_model_in(source, &active, allocator)
    }

    fn compile_model_in<M: Allocator + Clone>(
        &mut self,
        source: &Graph<'_>,
        model: &dyn DevicePolicies,
        allocator: M,
    ) -> Result<AxExecutable<M>, AxCompileError> {
        let graph = GraphBuf::copy_from_in(source.as_bytes(), allocator.clone())?;
        let parsed = graph.parse().expect("copy validated graph");
        let mut schedule = Schedule::analyze_in(&parsed, allocator.clone())?;
        let active_model = model.cpu_policy();
        let mut boolean_placement = Vec::new_in(allocator.clone());
        reserve(
            &mut boolean_placement,
            model.backend_count().saturating_sub(1) as usize,
        )?;
        for backend in 1..model.backend_count() {
            let policy = model
                .accelerator_policy(backend)
                .ok_or(AxCompileError::MissingAccelerator)?;
            let mut target = boolean_placement_in(&parsed, &schedule, policy, allocator.clone())?;
            target.backend = backend;
            boolean_placement.push(target);
        }
        let placement_report = loop {
            let placement_model = AxPlacementModel {
                model,
                targets: &boolean_placement,
            };
            let mut report = schedule.place_optimized(&parsed, &placement_model)?;
            if boolean_placement.iter_mut().fold(false, |changed, target| {
                target.normalize(&schedule) | changed
            }) {
                continue;
            }
            // Refinement and Boolean-component normalization can change backend
            // runs, so repeat the allocation-free pass for the final placement.
            report.regions_eliminated = report
                .regions_eliminated
                .saturating_add(schedule.cluster_by_placement()?);
            let placement_model = AxPlacementModel {
                model,
                targets: &boolean_placement,
            };
            report.aggregate_cost = schedule.aggregate_cost(&parsed, &placement_model)?;
            schedule
                .build_plan(&parsed)
                .map_err(AllocationError::from)?;
            let mut boolean_changed = false;
            let mut capability_changed = false;
            for target in &mut boolean_placement {
                boolean_changed |= target.normalize_partition_crossings(&parsed, &schedule)?;
                capability_changed |= forbid_unsupported_accelerator_partitions(
                    &parsed,
                    &schedule,
                    model
                        .accelerator_policy(target.backend)
                        .ok_or(AxCompileError::MissingAccelerator)?,
                    target.backend,
                    &mut target.accelerator_forbidden,
                )?;
            }
            if boolean_changed || capability_changed {
                continue;
            }
            break report;
        };
        let partition_count = schedule.plan().partitions.len();

        // Reserve the rollback journal before installing any provider program;
        // every later failure can then evict all successfully ensured keys
        // without another fallible allocation.
        let mut installed = Vec::new_in(allocator.clone());
        reserve(&mut installed, partition_count)?;
        let result = (|| -> Result<_, AxCompileError> {
            let mut compiled = Vec::new_in(allocator.clone());
            reserve(&mut compiled, partition_count)?;
            for partition_index in 0..partition_count {
                let partition = schedule.plan().partitions[partition_index];
                if partition.backend == CPU_BACKEND
                    && partition.implementation == CPU_KERNEL_IMPLEMENTATION
                {
                    compiled.push(compile_native_cpu_partition_in(
                        &parsed,
                        schedule.order(),
                        schedule.producers(),
                        partition,
                        &active_model,
                        allocator.clone(),
                    )?);
                    continue;
                }
                if partition.backend == CPU_BACKEND
                    && is_sparse_bsr_implementation(partition.implementation)
                {
                    compiled.push(compile_sparse_partition_in(
                        &parsed,
                        schedule.order(),
                        schedule.producers(),
                        partition,
                        active_model
                            .sparse_bsr
                            .as_ref()
                            .map(|policy| policy.block_k)
                            .ok_or(AxCompileError::InvalidSparsePartition)?,
                        allocator.clone(),
                    )?);
                    continue;
                }
                if partition.implementation != DENSE_IMPLEMENTATION {
                    return Err(AxCompileError::UnsupportedImplementation {
                        backend: partition.backend,
                        implementation: partition.implementation,
                    });
                }
                let accelerator_partition = partition.backend != CPU_BACKEND;
                let active_model = if accelerator_partition {
                    model
                        .accelerator_policy(partition.backend)
                        .ok_or(AxCompileError::MissingAccelerator)?
                } else {
                    model.cpu_policy()
                };
                let lowering = AxLoweringPolicy {
                    precision: if accelerator_partition && active_model.fp8_promotion {
                        AxPrecisionPolicy::PromoteFp8ToF16
                    } else {
                        AxPrecisionPolicy::Native
                    },
                    boolean_boundaries: if accelerator_partition
                        && active_model.f32_boolean_boundary_bridge
                    {
                        AxBooleanBoundaryPolicy::F32Bridge
                    } else {
                        AxBooleanBoundaryPolicy::Native
                    },
                };
                let lowered = artifact::ax_partition_with_lowering_policy_in(
                    &parsed,
                    schedule.order(),
                    schedule.producers(),
                    partition,
                    lowering,
                    allocator.clone(),
                )?;
                let key = self.next_program;
                self.next_program = self.next_program.wrapping_add(1);
                let artifact::AxPartitionArtifact {
                    bytes,
                    inputs,
                    input_transforms,
                    outputs,
                    output_transforms,
                    output_bytes,
                } = lowered;
                match partition.backend {
                    CPU_BACKEND => self.cpu.ensure(key, || bytes)?,
                    backend if backend != CPU_BACKEND => self
                        .accelerator_mut(backend)
                        .ok_or(AxCompileError::MissingAccelerator)?
                        .install(key, bytes)?,
                    _ => return Err(AxCompileError::MissingAccelerator),
                }
                installed.push((partition.backend, key));
                compiled.push(ExecutablePartition {
                    backend: partition.backend,
                    implementation: partition.implementation,
                    key,
                    inputs,
                    input_transforms,
                    outputs,
                    output_transforms,
                    output_bytes,
                    kind: ExecutablePartitionKind::Provider,
                });
            }

            let mut inputs = Vec::new_in(allocator.clone());
            let mut outputs = Vec::new_in(allocator.clone());
            reserve(&mut inputs, parsed.tensor_count())?;
            reserve(&mut outputs, parsed.tensor_count())?;
            for index in 0..parsed.tensor_count() {
                let tensor = parsed
                    .tensor(TensorId(index as u32))
                    .map_err(|_| AxCompileError::MissingTensor(TensorId(index as u32)))?;
                if tensor.flags.contains(TensorFlags::INPUT) {
                    inputs.push(tensor.id);
                }
                if tensor.flags.contains(TensorFlags::OUTPUT) {
                    outputs.push(tensor.id);
                }
            }
            let value_slots = inputs
                .iter()
                .chain(&outputs)
                .chain(
                    compiled
                        .iter()
                        .flat_map(|partition| partition.inputs().iter().chain(partition.outputs())),
                )
                .map(|id| id.0 as usize + 1)
                .max()
                .unwrap_or(0);
            Ok((compiled, inputs, outputs, value_slots))
        })();
        if result.is_err() {
            self.rollback_programs(&installed);
        }
        let (partitions, inputs, outputs, value_slots) = result?;
        Ok(AxExecutable {
            owner: self.owner,
            graph,
            schedule,
            placement_report,
            value_slots,
            inputs,
            outputs,
            partitions,
        })
    }

    fn rollback_programs(&mut self, installed: &[(u16, u64)]) {
        for &(backend, key) in installed.iter().rev() {
            let _ = match backend {
                CPU_BACKEND => self.cpu.evict_generic(key),
                backend if backend != CPU_BACKEND => self
                    .accelerator_mut(backend)
                    .ok_or(BackendError::InvalidArgument)
                    .and_then(|session| session.evict_generic(key)),
                _ => Err(BackendError::InvalidArgument),
            };
        }
    }

    /// Release the provider programs and fixed I/O owned by one executable.
    ///
    /// `Session` deliberately retains compiled programs for warm dispatch, so
    /// dropping only the allocator-owned schedule cannot release provider
    /// handles. Cache owners use this consuming operation when an exact graph
    /// or concrete viewport signature is evicted. Every provider partition is
    /// attempted even if an earlier release fails; rejected handles remain
    /// quarantined inside their session until runtime teardown.
    pub fn release_executable<M: Allocator>(
        &mut self,
        executable: AxExecutable<M>,
    ) -> Result<(), AxCompileError> {
        if executable.owner != self.owner {
            return Err(AxCompileError::WrongExecutor);
        }

        let mut first_error = None;
        for partition in executable.partitions {
            if !matches!(partition.kind, ExecutablePartitionKind::Provider) {
                continue;
            }
            let released = match partition.backend {
                CPU_BACKEND => self.cpu.evict_generic(partition.key),
                backend if backend != CPU_BACKEND => self
                    .accelerator_mut(backend)
                    .ok_or(BackendError::InvalidArgument)
                    .and_then(|session| session.evict_generic(partition.key)),
                _ => Err(BackendError::InvalidArgument),
            };
            if let Err(error) = released {
                first_error.get_or_insert(error);
            }
        }

        first_error.map_or(Ok(()), |error| Err(AxCompileError::Backend(error)))
    }

    /// Prepare a caller-driven graph invocation without submitting work.
    ///
    /// Input byte planes remain borrowed for the invocation lifetime. Each
    /// call to [`advance_in`](Self::advance_in) performs at most one provider
    /// submission, one bounded event poll, or one local sparse partition, so
    /// an event loop retains control between placed partitions.
    pub fn begin_in<'executable, 'input, M: Allocator, R: Allocator + Clone>(
        &self,
        executable: &'executable AxExecutable<M>,
        inputs: &[&'input [u8]],
        allocator: R,
    ) -> Result<AxInvocation<'executable, 'input, M, R>, AxCompileError> {
        self.begin_with_resources_in(executable, inputs, &[], allocator)
    }

    /// Begin an invocation borrowing both input planes and immutable resources.
    /// Resource validation precedes all allocation and provider submission.
    pub fn begin_with_resources_in<'executable, 'input, M: Allocator, R: Allocator + Clone>(
        &self,
        executable: &'executable AxExecutable<M>,
        inputs: &[&'input [u8]],
        resources: &'input [NativeResourceBinding<'input>],
        allocator: R,
    ) -> Result<AxInvocation<'executable, 'input, M, R>, AxCompileError> {
        if executable.owner != self.owner {
            return Err(AxCompileError::WrongExecutor);
        }
        if inputs.len() != executable.inputs.len() {
            return Err(AxCompileError::InvalidInputCount {
                expected: executable.inputs.len(),
                actual: inputs.len(),
            });
        }
        validate_executable_resources(executable, resources)?;
        let graph = executable.graph();
        let mut values = Vec::new_in(allocator.clone());
        reserve(&mut values, executable.value_slots)?;
        values.resize_with(executable.value_slots, || None);
        for (&tensor_id, &bytes) in executable.inputs.iter().zip(inputs) {
            let tensor = graph
                .tensor(tensor_id)
                .map_err(|_| AxCompileError::MissingTensor(tensor_id))?;
            let expected =
                tensor_storage_bytes(tensor).ok_or(AxCompileError::MissingTensor(tensor_id))?;
            if bytes.len() != expected {
                return Err(AxCompileError::InvalidInputSize {
                    tensor: tensor_id,
                    expected,
                    actual: bytes.len(),
                });
            }
            values[tensor_id.0 as usize] = Some(RuntimeValue::Borrowed(bytes));
        }
        Ok(AxInvocation {
            owner: self.owner,
            executable,
            values,
            resources,
            allocator,
            next_partition: 0,
            host_copies: AxHostCopyStats::default(),
            pending: None,
            finished: false,
        })
    }

    /// Owned-input form of [`begin_in`](Self::begin_in) for event loops whose
    /// control planes are cheaper to copy than to pin across polls.
    ///
    /// This performs one explicit host copy into `R`; provider upload remains
    /// a separate transfer at partition admission. Prefer `begin_in` for
    /// frame-sized or model-sized inputs, and this form for small dynamic
    /// scalars such as pointer-controlled shader parameters.
    pub fn begin_owned_in<'executable, M: Allocator, R: Allocator + Clone>(
        &self,
        executable: &'executable AxExecutable<M>,
        inputs: &[&[u8]],
        allocator: R,
    ) -> Result<AxInvocation<'executable, 'static, M, R>, AxCompileError> {
        if executable.owner != self.owner {
            return Err(AxCompileError::WrongExecutor);
        }
        validate_executable_resources(executable, &[])?;
        if inputs.len() != executable.inputs.len() {
            return Err(AxCompileError::InvalidInputCount {
                expected: executable.inputs.len(),
                actual: inputs.len(),
            });
        }
        let graph = executable.graph();
        let mut values = Vec::new_in(allocator.clone());
        reserve(&mut values, executable.value_slots)?;
        values.resize_with(executable.value_slots, || None);
        for (&tensor_id, &bytes) in executable.inputs.iter().zip(inputs) {
            let tensor = graph
                .tensor(tensor_id)
                .map_err(|_| AxCompileError::MissingTensor(tensor_id))?;
            let expected =
                tensor_storage_bytes(tensor).ok_or(AxCompileError::MissingTensor(tensor_id))?;
            if bytes.len() != expected {
                return Err(AxCompileError::InvalidInputSize {
                    tensor: tensor_id,
                    expected,
                    actual: bytes.len(),
                });
            }
            values[tensor_id.0 as usize] = Some(RuntimeValue::Owned(copy_bytes_in(
                bytes,
                allocator.clone(),
            )?));
        }
        Ok(AxInvocation {
            owner: self.owner,
            executable,
            values,
            resources: &[],
            allocator,
            next_partition: 0,
            host_copies: AxHostCopyStats::default(),
            pending: None,
            finished: false,
        })
    }

    /// Advance using the legacy owned byte-vector result ABI. Prefer
    /// [`advance_retained_in`](Self::advance_retained_in) to retain typed output
    /// backing, or [`advance_into_in`](Self::advance_into_in) for a host surface.
    #[allow(clippy::type_complexity)]
    pub fn advance_in<M: Allocator, R: Allocator + Clone>(
        &mut self,
        invocation: &mut AxInvocation<'_, '_, M, R>,
    ) -> Result<AxInvocationStatus<Vec<Vec<u8, R>, R>>, AxCompileError> {
        match self.advance_retained_in(invocation)? {
            AxInvocationStatus::Complete(outputs) => {
                let mut bytes = Vec::new_in(invocation.allocator.clone());
                reserve(&mut bytes, outputs.len())?;
                for output in outputs {
                    let copied = match &output.value {
                        RuntimeValue::Borrowed(_)
                        | RuntimeValue::Native(NativeValue::F32(_) | NativeValue::I32(_)) => {
                            output.len() as u64
                        }
                        _ => 0,
                    };
                    bytes.push(output.into_bytes_in(invocation.allocator.clone())?);
                    invocation.host_copies.output_materialization_bytes = invocation
                        .host_copies
                        .output_materialization_bytes
                        .saturating_add(copied);
                }
                Ok(AxInvocationStatus::Complete(bytes))
            }
            AxInvocationStatus::Progress(p) => Ok(AxInvocationStatus::Progress(p)),
            AxInvocationStatus::Cancelled(p) => Ok(AxInvocationStatus::Cancelled(p)),
        }
    }

    /// Advance into caller-owned packed or pitched output backing. All output
    /// lengths are validated before advancing. Destinations are borrowed only
    /// during this call; no provider retains their addresses. Publish only on
    /// `Complete(())`: a failed provider readback can leave partial output. Native results copy directly from
    /// typed storage, without first allocating/encoding a byte staging plane.
    /// Terminal provider outputs with identity ABI transforms read directly into
    /// these sinks. This is not device import: provider transfers remain, and
    /// native results need one final host copy with the current CPU kernels.
    pub fn advance_into_in<M: Allocator, R: Allocator + Clone>(
        &mut self,
        invocation: &mut AxInvocation<'_, '_, M, R>,
        outputs: &mut [AxOutput<'_>],
    ) -> Result<AxInvocationStatus<()>, AxCompileError> {
        use virtio_accel_core::ByteSink;
        if invocation.owner != self.owner {
            return Err(AxCompileError::WrongExecutor);
        }
        if outputs.len() != invocation.executable.outputs.len() {
            return Err(BackendError::InvalidArgument.into());
        }
        let graph = invocation.executable.graph();
        for (&id, output) in invocation.executable.outputs.iter().zip(outputs.iter()) {
            let tensor = graph
                .tensor(id)
                .map_err(|_| AxCompileError::MissingTensor(id))?;
            if tensor_storage_bytes(tensor).map(|v| v as u64) != Some(output.len()) {
                return Err(BackendError::InvalidArgument.into());
            }
        }
        // A terminal provider whose complete output set is the graph output
        // set can read directly into the caller's sinks. Earlier graph outputs
        // and representation bridges use the retained path below instead.
        if let Some(pending) = invocation.pending.as_ref() {
            let partition = &invocation.executable.partitions[pending.partition];
            let graph_outputs = &invocation.executable.outputs;
            if pending.partition + 1 == invocation.executable.partitions.len()
                && partition.outputs.len() == graph_outputs.len()
                && partition
                    .outputs
                    .iter()
                    .all(|id| graph_outputs.contains(id))
                && partition
                    .output_transforms
                    .iter()
                    .all(|t| *t == AxOutputTransform::Identity)
            {
                let token = pending.invocation;
                let backend = pending.backend;
                let partition_index = pending.partition;
                let mut ordered = Vec::new_in(invocation.allocator.clone());
                reserve(&mut ordered, outputs.len())?;
                for (&id, output) in graph_outputs.iter().zip(outputs.iter_mut()) {
                    let slot = partition
                        .outputs
                        .iter()
                        .position(|candidate| *candidate == id)
                        .ok_or(AxArtifactError::InvalidPartition)?;
                    ordered.push((slot, output as &mut dyn ByteSink));
                }
                ordered.sort_unstable_by_key(|(slot, _)| *slot);
                let mut sinks = Vec::new_in(invocation.allocator.clone());
                reserve(&mut sinks, ordered.len())?;
                for (_, sink) in ordered {
                    sinks.push(sink);
                }
                let polled = match backend {
                    CPU_BACKEND => self.cpu.poll_many_into(token, &mut sinks),
                    backend if backend != CPU_BACKEND => self
                        .accelerator_mut(backend)
                        .ok_or(AxCompileError::MissingAccelerator)?
                        .poll_many_into(token, &mut sinks),
                    _ => return Err(AxCompileError::MissingAccelerator),
                };
                let status = match polled {
                    Ok(status) => status,
                    Err(error) => {
                        let still_pending = match backend {
                            CPU_BACKEND => self.cpu.is_pending(token),
                            backend if backend != CPU_BACKEND => self
                                .accelerator(backend)
                                .is_some_and(|session| session.is_pending(token)),
                            _ => false,
                        };
                        if !still_pending {
                            invocation.pending = None;
                            invocation.finished = true;
                        }
                        return Err(error.into());
                    }
                };
                let progress = AxPartitionProgress {
                    partition: partition_index,
                    backend,
                    phase: AxPartitionPhase::Complete,
                    submit_error: None,
                };
                if matches!(status, SessionInvocationStatus::Pending) {
                    return Ok(AxInvocationStatus::Progress(AxPartitionProgress {
                        phase: AxPartitionPhase::Waiting,
                        ..progress
                    }));
                }
                invocation.pending = None;
                invocation.finished = true;
                invocation.next_partition = partition_index + 1;
                return match status {
                    SessionInvocationStatus::Complete(()) => Ok(AxInvocationStatus::Complete(())),
                    SessionInvocationStatus::Failed(error) => Err(error.into()),
                    SessionInvocationStatus::Cancelled => {
                        Ok(AxInvocationStatus::Cancelled(progress))
                    }
                    SessionInvocationStatus::Pending => unreachable!(),
                };
            }
        }
        match self.advance_retained_in(invocation)? {
            AxInvocationStatus::Complete(values) => {
                for (output, value) in outputs.iter_mut().zip(values) {
                    output.write_at(0, value.as_bytes())?;
                    invocation.host_copies.output_materialization_bytes = invocation
                        .host_copies
                        .output_materialization_bytes
                        .saturating_add(value.len() as u64);
                }
                Ok(AxInvocationStatus::Complete(()))
            }
            AxInvocationStatus::Progress(p) => Ok(AxInvocationStatus::Progress(p)),
            AxInvocationStatus::Cancelled(p) => Ok(AxInvocationStatus::Cancelled(p)),
        }
    }

    /// Synchronous caller-owned output form of [`run_in`](Self::run_in).
    pub fn run_into_in<M: Allocator, R: Allocator + Clone>(
        &mut self,
        executable: &AxExecutable<M>,
        inputs: &[&[u8]],
        outputs: &mut [AxOutput<'_>],
        allocator: R,
    ) -> Result<(), AxCompileError> {
        let mut invocation = self.begin_in(executable, inputs, allocator)?;
        loop {
            match self.advance_into_in(&mut invocation, outputs)? {
                AxInvocationStatus::Complete(()) => return Ok(()),
                AxInvocationStatus::Progress(_) => std::thread::yield_now(),
                AxInvocationStatus::Cancelled(_) => {
                    return Err(BackendError::DeadlineExpired.into());
                }
            }
        }
    }

    /// Advance one graph invocation by one bounded scheduling transition.
    ///
    /// A provider partition is submitted in one call and polled in later
    /// calls. Terminal outputs are transformed and retained before the next
    /// partition becomes eligible. Packed sparse CPU partitions remain local
    /// today, but only one such partition is executed per call.
    pub fn advance_retained_in<'input, M: Allocator, R: Allocator + Clone>(
        &mut self,
        invocation: &mut AxInvocation<'_, 'input, M, R>,
    ) -> Result<AxInvocationStatus<Vec<AxTensor<'input, R>, R>>, AxCompileError> {
        if invocation.owner != self.owner || invocation.executable.owner != self.owner {
            return Err(AxCompileError::WrongExecutor);
        }
        if invocation.finished {
            return Err(AxCompileError::InvocationFinished);
        }

        if let Some(pending) = invocation.pending.as_ref() {
            let partition_index = pending.partition;
            let backend = pending.backend;
            let token = pending.invocation;
            let polled = match backend {
                CPU_BACKEND => self.cpu.poll_many_with(token, |provider, buffers, sizes| {
                    read_partition_outputs_in(
                        provider,
                        buffers,
                        sizes,
                        &invocation.executable.graph(),
                        &invocation.executable.partitions[partition_index],
                        invocation.allocator.clone(),
                    )
                }),
                backend if backend != CPU_BACKEND => self
                    .accelerator_mut(backend)
                    .ok_or(AxCompileError::MissingAccelerator)?
                    .poll_with(token, |reader| {
                        read_partition_outputs_from_in(
                            reader,
                            &invocation.executable.graph(),
                            &invocation.executable.partitions[partition_index],
                            invocation.allocator.clone(),
                        )
                    }),
                _ => return Err(AxCompileError::MissingAccelerator),
            };
            let status = match polled {
                Ok(status) => status,
                Err(error) => {
                    let still_pending = match backend {
                        CPU_BACKEND => self.cpu.is_pending(token),
                        backend if backend != CPU_BACKEND => self
                            .accelerator(backend)
                            .is_some_and(|session| session.is_pending(token)),
                        _ => false,
                    };
                    if !still_pending {
                        invocation.pending = None;
                        invocation.finished = true;
                    }
                    return Err(error.into());
                }
            };
            match status {
                SessionInvocationStatus::Pending => {
                    return Ok(AxInvocationStatus::Progress(AxPartitionProgress {
                        partition: partition_index,
                        backend,
                        phase: AxPartitionPhase::Waiting,
                        submit_error: None,
                    }));
                }
                SessionInvocationStatus::Complete(outputs) => {
                    invocation.pending = None;
                    let partition = &invocation.executable.partitions[partition_index];
                    if let Err(error) =
                        commit_partition_outputs(partition, outputs, &mut invocation.values)
                    {
                        invocation.finished = true;
                        return Err(error);
                    }
                    invocation.next_partition = partition_index + 1;
                    if invocation.next_partition == invocation.executable.partitions.len() {
                        let outputs = collect_graph_outputs_in(
                            invocation.executable,
                            &mut invocation.values,
                            invocation.allocator.clone(),
                        );
                        invocation.finished = true;
                        return outputs.map(AxInvocationStatus::Complete);
                    }
                    return Ok(AxInvocationStatus::Progress(AxPartitionProgress {
                        partition: partition_index,
                        backend,
                        phase: AxPartitionPhase::Complete,
                        submit_error: None,
                    }));
                }
                SessionInvocationStatus::Failed(error) => {
                    invocation.pending = None;
                    invocation.finished = true;
                    return Err(error.into());
                }
                SessionInvocationStatus::Cancelled => {
                    invocation.pending = None;
                    invocation.finished = true;
                    return Ok(AxInvocationStatus::Cancelled(AxPartitionProgress {
                        partition: partition_index,
                        backend,
                        phase: AxPartitionPhase::Complete,
                        submit_error: None,
                    }));
                }
            }
        }

        if invocation.next_partition == invocation.executable.partitions.len() {
            let outputs = collect_graph_outputs_in(
                invocation.executable,
                &mut invocation.values,
                invocation.allocator.clone(),
            );
            invocation.finished = true;
            return outputs.map(AxInvocationStatus::Complete);
        }

        let partition_index = invocation.next_partition;
        let partition = &invocation.executable.partitions[partition_index];
        match &partition.kind {
            ExecutablePartitionKind::NativeCpu(native) => {
                let graph = invocation.executable.graph();
                let outputs = match run_native_cpu_partition_in(
                    native,
                    &graph,
                    &invocation.values,
                    &partition.outputs,
                    invocation.resources,
                    &mut invocation.host_copies,
                    invocation.allocator.clone(),
                ) {
                    Ok(outputs) => outputs,
                    Err(error) => {
                        invocation.finished = true;
                        return Err(error.into());
                    }
                };
                if let Err(error) =
                    commit_partition_outputs(partition, outputs, &mut invocation.values)
                {
                    invocation.finished = true;
                    return Err(error);
                }
            }
            ExecutablePartitionKind::SparseF32(sparse) => {
                let outputs = match run_sparse_f32_partition_in(
                    sparse,
                    &invocation.values,
                    &partition.outputs,
                    invocation.allocator.clone(),
                )
                .and_then(own_byte_outputs)
                {
                    Ok(outputs) => outputs,
                    Err(error) => {
                        invocation.finished = true;
                        return Err(error);
                    }
                };
                if let Err(error) =
                    commit_partition_outputs(partition, outputs, &mut invocation.values)
                {
                    invocation.finished = true;
                    return Err(error);
                }
            }
            ExecutablePartitionKind::SparseFp8(sparse) => {
                let outputs = match run_sparse_fp8_partition_in(
                    sparse,
                    &invocation.values,
                    &partition.outputs,
                    invocation.allocator.clone(),
                )
                .and_then(own_byte_outputs)
                {
                    Ok(outputs) => outputs,
                    Err(error) => {
                        invocation.finished = true;
                        return Err(error);
                    }
                };
                if let Err(error) =
                    commit_partition_outputs(partition, outputs, &mut invocation.values)
                {
                    invocation.finished = true;
                    return Err(error);
                }
            }
            ExecutablePartitionKind::Provider => {
                if partition.implementation != DENSE_IMPLEMENTATION
                    || partition.inputs.len() != partition.input_transforms.len()
                {
                    return Err(AxArtifactError::InvalidPartition.into());
                }
                let mut adapted = Vec::new_in(invocation.allocator.clone());
                if partition
                    .input_transforms
                    .iter()
                    .any(|transform| *transform != AxInputTransform::Identity)
                {
                    reserve(&mut adapted, partition.inputs.len())?;
                    adapted.resize_with(partition.inputs.len(), || None);
                }
                for (slot, (&id, &transform)) in partition
                    .inputs
                    .iter()
                    .zip(&partition.input_transforms)
                    .enumerate()
                {
                    if transform == AxInputTransform::Identity {
                        continue;
                    }
                    let source = invocation.values[id.0 as usize]
                        .as_ref()
                        .map(RuntimeValue::as_slice)
                        .ok_or(AxCompileError::MissingTensor(id))?;
                    adapted[slot] = Some(transform_input_in(
                        transform,
                        source,
                        invocation.allocator.clone(),
                    )?);
                }
                let mut input_refs = Vec::new_in(invocation.allocator.clone());
                reserve(&mut input_refs, partition.inputs.len())?;
                for (slot, id) in partition.inputs.iter().enumerate() {
                    let source = invocation.values[id.0 as usize]
                        .as_ref()
                        .map(RuntimeValue::as_slice)
                        .ok_or(AxCompileError::MissingTensor(*id))?;
                    input_refs.push(
                        adapted
                            .get(slot)
                            .and_then(Option::as_deref)
                            .unwrap_or(source),
                    );
                }
                let submitted = match partition.backend {
                    CPU_BACKEND => self.cpu.submit_many(
                        partition.key,
                        || unreachable!("compiled ax CPU partition missing"),
                        &input_refs,
                        &partition.output_bytes,
                    ),
                    backend if backend != CPU_BACKEND => self
                        .accelerator_mut(backend)
                        .ok_or(AxCompileError::MissingAccelerator)?
                        .submit(partition.key, &input_refs, &partition.output_bytes),
                    _ => return Err(AxCompileError::MissingAccelerator),
                };
                let (token, submit_error) = match submitted {
                    Ok(token) => (token, None),
                    Err(failure) => match failure.invocation {
                        Some(token) => (token, Some(failure.error)),
                        None => return Err(failure.error.into()),
                    },
                };
                invocation.pending = Some(PendingPartition {
                    partition: partition_index,
                    backend: partition.backend,
                    invocation: token,
                });
                return Ok(AxInvocationStatus::Progress(AxPartitionProgress {
                    partition: partition_index,
                    backend: partition.backend,
                    phase: AxPartitionPhase::Submitted,
                    submit_error,
                }));
            }
        }

        invocation.next_partition += 1;
        if invocation.next_partition == invocation.executable.partitions.len() {
            let outputs = collect_graph_outputs_in(
                invocation.executable,
                &mut invocation.values,
                invocation.allocator.clone(),
            );
            invocation.finished = true;
            return outputs.map(AxInvocationStatus::Complete);
        }
        Ok(AxInvocationStatus::Progress(AxPartitionProgress {
            partition: partition_index,
            backend: partition.backend,
            phase: AxPartitionPhase::Complete,
            submit_error: None,
        }))
    }

    /// Request nonblocking cancellation of the currently submitted provider
    /// partition. The invocation remains live and must be advanced until a
    /// terminal status is observed.
    pub fn cancel_in<M: Allocator, R: Allocator + Clone>(
        &self,
        invocation: &AxInvocation<'_, '_, M, R>,
    ) -> Result<(), AxCompileError> {
        if invocation.owner != self.owner || invocation.executable.owner != self.owner {
            return Err(AxCompileError::WrongExecutor);
        }
        if invocation.finished {
            return Err(AxCompileError::InvocationFinished);
        }
        let pending = invocation
            .pending
            .as_ref()
            .ok_or(AxCompileError::Backend(BackendError::Busy))?;
        match pending.backend {
            CPU_BACKEND => self.cpu.cancel(pending.invocation)?,
            backend if backend != CPU_BACKEND => self
                .accelerator(backend)
                .ok_or(AxCompileError::MissingAccelerator)?
                .cancel(pending.invocation)?,
            _ => return Err(AxCompileError::MissingAccelerator),
        }
        Ok(())
    }

    /// Execute with graph inputs in ascending tensor-ID order. Transient
    /// tensor storage and returned outputs belong to `allocator`, independent
    /// of the allocator that owns the executable.
    pub fn run_in<M: Allocator, R: Allocator + Clone>(
        &mut self,
        executable: &AxExecutable<M>,
        inputs: &[&[u8]],
        allocator: R,
    ) -> Result<Vec<Vec<u8, R>, R>, AxCompileError> {
        self.run_with_resources_in(executable, inputs, &[], allocator)
    }

    /// Execute with borrowed native resources and allocator-owned outputs.
    /// All required bindings are validated before any partition runs.
    pub fn run_with_resources_in<M: Allocator, R: Allocator + Clone>(
        &mut self,
        executable: &AxExecutable<M>,
        inputs: &[&[u8]],
        resources: &[NativeResourceBinding<'_>],
        allocator: R,
    ) -> Result<Vec<Vec<u8, R>, R>, AxCompileError> {
        let mut invocation =
            self.begin_with_resources_in(executable, inputs, resources, allocator)?;
        loop {
            match self.advance_in(&mut invocation)? {
                AxInvocationStatus::Complete(outputs) => return Ok(outputs),
                AxInvocationStatus::Progress(_) => std::thread::yield_now(),
                AxInvocationStatus::Cancelled(_) => {
                    return Err(BackendError::DeadlineExpired.into());
                }
            }
        }
    }

    /// Repeatedly execute any resource-free schedule, choosing provider-resident
    /// feedback when a single provider partition can express all bindings.
    /// Otherwise recurrent host tensor backing moves into the next invocation
    /// without byte-vector staging; mixed schedules still transfer their provider
    /// boundaries each iteration. Non-feedback inputs retain their original values.
    /// Only final outputs materialize the returned byte-vector ABI.
    /// Use `run_repeated_in` to require the stricter provider-residency guarantee.
    pub fn run_repeated_auto_in<M: Allocator, R: Allocator + Clone>(
        &mut self,
        executable: &AxExecutable<M>,
        inputs: &[&[u8]],
        repetitions: usize,
        feedback: &[AxFeedback],
        allocator: R,
    ) -> Result<Vec<Vec<u8, R>, R>, AxCompileError> {
        if executable.owner != self.owner {
            return Err(AxCompileError::WrongExecutor);
        }
        if repetitions == 0 {
            return Err(AxCompileError::InvalidRepetitionCount);
        }
        let graph = executable.graph();
        let mut feedback_slots = Vec::new_in(allocator.clone());
        reserve(&mut feedback_slots, feedback.len())?;
        for (index, binding) in feedback.iter().enumerate() {
            let invalid = || AxCompileError::InvalidFeedback {
                input: binding.input,
                output: binding.output,
            };
            if !executable.inputs.contains(&binding.input)
                || feedback[..index]
                    .iter()
                    .any(|prior| prior.input == binding.input || prior.output == binding.output)
            {
                return Err(invalid());
            }
            let output_slot = executable
                .outputs
                .iter()
                .position(|id| *id == binding.output)
                .ok_or_else(invalid)?;
            let input = graph.tensor(binding.input).map_err(|_| invalid())?;
            let output = graph.tensor(binding.output).map_err(|_| invalid())?;
            if input.dtype != output.dtype || !input.dimensions().eq(output.dimensions()) {
                return Err(invalid());
            }
            feedback_slots.push((binding.input, output_slot));
        }

        // Select by the compiled schedule, never by backend name or by retrying
        // a failed execution. A provider error must not replay accepted work.
        let resident = match executable.partitions.as_slice() {
            [partition] if matches!(&partition.kind, ExecutablePartitionKind::Provider) => {
                feedback.iter().all(|binding| {
                    let Some(input) = partition.inputs.iter().position(|id| *id == binding.input)
                    else {
                        return false;
                    };
                    let Some(output) = partition
                        .outputs
                        .iter()
                        .position(|id| *id == binding.output)
                    else {
                        return false;
                    };
                    matches!(
                        (
                            partition.input_transforms.get(input),
                            partition.output_transforms.get(output)
                        ),
                        (
                            Some(AxInputTransform::Identity),
                            Some(AxOutputTransform::Identity)
                        ) | (
                            Some(AxInputTransform::BoolU8ToF32),
                            Some(AxOutputTransform::BoolF32ToU8)
                        )
                    )
                })
            }
            _ => false,
        };
        if resident {
            return self.run_repeated_in(executable, inputs, repetitions, feedback, allocator);
        }

        let mut invocation = self.begin_in(executable, inputs, allocator.clone())?;
        for repetition in 0..repetitions {
            let outputs = loop {
                match self.advance_retained_in(&mut invocation)? {
                    AxInvocationStatus::Complete(outputs) => break outputs,
                    AxInvocationStatus::Progress(_) => std::thread::yield_now(),
                    AxInvocationStatus::Cancelled(_) => {
                        return Err(BackendError::DeadlineExpired.into());
                    }
                }
            };
            if repetition == repetitions - 1 {
                let mut bytes = Vec::new_in(allocator.clone());
                reserve(&mut bytes, outputs.len())?;
                for output in outputs {
                    bytes.push(output.into_bytes_in(allocator.clone())?);
                }
                return Ok(bytes);
            }
            // Every provider event has completed and been released before
            // Complete. Clear stale intermediates, restore invariant inputs
            // (including graph-output passthroughs), then move recurrent backing.
            for value in &mut invocation.values {
                *value = None;
            }
            for (&id, &bytes) in executable.inputs.iter().zip(inputs) {
                invocation.values[id.0 as usize] = Some(RuntimeValue::Borrowed(bytes));
            }
            for (slot, output) in outputs.into_iter().enumerate() {
                if let Some(&(input, _)) =
                    feedback_slots.iter().find(|&&(_, output)| output == slot)
                {
                    invocation.values[input.0 as usize] = Some(output.value);
                }
            }
            invocation.next_partition = 0;
            invocation.finished = false;
        }
        unreachable!("nonzero repetitions return final outputs")
    }

    /// Repeatedly execute a single-provider graph while keeping recurrent
    /// values in provider-owned buffers between submissions.
    ///
    /// `feedback` maps graph outputs to the graph inputs of the next
    /// invocation. Input transforms run once before the first submission,
    /// output transforms once after the final submission, and no recurrent
    /// value crosses the host/provider boundary between invocations. A
    /// multi-partition graph cannot yet preserve this invariant because the
    /// current provider contract has no cross-device buffer import/export.
    pub fn run_repeated_in<M: Allocator, R: Allocator + Clone>(
        &mut self,
        executable: &AxExecutable<M>,
        inputs: &[&[u8]],
        repetitions: usize,
        feedback: &[AxFeedback],
        allocator: R,
    ) -> Result<Vec<Vec<u8, R>, R>, AxCompileError> {
        if executable.owner != self.owner {
            return Err(AxCompileError::WrongExecutor);
        }
        if repetitions == 0 {
            return Err(AxCompileError::InvalidRepetitionCount);
        }
        if inputs.len() != executable.inputs.len() {
            return Err(AxCompileError::InvalidInputCount {
                expected: executable.inputs.len(),
                actual: inputs.len(),
            });
        }
        let [partition] = executable.partitions.as_slice() else {
            return Err(AxCompileError::ResidentRequiresSinglePartition);
        };
        if !matches!(&partition.kind, ExecutablePartitionKind::Provider) {
            return Err(AxCompileError::ResidentRequiresSinglePartition);
        }
        if partition.inputs.len() != partition.input_transforms.len()
            || partition.outputs.len() != partition.output_transforms.len()
            || partition.outputs.len() != partition.output_bytes.len()
        {
            return Err(AxArtifactError::InvalidPartition.into());
        }

        let graph = executable.graph();
        let mut values = Vec::new_in(allocator.clone());
        reserve(&mut values, executable.value_slots)?;
        values.resize_with(executable.value_slots, || None);
        for (&tensor_id, &bytes) in executable.inputs.iter().zip(inputs) {
            let tensor = graph
                .tensor(tensor_id)
                .map_err(|_| AxCompileError::MissingTensor(tensor_id))?;
            let expected =
                tensor_storage_bytes(tensor).ok_or(AxCompileError::MissingTensor(tensor_id))?;
            if bytes.len() != expected {
                return Err(AxCompileError::InvalidInputSize {
                    tensor: tensor_id,
                    expected,
                    actual: bytes.len(),
                });
            }
            values[tensor_id.0 as usize] = Some(RuntimeValue::Borrowed(bytes));
        }

        let mut adapted = Vec::new_in(allocator.clone());
        if partition
            .input_transforms
            .iter()
            .any(|transform| *transform != AxInputTransform::Identity)
        {
            reserve(&mut adapted, partition.inputs.len())?;
            adapted.resize_with(partition.inputs.len(), || None);
        }
        for (slot, (&id, &transform)) in partition
            .inputs
            .iter()
            .zip(&partition.input_transforms)
            .enumerate()
        {
            if transform == AxInputTransform::Identity {
                continue;
            }
            let source = values[id.0 as usize]
                .as_ref()
                .map(RuntimeValue::as_slice)
                .ok_or(AxCompileError::MissingTensor(id))?;
            adapted[slot] = Some(transform_input_in(transform, source, allocator.clone())?);
        }
        let mut input_refs = Vec::new_in(allocator.clone());
        reserve(&mut input_refs, partition.inputs.len())?;
        for (slot, id) in partition.inputs.iter().enumerate() {
            let source = values[id.0 as usize]
                .as_ref()
                .map(RuntimeValue::as_slice)
                .ok_or(AxCompileError::MissingTensor(*id))?;
            input_refs.push(
                adapted
                    .get(slot)
                    .and_then(Option::as_deref)
                    .unwrap_or(source),
            );
        }

        let mut provider_feedback = Vec::new_in(allocator.clone());
        reserve(&mut provider_feedback, feedback.len())?;
        for (index, binding) in feedback.iter().enumerate() {
            let Some(input_slot) = partition.inputs.iter().position(|id| *id == binding.input)
            else {
                return Err(AxCompileError::InvalidFeedback {
                    input: binding.input,
                    output: binding.output,
                });
            };
            let Some(output_slot) = partition
                .outputs
                .iter()
                .position(|id| *id == binding.output)
            else {
                return Err(AxCompileError::InvalidFeedback {
                    input: binding.input,
                    output: binding.output,
                });
            };
            let input_tensor = graph
                .tensor(binding.input)
                .map_err(|_| AxCompileError::MissingTensor(binding.input))?;
            let output_tensor = graph
                .tensor(binding.output)
                .map_err(|_| AxCompileError::MissingTensor(binding.output))?;
            let duplicate = feedback[..index]
                .iter()
                .any(|prior| prior.input == binding.input || prior.output == binding.output);
            let transforms_match = matches!(
                (
                    partition.input_transforms[input_slot],
                    partition.output_transforms[output_slot]
                ),
                (AxInputTransform::Identity, AxOutputTransform::Identity)
                    | (
                        AxInputTransform::BoolU8ToF32,
                        AxOutputTransform::BoolF32ToU8
                    )
            );
            if duplicate
                || !input_tensor.flags.contains(TensorFlags::INPUT)
                || !output_tensor.flags.contains(TensorFlags::OUTPUT)
                || input_tensor.dtype != output_tensor.dtype
                || !input_tensor.dimensions().eq(output_tensor.dimensions())
                || !transforms_match
                || input_refs[input_slot].len() != partition.output_bytes[output_slot]
            {
                return Err(AxCompileError::InvalidFeedback {
                    input: binding.input,
                    output: binding.output,
                });
            }
            provider_feedback.push(FeedbackBinding::new(input_slot, output_slot));
        }

        let provider_outputs = match partition.backend {
            CPU_BACKEND => self.cpu.run_many_repeated_in(
                partition.key,
                || unreachable!("compiled ax CPU partition missing"),
                &input_refs,
                &partition.output_bytes,
                RepeatedExecution::new(repetitions, &provider_feedback),
                allocator.clone(),
            )?,
            backend if backend != CPU_BACKEND => self
                .accelerator_mut(backend)
                .ok_or(AxCompileError::MissingAccelerator)?
                .run_repeated_in(
                    partition.key,
                    &input_refs,
                    &partition.output_bytes,
                    RepeatedExecution::new(repetitions, &provider_feedback),
                    allocator.clone(),
                )?,
            _ => return Err(AxCompileError::MissingAccelerator),
        };
        if provider_outputs.len() != partition.outputs.len() {
            return Err(AxArtifactError::InvalidPartition.into());
        }
        for ((&tensor, &transform), bytes) in partition
            .outputs
            .iter()
            .zip(&partition.output_transforms)
            .zip(provider_outputs)
        {
            let bytes = transform_output_in(transform, tensor, bytes)?;
            values[tensor.0 as usize] = Some(RuntimeValue::Owned(bytes));
        }

        let mut outputs = Vec::new_in(allocator.clone());
        reserve(&mut outputs, executable.outputs.len())?;
        for &id in &executable.outputs {
            if let Some(value) = values[id.0 as usize].take() {
                match value {
                    RuntimeValue::Borrowed(bytes) => {
                        outputs.push(copy_bytes_in(bytes, allocator.clone())?);
                    }
                    RuntimeValue::Owned(bytes) => outputs.push(bytes),
                    RuntimeValue::Native(value) => {
                        outputs.push(encode_native_value_in(value, allocator.clone())?)
                    }
                }
                continue;
            }
            let tensor = graph
                .tensor(id)
                .map_err(|_| AxCompileError::MissingTensor(id))?;
            if !tensor.flags.contains(TensorFlags::CONSTANT) {
                return Err(AxCompileError::MissingTensor(id));
            }
            outputs.push(copy_bytes_in(tensor.constant(), allocator.clone())?);
        }
        Ok(outputs)
    }
}

fn prepare_native_in<M: Allocator, R: Allocator + Clone>(
    executable: &AxExecutable<M>,
    allocator: R,
) -> Result<PreparedNative<R>, AxCompileError> {
    let [partition] = executable.partitions.as_slice() else {
        return Err(AxCompileError::NativeRequiresSinglePartition);
    };
    let ExecutablePartitionKind::NativeCpu(native) = &partition.kind else {
        return Err(AxCompileError::NativeRequiresSinglePartition);
    };
    if partition.backend != CPU_BACKEND || partition.implementation != CPU_KERNEL_IMPLEMENTATION {
        return Err(AxCompileError::UnsupportedImplementation {
            backend: partition.backend,
            implementation: partition.implementation,
        });
    }
    ax_exec::native::prepare_native_in(
        native,
        executable.graph(),
        &executable.inputs,
        &executable.outputs,
        allocator,
    )
    .map(|inner| PreparedNative { inner })
    .map_err(Into::into)
}

fn compile_native_cpu_partition_in<A: Allocator + Clone>(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    partition: Partition,
    model: &AxDeviceModel,
    allocator: A,
) -> Result<ExecutablePartition<A>, AxCompileError> {
    if partition.backend != CPU_BACKEND || partition.implementation != CPU_KERNEL_IMPLEMENTATION {
        return Err(AxArtifactError::InvalidPartition.into());
    }
    let policy = model.cpu_kernels.ok_or(AxArtifactError::InvalidPartition)?;
    let native = ax_exec::native::compile_native_cpu_partition_in(
        graph,
        order,
        producers,
        partition,
        policy,
        model.native_fusions,
        allocator.clone(),
    )?;
    let mut inputs = Vec::new_in(allocator.clone());
    reserve(&mut inputs, native.inputs.len())?;
    inputs.extend_from_slice(&native.inputs);
    let mut outputs = Vec::new_in(allocator.clone());
    reserve(&mut outputs, native.outputs.len())?;
    outputs.extend_from_slice(&native.outputs);
    let mut input_transforms = Vec::new_in(allocator.clone());
    reserve(&mut input_transforms, inputs.len())?;
    input_transforms.resize(inputs.len(), AxInputTransform::Identity);
    let mut output_transforms = Vec::new_in(allocator.clone());
    reserve(&mut output_transforms, outputs.len())?;
    output_transforms.resize(outputs.len(), AxOutputTransform::Identity);
    let mut output_bytes = Vec::new_in(allocator);
    reserve(&mut output_bytes, outputs.len())?;
    for &output in &outputs {
        let tensor = graph
            .tensor(output)
            .map_err(|_| AxCompileError::MissingTensor(output))?;
        output_bytes
            .push(tensor_storage_bytes(tensor).ok_or(AxCompileError::MissingTensor(output))?);
    }
    Ok(ExecutablePartition {
        backend: CPU_BACKEND,
        implementation: CPU_KERNEL_IMPLEMENTATION,
        key: 0,
        inputs,
        outputs,
        input_transforms,
        output_transforms,
        output_bytes,
        kind: ExecutablePartitionKind::NativeCpu(native),
    })
}

fn compile_sparse_partition_in<A: Allocator + Clone>(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    partition: Partition,
    block_k: usize,
    allocator: A,
) -> Result<ExecutablePartition<A>, AxCompileError> {
    if partition.backend != CPU_BACKEND
        || !is_sparse_bsr_implementation(partition.implementation)
        || partition.node_count == 0
        || producers.len() < graph.tensor_count()
    {
        return Err(AxCompileError::InvalidSparsePartition);
    }
    let start = partition.order_start as usize;
    let end = start
        .checked_add(partition.node_count as usize)
        .filter(|&end| end <= order.len())
        .ok_or(AxCompileError::InvalidSparsePartition)?;
    let mut inside = Vec::new_in(allocator.clone());
    reserve(&mut inside, graph.node_count())?;
    inside.resize(graph.node_count(), 0_u8);
    for &node_index in &order[start..end] {
        *inside
            .get_mut(node_index as usize)
            .ok_or(AxCompileError::InvalidSparsePartition)? = 1;
    }

    let mut inputs = Vec::new_in(allocator.clone());
    let mut outputs = Vec::new_in(allocator.clone());
    reserve(&mut inputs, partition.node_count as usize)?;
    reserve(&mut outputs, partition.node_count as usize)?;
    let mut f32_nodes = Vec::new_in(allocator.clone());
    let mut fp8_nodes = Vec::new_in(allocator.clone());
    reserve(&mut f32_nodes, partition.node_count as usize)?;
    reserve(&mut fp8_nodes, partition.node_count as usize)?;

    for &node_index in &order[start..end] {
        let node = graph
            .node(node_index as usize)
            .map_err(|_| AxCompileError::InvalidSparsePartition)?;
        let info =
            sparse_bsr_matmul_info(graph, node).ok_or(AxCompileError::InvalidSparsePartition)?;
        if info.precision.implementation() != partition.implementation {
            return Err(AxCompileError::InvalidSparsePartition);
        }
        let activations = graph
            .tensor(info.input)
            .map_err(|_| AxCompileError::InvalidSparsePartition)?;
        let input_constant = activations
            .flags
            .contains(TensorFlags::CONSTANT)
            .then(|| copy_bytes_in(activations.constant(), allocator.clone()))
            .transpose()?;
        let producer = producers[info.input.0 as usize];
        let produced_inside = producer != ax::schedule::NO_PRODUCER
            && inside
                .get(producer as usize)
                .is_some_and(|&member| member != 0);
        if input_constant.is_none() && !produced_inside && !inputs.contains(&info.input) {
            inputs.push(info.input);
        }
        let result = graph
            .tensor(info.output)
            .map_err(|_| AxCompileError::InvalidSparsePartition)?;
        let consumed_outside = (0..graph.node_count()).any(|consumer| {
            inside[consumer] == 0
                && graph
                    .node(consumer)
                    .is_ok_and(|consumer| consumer.inputs().any(|input| input == info.output))
        });
        if (result.flags.contains(TensorFlags::OUTPUT) || consumed_outside)
            && !outputs.contains(&info.output)
        {
            outputs.push(info.output);
        }

        let profile = info.precision.profile(&info, block_k)?;
        let values_len = profile
            .packed_values()
            .ok_or(AxCompileError::InvalidSparsePartition)?;
        let row_offset_len = info
            .output_features
            .checked_add(1)
            .ok_or(AxCompileError::InvalidSparsePartition)?;
        let mut row_offsets = Vec::new_in(allocator.clone());
        reserve(&mut row_offsets, row_offset_len)?;
        row_offsets.resize(row_offset_len, 0);
        let mut column_indices = Vec::new_in(allocator.clone());
        reserve(&mut column_indices, profile.nonzero_blocks)?;
        column_indices.resize(profile.nonzero_blocks, 0);

        match info.precision {
            SparseBsrPrecision::F32 => {
                let mut values = Vec::new_in(allocator.clone());
                reserve(&mut values, values_len)?;
                values.resize(values_len, 0.0);
                axnn_cpu::sparse::pack_f32_le_transposed(
                    info.weights.constant(),
                    info.output_features,
                    info.input_features,
                    block_k,
                    &mut row_offsets,
                    &mut column_indices,
                    &mut values,
                )?;
                f32_nodes.push(PackedSparseF32 {
                    input: info.input,
                    output: info.output,
                    input_constant,
                    output_features: info.output_features,
                    input_features: info.input_features,
                    activation_rows: info.activation_rows,
                    block_k,
                    row_offsets,
                    column_indices,
                    values,
                });
            }
            precision @ (SparseBsrPrecision::B8 | SparseBsrPrecision::F8) => {
                let mut values = Vec::new_in(allocator.clone());
                reserve(&mut values, values_len)?;
                values.resize(values_len, 0);
                axnn_cpu::sparse::pack_fp8_transposed(
                    info.weights.constant(),
                    info.output_features,
                    info.input_features,
                    block_k,
                    &mut row_offsets,
                    &mut column_indices,
                    &mut values,
                )?;
                fp8_nodes.push(PackedSparseFp8 {
                    input: info.input,
                    output: info.output,
                    input_constant,
                    precision,
                    output_features: info.output_features,
                    input_features: info.input_features,
                    activation_rows: info.activation_rows,
                    block_k,
                    row_offsets,
                    column_indices,
                    values,
                });
            }
        }
    }
    if outputs.is_empty() {
        return Err(AxArtifactError::NoPartitionOutputs.into());
    }
    let mut input_transforms = Vec::new_in(allocator.clone());
    reserve(&mut input_transforms, inputs.len())?;
    input_transforms.resize(inputs.len(), AxInputTransform::Identity);
    let mut output_transforms = Vec::new_in(allocator.clone());
    reserve(&mut output_transforms, outputs.len())?;
    output_transforms.resize(outputs.len(), AxOutputTransform::Identity);
    let mut output_bytes = Vec::new_in(allocator.clone());
    reserve(&mut output_bytes, outputs.len())?;
    for &output in &outputs {
        let tensor = graph
            .tensor(output)
            .map_err(|_| AxCompileError::InvalidSparsePartition)?;
        output_bytes
            .push(tensor_storage_bytes(tensor).ok_or(AxCompileError::InvalidSparsePartition)?);
    }

    let kind = match partition.implementation {
        CPU_BSR_F32_IMPLEMENTATION if f32_nodes.len() == partition.node_count as usize => {
            ExecutablePartitionKind::SparseF32(f32_nodes)
        }
        CPU_BSR_B8_IMPLEMENTATION | CPU_BSR_F8_IMPLEMENTATION
            if fp8_nodes.len() == partition.node_count as usize =>
        {
            ExecutablePartitionKind::SparseFp8(fp8_nodes)
        }
        _ => return Err(AxCompileError::InvalidSparsePartition),
    };

    Ok(ExecutablePartition {
        backend: CPU_BACKEND,
        implementation: partition.implementation,
        key: 0,
        inputs,
        input_transforms,
        outputs,
        output_transforms,
        output_bytes,
        kind,
    })
}

fn run_sparse_f32_partition_in<P: Allocator, R: Allocator + Clone>(
    packed: &[PackedSparseF32<P>],
    boundary_values: &[Option<RuntimeValue<'_, R>>],
    output_ids: &[TensorId],
    allocator: R,
) -> Result<Vec<Vec<u8, R>, R>, AxCompileError> {
    let slots = packed
        .iter()
        .flat_map(|node| [node.input, node.output])
        .chain(output_ids.iter().copied())
        .map(|id| id.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let mut local = Vec::new_in(allocator.clone());
    reserve(&mut local, slots)?;
    local.resize_with(slots, || None);
    for node in packed {
        let bytes = {
            let source = match node.input_constant.as_deref() {
                Some(source) => source,
                None => local
                    .get(node.input.0 as usize)
                    .and_then(Option::as_deref)
                    .or_else(|| {
                        boundary_values
                            .get(node.input.0 as usize)
                            .and_then(Option::as_ref)
                            .map(RuntimeValue::as_slice)
                    })
                    .ok_or(AxCompileError::MissingTensor(node.input))?,
            };
            run_sparse_f32_node_in(node, source, allocator.clone())?
        };
        local[node.output.0 as usize] = Some(bytes);
    }
    let mut outputs = Vec::new_in(allocator);
    reserve(&mut outputs, output_ids.len())?;
    for &output in output_ids {
        outputs.push(
            local
                .get_mut(output.0 as usize)
                .and_then(Option::take)
                .ok_or(AxCompileError::MissingTensor(output))?,
        );
    }
    Ok(outputs)
}

fn run_sparse_f32_node_in<P: Allocator, R: Allocator + Clone>(
    packed: &PackedSparseF32<P>,
    source: &[u8],
    allocator: R,
) -> Result<Vec<u8, R>, AxCompileError> {
    let elements = packed
        .activation_rows
        .checked_mul(packed.input_features)
        .ok_or(AxCompileError::InvalidSparsePartition)?;
    let expected = elements
        .checked_mul(size_of::<f32>())
        .ok_or(AxCompileError::InvalidSparsePartition)?;
    if source.len() != expected {
        return Err(AxCompileError::InvalidInputSize {
            tensor: packed.input,
            expected,
            actual: source.len(),
        });
    }

    // The `.ax` ABI uses canonical byte planes.  Decode the dynamic input
    // into an aligned F32 scratch once, while the constant remains packed and
    // cached in its SIMD-friendly BSR form across every run.
    let mut activations = Vec::new_in(allocator.clone());
    reserve(&mut activations, elements)?;
    activations.resize(elements, 0.0);
    for (index, value) in activations.iter_mut().enumerate() {
        let offset = index * size_of::<f32>();
        *value = f32::from_le_bytes(
            source[offset..offset + size_of::<f32>()]
                .try_into()
                .unwrap(),
        );
    }
    let output_elements = packed
        .activation_rows
        .checked_mul(packed.output_features)
        .ok_or(AxCompileError::InvalidSparsePartition)?;
    let mut result = Vec::new_in(allocator.clone());
    reserve(&mut result, output_elements)?;
    result.resize(output_elements, 0.0);
    let matrix = axnn_cpu::sparse::BsrMatrix::new(
        packed.output_features,
        packed.input_features,
        packed.block_k,
        &packed.row_offsets,
        &packed.column_indices,
        &packed.values,
    )?;
    axnn_cpu::sparse::gemm_f32(matrix, &activations, &mut result, packed.activation_rows)?;

    let output_len = output_elements
        .checked_mul(size_of::<f32>())
        .ok_or(AxCompileError::InvalidSparsePartition)?;
    let mut bytes = Vec::new_in(allocator.clone());
    reserve(&mut bytes, output_len)?;
    bytes.resize(output_len, 0);
    for (index, value) in result.iter().enumerate() {
        let offset = index * size_of::<f32>();
        bytes[offset..offset + size_of::<f32>()].copy_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn run_sparse_fp8_partition_in<P: Allocator, R: Allocator + Clone>(
    packed: &[PackedSparseFp8<P>],
    boundary_values: &[Option<RuntimeValue<'_, R>>],
    output_ids: &[TensorId],
    allocator: R,
) -> Result<Vec<Vec<u8, R>, R>, AxCompileError> {
    let slots = packed
        .iter()
        .flat_map(|node| [node.input, node.output])
        .chain(output_ids.iter().copied())
        .map(|id| id.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let mut local = Vec::new_in(allocator.clone());
    reserve(&mut local, slots)?;
    local.resize_with(slots, || None);
    for node in packed {
        let bytes = {
            let source = match node.input_constant.as_deref() {
                Some(source) => source,
                None => local
                    .get(node.input.0 as usize)
                    .and_then(Option::as_deref)
                    .or_else(|| {
                        boundary_values
                            .get(node.input.0 as usize)
                            .and_then(Option::as_ref)
                            .map(RuntimeValue::as_slice)
                    })
                    .ok_or(AxCompileError::MissingTensor(node.input))?,
            };
            run_sparse_fp8_node_in(node, source, allocator.clone())?
        };
        local[node.output.0 as usize] = Some(bytes);
    }
    let mut outputs = Vec::new_in(allocator);
    reserve(&mut outputs, output_ids.len())?;
    for &output in output_ids {
        outputs.push(
            local
                .get_mut(output.0 as usize)
                .and_then(Option::take)
                .ok_or(AxCompileError::MissingTensor(output))?,
        );
    }
    Ok(outputs)
}

fn run_sparse_fp8_node_in<P: Allocator, R: Allocator + Clone>(
    packed: &PackedSparseFp8<P>,
    source: &[u8],
    allocator: R,
) -> Result<Vec<u8, R>, AxCompileError> {
    let elements = packed
        .activation_rows
        .checked_mul(packed.input_features)
        .ok_or(AxCompileError::InvalidSparsePartition)?;
    if source.len() != elements {
        return Err(AxCompileError::InvalidInputSize {
            tensor: packed.input,
            expected: elements,
            actual: source.len(),
        });
    }

    let output_elements = packed
        .activation_rows
        .checked_mul(packed.output_features)
        .ok_or(AxCompileError::InvalidSparsePartition)?;
    let mut result = Vec::new_in(allocator.clone());
    reserve(&mut result, output_elements)?;
    result.resize(output_elements, 0.0);
    let matrix = axnn_cpu::sparse::BsrMatrix::new(
        packed.output_features,
        packed.input_features,
        packed.block_k,
        &packed.row_offsets,
        &packed.column_indices,
        &packed.values,
    )?;
    match packed.precision {
        SparseBsrPrecision::B8 => axnn_cpu::sparse::gemm_e4m3_e4m3_f32(
            matrix,
            source,
            &mut result,
            packed.activation_rows,
        )?,
        SparseBsrPrecision::F8 => axnn_cpu::sparse::gemm_e5m2_e5m2_f32(
            matrix,
            source,
            &mut result,
            packed.activation_rows,
        )?,
        SparseBsrPrecision::F32 => return Err(AxCompileError::InvalidSparsePartition),
    }

    let output_len = output_elements
        .checked_mul(size_of::<u16>())
        .ok_or(AxCompileError::InvalidSparsePartition)?;
    let mut bytes = Vec::new_in(allocator.clone());
    reserve(&mut bytes, output_len)?;
    bytes.resize(output_len, 0);
    for (index, value) in result.iter().enumerate() {
        let offset = index * size_of::<u16>();
        bytes[offset..offset + size_of::<u16>()]
            .copy_from_slice(&(*value as f16).to_bits().to_le_bytes());
    }
    Ok(bytes)
}

const NO_BOOLEAN_COMPONENT: u32 = u32::MAX;

struct BooleanPlacement<A: Allocator> {
    backend: u16,
    accelerator_forbidden: Vec<u8, A>,
    /// Root node for every Boolean-connected node; other nodes use the
    /// `NO_BOOLEAN_COMPONENT` sentinel.
    components: Vec<u32, A>,
}

impl<A: Allocator> BooleanPlacement<A> {
    /// Preserve the provider boundary contract after cost refinement. An
    /// internal component may stay accelerated only when every participating
    /// node did; otherwise the whole component returns to the CPU.
    fn normalize(&mut self, schedule: &Schedule<A>) -> bool {
        let mut changed = false;
        for (node, &root) in self.components.iter().enumerate() {
            if root != NO_BOOLEAN_COMPONENT && schedule.placements()[node] != self.backend {
                let root = root as usize;
                changed |= self.accelerator_forbidden[root] == 0;
                self.accelerator_forbidden[root] = 1;
            }
        }
        for (node, &root) in self.components.iter().enumerate() {
            if root != NO_BOOLEAN_COMPONENT && self.accelerator_forbidden[root as usize] != 0 {
                changed |= self.accelerator_forbidden[node] == 0;
                self.accelerator_forbidden[node] = 1;
            }
        }
        changed
    }

    /// Backend-atomic placement is not sufficient when two accelerator runs
    /// are separated by another implementation/backend. Inspect the finalized
    /// executable partitions and forbid every Boolean component whose U8 edge
    /// would become an unsupported accelerator block boundary.
    fn normalize_partition_crossings(
        &mut self,
        graph: &Graph<'_>,
        schedule: &Schedule<A>,
    ) -> Result<bool, AxCompileError> {
        let mut changed = false;
        for tensor_index in 0..graph.tensor_count() {
            let tensor_id = TensorId(tensor_index as u32);
            let tensor = graph
                .tensor(tensor_id)
                .map_err(|_| AxCompileError::MissingTensor(tensor_id))?;
            if tensor.dtype != DType::U8 || tensor.flags.contains(TensorFlags::CONSTANT) {
                continue;
            }
            let producer = schedule.producers()[tensor_index];
            let producer_partition = (producer != ax::schedule::NO_PRODUCER)
                .then(|| schedule_node_partition(schedule, producer as usize))
                .transpose()?;
            for consumer_index in 0..graph.node_count() {
                let consumer = graph
                    .node(consumer_index)
                    .map_err(|_| AxArtifactError::InvalidGraph)?;
                if !consumer.inputs().any(|input| input == tensor_id) {
                    continue;
                }
                let consumer_partition = schedule_node_partition(schedule, consumer_index)?;
                let crossing =
                    producer_partition.is_none_or(|(index, _)| index != consumer_partition.0);
                let accelerator_boundary = producer_partition
                    .is_some_and(|(_, backend)| backend == self.backend)
                    || consumer_partition.1 == self.backend;
                if crossing && accelerator_boundary {
                    let component_node = if producer != ax::schedule::NO_PRODUCER {
                        producer as usize
                    } else {
                        consumer_index
                    };
                    changed |= self.forbid_component(component_node);
                }
            }
            if tensor.flags.contains(TensorFlags::OUTPUT)
                && producer_partition.is_some_and(|(_, backend)| backend == self.backend)
            {
                changed |= self.forbid_component(producer as usize);
            }
        }
        if changed {
            self.propagate_forbidden();
        }
        Ok(changed)
    }

    fn forbid_component(&mut self, node: usize) -> bool {
        let Some(&root) = self.components.get(node) else {
            return false;
        };
        if root == NO_BOOLEAN_COMPONENT {
            return false;
        }
        let slot = &mut self.accelerator_forbidden[root as usize];
        let changed = *slot == 0;
        *slot = 1;
        changed
    }

    fn propagate_forbidden(&mut self) {
        for (node, &root) in self.components.iter().enumerate() {
            if root != NO_BOOLEAN_COMPONENT && self.accelerator_forbidden[root as usize] != 0 {
                self.accelerator_forbidden[node] = 1;
            }
        }
    }
}

fn schedule_node_partition<A: Allocator>(
    schedule: &Schedule<A>,
    node: usize,
) -> Result<(usize, u16), AxCompileError> {
    let position = schedule
        .order()
        .iter()
        .position(|&candidate| candidate as usize == node)
        .ok_or(AxArtifactError::InvalidPartition)?;
    schedule
        .plan()
        .partitions
        .iter()
        .enumerate()
        .find_map(|(index, partition)| {
            let start = partition.order_start as usize;
            let end = start.checked_add(partition.node_count as usize)?;
            (position >= start && position < end).then_some((index, partition.backend))
        })
        .ok_or_else(|| AxArtifactError::InvalidPartition.into())
}

fn forbid_unsupported_accelerator_partitions<A: Allocator>(
    graph: &Graph<'_>,
    schedule: &Schedule<A>,
    model: &AxDeviceModel,
    backend: u16,
    accelerator_forbidden: &mut [u8],
) -> Result<bool, AxCompileError> {
    let Some(capability) = model.accelerator_tosa_capability else {
        return Ok(false);
    };
    let mut changed = false;
    for partition in schedule.plan().partitions {
        if partition.backend != backend {
            continue;
        }
        if lowering::capability_supports_partition(
            capability,
            graph,
            schedule.order(),
            schedule.producers(),
            *partition,
            TosaProfile::Accelerator,
            model.f32_boolean_boundary_bridge,
            model.fp8_promotion,
        ) {
            continue;
        }
        let start = partition.order_start as usize;
        let end = start
            .checked_add(partition.node_count as usize)
            .filter(|&end| end <= schedule.order().len())
            .ok_or(AxArtifactError::InvalidPartition)?;
        for &node in &schedule.order()[start..end] {
            let slot = accelerator_forbidden
                .get_mut(node as usize)
                .ok_or(AxArtifactError::InvalidPartition)?;
            changed |= *slot == 0;
            *slot = 1;
        }
    }
    Ok(changed)
}

/// Analyze Boolean-connected node components when the provider can neither
/// expose native Boolean model boundaries nor bridge them through an admitted
/// dtype. Components tied to graph I/O or an unsupported accelerator node stay
/// on the CPU. A fully supported internal island remains eligible, then
/// [`BooleanPlacement::normalize`] makes its final placement atomic so
/// partitioning cannot manufacture an invalid Boolean transfer.
fn boolean_placement_in<A: Allocator + Clone>(
    graph: &Graph<'_>,
    schedule: &Schedule<A>,
    model: &AxDeviceModel,
    allocator: A,
) -> Result<BooleanPlacement<A>, AllocationError> {
    let nodes = graph.node_count();
    let tensors = graph.tensor_count();
    let mut blocked = Vec::new_in(allocator.clone());
    blocked
        .try_reserve_exact(nodes)
        .map_err(|_| AllocationError::AllocationFailed)?;
    blocked.resize(nodes, 0);
    let mut parent = Vec::new_in(allocator.clone());
    parent
        .try_reserve_exact(nodes)
        .map_err(|_| AllocationError::AllocationFailed)?;
    parent.resize(nodes, NO_BOOLEAN_COMPONENT);
    if model.accelerator_boolean_boundaries
        || model.f32_boolean_boundary_bridge
        || !model.accelerator
        || nodes == 0
    {
        return Ok(BooleanPlacement {
            backend: ACCEL_BACKEND,
            accelerator_forbidden: blocked,
            components: parent,
        });
    }

    let mut rank = Vec::new_in(allocator.clone());
    rank.try_reserve_exact(nodes)
        .map_err(|_| AllocationError::AllocationFailed)?;
    rank.resize(nodes, 0_u8);
    let mut owners = Vec::new_in(allocator.clone());
    owners
        .try_reserve_exact(tensors)
        .map_err(|_| AllocationError::AllocationFailed)?;
    owners.resize(tensors, ax::schedule::NO_PRODUCER);

    for (tensor_index, owner) in owners.iter_mut().enumerate() {
        let tensor = graph
            .tensor(TensorId(tensor_index as u32))
            .map_err(AllocationError::from)?;
        if tensor.dtype == DType::U8 {
            *owner = schedule.producers()[tensor_index];
            if *owner != ax::schedule::NO_PRODUCER {
                activate_component(&mut parent, *owner);
            }
        }
    }

    for node_index in 0..nodes {
        let node = graph.node(node_index).map_err(AllocationError::from)?;
        for tensor_id in node.inputs() {
            let tensor = graph.tensor(tensor_id).map_err(AllocationError::from)?;
            if tensor.dtype != DType::U8 {
                continue;
            }
            activate_component(&mut parent, node_index as u32);
            let owner = &mut owners[tensor_id.0 as usize];
            if *owner == ax::schedule::NO_PRODUCER {
                *owner = node_index as u32;
            } else {
                union_components(&mut parent, &mut rank, *owner, node_index as u32);
            }
        }
    }

    let mut seeded = Vec::new_in(allocator);
    seeded
        .try_reserve_exact(nodes)
        .map_err(|_| AllocationError::AllocationFailed)?;
    seeded.resize(nodes, 0_u8);
    for node in 0..nodes {
        if parent[node] == NO_BOOLEAN_COMPONENT {
            continue;
        }
        let root = component_root(&mut parent, node as u32);
        parent[node] = root;
        let view = graph.node(node).map_err(AllocationError::from)?;
        if !model.supports(ACCEL_BACKEND, graph, view) {
            seeded[root as usize] = 1;
        }
    }
    for (tensor_index, &owner) in owners.iter().enumerate() {
        if owner == ax::schedule::NO_PRODUCER {
            continue;
        }
        let tensor = graph
            .tensor(TensorId(tensor_index as u32))
            .map_err(AllocationError::from)?;
        if tensor.dtype == DType::U8
            && (tensor.flags.contains(TensorFlags::INPUT)
                || tensor.flags.contains(TensorFlags::OUTPUT))
        {
            let root = component_root(&mut parent, owner);
            seeded[root as usize] = 1;
        }
    }
    for (node, slot) in blocked.iter_mut().enumerate() {
        let root = parent[node];
        if root != NO_BOOLEAN_COMPONENT {
            *slot = seeded[root as usize];
        }
    }
    Ok(BooleanPlacement {
        backend: ACCEL_BACKEND,
        accelerator_forbidden: blocked,
        components: parent,
    })
}

fn activate_component(parent: &mut [u32], node: u32) {
    let slot = &mut parent[node as usize];
    if *slot == NO_BOOLEAN_COMPONENT {
        *slot = node;
    }
}

fn component_root(parent: &mut [u32], node: u32) -> u32 {
    let index = node as usize;
    let ancestor = parent[index];
    if ancestor == node {
        return node;
    }
    let root = component_root(parent, ancestor);
    parent[index] = root;
    root
}

fn union_components(parent: &mut [u32], rank: &mut [u8], left: u32, right: u32) {
    let mut left = component_root(parent, left);
    let mut right = component_root(parent, right);
    if left == right {
        return;
    }
    if rank[left as usize] < rank[right as usize] {
        core::mem::swap(&mut left, &mut right);
    }
    parent[right as usize] = left;
    if rank[left as usize] == rank[right as usize] {
        rank[left as usize] = rank[left as usize].saturating_add(1);
    }
}

fn reserve<T, A: Allocator>(
    values: &mut Vec<T, A>,
    additional: usize,
) -> Result<(), AxCompileError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| AllocationError::AllocationFailed.into())
}

fn copy_bytes_in<A: Allocator>(bytes: &[u8], allocator: A) -> Result<Vec<u8, A>, AxCompileError> {
    let mut copy = Vec::new_in(allocator);
    reserve(&mut copy, bytes.len())?;
    copy.extend_from_slice(bytes);
    Ok(copy)
}

fn transform_input_in<A: Allocator>(
    transform: AxInputTransform,
    source: &[u8],
    allocator: A,
) -> Result<Vec<u8, A>, AxCompileError> {
    if transform == AxInputTransform::Identity {
        return copy_bytes_in(source, allocator);
    }
    let element_bytes = match transform {
        AxInputTransform::B8ToF16 | AxInputTransform::F8ToF16 => 2,
        AxInputTransform::BoolU8ToF32 => 4,
        AxInputTransform::Identity => unreachable!(),
    };
    let bytes = source
        .len()
        .checked_mul(element_bytes)
        .ok_or(AllocationError::AllocationFailed)?;
    let mut expanded = Vec::new_in(allocator);
    reserve(&mut expanded, bytes)?;
    expanded.resize(bytes, 0);
    match transform {
        AxInputTransform::B8ToF16 => {
            axnn_cpu::fp8::dequant_e4m3_f16(source, &mut expanded, source.len())
        }
        AxInputTransform::F8ToF16 => {
            axnn_cpu::fp8::dequant_e5m2_f16(source, &mut expanded, source.len())
        }
        AxInputTransform::BoolU8ToF32 => {
            for (value, output) in source.iter().zip(expanded.as_chunks_mut::<4>().0) {
                output.copy_from_slice(&f32::from(*value != 0).to_le_bytes());
            }
        }
        AxInputTransform::Identity => unreachable!(),
    }
    Ok(expanded)
}

fn transform_output_in<A: Allocator>(
    transform: AxOutputTransform,
    tensor: TensorId,
    mut source: Vec<u8, A>,
) -> Result<Vec<u8, A>, AxCompileError> {
    if transform == AxOutputTransform::Identity {
        return Ok(source);
    }
    if !source.len().is_multiple_of(4) {
        return Err(AxCompileError::InvalidBooleanBoundary { tensor });
    }
    let elements = source.len() / 4;
    for index in 0..elements {
        let start = index * 4;
        let value = f32::from_le_bytes(
            source[start..start + 4]
                .try_into()
                .map_err(|_| AxCompileError::InvalidBooleanBoundary { tensor })?,
        );
        source[index] = if value == 0.0 {
            0
        } else if value == 1.0 {
            1
        } else {
            return Err(AxCompileError::InvalidBooleanBoundary { tensor });
        };
    }
    source.truncate(elements);
    Ok(source)
}

fn accel_supports(graph: &Graph<'_>, node: NodeView<'_>, model: &AxDeviceModel) -> bool {
    // BF16 operands with an FP32 accumulator are a distinct, explicit TOSA
    // row. Only advertise it when the actual provider declares that contract;
    // the legacy default FP16/FP32 profile must not silently broaden.
    if node.op == OpCode::MATMUL
        && matmul_dtypes(graph, node) == Some((DType::B16, DType::B16, DType::F32))
        && model.accelerator_tosa_capability.is_none()
    {
        return false;
    }
    // FP8 is an explicit tier for the same reason as the BF16 row above: the accelerator
    // profile admits it so a provider's descriptor can have a say, but with no descriptor there
    // is nothing that has said so, and FP8 belongs on the CPU. This gates only the native path --
    // the `fp8_promotion` route below deliberately serves accelerators that declare no FP8 tier.
    let touches_fp8 = node.inputs().chain(node.outputs()).any(|id| {
        graph
            .tensor(id)
            .is_ok_and(|tensor| matches!(tensor.dtype, DType::B8 | DType::F8))
    });
    let native_declared = !(touches_fp8 && model.accelerator_tosa_capability.is_none());
    if accelerator_native_supports(graph, node)
        && native_declared
        && model.accelerator_tosa_capability.is_none_or(|capability| {
            lowering::capability_supports_node(
                capability,
                graph,
                node,
                TosaProfile::Accelerator,
                model.f32_boolean_boundary_bridge,
                false,
            )
        })
    {
        return true;
    }
    if !model.fp8_promotion || node.op != OpCode::MATMUL {
        return false;
    }
    let mut inputs = node.inputs();
    let Some(left) = inputs.next().and_then(|id| graph.tensor(id).ok()) else {
        return false;
    };
    let Some(right) = inputs.next().and_then(|id| graph.tensor(id).ok()) else {
        return false;
    };
    if inputs.next().is_some()
        || left.dtype != right.dtype
        || !matches!(left.dtype, DType::B8 | DType::F8)
    {
        return false;
    }
    let supported_output = node.outputs().all(|id| {
        graph
            .tensor(id)
            .is_ok_and(|tensor| tensor.dtype == DType::F16)
    });
    supported_output
        && model.accelerator_tosa_capability.is_none_or(|capability| {
            lowering::capability_supports_node(
                capability,
                graph,
                node,
                TosaProfile::Accelerator,
                model.f32_boolean_boundary_bridge,
                true,
            )
        })
}

fn cpu_tosa_supports(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if let Some(request) = ax::node_kernel_request(graph, node) {
        return lowering::select_tosa_recipe(&request, TosaProfile::Cpu).is_some();
    }
    if node.op == OpCode::CAST {
        return unary_dtypes(graph, node).is_some_and(|(input, output)| {
            (float_storage_dtype(input) && float_storage_dtype(output))
                || (compute_float_dtype(input) && output == DType::I8)
        });
    }
    if node.op == OpCode::MATMUL {
        return matmul_dtypes(graph, node).is_some_and(|(left, right, output)| {
            matches!(
                (left, right, output),
                (DType::F32, DType::F32, DType::F32)
                    | (DType::F16, DType::F16, DType::F16)
                    | (DType::B8, DType::B8, DType::F16)
                    | (DType::F8, DType::F8, DType::F16)
                    | (DType::I8, DType::I8, DType::I32)
            )
        });
    }
    false
}

fn accelerator_native_supports(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if let Some(request) = ax::node_kernel_request(graph, node) {
        return lowering::select_tosa_recipe(&request, TosaProfile::Accelerator).is_some();
    }
    if node.op == OpCode::CAST {
        return unary_dtypes(graph, node).is_some_and(|(input, output)| {
            compute_float_dtype(input) && (compute_float_dtype(output) || output == DType::I8)
        });
    }
    if node.op == OpCode::MATMUL {
        return matmul_dtypes(graph, node).is_some_and(|(left, right, output)| {
            matches!(
                (left, right, output),
                (DType::F32, DType::F32, DType::F32)
                    | (DType::F16, DType::F16, DType::F16)
                    | (DType::B16, DType::B16, DType::F32)
                    // TOSA's FP8 MATMUL row. Like the BF16 row it is a declared tier, not a
                    // default: `accel_supports` refuses it when the provider advertises no
                    // capability descriptor, so listing it here only lets a declaring provider
                    // be asked.
                    | (DType::B8, DType::B8, DType::F16)
                    | (DType::F8, DType::F8, DType::F16)
            )
        });
    }
    false
}

const fn compute_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16)
}

const fn float_storage_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::B8 | DType::F8)
}

fn tensor_dtype(graph: &Graph<'_>, tensor: Option<TensorId>) -> Option<DType> {
    graph.tensor(tensor?).ok().map(|tensor| tensor.dtype)
}

fn unary_dtypes(graph: &Graph<'_>, node: NodeView<'_>) -> Option<(DType, DType)> {
    Some((
        tensor_dtype(graph, node.inputs().next())?,
        tensor_dtype(graph, node.outputs().next())?,
    ))
}

fn binary_dtypes(graph: &Graph<'_>, node: NodeView<'_>) -> Option<(DType, DType, DType)> {
    let mut inputs = node.inputs();
    Some((
        tensor_dtype(graph, inputs.next())?,
        tensor_dtype(graph, inputs.next())?,
        tensor_dtype(graph, node.outputs().next())?,
    ))
}

fn matmul_dtypes(graph: &Graph<'_>, node: NodeView<'_>) -> Option<(DType, DType, DType)> {
    binary_dtypes(graph, node)
}

fn node_work(graph: &Graph<'_>, node: NodeView<'_>) -> Option<u64> {
    if node.op == OpCode::MATMUL {
        let mut inputs = node.inputs();
        let left = graph.tensor(inputs.next()?).ok()?;
        let right = graph.tensor(inputs.next()?).ok()?;
        let (batch, m, k) = trailing_matrix_shape(left)?;
        let (_, right_k, n) = trailing_matrix_shape(right)?;
        if k != right_k {
            return None;
        }
        return batch
            .checked_mul(m)?
            .checked_mul(k)?
            .checked_mul(n)?
            .checked_mul(2);
    }
    let output = node.outputs().next()?;
    let tensor = graph.tensor(output).ok()?;
    let elements = tensor.dimensions().try_fold(1_u64, |total, dim| {
        (dim != ax::DYNAMIC_DIM)
            .then_some(())
            .and_then(|()| total.checked_mul(u64::from(dim)))
    })?;
    Some(if matches!(node.op, OpCode::TANH | OpCode::SIGMOID) {
        elements.saturating_mul(16)
    } else {
        elements
    })
}

/// Return `(batch, rows, columns)` without materializing a dimensions vector.
fn trailing_matrix_shape(tensor: TensorView<'_>) -> Option<(u64, u64, u64)> {
    let mut batch = 1_u64;
    let mut rows = None;
    let mut columns = None;
    for dimension in tensor.dimensions() {
        if dimension == ax::DYNAMIC_DIM {
            return None;
        }
        if let Some(previous_rows) = rows {
            batch = batch.checked_mul(previous_rows)?;
        }
        rows = columns;
        columns = Some(u64::from(dimension));
    }
    Some((batch, rows?, columns?))
}

/// Numeric storage/accumulation contract of one native sparse matmul.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SparseBsrPrecision {
    F32,
    B8,
    F8,
}

impl SparseBsrPrecision {
    const fn implementation(self) -> Implementation {
        match self {
            Self::F32 => CPU_BSR_F32_IMPLEMENTATION,
            Self::B8 => CPU_BSR_B8_IMPLEMENTATION,
            Self::F8 => CPU_BSR_F8_IMPLEMENTATION,
        }
    }

    fn profile(
        self,
        info: &SparseBsrMatmulInfo<'_>,
        block_k: usize,
    ) -> Result<axnn_cpu::sparse::BlockSparseProfile, axnn_cpu::sparse::SparseError> {
        match self {
            Self::F32 => axnn_cpu::sparse::profile_f32_le_transposed(
                info.weights.constant(),
                info.output_features,
                info.input_features,
                block_k,
            ),
            Self::B8 | Self::F8 => axnn_cpu::sparse::profile_fp8_transposed(
                info.weights.constant(),
                info.output_features,
                info.input_features,
                block_k,
            ),
        }
    }
}

const fn is_sparse_bsr_implementation(implementation: Implementation) -> bool {
    matches!(
        implementation,
        CPU_BSR_F32_IMPLEMENTATION | CPU_BSR_B8_IMPLEMENTATION | CPU_BSR_F8_IMPLEMENTATION
    )
}

/// Static-shape dense matmul that can be compiled as one CPU BSR kernel.
///
/// The graph continues to describe the conventional `A × B` operation.
/// `B[input, output]` is transposed during packing into the kernel's preferred
/// `W[output, input]` form, so this optimization never changes graph-visible
/// tensor layout. F32 stays F32; F8 accumulates in F32 and narrows to the
/// graph's F16 result just as the dense TOSA CPU path does.
#[derive(Clone, Copy)]
struct SparseBsrMatmulInfo<'a> {
    input: TensorId,
    output: TensorId,
    weights: TensorView<'a>,
    precision: SparseBsrPrecision,
    activation_rows: usize,
    input_features: usize,
    output_features: usize,
}

fn sparse_bsr_matmul_info<'a>(
    graph: &Graph<'a>,
    node: NodeView<'a>,
) -> Option<SparseBsrMatmulInfo<'a>> {
    if node.op != OpCode::MATMUL || !node.attributes().is_empty() || node.output_count() != 1 {
        return None;
    }
    let mut inputs = node.inputs();
    let input = inputs.next()?;
    let weights_id = inputs.next()?;
    if inputs.next().is_some() {
        return None;
    }
    let output = node.outputs().next()?;
    let activations = graph.tensor(input).ok()?;
    let weights = graph.tensor(weights_id).ok()?;
    let result = graph.tensor(output).ok()?;
    let precision = match (activations.dtype, weights.dtype, result.dtype) {
        (DType::F32, DType::F32, DType::F32) => SparseBsrPrecision::F32,
        (DType::B8, DType::B8, DType::F16) => SparseBsrPrecision::B8,
        (DType::F8, DType::F8, DType::F16) => SparseBsrPrecision::F8,
        _ => return None,
    };
    if !weights.flags.contains(TensorFlags::CONSTANT) {
        return None;
    }
    let (activation_batch, activation_rows, input_features) = trailing_matrix_shape(activations)?;
    let (weight_batch, weight_input_features, output_features) = trailing_matrix_shape(weights)?;
    let (output_batch, output_rows, output_columns) = trailing_matrix_shape(result)?;
    // The first direct executor slice supports the common immutable MLP
    // weight form: a single shared batch plane.  Batched weight constants can
    // follow once the packed cache carries one BSR matrix per batch.
    if activation_batch != 1
        || weight_batch != 1
        || output_batch != 1
        || activation_rows != output_rows
        || input_features != weight_input_features
        || output_features != output_columns
    {
        return None;
    }
    Some(SparseBsrMatmulInfo {
        input,
        output,
        weights,
        precision,
        activation_rows: usize::try_from(activation_rows).ok()?,
        input_features: usize::try_from(input_features).ok()?,
        output_features: usize::try_from(output_features).ok()?,
    })
}

fn tensor_storage_bytes(tensor: TensorView<'_>) -> Option<usize> {
    ax_exec::native::tensor_storage_bytes(tensor)
}

#[cfg(test)]
fn native_f32<A: Allocator>(
    values: &[Option<NativeValue<A>>],
    slot: usize,
) -> Result<&[f32], AxCompileError> {
    match values.get(slot).and_then(Option::as_ref) {
        Some(NativeValue::F32(values)) => Ok(values),
        _ => Err(AxArtifactError::InvalidPartition.into()),
    }
}

#[cfg(test)]
mod boundary_transform_tests {
    use core::alloc::{AllocError, Allocator, Layout};
    use core::cell::Cell;
    use core::ptr::NonNull;
    use std::alloc::Global;

    use crate::{AxHetero, AxTosaAccelerator, Session};
    use ax::{GraphBuilder, NodeFlags};

    use super::{
        ACCEL_BACKEND, AxCompileError, AxDeviceModel, AxInputTransform, AxOutputTransform,
        AxPlacementModel, CPU_BACKEND, CPU_KERNEL_IMPLEMENTATION, DENSE_IMPLEMENTATION, DType,
        ExecutablePartitionKind, OpCode, Schedule, TensorFlags, TensorId, boolean_placement_in,
        compile_native_cpu_partition_in, copy_bytes_in, transform_input_in, transform_output_in,
    };

    #[derive(Default)]
    struct PreparedAllocationCounts {
        allocations: Cell<usize>,
        live_bytes: Cell<usize>,
        deny: Cell<bool>,
        fail_at: Cell<Option<usize>>,
    }

    #[derive(Clone, Copy)]
    struct PreparedAllocator<'a>(&'a PreparedAllocationCounts);

    // SAFETY: Each successful allocation and deallocation delegates to Global
    // with the same layout; copied handles share only accounting state.
    unsafe impl Allocator for PreparedAllocator<'_> {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            self.0.allocations.set(self.0.allocations.get() + 1);
            if self.0.deny.get()
                || self
                    .0
                    .fail_at
                    .get()
                    .is_some_and(|limit| self.0.allocations.get() >= limit)
            {
                return Err(AllocError);
            }
            let allocation = Global.allocate(layout)?;
            self.0
                .live_bytes
                .set(self.0.live_bytes.get() + layout.size());
            Ok(allocation)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            self.0
                .live_bytes
                .set(self.0.live_bytes.get() - layout.size());
            // SAFETY: This allocation came from Global with this layout.
            unsafe { Global.deallocate(ptr, layout) };
        }
    }

    fn prepared_test_runtime() -> AxHetero<AxTosaAccelerator, AxTosaAccelerator> {
        let cpu = Session::new(AxTosaAccelerator::new(), AxTosaAccelerator::target()).unwrap();
        AxHetero::new(cpu, None)
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn native_ingress_borrows_alignment_and_decodes_unaligned_exact_bits() {
        use super::{NativeRef, NativeSlot, NativeValue, RuntimeValue, ensure_native_value_in};
        let mut builder = GraphBuilder::new_in(Global);
        let id = builder
            .tensor(DType::I32, TensorFlags::INPUT, &[5], &[])
            .unwrap();
        let graph_bytes = builder.finish().unwrap();
        let graph = graph_bytes.parse().unwrap();
        let input = NativeValue::I32(vec![i32::MIN, -1, 0, 16_777_217, i32::MAX]);
        let counts = PreparedAllocationCounts::default();
        counts.deny.set(true);
        let boundaries = [Some(RuntimeValue::Borrowed(input.as_native_bytes()))];
        let mut slots = [None];
        let mut copies = super::AxHostCopyStats::default();
        ensure_native_value_in(
            id,
            0,
            &graph,
            &boundaries,
            &mut slots,
            &mut copies,
            PreparedAllocator(&counts),
        )
        .unwrap();
        let Some(NativeSlot::Borrowed(NativeRef::I32(view))) = slots[0].as_ref() else {
            panic!("aligned ingress was copied")
        };
        assert_eq!(view.as_ptr().cast::<u8>(), input.as_native_bytes().as_ptr());
        assert_eq!(counts.allocations.get(), 0);
        assert_eq!(copies.native_ingress_bytes, 0);

        let mut misaligned = NativeValue::I32(vec![0; 6]);
        misaligned.as_native_bytes_mut()[1..21].copy_from_slice(input.as_native_bytes());
        let boundaries = [Some(RuntimeValue::Borrowed(
            &misaligned.as_native_bytes()[1..21],
        ))];
        let mut slots = [None];
        assert!(
            ensure_native_value_in(
                id,
                0,
                &graph,
                &boundaries,
                &mut slots,
                &mut copies,
                PreparedAllocator(&counts)
            )
            .is_err()
        );
        assert!(slots[0].is_none());
        counts.deny.set(false);
        ensure_native_value_in(
            id,
            0,
            &graph,
            &boundaries,
            &mut slots,
            &mut copies,
            PreparedAllocator(&counts),
        )
        .unwrap();
        let Some(NativeSlot::Owned(value)) = slots[0].as_ref() else {
            panic!("unaligned ingress was not decoded")
        };
        assert_eq!(value.as_native_bytes(), input.as_native_bytes());
        assert_eq!(copies.native_ingress_bytes, 20);
        drop(slots);
        assert_eq!(counts.live_bytes.get(), 0);
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn retained_outputs_keep_native_or_provider_typed_backing_and_passthrough_borrows() {
        use super::{AxInvocationStatus, NativeValue, RuntimeValue};
        let mut builder = GraphBuilder::new_in(Global);
        let input_id = builder
            .tensor(
                DType::F32,
                TensorFlags::INPUT | TensorFlags::OUTPUT,
                &[5],
                &[],
            )
            .unwrap();
        let copied = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[5], &[])
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
        let graph_bytes = builder.finish().unwrap();
        let input = NativeValue::F32(vec![
            0.0,
            -0.0,
            f32::INFINITY,
            f32::from_bits(0x7fc01234),
            -1.5,
        ]);
        for provider in [false, true] {
            let mut runtime = prepared_test_runtime();
            let model = if provider {
                AxDeviceModel {
                    accelerator: false,
                    cpu_kernels: None,
                    sparse_bsr: None,
                    ..AxDeviceModel::default()
                }
            } else {
                prepared_test_model()
            };
            let executable = runtime
                .compile_in(&graph_bytes.parse().unwrap(), &model, Global)
                .unwrap();
            let mut invocation = runtime
                .begin_in(&executable, &[input.as_native_bytes()], Global)
                .unwrap();
            let outputs = loop {
                match runtime.advance_retained_in(&mut invocation).unwrap() {
                    AxInvocationStatus::Complete(outputs) => break outputs,
                    AxInvocationStatus::Progress(_) => {}
                    other => panic!("unexpected {other:?}"),
                }
            };
            assert_eq!(
                invocation.host_copy_stats(),
                super::AxHostCopyStats::default()
            );
            assert_eq!(
                outputs[0].as_bytes().as_ptr(),
                input.as_native_bytes().as_ptr()
            );
            let RuntimeValue::Native(value) = &outputs[1].value else {
                panic!("typed output was materialized")
            };
            assert_eq!(
                outputs[1].as_bytes().as_ptr(),
                value.as_native_bytes().as_ptr()
            );
            assert_eq!(outputs[1].as_bytes(), input.as_native_bytes());
            // The legacy conversion remains explicit and bit-exact.
            let mut outputs = outputs;
            assert_eq!(
                outputs.pop().unwrap().into_bytes_in(Global).unwrap(),
                input.as_native_bytes()
            );
            drop(invocation);
            runtime.release_executable(executable).unwrap();
        }
    }

    #[test]
    fn pitched_output_checks_overflow_and_writes_partial_ranges_without_touching_guards() {
        use super::AxOutput;
        use virtio_accel_core::ByteSink;
        let mut backing = [0xa5; 32];
        assert!(AxOutput::pitched(&mut backing, usize::MAX, 4, 2, 8).is_err());
        assert!(AxOutput::pitched(&mut backing, 0, 4, usize::MAX, 8).is_err());
        assert!(AxOutput::pitched(&mut backing, 0, 4, 2, 3).is_err());
        assert!(AxOutput::pitched(&mut backing, 28, 4, 2, 8).is_err());
        let mut output = AxOutput::pitched(&mut backing, 3, 4, 3, 8).unwrap();
        assert!(output.as_contiguous_mut().is_none());
        assert!(output.write_at(u64::MAX, &[0]).is_err());
        assert!(output.write_at(11, &[0, 0]).is_err());
        output.write_at(2, &[1, 2, 3, 4, 5, 6, 7]).unwrap();
        output.write_at(12, &[]).unwrap();
        let mut expected = [0xa5; 32];
        for (index, value) in [(5, 1), (6, 2), (11, 3), (12, 4), (13, 5), (14, 6), (19, 7)] {
            expected[index] = value;
        }
        assert_eq!(backing, expected);
        let mut empty = [];
        AxOutput::packed(&mut empty).write_at(0, &[]).unwrap();
    }

    fn prepared_test_model() -> AxDeviceModel {
        AxDeviceModel {
            accelerator: false,
            cpu_tosa: false,
            sparse_bsr: None,
            ..AxDeviceModel::default()
        }
    }

    fn prepared_mul_add_graph(
        scalar_mask: u8,
        reverse_multiply: bool,
        product_first: bool,
    ) -> ax::GraphBuf<Global> {
        let mut builder = GraphBuilder::new_in(Global);
        let inputs: [TensorId; 3] = core::array::from_fn(|index| {
            let shape: &[u32] = if scalar_mask & (1 << index) != 0 {
                &[1, 1]
            } else {
                &[3, 5]
            };
            builder
                .tensor(DType::F32, TensorFlags::INPUT, shape, &[])
                .unwrap()
        });
        let product_shape: &[u32] = if scalar_mask & 3 == 3 {
            &[1, 1]
        } else {
            &[3, 5]
        };
        let output_shape: &[u32] = if scalar_mask == 7 { &[1, 1] } else { &[3, 5] };
        let product = builder
            .tensor(DType::F32, TensorFlags::default(), product_shape, &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, output_shape, &[])
            .unwrap();
        let multiply = if reverse_multiply {
            [inputs[1], inputs[0]]
        } else {
            [inputs[0], inputs[1]]
        };
        let add = if product_first {
            [product, inputs[2]]
        } else {
            [inputs[2], product]
        };
        builder
            .node(
                OpCode::MUL,
                NodeFlags::default(),
                &multiply,
                &[product],
                &[],
            )
            .unwrap();
        builder
            .node(OpCode::ADD, NodeFlags::default(), &add, &[output], &[])
            .unwrap();
        builder.finish().unwrap()
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn default_dispatch_selects_exact_fusions_without_preparation_or_model_knobs() {
        use super::{AxInvocationStatus, NativeValue};
        for (index, encoded) in [
            prepared_mul_add_graph(0, false, true),
            prepared_scale_interleave_graph([3, 5], [false, true, false]),
            prepared_cast_shift_or_graph(15, 7, false),
        ]
        .into_iter()
        .enumerate()
        {
            let graph = encoded.parse().unwrap();
            let mut runtime = prepared_test_runtime();
            let executable = runtime.compile_auto_in(&graph, Global).unwrap();
            let stats = executable.native_plan_stats();
            assert_eq!(
                [
                    stats.mul_add_fusions,
                    stats.scale_interleave3_fusions,
                    stats.cast_shift_or3_fusions
                ][index],
                1
            );
            assert_eq!(stats.instructions, 1);
            assert!(stats.logical_nodes > stats.instructions);
            let reference = runtime
                .compile_in(
                    &graph,
                    &AxDeviceModel {
                        native_fusions: false,
                        ..runtime.automatic_device_model()
                    },
                    Global,
                )
                .unwrap();
            assert_eq!(
                reference.native_plan_stats().instructions,
                stats.logical_nodes
            );
            assert_eq!(
                reference
                    .prepare_native_in(Global)
                    .unwrap()
                    .instruction_count(),
                stats.logical_nodes
            );
            let input: Vec<_> = executable
                .input_ids()
                .iter()
                .enumerate()
                .map(|(index, &id)| {
                    let tensor = graph.tensor(id).unwrap();
                    let elements = super::tensor_storage_bytes(tensor).unwrap() / 4;
                    match tensor.dtype {
                        DType::F32 => NativeValue::F32(
                            (0..elements)
                                .map(|lane| {
                                    [
                                        f32::from_bits(0x3f800001),
                                        f32::from_bits(0x3f7ffffe),
                                        -1.0,
                                        -0.0,
                                        0.0,
                                        f32::MIN_POSITIVE,
                                        17.25,
                                    ][(lane + index) % 7]
                                })
                                .collect::<Vec<_>>(),
                        ),
                        DType::I32 => NativeValue::I32(vec![[16, 8, -31][index % 3]; elements]),
                        _ => panic!("unexpected fixture input"),
                    }
                })
                .collect();
            let inputs: Vec<_> = input.iter().map(NativeValue::as_native_bytes).collect();
            let expected = runtime.run_in(&reference, &inputs, Global).unwrap();
            // Both the compatible byte ABI and retained completion use the
            // same automatically fused instruction schedule.
            assert_eq!(
                runtime.run_in(&executable, &inputs, Global).unwrap(),
                expected
            );
            let mut invocation = runtime.begin_in(&executable, &inputs, Global).unwrap();
            let AxInvocationStatus::Complete(output) =
                runtime.advance_retained_in(&mut invocation).unwrap()
            else {
                panic!("native default should complete inline")
            };
            assert_eq!(
                invocation.host_copy_stats(),
                super::AxHostCopyStats::default()
            );
            for (value, expected) in output.iter().zip(expected) {
                assert_eq!(value.as_bytes(), expected);
            }
            assert_eq!(runtime.cpu.provider().buffer_allocations(), 0);
        }
    }

    #[test]
    fn prepared_mul_add_keeps_order_rounding_and_scalar_policy_without_warm_allocations() {
        use super::{CpuCapabilities, PreparedMulAddKernel, PreparedNativeInstruction};

        // The first tuple separates two-rounding MUL+ADD from IEEE FMA.
        // The remaining lanes include subnormal products, signed zeros,
        // overflow, infinities and distinct signed quiet-NaN payloads.
        let samples: [[f32; 15]; 3] = [
            [
                f32::from_bits(0x3f80_0001),
                f32::MIN_POSITIVE,
                f32::from_bits(1),
                -0.0,
                0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(0x7fc0_1234),
                f32::from_bits(0xffc0_4321),
                f32::MAX,
                1.0e-20,
                -f32::MIN_POSITIVE,
                -1.0,
                3.5,
                f32::from_bits(3),
            ],
            [
                f32::from_bits(0x3f7f_fffe),
                0.5,
                0.5,
                1.0,
                -1.0,
                0.0,
                -1.0,
                f32::from_bits(0xffc0_2468),
                2.0,
                2.0,
                1.0e-20,
                0.5,
                -0.0,
                -2.0,
                0.5,
            ],
            [
                -1.0,
                f32::from_bits(1),
                f32::from_bits(1),
                -0.0,
                0.0,
                3.0,
                f32::NEG_INFINITY,
                f32::from_bits(0x7fc0_5678),
                f32::from_bits(0x7fc0_8765),
                -f32::MAX,
                0.0,
                f32::MIN_POSITIVE,
                -0.0,
                7.0,
                -f32::from_bits(1),
            ],
        ];
        assert_ne!(
            samples[0][0]
                .mul_add(samples[1][0], samples[2][0])
                .to_bits(),
            0
        );
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
            for scalar_mask in 0..8_u8 {
                for reverse_multiply in [false, true] {
                    for product_first in [false, true] {
                        let graph =
                            prepared_mul_add_graph(scalar_mask, reverse_multiply, product_first);
                        let mut runtime = prepared_test_runtime();
                        let mut model = prepared_test_model();
                        model.cpu_kernels.as_mut().unwrap().capabilities = capabilities;
                        let executable = runtime
                            .compile_in(&graph.parse().unwrap(), &model, Global)
                            .unwrap();
                        let ExecutablePartitionKind::NativeCpu(native) =
                            &executable.partitions[0].kind
                        else {
                            panic!("native region")
                        };
                        assert_eq!(native.capabilities, capabilities);
                        assert_eq!(native.nodes.len(), 2);
                        let counts = PreparedAllocationCounts::default();
                        let mut prepared = executable
                            .prepare_native_in(PreparedAllocator(&counts))
                            .unwrap();
                        assert_eq!(prepared.instruction_count(), 1);
                        assert_eq!(prepared.mul_add_fusion_count(), 1);
                        let simd = capabilities != CpuCapabilities::SCALAR
                            && (matches!(scalar_mask & 3, 1 | 2) || scalar_mask == 0);
                        assert_eq!(prepared.simd_mul_add_fusion_count(), usize::from(simd));
                        let PreparedNativeInstruction::MulAdd(fused) = prepared.nodes[0] else {
                            panic!("strict combination")
                        };
                        assert_eq!(
                            fused.inputs[0] as usize,
                            prepared.inputs[usize::from(reverse_multiply)].slot
                        );
                        assert_eq!(
                            fused.inputs[1] as usize,
                            prepared.inputs[usize::from(!reverse_multiply)].slot
                        );
                        assert_eq!(fused.inputs[2] as usize, prepared.inputs[2].slot);
                        assert_eq!(
                            fused.order,
                            if product_first {
                                super::MulAddOrder::ProductAddend
                            } else {
                                super::MulAddOrder::AddendProduct
                            }
                        );
                        if capabilities == CpuCapabilities::SCALAR {
                            assert!(matches!(fused.kernel, PreparedMulAddKernel::Scalar));
                        }
                        assert!(!fused.inputs.contains(&fused.output_slots[0]));
                        assert!(prepared.buffer_count() < prepared.logical_buffer_count());
                        let allocations = counts.allocations.get();
                        counts.deny.set(true);
                        // Rotate both dense inputs and scalar uniforms, then
                        // repeat a prior run to expose stale/reused storage.
                        for offset in [0, 7, 12, 0] {
                            let bytes: [Vec<u8>; 3] = core::array::from_fn(|input| {
                                let count = if scalar_mask & (1 << input) != 0 {
                                    1
                                } else {
                                    15
                                };
                                (0..count)
                                    .flat_map(|index| {
                                        samples[input][(index + offset) % 15].to_le_bytes()
                                    })
                                    .collect()
                            });
                            let inputs = bytes.each_ref().map(Vec::as_slice);
                            let expected = runtime.run_in(&executable, &inputs, Global).unwrap();
                            prepared.run(&inputs).unwrap();
                            let actual = prepared.output_f32(0).unwrap();
                            if offset == 0 {
                                assert_eq!(
                                    u32::from_le_bytes(expected[0][..4].try_into().unwrap()),
                                    0
                                );
                            }
                            for (lane, (actual, expected)) in actual
                                .iter()
                                .zip(expected[0].as_chunks::<4>().0)
                                .enumerate()
                            {
                                let expected = f32::from_le_bytes(*expected);
                                // Arithmetic NaN sign/payload selection is
                                // unspecified in Rust, unlike raw identity or
                                // byte-preserving storage. In particular LLVM
                                // may select a different input NaN payload in
                                // optimized scalar broadcast loops. Preserve
                                // authored operand order above, and compare
                                // exact bits for every non-NaN result below.
                                if expected.is_nan() {
                                    assert!(actual.is_nan());
                                } else {
                                    assert_eq!(
                                        actual.to_bits(),
                                        expected.to_bits(),
                                        "caps={capabilities:?} mask={scalar_mask} mul_reverse={reverse_multiply} product_first={product_first} offset={offset} lane={lane}"
                                    );
                                }
                            }
                            assert_eq!(counts.allocations.get(), allocations);
                        }
                        drop(prepared);
                        assert_eq!(counts.live_bytes.get(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_mul_add_extends_repeated_operands_past_intervening_nodes() {
        let mut builder = GraphBuilder::new_in(Global);
        let x = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[3, 5], &[])
            .unwrap();
        let y = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[3, 5], &[])
            .unwrap();
        let a = builder
            .tensor(DType::F32, TensorFlags::default(), &[3, 5], &[])
            .unwrap();
        let product = builder
            .tensor(DType::F32, TensorFlags::default(), &[3, 5], &[])
            .unwrap();
        let first = builder
            .tensor(DType::F32, TensorFlags::default(), &[3, 5], &[])
            .unwrap();
        let second = builder
            .tensor(DType::F32, TensorFlags::default(), &[3, 5], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[3, 5], &[])
            .unwrap();
        let copy = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[3, 5], &[])
            .unwrap();
        for (op, inputs, result) in [
            (OpCode::ABS, vec![x], a),
            (OpCode::MUL, vec![a, a], product),
            (OpCode::SUB, vec![a, y], first),
            (OpCode::SUB, vec![first, y], second),
            (OpCode::ADD, vec![second, product], output),
            (OpCode::IDENTITY, vec![output], copy),
        ] {
            builder
                .node(op, NodeFlags::default(), &inputs, &[result], &[])
                .unwrap();
        }
        let mut runtime = prepared_test_runtime();
        let executable = runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap();
        let ExecutablePartitionKind::NativeCpu(native) = &executable.partitions[0].kind else {
            panic!("native region")
        };
        let multiply = native
            .nodes
            .iter()
            .position(|node| node.plan.semantic == super::SemanticOp::Binary(axnn::BinaryOp::Mul))
            .unwrap();
        let add = native
            .nodes
            .iter()
            .position(|node| node.plan.semantic == super::SemanticOp::Binary(axnn::BinaryOp::Add))
            .unwrap();
        assert!(
            add >= multiply + 3,
            "two intervening nodes must exercise delayed liveness"
        );
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert_eq!(prepared.instruction_count(), 5);
        assert_eq!(prepared.mul_add_fusion_count(), 1);
        for instruction in &prepared.nodes {
            for output in instruction.outputs() {
                assert!(!instruction.inputs().contains(&output));
            }
        }
        assert_ne!(prepared.outputs[0].slot, prepared.outputs[1].slot);
        let allocations = counts.allocations.get();
        counts.deny.set(true);
        for shift in [0.0, -5.0, 1.25, 0.0] {
            let x: Vec<u8> = (0..15)
                .flat_map(|i| (i as f32 * -0.375 + shift).to_le_bytes())
                .collect();
            let y: Vec<u8> = (0..15)
                .flat_map(|i| (i as f32 * 0.125 - shift).to_le_bytes())
                .collect();
            let expected = runtime.run_in(&executable, &[&x, &y], Global).unwrap();
            prepared.run(&[&x, &y]).unwrap();
            for (index, bytes) in expected.iter().enumerate() {
                for (actual, expected) in prepared
                    .output_f32(index)
                    .unwrap()
                    .iter()
                    .zip(bytes.as_chunks::<4>().0)
                {
                    assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
                }
            }
            assert_eq!(counts.allocations.get(), allocations);
        }
    }

    #[test]
    fn prepared_mul_add_rejects_pinned_products_multiple_uses_and_partial_broadcasts() {
        for case in 0..4 {
            let mut builder = GraphBuilder::new_in(Global);
            let x_shape: &[u32] = if case == 3 { &[3, 1] } else { &[3, 5] };
            let y_shape: &[u32] = if case == 3 { &[1, 5] } else { &[3, 5] };
            let x = builder
                .tensor(DType::F32, TensorFlags::INPUT, x_shape, &[])
                .unwrap();
            let y = builder
                .tensor(DType::F32, TensorFlags::INPUT, y_shape, &[])
                .unwrap();
            let addend = builder
                .tensor(DType::F32, TensorFlags::INPUT, &[3, 5], &[])
                .unwrap();
            let product = builder
                .tensor(
                    DType::F32,
                    if case == 0 {
                        TensorFlags::OUTPUT
                    } else {
                        TensorFlags::default()
                    },
                    &[3, 5],
                    &[],
                )
                .unwrap();
            let output = builder
                .tensor(DType::F32, TensorFlags::OUTPUT, &[3, 5], &[])
                .unwrap();
            builder
                .node(OpCode::MUL, NodeFlags::default(), &[x, y], &[product], &[])
                .unwrap();
            builder
                .node(
                    OpCode::ADD,
                    NodeFlags::default(),
                    &[product, if case == 2 { product } else { addend }],
                    &[output],
                    &[],
                )
                .unwrap();
            if case == 1 {
                let copy = builder
                    .tensor(DType::F32, TensorFlags::OUTPUT, &[3, 5], &[])
                    .unwrap();
                builder
                    .node(
                        OpCode::IDENTITY,
                        NodeFlags::default(),
                        &[product],
                        &[copy],
                        &[],
                    )
                    .unwrap();
            }
            let mut runtime = prepared_test_runtime();
            let executable = runtime
                .compile_in(
                    &builder.finish().unwrap().parse().unwrap(),
                    &prepared_test_model(),
                    Global,
                )
                .unwrap();
            let mut prepared = executable.prepare_native_in(Global).unwrap();
            assert_eq!(prepared.mul_add_fusion_count(), 0, "case {case}");
            let x: Vec<u8> = (0..if case == 3 { 3 } else { 15 })
                .flat_map(|i| (i as f32 * 0.25).to_le_bytes())
                .collect();
            let y: Vec<u8> = (0..if case == 3 { 5 } else { 15 })
                .flat_map(|i| (i as f32 - 0.75).to_le_bytes())
                .collect();
            let addend: Vec<u8> = (0..15)
                .flat_map(|i| (i as f32 + 1.0).to_le_bytes())
                .collect();
            let expected = runtime
                .run_in(&executable, &[&x, &y, &addend], Global)
                .unwrap();
            prepared.run(&[&x, &y, &addend]).unwrap();
            for (index, bytes) in expected.iter().enumerate() {
                for (actual, expected) in prepared
                    .output_f32(index)
                    .unwrap()
                    .iter()
                    .zip(bytes.as_chunks::<4>().0)
                {
                    assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
                }
            }
        }
    }

    #[test]
    fn prepared_mul_add_reports_allocation_failure_and_restores_failed_output() {
        let mut runtime = prepared_test_runtime();
        let graph = prepared_mul_add_graph(2, false, true);
        let executable = runtime
            .compile_in(&graph.parse().unwrap(), &prepared_test_model(), Global)
            .unwrap();
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert_eq!(prepared.mul_add_fusion_count(), 1);
        let allocations = counts.allocations.get();
        counts.deny.set(true);
        let field: Vec<u8> = (0..15)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        let scale = 2.5_f32.to_le_bytes();
        let inputs = [field.as_slice(), scale.as_slice(), field.as_slice()];
        prepared.run(&inputs).unwrap();
        let output = prepared.outputs[0].slot;
        let Some(super::NativeValue::F32(values)) = prepared.values[output].as_mut() else {
            panic!("F32 output")
        };
        values.pop();
        assert!(prepared.run(&inputs).is_err());
        assert!(prepared.output_f32(0).is_none());
        assert!(prepared.values.iter().all(Option::is_some));
        let Some(super::NativeValue::F32(values)) = prepared.values[output].as_mut() else {
            panic!("restored F32 output")
        };
        values.push(0.0);
        prepared.run(&inputs).unwrap();
        let expected = runtime.run_in(&executable, &inputs, Global).unwrap();
        for (actual, expected) in prepared
            .output_f32(0)
            .unwrap()
            .iter()
            .zip(expected[0].as_chunks::<4>().0)
        {
            assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
        }
        assert_eq!(counts.allocations.get(), allocations);
        drop(prepared);
        assert_eq!(counts.live_bytes.get(), 0);
        for fail_at in 1..=allocations {
            let partial = PreparedAllocationCounts::default();
            partial.fail_at.set(Some(fail_at));
            assert!(matches!(
                executable.prepare_native_in(PreparedAllocator(&partial)),
                Err(AxCompileError::Ax(ax::AllocationError::AllocationFailed))
            ));
            assert_eq!(partial.allocations.get(), fail_at);
            assert_eq!(partial.live_bytes.get(), 0);
        }
    }

    #[test]
    fn prepared_mul_add_requires_strict_canonical_plans_at_both_boundaries() {
        use super::{KernelAccuracy, strict_canonical_f32_binary};
        use axnn::{BinaryOp, Shape};

        let graph = prepared_mul_add_graph(2, false, true);
        let mut runtime = prepared_test_runtime();
        for changed_node in 0..2 {
            let mut executable = runtime
                .compile_in(&graph.parse().unwrap(), &prepared_test_model(), Global)
                .unwrap();
            let ExecutablePartitionKind::NativeCpu(native) = &mut executable.partitions[0].kind
            else {
                panic!("native region")
            };
            let original = native.nodes[changed_node];
            let op = if changed_node == 0 {
                BinaryOp::Mul
            } else {
                BinaryOp::Add
            };
            assert!(strict_canonical_f32_binary(&original, op));
            for accuracy in [KernelAccuracy::Faithful, KernelAccuracy::Approximate] {
                let mut modified = original;
                modified.plan.accuracy = accuracy;
                assert!(!strict_canonical_f32_binary(&modified, op));
            }
            let mut scratch = original;
            scratch.plan.temporary_bytes = 4;
            assert!(!strict_canonical_f32_binary(&scratch, op));
            let mut shape = original;
            shape.plan.input_shapes[0] = Some(Shape::from([15]));
            assert!(!strict_canonical_f32_binary(&shape, op));
            let mut dtype = original;
            dtype.plan.input_dtypes[0] = Some(DType::F16);
            assert!(!strict_canonical_f32_binary(&dtype, op));
            native.nodes[changed_node].plan.accuracy = KernelAccuracy::Faithful;
            let prepared = executable.prepare_native_in(Global).unwrap();
            assert_eq!(prepared.mul_add_fusion_count(), 0);
            assert_eq!(prepared.instruction_count(), 2);
        }
    }

    fn prepared_cast_shift_or_graph(
        elements: u32,
        shift_mask: u8,
        right_tree: bool,
    ) -> ax::GraphBuf<Global> {
        let mut builder = GraphBuilder::new_in(Global);
        let field = [1, elements];
        let planes: [TensorId; 3] = core::array::from_fn(|_| {
            builder
                .tensor(DType::F32, TensorFlags::INPUT, &field, &[])
                .unwrap()
        });
        let shifts: [Option<TensorId>; 3] = core::array::from_fn(|axis| {
            (shift_mask & (1 << axis) != 0).then(|| {
                builder
                    .tensor(DType::I32, TensorFlags::INPUT, &[1], &[])
                    .unwrap()
            })
        });
        let words: [TensorId; 3] = core::array::from_fn(|axis| {
            let cast = builder
                .tensor(DType::I32, TensorFlags::default(), &field, &[])
                .unwrap();
            builder
                .node(
                    OpCode::CAST,
                    NodeFlags::default(),
                    &[planes[axis]],
                    &[cast],
                    &[],
                )
                .unwrap();
            if let Some(shift) = shifts[axis] {
                let word = builder
                    .tensor(DType::I32, TensorFlags::default(), &field, &[])
                    .unwrap();
                builder
                    .node(
                        OpCode::SHIFT_LEFT,
                        NodeFlags::default(),
                        &[cast, shift],
                        &[word],
                        &[],
                    )
                    .unwrap();
                word
            } else {
                cast
            }
        });
        let inner = builder
            .tensor(DType::I32, TensorFlags::default(), &field, &[])
            .unwrap();
        let output = builder
            .tensor(DType::I32, TensorFlags::OUTPUT, &field, &[])
            .unwrap();
        let (inner_inputs, outer_inputs) = if right_tree {
            ([words[1], words[2]], [words[0], inner])
        } else {
            ([words[0], words[1]], [inner, words[2]])
        };
        builder
            .node(
                OpCode::BITWISE_OR,
                NodeFlags::default(),
                &inner_inputs,
                &[inner],
                &[],
            )
            .unwrap();
        builder
            .node(
                OpCode::BITWISE_OR,
                NodeFlags::default(),
                &outer_inputs,
                &[output],
                &[],
            )
            .unwrap();
        builder.finish().unwrap()
    }

    #[test]
    fn prepared_cast_shift_or_keeps_cast_edges_shifts_and_tails_without_warm_allocations() {
        let samples = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_5678),
            2_147_483_520.0,
            2_147_483_648.0,
            -2_147_483_648.0,
            -2_147_483_904.0,
            16_777_216.0,
            255.999_98,
            -255.999_98,
            1.999_999_9,
        ];
        for capabilities in [
            super::CpuCapabilities::SCALAR,
            super::CpuCapabilities::PORTABLE_SIMD,
        ] {
            for elements in [1, 3, 7, 8, 9, 15, 16, 17, 271] {
                for shift_mask in [0_u8, 3, 7] {
                    for right_tree in [false, true] {
                        let graph = prepared_cast_shift_or_graph(elements, shift_mask, right_tree);
                        let mut runtime = prepared_test_runtime();
                        let mut model = prepared_test_model();
                        model.cpu_kernels.as_mut().unwrap().capabilities = capabilities;
                        let executable = runtime
                            .compile_in(&graph.parse().unwrap(), &model, Global)
                            .unwrap();
                        let counts = PreparedAllocationCounts::default();
                        let mut prepared = executable
                            .prepare_native_in(PreparedAllocator(&counts))
                            .unwrap();
                        let scalar_count = shift_mask.count_ones() as usize;
                        assert_eq!(prepared.cast_shift_or3_fusion_count(), 1);
                        assert_eq!(prepared.instruction_count(), 1);
                        assert_eq!(prepared.buffer_count(), 4 + scalar_count);
                        assert_eq!(
                            prepared.buffer_bytes(),
                            elements as usize * 16 + scalar_count * 4
                        );
                        assert_eq!(prepared.logical_buffer_count(), 8 + scalar_count * 2);
                        assert!(prepared.output_i32(0).is_none());
                        let super::PreparedNativeInstruction::CastShiftOr3(fused) =
                            prepared.nodes[0]
                        else {
                            panic!("cast/shift/OR sink")
                        };
                        assert!(
                            !fused.inputs[..fused.input_count as usize]
                                .contains(&fused.output_slots[0])
                        );
                        let allocations = counts.allocations.get();
                        counts.deny.set(true);
                        for (generation, shifts) in
                            [[16_i32, 8, 0], [i32::MIN, i32::MAX, -1], [32, 65, -31]]
                                .into_iter()
                                .enumerate()
                        {
                            let plane_bytes: [Vec<u8>; 3] = core::array::from_fn(|axis| {
                                (0..elements as usize)
                                    .flat_map(|lane| {
                                        samples[(lane + axis * 5 + generation * 3) % samples.len()]
                                            .to_le_bytes()
                                    })
                                    .collect()
                            });
                            let shift_bytes = shifts.map(i32::to_le_bytes);
                            let mut inputs: Vec<&[u8]> =
                                plane_bytes.iter().map(Vec::as_slice).collect();
                            for (axis, bytes) in shift_bytes.iter().enumerate() {
                                if shift_mask & (1 << axis) != 0 {
                                    inputs.push(bytes);
                                }
                            }
                            let expected = runtime.run_in(&executable, &inputs, Global).unwrap();
                            prepared.run(&inputs).unwrap();
                            let actual = prepared.output_i32(0).unwrap();
                            for (&word, bytes) in actual.iter().zip(expected[0].as_chunks::<4>().0)
                            {
                                assert_eq!(word, i32::from_le_bytes(*bytes));
                            }
                            assert_eq!(counts.allocations.get(), allocations);
                            // Invalid ingress never publishes a stale packed plane.
                            assert!(prepared.run(&inputs[..inputs.len() - 1]).is_err());
                            assert!(prepared.output_i32(0).is_none());
                            assert_eq!(counts.allocations.get(), allocations);
                        }
                        drop(prepared);
                        assert_eq!(counts.live_bytes.get(), 0);
                    }
                }
            }
        }
    }

    fn prepared_word_pack_boundary_graph(
        boundary: Option<(usize, bool)>,
        partial_plane: u8,
        dense_shift: bool,
    ) -> ax::GraphBuf<Global> {
        let mut builder = GraphBuilder::new_in(Global);
        let field: &[u32] = &[1, 15];
        let first_shape: &[u32] = match partial_plane {
            1 => &[1, 1],
            2 => &[15], // Same element count, different broadcast descriptor.
            _ => field,
        };
        let shapes = [first_shape, field, field];
        let planes: [TensorId; 3] = core::array::from_fn(|axis| {
            builder
                .tensor(DType::F32, TensorFlags::INPUT, shapes[axis], &[])
                .unwrap()
        });
        let shifts: [TensorId; 3] = core::array::from_fn(|axis| {
            builder
                .tensor(
                    DType::I32,
                    TensorFlags::INPUT,
                    if dense_shift && axis == 0 {
                        field
                    } else {
                        &[1]
                    },
                    &[],
                )
                .unwrap()
        });
        // Stages 0..3 are casts, 3..6 shifts, 6 the inner OR, 7 the sink.
        let mut append = |op, inputs: &[TensorId], shape: &[u32], stage| {
            let flags = if stage == 7 || boundary == Some((stage, true)) {
                TensorFlags::OUTPUT
            } else {
                TensorFlags::default()
            };
            let value = builder.tensor(DType::I32, flags, shape, &[]).unwrap();
            builder
                .node(op, NodeFlags::default(), inputs, &[value], &[])
                .unwrap();
            if boundary == Some((stage, false)) {
                let copy = builder
                    .tensor(DType::I32, TensorFlags::OUTPUT, shape, &[])
                    .unwrap();
                builder
                    .node(
                        OpCode::IDENTITY,
                        NodeFlags::default(),
                        &[value],
                        &[copy],
                        &[],
                    )
                    .unwrap();
            }
            value
        };
        let casts: [TensorId; 3] =
            core::array::from_fn(|axis| append(OpCode::CAST, &[planes[axis]], shapes[axis], axis));
        let words: [TensorId; 3] = core::array::from_fn(|axis| {
            append(
                OpCode::SHIFT_LEFT,
                &[casts[axis], shifts[axis]],
                shapes[axis],
                3 + axis,
            )
        });
        let inner = append(OpCode::BITWISE_OR, &words[..2], field, 6);
        append(OpCode::BITWISE_OR, &[inner, words[2]], field, 7);
        builder.finish().unwrap()
    }

    #[test]
    fn prepared_cast_shift_or_rejects_shared_pinned_broadcast_and_non_strict_nodes() {
        let field: Vec<u8> = (0..15)
            .flat_map(|lane| (lane as f32 * 0.75 - 5.25).to_le_bytes())
            .collect();
        let shift = 8_i32.to_le_bytes();
        for stage in 0..7 {
            for pinned in [false, true] {
                let graph = prepared_word_pack_boundary_graph(Some((stage, pinned)), 0, false);
                let mut runtime = prepared_test_runtime();
                let executable = runtime
                    .compile_in(&graph.parse().unwrap(), &prepared_test_model(), Global)
                    .unwrap();
                let mut prepared = executable.prepare_native_in(Global).unwrap();
                assert_eq!(
                    prepared.cast_shift_or3_fusion_count(),
                    0,
                    "stage {stage}, pinned {pinned}"
                );
                let inputs: [&[u8]; 6] = [&field, &field, &field, &shift, &shift, &shift];
                let expected = runtime.run_in(&executable, &inputs, Global).unwrap();
                prepared.run(&inputs).unwrap();
                for (output, bytes) in expected.iter().enumerate() {
                    for (&actual, expected) in prepared
                        .output_i32(output)
                        .unwrap()
                        .iter()
                        .zip(bytes.as_chunks::<4>().0)
                    {
                        assert_eq!(actual, i32::from_le_bytes(*expected));
                    }
                }
            }
        }
        for (partial_plane, dense_shift) in [(1, false), (2, false), (0, true)] {
            let graph = prepared_word_pack_boundary_graph(None, partial_plane, dense_shift);
            let mut runtime = prepared_test_runtime();
            let executable = runtime
                .compile_in(&graph.parse().unwrap(), &prepared_test_model(), Global)
                .unwrap();
            assert_eq!(
                executable
                    .prepare_native_in(Global)
                    .unwrap()
                    .cast_shift_or3_fusion_count(),
                0
            );
        }
        let graph = prepared_cast_shift_or_graph(15, 7, false);
        let mut runtime = prepared_test_runtime();
        for changed_node in 0..8 {
            let mut executable = runtime
                .compile_in(&graph.parse().unwrap(), &prepared_test_model(), Global)
                .unwrap();
            let ExecutablePartitionKind::NativeCpu(native) = &mut executable.partitions[0].kind
            else {
                panic!("native")
            };
            native.nodes[changed_node].plan.accuracy = super::KernelAccuracy::Faithful;
            assert_eq!(
                executable
                    .prepare_native_in(Global)
                    .unwrap()
                    .cast_shift_or3_fusion_count(),
                0,
                "changed {changed_node}"
            );
        }
    }

    #[test]
    fn prepared_cast_shift_or_keeps_repeated_sources_live_across_intervening_writes() {
        let mut builder = GraphBuilder::new_in(Global);
        let field: &[u32] = &[1, 17];
        let a = builder
            .tensor(DType::F32, TensorFlags::INPUT, field, &[])
            .unwrap();
        let b = builder
            .tensor(DType::F32, TensorFlags::INPUT, field, &[])
            .unwrap();
        let count = builder
            .tensor(DType::I32, TensorFlags::INPUT, &[1], &[])
            .unwrap();
        let mut authored_order = Vec::new();
        let mut append = |op, dtype, inputs: &[TensorId], shape: &[u32], flags| {
            let value = builder.tensor(dtype, flags, shape, &[]).unwrap();
            builder
                .node(op, NodeFlags::default(), inputs, &[value], &[])
                .unwrap();
            authored_order.push(value);
            value
        };
        let shared = append(OpCode::ABS, DType::F32, &[a], field, TensorFlags::default());
        let pinned = append(OpCode::NEGATE, DType::F32, &[b], field, TensorFlags::OUTPUT);
        let shift = append(
            OpCode::ABS,
            DType::I32,
            &[count],
            &[1],
            TensorFlags::default(),
        );
        let planes = [shared, shared, pinned];
        let words: [TensorId; 3] = core::array::from_fn(|axis| {
            let cast = append(
                OpCode::CAST,
                DType::I32,
                &[planes[axis]],
                field,
                TensorFlags::default(),
            );
            append(
                OpCode::SHIFT_LEFT,
                DType::I32,
                &[cast, shift],
                field,
                TensorFlags::default(),
            )
        });
        let inner = append(
            OpCode::BITWISE_OR,
            DType::I32,
            &words[..2],
            field,
            TensorFlags::default(),
        );
        // These equally shaped destinations must not recycle either the
        // delayed F32 source or I32 scalar count before the final OR sink.
        for (input, dtype, shape) in [(shared, DType::F32, field), (shift, DType::I32, &[1][..])] {
            let negated = append(
                OpCode::NEGATE,
                dtype,
                &[input],
                shape,
                TensorFlags::default(),
            );
            append(OpCode::ABS, dtype, &[negated], shape, TensorFlags::OUTPUT);
        }
        append(
            OpCode::BITWISE_OR,
            DType::I32,
            &[inner, words[2]],
            field,
            TensorFlags::OUTPUT,
        );
        let graph = builder.finish().unwrap();
        let mut runtime = prepared_test_runtime();
        let mut executable = runtime
            .compile_in(&graph.parse().unwrap(), &prepared_test_model(), Global)
            .unwrap();
        // Ordinary dispatch now caches its instruction schedule at compile
        // time. Keep a separate unfused reference before mutating the private
        // canonical nodes to stress explicit preparation's authored ordering.
        let reference = runtime
            .compile_in(
                &executable.graph(),
                &AxDeviceModel {
                    native_fusions: false,
                    ..prepared_test_model()
                },
                Global,
            )
            .unwrap();
        let ExecutablePartitionKind::NativeCpu(native) = &mut executable.partitions[0].kind else {
            panic!("native")
        };
        // The scheduler can otherwise legally put the sink before clobbers.
        let tensors = &native.tensors;
        native.nodes.sort_by_key(|node| {
            authored_order
                .iter()
                .position(|&tensor| tensor == tensors[node.output_slots[0] as usize])
                .unwrap()
        });
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert_eq!(prepared.cast_shift_or3_fusion_count(), 1);
        assert_eq!(prepared.instruction_count(), 8);
        let super::PreparedNativeInstruction::CastShiftOr3(fused) = prepared.nodes.last().unwrap()
        else {
            panic!("packing last")
        };
        assert_eq!(fused.inputs[0], fused.inputs[1]);
        assert_eq!(fused.inputs[3..6], [fused.inputs[3]; 3]);
        for node in &prepared.nodes {
            for output in node.outputs() {
                assert!(!node.inputs().contains(&output));
            }
        }
        let allocations = counts.allocations.get();
        counts.deny.set(true);
        for generation in 0..4 {
            let a: Vec<u8> = (0..17)
                .flat_map(|lane| (lane as f32 - 8.75 + generation as f32 * 0.125).to_le_bytes())
                .collect();
            let b: Vec<u8> = (0..17)
                .flat_map(|lane| (7.5 - lane as f32 * 0.25).to_le_bytes())
                .collect();
            let count = [16_i32, -31, i32::MIN, 8][generation].to_le_bytes();
            let inputs: [&[u8]; 3] = [&a, &b, &count];
            let expected = runtime.run_in(&reference, &inputs, Global).unwrap();
            prepared.run(&inputs).unwrap();
            for (index, bytes) in expected.iter().enumerate() {
                if let Some(values) = prepared.output_f32(index) {
                    for (actual, expected) in values.iter().zip(bytes.as_chunks::<4>().0) {
                        assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
                    }
                } else {
                    for (&actual, expected) in prepared
                        .output_i32(index)
                        .unwrap()
                        .iter()
                        .zip(bytes.as_chunks::<4>().0)
                    {
                        assert_eq!(actual, i32::from_le_bytes(*expected));
                    }
                }
            }
            assert_eq!(counts.allocations.get(), allocations);
        }
    }

    #[test]
    fn prepared_cast_shift_or_removes_six_instructions_and_integer_scratch_planes() {
        let graph = prepared_cast_shift_or_graph(4096, 3, false);
        let mut runtime = prepared_test_runtime();
        let mut executable = runtime
            .compile_in(&graph.parse().unwrap(), &prepared_test_model(), Global)
            .unwrap();
        let fused = executable.prepare_native_in(Global).unwrap();
        // This admission-only change preserves each ordinary kernel's
        // arithmetic and provides an unfused prepared storage comparison.
        let ExecutablePartitionKind::NativeCpu(native) = &mut executable.partitions[0].kind else {
            panic!("native")
        };
        for node in &mut native.nodes {
            node.plan.accuracy = super::KernelAccuracy::Faithful;
        }
        let unfused = executable.prepare_native_in(Global).unwrap();
        assert_eq!(unfused.instruction_count() - fused.instruction_count(), 6);
        assert_eq!(fused.buffer_count(), 6);
        assert!(unfused.buffer_count() > fused.buffer_count());
        assert!(unfused.buffer_bytes() > fused.buffer_bytes());
        eprintln!(
            "CAST/shift/OR 4096-lane tail: instructions {} -> {}, buffers {} -> {}, payload {} -> {} bytes",
            unfused.instruction_count(),
            fused.instruction_count(),
            unfused.buffer_count(),
            fused.buffer_count(),
            unfused.buffer_bytes(),
            fused.buffer_bytes()
        );
    }

    fn prepared_scale_interleave_graph(
        field: [u32; 2],
        scale_first: [bool; 3],
    ) -> ax::GraphBuf<Global> {
        let mut builder = GraphBuilder::new_in(Global);
        let planes: [TensorId; 3] = core::array::from_fn(|_| {
            builder
                .tensor(
                    DType::F32,
                    TensorFlags::INPUT | TensorFlags::OUTPUT,
                    &field,
                    &[],
                )
                .unwrap()
        });
        let scale = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 1], &[])
            .unwrap();
        let products: [TensorId; 3] = core::array::from_fn(|axis| {
            let product = builder
                .tensor(DType::F32, TensorFlags::default(), &field, &[])
                .unwrap();
            let operands = if scale_first[axis] {
                [scale, planes[axis]]
            } else {
                [planes[axis], scale]
            };
            builder
                .node(
                    OpCode::MUL,
                    NodeFlags::default(),
                    &operands,
                    &[product],
                    &[],
                )
                .unwrap();
            product
        });
        let packed = builder
            .tensor(
                DType::F32,
                TensorFlags::OUTPUT,
                &[field[0], field[1], 3],
                &[],
            )
            .unwrap();
        builder
            .node(
                OpCode::INTERLEAVE3_F32,
                NodeFlags::default(),
                &products,
                &[packed],
                &[],
            )
            .unwrap();
        builder.finish().unwrap()
    }

    #[test]
    fn prepared_scale_interleave_preserves_orders_pinned_directions_and_policy() {
        use super::{CpuCapabilities, PreparedNativeInstruction};

        let samples = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(3),
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::from_bits(0x3f80_0001),
            -1.25,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_4321),
        ];
        for capabilities in [CpuCapabilities::SCALAR, CpuCapabilities::PORTABLE_SIMD] {
            for mask in 0..8 {
                let order = core::array::from_fn(|axis| mask & (1 << axis) != 0);
                let graph = prepared_scale_interleave_graph([3, 5], order);
                let mut runtime = prepared_test_runtime();
                let mut model = prepared_test_model();
                model.cpu_kernels.as_mut().unwrap().capabilities = capabilities;
                let executable = runtime
                    .compile_in(&graph.parse().unwrap(), &model, Global)
                    .unwrap();
                let counts = PreparedAllocationCounts::default();
                let mut prepared = executable
                    .prepare_native_in(PreparedAllocator(&counts))
                    .unwrap();
                assert_eq!(prepared.instruction_count(), 1);
                assert_eq!(prepared.mul_add_fusion_count(), 0);
                assert_eq!(prepared.scale_interleave3_fusion_count(), 1);
                assert_eq!(
                    prepared.simd_scale_interleave3_fusion_count(),
                    usize::from(capabilities != CpuCapabilities::SCALAR)
                );
                assert_eq!(prepared.buffer_count(), 5);
                assert_eq!(prepared.logical_buffer_count(), 8);
                let PreparedNativeInstruction::ScaleInterleave3(fused) = prepared.nodes[0] else {
                    panic!("fused packing")
                };
                assert_eq!(fused.scale_first, order);
                for (input, binding) in fused.inputs.iter().zip(&prepared.inputs) {
                    assert_eq!(*input as usize, binding.slot);
                }
                assert!(!fused.inputs.contains(&fused.output_slots[0]));
                let allocations = counts.allocations.get();
                counts.deny.set(true);
                for (offset, scale) in [
                    (0, 0.5),
                    (3, -0.0),
                    (7, f32::INFINITY),
                    (0, f32::from_bits(0x7fc0_5678)),
                    (0, 0.5),
                ] {
                    let bytes: [Vec<u8>; 3] = core::array::from_fn(|axis| {
                        (0..15)
                            .flat_map(|lane| {
                                samples[(lane + offset + axis * 3) % samples.len()].to_le_bytes()
                            })
                            .collect()
                    });
                    let scale_bytes = scale.to_le_bytes();
                    let inputs = [
                        bytes[0].as_slice(),
                        bytes[1].as_slice(),
                        bytes[2].as_slice(),
                        scale_bytes.as_slice(),
                    ];
                    let expected = runtime.run_in(&executable, &inputs, Global).unwrap();
                    prepared.run(&inputs).unwrap();
                    for (output, bytes) in expected.iter().enumerate() {
                        for (actual, encoded) in prepared
                            .output_f32(output)
                            .unwrap()
                            .iter()
                            .zip(bytes.as_chunks::<4>().0)
                        {
                            let expected = f32::from_le_bytes(*encoded);
                            if output == 3 && expected.is_nan() {
                                assert!(actual.is_nan());
                            } else {
                                // Unscaled output planes preserve even raw
                                // NaN payloads; only arithmetic NaN selection
                                // in the packed output is unspecified.
                                assert_eq!(actual.to_bits(), expected.to_bits());
                            }
                        }
                    }
                    assert_eq!(counts.allocations.get(), allocations);
                }
                drop(prepared);
                assert_eq!(counts.live_bytes.get(), 0);
            }
        }
    }

    #[test]
    fn prepared_scale_interleave_recomputes_all_four_lifetimes() {
        let mut builder = GraphBuilder::new_in(Global);
        let a = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 5], &[])
            .unwrap();
        let b = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 5], &[])
            .unwrap();
        let scale_input = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 1], &[])
            .unwrap();
        let xy = builder
            .tensor(DType::F32, TensorFlags::default(), &[1, 5], &[])
            .unwrap();
        let z = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 5], &[])
            .unwrap();
        let scale = builder
            .tensor(DType::F32, TensorFlags::default(), &[1, 1], &[])
            .unwrap();
        for (op, input, output) in [
            (OpCode::ABS, a, xy),
            (OpCode::NEGATE, b, z),
            (OpCode::ABS, scale_input, scale),
        ] {
            builder
                .node(op, NodeFlags::default(), &[input], &[output], &[])
                .unwrap();
        }
        let products: [TensorId; 3] = core::array::from_fn(|axis| {
            let output = builder
                .tensor(DType::F32, TensorFlags::default(), &[1, 5], &[])
                .unwrap();
            builder
                .node(
                    OpCode::MUL,
                    NodeFlags::default(),
                    &[if axis < 2 { xy } else { z }, scale],
                    &[output],
                    &[],
                )
                .unwrap();
            output
        });
        let mut authored_order = vec![xy, z, scale];
        authored_order.extend_from_slice(&products);
        // Both a dense source and the shared scalar have later consumers.
        // Without recounting the delayed fusion's four uses, the second
        // clobber could recycle either source before INTERLEAVE3 runs.
        for (input, shape) in [(xy, [1, 5]), (scale, [1, 1])] {
            let first = builder
                .tensor(DType::F32, TensorFlags::default(), &shape, &[])
                .unwrap();
            let second = builder
                .tensor(DType::F32, TensorFlags::OUTPUT, &shape, &[])
                .unwrap();
            builder
                .node(
                    OpCode::NEGATE,
                    NodeFlags::default(),
                    &[input],
                    &[first],
                    &[],
                )
                .unwrap();
            builder
                .node(OpCode::ABS, NodeFlags::default(), &[first], &[second], &[])
                .unwrap();
            authored_order.extend_from_slice(&[first, second]);
        }
        let packed = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 5, 3], &[])
            .unwrap();
        builder
            .node(
                OpCode::INTERLEAVE3_F32,
                NodeFlags::default(),
                &products,
                &[packed],
                &[],
            )
            .unwrap();
        let mut runtime = prepared_test_runtime();
        let mut executable = runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap();
        // The placement planner may validly run packing before independent
        // clobbers. Force this equally valid authored topology so the test
        // actually exercises operands surviving intervening instructions.
        authored_order.push(packed);
        // Ordinary dispatch now caches its instruction schedule at compile
        // time. Keep a separate unfused reference before mutating the private
        // canonical nodes to stress explicit preparation's authored ordering.
        let reference = runtime
            .compile_in(
                &executable.graph(),
                &AxDeviceModel {
                    native_fusions: false,
                    ..prepared_test_model()
                },
                Global,
            )
            .unwrap();
        let ExecutablePartitionKind::NativeCpu(native) = &mut executable.partitions[0].kind else {
            panic!("native")
        };
        let tensors = &native.tensors;
        native.nodes.sort_by_key(|node| {
            authored_order
                .iter()
                .position(|&id| id == tensors[node.output_slots[0] as usize])
                .unwrap()
        });
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert_eq!(prepared.scale_interleave3_fusion_count(), 1);
        assert_eq!(prepared.instruction_count(), 8);
        let super::PreparedNativeInstruction::ScaleInterleave3(fused) =
            prepared.nodes.last().unwrap()
        else {
            panic!("packing last")
        };
        assert_eq!(fused.inputs[0], fused.inputs[1]);
        for node in &prepared.nodes {
            for output in node.outputs() {
                assert!(!node.inputs().contains(&output));
            }
        }
        let allocations = counts.allocations.get();
        counts.deny.set(true);
        for shift in [0.0_f32, 2.5, -4.0, 0.0] {
            let a: Vec<u8> = (0..5)
                .flat_map(|i| (shift + i as f32 * -0.25).to_le_bytes())
                .collect();
            let b: Vec<u8> = (0..5)
                .flat_map(|i| (shift + i as f32 * 0.125).to_le_bytes())
                .collect();
            let scale = (shift - 0.75).to_le_bytes();
            let inputs = [a.as_slice(), b.as_slice(), scale.as_slice()];
            let expected = runtime.run_in(&reference, &inputs, Global).unwrap();
            prepared.run(&inputs).unwrap();
            for (index, bytes) in expected.iter().enumerate() {
                for (actual, expected) in prepared
                    .output_f32(index)
                    .unwrap()
                    .iter()
                    .zip(bytes.as_chunks::<4>().0)
                {
                    assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
                }
            }
            assert_eq!(counts.allocations.get(), allocations);
        }
    }

    #[test]
    fn prepared_scale_interleave_rejects_non_strict_or_non_shared_products() {
        for case in 0..7 {
            let mut builder = GraphBuilder::new_in(Global);
            let field = if case == 5 {
                [1, 1]
            } else if case == 6 {
                [1, 15]
            } else {
                [3, 5]
            };
            let input_shape: &[u32] = if case == 4 {
                &[3, 1]
            } else if case == 6 {
                &[15]
            } else {
                &field
            };
            let planes: [TensorId; 3] = core::array::from_fn(|_| {
                builder
                    .tensor(DType::F32, TensorFlags::INPUT, input_shape, &[])
                    .unwrap()
            });
            let scale = builder
                .tensor(DType::F32, TensorFlags::INPUT, &[1], &[])
                .unwrap();
            let other_scale = builder
                .tensor(DType::F32, TensorFlags::INPUT, &[1], &[])
                .unwrap();
            let products: [TensorId; 3] = core::array::from_fn(|axis| {
                let output = builder
                    .tensor(
                        DType::F32,
                        if case == 0 && axis == 0 {
                            TensorFlags::OUTPUT
                        } else {
                            TensorFlags::default()
                        },
                        input_shape,
                        &[],
                    )
                    .unwrap();
                builder
                    .node(
                        OpCode::MUL,
                        NodeFlags::default(),
                        &[
                            planes[axis],
                            if case == 3 && axis == 1 {
                                other_scale
                            } else {
                                scale
                            },
                        ],
                        &[output],
                        &[],
                    )
                    .unwrap();
                output
            });
            let packed = builder
                .tensor(
                    DType::F32,
                    TensorFlags::OUTPUT,
                    &[field[0], field[1], 3],
                    &[],
                )
                .unwrap();
            let components = if case == 1 {
                [products[0], products[0], products[2]]
            } else {
                products
            };
            builder
                .node(
                    OpCode::INTERLEAVE3_F32,
                    NodeFlags::default(),
                    &components,
                    &[packed],
                    &[],
                )
                .unwrap();
            if case == 2 {
                let copy = builder
                    .tensor(DType::F32, TensorFlags::OUTPUT, input_shape, &[])
                    .unwrap();
                builder
                    .node(
                        OpCode::IDENTITY,
                        NodeFlags::default(),
                        &[products[2]],
                        &[copy],
                        &[],
                    )
                    .unwrap();
            }
            let mut runtime = prepared_test_runtime();
            let executable = runtime
                .compile_in(
                    &builder.finish().unwrap().parse().unwrap(),
                    &prepared_test_model(),
                    Global,
                )
                .unwrap();
            assert_eq!(
                executable
                    .prepare_native_in(Global)
                    .unwrap()
                    .scale_interleave3_fusion_count(),
                0,
                "case {case}"
            );
        }
        let graph = prepared_scale_interleave_graph([3, 5], [false; 3]);
        let mut runtime = prepared_test_runtime();
        for changed_node in 0..4 {
            let mut executable = runtime
                .compile_in(&graph.parse().unwrap(), &prepared_test_model(), Global)
                .unwrap();
            let ExecutablePartitionKind::NativeCpu(native) = &mut executable.partitions[0].kind
            else {
                panic!("native")
            };
            native.nodes[changed_node].plan.accuracy = super::KernelAccuracy::Faithful;
            assert_eq!(
                executable
                    .prepare_native_in(Global)
                    .unwrap()
                    .scale_interleave3_fusion_count(),
                0
            );
        }
    }

    #[test]
    fn prepared_scale_interleave_reports_allocations_and_restores_failed_output() {
        let graph = prepared_scale_interleave_graph([3, 5], [true, false, true]);
        let mut runtime = prepared_test_runtime();
        let executable = runtime
            .compile_in(&graph.parse().unwrap(), &prepared_test_model(), Global)
            .unwrap();
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert_eq!(prepared.scale_interleave3_fusion_count(), 1);
        let allocations = counts.allocations.get();
        counts.deny.set(true);
        let field: Vec<u8> = (0..15)
            .flat_map(|i| (i as f32 * 0.125).to_le_bytes())
            .collect();
        let scale = 0.5_f32.to_le_bytes();
        let inputs = [
            field.as_slice(),
            field.as_slice(),
            field.as_slice(),
            scale.as_slice(),
        ];
        prepared.run(&inputs).unwrap();
        let output = prepared.outputs[3].slot;
        let Some(super::NativeValue::F32(values)) = prepared.values[output].as_mut() else {
            panic!("F32")
        };
        values.pop();
        assert!(prepared.run(&inputs).is_err());
        assert!((0..4).all(|index| prepared.output_f32(index).is_none()));
        assert!(prepared.values.iter().all(Option::is_some));
        let Some(super::NativeValue::F32(values)) = prepared.values[output].as_mut() else {
            panic!("restored F32")
        };
        values.push(0.0);
        prepared.run(&inputs).unwrap();
        assert_eq!(counts.allocations.get(), allocations);
        drop(prepared);
        assert_eq!(counts.live_bytes.get(), 0);
        for fail_at in 1..=allocations {
            let partial = PreparedAllocationCounts::default();
            partial.fail_at.set(Some(fail_at));
            assert!(matches!(
                executable.prepare_native_in(PreparedAllocator(&partial)),
                Err(AxCompileError::Ax(ax::AllocationError::AllocationFailed))
            ));
            assert_eq!(partial.allocations.get(), fail_at);
            assert_eq!(partial.live_bytes.get(), 0);
        }
    }

    const VOXEL_RESOURCE: ax::voxel::ResourceKey = ax::voxel::ResourceKey {
        group: 1,
        binding: 0,
    };

    fn voxel_test_executable(
        runtime: &mut AxHetero<AxTosaAccelerator, AxTosaAccelerator>,
        dims: [u32; 3],
    ) -> super::AxExecutable<Global> {
        let mut builder = GraphBuilder::new_in(Global);
        let x = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 5], &[])
            .unwrap();
        let z = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 1], &[])
            .unwrap();
        let mut constant = |value: f32| {
            builder
                .tensor(
                    DType::F32,
                    TensorFlags::CONSTANT,
                    &[1, 1],
                    &value.to_le_bytes(),
                )
                .unwrap()
        };
        let y = constant(1.5);
        let zero = constant(0.0);
        let one = constant(1.0);
        let limit = constant(700.0);
        let origins = builder.interleave3_f32([x, y, z], [1, 5]).unwrap();
        let directions = builder.interleave3_f32([one, zero, zero], [1, 5]).unwrap();
        let t = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 5], &[])
            .unwrap();
        let id = builder
            .tensor(DType::I32, TensorFlags::OUTPUT, &[1, 5], &[])
            .unwrap();
        let params = ax::voxel::VoxelBrickTraceParams::new(VOXEL_RESOURCE, dims).unwrap();
        builder
            .node(
                OpCode::VOXEL_BRICK_TRACE,
                NodeFlags::READS_RESOURCE,
                &[origins, directions, limit],
                &[t, id],
                &ax::voxel::encode_brick_trace_attributes(params).unwrap(),
            )
            .unwrap();
        // Both query results stay pinned even while later nodes consume them.
        let t_copy = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 5], &[])
            .unwrap();
        let id_copy = builder
            .tensor(DType::I32, TensorFlags::OUTPUT, &[1, 5], &[])
            .unwrap();
        let hit = builder
            .tensor(DType::U8, TensorFlags::OUTPUT, &[1, 5], &[])
            .unwrap();
        let miss = builder
            .tensor(
                DType::I32,
                TensorFlags::CONSTANT,
                &[1, 1],
                &(-1_i32).to_le_bytes(),
            )
            .unwrap();
        for (source, output) in [(t, t_copy), (id, id_copy)] {
            builder
                .node(
                    OpCode::IDENTITY,
                    NodeFlags::default(),
                    &[source],
                    &[output],
                    &[],
                )
                .unwrap();
        }
        builder
            .node(
                OpCode::GREATER,
                NodeFlags::default(),
                &[id, miss],
                &[hit],
                &[],
            )
            .unwrap();
        runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap()
    }

    fn volume_test_executable(
        runtime: &mut AxHetero<AxTosaAccelerator, AxTosaAccelerator>,
        storage: super::VolumeRgbStorage,
    ) -> super::AxExecutable<Global> {
        let mut builder = GraphBuilder::new_in(Global);
        let x = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 5], &[])
            .unwrap();
        let z = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 1], &[])
            .unwrap();
        let y = builder
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1, 1],
                &0.25_f32.to_le_bytes(),
            )
            .unwrap();
        let params = super::VolumeRgbParams::new(VOXEL_RESOURCE, [2, 2, 2], storage).unwrap();
        let attributes = ax::sampling::encode_volume_rgb_attributes(params).unwrap();
        for flags in [TensorFlags::OUTPUT, TensorFlags::default()] {
            let rgb: [TensorId; 3] =
                core::array::from_fn(|_| builder.tensor(DType::F32, flags, &[1, 5], &[]).unwrap());
            builder
                .node(
                    OpCode::VOLUME_TRILINEAR_RGB,
                    NodeFlags::READS_RESOURCE,
                    &[x, y, z],
                    &rgb,
                    &attributes,
                )
                .unwrap();
            if flags == TensorFlags::default() {
                // Consume every unpinned result, while the first RGB triple
                // stays available through the entire later calculation.
                let rg = builder
                    .tensor(DType::F32, TensorFlags::default(), &[1, 5], &[])
                    .unwrap();
                let total = builder
                    .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 5], &[])
                    .unwrap();
                builder
                    .node(
                        OpCode::ADD,
                        NodeFlags::default(),
                        &[rgb[0], rgb[1]],
                        &[rg],
                        &[],
                    )
                    .unwrap();
                builder
                    .node(
                        OpCode::ADD,
                        NodeFlags::default(),
                        &[rg, rgb[2]],
                        &[total],
                        &[],
                    )
                    .unwrap();
            }
        }
        runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap()
    }

    #[test]
    fn prepared_rgb_volume_preserves_all_outputs_and_borrows_without_warm_allocations() {
        use super::{
            NativeResourceBinding, NativeResourceRequirement, RgbVolumeRef, VolumeRgbStorage,
        };
        for storage in [
            VolumeRgbStorage::F32,
            VolumeRgbStorage::Uf16E5M11,
            VolumeRgbStorage::Uf8E5M3,
        ] {
            let mut runtime = prepared_test_runtime();
            let executable = volume_test_executable(&mut runtime, storage);
            let counts = PreparedAllocationCounts::default();
            let mut prepared = executable
                .prepare_native_in(PreparedAllocator(&counts))
                .unwrap();
            assert_eq!(
                prepared.resource_requirements,
                [NativeResourceRequirement::RgbVolume(
                    super::VolumeRgbParams::new(VOXEL_RESOURCE, [2, 2, 2], storage).unwrap()
                )]
            );
            assert_eq!(prepared.scratch_bytes(), 0);
            for node in &prepared.nodes {
                for (index, slot) in node.outputs().enumerate() {
                    assert!(!node.output_slots()[..index].contains(&slot));
                    assert!(!node.inputs().contains(&slot));
                }
            }
            let before = (
                runtime.cpu.provider().buffer_write_bytes(),
                runtime.cpu.provider().buffer_read_bytes(),
            );
            let mut packed_x = [0xa5_u8; 21];
            for (bytes, value) in packed_x[1..].as_chunks_mut::<4>().0.iter_mut().zip([
                f32::NEG_INFINITY,
                0.0,
                0.5,
                1.0,
                f32::INFINITY,
            ]) {
                *bytes = value.to_le_bytes();
            }
            let z = 0.75_f32.to_le_bytes();
            let inputs: [&[u8]; 2] = [&packed_x[1..], &z];
            let allocations = counts.allocations.get();
            counts.deny.set(true);
            for offset in [0.0_f32, 8.0, 0.0] {
                let rgb: [[f32; 3]; 8] = core::array::from_fn(|i| {
                    let value = offset + 1.0 + i as f32;
                    [value, value * 2.0, value * 4.0]
                });
                let words = rgb.map(|rgb| rgb.map(axnn_cpu::uf::f32_to_uf16e5m11_rne));
                let bytes = rgb.map(|rgb| rgb.map(axnn_cpu::uf::f32_to_uf8e5m3_rne));
                let volume = match storage {
                    VolumeRgbStorage::F32 => RgbVolumeRef::new_f32(&rgb, [2, 2, 2]).unwrap(),
                    VolumeRgbStorage::Uf16E5M11 => {
                        RgbVolumeRef::new_uf16(&words, [2, 2, 2]).unwrap()
                    }
                    VolumeRgbStorage::Uf8E5M3 => {
                        RgbVolumeRef::new_uf8e5m3(&bytes, [2, 2, 2]).unwrap()
                    }
                };
                let resources = [NativeResourceBinding::rgb_volume(VOXEL_RESOURCE, volume)];
                let reference = runtime
                    .run_with_resources_in(&executable, &inputs, &resources, Global)
                    .unwrap();
                prepared.run_with_resources(&inputs, &resources).unwrap();
                for (channel, multiplier) in [1.0, 2.0, 4.0, 7.0].into_iter().enumerate() {
                    let expected =
                        [4.5, 4.5, 5.0, 5.5, 5.5].map(|value| (offset + value) * multiplier);
                    assert_eq!(prepared.output_f32(channel).unwrap(), expected);
                    assert_eq!(
                        reference[channel],
                        expected
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .collect::<Vec<_>>()
                    );
                    assert!(prepared.output_i32(channel).is_none());
                }
            }
            assert_eq!(counts.allocations.get(), allocations);
            assert_eq!(
                (
                    runtime.cpu.provider().buffer_write_bytes(),
                    runtime.cpu.provider().buffer_read_bytes()
                ),
                before
            );
            drop(prepared);
            assert_eq!(counts.live_bytes.get(), 0);
        }
    }

    #[test]
    fn prepared_rgb_volume_failures_preflight_and_restore_every_output_handle() {
        use super::{
            NativeResourceBinding, NativeResourceKind, NativeValue, RgbVolumeRef, VolumeRgbStorage,
        };
        let mut runtime = prepared_test_runtime();
        let executable = volume_test_executable(&mut runtime, VolumeRgbStorage::Uf16E5M11);
        let mut prepared = executable.prepare_native_in(Global).unwrap();
        let x = [0_u8; 20];
        let z = 0.5_f32.to_le_bytes();
        let inputs: [&[u8]; 2] = [&x, &z];
        let words = [[0_u16; 3]; 8];
        let rgb = [[0.0_f32; 3]; 8];
        let bricks = [0_u64];
        let valid = NativeResourceBinding::rgb_volume(
            VOXEL_RESOURCE,
            RgbVolumeRef::new_uf16(&words, [2, 2, 2]).unwrap(),
        );
        let wrong_kind = NativeResourceBinding::brick_grid(
            VOXEL_RESOURCE,
            super::BrickGridRef::new(&bricks, [4, 4, 4]).unwrap(),
        );
        let wrong_storage = NativeResourceBinding::rgb_volume(
            VOXEL_RESOURCE,
            RgbVolumeRef::new_f32(&rgb, [2, 2, 2]).unwrap(),
        );
        let wrong_dims = NativeResourceBinding::rgb_volume(
            VOXEL_RESOURCE,
            RgbVolumeRef::new_uf16(&words, [4, 2, 1]).unwrap(),
        );
        assert!(matches!(
            prepared.run_with_resources(&inputs, &[wrong_kind]),
            Err(AxCompileError::ResourceKind {
                expected: NativeResourceKind::RgbVolume,
                actual: NativeResourceKind::BrickGrid,
                ..
            })
        ));
        assert!(matches!(
            prepared.run_with_resources(&inputs, &[wrong_storage]),
            Err(AxCompileError::ResourceStorage {
                expected: VolumeRgbStorage::Uf16E5M11,
                actual: VolumeRgbStorage::F32,
                ..
            })
        ));
        assert!(matches!(
            prepared.run_with_resources(&inputs, &[wrong_dims]),
            Err(AxCompileError::ResourceDimensions { .. })
        ));
        // Even the scalar input has not been decoded by these failed runs.
        assert_eq!(
            super::native_f32(&prepared.values, prepared.inputs[1].slot).unwrap(),
            &[0.0]
        );
        prepared.run_with_resources(&inputs, &[valid]).unwrap();
        assert!(matches!(
            prepared.run_with_resources(&inputs, &[valid, valid]),
            Err(AxCompileError::DuplicateResource(_))
        ));
        assert!((0..4).all(|index| prepared.output_f32(index).is_none()));
        assert!(matches!(
            prepared.run(&inputs),
            Err(AxCompileError::MissingResource(_))
        ));

        let node = *prepared
            .nodes
            .iter()
            .find(|node| {
                matches!(node, super::PreparedNativeInstruction::Kernel(kernel)
                if kernel.plan.semantic == super::SemanticOp::VolumeTrilinearRgb)
            })
            .unwrap();
        let third = node.output_slots()[2] as usize;
        let saved = prepared.values[third].take().unwrap();
        assert!(prepared.run_with_resources(&inputs, &[valid]).is_err());
        for slot in node.outputs().take(2) {
            assert!(prepared.values[slot as usize].is_some());
        }
        prepared.values[third] = Some(saved);
        let Some(NativeValue::F32(values)) = prepared.values[third].as_mut() else {
            panic!("F32 plane")
        };
        values.pop();
        assert!(matches!(
            prepared.run_with_resources(&inputs, &[valid]),
            Err(AxCompileError::Cpu(_))
        ));
        assert!(prepared.values.iter().all(Option::is_some));
        assert!((0..4).all(|index| prepared.output_f32(index).is_none()));
        let Some(NativeValue::F32(values)) = prepared.values[third].as_mut() else {
            panic!("F32 plane")
        };
        values.push(0.0);
        prepared.run_with_resources(&inputs, &[valid]).unwrap();
        for index in 0..4 {
            assert_eq!(prepared.output_f32(index).unwrap(), &[0.0; 5]);
        }
    }

    #[test]
    fn prepared_voxel_queries_borrow_edits_preserve_large_ids_and_allocate_no_warm_storage() {
        use super::{BrickGridRef, NativeResourceBinding};
        use axnn_cpu::CpuImplementation;

        let dims = [4, 4, 1_048_580];
        let mut runtime = prepared_test_runtime();
        let executable = voxel_test_executable(&mut runtime, dims);
        let [partition] = executable.partitions.as_slice() else {
            panic!("one CPU region")
        };
        assert_eq!(partition.implementation, CPU_KERNEL_IMPLEMENTATION);
        let ExecutablePartitionKind::NativeCpu(native) = &partition.kind else {
            panic!("native query")
        };
        let query = native
            .nodes
            .iter()
            .find(|node| node.plan.semantic == axnn::lowering::SemanticOp::VoxelBrickTrace)
            .unwrap();
        assert_eq!(query.plan.implementation, CpuImplementation::PortableSimd);
        assert_eq!(query.plan.temporary_bytes, 5 * size_of::<u32>());
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert_eq!(prepared.scratch_bytes(), 20);
        assert_eq!(prepared.resource_requirements.len(), 1);
        for node in &prepared.nodes {
            for output in node.outputs() {
                assert!(!node.inputs().contains(&output));
            }
            for (index, slot) in node.outputs().enumerate() {
                assert!(!node.output_slots()[..index].contains(&slot));
            }
        }

        let mut bricks = vec![0_u64; dims[2] as usize / 4];
        let brick = 1_048_576 / 4;
        bricks[brick] = 1 << (1 + 4); // cell(1,1,1048576), x-major id16777221
        let mut x_bytes = [0xa5_u8; 21];
        for (slot, value) in
            x_bytes[1..]
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip([-1.0_f32, -2.0, 0.0, 1.5, f32::NAN])
        {
            *slot = value.to_le_bytes();
        }
        let z_bytes = 1_048_576.5_f32.to_le_bytes();
        let inputs: [&[u8]; 2] = [&x_bytes[1..], &z_bytes];
        let allocated = counts.allocations.get();
        counts.deny.set(true);
        for occupied in [true, false, true] {
            bricks[brick] = if occupied { 1 << 5 } else { 0 };
            let resources = [NativeResourceBinding::brick_grid(
                VOXEL_RESOURCE,
                BrickGridRef::new(&bricks, dims).unwrap(),
            )];
            let reference = runtime
                .run_with_resources_in(&executable, &inputs, &resources, Global)
                .unwrap();
            prepared.run_with_resources(&inputs, &resources).unwrap();
            let expected_ids = if occupied {
                [16_777_221, 16_777_221, 16_777_221, 16_777_221, -1]
            } else {
                [-1; 5]
            };
            assert_eq!(prepared.output_i32(1).unwrap(), &expected_ids);
            assert_eq!(prepared.output_i32(3).unwrap(), &expected_ids);
            for index in [0, 2] {
                for (value, bytes) in prepared
                    .output_f32(index)
                    .unwrap()
                    .iter()
                    .zip(reference[index].as_chunks::<4>().0)
                {
                    assert_eq!(value.to_bits(), u32::from_le_bytes(*bytes));
                }
            }
            for index in [1, 3] {
                for (value, bytes) in prepared
                    .output_i32(index)
                    .unwrap()
                    .iter()
                    .zip(reference[index].as_chunks::<4>().0)
                {
                    assert_eq!(*value, i32::from_le_bytes(*bytes));
                }
            }
            assert_eq!(prepared.output_u8(4).unwrap(), reference[4]);
            assert!(prepared.output_f32(1).is_none());
            assert!(prepared.output_i32(0).is_none());
        }
        assert_eq!(counts.allocations.get(), allocated);

        // Preparation depends on the tile, not the external world volume.
        let small_counts = PreparedAllocationCounts::default();
        let small = voxel_test_executable(&mut runtime, [4, 4, 4])
            .prepare_native_in(PreparedAllocator(&small_counts))
            .unwrap();
        assert_eq!(small.buffer_bytes(), prepared.buffer_bytes());
        assert_eq!(small.scratch_bytes(), prepared.scratch_bytes());
        assert_eq!(small_counts.live_bytes.get(), counts.live_bytes.get());
        drop(prepared);
        assert_eq!(counts.live_bytes.get(), 0);
    }

    #[test]
    fn prepared_resource_requirements_deduplicate_queries_without_caching_borrows() {
        use super::{
            BrickGridRef, NativeResourceBinding, NativeResourceRequirement, VoxelBrickTraceParams,
        };

        fn compile(
            runtime: &mut AxHetero<AxTosaAccelerator, AxTosaAccelerator>,
            requirements: &[VoxelBrickTraceParams],
        ) -> super::AxExecutable<Global> {
            let mut builder = GraphBuilder::new_in(Global);
            let origin = builder
                .tensor(DType::F32, TensorFlags::INPUT, &[1, 1, 3], &[])
                .unwrap();
            let direction = builder
                .tensor(
                    DType::F32,
                    TensorFlags::CONSTANT,
                    &[1, 1, 3],
                    &[1.0_f32, 0.0, 0.0]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let limit = builder
                .tensor(
                    DType::F32,
                    TensorFlags::CONSTANT,
                    &[1],
                    &700.0_f32.to_le_bytes(),
                )
                .unwrap();
            for &params in requirements {
                let t = builder
                    .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 1], &[])
                    .unwrap();
                let id = builder
                    .tensor(DType::I32, TensorFlags::OUTPUT, &[1, 1], &[])
                    .unwrap();
                builder
                    .node(
                        OpCode::VOXEL_BRICK_TRACE,
                        NodeFlags::READS_RESOURCE,
                        &[origin, direction, limit],
                        &[t, id],
                        &ax::voxel::encode_brick_trace_attributes(params).unwrap(),
                    )
                    .unwrap();
            }
            runtime
                .compile_in(
                    &builder.finish().unwrap().parse().unwrap(),
                    &prepared_test_model(),
                    Global,
                )
                .unwrap()
        }

        let other = ax::voxel::ResourceKey {
            group: 1,
            binding: 1,
        };
        let first = VoxelBrickTraceParams::new(VOXEL_RESOURCE, [4, 4, 4]).unwrap();
        let second = VoxelBrickTraceParams::new(other, [4, 4, 4]).unwrap();
        let mut runtime = prepared_test_runtime();
        let executable = compile(&mut runtime, &[first, first, second]);
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert_eq!(
            prepared.resource_requirements,
            [first, second].map(NativeResourceRequirement::BrickGrid)
        );
        let allocations = counts.allocations.get();
        counts.deny.set(true);

        let input: Vec<u8> = [-1.0_f32, 1.5, 1.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let initial = [1_u64 << 21];
        let binding = NativeResourceBinding::brick_grid(
            VOXEL_RESOURCE,
            BrickGridRef::new(&initial, [4, 4, 4]).unwrap(),
        );
        assert!(matches!(
            prepared.run_with_resources(&[&input], &[binding]),
            Err(AxCompileError::MissingResource(key)) if key == other
        ));
        // The missing later query is rejected before even decoding the first input.
        assert_eq!(
            super::native_f32(&prepared.values, prepared.inputs[0].slot).unwrap(),
            &[0.0; 3],
        );
        assert!(prepared.values.iter().all(Option::is_some));

        for occupied in [true, false, true] {
            let bricks = [if occupied { 1_u64 << 21 } else { 0 }];
            let other_bricks = [1_u64 << 22];
            // Supply the bindings in a different order from the prepared list.
            let resources = [
                NativeResourceBinding::brick_grid(
                    other,
                    BrickGridRef::new(&other_bricks, [4, 4, 4]).unwrap(),
                ),
                NativeResourceBinding::brick_grid(
                    VOXEL_RESOURCE,
                    BrickGridRef::new(&bricks, [4, 4, 4]).unwrap(),
                ),
            ];
            let reference = runtime
                .run_with_resources_in(&executable, &[&input], &resources, Global)
                .unwrap();
            prepared.run_with_resources(&[&input], &resources).unwrap();
            for index in 0..3 {
                let expected_id = if index == 2 {
                    22
                } else if occupied {
                    21
                } else {
                    -1
                };
                assert_eq!(prepared.output_i32(index * 2 + 1).unwrap(), &[expected_id]);
                let t = prepared.output_f32(index * 2).unwrap()[0];
                assert_eq!(
                    t.to_bits(),
                    u32::from_le_bytes(reference[index * 2][..4].try_into().unwrap())
                );
            }
            assert!(matches!(
                prepared.run_with_resources(&[&input], &[resources[0], resources[1], resources[0]]),
                Err(AxCompileError::DuplicateResource(key)) if key == other
            ));
            assert!(prepared.output_f32(0).is_none());
            assert!(prepared.output_i32(5).is_none());
        }
        assert_eq!(counts.allocations.get(), allocations);
        drop(prepared);
        assert_eq!(counts.live_bytes.get(), 0);

        // Deduplicating by key alone would silently erase this conflict.
        let different_dimensions = VoxelBrickTraceParams::new(VOXEL_RESOURCE, [8, 4, 4]).unwrap();
        let conflicting = compile(&mut runtime, &[first, different_dimensions]);
        let mut prepared = conflicting.prepare_native_in(Global).unwrap();
        assert_eq!(
            prepared.resource_requirements,
            [first, different_dimensions].map(NativeResourceRequirement::BrickGrid)
        );
        assert!(matches!(
            prepared.run_with_resources(&[&input], &[binding]),
            Err(AxCompileError::ResourceDimensions {
                expected: [8, 4, 4],
                actual: [4, 4, 4],
                ..
            })
        ));
        assert_eq!(
            super::native_f32(&prepared.values, prepared.inputs[0].slot).unwrap(),
            &[0.0; 3]
        );
    }

    #[test]
    fn prepared_voxel_binding_and_kernel_failures_hide_outputs_and_recover_both_handles() {
        use super::{BrickGridRef, NativeResourceBinding};
        let mut runtime = prepared_test_runtime();
        let executable = voxel_test_executable(&mut runtime, [4, 4, 4]);
        let mut prepared = executable.prepare_native_in(Global).unwrap();
        let x: Vec<u8> = [-1.0_f32; 5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let z = 1.5_f32.to_le_bytes();
        let inputs: [&[u8]; 2] = [&x, &z];
        let bricks = [1 << 21];
        let binding = NativeResourceBinding::brick_grid(
            VOXEL_RESOURCE,
            BrickGridRef::new(&bricks, [4, 4, 4]).unwrap(),
        );
        let resources = [binding];
        prepared.run_with_resources(&inputs, &resources).unwrap();
        assert_eq!(prepared.output_i32(1).unwrap(), &[21; 5]);
        assert!(
            matches!(prepared.run(&inputs), Err(AxCompileError::MissingResource(key)) if key == VOXEL_RESOURCE)
        );
        assert!(prepared.output_i32(1).is_none());
        assert!(
            matches!(prepared.run_with_resources(&inputs, &[binding, binding]), Err(AxCompileError::DuplicateResource(key)) if key == VOXEL_RESOURCE)
        );
        let wrong_grid = [0; 2];
        let wrong = [NativeResourceBinding::brick_grid(
            VOXEL_RESOURCE,
            BrickGridRef::new(&wrong_grid, [8, 4, 4]).unwrap(),
        )];
        assert!(matches!(
            prepared.run_with_resources(&inputs, &wrong),
            Err(AxCompileError::ResourceDimensions { .. })
        ));
        assert!(prepared.output_f32(0).is_none());
        prepared.voxel_ids.clear();
        assert!(matches!(
            prepared.run_with_resources(&inputs, &resources),
            Err(AxCompileError::Cpu(_))
        ));
        assert!(prepared.values.iter().all(Option::is_some));
        assert!(prepared.output_i32(1).is_none());
        prepared.voxel_ids.resize(5, 0);
        prepared.run_with_resources(&inputs, &resources).unwrap();
        assert_eq!(prepared.output_i32(1).unwrap(), &[21; 5]);

        assert!(matches!(
            runtime.begin_in(&executable, &inputs, Global),
            Err(AxCompileError::MissingResource(_))
        ));
        let mut invocation = runtime
            .begin_with_resources_in(&executable, &inputs, &resources, Global)
            .unwrap();
        let super::AxInvocationStatus::Complete(outputs) =
            runtime.advance_in(&mut invocation).unwrap()
        else {
            panic!("one native region completes in one advance")
        };
        assert_eq!(
            outputs[1],
            [21_i32; 5]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn voxel_resources_and_paired_results_cross_provider_partitions_and_async_steps() {
        use super::{AxInvocationStatus, AxPartitionPhase, BrickGridRef, NativeResourceBinding};

        let mut builder = GraphBuilder::new_in(Global);
        let packed_half = builder
            .tensor(DType::F16, TensorFlags::INPUT, &[1, 1, 3], &[])
            .unwrap();
        let origins = builder
            .tensor(DType::F32, TensorFlags::default(), &[1, 1, 3], &[])
            .unwrap();
        builder
            .node(
                OpCode::CAST,
                NodeFlags::default(),
                &[packed_half],
                &[origins],
                &[],
            )
            .unwrap();
        let direction_bytes: Vec<u8> = [1.0_f32, 0.0, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let directions = builder
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1, 1, 3],
                &direction_bytes,
            )
            .unwrap();
        let limit = builder
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT,
                &[1, 1],
                &700.0_f32.to_le_bytes(),
            )
            .unwrap();
        let t = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 1], &[])
            .unwrap();
        let id = builder
            .tensor(DType::I32, TensorFlags::OUTPUT, &[1, 1], &[])
            .unwrap();
        let params = ax::voxel::VoxelBrickTraceParams::new(VOXEL_RESOURCE, [4, 4, 4]).unwrap();
        let attributes = ax::voxel::encode_brick_trace_attributes(params).unwrap();
        builder
            .node(
                OpCode::VOXEL_BRICK_TRACE,
                NodeFlags::READS_RESOURCE,
                &[origins, directions, limit],
                &[t, id],
                &attributes,
            )
            .unwrap();

        // Use only supported, finite CAST pairs. A second query depends on
        // the round-tripped distance, forcing id across the provider region
        // into a later native I32 consumer without casting either ID.
        let half_t = builder
            .tensor(DType::F16, TensorFlags::default(), &[1, 1], &[])
            .unwrap();
        let roundtrip_t = builder
            .tensor(DType::F32, TensorFlags::default(), &[1, 1], &[])
            .unwrap();
        builder
            .node(OpCode::CAST, NodeFlags::default(), &[t], &[half_t], &[])
            .unwrap();
        builder
            .node(
                OpCode::CAST,
                NodeFlags::default(),
                &[half_t],
                &[roundtrip_t],
                &[],
            )
            .unwrap();
        let second = builder
            .voxel_brick_trace(origins, directions, roundtrip_t, params, [1, 1])
            .unwrap();
        let equal = builder
            .tensor(DType::U8, TensorFlags::OUTPUT, &[1, 1], &[])
            .unwrap();
        builder
            .node(
                OpCode::EQUAL,
                NodeFlags::default(),
                &[id, second.cell_id],
                &[equal],
                &[],
            )
            .unwrap();

        let mut runtime = prepared_test_runtime();
        let model = AxDeviceModel {
            cpu_tosa: true,
            ..prepared_test_model()
        };
        let executable = runtime
            .compile_in(&builder.finish().unwrap().parse().unwrap(), &model, Global)
            .unwrap();
        let implementations: Vec<_> = executable
            .partitions
            .iter()
            .map(|partition| partition.implementation)
            .collect();
        assert_eq!(
            implementations,
            [
                DENSE_IMPLEMENTATION,
                CPU_KERNEL_IMPLEMENTATION,
                DENSE_IMPLEMENTATION,
                CPU_KERNEL_IMPLEMENTATION
            ]
        );
        assert!(executable.partitions[3].inputs.contains(&id));
        assert!(matches!(
            executable.prepare_native_in(Global),
            Err(AxCompileError::NativeRequiresSinglePartition)
        ));
        let input: Vec<u8> = [-1.0_f32, 1.5, 1.5]
            .into_iter()
            .flat_map(|value| (value as f16).to_bits().to_le_bytes())
            .collect();
        let inputs: [&[u8]; 1] = [&input];
        let bricks = [1_u64 << 21];
        let binding = NativeResourceBinding::brick_grid(
            VOXEL_RESOURCE,
            BrickGridRef::new(&bricks, [4, 4, 4]).unwrap(),
        );
        let resources = [binding];
        let before = (
            runtime.cpu.provider().buffer_allocations(),
            runtime.cpu.provider().buffer_write_bytes(),
            runtime.cpu.provider().buffer_read_bytes(),
        );

        // A missing resource in a later native partition must be reported
        // before any transient allocation or the FIRST provider submission.
        let rejected = PreparedAllocationCounts::default();
        rejected.deny.set(true);
        assert!(
            matches!(runtime.run_with_resources_in(&executable, &inputs, &[], PreparedAllocator(&rejected)), Err(AxCompileError::MissingResource(key)) if key == VOXEL_RESOURCE)
        );
        assert!(
            matches!(runtime.begin_with_resources_in(&executable, &inputs, &[], PreparedAllocator(&rejected)), Err(AxCompileError::MissingResource(key)) if key == VOXEL_RESOURCE)
        );
        assert!(matches!(
            runtime.run_in(&executable, &inputs, PreparedAllocator(&rejected)),
            Err(AxCompileError::MissingResource(_))
        ));
        assert!(matches!(
            runtime.begin_owned_in(&executable, &inputs, PreparedAllocator(&rejected)),
            Err(AxCompileError::MissingResource(_))
        ));
        assert!(matches!(
            runtime.run_with_resources_in(
                &executable,
                &inputs,
                &[binding, binding],
                PreparedAllocator(&rejected)
            ),
            Err(AxCompileError::DuplicateResource(_))
        ));
        let other_grid = [0_u64; 2];
        let wrong = [NativeResourceBinding::brick_grid(
            VOXEL_RESOURCE,
            BrickGridRef::new(&other_grid, [8, 4, 4]).unwrap(),
        )];
        assert!(matches!(
            runtime.begin_with_resources_in(
                &executable,
                &inputs,
                &wrong,
                PreparedAllocator(&rejected)
            ),
            Err(AxCompileError::ResourceDimensions { .. })
        ));
        assert_eq!(rejected.allocations.get(), 0);
        assert_eq!(rejected.live_bytes.get(), 0);
        assert_eq!(runtime.cpu.pending_count(), 0);
        assert_eq!(
            (
                runtime.cpu.provider().buffer_allocations(),
                runtime.cpu.provider().buffer_write_bytes(),
                runtime.cpu.provider().buffer_read_bytes()
            ),
            before
        );

        let reference = runtime
            .run_with_resources_in(&executable, &inputs, &resources, Global)
            .unwrap();
        assert_eq!(
            reference.as_slice(),
            [
                2.0_f32.to_le_bytes().to_vec(),
                21_i32.to_le_bytes().to_vec(),
                vec![1]
            ]
        );
        // Only packed ray bytes and the distance plane enter the two CAST
        // programs. Occupancy and I32 IDs never become provider inputs.
        assert_eq!(
            runtime.cpu.provider().buffer_write_bytes() - before.1,
            6 + 4
        );
        assert_eq!(
            runtime.cpu.provider().buffer_read_bytes() - before.2,
            12 + 4
        );

        let mut invocation = runtime
            .begin_with_resources_in(&executable, &inputs, &resources, Global)
            .unwrap();
        assert_eq!(runtime.cpu.pending_count(), 0);
        let mut submitted = Vec::new();
        let mut output = None;
        for _ in 0..16 {
            match runtime.advance_in(&mut invocation).unwrap() {
                AxInvocationStatus::Progress(progress) => {
                    if progress.phase == AxPartitionPhase::Submitted {
                        submitted.push(progress.partition);
                    }
                }
                AxInvocationStatus::Complete(outputs) => {
                    output = Some(outputs);
                    break;
                }
                AxInvocationStatus::Cancelled(_) => panic!("no cancellation was requested"),
            }
        }
        assert_eq!(submitted, [0, 2]);
        assert_eq!(
            output.expect("four partitions finish within the step budget"),
            reference
        );
        assert!(invocation.is_finished());
        assert_eq!(runtime.cpu.pending_count(), 0);
        assert!(matches!(
            runtime.advance_in(&mut invocation),
            Err(AxCompileError::InvocationFinished)
        ));
    }

    #[test]
    fn prepared_i32_ingress_preserves_unaligned_extremes_and_graph_abi() {
        let mut builder = GraphBuilder::new_in(Global);
        let input = builder
            .tensor(
                DType::I32,
                TensorFlags::INPUT | TensorFlags::OUTPUT,
                &[1, 5],
                &[],
            )
            .unwrap();
        let threshold = builder
            .tensor(DType::I32, TensorFlags::INPUT, &[1, 1], &[])
            .unwrap();
        let comparison = builder
            .tensor(DType::U8, TensorFlags::OUTPUT, &[1, 5], &[])
            .unwrap();
        let copied = builder
            .tensor(DType::I32, TensorFlags::OUTPUT, &[1, 5], &[])
            .unwrap();
        builder
            .node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[input],
                &[copied],
                &[],
            )
            .unwrap();
        builder
            .node(
                OpCode::GREATER,
                NodeFlags::default(),
                &[copied, threshold],
                &[comparison],
                &[],
            )
            .unwrap();
        let mut runtime = prepared_test_runtime();
        let executable = runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap();
        assert_eq!(executable.output_ids(), [input, comparison, copied]);
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        let allocations = counts.allocations.get();
        counts.deny.set(true);
        for (values, threshold) in [
            ([i32::MIN, -1, 0, 16_777_217, i32::MAX], 16_777_216_i32),
            ([i32::MAX, 16_777_216, 16_777_217, -1, i32::MIN], 16_777_217),
            ([i32::MIN, -16_777_217, -1, 0, i32::MAX], -1),
        ] {
            let mut input_bytes = [0xa5_u8; 21];
            for (slot, value) in input_bytes[1..]
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(values)
            {
                *slot = value.to_le_bytes();
            }
            let mut threshold_bytes = [0xa5_u8; 5];
            threshold_bytes[1..].copy_from_slice(&threshold.to_le_bytes());
            let inputs: [&[u8]; 2] = [&input_bytes[1..], &threshold_bytes[1..]];
            let expected_mask = values.map(|value| u8::from(value > threshold));
            let reference = runtime.run_in(&executable, &inputs, Global).unwrap();
            assert_eq!(reference[0], inputs[0]);
            assert_eq!(reference[1], expected_mask);
            assert_eq!(reference[2], inputs[0]);
            prepared.run(&inputs).unwrap();
            assert_eq!(prepared.output_i32(0).unwrap(), values);
            assert_eq!(prepared.output_u8(1).unwrap(), expected_mask);
            assert_eq!(prepared.output_i32(2).unwrap(), values);
            assert_eq!(counts.allocations.get(), allocations);
            assert!(matches!(
                prepared.run(&[&input_bytes[1..17], &threshold_bytes[1..]]),
                Err(AxCompileError::InvalidInputSize { .. })
            ));
            assert!(prepared.output_i32(0).is_none());
            assert!(prepared.output_u8(1).is_none());
            assert!(prepared.output_i32(2).is_none());
        }
        assert_eq!(counts.allocations.get(), allocations);
        drop(prepared);
        assert_eq!(counts.live_bytes.get(), 0);
    }

    #[test]
    fn prepared_resource_results_report_every_allocation_failure_without_leaking() {
        let mut runtime = prepared_test_runtime();
        for storage in [
            None,
            Some(super::VolumeRgbStorage::F32),
            Some(super::VolumeRgbStorage::Uf16E5M11),
        ] {
            let executable = match storage {
                None => voxel_test_executable(&mut runtime, [4, 4, 4]),
                Some(storage) => volume_test_executable(&mut runtime, storage),
            };
            let successful = PreparedAllocationCounts::default();
            let prepared = executable
                .prepare_native_in(PreparedAllocator(&successful))
                .unwrap();
            assert_eq!(
                prepared.scratch_bytes(),
                if storage.is_none() {
                    5 * size_of::<u32>()
                } else {
                    0
                }
            );
            let allocation_count = successful.allocations.get();
            drop(prepared);
            assert_eq!(successful.live_bytes.get(), 0);
            // Includes every paired or triple payload allocation, plus query
            // scratch after all plans and physical slots have been created.
            for fail_at in 1..=allocation_count {
                let partial = PreparedAllocationCounts::default();
                partial.fail_at.set(Some(fail_at));
                assert!(matches!(
                    executable.prepare_native_in(PreparedAllocator(&partial)),
                    Err(AxCompileError::Ax(ax::AllocationError::AllocationFailed))
                ));
                assert_eq!(partial.allocations.get(), fail_at);
                assert_eq!(partial.live_bytes.get(), 0, "allocation {fail_at}");
            }
        }
    }

    #[test]
    fn prepared_native_reuses_owned_typed_slots_and_preserves_graph_abi() {
        let mut builder = GraphBuilder::new_in(Global);
        let unused = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[2], &[])
            .unwrap();
        let input = builder
            .tensor(
                DType::F32,
                TensorFlags::INPUT | TensorFlags::OUTPUT,
                &[2, 2],
                &[],
            )
            .unwrap();
        let mask = builder
            .tensor(DType::U8, TensorFlags::INPUT, &[2, 2], &[])
            .unwrap();
        let constant = builder
            .tensor(
                DType::F32,
                TensorFlags::CONSTANT | TensorFlags::OUTPUT,
                &[1, 1],
                &0.5_f32.to_le_bytes(),
            )
            .unwrap();
        let absolute = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 2], &[])
            .unwrap();
        let sum = builder
            .tensor(DType::F32, TensorFlags::default(), &[2, 2], &[])
            .unwrap();
        let comparison = builder
            .tensor(DType::U8, TensorFlags::default(), &[2, 2], &[])
            .unwrap();
        let negated = builder
            .tensor(DType::U8, TensorFlags::default(), &[2, 2], &[])
            .unwrap();
        let logical = builder
            .tensor(DType::U8, TensorFlags::OUTPUT, &[2, 2], &[])
            .unwrap();
        let selected = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 2], &[])
            .unwrap();
        let fused = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 2], &[])
            .unwrap();
        let mask_copy = builder
            .tensor(DType::U8, TensorFlags::OUTPUT, &[2, 2], &[])
            .unwrap();
        for (op, inputs, output) in [
            (OpCode::ABS, vec![input], absolute),
            (OpCode::ADD, vec![absolute, absolute], sum),
            (OpCode::GREATER, vec![sum, constant], comparison),
            (OpCode::LOGICAL_NOT, vec![comparison], negated),
            (OpCode::LOGICAL_XOR, vec![negated, mask], logical),
            (OpCode::SELECT, vec![logical, sum, input], selected),
            (OpCode::FMA, vec![selected, constant, input], fused),
            (OpCode::IDENTITY, vec![mask], mask_copy),
        ] {
            builder
                .node(op, NodeFlags::default(), &inputs, &[output], &[])
                .unwrap();
        }
        let mut runtime = prepared_test_runtime();
        let executable = runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap();
        assert_eq!(executable.input_ids(), [unused, input, mask]);

        let encoded = [[-0.0_f32, -2.0, 1.0, 4.0], [3.0, 0.0, -0.25, -8.0]].map(|values| {
            // Deliberately unaligned byte slices must never be cast to f32.
            let mut bytes = [0xa5_u8; 17];
            for (value, bytes) in values.iter().zip(bytes[1..].as_chunks_mut::<4>().0) {
                *bytes = value.to_le_bytes();
            }
            bytes
        });
        let masks = [[0_u8, 2, 7, 0], [9, 0, 0, 3]];
        let expected: Vec<_> = (0..2)
            .map(|index| {
                runtime
                    .run_in(
                        &executable,
                        &[&[0; 8], &encoded[index][1..], &masks[index]],
                        Global,
                    )
                    .unwrap()
            })
            .collect();
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert_eq!(prepared.input_count(), 3);
        assert_eq!(prepared.output_count(), 7);
        assert!(prepared.resource_requirements.is_empty());
        assert!(prepared.buffer_count() < prepared.logical_buffer_count());
        assert!(prepared.buffer_bytes() < prepared.logical_buffer_bytes());
        for node in &prepared.nodes {
            for output in node.outputs() {
                assert!(!node.inputs().contains(&output));
            }
        }
        assert!(prepared.output_f32(0).is_none());
        let pointers: Vec<_> = prepared
            .values
            .iter()
            .map(|value| match value.as_ref().unwrap() {
                super::NativeValue::F32(values) => values.as_ptr() as usize,
                super::NativeValue::I32(values) => values.as_ptr() as usize,
                super::NativeValue::U8(values) => values.as_ptr() as usize,
                super::NativeValue::Packed(_, values) => values.as_ptr() as usize,
            })
            .collect();
        // The prepared object must not borrow plans, constants or providers.
        drop(executable);
        drop(runtime);
        let allocations = counts.allocations.get();
        assert!(allocations > 0 && counts.live_bytes.get() > 0);
        counts.deny.set(true);

        for index in [0, 1, 0] {
            prepared
                .run(&[&[0; 8], &encoded[index][1..], &masks[index]])
                .unwrap();
            for (output, expected) in expected[index].iter().enumerate() {
                if let Some(actual) = prepared.output_f32(output) {
                    assert_eq!(actual.len() * 4, expected.len());
                    for (actual, expected) in actual.iter().zip(expected.as_chunks::<4>().0) {
                        assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
                    }
                } else {
                    assert_eq!(prepared.output_u8(output).unwrap(), expected);
                }
            }
            for (value, &pointer) in prepared.values.iter().zip(&pointers) {
                let actual = match value.as_ref().unwrap() {
                    super::NativeValue::F32(values) => values.as_ptr() as usize,
                    super::NativeValue::I32(values) => values.as_ptr() as usize,
                    super::NativeValue::U8(values) => values.as_ptr() as usize,
                    super::NativeValue::Packed(_, values) => values.as_ptr() as usize,
                };
                assert_eq!(actual, pointer);
            }
        }
        assert_eq!(counts.allocations.get(), allocations);
        assert!(prepared.output_u8(0).is_none());
        assert!(prepared.output_f32(3).is_none());
        assert!(prepared.output_f32(usize::MAX).is_none());
        assert!(matches!(
            prepared.run(&[&[0; 4], &encoded[0][1..], &masks[0]]),
            Err(AxCompileError::InvalidInputSize { tensor, expected: 8, actual: 4 }) if tensor == unused
        ));
        assert!(prepared.output_f32(0).is_none());
        assert!(matches!(
            prepared.run(&[]),
            Err(AxCompileError::InvalidInputCount {
                expected: 3,
                actual: 0
            })
        ));
        prepared
            .run(&[&[0; 8], &encoded[1][1..], &masks[1]])
            .unwrap();
        assert_eq!(prepared.output_f32(1).unwrap(), [0.5]);
        assert_eq!(counts.allocations.get(), allocations);
        drop(prepared);
        assert_eq!(counts.live_bytes.get(), 0);
    }

    #[test]
    fn prepared_native_reuses_only_dead_storage_and_pins_constants_and_outputs() {
        let mut builder = GraphBuilder::new_in(Global);
        let unused = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let input = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let late = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let passthrough = builder
            .tensor(
                DType::F32,
                TensorFlags::INPUT | TensorFlags::OUTPUT,
                &[4],
                &[],
            )
            .unwrap();
        let constant_bytes: Vec<_> = [0.25_f32, -2.0, 0.5, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        // Not a graph output: it must still survive every run after its one
        // early consumer, even with many later outputs of this same layout.
        let constant = builder
            .tensor(DType::F32, TensorFlags::CONSTANT, &[4], &constant_bytes)
            .unwrap();
        let first = builder
            .tensor(DType::F32, TensorFlags::default(), &[4], &[])
            .unwrap();
        let early = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        let doubled = builder
            .tensor(DType::F32, TensorFlags::default(), &[4], &[])
            .unwrap();
        let branch = builder
            .tensor(DType::F32, TensorFlags::default(), &[4], &[])
            .unwrap();
        let joined = builder
            .tensor(DType::F32, TensorFlags::default(), &[4], &[])
            .unwrap();
        let folded = builder
            .tensor(DType::F32, TensorFlags::default(), &[4], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        let alias = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        for (op, inputs, output) in [
            (OpCode::ADD, vec![input, constant], first),
            (OpCode::ABS, vec![first], early),
            (OpCode::ADD, vec![first, first], doubled),
            (OpCode::MUL, vec![doubled, late], branch),
            (OpCode::SUB, vec![branch, doubled], joined),
            (OpCode::ADD, vec![joined, joined], folded),
            (OpCode::ADD, vec![folded, input], output),
            (OpCode::IDENTITY, vec![early], alias),
        ] {
            builder
                .node(op, NodeFlags::default(), &inputs, &[output], &[])
                .unwrap();
        }
        let mut runtime = prepared_test_runtime();
        let executable = runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap();
        assert_eq!(executable.input_ids(), [unused, input, late, passthrough]);
        assert_eq!(executable.output_ids(), [passthrough, early, output, alias]);
        let cases: Vec<Vec<Vec<u8>>> = (0..5)
            .map(|run| {
                (0..4)
                    .map(|plane| {
                        (0..4)
                            .flat_map(|lane| {
                                ((run * 7 + plane * 5 + lane * 3) as f32 * 0.125 - 3.0)
                                    .to_le_bytes()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let expected: Vec<_> = cases
            .iter()
            .map(|case| {
                let inputs: Vec<_> = case.iter().map(Vec::as_slice).collect();
                runtime.run_in(&executable, &inputs, Global).unwrap()
            })
            .collect();
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert!(prepared.buffer_count() < prepared.logical_buffer_count());
        assert_eq!(prepared.buffer_bytes(), prepared.buffer_count() * 16);
        assert_eq!(
            prepared.logical_buffer_bytes(),
            prepared.logical_buffer_count() * 16
        );
        // Ingress is simultaneous, including inputs first consumed late and
        // unused inputs whose buffers are eligible for subsequent reuse.
        for (index, binding) in prepared.inputs.iter().enumerate() {
            assert!(
                prepared.inputs[..index]
                    .iter()
                    .all(|other| other.slot != binding.slot)
            );
        }
        assert_eq!(prepared.inputs[3].slot, prepared.outputs[0].slot);
        for (index, binding) in prepared.outputs.iter().enumerate() {
            assert!(
                prepared.outputs[..index]
                    .iter()
                    .all(|other| other.slot != binding.slot)
            );
        }
        for node in &prepared.nodes {
            for output in node.outputs() {
                assert!(!node.inputs().contains(&output));
            }
        }
        let allocations = counts.allocations.get();
        counts.deny.set(true);
        for index in [0, 1, 2, 3, 4, 0, 4, 2] {
            let case = &cases[index];
            prepared
                .run(&[&case[0], &case[1], &case[2], &case[3]])
                .unwrap();
            for (output, bytes) in expected[index].iter().enumerate() {
                let actual = prepared.output_f32(output).unwrap();
                assert_eq!(actual.len() * 4, bytes.len());
                for (actual, expected) in actual.iter().zip(bytes.as_chunks::<4>().0) {
                    assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
                }
            }
            // Borrowing an output is read-only and must not let a later output
            // overwrite its storage; the identity result is a separate plane.
            assert_eq!(prepared.output_f32(1), prepared.output_f32(3));
        }
        assert_eq!(counts.allocations.get(), allocations);
        drop(prepared);
        assert_eq!(counts.live_bytes.get(), 0);
    }

    #[test]
    fn prepared_native_reuse_keeps_equal_byte_count_dtypes_distinct() {
        let mut builder = GraphBuilder::new_in(Global);
        let scalar = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 1], &[])
            .unwrap();
        let field = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 4], &[])
            .unwrap();
        let sum = builder
            .tensor(DType::F32, TensorFlags::default(), &[1, 4], &[])
            .unwrap();
        let compared = builder
            .tensor(DType::U8, TensorFlags::default(), &[1, 4], &[])
            .unwrap();
        let negated = builder
            .tensor(DType::U8, TensorFlags::default(), &[1, 4], &[])
            .unwrap();
        let repeated = builder
            .tensor(DType::U8, TensorFlags::OUTPUT, &[1, 4], &[])
            .unwrap();
        let selected = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[1, 4], &[])
            .unwrap();
        for (op, inputs, output) in [
            (OpCode::ADD, vec![scalar, field], sum),
            (OpCode::GREATER, vec![sum, field], compared),
            (OpCode::LOGICAL_NOT, vec![compared], negated),
            (OpCode::LOGICAL_XOR, vec![negated, negated], repeated),
            (OpCode::SELECT, vec![repeated, sum, field], selected),
        ] {
            builder
                .node(op, NodeFlags::default(), &inputs, &[output], &[])
                .unwrap();
        }
        let mut runtime = prepared_test_runtime();
        let executable = runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap();
        let field_bytes: Vec<_> = [-2.0_f32, -0.0, 1.0, 7.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let expected: Vec<_> = [-1.0_f32, 0.0, 2.0]
            .into_iter()
            .map(|scalar| {
                runtime
                    .run_in(&executable, &[&scalar.to_le_bytes(), &field_bytes], Global)
                    .unwrap()
            })
            .collect();
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        assert!(prepared.buffer_count() < prepared.logical_buffer_count());
        let scalar_slot = prepared.inputs[0].slot;
        for node in &prepared.nodes {
            if matches!(node, super::PreparedNativeInstruction::Kernel(kernel)
                if kernel.plan.output_dtype == DType::U8)
            {
                // A dead scalar f32 has the same byte count as the U8 mask,
                // but a typed Vec<f32> cannot become the mask's Vec<u8>.
                for output in node.outputs() {
                    assert_ne!(output as usize, scalar_slot);
                }
            }
        }
        let allocations = counts.allocations.get();
        counts.deny.set(true);
        for (index, scalar) in [-1.0_f32, 0.0, 2.0].into_iter().enumerate() {
            prepared
                .run(&[&scalar.to_le_bytes(), &field_bytes])
                .unwrap();
            assert_eq!(prepared.output_u8(0).unwrap(), expected[index][0]);
            for (actual, expected) in prepared
                .output_f32(1)
                .unwrap()
                .iter()
                .zip(expected[index][1].as_chunks::<4>().0)
            {
                assert_eq!(actual.to_bits(), u32::from_le_bytes(*expected));
            }
        }
        assert_eq!(counts.allocations.get(), allocations);
        drop(prepared);
        assert_eq!(counts.live_bytes.get(), 0);
    }

    #[test]
    fn prepared_native_resets_pure_matmul_destinations_between_runs() {
        let mut builder = GraphBuilder::new_in(Global);
        let _unused = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[2, 2], &[])
            .unwrap();
        let lhs = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[2, 2], &[])
            .unwrap();
        let rhs = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[2, 2], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[2, 2], &[])
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
        let mut runtime = prepared_test_runtime();
        let executable = runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap();
        let counts = PreparedAllocationCounts::default();
        let mut prepared = executable
            .prepare_native_in(PreparedAllocator(&counts))
            .unwrap();
        // This destination receives nonzero ingress bytes before each run.
        // Liveness reuse must preserve the pure MATMUL zero-initialization rule.
        assert_eq!(
            prepared.nodes[0].output_slots()[0] as usize,
            prepared.inputs[0].slot
        );
        let lhs: Vec<_> = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let rhs: Vec<_> = [5.0_f32, 6.0, 7.0, 8.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let allocations = counts.allocations.get();
        counts.deny.set(true);
        for _ in 0..3 {
            prepared.run(&[&rhs, &lhs, &rhs]).unwrap();
            assert_eq!(prepared.output_f32(0).unwrap(), [19.0, 22.0, 43.0, 50.0]);
            prepared.run(&[&rhs, &[0; 16], &rhs]).unwrap();
            assert_eq!(prepared.output_f32(0).unwrap(), [0.0; 4]);
        }
        assert_eq!(counts.allocations.get(), allocations);
    }

    #[test]
    fn prepared_native_rejects_provider_and_effectful_graphs_and_reports_allocation_failure() {
        for flags in [
            NodeFlags::default(),
            NodeFlags::ORDERED,
            NodeFlags::MUTATING,
        ] {
            let mut builder = GraphBuilder::new_in(Global);
            let input = builder
                .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
                .unwrap();
            let output = builder
                .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
                .unwrap();
            builder
                .node(OpCode::ABS, flags, &[input], &[output], &[])
                .unwrap();
            let mut runtime = prepared_test_runtime();
            let executable = runtime
                .compile_in(
                    &builder.finish().unwrap().parse().unwrap(),
                    &prepared_test_model(),
                    Global,
                )
                .unwrap();
            let counts = PreparedAllocationCounts::default();
            counts.deny.set(true);
            if flags == NodeFlags::default() {
                assert!(matches!(
                    executable.prepare_native_in(PreparedAllocator(&counts)),
                    Err(AxCompileError::Ax(ax::AllocationError::AllocationFailed))
                ));
                let successful = PreparedAllocationCounts::default();
                drop(
                    executable
                        .prepare_native_in(PreparedAllocator(&successful))
                        .unwrap(),
                );
                assert_eq!(successful.live_bytes.get(), 0);
                // Exercise every metadata and payload allocation, including
                // failures after the entire physical plan has been built.
                for fail_at in 1..=successful.allocations.get() {
                    let partial = PreparedAllocationCounts::default();
                    partial.fail_at.set(Some(fail_at));
                    assert!(matches!(
                        executable.prepare_native_in(PreparedAllocator(&partial)),
                        Err(AxCompileError::Ax(ax::AllocationError::AllocationFailed))
                    ));
                    assert_eq!(partial.live_bytes.get(), 0);
                }
            } else {
                assert!(matches!(
                    executable.prepare_native_in(PreparedAllocator(&counts)),
                    Err(AxCompileError::UnsupportedImplementation { .. })
                ));
                assert_eq!(counts.allocations.get(), 0);
            }
            assert_eq!(counts.live_bytes.get(), 0);
        }

        let mut builder = GraphBuilder::new_in(Global);
        let input = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        builder
            .node(OpCode::ABS, NodeFlags::default(), &[input], &[output], &[])
            .unwrap();
        let mut runtime = prepared_test_runtime();
        let model = AxDeviceModel {
            cpu_tosa: true,
            cpu_kernels: None,
            ..prepared_test_model()
        };
        let executable = runtime
            .compile_in(&builder.finish().unwrap().parse().unwrap(), &model, Global)
            .unwrap();
        assert!(matches!(
            executable.prepare_native_in(Global),
            Err(AxCompileError::NativeRequiresSinglePartition)
        ));
    }

    #[test]
    fn prepared_native_rejects_mixed_partitions_and_unsupported_boundary_types() {
        let mut builder = GraphBuilder::new_in(Global);
        let input = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let rounded = builder
            .tensor(DType::F32, TensorFlags::default(), &[4], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F16, TensorFlags::OUTPUT, &[4], &[])
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
                OpCode::CAST,
                NodeFlags::default(),
                &[rounded],
                &[output],
                &[],
            )
            .unwrap();
        let mut runtime = prepared_test_runtime();
        let model = AxDeviceModel {
            cpu_tosa: true,
            ..prepared_test_model()
        };
        let executable = runtime
            .compile_in(&builder.finish().unwrap().parse().unwrap(), &model, Global)
            .unwrap();
        assert_eq!(executable.partitions.len(), 2);
        assert!(matches!(
            executable.prepare_native_in(Global),
            Err(AxCompileError::NativeRequiresSinglePartition)
        ));

        let mut builder = GraphBuilder::new_in(Global);
        builder
            .tensor(DType::I64, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let input = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        builder
            .node(OpCode::ABS, NodeFlags::default(), &[input], &[output], &[])
            .unwrap();
        let executable = runtime
            .compile_in(
                &builder.finish().unwrap().parse().unwrap(),
                &prepared_test_model(),
                Global,
            )
            .unwrap();
        assert!(matches!(
            executable.prepare_native_in(Global),
            Err(AxCompileError::UnsupportedImplementation { .. })
        ));
    }

    #[test]
    fn boolean_input_bridge_preserves_nonzero_truth_semantics() {
        let actual = transform_input_in(AxInputTransform::BoolU8ToF32, &[0, 1, 7], Global)
            .expect("boolean input bridge");
        let expected: Vec<u8> = [0.0_f32, 1.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn boolean_output_bridge_is_exact_and_rejects_non_boolean_values() {
        let tensor = TensorId(9);
        let exact: Vec<u8> = [0.0_f32, 1.0, -0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let exact = copy_bytes_in(&exact, Global).unwrap();
        let actual = transform_output_in(AxOutputTransform::BoolF32ToU8, tensor, exact).unwrap();
        assert_eq!(actual, [0_u8, 1, 0]);

        let invalid = copy_bytes_in(&0.5_f32.to_le_bytes(), Global).unwrap();
        assert!(matches!(
            transform_output_in(AxOutputTransform::BoolF32ToU8, tensor, invalid),
            Err(AxCompileError::InvalidBooleanBoundary {
                tensor: TensorId(9)
            })
        ));
    }

    #[test]
    fn placement_wrapper_forwards_implementation_partition_costs() {
        let mut builder = GraphBuilder::new_in(Global);
        let input = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        builder
            .node(OpCode::ABS, NodeFlags::default(), &[input], &[output], &[])
            .unwrap();
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        let model = AxDeviceModel {
            accelerator: false,
            cpu_provider_launch_cost: 7,
            sparse_bsr: None,
            cpu_kernels: None,
            ..AxDeviceModel::default()
        };
        let wrapped: AxPlacementModel<Global> = AxPlacementModel {
            model: &model,
            targets: &[],
        };
        let mut direct = Schedule::analyze_in(&graph, Global).unwrap();
        let mut through_wrapper = Schedule::analyze_in(&graph, Global).unwrap();

        let direct_report = direct.place_optimized(&graph, &model).unwrap();
        let wrapped_report = through_wrapper.place_optimized(&graph, &wrapped).unwrap();

        assert_eq!(direct_report.aggregate_cost, 11);
        assert_eq!(wrapped_report.aggregate_cost, direct_report.aggregate_cost);
    }

    #[test]
    fn native_partition_storage_scales_with_the_region_not_the_graph() {
        let mut builder = GraphBuilder::new_in(Global);
        for value in 0..128_u32 {
            builder
                .tensor(
                    DType::F32,
                    TensorFlags::CONSTANT,
                    &[1],
                    &(value as f32).to_le_bytes(),
                )
                .unwrap();
        }
        let input = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        builder
            .node(
                OpCode::ROUND,
                NodeFlags::default(),
                &[input],
                &[output],
                &[],
            )
            .unwrap();
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        let model = AxDeviceModel {
            accelerator: false,
            cpu_tosa: false,
            sparse_bsr: None,
            ..AxDeviceModel::default()
        };
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();
        schedule.place_optimized(&graph, &model).unwrap();
        schedule.cluster_by_placement().unwrap();
        schedule.build_plan(&graph).unwrap();
        assert_eq!(schedule.plan().partitions.len(), 1);

        let executable = compile_native_cpu_partition_in(
            &graph,
            schedule.order(),
            schedule.producers(),
            schedule.plan().partitions[0],
            &model,
            Global,
        )
        .unwrap();
        let ExecutablePartitionKind::NativeCpu(native) = &executable.kind else {
            panic!("ROUND should compile as a native CPU region");
        };
        assert_eq!(native.nodes.len(), 1);
        assert_eq!(native.tensors, [input, output]);
        assert_eq!(native.takes.len(), 2);
        assert!(executable.inputs.capacity() < graph.tensor_count());
        assert!(executable.outputs.capacity() < graph.tensor_count());
        assert!(native.tensors.capacity() < graph.tensor_count());
        assert!(native.takes.capacity() < graph.tensor_count());
    }

    #[test]
    fn boolean_normalization_replaces_the_whole_component_through_catalog_placement() {
        let mut builder = GraphBuilder::new_in(Global);
        let left = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let right = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let first_mask = builder
            .tensor(DType::U8, TensorFlags::default(), &[4], &[])
            .unwrap();
        let second_mask = builder
            .tensor(DType::U8, TensorFlags::default(), &[4], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        builder
            .node(
                OpCode::GREATER,
                NodeFlags::default(),
                &[left, right],
                &[first_mask],
                &[],
            )
            .unwrap();
        builder
            .node(
                OpCode::LOGICAL_NOT,
                NodeFlags::default(),
                &[first_mask],
                &[second_mask],
                &[],
            )
            .unwrap();
        builder
            .node(
                OpCode::SELECT,
                NodeFlags::default(),
                &[second_mask, left, right],
                &[output],
                &[],
            )
            .unwrap();
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        let model = AxDeviceModel {
            cpu_tosa: false,
            accelerator_boolean_boundaries: false,
            f32_boolean_boundary_bridge: false,
            sparse_bsr: None,
            ..AxDeviceModel::default()
        };
        let mut schedule = Schedule::analyze_in(&graph, Global).unwrap();
        schedule.place(&graph, &model).unwrap();
        schedule
            .reassign_node_with_implementation(0, ACCEL_BACKEND, DENSE_IMPLEMENTATION)
            .unwrap();
        schedule
            .reassign_node_with_implementation(1, CPU_BACKEND, CPU_KERNEL_IMPLEMENTATION)
            .unwrap();
        schedule
            .reassign_node_with_implementation(2, ACCEL_BACKEND, DENSE_IMPLEMENTATION)
            .unwrap();
        let mut boolean = boolean_placement_in(&graph, &schedule, &model, Global).unwrap();

        assert!(boolean.normalize(&schedule));
        assert_eq!(boolean.accelerator_forbidden, [1, 1, 1]);

        let wrapped = AxPlacementModel {
            model: &model,
            targets: core::slice::from_ref(&boolean),
        };
        schedule.place_optimized(&graph, &wrapped).unwrap();
        assert_eq!(schedule.placements(), [CPU_BACKEND; 3]);
        assert_eq!(schedule.implementations(), [CPU_KERNEL_IMPLEMENTATION; 3]);
        for node_index in 0..3 {
            assert!(model.supports_implementation(
                CPU_BACKEND,
                &graph,
                graph.node(node_index).unwrap(),
                schedule.implementations()[node_index],
            ));
        }
    }
}
