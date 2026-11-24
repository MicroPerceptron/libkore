//! Shared native execution implementation. Host adapters and portable programs
//! use this same planner, typed storage and checked kernel invocation path.
use crate::storage::{NativeInputs, NativeRef, NativeSlot, RuntimeValue};
use crate::{Error as AxCompileError, PlanError as AxArtifactError};
use alloc::{sync::Arc, vec::Vec};
use ax::{
    AllocationError, DType, Graph, Implementation, Partition, TensorFlags, TensorId, TensorView,
};
use axnn::lowering::{
    IntegerBinaryOp, KernelAccuracy, KernelEffects, KernelParams, KernelPolicy, KernelRequest,
    KernelResult, KernelScalar, KvCacheParams, MAX_TENSOR_RESULTS, MoeParams, RasterNodeParams,
    ResourceKey, SemanticOp, StateBinding, U8BufferParams, VolumeRgbParams, VoxelBrickTraceParams,
};
use axnn::{BinaryOp, TensorDesc, TernaryOp};
use axnn_cpu::elementwise::{self, MulAddOrder, ScaleAddOrder};
use axnn_cpu::{
    BrickGridRef, CoverageAtlasRef, CpuCapabilities, CpuKernelPlan, ExpertStoreRef, ExpertWeights,
    GlyphTableRef, KvCacheRef, RgbVolumeRef, StateSourceRef, StateTargetRef, U8BufferRef,
    XrgbSurfaceRef, XrgbTargetRef,
};
use axnn_cpu::{ForkJoin, Serial};
use core::{alloc::Allocator, mem::size_of, ops::Deref};
/// Optional host execution of a dense projection. Returning `false` leaves
/// execution to the native kernel; the delegate must leave output untouched
/// in that case. A successful delegate must implement the complete plan,
/// including its scale and activation. Resource-bearing kernels are never
/// delegated. Inputs are borrowed only for the duration of this call.
pub trait ProjectionExecutor {
    /// Whether this executor may be invoked for latency-oriented one-token decode. Generic
    /// precision-changing projection delegates keep the default false; exact native-format
    /// delegates can opt in and reject unsupported individual kernels from `execute`.
    fn supports_decode(&self) -> bool {
        false
    }

    /// Whether multi-token target verification may use this executor even
    /// when one-token decode remains on the native kernel. Opt in only when
    /// projections preserve the target model's numerical contract: a draft
    /// is accepted using these results. Prefill-only precision conversions
    /// must retain the default. This does not delegate recurrent/KV state or
    /// authorize committing rejected tokens.
    fn supports_verification(&self) -> bool {
        false
    }

    fn execute(
        &mut self,
        plan: &CpuKernelPlan,
        activation: &[f32],
        weight: NativeRef<'_>,
        output: &mut [f32],
        parallel: &dyn ForkJoin,
    ) -> Result<bool, AxCompileError>;

    /// Optional heterogeneous execution of a checkpoint-native NVFP4 projection. Returning
    /// `false` preserves the selected CPU kernel. `scratch` is the plan's reusable native
    /// scratch and may be used by a concurrent CPU shard.
    #[allow(clippy::too_many_arguments)]
    fn execute_nvfp4(
        &mut self,
        _plan: &CpuKernelPlan,
        _activation: &[f32],
        _packed: &[u8],
        _block_scales: &[u8],
        _output: &mut [f32],
        _scratch: &mut [f32],
        _parallel: &dyn ForkJoin,
    ) -> Result<bool, AxCompileError> {
        Ok(false)
    }

    /// Optional execution of the **up projections alone** of one token's
    /// routed NVFP4 experts: write each expert's pre-activation output
    /// (`experts.len() × intermediate`, in the order given, weight scales
    /// applied, no epilogue) into `activations`.  The CPU kernel then
    /// applies the activation, fetches the down rows and mixes — the grain
    /// a streamed-expert path needs, whose down rows are sparse and chosen
    /// by the activation, and the one an accelerator's expert-up package
    /// computes.  Returning `false` keeps the CPU up projections.
    #[allow(clippy::too_many_arguments)]
    fn execute_moe_up_nvfp4(
        &mut self,
        _plan: &CpuKernelPlan,
        _params: MoeParams,
        _hidden: &[f32],
        _experts: &[ExpertWeights<'_>],
        _activations: &mut [f32],
        _parallel: &dyn ForkJoin,
    ) -> Result<bool, AxCompileError> {
        Ok(false)
    }
    /// Optional execution of one routed and resident NVFP4 expert batch.
    /// Experts and `mixture_weights` are aligned in ascending expert-number
    /// order. Returning `false` invokes the selected CPU MoE kernel.
    #[allow(clippy::too_many_arguments)]
    fn execute_moe_nvfp4(
        &mut self,
        _plan: &CpuKernelPlan,
        _params: MoeParams,
        _hidden: &[f32],
        _experts: &[ExpertWeights<'_>],
        _mixture_weights: &[f32],
        _output: &mut [f32],
        _scratch: &mut [f32],
        _parallel: &dyn ForkJoin,
    ) -> Result<bool, AxCompileError> {
        Ok(false)
    }
}

/// A [`ProjectionExecutor`] as the MoE kernel's delegate: a whole token to
/// [`execute_moe_nvfp4`](ProjectionExecutor::execute_moe_nvfp4), or its up
/// projections alone to
/// [`execute_moe_up_nvfp4`](ProjectionExecutor::execute_moe_up_nvfp4).
struct ProjectionMoeDelegate<'e, 'p> {
    executor: &'e mut dyn ProjectionExecutor,
    parallel: &'p dyn ForkJoin,
}

/// Bind resource-local token identity and graph accuracy to the same runtime
/// executor that owns pure-kernel admission and completion.
struct PreparedMoeDelegate<'e> {
    executor: &'e mut dyn crate::PreparedKernelExecutor,
    instruction: usize,
    minimum_accuracy: KernelAccuracy,
    check: Option<&'e mut DelegateCheck>,
    parallel: &'e dyn ForkJoin,
    profile: Option<&'e mut (fn() -> u64, OpProfile)>,
}

impl axnn_cpu::ResumableMoeDelegate<AxCompileError> for PreparedMoeDelegate<'_> {
    fn wave_capacity(
        &self,
        params: MoeParams,
        tokens: usize,
        routes: &[axnn_cpu::MoeBatchRow],
    ) -> usize {
        self.executor.moe_wave_capacity(params, tokens, routes)
    }

    fn experts(
        &mut self,
        wave: usize,
        plan: &CpuKernelPlan,
        params: MoeParams,
        hidden: &[f32],
        routes: &[axnn_cpu::MoeBatchRow],
        experts: &[ExpertWeights<'_>],
        output: &mut [f32],
    ) -> Result<axnn_cpu::MoeProgress, AxCompileError> {
        // A checked wave keeps the output as the wave found it: the first
        // offer, not a retry of the same wave.
        if let Some(check) = self.check.as_deref_mut()
            && check.wave != Some((self.instruction, wave))
        {
            check.before.clear();
            check.before.extend_from_slice(output);
            check.wave = Some((self.instruction, wave));
        }
        let progress = self.executor.advance_moe_experts(
            self.instruction,
            wave,
            plan,
            params,
            hidden,
            routes,
            experts,
            output,
            self.minimum_accuracy,
        )?;
        if progress == crate::KernelProgress::Complete
            && let Some(check) = self.check.as_deref_mut()
        {
            check.wave = None;
            let mut native = core::mem::take(&mut check.before);
            check.contribution.resize(params.hidden as usize, 0.0);
            check
                .scratch
                .resize(axnn_cpu::moe_scratch_elements(params), 0.0);
            let started = self.profile.as_ref().map(|(clock, _)| clock());
            let result = axnn_cpu::native_wave(
                plan,
                params,
                hidden,
                routes,
                experts,
                &mut native,
                &mut check.contribution,
                &mut check.scratch,
                self.parallel,
            );
            if let (Some(started), Some((clock, profile))) = (started, self.profile.as_mut()) {
                profile.reference.add(0, clock().saturating_sub(started));
            }
            result?;
            check.report.record(0, self.instruction, output, &native);
            output.copy_from_slice(&native);
            check.before = native;
        }
        Ok(match progress {
            crate::KernelProgress::Unsupported => axnn_cpu::MoeProgress::Unsupported,
            crate::KernelProgress::Pending => axnn_cpu::MoeProgress::Pending,
            crate::KernelProgress::Complete => axnn_cpu::MoeProgress::Complete,
        })
    }
}

impl axnn_cpu::MoeDelegate<AxCompileError> for ProjectionMoeDelegate<'_, '_> {
    fn token(
        &mut self,
        plan: &CpuKernelPlan,
        params: MoeParams,
        hidden: &[f32],
        experts: &[ExpertWeights<'_>],
        mixture_weights: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<bool, AxCompileError> {
        self.executor.execute_moe_nvfp4(
            plan,
            params,
            hidden,
            experts,
            mixture_weights,
            output,
            scratch,
            self.parallel,
        )
    }

    fn up(
        &mut self,
        plan: &CpuKernelPlan,
        params: MoeParams,
        hidden: &[f32],
        experts: &[ExpertWeights<'_>],
        activations: &mut [f32],
    ) -> Result<bool, AxCompileError> {
        self.executor.execute_moe_up_nvfp4(
            plan,
            params,
            hidden,
            experts,
            activations,
            self.parallel,
        )
    }
}

pub const CPU_BACKEND: u16 = 0;
pub const CPU_KERNEL_IMPLEMENTATION: Implementation = Implementation(4);
/// One immutable native resource borrowed for a graph invocation.
///
/// The payload is never copied into tensor ingress or retained in an executable
/// or prepared plan. Host edits become visible on the next invocation with
/// the updated borrow, even when every tensor input is unchanged.
#[derive(Clone, Copy, Debug)]
pub struct NativeResourceBinding<'a> {
    resource: ResourceKey,
    value: NativeResourceValue<'a>,
}

/// The semantic layout of a borrowed native resource, independent of tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeResourceKind {
    BrickGrid,
    RgbVolume,
    U8Buffer,
    /// A packed-XRGB surface a raster node samples.
    XrgbSurface,
    /// The packed-XRGB target raster nodes paint.
    XrgbTarget,
    /// A glyph coverage atlas.
    CoverageAtlas,
    /// The placed glyphs a frame's text runs index.
    GlyphTable,
    /// One attention layer's key/value cache, which appends write.
    KvCache,
    /// Recurrent F32 state before a step, read only.
    StateSource,
    /// The F32 buffer a recurrent step writes its successor state into.
    StateTarget,
    /// One mixture-of-experts layer's expert weights, read only.
    ExpertStore,
}

#[derive(Clone, Copy, Debug)]
enum NativeResourceValue<'a> {
    BrickGrid(BrickGridRef<'a>),
    RgbVolume(RgbVolumeRef<'a>),
    U8Buffer(U8BufferRef<'a>),
    XrgbSurface(XrgbSurfaceRef<'a>),
    XrgbTarget(XrgbTargetRef<'a>),
    CoverageAtlas(CoverageAtlasRef<'a>),
    GlyphTable(GlyphTableRef<'a>),
    KvCache(KvCacheRef<'a>),
    ResidentKvCache(crate::ResidentKvCache),
    StateSource(StateSourceRef<'a>),
    StateTarget(StateTargetRef<'a>),
    ExpertStore(ExpertStoreRef<'a>),
}

impl<'a> NativeResourceBinding<'a> {
    pub const fn brick_grid(resource: ResourceKey, grid: BrickGridRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::BrickGrid(grid),
        }
    }

    pub const fn rgb_volume(resource: ResourceKey, volume: RgbVolumeRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::RgbVolume(volume),
        }
    }

    pub const fn resource(&self) -> ResourceKey {
        self.resource
    }

    pub const fn u8_buffer(resource: ResourceKey, buffer: U8BufferRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::U8Buffer(buffer),
        }
    }

    pub const fn xrgb_surface(resource: ResourceKey, surface: XrgbSurfaceRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::XrgbSurface(surface),
        }
    }

    pub const fn xrgb_target(resource: ResourceKey, target: XrgbTargetRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::XrgbTarget(target),
        }
    }

    pub const fn coverage_atlas(resource: ResourceKey, atlas: CoverageAtlasRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::CoverageAtlas(atlas),
        }
    }

    pub const fn glyph_table(resource: ResourceKey, glyphs: GlyphTableRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::GlyphTable(glyphs),
        }
    }

    pub const fn kv_cache(resource: ResourceKey, cache: KvCacheRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::KvCache(cache),
        }
    }

    /// Bind runtime-owned storage without creating or borrowing a CPU mirror.
    pub const fn resident_kv_cache(cache: crate::ResidentKvCache) -> Self {
        Self {
            resource: cache.params().resource,
            value: NativeResourceValue::ResidentKvCache(cache),
        }
    }

    pub const fn state_source(resource: ResourceKey, state: StateSourceRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::StateSource(state),
        }
    }

    pub const fn state_target(resource: ResourceKey, state: StateTargetRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::StateTarget(state),
        }
    }

    pub const fn expert_store(resource: ResourceKey, store: ExpertStoreRef<'a>) -> Self {
        Self {
            resource,
            value: NativeResourceValue::ExpertStore(store),
        }
    }

    pub const fn kind(&self) -> NativeResourceKind {
        match self.value {
            NativeResourceValue::BrickGrid(_) => NativeResourceKind::BrickGrid,
            NativeResourceValue::RgbVolume(_) => NativeResourceKind::RgbVolume,
            NativeResourceValue::U8Buffer(_) => NativeResourceKind::U8Buffer,
            NativeResourceValue::XrgbSurface(_) => NativeResourceKind::XrgbSurface,
            NativeResourceValue::XrgbTarget(_) => NativeResourceKind::XrgbTarget,
            NativeResourceValue::CoverageAtlas(_) => NativeResourceKind::CoverageAtlas,
            NativeResourceValue::GlyphTable(_) => NativeResourceKind::GlyphTable,
            NativeResourceValue::KvCache(_) | NativeResourceValue::ResidentKvCache(_) => {
                NativeResourceKind::KvCache
            }
            NativeResourceValue::StateSource(_) => NativeResourceKind::StateSource,
            NativeResourceValue::StateTarget(_) => NativeResourceKind::StateTarget,
            NativeResourceValue::ExpertStore(_) => NativeResourceKind::ExpertStore,
        }
    }
}

/// Runtime-owned admission policy for automatic native CPU kernel lowering.
///
/// A host may use [`Default`] at its final `std` boundary. Kore can instead
/// supply the capability mask discovered from its CPU/device service and its
/// own fidelity or scratch-memory limits; this adapter never probes the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxCpuKernelPolicy {
    pub capabilities: CpuCapabilities,
    pub minimum_accuracy: KernelAccuracy,
    pub max_temporary_bytes: usize,
    pub temporary_byte_cost: u64,
}

impl AxCpuKernelPolicy {
    pub fn plan(self, request: &KernelRequest) -> Option<CpuKernelPlan> {
        axnn_cpu::select_cpu_kernel(
            request,
            KernelPolicy::new(self.capabilities.mask())
                .with_minimum_accuracy(self.minimum_accuracy)
                .with_max_temporary_bytes(self.max_temporary_bytes)
                .with_temporary_byte_cost(self.temporary_byte_cost),
        )
    }
}

impl Default for AxCpuKernelPolicy {
    fn default() -> Self {
        Self {
            capabilities: CpuCapabilities::compiled(),
            minimum_accuracy: KernelAccuracy::Faithful,
            max_temporary_bytes: usize::MAX,
            temporary_byte_cost: 0,
        }
    }
}

impl From<KernelPolicy> for AxCpuKernelPolicy {
    fn from(policy: KernelPolicy) -> Self {
        Self {
            capabilities: CpuCapabilities::from_mask(policy.capabilities),
            minimum_accuracy: policy.minimum_accuracy,
            max_temporary_bytes: policy.max_temporary_bytes,
            temporary_byte_cost: policy.temporary_byte_cost,
        }
    }
}

/// Persistent, allocator-owned native CPU plan for one scheduled region.
/// Individual scalar/SIMD/ISA choices stay inside each `CpuKernelPlan`; the
/// scheduler sees only [`CPU_KERNEL_IMPLEMENTATION`] for the whole region.
#[derive(Debug)]
pub struct NativeCpuPartition<A: Allocator> {
    pub inputs: Vec<TensorId, A>,
    pub outputs: Vec<TensorId, A>,
    pub nodes: Vec<NativeCpuNode, A>,
    pub instructions: Vec<PreparedNativeScheduleEntry, A>,
    pub fusions: bool,
    /// Admission policy also applies to any prepared execution specialization.
    pub capabilities: CpuCapabilities,
    /// Partition-local tensor table used by native nodes and liveness state.
    pub tensors: Vec<TensorId, A>,
    /// Remaining-use seed for invocation-local value reclamation. Graph
    /// outputs contribute one final use so intermediates can be dropped as
    /// soon as their last planned consumer completes.
    pub takes: Vec<u32, A>,
}

/// `elements` elements of an operand's source slot, from `offset`: a view
/// resolved at compile time. The node's input names the source, so liveness
/// and slot reuse count the view's reads as reads of the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeRange {
    pub offset: usize,
    pub elements: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeCpuNode {
    pub plan: CpuKernelPlan,
    pub inputs: [u32; 3],
    pub input_count: u8,
    pub output_slots: [u32; MAX_TENSOR_RESULTS],
    pub output_count: u8,
    /// The part of each input's slot the operand reads, when it is a view.
    /// Operands naming one slot always read the same part.
    pub input_ranges: [Option<NativeRange>; 3],
}

impl NativeCpuNode {
    pub fn has_ranges(&self) -> bool {
        self.input_ranges.iter().any(Option::is_some)
    }

    /// The part of `slot` this node's operands read, if a view.
    pub fn range_for(&self, slot: u32) -> Option<NativeRange> {
        self.inputs[..self.input_count as usize]
            .iter()
            .zip(&self.input_ranges)
            .find_map(|(&input, range)| (input == slot).then_some(*range).flatten())
    }

    pub fn outputs(&self) -> impl Iterator<Item = u32> + '_ {
        self.output_slots[..self.output_count as usize]
            .iter()
            .copied()
    }
}

/// Owned metadata only: resource borrows are resolved afresh for each run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeResourceRequirement {
    BrickGrid(VoxelBrickTraceParams),
    RgbVolume(VolumeRgbParams),
    U8Buffer(U8BufferParams),
    /// A raster node's target must be bound; its surfaces, atlas and glyph
    /// table are resolved per run and paint nothing when absent.  Keyed by
    /// the target alone, so every node of a program painting one target
    /// shares one requirement and one check per run.
    RasterTarget(ResourceKey),
    /// Appends and attention of one layer share its cache and requirement.
    KvCache(KvCacheParams),
    /// A recurrent step's source and target buffers, each of exactly
    /// `elements` F32 values.
    RecurrentState {
        state: StateBinding,
        elements: usize,
    },
    /// An expert store with exactly the layer's experts and projection sizes.
    ExpertStore(MoeParams),
}

impl NativeResourceRequirement {
    fn for_plan(plan: &CpuKernelPlan) -> Result<Option<Self>, AxCompileError> {
        match (plan.semantic, plan.params()) {
            (SemanticOp::VoxelBrickTrace, KernelParams::VoxelBrickTrace(params))
                if params.is_valid() =>
            {
                Ok(Some(Self::BrickGrid(params)))
            }
            (SemanticOp::VolumeTrilinearRgb, KernelParams::VolumeTrilinearRgb(params))
                if params.is_valid() =>
            {
                Ok(Some(Self::RgbVolume(params)))
            }
            (SemanticOp::BufferLoadU8, KernelParams::BufferLoadU8(params)) if params.is_valid() => {
                Ok(Some(Self::U8Buffer(params)))
            }
            (SemanticOp::Raster(_), KernelParams::Raster(params)) if params.is_valid() => {
                Ok(Some(Self::RasterTarget(params.target)))
            }
            (SemanticOp::KvAppend, KernelParams::KvAppend(params)) if params.is_valid() => {
                Ok(Some(Self::KvCache(params)))
            }
            (SemanticOp::AttentionDecode, KernelParams::AttentionDecode(params))
                if params.is_valid() =>
            {
                Ok(Some(Self::KvCache(params.cache)))
            }
            (SemanticOp::CausalConv1d, KernelParams::CausalConv1d(params)) if params.is_valid() => {
                Ok(params
                    .state_elements()
                    .map(|elements| Self::RecurrentState {
                        state: params.state,
                        elements,
                    }))
            }
            (SemanticOp::MoeExperts, KernelParams::MoeExperts(params)) if params.is_valid() => {
                Ok(Some(Self::ExpertStore(params)))
            }
            (SemanticOp::SsmScan, KernelParams::SsmScan(params)) if params.is_valid() => Ok(params
                .state_elements()
                .map(|elements| Self::RecurrentState {
                    state: params.state,
                    elements,
                })),
            (SemanticOp::GatedDeltaRule, KernelParams::GatedDeltaRule(params))
                if params.is_valid() =>
            {
                Ok(params
                    .state_elements()
                    .map(|elements| Self::RecurrentState {
                        state: params.state,
                        elements,
                    }))
            }
            (
                SemanticOp::VoxelBrickTrace
                | SemanticOp::VolumeTrilinearRgb
                | SemanticOp::BufferLoadU8
                | SemanticOp::Raster(_)
                | SemanticOp::KvAppend
                | SemanticOp::AttentionDecode
                | SemanticOp::CausalConv1d
                | SemanticOp::SsmScan
                | SemanticOp::GatedDeltaRule
                | SemanticOp::MoeExperts,
                _,
            ) => Err(AxArtifactError::InvalidPartition.into()),
            _ => Ok(None),
        }
    }

    pub fn validate(self, resources: &[NativeResourceBinding<'_>]) -> Result<(), AxCompileError> {
        match self {
            Self::BrickGrid(params) => {
                voxel_resource_for_params(params, resources)?;
            }
            Self::RgbVolume(params) => {
                volume_resource_for_params(params, resources)?;
            }
            Self::U8Buffer(params) => {
                buffer_resource_for_params(params, resources)?;
            }
            Self::RasterTarget(target) => {
                raster_target(target, resources)?;
            }
            Self::KvCache(params) => {
                kv_binding_for_params(params, resources)?;
            }
            Self::RecurrentState { state, elements } => {
                recurrent_state(state, elements, resources)?;
            }
            Self::ExpertStore(params) => {
                expert_store_for_params(params, resources)?;
            }
        }
        Ok(())
    }
}

/// The target a raster node paints, by its static key.
pub fn raster_target_for_params<'a>(
    params: RasterNodeParams,
    resources: &[NativeResourceBinding<'a>],
) -> Result<XrgbTargetRef<'a>, AxCompileError> {
    raster_target(params.target, resources)
}

/// The bound packed-XRGB target at `key`.
pub fn raster_target<'a>(
    key: ResourceKey,
    resources: &[NativeResourceBinding<'a>],
) -> Result<XrgbTargetRef<'a>, AxCompileError> {
    let binding = resources
        .iter()
        .find(|binding| binding.resource == key)
        .ok_or(AxCompileError::MissingResource(key))?;
    let NativeResourceValue::XrgbTarget(target) = binding.value else {
        return Err(AxCompileError::ResourceKind {
            resource: key,
            expected: NativeResourceKind::XrgbTarget,
            actual: binding.kind(),
        });
    };
    Ok(target)
}

/// A raster node's surface by slot in its surface group, when bound.
fn raster_surface<'a>(
    params: RasterNodeParams,
    slot: u32,
    resources: &[NativeResourceBinding<'a>],
) -> Option<XrgbSurfaceRef<'a>> {
    let key = ResourceKey {
        group: params.surfaces,
        binding: slot,
    };
    resources
        .iter()
        .find(|binding| binding.resource == key)
        .and_then(|binding| match binding.value {
            NativeResourceValue::XrgbSurface(surface) => Some(surface),
            _ => None,
        })
}

pub fn validate_resource_keys(
    resources: &[NativeResourceBinding<'_>],
) -> Result<(), AxCompileError> {
    for (index, binding) in resources.iter().enumerate() {
        if resources[..index]
            .iter()
            .any(|other| other.resource == binding.resource)
        {
            return Err(AxCompileError::DuplicateResource(binding.resource));
        }
    }
    Ok(())
}

pub fn voxel_resource<'a>(
    plan: &CpuKernelPlan,
    resources: &[NativeResourceBinding<'a>],
) -> Result<BrickGridRef<'a>, AxCompileError> {
    let KernelParams::VoxelBrickTrace(params) = plan.params() else {
        return Err(AxArtifactError::InvalidPartition.into());
    };
    voxel_resource_for_params(params, resources)
}

pub fn voxel_resource_for_params<'a>(
    params: VoxelBrickTraceParams,
    resources: &[NativeResourceBinding<'a>],
) -> Result<BrickGridRef<'a>, AxCompileError> {
    let binding = resources
        .iter()
        .find(|binding| binding.resource == params.resource)
        .ok_or(AxCompileError::MissingResource(params.resource))?;
    let NativeResourceValue::BrickGrid(grid) = binding.value else {
        return Err(AxCompileError::ResourceKind {
            resource: params.resource,
            expected: NativeResourceKind::BrickGrid,
            actual: binding.kind(),
        });
    };
    if grid.dims() != params.dims {
        return Err(AxCompileError::ResourceDimensions {
            resource: params.resource,
            expected: params.dims,
            actual: grid.dims(),
        });
    }
    Ok(grid)
}

pub fn volume_resource<'a>(
    plan: &CpuKernelPlan,
    resources: &[NativeResourceBinding<'a>],
) -> Result<RgbVolumeRef<'a>, AxCompileError> {
    let KernelParams::VolumeTrilinearRgb(params) = plan.params() else {
        return Err(AxArtifactError::InvalidPartition.into());
    };
    volume_resource_for_params(params, resources)
}

pub fn volume_resource_for_params<'a>(
    params: VolumeRgbParams,
    resources: &[NativeResourceBinding<'a>],
) -> Result<RgbVolumeRef<'a>, AxCompileError> {
    let binding = resources
        .iter()
        .find(|binding| binding.resource == params.resource)
        .ok_or(AxCompileError::MissingResource(params.resource))?;
    let NativeResourceValue::RgbVolume(volume) = binding.value else {
        return Err(AxCompileError::ResourceKind {
            resource: params.resource,
            expected: NativeResourceKind::RgbVolume,
            actual: binding.kind(),
        });
    };
    if volume.dims() != params.dims {
        return Err(AxCompileError::ResourceDimensions {
            resource: params.resource,
            expected: params.dims,
            actual: volume.dims(),
        });
    }
    if volume.storage() != params.storage {
        return Err(AxCompileError::ResourceStorage {
            resource: params.resource,
            expected: params.storage,
            actual: volume.storage(),
        });
    }
    Ok(volume)
}

pub fn buffer_resource_for_params<'a>(
    params: U8BufferParams,
    resources: &[NativeResourceBinding<'a>],
) -> Result<U8BufferRef<'a>, AxCompileError> {
    let binding = resources
        .iter()
        .find(|binding| binding.resource == params.resource)
        .ok_or(AxCompileError::MissingResource(params.resource))?;
    let NativeResourceValue::U8Buffer(buffer) = binding.value else {
        return Err(AxCompileError::ResourceKind {
            resource: params.resource,
            expected: NativeResourceKind::U8Buffer,
            actual: binding.kind(),
        });
    };
    if buffer.elements() != params.elements {
        return Err(AxCompileError::ResourceLength {
            resource: params.resource,
            expected: params.elements as usize,
            actual: buffer.bytes().len(),
        });
    }
    Ok(buffer)
}

/// The bound key/value cache with exactly the node's geometry and storage.
pub fn kv_cache_for_params<'a>(
    params: KvCacheParams,
    resources: &[NativeResourceBinding<'a>],
) -> Result<KvCacheRef<'a>, AxCompileError> {
    match kv_binding_for_params(params, resources)? {
        NativeResourceValue::KvCache(cache) => Ok(cache),
        _ => Err(AxCompileError::ResidentResourceRequiresExecutor(
            params.resource,
        )),
    }
}

fn kv_binding_for_params<'a>(
    params: KvCacheParams,
    resources: &[NativeResourceBinding<'a>],
) -> Result<NativeResourceValue<'a>, AxCompileError> {
    let binding = resources
        .iter()
        .find(|binding| binding.resource == params.resource)
        .ok_or(AxCompileError::MissingResource(params.resource))?;
    let actual = match binding.value {
        NativeResourceValue::KvCache(cache) => cache.params(),
        NativeResourceValue::ResidentKvCache(cache) => cache.params(),
        _ => {
            return Err(AxCompileError::ResourceKind {
                resource: params.resource,
                expected: NativeResourceKind::KvCache,
                actual: binding.kind(),
            });
        }
    };
    if actual != params {
        return Err(AxCompileError::KvCacheGeometry {
            resource: params.resource,
            expected: params,
            actual,
        });
    }
    Ok(binding.value)
}

/// The bound expert store, checked against the layer's geometry.
pub fn expert_store_for_params<'a>(
    params: MoeParams,
    resources: &[NativeResourceBinding<'a>],
) -> Result<ExpertStoreRef<'a>, AxCompileError> {
    let binding = resources
        .iter()
        .find(|binding| binding.resource == params.resource)
        .ok_or(AxCompileError::MissingResource(params.resource))?;
    let NativeResourceValue::ExpertStore(store) = binding.value else {
        return Err(AxCompileError::ResourceKind {
            resource: params.resource,
            expected: NativeResourceKind::ExpertStore,
            actual: binding.kind(),
        });
    };
    store.check(params)?;
    Ok(store)
}

/// A recurrent step's bound source and target, each of exactly `elements`.
/// The state binding and state elements of a recurrent kernel (a Mamba
/// causal convolution or state-space scan, or a gated delta rule), the
/// instructions an admission window offers with their state.
fn recurrent_binding(plan: &CpuKernelPlan) -> Option<(StateBinding, usize)> {
    match (plan.semantic, plan.params()) {
        (SemanticOp::CausalConv1d, KernelParams::CausalConv1d(params)) => {
            Some((params.state, params.state_elements()?))
        }
        (SemanticOp::SsmScan, KernelParams::SsmScan(params)) => {
            Some((params.state, params.state_elements()?))
        }
        (SemanticOp::GatedDeltaRule, KernelParams::GatedDeltaRule(params)) => {
            Some((params.state, params.state_elements()?))
        }
        _ => None,
    }
}

pub fn recurrent_state<'a>(
    state: StateBinding,
    elements: usize,
    resources: &[NativeResourceBinding<'a>],
) -> Result<(StateSourceRef<'a>, StateTargetRef<'a>), AxCompileError> {
    let find = |key: ResourceKey| {
        resources
            .iter()
            .find(|binding| binding.resource == key)
            .ok_or(AxCompileError::MissingResource(key))
    };
    let (source, target) = (find(state.source)?, find(state.target)?);
    let NativeResourceValue::StateSource(source_ref) = source.value else {
        return Err(AxCompileError::ResourceKind {
            resource: state.source,
            expected: NativeResourceKind::StateSource,
            actual: source.kind(),
        });
    };
    let NativeResourceValue::StateTarget(target_ref) = target.value else {
        return Err(AxCompileError::ResourceKind {
            resource: state.target,
            expected: NativeResourceKind::StateTarget,
            actual: target.kind(),
        });
    };
    // A target may hold one state per token of the run (whole multiples of
    // the state): the kernel then keeps every intermediate state, and checks
    // the count against the tokens it runs.
    let target_fits = if elements == 0 {
        target_ref.len() == 0
    } else {
        target_ref.len() >= elements && target_ref.len().is_multiple_of(elements)
    };
    for (resource, actual, fits) in [
        (state.source, source_ref.len(), source_ref.len() == elements),
        (state.target, target_ref.len(), target_fits),
    ] {
        if !fits {
            return Err(AxCompileError::ResourceLength {
                resource,
                expected: elements,
                actual,
            });
        }
    }
    Ok((source_ref, target_ref))
}

pub fn validate_native_resources(
    nodes: &[NativeCpuNode],
    resources: &[NativeResourceBinding<'_>],
) -> Result<(), AxCompileError> {
    validate_resource_keys(resources)?;
    for node in nodes {
        if let Some(requirement) = NativeResourceRequirement::for_plan(&node.plan)? {
            requirement.validate(resources)?;
        }
    }
    Ok(())
}

/// Aligned native storage retained across CPU and provider boundaries.
///
/// Aligned little-endian inputs are borrowed. Unaligned or non-native-endian
/// bytes are decoded explicitly. Owned outputs keep their typed allocation
/// until a caller requests the legacy byte-vector ABI.
pub enum NativeValue<A: Allocator> {
    F32(Vec<f32, A>),
    I32(Vec<i32, A>),
    U8(Vec<u8, A>),
    /// Canonical packed blocks; the dtype is preserved across slot reuse.
    Packed(DType, Vec<u8, A>),
}

#[derive(Clone, Copy)]
pub struct PreparedNativeBinding {
    pub tensor: TensorId,
    pub slot: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PreparedNativeLayout {
    pub dtype: DType,
    pub bytes: usize,
}

impl PreparedNativeLayout {
    fn elements(self) -> usize {
        self.bytes / self.dtype.block_byte_size() * self.dtype.block_size()
    }

    fn matches(self, dtype: DType, elements: usize) -> bool {
        self.dtype == dtype
            && elements.is_multiple_of(dtype.block_size())
            && self.elements() == elements
    }
}

/// Preparation-only logical state. Physical slots are assigned only after the
/// original logical graph and every node descriptor have been validated.
pub struct PreparedNativeLifetime {
    pub layout: PreparedNativeLayout,
    pub ingress: bool,
    pub pinned: bool,
    pub available: bool,
    pub remaining_uses: usize,
    pub slot: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum PreparedMulAddKernel {
    Scalar,
    Scale {
        input: usize,
        order: ScaleAddOrder,
        scalar_addend: bool,
    },
    Dense,
}

/// A strict two-operation instruction, not an IEEE fused multiply-add. Inputs
/// keep the original MUL lhs/rhs followed by ADD's other operand. No resource
/// pointer or intermediate product allocation is retained.
#[derive(Clone, Copy, Debug)]
pub struct PreparedMulAdd {
    pub inputs: [u32; 3],
    pub output_slots: [u32; 1],
    pub input_elements: [usize; 3],
    pub elements: usize,
    pub order: MulAddOrder,
    pub kernel: PreparedMulAddKernel,
}

/// A SUB feeding the first FMA operand, preserving subtraction and the
/// separate multiply/add. Four operand occurrences retain their logical
/// identities even when a color blend reads its base twice.
#[derive(Clone, Copy, Debug)]
pub struct PreparedSubMulAdd {
    pub inputs: [u32; 4],
    pub output_slots: [u32; 1],
    pub input_elements: [usize; 4],
    pub elements: usize,
    pub simd: bool,
}

/// Three dense planes multiplied by one shared scalar, then packed as xyz.
/// Each axis retains the authored MUL operand order. The four logical uses
/// participate in the same lifetime planning as ordinary prepared operands.
#[derive(Clone, Copy, Debug)]
pub struct PreparedScaleInterleave3 {
    pub inputs: [u32; 4],
    pub output_slots: [u32; 1],
    pub elements: usize,
    pub scale_first: [bool; 3],
    pub simd: bool,
}

/// Three dense F32 casts and their optional scalar shifts, combined by OR.
/// Plane operands come first, followed by zero to three scalar shift operands.
/// Keeping each occurrence lets liveness account for shared planes/counts.
#[derive(Clone, Copy, Debug)]
pub struct PreparedCastShiftOr3 {
    pub inputs: [u32; 6],
    pub input_count: u8,
    pub shift_operands: [Option<u8>; 3],
    pub output_slots: [u32; 1],
    pub elements: usize,
}

// Plans remain inline in the caller-allocated instruction vector. Boxing the
// checked kernel variant would add a separate allocation for every node.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug)]
pub enum PreparedNativeInstruction {
    Kernel(NativeCpuNode),
    SubMulAdd(PreparedSubMulAdd),
    MulAdd(PreparedMulAdd),
    ScaleInterleave3(PreparedScaleInterleave3),
    CastShiftOr3(PreparedCastShiftOr3),
}

impl PreparedNativeInstruction {
    pub fn inputs(&self) -> &[u32] {
        match self {
            Self::Kernel(node) => &node.inputs[..node.input_count as usize],
            Self::SubMulAdd(node) => &node.inputs,
            Self::MulAdd(node) => &node.inputs,
            Self::ScaleInterleave3(node) => &node.inputs,
            Self::CastShiftOr3(node) => &node.inputs[..node.input_count as usize],
        }
    }

    pub fn inputs_mut(&mut self) -> &mut [u32] {
        match self {
            Self::Kernel(node) => &mut node.inputs[..node.input_count as usize],
            Self::SubMulAdd(node) => &mut node.inputs,
            Self::MulAdd(node) => &mut node.inputs,
            Self::ScaleInterleave3(node) => &mut node.inputs,
            Self::CastShiftOr3(node) => &mut node.inputs[..node.input_count as usize],
        }
    }

    pub fn output_slots(&self) -> &[u32] {
        match self {
            Self::Kernel(node) => &node.output_slots[..node.output_count as usize],
            Self::SubMulAdd(node) => &node.output_slots,
            Self::MulAdd(node) => &node.output_slots,
            Self::ScaleInterleave3(node) => &node.output_slots,
            Self::CastShiftOr3(node) => &node.output_slots,
        }
    }

    pub fn output_slots_mut(&mut self) -> &mut [u32] {
        match self {
            Self::Kernel(node) => &mut node.output_slots[..node.output_count as usize],
            Self::SubMulAdd(node) => &mut node.output_slots,
            Self::MulAdd(node) => &mut node.output_slots,
            Self::ScaleInterleave3(node) => &mut node.output_slots,
            Self::CastShiftOr3(node) => &mut node.output_slots,
        }
    }

    pub fn outputs(&self) -> impl Iterator<Item = u32> + '_ {
        self.output_slots().iter().copied()
    }
}

/// Preparation metadata uses indices for unchanged kernels rather than a
/// second copy of every (large) CpuKernelPlan. Elided entries never execute.
#[derive(Clone, Copy, Debug)]
pub enum PreparedNativeScheduleEntry {
    Kernel(usize),
    SubMulAdd(PreparedSubMulAdd),
    MulAdd(PreparedMulAdd),
    ScaleInterleave3(PreparedScaleInterleave3),
    CastShiftOr3(PreparedCastShiftOr3),
    Elided,
}

impl PreparedNativeScheduleEntry {
    pub fn instruction(self, nodes: &[NativeCpuNode]) -> Option<PreparedNativeInstruction> {
        match self {
            Self::Kernel(index) => Some(PreparedNativeInstruction::Kernel(nodes[index])),
            Self::SubMulAdd(node) => Some(PreparedNativeInstruction::SubMulAdd(node)),
            Self::MulAdd(node) => Some(PreparedNativeInstruction::MulAdd(node)),
            Self::ScaleInterleave3(node) => Some(PreparedNativeInstruction::ScaleInterleave3(node)),
            Self::CastShiftOr3(node) => Some(PreparedNativeInstruction::CastShiftOr3(node)),
            Self::Elided => None,
        }
    }
}

/// Immutable, allocator-owned instructions shared by independent native arenas.
/// The allocator that owns the instruction allocation stays with the last
/// reference, even when the original prepared arena has already been dropped.
pub struct PreparedNativeInstructions<A: Allocator>(Arc<Vec<PreparedNativeInstruction, A>, A>);

impl<A: Allocator> Deref for PreparedNativeInstructions<A> {
    type Target = [PreparedNativeInstruction];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, A: Allocator> IntoIterator for &'a PreparedNativeInstructions<A> {
    type Item = &'a PreparedNativeInstruction;
    type IntoIter = core::slice::Iter<'a, PreparedNativeInstruction>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Reusable storage for one native CPU `.ax` region with pure operations and
/// explicitly borrowed, immutable occupancy and RGB volume resources.
///
/// Preparation copies the executable's selected CPU plans and assigns aligned
/// storage with the supplied allocator. Tensors with disjoint live ranges may
/// share storage, but each node's output is distinct from every operand. All
/// inputs are live at ingress, constants survive across runs, and graph outputs
/// remain pinned until the next run. Single-use strict F32 MUL→ADD pairs may
/// share an instruction, preserving both rounding boundaries and operand
/// orders. Three dense MULs by one scalar may also share their subsequent
/// INTERLEAVE3 instruction without storing intermediate product planes.
/// Three dense F32→I32 casts, optionally shifted by scalar counts, may share
/// their OR sink without intermediate integer planes. Cast saturation and
/// wrapping shifts retain their exact semantics; no color transform is added.
/// There is no FMA contraction or reassociation. Each run validates
/// its inputs and invokes kernels without allocating or encoding output byte
/// planes; any SIMD specialization respects the original CPU policy.
/// Arithmetic NaN results follow Rust's unspecified payload/sign selection;
/// only their NaN classification is preserved. Finite results, signed zeros
/// and infinities retain the unfused operations' exact bits. Identity/copy
/// operations continue to preserve stored NaN payloads.
///
/// This object owns its plans and constants and may outlive the source
/// executable and provider sessions. Output slices borrow the latest successful
/// run; starting another run invalidates them, including when that run fails.
pub struct PreparedNative<A: Allocator> {
    pub nodes: PreparedNativeInstructions<A>,
    /// Last reader of each instruction's single result before its physical
    /// slot is overwritten. MAX means observable at the graph boundary.
    /// Shared with forks; computed once, never scanned on the execution path.
    output_last_use: Arc<Vec<usize, A>, A>,
    // Preparation deduplicates exact key/kind/dimension/storage requirements. No resource
    // view or pointer survives a run, and pure graphs have an empty list.
    pub resource_requirements: Vec<NativeResourceRequirement, A>,
    pub values: Vec<Option<NativeValue<A>>, A>,
    pub inputs: Vec<PreparedNativeBinding, A>,
    /// Logical-operand use counts for each graph input after fusion. Borrowed
    /// ingress must stop shadowing a physical slot once its last consumer
    /// retires because the planner may recycle that slot for a later result.
    pub borrowed_input_takes: Vec<u32, A>,
    pub borrowed_input_remaining: Vec<u32, A>,
    /// Physical slot to graph-input index, or `usize::MAX` when the slot is
    /// owned by the arena. This keeps operand lookup constant-time in kernels.
    pub borrowed_input_by_slot: Vec<usize, A>,
    /// Passthrough graph inputs are also outputs and therefore must remain in
    /// arena-owned storage after the call returns.
    pub borrowed_input_copy: Vec<bool, A>,
    pub outputs: Vec<PreparedNativeBinding, A>,
    pub buffer_bytes: usize,
    pub logical_buffers: usize,
    pub logical_bytes: usize,
    pub voxel_ids: Vec<u32, A>,
    /// F32 kernel scratch sized to the largest node that needs it: cache
    /// attention rows and softmax state, and expert activations.
    pub kernel_scratch: Vec<f32, A>,
    /// Inputs have no arena storage and must be borrowed by every run.
    pub borrowed_ingress: bool,
    pub complete: bool,
    /// The profiler clock and its tally, while one is installed.
    pub profile: Option<(fn() -> u64, OpProfile)>,
    /// Recompute delegated results natively and compare, while enabled
    /// ([`Self::set_delegate_check`]).
    pub check: Option<DelegateCheck>,
}

impl<A: Allocator> PreparedNative<A> {
    /// Create an independent execution arena while sharing immutable instructions.
    ///
    /// Constants, ingress, output slots and voxel scratch belong to the supplied
    /// allocator. No tensor storage or resource borrow is shared. The fork may
    /// execute concurrently and outlive this arena; it publishes no output until
    /// its own first successful run. Allocator types match, but allocator values
    /// may name different heaps with compatible lifetimes.
    pub fn fork_in(&self, allocator: A) -> Result<Self, AxCompileError>
    where
        A: Clone,
    {
        let mut values = Vec::new_in(allocator.clone());
        reserve(&mut values, self.values.len())?;
        for value in &self.values {
            values.push(match value {
                Some(NativeValue::F32(v)) => {
                    Some(NativeValue::F32(copy_slice_in(v, allocator.clone())?))
                }
                Some(NativeValue::I32(v)) => {
                    Some(NativeValue::I32(copy_slice_in(v, allocator.clone())?))
                }
                Some(NativeValue::U8(v)) => {
                    Some(NativeValue::U8(copy_slice_in(v, allocator.clone())?))
                }
                Some(NativeValue::Packed(dtype, v)) => Some(NativeValue::Packed(
                    *dtype,
                    copy_slice_in(v, allocator.clone())?,
                )),
                None => None,
            });
        }
        let inputs = copy_slice_in(&self.inputs, allocator.clone())?;
        let borrowed_input_takes = copy_slice_in(&self.borrowed_input_takes, allocator.clone())?;
        let borrowed_input_remaining =
            copy_slice_in(&self.borrowed_input_remaining, allocator.clone())?;
        let borrowed_input_by_slot =
            copy_slice_in(&self.borrowed_input_by_slot, allocator.clone())?;
        let borrowed_input_copy = copy_slice_in(&self.borrowed_input_copy, allocator.clone())?;
        let outputs = copy_slice_in(&self.outputs, allocator.clone())?;
        let resource_requirements = copy_slice_in(&self.resource_requirements, allocator.clone())?;
        let mut voxel_ids = Vec::new_in(allocator.clone());
        reserve(&mut voxel_ids, self.voxel_ids.len())?;
        voxel_ids.resize(self.voxel_ids.len(), 0);
        let mut kernel_scratch = Vec::new_in(allocator);
        reserve(&mut kernel_scratch, self.kernel_scratch.len())?;
        kernel_scratch.resize(self.kernel_scratch.len(), 0.0);
        Ok(Self {
            nodes: PreparedNativeInstructions(Arc::clone(&self.nodes.0)),
            output_last_use: Arc::clone(&self.output_last_use),
            resource_requirements,
            values,
            inputs,
            borrowed_input_takes,
            borrowed_input_remaining,
            borrowed_input_by_slot,
            borrowed_input_copy,
            outputs,
            buffer_bytes: self.buffer_bytes,
            logical_buffers: self.logical_buffers,
            logical_bytes: self.logical_bytes,
            voxel_ids,
            kernel_scratch,
            borrowed_ingress: self.borrowed_ingress,
            complete: false,
            profile: None,
            check: None,
        })
    }

    /// Execute changing little-endian F32, I32 or U8 inputs in graph input order.
    ///
    /// All input counts and lengths are checked before any input is copied.
    /// Pure MATMUL destinations are reset on every run because the existing
    /// GEMM invoker accumulates into its caller-provided destination.
    pub fn run(&mut self, inputs: &[&[u8]]) -> Result<(), AxCompileError> {
        self.run_with_resources(inputs, &[])
    }

    /// Time every instruction from now on, by kernel class, on `clock`
    /// (monotonic nanoseconds).
    pub fn set_profiler(&mut self, clock: fn() -> u64) {
        self.profile = Some((clock, OpProfile::default()));
    }

    /// The tally so far; zero when no profiler is installed.
    pub fn profile(&self) -> OpProfile {
        self.profile.map(|(_, profile)| profile).unwrap_or_default()
    }

    /// Recompute every result a delegate (an executor or a MoE wave's
    /// device) completes with the native kernel, from the same inputs, and
    /// compare ([`Self::delegate_check`]); the native result then stands, so
    /// one wrong kernel does not spoil the inputs of the next. A diagnostic:
    /// it spends the CPU time the delegate saved.
    pub fn set_delegate_check(&mut self, enabled: bool) {
        self.check = enabled.then(DelegateCheck::default);
    }

    /// The comparison so far; empty unless checking.
    pub fn delegate_check(&self) -> CheckReport {
        self.check
            .as_ref()
            .map(|check| check.report)
            .unwrap_or_default()
    }

    /// Execute with immutable resources borrowed only for this call.
    ///
    /// Binding keys must be unique. Required kinds, dimensions and storage are
    /// checked before decoding inputs or running any kernel. An invalid run
    /// hides previous outputs and leaves the plan usable for a later run.
    pub fn run_with_resources(
        &mut self,
        inputs: &[&[u8]],
        resources: &[NativeResourceBinding<'_>],
    ) -> Result<(), AxCompileError> {
        self.complete = false;
        if self.borrowed_ingress {
            // Byte ingress decodes into arena slots this arena never
            // allocated.
            return Err(AxCompileError::UnsupportedImplementation {
                backend: CPU_BACKEND,
                implementation: CPU_KERNEL_IMPLEMENTATION,
            });
        }
        if inputs.len() != self.inputs.len() {
            return Err(AxCompileError::InvalidInputCount {
                expected: self.inputs.len(),
                actual: inputs.len(),
            });
        }
        for (binding, bytes) in self.inputs.iter().zip(inputs) {
            if bytes.len() != binding.bytes {
                return Err(AxCompileError::InvalidInputSize {
                    tensor: binding.tensor,
                    expected: binding.bytes,
                    actual: bytes.len(),
                });
            }
        }
        validate_resource_keys(resources)?;
        for &requirement in &self.resource_requirements {
            requirement.validate(resources)?;
        }
        for (binding, bytes) in self.inputs.iter().zip(inputs) {
            let value = self.values[binding.slot]
                .as_mut()
                .ok_or(AxArtifactError::InvalidPartition)?;
            decode_prepared_native(value, bytes);
        }
        for node in &self.nodes {
            // Move only the Vec handle, not its allocation. This gives the
            // outputs unique mutable borrows while repeated input operands may
            // share immutable borrows. Preparation rejects all output aliasing.
            let mut outputs: [Option<NativeValue<A>>; MAX_TENSOR_RESULTS] =
                core::array::from_fn(|_| None);
            let result = (|| {
                for (destination, slot) in outputs.iter_mut().zip(node.outputs()) {
                    *destination = self
                        .values
                        .get_mut(slot as usize)
                        .ok_or(AxArtifactError::InvalidPartition)?
                        .take();
                    if destination.is_none() {
                        return Err(AxArtifactError::InvalidPartition.into());
                    }
                }
                invoke_native_instruction(
                    node,
                    self.values.as_slice(),
                    &mut outputs[..node.output_slots().len()],
                    resources,
                    &mut self.voxel_ids,
                    &mut self.kernel_scratch,
                    &Serial,
                )
            })();
            // Restore every moved handle even if extraction or invocation
            // failed. A failed run never publishes a subset of its results.
            for (source, slot) in outputs.iter_mut().zip(node.outputs()) {
                if let Some(value) = source.take() {
                    self.values[slot as usize] = Some(value);
                }
            }
            result?;
        }
        self.complete = true;
        Ok(())
    }

    /// Execute with call-scoped typed ingress without encoding or copying
    /// ordinary inputs into the arena. Passthrough outputs are the sole copy:
    /// they must remain owned and observable after this call returns.
    pub fn run_with_native_inputs_and_resources(
        &mut self,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'_>],
    ) -> Result<(), AxCompileError> {
        self.run_native_on(inputs, resources, &Serial)
    }

    /// [`Self::run_with_native_inputs_and_resources`], splitting large
    /// kernels across the caller's workers. Results are identical to a
    /// serial run.
    pub fn run_native_on(
        &mut self,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'_>],
        parallel: &dyn ForkJoin,
    ) -> Result<(), AxCompileError> {
        self.run_native_with_projections(inputs, resources, parallel, None)
    }

    /// Execute with an optional projection dispatcher, preserving the native
    /// executor's resource validation and state transition order.
    pub fn run_native_with_projections(
        &mut self,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'_>],
        parallel: &dyn ForkJoin,
        mut projections: Option<&mut dyn ProjectionExecutor>,
    ) -> Result<(), AxCompileError> {
        self.begin_native(inputs, resources)?;
        for index in 0..self.nodes.len() {
            let started = self.profile.as_ref().map(|(clock, _)| clock());
            let result = self.run_native_node(
                index,
                inputs,
                resources,
                parallel,
                &mut projections,
                None,
                false,
                KernelAccuracy::Strict,
                &mut None,
                CpuExecution::Planned,
            );
            self.profile_node(index, started);
            result?;
        }
        self.complete = true;
        Ok(())
    }

    pub(crate) fn begin_native(
        &mut self,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'_>],
    ) -> Result<(), AxCompileError> {
        self.complete = false;
        if inputs.len() != self.inputs.len() {
            return Err(AxCompileError::InvalidInputCount {
                expected: self.inputs.len(),
                actual: inputs.len(),
            });
        }
        for (binding, input) in self.inputs.iter().zip(inputs) {
            if input.as_bytes().len() != binding.bytes {
                return Err(AxCompileError::InvalidInputSize {
                    tensor: binding.tensor,
                    expected: binding.bytes,
                    actual: input.as_bytes().len(),
                });
            }
            let expected = self
                .values
                .get(binding.slot)
                .and_then(Option::as_ref)
                .ok_or(AxArtifactError::InvalidPartition)?;
            if input.dtype() != expected.view().dtype()
                || !matches!(
                    (input, expected),
                    (NativeRef::F32(_), NativeValue::F32(_))
                        | (NativeRef::I32(_), NativeValue::I32(_))
                        | (NativeRef::U8(_), NativeValue::U8(_))
                        | (NativeRef::Packed(_, _), NativeValue::Packed(_, _))
                )
            {
                return Err(AxArtifactError::InvalidPartition.into());
            }
        }
        validate_resource_keys(resources)?;
        for &requirement in &self.resource_requirements {
            requirement.validate(resources)?;
        }
        self.borrowed_input_remaining
            .copy_from_slice(&self.borrowed_input_takes);
        for ((binding, input), &copy) in self
            .inputs
            .iter()
            .zip(inputs)
            .zip(&self.borrowed_input_copy)
        {
            if !copy {
                continue;
            }
            match (
                input,
                self.values[binding.slot]
                    .as_mut()
                    .ok_or(AxArtifactError::InvalidPartition)?,
            ) {
                (NativeRef::F32(source), NativeValue::F32(destination)) => {
                    destination.copy_from_slice(source)
                }
                (NativeRef::I32(source), NativeValue::I32(destination)) => {
                    destination.copy_from_slice(source)
                }
                (NativeRef::U8(source), NativeValue::U8(destination)) => {
                    destination.copy_from_slice(source)
                }
                (NativeRef::Packed(_, source), NativeValue::Packed(_, destination)) => {
                    destination.copy_from_slice(source)
                }
                _ => return Err(AxArtifactError::InvalidPartition.into()),
            }
        }
        Ok(())
    }

    pub(crate) fn profile_node(&mut self, index: usize, started: Option<u64>) {
        if let (Some(started), Some((clock, profile))) = (started, self.profile.as_mut()) {
            profile.add(
                op_class(&self.nodes[index]),
                clock().saturating_sub(started),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_native_node<'r>(
        &mut self,
        index: usize,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'r>],
        parallel: &dyn ForkJoin,
        projections: &mut Option<&mut dyn ProjectionExecutor>,
        mut delegate: Option<&mut dyn crate::PreparedKernelExecutor>,
        pending: bool,
        minimum_accuracy: KernelAccuracy,
        moe: &mut Option<axnn_cpu::MoeInvocation<'r>>,
        mut cpu_execution: CpuExecution,
    ) -> Result<crate::KernelProgress, AxCompileError> {
        let node = &self.nodes[index];
        let mut outputs: [Option<NativeValue<A>>; MAX_TENSOR_RESULTS] =
            core::array::from_fn(|_| None);
        let mut checked = false;
        let result = (|| {
            for (destination, slot) in outputs.iter_mut().zip(node.outputs()) {
                *destination = self
                    .values
                    .get_mut(slot as usize)
                    .ok_or(AxArtifactError::InvalidPartition)?
                    .take();
                if destination.is_none() {
                    return Err(AxArtifactError::InvalidPartition.into());
                }
            }
            let values = PreparedNativeInputs {
                values: self.values.as_slice(),
                input_by_slot: &self.borrowed_input_by_slot,
                inputs,
                remaining: &self.borrowed_input_remaining,
            };
            if let PreparedNativeInstruction::Kernel(kernel) = node {
                let params = match kernel.plan.params() {
                    KernelParams::KvAppend(params) => Some(params),
                    KernelParams::AttentionDecode(params) => Some(params.cache),
                    _ => None,
                };
                if let Some(params) = params
                    && let NativeResourceValue::ResidentKvCache(cache) =
                        kv_binding_for_params(params, resources)?
                {
                    // There is no host mirror to compare or fall back to.
                    // Refuse before publishing any cache operation in checked mode.
                    if self.check.is_some() {
                        return Err(AxCompileError::ResidentResourceRequiresExecutor(
                            params.resource,
                        ));
                    }
                    let executor = delegate.as_deref_mut().ok_or(
                        AxCompileError::ResidentResourceRequiresExecutor(params.resource),
                    )?;
                    let operands = OperandInputs {
                        values: &values,
                        node: kernel,
                    };
                    let mut borrowed = [None; 3];
                    for (input, &slot) in borrowed
                        .iter_mut()
                        .zip(&kernel.inputs[..kernel.input_count as usize])
                    {
                        *input = Some(
                            operands
                                .view(slot as usize)
                                .ok_or(AxArtifactError::InvalidPartition)?,
                        );
                    }
                    let output = match (&kernel.plan.params(), &mut outputs[0]) {
                        (KernelParams::KvAppend(_), Some(NativeValue::I32(token))) => {
                            crate::KvOutput::Appended(token)
                        }
                        (KernelParams::AttentionDecode(_), Some(NativeValue::F32(output))) => {
                            crate::KvOutput::Attention(output)
                        }
                        _ => return Err(AxArtifactError::InvalidPartition.into()),
                    };
                    return match executor.advance_kv(
                        index,
                        &kernel.plan,
                        cache,
                        &borrowed,
                        output,
                        minimum_accuracy,
                    )? {
                        crate::KernelProgress::Unsupported => Err(
                            AxCompileError::ResidentResourceRequiresExecutor(params.resource),
                        ),
                        progress => Ok(progress),
                    };
                }
            }
            if let Some(executor) = delegate.as_deref_mut()
                && let PreparedNativeInstruction::Kernel(kernel) = node
                && kernel.plan.effects() == KernelEffects::PURE
                && let [Some(NativeValue::F32(output))] = &mut outputs[..node.output_slots().len()]
            {
                let operands = OperandInputs {
                    values: &values,
                    node: kernel,
                };
                let mut inputs = [None; 3];
                for (input, slot) in inputs
                    .iter_mut()
                    .zip(&kernel.inputs[..kernel.input_count as usize])
                {
                    *input = Some(
                        operands
                            .view(*slot as usize)
                            .ok_or(AxArtifactError::InvalidPartition)?,
                    );
                }
                match executor.advance(index, &kernel.plan, &inputs, output, minimum_accuracy)? {
                    crate::KernelProgress::Unsupported if !pending => {
                        cpu_execution = CpuExecution::Fallback;
                    }
                    crate::KernelProgress::Unsupported => {
                        return Err(AxArtifactError::InvalidPartition.into());
                    }
                    // Checked: keep the delegate's result and run the
                    // native kernel over the same inputs below.
                    crate::KernelProgress::Complete if self.check.is_some() => {
                        let check = self.check.as_mut().expect("checked");
                        check.delegated.clear();
                        check.delegated.extend_from_slice(output);
                        checked = true;
                        cpu_execution = CpuExecution::Reference;
                    }
                    progress => return Ok(progress),
                }
            }
            if let Some(executor) = delegate.filter(|_| !checked)
                && let PreparedNativeInstruction::Kernel(kernel) = node
                && kernel.plan.semantic == SemanticOp::MoeExperts
                && let KernelParams::MoeExperts(params) = kernel.plan.params()
                && let [Some(NativeValue::F32(output))] = &mut outputs[..node.output_slots().len()]
            {
                let operands = OperandInputs {
                    values: &values,
                    node: kernel,
                };
                let hidden = operands.f32(kernel.inputs[0] as usize)?;
                let scores = operands.f32(kernel.inputs[1] as usize)?;
                let bias = operands.f32(kernel.inputs[2] as usize)?;
                if moe.is_none() {
                    if pending {
                        return Err(AxCompileError::InvocationFinished);
                    }
                    let invocation = axnn_cpu::MoeInvocation::new(
                        &kernel.plan,
                        expert_store_for_params(params, resources)?,
                        hidden,
                        scores,
                        bias,
                        output,
                        &self.kernel_scratch,
                    )?;
                    *moe = Some(match self.profile {
                        Some((clock, _)) => invocation.with_clock(clock),
                        None => invocation,
                    });
                }
                let invocation = moe.as_mut().ok_or(AxCompileError::InvocationFinished)?;
                let before = invocation.phases();
                let progress = invocation.advance(
                    hidden,
                    scores,
                    bias,
                    output,
                    &mut self.kernel_scratch,
                    parallel,
                    &mut PreparedMoeDelegate {
                        executor,
                        instruction: index,
                        minimum_accuracy,
                        check: self.check.as_mut(),
                        parallel,
                        profile: self.profile.as_mut(),
                    },
                );
                // Publish completed work even if a later wave is pending or
                // fails. Merging only at the end hides mid-operation fallback
                // from a live or failed invocation's profile.
                if let Some((_, profile)) = self.profile.as_mut() {
                    profile.moe.merge(&invocation.phases().since(&before));
                }
                match progress? {
                    axnn_cpu::MoeProgress::Pending => return Ok(crate::KernelProgress::Pending),
                    axnn_cpu::MoeProgress::Complete => {
                        *moe = None;
                        return Ok(crate::KernelProgress::Complete);
                    }
                    axnn_cpu::MoeProgress::Unsupported => {
                        return Err(AxCompileError::InvocationFinished);
                    }
                }
            }
            if let Some(executor) = projections.as_deref_mut()
                && let PreparedNativeInstruction::Kernel(kernel) = node
                && kernel.plan.semantic == SemanticOp::MatMulTransposed
                && kernel.plan.input(0).is_some_and(|t| t.dtype == DType::F32)
                && let [Some(NativeValue::F32(output))] = &mut outputs[..node.output_slots().len()]
            {
                let operands = OperandInputs {
                    values: &values,
                    node: kernel,
                };
                if executor.execute(
                    &kernel.plan,
                    operands.f32(kernel.inputs[0] as usize)?,
                    operands
                        .view(kernel.inputs[1] as usize)
                        .ok_or(AxArtifactError::InvalidPartition)?,
                    output,
                    parallel,
                )? {
                    return Ok(crate::KernelProgress::Complete);
                }
                cpu_execution = CpuExecution::Fallback;
            }
            if let Some(executor) = projections.as_deref_mut()
                && let PreparedNativeInstruction::Kernel(kernel) = node
                && kernel.plan.semantic == SemanticOp::MatMulNvfp4Transposed
                && let [Some(NativeValue::F32(output))] = &mut outputs[..node.output_slots().len()]
            {
                let operands = OperandInputs {
                    values: &values,
                    node: kernel,
                };
                if executor.execute_nvfp4(
                    &kernel.plan,
                    operands.f32(kernel.inputs[0] as usize)?,
                    operands.u8(kernel.inputs[1] as usize)?,
                    operands.packed(kernel.inputs[2] as usize, DType::B8)?,
                    output,
                    &mut self.kernel_scratch,
                    parallel,
                )? {
                    return Ok(crate::KernelProgress::Complete);
                }
                cpu_execution = CpuExecution::Fallback;
            }
            if let Some(executor) = projections.as_deref_mut()
                && let PreparedNativeInstruction::Kernel(kernel) = node
                && kernel.plan.semantic == SemanticOp::MoeExperts
                && let KernelParams::MoeExperts(params) = kernel.plan.params()
                && let [Some(NativeValue::F32(output))] = &mut outputs[..node.output_slots().len()]
            {
                let operands = OperandInputs {
                    values: &values,
                    node: kernel,
                };
                axnn_cpu::invoke_moe_experts_delegated(
                    &kernel.plan,
                    expert_store_for_params(params, resources)?,
                    operands.f32(kernel.inputs[0] as usize)?,
                    operands.f32(kernel.inputs[1] as usize)?,
                    operands.f32(kernel.inputs[2] as usize)?,
                    output,
                    &mut self.kernel_scratch,
                    parallel,
                    &mut ProjectionMoeDelegate { executor, parallel },
                )?;
                return Ok(crate::KernelProgress::Complete);
            }
            // Count at the execution boundary: retries, failed fusions and
            // device waits never enter this interval. Include synchronous
            // worker dispatch/join because it is part of executing on CPU.
            let started = self.profile.as_ref().map(|(clock, _)| clock());
            let result = invoke_native_instruction(
                node,
                &values,
                &mut outputs[..node.output_slots().len()],
                resources,
                &mut self.voxel_ids,
                &mut self.kernel_scratch,
                parallel,
            );
            if let (Some(started), Some((clock, profile))) = (started, self.profile.as_mut()) {
                let tally = match cpu_execution {
                    CpuExecution::Planned => &mut profile.cpu,
                    CpuExecution::Fallback => &mut profile.fallback,
                    CpuExecution::Reference => &mut profile.reference,
                };
                tally.add(op_class(node), clock().saturating_sub(started));
            }
            result.map(|()| crate::KernelProgress::Complete)
        })();
        for (source, slot) in outputs.iter_mut().zip(node.outputs()) {
            if let Some(value) = source.take() {
                self.values[slot as usize] = Some(value);
            }
        }
        let progress = result?;
        if progress == crate::KernelProgress::Pending {
            return Ok(progress);
        }
        if checked {
            self.record_check(index, index)?;
        }
        self.retire_borrowed_inputs(index)?;
        Ok(crate::KernelProgress::Complete)
    }

    /// Compare the delegate's result for `first..=last` (kept in the check)
    /// with the native one now in `last`'s output.
    fn record_check(&mut self, first: usize, last: usize) -> Result<(), AxCompileError> {
        let class = op_class(&self.nodes[first]);
        let slot = self.nodes[last].output_slots()[0] as usize;
        let Some(NativeValue::F32(native)) = self.values.get(slot).and_then(Option::as_ref) else {
            return Err(AxArtifactError::InvalidPartition.into());
        };
        let check = self
            .check
            .as_mut()
            .ok_or(AxArtifactError::InvalidPartition)?;
        let delegated = core::mem::take(&mut check.delegated);
        check.report.record(class, first, &delegated, native);
        check.delegated = delegated;
        Ok(())
    }

    /// After a delegate completed `first..=last` (one instruction, or a
    /// fused run): retire their borrowed inputs, or, when checking, keep
    /// the delegate's result, run them natively (which retires them) and
    /// compare.
    pub(crate) fn complete_delegated<'r>(
        &mut self,
        first: usize,
        last: usize,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'r>],
        parallel: &dyn ForkJoin,
        minimum_accuracy: KernelAccuracy,
    ) -> Result<(), AxCompileError> {
        // A routed-expert instruction retired here was routed by the device
        // (the host-routed path completes through the CPU continuation,
        // which counts the waves it delegates): one delegated wave of every
        // token's routes.
        for index in first..=last {
            let PreparedNativeInstruction::Kernel(kernel) = &self.nodes[index] else {
                continue;
            };
            let KernelParams::MoeExperts(params) = kernel.plan.params() else {
                continue;
            };
            let slot = kernel.output_slots[0] as usize;
            let rows = match self.values.get(slot).and_then(Option::as_ref) {
                Some(NativeValue::F32(rows)) => rows.len() / (params.hidden.max(1) as usize),
                _ => 0,
            };
            if let Some((_, profile)) = self.profile.as_mut() {
                profile.moe.delegated_waves += 1;
                profile.moe.delegated_routes += (rows * params.top_k as usize) as u64;
            }
        }
        if self.check.is_none() {
            for index in first..=last {
                self.retire_borrowed_inputs(index)?;
            }
            return Ok(());
        }
        let slot = self.nodes[last].output_slots()[0] as usize;
        let Some(NativeValue::F32(delegated)) = self.values.get(slot).and_then(Option::as_ref)
        else {
            return Err(AxArtifactError::InvalidPartition.into());
        };
        let check = self
            .check
            .as_mut()
            .ok_or(AxArtifactError::InvalidPartition)?;
        check.delegated.clear();
        check.delegated.extend_from_slice(delegated);
        for index in first..=last {
            self.run_declined(
                index,
                inputs,
                resources,
                parallel,
                minimum_accuracy,
                CpuExecution::Reference,
            )?;
        }
        self.record_check(first, last)
    }

    /// Whether instruction `index` may join an admission window: a pure
    /// kernel with exactly one F32 result, the same instructions
    /// `run_native_node` offers through `advance`, or a recurrent kernel,
    /// which the window offers with its state through `advance_recurrent`.
    pub(crate) fn admissible(&self, index: usize) -> bool {
        let Some(PreparedNativeInstruction::Kernel(kernel)) = self.nodes.get(index) else {
            return false;
        };
        (kernel.plan.effects() == KernelEffects::PURE || recurrent_binding(&kernel.plan).is_some())
            && kernel.output_count == 1
            && matches!(
                self.values.get(kernel.output_slots[0] as usize),
                Some(Some(NativeValue::F32(_)))
            )
    }

    /// Whether instruction `index` is a routed-expert kernel an admission
    /// window may offer whole, through
    /// [`advance_moe_routed`](crate::PreparedKernelExecutor::advance_moe_routed):
    /// one F32 result, and no reference check, which runs it natively.
    pub(crate) fn routable_moe(&self, index: usize) -> bool {
        let Some(PreparedNativeInstruction::Kernel(kernel)) = self.nodes.get(index) else {
            return false;
        };
        self.check.is_none()
            && kernel.plan.semantic == SemanticOp::MoeExperts
            && kernel.output_count == 1
            && matches!(
                self.values.get(kernel.output_slots[0] as usize),
                Some(Some(NativeValue::F32(_)))
            )
    }

    /// Whether instruction `index` is a KV append or attention read over a
    /// resident cache, which an admission window may offer whole through
    /// [`advance_kv_chained`](crate::PreparedKernelExecutor::advance_kv_chained).
    /// Reference checking refuses resident caches, so it never offers one.
    pub(crate) fn routable_kv(
        &self,
        index: usize,
        resources: &[NativeResourceBinding<'_>],
    ) -> bool {
        let Some(PreparedNativeInstruction::Kernel(kernel)) = self.nodes.get(index) else {
            return false;
        };
        let params = match kernel.plan.params() {
            KernelParams::KvAppend(params) => params,
            KernelParams::AttentionDecode(params) => params.cache,
            _ => return false,
        };
        self.check.is_none()
            && kernel.output_count == 1
            && matches!(
                kv_binding_for_params(params, resources),
                Ok(NativeResourceValue::ResidentKvCache(_))
            )
    }

    pub(crate) fn can_retire_unobserved(&self, index: usize, cursor: usize) -> bool {
        self.check.is_none()
            && self.output_last_use[index] < cursor
            && matches!(&self.nodes[index], PreparedNativeInstruction::Kernel(node)
                if node.output_count == 1 && node.plan.effects() == KernelEffects::PURE)
    }

    pub(crate) fn can_retire_moe_unobserved(&self, index: usize, cursor: usize) -> bool {
        self.check.is_none()
            && self.output_last_use[index] < cursor
            && matches!(&self.nodes[index], PreparedNativeInstruction::Kernel(node)
                if node.output_count == 1 && node.plan.semantic == SemanticOp::MoeExperts)
    }

    pub(crate) fn retire_moe_unobserved(
        &mut self,
        index: usize,
        moe: &mut Option<axnn_cpu::MoeInvocation<'_>>,
    ) -> Result<(), AxCompileError> {
        let invocation = moe.as_mut().ok_or(AxCompileError::InvocationFinished)?;
        let before = invocation.phases();
        invocation.retire_final_wave()?;
        if let Some((_, profile)) = self.profile.as_mut() {
            profile.moe.merge(&invocation.phases().since(&before));
        }
        *moe = None;
        self.retire_borrowed_inputs(index)
    }

    /// Which of instruction `index`'s operands the admitted instructions
    /// (`admitted`, oldest first) produce. Each operand names the most recent
    /// admitted writer of its slot, and the part of that output it reads.
    pub(crate) fn chain_sources(
        &self,
        index: usize,
        admitted: impl DoubleEndedIterator<Item = usize> + Clone,
    ) -> [Option<crate::continuation::ChainSource>; 3] {
        let mut sources = [None; 3];
        let Some(PreparedNativeInstruction::Kernel(kernel)) = self.nodes.get(index) else {
            return sources;
        };
        for (source, &slot) in sources
            .iter_mut()
            .zip(&kernel.inputs[..kernel.input_count as usize])
        {
            *source = self.chain_source(index, slot, admitted.clone());
        }
        sources
    }

    /// Whether an admitted instruction produces what `consumer` reads from
    /// `slot`: the most recent admitted writer of the slot, and the part of
    /// its output the consumer's operand covers.
    pub(crate) fn chain_source(
        &self,
        consumer: usize,
        slot: u32,
        admitted: impl DoubleEndedIterator<Item = usize>,
    ) -> Option<crate::continuation::ChainSource> {
        let producer = admitted.rev().find(|&producer| {
            matches!(
                self.nodes.get(producer),
                Some(PreparedNativeInstruction::Kernel(node))
                    if node.output_count == 1 && node.output_slots[0] == slot
            )
        })?;
        let whole = match self.values.get(slot as usize) {
            Some(Some(NativeValue::F32(values))) => values.len(),
            // Only a resident KV append's token: no other I32 producer joins
            // a window.
            Some(Some(NativeValue::I32(values))) => values.len(),
            _ => return None,
        };
        let range = match self.nodes.get(consumer) {
            Some(PreparedNativeInstruction::Kernel(node)) => node.range_for(slot),
            _ => None,
        };
        let (offset, elements) = range.map_or((0, whole), |range| (range.offset, range.elements));
        Some(crate::continuation::ChainSource {
            producer,
            offset,
            elements,
        })
    }

    /// Whether the value `producer` writes to `slot` is read by `consumer`
    /// and by nothing else before the slot is written again, and is not a
    /// graph output: a fused run may then leave it unpublished.
    fn sole_consumer(&self, producer: usize, slot: u32, consumer: usize) -> bool {
        for (index, node) in self.nodes.iter().enumerate().skip(producer + 1) {
            if index != consumer && node.inputs().contains(&slot) {
                return false;
            }
            if node.output_slots().contains(&slot) {
                return true;
            }
        }
        !self
            .outputs
            .iter()
            .any(|binding| binding.slot == slot as usize)
    }

    /// The fused run starting at `first`, if the prepared instructions form
    /// one: its kind, its last instruction, and (for an accumulate run) the
    /// base operand's slot. Every instruction of the run is admissible, the
    /// first is a projection without an epilogue, and every intermediate is
    /// consumed only inside the run, whole (no view).
    pub(crate) fn fused_run(&self, first: usize) -> Option<(crate::FusedRun, usize, Option<u32>)> {
        let kernel = |index: usize| match self.nodes.get(index) {
            Some(PreparedNativeInstruction::Kernel(node)) if self.admissible(index) => Some(node),
            _ => None,
        };
        let projection = kernel(first)?;
        let next = kernel(first + 1)?;
        if matches!(
            projection.plan.semantic,
            SemanticOp::Binary(axnn::BinaryOp::Add | axnn::BinaryOp::Mul)
        ) && next.plan.semantic == SemanticOp::RmsNorm
            && let KernelParams::Normalization { epsilon, axis } = next.plan.params()
        {
            let value = projection.output_slots[0];
            let normalized = next.plan.input(0)?;
            let elements = projection.plan.shape.checked_num_elements()?;
            // No binary broadcasting: both operands and the full (possibly
            // reshaped) intermediate cover the normalization's whole tensor.
            let same_shape = projection.plan.input_shapes[..2]
                .iter()
                .all(|shape| shape.as_ref() == Some(&projection.plan.shape));
            let whole = next
                .range_for(value)
                .is_none_or(|range| range.offset == 0 && range.elements == elements);
            if same_shape
                && next.inputs[0] == value
                && whole
                && normalized.shape.checked_num_elements()? == elements
                && self.sole_consumer(first, value, first + 1)
                && !projection.inputs[..2].contains(&next.output_slots[0])
            {
                let width = normalized.shape.dims()[axis as usize..]
                    .iter()
                    .try_fold(1usize, |size, dim| size.checked_mul(*dim))?;
                return Some((
                    crate::FusedRun::RmsNorm { width, epsilon },
                    first + 1,
                    Some(next.inputs[1]),
                ));
            }
            return None;
        }
        let plain = match (projection.plan.semantic, projection.plan.params()) {
            (
                SemanticOp::MatMulNvfp4Transposed | SemanticOp::MatMulTransposed,
                KernelParams::MatmulEpilogue {
                    activation: axnn::lowering::EpilogueActivation::None,
                    ..
                },
            ) => true,
            (SemanticOp::MatMulTransposed, KernelParams::None) => true,
            _ => false,
        };
        if !plain {
            return None;
        }
        let value = projection.output_slots[0];
        let whole = |node: &NativeCpuNode, slot: u32| node.range_for(slot).is_none();
        let next = kernel(first + 1)?;
        let found = match next.plan.semantic {
            SemanticOp::Unary(axnn::UnaryOp::Relu)
                if next.inputs[..next.input_count as usize] == [value]
                    && whole(next, value)
                    && self.sole_consumer(first, value, first + 1) =>
            {
                let rectified = next.output_slots[0];
                let square = kernel(first + 2)?;
                (square.plan.semantic == SemanticOp::Binary(axnn::BinaryOp::Mul)
                    && square.inputs[..square.input_count as usize] == [rectified, rectified]
                    && whole(square, rectified)
                    && self.sole_consumer(first + 1, rectified, first + 2))
                .then_some((crate::FusedRun::SquaredRelu, first + 2, None))
            }
            // Element-for-element only: `Binary` broadcasts, and a biased
            // linear (`proj [t,n] + bias [n]`) is not an accumulate.
            SemanticOp::Binary(axnn::BinaryOp::Add)
                if next.input_count == 2
                    && self.sole_consumer(first, value, first + 1)
                    && whole(next, value)
                    && next.plan.input_shapes[0].as_ref() == Some(&next.plan.shape)
                    && next.plan.input_shapes[1].as_ref() == Some(&next.plan.shape) =>
            {
                let base = match next.inputs[..2] {
                    [a, b] if a == value && b != value => b,
                    [a, b] if b == value && a != value => a,
                    _ => return None,
                };
                Some((crate::FusedRun::Accumulate, first + 1, Some(base)))
            }
            _ => None,
        };
        let (run, last, base) = found?;
        // `offer_fused` takes the run's output slot before viewing the
        // projection's operands, and the slot allocator may give the tail a
        // projection input that dies at the projection (a square projection
        // with a residual add). Such a run is offered one instruction at a
        // time instead.
        let output = kernel(last)?.output_slots[0];
        (!projection.inputs[..projection.input_count as usize].contains(&output))
            .then_some((run, last, base))
    }

    /// Offer the fused run `first..=last` through `advance_fused`: the
    /// projection's operands and an accumulate run's base as pending
    /// references or host views, the last instruction's output slot as the
    /// output. On Complete the caller retires every instruction of the run
    /// ([`Self::complete_delegated`]); the intermediates are never
    /// published.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn offer_fused(
        &mut self,
        first: usize,
        last: usize,
        run: crate::FusedRun,
        inputs: &[NativeRef<'_>],
        executor: &mut dyn crate::PreparedKernelExecutor,
        sources: &[Option<crate::continuation::ChainSource>; 3],
        base: Option<crate::continuation::ChainSource>,
        minimum_accuracy: KernelAccuracy,
    ) -> Result<crate::KernelProgress, AxCompileError> {
        let (
            Some(PreparedNativeInstruction::Kernel(projection)),
            Some(PreparedNativeInstruction::Kernel(tail)),
        ) = (self.nodes.get(first), self.nodes.get(last))
        else {
            return Err(AxArtifactError::InvalidPartition.into());
        };
        let base_slot = match run {
            crate::FusedRun::Accumulate => {
                let value = projection.output_slots[0];
                Some(if tail.inputs[0] == value {
                    tail.inputs[1]
                } else {
                    tail.inputs[0]
                })
            }
            crate::FusedRun::SquaredRelu => None,
            crate::FusedRun::RmsNorm { .. } => Some(tail.inputs[1]),
        };
        let slot = tail.output_slots[0] as usize;
        let mut output = self
            .values
            .get_mut(slot)
            .ok_or(AxArtifactError::InvalidPartition)?
            .take()
            .ok_or(AxArtifactError::InvalidPartition)?;
        let result = (|| {
            let NativeValue::F32(values_out) = &mut output else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            let values = PreparedNativeInputs {
                values: self.values.as_slice(),
                input_by_slot: &self.borrowed_input_by_slot,
                inputs,
                remaining: &self.borrowed_input_remaining,
            };
            let operands = OperandInputs {
                values: &values,
                node: projection,
            };
            let mut chained = [None; 3];
            for ((input, &slot), source) in chained
                .iter_mut()
                .zip(&projection.inputs[..projection.input_count as usize])
                .zip(sources)
            {
                *input = Some(match source {
                    Some(source) => crate::ChainedInput::Pending {
                        instruction: source.producer,
                        offset: source.offset,
                        elements: source.elements,
                    },
                    None => crate::ChainedInput::Host(
                        operands
                            .view(slot as usize)
                            .ok_or(AxArtifactError::InvalidPartition)?,
                    ),
                });
            }
            let tail_operands = OperandInputs {
                values: &values,
                node: tail,
            };
            let base = match (base_slot, base) {
                (None, _) => None,
                (Some(_), Some(source)) => Some(crate::ChainedInput::Pending {
                    instruction: source.producer,
                    offset: source.offset,
                    elements: source.elements,
                }),
                (Some(slot), None) => Some(crate::ChainedInput::Host(
                    tail_operands
                        .view(slot as usize)
                        .ok_or(AxArtifactError::InvalidPartition)?,
                )),
            };
            executor.advance_fused(
                last,
                run,
                &projection.plan,
                &chained,
                base,
                values_out,
                minimum_accuracy,
            )
        })();
        self.values[slot] = Some(output);
        // On Complete the caller retires the run (`complete_delegated`).
        result
    }

    /// Offer admissible instruction `index` through `advance_chained`: its
    /// operands produced by pending admissions (`sources`) as references,
    /// the rest as host views. On Complete the caller retires the
    /// instruction's borrowed ingress uses ([`Self::complete_delegated`]),
    /// exactly as after a sequential run; on Pending the output slot is
    /// restored untouched and nothing retires.
    pub(crate) fn offer_chained(
        &mut self,
        index: usize,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'_>],
        executor: &mut dyn crate::PreparedKernelExecutor,
        sources: &[Option<crate::continuation::ChainSource>; 3],
        minimum_accuracy: KernelAccuracy,
    ) -> Result<crate::KernelProgress, AxCompileError> {
        let PreparedNativeInstruction::Kernel(kernel) = &self.nodes[index] else {
            return Err(AxArtifactError::InvalidPartition.into());
        };
        let slot = kernel.output_slots[0] as usize;
        let mut output = self
            .values
            .get_mut(slot)
            .ok_or(AxArtifactError::InvalidPartition)?
            .take()
            .ok_or(AxArtifactError::InvalidPartition)?;
        let result = (|| {
            let values = PreparedNativeInputs {
                values: self.values.as_slice(),
                input_by_slot: &self.borrowed_input_by_slot,
                inputs,
                remaining: &self.borrowed_input_remaining,
            };
            let operands = OperandInputs {
                values: &values,
                node: kernel,
            };
            let mut chained = [None; 3];
            for ((input, &slot), source) in chained
                .iter_mut()
                .zip(&kernel.inputs[..kernel.input_count as usize])
                .zip(sources)
            {
                *input = Some(match source {
                    Some(source) => crate::ChainedInput::Pending {
                        instruction: source.producer,
                        offset: source.offset,
                        elements: source.elements,
                    },
                    None => crate::ChainedInput::Host(
                        operands
                            .view(slot as usize)
                            .ok_or(AxArtifactError::InvalidPartition)?,
                    ),
                });
            }
            // A resident KV operation: the append's I32 token or the read's
            // F32 rows, against the executor's own cache.
            let kv = match kernel.plan.params() {
                KernelParams::KvAppend(params) => Some(params),
                KernelParams::AttentionDecode(params) => Some(params.cache),
                _ => None,
            };
            if let Some(params) = kv {
                let NativeResourceValue::ResidentKvCache(cache) =
                    kv_binding_for_params(params, resources)?
                else {
                    return Err(AxArtifactError::InvalidPartition.into());
                };
                let output = match (kernel.plan.params(), &mut output) {
                    (KernelParams::KvAppend(_), NativeValue::I32(token)) => {
                        crate::KvOutput::Appended(token)
                    }
                    (KernelParams::AttentionDecode(_), NativeValue::F32(rows)) => {
                        crate::KvOutput::Attention(rows)
                    }
                    _ => return Err(AxArtifactError::InvalidPartition.into()),
                };
                return executor.advance_kv_chained(
                    index,
                    &kernel.plan,
                    cache,
                    &chained,
                    output,
                    minimum_accuracy,
                );
            }
            let NativeValue::F32(values_out) = &mut output else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            if kernel.plan.semantic == SemanticOp::MoeExperts {
                let KernelParams::MoeExperts(params) = kernel.plan.params() else {
                    return Err(AxArtifactError::InvalidPartition.into());
                };
                return executor.advance_moe_routed(
                    index,
                    &kernel.plan,
                    &chained,
                    expert_store_for_params(params, resources)?,
                    values_out,
                    minimum_accuracy,
                );
            }
            match recurrent_binding(&kernel.plan) {
                Some((state, elements)) => {
                    let (source, target) = recurrent_state(state, elements, resources)?;
                    // Keep the exclusive resource as address/length metadata:
                    // a retry can occur while an independently pinned target
                    // is being written by the device. A CPU slice here would
                    // assert exclusive CPU access before completion.
                    executor.advance_recurrent(
                        index,
                        &kernel.plan,
                        &chained,
                        crate::RecurrentState {
                            source: source.values(),
                            tree: source.tree(),
                            target,
                        },
                        values_out,
                        minimum_accuracy,
                    )
                }
                None => executor.advance_chained(
                    index,
                    &kernel.plan,
                    &chained,
                    values_out,
                    minimum_accuracy,
                ),
            }
        })();
        self.values[slot] = Some(output);
        // On Complete the caller retires the instruction (`complete_delegated`).
        result
    }

    /// Run instruction `index` with its native kernel after the executor
    /// declined it with nothing admitted: no second offer, same retirement.
    pub(crate) fn run_declined<'r>(
        &mut self,
        index: usize,
        inputs: &[NativeRef<'_>],
        resources: &[NativeResourceBinding<'r>],
        parallel: &dyn ForkJoin,
        minimum_accuracy: KernelAccuracy,
        cpu_execution: CpuExecution,
    ) -> Result<(), AxCompileError> {
        let mut moe = None;
        match self.run_native_node(
            index,
            inputs,
            resources,
            parallel,
            &mut None,
            None,
            false,
            minimum_accuracy,
            &mut moe,
            cpu_execution,
        )? {
            crate::KernelProgress::Complete => Ok(()),
            _ => Err(AxArtifactError::InvalidPartition.into()),
        }
    }

    /// Count one completed use of each borrowed ingress operand of `index`.
    fn retire_borrowed_inputs(&mut self, index: usize) -> Result<(), AxCompileError> {
        for &slot in self.nodes[index].inputs() {
            if let Some(&input) = self.borrowed_input_by_slot.get(slot as usize)
                && input != usize::MAX
                && self.borrowed_input_remaining[input] != 0
            {
                self.borrowed_input_remaining[input] = self.borrowed_input_remaining[input]
                    .checked_sub(1)
                    .ok_or(AxArtifactError::InvalidPartition)?;
            }
        }
        Ok(())
    }

    /// Borrow an F32 output in graph output order after a successful run.
    /// Returns `None` before a run, after a failed run, for another dtype, or
    /// for an out-of-range index.
    pub fn output_f32(&self, index: usize) -> Option<&[f32]> {
        if !self.complete {
            return None;
        }
        let binding = self.outputs.get(index)?;
        match self.values.get(binding.slot)?.as_ref()? {
            NativeValue::F32(values) => Some(values),
            NativeValue::I32(_) | NativeValue::U8(_) | NativeValue::Packed(_, _) => None,
        }
    }

    /// Compact selected rows of a completed F32 output into its existing
    /// allocation. Increasing source indices make forward copies safe; every
    /// bound is checked before mutation. The allocation and output shape stay
    /// unchanged, so retained capture views keep their original geometry.
    pub fn compact_output_f32_rows(
        &mut self,
        index: usize,
        rows: usize,
        path: &[u32],
    ) -> Option<()> {
        if !self.complete
            || rows == 0
            || path.is_empty()
            || path.len() > rows
            || path.iter().any(|&row| row as usize >= rows)
            || !path.windows(2).all(|pair| pair[0] < pair[1])
        {
            return None;
        }
        let binding = self.outputs.get(index)?;
        let NativeValue::F32(values) = self.values.get_mut(binding.slot)?.as_mut()? else {
            return None;
        };
        if !values.len().is_multiple_of(rows) {
            return None;
        }
        let width = values.len() / rows;
        for (destination, &source) in path.iter().enumerate() {
            let start = source as usize * width;
            values.copy_within(start..start + width, destination * width);
        }
        Some(())
    }

    /// Borrow an exact I32 output using the same ordering as `output_f32`.
    pub fn output_i32(&self, index: usize) -> Option<&[i32]> {
        if !self.complete {
            return None;
        }
        let binding = self.outputs.get(index)?;
        match self.values.get(binding.slot)?.as_ref()? {
            NativeValue::I32(values) => Some(values),
            NativeValue::F32(_) | NativeValue::U8(_) | NativeValue::Packed(_, _) => None,
        }
    }

    /// Borrow a U8 output using the same lifetime and ordering as `output_f32`.
    pub fn output_u8(&self, index: usize) -> Option<&[u8]> {
        if !self.complete {
            return None;
        }
        let binding = self.outputs.get(index)?;
        match self.values.get(binding.slot)?.as_ref()? {
            NativeValue::U8(values) => Some(values),
            NativeValue::F32(_) | NativeValue::I32(_) | NativeValue::Packed(_, _) => None,
        }
    }

    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Number of executed instructions after strict prepared combinations.
    pub fn instruction_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of single-use MUL→ADD pairs sharing one prepared instruction.
    pub fn mul_add_fusion_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node, PreparedNativeInstruction::MulAdd(_)))
            .count()
    }

    /// Fused pairs routed to portable-SIMD helpers by the original CPU policy.
    /// Other fused pairs use scalar separate multiply and add operations.
    pub fn simd_mul_add_fusion_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| {
                matches!(node, PreparedNativeInstruction::MulAdd(node)
                    if !matches!(node.kernel, PreparedMulAddKernel::Scalar))
            })
            .count()
    }

    /// Groups of three dense F32 MULs by one scalar, combined with INTERLEAVE3.
    /// Each group removes three product instructions and their intermediates.
    pub fn scale_interleave3_fusion_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node, PreparedNativeInstruction::ScaleInterleave3(_)))
            .count()
    }

    /// Scale/INTERLEAVE3 groups admitted to the portable-SIMD helper.
    pub fn simd_scale_interleave3_fusion_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node, PreparedNativeInstruction::ScaleInterleave3(node) if node.simd))
            .count()
    }

    /// Three dense F32→I32 casts, optional scalar shifts, and two ORs sharing
    /// one storage sink. Floating-point operations before the casts stay intact.
    pub fn cast_shift_or3_fusion_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node, PreparedNativeInstruction::CastShiftOr3(_)))
            .count()
    }

    /// Number of physical tensor buffers retained after liveness reuse.
    pub fn buffer_count(&self) -> usize {
        self.values.len()
    }

    /// Retained tensor payload bytes, excluding plan and allocation metadata.
    pub fn buffer_bytes(&self) -> usize {
        self.buffer_bytes
    }

    /// Number of logical tensor buffers before liveness reuse, including all
    /// graph inputs and outputs even when they are not used by a native node.
    pub fn logical_buffer_count(&self) -> usize {
        self.logical_buffers
    }

    /// Tensor payload bytes required by one distinct buffer per logical tensor.
    pub fn logical_buffer_bytes(&self) -> usize {
        self.logical_bytes
    }

    /// Reusable kernel scratch: U32-to-I32 voxel results and cache
    /// attention rows. Excludes borrowed resource bytes.
    pub fn scratch_bytes(&self) -> usize {
        self.voxel_ids.len() * size_of::<u32>() + self.kernel_scratch.len() * size_of::<f32>()
    }
}

struct PreparedNativeInputs<'a, A: Allocator> {
    values: &'a [Option<NativeValue<A>>],
    input_by_slot: &'a [usize],
    inputs: &'a [NativeRef<'a>],
    remaining: &'a [u32],
}

impl<A: Allocator> NativeInputs for PreparedNativeInputs<'_, A> {
    fn view(&self, slot: usize) -> Option<NativeRef<'_>> {
        if let Some(&index) = self.input_by_slot.get(slot)
            && index != usize::MAX
            && self.remaining.get(index).copied().unwrap_or(0) != 0
        {
            return self.inputs.get(index).copied();
        }
        self.values.view(slot)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxHostCopyStats {
    pub native_ingress_bytes: u64,
    pub native_boundary_encode_bytes: u64,
    pub output_materialization_bytes: u64,
}

/// Why the engine is entering a CPU instruction. This is selected only after
/// device admission has settled, not by the adapter's tentative offers.
#[derive(Clone, Copy)]
pub(crate) enum CpuExecution {
    Planned,
    Fallback,
    Reference,
}

/// Synchronous CPU execution wall time by operation class. Includes worker
/// dispatch and joining, excludes GPU admission/waiting. This is not CPU busy
/// time and does not inspect work inside opaque runtime/legacy callbacks.
/// Attempts that return errors count too. Requires a profiler clock; no clock
/// reads when profiling is disabled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuProfile {
    pub ns: [u64; OP_CLASSES],
    pub runs: [u64; OP_CLASSES],
}

impl CpuProfile {
    fn add(&mut self, class: usize, ns: u64) {
        self.ns[class] += ns;
        self.runs[class] += 1;
    }

    fn merge(&mut self, other: &Self) {
        for class in 0..OP_CLASSES {
            self.ns[class] += other.ns[class];
            self.runs[class] += other.runs[class];
        }
    }

    pub fn since(&self, before: &Self) -> Self {
        Self {
            ns: core::array::from_fn(|i| self.ns[i].saturating_sub(before.ns[i])),
            runs: core::array::from_fn(|i| self.runs[i].saturating_sub(before.runs[i])),
        }
    }
}

impl core::fmt::Display for CpuProfile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        for (class, &runs) in self.runs.iter().enumerate() {
            if runs == 0 {
                continue;
            }
            if !first {
                f.write_str(", ")?;
            }
            first = false;
            write!(
                f,
                "{} {} us/{}",
                OP_CLASS_NAMES[class],
                self.ns[class] / 1_000,
                runs
            )?;
        }
        if first {
            f.write_str("none")?;
        }
        Ok(())
    }
}

/// Prepared instruction lifetimes by kernel class. A delegated instruction's
/// interval includes waiting and can overlap other admissions: `ns` is not CPU
/// execution time and its sum is not elapsed turn time. The CPU tallies below
/// record actual synchronous execution at its final decision point instead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpProfile {
    pub ns: [u64; OP_CLASSES],
    pub runs: [u64; OP_CLASSES],
    /// The `moe` class's host time by phase ([`axnn_cpu::MoePhases`]), for
    /// completed work, including in-progress invocations. What the phases do
    /// not cover is waiting on a delegate's device between calls.
    pub moe: axnn_cpu::MoePhases,
    /// Instructions executed directly, without a final declined device offer.
    pub cpu: CpuProfile,
    /// Instructions executed after a final unsupported offer (not retries).
    /// Expert-wave fallback is reported separately by `moe.native_*`.
    pub fallback: CpuProfile,
    /// Explicit CPU recomputation requested by `set_delegate_check`, including
    /// expert waves. These are not fallback and are absent in ordinary runs.
    pub reference: CpuProfile,
}

/// How far delegated results were from the native kernels', by op class
/// ([`PreparedNative::set_delegate_check`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CheckReport {
    /// Results compared: instructions, fused runs and MoE waves.
    pub checked: [u64; OP_CLASSES],
    /// The largest max|delegate − native| / max|native| of one result.
    pub worst: [f32; OP_CLASSES],
    /// The instruction (a fused run's first) whose result it was.
    pub worst_at: [usize; OP_CLASSES],
    /// Delegate values NaN or infinite where the native one is finite.
    pub nonfinite: [u64; OP_CLASSES],
    /// The first result off by more than [`CHECK_THRESHOLD`], as
    /// (instruction, class): where a wrong kernel shows first. Later
    /// results of a device chain may inherit its error from their operands.
    pub first: Option<(usize, usize)>,
    /// The first result over the threshold of each (class, length), up to
    /// [`CHECK_OFFENDERS`]: one per kernel shape, so a report names every
    /// wrong kernel, not only the first.
    pub offenders: [Option<CheckOffender>; CHECK_OFFENDERS],
}

/// Distinct kernel shapes a [`CheckReport`] names at most.
pub const CHECK_OFFENDERS: usize = 12;

/// One result over [`CHECK_THRESHOLD`] ([`CheckReport::offenders`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CheckOffender {
    pub instruction: usize,
    pub class: usize,
    /// Values in the result.
    pub len: usize,
    /// Delegate values NaN or infinite where the native one is finite.
    pub nonfinite: u64,
    pub error: f32,
    /// The first value off: the first non-finite one, else the largest
    /// difference. Its index, the delegate's value and the native one.
    pub at: usize,
    pub delegated: f32,
    pub native: f32,
}

impl core::fmt::Display for CheckOffender {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "#{} {} of {}: error {:.2e}, {} non-finite; [{}] {:e} vs {:e}",
            self.instruction,
            OP_CLASS_NAMES[self.class],
            self.len,
            self.error,
            self.nonfinite,
            self.at,
            self.delegated,
            self.native
        )
    }
}

/// The relative error past which [`CheckReport::first`] names a result.
pub const CHECK_THRESHOLD: f32 = 1e-2;

impl CheckReport {
    fn record(&mut self, class: usize, instruction: usize, delegated: &[f32], native: &[f32]) {
        let (mut diff, mut scale, mut nonfinite) = (0.0f32, 0.0f32, 0u64);
        let (mut widest, mut first_nonfinite) = (0usize, None);
        for (index, (&d, &n)) in delegated.iter().zip(native).enumerate() {
            if !n.is_finite() {
                continue;
            }
            scale = scale.max(n.abs());
            if d.is_finite() {
                if (d - n).abs() > diff {
                    diff = (d - n).abs();
                    widest = index;
                }
            } else {
                nonfinite += 1;
                first_nonfinite.get_or_insert(index);
            }
        }
        let error = if delegated.len() != native.len() || nonfinite > 0 {
            f32::INFINITY
        } else if scale > 0.0 {
            diff / scale
        } else if diff > 0.0 {
            f32::INFINITY
        } else {
            0.0
        };
        self.checked[class] += 1;
        self.nonfinite[class] += nonfinite;
        if error > CHECK_THRESHOLD && self.first.is_none() {
            self.first = Some((instruction, class));
        }
        if error > CHECK_THRESHOLD {
            let at = first_nonfinite.unwrap_or(widest);
            self.offend(CheckOffender {
                instruction,
                class,
                len: delegated.len(),
                nonfinite,
                error,
                at,
                delegated: delegated.get(at).copied().unwrap_or(f32::NAN),
                native: native.get(at).copied().unwrap_or(f32::NAN),
            });
        }
        if error > self.worst[class] || self.checked[class] == 1 {
            self.worst[class] = error;
            self.worst_at[class] = instruction;
        }
    }

    /// Keep `offender` unless its (class, length) is already named or the
    /// list is full.
    fn offend(&mut self, offender: CheckOffender) {
        if self
            .offenders
            .iter()
            .flatten()
            .any(|kept| (kept.class, kept.len) == (offender.class, offender.len))
        {
            return;
        }
        if let Some(slot) = self.offenders.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(offender);
        }
    }

    /// The offenders, one per kernel shape, in the order they were found.
    pub fn offenders(&self) -> impl Iterator<Item = &CheckOffender> {
        self.offenders.iter().flatten()
    }

    pub fn merge(&mut self, other: &Self) {
        self.first = self.first.or(other.first);
        for offender in other.offenders.iter().flatten() {
            self.offend(*offender);
        }
        for class in 0..OP_CLASSES {
            if other.checked[class] == 0 {
                continue;
            }
            if self.checked[class] == 0 || other.worst[class] > self.worst[class] {
                self.worst[class] = other.worst[class];
                self.worst_at[class] = other.worst_at[class];
            }
            self.checked[class] += other.checked[class];
            self.nonfinite[class] += other.nonfinite[class];
        }
    }
}

impl core::fmt::Display for CheckReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        for class in 0..OP_CLASSES {
            if self.checked[class] == 0 {
                continue;
            }
            if !first {
                f.write_str(", ")?;
            }
            first = false;
            write!(
                f,
                "{} {} (worst {:.2e} at #{}",
                OP_CLASS_NAMES[class], self.checked[class], self.worst[class], self.worst_at[class]
            )?;
            if self.nonfinite[class] > 0 {
                write!(f, ", {} non-finite", self.nonfinite[class])?;
            }
            f.write_str(")")?;
        }
        if first {
            return f.write_str("nothing delegated");
        }
        match self.first {
            Some((instruction, class)) => write!(
                f,
                "; first over {CHECK_THRESHOLD:.0e}: #{instruction} {}",
                OP_CLASS_NAMES[class]
            ),
            None => write!(f, "; none over {CHECK_THRESHOLD:.0e}"),
        }
    }
}

/// A delegate check's working state: the report, the delegate's copy of
/// the result being compared, and a MoE wave's output as it found it.
#[derive(Debug, Default)]
pub struct DelegateCheck {
    pub report: CheckReport,
    delegated: Vec<f32>,
    before: Vec<f32>,
    wave: Option<(usize, usize)>,
    contribution: Vec<f32>,
    scratch: Vec<f32>,
}

pub const OP_CLASSES: usize = 10;
pub const OP_CLASS_NAMES: [&str; OP_CLASSES] = [
    "moe", "nvfp4", "matmul", "scan", "conv", "attn", "norm", "gather", "fused", "other",
];

impl OpProfile {
    pub fn add(&mut self, class: usize, ns: u64) {
        self.ns[class] += ns;
        self.runs[class] += 1;
    }

    pub fn merge(&mut self, other: &Self) {
        for class in 0..OP_CLASSES {
            self.ns[class] += other.ns[class];
            self.runs[class] += other.runs[class];
        }
        self.moe.merge(&other.moe);
        self.cpu.merge(&other.cpu);
        self.fallback.merge(&other.fallback);
        self.reference.merge(&other.reference);
    }

    pub fn since(&self, before: &Self) -> Self {
        let mut out = Self::default();
        for class in 0..OP_CLASSES {
            out.ns[class] = self.ns[class].saturating_sub(before.ns[class]);
            out.runs[class] = self.runs[class].saturating_sub(before.runs[class]);
        }
        out.moe = self.moe.since(&before.moe);
        out.cpu = self.cpu.since(&before.cpu);
        out.fallback = self.fallback.since(&before.fallback);
        out.reference = self.reference.since(&before.reference);
        out
    }

    /// Sum of instruction lifetimes, which may overlap. Not elapsed turn time
    /// or CPU execution time.
    pub fn total_ns(&self) -> u64 {
        self.ns.iter().sum()
    }
}

impl core::fmt::Display for OpProfile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut first = true;
        for class in 0..OP_CLASSES {
            if self.runs[class] == 0 {
                continue;
            }
            if !first {
                f.write_str(", ")?;
            }
            first = false;
            write!(
                f,
                "{} {} ms/{}",
                OP_CLASS_NAMES[class],
                self.ns[class] / 1_000_000,
                self.runs[class]
            )?;
            // MoE: where its host time went; the rest is device waits.
            if class == 0 && self.moe.ns.iter().any(|&ns| ns > 0) {
                f.write_str(" (")?;
                let mut covered = 0;
                for (name, ns) in axnn_cpu::MOE_PHASE_NAMES.iter().zip(self.moe.ns) {
                    write!(f, "{name} {} ", ns / 1_000_000)?;
                    covered += ns;
                }
                write!(
                    f,
                    "wait {} ms)",
                    self.ns[0].saturating_sub(covered) / 1_000_000
                )?;
            }
        }
        if first {
            f.write_str("none")?;
        }
        write!(
            f,
            "; CPU direct [{}]; CPU fallback [{}]; CPU reference [{}]",
            self.cpu, self.fallback, self.reference
        )?;
        if self.moe.delegated_waves != 0 || self.moe.native_waves != 0 {
            write!(
                f,
                "; expert placement: delegated {}/{} waves/routes, CPU {}/{} waves/routes, mixed {}",
                self.moe.delegated_waves,
                self.moe.delegated_routes,
                self.moe.native_waves,
                self.moe.native_routes,
                self.moe.mixed_invocations
            )?;
        }
        Ok(())
    }
}

fn op_class(node: &PreparedNativeInstruction) -> usize {
    let PreparedNativeInstruction::Kernel(kernel) = node else {
        return 8;
    };
    match kernel.plan.semantic {
        SemanticOp::MoeExperts | SemanticOp::MoeGateUpNvfp4 | SemanticOp::MoeDownNvfp4 => 0,
        SemanticOp::MatMulNvfp4Transposed => 1,
        SemanticOp::MatMul | SemanticOp::MatMulTransposed | SemanticOp::VecDot => 2,
        SemanticOp::SsmScan | SemanticOp::GatedDeltaRule => 3,
        SemanticOp::SsmConv | SemanticOp::CausalConv1d | SemanticOp::Conv1d => 4,
        SemanticOp::AttentionDecode | SemanticOp::KvAppend => 5,
        SemanticOp::RmsNorm
        | SemanticOp::LayerNorm
        | SemanticOp::Softmax
        | SemanticOp::Reduce(_)
        | SemanticOp::ArgReduce(_)
        | SemanticOp::TopK
        | SemanticOp::MoeRoute => 6,
        SemanticOp::Gather => 7,
        _ => 9,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxNativePlanStats {
    pub logical_nodes: usize,
    pub instructions: usize,
    pub mul_add_fusions: usize,
    pub sub_mul_add_fusions: usize,
    pub scale_interleave3_fusions: usize,
    pub cast_shift_or3_fusions: usize,
}

pub fn strict_canonical_f32_binary(node: &NativeCpuNode, op: BinaryOp) -> bool {
    let plan = &node.plan;
    if plan.semantic != SemanticOp::Binary(op)
        || plan.accuracy != KernelAccuracy::Strict
        || plan.effects() != KernelEffects::PURE
        || plan.params() != KernelParams::None
        || plan.temporary_bytes != 0
        || node.input_count != 2
        || node.output_count != 1
        || plan.input_dtypes != [Some(DType::F32), Some(DType::F32), None]
        || plan.output_dtype != DType::F32
    {
        return false;
    }
    let (Some(left), Some(right), Some(output)) =
        (plan.input(0), plan.input(1), plan.result_tensor(0))
    else {
        return false;
    };
    if plan.input_shapes != [Some(left.shape), Some(right.shape), None]
        || plan.shape != output.shape
        || [left, right, output]
            .into_iter()
            .any(|desc| TensorDesc::try_new(desc.shape, DType::F32).as_ref() != Some(desc))
    {
        return false;
    }
    // Reconstruct the typed binary request as well as checking canonical
    // layouts. Equal element counts alone do not prove broadcast semantics.
    KernelRequest::binary(op, left, right).is_some_and(|request| request.result() == plan.result())
}

fn prepared_sub_mul_add(
    sub: &NativeCpuNode,
    fma: &NativeCpuNode,
    capabilities: CpuCapabilities,
) -> Option<PreparedSubMulAdd> {
    let plan = &fma.plan;
    if !strict_canonical_f32_binary(sub, BinaryOp::Sub)
        || plan.semantic != SemanticOp::Ternary(TernaryOp::Fma)
        || plan.accuracy != KernelAccuracy::Strict
        || plan.effects() != KernelEffects::PURE
        || plan.params() != KernelParams::None
        || plan.temporary_bytes != 0
        || fma.input_count != 3
        || fma.output_count != 1
        || plan.input_dtypes != [Some(DType::F32); 3]
        || plan.output_dtype != DType::F32
        || sub.output_slots[0] != fma.inputs[0]
        || sub.plan.result_tensor(0) != plan.input(0)
    {
        return None;
    }
    let operands = [plan.input(0)?, plan.input(1)?, plan.input(2)?];
    let output = plan.result_tensor(0)?;
    if plan.shape != output.shape
        || plan.input_shapes != operands.map(|d| Some(d.shape))
        || operands
            .into_iter()
            .chain([output])
            .any(|d| TensorDesc::try_new(d.shape, DType::F32).as_ref() != Some(d))
        || KernelRequest::ternary(TernaryOp::Fma, operands[0], operands[1], operands[2])?.result()
            != plan.result()
    {
        return None;
    }
    let elements = output.shape.checked_num_elements()?;
    let input_elements = [
        sub.plan.input(0)?,
        sub.plan.input(1)?,
        operands[1],
        operands[2],
    ]
    .map(|d| d.num_elements());
    // Partial row/column broadcasts and empty fields keep their checked nodes.
    if elements == 0
        || input_elements
            .iter()
            .chain([&operands[0].num_elements()])
            .any(|&n| n != 1 && n != elements)
    {
        return None;
    }
    Some(PreparedSubMulAdd {
        inputs: [sub.inputs[0], sub.inputs[1], fma.inputs[1], fma.inputs[2]],
        output_slots: [fma.output_slots[0]],
        input_elements,
        elements,
        simd: capabilities
            .mask()
            .contains(CpuCapabilities::PORTABLE_SIMD.mask()),
    })
}

pub fn prepared_mul_add(
    multiply: &NativeCpuNode,
    add: &NativeCpuNode,
    product_operand: usize,
    capabilities: CpuCapabilities,
) -> Option<PreparedMulAdd> {
    if !strict_canonical_f32_binary(multiply, BinaryOp::Mul)
        || !strict_canonical_f32_binary(add, BinaryOp::Add)
        || product_operand > 1
        || multiply.output_slots[0] != add.inputs[product_operand]
        || multiply.plan.result_tensor(0) != add.plan.input(product_operand)
    {
        return None;
    }
    let elements = add.plan.shape.checked_num_elements()?;
    if elements == 0 {
        return None;
    }
    let product_elements = multiply.plan.shape.checked_num_elements()?;
    let inputs = [
        multiply.plan.input(0)?,
        multiply.plan.input(1)?,
        add.plan.input(1 - product_operand)?,
    ];
    let input_elements = inputs.map(|desc| desc.shape.num_elements());
    // Only scalar or complete dense planes may cross either boundary.
    // Partial row/column broadcasts retain their ordinary checked nodes.
    if (product_elements != 1 && product_elements != elements)
        || input_elements
            .iter()
            .any(|&count| count != 1 && count != elements)
    {
        return None;
    }
    let order = if product_operand == 0 {
        MulAddOrder::ProductAddend
    } else {
        MulAddOrder::AddendProduct
    };
    let scalar = input_elements.map(|count| count == 1);
    // These helpers explicitly invoke portable::vec, not the architecture
    // alias which may select AVX2. Never infer permission from the host or
    // call them under a scalar-only policy. Compiler target-baseline choices
    // remain the same artifact-wide contract as in CpuCapabilities.
    let kernel = if !capabilities
        .mask()
        .contains(CpuCapabilities::PORTABLE_SIMD.mask())
    {
        PreparedMulAddKernel::Scalar
    } else if scalar[0] != scalar[1] {
        let input = usize::from(scalar[0]);
        let scale_order = match (input, order) {
            (0, MulAddOrder::ProductAddend) => ScaleAddOrder::InputScaleAddend,
            (1, MulAddOrder::ProductAddend) => ScaleAddOrder::ScaleInputAddend,
            (0, MulAddOrder::AddendProduct) => ScaleAddOrder::AddendInputScale,
            (1, MulAddOrder::AddendProduct) => ScaleAddOrder::AddendScaleInput,
            _ => return None,
        };
        PreparedMulAddKernel::Scale {
            input,
            order: scale_order,
            scalar_addend: scalar[2],
        }
    } else if !scalar[0] && !scalar[2] {
        PreparedMulAddKernel::Dense
    } else {
        // The current helper surface has no dense*dense+scalar or
        // scalar*scalar+broadcast form; keep their exact scalar arithmetic.
        PreparedMulAddKernel::Scalar
    };
    Some(PreparedMulAdd {
        inputs: [
            multiply.inputs[0],
            multiply.inputs[1],
            add.inputs[1 - product_operand],
        ],
        output_slots: [add.output_slots[0]],
        input_elements,
        elements,
        order,
        kernel,
    })
}

pub fn prepared_scale_interleave3(
    interleave: &NativeCpuNode,
    multiplies: [&NativeCpuNode; 3],
    capabilities: CpuCapabilities,
) -> Option<PreparedScaleInterleave3> {
    let plan = &interleave.plan;
    if plan.semantic != SemanticOp::Interleave3F32
        || plan.accuracy != KernelAccuracy::Strict
        || plan.effects() != KernelEffects::PURE
        || plan.params() != KernelParams::None
        || plan.temporary_bytes != 0
        || interleave.input_count != 3
        || interleave.output_count != 1
        || plan.input_dtypes != [Some(DType::F32); 3]
        || plan.output_dtype != DType::F32
    {
        return None;
    }
    let output = plan.result_tensor(0)?;
    let inputs = [plan.input(0)?, plan.input(1)?, plan.input(2)?];
    if plan.shape != output.shape
        || plan.input_shapes != inputs.map(|desc| Some(desc.shape))
        || inputs
            .into_iter()
            .chain([output])
            .any(|desc| TensorDesc::try_new(desc.shape, DType::F32).as_ref() != Some(desc))
        || KernelRequest::interleave3_f32(inputs[0], inputs[1], inputs[2], output)?.result()
            != plan.result()
    {
        return None;
    }
    let field = axnn::Shape::try_from(&output.shape.dims()[..2]).ok()?;
    let field_desc = TensorDesc::try_new(field, DType::F32)?;
    let elements = field.checked_num_elements()?;
    // For N=1 the dense and scalar roles can be ambiguous. Leave that small
    // case to the existing checked nodes rather than inferring intent.
    if elements <= 1 {
        return None;
    }
    for (axis, multiply) in multiplies.iter().enumerate() {
        if !strict_canonical_f32_binary(multiply, BinaryOp::Mul)
            || multiply.output_slots[0] != interleave.inputs[axis]
            || multiply.plan.result_tensor(0) != Some(&field_desc)
            || inputs[axis] != &field_desc
        {
            return None;
        }
    }
    let scalar_side = (0..2).find(|&index| {
        multiplies[0]
            .plan
            .input(index)
            .is_some_and(|desc| desc.num_elements() == 1)
    })?;
    let scalar = multiplies[0].inputs[scalar_side];
    let scalar_desc = multiplies[0].plan.input(scalar_side)?;
    let mut operands = [0_u32; 4];
    let mut scale_first = [false; 3];
    for (axis, multiply) in multiplies.iter().enumerate() {
        // Sharing numerical values is insufficient: separate uniforms may
        // change independently on the next run. Require one tensor identity
        // and its complete canonical descriptor on all three MULs.
        let side = (0..2).find(|&index| {
            multiply.inputs[index] == scalar
                && multiply.plan.input(index) == Some(scalar_desc)
                && multiply.plan.input(1 - index) == Some(&field_desc)
        })?;
        operands[axis] = multiply.inputs[1 - side];
        scale_first[axis] = side == 0;
    }
    operands[3] = scalar;
    Some(PreparedScaleInterleave3 {
        inputs: operands,
        output_slots: [interleave.output_slots[0]],
        elements,
        scale_first,
        simd: capabilities
            .mask()
            .contains(CpuCapabilities::PORTABLE_SIMD.mask()),
    })
}

pub fn strict_canonical_word_node(node: &NativeCpuNode, semantic: SemanticOp) -> bool {
    let plan = &node.plan;
    let input_dtypes = match semantic {
        SemanticOp::Cast => [Some(DType::F32), None, None],
        SemanticOp::IntegerBinary(_) => [Some(DType::I32), Some(DType::I32), None],
        _ => return false,
    };
    if plan.semantic != semantic
        || plan.implementation != axnn_cpu::CpuImplementation::Scalar
        || plan.accuracy != KernelAccuracy::Strict
        || plan.effects() != KernelEffects::PURE
        || plan.params() != KernelParams::None
        || plan.temporary_bytes != 0
        || node.input_count != semantic.arity()
        || node.output_count != 1
        || plan.input_dtypes != input_dtypes
        || plan.output_dtype != DType::I32
    {
        return false;
    }
    let (Some(first), Some(output)) = (plan.input(0), plan.result_tensor(0)) else {
        return false;
    };
    let second = plan.input(1);
    if plan.input_shapes != [Some(first.shape), second.map(|desc| desc.shape), None]
        || plan.shape != output.shape
        || plan.input(2).is_some()
        || [Some(first), second, Some(output)]
            .into_iter()
            .flatten()
            .any(|desc| TensorDesc::try_new(desc.shape, desc.dtype).as_ref() != Some(desc))
    {
        return false;
    }
    let request = match semantic {
        SemanticOp::Cast if second.is_none() => KernelRequest::cast(first, DType::I32),
        SemanticOp::IntegerBinary(op) => {
            second.and_then(|second| KernelRequest::integer_binary(op, first, second))
        }
        _ => None,
    };
    request.is_some_and(|request| request.result() == plan.result())
}

pub fn single_use_prepared_producer(
    slot: u32,
    before: usize,
    producers: &[Option<usize>],
    lifetimes: &[PreparedNativeLifetime],
    instructions: &[PreparedNativeScheduleEntry],
) -> Option<usize> {
    let lifetime = lifetimes.get(slot as usize)?;
    if lifetime.pinned || lifetime.remaining_uses != 1 {
        return None;
    }
    let producer = producers
        .get(slot as usize)?
        .filter(|&producer| producer < before)?;
    matches!(
        instructions.get(producer)?,
        PreparedNativeScheduleEntry::Kernel(_)
    )
    .then_some(producer)
}

/// Recognize only an exact three-leaf word OR tree. Each leaf is a dense
/// F32→I32 cast, optionally followed by a scalar-count left shift. The
/// complete canonical descriptors must match at every consumed edge; equal
/// element counts never imply safe broadcasting or layout conversion.
pub fn prepared_cast_shift_or3(
    index: usize,
    nodes: &[NativeCpuNode],
    producers: &[Option<usize>],
    lifetimes: &[PreparedNativeLifetime],
    instructions: &[PreparedNativeScheduleEntry],
) -> Option<(PreparedCastShiftOr3, [Option<usize>; 7])> {
    let root = &nodes[index];
    let or = SemanticOp::IntegerBinary(IntegerBinaryOp::Or);
    if !strict_canonical_word_node(root, or)
        || !matches!(instructions[index], PreparedNativeScheduleEntry::Kernel(_))
    {
        return None;
    }
    let word_desc = root.plan.result_tensor(0)?;
    if (0..2).any(|side| root.plan.input(side) != Some(word_desc)) {
        return None;
    }
    let elements = word_desc.shape.checked_num_elements()?;
    if elements == 0 {
        return None;
    }
    let plane_desc = TensorDesc::try_new(word_desc.shape, DType::F32)?;
    let (inner_side, inner_index) = (0..2).find_map(|side| {
        let producer = single_use_prepared_producer(
            root.inputs[side],
            index,
            producers,
            lifetimes,
            instructions,
        )?;
        strict_canonical_word_node(&nodes[producer], or).then_some((side, producer))
    })?;
    let inner = &nodes[inner_index];
    if inner.plan.result_tensor(0) != Some(word_desc)
        || (0..2).any(|side| inner.plan.input(side) != Some(word_desc))
    {
        return None;
    }
    let leaves = [
        (inner.inputs[0], inner_index),
        (inner.inputs[1], inner_index),
        (root.inputs[1 - inner_side], index),
    ];
    let mut fused = PreparedCastShiftOr3 {
        inputs: [0; 6],
        input_count: 3,
        shift_operands: [None; 3],
        output_slots: [root.output_slots[0]],
        elements,
    };
    let mut elided = [None; 7];
    elided[0] = Some(inner_index);
    for (axis, (leaf, consumer)) in leaves.into_iter().enumerate() {
        let mut producer =
            single_use_prepared_producer(leaf, consumer, producers, lifetimes, instructions)?;
        let node = &nodes[producer];
        if node.plan.semantic == SemanticOp::IntegerBinary(IntegerBinaryOp::ShiftLeft) {
            if !strict_canonical_word_node(node, node.plan.semantic)
                || node.plan.result_tensor(0) != Some(word_desc)
                || node.plan.input(0) != Some(word_desc)
                || node.plan.input(1)?.num_elements() != 1
            {
                return None;
            }
            let shift_operand = fused.input_count;
            fused.inputs[shift_operand as usize] = node.inputs[1];
            fused.input_count += 1;
            fused.shift_operands[axis] = Some(shift_operand);
            elided[2 + axis * 2] = Some(producer);
            producer = single_use_prepared_producer(
                node.inputs[0],
                producer,
                producers,
                lifetimes,
                instructions,
            )?;
        }
        let cast = &nodes[producer];
        if !strict_canonical_word_node(cast, SemanticOp::Cast)
            || cast.plan.result_tensor(0) != Some(word_desc)
            || cast.plan.input(0) != Some(&plane_desc)
        {
            return None;
        }
        elided[1 + axis * 2] = Some(producer);
        fused.inputs[axis] = cast.inputs[0];
    }
    Some((fused, elided))
}

pub fn native_schedule_in<A: Allocator + Clone>(
    nodes: &[NativeCpuNode],
    capabilities: CpuCapabilities,
    lifetimes: &mut [PreparedNativeLifetime],
    enabled: bool,
    allocator: A,
) -> Result<Vec<PreparedNativeScheduleEntry, A>, AxCompileError> {
    // Discover combinations on validated logical SSA, before physical slots
    // can hide producer identities. Operand occurrences exclude graph outputs
    // here because the separate pin bit forbids eliding their producers.
    let mut producers = Vec::new_in(allocator.clone());
    reserve(&mut producers, lifetimes.len())?;
    producers.resize(lifetimes.len(), None);
    let mut instructions = Vec::new_in(allocator.clone());
    reserve(&mut instructions, nodes.len())?;
    for (index, node) in nodes.iter().enumerate() {
        instructions.push(PreparedNativeScheduleEntry::Kernel(index));
        for output in node.outputs() {
            producers[output as usize] = Some(index);
        }
    }
    if !enabled {
        return Ok(instructions);
    }
    // A node that reads a view is never fused, as consumer or as elided
    // producer: fused instructions carry whole slots, not ranges.
    for (index, add) in nodes.iter().enumerate() {
        if !strict_canonical_f32_binary(add, BinaryOp::Add) || add.has_ranges() {
            continue;
        }
        for product_operand in 0..2 {
            let product = add.inputs[product_operand] as usize;
            if lifetimes[product].pinned || lifetimes[product].remaining_uses != 1 {
                continue;
            }
            let Some(producer) = producers[product]
                .filter(|&producer| producer < index && !nodes[producer].has_ranges())
            else {
                continue;
            };
            if let Some(fused) =
                prepared_mul_add(&nodes[producer], add, product_operand, capabilities)
            {
                instructions[producer] = PreparedNativeScheduleEntry::Elided;
                instructions[index] = PreparedNativeScheduleEntry::MulAdd(fused);
                break;
            }
        }
    }
    for (index, fma) in nodes.iter().enumerate() {
        if fma.plan.semantic != SemanticOp::Ternary(TernaryOp::Fma)
            || fma.has_ranges()
            || !matches!(instructions[index], PreparedNativeScheduleEntry::Kernel(_))
        {
            continue;
        }
        let difference = fma.inputs[0] as usize;
        if lifetimes[difference].pinned || lifetimes[difference].remaining_uses != 1 {
            continue;
        }
        let Some(producer) = producers[difference].filter(|&p| p < index && !nodes[p].has_ranges())
        else {
            continue;
        };
        if !matches!(
            instructions[producer],
            PreparedNativeScheduleEntry::Kernel(_)
        ) {
            continue;
        }
        if let Some(fused) = prepared_sub_mul_add(&nodes[producer], fma, capabilities) {
            instructions[producer] = PreparedNativeScheduleEntry::Elided;
            instructions[index] = PreparedNativeScheduleEntry::SubMulAdd(fused);
        }
    }
    for (index, interleave) in nodes.iter().enumerate() {
        if interleave.plan.semantic != SemanticOp::Interleave3F32 || interleave.has_ranges() {
            continue;
        }
        let mut multiply_indices = [0_usize; 3];
        let candidates = interleave
            .inputs
            .iter()
            .enumerate()
            .all(|(axis, &product)| {
                let product = product as usize;
                if lifetimes[product].pinned || lifetimes[product].remaining_uses != 1 {
                    return false;
                }
                let Some(producer) = producers[product]
                    .filter(|&producer| producer < index && !nodes[producer].has_ranges())
                else {
                    return false;
                };
                if !matches!(
                    instructions[producer],
                    PreparedNativeScheduleEntry::Kernel(_)
                ) || multiply_indices[..axis].contains(&producer)
                {
                    return false;
                }
                multiply_indices[axis] = producer;
                true
            });
        if candidates
            && let Some(fused) = prepared_scale_interleave3(
                interleave,
                multiply_indices.map(|index| &nodes[index]),
                capabilities,
            )
        {
            for producer in multiply_indices {
                instructions[producer] = PreparedNativeScheduleEntry::Elided;
            }
            instructions[index] = PreparedNativeScheduleEntry::ScaleInterleave3(fused);
        }
    }
    // Packing chains span several nodes; keep the pass away from views.
    let viewed = nodes.iter().any(NativeCpuNode::has_ranges);
    for index in (0..nodes.len()).filter(|_| !viewed) {
        if let Some((fused, elided)) =
            prepared_cast_shift_or3(index, nodes, &producers, lifetimes, &instructions)
        {
            for producer in elided.into_iter().flatten() {
                instructions[producer] = PreparedNativeScheduleEntry::Elided;
            }
            instructions[index] = PreparedNativeScheduleEntry::CastShiftOr3(fused);
        }
    }
    // Moving operations to their ADD/FMA, INTERLEAVE3 or word OR consumer extends
    // original operands' live ranges. Recompute every occurrence, including
    // repeated planes and scalar shift counts, from the executable schedule;
    // the old counts would free those inputs before intervening nodes finish.
    for lifetime in lifetimes.iter_mut() {
        lifetime.remaining_uses = 0;
    }
    // Reuse the logical producer table as a last-consumer table. Together
    // with occurrence counts it identifies sole readers without rescanning
    // the graph for each candidate projection.
    producers.fill(None);
    for (index, instruction) in instructions
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| entry.instruction(nodes).map(|node| (index, node)))
    {
        for &input in instruction.inputs() {
            let lifetime = &mut lifetimes[input as usize];
            lifetime.remaining_uses = lifetime
                .remaining_uses
                .checked_add(1)
                .ok_or(AxArtifactError::InvalidPartition)?;
            producers[input as usize] = Some(index);
        }
    }

    // A topological schedule can interleave an independent resource operation
    // between a projection and its pure epilogue. The windowed executor only
    // offers adjacent fused runs; restore that adjacency on logical SSA, before
    // assigning physical slots. Move only the sole-reader RELU/square tail:
    // neither reads anything except its predecessor, and resource operations
    // retain their original relative order. CPU execution remains unfused and
    // numerically identical when the device declines the run.
    let mut compact = Vec::new_in(allocator);
    reserve(&mut compact, instructions.len())?;
    for index in 0..instructions.len() {
        let entry = instructions[index];
        if matches!(entry, PreparedNativeScheduleEntry::Elided) {
            continue;
        }
        compact.push(entry);
        let PreparedNativeScheduleEntry::Kernel(first) = entry else {
            continue;
        };
        let projection = &nodes[first];
        let plain = matches!(
            (projection.plan.semantic, projection.plan.params()),
            (
                SemanticOp::MatMulNvfp4Transposed | SemanticOp::MatMulTransposed,
                KernelParams::MatmulEpilogue {
                    activation: axnn::lowering::EpilogueActivation::None,
                    ..
                }
            ) | (SemanticOp::MatMulTransposed, KernelParams::None)
        );
        if !plain
            || projection.has_ranges()
            || projection.output_count != 1
            || projection.plan.effects() != KernelEffects::PURE
        {
            continue;
        }
        let value = projection.output_slots[0] as usize;
        if lifetimes[value].pinned || lifetimes[value].remaining_uses != 1 {
            continue;
        }
        let Some(relu_index) = producers[value].filter(|&i| i > index) else {
            continue;
        };
        let PreparedNativeScheduleEntry::Kernel(relu_node) = instructions[relu_index] else {
            continue;
        };
        let relu = &nodes[relu_node];
        if relu.plan.semantic != SemanticOp::Unary(axnn::UnaryOp::Relu)
            || relu.plan.effects() != KernelEffects::PURE
            || relu.has_ranges()
            || relu.input_count != 1
            || relu.output_count != 1
            || relu.inputs[0] != value as u32
            || relu.plan.input(0) != projection.plan.result_tensor(0)
            || relu.plan.result_tensor(0) != projection.plan.result_tensor(0)
        {
            continue;
        }
        let rectified = relu.output_slots[0] as usize;
        if lifetimes[rectified].pinned || lifetimes[rectified].remaining_uses != 2 {
            continue;
        }
        let Some(square_index) = producers[rectified].filter(|&i| i > relu_index) else {
            continue;
        };
        let PreparedNativeScheduleEntry::Kernel(square_node) = instructions[square_index] else {
            continue;
        };
        let square = &nodes[square_node];
        if !strict_canonical_f32_binary(square, BinaryOp::Mul)
            || square.has_ranges()
            || square.inputs[..2] != [rectified as u32; 2]
            || square.plan.result_tensor(0) != relu.plan.result_tensor(0)
        {
            continue;
        }
        compact.push(core::mem::replace(
            &mut instructions[relu_index],
            PreparedNativeScheduleEntry::Elided,
        ));
        compact.push(core::mem::replace(
            &mut instructions[square_index],
            PreparedNativeScheduleEntry::Elided,
        ));
    }
    Ok(compact)
}

/// Storage descriptor for one graph-visible result. `axnn` models reductions
/// as host scalars, while `.ax` represents every edge as a tensor; the scalar
/// bridge is therefore the canonical rank-zero F32 tensor and nothing else.
fn native_result_desc(plan: &CpuKernelPlan, index: usize) -> Option<TensorDesc> {
    match plan.result() {
        KernelResult::Scalar(KernelScalar::F32) if index == 0 => {
            TensorDesc::try_new(axnn::Shape::SCALAR, DType::F32)
        }
        _ => plan.result_tensor(index).copied(),
    }
}

/// Build a reusable arena that owns storage for every graph input.
pub fn prepare_native_in<M: Allocator, R: Allocator + Clone>(
    native: &NativeCpuPartition<M>,
    graph: Graph<'_>,
    input_ids: &[TensorId],
    output_ids: &[TensorId],
    allocator: R,
) -> Result<PreparedNative<R>, AxCompileError> {
    prepare_native_ingress_in(native, graph, input_ids, output_ids, false, allocator)
}

/// Build a reusable arena. With `borrow_inputs`, graph inputs that are not
/// also outputs get no storage: every run borrows them in place, their slots
/// are never recycled, and only typed ingress can run the arena.
pub fn prepare_native_ingress_in<M: Allocator, R: Allocator + Clone>(
    native: &NativeCpuPartition<M>,
    graph: Graph<'_>,
    input_ids: &[TensorId],
    output_ids: &[TensorId],
    borrow_inputs: bool,
    allocator: R,
) -> Result<PreparedNative<R>, AxCompileError> {
    let unsupported = || AxCompileError::UnsupportedImplementation {
        backend: CPU_BACKEND,
        implementation: CPU_KERNEL_IMPLEMENTATION,
    };
    // Pure and resource-reading nodes, plus raster and key/value cache
    // nodes with exactly their ordered effect sets: the schedule already
    // chains ordered nodes, and no fusion touches an effectful node.
    if (0..graph.node_count()).any(|index| {
        !graph.node(index).is_ok_and(|node| {
            if ax::raster::raster_op(node.op).is_some() {
                node.flags == ax::RASTER_NODE_FLAGS
            } else if node.op == ax::OpCode::KV_APPEND {
                node.flags == ax::kv::KV_APPEND_NODE_FLAGS
            } else if node.op == ax::OpCode::ATTENTION_DECODE {
                node.flags == ax::kv::ATTENTION_DECODE_NODE_FLAGS
            } else if matches!(
                node.op,
                ax::OpCode::CAUSAL_CONV1D | ax::OpCode::SSM_SCAN | ax::OpCode::GATED_DELTA_RULE
            ) {
                node.flags == ax::recurrent::RECURRENT_NODE_FLAGS
            } else {
                node.flags.0 & !ax::NodeFlags::READS_RESOURCE.0 == 0
            }
        })
    }) {
        return Err(unsupported());
    }

    // Retain the region-local numbering already encoded in NativeCpuNode.
    // Unused graph inputs and passthrough/constant outputs are added only as
    // needed to preserve the complete graph ABI.
    let capacity = native
        .tensors
        .len()
        .checked_add(input_ids.len())
        .and_then(|count| count.checked_add(output_ids.len()))
        .ok_or(AllocationError::AllocationFailed)?;
    let mut tensors = Vec::new_in(allocator.clone());
    reserve(&mut tensors, capacity)?;
    tensors.extend_from_slice(&native.tensors);
    // Keep the boundary lookup results in ABI order. Reuse these logical
    // slots for fused use counts and physical bindings instead of searching
    // the tensor list again. This temporary storage scales with the boundary,
    // not the containing graph's potentially much larger tensor ID space.
    let boundary_count = input_ids
        .len()
        .checked_add(output_ids.len())
        .ok_or(AllocationError::AllocationFailed)?;
    let mut boundary_slots = Vec::new_in(allocator.clone());
    reserve(&mut boundary_slots, boundary_count)?;
    for &id in input_ids.iter().chain(output_ids) {
        boundary_slots.push(native_tensor_slot(&mut tensors, id)?);
    }

    let mut lifetimes = Vec::new_in(allocator.clone());
    reserve(&mut lifetimes, tensors.len())?;
    let mut logical_bytes = 0_usize;
    for &id in &tensors {
        let tensor = graph
            .tensor(id)
            .map_err(|_| AxCompileError::MissingTensor(id))?;
        let bytes = tensor_storage_bytes(tensor).ok_or(AxCompileError::MissingTensor(id))?;
        let input = tensor.flags.contains(TensorFlags::INPUT);
        let constant = tensor.flags.contains(TensorFlags::CONSTANT);
        if (input && constant)
            || !matches!(
                tensor.dtype,
                DType::F32
                    | DType::I32
                    | DType::U8
                    | DType::Q4
                    | DType::Q6
                    | DType::Q8
                    | DType::B8
                    | DType::B16
            )
        {
            return Err(unsupported());
        }
        if constant && tensor.constant().len() != bytes {
            return Err(AxCompileError::InvalidInputSize {
                tensor: id,
                expected: bytes,
                actual: tensor.constant().len(),
            });
        }
        logical_bytes = logical_bytes
            .checked_add(bytes)
            .ok_or(AllocationError::AllocationFailed)?;
        let output = tensor.flags.contains(TensorFlags::OUTPUT);
        lifetimes.push(PreparedNativeLifetime {
            layout: PreparedNativeLayout {
                dtype: tensor.dtype,
                bytes,
            },
            ingress: input || constant,
            // A borrowed input's slot holds no storage to lend: never
            // recycle it.
            pinned: constant || output || (borrow_inputs && input),
            available: input || constant,
            remaining_uses: 0,
            slot: None,
        });
    }

    // Validate the original logical graph before planning or allocating its
    // storage. In-place, accumulating, or multiply-defined logical tensors are
    // not made safe by physical buffer reuse and remain unsupported.
    let mut voxel_scratch_elements = 0_usize;
    let mut kernel_scratch_elements = 0_usize;
    let mut resource_requirements = Vec::new_in(allocator.clone());
    for node in &native.nodes {
        let voxel = node.plan.semantic == SemanticOp::VoxelBrickTrace;
        let requirement = NativeResourceRequirement::for_plan(&node.plan)?;
        let effects = if matches!(node.plan.semantic, SemanticOp::Raster(_)) {
            KernelEffects::RASTER
        } else if node.plan.semantic == SemanticOp::KvAppend {
            KernelEffects::WRITES_RESOURCE
        } else if matches!(
            node.plan.semantic,
            SemanticOp::CausalConv1d | SemanticOp::SsmScan | SemanticOp::GatedDeltaRule
        ) {
            KernelEffects::UPDATES_RESOURCE
        } else if requirement.is_some() {
            KernelEffects::READS_RESOURCE
        } else {
            KernelEffects::PURE
        };
        let result_count = match node.plan.semantic {
            SemanticOp::VoxelBrickTrace | SemanticOp::TopK => 2,
            SemanticOp::VolumeTrilinearRgb => 3,
            _ => 1,
        };
        let temporary_bytes = if voxel {
            let elements = node.plan.shape.num_elements();
            voxel_scratch_elements = voxel_scratch_elements.max(elements);
            elements
                .checked_mul(size_of::<u32>())
                .ok_or_else(unsupported)?
        } else if let KernelParams::MoeExperts(params) = node.plan.params() {
            let tokens = node
                .plan
                .input(0)
                .ok_or_else(unsupported)?
                .shape
                .num_elements()
                .checked_div(params.hidden as usize)
                .ok_or_else(unsupported)?;
            let elements =
                axnn_cpu::moe_prepared_scratch_elements(params, tokens).ok_or_else(unsupported)?;
            kernel_scratch_elements = kernel_scratch_elements.max(elements);
            elements
                .checked_mul(size_of::<f32>())
                .ok_or_else(unsupported)?
        } else if let KernelParams::AttentionDecode(params) = node.plan.params() {
            let elements = axnn_cpu::attention_scratch_elements(params);
            kernel_scratch_elements = kernel_scratch_elements.max(elements);
            elements
                .checked_mul(size_of::<f32>())
                .ok_or_else(unsupported)?
        } else if node.plan.semantic == SemanticOp::MatMulNvfp4Transposed {
            let elements = node.plan.temporary_bytes.div_ceil(size_of::<f32>());
            kernel_scratch_elements = kernel_scratch_elements.max(elements);
            node.plan.temporary_bytes
        } else {
            0
        };
        if node.input_count as usize > node.inputs.len()
            || node.input_count != node.plan.semantic.arity()
            || node.plan.effects() != effects
            || node.plan.temporary_bytes != temporary_bytes
            || node.output_count as usize != result_count
            || node.plan.result_count() != result_count
            || !matches!(
                node.plan.result(),
                KernelResult::Tensor(_)
                    | KernelResult::TensorPair(_)
                    | KernelResult::Tensors(_)
                    | KernelResult::Scalar(KernelScalar::F32)
            )
            || !native_result_desc(&node.plan, 0).is_some_and(|desc| {
                desc.dtype == node.plan.output_dtype && desc.shape == node.plan.shape
            })
            || !matches!(
                node.plan.semantic,
                SemanticOp::Identity
                    | SemanticOp::Cast
                    | SemanticOp::IntegerBinary(_)
                    | SemanticOp::BufferLoadU8
                    | SemanticOp::Unary(_)
                    | SemanticOp::Binary(_)
                    | SemanticOp::Ternary(_)
                    | SemanticOp::Logical(_)
                    | SemanticOp::Compare(_)
                    | SemanticOp::Select
                    | SemanticOp::MatMul
                    | SemanticOp::MatMulTransposed
                    | SemanticOp::MatMulNvfp4Transposed
                    | SemanticOp::Softmax
                    | SemanticOp::RmsNorm
                    | SemanticOp::LayerNorm
                    | SemanticOp::Rope(_)
                    | SemanticOp::Reduce(_)
                    | SemanticOp::Interleave3F32
                    | SemanticOp::VoxelBrickTrace
                    | SemanticOp::VolumeTrilinearRgb
                    | SemanticOp::Raster(_)
                    | SemanticOp::KvAppend
                    | SemanticOp::AttentionDecode
                    | SemanticOp::Gather
                    | SemanticOp::TopK
                    | SemanticOp::CausalConv1d
                    | SemanticOp::SsmScan
                    | SemanticOp::GatedDeltaRule
                    | SemanticOp::MoeExperts
                    | SemanticOp::MoeRoute
                    | SemanticOp::MoeGateUpNvfp4
                    | SemanticOp::MoeDownNvfp4
            )
        {
            return Err(unsupported());
        }
        if let Some(requirement) = requirement {
            // Keep conflicting metadata for one key as separate requirements:
            // no binding can satisfy both, just as in the checked node path.
            if !resource_requirements.contains(&requirement) {
                reserve(&mut resource_requirements, 1)?;
                resource_requirements.push(requirement);
            }
        }
        for (index, &input) in node.inputs[..node.input_count as usize].iter().enumerate() {
            let desc = node
                .plan
                .input(index)
                .ok_or(AxArtifactError::InvalidPartition)?;
            let lifetime = lifetimes
                .get_mut(input as usize)
                .ok_or(AxArtifactError::InvalidPartition)?;
            let fits = match node.input_ranges[index] {
                // A view reads a whole-block part of its source's storage.
                Some(range) => {
                    lifetime.layout.dtype == desc.dtype
                        && range.elements == desc.shape.num_elements()
                        && range
                            .offset
                            .checked_add(range.elements)
                            .is_some_and(|end| end <= lifetime.layout.elements())
                }
                None => lifetime
                    .layout
                    .matches(desc.dtype, desc.shape.num_elements()),
            };
            if node.outputs().any(|output| input == output) || !lifetime.available || !fits {
                return Err(AxArtifactError::InvalidPartition.into());
            }
            // Count operand occurrences, not just distinct consumers: a node
            // such as ADD(x, x) must release x only after both uses retire.
            lifetime.remaining_uses = lifetime
                .remaining_uses
                .checked_add(1)
                .ok_or(AxArtifactError::InvalidPartition)?;
        }
        for (index, slot) in node.outputs().enumerate() {
            let desc = native_result_desc(&node.plan, index).ok_or_else(unsupported)?;
            let output = lifetimes
                .get_mut(slot as usize)
                .ok_or(AxArtifactError::InvalidPartition)?;
            if !desc.is_contiguous()
                || output.available
                || !output.layout.matches(desc.dtype, desc.shape.num_elements())
            {
                return Err(AxArtifactError::InvalidPartition.into());
            }
            output.available = true;
        }
    }

    let instructions = native_schedule_in(
        &native.nodes,
        native.capabilities,
        &mut lifetimes,
        native.fusions,
        allocator.clone(),
    )?;

    // Capture each graph input's uses while operands still have logical
    // identities. Counting after physical assignment also counts consumers of
    // later values that recycle the input slot, keeping borrowed ingress live
    // past its final use and hiding those later values from their consumers.
    let mut borrowed_input_takes = Vec::new_in(allocator.clone());
    reserve(&mut borrowed_input_takes, input_ids.len())?;
    for &logical in &boundary_slots[..input_ids.len()] {
        borrowed_input_takes.push(
            u32::try_from(lifetimes[logical as usize].remaining_uses)
                .map_err(|_| AxArtifactError::InvalidPartition)?,
        );
    }

    let mut layouts = Vec::new_in(allocator.clone());
    reserve(&mut layouts, tensors.len())?;
    let mut free = Vec::new_in(allocator.clone());
    reserve(&mut free, tensors.len())?;
    let mut nodes = Vec::new_in(allocator.clone());
    reserve(&mut nodes, native.nodes.len())?;

    // Every input is decoded before the first node runs, even an input whose
    // first consumer comes much later. Assign distinct ingress slots first;
    // unused, unpinned inputs can then lend their storage to a later output.
    for lifetime in &mut lifetimes {
        if lifetime.ingress {
            let slot =
                u32::try_from(layouts.len()).map_err(|_| AxArtifactError::InvalidPartition)?;
            layouts.push(lifetime.layout);
            lifetime.slot = Some(slot);
        }
    }
    for lifetime in &lifetimes {
        if lifetime.ingress && !lifetime.pinned && lifetime.remaining_uses == 0 {
            free.push(lifetime.slot.ok_or(AxArtifactError::InvalidPartition)?);
        }
    }

    for node in instructions
        .into_iter()
        .filter_map(|entry| entry.instruction(&native.nodes))
    {
        // Assign all outputs BEFORE retiring any operand. Even an operand's
        // last use cannot share this node's destination: the checked invoker
        // requires immutable inputs and distinct mutable output slices.
        let mut physical = node;
        for (index, output) in node.outputs().enumerate() {
            let layout = lifetimes[output as usize].layout;
            let slot = if let Some(index) = free
                .iter()
                .position(|&slot| layouts[slot as usize] == layout)
            {
                free.swap_remove(index)
            } else {
                let slot =
                    u32::try_from(layouts.len()).map_err(|_| AxArtifactError::InvalidPartition)?;
                layouts.push(layout);
                slot
            };
            lifetimes[output as usize].slot = Some(slot);
            if physical.output_slots()[..index].contains(&slot) {
                return Err(AxArtifactError::InvalidPartition.into());
            }
            physical.output_slots_mut()[index] = slot;
        }
        for (index, &input) in node.inputs().iter().enumerate() {
            let input_slot = lifetimes[input as usize]
                .slot
                .ok_or(AxArtifactError::InvalidPartition)?;
            if physical.outputs().any(|output| input_slot == output) {
                return Err(AxArtifactError::InvalidPartition.into());
            }
            physical.inputs_mut()[index] = input_slot;
        }
        nodes.push(physical);
        for &input in node.inputs() {
            let lifetime = &mut lifetimes[input as usize];
            lifetime.remaining_uses = lifetime
                .remaining_uses
                .checked_sub(1)
                .ok_or(AxArtifactError::InvalidPartition)?;
            if lifetime.remaining_uses == 0 && !lifetime.pinned {
                free.push(lifetime.slot.ok_or(AxArtifactError::InvalidPartition)?);
            }
        }
        for output in node.outputs() {
            let lifetime = &lifetimes[output as usize];
            if lifetime.remaining_uses == 0 && !lifetime.pinned {
                free.push(lifetime.slot.ok_or(AxArtifactError::InvalidPartition)?);
            }
        }
    }

    let mut inputs = Vec::new_in(allocator.clone());
    reserve(&mut inputs, input_ids.len())?;
    let mut outputs = Vec::new_in(allocator.clone());
    reserve(&mut outputs, output_ids.len())?;
    for ((&id, destination), &logical) in input_ids
        .iter()
        .map(|id| (id, false))
        .chain(output_ids.iter().map(|id| (id, true)))
        .zip(&boundary_slots)
    {
        let slot = logical as usize;
        if !lifetimes[slot].available {
            return Err(AxCompileError::MissingTensor(id));
        }
        let binding = PreparedNativeBinding {
            tensor: id,
            slot: lifetimes[slot]
                .slot
                .ok_or(AxCompileError::MissingTensor(id))? as usize,
            bytes: lifetimes[slot].layout.bytes,
        };
        if destination {
            outputs.push(binding);
        } else {
            inputs.push(binding);
        }
    }
    let mut borrowed_input_by_slot = Vec::new_in(allocator.clone());
    reserve(&mut borrowed_input_by_slot, layouts.len())?;
    borrowed_input_by_slot.resize(layouts.len(), usize::MAX);
    for (index, binding) in inputs.iter().enumerate() {
        let mapped = borrowed_input_by_slot
            .get_mut(binding.slot)
            .ok_or(AxArtifactError::InvalidPartition)?;
        if *mapped != usize::MAX {
            return Err(AxArtifactError::InvalidPartition.into());
        }
        *mapped = index;
    }
    let borrowed_input_remaining = copy_slice_in(&borrowed_input_takes, allocator.clone())?;
    let mut borrowed_input_copy = Vec::new_in(allocator.clone());
    reserve(&mut borrowed_input_copy, inputs.len())?;
    for binding in &inputs {
        borrowed_input_copy.push(output_ids.contains(&binding.tensor));
    }
    // Only physical slots allocate tensor payloads. Constants are decoded once
    // into permanently pinned slots, so subsequent runs cannot overwrite them.
    // Borrowed input slots keep only their dtype, as an empty value.
    let mut values = Vec::new_in(allocator.clone());
    reserve(&mut values, layouts.len())?;
    let mut buffer_bytes = 0_usize;
    for (slot, layout) in layouts.iter().enumerate() {
        let borrowed = borrow_inputs
            && match borrowed_input_by_slot[slot] {
                usize::MAX => false,
                index => !borrowed_input_copy[index],
            };
        let elements = if borrowed { 0 } else { layout.elements() };
        let value = allocate_native_value_in(layout.dtype, elements, allocator.clone())?;
        if !borrowed {
            buffer_bytes = buffer_bytes
                .checked_add(layout.bytes)
                .ok_or(AllocationError::AllocationFailed)?;
        }
        values.push(Some(value));
    }
    for (&id, lifetime) in tensors.iter().zip(&lifetimes) {
        let tensor = graph
            .tensor(id)
            .map_err(|_| AxCompileError::MissingTensor(id))?;
        if tensor.flags.contains(TensorFlags::CONSTANT) {
            let slot = lifetime.slot.ok_or(AxArtifactError::InvalidPartition)? as usize;
            let value = values[slot]
                .as_mut()
                .ok_or(AxArtifactError::InvalidPartition)?;
            decode_prepared_native(value, tensor.constant());
        }
    }
    // Walk physical lifetimes backwards in O(instructions + operands + slots).
    // A later writer kills the previous value even if this physical slot is
    // eventually a graph output. Readers are processed after the kill because
    // the planner forbids aliasing an instruction's inputs and outputs.
    let mut last_use = Vec::new_in(allocator.clone());
    reserve(&mut last_use, layouts.len())?;
    last_use.resize(layouts.len(), None);
    for binding in &outputs {
        last_use[binding.slot] = Some(usize::MAX);
    }
    let mut output_last_use = Vec::new_in(allocator.clone());
    reserve(&mut output_last_use, nodes.len())?;
    output_last_use.resize(nodes.len(), usize::MAX);
    for (index, node) in nodes.iter().enumerate().rev() {
        if let [slot] = node.output_slots() {
            output_last_use[index] = last_use[*slot as usize].unwrap_or(index);
        }
        for slot in node.outputs() {
            last_use[slot as usize] = None;
        }
        for &slot in node.inputs() {
            last_use[slot as usize].get_or_insert(index);
        }
    }
    let output_last_use = Arc::try_new_in(output_last_use, allocator.clone())
        .map_err(|_| AllocationError::AllocationFailed)?;
    let nodes = PreparedNativeInstructions(
        Arc::try_new_in(nodes, allocator.clone()).map_err(|_| AllocationError::AllocationFailed)?,
    );
    let mut voxel_ids = Vec::new_in(allocator.clone());
    reserve(&mut voxel_ids, voxel_scratch_elements)?;
    voxel_ids.resize(voxel_scratch_elements, 0_u32);
    let mut kernel_scratch = Vec::new_in(allocator);
    reserve(&mut kernel_scratch, kernel_scratch_elements)?;
    kernel_scratch.resize(kernel_scratch_elements, 0.0_f32);
    Ok(PreparedNative {
        nodes,
        output_last_use,
        resource_requirements,
        values,
        inputs,
        borrowed_input_takes,
        borrowed_input_remaining,
        borrowed_input_by_slot,
        borrowed_input_copy,
        outputs,
        buffer_bytes,
        logical_buffers: tensors.len(),
        logical_bytes,
        voxel_ids,
        kernel_scratch,
        borrowed_ingress: borrow_inputs,
        complete: false,
        profile: None,
        check: None,
    })
}

pub fn allocate_native_value_in<A: Allocator>(
    dtype: DType,
    elements: usize,
    allocator: A,
) -> Result<NativeValue<A>, AxCompileError> {
    Ok(match dtype {
        DType::F32 => {
            let mut values = Vec::new_in(allocator);
            reserve(&mut values, elements)?;
            values.resize(elements, 0.0_f32);
            NativeValue::F32(values)
        }
        DType::I32 => {
            let mut values = Vec::new_in(allocator);
            reserve(&mut values, elements)?;
            values.resize(elements, 0_i32);
            NativeValue::I32(values)
        }
        dtype @ (DType::Q4 | DType::Q6 | DType::Q8 | DType::B8 | DType::B16) => {
            if !elements.is_multiple_of(dtype.block_size()) {
                return Err(AxArtifactError::InvalidPartition.into());
            }
            let bytes = (elements / dtype.block_size())
                .checked_mul(dtype.block_byte_size())
                .ok_or(AxArtifactError::InvalidPartition)?;
            let mut values = Vec::new_in(allocator);
            reserve(&mut values, bytes)?;
            values.resize(bytes, 0);
            NativeValue::Packed(dtype, values)
        }
        DType::U8 => {
            let mut values = Vec::new_in(allocator);
            reserve(&mut values, elements)?;
            values.resize(elements, 0_u8);
            NativeValue::U8(values)
        }
        _ => {
            return Err(AxCompileError::UnsupportedImplementation {
                backend: CPU_BACKEND,
                implementation: CPU_KERNEL_IMPLEMENTATION,
            });
        }
    })
}

/// Lengths are checked at preparation or invocation ingress, never inferred
/// from a raw byte cast. Unaligned little-endian inputs are therefore valid.
pub fn decode_prepared_native<A: Allocator>(value: &mut NativeValue<A>, bytes: &[u8]) {
    match value {
        NativeValue::F32(values) => {
            for (value, bytes) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *value = f32::from_le_bytes(*bytes);
            }
        }
        NativeValue::I32(values) => {
            for (value, bytes) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *value = i32::from_le_bytes(*bytes);
            }
        }
        NativeValue::U8(values) | NativeValue::Packed(_, values) => values.copy_from_slice(bytes),
    }
}

/// A kernel node's inputs, with each viewed operand narrowed to its range of
/// the source slot. Nothing is copied.
struct OperandInputs<'a, V: ?Sized> {
    values: &'a V,
    node: &'a NativeCpuNode,
}

impl<V: NativeInputs + ?Sized> NativeInputs for OperandInputs<'_, V> {
    fn view(&self, slot: usize) -> Option<NativeRef<'_>> {
        let whole = self.values.view(slot)?;
        match self.node.range_for(u32::try_from(slot).ok()?) {
            Some(range) => whole.range(range.offset, range.elements),
            None => Some(whole),
        }
    }
}

pub fn invoke_native_instruction<A: Allocator>(
    instruction: &PreparedNativeInstruction,
    values: &(impl NativeInputs + ?Sized),
    outputs: &mut [Option<NativeValue<A>>],
    resources: &[NativeResourceBinding<'_>],
    voxel_ids: &mut [u32],
    kernel_scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), AxCompileError> {
    if outputs.len() != instruction.output_slots().len() {
        return Err(AxArtifactError::InvalidPartition.into());
    }
    // Scene lowering marks inactive raster rows with zero visible width. An
    // all-inactive static range has the exact checked-kernel result `token=0`;
    // avoid resource resolution and the full row walk in that case. Required
    // resources were already validated at run ingress.
    if let PreparedNativeInstruction::Kernel(node) = instruction
        && let (SemanticOp::Raster(_), KernelParams::Raster(params)) =
            (node.plan.semantic, node.plan.params())
    {
        let rows = values.i32(node.inputs[0] as usize)?;
        let start = params.row as usize * axnn::RASTER_ROW_WORDS;
        let end = params.end_row() as usize * axnn::RASTER_ROW_WORDS;
        if params.is_valid()
            && end <= rows.len()
            && rows[start..end]
                .chunks_exact(axnn::RASTER_ROW_WORDS)
                .all(|row| row[6] == 0)
        {
            let [Some(NativeValue::I32(token))] = outputs else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            if token.len() != 1 {
                return Err(AxArtifactError::InvalidPartition.into());
            }
            token[0] = 0;
            return Ok(());
        }
    }
    match instruction {
        PreparedNativeInstruction::Kernel(node) => invoke_native_node(
            node,
            &OperandInputs { values, node },
            outputs,
            resources,
            voxel_ids,
            kernel_scratch,
            parallel,
        ),
        PreparedNativeInstruction::MulAdd(node) => {
            invoke_prepared_mul_add(node, values, &mut outputs[0])
        }
        PreparedNativeInstruction::SubMulAdd(node) => {
            let Some(NativeValue::F32(output)) = &mut outputs[0] else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            let inputs = [
                values.f32(node.inputs[0] as usize)?,
                values.f32(node.inputs[1] as usize)?,
                values.f32(node.inputs[2] as usize)?,
                values.f32(node.inputs[3] as usize)?,
            ];
            if output.len() != node.elements
                || inputs
                    .iter()
                    .zip(node.input_elements)
                    .any(|(p, n)| p.len() != n)
            {
                return Err(AxArtifactError::InvalidPartition.into());
            }
            if node.simd {
                elementwise::sub_mul_add_f32(inputs, output);
            } else {
                let strides = node.input_elements.map(|n| usize::from(n != 1));
                for (i, out) in output.iter_mut().enumerate() {
                    let [a, b, scale, addend] = core::array::from_fn(|c| inputs[c][i * strides[c]]);
                    *out = (a - b) * scale + addend;
                }
            }
            Ok(())
        }
        PreparedNativeInstruction::ScaleInterleave3(node) => {
            invoke_prepared_scale_interleave3(node, values, &mut outputs[0])
        }
        PreparedNativeInstruction::CastShiftOr3(node) => {
            invoke_prepared_cast_shift_or3(node, values, &mut outputs[0])
        }
    }
}

pub fn invoke_prepared_mul_add<A: Allocator>(
    node: &PreparedMulAdd,
    values: &(impl NativeInputs + ?Sized),
    output: &mut Option<NativeValue<A>>,
) -> Result<(), AxCompileError> {
    let Some(NativeValue::F32(output)) = output else {
        return Err(AxArtifactError::InvalidPartition.into());
    };
    let inputs = [
        values.f32(node.inputs[0] as usize)?,
        values.f32(node.inputs[1] as usize)?,
        values.f32(node.inputs[2] as usize)?,
    ];
    if output.len() != node.elements
        || inputs
            .iter()
            .zip(node.input_elements)
            .any(|(input, elements)| input.len() != elements)
    {
        return Err(AxArtifactError::InvalidPartition.into());
    }
    match node.kernel {
        PreparedMulAddKernel::Scale {
            input,
            order,
            scalar_addend,
        } => {
            let scale = inputs[1 - input][0];
            if scalar_addend {
                elementwise::scale_add_scalar_f32(
                    inputs[input],
                    scale,
                    inputs[2][0],
                    output,
                    order,
                );
            } else {
                elementwise::scale_add_f32(inputs[input], scale, inputs[2], output, order);
            }
        }
        PreparedMulAddKernel::Dense => {
            elementwise::mul_add_f32(inputs[0], inputs[1], inputs[2], output, node.order);
        }
        PreparedMulAddKernel::Scalar => {
            // Hoist order and scalar/dense indexing out of the loop. These
            // are two ordinary F32 operations, never f32::mul_add or an
            // architecture-dispatched helper under a scalar-only policy.
            let strides = node
                .input_elements
                .map(|elements| usize::from(elements != 1));
            macro_rules! apply {
                ($operation:expr) => {{
                    let operation = $operation;
                    for (index, result) in output.iter_mut().enumerate() {
                        *result = operation(
                            inputs[0][index * strides[0]],
                            inputs[1][index * strides[1]],
                            inputs[2][index * strides[2]],
                        );
                    }
                }};
            }
            match node.order {
                MulAddOrder::ProductAddend => apply!(|left, right, addend| (left * right) + addend),
                MulAddOrder::AddendProduct => apply!(|left, right, addend| addend + (left * right)),
            }
        }
    }
    Ok(())
}

// Preserve authored floating-point operand order, even though the lint treats
// these multiplications as interchangeable. No commutative rewrite is needed.
#[allow(clippy::if_same_then_else)]
pub fn invoke_prepared_scale_interleave3<A: Allocator>(
    node: &PreparedScaleInterleave3,
    values: &(impl NativeInputs + ?Sized),
    output: &mut Option<NativeValue<A>>,
) -> Result<(), AxCompileError> {
    let Some(NativeValue::F32(output)) = output else {
        return Err(AxArtifactError::InvalidPartition.into());
    };
    let planes = [
        values.f32(node.inputs[0] as usize)?,
        values.f32(node.inputs[1] as usize)?,
        values.f32(node.inputs[2] as usize)?,
    ];
    let scale = values.f32(node.inputs[3] as usize)?;
    if node.elements.checked_mul(3) != Some(output.len())
        || planes.iter().any(|plane| plane.len() != node.elements)
        || scale.len() != 1
    {
        return Err(AxArtifactError::InvalidPartition.into());
    }
    if node.simd {
        elementwise::scale_interleave3_f32(planes, scale[0], output, node.scale_first);
    } else {
        // Scalar policy must not enter the architecture-dispatched helper.
        // The packed store follows each single F32 multiply without changing
        // the rounding boundary or choosing another multiplication order.
        let scale = scale[0];
        for (index, xyz) in output.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            for (axis, result) in xyz.iter_mut().enumerate() {
                let value = planes[axis][index];
                *result = if node.scale_first[axis] {
                    scale * value
                } else {
                    value * scale
                };
            }
        }
    }
    Ok(())
}

pub fn invoke_prepared_cast_shift_or3<A: Allocator>(
    node: &PreparedCastShiftOr3,
    values: &(impl NativeInputs + ?Sized),
    output: &mut Option<NativeValue<A>>,
) -> Result<(), AxCompileError> {
    let Some(NativeValue::I32(output)) = output else {
        return Err(AxArtifactError::InvalidPartition.into());
    };
    let planes = [
        values.f32(node.inputs[0] as usize)?,
        values.f32(node.inputs[1] as usize)?,
        values.f32(node.inputs[2] as usize)?,
    ];
    if output.len() != node.elements {
        return Err(AxArtifactError::InvalidPartition.into());
    }
    let mut shifts = [0; 3];
    for (axis, operand) in node.shift_operands.iter().enumerate() {
        if let Some(operand) = operand {
            let shift = values.i32(node.inputs[*operand as usize] as usize)?;
            let [count] = shift else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            shifts[axis] = *count as u32;
        }
    }
    axnn_cpu::cast_shift_or3_f32_i32(planes, shifts, output)?;
    Ok(())
}

pub fn invoke_native_node<A: Allocator>(
    node: &NativeCpuNode,
    values: &(impl NativeInputs + ?Sized),
    outputs: &mut [Option<NativeValue<A>>],
    resources: &[NativeResourceBinding<'_>],
    voxel_ids: &mut [u32],
    kernel_scratch: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), AxCompileError> {
    if outputs.len() != node.plan.result_count() {
        return Err(AxArtifactError::InvalidPartition.into());
    }
    if node.plan.semantic == SemanticOp::VoxelBrickTrace {
        let [
            Some(NativeValue::F32(distances)),
            Some(NativeValue::I32(ids)),
        ] = outputs
        else {
            return Err(AxArtifactError::InvalidPartition.into());
        };
        axnn_cpu::invoke_voxel_brick_trace(
            &node.plan,
            voxel_resource(&node.plan, resources)?,
            values.f32(node.inputs[0] as usize)?,
            values.f32(node.inputs[1] as usize)?,
            values.f32(node.inputs[2] as usize)?,
            distances,
            ids,
            voxel_ids,
        )?;
        return Ok(());
    }
    if node.plan.semantic == SemanticOp::TopK {
        let [
            Some(NativeValue::F32(values_out)),
            Some(NativeValue::I32(indices)),
        ] = outputs
        else {
            return Err(AxArtifactError::InvalidPartition.into());
        };
        axnn_cpu::invoke_top_k(
            &node.plan,
            values.f32(node.inputs[0] as usize)?,
            values_out,
            indices,
        )?;
        return Ok(());
    }
    if node.plan.semantic == SemanticOp::VolumeTrilinearRgb {
        let [
            Some(NativeValue::F32(r)),
            Some(NativeValue::F32(g)),
            Some(NativeValue::F32(b)),
        ] = outputs
        else {
            return Err(AxArtifactError::InvalidPartition.into());
        };
        axnn_cpu::invoke_volume_trilinear_rgb(
            &node.plan,
            volume_resource(&node.plan, resources)?,
            values.f32(node.inputs[0] as usize)?,
            values.f32(node.inputs[1] as usize)?,
            values.f32(node.inputs[2] as usize)?,
            r,
            g,
            b,
        )?;
        return Ok(());
    }
    let [Some(output)] = outputs else {
        return Err(AxArtifactError::InvalidPartition.into());
    };
    match (node.plan.semantic, output) {
        (SemanticOp::Raster(_), NativeValue::I32(output)) => {
            let KernelParams::Raster(params) = node.plan.params() else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            let target = raster_target_for_params(params, resources)?;
            let find = |key: ResourceKey| resources.iter().find(|b| b.resource == key);
            let atlas = find(params.atlas).and_then(|b| match b.value {
                NativeResourceValue::CoverageAtlas(atlas) => Some(atlas),
                _ => None,
            });
            let glyphs = find(params.glyphs).and_then(|b| match b.value {
                NativeResourceValue::GlyphTable(glyphs) => Some(glyphs),
                _ => None,
            });
            let source = |slot: u32| raster_surface(params, slot, resources);
            axnn_cpu::invoke_raster(
                &node.plan,
                values.i32(node.inputs[0] as usize)?,
                values.u8(node.inputs[1] as usize)?,
                target,
                &source,
                atlas,
                glyphs,
                output,
            )?;
        }
        (SemanticOp::MoeExperts, NativeValue::F32(output)) => {
            let KernelParams::MoeExperts(params) = node.plan.params() else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            axnn_cpu::invoke_moe_experts(
                &node.plan,
                expert_store_for_params(params, resources)?,
                values.f32(node.inputs[0] as usize)?,
                values.f32(node.inputs[1] as usize)?,
                values.f32(node.inputs[2] as usize)?,
                output,
                kernel_scratch,
                parallel,
            )?;
        }
        (SemanticOp::CausalConv1d, NativeValue::F32(output)) => {
            let KernelParams::CausalConv1d(params) = node.plan.params() else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            let elements = params
                .state_elements()
                .ok_or(AxArtifactError::InvalidPartition)?;
            let (source, target) = recurrent_state(params.state, elements, resources)?;
            axnn_cpu::invoke_causal_conv1d(
                &node.plan,
                source,
                target,
                values.f32(node.inputs[0] as usize)?,
                values.f32(node.inputs[1] as usize)?,
                values.f32(node.inputs[2] as usize)?,
                output,
                parallel,
            )?;
        }
        (SemanticOp::SsmScan, NativeValue::F32(output)) => {
            let KernelParams::SsmScan(params) = node.plan.params() else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            let elements = params
                .state_elements()
                .ok_or(AxArtifactError::InvalidPartition)?;
            let (source, target) = recurrent_state(params.state, elements, resources)?;
            axnn_cpu::invoke_ssm_scan(
                &node.plan,
                source,
                target,
                values.f32(node.inputs[0] as usize)?,
                values.f32(node.inputs[1] as usize)?,
                values.f32(node.inputs[2] as usize)?,
                output,
                parallel,
            )?;
        }
        (SemanticOp::GatedDeltaRule, NativeValue::F32(output)) => {
            let KernelParams::GatedDeltaRule(params) = node.plan.params() else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            let elements = params
                .state_elements()
                .ok_or(AxArtifactError::InvalidPartition)?;
            let (source, target) = recurrent_state(params.state, elements, resources)?;
            axnn_cpu::invoke_gated_delta_rule(
                &node.plan,
                source,
                target,
                values.f32(node.inputs[0] as usize)?,
                values.f32(node.inputs[1] as usize)?,
                values.f32(node.inputs[2] as usize)?,
                output,
                parallel,
            )?;
        }
        (SemanticOp::Gather, NativeValue::F32(output)) => {
            let table = match node.plan.input(0).map(|d| d.dtype) {
                Some(DType::F32) => {
                    axnn_cpu::GatherTable::F32(values.f32(node.inputs[0] as usize)?)
                }
                Some(DType::Q8) => {
                    axnn_cpu::GatherTable::Q8(values.packed(node.inputs[0] as usize, DType::Q8)?)
                }
                Some(DType::B16) => {
                    axnn_cpu::GatherTable::B16(values.packed(node.inputs[0] as usize, DType::B16)?)
                }
                _ => return Err(AxArtifactError::InvalidPartition.into()),
            };
            axnn_cpu::invoke_gather(
                &node.plan,
                table,
                values.i32(node.inputs[1] as usize)?,
                output,
            )?;
        }
        (SemanticOp::KvAppend, NativeValue::I32(token)) => {
            let KernelParams::KvAppend(params) = node.plan.params() else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            axnn_cpu::invoke_kv_append(
                &node.plan,
                kv_cache_for_params(params, resources)?,
                values.i32(node.inputs[0] as usize)?,
                values.f32(node.inputs[1] as usize)?,
                values.f32(node.inputs[2] as usize)?,
                token,
            )?;
        }
        (SemanticOp::AttentionDecode, NativeValue::F32(output)) => {
            let KernelParams::AttentionDecode(params) = node.plan.params() else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            axnn_cpu::invoke_attention_decode(
                &node.plan,
                kv_cache_for_params(params.cache, resources)?,
                values.i32(node.inputs[0] as usize)?,
                values.i32(node.inputs[1] as usize)?,
                values.f32(node.inputs[2] as usize)?,
                output,
                kernel_scratch,
                parallel,
            )?;
        }
        (SemanticOp::BufferLoadU8, NativeValue::I32(output)) => {
            let KernelParams::BufferLoadU8(params) = node.plan.params() else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            axnn_cpu::invoke_buffer_load_u8(
                &node.plan,
                buffer_resource_for_params(params, resources)?,
                values.i32(node.inputs[0] as usize)?,
                output,
            )?;
        }
        (SemanticOp::Cast, NativeValue::I32(output)) => {
            axnn_cpu::invoke_cast_f32_i32(
                &node.plan,
                values.f32(node.inputs[0] as usize)?,
                output,
            )?;
        }
        (SemanticOp::Cast, NativeValue::F32(output))
            if matches!(
                node.plan.input(0).map(|d| d.dtype),
                Some(DType::Q4 | DType::Q6 | DType::Q8)
            ) =>
        {
            axnn_cpu::invoke_dequantize_f32(
                &node.plan,
                values.packed(node.inputs[0] as usize, node.plan.input(0).unwrap().dtype)?,
                output,
            )?;
        }
        (SemanticOp::Cast, NativeValue::F32(output)) => {
            axnn_cpu::invoke_cast_i32_f32(
                &node.plan,
                values.i32(node.inputs[0] as usize)?,
                output,
            )?;
        }
        (SemanticOp::Select, NativeValue::I32(output)) => {
            axnn_cpu::invoke_select_i32(
                &node.plan,
                values.u8(node.inputs[0] as usize)?,
                values.i32(node.inputs[1] as usize)?,
                values.i32(node.inputs[2] as usize)?,
                output,
            )?;
        }
        (
            SemanticOp::Unary(_)
            | SemanticOp::Binary(_)
            | SemanticOp::Ternary(_)
            | SemanticOp::IntegerBinary(_),
            NativeValue::I32(output),
        ) => {
            let first = values.i32(node.inputs[0] as usize)?;
            let second = (node.input_count >= 2)
                .then(|| values.i32(node.inputs[1] as usize))
                .transpose()?;
            let third = (node.input_count >= 3)
                .then(|| values.i32(node.inputs[2] as usize))
                .transpose()?;
            axnn_cpu::invoke_integer_i32(&node.plan, first, second, third, output)?;
        }
        (SemanticOp::Logical(_) | SemanticOp::Identity, NativeValue::U8(output)) => {
            let first = values.u8(node.inputs[0] as usize)?;
            let second = (node.input_count >= 2)
                .then(|| values.u8(node.inputs[1] as usize))
                .transpose()?;
            axnn_cpu::invoke_u8(&node.plan, first, second, output)?;
        }
        (SemanticOp::Compare(_), NativeValue::U8(output))
            if node.plan.input_dtypes[0] == Some(DType::I32) =>
        {
            let first = values.i32(node.inputs[0] as usize)?;
            let second = values.i32(node.inputs[1] as usize)?;
            axnn_cpu::invoke_compare_i32(&node.plan, first, second, output)?;
        }
        (SemanticOp::Identity, NativeValue::I32(output)) => {
            let first = values.i32(node.inputs[0] as usize)?;
            axnn_cpu::invoke_i32(&node.plan, first, output)?;
        }
        (SemanticOp::Compare(_), NativeValue::U8(output)) => {
            let first = values.f32(node.inputs[0] as usize)?;
            let second = values.f32(node.inputs[1] as usize)?;
            axnn_cpu::invoke_compare_f32(&node.plan, first, second, output)?;
        }
        (SemanticOp::Select, NativeValue::F32(output)) => {
            let mask = values.u8(node.inputs[0] as usize)?;
            let first = values.f32(node.inputs[1] as usize)?;
            let second = values.f32(node.inputs[2] as usize)?;
            axnn_cpu::invoke_select_f32(&node.plan, mask, first, second, output)?;
        }
        (SemanticOp::Cast, NativeValue::Packed(dtype, output)) => {
            if *dtype != node.plan.output_dtype {
                return Err(AxArtifactError::InvalidPartition.into());
            }
            axnn_cpu::invoke_quantize_f32(
                &node.plan,
                values.f32(node.inputs[0] as usize)?,
                output,
            )?;
        }

        (SemanticOp::MatMulTransposed, NativeValue::F32(output))
            if node.plan.input(0).map(|d| d.dtype) == Some(DType::F32) =>
        {
            let weights = node
                .plan
                .input(1)
                .ok_or(AxArtifactError::InvalidPartition)?;
            let activations = values.f32(node.inputs[0] as usize)?;
            if weights.dtype == DType::F32 {
                axnn_cpu::invoke_matmul_transposed_f32_weights(
                    &node.plan,
                    activations,
                    values.f32(node.inputs[1] as usize)?,
                    output,
                    parallel,
                )?;
            } else {
                axnn_cpu::invoke_matmul_transposed_f32(
                    &node.plan,
                    activations,
                    values.packed(node.inputs[1] as usize, weights.dtype)?,
                    output,
                    parallel,
                )?;
            }
        }
        (SemanticOp::MatMulTransposed, NativeValue::F32(output)) => {
            axnn_cpu::invoke_matmul_transposed(
                &node.plan,
                values.packed(
                    node.inputs[0] as usize,
                    node.plan
                        .input(0)
                        .ok_or(AxArtifactError::InvalidPartition)?
                        .dtype,
                )?,
                values.packed(node.inputs[1] as usize, DType::Q8)?,
                output,
            )?;
        }
        (SemanticOp::MoeRoute, NativeValue::I32(output)) => {
            axnn_cpu::invoke_moe_route(&node.plan, values.f32(node.inputs[0] as usize)?, output)?;
        }
        (SemanticOp::MoeGateUpNvfp4, NativeValue::F32(output)) => {
            axnn_cpu::invoke_moe_gate_up_nvfp4(
                &node.plan,
                values.f32(node.inputs[0] as usize)?,
                values.u8(node.inputs[1] as usize)?,
                values.i32(node.inputs[2] as usize)?,
                output,
                parallel,
            )?;
        }
        (SemanticOp::MoeDownNvfp4, NativeValue::F32(output)) => {
            axnn_cpu::invoke_moe_down_nvfp4(
                &node.plan,
                values.f32(node.inputs[0] as usize)?,
                values.u8(node.inputs[1] as usize)?,
                values.i32(node.inputs[2] as usize)?,
                output,
                parallel,
            )?;
        }
        (SemanticOp::MatMulNvfp4Transposed, NativeValue::F32(output)) => {
            axnn_cpu::invoke_matmul_nvfp4_transposed(
                &node.plan,
                values.f32(node.inputs[0] as usize)?,
                values.u8(node.inputs[1] as usize)?,
                values.packed(node.inputs[2] as usize, DType::B8)?,
                output,
                kernel_scratch,
                parallel,
            )?;
        }
        (
            SemanticOp::Softmax
            | SemanticOp::RmsNorm
            | SemanticOp::LayerNorm
            | SemanticOp::Rope(_),
            NativeValue::F32(output),
        ) => {
            axnn_cpu::invoke_neural_f32(
                &node.plan,
                values.f32(node.inputs[0] as usize)?,
                (node.input_count >= 2)
                    .then(|| values.f32(node.inputs[1] as usize))
                    .transpose()?,
                (node.input_count >= 3)
                    .then(|| values.f32(node.inputs[2] as usize))
                    .transpose()?,
                output,
            )?;
        }
        (SemanticOp::Reduce(_), NativeValue::F32(output)) => {
            axnn_cpu::invoke_reduce_f32(&node.plan, values.f32(node.inputs[0] as usize)?, output)?;
        }
        (SemanticOp::MatMul, NativeValue::F32(output)) => {
            let first = values.f32(node.inputs[0] as usize)?;
            let second = values.f32(node.inputs[1] as usize)?;
            output.fill(0.0);
            axnn_cpu::invoke_gemm_f32(&node.plan, first, second, output)?;
        }
        (SemanticOp::Interleave3F32, NativeValue::F32(output)) => {
            axnn_cpu::invoke_interleave3_f32(
                &node.plan,
                values.f32(node.inputs[0] as usize)?,
                values.f32(node.inputs[1] as usize)?,
                values.f32(node.inputs[2] as usize)?,
                output,
            )?;
        }
        (
            SemanticOp::Identity
            | SemanticOp::Unary(_)
            | SemanticOp::Binary(_)
            | SemanticOp::Ternary(_),
            NativeValue::F32(output),
        ) => {
            let first = values.f32(node.inputs[0] as usize)?;
            let second = (node.input_count >= 2)
                .then(|| values.f32(node.inputs[1] as usize))
                .transpose()?;
            let third = (node.input_count >= 3)
                .then(|| values.f32(node.inputs[2] as usize))
                .transpose()?;
            axnn_cpu::invoke_f32(&node.plan, first, second, third, output)?;
        }
        _ => {
            return Err(AxCompileError::UnsupportedImplementation {
                backend: CPU_BACKEND,
                implementation: CPU_KERNEL_IMPLEMENTATION,
            });
        }
    }
    Ok(())
}

pub fn compile_native_cpu_partition_in<A: Allocator + Clone>(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    partition: Partition,
    policy: AxCpuKernelPolicy,
    fusions: bool,
    allocator: A,
) -> Result<NativeCpuPartition<A>, AxCompileError> {
    if (partition.node_count == 0 && graph.node_count() != 0)
        || producers.len() < graph.tensor_count()
    {
        return Err(AxArtifactError::InvalidPartition.into());
    }
    let start = partition.order_start as usize;
    let end = start
        .checked_add(partition.node_count as usize)
        .filter(|&end| end <= order.len())
        .ok_or(AxArtifactError::InvalidPartition)?;

    let mut inside = Vec::new_in(allocator.clone());
    reserve(&mut inside, graph.node_count())?;
    inside.resize(graph.node_count(), 0_u8);
    for &node_index in &order[start..end] {
        let slot = inside
            .get_mut(node_index as usize)
            .ok_or(AxArtifactError::InvalidPartition)?;
        if *slot != 0 {
            return Err(AxArtifactError::InvalidPartition.into());
        }
        *slot = 1;
    }

    let mut nodes = Vec::new_in(allocator.clone());
    reserve(&mut nodes, partition.node_count as usize)?;
    let mut tensors = Vec::new_in(allocator.clone());
    reserve(
        &mut tensors,
        (partition.node_count as usize).saturating_mul(3 + MAX_TENSOR_RESULTS),
    )?;
    let mut inputs = Vec::new_in(allocator.clone());
    let mut outputs = Vec::new_in(allocator.clone());
    reserve(
        &mut inputs,
        (partition.node_count as usize).saturating_mul(3),
    )?;
    reserve(
        &mut outputs,
        (partition.node_count as usize).saturating_mul(MAX_TENSOR_RESULTS),
    )?;

    // Views never materialize: each maps its tensor to the root tensor whose
    // storage it reads, and the element range it covers there.
    let mut views: Vec<(TensorId, TensorId, NativeRange), A> = Vec::new_in(allocator.clone());
    let unsupported = || AxCompileError::UnsupportedImplementation {
        backend: CPU_BACKEND,
        implementation: CPU_KERNEL_IMPLEMENTATION,
    };
    for &node_index in &order[start..end] {
        let node = graph
            .node(node_index as usize)
            .map_err(|_| AxArtifactError::InvalidPartition)?;
        let plan = ax::node_kernel_request(graph, node)
            .and_then(|request| policy.plan(&request))
            .ok_or_else(unsupported)?;
        // Register a tensor this region reads but does not produce.
        let mut ingress = |input: TensorId| -> Result<(), AxCompileError> {
            let tensor = graph
                .tensor(input)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            let producer = producers[input.0 as usize];
            let produced_inside = producer != ax::schedule::NO_PRODUCER
                && inside
                    .get(producer as usize)
                    .is_some_and(|&inside| inside != 0);
            if !produced_inside
                && !tensor.flags.contains(TensorFlags::CONSTANT)
                && !inputs.contains(&input)
            {
                reserve(&mut inputs, 1)?;
                inputs.push(input);
            }
            Ok(())
        };
        if let KernelParams::View { offset } = plan.params() {
            let (Some(input), Some(output)) = (
                node.inputs().next().filter(|_| node.input_count() == 1),
                node.outputs().next().filter(|_| node.output_count() == 1),
            ) else {
                return Err(AxArtifactError::InvalidPartition.into());
            };
            let (root, base) = views
                .iter()
                .find(|(view, _, _)| *view == input)
                .map_or((input, 0), |&(_, root, range)| (root, range.offset));
            ingress(root)?;
            // A view has no storage of its own, so it cannot leave the region.
            let exported = graph
                .tensor(output)
                .map_err(|_| AxArtifactError::InvalidGraph)?
                .flags
                .contains(TensorFlags::OUTPUT)
                || (0..graph.node_count()).any(|consumer| {
                    inside[consumer] == 0
                        && graph
                            .node(consumer)
                            .is_ok_and(|consumer| consumer.inputs().any(|input| input == output))
                });
            let elements = plan.shape.checked_num_elements().ok_or_else(unsupported)?;
            let offset = usize::try_from(offset)
                .ok()
                .and_then(|offset| offset.checked_add(base))
                .ok_or_else(unsupported)?;
            if exported {
                return Err(unsupported());
            }
            reserve(&mut views, 1)?;
            views.push((output, root, NativeRange { offset, elements }));
            continue;
        }
        let mut node_inputs = [0_u32; 3];
        let mut input_ranges = [None; 3];
        let mut input_count = 0_usize;
        for input in node.inputs() {
            let slot = node_inputs
                .get_mut(input_count)
                .ok_or(AxArtifactError::InvalidPartition)?;
            match views.iter().find(|(view, _, _)| *view == input) {
                Some(&(_, root, range)) => {
                    *slot = native_tensor_slot(&mut tensors, root)?;
                    input_ranges[input_count] = Some(range);
                }
                None => {
                    *slot = native_tensor_slot(&mut tensors, input)?;
                    ingress(input)?;
                }
            }
            input_count += 1;
        }
        // Operands naming one slot must read the same part of it.
        for a in 0..input_count {
            for b in 0..a {
                if node_inputs[a] == node_inputs[b] && input_ranges[a] != input_ranges[b] {
                    return Err(unsupported());
                }
            }
        }
        if input_count != plan.semantic.arity() as usize {
            return Err(AxArtifactError::InvalidPartition.into());
        }
        if node.output_count() != plan.result_count()
            || !(1..=MAX_TENSOR_RESULTS).contains(&node.output_count())
        {
            return Err(AxArtifactError::InvalidPartition.into());
        }
        let mut output_slots = [0_u32; MAX_TENSOR_RESULTS];
        for (index, id) in node.outputs().enumerate() {
            let slot = native_tensor_slot(&mut tensors, id)?;
            if output_slots[..index].contains(&slot) || node_inputs[..input_count].contains(&slot) {
                return Err(AxArtifactError::InvalidPartition.into());
            }
            output_slots[index] = slot;
        }
        nodes.push(NativeCpuNode {
            plan,
            inputs: node_inputs,
            input_count: input_count as u8,
            output_slots,
            output_count: node.output_count() as u8,
            input_ranges,
        });

        for output in node.outputs() {
            let tensor = graph
                .tensor(output)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            let consumed_outside = (0..graph.node_count()).any(|consumer| {
                inside[consumer] == 0
                    && graph
                        .node(consumer)
                        .is_ok_and(|consumer| consumer.inputs().any(|input| input == output))
            });
            if (tensor.flags.contains(TensorFlags::OUTPUT) || consumed_outside)
                && !outputs.contains(&output)
            {
                outputs.push(output);
            }
        }
    }
    // A standalone graph may expose only input/constant passthroughs. It
    // still executes its admitted nodes (if any), reclaiming dead results.
    // An arbitrary empty region inside a larger graph remains invalid.
    let standalone_passthrough = start == 0
        && end == graph.node_count()
        && (0..graph.tensor_count()).any(|i| {
            graph.tensor(TensorId(i as u32)).is_ok_and(|tensor| {
                tensor.flags.contains(TensorFlags::OUTPUT)
                    && (tensor.flags.contains(TensorFlags::INPUT)
                        || tensor.flags.contains(TensorFlags::CONSTANT))
            })
        });
    if outputs.is_empty() && !standalone_passthrough {
        return Err(AxArtifactError::NoPartitionOutputs.into());
    }

    let mut takes = Vec::new_in(allocator.clone());
    reserve(&mut takes, tensors.len())?;
    takes.resize(tensors.len(), 0_u32);
    for node in &nodes {
        for &input in &node.inputs[..node.input_count as usize] {
            takes[input as usize] = takes[input as usize]
                .checked_add(1)
                .ok_or(AxArtifactError::InvalidPartition)?;
        }
    }
    let capabilities = policy.capabilities;
    let mut lifetimes = Vec::new_in(allocator.clone());
    reserve(&mut lifetimes, tensors.len())?;
    for (&id, &uses) in tensors.iter().zip(&takes) {
        let tensor = graph
            .tensor(id)
            .map_err(|_| AxCompileError::MissingTensor(id))?;
        let constant = tensor.flags.contains(TensorFlags::CONSTANT);
        lifetimes.push(PreparedNativeLifetime {
            layout: PreparedNativeLayout {
                dtype: tensor.dtype,
                bytes: tensor_storage_bytes(tensor).ok_or(AxCompileError::MissingTensor(id))?,
            },
            ingress: inputs.contains(&id) || constant,
            pinned: outputs.contains(&id) || constant,
            available: true,
            remaining_uses: uses as usize,
            slot: None,
        });
    }
    let instructions = native_schedule_in(
        &nodes,
        capabilities,
        &mut lifetimes,
        fusions,
        allocator.clone(),
    )?;
    for (takes, lifetime) in takes.iter_mut().zip(&lifetimes) {
        *takes = u32::try_from(lifetime.remaining_uses)
            .map_err(|_| AxArtifactError::InvalidPartition)?;
    }
    for &output in &outputs {
        let slot = tensors
            .iter()
            .position(|&tensor| tensor == output)
            .ok_or(AxArtifactError::InvalidPartition)?;
        takes[slot] = takes[slot]
            .checked_add(1)
            .ok_or(AxArtifactError::InvalidPartition)?;
    }

    Ok(NativeCpuPartition {
        nodes,
        capabilities,
        instructions,
        fusions,
        tensors,
        takes,
        inputs,
        outputs,
    })
}

pub fn native_tensor_slot<A: Allocator>(
    tensors: &mut Vec<TensorId, A>,
    tensor: TensorId,
) -> Result<u32, AxCompileError> {
    if let Some(slot) = tensors.iter().position(|&candidate| candidate == tensor) {
        return u32::try_from(slot).map_err(|_| AxArtifactError::InvalidPartition.into());
    }
    let slot = u32::try_from(tensors.len()).map_err(|_| AxArtifactError::InvalidPartition)?;
    tensors.push(tensor);
    Ok(slot)
}

pub fn run_native_cpu_partition_in<M: Allocator, R: Allocator + Clone>(
    native: &NativeCpuPartition<M>,
    graph: &Graph<'_>,
    boundary_values: &[Option<RuntimeValue<'_, R>>],
    output_ids: &[TensorId],
    resources: &[NativeResourceBinding<'_>],
    copies: &mut AxHostCopyStats,
    allocator: R,
) -> Result<Vec<RuntimeValue<'static, R>, R>, AxCompileError> {
    let mut invocation = crate::program::NativeInvocation::from_partition(
        native,
        *graph,
        output_ids,
        resources,
        |id| {
            boundary_values
                .get(id.0 as usize)
                .and_then(Option::as_ref)
                .map(RuntimeValue::as_slice)
        },
        allocator.clone(),
    )?;
    let result = (|| {
        invocation.advance(usize::MAX)?;
        invocation.take_values_map_in(|_, value| {
            // Partition outputs are produced here, never ingress passthroughs.
            match value {
                RuntimeValue::Owned(bytes) => Ok(RuntimeValue::Owned(bytes)),
                RuntimeValue::Native(value) => Ok(RuntimeValue::Native(value)),
                RuntimeValue::Borrowed(_) => Err(AxArtifactError::InvalidPartition.into()),
            }
        })
    })();
    let observed = invocation.host_copy_stats();
    copies.native_ingress_bytes = copies
        .native_ingress_bytes
        .saturating_add(observed.native_ingress_bytes);
    copies.native_boundary_encode_bytes = copies
        .native_boundary_encode_bytes
        .saturating_add(observed.native_boundary_encode_bytes);
    result
}

pub fn ensure_native_value_in<'a, R: Allocator + Clone>(
    id: TensorId,
    value_slot: usize,
    graph: &Graph<'a>,
    boundary_values: &'a [Option<RuntimeValue<'_, R>>],
    values: &mut [Option<NativeSlot<'a, R>>],
    copies: &mut AxHostCopyStats,
    allocator: R,
) -> Result<(), AxCompileError> {
    let slot = values
        .get(value_slot)
        .ok_or(AxCompileError::MissingTensor(id))?;
    if slot.is_some() {
        return Ok(());
    }
    let tensor = graph
        .tensor(id)
        .map_err(|_| AxCompileError::MissingTensor(id))?;
    let source = if tensor.flags.contains(TensorFlags::CONSTANT) {
        tensor.constant()
    } else {
        boundary_values
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .map(RuntimeValue::as_slice)
            .ok_or(AxCompileError::MissingTensor(id))?
    };
    values[value_slot] = Some(crate::program::decode_source(
        tensor, source, copies, allocator,
    )?);
    Ok(())
}

pub fn encode_native_value_in<A: Allocator>(
    value: NativeValue<A>,
    allocator: A,
) -> Result<Vec<u8, A>, AxCompileError> {
    match value {
        NativeValue::U8(values) | NativeValue::Packed(_, values) => Ok(values),
        NativeValue::I32(values) => {
            let bytes = values
                .len()
                .checked_mul(size_of::<i32>())
                .ok_or(AllocationError::AllocationFailed)?;
            let mut encoded = Vec::new_in(allocator);
            reserve(&mut encoded, bytes)?;
            for value in values {
                encoded.extend_from_slice(&value.to_le_bytes());
            }
            Ok(encoded)
        }
        NativeValue::F32(values) => {
            let bytes = values
                .len()
                .checked_mul(4)
                .ok_or(AllocationError::AllocationFailed)?;
            let mut encoded = Vec::new_in(allocator);
            reserve(&mut encoded, bytes)?;
            for value in values {
                encoded.extend_from_slice(&value.to_le_bytes());
            }
            Ok(encoded)
        }
    }
}

/// Amortized, never exact — see `ax::builder::reserve`: a per-push exact
/// reservation reallocates on every push, and an allocator that serves
/// each large block from a fresh extent then maps the sum of every size
/// the vector ever had.
pub fn reserve<T, A: Allocator>(
    values: &mut Vec<T, A>,
    additional: usize,
) -> Result<(), AxCompileError> {
    values
        .try_reserve(additional)
        .map_err(|_| AllocationError::AllocationFailed.into())
}

pub fn copy_bytes_in<A: Allocator>(
    bytes: &[u8],
    allocator: A,
) -> Result<Vec<u8, A>, AxCompileError> {
    let mut copy = Vec::new_in(allocator);
    reserve(&mut copy, bytes.len())?;
    copy.extend_from_slice(bytes);
    Ok(copy)
}

fn copy_slice_in<T: Copy, A: Allocator>(
    values: &[T],
    allocator: A,
) -> Result<Vec<T, A>, AxCompileError> {
    let mut copy = Vec::new_in(allocator);
    reserve(&mut copy, values.len())?;
    copy.extend_from_slice(values);
    Ok(copy)
}

pub fn tensor_storage_bytes(tensor: TensorView<'_>) -> Option<usize> {
    let elements = tensor.dimensions().try_fold(1_usize, |total, dim| {
        (dim != ax::DYNAMIC_DIM)
            .then_some(())
            .and_then(|()| total.checked_mul(dim as usize))
    })?;
    if tensor.dtype.block_size() != 1 {
        let width = tensor.dimensions().last().unwrap_or(1) as usize;
        if !width.is_multiple_of(tensor.dtype.block_size()) {
            return None;
        }
    }
    (elements / tensor.dtype.block_size()).checked_mul(tensor.dtype.block_byte_size())
}
