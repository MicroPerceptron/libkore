//! Xe implementation of Axiom graph admission over the existing kernel family.
//!
//! This adapter consumes canonical `axnn` requests and produces bindings for
//! the possessed accelerator face. It owns no tensors, DMA routes or allocator.
//! A plan is metadata: only successful program loading establishes that the
//! current provider has a native variant. Costs are supplied by the runtime.

use ax::{CostModel, Graph, NodeFlags, NodeView, TensorView};
use axnn::lowering::{
    EpilogueActivation, KernelAccuracy, KernelParams, KernelPlan, KernelRequest, SemanticOp,
};
use kore_svc_accel::{
    client::{LaunchBuilder, LaunchExecution},
    contract::{AccessMode, LaunchShape, Workgroups},
    service::{AccelBinding, AccelBindings, AccelRefusal, MAX_ACCEL_BINDINGS},
};
use kore_xe_kernels::{
    CausalConv, DeltaRule, DeltaRuleGeometry, DenseProjection, KvGeometry, KvKernel, MoeWave,
    Nvfp4Projection, QwenDense, QwenProjection, QwenRms, QwenRope, RmsNorm, RoutedGeometry,
    RoutedKernel, XeKernelArtifact,
    xve::{F32Elements, F32Map},
};

use crate::accelerator::Error;
use crate::xe_moe;

/// At most three canonical inputs, one parameter port and one output port.
/// Unused tail entries stay absent from the submitted binding list.
pub const KERNEL_BUFFER_COUNT: usize = 3 + 2;

/// Checked semantic launch metadata stored alongside `BufferExecution` in a
/// runtime owner. Its fixed type permits replacing a completed graph node's
/// plan without a boxed closure or a new execution/retirement implementation.
/// Program, queue and buffer IDs remain scoped to the caller's possessed face.
#[derive(Clone, Copy, Debug)]
pub struct KernelLaunch {
    kernel: Kernel,
    queue: u64,
    program: u64,
    offsets: [u64; KERNEL_BUFFER_COUNT],
    timeout_ns: u64,
}

impl LaunchBuilder<KERNEL_BUFFER_COUNT> for KernelLaunch {
    fn build(&self, buffers: &[u64; KERNEL_BUFFER_COUNT]) -> Result<LaunchExecution, AccelRefusal> {
        let count = self.kernel.input_count;
        let inputs = core::array::from_fn::<_, 3, _>(|index| (buffers[index], self.offsets[index]));
        let bindings = self.kernel.bindings(
            &inputs[..count],
            (buffers[count], self.offsets[count]),
            (buffers[count + 1], self.offsets[count + 1]),
        );
        self.kernel
            .execution(
                self.queue,
                self.program,
                bindings.map_err(|_| AccelRefusal::InvalidArgument)?,
                self.timeout_ns,
            )
            .map_err(|_| AccelRefusal::InvalidArgument)
    }
}

/// Where one package storage slot takes its buffer. A package's slots are
/// its storage array in order, and the provider requires the final slot to be
/// the one written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Port {
    /// The request's input `n`.
    Input(u8),
    /// The eight-byte control block: tensor scale/element count and rows.
    Parameter,
    /// A read-only run of this many bytes of F32 ones that the executor
    /// provides: the multiplicative identity where a package takes a second
    /// operand the graph does not have (the Mamba group RMS's gate).
    Ones(u64),
    /// The single result.
    Output,
    /// The single result, which the kernel also reads: the executor fills
    /// it with the accumulate base before submission, and the package adds
    /// its projection onto it.
    Accumulator,
    /// Expert-cache import `slot` of a MoE wave's chunk set, `bytes` long:
    /// a resident region bound whole, read by page.
    Chunk { slot: u8, bytes: u64 },
    /// A MoE wave's region (`crate::xe_moe`), `bytes` from this launch's
    /// header on: read and written.
    Wave(u64),
    /// A recurrent package's state before the run, `bytes` long, followed
    /// by `parameter_bytes()` (token and successor-slot counts). Staged
    /// from the host together and read-only for the entire submission.
    StateSource(u64),
    /// A recurrent package's successor states, `bytes` long: one slot, or
    /// one per token of the run ([`Kernel::with_state_slots`]); written, and
    /// read back to the host beside the output.
    StateTarget(u64),
    /// A resident KV cache (`crate::xe_executor::Executor::with_resident_kv`),
    /// `bytes` long, bound whole in place: written by an append, read by an
    /// attention read. Nothing of it reaches the host.
    KvCache { bytes: u64, written: bool },
    /// An attention read's split partials, `bytes` of fresh scratch the
    /// kernel writes and its reduction reads; never read back.
    Partials(u64),
    /// A result the package writes that nothing reads (a top-1's selected
    /// values beside its indices): `bytes` of fresh scratch, never read
    /// back. Only sequences that place every binding themselves (a frozen
    /// decode) bind it.
    Discard(u64),
}

/// Most storage slots one package binds (the provider's binding ceiling).
pub const MAX_PORTS: usize = MAX_ACCEL_BINDINGS;
/// The widest control block a package reads: the attention read's six
/// words, or two shape words followed by the bounded candidate-parent
/// table of a tree-shaped recurrent package, whichever is longer.
const CONTROL_BYTES: usize = {
    const ATTENTION_READ: usize = 24;
    const TREE: usize = 8 + 4 * axnn::lowering::TokenTree::MAX_NODES;
    if ATTENTION_READ > TREE { ATTENTION_READ } else { TREE }
};
/// Cache slots per append workgroup: the four whose scale bytes share a
/// dword, aligned, so no two workgroups write one dword.
const KV_SLOT_GROUP: u32 = 4;

/// Bytes of the F32 ones run a group RMS gate needs for `tokens` rows.
const GROUP_RMS_WIDTH: usize = 8 * 512;
/// The checkpoint epsilon the captured RMS packages fold in.
const RMS_EPSILON: f32 = 1e-5;
/// The captured BF16 embedding gather's vocabulary and width.
const GATHER_ROWS: usize = 131_072;
const HIDDEN: usize = 2688;
// Qwen geometry (res/QWEN_XE_CATALOG.md), the second family the Xe graph
// dispatches beside Nemotron's 2688-wide shapes: Qwen3-32B and Qwen3.8-27B
// share the 5120-wide hidden state, Qwen3.6-35B-A3B's is 2048, all at
// epsilon 1e-6; their heads, norms and rotations are the catalog's
// (`QWEN_HEADS`, `QwenRms`, `QwenRope`).
/// The most rows of a Qwen verification (or prompt chunk): the multi-row
/// packages' baked M, which take a pass's own count from the parameter block.
const QWEN_ROWS: usize = kore_xe_kernels::QWEN_ROWS as usize;
const QWEN_RMS_EPSILON: f32 = 1e-6;
/// The captured recurrent geometries: convolutions of four taps
/// (xe-nemotron-conv.comp, at each width `CausalConv::CHANNELS` lists), and
/// the Mamba scan's 64 heads of 64 over eight groups of 128 state elements
/// (xe-nemotron-scan.comp); 256 invocations per workgroup.
const CONV_TAPS: u32 = 4;
const MAMBA_HEADS: u32 = 64;
const MAMBA_HEAD_DIM: u32 = 64;
const MAMBA_GROUPS: u32 = 8;
const MAMBA_STATE_SIZE: u32 = 128;
const RECURRENT_LANES: u32 = 256;

/// The recurrence a recurrent package steps: enough to swap in its
/// candidate-tree package, and for the delta rule the geometry that
/// package was captured at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Recurrence {
    /// A convolution over this many channels.
    Conv(u32),
    Scan,
    Delta(DeltaRuleGeometry),
}

/// One graph node lowered to a checked package and its exact port contract.
/// No model-specific node numbers or tensor names participate in selection.
#[derive(Clone, Copy, Debug)]
pub struct Kernel {
    artifact: XeKernelArtifact,
    launch: LaunchShape,
    /// The control block: word 0 the tensor scale or element count, word
    /// 1 the rows M a runtime-M projection package reads (zero otherwise).
    /// Recurrent packages use both words for tokens and successor slots.
    /// The resident KV kernels read `parameter_len` bytes of it: the
    /// append five words, the attention read six.
    parameter: [u8; CONTROL_BYTES],
    parameter_len: usize,
    recurrent: Option<Recurrence>,
    input_bytes: [u64; 3],
    input_count: usize,
    output_bytes: u64,
    work: u64,
    ports: [Port; MAX_PORTS],
    port_count: usize,
    /// Inputs (bit per index) the package reads as 32-bit words, whose views
    /// must be 4-byte aligned; a misaligned one stays on the CPU.
    word_inputs: u8,
    /// A dense E4M3/BF16 projection (fixed or runtime-M package), which an
    /// executor may decline wholesale after a failed self-test.
    dense: bool,
    /// A routed gate/up or down projection over an expert table, which
    /// [`Self::with_pool`] can move onto a slot pool.
    routed: Option<(RoutedKernel, RoutedGeometry)>,
}

impl Kernel {
    /// Reject resource effects, broadcasting, unsupported shapes and stricter
    /// fidelity policies before any publication. XMX projections use FP16
    /// operands with F32 accumulation; F32 vector projections also change
    /// reduction order. Storage dtype alone is not an accuracy guarantee.
    pub fn from_node(
        graph: &Graph<'_>,
        node: NodeView<'_>,
        minimum: KernelAccuracy,
    ) -> Option<Self> {
        if node.flags != NodeFlags::default() || minimum != KernelAccuracy::Approximate {
            return None;
        }
        Self::select(&ax::node_kernel_request(graph, node)?, minimum)
    }

    /// The same admission for an already lowered semantic request. Rebuild its
    /// canonical form because request compatibility fields are public metadata;
    /// changing them must not bypass dtype, shape, effect or epilogue checks.
    pub fn select(request: &KernelRequest, minimum: KernelAccuracy) -> Option<Self> {
        if minimum != KernelAccuracy::Approximate {
            return None;
        }
        let canonical = Self::canonical(
            request.semantic,
            request.inputs(),
            request.result_tensor(0),
            request.params(),
        )?;
        if *request != canonical {
            return None;
        }
        Self::from_request(&canonical)
    }

    /// Admit the semantic contract carried by a prepared CPU plan. The CPU's
    /// chosen implementation and cost do not determine GPU fidelity: the
    /// caller's minimum policy still governs the replacement. Recheck public
    /// compatibility fields against the private typed descriptors before DMA.
    pub fn from_plan<I>(plan: &KernelPlan<I>, minimum: KernelAccuracy) -> Option<Self> {
        if minimum != KernelAccuracy::Approximate {
            return None;
        }
        let request = Self::canonical(
            plan.semantic,
            plan.inputs(),
            plan.result_tensor(0),
            plan.params(),
        )?;
        if plan.input_dtypes != request.input_dtypes
            || plan.input_shapes != request.input_shapes
            || plan.output_dtype != request.output_dtype
            || plan.shape != request.shape
            || plan.work_items != request.work_items()
            || plan.inputs() != request.inputs()
            || plan.result() != request.result()
            || plan.params() != request.params()
            || plan.effects() != request.effects()
        {
            return None;
        }
        Self::from_request(&request)
    }

    /// Construct the same checked semantic admission from a captured family
    /// before loading it. A package whose epilogue is not representable by this
    /// executor is unavailable, even if its native bytes exist in the catalog.
    pub fn nvfp4_projection(
        kind: Nvfp4Projection,
        scale: f32,
        minimum: KernelAccuracy,
    ) -> Option<Self> {
        let &(m, n, k, activation, _) = PROJECTIONS.iter().find(|entry| entry.4 == kind)?;
        let desc = |shape, dtype| axnn::TensorDesc::try_new(shape, dtype);
        let input = desc(axnn::Shape::from([m, k]), axnn::DType::F32)?;
        let weights = desc(axnn::Shape::from([n, k / 2]), axnn::DType::U8)?;
        let scales = desc(axnn::Shape::from([n, k / 16]), axnn::DType::B8)?;
        let output = desc(axnn::Shape::from([m, n]), axnn::DType::F32)?;
        let request = KernelRequest::matmul_nvfp4_transposed(
            &input, &weights, &scales, &output, scale, activation,
        )?;
        Self::select(&request, minimum)
    }

    fn canonical(
        semantic: SemanticOp,
        inputs: &[Option<axnn::TensorDesc>; 3],
        output: Option<&axnn::TensorDesc>,
        params: KernelParams,
    ) -> Option<KernelRequest> {
        match semantic {
            SemanticOp::MatMulNvfp4Transposed => {
                let KernelParams::MatmulEpilogue { scale, activation } = params else {
                    return None;
                };
                KernelRequest::matmul_nvfp4_transposed(
                    inputs[0].as_ref()?,
                    inputs[1].as_ref()?,
                    inputs[2].as_ref()?,
                    output?,
                    scale.to_f32(),
                    activation,
                )
            }
            SemanticOp::MatMulTransposed => {
                let KernelParams::MatmulEpilogue { scale, activation } = params else {
                    return None;
                };
                KernelRequest::matmul_transposed_activated(
                    inputs[0].as_ref()?,
                    inputs[1].as_ref()?,
                    output?,
                    scale.to_f32(),
                    activation,
                )
            }
            SemanticOp::Binary(op) => {
                KernelRequest::binary(op, inputs[0].as_ref()?, inputs[1].as_ref()?)
            }
            SemanticOp::Unary(op) => Some(KernelRequest::unary(op, inputs[0].as_ref()?)),
            SemanticOp::RmsNorm => {
                let KernelParams::Normalization { epsilon, axis } = params else {
                    return None;
                };
                KernelRequest::rms_norm_over(
                    inputs[0].as_ref()?,
                    inputs[1].as_ref()?,
                    epsilon.to_f32(),
                    isize::try_from(axis).ok()?,
                )
            }
            SemanticOp::Gather => KernelRequest::gather(inputs[0].as_ref()?, inputs[1].as_ref()?),
            // Without this arm every ROPE node was refused before
            // `from_request`'s Qwen RoPE arm could see it.
            SemanticOp::Rope(mode) => {
                let KernelParams::Rope {
                    mode: kind,
                    rotary_dims,
                } = params
                else {
                    return None;
                };
                if kind != mode {
                    return None;
                }
                KernelRequest::rope_partial(
                    inputs[0].as_ref()?,
                    inputs[1].as_ref()?,
                    inputs[2].as_ref()?,
                    mode,
                    rotary_dims,
                )
            }
            SemanticOp::Ternary(op) => KernelRequest::ternary(
                op,
                inputs[0].as_ref()?,
                inputs[1].as_ref()?,
                inputs[2].as_ref()?,
            ),
            SemanticOp::CausalConv1d => {
                let KernelParams::CausalConv1d(conv) = params else {
                    return None;
                };
                KernelRequest::causal_conv1d(
                    inputs[0].as_ref()?,
                    inputs[1].as_ref()?,
                    inputs[2].as_ref()?,
                    conv,
                )
            }
            SemanticOp::SsmScan => {
                let KernelParams::SsmScan(scan) = params else {
                    return None;
                };
                KernelRequest::ssm_scan(
                    inputs[0].as_ref()?,
                    inputs[1].as_ref()?,
                    inputs[2].as_ref()?,
                    scan,
                )
            }
            SemanticOp::GatedDeltaRule => {
                let KernelParams::GatedDeltaRule(rule) = params else {
                    return None;
                };
                KernelRequest::gated_delta_rule(
                    inputs[0].as_ref()?,
                    inputs[1].as_ref()?,
                    inputs[2].as_ref()?,
                    rule,
                )
            }
            SemanticOp::MoeRoute => {
                let KernelParams::MoeRoute(route) = params else {
                    return None;
                };
                KernelRequest::moe_route(inputs[0].as_ref()?, route)
            }
            SemanticOp::MoeGateUpNvfp4 | SemanticOp::MoeDownNvfp4 => {
                let KernelParams::RoutedExperts(routed) = params else {
                    return None;
                };
                let request = if semantic == SemanticOp::MoeGateUpNvfp4 {
                    KernelRequest::moe_gate_up_nvfp4
                } else {
                    KernelRequest::moe_down_nvfp4
                };
                request(
                    inputs[0].as_ref()?,
                    inputs[1].as_ref()?,
                    inputs[2].as_ref()?,
                    routed,
                )
            }
            _ => None,
        }
    }

    fn from_request(request: &KernelRequest) -> Option<Self> {
        let output = request.result_tensor(0)?;
        if request.result_count() != 1 || !output.is_contiguous() || output.layout.offset() != 0 {
            return None;
        }
        let mut input_bytes = [0; 3];
        let mut input_count = 0;
        for (index, input) in request.inputs().iter().enumerate() {
            if let Some(input) = input {
                if !input.is_contiguous() || input.layout.offset() != 0 {
                    return None;
                }
                input_bytes[index] = input.checked_storage_bytes()? as u64;
                input_count += 1;
            }
        }
        let mut word_inputs = 0;
        let mut pool_form = None;
        let legacy = |count: usize| {
            let mut ports = [Port::Output; MAX_PORTS];
            for (index, port) in ports[..count].iter_mut().enumerate() {
                *port = Port::Input(index as u8);
            }
            ports[count] = Port::Parameter;
            (ports, count + 2)
        };
        let (artifact, launch, parameter, (ports, port_count)) = match request.semantic {
            SemanticOp::MatMulNvfp4Transposed => {
                let KernelParams::MatmulEpilogue { scale, activation } = request.params() else {
                    return None;
                };
                let shape = request.input(0)?.shape;
                let (m, n, k) = (shape.dim(0), output.shape.dim(1), shape.dim(1));
                let (artifact, group_width, columns, tile_rows) = if let Some(kind) = projection(m, n, k, activation) {
                    (kore_xe_kernels::nvfp4_projection(kind), kind.group_width(), 16, 8)
                } else if (1..=QWEN_ROWS).contains(&m) {
                    // A decode row on the vector engines, or a
                    // verification's (prompt chunk's) rows on XMX, whose
                    // package takes the count from the parameter block. More
                    // rows, or a shape with no captured package, is
                    // declined: the executor refuses it (member_declined)
                    // and, under strict placement, fails the invocation — a
                    // loud CPU fallback, not a silent one.
                    let kind = qwen_projection_of(n, k, activation)?;
                    if m == 1 {
                        (
                            kore_xe_kernels::qwen_projection(kind)
                                .expect("the Qwen decode projections are captured"),
                            kind.group_width(),
                            16,
                            8,
                        )
                    } else {
                        (
                            kore_xe_kernels::qwen_projection_rows(kind)?,
                            kind.rows_group_width(),
                            kind.rows_group_columns(),
                            8,
                        )
                    }
                } else if m == kore_xe_kernels::QWEN_PREFILL16_ROWS as usize {
                    let kind = qwen_projection_of(n, k, activation)?;
                    (
                        kore_xe_kernels::qwen_projection_prefill16(kind)?,
                        kind.prefill16_group_width(),
                        kind.prefill16_group_columns(),
                        16,
                    )
                } else {
                    return None;
                };
                // Packed words may read the final partial word. Require the
                // caller's imported view to cover that word, not just its
                // logical byte tensor (e.g. the 17-row witness's scales).
                for bytes in &mut input_bytes[1..3] {
                    *bytes = bytes.checked_add(3)? & !3;
                }
                // The selected package fixes its row tile: legacy and
                // Nemotron runtime-M packages process eight rows per Y
                // group; only the dedicated Qwen M16 package processes 16.
                let groups = projection_groups(m, n, k, group_width, columns, tile_rows)?;
                (
                    artifact,
                    LaunchShape::Workgroups(groups),
                    projection_parameter(scale.to_f32(), m),
                    legacy(input_count),
                )
            }
            SemanticOp::MatMulTransposed => {
                let KernelParams::MatmulEpilogue { scale, activation } = request.params() else {
                    return None;
                };
                let (input, weights) = (request.input(0)?, request.input(1)?);
                if input.dtype != axnn::DType::F32 || output.dtype != axnn::DType::F32 {
                    return None;
                }
                let (m, n, k) = (input.shape.dim(0), output.shape.dim(1), input.shape.dim(1));
                if weights.dtype == axnn::DType::F32 {
                    use kore_xe_kernels::xve::{F32Projection, F32ProjectionShape};
                    let epilogue = match activation {
                        EpilogueActivation::None => F32Projection::Linear,
                        EpilogueActivation::Silu => F32Projection::Silu,
                        EpilogueActivation::Sigmoid => F32Projection::Sigmoid,
                    };
                    let shape = F32ProjectionShape::new(
                        u32::try_from(m).ok()?,
                        u32::try_from(n).ok()?,
                        u32::try_from(k).ok()?,
                    )?;
                    let [activation_bytes, weight_bytes, result_bytes] = shape.data_bytes();
                    if input_bytes != [activation_bytes, weight_bytes, 0]
                        || output.checked_storage_bytes()? as u64 != result_bytes
                    {
                        return None;
                    }
                    word_inputs = 0b11;
                    (
                        kore_xe_kernels::xve::f32_projection(epilogue),
                        shape.launch().ok()?,
                        control(shape.parameter_bytes(scale.to_f32())?),
                        legacy(input_count),
                    )
                } else {
                    let (vocab, hidden) = (u32::try_from(n).ok()?, u32::try_from(k).ok()?);
                    let head = (weights.dtype, activation) == (axnn::DType::B16, EpilogueActivation::None)
                        && kore_xe_kernels::QWEN_HEADS.contains(&(vocab, hidden));
                    let small = QwenDense::ALL.into_iter().find(|kind| {
                        weights.dtype == axnn::DType::B16
                            && (kind.n(), kind.k(), kind.epilogue()) == (vocab, hidden, epilogue_code(activation))
                    });
                    let head = match m {
                        1 if head => kore_xe_kernels::qwen_lm_head(vocab, hidden)
                            .map(|head| (head, kore_xe_kernels::qwen_lm_head_lanes(hidden), 16)),
                        2..=QWEN_ROWS if head => kore_xe_kernels::qwen_lm_head_rows(vocab, hidden).map(|head| {
                            let (lanes, columns) = kore_xe_kernels::qwen_lm_head_rows_geometry(hidden);
                            (head, lanes, columns)
                        }),
                        16 if head => kore_xe_kernels::qwen_lm_head_prefill16(vocab, hidden).map(|head| {
                            let (lanes, columns) = kore_xe_kernels::qwen_lm_head_prefill16_geometry(hidden);
                            (head, lanes, columns)
                        }),
                        // The 35B-A3B's router and gate inputs, BF16 as
                        // published: sixteen columns a 128-lane workgroup at
                        // one row or a verification's, and 32 lanes at M16.
                        1 => small.and_then(|kind| kore_xe_kernels::qwen_dense(kind).map(|p| (p, kind.group_width(), 16))),
                        2..=QWEN_ROWS => {
                            small.and_then(|kind| kore_xe_kernels::qwen_dense_rows(kind).map(|p| (p, kind.group_width(), 16)))
                        }
                        16 => small.and_then(|kind| {
                            kore_xe_kernels::qwen_dense_prefill16(kind)
                                .map(|p| (p, kind.prefill16_group_width(), 16))
                        }),
                        _ => None,
                    };
                    if let Some((head, lanes, columns)) = head {
                        // An untied dense lm_head [vocab, hidden] BF16 of a
                        // captured head: one row on the vector engines, or
                        // a verification's on XMX (the row count in the
                        // parameter block); or a small dense projection
                        // the same way.
                        word_inputs = 1 << 1;
                        let groups = Workgroups::new(
                            [
                                u32::try_from(n.div_ceil(columns as usize)).ok()?,
                                u32::try_from(m.div_ceil(if m == 16 { 16 } else { 8 })).ok()?,
                                1,
                            ],
                            [lanes, 1, 1],
                        )
                        .ok()?;
                        (
                            head,
                            LaunchShape::Workgroups(groups),
                            projection_parameter(scale.to_f32(), m),
                            legacy(input_count),
                        )
                    } else {
                        let (kind, wide) = dense_projection(m, n, k, weights.dtype, activation)?;
                        word_inputs = 1 << 1;
                        // xe-dense-projection.comp: one 8x16 tile per workgroup,
                        // K split across the package's teams.
                        let groups = Workgroups::new(
                            [
                                u32::try_from(n.div_ceil(16)).ok()?,
                                u32::try_from(m.div_ceil(8)).ok()?,
                                1,
                            ],
                            [kind.group_width(), 1, 1],
                        )
                        .ok()?;
                        (
                            if wide {
                                kore_xe_kernels::dense_projection_wide(kind)?
                            } else {
                                kore_xe_kernels::dense_projection(kind)?
                            },
                            LaunchShape::Workgroups(groups),
                            projection_parameter(scale.to_f32(), m),
                            legacy(input_count),
                        )
                    }
                }
            }
            SemanticOp::RmsNorm => {
                let KernelParams::Normalization { epsilon, axis } = request.params() else {
                    return None;
                };
                let (input, weight) = (request.input(0)?, request.input(1)?);
                let epsilon_bits = epsilon.to_f32().to_bits();
                if (epsilon_bits != RMS_EPSILON.to_bits() && epsilon_bits != QWEN_RMS_EPSILON.to_bits())
                    || input.dtype != axnn::DType::F32
                    || output.dtype != axnn::DType::F32
                {
                    return None;
                }
                let dims = input.shape.dims();
                let rows = u32::try_from(dims[0]).ok()?;
                let mut ports = [Port::Output; MAX_PORTS];
                // rms-op0: one workgroup per row, x/weight/out. The Qwen layer
                // norm is 5120-wide (one workgroup per token); a per-head norm
                // one row per (token, head) at a captured width (128: Qwen3's
                // QK-norm and Qwen3.5's delta-rule head norm; 256: Qwen3.5's
                // gated QK-norm). The Nemotron captures fold in epsilon 1e-5,
                // the Qwen ones 1e-6; an epsilon must not cross into the other
                // family's width (epsilon is baked, not runtime).
                let layered = epsilon_bits == RMS_EPSILON.to_bits();
                let (artifact, groups, count) = match (dims, axis, weight.shape.dims()) {
                    ([_, HIDDEN], 1, [HIDDEN]) if layered => {
                        ports[..2].copy_from_slice(&[Port::Input(0), Port::Input(1)]);
                        (kore_xe_kernels::rms_norm(RmsNorm::Hidden), rows, 3)
                    }
                    ([tokens, 8, 512], 2, [8, 512]) if layered => {
                        let gate = (tokens * GROUP_RMS_WIDTH * size_of::<f32>()) as u64;
                        ports[..3].copy_from_slice(&[
                            Port::Input(0),
                            Port::Ones(gate),
                            Port::Input(1),
                        ]);
                        (
                            kore_xe_kernels::rms_norm(RmsNorm::MambaGateGroup),
                            rows.checked_mul(8)?,
                            4,
                        )
                    }
                    ([_, width], 1, [weight])
                        if !layered
                            && width == weight
                            && QwenRms::of_width(u32::try_from(*width).ok()?).is_some_and(QwenRms::is_layer) =>
                    {
                        let kind = QwenRms::of_width(u32::try_from(*width).ok()?)?;
                        ports[..2].copy_from_slice(&[Port::Input(0), Port::Input(1)]);
                        (kore_xe_kernels::qwen_rms(kind)?, rows, 3)
                    }
                    ([_, heads, width], 2, [weight]) if !layered && width == weight => {
                        let kind = QwenRms::of_width(u32::try_from(*width).ok()?)?;
                        ports[..2].copy_from_slice(&[Port::Input(0), Port::Input(1)]);
                        let heads = u32::try_from(*heads).ok()?;
                        (kore_xe_kernels::qwen_rms(kind)?, rows.checked_mul(heads)?, 3)
                    }
                    _ => return None,
                };
                (
                    artifact,
                    LaunchShape::Workgroups(Workgroups::new([groups, 1, 1], [256, 1, 1]).ok()?),
                    [0; CONTROL_BYTES],
                    (ports, count),
                )
            }
            SemanticOp::Gather => {
                let (table, indices) = (request.input(0)?, request.input(1)?);
                if indices.dtype != axnn::DType::I32
                    || indices.shape.rank() != 1
                    || request.params() != KernelParams::None
                {
                    return None;
                }
                // BF16 embeddings widen to F32 rows; an F32 table (Qwen3.5's
                // q/gate split) is copied row for row.
                let artifact = match (table.dtype, table.shape.dims()) {
                    (axnn::DType::B16, [GATHER_ROWS, HIDDEN]) => kore_xe_kernels::embedding_gather_bf16(),
                    (axnn::DType::B16, [vocab, hidden]) => kore_xe_kernels::qwen_gather(
                        u32::try_from(*vocab).ok()?,
                        u32::try_from(*hidden).ok()?,
                    )?,
                    (axnn::DType::F32, [rows, width]) if output.dtype == axnn::DType::F32 => {
                        let (rows, width) = (u32::try_from(*rows).ok()?, u32::try_from(*width).ok()?);
                        kore_xe_kernels::f32_row_gather(rows, width)
                            .or_else(|| kore_xe_kernels::f32_row_gather_prefill16(rows, width))?
                    }
                    _ => return None,
                };
                let rows = u32::try_from(indices.shape.dim(0)).ok()?;
                let mut ports = [Port::Output; MAX_PORTS];
                ports[..2].copy_from_slice(&[Port::Input(0), Port::Input(1)]);
                (
                    artifact,
                    LaunchShape::Workgroups(Workgroups::new([rows, 1, 1], [256, 1, 1]).ok()?),
                    [0; CONTROL_BYTES],
                    (ports, 3),
                )
            }
            SemanticOp::Rope(mode) => {
                let KernelParams::Rope {
                    mode: kind,
                    rotary_dims,
                } = request.params()
                else {
                    return None;
                };
                // A captured rotation (heads, head width, rotated columns):
                // Qwen3's whole 128-wide heads, Qwen3.5's leading 64 of 256.
                if mode != kind || kind != axnn::RopeMode::NeoX {
                    return None;
                }
                let x = request.input(0)?;
                if x.dtype != axnn::DType::F32 || output.dtype != axnn::DType::F32 {
                    return None;
                }
                let dims = x.shape.dims();
                if dims.len() != 3 {
                    return None;
                }
                let rope = QwenRope::of(
                    u32::try_from(dims[1]).ok()?,
                    u32::try_from(dims[2]).ok()?,
                    rotary_dims,
                )?;
                let tokens = u32::try_from(dims[0]).ok()?;
                let groups = tokens.checked_mul(u32::try_from(dims[1]).ok()?)?;
                let mut ports = [Port::Output; MAX_PORTS];
                ports[..3].copy_from_slice(&[Port::Input(0), Port::Input(1), Port::Input(2)]);
                (
                    kore_xe_kernels::qwen_rope(rope)?,
                    LaunchShape::Workgroups(Workgroups::new([groups, 1, 1], [64, 1, 1]).ok()?),
                    [0; CONTROL_BYTES],
                    (ports, 4),
                )
            }
            SemanticOp::CausalConv1d => {
                let KernelParams::CausalConv1d(conv) = request.params() else {
                    return None;
                };
                let input = request.input(0)?;
                // xe-nemotron-conv.comp at a captured width: four taps with
                // the SiLU folded in; one invocation per channel through the
                // tokens, carried in the source-state control trailer.
                if conv.kernel != CONV_TAPS
                    || conv.activation != EpilogueActivation::Silu
                    || input.dtype != axnn::DType::F32
                    || output.dtype != axnn::DType::F32
                {
                    return None;
                }
                let tokens = u32::try_from(input.shape.dim(0)).ok()?;
                let state = (conv.state_elements()? as u64).checked_mul(4)?;
                (
                    kore_xe_kernels::causal_conv(CausalConv::Runtime, conv.channels)?,
                    recurrent_launch(conv.channels as usize, tokens)?,
                    recurrent_parameter(tokens),
                    recurrent_ports(state),
                )
            }
            SemanticOp::SsmScan => {
                let KernelParams::SsmScan(scan) = request.params() else {
                    return None;
                };
                let xbc = request.input(0)?;
                // xe-nemotron-scan.comp: 64 heads of 64 over eight groups of
                // 128, dt limited only by the thresholded softplus; one
                // invocation per (head, channel) state row through the
                // tokens, carried in the source-state control trailer.
                if (scan.heads, scan.head_dim, scan.groups, scan.state_size)
                    != (MAMBA_HEADS, MAMBA_HEAD_DIM, MAMBA_GROUPS, MAMBA_STATE_SIZE)
                    || scan.dt_min.to_f32() != 0.0
                    || scan.dt_max.to_f32() != f32::INFINITY
                    || xbc.dtype != axnn::DType::F32
                    || output.dtype != axnn::DType::F32
                {
                    return None;
                }
                let tokens = u32::try_from(xbc.shape.dim(0)).ok()?;
                let state = (scan.state_elements()? as u64).checked_mul(4)?;
                (
                    kore_xe_kernels::mamba_scan_runtime()?,
                    recurrent_launch((MAMBA_HEADS * MAMBA_HEAD_DIM) as usize, tokens)?,
                    recurrent_parameter(tokens),
                    recurrent_ports(state),
                )
            }
            SemanticOp::GatedDeltaRule => {
                let KernelParams::GatedDeltaRule(rule) = request.params() else {
                    return None;
                };
                let qkv = request.input(0)?;
                // xe-delta-rule.comp: the captured geometry, the query and key
                // L2-normalized in the kernel; one workgroup per value head,
                // tokens and slots in the source-state control trailer.
                if !rule.qk_l2norm
                    || qkv.dtype != axnn::DType::F32
                    || output.dtype != axnn::DType::F32
                {
                    return None;
                }
                let geometry = delta_geometry(rule);
                let tokens = u32::try_from(qkv.shape.dim(0)).ok()?;
                let state = (rule.state_elements()? as u64).checked_mul(4)?;
                (
                    kore_xe_kernels::delta_rule(DeltaRule::Runtime, geometry)?,
                    delta_launch(geometry, DeltaRule::Runtime)?,
                    recurrent_parameter(tokens),
                    recurrent_ports(state),
                )
            }
            SemanticOp::MoeRoute => {
                let KernelParams::MoeRoute(route) = request.params() else {
                    return None;
                };
                // xe-moe-route.comp at a captured routing: one workgroup a
                // token, the routes written for the routed projections.
                let geometry = RoutedGeometry::ALL.into_iter().find(|geometry| {
                    (geometry.experts, geometry.top_k, geometry.shared)
                        == (route.experts, route.top_k, route.shared)
                })?;
                let tokens = u32::try_from(request.input(0)?.shape.dim(0)).ok()?;
                let mut ports = [Port::Output; MAX_PORTS];
                ports[0] = Port::Input(0);
                (
                    kore_xe_kernels::routed_kernel(RoutedKernel::Route, geometry)?,
                    routed_launch(RoutedKernel::Route, geometry, tokens)?,
                    [0; CONTROL_BYTES],
                    (ports, 2),
                )
            }
            SemanticOp::MoeGateUpNvfp4 | SemanticOp::MoeDownNvfp4 => {
                let KernelParams::RoutedExperts(routed) = request.params() else {
                    return None;
                };
                // xe-nvfp4-routed.comp at a captured geometry: the
                // activations, the expert table (read as words: code pairs,
                // block scales and tensor scales), the routes, the output.
                let geometry = RoutedGeometry::ALL.into_iter().find(|geometry| {
                    (geometry.records(), geometry.slots(), geometry.hidden, geometry.inner)
                        == (routed.experts, routed.slots, routed.hidden, routed.intermediate)
                })?;
                let kind = if request.semantic == SemanticOp::MoeGateUpNvfp4 {
                    RoutedKernel::GateUp
                } else {
                    RoutedKernel::Down
                };
                let tokens = u32::try_from(request.input(0)?.shape.dim(0)).ok()?;
                word_inputs = 1 << 1;
                pool_form = Some((kind, geometry));
                let mut ports = [Port::Output; MAX_PORTS];
                ports[..3].copy_from_slice(&[Port::Input(0), Port::Input(1), Port::Input(2)]);
                (
                    kore_xe_kernels::routed_kernel(kind, geometry)?,
                    routed_launch(kind, geometry, tokens)?,
                    [0; CONTROL_BYTES],
                    (ports, 4),
                )
            }
            semantic => {
                let op = match semantic {
                    SemanticOp::Binary(axnn::BinaryOp::Add) => F32Map::Add,
                    SemanticOp::Binary(axnn::BinaryOp::Mul) => F32Map::Multiply,
                    SemanticOp::Unary(axnn::UnaryOp::Silu) => F32Map::Silu,
                    SemanticOp::Unary(axnn::UnaryOp::Softplus) => F32Map::Softplus,
                    SemanticOp::Unary(axnn::UnaryOp::Sigmoid) => F32Map::Sigmoid,
                    SemanticOp::Ternary(axnn::TernaryOp::Fma) => F32Map::Affine,
                    _ => return None,
                };
                if output.dtype != axnn::DType::F32
                    || request.params() != KernelParams::None
                    || request
                        .inputs()
                        .iter()
                        .flatten()
                        .any(|input| input.dtype != output.dtype || input.shape != output.shape)
                {
                    return None;
                }
                let count = F32Elements::new(u32::try_from(request.elements()).ok()?)?;
                (
                    kore_xe_kernels::xve::f32_map(op),
                    count.launch().ok()?,
                    word_parameter(count.parameter_bytes()),
                    legacy(input_count),
                )
            }
        };
        Some(Self {
            artifact,
            launch,
            parameter,
            parameter_len: 8,
            recurrent: match request.params() {
                KernelParams::CausalConv1d(conv) => Some(Recurrence::Conv(conv.channels)),
                KernelParams::SsmScan(_) => Some(Recurrence::Scan),
                KernelParams::GatedDeltaRule(rule) => Some(Recurrence::Delta(delta_geometry(rule))),
                _ => None,
            },
            input_bytes,
            input_count,
            output_bytes: output.checked_storage_bytes()? as u64,
            work: request.work_items(),
            ports,
            port_count,
            word_inputs,
            dense: request.semantic == SemanticOp::MatMulTransposed,
            routed: pool_form,
        })
    }

    /// Whether every input view the package reads as 32-bit words is
    /// 4-byte aligned (`inputs` in slot order, absent ones skipped).
    pub fn inputs_aligned<'b>(&self, inputs: impl IntoIterator<Item = Option<&'b [u8]>>) -> bool {
        inputs.into_iter().enumerate().all(|(index, input)| {
            self.word_inputs & (1 << index) == 0
                || input.is_none_or(|bytes| (bytes.as_ptr() as usize).is_multiple_of(4))
        })
    }

    /// The package's storage slots, in slot order.
    pub fn ports(&self) -> &[Port] {
        &self.ports[..self.port_count]
    }

    /// Whether the slots are the inputs, the parameter, then the output: the
    /// only layout the per-port single-job path (`launch_builder`) binds.
    pub fn legacy_layout(&self) -> bool {
        self.port_count == self.input_count + 2
            && self.ports[self.input_count] == Port::Parameter
            && self.ports[self.input_count + 1] == Port::Output
            && self.ports[..self.input_count]
                .iter()
                .enumerate()
                .all(|(index, port)| *port == Port::Input(index as u8))
    }

    /// The bytes one port binds.
    pub fn port_bytes(&self, port: Port) -> u64 {
        match port {
            Port::Input(index) => self.input_bytes[usize::from(index)],
            Port::Parameter => self.parameter_len as u64,
            // StateSource's payload is the captured fixed geometry; the
            // control trailer shares its retained read-only allocation.
            Port::StateSource(bytes) => bytes + self.parameter_len as u64,
            Port::Ones(bytes) => bytes,
            Port::Output | Port::Accumulator => self.output_bytes,
            Port::Chunk { bytes, .. }
            | Port::Wave(bytes)
            | Port::StateTarget(bytes)
            | Port::KvCache { bytes, .. }
            | Port::Partials(bytes)
            | Port::Discard(bytes) => bytes,
        }
    }

    /// A recurrent package's state bytes: the state before the run, then
    /// its successor states (every slot). `None` for any other package.
    pub fn state_bytes(&self) -> Option<(u64, u64)> {
        let (mut source, mut target) = (None, None);
        for port in self.ports() {
            match *port {
                Port::StateSource(bytes) => source = Some(bytes),
                Port::StateTarget(bytes) => target = Some(bytes),
                _ => {}
            }
        }
        Some((source?, target?))
    }

    /// Whether this is a recurrent package (a Mamba convolution or scan, or a
    /// gated delta rule), which carries a state pair beside its operands.
    pub fn is_recurrent(&self) -> bool {
        self.state_bytes().is_some()
    }

    /// This recurrent package with its target holding `slots` successor
    /// states, which the kernel reads from its control trailer: one
    /// (updated in place, a prompt run's) or one per token of the run (a
    /// verification's, for its commit). Any other count, or a package
    /// without a state pair, is declined.
    pub fn with_state_slots(mut self, slots: u32) -> Option<Self> {
        let (source, _) = self.state_bytes()?;
        let tokens = u32::from_le_bytes(self.parameter[..4].try_into().ok()?);
        if (slots != 1 && slots != tokens) || (self.parameter_len > 8 && slots != tokens) {
            return None;
        }
        self.parameter[4..8].copy_from_slice(&slots.to_le_bytes());
        let target = source.checked_mul(u64::from(slots))?;
        for port in &mut self.ports[..self.port_count] {
            if let Port::StateTarget(bytes) = port {
                *bytes = target;
            }
        }
        Some(self)
    }

    /// This recurrent package as its banked form (`DeltaRule::Bank`,
    /// `CausalConv::Bank`): `slots` successor states (one, or one per token)
    /// each [`kore_xe_kernels::BANK_SLOT_ROOM`] bytes apart past the state,
    /// so every slot carries its own trailer and an accepted successor is a
    /// later run's source where it lies (a speculative verification's
    /// commit). A Mamba scan has no banked package: declined, as is any
    /// count `with_state_slots` declines.
    pub fn with_slot_bank(self, slots: u32) -> Option<Self> {
        let mut bank = self.with_state_slots(slots)?;
        let (state, _) = bank.state_bytes()?;
        match bank.recurrent? {
            Recurrence::Conv(channels) => {
                bank.artifact = kore_xe_kernels::causal_conv(CausalConv::Bank, channels)?;
            }
            Recurrence::Delta(geometry) => {
                bank.artifact = kore_xe_kernels::delta_rule(DeltaRule::Bank, geometry)?;
                bank.launch = delta_launch(geometry, DeltaRule::Bank)?;
            }
            Recurrence::Scan => return None,
        }
        let stride = state.checked_add(kore_xe_kernels::BANK_SLOT_ROOM)?;
        let target = stride.checked_mul(u64::from(slots - 1))?.checked_add(state)?;
        for port in &mut bank.ports[..bank.port_count] {
            if let Port::StateTarget(bytes) = port {
                *bytes = target;
            }
        }
        Some(bank)
    }

    /// This recurrent package over the first `tokens` rows of its operands
    /// only: the runtime-shaped packages read the count from their control
    /// trailer and loop to it, so rows past it are neither read nor written
    /// and their successors never stored. With [`Self::with_slot_bank`] at
    /// the same count, a speculative verification of more rows than it can
    /// keep banks only the rows it can (a pass's padding past them is
    /// garbage the host never accepts). Declined past the operands' rows,
    /// for none, for a token tree, and for any package without a state pair.
    pub fn with_run_tokens(mut self, tokens: u32) -> Option<Self> {
        self.recurrent?;
        let rows = u32::from_le_bytes(self.parameter[..4].try_into().ok()?);
        if tokens == 0 || tokens > rows || self.parameter_len > 8 {
            return None;
        }
        self.parameter[..4].copy_from_slice(&tokens.to_le_bytes());
        Some(self)
    }

    /// The routed projection this kernel computes over an expert table, which
    /// [`Self::with_pool`] can move onto a slot pool; `None` for any other
    /// kernel, a pooled one included.
    pub fn routed(&self) -> Option<(RoutedKernel, RoutedGeometry)> {
        self.routed
    }

    /// This routed gate/up or down projection over a slot pool of `records`
    /// slots instead of its expert table (`res/QWEN_EXPERT_STREAMING.md`,
    /// [`kore_xe_kernels::routed_pool_kernel`]): its table slot binds the
    /// pool — the down projection's from the first slot's down record
    /// ([`RoutedGeometry::pool_down_offset`]), through the last slot's
    /// record — its routes name slots, and the record count is the control
    /// block's first word, between the routes and the output, the clamp's
    /// bound. Any other kernel, or an empty pool, is declined.
    pub fn with_pool(mut self, records: u32) -> Option<Self> {
        let (kind, geometry) = self.routed?;
        let record = match kind {
            RoutedKernel::GateUp => geometry.gate_up_record_bytes(),
            RoutedKernel::Down => geometry.down_record_bytes(),
            RoutedKernel::Route => return None,
        };
        let last = u64::from(records.checked_sub(1)?).checked_mul(geometry.pool_slot_bytes())?;
        self.artifact = kore_xe_kernels::routed_pool_kernel(kind, geometry)?;
        self.input_bytes[1] = last.checked_add(record)?;
        self.parameter = [0; CONTROL_BYTES];
        self.parameter[..4].copy_from_slice(&records.to_le_bytes());
        self.parameter_len = 8;
        self.ports[..5].copy_from_slice(&[
            Port::Input(0),
            Port::Input(1),
            Port::Input(2),
            Port::Parameter,
            Port::Output,
        ]);
        self.port_count = 5;
        // A pool is the projection's last form: it is not moved again.
        self.routed = None;
        Some(self)
    }

    /// Bind a validated parent table to the tree-specific recurrent package.
    /// Every node must retain its own successor for later children and commit.
    pub fn with_token_tree(mut self, tree: axnn::lowering::TokenTree<'_>) -> Option<Self> {
        let tokens = u32::from_le_bytes(self.parameter[..4].try_into().ok()?) as usize;
        let slots = u32::from_le_bytes(self.parameter[4..8].try_into().ok()?) as usize;
        if tree.len() != tokens || slots != tokens {
            return None;
        }
        match self.recurrent? {
            Recurrence::Conv(channels) => {
                self.artifact = kore_xe_kernels::causal_conv(CausalConv::Tree, channels)?;
            }
            Recurrence::Scan => self.artifact = kore_xe_kernels::mamba_scan_tree()?,
            // The tree package slices each column finer, at its own width.
            Recurrence::Delta(geometry) => {
                self.artifact = kore_xe_kernels::delta_rule(DeltaRule::Tree, geometry)?;
                self.launch = delta_launch(geometry, DeltaRule::Tree)?;
            }
        }
        self.parameter_len = 8 + 4 * tree.len();
        for (word, parent) in self.parameter[8..self.parameter_len]
            .chunks_exact_mut(4)
            .zip(tree.parents())
        {
            word.copy_from_slice(&parent.to_le_bytes());
        }
        Some(self)
    }

    /// One member of a MoE wave (`kore_xe_kernels::moe_wave`), or `None`
    /// while its package is unsealed. `rows` is the launch's tiles (grouped
    /// up and down) or the wave's tokens (combine); `wave_bytes` the region
    /// it binds from its header on; `chunks` the bytes of the set's five
    /// imports (grouped only; an unused slot repeats a used one).
    pub fn moe_wave(
        part: MoeWave,
        rows: usize,
        wave_bytes: u64,
        chunks: [u64; xe_moe::CHUNK_SLOTS],
    ) -> Option<Self> {
        let artifact = kore_xe_kernels::moe_wave(part)?;
        if rows == 0 || wave_bytes == 0 || !wave_bytes.is_multiple_of(4) {
            return None;
        }
        let width = part.group_width();
        let columns = match part {
            MoeWave::Up => xe_moe::MOE_INTERMEDIATE.div_ceil(16),
            MoeWave::Down => xe_moe::MOE_HIDDEN.div_ceil(16),
            MoeWave::Combine => xe_moe::MOE_HIDDEN.div_ceil(4 * width as usize),
        };
        let groups = Workgroups::new(
            [u32::try_from(columns).ok()?, u32::try_from(rows).ok()?, 1],
            [width, 1, 1],
        )
        .ok()?;
        let mut ports = [Port::Output; MAX_PORTS];
        let port_count = if part == MoeWave::Combine {
            ports[0] = Port::Wave(wave_bytes);
            1
        } else {
            for (slot, &bytes) in chunks.iter().enumerate() {
                if bytes == 0 {
                    return None;
                }
                ports[slot] = Port::Chunk {
                    slot: slot as u8,
                    bytes,
                };
            }
            ports[xe_moe::CHUNK_SLOTS] = Port::Wave(wave_bytes);
            xe_moe::CHUNK_SLOTS + 1
        };
        Some(Self {
            artifact,
            launch: LaunchShape::Workgroups(groups),
            parameter: [0; CONTROL_BYTES],
            parameter_len: 8,
            recurrent: None,
            input_bytes: [0; 3],
            input_count: 0,
            output_bytes: wave_bytes,
            work: (columns * rows) as u64,
            ports,
            port_count,
            word_inputs: 0,
            dense: false,
            routed: None,
        })
    }

    /// The device's plan for a routed wave of `tokens` tokens: one
    /// workgroup reading the router's scores and the hidden rows (the MoE
    /// instruction's inputs 1 and 0, in the package's binding order) and
    /// writing the region (`wave_bytes`, from header 0).
    pub fn moe_plan(tokens: usize, wave_bytes: u64) -> Option<Self> {
        let artifact = kore_xe_kernels::moe_plan()?;
        if tokens == 0
            || tokens > xe_moe::ROUTED_MAX_TOKENS
            || wave_bytes == 0
            || !wave_bytes.is_multiple_of(4)
        {
            return None;
        }
        let groups = Workgroups::new([1, 1, 1], [kore_xe_kernels::MOE_PLAN_LANES, 1, 1]).ok()?;
        let hidden = (tokens * xe_moe::MOE_HIDDEN * 4) as u64;
        let scores = (tokens * xe_moe::ROUTED_EXPERTS * 4) as u64;
        let mut ports = [Port::Output; MAX_PORTS];
        ports[0] = Port::Input(1);
        ports[1] = Port::Input(0);
        ports[2] = Port::Wave(wave_bytes);
        Some(Self {
            artifact,
            launch: LaunchShape::Workgroups(groups),
            parameter: [0; CONTROL_BYTES],
            parameter_len: 8,
            recurrent: None,
            input_bytes: [hidden, scores, 0],
            input_count: 2,
            output_bytes: wave_bytes,
            work: (tokens * xe_moe::ROUTED_EXPERTS) as u64,
            ports,
            port_count: 3,
            word_inputs: 0,
            dense: false,
            routed: None,
        })
    }

    /// The cache bytes a read or append over slots below `end` binds: from
    /// the cache's start through the last page it touches, so a short
    /// context binds (and the lane publishes each submit) kilobytes rather
    /// than the whole cache. The kernels index the cache from its start.
    fn kv_bound(params: ax::kv::KvCacheParams, end: u32) -> Option<u64> {
        if end == 0 || end > params.capacity {
            return None;
        }
        let pages = end.div_ceil(params.page_tokens) as u64;
        pages.checked_mul(params.page_bytes()? as u64)
    }

    /// A resident KV append of `tokens` rows at cache slot `first`
    /// (`xe-kv-append.comp`): the keys and values (the instruction's inputs
    /// 1 and 2), the cache bound whole and written, and the control block.
    /// One group of a lane per column per KV head and aligned four slots
    /// the rows touch (the four whose scale bytes share a dword), so pages
    /// are a multiple of four slots. The cache's geometry is axnn-cpu's; only
    /// a captured one ([`KvGeometry`]: two or eight KV heads of 128, four of
    /// 256) admits.
    pub fn kv_append(tokens: usize, first: u32, params: ax::kv::KvCacheParams) -> Option<Self> {
        let geometry = KvGeometry::of_cache(params.kv_heads, params.head_dim)?;
        let artifact = kore_xe_kernels::kv_kernel(KvKernel::Append, geometry)?;
        let end = first.checked_add(u32::try_from(tokens).ok()?)?;
        let cache_bytes = Self::kv_bound(params, end)?;
        if tokens == 0 || params.page_tokens % KV_SLOT_GROUP != 0 || end > params.capacity {
            return None;
        }
        let slot_groups = end.div_ceil(KV_SLOT_GROUP) - first / KV_SLOT_GROUP;
        let groups = Workgroups::new(
            [slot_groups, params.kv_heads, 1],
            [KvKernel::Append.group_width(geometry), 1, 1],
        )
        .ok()?;
        let rows = (tokens * params.kv_heads as usize * params.head_dim as usize * 4) as u64;
        let mut parameter = [0; CONTROL_BYTES];
        for (word, value) in parameter.chunks_exact_mut(4).zip([
            first,
            tokens as u32,
            params.capacity,
            params.page_tokens,
            params.kv_heads,
        ]) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        let mut ports = [Port::Output; MAX_PORTS];
        ports[0] = Port::Input(1);
        ports[1] = Port::Input(2);
        ports[2] = Port::KvCache {
            bytes: cache_bytes,
            written: true,
        };
        ports[3] = Port::Parameter;
        Some(Self {
            artifact,
            launch: LaunchShape::Workgroups(groups),
            parameter,
            parameter_len: 20,
            recurrent: None,
            input_bytes: [0, rows, rows],
            input_count: 3,
            output_bytes: 0,
            work: (slot_groups * params.kv_heads) as u64,
            ports,
            port_count: 4,
            word_inputs: 0,
            dense: false,
            routed: None,
        })
    }

    /// A resident attention read of `tokens` query rows whose first row sits
    /// at cache slot `first` (`xe-kv-attention.comp`): the queries (input
    /// 2), the cache bound whole, the control block, the split partials and
    /// the output. Each query row is one subgroup; `splits` workgroups along
    /// Y share its causal prefix, and a second kernel ([`Self::kv_reduce`])
    /// merges them when `splits > 1`. Captured ([`KvGeometry`]) for 32 query
    /// heads of 128 over two KV heads (Nemotron), 64 over eight (Qwen3 GQA
    /// 8:1) and 24 of 256 over four (Qwen3.5).
    pub fn kv_attention(
        tokens: usize,
        first: u32,
        params: ax::kv::AttentionDecodeParams,
        splits: u32,
    ) -> Option<Self> {
        let geometry = Self::kv_read_geometry(params)?;
        let artifact = kore_xe_kernels::kv_kernel(KvKernel::Attention, geometry)?;
        let cache = params.cache;
        let end = first.checked_add(u32::try_from(tokens).ok()?)?;
        if tokens == 0 || splits == 0 || end > cache.capacity {
            return None;
        }
        let cache_bytes = Self::kv_bound(cache, end)?;
        let width = KvKernel::Attention.group_width(geometry);
        // One SIMD32 subgroup per query row, the geometry's rows per
        // workgroup (its other subgroups only stage the shared tile).
        let rows = tokens * params.query_heads as usize;
        let groups = Workgroups::new(
            [u32::try_from(rows.div_ceil(geometry.rows as usize)).ok()?, splits, 1],
            [width, 1, 1],
        )
        .ok()?;
        let mut parameter = [0; CONTROL_BYTES];
        for (word, value) in parameter.chunks_exact_mut(4).zip([
            first,
            tokens as u32,
            cache.capacity,
            cache.page_tokens,
            splits,
            params.logit_scale.bits(),
        ]) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        let query_bytes = (rows * cache.head_dim as usize * 4) as u64;
        let partial_bytes = (rows * splits as usize * (cache.head_dim as usize + 2) * 4) as u64;
        let mut ports = [Port::Output; MAX_PORTS];
        ports[0] = Port::Input(2);
        ports[1] = Port::KvCache {
            bytes: cache_bytes,
            written: false,
        };
        ports[2] = Port::Parameter;
        ports[3] = Port::Partials(partial_bytes);
        ports[4] = Port::Output;
        Some(Self {
            artifact,
            launch: LaunchShape::Workgroups(groups),
            parameter,
            parameter_len: 24,
            recurrent: None,
            input_bytes: [0, 0, query_bytes],
            input_count: 3,
            output_bytes: query_bytes,
            work: (rows * splits as usize) as u64,
            ports,
            port_count: 5,
            word_inputs: 0,
            dense: false,
            routed: None,
        })
    }

    /// A fill of `words` 32-bit words with `value` (`xe-fill-u32.comp`):
    /// the control block `{count, value}`, then the words, written. It is
    /// how a frozen decode writes into a region the host cannot map: the
    /// `{tokens, slots}` trailer after each recurrent state slot.
    pub fn fill(words: u32, value: u32) -> Option<Self> {
        if words == 0 {
            return None;
        }
        let lanes = kore_xe_kernels::FILL_LANES;
        let groups = Workgroups::new([words.div_ceil(lanes), 1, 1], [lanes, 1, 1]).ok()?;
        let mut parameter = [0; CONTROL_BYTES];
        parameter[..4].copy_from_slice(&words.to_le_bytes());
        parameter[4..8].copy_from_slice(&value.to_le_bytes());
        let mut ports = [Port::Output; MAX_PORTS];
        ports[0] = Port::Parameter;
        ports[1] = Port::Output;
        Some(Self {
            artifact: kore_xe_kernels::fill_u32(),
            launch: LaunchShape::Workgroups(groups),
            parameter,
            parameter_len: 8,
            recurrent: None,
            input_bytes: [0; 3],
            input_count: 0,
            output_bytes: u64::from(words) * 4,
            work: u64::from(words),
            ports,
            port_count: 2,
            word_inputs: 0,
            dense: false,
            routed: None,
        })
    }

    /// The device's greedy pick over `rows` rows of F32 logits `[rows,
    /// vocab]` at a captured Qwen vocabulary (`xe-nemotron-top1.comp`): slot
    /// 0 the logits, slot 1 each row's best value (written, read by
    /// nothing), slot 2 its I32 index, the result. One 256-lane workgroup a
    /// row; ties go to the lower index, as `f32::total_cmp` orders. It reads
    /// no control block. A frozen decode chains its drafts with it: the
    /// index lands where the next pass reads its token.
    pub fn top1(rows: u32, vocab: u32) -> Option<Self> {
        if rows == 0 {
            return None;
        }
        let artifact = kore_xe_kernels::qwen_top1(vocab)?;
        let logits = u64::from(rows).checked_mul(u64::from(vocab))?.checked_mul(4)?;
        let groups = Workgroups::new([rows, 1, 1], [256, 1, 1]).ok()?;
        let mut ports = [Port::Output; MAX_PORTS];
        ports[0] = Port::Input(0);
        ports[1] = Port::Discard(u64::from(rows) * 4);
        ports[2] = Port::Output;
        Some(Self {
            artifact,
            launch: LaunchShape::Workgroups(groups),
            parameter: [0; CONTROL_BYTES],
            parameter_len: 8,
            recurrent: None,
            input_bytes: [logits, 0, 0],
            input_count: 1,
            output_bytes: u64::from(rows) * 4,
            work: logits / 4,
            ports,
            port_count: 3,
            word_inputs: 0,
            dense: false,
            routed: None,
        })
    }

    /// The captured geometry of an attention read: its cache's, with the
    /// query heads the package bakes.
    fn kv_read_geometry(params: ax::kv::AttentionDecodeParams) -> Option<KvGeometry> {
        KvGeometry::of_cache(params.cache.kv_heads, params.cache.head_dim)
            .filter(|geometry| geometry.query_heads == params.query_heads)
    }

    /// The reduction of an attention read's `splits` partials into its
    /// output (`xe-kv-attention.comp` with `REDUCE=1`): the control block,
    /// the partials and the output, one subgroup per query row.
    pub fn kv_reduce(
        tokens: usize,
        first: u32,
        params: ax::kv::AttentionDecodeParams,
        splits: u32,
    ) -> Option<Self> {
        let geometry = Self::kv_read_geometry(params)?;
        let artifact = kore_xe_kernels::kv_kernel(KvKernel::Reduce, geometry)?;
        let read = Self::kv_attention(tokens, first, params, splits)?;
        if splits < 2 {
            return None;
        }
        let width = KvKernel::Reduce.group_width(geometry);
        let rows = tokens * params.query_heads as usize;
        let groups = Workgroups::new(
            [u32::try_from(rows.div_ceil(geometry.rows as usize)).ok()?, 1, 1],
            [width, 1, 1],
        )
        .ok()?;
        let Port::Partials(partial_bytes) = read.ports[3] else {
            return None;
        };
        let mut ports = [Port::Output; MAX_PORTS];
        ports[0] = Port::Parameter;
        ports[1] = Port::Partials(partial_bytes);
        ports[2] = Port::Output;
        Some(Self {
            artifact,
            launch: LaunchShape::Workgroups(groups),
            parameter: read.parameter,
            parameter_len: read.parameter_len,
            recurrent: None,
            input_bytes: [0, 0, 0],
            input_count: 0,
            output_bytes: read.output_bytes,
            work: rows as u64,
            ports,
            port_count: 3,
            word_inputs: 0,
            dense: false,
            routed: None,
        })
    }

    /// A kernel that takes a whole fused run of the prepared engine:
    /// `SquaredRelu` (the projection's squared-ReLU epilogue, e3) or
    /// `Accumulate` (the projection added onto an accumulator the executor
    /// fills with the run's base, e4, route scale 1.0 from the ones run).
    /// Only captured shapes admit; the plan is re-checked like `from_plan`.
    pub fn fused_run<I>(
        plan: &KernelPlan<I>,
        run: ax_exec::FusedRun,
        minimum: KernelAccuracy,
    ) -> Option<Self> {
        if let ax_exec::FusedRun::RmsNorm { width, epsilon } = run {
            // The captured group kernel consumes the actual gate here. Plain
            // group RMS binds a ones gate; the fused form removes that
            // separate multiply and its intermediate staging/publication.
            if plan.semantic != SemanticOp::Binary(axnn::BinaryOp::Mul)
                || width != GROUP_RMS_WIDTH / 8
                || epsilon.to_f32().to_bits() != RMS_EPSILON.to_bits()
                || plan.shape.rank() != 2
                || plan.shape.dim(1) != GROUP_RMS_WIDTH
                || plan.input(0)?.shape != plan.shape
                || plan.input(1)?.shape != plan.shape
                || plan.input(0)?.dtype != axnn::DType::F32
                || plan.input(1)?.dtype != axnn::DType::F32
            {
                return None;
            }
            let mut kernel = Self::from_plan(plan, minimum)?;
            let rows = u32::try_from(plan.shape.dim(0)).ok()?.checked_mul(8)?;
            kernel.artifact = kore_xe_kernels::rms_norm(RmsNorm::MambaGateGroup);
            kernel.launch =
                LaunchShape::Workgroups(Workgroups::new([rows, 1, 1], [256, 1, 1]).ok()?);
            kernel.input_count = 3;
            kernel.input_bytes[2] = (GROUP_RMS_WIDTH * size_of::<f32>()) as u64;
            kernel.word_inputs = 0b111;
            kernel.ports[..4].copy_from_slice(&[
                Port::Input(0),
                Port::Input(1),
                Port::Input(2),
                Port::Output,
            ]);
            kernel.port_count = 4;
            return Some(kernel);
        }
        let KernelParams::MatmulEpilogue {
            activation: EpilogueActivation::None,
            ..
        } = plan.params()
        else {
            return None;
        };
        let mut kernel = Self::from_plan_unlisted(plan, minimum)?;
        let shape = plan.input(0)?.shape;
        let (m, k, n) = (
            shape.dim(0),
            shape.dim(1),
            plan.result_tensor(0)?.shape.dim(1),
        );
        let kind = FUSED_PROJECTIONS
            .iter()
            .find_map(|&(rows, columns, inner, fused, kind)| {
                ((n, k, fused) == (columns, inner, run) && rows_fit(rows, m, kind.runtime_m()))
                    .then_some(kind)
            })?;
        kernel.artifact = kore_xe_kernels::nvfp4_projection(kind);
        if run == ax_exec::FusedRun::Accumulate {
            kernel.ports[kernel.input_count + 1] = Port::Accumulator;
            kernel.ports[kernel.input_count + 2] = Port::Ones(core::mem::size_of::<f32>() as u64);
            kernel.port_count = kernel.input_count + 3;
        }
        Some(kernel)
    }

    /// `from_plan`'s checks for an NVFP4 projection whose shape the plain
    /// catalog need not list: the fused forms have their own table.
    fn from_plan_unlisted<I>(plan: &KernelPlan<I>, minimum: KernelAccuracy) -> Option<Self> {
        if minimum != KernelAccuracy::Approximate
            || plan.semantic != SemanticOp::MatMulNvfp4Transposed
        {
            return None;
        }
        let request = Self::canonical(
            plan.semantic,
            plan.inputs(),
            plan.result_tensor(0),
            plan.params(),
        )?;
        if plan.input_dtypes != request.input_dtypes
            || plan.input_shapes != request.input_shapes
            || plan.output_dtype != request.output_dtype
            || plan.shape != request.shape
            || plan.work_items != request.work_items()
            || plan.inputs() != request.inputs()
            || plan.result() != request.result()
            || plan.params() != request.params()
            || plan.effects() != request.effects()
        {
            return None;
        }
        let KernelParams::MatmulEpilogue { scale, .. } = request.params() else {
            return None;
        };
        let output = request.result_tensor(0)?;
        let shape = request.input(0)?.shape;
        let (m, n) = (shape.dim(0), output.shape.dim(1));
        let mut input_bytes = [0; 3];
        for (index, input) in request.inputs().iter().enumerate() {
            let input = input.as_ref()?;
            if !input.is_contiguous() || input.layout.offset() != 0 {
                return None;
            }
            input_bytes[index] = (input.checked_storage_bytes()? as u64).checked_add(3)? & !3;
        }
        input_bytes[0] = request.input(0)?.checked_storage_bytes()? as u64;
        let groups = projection_groups(
            m,
            n,
            shape.dim(1),
            Nvfp4Projection::DecodeTile.group_width(),
            16,
            8,
        )?;
        let mut ports = [Port::Output; MAX_PORTS];
        ports[..5].copy_from_slice(&[
            Port::Input(0),
            Port::Input(1),
            Port::Input(2),
            Port::Parameter,
            Port::Output,
        ]);
        Some(Self {
            artifact: kore_xe_kernels::nvfp4_projection(Nvfp4Projection::DecodeTile),
            launch: LaunchShape::Workgroups(groups),
            parameter: projection_parameter(scale.to_f32(), m),
            parameter_len: 8,
            recurrent: None,
            input_bytes,
            input_count: 3,
            output_bytes: output.checked_storage_bytes()? as u64,
            work: request.work_items(),
            ports,
            port_count: 5,
            word_inputs: 0,
            dense: false,
            routed: None,
        })
    }

    /// Bind every port to `(buffer, offset)` from `place`, in slot order.
    /// No two bindings may overlap, reads included: the Xe provider refuses
    /// such a launch. Offsets are word aligned (the shaders access storage
    /// as words). The provider still checks bounds and DMA authority.
    pub fn port_bindings(
        &self,
        mut place: impl FnMut(Port) -> (u64, u64),
    ) -> Result<AccelBindings, Error> {
        let empty = AccelBinding {
            buffer: 0,
            offset: 0,
            len: 0,
            slot: 0,
            access: AccessMode::Read,
        };
        let mut bindings = [empty; MAX_ACCEL_BINDINGS];
        let mut output = None;
        let last = self.port_count - 1;
        for (slot, &port) in self.ports().iter().enumerate() {
            let (buffer, offset) = place(port);
            let len = self.port_bytes(port);
            if buffer == 0
                || len == 0
                || offset.checked_add(len).is_none()
                || !offset.is_multiple_of(core::mem::size_of::<u32>() as u64)
            {
                return Err(Error::Binding);
            }
            // The provider binds the final slot writable and every other one
            // readable (a written slot that is not last is read-write); an
            // accumulate package's final slot is its read-only route scale,
            // and a recurrent package writes its successor states beside
            // its output.
            let access = match port {
                Port::Output if slot == last => AccessMode::Write,
                Port::Output
                | Port::Accumulator
                | Port::Wave(_)
                | Port::StateTarget(_)
                | Port::KvCache { written: true, .. }
                | Port::Partials(_)
                | Port::Discard(_) => AccessMode::ReadWrite,
                _ if slot == last => AccessMode::ReadWrite,
                _ => AccessMode::Read,
            };
            // The KV append's result is its cache: nothing of it is read back.
            if matches!(
                port,
                Port::Output | Port::Accumulator | Port::Wave(_) | Port::KvCache { written: true, .. }
            ) {
                output = Some(slot);
            }
            bindings[slot] = AccelBinding {
                buffer,
                offset,
                len,
                slot: slot as u32,
                access,
            };
        }
        let live = &bindings[..self.port_count];
        output.ok_or(Error::Binding)?;
        if live.iter().enumerate().any(|(index, a)| {
            live[index + 1..].iter().any(|b| {
                a.buffer == b.buffer && a.offset < b.offset + b.len && b.offset < a.offset + a.len
            })
        }) {
            return Err(Error::Binding);
        }
        AccelBindings::new(live).ok_or(Error::Binding)
    }

    pub fn artifact(&self) -> XeKernelArtifact {
        self.artifact
    }
    pub fn launch(&self) -> LaunchShape {
        self.launch
    }
    /// Eight bytes in a retained read-only control buffer: the tensor scale
    /// or XVE element count, then the rows a runtime-M projection reads
    /// (or K for a runtime-shaped F32 projection).
    /// Recurrent packages instead append these two words (tokens, slots)
    /// to the source-state buffer. Tensor payloads never enter IPC records.
    pub fn parameter_bytes(&self) -> [u8; 8] {
        self.parameter[..8].try_into().expect("eight bytes")
    }
    /// The whole control block the package reads: the eight bytes above,
    /// or a resident KV kernel's five or six words.
    pub fn control_bytes(&self) -> &[u8] {
        &self.parameter[..self.parameter_len]
    }
    pub fn work_items(&self) -> u64 {
        self.work
    }
    pub fn input_bytes(&self) -> &[u64] {
        &self.input_bytes[..self.input_count]
    }
    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }

    /// Whether this is a dense E4M3/BF16/F32 projection package.
    pub fn is_dense(&self) -> bool {
        self.dense
    }

    /// Prepare a storable binding callback before transferring any allocation.
    /// Ports are inputs, parameter, output; offsets are within the imported
    /// views. Geometry is checked now. The final import IDs are checked again
    /// during binding, including aliases between resident and new buffers.
    pub fn launch_builder(
        self,
        queue: u64,
        program: u64,
        offsets: [u64; KERNEL_BUFFER_COUNT],
        timeout_ns: u64,
    ) -> Result<KernelLaunch, Error> {
        if !self.legacy_layout() {
            return Err(Error::Binding);
        }
        let plan = KernelLaunch {
            kernel: self,
            queue,
            program,
            offsets,
            timeout_ns,
        };
        // Distinct placeholder IDs prove local geometry only. Actual IDs are
        // provided after imports and cannot inherit this alias assumption.
        let distinct = core::array::from_fn(|index| index as u64 + 1);
        plan.build(&distinct).map_err(|_| Error::Binding)?;
        Ok(plan)
    }

    /// Bind pre-imported buffer IDs and byte offsets in semantic port order.
    /// No two bindings may overlap, reads included: the Xe provider refuses
    /// such a launch. Bounds and DMA authority remain checked
    /// by the provider. Retain all routes until the execution is retired.
    pub fn bindings(
        &self,
        inputs: &[(u64, u64)],
        parameter: (u64, u64),
        output: (u64, u64),
    ) -> Result<AccelBindings, Error> {
        if inputs.len() != self.input_count {
            return Err(Error::Binding);
        }
        let empty = AccelBinding {
            buffer: 0,
            offset: 0,
            len: 0,
            slot: 0,
            access: AccessMode::Read,
        };
        let mut bindings = [empty; MAX_ACCEL_BINDINGS];
        for (index, (&(buffer, offset), &len)) in inputs.iter().zip(self.input_bytes()).enumerate()
        {
            bindings[index] = AccelBinding {
                buffer,
                offset,
                len,
                slot: index as u32,
                access: AccessMode::Read,
            };
        }
        bindings[self.input_count] = AccelBinding {
            buffer: parameter.0,
            offset: parameter.1,
            len: self.parameter_len as u64,
            slot: self.input_count as u32,
            access: AccessMode::Read,
        };
        let output_slot = self.input_count + 1;
        bindings[output_slot] = AccelBinding {
            buffer: output.0,
            offset: output.1,
            len: self.output_bytes,
            slot: output_slot as u32,
            access: AccessMode::Write,
        };
        let live = &bindings[..=output_slot];
        for binding in live {
            if binding.buffer == 0
                || binding.len == 0
                || binding.offset.checked_add(binding.len).is_none()
            {
                return Err(Error::Binding);
            }
            // The captured shaders access storage as uint/float words, even
            // for the packed byte tensors; byte-aligned aliases are not valid.
            if !binding
                .offset
                .is_multiple_of(core::mem::size_of::<u32>() as u64)
            {
                return Err(Error::Binding);
            }
        }
        if live.iter().enumerate().any(|(index, a)| {
            live[index + 1..].iter().any(|b| {
                a.buffer == b.buffer && a.offset < b.offset + b.len && b.offset < a.offset + a.len
            })
        }) {
            return Err(Error::Binding);
        }
        AccelBindings::new(live).ok_or(Error::Binding)
    }

    /// Construct work for the shared asynchronous execution ledger. The owner
    /// retains its possessed face and imported buffers until acknowledged
    /// retirement, including after cancellation or an observation timeout.
    pub fn execution(
        &self,
        queue: u64,
        program: u64,
        bindings: AccelBindings,
        timeout_ns: u64,
    ) -> Result<LaunchExecution, Error> {
        if queue == 0 || program == 0 || timeout_ns == 0 {
            return Err(Error::Binding);
        }
        Ok(LaunchExecution {
            queue,
            program,
            bindings,
            timeout_ns,
            launch: self.launch,
        })
    }
}

// Package availability, not a shape restriction on Axiom. Both graph selection
// and pre-load admission consume this bounded catalog of supported contracts.
// Squared-ReLU and accumulating packages require different semantic interfaces.
const PROJECTIONS: &[(usize, usize, usize, EpilogueActivation, Nvfp4Projection)] = &[
    (
        1,
        17,
        32,
        EpilogueActivation::None,
        Nvfp4Projection::DecodeTile,
    ),
    (
        8,
        32,
        32,
        EpilogueActivation::Silu,
        Nvfp4Projection::PrefillSiluTile,
    ),
    (
        1,
        4096,
        2688,
        EpilogueActivation::Silu,
        Nvfp4Projection::MambaGateSiluDecode,
    ),
    (
        8,
        4096,
        2688,
        EpilogueActivation::Silu,
        Nvfp4Projection::MambaGateSiluPrefill,
    ),
    (
        1,
        6144,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::MambaXbcDecode,
    ),
    (
        8,
        6144,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::MambaXbcPrefill,
    ),
    (
        1,
        64,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::MambaDtDecode,
    ),
    (
        8,
        64,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::MambaDtPrefill,
    ),
    (
        1,
        2688,
        4096,
        EpilogueActivation::None,
        Nvfp4Projection::HiddenOutDecode,
    ),
    (
        8,
        2688,
        4096,
        EpilogueActivation::None,
        Nvfp4Projection::HiddenOutPrefill,
    ),
    (
        1,
        4096,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::AttentionQDecode,
    ),
    (
        8,
        4096,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::AttentionQPrefill,
    ),
    (
        1,
        256,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::AttentionKvDecode,
    ),
    (
        8,
        256,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::AttentionKvPrefill,
    ),
    (
        1,
        131072,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::LmHeadDecode,
    ),
    (
        8,
        131072,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::LmHeadPrefill,
    ),
    // Fixed 1/8-row packages retain priority. This witness was captured at
    // 16 rows; its parameter supplies M for every other multi-row width.
    (
        16,
        131072,
        2688,
        EpilogueActivation::None,
        Nvfp4Projection::LmHeadWide,
    ),
];

/// Captured projections that take a whole fused run (their epilogue is the
/// run's tail). The routed-expert forms are Xe2-only; loading refuses the
/// missing variant on Xe3.
const FUSED_PROJECTIONS: &[(usize, usize, usize, ax_exec::FusedRun, Nvfp4Projection)] = &[
    (
        8,
        3712,
        2688,
        ax_exec::FusedRun::SquaredRelu,
        Nvfp4Projection::SharedUpSquaredReluPrefill,
    ),
    (
        1,
        3712,
        2688,
        ax_exec::FusedRun::SquaredRelu,
        Nvfp4Projection::SharedUpSquaredReluDecode,
    ),
    (
        8,
        2688,
        3712,
        ax_exec::FusedRun::Accumulate,
        Nvfp4Projection::SharedDownAccumulatePrefill,
    ),
    (
        1,
        2688,
        3712,
        ax_exec::FusedRun::Accumulate,
        Nvfp4Projection::SharedDownAccumulateDecode,
    ),
    (
        16,
        3712,
        2688,
        ax_exec::FusedRun::SquaredRelu,
        Nvfp4Projection::SharedUpSquaredReluWide,
    ),
    (
        16,
        2688,
        3712,
        ax_exec::FusedRun::Accumulate,
        Nvfp4Projection::SharedDownAccumulateWide,
    ),
];

fn projection(
    m: usize,
    n: usize,
    k: usize,
    activation: EpilogueActivation,
) -> Option<Nvfp4Projection> {
    PROJECTIONS
        .iter()
        .find_map(|&(rows, columns, inner, epilogue, kind)| {
            ((n, k, activation) == (columns, inner, epilogue)
                && rows_fit(rows, m, kind.runtime_m()))
            .then_some(kind)
        })
}

/// The Qwen dense decode projection captured at (N, K) with `activation`
/// fused (res/QWEN_XE_CATALOG.md), read from the catalog itself. Qwen3-32B's
/// are all the identity (its graph keeps SiLU, the residual add and the
/// gate·up multiply as elementwise ops; fusion is neutral at M=1), so its
/// gate and up share one package; Qwen3.8-27B's graph fuses SiLU into its z
/// and MLP gate projections and sigmoid into its write strength, so an
/// (N, K) can carry a package per epilogue.
fn qwen_projection_of(n: usize, k: usize, activation: EpilogueActivation) -> Option<QwenProjection> {
    let epilogue = epilogue_code(activation);
    QwenProjection::ALL.into_iter().find(|kind| {
        (kind.n() as usize, kind.k() as usize, kind.epilogue()) == (n, k, epilogue)
    })
}

/// The projection kernels' `EPILOGUE` code of an activation.
const fn epilogue_code(activation: EpilogueActivation) -> u32 {
    match activation {
        EpilogueActivation::None => 0,
        EpilogueActivation::Silu => 1,
        EpilogueActivation::Sigmoid => 2,
    }
}

/// Whether a package captured at `rows` serves `m` rows: its own M, or any
/// M > 1 for a runtime-M multi-row package.
fn rows_fit(rows: usize, m: usize, runtime_m: bool) -> bool {
    rows == m || (runtime_m && rows > 1 && m > 1)
}

/// NVFP4 shaders use u32 indices and storage-buffer byte offsets. Include
/// the F32 stride so a wide buffer cannot admit wrapped address arithmetic.
fn projection_groups(m: usize, n: usize, k: usize, width: u32, columns: u32, tile_rows: u32) -> Option<Workgroups> {
    let tile_rows = usize::try_from(tile_rows).ok()?;
    if tile_rows == 0 {
        return None;
    }
    u32::try_from(m).ok()?;
    u32::try_from(m.checked_mul(k)?.checked_mul(core::mem::size_of::<f32>())?).ok()?;
    u32::try_from(m.checked_mul(n)?.checked_mul(core::mem::size_of::<f32>())?).ok()?;
    u32::try_from(n.checked_mul(k)?).ok()?;
    Workgroups::new(
        [
            u32::try_from(n.div_ceil(columns as usize)).ok()?,
            u32::try_from(m.div_ceil(tile_rows)).ok()?,
            1,
        ],
        [width, 1, 1],
    )
    .ok()
}

/// The launch of a runtime-shaped recurrent package: one invocation per
/// channel or state row (`rows` of them). Shape is data, never extra Y/Z
/// workgroups that would launch `tokens * slots` times the useful work.
fn recurrent_launch(rows: usize, tokens: u32) -> Option<LaunchShape> {
    if tokens == 0 {
        return None;
    }
    let groups = u32::try_from(rows.div_ceil(RECURRENT_LANES as usize)).ok()?;
    Some(LaunchShape::Workgroups(
        Workgroups::new([groups, 1, 1], [RECURRENT_LANES, 1, 1]).ok()?,
    ))
}

/// The head geometry of a gated delta rule's params.
fn delta_geometry(rule: axnn::lowering::GatedDeltaRuleParams) -> DeltaRuleGeometry {
    DeltaRuleGeometry {
        key_heads: rule.key_heads,
        key_dim: rule.key_dim,
        value_heads: rule.value_heads,
        value_dim: rule.value_dim,
    }
}

/// A delta rule package's launch: one workgroup per value head, at the
/// width its variant was captured with.
fn delta_launch(geometry: DeltaRuleGeometry, kind: DeltaRule) -> Option<LaunchShape> {
    Some(LaunchShape::Workgroups(
        Workgroups::new(
            [geometry.value_heads, 1, 1],
            [kind.group_width(geometry), 1, 1],
        )
        .ok()?,
    ))
}

/// A routed-expert package's launch for `tokens` rows, at the width it was
/// captured with ([`RoutedKernel::groups`]).
fn routed_launch(kind: RoutedKernel, geometry: RoutedGeometry, tokens: u32) -> Option<LaunchShape> {
    Some(LaunchShape::Workgroups(
        Workgroups::new(kind.groups(geometry, tokens)?, [kind.group_width(), 1, 1]).ok()?,
    ))
}

/// Source-state trailer: token count then successor-state slots, LE u32s.
fn recurrent_parameter(tokens: u32) -> [u8; CONTROL_BYTES] {
    let mut block = [0; CONTROL_BYTES];
    block[..4].copy_from_slice(&tokens.to_le_bytes());
    block[4..8].copy_from_slice(&1u32.to_le_bytes());
    block
}

/// A recurrent package's slots: its three operands, the state before the
/// run, its successor states (one slot, `state` bytes) and the output.
fn recurrent_ports(state: u64) -> ([Port; MAX_PORTS], usize) {
    let mut ports = [Port::Output; MAX_PORTS];
    ports[..6].copy_from_slice(&[
        Port::Input(0),
        Port::Input(1),
        Port::Input(2),
        Port::StateSource(state),
        Port::StateTarget(state),
        Port::Output,
    ]);
    (ports, 6)
}

/// A projection's parameter block: the F32 tensor scale, then M.
fn projection_parameter(scale: f32, m: usize) -> [u8; CONTROL_BYTES] {
    let mut block = [0; CONTROL_BYTES];
    block[..4].copy_from_slice(&scale.to_le_bytes());
    block[4..8].copy_from_slice(&(m as u32).to_le_bytes());
    block
}

/// An eight-byte block as a control block.
fn control(block: [u8; 8]) -> [u8; CONTROL_BYTES] {
    let mut control = [0; CONTROL_BYTES];
    control[..8].copy_from_slice(&block);
    control
}

/// A one-word parameter in the eight-byte block.
fn word_parameter(word: [u8; 4]) -> [u8; CONTROL_BYTES] {
    let mut block = [0; CONTROL_BYTES];
    block[..4].copy_from_slice(&word);
    block
}

/// Captured dense projections: (rows, columns, inner, weight dtype,
/// epilogue, kind). E4M3 rows are `B8`, BF16 rows `B16`.
const DENSE_PROJECTIONS: &[(
    usize,
    usize,
    usize,
    axnn::DType,
    EpilogueActivation,
    DenseProjection,
)] = &[
    (
        8,
        4096,
        2688,
        axnn::DType::B8,
        EpilogueActivation::Silu,
        DenseProjection::MambaGateSiluE4m3Prefill,
    ),
    (
        8,
        6144,
        2688,
        axnn::DType::B8,
        EpilogueActivation::None,
        DenseProjection::MambaXbcE4m3Prefill,
    ),
    (
        8,
        64,
        2688,
        axnn::DType::B8,
        EpilogueActivation::None,
        DenseProjection::MambaDtE4m3Prefill,
    ),
    (
        8,
        2688,
        4096,
        axnn::DType::B8,
        EpilogueActivation::None,
        DenseProjection::MambaOutE4m3Prefill,
    ),
    (
        8,
        4096,
        2688,
        axnn::DType::B16,
        EpilogueActivation::None,
        DenseProjection::AttentionQBf16Prefill,
    ),
    (
        8,
        256,
        2688,
        axnn::DType::B16,
        EpilogueActivation::None,
        DenseProjection::AttentionKvBf16Prefill,
    ),
    (
        8,
        2688,
        4096,
        axnn::DType::B16,
        EpilogueActivation::None,
        DenseProjection::AttentionOutBf16Prefill,
    ),
];

/// The dense kind for this projection, and whether its runtime-M package
/// serves it: the fixed package at its own M, the runtime-M one (when
/// captured) at any other M > 1.
pub(crate) fn dense_projection(
    m: usize,
    n: usize,
    k: usize,
    weights: axnn::DType,
    activation: EpilogueActivation,
) -> Option<(DenseProjection, bool)> {
    DENSE_PROJECTIONS
        .iter()
        .find_map(|&(rows, columns, inner, dtype, epilogue, kind)| {
            if (n, k, weights, activation) != (columns, inner, dtype, epilogue) {
                return None;
            }
            if rows == m {
                return Some((kind, false));
            }
            (rows > 1 && m > 1 && kore_xe_kernels::dense_projection_wide(kind).is_some())
                .then_some((kind, true))
        })
}

/// A single Xe target for `ax` placement. The runtime supplies measured costs
/// in the same units as its other targets; there is no assumed TOPS or bandwidth.
/// Transfer callbacks price only routes the runtime can actually realize.
pub struct Target<A, C, T> {
    pub minimum_accuracy: KernelAccuracy,
    /// True only after this package has been admitted by the possessed GPU.
    /// Some captured families have no variant on every supported generation.
    pub available: A,
    pub execution_cost: C,
    pub transfer_cost: T,
}

impl<A, C, T> CostModel for Target<A, C, T>
where
    A: Fn(&Kernel) -> bool,
    C: Fn(&Kernel) -> u64,
    T: Fn(TensorView<'_>, Option<u16>, u16) -> u64,
{
    fn backend_count(&self) -> u16 {
        1
    }
    fn supports(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> bool {
        backend == 0
            && Kernel::from_node(graph, node, self.minimum_accuracy)
                .is_some_and(|kernel| (self.available)(&kernel))
    }
    fn execute_cost(&self, backend: u16, graph: &Graph<'_>, node: NodeView<'_>) -> u64 {
        if backend != 0 {
            return u64::MAX;
        }
        Kernel::from_node(graph, node, self.minimum_accuracy)
            .filter(|kernel| (self.available)(kernel))
            .map_or(u64::MAX, |kernel| (self.execution_cost)(&kernel))
    }
    fn transfer_cost(&self, tensor: TensorView<'_>, source: Option<u16>, destination: u16) -> u64 {
        if destination != 0 || source.is_some_and(|source| source != 0) {
            return u64::MAX;
        }
        if source == Some(destination) {
            return 0;
        }
        (self.transfer_cost)(tensor, source, destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axnn::{DType, Shape, TensorDesc};

    fn desc(shape: [usize; 2], dtype: DType) -> TensorDesc {
        TensorDesc::try_new(Shape::from(shape), dtype).unwrap()
    }

    #[test]
    fn resident_kv_kernels_bind_the_cache_whole_and_split_a_long_prefix() {
        use alloc::vec::Vec;
        use axnn::lowering::{AttentionDecodeParams, KvCacheParams, KvCacheStorage, ResourceKey};
        use kore_xe_kernels::{KvGeometry, KvKernel, kv_kernel};
        let words = |values: &[u32]| -> Vec<u8> {
            values.iter().flat_map(|value| value.to_le_bytes()).collect()
        };
        let cache = KvCacheParams::new(
            ResourceKey {
                group: 0,
                binding: 0,
            },
            2,
            128,
            4096,
            16,
            KvCacheStorage::E4M3TokenScaleE8M0,
        )
        .unwrap();
        // Slots below 43 lie in the first three pages of sixteen.
        let page = cache.page_bytes().unwrap() as u64;
        let bytes = 3 * page;

        // An append of three rows at slot 40: slots 40..43 are one aligned
        // group of four, one workgroup per KV head; the cache bound through
        // its third page and written, the control block last.
        let append = Kernel::kv_append(3, 40, cache).unwrap();
        assert_eq!(
            append.artifact().bytes(),
            kv_kernel(KvKernel::Append, KvGeometry::NEMOTRON).unwrap().bytes()
        );
        assert_eq!(
            append.launch(),
            LaunchShape::Workgroups(Workgroups::new([1, 2, 1], [128, 1, 1]).unwrap())
        );
        // Slots 3..11 straddle three groups; slots 7..8 two.
        assert_eq!(
            Kernel::kv_append(8, 3, cache).unwrap().launch(),
            LaunchShape::Workgroups(Workgroups::new([3, 2, 1], [128, 1, 1]).unwrap())
        );
        assert_eq!(
            Kernel::kv_append(1, 7, cache).unwrap().launch(),
            LaunchShape::Workgroups(Workgroups::new([1, 2, 1], [128, 1, 1]).unwrap())
        );
        assert_eq!(
            Kernel::kv_append(2, 7, cache).unwrap().launch(),
            LaunchShape::Workgroups(Workgroups::new([2, 2, 1], [128, 1, 1]).unwrap())
        );
        assert_eq!(append.control_bytes(), &words(&[40, 3, 4096, 16, 2])[..]);
        assert_eq!(
            append.ports(),
            &[
                Port::Input(1),
                Port::Input(2),
                Port::KvCache {
                    bytes,
                    written: true
                },
                Port::Parameter
            ]
        );
        assert_eq!(append.input_bytes(), &[0, 3 * 2 * 128 * 4, 3 * 2 * 128 * 4]);
        assert_eq!(append.output_bytes(), 0);
        // No rows, rows past the capacity, or a geometry the kernel was not
        // captured for are refused.
        assert!(Kernel::kv_append(0, 0, cache).is_none());
        assert!(Kernel::kv_append(1, 4096, cache).is_none());
        assert!(Kernel::kv_append(4096, 1, cache).is_none());
        // The last page is bound whole; one slot binds one page.
        assert_eq!(
            Kernel::kv_append(1, 4095, cache).unwrap().ports()[2],
            Port::KvCache {
                bytes: cache.storage_bytes().unwrap() as u64,
                written: true
            }
        );
        assert_eq!(
            Kernel::kv_append(1, 0, cache).unwrap().ports()[2],
            Port::KvCache {
                bytes: page,
                written: true
            }
        );
        let wide = KvCacheParams::new(
            ResourceKey {
                group: 0,
                binding: 0,
            },
            4,
            128,
            4096,
            16,
            KvCacheStorage::E4M3TokenScaleE8M0,
        )
        .unwrap();
        assert!(Kernel::kv_append(1, 0, wide).is_none());
        let two_slot_pages = KvCacheParams::new(
            ResourceKey {
                group: 0,
                binding: 0,
            },
            2,
            128,
            4096,
            2,
            KvCacheStorage::E4M3TokenScaleE8M0,
        )
        .unwrap();
        assert!(Kernel::kv_append(1, 0, two_slot_pages).is_none());

        // A read of two query rows over four splits: eight subgroups per
        // workgroup, one per (token, query head), the splits along Y, and
        // partials of (head_dim + 2) floats per (row, split).
        let attention = AttentionDecodeParams::new(cache, 32, 0.088_388_35).unwrap();
        let read = Kernel::kv_attention(2, 40, attention, 4).unwrap();
        assert_eq!(
            read.artifact().bytes(),
            kv_kernel(KvKernel::Attention, KvGeometry::NEMOTRON).unwrap().bytes()
        );
        assert_eq!(
            read.launch(),
            LaunchShape::Workgroups(Workgroups::new([8, 4, 1], [256, 1, 1]).unwrap())
        );
        assert_eq!(
            read.control_bytes(),
            &words(&[40, 2, 4096, 16, 4, attention.logit_scale.bits()])[..]
        );
        let partials = 2 * 32 * 4 * 130 * 4;
        assert_eq!(
            read.ports(),
            &[
                Port::Input(2),
                Port::KvCache {
                    bytes,
                    written: false
                },
                Port::Parameter,
                Port::Partials(partials),
                Port::Output
            ]
        );
        assert_eq!(read.input_bytes(), &[0, 0, 2 * 32 * 128 * 4]);
        assert_eq!(read.output_bytes(), 2 * 32 * 128 * 4);
        // Nine rows take two workgroups; no splits is refused.
        let nine = Kernel::kv_attention(9, 0, attention, 1).unwrap();
        assert_eq!(
            nine.launch(),
            LaunchShape::Workgroups(Workgroups::new([36, 1, 1], [256, 1, 1]).unwrap())
        );
        assert!(Kernel::kv_attention(2, 40, attention, 0).is_none());
        assert!(Kernel::kv_attention(1, 4096, attention, 1).is_none());

        // The reduction merges the same partials into the same output under
        // the same control block, and only exists for more than one split.
        let reduce = Kernel::kv_reduce(2, 40, attention, 4).unwrap();
        assert_eq!(
            reduce.artifact().bytes(),
            kv_kernel(KvKernel::Reduce, KvGeometry::NEMOTRON).unwrap().bytes()
        );
        assert_eq!(
            reduce.launch(),
            LaunchShape::Workgroups(Workgroups::new([8, 1, 1], [256, 1, 1]).unwrap())
        );
        assert_eq!(reduce.control_bytes(), read.control_bytes());
        assert_eq!(
            reduce.ports(),
            &[Port::Parameter, Port::Partials(partials), Port::Output]
        );
        assert_eq!(reduce.output_bytes(), read.output_bytes());
        assert!(Kernel::kv_reduce(2, 40, attention, 1).is_none());

        // The provider's rule: the last slot is the written one, a written
        // slot ahead of it is read-write, and the rest are read. The append
        // writes its cache in place and the read scribbles its partials.
        let place = |port: Port| match port {
            Port::KvCache { .. } => (7, 0),
            Port::Partials(_) => (8, 0),
            Port::Input(index) => (9, u64::from(index) << 20),
            Port::Parameter => (10, 0),
            _ => (11, 0),
        };
        let bound: Vec<_> = append.port_bindings(place).unwrap().iter().copied().collect();
        let modes: Vec<_> = bound.iter().map(|b| b.access).collect();
        assert_eq!(
            modes,
            [
                AccessMode::Read,
                AccessMode::Read,
                AccessMode::ReadWrite,
                AccessMode::ReadWrite
            ]
        );
        assert_eq!(bound[2].buffer, 7);
        assert_eq!(bound[2].len, bytes);
        let bound: Vec<_> = read.port_bindings(place).unwrap().iter().copied().collect();
        let modes: Vec<_> = bound.iter().map(|b| b.access).collect();
        assert_eq!(
            modes,
            [
                AccessMode::Read,
                AccessMode::Read,
                AccessMode::Read,
                AccessMode::ReadWrite,
                AccessMode::Write
            ]
        );
        assert_eq!(bound[3].len, partials);
    }

    #[test]
    fn f32_projections_admit_dynamic_shapes_with_checked_ports_and_accuracy() {
        use kore_xe_kernels::xve::{F32Projection, f32_projection};
        for (m, n, k) in [(1, 1, 1), (3, 17, 33), (8, 128, 2688), (311, 128, 2688)] {
            for (activation, artifact) in [
                (EpilogueActivation::None, F32Projection::Linear),
                (EpilogueActivation::Silu, F32Projection::Silu),
                (EpilogueActivation::Sigmoid, F32Projection::Sigmoid),
            ] {
                let request = KernelRequest::matmul_transposed_activated(
                    &desc([m, k], DType::F32),
                    &desc([n, k], DType::F32),
                    &desc([m, n], DType::F32),
                    0.5,
                    activation,
                )
                .unwrap();
                let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
                assert_eq!(kernel.artifact().bytes(), f32_projection(artifact).bytes());
                assert_eq!(
                    kernel.launch(),
                    LaunchShape::Workgroups(
                        Workgroups::new([n as u32, m as u32, 1], [32, 1, 1]).unwrap()
                    )
                );
                assert_eq!(
                    kernel.input_bytes(),
                    &[(m * k * 4) as u64, (n * k * 4) as u64]
                );
                assert_eq!(kernel.output_bytes(), (m * n * 4) as u64);
                assert_eq!(
                    kernel.ports(),
                    &[
                        Port::Input(0),
                        Port::Input(1),
                        Port::Parameter,
                        Port::Output
                    ]
                );
                assert_eq!(
                    kernel.parameter_bytes(),
                    &[0.5f32.to_le_bytes(), (k as u32).to_le_bytes()].concat()[..]
                );
                assert!(Kernel::select(&request, KernelAccuracy::Faithful).is_none());
                assert!(Kernel::select(&request, KernelAccuracy::Strict).is_none());
                let words = [0u32; 4];
                let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 16) };
                assert!(kernel.inputs_aligned([Some(&bytes[..4]), Some(&bytes[..4])]));
                assert!(!kernel.inputs_aligned([Some(&bytes[1..5]), Some(&bytes[..4])]));
                assert!(!kernel.inputs_aligned([Some(&bytes[..4]), Some(&bytes[1..5])]));
                let mut altered = request;
                altered.output_dtype = DType::B16;
                assert!(Kernel::select(&altered, KernelAccuracy::Approximate).is_none());
            }
        }
    }

    /// Runtime shape is immutable control data, not empty workgroups. The
    /// six ports include the shape trailer in the source binding and keep
    /// successor-state sizing separate from the source's logical payload.
    /// The gated delta rule lowers at its captured geometry to the six-slot
    /// recurrent contract: one workgroup per value head, the shape in the
    /// source-state trailer, a slot per token for verification, and the
    /// tree package (finer column slices, so wider groups) for a candidate
    /// tree. Another geometry, or a rule without its in-kernel L2 norm, has
    /// no package and stays on the CPU.
    #[test]
    fn the_delta_rule_lowers_at_its_captured_geometry() {
        use axnn::lowering::{GatedDeltaRuleParams, ResourceKey, StateBinding};
        let state = StateBinding {
            source: ResourceKey {
                group: 1,
                binding: 3,
            },
            target: ResourceKey {
                group: 1,
                binding: 4,
            },
        };
        let request = |tokens: usize, value_heads: u32, l2: bool| {
            let rule = GatedDeltaRuleParams::new(state, (16, 128), (value_heads, 128), l2).unwrap();
            let rows = value_heads as usize;
            KernelRequest::gated_delta_rule(
                &desc([tokens, rule.qkv_dim().unwrap()], DType::F32),
                &desc([tokens, rows], DType::F32),
                &desc([tokens, rows], DType::F32),
                rule,
            )
            .unwrap()
        };
        let kernel = Kernel::select(&request(4, 48, true), KernelAccuracy::Approximate).unwrap();
        let geometry = kore_xe_kernels::DeltaRuleGeometry::QWEN3_5_27B;
        assert_eq!(
            kernel.artifact().bytes(),
            kore_xe_kernels::delta_rule(DeltaRule::Runtime, geometry)
                .unwrap()
                .bytes()
        );
        let state_bytes = 48 * 128 * 128 * 4;
        assert!(kernel.is_recurrent());
        assert_eq!(
            kernel.ports(),
            &[
                Port::Input(0),
                Port::Input(1),
                Port::Input(2),
                Port::StateSource(state_bytes),
                Port::StateTarget(state_bytes),
                Port::Output,
            ]
        );
        assert_eq!(kernel.parameter_bytes(), [4, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(kernel.output_bytes(), 4 * 48 * 128 * 4);
        let LaunchShape::Workgroups(groups) = kernel.launch() else {
            panic!("explicit workgroups");
        };
        assert_eq!(
            (groups.group_count(), groups.local_size()),
            ([48, 1, 1], [512, 1, 1])
        );

        let per_token = kernel.with_state_slots(4).unwrap();
        assert_eq!(
            per_token.state_bytes(),
            Some((state_bytes, 4 * state_bytes))
        );
        // The banked form: its own package, each slot a page past the last
        // state, the same workgroups; one slot is the state alone.
        let banked = kernel.with_slot_bank(4).unwrap();
        assert_eq!(
            banked.artifact().bytes(),
            kore_xe_kernels::delta_rule(DeltaRule::Bank, geometry).unwrap().bytes()
        );
        assert_eq!(banked.parameter_bytes(), [4, 0, 0, 0, 4, 0, 0, 0]);
        let stride = state_bytes + kore_xe_kernels::BANK_SLOT_ROOM;
        assert_eq!(banked.state_bytes(), Some((state_bytes, 3 * stride + state_bytes)));
        assert_eq!(banked.launch(), kernel.launch());
        assert_eq!(kernel.with_slot_bank(1).unwrap().state_bytes(), Some((state_bytes, state_bytes)));
        assert!(kernel.with_slot_bank(3).is_none());
        // A run of the first two of its four rows banks two: the trailer
        // says so, the operands keep their rows, the launch its workgroups.
        let shallow = kernel.with_run_tokens(2).unwrap().with_slot_bank(2).unwrap();
        assert_eq!(shallow.parameter_bytes(), [2, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(shallow.artifact().bytes(), banked.artifact().bytes());
        assert_eq!(shallow.state_bytes(), Some((state_bytes, stride + state_bytes)));
        assert_eq!(shallow.output_bytes(), kernel.output_bytes());
        assert_eq!(shallow.launch(), kernel.launch());
        assert!(kernel.with_run_tokens(5).is_none() && kernel.with_run_tokens(0).is_none());
        let parents = [u32::MAX, 0, 0, 1];
        let tree = axnn::lowering::TokenTree::new(&parents).unwrap();
        let branching = per_token.with_token_tree(tree).unwrap();
        assert_eq!(
            branching.artifact().bytes(),
            kore_xe_kernels::delta_rule(DeltaRule::Tree, geometry)
                .unwrap()
                .bytes()
        );
        let LaunchShape::Workgroups(groups) = branching.launch() else {
            panic!("explicit workgroups");
        };
        assert_eq!(
            (groups.group_count(), groups.local_size()),
            ([48, 1, 1], [1024, 1, 1])
        );
        assert_eq!(
            branching.port_bytes(Port::StateSource(state_bytes)),
            state_bytes + 8 + 4 * 4
        );

        assert!(Kernel::select(&request(1, 16, true), KernelAccuracy::Approximate).is_none());
        assert!(Kernel::select(&request(1, 48, false), KernelAccuracy::Approximate).is_none());
    }

    /// A mixture-of-experts layer over expert tables lowers at its captured
    /// geometry: the route one workgroup a token, gate/up each slot's eight
    /// outputs a workgroup, down a token's sixteen rows; the tables bound
    /// whole and read as words. Another routing or width has no package and
    /// stays on the CPU.
    #[test]
    fn routed_experts_lower_at_their_captured_geometry() {
        use axnn::lowering::{MoeRouteParams, RoutedExpertsParams};
        let qwen = kore_xe_kernels::RoutedGeometry::QWEN3_6_35B_A3B;
        let route = |experts: u32, top_k: u32| MoeRouteParams::new(experts, top_k, 1).unwrap();
        let routed = |intermediate: u32| RoutedExpertsParams::new(257, 9, 2048, intermediate).unwrap();
        let workgroups = |kernel: &Kernel| {
            let LaunchShape::Workgroups(groups) = kernel.launch() else {
                panic!("explicit workgroups");
            };
            (groups.group_count(), groups.local_size())
        };
        let same = |kernel: &Kernel, kind: RoutedKernel| {
            kernel.artifact().same(kore_xe_kernels::routed_kernel(kind, qwen).unwrap())
        };
        let cube = |dims: [usize; 3], dtype| TensorDesc::try_new(Shape::from(dims), dtype).unwrap();
        let bytes = [0u8; 16];
        let skew = (4 - bytes.as_ptr() as usize % 4) % 4;
        let (aligned, misaligned) = (&bytes[skew..skew + 8], &bytes[skew + 2..skew + 10]);
        for tokens in [1_usize, 8] {
            let t = tokens as u32;
            let params = route(256, 8);
            let request =
                KernelRequest::moe_route(&desc([tokens, 257], DType::F32), params).unwrap();
            let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
            assert!(same(&kernel, RoutedKernel::Route));
            assert_eq!(kernel.ports(), &[Port::Input(0), Port::Output]);
            assert_eq!(kernel.output_bytes(), (tokens * 9 * 2 * 4) as u64);
            assert_eq!(workgroups(&kernel), ([t, 1, 1], [32, 1, 1]));

            let params = routed(512);
            let gate_up_table = desc([257, params.gate_up_record().bytes().unwrap()], DType::U8);
            let down_table = desc([257, params.down_record().bytes().unwrap()], DType::U8);
            let routes = cube([tokens, 9, 2], DType::I32);
            let request = KernelRequest::moe_gate_up_nvfp4(
                &desc([tokens, 2048], DType::F32),
                &gate_up_table,
                &routes,
                params,
            )
            .unwrap();
            let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
            assert!(same(&kernel, RoutedKernel::GateUp));
            let ports = [Port::Input(0), Port::Input(1), Port::Input(2), Port::Output];
            assert_eq!(kernel.ports(), &ports);
            assert_eq!(kernel.port_bytes(Port::Input(1)), 257 * 1_179_712);
            assert_eq!(kernel.output_bytes(), (tokens * 9 * 512 * 4) as u64);
            assert_eq!(workgroups(&kernel), ([64, 9 * t, 1], [128, 1, 1]));
            // The table is read as words: a misaligned view stays on the CPU.
            assert!(!kernel.inputs_aligned([None, Some(misaligned), None]));
            assert!(kernel.inputs_aligned([Some(misaligned), Some(aligned), None]));

            let request = KernelRequest::moe_down_nvfp4(
                &cube([tokens, 9, 512], DType::F32),
                &down_table,
                &routes,
                params,
            )
            .unwrap();
            let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
            assert!(same(&kernel, RoutedKernel::Down));
            assert_eq!(kernel.ports(), &ports);
            assert_eq!(kernel.output_bytes(), (tokens * 2048 * 4) as u64);
            assert_eq!(workgroups(&kernel), ([128, t, 1], [64, 1, 1]));
        }
        // Over a slot pool: the pooled package at the table's launch, the
        // pool bound through the last slot's record (the down projection's
        // from the first slot's down record), and the record count in the
        // control block between the routes and the output.
        let pooled = |kind: RoutedKernel| kore_xe_kernels::routed_pool_kernel(kind, qwen).unwrap();
        let params = routed(512);
        let request = KernelRequest::moe_gate_up_nvfp4(
            &desc([8, 2048], DType::F32),
            &desc([257, params.gate_up_record().bytes().unwrap()], DType::U8),
            &cube([8, 9, 2], DType::I32),
            params,
        )
        .unwrap();
        let table = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
        let pool = table.with_pool(100).unwrap();
        assert!(pool.artifact().same(pooled(RoutedKernel::GateUp)));
        let ports = [Port::Input(0), Port::Input(1), Port::Input(2), Port::Parameter, Port::Output];
        assert_eq!(pool.ports(), &ports);
        assert_eq!(pool.port_bytes(Port::Input(1)), 99 * 1_777_664 + 1_179_712);
        assert_eq!(pool.port_bytes(Port::Parameter), 8);
        assert_eq!(u32::from_le_bytes(pool.parameter[..4].try_into().unwrap()), 100);
        assert_eq!(workgroups(&pool), workgroups(&table));
        assert_eq!(pool.output_bytes(), table.output_bytes());
        assert!(pool.with_pool(100).is_none(), "a pool is not moved again");
        assert!(table.with_pool(0).is_none(), "an empty pool");
        let request = KernelRequest::moe_down_nvfp4(
            &cube([1, 9, 512], DType::F32),
            &desc([257, params.down_record().bytes().unwrap()], DType::U8),
            &cube([1, 9, 2], DType::I32),
            params,
        )
        .unwrap();
        let pool = Kernel::select(&request, KernelAccuracy::Approximate).unwrap().with_pool(1).unwrap();
        assert!(pool.artifact().same(pooled(RoutedKernel::Down)));
        assert_eq!(pool.port_bytes(Port::Input(1)), 589_888);
        let request = KernelRequest::moe_route(&desc([1, 257], DType::F32), route(256, 8)).unwrap();
        let route_kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
        assert!(route_kernel.with_pool(9).is_none(), "the route reads no expert");
        // Six picks, or 768-wide experts, were never captured.
        let request = KernelRequest::moe_route(&desc([1, 257], DType::F32), route(256, 6)).unwrap();
        assert!(Kernel::select(&request, KernelAccuracy::Approximate).is_none());
        let params = routed(768);
        let request = KernelRequest::moe_down_nvfp4(
            &cube([1, 9, 768], DType::F32),
            &desc([257, params.down_record().bytes().unwrap()], DType::U8),
            &cube([1, 9, 2], DType::I32),
            params,
        )
        .unwrap();
        assert!(Kernel::select(&request, KernelAccuracy::Approximate).is_none());
    }

    /// A convolution lowers at each captured width to that width's package
    /// (one invocation per channel, so 40 groups for the Qwen3.5 27B's
    /// query/key/value) and swaps in that width's tree package; a width
    /// without a capture stays on the CPU.
    #[test]
    fn a_convolution_lowers_to_the_package_of_its_width() {
        use axnn::lowering::{CausalConv1dParams, ResourceKey, StateBinding};
        let state = StateBinding {
            source: ResourceKey {
                group: 1,
                binding: 1,
            },
            target: ResourceKey {
                group: 1,
                binding: 2,
            },
        };
        let request = |channels: usize, tokens: usize| {
            let conv = CausalConv1dParams::new_activated(
                state,
                channels as u32,
                4,
                EpilogueActivation::Silu,
            )
            .unwrap();
            KernelRequest::causal_conv1d(
                &desc([tokens, channels], DType::F32),
                &desc([channels, 4], DType::F32),
                &TensorDesc::try_new(Shape::from([channels]), DType::F32).unwrap(),
                conv,
            )
            .unwrap()
        };
        for (channels, groups) in [(6144, 24), (10240, 40)] {
            let kernel = Kernel::select(&request(channels, 3), KernelAccuracy::Approximate)
                .unwrap()
                .with_state_slots(3)
                .unwrap();
            assert_eq!(
                kernel.artifact().bytes(),
                kore_xe_kernels::causal_conv(CausalConv::Runtime, channels as u32)
                    .unwrap()
                    .bytes()
            );
            let LaunchShape::Workgroups(launch) = kernel.launch() else {
                panic!("workgroups");
            };
            assert_eq!(
                (launch.group_count(), launch.local_size()),
                ([groups, 1, 1], [256, 1, 1])
            );
            let state_bytes = (channels * 3 * 4) as u64;
            assert_eq!(kernel.state_bytes(), Some((state_bytes, 3 * state_bytes)));
            let tree = axnn::lowering::TokenTree::new(&[u32::MAX, 0, 0]).unwrap();
            let branching = kernel.with_token_tree(tree).unwrap();
            assert_eq!(
                branching.artifact().bytes(),
                kore_xe_kernels::causal_conv(CausalConv::Tree, channels as u32)
                    .unwrap()
                    .bytes()
            );
            assert_eq!(branching.launch(), kernel.launch());
            // Only Qwen3.5's width has a banked package.
            let unbanked = Kernel::select(&request(channels, 3), KernelAccuracy::Approximate).unwrap();
            let banked = unbanked.with_slot_bank(3);
            assert_eq!(banked.is_some(), channels == 10240);
            if let Some(banked) = banked {
                assert_eq!(
                    banked.artifact().bytes(),
                    kore_xe_kernels::causal_conv(CausalConv::Bank, 10240).unwrap().bytes()
                );
                let stride = state_bytes + kore_xe_kernels::BANK_SLOT_ROOM;
                assert_eq!(banked.state_bytes(), Some((state_bytes, 2 * stride + state_bytes)));
                assert_eq!(banked.launch(), kernel.launch());
            }
        }
        assert!(Kernel::select(&request(4096, 1), KernelAccuracy::Approximate).is_none());
    }

    #[test]
    fn recurrent_packages_keep_shape_in_the_source_state_trailer() {
        use axnn::lowering::{CausalConv1dParams, ResourceKey, SsmScanParams, StateBinding};
        let state = StateBinding {
            source: ResourceKey {
                group: 1,
                binding: 0,
            },
            target: ResourceKey {
                group: 1,
                binding: 1,
            },
        };
        let scan = SsmScanParams::new(state, 64, 64, 8, 128, 0.0, f32::INFINITY).unwrap();
        let request = KernelRequest::ssm_scan(
            &desc([5, 6144], DType::F32),
            &desc([5, 64], DType::F32),
            &TensorDesc::try_new(Shape::from([3, 64]), DType::F32).unwrap(),
            scan,
        )
        .unwrap();
        let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
        let state_bytes = 4096 * 128 * 4;
        assert!(kernel.is_recurrent() && !kernel.legacy_layout());
        assert_eq!(kernel.state_bytes(), Some((state_bytes, state_bytes)));
        assert_eq!(
            kernel.ports(),
            &[
                Port::Input(0),
                Port::Input(1),
                Port::Input(2),
                Port::StateSource(state_bytes),
                Port::StateTarget(state_bytes),
                Port::Output,
            ]
        );
        assert_eq!(
            kernel.port_bytes(Port::StateSource(state_bytes)),
            state_bytes + 8
        );
        assert_eq!(kernel.parameter_bytes(), [5, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(kernel.output_bytes(), 5 * 4096 * 4);
        let LaunchShape::Workgroups(groups) = kernel.launch() else {
            panic!("explicit workgroups");
        };
        assert_eq!(
            (groups.group_count(), groups.local_size()),
            ([16, 1, 1], [256, 1, 1])
        );

        // A verification keeps one state per token; a prompt run one.
        let per_token = kernel.with_state_slots(5).unwrap();
        assert_eq!(per_token.parameter_bytes(), [5, 0, 0, 0, 5, 0, 0, 0]);
        assert!(kernel.with_state_slots(0).is_none());
        assert_eq!(
            per_token.with_state_slots(1).unwrap().parameter_bytes(),
            kernel.parameter_bytes()
        );
        let LaunchShape::Workgroups(groups) = per_token.launch() else {
            panic!("explicit workgroups");
        };
        assert_eq!(groups.group_count(), [16, 1, 1]);
        assert_eq!(
            per_token.state_bytes(),
            Some((state_bytes, 5 * state_bytes))
        );
        assert_eq!(
            kernel.with_state_slots(1).unwrap().state_bytes(),
            kernel.state_bytes()
        );
        assert!(kernel.with_state_slots(3).is_none());
        let parents = [u32::MAX, 0, 0, 1, 2];
        let tree = axnn::lowering::TokenTree::new(&parents).unwrap();
        assert!(kernel.with_token_tree(tree).is_none());
        let branching = per_token.with_token_tree(tree).unwrap();
        assert_eq!(branching.control_bytes().len(), 8 + 4 * parents.len());
        assert_eq!(
            branching.port_bytes(Port::StateSource(state_bytes)),
            state_bytes + 28
        );
        assert_eq!(branching.state_bytes(), per_token.state_bytes());
        assert_eq!(branching.launch(), per_token.launch());
        assert_eq!(
            branching.artifact().bytes(),
            kore_xe_kernels::mamba_scan_tree().unwrap().bytes()
        );
        for (word, parent) in branching.control_bytes()[8..]
            .chunks_exact(4)
            .zip(parents)
        {
            assert_eq!(u32::from_le_bytes(word.try_into().unwrap()), parent);
        }
        assert!(branching.with_state_slots(1).is_none());
        assert!(
            per_token
                .with_token_tree(axnn::lowering::TokenTree::new(&parents[..4]).unwrap())
                .is_none()
        );
        let mut next = 0x1000u64;
        let bindings = per_token
            .port_bindings(|port| {
                let here = next;
                next += per_token.port_bytes(port).next_multiple_of(64);
                (9, here)
            })
            .unwrap();
        assert!(bindings.iter().map(|binding| binding.access).eq([
            AccessMode::Read,
            AccessMode::Read,
            AccessMode::Read,
            AccessMode::Read,
            AccessMode::ReadWrite,
            AccessMode::Write,
        ]));

        let conv =
            CausalConv1dParams::new_activated(state, 6144, 4, EpilogueActivation::Silu).unwrap();
        let taps = desc([6144, 4], DType::F32);
        let bias = TensorDesc::try_new(Shape::from([6144]), DType::F32).unwrap();
        let request =
            KernelRequest::causal_conv1d(&desc([8, 6144], DType::F32), &taps, &bias, conv).unwrap();
        let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
        let LaunchShape::Workgroups(groups) = kernel.launch() else {
            panic!("explicit workgroups");
        };
        assert_eq!(groups.group_count(), [24, 1, 1]);
        assert_eq!(kernel.state_bytes(), Some((6144 * 3 * 4, 6144 * 3 * 4)));
        // Include short speculative prefixes and prompts above u8::MAX:
        // both shader readers must consume the entire LE u32 count.
        for tokens in [1usize, 3, 8, 17, 256, 512] {
            let requests = [
                KernelRequest::causal_conv1d(&desc([tokens, 6144], DType::F32), &taps, &bias, conv)
                    .unwrap(),
                KernelRequest::ssm_scan(
                    &desc([tokens, 6144], DType::F32),
                    &desc([tokens, 64], DType::F32),
                    &desc([3, 64], DType::F32),
                    scan,
                )
                .unwrap(),
            ];
            for (request, groups) in requests.iter().zip([24, 16]) {
                for slots in [1, tokens as u32] {
                    let kernel = Kernel::select(request, KernelAccuracy::Approximate)
                        .unwrap()
                        .with_state_slots(slots)
                        .unwrap();
                    assert_eq!(
                        u32::from_le_bytes(kernel.parameter_bytes()[..4].try_into().unwrap()),
                        tokens as u32
                    );
                    assert_eq!(
                        u32::from_le_bytes(kernel.parameter_bytes()[4..].try_into().unwrap()),
                        slots
                    );
                    let LaunchShape::Workgroups(launch) = kernel.launch() else {
                        panic!("workgroups");
                    };
                    assert_eq!(launch.group_count(), [groups, 1, 1]);
                    let (source, target) = kernel.state_bytes().unwrap();
                    assert_eq!(target, source * u64::from(slots));
                    assert_eq!(kernel.port_bytes(Port::StateSource(source)), source + 8);
                }
            }
        }
        assert!(recurrent_launch(6144, 0).is_none());
        // Without the folded SiLU the package computes something else.
        let plain = CausalConv1dParams::new(state, 6144, 4).unwrap();
        let request =
            KernelRequest::causal_conv1d(&desc([8, 6144], DType::F32), &taps, &bias, plain)
                .unwrap();
        assert!(Kernel::select(&request, KernelAccuracy::Approximate).is_none());
        // Nor does a pure package carry a state pair.
        let add = KernelRequest::binary(
            axnn::BinaryOp::Add,
            &desc([8, 2688], DType::F32),
            &desc([8, 2688], DType::F32),
        )
        .unwrap();
        let add = Kernel::select(&add, KernelAccuracy::Approximate).unwrap();
        assert!(!add.is_recurrent() && add.with_state_slots(1).is_none());
    }

    /// Every dense package has exactly one shape row, and only its exact
    /// weight dtype and epilogue select it.
    /// A package serves its own M; a runtime-M multi-row one serves any M
    /// above one (never a decode row, which has its own path), and reads it
    /// from word 1 of the parameter block.
    #[test]
    fn runtime_m_packages_serve_any_multi_row_width() {
        assert!(rows_fit(8, 8, false) && rows_fit(1, 1, true));
        assert!(!rows_fit(8, 256, false));
        assert!(rows_fit(8, 256, true) && rows_fit(8, 2, true));
        assert!(!rows_fit(8, 1, true) && !rows_fit(1, 256, true));
        // The parameter block is the control block's first two words; the
        // rest of the block is the resident KV kernels' and stays zero here.
        let control = projection_parameter(0.25, 256);
        assert_eq!(
            control[..8],
            [0.25f32.to_le_bytes(), 256u32.to_le_bytes()].concat()[..]
        );
        assert!(control[8..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn runtime_nvfp4_launch_rejects_wrapped_element_indices() {
        let max_rows = u32::MAX as usize / (131072 * core::mem::size_of::<f32>());
        assert!(projection_groups(max_rows, 131072, 2688, 32, 16, 8).is_some());
        assert!(projection_groups(max_rows + 1, 131072, 2688, 32, 16, 8).is_none());
        assert!(projection_groups(usize::MAX, 131072, 2688, 32, 16, 8).is_none());
        assert!(projection_groups(2, 1, usize::MAX, 32, 16, 8).is_none());
        assert!(projection_groups(16, 16, 16, 32, 16, 0).is_none());
    }

    #[test]
    fn projection_row_tile_follows_the_selected_package() {
        let nemotron = projection_groups(16, 131072, 2688, 32, 16, 8).unwrap();
        assert_eq!(nemotron.group_count(), [8192, 2, 1]);
        assert_eq!(projection_groups(17, 131072, 2688, 32, 16, 8).unwrap().group_count(), [8192, 3, 1]);
        let qwen = kore_xe_kernels::QwenProjection::DeltaGate;
        let n = qwen.n() as usize;
        let m16 = projection_groups(16, n, qwen.k() as usize,
            qwen.prefill16_group_width(), qwen.prefill16_group_columns(), 16).unwrap();
        assert_eq!(m16.group_count(), [n.div_ceil(qwen.prefill16_group_columns() as usize) as u32, 1, 1]);
        assert_eq!(projection_groups(8, n, qwen.k() as usize,
            qwen.rows_group_width(), qwen.rows_group_columns(), 8).unwrap().group_count()[1], 1);
    }

    #[test]
    fn dense_projection_table_names_each_kind_once_and_exactly() {
        for kind in DenseProjection::ALL {
            assert_eq!(
                DENSE_PROJECTIONS.iter().filter(|row| row.5 == kind).count(),
                1,
                "{kind:?}"
            );
            let &(m, n, k, dtype, activation, _) =
                DENSE_PROJECTIONS.iter().find(|row| row.5 == kind).unwrap();
            assert_eq!(
                dense_projection(m, n, k, dtype, activation),
                Some((kind, false))
            );
            // Any other multi-row M takes the runtime-M package; one row
            // (decode) has neither.
            assert_eq!(
                dense_projection(256, n, k, dtype, activation),
                Some((kind, true))
            );
            let other = if dtype == DType::B8 {
                DType::B16
            } else {
                DType::B8
            };
            // Two kinds may share a shape (Mamba out, attention out); the
            // other dtype never selects this one.
            assert!(
                dense_projection(m, n, k, other, activation).is_none_or(|(found, _)| found != kind)
            );
            let sigmoid = EpilogueActivation::Sigmoid;
            assert_eq!(dense_projection(m, n, k, dtype, sigmoid), None);
            assert_eq!(dense_projection(1, n, k, dtype, activation), None);
        }
    }

    /// The Mamba gate rows lower exactly as the checkpoint graph lowers them
    /// (a scaled, activated MATMUL_TRANSPOSED) and canonicalize to the
    /// request the dense table names; an unsealed package admits nothing.
    #[test]
    fn a_scaled_e4m3_projection_canonicalizes_to_its_dense_kind() {
        let (input, weights, output) = (
            desc([8, 2688], DType::F32),
            desc([4096, 2688], DType::B8),
            desc([8, 4096], DType::F32),
        );
        let request = KernelRequest::matmul_transposed_activated(
            &input,
            &weights,
            &output,
            0.0053,
            EpilogueActivation::Silu,
        )
        .unwrap();
        let canonical = Kernel::canonical(
            request.semantic,
            request.inputs(),
            request.result_tensor(0),
            request.params(),
        )
        .unwrap();
        assert_eq!(canonical, request);
        assert_eq!(
            dense_projection(8, 4096, 2688, DType::B8, EpilogueActivation::Silu),
            Some((DenseProjection::MambaGateSiluE4m3Prefill, false))
        );
        let kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
        assert_eq!(
            kernel.artifact().bytes(),
            kore_xe_kernels::dense_projection(DenseProjection::MambaGateSiluE4m3Prefill)
                .unwrap()
                .bytes()
        );
        // One 8x16 tile per workgroup, four K-splitting teams.
        assert_eq!(
            kernel.launch(),
            LaunchShape::Workgroups(Workgroups::new([256, 1, 1], [128, 1, 1]).unwrap())
        );
        assert_eq!(
            kernel.ports(),
            &[
                Port::Input(0),
                Port::Input(1),
                Port::Parameter,
                Port::Output
            ]
        );
        assert_eq!(kernel.parameter_bytes()[..4], 0.0053f32.to_le_bytes());
        assert_eq!(kernel.input_bytes(), &[8 * 2688 * 4, 4096 * 2688]);
        assert_eq!(kernel.output_bytes(), 8 * 4096 * 4);
        // The weights are read as words; the activation may sit anywhere.
        let words = [0u32; 4];
        let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 16) };
        assert!(kernel.inputs_aligned([Some(&bytes[1..5]), Some(&bytes[..8])]));
        assert!(!kernel.inputs_aligned([Some(&bytes[..8]), Some(&bytes[2..10])]));
        assert!(Kernel::select(&request, KernelAccuracy::Faithful).is_none());
    }

    /// Only the inputs a package reads as words must be 4-byte aligned.
    #[test]
    fn word_inputs_must_be_aligned() {
        let bytes = [0u8; 16];
        let (aligned, misaligned) = if (bytes.as_ptr() as usize).is_multiple_of(4) {
            (&bytes[..8], &bytes[2..10])
        } else {
            let skew = 4 - bytes.as_ptr() as usize % 4;
            (&bytes[skew..skew + 8], &bytes[skew + 2..skew + 10])
        };
        let request = KernelRequest::binary(
            axnn::BinaryOp::Add,
            &TensorDesc::try_new(Shape::from([2]), DType::F32).unwrap(),
            &TensorDesc::try_new(Shape::from([2]), DType::F32).unwrap(),
        )
        .unwrap();
        let mut kernel = Kernel::select(&request, KernelAccuracy::Approximate).unwrap();
        // No word input: any alignment binds.
        assert!(kernel.inputs_aligned([Some(misaligned), Some(misaligned)]));
        kernel.word_inputs = 1 << 1;
        assert!(kernel.inputs_aligned([Some(misaligned), Some(aligned)]));
        assert!(!kernel.inputs_aligned([Some(aligned), Some(misaligned)]));
        // A pending (device-resident) operand has no host view to check.
        assert!(kernel.inputs_aligned([Some(misaligned), None]));
    }

    /// A Qwen RoPE request must reach `from_request` through `select`: the
    /// canonical form once had no Rope arm, so every ROPE node of the Qwen
    /// decode graph was refused. Query (64 heads) and key (8 heads) rotate
    /// with their own captures; any other head count stays unlowered.
    #[test]
    fn qwen_rope_selects_its_captures_through_the_canonical_form() {
        let rotary = desc([1, 64], DType::F32);
        let rope = |heads: usize| {
            let x = TensorDesc::try_new(Shape::from([1, heads, 128]), DType::F32).unwrap();
            KernelRequest::rope(&x, &rotary, &rotary, axnn::RopeMode::NeoX).unwrap()
        };
        for (heads, kind) in [
            (64, kore_xe_kernels::QwenRope::Query),
            (8, kore_xe_kernels::QwenRope::Key),
        ] {
            let kernel = Kernel::select(&rope(heads), KernelAccuracy::Approximate)
                .expect("a captured Qwen RoPE");
            assert_eq!(
                kernel.artifact().bytes(),
                kore_xe_kernels::qwen_rope(kind).unwrap().bytes()
            );
            assert_eq!(
                kernel.ports(),
                &[Port::Input(0), Port::Input(1), Port::Input(2), Port::Output][..]
            );
            let LaunchShape::Workgroups(groups) = kernel.launch() else {
                panic!("a RoPE launches workgroups");
            };
            assert_eq!(groups.group_count(), [heads as u32, 1, 1]);
            assert_eq!(groups.local_size(), [64, 1, 1]);
        }
        assert!(Kernel::select(&rope(32), KernelAccuracy::Approximate).is_none());
        // Qwen3's captures rotate whole heads: a partial rotation of the same
        // heads (phase rows for its quarter) must not reach them.
        let quarter = desc([1, 16], DType::F32);
        let x = TensorDesc::try_new(Shape::from([1, 64, 128]), DType::F32).unwrap();
        let partial =
            KernelRequest::rope_partial(&x, &quarter, &quarter, axnn::RopeMode::NeoX, 32).unwrap();
        assert!(Kernel::select(&partial, KernelAccuracy::Approximate).is_none());
        // Qwen3.5's rotate the leading 64 of each 256-wide head, two tokens
        // here; the whole-head rotation of those heads is not captured.
        let phase = desc([2, 32], DType::F32);
        for (heads, kind) in [
            (24, kore_xe_kernels::QwenRope::GatedQuery),
            (4, kore_xe_kernels::QwenRope::GatedKey),
        ] {
            let x = TensorDesc::try_new(Shape::from([2, heads, 256]), DType::F32).unwrap();
            let request =
                KernelRequest::rope_partial(&x, &phase, &phase, axnn::RopeMode::NeoX, 64).unwrap();
            let kernel = Kernel::select(&request, KernelAccuracy::Approximate)
                .expect("a captured Qwen3.5 RoPE");
            assert_eq!(
                kernel.artifact().bytes(),
                kore_xe_kernels::qwen_rope(kind).unwrap().bytes()
            );
            let LaunchShape::Workgroups(groups) = kernel.launch() else {
                panic!("a RoPE launches workgroups");
            };
            assert_eq!(groups.group_count(), [2 * heads as u32, 1, 1]);
            let whole = desc([2, 128], DType::F32);
            let request = KernelRequest::rope(&x, &whole, &whole, axnn::RopeMode::NeoX).unwrap();
            assert!(Kernel::select(&request, KernelAccuracy::Approximate).is_none());
        }
    }

    /// Qwen3.8-27B's decode ops reach their captures: every projection at its
    /// (N, K, fused epilogue) and team width, the 248320-row head and
    /// embedding, the q/gate split's F32 row gather, the 256- and 128-wide
    /// head norms, the softplus and sigmoid maps, and the four-KV-head cache
    /// with six query rows per eight-subgroup read workgroup.
    #[test]
    fn qwen3_5_decode_ops_lower_to_their_captures() {
        use axnn::lowering::{AttentionDecodeParams, KvCacheParams, KvCacheStorage, ResourceKey};
        use kore_xe_kernels::{KvGeometry, KvKernel, QwenProjection, QwenRms, kv_kernel, xve::F32Map};
        let lower = |request: &KernelRequest| Kernel::select(request, KernelAccuracy::Approximate);
        let groups = |count: [u32; 3], width: u32| {
            LaunchShape::Workgroups(Workgroups::new(count, [width, 1, 1]).unwrap())
        };
        let rank3 = |dims: [usize; 3]| TensorDesc::try_new(Shape::from(dims), DType::F32).unwrap();
        let rank1 = |dims: [usize; 1], dtype| TensorDesc::try_new(Shape::from(dims), dtype).unwrap();
        let rows_projection = |m: usize, n: usize, k: usize, activation| {
            KernelRequest::matmul_nvfp4_transposed(
                &desc([m, k], DType::F32),
                &desc([n, k / 2], DType::U8),
                &desc([n, k / 16], DType::B8),
                &desc([m, n], DType::F32),
                0.375,
                activation,
            )
            .unwrap()
        };
        let projection = |n: usize, k: usize, activation| rows_projection(1, n, k, activation);
        for kind in QwenProjection::QWEN3_5 {
            let (n, k) = (kind.n() as usize, kind.k() as usize);
            let activation = match kind.epilogue() {
                0 => EpilogueActivation::None,
                1 => EpilogueActivation::Silu,
                _ => EpilogueActivation::Sigmoid,
            };
            let kernel = lower(&projection(n, k, activation)).expect("a captured projection");
            assert_eq!(
                kernel.artifact().bytes(),
                kore_xe_kernels::qwen_projection(kind).unwrap().bytes(),
                "{kind:?}"
            );
            assert_eq!(
                kernel.launch(),
                groups([n.div_ceil(16) as u32, 1, 1], kind.group_width()),
                "{kind:?}"
            );
        }
        // A captured shape with an epilogue it was not built with stays off.
        assert!(lower(&projection(10240, 5120, EpilogueActivation::Silu)).is_none());
        assert!(lower(&projection(48, 5120, EpilogueActivation::Silu)).is_none());
        // A verification's rows, two to eight, reach the multi-row packages
        // at their own workgroups (sixteen columns a subgroup), the count in
        // the parameter block; nine stay off.
        for kind in QwenProjection::ROWS {
            let (n, k) = (kind.n() as usize, kind.k() as usize);
            let activation = match kind.epilogue() {
                0 => EpilogueActivation::None,
                1 => EpilogueActivation::Silu,
                _ => EpilogueActivation::Sigmoid,
            };
            for m in 2..=QWEN_ROWS {
                let kernel = lower(&rows_projection(m, n, k, activation)).expect("a multi-row projection");
                assert_eq!(
                    kernel.artifact().bytes(),
                    kore_xe_kernels::qwen_projection_rows(kind).unwrap().bytes(),
                    "{kind:?}"
                );
                let columns = kind.rows_group_columns() as usize;
                assert_eq!(
                    kernel.launch(),
                    groups([n.div_ceil(columns) as u32, 1, 1], kind.rows_group_width()),
                    "{kind:?}"
                );
                assert_eq!(kernel.parameter_bytes()[4..8], (m as u32).to_le_bytes(), "{kind:?} at {m}");
            }
            assert!(lower(&rows_projection(QWEN_ROWS + 1, n, k, activation)).is_none(), "{kind:?}");
            if !kore_xe_kernels::QWEN_PREFILL16_NATIVE {
                assert!(lower(&rows_projection(16, n, k, activation)).is_none(),
                    "M16 source alone must not admit {kind:?}");
            } else {
                let kernel = lower(&rows_projection(16, n, k, activation))
                    .expect("a sealed M16 projection");
                assert_eq!(kernel.artifact().bytes(),
                    kore_xe_kernels::qwen_projection_prefill16(kind).unwrap().bytes());
                assert_eq!(kernel.launch(),
                    groups([n.div_ceil(kind.prefill16_group_columns() as usize) as u32, 1, 1],
                        kind.prefill16_group_width()), "M16 package writes all sixteen rows");
            }
        }
        // The prediction head's NVFP4 draft head, over the vocabulary's
        // prefix, is one row only; the whole vocabulary has no NVFP4 head.
        let draft = kore_xe_kernels::QWEN3_5_DRAFT_VOCAB as usize;
        let draft_head = lower(&projection(draft, 5120, EpilogueActivation::None)).expect("the draft head");
        assert_eq!(
            draft_head.artifact().bytes(),
            kore_xe_kernels::qwen_projection(QwenProjection::DraftHead).unwrap().bytes()
        );
        assert!(lower(&rows_projection(QWEN_ROWS, draft, 5120, EpilogueActivation::None)).is_none());
        assert!(lower(&projection(248320, 5120, EpilogueActivation::None)).is_none());

        let head = KernelRequest::matmul_transposed_activated(
            &desc([1, 5120], DType::F32),
            &desc([248320, 5120], DType::B16),
            &desc([1, 248320], DType::F32),
            1.0,
            EpilogueActivation::None,
        )
        .unwrap();
        let kernel = lower(&head).expect("the 27B lm_head");
        assert_eq!(
            kernel.artifact().bytes(),
            kore_xe_kernels::qwen_lm_head(248320, 5120).unwrap().bytes()
        );
        assert_eq!(kernel.launch(), groups([248320 / 16, 1, 1], 128));
        let rows_head = |m: usize| {
            KernelRequest::matmul_transposed_activated(
                &desc([m, 5120], DType::F32),
                &desc([248320, 5120], DType::B16),
                &desc([m, 248320], DType::F32),
                1.0,
                EpilogueActivation::None,
            )
            .unwrap()
        };
        for m in [2, 4, QWEN_ROWS] {
            let kernel = lower(&rows_head(m)).expect("the 27B lm_head over a verification's rows");
            assert_eq!(
                kernel.artifact().bytes(),
                kore_xe_kernels::qwen_lm_head_rows(248320, 5120).unwrap().bytes()
            );
            let (lanes, columns) = kore_xe_kernels::qwen_lm_head_rows_geometry(5120);
            assert_eq!(kernel.launch(), groups([248320 / columns, 1, 1], lanes));
        }
        assert!(lower(&rows_head(QWEN_ROWS + 1)).is_none());
        if !kore_xe_kernels::QWEN_PREFILL16_NATIVE {
            assert!(lower(&rows_head(16)).is_none(), "no sealed M16 BF16 head");
        }

        // The device's greedy pick a frozen decode chains its drafts with.
        let pick = Kernel::top1(4, 248320).expect("the 248320-row top-1");
        assert_eq!(
            pick.artifact().bytes(),
            kore_xe_kernels::qwen_top1(248320).unwrap().bytes()
        );
        assert_eq!(pick.ports(), &[Port::Input(0), Port::Discard(16), Port::Output]);
        assert_eq!((pick.port_bytes(Port::Input(0)), pick.output_bytes()), (4 * 248320 * 4, 16));
        assert_eq!(pick.launch(), groups([4, 1, 1], 256));
        assert!(Kernel::top1(0, 248320).is_none() && Kernel::top1(1, 131072).is_none());

        let ids = rank1([3], DType::I32);
        let embed = KernelRequest::gather(&desc([248320, 5120], DType::B16), &ids).unwrap();
        assert_eq!(
            lower(&embed).expect("the 27B embedding").artifact().bytes(),
            kore_xe_kernels::qwen_gather(248320, 5120).unwrap().bytes()
        );
        let order = rank1([48], DType::I32);
        let split = KernelRequest::gather(&desc([48, 256], DType::F32), &order).unwrap();
        let kernel = lower(&split).expect("the q/gate split");
        assert_eq!(
            kernel.artifact().bytes(),
            kore_xe_kernels::f32_row_gather(48, 256).unwrap().bytes()
        );
        assert_eq!(kernel.launch(), groups([48, 1, 1], 256));
        let wider = KernelRequest::gather(&desc([96, 256], DType::F32), &rank1([96], DType::I32));
        assert!(lower(&wider.unwrap()).is_none(), "two tokens' rows are not captured");
        let rows = 48 * QWEN_ROWS;
        let eight = KernelRequest::gather(&desc([rows, 256], DType::F32), &rank1([rows], DType::I32)).unwrap();
        let kernel = lower(&eight).expect("a verification's q/gate split");
        assert_eq!(
            kernel.artifact().bytes(),
            kore_xe_kernels::f32_row_gather(rows as u32, 256).unwrap().bytes()
        );
        assert_eq!(kernel.launch(), groups([rows as u32, 1, 1], 256));
        let four = KernelRequest::gather(&desc([192, 256], DType::F32), &rank1([192], DType::I32));
        assert!(lower(&four.unwrap()).is_none(), "four tokens' rows are no longer captured");

        for (dims, width, kind) in [
            ([1, 24, 256], 256, QwenRms::GatedHeadQk),
            ([1, 4, 256], 256, QwenRms::GatedHeadQk),
            ([1, 48, 128], 128, QwenRms::HeadQk),
        ] {
            let norm =
                KernelRequest::rms_norm_over(&rank3(dims), &rank1([width], DType::F32), 1e-6, 2).unwrap();
            let kernel = lower(&norm).expect("a captured head norm");
            assert_eq!(kernel.artifact().bytes(), kore_xe_kernels::qwen_rms(kind).unwrap().bytes());
            assert_eq!(kernel.launch(), groups([dims[1] as u32, 1, 1], 256));
        }

        for (op, map) in [
            (axnn::UnaryOp::Softplus, F32Map::Softplus),
            (axnn::UnaryOp::Sigmoid, F32Map::Sigmoid),
        ] {
            let kernel = lower(&KernelRequest::unary(op, &desc([1, 6144], DType::F32))).unwrap();
            assert_eq!(kernel.artifact().bytes(), kore_xe_kernels::xve::f32_map(map).bytes());
        }

        let cache = KvCacheParams::new(
            ResourceKey {
                group: 4,
                binding: 0,
            },
            4,
            256,
            8192,
            16,
            KvCacheStorage::E4M3TokenScaleE8M0,
        )
        .unwrap();
        let geometry = KvGeometry::QWEN3_5;
        let append = Kernel::kv_append(1, 100, cache).unwrap();
        assert_eq!(
            append.artifact().bytes(),
            kv_kernel(KvKernel::Append, geometry).unwrap().bytes()
        );
        assert_eq!(append.launch(), groups([1, 4, 1], 256));
        let read = AttentionDecodeParams::new(cache, 24, 0.0625).unwrap();
        let attention = Kernel::kv_attention(1, 100, read, 3).unwrap();
        assert_eq!(
            attention.artifact().bytes(),
            kv_kernel(KvKernel::Attention, geometry).unwrap().bytes()
        );
        // 24 query rows, six per workgroup of eight subgroups.
        assert_eq!(attention.launch(), groups([4, 3, 1], 256));
        let reduce = Kernel::kv_reduce(1, 100, read, 3).unwrap();
        assert_eq!(reduce.launch(), groups([4, 1, 1], 256));
        assert_eq!(
            reduce.artifact().bytes(),
            kv_kernel(KvKernel::Reduce, geometry).unwrap().bytes()
        );
        // Partials of (head_dim + 2) floats per (row, split).
        assert_eq!(attention.ports()[3], Port::Partials(24 * 3 * 258 * 4));
        // The package bakes 24 query heads: any other count stays off.
        let other = AttentionDecodeParams::new(cache, 16, 0.0625).unwrap();
        assert!(Kernel::kv_attention(1, 100, other, 1).is_none());
    }

    /// A fill writes its words after its control block, a word per lane.
    /// Qwen3.6-35B-A3B's decode at its 2048-wide geometry lowers to its own
    /// captures at one row and at a verification's eight: the projections,
    /// the BF16 head, router and gate inputs, the layer norm, the embedding,
    /// the rotations, the two-KV-head cache, the convolution and the delta
    /// rule.
    #[test]
    fn qwen3_5_moe_decode_ops_lower_to_their_captures() {
        let lower = |request: &KernelRequest| Kernel::select(request, KernelAccuracy::Approximate);
        let groups = |count: [u32; 3], lanes: u32| {
            LaunchShape::Workgroups(Workgroups::new(count, [lanes, 1, 1]).unwrap())
        };
        let rank1 = |shape: [usize; 1], dtype| TensorDesc::try_new(Shape::from(shape), dtype).unwrap();
        let same = |kernel: &Kernel, artifact: XeKernelArtifact| kernel.artifact().same(artifact);
        let nvfp4 = |m: usize, n: usize, k: usize, activation| {
            KernelRequest::matmul_nvfp4_transposed(
                &desc([m, k], DType::F32),
                &desc([n, k / 2], DType::U8),
                &desc([n, k / 16], DType::B8),
                &desc([m, n], DType::F32),
                0.25,
                activation,
            )
            .unwrap()
        };
        for kind in QwenProjection::QWEN3_5_MOE.into_iter().chain([QwenProjection::MoeMtpFc]) {
            let (n, k) = (kind.n() as usize, kind.k() as usize);
            let activation = if kind.epilogue() == 1 { EpilogueActivation::Silu } else { EpilogueActivation::None };
            let one = lower(&nvfp4(1, n, k, activation)).unwrap();
            assert!(same(&one, kore_xe_kernels::qwen_projection(kind).unwrap()), "{kind:?}");
            let rows = lower(&nvfp4(8, n, k, activation)).unwrap();
            assert!(same(&rows, kore_xe_kernels::qwen_projection_rows(kind).unwrap()), "{kind:?}");
            let LaunchShape::Workgroups(groups) = rows.launch() else { panic!() };
            assert_eq!(groups.group_count()[0], (n as u32).div_ceil(kind.rows_group_columns()), "{kind:?}");
        }
        let draft = lower(&nvfp4(1, 98304, 2048, EpilogueActivation::None)).unwrap();
        assert!(same(&draft, kore_xe_kernels::qwen_projection(QwenProjection::MoeDraftHead).unwrap()));
        let dense = |m: usize, n: usize, activation| {
            KernelRequest::matmul_transposed_activated(
                &desc([m, 2048], DType::F32),
                &desc([n, 2048], DType::B16),
                &desc([m, n], DType::F32),
                1.0,
                activation,
            )
            .unwrap()
        };
        let head = lower(&dense(1, 248320, EpilogueActivation::None)).unwrap();
        assert!(same(&head, kore_xe_kernels::qwen_lm_head(248320, 2048).unwrap()));
        assert_eq!(head.launch(), groups([248320 / 16, 1, 1], 256));
        let head = lower(&dense(8, 248320, EpilogueActivation::None)).unwrap();
        assert!(same(&head, kore_xe_kernels::qwen_lm_head_rows(248320, 2048).unwrap()));
        assert_eq!(head.launch(), groups([248320 / 128, 1, 1], 512));
        for (kind, activation) in [
            (QwenDense::MoeRouter, EpilogueActivation::None),
            (QwenDense::MoeStep, EpilogueActivation::None),
            (QwenDense::MoeBeta, EpilogueActivation::Sigmoid),
        ] {
            let n = kind.n() as usize;
            let one = lower(&dense(1, n, activation)).unwrap();
            assert!(same(&one, kore_xe_kernels::qwen_dense(kind).unwrap()), "{kind:?}");
            assert_eq!(one.launch(), groups([(n as u32).div_ceil(16), 1, 1], 128), "{kind:?}");
            let rows = lower(&dense(3, n, activation)).unwrap();
            assert!(same(&rows, kore_xe_kernels::qwen_dense_rows(kind).unwrap()), "{kind:?}");
            assert_eq!(rows.parameter_bytes()[4..8], 3u32.to_le_bytes(), "{kind:?}");
        }
        // A gate input without its sigmoid is the step's package, never the beta's.
        assert!(same(&lower(&dense(1, 32, EpilogueActivation::None)).unwrap(),
                     kore_xe_kernels::qwen_dense(QwenDense::MoeStep).unwrap()));
        assert!(lower(&dense(1, 32, EpilogueActivation::Silu)).is_none());
        let norm = KernelRequest::rms_norm_over(&desc([8, 2048], DType::F32), &rank1([2048], DType::F32), 1e-6, 1)
            .unwrap();
        assert!(same(&lower(&norm).unwrap(), kore_xe_kernels::qwen_rms(QwenRms::MoeLayer).unwrap()));
        let ids = rank1([8], DType::I32);
        let embed = KernelRequest::gather(&desc([248320, 2048], DType::B16), &ids).unwrap();
        assert!(same(&lower(&embed).unwrap(), kore_xe_kernels::qwen_gather(248320, 2048).unwrap()));
    }

    #[test]
    fn a_fill_binds_its_control_block_then_its_words() {
        let fill = Kernel::fill(2, 1).unwrap();
        assert!(fill.artifact().same(kore_xe_kernels::fill_u32()));
        assert_eq!(fill.ports(), &[Port::Parameter, Port::Output][..]);
        assert!(fill.legacy_layout());
        assert_eq!(fill.control_bytes(), &[2, 0, 0, 0, 1, 0, 0, 0][..]);
        assert_eq!(fill.port_bytes(Port::Output), 8);
        let LaunchShape::Workgroups(groups) = fill.launch() else {
            panic!("a fill launches workgroups");
        };
        assert_eq!((groups.group_count(), groups.local_size()), ([1, 1, 1], [256, 1, 1]));
        let LaunchShape::Workgroups(groups) = Kernel::fill(786_433, 0).unwrap().launch() else {
            panic!("a fill launches workgroups");
        };
        assert_eq!(groups.group_count(), [3073, 1, 1]);
        assert!(Kernel::fill(0, 0).is_none());
    }
}
