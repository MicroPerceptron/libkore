//! Automatic `axnn` semantic lowering into provider-neutral TOSA recipes.
//!
//! `.ax` deliberately carries only semantic opcodes. This module is the TOSA
//! provider's allocation-free implementation catalog: it consumes the same
//! [`ax::node_kernel_request`] used by native CPU lowering, chooses a legal
//! primitive recipe through `axnn`, and keeps every provider-specific detail
//! out of the serialized graph.

use ax::{DType, Graph, NodeView, OpCode, Partition, TensorFlags, TensorId, TensorView};
use axnn::lowering::{
    CandidateVisitor, DTypeSet, KernelAccuracy, KernelCandidate, KernelCatalog, KernelConstraints,
    KernelCost, KernelPlan, KernelPolicy, KernelRequest, KernelShapePolicy, LogicalOp, SemanticOp,
    select_kernel,
};
use axnn::{BinaryOp, CompareOp, TensorDesc, TernaryOp, UnaryOp};
use virtio_accel_tosa::{CapabilityDescriptor, DType as TosaDType, Op as TosaOp, ValueRoles};
use virtio_accel_tosa_build::OperatorKind;

/// Dtype surface exposed by one TOSA execution lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TosaProfile {
    /// The in-tree `axtosa` CPU provider, including its storage-only types.
    Cpu,
    /// The conservative cross-provider accelerator surface.
    Accelerator,
}

/// Provider-neutral implementation of one semantic `.ax` operation.
///
/// Recipes are private implementation choices. Adding a better native TOSA
/// operator or another legal expansion never consumes an `.ax` opcode or a
/// scheduler implementation ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AxTosaRecipe {
    Direct(OperatorKind),
    /// Floating-point division as `lhs * reciprocal(rhs)`; TOSA's native
    /// division is integer-only.
    Divide,
    /// Square root as `reciprocal(rsqrt(x))`, preserving zero and infinity
    /// behavior better than `x * rsqrt(x)`.
    SquareRoot,
    /// Dynamic three-input clamp as `minimum(maximum(x, lo), hi)`.
    Clamp,
    /// WGSL permits `fma` to execute as multiply followed by add.
    Fma,
    /// `pow(2, x)` with a synthesized scalar base.
    Exp2,
    /// `maximum(x, 0)` with a synthesized scalar zero: RELU stays inside the
    /// provider partition instead of splitting it around a host hop.
    Relu,
    /// `log(x) * (1 / ln(2))`.
    Log2,
    /// `sin(x) * reciprocal(cos(x))`.
    Tan,
}

impl AxTosaRecipe {
    pub(crate) const fn operator_count(self) -> u64 {
        match self {
            Self::Direct(_) | Self::Exp2 | Self::Relu => 1,
            Self::Divide | Self::SquareRoot | Self::Clamp | Self::Fma | Self::Log2 => 2,
            Self::Tan => 4,
        }
    }

    fn visit_operators(self, mut visit: impl FnMut(TosaOp)) {
        match self {
            Self::Direct(kind) => visit(kind.op()),
            Self::Divide => {
                visit(TosaOp::RECIPROCAL);
                visit(TosaOp::MUL);
            }
            Self::SquareRoot => {
                visit(TosaOp::RSQRT);
                visit(TosaOp::RECIPROCAL);
            }
            Self::Clamp => {
                visit(TosaOp::MAXIMUM);
                visit(TosaOp::MINIMUM);
            }
            Self::Fma => {
                visit(TosaOp::MUL);
                visit(TosaOp::ADD);
            }
            Self::Exp2 => visit(TosaOp::POW),
            Self::Relu => visit(TosaOp::MAXIMUM),
            Self::Log2 => {
                visit(TosaOp::LOG);
                visit(TosaOp::MUL);
            }
            Self::Tan => {
                visit(TosaOp::SIN);
                visit(TosaOp::COS);
                visit(TosaOp::RECIPROCAL);
                visit(TosaOp::MUL);
            }
        }
    }

    /// Visit every value synthesized by artifact authoring. Repeated visits
    /// are intentional: they mirror actual tensors (for example, TAN has
    /// three intermediates and MATMUL has two zero-point constants).
    fn visit_synthetic_values(
        self,
        data_dtype: TosaDType,
        mut visit: impl FnMut(TosaDType, ValueRoles),
    ) {
        match self {
            Self::Direct(OperatorKind::MatMul | OperatorKind::Negate) => {
                visit(data_dtype, ValueRoles::CONSTANT);
                visit(data_dtype, ValueRoles::CONSTANT);
            }
            Self::Direct(OperatorKind::Mul) => {
                visit(TosaDType::INT8, ValueRoles::CONSTANT);
            }
            Self::Direct(_) => {}
            Self::Divide => {
                visit(data_dtype, ValueRoles::INTERMEDIATE);
                visit(TosaDType::INT8, ValueRoles::CONSTANT);
            }
            Self::SquareRoot | Self::Clamp => {
                visit(data_dtype, ValueRoles::INTERMEDIATE);
            }
            Self::Fma => {
                visit(data_dtype, ValueRoles::INTERMEDIATE);
                visit(TosaDType::INT8, ValueRoles::CONSTANT);
            }
            Self::Exp2 | Self::Relu => visit(data_dtype, ValueRoles::CONSTANT),
            Self::Log2 => {
                visit(data_dtype, ValueRoles::INTERMEDIATE);
                visit(data_dtype, ValueRoles::CONSTANT);
                visit(TosaDType::INT8, ValueRoles::CONSTANT);
            }
            Self::Tan => {
                visit(data_dtype, ValueRoles::INTERMEDIATE);
                visit(data_dtype, ValueRoles::INTERMEDIATE);
                visit(data_dtype, ValueRoles::INTERMEDIATE);
                visit(TosaDType::INT8, ValueRoles::CONSTANT);
            }
        }
    }

    pub(crate) const fn needs_i8_shift(self) -> bool {
        matches!(
            self,
            Self::Direct(OperatorKind::Mul) | Self::Divide | Self::Fma | Self::Log2 | Self::Tan
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct TosaCatalog {
    profile: TosaProfile,
}

impl KernelCatalog for TosaCatalog {
    type Implementation = AxTosaRecipe;

    fn visit_candidates<V: CandidateVisitor<Self::Implementation>>(
        &self,
        request: &KernelRequest,
        visitor: &mut V,
    ) {
        if request.semantic == SemanticOp::MatMul
            && !matmul_request_supported(request, self.profile)
        {
            return;
        }
        let Some(recipe) = semantic_recipe(request.semantic) else {
            return;
        };
        let Some(constraints) = semantic_constraints(request.semantic, self.profile) else {
            return;
        };
        let temporary_bytes = recipe_temporary_bytes(recipe, request);
        visitor.visit(KernelCandidate {
            semantic: request.semantic,
            implementation: recipe,
            required_capabilities: axnn::lowering::CapabilityMask::NONE,
            constraints,
            accuracy: KernelAccuracy::Faithful,
            cost: KernelCost::new(0, recipe.operator_count()).with_temporary_bytes(temporary_bytes),
        });
    }
}

/// Select the cheapest legal TOSA implementation of a canonical request.
pub(crate) fn select_tosa_recipe(
    request: &KernelRequest,
    profile: TosaProfile,
) -> Option<KernelPlan<AxTosaRecipe>> {
    select_kernel(
        &TosaCatalog { profile },
        request,
        KernelPolicy::new(axnn::lowering::CapabilityMask::NONE)
            .with_minimum_accuracy(KernelAccuracy::Faithful),
    )
}

/// Select a TOSA recipe for one graph node. Canonical WGSL/elementwise nodes
/// must pass through `ax::node_kernel_request`; only the older structural and
/// matrix operations retain an opcode-specific compatibility path.
pub(crate) fn recipe_for_node(
    graph: &Graph<'_>,
    node: NodeView<'_>,
    profile: TosaProfile,
) -> Option<AxTosaRecipe> {
    if let Some(request) = ax::node_kernel_request(graph, node) {
        return select_tosa_recipe(&request, profile).map(|plan| plan.implementation);
    }
    legacy_recipe(node.op)
}

/// Recipe lookup for callers that already own a validated partition.
///
/// Artifact authoring historically supports a few wider TOSA dtype rows than
/// the placement model advertises (notably integer NEGATE with explicit zero
/// points). Preserve that low-level surface without making it an automatically
/// selected provider candidate. The operation identity still comes from the
/// shared semantic request rather than a second opcode table.
pub(crate) fn artifact_recipe_for_node(
    graph: &Graph<'_>,
    node: NodeView<'_>,
) -> Option<AxTosaRecipe> {
    if let Some(request) = ax::node_kernel_request(graph, node) {
        return semantic_recipe(request.semantic);
    }
    legacy_recipe(node.op)
}

/// Provider capability admission for the exact artifact that authoring will
/// emit, including all primitive operators, graph constants, recipe scratch
/// tensors, literals, shifts, zero points, and Boolean boundary bridges.
pub(crate) fn capability_supports_node(
    capability: CapabilityDescriptor,
    graph: &Graph<'_>,
    node: NodeView<'_>,
    profile: TosaProfile,
    f32_boolean_boundary_bridge: bool,
    promote_fp8: bool,
) -> bool {
    let recipe = if promote_fp8 && node.op == OpCode::MATMUL {
        semantic_recipe(SemanticOp::MatMul)
    } else {
        recipe_for_node(graph, node, profile)
    };
    let Some(recipe) = recipe else {
        return false;
    };

    let mut operators_supported = true;
    recipe.visit_operators(|op| operators_supported &= capability.supports_operator(op));
    if !operators_supported {
        return false;
    }

    let mut has_authored_constant = false;
    for tensor_id in node.inputs().chain(node.outputs()) {
        let Ok(tensor) = graph.tensor(tensor_id) else {
            return false;
        };
        let Some(dtype) = emitted_tosa_dtype(tensor.dtype, promote_fp8) else {
            return false;
        };
        let role = value_role(tensor);
        if tensor.flags.contains(TensorFlags::CONSTANT) {
            has_authored_constant = true;
        }
        if f32_boolean_boundary_bridge
            && tensor.dtype == DType::U8
            && !tensor.flags.contains(TensorFlags::CONSTANT)
        {
            // The graph value stays BOOL inside the model, while the provider
            // model feature is FP32. Check both sides of that generated bridge.
            if !capability.supports_dtype(TosaDType::BOOL, ValueRoles::INTERMEDIATE)
                || !capability.supports_dtype(TosaDType::FP32, role)
            {
                return false;
            }
            if tensor.flags.contains(TensorFlags::INPUT)
                && (!capability.supports_operator(TosaOp::GREATER)
                    || !capability.supports_operator(TosaOp::CONST))
            {
                return false;
            }
            if tensor.flags.contains(TensorFlags::OUTPUT)
                && (!capability.supports_operator(TosaOp::SELECT)
                    || !capability.supports_operator(TosaOp::CONST))
            {
                return false;
            }
            if (tensor.flags.contains(TensorFlags::INPUT)
                || tensor.flags.contains(TensorFlags::OUTPUT))
                && !capability.supports_dtype(TosaDType::FP32, ValueRoles::CONSTANT)
            {
                return false;
            }
        } else if !capability.supports_dtype(dtype, role) {
            return false;
        }
    }

    let Some(data_dtype) = node
        .inputs()
        .next()
        .and_then(|id| graph.tensor(id).ok())
        .and_then(|tensor| emitted_tosa_dtype(tensor.dtype, promote_fp8))
    else {
        return false;
    };
    let mut synthetic_supported = true;
    let mut has_synthetic_constant = false;
    recipe.visit_synthetic_values(data_dtype, |dtype, role| {
        synthetic_supported &= capability.supports_dtype(dtype, role);
        has_synthetic_constant |= role.contains(ValueRoles::CONSTANT);
    });
    synthetic_supported
        && (!(has_authored_constant || has_synthetic_constant)
            || capability.supports_operator(TosaOp::CONST))
}

/// Validate the exact block ABI emitted for a finalized scheduled partition.
/// Per-node admission cannot know when an authored intermediate becomes a
/// provider input/output or when a Boolean bridge adds boundary operators.
pub(crate) fn capability_supports_partition(
    capability: CapabilityDescriptor,
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    partition: Partition,
    profile: TosaProfile,
    f32_boolean_boundary_bridge: bool,
    promote_fp8: bool,
) -> bool {
    let start = partition.order_start as usize;
    let Some(end) = start.checked_add(partition.node_count as usize) else {
        return false;
    };
    if start == end || end > order.len() || producers.len() < graph.tensor_count() {
        return false;
    }
    let region = &order[start..end];
    let node_inside = |node: u32| region.contains(&node);

    for &node_index in region {
        let Ok(node) = graph.node(node_index as usize) else {
            return false;
        };
        let recipe = if promote_fp8 && node.op == OpCode::MATMUL {
            semantic_recipe(SemanticOp::MatMul)
        } else {
            recipe_for_node(graph, node, profile)
        };
        let Some(recipe) = recipe else {
            return false;
        };
        let mut operators_supported = true;
        recipe.visit_operators(|op| operators_supported &= capability.supports_operator(op));
        if !operators_supported {
            return false;
        }
        let Some(data_dtype) = node
            .inputs()
            .next()
            .and_then(|id| graph.tensor(id).ok())
            .and_then(|tensor| emitted_tosa_dtype(tensor.dtype, promote_fp8))
        else {
            return false;
        };
        let mut synthetic_supported = true;
        let mut has_synthetic_constant = false;
        recipe.visit_synthetic_values(data_dtype, |dtype, role| {
            synthetic_supported &= capability.supports_dtype(dtype, role);
            has_synthetic_constant |= role.contains(ValueRoles::CONSTANT);
        });
        if !synthetic_supported
            || (has_synthetic_constant && !capability.supports_operator(TosaOp::CONST))
        {
            return false;
        }
    }

    let mut has_authored_constant = false;
    let mut bridge_input = false;
    let mut bridge_output = false;
    for tensor_index in 0..graph.tensor_count() {
        let id = TensorId(tensor_index as u32);
        let Ok(tensor) = graph.tensor(id) else {
            return false;
        };
        let included = region.iter().any(|&node_index| {
            graph.node(node_index as usize).is_ok_and(|node| {
                node.inputs()
                    .chain(node.outputs())
                    .any(|candidate| candidate == id)
            })
        });
        if !included {
            continue;
        }
        let constant = tensor.flags.contains(TensorFlags::CONSTANT);
        let producer = producers[tensor_index];
        let produced_inside = producer != ax::schedule::NO_PRODUCER && node_inside(producer);
        let boundary_input = !constant
            && region.iter().any(|&node_index| {
                graph
                    .node(node_index as usize)
                    .is_ok_and(|node| node.inputs().any(|candidate| candidate == id))
            })
            && !produced_inside;
        let boundary_output = produced_inside
            && (tensor.flags.contains(TensorFlags::OUTPUT)
                || (0..graph.node_count()).any(|consumer| {
                    !node_inside(consumer as u32)
                        && graph
                            .node(consumer)
                            .is_ok_and(|node| node.inputs().any(|candidate| candidate == id))
                }));
        let Some(dtype) = emitted_tosa_dtype(tensor.dtype, promote_fp8) else {
            return false;
        };
        if constant {
            has_authored_constant = true;
            if !capability.supports_dtype(dtype, ValueRoles::CONSTANT) {
                return false;
            }
            continue;
        }
        if f32_boolean_boundary_bridge && tensor.dtype == DType::U8 {
            if !capability.supports_dtype(TosaDType::BOOL, ValueRoles::INTERMEDIATE) {
                return false;
            }
            if boundary_input {
                bridge_input = true;
                if !capability.supports_dtype(TosaDType::FP32, ValueRoles::INPUT) {
                    return false;
                }
            }
            if boundary_output {
                bridge_output = true;
                if !capability.supports_dtype(TosaDType::FP32, ValueRoles::OUTPUT) {
                    return false;
                }
            }
            continue;
        }
        if boundary_input && !capability.supports_dtype(dtype, ValueRoles::INPUT) {
            return false;
        }
        if boundary_output && !capability.supports_dtype(dtype, ValueRoles::OUTPUT) {
            return false;
        }
        if !boundary_input
            && !boundary_output
            && !capability.supports_dtype(dtype, ValueRoles::INTERMEDIATE)
        {
            return false;
        }
    }
    if has_authored_constant && !capability.supports_operator(TosaOp::CONST) {
        return false;
    }
    if bridge_input
        && (!capability.supports_operator(TosaOp::GREATER)
            || !capability.supports_operator(TosaOp::CONST)
            || !capability.supports_dtype(TosaDType::FP32, ValueRoles::CONSTANT))
    {
        return false;
    }
    if bridge_output
        && (!capability.supports_operator(TosaOp::SELECT)
            || !capability.supports_operator(TosaOp::CONST)
            || !capability.supports_dtype(TosaDType::FP32, ValueRoles::CONSTANT))
    {
        return false;
    }
    true
}

fn semantic_recipe(semantic: SemanticOp) -> Option<AxTosaRecipe> {
    Some(match semantic {
        SemanticOp::Identity => AxTosaRecipe::Direct(OperatorKind::Identity),
        SemanticOp::Unary(op) => match op {
            UnaryOp::Abs => AxTosaRecipe::Direct(OperatorKind::Abs),
            UnaryOp::Cos => AxTosaRecipe::Direct(OperatorKind::Cos),
            UnaryOp::Exp => AxTosaRecipe::Direct(OperatorKind::Exp),
            UnaryOp::Log => AxTosaRecipe::Direct(OperatorKind::Log),
            UnaryOp::Neg => AxTosaRecipe::Direct(OperatorKind::Negate),
            UnaryOp::Reciprocal => AxTosaRecipe::Direct(OperatorKind::Reciprocal),
            UnaryOp::InverseSqrt => AxTosaRecipe::Direct(OperatorKind::Rsqrt),
            UnaryOp::Sin => AxTosaRecipe::Direct(OperatorKind::Sin),
            UnaryOp::Tanh => AxTosaRecipe::Direct(OperatorKind::Tanh),
            UnaryOp::Sigmoid => AxTosaRecipe::Direct(OperatorKind::Sigmoid),
            UnaryOp::Sqrt => AxTosaRecipe::SquareRoot,
            UnaryOp::Exp2 => AxTosaRecipe::Exp2,
            UnaryOp::Relu => AxTosaRecipe::Relu,
            UnaryOp::Log2 => AxTosaRecipe::Log2,
            UnaryOp::Tan => AxTosaRecipe::Tan,
            // TOSA 1.0 and the in-tree executor understand CEIL/FLOOR, but
            // virtio-accel-tosa-build 0.3.3 cannot author those variants yet.
            // Do not advertise them until the builder grows the typed surface.
            UnaryOp::Ceil | UnaryOp::Floor => return None,
            _ => return None,
        },
        SemanticOp::Binary(op) => match op {
            BinaryOp::Add => AxTosaRecipe::Direct(OperatorKind::Add),
            BinaryOp::Sub => AxTosaRecipe::Direct(OperatorKind::Sub),
            BinaryOp::Mul => AxTosaRecipe::Direct(OperatorKind::Mul),
            BinaryOp::Div => AxTosaRecipe::Divide,
            BinaryOp::Pow => AxTosaRecipe::Direct(OperatorKind::Pow),
            BinaryOp::Max => AxTosaRecipe::Direct(OperatorKind::Maximum {
                nan_mode: virtio_accel_tosa::NanPropagationMode::PROPAGATE,
            }),
            BinaryOp::Min => AxTosaRecipe::Direct(OperatorKind::Minimum {
                nan_mode: virtio_accel_tosa::NanPropagationMode::PROPAGATE,
            }),
            _ => return None,
        },
        SemanticOp::Ternary(TernaryOp::Clamp) => AxTosaRecipe::Clamp,
        SemanticOp::Ternary(TernaryOp::Fma) => AxTosaRecipe::Fma,
        SemanticOp::Compare(op) => AxTosaRecipe::Direct(match op {
            CompareOp::Equal => OperatorKind::Equal,
            CompareOp::Greater => OperatorKind::Greater,
            CompareOp::GreaterEqual => OperatorKind::GreaterEqual,
        }),
        SemanticOp::Logical(op) => AxTosaRecipe::Direct(match op {
            LogicalOp::Not => OperatorKind::LogicalNot,
            LogicalOp::And => OperatorKind::LogicalAnd,
            LogicalOp::Or => OperatorKind::LogicalOr,
            LogicalOp::Xor => OperatorKind::LogicalXor,
        }),
        SemanticOp::Select => AxTosaRecipe::Direct(OperatorKind::Select),
        SemanticOp::MatMul => AxTosaRecipe::Direct(OperatorKind::MatMul),
        _ => return None,
    })
}

fn semantic_constraints(semantic: SemanticOp, profile: TosaProfile) -> Option<KernelConstraints> {
    let floats = DTypeSet::one(DType::F32) | DTypeSet::one(DType::F16);
    let boolean = DTypeSet::one(DType::U8);
    let identity = match profile {
        TosaProfile::Cpu => {
            floats
                | DTypeSet::one(DType::B16)
                | DTypeSet::one(DType::I32)
                | DTypeSet::one(DType::I16)
                | DTypeSet::one(DType::I8)
                | boolean
                | DTypeSet::one(DType::B8)
                | DTypeSet::one(DType::F8)
        }
        TosaProfile::Accelerator => {
            floats
                | DTypeSet::one(DType::B16)
                | boolean
                | DTypeSet::one(DType::B8)
                | DTypeSet::one(DType::F8)
        }
    };
    let (inputs, outputs) = match semantic {
        SemanticOp::Identity => ([identity, DTypeSet::NONE, DTypeSet::NONE], identity),
        SemanticOp::Unary(_) => ([floats, DTypeSet::NONE, DTypeSet::NONE], floats),
        SemanticOp::Binary(_) => ([floats, floats, DTypeSet::NONE], floats),
        SemanticOp::Ternary(_) => ([floats, floats, floats], floats),
        SemanticOp::Compare(_) => ([floats, floats, DTypeSet::NONE], boolean),
        SemanticOp::Logical(LogicalOp::Not) => ([boolean, DTypeSet::NONE, DTypeSet::NONE], boolean),
        SemanticOp::Logical(_) => ([boolean, boolean, DTypeSet::NONE], boolean),
        SemanticOp::Select => ([boolean, floats, floats], floats),
        SemanticOp::MatMul => {
            let matrix_inputs = match profile {
                TosaProfile::Cpu => {
                    floats
                        | DTypeSet::one(DType::B8)
                        | DTypeSet::one(DType::F8)
                        | DTypeSet::one(DType::I8)
                }
                TosaProfile::Accelerator => {
                    floats
                        | DTypeSet::one(DType::B16)
                        | DTypeSet::one(DType::B8)
                        | DTypeSet::one(DType::F8)
                }
            };
            let matrix_outputs = match profile {
                TosaProfile::Cpu => floats | DTypeSet::one(DType::I32),
                TosaProfile::Accelerator => floats,
            };
            (
                [matrix_inputs, matrix_inputs, DTypeSet::NONE],
                matrix_outputs,
            )
        }
        _ => return None,
    };
    Some(KernelConstraints {
        input_dtypes: inputs,
        output_dtypes: outputs,
        min_elements: 0,
        max_elements: usize::MAX,
        element_multiple: 1,
        min_work_items: 0,
        max_work_items: u64::MAX,
        work_item_multiple: 1,
        max_rank: axnn::MAX_RANK as u8,
        contiguous_only: false,
        shape_policy: if semantic == SemanticOp::MatMul {
            KernelShapePolicy::Typed
        } else {
            KernelShapePolicy::Broadcast
        },
    })
}

fn matmul_request_supported(request: &KernelRequest, profile: TosaProfile) -> bool {
    let dtypes = (
        request.input_dtypes[0],
        request.input_dtypes[1],
        request.output_dtype,
    );
    match profile {
        TosaProfile::Cpu => matches!(
            dtypes,
            (Some(DType::F32), Some(DType::F32), DType::F32)
                | (Some(DType::F16), Some(DType::F16), DType::F16)
                | (Some(DType::B8), Some(DType::B8), DType::F16)
                | (Some(DType::F8), Some(DType::F8), DType::F16)
                | (Some(DType::I8), Some(DType::I8), DType::I32)
        ),
        TosaProfile::Accelerator => matches!(
            dtypes,
            (Some(DType::F32), Some(DType::F32), DType::F32)
                | (Some(DType::F16), Some(DType::F16), DType::F16)
                | (Some(DType::B16), Some(DType::B16), DType::F32)
                // TOSA's own FP8 MATMUL row, which the Vulkan and OpenVINO FP8 tiers implement.
                // Like the BF16 row, a provider must declare it: `accel_supports` refuses FP8 on
                // the accelerator when no capability descriptor is present.
                | (Some(DType::B8), Some(DType::B8), DType::F16)
                | (Some(DType::F8), Some(DType::F8), DType::F16)
        ),
    }
}

fn recipe_temporary_bytes(recipe: AxTosaRecipe, request: &KernelRequest) -> usize {
    let output = request.output_dtype.storage_bytes(request.elements());
    match recipe {
        AxTosaRecipe::Divide => request
            .input(1)
            .and_then(|input| TensorDesc::try_new(input.shape, input.dtype))
            .map(|input| input.storage_bytes())
            .unwrap_or(output),
        AxTosaRecipe::SquareRoot | AxTosaRecipe::Clamp | AxTosaRecipe::Fma | AxTosaRecipe::Log2 => {
            output
        }
        AxTosaRecipe::Tan => output.saturating_mul(3),
        AxTosaRecipe::Direct(_) | AxTosaRecipe::Exp2 | AxTosaRecipe::Relu => 0,
    }
}

fn legacy_recipe(op: OpCode) -> Option<AxTosaRecipe> {
    Some(AxTosaRecipe::Direct(if op == OpCode::CAST {
        OperatorKind::Cast
    } else if op == OpCode::MATMUL {
        OperatorKind::MatMul
    } else if op == OpCode::RESHAPE {
        OperatorKind::Reshape
    } else {
        return None;
    }))
}

fn value_role(tensor: TensorView<'_>) -> ValueRoles {
    if tensor.flags.contains(TensorFlags::CONSTANT) {
        ValueRoles::CONSTANT
    } else if tensor.flags.contains(TensorFlags::INPUT) {
        ValueRoles::INPUT
    } else if tensor.flags.contains(TensorFlags::OUTPUT) {
        ValueRoles::OUTPUT
    } else {
        ValueRoles::INTERMEDIATE
    }
}

fn emitted_tosa_dtype(dtype: DType, promote_fp8: bool) -> Option<TosaDType> {
    if promote_fp8 && matches!(dtype, DType::B8 | DType::F8) {
        return Some(TosaDType::FP16);
    }
    Some(match dtype {
        DType::F32 => TosaDType::FP32,
        DType::F16 => TosaDType::FP16,
        DType::B16 => TosaDType::BF16,
        DType::I32 => TosaDType::INT32,
        DType::I16 => TosaDType::INT16,
        DType::I8 => TosaDType::INT8,
        DType::U8 => TosaDType::BOOL,
        DType::B8 => TosaDType::FP8E4M3,
        DType::F8 => TosaDType::FP8E5M2,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use std::alloc::Global;

    use super::*;
    use ax::{GraphBuilder, NodeFlags, TensorFlags};
    use axnn::{Shape, TensorDesc};
    use virtio_accel_tosa::{
        DTypeCapability, ExtensionSet, GraphCapabilities, Level, OperatorCapability, ProfileSet,
        RuntimeConditionSupport, Target, Version,
    };

    const TEST_TARGET: Target = Target::new(
        Version::TOSA_1_0,
        ProfileSet::FLOATING_POINT,
        Level::Level8K,
        ExtensionSet::NONE,
    );
    const TEST_GRAPH: GraphCapabilities = GraphCapabilities {
        max_regions: 1,
        max_blocks: 1,
        dynamic_shapes: false,
        runtime_conditions: RuntimeConditionSupport::None,
    };
    const DIVISION_OPERATORS: &[OperatorCapability] = &[
        OperatorCapability::new(TosaOp::RECIPROCAL),
        OperatorCapability::new(TosaOp::MUL),
        OperatorCapability::new(TosaOp::CONST),
    ];
    const FLOAT_ONLY: &[DTypeCapability] =
        &[DTypeCapability::new(TosaDType::FP32, ValueRoles::ALL)];
    const FLOAT_AND_SHIFT: &[DTypeCapability] = &[
        DTypeCapability::new(TosaDType::FP32, ValueRoles::ALL),
        DTypeCapability::new(TosaDType::INT8, ValueRoles::CONSTANT),
    ];

    #[test]
    fn semantic_catalog_selects_composite_recipes_and_prices_scratch() {
        let field = TensorDesc::new(Shape::try_from(&[8, 16][..]).unwrap(), DType::F32);
        let request = KernelRequest::new(SemanticOp::Unary(UnaryOp::Tan), &field);
        let plan = select_tosa_recipe(&request, TosaProfile::Accelerator).unwrap();
        assert_eq!(plan.implementation, AxTosaRecipe::Tan);
        assert_eq!(plan.estimated_cost, 4 * 8 * 16);
        assert_eq!(plan.temporary_bytes, 3 * 8 * 16 * 4);
    }

    #[test]
    fn semantic_catalog_selects_typed_matmul_and_prices_reduction_work() {
        let lhs = TensorDesc::new(Shape::try_from(&[2, 3][..]).unwrap(), DType::F32);
        let rhs = TensorDesc::new(Shape::try_from(&[3, 4][..]).unwrap(), DType::F32);
        let output = TensorDesc::new(Shape::try_from(&[2, 4][..]).unwrap(), DType::F32);
        let request = KernelRequest::matmul(&lhs, &rhs, &output).unwrap();

        let plan = select_tosa_recipe(&request, TosaProfile::Accelerator).unwrap();

        assert_eq!(
            plan.implementation,
            AxTosaRecipe::Direct(OperatorKind::MatMul)
        );
        assert_eq!(plan.work_items, 24);
        assert_eq!(plan.estimated_cost, 24);

        let narrow = TensorDesc::new(rhs.shape, DType::F16);
        let invalid = KernelRequest::matmul(&lhs, &narrow, &output).unwrap();
        assert!(select_tosa_recipe(&invalid, TosaProfile::Cpu).is_none());
    }

    #[test]
    fn authoring_gap_does_not_falsely_admit_ceil_or_floor() {
        let desc = TensorDesc::new(Shape::try_from(&[4][..]).unwrap(), DType::F32);
        for op in [UnaryOp::Ceil, UnaryOp::Floor] {
            let request = KernelRequest::new(SemanticOp::Unary(op), &desc);
            assert!(select_tosa_recipe(&request, TosaProfile::Cpu).is_none());
        }
    }

    #[test]
    fn catalog_rejects_non_float_select_values() {
        let shape = Shape::try_from(&[4][..]).unwrap();
        let mask = TensorDesc::new(shape, DType::U8);
        let values = TensorDesc::new(shape, DType::U8);
        let request = KernelRequest::select(&mask, &values, &values).unwrap();
        assert!(select_tosa_recipe(&request, TosaProfile::Cpu).is_none());
    }

    #[test]
    fn capability_admission_includes_synthesized_shift_constants() {
        let mut builder = GraphBuilder::new_in(Global);
        let left = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let right = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let output = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[4], &[])
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
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        let node = graph.node(0).unwrap();
        let without_shift = CapabilityDescriptor {
            target: TEST_TARGET,
            dtypes: FLOAT_ONLY,
            operators: DIVISION_OPERATORS,
            graph: TEST_GRAPH,
        };
        assert!(!capability_supports_node(
            without_shift,
            &graph,
            node,
            TosaProfile::Accelerator,
            false,
            false,
        ));

        let complete = CapabilityDescriptor {
            target: TEST_TARGET,
            dtypes: FLOAT_AND_SHIFT,
            operators: DIVISION_OPERATORS,
            graph: TEST_GRAPH,
        };
        assert!(capability_supports_node(
            complete,
            &graph,
            node,
            TosaProfile::Accelerator,
            false,
            false,
        ));
    }
}
