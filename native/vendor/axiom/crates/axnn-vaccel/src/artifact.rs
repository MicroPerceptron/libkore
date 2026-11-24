//! TOSA 1.0 artifact authoring for axiom execution and scheduled partitions.
//!
//! `virtio-accel-tosa-build` owns the FlatBuffer schema details and validates
//! every completed graph before it can reach a provider. Partition metadata
//! continues to honor the caller allocator; artifact authoring remains a cold,
//! one-time compile path that returns the provider-owned byte vector.

use core::alloc::Allocator;

use ax::{DType, Graph, OpCode, Partition, TensorFlags, TensorId};
use virtio_accel_tosa::{DType as TosaDType, NanPropagationMode};
use virtio_accel_tosa_build::{
    Graph as TosaGraph, Operator, OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor, Tensor,
};

use crate::lowering::{AxTosaRecipe, artifact_recipe_for_node};

/// Element type of a generated MATMUL graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatMulDType {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 float.
    F16,
    /// 8-bit brain float (OCP E4M3).
    B8,
    /// 8-bit IEEE float (OCP E5M2).
    F8,
}

/// Failure while authoring a fixed-shape artifact.
///
/// The public helpers in this module accept host-provided dimensions and
/// weights.  Invalid input is therefore a recoverable boundary error, not a
/// programmer-only assertion: callers such as [`crate::Session`] must be able
/// to preserve provider fallback and error reporting when a shape does not fit
/// the selected TOSA target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    /// A tensor extent was zero.
    ZeroDimension,
    /// A tensor extent cannot be represented by TOSA's signed shape fields.
    DimensionTooLarge,
    /// An MLP needs at least one layer and one weight matrix per layer.
    InvalidLayerShape,
    /// A layer's supplied weight plane does not match its declared shape.
    InvalidWeightShape { layer: usize },
    /// Computing a host-side tensor size overflowed `usize`.
    SizeOverflow,
    /// The completed graph violates the selected TOSA target's semantic
    /// limits (for example, its maximum tensor storage size).
    AuthoringFailed,
}

impl core::fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroDimension => formatter.write_str("tensor dimensions must be nonzero"),
            Self::DimensionTooLarge => formatter
                .write_str("tensor dimension does not fit TOSA's signed shape representation"),
            Self::InvalidLayerShape => {
                formatter.write_str("MLP dimensions and layer count are inconsistent")
            }
            Self::InvalidWeightShape { layer } => {
                write!(
                    formatter,
                    "MLP weight plane {layer} does not match its declared shape"
                )
            }
            Self::SizeOverflow => {
                formatter.write_str("tensor size overflows the host address space")
            }
            Self::AuthoringFailed => {
                formatter.write_str("artifact violates the selected TOSA target")
            }
        }
    }
}

impl std::error::Error for ArtifactError {}

impl MatMulDType {
    const fn input_dtype(self) -> TosaDType {
        match self {
            Self::F32 => TosaDType::FP32,
            Self::F16 => TosaDType::FP16,
            Self::B8 => TosaDType::FP8E4M3,
            Self::F8 => TosaDType::FP8E5M2,
        }
    }

    /// TOSA 1.0 MATMUL output row: fp32→fp32, fp16→fp16, fp8→fp16.
    const fn output_dtype(self) -> TosaDType {
        match self {
            Self::F32 => TosaDType::FP32,
            Self::F16 => TosaDType::FP16,
            Self::B8 | Self::F8 => TosaDType::FP16,
        }
    }

    /// Bytes per input element.
    pub const fn input_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::B8 | Self::F8 => 1,
        }
    }

    /// Bytes per output element.
    pub const fn output_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::B8 | Self::F8 => 2,
        }
    }

    /// Stable one-byte tag for cache keys and routing tables.
    pub const fn tag(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::B8 => 2,
            Self::F8 => 3,
        }
    }
}

/// Serialize a TOSA 1.0 artifact computing
/// `out[1, m, n] = MATMUL(a[1, m, k], b[1, k, n])` in `dt`, with the
/// zero-point operands the spec requires bound to CONST zeros.
///
/// Block inputs are `a` (slot 0) and `b` (slot 1); the sole output is
/// `out` (slot 2). Element encodings are little-endian; F8 inputs
/// produce an FP16 output plane.
pub fn matmul(dt: MatMulDType, m: usize, k: usize, n: usize) -> Result<Vec<u8>, ArtifactError> {
    let (m, k, n) = (tosa_dim(m)?, tosa_dim(k)?, tosa_dim(n)?);
    let in_ty = dt.input_dtype();
    let zero = [0u8; 4];
    let a_shape = [1, m, k];
    let b_shape = [1, k, n];
    let out_shape = [1, m, n];
    let tensors = [
        Tensor::new("a", &a_shape, in_ty),
        Tensor::new("b", &b_shape, in_ty),
        Tensor::constant("a_zp", &[1], in_ty, &zero[..dt.input_size()]),
        Tensor::constant("b_zp", &[1], in_ty, &zero[..dt.input_size()]),
        Tensor::new("out", &out_shape, dt.output_dtype()),
    ];
    let operators = [
        Operator::new(OperatorKind::Const, &[], &["a_zp"]),
        Operator::new(OperatorKind::Const, &[], &["b_zp"]),
        Operator::new(OperatorKind::MatMul, &["a", "b", "a_zp", "b_zp"], &["out"]),
    ];

    TosaGraph::new("main", &tensors, &operators, &["a", "b"], &["out"])
        .build(axtosa::target())
        .map_err(|_| ArtifactError::AuthoringFailed)
}

/// Serialize a fused FP32 MLP as one TOSA 1.0 graph:
///
/// ```text
/// x[1, m, dims[0]] → MATMUL(w0) → TANH → MATMUL(w1) → TANH → …
///                  → MATMUL(w_last) → SIGMOID → out[1, m, dims.last()]
/// ```
///
/// `weights[i]` is the row-major `dims[i] × dims[i+1]` matrix, **baked
/// into the artifact as a CONST tensor** — the graph's only runtime
/// input is `x`, so a whole inference is one submission. Zero-point
/// operands are emitted as CONST zeros per MATMUL, as TOSA 1.0
/// requires. Hidden activations are TANH; the final layer is SIGMOID.
///
/// TOSA validates the completed tensor storage against the selected target;
/// dimensions are not independently capped at 8192 (that Level 8K limit
/// applies to specific kernel and stride attributes, not MATMUL extents).
pub fn mlp_f32(m: usize, dims: &[usize], weights: &[Vec<f32>]) -> Result<Vec<u8>, ArtifactError> {
    mlp(m, dims, weights, false)
}

/// [`mlp_f32`] with every tensor in FP16 — the native precision of most
/// NPUs (the ANE computes in f16 regardless; f16 planes halve the I/O
/// and skip the boundary conversion). Weights are narrowed RNE.
pub fn mlp_f16(m: usize, dims: &[usize], weights: &[Vec<f32>]) -> Result<Vec<u8>, ArtifactError> {
    mlp(m, dims, weights, true)
}

fn tosa_dim(dim: usize) -> Result<i32, ArtifactError> {
    if dim == 0 {
        return Err(ArtifactError::ZeroDimension);
    }
    i32::try_from(dim).map_err(|_| ArtifactError::DimensionTooLarge)
}

fn validate_mlp_shape<T>(
    m: usize,
    dims: &[usize],
    weights: &[Vec<T>],
) -> Result<Vec<i32>, ArtifactError> {
    if dims.len() < 2 || weights.len() != dims.len() - 1 {
        return Err(ArtifactError::InvalidLayerShape);
    }
    tosa_dim(m)?;
    for (layer, weight) in weights.iter().enumerate() {
        let expected = dims[layer]
            .checked_mul(dims[layer + 1])
            .ok_or(ArtifactError::SizeOverflow)?;
        if weight.len() != expected {
            return Err(ArtifactError::InvalidWeightShape { layer });
        }
    }
    dims.iter().copied().map(tosa_dim).collect()
}

fn build_fixed_graph<'a>(
    tensors: Vec<OwnedTensor<'a>>,
    operators: Vec<OwnedOperator>,
    inputs: &[&str],
    outputs: &[&str],
) -> Result<Vec<u8>, ArtifactError> {
    OwnedGraph {
        name: "main".to_owned(),
        tensors,
        shapes: Vec::new(),
        operators,
        inputs: inputs.iter().map(|name| (*name).to_owned()).collect(),
        outputs: outputs.iter().map(|name| (*name).to_owned()).collect(),
    }
    .build(axtosa::target())
    .map_err(|_| ArtifactError::AuthoringFailed)
}

fn build_owned_graph<'a>(
    tensors: Vec<OwnedTensor<'a>>,
    operators: Vec<OwnedOperator>,
    inputs: &[&str],
    outputs: &[&str],
) -> Result<Vec<u8>, AxArtifactError> {
    // This is a device-neutral authoring envelope, not CPU admission. BF16
    // partitions are valid for providers such as XDNA even though axtosa's
    // CPU target does not implement BF16. Placement still requires the actual
    // provider's dtype/operator descriptor and load_program validates shapes.
    let mut target = axtosa::target();
    if tensors.iter().any(|tensor| tensor.dtype == TosaDType::BF16) {
        target.extensions = target
            .extensions
            .union(virtio_accel_tosa::ExtensionSet::BF16);
    }
    OwnedGraph {
        name: "main".to_owned(),
        tensors,
        shapes: Vec::new(),
        operators,
        inputs: inputs.iter().map(|name| (*name).to_owned()).collect(),
        outputs: outputs.iter().map(|name| (*name).to_owned()).collect(),
    }
    .build(target)
    .map_err(|_| AxArtifactError::AuthoringFailed)
}

/// Serialize a B8 (OCP E4M3)-storage MLP whose MATMUL results compute in FP16.
/// Hidden activations pass through `TANH` in FP16 and are explicitly cast
/// back to B8 before the next layer; the final FP16 activation uses
/// `SIGMOID`. This preserves narrow activation semantics while allowing an
/// adapter to widen each F8 MATMUL boundary for an FP16-only accelerator.
pub fn mlp_b8(m: usize, dims: &[usize], weights: &[Vec<u8>]) -> Result<Vec<u8>, ArtifactError> {
    let dims = validate_mlp_shape(m, dims, weights)?;

    let layers = weights.len();
    let m = tosa_dim(m)?;
    let weight_names: Vec<String> = (0..layers).map(|i| format!("w{i}")).collect();
    let input_zp_names: Vec<String> = (0..layers).map(|i| format!("x{i}_zp")).collect();
    let weight_zp_names: Vec<String> = (0..layers).map(|i| format!("w{i}_zp")).collect();
    let matmul_names: Vec<String> = (0..layers).map(|i| format!("mm{i}")).collect();
    let activation_names: Vec<String> = (0..layers).map(|i| format!("act{i}")).collect();
    let quantized_names: Vec<String> = (0..layers.saturating_sub(1))
        .map(|i| format!("q{i}"))
        .collect();
    let layer_inputs: Vec<String> = core::iter::once("x".to_string())
        .chain(quantized_names.iter().cloned())
        .collect();
    let mut tensors = Vec::new();
    tensors.push(OwnedTensor::new(
        "x",
        vec![1, m, dims[0]],
        TosaDType::FP8E4M3,
    ));
    let mut operators = Vec::new();
    for i in 0..layers {
        let (k, n) = (dims[i], dims[i + 1]);
        tensors.push(OwnedTensor::constant(
            &weight_names[i],
            vec![1, k, n],
            TosaDType::FP8E4M3,
            weights[i].clone(),
        ));
        tensors.push(OwnedTensor::constant(
            &input_zp_names[i],
            vec![1],
            TosaDType::FP8E4M3,
            vec![0],
        ));
        tensors.push(OwnedTensor::constant(
            &weight_zp_names[i],
            vec![1],
            TosaDType::FP8E4M3,
            vec![0],
        ));
        tensors.push(OwnedTensor::new(
            &matmul_names[i],
            vec![1, m, n],
            TosaDType::FP16,
        ));
        tensors.push(OwnedTensor::new(
            &activation_names[i],
            vec![1, m, n],
            TosaDType::FP16,
        ));
        if i + 1 < layers {
            tensors.push(OwnedTensor::new(
                &quantized_names[i],
                vec![1, m, n],
                TosaDType::FP8E4M3,
            ));
        }

        operators.push(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec![weight_names[i].clone()],
        ));
        operators.push(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec![input_zp_names[i].clone()],
        ));
        operators.push(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec![weight_zp_names[i].clone()],
        ));
        operators.push(OwnedOperator::new(
            OperatorKind::MatMul,
            vec![
                layer_inputs[i].clone(),
                weight_names[i].clone(),
                input_zp_names[i].clone(),
                weight_zp_names[i].clone(),
            ],
            vec![matmul_names[i].clone()],
        ));
        operators.push(OwnedOperator::new(
            if i + 1 == layers {
                OperatorKind::Sigmoid
            } else {
                OperatorKind::Tanh
            },
            vec![matmul_names[i].clone()],
            vec![activation_names[i].clone()],
        ));
        if i + 1 < layers {
            operators.push(OwnedOperator::new(
                OperatorKind::Cast,
                vec![activation_names[i].clone()],
                vec![quantized_names[i].clone()],
            ));
        }
    }

    build_fixed_graph(
        tensors,
        operators,
        &["x"],
        &[activation_names[layers - 1].as_str()],
    )
}

fn mlp(
    m: usize,
    dims: &[usize],
    weights: &[Vec<f32>],
    half: bool,
) -> Result<Vec<u8>, ArtifactError> {
    let dims = validate_mlp_shape(m, dims, weights)?;
    let elem = if half {
        TosaDType::FP16
    } else {
        TosaDType::FP32
    };
    let enc = |v: f32| -> Vec<u8> {
        if half {
            (v as f16).to_bits().to_le_bytes().to_vec()
        } else {
            v.to_le_bytes().to_vec()
        }
    };
    let layers = weights.len();
    let m = tosa_dim(m)?;

    // Owned names so tensor/operator tables can reference them.
    let wname: Vec<String> = (0..layers).map(|i| format!("w{i}")).collect();
    let xzp: Vec<String> = (0..layers).map(|i| format!("x{i}_zp")).collect();
    let wzp: Vec<String> = (0..layers).map(|i| format!("w{i}_zp")).collect();
    let mm: Vec<String> = (0..layers).map(|i| format!("mm{i}")).collect();
    let act: Vec<String> = (0..layers).map(|i| format!("act{i}")).collect();
    // Layer input names: x, act0, act1, …
    let lin: Vec<String> = core::iter::once("x".to_string())
        .chain(act.iter().take(layers - 1).cloned())
        .collect();

    let zero = enc(0.0);
    let mut tensors = Vec::new();
    tensors.push(OwnedTensor::new("x", vec![1, m, dims[0]], elem));
    let mut operators = Vec::new();
    for i in 0..layers {
        let (k, n) = (dims[i], dims[i + 1]);
        let bytes: Vec<u8> = weights[i].iter().flat_map(|value| enc(*value)).collect();
        tensors.push(OwnedTensor::constant(&wname[i], vec![1, k, n], elem, bytes));
        tensors.push(OwnedTensor::constant(&xzp[i], vec![1], elem, zero.clone()));
        tensors.push(OwnedTensor::constant(&wzp[i], vec![1], elem, zero.clone()));
        tensors.push(OwnedTensor::new(&mm[i], vec![1, m, n], elem));
        tensors.push(OwnedTensor::new(&act[i], vec![1, m, n], elem));

        operators.push(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec![wname[i].clone()],
        ));
        operators.push(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec![xzp[i].clone()],
        ));
        operators.push(OwnedOperator::new(
            OperatorKind::Const,
            vec![],
            vec![wzp[i].clone()],
        ));
        operators.push(OwnedOperator::new(
            OperatorKind::MatMul,
            vec![
                lin[i].clone(),
                wname[i].clone(),
                xzp[i].clone(),
                wzp[i].clone(),
            ],
            vec![mm[i].clone()],
        ));
        operators.push(OwnedOperator::new(
            if i + 1 == layers {
                OperatorKind::Sigmoid
            } else {
                OperatorKind::Tanh
            },
            vec![mm[i].clone()],
            vec![act[i].clone()],
        ));
    }

    build_fixed_graph(tensors, operators, &["x"], &[act[layers - 1].as_str()])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AxArtifactError {
    AllocationFailed,
    InvalidGraph,
    InvalidPartition,
    UnsupportedDType(DType),
    UnsupportedOperator(OpCode),
    DynamicShape(TensorId),
    DimensionTooLarge(TensorId),
    SizeOverflow(TensorId),
    InvalidConstant(TensorId),
    UnsupportedPromotion(TensorId),
    NoPartitionOutputs,
    AuthoringFailed,
}

/// Precision adaptation applied while lowering one provider partition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AxPrecisionPolicy {
    #[default]
    Native,
    /// Present F8 boundary tensors and constants to the provider as FP16.
    /// Graph-visible F8 outputs are rejected, so narrowing remains an
    /// explicit partition on a backend that supports it.
    PromoteFp8ToF16,
}

/// Provider-facing representation for Boolean partition boundaries.
///
/// TOSA Boolean values remain Boolean inside the partition. The F32 bridge
/// exists only for providers such as Core ML whose model feature interface
/// cannot expose a Boolean tensor even though its graph supports comparisons,
/// logical operators, and select internally.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AxBooleanBoundaryPolicy {
    #[default]
    Native,
    F32Bridge,
}

/// Independent precision and boundary adaptations for one partition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AxLoweringPolicy {
    pub precision: AxPrecisionPolicy,
    pub boolean_boundaries: AxBooleanBoundaryPolicy,
}

/// Host-side conversion required for one compiled partition input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AxInputTransform {
    #[default]
    Identity,
    B8ToF16,
    F8ToF16,
    BoolU8ToF32,
}

/// Host-side conversion required for one compiled partition output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AxOutputTransform {
    #[default]
    Identity,
    BoolF32ToU8,
}

/// One scheduled `.ax` partition lowered to a provider-neutral TOSA
/// artifact, including its boundary tensor order.
#[derive(Debug)]
pub struct AxPartitionArtifact<A: Allocator> {
    pub bytes: Vec<u8>,
    pub inputs: Vec<TensorId, A>,
    pub input_transforms: Vec<AxInputTransform, A>,
    pub outputs: Vec<TensorId, A>,
    pub output_transforms: Vec<AxOutputTransform, A>,
    /// Provider-visible sizes. These may differ from the graph tensor sizes
    /// when a boundary bridge is active.
    pub output_bytes: Vec<usize, A>,
}

/// Lower a scheduled `.ax` partition into the TOSA contract consumed by
/// both `AxTosaAccelerator` and host accelerator providers.
pub fn ax_partition_in<A: Allocator + Clone>(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    partition: Partition,
    allocator: A,
) -> Result<AxPartitionArtifact<A>, AxArtifactError> {
    ax_partition_with_policy_in(
        graph,
        order,
        producers,
        partition,
        AxPrecisionPolicy::Native,
        allocator,
    )
}

/// [`ax_partition_in`] with an explicit provider precision policy.
pub fn ax_partition_with_policy_in<A: Allocator + Clone>(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    partition: Partition,
    precision: AxPrecisionPolicy,
    allocator: A,
) -> Result<AxPartitionArtifact<A>, AxArtifactError> {
    ax_partition_with_lowering_policy_in(
        graph,
        order,
        producers,
        partition,
        AxLoweringPolicy {
            precision,
            boolean_boundaries: AxBooleanBoundaryPolicy::Native,
        },
        allocator,
    )
}

/// Lower a scheduled partition with composable precision and provider-boundary
/// adaptations.
pub fn ax_partition_with_lowering_policy_in<A: Allocator + Clone>(
    graph: &Graph<'_>,
    order: &[u32],
    producers: &[u32],
    partition: Partition,
    policy: AxLoweringPolicy,
    allocator: A,
) -> Result<AxPartitionArtifact<A>, AxArtifactError> {
    let precision = policy.precision;
    let start = partition.order_start as usize;
    let end = start
        .checked_add(partition.node_count as usize)
        .filter(|end| *end <= order.len())
        .ok_or(AxArtifactError::InvalidPartition)?;
    if start == end || producers.len() < graph.tensor_count() {
        return Err(AxArtifactError::InvalidPartition);
    }
    let region_order = &order[start..end];
    let mut inside = filled_in(graph.node_count(), false, allocator.clone())?;
    let mut included = filled_in(graph.tensor_count(), false, allocator.clone())?;
    for &node_index in region_order {
        let node_index = node_index as usize;
        if node_index >= graph.node_count() || inside[node_index] {
            return Err(AxArtifactError::InvalidPartition);
        }
        inside[node_index] = true;
        let node = graph
            .node(node_index)
            .map_err(|_| AxArtifactError::InvalidGraph)?;
        artifact_recipe_for_node(graph, node)
            .ok_or(AxArtifactError::UnsupportedOperator(node.op))?;
        if !node.attributes().is_empty() {
            return Err(AxArtifactError::UnsupportedOperator(node.op));
        }
        for tensor in node.inputs().chain(node.outputs()) {
            included[tensor.0 as usize] = true;
        }
    }

    let mut inputs = Vec::new_in(allocator.clone());
    let mut outputs = Vec::new_in(allocator.clone());
    reserve_in(&mut inputs, graph.tensor_count())?;
    reserve_in(&mut outputs, graph.tensor_count())?;
    for &node_index in region_order {
        let node = graph
            .node(node_index as usize)
            .map_err(|_| AxArtifactError::InvalidGraph)?;
        for input in node.inputs() {
            let tensor = graph
                .tensor(input)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            let producer = producers[input.0 as usize];
            let produced_inside = producer != ax::schedule::NO_PRODUCER
                && (producer as usize) < inside.len()
                && inside[producer as usize];
            if !produced_inside
                && !tensor.flags.contains(TensorFlags::CONSTANT)
                && !inputs.contains(&input)
            {
                inputs.push(input);
            }
        }
        for output in node.outputs() {
            let tensor = graph
                .tensor(output)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            let consumed_outside = (0..graph.node_count()).any(|consumer| {
                !inside[consumer]
                    && graph
                        .node(consumer)
                        .is_ok_and(|node| node.inputs().any(|input| input == output))
            });
            if (tensor.flags.contains(TensorFlags::OUTPUT) || consumed_outside)
                && !outputs.contains(&output)
            {
                outputs.push(output);
            }
        }
    }
    if outputs.is_empty() {
        return Err(AxArtifactError::NoPartitionOutputs);
    }
    if precision == AxPrecisionPolicy::PromoteFp8ToF16 {
        for &output in &outputs {
            let tensor = graph
                .tensor(output)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            if matches!(tensor.dtype, DType::B8 | DType::F8) {
                return Err(AxArtifactError::UnsupportedPromotion(output));
            }
        }
    }

    let mut input_transforms = Vec::new_in(allocator.clone());
    reserve_in(&mut input_transforms, inputs.len())?;
    for &input in &inputs {
        let tensor = graph
            .tensor(input)
            .map_err(|_| AxArtifactError::InvalidGraph)?;
        input_transforms.push(input_transform(tensor.dtype, policy));
    }
    let mut output_transforms = Vec::new_in(allocator.clone());
    reserve_in(&mut output_transforms, outputs.len())?;
    for &output in &outputs {
        let tensor = graph
            .tensor(output)
            .map_err(|_| AxArtifactError::InvalidGraph)?;
        output_transforms.push(output_transform(tensor.dtype, policy));
    }

    let tensor_names: Vec<String> = (0..graph.tensor_count())
        .map(|index| format!("t{index}"))
        .collect();
    let boundary_names: Vec<String> = (0..graph.tensor_count())
        .map(|index| format!("t{index}_f32_boundary"))
        .collect();
    let azp_names: Vec<String> = (0..graph.node_count())
        .map(|index| format!("n{index}_azp"))
        .collect();
    let bzp_names: Vec<String> = (0..graph.node_count())
        .map(|index| format!("n{index}_bzp"))
        .collect();
    let shift_names: Vec<String> = (0..graph.node_count())
        .map(|index| format!("n{index}_shift"))
        .collect();
    let has_boolean_inputs = input_transforms.contains(&AxInputTransform::BoolU8ToF32);
    let has_boolean_outputs = output_transforms.contains(&AxOutputTransform::BoolF32ToU8);
    const BOOLEAN_ZERO: &str = "boundary_bool_zero";
    const BOOLEAN_ONE: &str = "boundary_bool_one";

    let mut artifact_tensors = Vec::new();
    {
        for (index, &is_included) in included.iter().enumerate() {
            if !is_included {
                continue;
            }
            let tensor_id = TensorId(index as u32);
            let view = graph
                .tensor(tensor_id)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            let shape = tensor_shape(view)?;
            let emitted_dtype = precision_dtype(view.dtype, precision);
            let dtype = tosa_dtype(emitted_dtype)?;
            let constant = view.flags.contains(TensorFlags::CONSTANT);
            let source = constant.then(|| view.constant());
            if source.is_some_and(|bytes| tensor_bytes(view).ok() != Some(bytes.len())) {
                return Err(AxArtifactError::InvalidConstant(tensor_id));
            }
            let expanded = match (source, precision_input_transform(view.dtype, precision)) {
                (Some(bytes), AxInputTransform::B8ToF16) => {
                    let len = bytes
                        .len()
                        .checked_mul(2)
                        .ok_or(AxArtifactError::SizeOverflow(tensor_id))?;
                    let mut expanded = vec![0_u8; len];
                    axnn_cpu::fp8::dequant_e4m3_f16(bytes, &mut expanded, bytes.len());
                    Some(expanded)
                }
                (Some(bytes), AxInputTransform::F8ToF16) => {
                    let len = bytes
                        .len()
                        .checked_mul(2)
                        .ok_or(AxArtifactError::SizeOverflow(tensor_id))?;
                    let mut expanded = vec![0_u8; len];
                    axnn_cpu::fp8::dequant_e5m2_f16(bytes, &mut expanded, bytes.len());
                    Some(expanded)
                }
                _ => None,
            };
            match expanded {
                Some(data) => artifact_tensors.push(OwnedTensor::constant(
                    &tensor_names[index],
                    shape,
                    dtype,
                    data,
                )),
                None => match source {
                    Some(data) => artifact_tensors.push(OwnedTensor::constant(
                        &tensor_names[index],
                        shape,
                        dtype,
                        data,
                    )),
                    None => {
                        artifact_tensors.push(OwnedTensor::new(&tensor_names[index], shape, dtype))
                    }
                },
            }
        }
        for (&input, &transform) in inputs.iter().zip(&input_transforms) {
            if transform != AxInputTransform::BoolU8ToF32 {
                continue;
            }
            let view = graph
                .tensor(input)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            let shape = tensor_shape(view)?;
            artifact_tensors.push(OwnedTensor::new(
                &boundary_names[input.0 as usize],
                shape,
                TosaDType::FP32,
            ));
        }
        for (&output, &transform) in outputs.iter().zip(&output_transforms) {
            if transform != AxOutputTransform::BoolF32ToU8 {
                continue;
            }
            let view = graph
                .tensor(output)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            let shape = tensor_shape(view)?;
            artifact_tensors.push(OwnedTensor::new(
                &boundary_names[output.0 as usize],
                shape,
                TosaDType::FP32,
            ));
        }
        if has_boolean_inputs || has_boolean_outputs {
            artifact_tensors.push(OwnedTensor::constant(
                BOOLEAN_ZERO,
                vec![1],
                TosaDType::FP32,
                0.0_f32.to_le_bytes().to_vec(),
            ));
        }
        if has_boolean_outputs {
            artifact_tensors.push(OwnedTensor::constant(
                BOOLEAN_ONE,
                vec![1],
                TosaDType::FP32,
                1.0_f32.to_le_bytes().to_vec(),
            ));
        }
        for &node_index in region_order {
            let node = graph
                .node(node_index as usize)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            let recipe = artifact_recipe_for_node(graph, node)
                .ok_or(AxArtifactError::UnsupportedOperator(node.op))?;
            if recipe.needs_i8_shift() {
                artifact_tensors.push(OwnedTensor::constant(
                    &shift_names[node_index as usize],
                    vec![1],
                    TosaDType::INT8,
                    vec![0],
                ));
            }
            if matches!(
                recipe,
                AxTosaRecipe::Direct(OperatorKind::MatMul | OperatorKind::Negate)
            ) {
                let input = node.inputs().next().ok_or(AxArtifactError::InvalidGraph)?;
                let input = graph
                    .tensor(input)
                    .map_err(|_| AxArtifactError::InvalidGraph)?;
                let emitted_dtype = precision_dtype(input.dtype, precision);
                let dtype = tosa_dtype(emitted_dtype)?;
                let zero = vec![0_u8; emitted_dtype.storage_bytes(1)];
                artifact_tensors.push(OwnedTensor::constant(
                    &azp_names[node_index as usize],
                    vec![1],
                    dtype,
                    zero.clone(),
                ));
                artifact_tensors.push(OwnedTensor::constant(
                    &bzp_names[node_index as usize],
                    vec![1],
                    dtype,
                    zero,
                ));
            }

            let output = node
                .outputs()
                .next()
                .and_then(|id| graph.tensor(id).ok())
                .ok_or(AxArtifactError::InvalidGraph)?;
            let output_shape = tensor_shape(output)?;
            let emitted_output_dtype = precision_dtype(output.dtype, precision);
            let tosa_output_dtype = tosa_dtype(emitted_output_dtype)?;
            let add_temp = |slot: usize, tensors: &mut Vec<OwnedTensor<'_>>| {
                tensors.push(OwnedTensor::new(
                    recipe_temp_name(node_index, slot),
                    output_shape.clone(),
                    tosa_output_dtype,
                ));
            };
            // A synthesized scalar operand is authored at the output's rank
            // with every dimension 1. TOSA 1.0 broadcasts per dimension but
            // requires equal ranks, so a rank-1 literal against a rank-3
            // result is not a valid graph and providers reject the program.
            // A rank-0 output takes a rank-0 literal; clamping the rank up
            // would recreate the mismatch this exists to avoid.
            let literal_shape = vec![1; output_shape.len()];
            match recipe {
                AxTosaRecipe::Divide => {
                    let divisor = node
                        .inputs()
                        .nth(1)
                        .and_then(|id| graph.tensor(id).ok())
                        .ok_or(AxArtifactError::InvalidGraph)?;
                    artifact_tensors.push(OwnedTensor::new(
                        recipe_temp_name(node_index, 0),
                        tensor_shape(divisor)?,
                        tosa_dtype(precision_dtype(divisor.dtype, precision))?,
                    ));
                }
                AxTosaRecipe::SquareRoot | AxTosaRecipe::Clamp | AxTosaRecipe::Fma => {
                    add_temp(0, &mut artifact_tensors);
                }
                AxTosaRecipe::Exp2 => {
                    artifact_tensors.push(OwnedTensor::constant(
                        recipe_literal_name(node_index, 0),
                        literal_shape,
                        tosa_output_dtype,
                        float_constant_bytes(emitted_output_dtype, 2.0)?,
                    ));
                }
                AxTosaRecipe::Relu => {
                    artifact_tensors.push(OwnedTensor::constant(
                        recipe_literal_name(node_index, 0),
                        literal_shape,
                        tosa_output_dtype,
                        float_constant_bytes(emitted_output_dtype, 0.0)?,
                    ));
                }
                AxTosaRecipe::Log2 => {
                    add_temp(0, &mut artifact_tensors);
                    artifact_tensors.push(OwnedTensor::constant(
                        recipe_literal_name(node_index, 0),
                        literal_shape,
                        tosa_output_dtype,
                        float_constant_bytes(emitted_output_dtype, core::f32::consts::LOG2_E)?,
                    ));
                }
                AxTosaRecipe::Tan => {
                    add_temp(0, &mut artifact_tensors);
                    add_temp(1, &mut artifact_tensors);
                    add_temp(2, &mut artifact_tensors);
                }
                AxTosaRecipe::Direct(_) => {}
            }
        }
    }

    let mut artifact_operators = Vec::new();
    {
        for (index, &is_included) in included.iter().enumerate() {
            if !is_included {
                continue;
            }
            let view = graph
                .tensor(TensorId(index as u32))
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            if view.flags.contains(TensorFlags::CONSTANT) {
                artifact_operators.push(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec![tensor_names[index].clone()],
                ));
            }
        }
        if has_boolean_inputs || has_boolean_outputs {
            artifact_operators.push(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec![BOOLEAN_ZERO.to_string()],
            ));
        }
        if has_boolean_outputs {
            artifact_operators.push(OwnedOperator::new(
                OperatorKind::Const,
                vec![],
                vec![BOOLEAN_ONE.to_string()],
            ));
        }
        for (&input, &transform) in inputs.iter().zip(&input_transforms) {
            if transform == AxInputTransform::BoolU8ToF32 {
                artifact_operators.push(OwnedOperator::new(
                    OperatorKind::Greater,
                    vec![
                        boundary_names[input.0 as usize].clone(),
                        BOOLEAN_ZERO.to_string(),
                    ],
                    vec![tensor_names[input.0 as usize].clone()],
                ));
            }
        }
        for &node_index in region_order {
            let node = graph
                .node(node_index as usize)
                .map_err(|_| AxArtifactError::InvalidGraph)?;
            let recipe = artifact_recipe_for_node(graph, node)
                .ok_or(AxArtifactError::UnsupportedOperator(node.op))?;
            if recipe.needs_i8_shift() {
                artifact_operators.push(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec![shift_names[node_index as usize].clone()],
                ));
            }
            if matches!(
                recipe,
                AxTosaRecipe::Direct(OperatorKind::MatMul | OperatorKind::Negate)
            ) {
                artifact_operators.push(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec![azp_names[node_index as usize].clone()],
                ));
                artifact_operators.push(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec![bzp_names[node_index as usize].clone()],
                ));
            }
            if matches!(
                recipe,
                AxTosaRecipe::Exp2 | AxTosaRecipe::Log2 | AxTosaRecipe::Relu
            ) {
                artifact_operators.push(OwnedOperator::new(
                    OperatorKind::Const,
                    vec![],
                    vec![recipe_literal_name(node_index, 0)],
                ));
            }
            let node_inputs: Vec<String> = node
                .inputs()
                .map(|id| tensor_names[id.0 as usize].clone())
                .collect();
            let node_outputs: Vec<String> = node
                .outputs()
                .map(|id| tensor_names[id.0 as usize].clone())
                .collect();
            emit_ax_tosa_recipe(
                &mut artifact_operators,
                recipe,
                &node_inputs,
                &node_outputs,
                RecipeBindings {
                    node: node_index,
                    azp: &azp_names[node_index as usize],
                    bzp: &bzp_names[node_index as usize],
                    shift: &shift_names[node_index as usize],
                },
            )?;
        }
        for (&output, &transform) in outputs.iter().zip(&output_transforms) {
            if transform == AxOutputTransform::BoolF32ToU8 {
                artifact_operators.push(OwnedOperator::new(
                    OperatorKind::Select,
                    vec![
                        tensor_names[output.0 as usize].clone(),
                        BOOLEAN_ONE.to_string(),
                        BOOLEAN_ZERO.to_string(),
                    ],
                    vec![boundary_names[output.0 as usize].clone()],
                ));
            }
        }
    }

    let input_names: Vec<&str> = inputs
        .iter()
        .zip(&input_transforms)
        .map(|(id, transform)| match transform {
            AxInputTransform::BoolU8ToF32 => boundary_names[id.0 as usize].as_str(),
            _ => tensor_names[id.0 as usize].as_str(),
        })
        .collect();
    let output_names: Vec<&str> = outputs
        .iter()
        .zip(&output_transforms)
        .map(|(id, transform)| match transform {
            AxOutputTransform::BoolF32ToU8 => boundary_names[id.0 as usize].as_str(),
            AxOutputTransform::Identity => tensor_names[id.0 as usize].as_str(),
        })
        .collect();
    let bytes = build_owned_graph(
        artifact_tensors,
        artifact_operators,
        &input_names,
        &output_names,
    )?;

    let mut output_bytes = Vec::new_in(allocator);
    reserve_in(&mut output_bytes, outputs.len())?;
    for (&id, &transform) in outputs.iter().zip(&output_transforms) {
        let native_bytes = graph
            .tensor(id)
            .map_err(|_| AxArtifactError::InvalidGraph)
            .and_then(tensor_bytes)?;
        output_bytes.push(match transform {
            AxOutputTransform::Identity => native_bytes,
            AxOutputTransform::BoolF32ToU8 => native_bytes
                .checked_mul(4)
                .ok_or(AxArtifactError::SizeOverflow(id))?,
        });
    }
    Ok(AxPartitionArtifact {
        bytes,
        inputs,
        input_transforms,
        outputs,
        output_transforms,
        output_bytes,
    })
}

fn recipe_temp_name(node: u32, slot: usize) -> String {
    format!("n{node}_recipe_tmp{slot}")
}

fn recipe_literal_name(node: u32, slot: usize) -> String {
    format!("n{node}_recipe_literal{slot}")
}

fn float_constant_bytes(dtype: DType, value: f32) -> Result<Vec<u8>, AxArtifactError> {
    Ok(match dtype {
        DType::F32 => value.to_le_bytes().to_vec(),
        DType::F16 => (value as f16).to_bits().to_le_bytes().to_vec(),
        other => return Err(AxArtifactError::UnsupportedDType(other)),
    })
}

#[derive(Clone, Copy)]
struct RecipeBindings<'a> {
    node: u32,
    azp: &'a str,
    bzp: &'a str,
    shift: &'a str,
}

fn emit_ax_tosa_recipe(
    operators: &mut Vec<OwnedOperator>,
    recipe: AxTosaRecipe,
    inputs: &[String],
    outputs: &[String],
    bindings: RecipeBindings<'_>,
) -> Result<(), AxArtifactError> {
    let output = outputs
        .first()
        .ok_or(AxArtifactError::InvalidGraph)?
        .clone();
    let input = |slot: usize| {
        inputs
            .get(slot)
            .cloned()
            .ok_or(AxArtifactError::InvalidGraph)
    };
    let temp = |slot| recipe_temp_name(bindings.node, slot);
    let literal = |slot| recipe_literal_name(bindings.node, slot);
    let mut push = |kind, ins: Vec<String>, out: String| {
        operators.push(OwnedOperator::new(kind, ins, vec![out]));
    };

    match recipe {
        AxTosaRecipe::Direct(kind) => {
            let mut direct_inputs = inputs.to_vec();
            if matches!(kind, OperatorKind::MatMul | OperatorKind::Negate) {
                direct_inputs.push(bindings.azp.to_string());
                direct_inputs.push(bindings.bzp.to_string());
            } else if matches!(kind, OperatorKind::Mul) {
                direct_inputs.push(bindings.shift.to_string());
            }
            push(kind, direct_inputs, output);
        }
        AxTosaRecipe::Divide => {
            push(OperatorKind::Reciprocal, vec![input(1)?], temp(0));
            push(
                OperatorKind::Mul,
                vec![input(0)?, temp(0), bindings.shift.to_string()],
                output,
            );
        }
        AxTosaRecipe::SquareRoot => {
            push(OperatorKind::Rsqrt, vec![input(0)?], temp(0));
            push(OperatorKind::Reciprocal, vec![temp(0)], output);
        }
        AxTosaRecipe::Clamp => {
            push(
                OperatorKind::Maximum {
                    nan_mode: NanPropagationMode::PROPAGATE,
                },
                vec![input(0)?, input(1)?],
                temp(0),
            );
            push(
                OperatorKind::Minimum {
                    nan_mode: NanPropagationMode::PROPAGATE,
                },
                vec![temp(0), input(2)?],
                output,
            );
        }
        AxTosaRecipe::Fma => {
            push(
                OperatorKind::Mul,
                vec![input(0)?, input(1)?, bindings.shift.to_string()],
                temp(0),
            );
            push(OperatorKind::Add, vec![temp(0), input(2)?], output);
        }
        AxTosaRecipe::Exp2 => {
            push(OperatorKind::Pow, vec![literal(0), input(0)?], output);
        }
        AxTosaRecipe::Relu => {
            push(
                OperatorKind::Maximum {
                    nan_mode: NanPropagationMode::PROPAGATE,
                },
                vec![input(0)?, literal(0)],
                output,
            );
        }
        AxTosaRecipe::Log2 => {
            push(OperatorKind::Log, vec![input(0)?], temp(0));
            push(
                OperatorKind::Mul,
                vec![temp(0), literal(0), bindings.shift.to_string()],
                output,
            );
        }
        AxTosaRecipe::Tan => {
            push(OperatorKind::Sin, vec![input(0)?], temp(0));
            push(OperatorKind::Cos, vec![input(0)?], temp(1));
            push(OperatorKind::Reciprocal, vec![temp(1)], temp(2));
            push(
                OperatorKind::Mul,
                vec![temp(0), temp(2), bindings.shift.to_string()],
                output,
            );
        }
    }
    Ok(())
}

fn reserve_in<T, A: Allocator>(
    values: &mut Vec<T, A>,
    additional: usize,
) -> Result<(), AxArtifactError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| AxArtifactError::AllocationFailed)
}

fn filled_in<T: Clone, A: Allocator>(
    len: usize,
    value: T,
    allocator: A,
) -> Result<Vec<T, A>, AxArtifactError> {
    let mut values = Vec::new_in(allocator);
    reserve_in(&mut values, len)?;
    values.resize(len, value);
    Ok(values)
}

const fn precision_input_transform(dtype: DType, precision: AxPrecisionPolicy) -> AxInputTransform {
    match (precision, dtype) {
        (AxPrecisionPolicy::PromoteFp8ToF16, DType::B8) => AxInputTransform::B8ToF16,
        (AxPrecisionPolicy::PromoteFp8ToF16, DType::F8) => AxInputTransform::F8ToF16,
        _ => AxInputTransform::Identity,
    }
}

const fn input_transform(dtype: DType, policy: AxLoweringPolicy) -> AxInputTransform {
    if matches!(
        (policy.boolean_boundaries, dtype),
        (AxBooleanBoundaryPolicy::F32Bridge, DType::U8)
    ) {
        AxInputTransform::BoolU8ToF32
    } else {
        precision_input_transform(dtype, policy.precision)
    }
}

const fn output_transform(dtype: DType, policy: AxLoweringPolicy) -> AxOutputTransform {
    if matches!(
        (policy.boolean_boundaries, dtype),
        (AxBooleanBoundaryPolicy::F32Bridge, DType::U8)
    ) {
        AxOutputTransform::BoolF32ToU8
    } else {
        AxOutputTransform::Identity
    }
}

const fn precision_dtype(dtype: DType, precision: AxPrecisionPolicy) -> DType {
    match precision_input_transform(dtype, precision) {
        AxInputTransform::B8ToF16 | AxInputTransform::F8ToF16 => DType::F16,
        AxInputTransform::Identity | AxInputTransform::BoolU8ToF32 => dtype,
    }
}

fn tosa_dtype(dtype: DType) -> Result<TosaDType, AxArtifactError> {
    Ok(match dtype {
        DType::F32 => TosaDType::FP32,
        DType::F16 => TosaDType::FP16,
        DType::B16 => TosaDType::BF16,
        DType::I32 => TosaDType::INT32,
        DType::I16 => TosaDType::INT16,
        DType::I8 => TosaDType::INT8,
        DType::U8 => TosaDType::BOOL,
        DType::B8 => TosaDType::FP8E4M3,
        DType::F8 => TosaDType::FP8E5M2,
        other => return Err(AxArtifactError::UnsupportedDType(other)),
    })
}

fn tensor_shape(tensor: ax::TensorView<'_>) -> Result<Vec<i32>, AxArtifactError> {
    tensor
        .dimensions()
        .map(|dimension| {
            if dimension == ax::DYNAMIC_DIM {
                Err(AxArtifactError::DynamicShape(tensor.id))
            } else {
                i32::try_from(dimension).map_err(|_| AxArtifactError::DimensionTooLarge(tensor.id))
            }
        })
        .collect()
}

fn tensor_bytes(tensor: ax::TensorView<'_>) -> Result<usize, AxArtifactError> {
    let mut elements = 1_usize;
    for dimension in tensor.dimensions() {
        if dimension == ax::DYNAMIC_DIM {
            return Err(AxArtifactError::DynamicShape(tensor.id));
        }
        elements = elements
            .checked_mul(dimension as usize)
            .ok_or(AxArtifactError::SizeOverflow(tensor.id))?;
    }
    elements
        .checked_mul(tensor.dtype.storage_bytes(1))
        .ok_or(AxArtifactError::SizeOverflow(tensor.id))
}
