//! Admission of a trusted offline compilation for a concrete ax graph.
//!
//! Hashes bind immutable inputs; they do not prove compiler correctness or
//! authenticate an arbitrary build record. The application must obtain the
//! record from its trusted build/package path. Device compatibility and real
//! compiler-port bounds remain the Accelerator provider's responsibility.
//! No allocation, device operation, or ownership transfer occurs at admission.

use ax::{DType, Graph, NodeFlags, OpCode, ScheduleScratch, TensorFlags, TensorId, TensorView};
use axnn::lowering::KernelAccuracy;
use kore_svc_accel::{
    contract::{AccessMode, ArtifactFormat},
    service::{AccelBinding, AccelBindings, AccelRequest, MAX_ACCEL_BINDINGS},
};

/// The shared host/native profile. F32 interfaces with approximate FP16
/// computation, never a faithful or strict F32 implementation.
pub const PROFILE: u32 = 1;
/// Exact fixed-geometry sink-GQA block profile; I32 controls are its only
/// non-F32 tensor. Other graph shapes remain under [`PROFILE`].
pub const PROFILE_SINK_GQA_BLOCK: u32 = 2;
/// Per-output-channel I8 weight projection with F32 activations and scales.
/// This profile does not assert I8 activation execution.
pub const PROFILE_I8_PER_CHANNEL: u32 = 3;
/// Exact six-port, two-output Q/K/V preparation for a sink-GQA block.
pub const PROFILE_QKV_PREP_BLOCK: u32 = 4;
/// Fixed six-port gated MLP over two dynamic signed-I8 weight matrices.
pub const PROFILE_MLP_GATE_UP_I8_BLOCK: u32 = 5;
/// Three separate DSpark captures, one I8 weight view, one scale view, and
/// one partial output. The two halves use the same artifact with disjoint
/// persistent weight views.
pub const PROFILE_CONTEXT_PARTIAL_I8_BLOCK: u32 = 6;
/// Fixed five-port I8 down projection with device-side residual addition.
pub const PROFILE_MLP_DOWN_RESIDUAL_I8_BLOCK: u32 = 7;
/// DSpark context partial sum, hidden norm, and one layer's K/V projection.
pub const PROFILE_CONTEXT_KV_BLOCK: u32 = 8;
/// Fused DSpark sink attention, I8 O projection, and residual. Its packed
/// 1032-slot ring/block view remains a borrowed DMA input, never embedded.
pub const PROFILE_SINK_GQA_O_RESIDUAL_BLOCK: u32 = 9;
/// One exact eight-row DSpark residual RMSNorm. The per-layer or final scale
/// vector stays a dynamic input so the same compiled program serves both.
pub const PROFILE_DSPARK_RMS_NORM: u32 = 15;
/// Exact resident I8 K/V producer using canonical archive rows and scales.
pub const PROFILE_CONTEXT_KV_I8_BLOCK: u32 = 13;
/// BF16 target-anchor and DSpark mask rows are gathered into one eight-row
/// F32 input without a CPU-side tensor copy. The graph uses portable AXIR
/// GATHER and SELECT operations; this number identifies its native artifact.
pub const PROFILE_DSPARK_EMBED_ROWS: u32 = 14;
/// Shared target LM head over eight final-normed rows and one I8 weight view.
pub const PROFILE_DSPARK_LM_HEAD_I8: u32 = 10;
/// Eight serial Markov choices, with feedback kept inside one device event.
pub const PROFILE_DSPARK_MARKOV_FEEDBACK: u32 = 11;
/// Eight-row Q/K/V preparation against output-major resident I8 weights.
pub const PROFILE_QKV_PREP_I8_BLOCK: u32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Graph,
    Schedule,
    Shape,
    Constant,
    UnsupportedOperation(u16),
    Effects,
    Profile,
    Accuracy,
    Identity,
    Ports,
    Budget,
    Binding,
    Region,
}

/// Metadata, not authority to a device buffer. Slots follow the selected
/// provider's binding ABI and must be unique across all input/output ports.
pub struct ArtifactPort {
    pub tensor: TensorId,
    pub slot: u32,
}

/// A trusted compiler/package record. Keep its storage immutable for the
/// lifetime of the admitted graph. It is not a certificate for untrusted JSON.
pub struct ArtifactBuild<'a> {
    pub profile: u32,
    pub graph_blake3: [u8; 32],
    pub artifact_blake3: [u8; 32],
    pub format: u32,
    pub target: [u32; 12],
    pub inputs: &'a [ArtifactPort],
    pub outputs: &'a [ArtifactPort],
}

/// Caller-supplied admission budget, not an implicit DMA reservation. Actual
/// residency is enforced again by the provider while loading the artifact.
pub struct Policy {
    pub minimum_accuracy: KernelAccuracy,
    pub max_tensor_bytes: u64,
    pub max_resident_bytes: u64,
}

pub fn tensor_bytes(tensor: TensorView<'_>) -> Result<u64, Error> {
    if !matches!(
        tensor.dtype,
        DType::F32 | DType::I32 | DType::I8 | DType::B16 | DType::U8
    ) || tensor.rank() == 0
    {
        return Err(Error::Shape);
    }
    let element_bytes: u64 = match tensor.dtype {
        DType::I8 | DType::U8 => 1,
        DType::B16 => 2,
        _ => 4,
    };
    tensor.dimensions().try_fold(element_bytes, |n, d| {
        if d == 0 || d == ax::DYNAMIC_DIM {
            return Err(Error::Shape);
        }
        n.checked_mul(u64::from(d)).ok_or(Error::Shape)
    })
}

/// One semantic profile used by both native admission and the host exporter.
/// DAG ordering is separately checked by ax's scheduler with caller scratch.
pub fn validate_profile(graph: &Graph<'_>, max_tensor_bytes: u64) -> Result<(), Error> {
    let mut inputs = 0;
    let mut outputs = 0;
    if graph.node_count() == 0 {
        return Err(Error::Graph);
    }
    for index in 0..graph.tensor_count() {
        let t = graph
            .tensor(TensorId(index as u32))
            .map_err(|_| Error::Graph)?;
        if t.dtype != DType::F32 {
            return Err(Error::Shape);
        }
        let bytes = tensor_bytes(t)?;
        if bytes > max_tensor_bytes {
            return Err(Error::Budget);
        }
        if t.flags.contains(TensorFlags::INPUT) {
            inputs += 1;
            if t.flags.contains(TensorFlags::CONSTANT) {
                return Err(Error::Constant);
            }
        }
        if t.flags.contains(TensorFlags::OUTPUT) {
            outputs += 1;
        }
        if t.flags.contains(TensorFlags::CONSTANT)
            && (t.constant().len() as u64 != bytes
                || t.constant()
                    .chunks_exact(4)
                    .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite()))
        {
            return Err(Error::Constant);
        }
    }
    if inputs == 0 || outputs == 0 || inputs + outputs > MAX_ACCEL_BINDINGS {
        return Err(Error::Ports);
    }
    for index in 0..graph.node_count() {
        let node = graph.node(index).map_err(|_| Error::Graph)?;
        if node.flags != NodeFlags::default() {
            return Err(Error::Effects);
        }
        if !matches!(
            node.op,
            OpCode::IDENTITY
                | OpCode::COPY
                | OpCode::ADD
                | OpCode::SUB
                | OpCode::MUL
                | OpCode::MINIMUM
                | OpCode::EXP
                | OpCode::MATMUL
                | OpCode::RELU
        ) {
            return Err(Error::UnsupportedOperation(node.op.0));
        }
        if ax::node_kernel_request(graph, node).is_none() {
            return Err(Error::Shape);
        }
    }
    Ok(())
}

/// Admit only the six tensors and one exact, pure sink-GQA block node.
/// The I32 control record remains typed; no arbitrary I32 graph is admitted.
pub fn validate_sink_gqa_profile(graph: &Graph<'_>, max_tensor_bytes: u64) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 6 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if !ax::sink_gqa::validate_node(graph, node) {
        return Err(Error::Shape);
    }
    for id in 0..6 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.flags
            != if id < 5 {
                TensorFlags::INPUT
            } else {
                TensorFlags::OUTPUT
            }
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit only the fixed, weight-free Q/K/V producer signature. Its two
/// outputs are the query and block-K/V inputs of `SINK_GQA_BLOCK`.
pub fn validate_qkv_prep_profile(graph: &Graph<'_>, max_tensor_bytes: u64) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 6 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if !ax::qkv_prep::validate_node(graph, node)
        || !node
            .inputs()
            .eq([TensorId(0), TensorId(1), TensorId(2), TensorId(3)])
        || !node.outputs().eq([TensorId(4), TensorId(5)])
    {
        return Err(Error::Shape);
    }
    for id in 0..6 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.flags
            != if id < 4 {
                TensorFlags::INPUT
            } else {
                TensorFlags::OUTPUT
            }
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit exactly one per-channel I8-weight projection with four dynamic ports.
/// The graph contract leaves activation values in F32; compiler precision is
/// separate evidence and must not be inferred from the I8 storage dtype.
pub fn validate_i8_per_channel_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 4 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if node.op != OpCode::MATMUL_I8_PER_CHANNEL
        || node.flags != NodeFlags::default()
        || !node.inputs().eq([TensorId(0), TensorId(1), TensorId(2)])
        || !node.outputs().eq([TensorId(3)])
        || ax::node_kernel_request(graph, node).is_none()
    {
        return Err(Error::Shape);
    }
    for id in 0..4 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.flags
            != if id < 3 {
                TensorFlags::INPUT
            } else {
                TensorFlags::OUTPUT
            }
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// A complete gate/up producer for one eight-row DSpark block. The hidden
/// input has already passed the layer's post-attention RMSNorm; the two
/// weight/scale pairs remain dynamic ports and can be bound to resident DMA.
pub fn validate_mlp_gate_up_i8_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 6 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if !ax::mlp_gate_up_i8::validate_node(graph, node) {
        return Err(Error::Shape);
    }
    for id in 0..6 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.flags
            != if id < 5 {
                TensorFlags::INPUT
            } else {
                TensorFlags::OUTPUT
            }
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit only the exact three-capture FC partial. Its six distinct tensor
/// ports can bind separate decoder capture DMA views without a host concat.
pub fn validate_context_partial_i8_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 6 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if !ax::context_partial_i8::validate_node(graph, node) {
        return Err(Error::Shape);
    }
    for id in 0..6 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.flags
            != if id < 5 {
                TensorFlags::INPUT
            } else {
                TensorFlags::OUTPUT
            }
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit the exact down/residual producer for one eight-row DSpark layer.
/// Weight rows and scales remain external; the residual is a separate input
/// and the output is a distinct destination, preserving DMA ownership.
pub fn validate_mlp_down_residual_i8_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 5 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if !ax::mlp_down_residual_i8::validate_node(graph, node) {
        return Err(Error::Shape);
    }
    for id in 0..5 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.flags
            != if id < 4 {
                TensorFlags::INPUT
            } else {
                TensorFlags::OUTPUT
            }
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit the exact six-port context consumer. The packed partials occupy one
/// borrowed DMA view; K and V remain separate output destinations.
pub fn validate_context_kv_profile(graph: &Graph<'_>, max_tensor_bytes: u64) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 6 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if !ax::context_kv::validate_node(graph, node) {
        return Err(Error::Shape);
    }
    for id in 0..6 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.flags
            != if id < 4 {
                TensorFlags::INPUT
            } else {
                TensorFlags::OUTPUT
            }
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// The dynamic weight and parameter ports borrow the model archive directly.
/// Output rows retain opcode 86's separate K/V ring binding geometry.
pub fn validate_context_kv_i8_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 6 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if !ax::context_kv::validate_i8_node(graph, node) {
        return Err(Error::Shape);
    }
    for id in 0..6 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.flags
            != if id < 4 {
                TensorFlags::INPUT
            } else {
                TensorFlags::OUTPUT
            }
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit one exact copy-free embedding prelude. A binding may point at a
/// single row inside the retained target embedding import, so the graph
/// itself needs only one BF16 anchor row rather than a 704 MiB table input.
pub fn validate_dspark_embed_rows_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.tensor_count() != 7 || graph.node_count() != 3 {
        return Err(Error::Graph);
    }
    let expected = [
        (DType::B16, TensorFlags::INPUT, &[1, 2688][..]),
        (DType::B16, TensorFlags::INPUT, &[1, 2688][..]),
        (DType::I32, TensorFlags::INPUT, &[8][..]),
        (DType::U8, TensorFlags::INPUT, &[8, 1][..]),
        (DType::F32, TensorFlags::default(), &[8, 2688][..]),
        (DType::F32, TensorFlags::default(), &[8, 2688][..]),
        (DType::F32, TensorFlags::OUTPUT, &[8, 2688][..]),
    ];
    for (id, (dtype, flags, shape)) in expected.into_iter().enumerate() {
        let tensor = graph
            .tensor(TensorId(id as u32))
            .map_err(|_| Error::Graph)?;
        if tensor.dtype != dtype
            || tensor.flags != flags
            || !tensor.dimensions().eq(shape.iter().copied())
            || !tensor.constant().is_empty()
        {
            return Err(Error::Shape);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    let nodes = [
        (OpCode::GATHER, &[0, 2][..], &[4][..]),
        (OpCode::GATHER, &[1, 2][..], &[5][..]),
        (OpCode::SELECT, &[3, 4, 5][..], &[6][..]),
    ];
    for (id, (op, inputs, outputs)) in nodes.into_iter().enumerate() {
        let node = graph.node(id).map_err(|_| Error::Graph)?;
        if node.op != op
            || node.flags != NodeFlags::default()
            || !node.attributes().is_empty()
            || !node
                .inputs()
                .map(|tensor| tensor.0)
                .eq(inputs.iter().copied())
            || !node
                .outputs()
                .map(|tensor| tensor.0)
                .eq(outputs.iter().copied())
            || ax::node_kernel_request(graph, node).is_none()
        {
            return Err(Error::Shape);
        }
    }
    Ok(())
}

/// Admit exactly the six dynamic ports of AXIR op87. This authenticates a
/// single artifact's shape; publication of the shadow ring and the op82
/// producer/consumer alias proof belong to the ordered caller.
pub fn validate_sink_gqa_o_residual_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 6 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if !ax::sink_gqa_o_residual::validate_node(graph, node) {
        return Err(Error::Shape);
    }
    for id in 0..6 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.flags
            != if id < 5 {
                TensorFlags::INPUT
            } else {
                TensorFlags::OUTPUT
            }
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit the existing AXIR RMS_NORM operation only at the DSpark hidden
/// geometry and epsilon. The dynamic norm scale is borrowed from the model's
/// resident DMA view; no checkpoint value is embedded in the artifact.
pub fn validate_dspark_rms_norm_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 3 {
        return Err(Error::Graph);
    }
    let node = graph.node(0).map_err(|_| Error::Graph)?;
    if node.op != OpCode::RMS_NORM
        || node.flags != NodeFlags::default()
        || !node.inputs().eq([TensorId(0), TensorId(1)])
        || !node.outputs().eq([TensorId(2)])
        || node.attributes().len() != ax::tensor_ops::RMS_NORM_AXIS_ATTRIBUTE_BYTES
        || ax::tensor_ops::decode_rms_norm_attributes(node.attributes()) != Some((1.0e-6_f32, -1))
        || ax::node_kernel_request(graph, node).is_none()
    {
        return Err(Error::Shape);
    }
    for (id, shape) in [(0, &[8, 2688][..]), (1, &[2688][..]), (2, &[8, 2688][..])] {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor.dtype != DType::F32
            || tensor.flags
                != if id < 2 {
                    TensorFlags::INPUT
                } else {
                    TensorFlags::OUTPUT
                }
            || !tensor.dimensions().eq(shape.iter().copied())
        {
            return Err(Error::Ports);
        }
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit only the typed target-head I8 profile. Conversion quality is a
/// separate model-local gate; admission authenticates its ports and artifact.
pub fn validate_dspark_lm_head_i8_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 4 {
        return Err(Error::Graph);
    }
    if !ax::dspark_lm_head_i8::validate_node(graph, graph.node(0).map_err(|_| Error::Graph)?) {
        return Err(Error::Shape);
    }
    for id in 0..4 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit the complete six-port Markov feedback profile. The caller retains
/// every weight and intermediate DMA lease through event retirement.
pub fn validate_dspark_markov_feedback_profile(
    graph: &Graph<'_>,
    max_tensor_bytes: u64,
) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 6 {
        return Err(Error::Graph);
    }
    if !ax::dspark_markov_feedback::validate_node(graph, graph.node(0).map_err(|_| Error::Graph)?) {
        return Err(Error::Shape);
    }
    for id in 0..6 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Admit only the typed I8 QKV geometry and one six-port native graph.
pub fn validate_qkv_prep_i8_profile(graph: &Graph<'_>, max_tensor_bytes: u64) -> Result<(), Error> {
    if graph.node_count() != 1 || graph.tensor_count() != 6 {
        return Err(Error::Graph);
    }
    if !ax::qkv_prep_i8::validate_node(graph, graph.node(0).map_err(|_| Error::Graph)?) {
        return Err(Error::Shape);
    }
    for id in 0..6 {
        let tensor = graph.tensor(TensorId(id)).map_err(|_| Error::Graph)?;
        if tensor_bytes(tensor)? > max_tensor_bytes {
            return Err(Error::Budget);
        }
    }
    Ok(())
}

/// Immutable admission witness. Execution handles and their cleanup remain
/// owned by the client issuing requests through its possessed Accelerator face.
pub struct AdmittedGraph<'a> {
    graph: Graph<'a>,
    artifact: &'a [u8],
    build: &'a ArtifactBuild<'a>,
    resident_bytes: u64,
}

impl<'a> AdmittedGraph<'a> {
    pub fn admit(
        graph_bytes: &'a [u8],
        artifact: &'a [u8],
        build: &'a ArtifactBuild<'a>,
        policy: Policy,
        resident_bytes: u64,
        scratch: &mut ScheduleScratch<'_>,
    ) -> Result<Self, Error> {
        if !matches!(
            build.profile,
            PROFILE
                | PROFILE_SINK_GQA_BLOCK
                | PROFILE_I8_PER_CHANNEL
                | PROFILE_QKV_PREP_BLOCK
                | PROFILE_MLP_GATE_UP_I8_BLOCK
                | PROFILE_CONTEXT_PARTIAL_I8_BLOCK
                | PROFILE_MLP_DOWN_RESIDUAL_I8_BLOCK
                | PROFILE_CONTEXT_KV_BLOCK
                | PROFILE_SINK_GQA_O_RESIDUAL_BLOCK
                | PROFILE_DSPARK_RMS_NORM
                | PROFILE_CONTEXT_KV_I8_BLOCK
                | PROFILE_DSPARK_EMBED_ROWS
                | PROFILE_DSPARK_LM_HEAD_I8
                | PROFILE_DSPARK_MARKOV_FEEDBACK
                | PROFILE_QKV_PREP_I8_BLOCK
        ) || ArtifactFormat::new(build.format).is_none()
        {
            return Err(Error::Profile);
        }
        if policy.minimum_accuracy > KernelAccuracy::Approximate {
            return Err(Error::Accuracy);
        }
        if build
            .inputs
            .len()
            .checked_add(build.outputs.len())
            .is_none_or(|n| n > MAX_ACCEL_BINDINGS)
        {
            return Err(Error::Ports);
        }
        if artifact.is_empty()
            || resident_bytes < artifact.len() as u64
            || resident_bytes > policy.max_resident_bytes
        {
            return Err(Error::Budget);
        }
        if *blake3::hash(graph_bytes).as_bytes() != build.graph_blake3
            || *blake3::hash(artifact).as_bytes() != build.artifact_blake3
        {
            return Err(Error::Identity);
        }
        let graph = Graph::parse(graph_bytes).map_err(|_| Error::Graph)?;
        match build.profile {
            PROFILE_SINK_GQA_BLOCK => validate_sink_gqa_profile(&graph, policy.max_tensor_bytes)?,
            PROFILE_I8_PER_CHANNEL => {
                validate_i8_per_channel_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_QKV_PREP_BLOCK => validate_qkv_prep_profile(&graph, policy.max_tensor_bytes)?,
            PROFILE_MLP_GATE_UP_I8_BLOCK => {
                validate_mlp_gate_up_i8_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_CONTEXT_PARTIAL_I8_BLOCK => {
                validate_context_partial_i8_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_MLP_DOWN_RESIDUAL_I8_BLOCK => {
                validate_mlp_down_residual_i8_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_CONTEXT_KV_BLOCK => {
                validate_context_kv_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_CONTEXT_KV_I8_BLOCK => {
                validate_context_kv_i8_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_DSPARK_EMBED_ROWS => {
                validate_dspark_embed_rows_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_SINK_GQA_O_RESIDUAL_BLOCK => {
                validate_sink_gqa_o_residual_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_DSPARK_RMS_NORM => {
                validate_dspark_rms_norm_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_DSPARK_LM_HEAD_I8 => {
                validate_dspark_lm_head_i8_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_DSPARK_MARKOV_FEEDBACK => {
                validate_dspark_markov_feedback_profile(&graph, policy.max_tensor_bytes)?
            }
            PROFILE_QKV_PREP_I8_BLOCK => {
                validate_qkv_prep_i8_profile(&graph, policy.max_tensor_bytes)?
            }
            _ => validate_profile(&graph, policy.max_tensor_bytes)?,
        }
        ax::analyze(&graph, scratch).map_err(|_| Error::Schedule)?;
        for (flag, ports) in [
            (TensorFlags::INPUT, build.inputs),
            (TensorFlags::OUTPUT, build.outputs),
        ] {
            let mut next = ports.iter();
            for index in 0..graph.tensor_count() {
                let tensor = graph
                    .tensor(TensorId(index as u32))
                    .map_err(|_| Error::Graph)?;
                if tensor.flags.contains(flag) && next.next().is_none_or(|p| p.tensor != tensor.id)
                {
                    return Err(Error::Ports);
                }
            }
            if next.next().is_some() {
                return Err(Error::Ports);
            }
        }
        let ports = build.inputs.iter().chain(build.outputs);
        for (index, port) in ports.clone().enumerate() {
            if ports.clone().take(index).any(|p| p.slot == port.slot) {
                return Err(Error::Ports);
            }
        }
        Ok(Self {
            graph,
            artifact,
            build,
            resident_bytes,
        })
    }

    pub fn artifact(&self) -> &'a [u8] {
        self.artifact
    }
    pub fn graph(&self) -> Graph<'a> {
        self.graph
    }
    pub fn input_bytes(&self, index: usize) -> Result<u64, Error> {
        self.port_bytes(self.build.inputs.get(index).ok_or(Error::Ports)?)
    }
    pub fn output_bytes(&self, index: usize) -> Result<u64, Error> {
        self.port_bytes(self.build.outputs.get(index).ok_or(Error::Ports)?)
    }
    fn port_bytes(&self, port: &ArtifactPort) -> Result<u64, Error> {
        tensor_bytes(self.graph.tensor(port.tensor).map_err(|_| Error::Graph)?)
    }
    pub fn check_input_f32(&self, index: usize, data: &[f32]) -> Result<(), Error> {
        let port = self.build.inputs.get(index).ok_or(Error::Ports)?;
        if self
            .graph
            .tensor(port.tensor)
            .map_err(|_| Error::Graph)?
            .dtype
            != DType::F32
            || self.input_bytes(index)? != core::mem::size_of_val(data) as u64
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    pub fn load_request(&self, ctx: u64, stage: u64) -> Result<AccelRequest, Error> {
        stage
            .checked_add(self.artifact.len() as u64)
            .ok_or(Error::Budget)?;
        Ok(AccelRequest::LoadProgram {
            ctx,
            stage,
            len: self.artifact.len() as u64,
            resident_bytes: self.resident_bytes,
            format: self.build.format,
            target: self.build.target,
        })
    }
    /// Buffer IDs are scoped by the possessed client face and checked again
    /// by the service. The list is in graph input order followed by outputs.
    /// This operation neither allocates nor transfers those buffers.
    pub fn bindings(&self, buffers: &[u64]) -> Result<AccelBindings, Error> {
        self.bind_ports(buffers.len(), |index| (buffers[index], 0))
    }

    /// Bind tensors inside already imported buffers without copying them to
    /// offset zero. Each tuple is `(buffer ID, byte offset)` in graph input
    /// order followed by outputs. Offsets are relative to the imported buffer,
    /// including when its fabric route starts at a page boundary before the
    /// tensor. The service still checks the buffer's actual extent and usage.
    pub fn bindings_with_offsets(&self, buffers: &[(u64, u64)]) -> Result<AccelBindings, Error> {
        self.bind_ports(buffers.len(), |index| buffers[index])
    }

    fn bind_ports(
        &self,
        count: usize,
        mut at: impl FnMut(usize) -> (u64, u64),
    ) -> Result<AccelBindings, Error> {
        if count != self.build.inputs.len() + self.build.outputs.len() {
            return Err(Error::Binding);
        }
        let mut list = [AccelBinding {
            buffer: 0,
            offset: 0,
            len: 0,
            slot: 0,
            access: AccessMode::Read,
        }; MAX_ACCEL_BINDINGS];
        if count > list.len() {
            return Err(Error::Binding);
        }
        for (index, port) in self
            .build
            .inputs
            .iter()
            .chain(self.build.outputs)
            .enumerate()
        {
            let (buffer, offset) = at(index);
            if buffer == 0 {
                return Err(Error::Binding);
            }
            let len = self.port_bytes(port)?;
            let end = offset.checked_add(len).ok_or(Error::Binding)?;
            // Pure graph outputs may share an allocation with inputs or other
            // outputs only when their byte ranges are disjoint. Repeated
            // read-only inputs may overlap.
            if index >= self.build.inputs.len()
                && list[..index].iter().any(|prior| {
                    prior.buffer == buffer
                        && prior.offset < end
                        && offset < prior.offset + prior.len
                })
            {
                return Err(Error::Binding);
            }
            list[index] = AccelBinding {
                buffer,
                offset,
                len,
                slot: port.slot,
                access: if index < self.build.inputs.len() {
                    AccessMode::Read
                } else {
                    AccessMode::Write
                },
            };
        }
        AccelBindings::new(&list[..count]).ok_or(Error::Binding)
    }
}

/// Trusted extraction record. `tensors[local_id]` names the parent tensor;
/// entries are strictly ascending, matching canonical extraction. Coordinates
/// select a contiguous interval in the parent's canonical topological order.
pub struct RegionBuild<'a> {
    pub source_graph_blake3: [u8; 32],
    pub order_start: u32,
    pub node_count: u32,
    pub tensors: &'a [TensorId],
}

/// A compiled graph whose exact relationship to its parent has been checked.
/// Buffer IDs still belong to the possessed Accelerator face; this witness
/// grants no allocation, mapping, completion, or transfer authority.
pub struct AdmittedRegion<'a> {
    graph: AdmittedGraph<'a>,
    parent: Graph<'a>,
    build: &'a RegionBuild<'a>,
}

impl<'a> AdmittedGraph<'a> {
    /// Validate extraction using caller-owned scratch, without allocation.
    /// Both scratch areas are clobbered. The parent may contain non-NPU work
    /// outside this region; its complete dependency graph must still be valid.
    pub fn into_region(
        self,
        parent_bytes: &'a [u8],
        build: &'a RegionBuild<'a>,
        scratch: &mut ScheduleScratch<'_>,
        tensor_scratch: &mut [u8],
    ) -> Result<AdmittedRegion<'a>, Error> {
        if *blake3::hash(parent_bytes).as_bytes() != build.source_graph_blake3 {
            return Err(Error::Identity);
        }
        let parent = Graph::parse(parent_bytes).map_err(|_| Error::Graph)?;
        let end = build
            .order_start
            .checked_add(build.node_count)
            .ok_or(Error::Region)? as usize;
        if build.node_count == 0
            || end > parent.node_count()
            || build.node_count as usize != self.graph.node_count()
            || build.tensors.len() != self.graph.tensor_count()
        {
            return Err(Error::Region);
        }
        ax::analyze(&parent, scratch).map_err(|_| Error::Schedule)?;
        let flags = tensor_scratch
            .get_mut(..parent.tensor_count())
            .ok_or(Error::Schedule)?;
        flags.fill(0);
        let selected = &mut scratch.indegree[..parent.node_count()];
        selected.fill(0);
        let order = &scratch.order[build.order_start as usize..end];
        for &node in order {
            selected[node as usize] = 1;
        }
        const USED: u8 = 1;
        const PRODUCED: u8 = 2;
        const EXTERNAL_READ: u8 = 4;
        for (id, &inside) in selected.iter().enumerate() {
            let node = parent.node(id).map_err(|_| Error::Graph)?;
            for tensor in node.inputs() {
                flags[tensor.0 as usize] |= if inside != 0 { USED } else { EXTERNAL_READ };
            }
            if inside != 0 {
                for tensor in node.outputs() {
                    flags[tensor.0 as usize] |= USED | PRODUCED;
                }
            }
        }
        let mut local_id = 0;
        for (id, &use_flags) in flags.iter().enumerate() {
            if use_flags & USED == 0 {
                continue;
            }
            if build.tensors.get(local_id) != Some(&TensorId(id as u32)) {
                return Err(Error::Region);
            }
            let source = parent
                .tensor(TensorId(id as u32))
                .map_err(|_| Error::Graph)?;
            let local = self
                .graph
                .tensor(TensorId(local_id as u32))
                .map_err(|_| Error::Region)?;
            let constant = source.flags.contains(TensorFlags::CONSTANT);
            let input = use_flags & PRODUCED == 0 && !constant;
            let output = use_flags & PRODUCED != 0
                && (use_flags & EXTERNAL_READ != 0 || source.flags.contains(TensorFlags::OUTPUT));
            let expected_flags = TensorFlags(
                if constant { TensorFlags::CONSTANT.0 } else { 0 }
                    | if input { TensorFlags::INPUT.0 } else { 0 }
                    | if output { TensorFlags::OUTPUT.0 } else { 0 },
            );
            if local.dtype != source.dtype
                || !local.dimensions().eq(source.dimensions())
                || local.constant() != source.constant()
                || local.flags != expected_flags
            {
                return Err(Error::Region);
            }
            local_id += 1;
        }
        if local_id != build.tensors.len() {
            return Err(Error::Region);
        }
        for (local_id, &source_id) in order.iter().enumerate() {
            let source = parent.node(source_id as usize).map_err(|_| Error::Graph)?;
            let local = self.graph.node(local_id).map_err(|_| Error::Region)?;
            if local.op != source.op
                || local.flags != source.flags
                || local.attributes() != source.attributes()
                || !local
                    .inputs()
                    .map(|id| build.tensors[id.0 as usize])
                    .eq(source.inputs())
                || !local
                    .outputs()
                    .map(|id| build.tensors[id.0 as usize])
                    .eq(source.outputs())
            {
                return Err(Error::Region);
            }
        }
        Ok(AdmittedRegion {
            graph: self,
            parent,
            build,
        })
    }
}

impl<'a> AdmittedRegion<'a> {
    pub fn graph(&self) -> &AdmittedGraph<'a> {
        &self.graph
    }

    /// Resolve only this region's ports from a parent-tensor-indexed residency
    /// table. Zero is absent; each required buffer must be live on this face.
    /// Graph-level range/alias checks remain in AdmittedGraph::bindings;
    /// the service checks each imported buffer's actual extent and usage.
    pub fn bindings(&self, buffers: &[u64]) -> Result<AccelBindings, Error> {
        self.bind_parent_ports(buffers.len(), |index| (buffers[index], 0))
    }

    /// Resolve parent-tensor-indexed `(buffer ID, byte offset)` residency for
    /// this region. This is useful when distinct tensors occupy disjoint
    /// spans of one imported, page-rounded shared-DMA allocation.
    pub fn bindings_with_offsets(&self, buffers: &[(u64, u64)]) -> Result<AccelBindings, Error> {
        self.bind_parent_ports(buffers.len(), |index| buffers[index])
    }

    fn bind_parent_ports(
        &self,
        count: usize,
        mut at: impl FnMut(usize) -> (u64, u64),
    ) -> Result<AccelBindings, Error> {
        if count != self.parent.tensor_count() {
            return Err(Error::Binding);
        }
        let mut local = [(0, 0); MAX_ACCEL_BINDINGS];
        let ports = self
            .graph
            .build
            .inputs
            .iter()
            .chain(self.graph.build.outputs);
        let mut count = 0;
        for port in ports {
            local[count] = at(self.build.tensors[port.tensor.0 as usize].0 as usize);
            count += 1;
        }
        self.graph.bindings_with_offsets(&local[..count])
    }
}

/// Caller-owned scratch for lifetime-checked sequence residency. Fields are
/// recomputed from admitted compiler ports; caller contents grant no authority.
#[derive(Clone, Copy, Default)]
pub struct ResidencyInterval {
    tensor: usize,
    buffer: u64,
    first: usize,
    last: usize,
}

/// Caller scratch for checking disjoint tensor ranges in a shared DMA buffer.
/// It carries no allocation or authority; the provider validates actual
/// imported-buffer bounds and usage at submission.
#[derive(Clone, Copy, Default)]
pub struct BufferRange {
    buffer: u64,
    start: u64,
    end: u64,
}

/// One explicitly declared aggregate read of complete, consecutive outputs.
/// The order of `producers` is their byte order in the consumer's buffer.
/// This is a storage relationship between parent tensors, not an AXIR node.
pub struct PackedView<'a> {
    pub consumer: TensorId,
    pub producers: &'a [TensorId],
}

fn check_disjoint_buffer_ranges(ranges: &mut [BufferRange]) -> Result<(), Error> {
    ranges.sort_unstable_by_key(|range| (range.buffer, range.start));
    if ranges.windows(2).any(|pair| {
        pair[0].buffer != 0 && pair[0].buffer == pair[1].buffer && pair[0].end > pair[1].start
    }) {
        return Err(Error::Binding);
    }
    Ok(())
}

/// One planned Shared buffer, not an allocated object or DMA reservation.
#[derive(Clone, Copy, Default)]
pub struct IntermediateStorage {
    bytes: u64,
    last: usize,
}
impl IntermediateStorage {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn create_request(&self, ctx: u64) -> AccelRequest {
        AccelRequest::CreateBuffer {
            ctx,
            bytes: self.bytes,
            align: 4096,
            domain: 3,
            usage: kore_svc_accel::contract::BufferUsage::MUTABLE_STATE.bits(),
        }
    }
}

/// Automatic intermediate assignment in caller storage. External inputs and
/// outputs are excluded: their existing residency remains caller-owned.
pub struct IntermediatePlan<'a> {
    assignments: &'a [usize],
    storage: &'a [IntermediateStorage],
}
impl IntermediatePlan<'_> {
    pub fn storage(&self) -> &[IntermediateStorage] {
        self.storage
    }
    pub fn bytes(&self) -> u64 {
        self.storage.iter().map(|slot| slot.bytes).sum()
    }

    /// Install the IDs returned by actual allocations, preserving external
    /// residency. This copies IDs only; call bind_reusing for final admission.
    /// An error leaves the residency table unchanged.
    pub fn assign_buffers(&self, buffers: &[u64], residency: &mut [u64]) -> Result<(), Error> {
        if buffers.len() != self.storage.len()
            || buffers.contains(&0)
            || residency.len() != self.assignments.len()
        {
            return Err(Error::Binding);
        }
        for (resident, &slot) in residency.iter_mut().zip(self.assignments) {
            if slot != usize::MAX {
                *resident = buffers[slot];
            }
        }
        Ok(())
    }
}

fn check_residency_intervals(intervals: &mut [ResidencyInterval]) -> Result<(), Error> {
    intervals.sort_unstable_by_key(|span| (span.buffer, span.first));
    if intervals.windows(2).any(|pair| {
        pair[0].buffer != 0 && pair[0].buffer == pair[1].buffer && pair[0].last >= pair[1].first
    }) {
        return Err(Error::Binding);
    }
    Ok(())
}

/// A complete, ordered cover of one parent graph by admitted regions.
/// This is a native execution plan, not a placement decision or transfer fence.
/// All artifacts must already have been admitted under the caller's policy.
pub struct AdmittedSequence<'s, 'g> {
    regions: &'s [AdmittedRegion<'g>],
}

impl<'s, 'g> AdmittedSequence<'s, 'g> {
    pub fn new(regions: &'s [AdmittedRegion<'g>]) -> Result<Self, Error> {
        let first = regions.first().ok_or(Error::Region)?;
        let mut end = 0u32;
        for region in regions {
            if region.build.source_graph_blake3 != first.build.source_graph_blake3
                || region.build.order_start != end
            {
                return Err(Error::Region);
            }
            end = end
                .checked_add(region.build.node_count)
                .ok_or(Error::Region)?;
        }
        if end as usize != first.parent.node_count() {
            return Err(Error::Region);
        }
        Ok(Self { regions })
    }

    /// Identity of ordered artifacts, extracted graphs and their parent mapping.
    /// Encoding: domain, parent digest, LE u32 region count; then per region
    /// graph digest, artifact digest, LE u32 start/count/map-length/map entries.
    /// This binds the capture to a sequence; it does not certify the compiler.
    pub fn identity(&self) -> [u8; 32] {
        let mut hash = blake3::Hasher::new();
        hash.update(b"kore-ax-sequence-v1\0");
        hash.update(&self.regions[0].build.source_graph_blake3);
        hash.update(&(self.regions.len() as u32).to_le_bytes());
        for region in self.regions {
            hash.update(&region.graph.build.graph_blake3);
            hash.update(&region.graph.build.artifact_blake3);
            for value in [
                region.build.order_start,
                region.build.node_count,
                region.build.tensors.len() as u32,
            ] {
                hash.update(&value.to_le_bytes());
            }
            for tensor in region.build.tensors {
                hash.update(&tensor.0.to_le_bytes());
            }
        }
        *hash.finalize().as_bytes()
    }

    fn residency_intervals<'a>(
        &self,
        scratch: &'a mut [ResidencyInterval],
    ) -> Result<&'a mut [ResidencyInterval], Error> {
        let parent = &self.regions[0].parent;
        let spans = scratch
            .get_mut(..parent.tensor_count())
            .ok_or(Error::Schedule)?;
        for (tensor, span) in spans.iter_mut().enumerate() {
            *span = ResidencyInterval {
                tensor,
                buffer: 0,
                first: usize::MAX,
                last: 0,
            };
        }
        for (index, region) in self.regions.iter().enumerate() {
            for port in region
                .graph
                .build
                .inputs
                .iter()
                .chain(region.graph.build.outputs)
            {
                let tensor = region.build.tensors[port.tensor.0 as usize];
                let span = &mut spans[tensor.0 as usize];
                span.first = span.first.min(index);
                span.last = span.last.max(index);
            }
        }
        for span in spans.iter_mut() {
            let tensor = parent
                .tensor(TensorId(span.tensor as u32))
                .map_err(|_| Error::Graph)?;
            if tensor.flags.contains(TensorFlags::INPUT)
                || tensor.flags.contains(TensorFlags::OUTPUT)
            {
                span.first = 0;
                span.last = self.regions.len();
            }
        }
        Ok(spans)
    }

    /// Assign storage automatically using inclusive region lifetimes. Reuse an
    /// expired slot with the least capacity growth, breaking ties by size then
    /// slot order. Slots may grow during planning, before any allocation.
    /// This deterministic greedy policy is bounded, not an optimal packing claim.
    ///
    /// Every allocation is page-rounded and the sum must fit budget_bytes.
    /// Inputs, outputs and constants are never temporary slots. Scratch and
    /// assignments need one entry per parent tensor; storage bounds the number
    /// of distinct allocations. On failure all scratch contents are unspecified.
    pub fn plan_intermediates<'a>(
        &self,
        budget_bytes: u64,
        assignments: &'a mut [usize],
        storage: &'a mut [IntermediateStorage],
        scratch: &mut [ResidencyInterval],
    ) -> Result<IntermediatePlan<'a>, Error> {
        let parent = &self.regions[0].parent;
        let assignments = assignments
            .get_mut(..parent.tensor_count())
            .ok_or(Error::Schedule)?;
        assignments.fill(usize::MAX);
        let spans = self.residency_intervals(scratch)?;
        spans.sort_unstable_by_key(|span| (span.first, span.tensor));
        let mut used = 0;
        let mut total = 0u64;
        for span in spans {
            if span.first == usize::MAX {
                continue;
            }
            let tensor = parent
                .tensor(TensorId(span.tensor as u32))
                .map_err(|_| Error::Graph)?;
            if tensor.flags.contains(TensorFlags::INPUT)
                || tensor.flags.contains(TensorFlags::OUTPUT)
                || tensor.flags.contains(TensorFlags::CONSTANT)
            {
                continue;
            }
            let bytes = tensor_bytes(tensor)?
                .checked_add(4095)
                .ok_or(Error::Budget)?
                & !4095;
            let reuse = storage[..used]
                .iter()
                .enumerate()
                .filter(|(_, slot)| slot.last < span.first)
                .min_by_key(|(index, slot)| (bytes.saturating_sub(slot.bytes), slot.bytes, *index))
                .map(|(index, _)| index);
            let index = reuse.unwrap_or(used);
            let previous = if reuse.is_some() {
                storage[index].bytes
            } else {
                0
            };
            let capacity = previous.max(bytes);
            total = total
                .checked_add(capacity - previous)
                .filter(|sum| *sum <= budget_bytes)
                .ok_or(Error::Budget)?;
            let slot = storage.get_mut(index).ok_or(Error::Schedule)?;
            *slot = IntermediateStorage {
                bytes: capacity,
                last: span.last,
            };
            assignments[span.tensor] = index;
            if reuse.is_none() {
                used += 1;
            }
        }
        Ok(IntermediatePlan {
            assignments,
            storage: &storage[..used],
        })
    }

    /// Bind caller-selected buffer reuse after proving disjoint intermediate
    /// lifetimes across the complete region sequence. Inputs and final outputs
    /// stay pinned for the whole sequence. Two ports used by the same region
    /// cannot alias, even when that region is their producer/last consumer.
    ///
    /// Execute this iterator with `RingAccel::execution_sequence` or equivalent
    /// ordered completion and event retirement. Parallel region submission
    /// invalidates the lifetime proof. Actual buffer size, usage and ownership
    /// are still checked by the provider; this does not allocate storage.
    /// Scratch requires one entry per parent tensor and is clobbered.
    pub fn bind_reusing<'b>(
        &'b self,
        queue: u64,
        programs: &'b [u64],
        buffers: &'b [u64],
        timeout_ns: u64,
        scratch: &mut [ResidencyInterval],
    ) -> Result<impl Iterator<Item = kore_svc_accel::client::ArtifactExecution> + 'b, Error> {
        let parent = &self.regions[0].parent;
        if queue == 0
            || programs.len() != self.regions.len()
            || programs.contains(&0)
            || buffers.len() != parent.tensor_count()
        {
            return Err(Error::Binding);
        }
        for region in self.regions {
            region.bindings(buffers)?;
        }
        let spans = self.residency_intervals(scratch)?;
        for span in spans.iter_mut() {
            if span.first != usize::MAX {
                span.buffer = buffers[span.tensor];
            }
        }
        check_residency_intervals(spans)?;
        Ok(self.launches(queue, programs, buffers, timeout_ns))
    }

    /// Validate every launch before the first submission. Residency is indexed
    /// by parent tensor ID and belongs to the same possessed accelerator face.
    /// Programs must have been loaded from these regions in order. Their IDs
    /// alone cannot attest artifact identity. Caller scratch provides one u64
    /// per parent tensor for allocation-free O(n log n) alias validation.
    /// Distinct tensor entries must not share a buffer: reusing dead ranges
    /// requires a separately proven liveness allocator, not ID equality.
    pub fn bind<'b>(
        &'b self,
        queue: u64,
        programs: &'b [u64],
        buffers: &'b [u64],
        timeout_ns: u64,
        scratch: &mut [u64],
    ) -> Result<impl Iterator<Item = kore_svc_accel::client::ArtifactExecution> + 'b, Error> {
        if queue == 0
            || programs.len() != self.regions.len()
            || programs.contains(&0)
            || buffers.len() != self.regions[0].parent.tensor_count()
        {
            return Err(Error::Binding);
        }
        let sorted = scratch.get_mut(..buffers.len()).ok_or(Error::Schedule)?;
        sorted.copy_from_slice(buffers);
        sorted.sort_unstable();
        if sorted
            .windows(2)
            .any(|pair| pair[0] != 0 && pair[0] == pair[1])
        {
            return Err(Error::Binding);
        }
        for region in self.regions {
            region.bindings(buffers)?;
        }
        Ok(self.launches(queue, programs, buffers, timeout_ns))
    }

    /// Bind a complete sequence to offsets inside already imported buffers.
    /// Distinct parent tensors may share one buffer only at disjoint byte
    /// ranges; this intentionally does not reuse a range after its last read.
    /// The returned iterator retains the residency table through ordered
    /// execution and event retirement. No tensor bytes are copied here.
    ///
    /// Scratch needs one entry per parent tensor. The provider separately
    /// validates each buffer's actual extent, usage, and ownership.
    pub fn bind_with_offsets<'b>(
        &'b self,
        queue: u64,
        programs: &'b [u64],
        buffers: &'b [(u64, u64)],
        timeout_ns: u64,
        scratch: &mut [BufferRange],
    ) -> Result<impl Iterator<Item = kore_svc_accel::client::ArtifactExecution> + 'b, Error> {
        let parent = &self.regions[0].parent;
        if queue == 0
            || programs.len() != self.regions.len()
            || programs.contains(&0)
            || buffers.len() != parent.tensor_count()
        {
            return Err(Error::Binding);
        }
        let ranges = self.offset_ranges(buffers, scratch)?;
        check_disjoint_buffer_ranges(ranges)?;
        Ok(self.offset_launches(queue, programs, buffers, timeout_ns))
    }

    fn offset_ranges<'a>(
        &self,
        buffers: &[(u64, u64)],
        scratch: &'a mut [BufferRange],
    ) -> Result<&'a mut [BufferRange], Error> {
        let parent = &self.regions[0].parent;
        let ranges = scratch.get_mut(..buffers.len()).ok_or(Error::Schedule)?;
        ranges.fill(BufferRange::default());
        for region in self.regions {
            region.bindings_with_offsets(buffers)?;
            for port in region
                .graph
                .build
                .inputs
                .iter()
                .chain(region.graph.build.outputs)
            {
                let tensor = region.build.tensors[port.tensor.0 as usize];
                let index = tensor.0 as usize;
                let (buffer, start) = buffers[index];
                let bytes = tensor_bytes(parent.tensor(tensor).map_err(|_| Error::Graph)?)?;
                let end = start.checked_add(bytes).ok_or(Error::Binding)?;
                ranges[index] = BufferRange { buffer, start, end };
            }
        }
        Ok(ranges)
    }

    fn offset_launches<'b>(
        &'b self,
        queue: u64,
        programs: &'b [u64],
        buffers: &'b [(u64, u64)],
        timeout_ns: u64,
    ) -> impl Iterator<Item = kore_svc_accel::client::ArtifactExecution> + 'b {
        self.regions
            .iter()
            .zip(programs)
            .map(
                move |(region, &program)| kore_svc_accel::client::ArtifactExecution {
                    queue,
                    program,
                    timeout_ns,
                    bindings: region
                        .bindings_with_offsets(buffers)
                        .expect("validated immutable sequence residency"),
                },
            )
    }

    /// Admit complete producer outputs as one later aggregate input without
    /// moving their bytes. Every declared source must be an earlier region's
    /// output, and its ranges must exactly cover the consumer's input in the
    /// stated order. All other parent tensors remain strictly disjoint.
    ///
    /// The returned launches must execute in order with event retirement (or
    /// one provider-ordered event). The provider still checks buffer geometry,
    /// usage and ownership. The caller retains routes through that event.
    pub fn bind_with_packed_views<'b>(
        &'b self,
        queue: u64,
        programs: &'b [u64],
        buffers: &'b [(u64, u64)],
        views: &[PackedView<'_>],
        timeout_ns: u64,
        scratch: &mut [BufferRange],
    ) -> Result<impl Iterator<Item = kore_svc_accel::client::ArtifactExecution> + 'b, Error> {
        let parent = &self.regions[0].parent;
        if queue == 0
            || programs.len() != self.regions.len()
            || programs.contains(&0)
            || buffers.len() != parent.tensor_count()
            || views.is_empty()
            || self.regions.len() > kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES
            || views.len() > kore_abi::svc::MAX_ACCEL_SEQUENCE_ENTRIES
        {
            return Err(Error::Binding);
        }
        let ranges = self.offset_ranges(buffers, scratch)?;
        for view in views {
            let consumer = view.consumer.0 as usize;
            let tensor = parent.tensor(view.consumer).map_err(|_| Error::Graph)?;
            if !tensor.flags.contains(TensorFlags::INPUT)
                || tensor.flags.contains(TensorFlags::OUTPUT)
                || view.producers.is_empty()
                || view.producers.len() > self.regions.len()
            {
                return Err(Error::Binding);
            }
            let aggregate = *ranges.get(consumer).ok_or(Error::Binding)?;
            if aggregate.buffer == 0 {
                return Err(Error::Binding);
            }
            let first_read =
                self.regions
                    .iter()
                    .position(|region| {
                        region.graph.build.inputs.iter().any(|port| {
                            region.build.tensors[port.tensor.0 as usize] == view.consumer
                        })
                    })
                    .ok_or(Error::Binding)?;
            if self.regions.iter().any(|region| {
                region
                    .graph
                    .build
                    .outputs
                    .iter()
                    .any(|port| region.build.tensors[port.tensor.0 as usize] == view.consumer)
            }) {
                return Err(Error::Binding);
            }
            let mut next = aggregate.start;
            for &source in view.producers {
                let source_tensor = parent.tensor(source).map_err(|_| Error::Graph)?;
                let span = *ranges.get(source.0 as usize).ok_or(Error::Binding)?;
                if source == view.consumer
                    || source_tensor.dtype != tensor.dtype
                    || source_tensor.flags.contains(TensorFlags::INPUT)
                    || source_tensor.flags.contains(TensorFlags::CONSTANT)
                    || span.buffer != aggregate.buffer
                    || span.start != next
                    || span.end > aggregate.end
                {
                    return Err(Error::Binding);
                }
                let mut producers = self.regions.iter().enumerate().filter(|(_, region)| {
                    region
                        .graph
                        .build
                        .outputs
                        .iter()
                        .any(|port| region.build.tensors[port.tensor.0 as usize] == source)
                });
                let (written_at, _) = producers.next().ok_or(Error::Binding)?;
                if written_at >= first_read || producers.next().is_some() {
                    return Err(Error::Binding);
                }
                next = span.end;
            }
            if next != aggregate.end {
                return Err(Error::Binding);
            }
            // A read-only aggregate is the one permitted overlap. Its sources
            // stay in the table, so the ordinary disjoint check rejects any
            // third tensor sharing even part of the covered range.
            ranges[consumer] = BufferRange::default();
        }
        check_disjoint_buffer_ranges(ranges)?;
        Ok(self.offset_launches(queue, programs, buffers, timeout_ns))
    }

    fn launches<'b>(
        &'b self,
        queue: u64,
        programs: &'b [u64],
        buffers: &'b [u64],
        timeout_ns: u64,
    ) -> impl Iterator<Item = kore_svc_accel::client::ArtifactExecution> + 'b {
        self.regions
            .iter()
            .zip(programs)
            .map(move |(region, &program)| {
                kore_svc_accel::client::ArtifactExecution {
                    queue,
                    program,
                    timeout_ns,
                    // Both metadata and residency remain immutably borrowed by
                    // this iterator after the complete validation pass above.
                    bindings: region
                        .bindings(buffers)
                        .expect("validated immutable sequence residency"),
                }
            })
    }
}
