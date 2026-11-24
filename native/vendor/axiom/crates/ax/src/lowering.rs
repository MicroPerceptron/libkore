//! Canonical bridge from serialized `.ax` nodes to `axnn` kernel requests.
//!
//! The file format records semantic intent, never implementation variants.
//! Adapters use this allocation-free bridge to ask an `axnn` catalog for a
//! concrete kernel plan. Keeping the opcode mapping here prevents CPU, TOSA,
//! and future kore adapters from drifting into separate semantic tables.

use axnn::lowering::{IntegerBinaryOp, KernelRequest, LogicalOp, RasterOp};
use axnn::{BinaryOp, CompareOp, ReduceOp, Shape, TensorDesc, TernaryOp, UnaryOp};

use crate::{DYNAMIC_DIM, Graph, NodeFlags, NodeView, OpCode, TensorId};

/// Translate one concrete `.ax` node into a typed kernel request. Attributes
/// are admitted only for explicitly supported, checked operation schemas.
///
/// Returns `None` when the operation is not yet represented by the `axnn`
/// catalog vocabulary, a dimension is still dynamic, arity is noncanonical,
/// broadcasting is invalid, or the authored output descriptor disagrees with
/// the semantic result. This is capability discovery, so malformed or
/// unsupported nodes are rejected rather than approximated.
pub fn node_kernel_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if node.op == OpCode::VOXEL_BRICK_TRACE {
        return voxel_brick_trace_request(graph, node);
    }
    if node.op == OpCode::VOLUME_TRILINEAR_RGB {
        return volume_trilinear_rgb_request(graph, node);
    }
    if node.op == OpCode::BUFFER_LOAD_U8 {
        return buffer_load_u8_request(graph, node);
    }
    if let Some(op) = crate::raster::raster_op(node.op) {
        return raster_request(graph, node, op);
    }
    if node.op == OpCode::KV_APPEND {
        return kv_append_request(graph, node);
    }
    if node.op == OpCode::ATTENTION_DECODE {
        return attention_decode_request(graph, node);
    }
    if matches!(
        node.op,
        OpCode::CAUSAL_CONV1D | OpCode::SSM_SCAN | OpCode::GATED_DELTA_RULE
    ) {
        return recurrent_request(graph, node);
    }
    if node.op == OpCode::MOE_EXPERTS {
        return moe_request(graph, node);
    }
    if matches!(
        node.op,
        OpCode::MOE_ROUTE | OpCode::MOE_GATE_UP_NVFP4 | OpCode::MOE_DOWN_NVFP4
    ) {
        return routed_request(graph, node);
    }
    if node.op == OpCode::MATMUL_NVFP4_TRANSPOSED {
        let [activations, packed, block_scales] = exact_inputs::<3>(node)?;
        let (scale, activation) =
            crate::tensor_ops::decode_matmul_epilogue_attributes(node.attributes())?;
        return KernelRequest::matmul_nvfp4_transposed(
            &tensor_desc(graph, activations)?,
            &tensor_desc(graph, packed)?,
            &tensor_desc(graph, block_scales)?,
            &tensor_desc(graph, only(node.outputs())?)?,
            scale,
            activation,
        );
    }
    if node.op == OpCode::MATMUL_I8_PER_CHANNEL {
        let [activations, weights, row_scales] = exact_inputs::<3>(node)?;
        if !node.attributes().is_empty() {
            return None;
        }
        return KernelRequest::matmul_i8_per_channel(
            &tensor_desc(graph, activations)?,
            &tensor_desc(graph, weights)?,
            &tensor_desc(graph, row_scales)?,
            &tensor_desc(graph, only(node.outputs())?)?,
        );
    }
    if node.op == OpCode::ROPE {
        let [qk, cosine, sine] = exact_inputs::<3>(node)?;
        let (mode, rotary) = crate::tensor_ops::decode_rope_attributes(node.attributes())?;
        let (qk, cosine, sine) = (
            tensor_desc(graph, qk)?,
            tensor_desc(graph, cosine)?,
            tensor_desc(graph, sine)?,
        );
        return match rotary {
            None => KernelRequest::rope(&qk, &cosine, &sine, mode),
            Some(rotary) => KernelRequest::rope_partial(&qk, &cosine, &sine, mode, rotary),
        };
    }
    // An old pure mapping must not discard an authored resource dependency.
    if node.flags.contains(NodeFlags::READS_RESOURCE)
        || node.flags.contains(NodeFlags::WRITES_RESOURCE)
    {
        return None;
    }
    if matches!(
        node.op,
        OpCode::SOFTMAX | OpCode::RMS_NORM | OpCode::LAYER_NORM
    ) {
        return neural_request(graph, node);
    }
    if node.op == OpCode::VIEW || node.op == OpCode::TOP_K {
        return selection_request(graph, node);
    }
    // A 1.8 matmul carrying a fused epilogue record (#107). Without the
    // record it falls through to the plain two-input form below.
    if node.op == OpCode::MATMUL_TRANSPOSED && !node.attributes().is_empty() {
        let (scale, activation) =
            crate::tensor_ops::decode_matmul_epilogue_attributes(node.attributes())?;
        let [left, right] = exact_inputs::<2>(node)?;
        return KernelRequest::matmul_transposed_activated(
            &tensor_desc(graph, left)?,
            &tensor_desc(graph, right)?,
            &tensor_desc(graph, only(node.outputs())?)?,
            scale,
            activation,
        );
    }
    if !node.attributes().is_empty() || node.output_count() != 1 {
        return None;
    }
    let output = tensor_desc(graph, only(node.outputs())?)?;

    let request = if matches!(node.op, OpCode::IDENTITY | OpCode::COPY) {
        let input = tensor_desc(graph, only(node.inputs())?)?;
        KernelRequest::identity(&input)
    } else if node.op == OpCode::CAST {
        let input = tensor_desc(graph, only(node.inputs())?)?;
        KernelRequest::cast(&input, output.dtype)?
    } else if let Some(op) = integer_binary_op(node.op) {
        let [left, right] = exact_inputs::<2>(node)?;
        KernelRequest::integer_binary(op, &tensor_desc(graph, left)?, &tensor_desc(graph, right)?)?
    } else if node.op == OpCode::LOGICAL_NOT {
        let input = tensor_desc(graph, only(node.inputs())?)?;
        KernelRequest::logical_not(&input)?
    } else if let Some(op) = logical_binary_op(node.op) {
        let [left, right] = exact_inputs::<2>(node)?;
        KernelRequest::logical_binary(op, &tensor_desc(graph, left)?, &tensor_desc(graph, right)?)?
    } else if let Some(op) = unary_op(node.op) {
        let input = tensor_desc(graph, only(node.inputs())?)?;
        KernelRequest::unary(op, &input)
    } else if node.op == OpCode::INTERLEAVE3_F32 {
        let [x, y, z] = exact_inputs::<3>(node)?;
        KernelRequest::interleave3_f32(
            &tensor_desc(graph, x)?,
            &tensor_desc(graph, y)?,
            &tensor_desc(graph, z)?,
            &output,
        )?
    } else if node.op == OpCode::GATHER {
        let [table, indices] = exact_inputs::<2>(node)?;
        KernelRequest::gather(&tensor_desc(graph, table)?, &tensor_desc(graph, indices)?)?
    } else if node.op == OpCode::MATMUL_TRANSPOSED {
        let [left, right] = exact_inputs::<2>(node)?;
        KernelRequest::matmul_transposed(
            &tensor_desc(graph, left)?,
            &tensor_desc(graph, right)?,
            &output,
        )?
    } else if node.op == OpCode::MATMUL {
        let [left, right] = exact_inputs::<2>(node)?;
        KernelRequest::matmul(
            &tensor_desc(graph, left)?,
            &tensor_desc(graph, right)?,
            &output,
        )?
    } else if let Some(op) = reduce_op(node.op) {
        let input = tensor_desc(graph, only(node.inputs())?)?;
        KernelRequest::reduce(op, &input)
    } else if let Some(op) = binary_op(node.op) {
        let [left, right] = exact_inputs::<2>(node)?;
        KernelRequest::binary(op, &tensor_desc(graph, left)?, &tensor_desc(graph, right)?)?
    } else if let Some(op) = ternary_op(node.op) {
        let [first, second, third] = exact_inputs::<3>(node)?;
        KernelRequest::ternary(
            op,
            &tensor_desc(graph, first)?,
            &tensor_desc(graph, second)?,
            &tensor_desc(graph, third)?,
        )?
    } else if let Some(op) = compare_op(node.op) {
        let [left, right] = exact_inputs::<2>(node)?;
        KernelRequest::compare(op, &tensor_desc(graph, left)?, &tensor_desc(graph, right)?)?
    } else if node.op == OpCode::SELECT {
        let [mask, on_true, on_false] = exact_inputs::<3>(node)?;
        KernelRequest::select(
            &tensor_desc(graph, mask)?,
            &tensor_desc(graph, on_true)?,
            &tensor_desc(graph, on_false)?,
        )?
    } else {
        return None;
    };

    (request.output_dtype == output.dtype && request.shape == output.shape).then_some(request)
}

fn reduce_op(op: OpCode) -> Option<ReduceOp> {
    Some(if op == OpCode::REDUCE_SUM {
        ReduceOp::Sum
    } else if op == OpCode::REDUCE_MAX {
        ReduceOp::Max
    } else {
        return None;
    })
}

fn integer_binary_op(op: OpCode) -> Option<IntegerBinaryOp> {
    Some(match op {
        OpCode::BITWISE_AND => IntegerBinaryOp::And,
        OpCode::BITWISE_OR => IntegerBinaryOp::Or,
        OpCode::BITWISE_XOR => IntegerBinaryOp::Xor,
        OpCode::SHIFT_LEFT => IntegerBinaryOp::ShiftLeft,
        OpCode::SHIFT_RIGHT => IntegerBinaryOp::ShiftRight,
        _ => return None,
    })
}

fn buffer_load_u8_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if !node.flags.contains(NodeFlags::READS_RESOURCE) || node.flags.contains(NodeFlags::MUTATING) {
        return None;
    }
    let input = only(node.inputs())?;
    let output = only(node.outputs())?;
    if input == output {
        return None;
    }
    let output = tensor_desc(graph, output)?;
    let request = KernelRequest::buffer_load_u8(
        &tensor_desc(graph, input)?,
        crate::buffer::decode_u8_buffer_attributes(node.attributes())?,
        output.shape,
    )?;
    (request.result_tensor(0) == Some(&output)).then_some(request)
}

fn raster_request(graph: &Graph<'_>, node: NodeView<'_>, op: RasterOp) -> Option<KernelRequest> {
    if node.flags != crate::raster::RASTER_NODE_FLAGS {
        return None;
    }
    let [params, dirty] = exact_inputs::<2>(node)?;
    let output = only(node.outputs())?;
    if output == params || output == dirty {
        return None;
    }
    let output = tensor_desc(graph, output)?;
    let request = KernelRequest::raster(
        op,
        &tensor_desc(graph, params)?,
        &tensor_desc(graph, dirty)?,
        crate::raster::decode_raster_attributes(node.attributes())?,
    )?;
    (request.result_tensor(0) == Some(&output)).then_some(request)
}

fn kv_append_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if node.flags != crate::kv::KV_APPEND_NODE_FLAGS {
        return None;
    }
    let inputs @ [position, keys, values] = exact_inputs::<3>(node)?;
    let token = only(node.outputs())?;
    if inputs.contains(&token) {
        return None;
    }
    let request = KernelRequest::kv_append(
        &tensor_desc(graph, position)?,
        &tensor_desc(graph, keys)?,
        &tensor_desc(graph, values)?,
        crate::kv::decode_kv_append_attributes(node.attributes())?,
    )?;
    (request.result_tensor(0) == Some(&tensor_desc(graph, token)?)).then_some(request)
}

/// A cache read is admitted only directly after its own write: `token` must
/// be produced by a `KV_APPEND` of the same cache, from the same `position`
/// tensor, of as many token rows as there are query rows. Slots before
/// `position` were written by earlier invocations; this one wrote exactly the
/// slots its queries newly reach.
fn attention_decode_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if node.flags != crate::kv::ATTENTION_DECODE_NODE_FLAGS {
        return None;
    }
    let inputs @ [token, position, queries] = exact_inputs::<3>(node)?;
    let output = only(node.outputs())?;
    if inputs.contains(&output) {
        return None;
    }
    let params = crate::kv::decode_attention_decode_attributes(node.attributes())?;
    let append = (0..graph.node_count())
        .filter_map(|index| graph.node(index).ok())
        .find(|candidate| candidate.outputs().any(|id| id == token))?;
    let [append_position, append_keys, _] = exact_inputs::<3>(append)?;
    if append.op != OpCode::KV_APPEND
        || crate::kv::decode_kv_append_attributes(append.attributes())? != params.cache
        || append_position != position
        || tensor_desc(graph, append_keys)?.shape.dim(0)
            != tensor_desc(graph, queries)?.shape.dim(0)
    {
        return None;
    }
    let request = KernelRequest::attention_decode(
        &tensor_desc(graph, token)?,
        &tensor_desc(graph, position)?,
        &tensor_desc(graph, queries)?,
        params,
    )?;
    (request.result_tensor(0) == Some(&tensor_desc(graph, output)?)).then_some(request)
}

/// Expert dispatch: a read of its store, three operands, one fresh result.
fn moe_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if node.flags != NodeFlags::READS_RESOURCE {
        return None;
    }
    let inputs @ [hidden, scores, bias] = exact_inputs::<3>(node)?;
    let output = only(node.outputs())?;
    if inputs.contains(&output) {
        return None;
    }
    let request = KernelRequest::moe_experts(
        &tensor_desc(graph, hidden)?,
        &tensor_desc(graph, scores)?,
        &tensor_desc(graph, bias)?,
        crate::moe::decode_moe_attributes(node.attributes())?,
    )?;
    (request.result_tensor(0) == Some(&tensor_desc(graph, output)?)).then_some(request)
}

/// Routing or a routed NVFP4 projection over an expert table of graph
/// weights: pure, one fresh result.
fn routed_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if node.flags != NodeFlags::default() {
        return None;
    }
    let output = only(node.outputs())?;
    let request = if node.op == OpCode::MOE_ROUTE {
        let params = crate::routed::decode_route_attributes(node.attributes())?;
        let logits = only(node.inputs())?;
        if logits == output {
            return None;
        }
        KernelRequest::moe_route(&tensor_desc(graph, logits)?, params)?
    } else {
        let params = crate::routed::decode_routed_attributes(node.attributes())?;
        let inputs @ [x, experts, routes] = exact_inputs::<3>(node)?;
        if inputs.contains(&output) {
            return None;
        }
        let (x, experts, routes) = (
            tensor_desc(graph, x)?,
            tensor_desc(graph, experts)?,
            tensor_desc(graph, routes)?,
        );
        if node.op == OpCode::MOE_GATE_UP_NVFP4 {
            KernelRequest::moe_gate_up_nvfp4(&x, &experts, &routes, params)?
        } else {
            KernelRequest::moe_down_nvfp4(&x, &experts, &routes, params)?
        }
    };
    (request.result_tensor(0) == Some(&tensor_desc(graph, output)?)).then_some(request)
}

/// A recurrent step: exact flags, three operands, one fresh result.
fn recurrent_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if node.flags != crate::recurrent::RECURRENT_NODE_FLAGS {
        return None;
    }
    let inputs @ [first, second, third] = exact_inputs::<3>(node)?;
    let output = only(node.outputs())?;
    if inputs.contains(&output) {
        return None;
    }
    let (first, second, third) = (
        tensor_desc(graph, first)?,
        tensor_desc(graph, second)?,
        tensor_desc(graph, third)?,
    );
    let request = if node.op == OpCode::CAUSAL_CONV1D {
        KernelRequest::causal_conv1d(
            &first,
            &second,
            &third,
            crate::recurrent::decode_causal_conv1d_attributes(node.attributes())?,
        )?
    } else if node.op == OpCode::SSM_SCAN {
        KernelRequest::ssm_scan(
            &first,
            &second,
            &third,
            crate::recurrent::decode_ssm_scan_attributes(node.attributes())?,
        )?
    } else {
        KernelRequest::gated_delta_rule(
            &first,
            &second,
            &third,
            crate::recurrent::decode_gated_delta_rule_attributes(node.attributes())?,
        )?
    };
    (request.result_tensor(0) == Some(&tensor_desc(graph, output)?)).then_some(request)
}

fn voxel_brick_trace_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if !node.flags.contains(NodeFlags::READS_RESOURCE)
        || node.flags.contains(NodeFlags::MUTATING)
        || node.output_count() != 2
    {
        return None;
    }
    let params = crate::voxel::decode_brick_trace_attributes(node.attributes())?;
    let inputs @ [origins, directions, limit] = exact_inputs::<3>(node)?;
    let mut outputs = node.outputs();
    let t = outputs.next()?;
    let cell_id = outputs.next()?;
    // Queries produce two independent values. They neither replace an input
    // nor alias each other, even when a one-ray shape happens to fit.
    if t == cell_id || inputs.contains(&t) || inputs.contains(&cell_id) {
        return None;
    }
    let request = KernelRequest::voxel_brick_trace(
        &tensor_desc(graph, origins)?,
        &tensor_desc(graph, directions)?,
        &tensor_desc(graph, limit)?,
        params,
    )?;
    for (index, id) in [t, cell_id].into_iter().enumerate() {
        if request.result_tensor(index) != Some(&tensor_desc(graph, id)?) {
            return None;
        }
    }
    Some(request)
}

fn volume_trilinear_rgb_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if !node.flags.contains(NodeFlags::READS_RESOURCE)
        || node.flags.contains(NodeFlags::MUTATING)
        || node.output_count() != 3
    {
        return None;
    }
    let params = crate::sampling::decode_volume_rgb_attributes(node.attributes())?;
    let inputs @ [x, y, z] = exact_inputs::<3>(node)?;
    let mut outputs = node.outputs();
    let outputs = [outputs.next()?, outputs.next()?, outputs.next()?];
    // RGB planes are distinct values, including when scalar coordinates or
    // empty fields make the input and output byte lengths happen to match.
    for (index, output) in outputs.iter().enumerate() {
        if inputs.contains(output) || outputs[..index].contains(output) {
            return None;
        }
    }
    let field = tensor_desc(graph, outputs[0])?.shape;
    let request = KernelRequest::volume_trilinear_rgb(
        &tensor_desc(graph, x)?,
        &tensor_desc(graph, y)?,
        &tensor_desc(graph, z)?,
        params,
        field,
    )?;
    for (index, id) in outputs.into_iter().enumerate() {
        if request.result_tensor(index) != Some(&tensor_desc(graph, id)?) {
            return None;
        }
    }
    Some(request)
}

fn tensor_desc(graph: &Graph<'_>, id: TensorId) -> Option<TensorDesc> {
    let tensor = graph.tensor(id).ok()?;
    let mut dimensions = [0_usize; axnn::MAX_RANK];
    if tensor.rank() > dimensions.len() {
        return None;
    }
    for (slot, dimension) in dimensions.iter_mut().zip(tensor.dimensions()) {
        if dimension == DYNAMIC_DIM {
            return None;
        }
        *slot = usize::try_from(dimension).ok()?;
    }
    let shape = Shape::try_from(&dimensions[..tensor.rank()]).ok()?;
    TensorDesc::try_new(shape, tensor.dtype)
}

fn only(mut ids: impl Iterator<Item = TensorId>) -> Option<TensorId> {
    let value = ids.next()?;
    ids.next().is_none().then_some(value)
}

fn exact_inputs<const N: usize>(node: NodeView<'_>) -> Option<[TensorId; N]> {
    let mut inputs = node.inputs();
    let mut values = [TensorId(0); N];
    for value in &mut values {
        *value = inputs.next()?;
    }
    inputs.next().is_none().then_some(values)
}

fn unary_op(op: OpCode) -> Option<UnaryOp> {
    Some(if op == OpCode::ABS {
        UnaryOp::Abs
    } else if op == OpCode::ACOS {
        UnaryOp::Acos
    } else if op == OpCode::ASIN {
        UnaryOp::Asin
    } else if op == OpCode::ATAN {
        UnaryOp::Atan
    } else if op == OpCode::CEIL {
        UnaryOp::Ceil
    } else if op == OpCode::COS {
        UnaryOp::Cos
    } else if op == OpCode::EXP {
        UnaryOp::Exp
    } else if op == OpCode::EXP2 {
        UnaryOp::Exp2
    } else if op == OpCode::FLOOR {
        UnaryOp::Floor
    } else if op == OpCode::LOG {
        UnaryOp::Log
    } else if op == OpCode::LOG2 {
        UnaryOp::Log2
    } else if op == OpCode::NEGATE {
        UnaryOp::Neg
    } else if op == OpCode::RECIPROCAL {
        UnaryOp::Reciprocal
    } else if op == OpCode::RELU {
        UnaryOp::Relu
    } else if op == OpCode::ROUND {
        UnaryOp::Round
    } else if op == OpCode::RSQRT {
        UnaryOp::InverseSqrt
    } else if op == OpCode::SIGN {
        UnaryOp::Sign
    } else if op == OpCode::SIGMOID {
        UnaryOp::Sigmoid
    } else if op == OpCode::SILU {
        UnaryOp::Silu
    } else if op == OpCode::SIN {
        UnaryOp::Sin
    } else if op == OpCode::SQRT {
        UnaryOp::Sqrt
    } else if op == OpCode::TAN {
        UnaryOp::Tan
    } else if op == OpCode::TANH {
        UnaryOp::Tanh
    } else if op == OpCode::TRUNC {
        UnaryOp::Trunc
    } else if op == OpCode::SOFTPLUS {
        UnaryOp::Softplus
    } else {
        return None;
    })
}

fn binary_op(op: OpCode) -> Option<BinaryOp> {
    Some(if op == OpCode::ADD {
        BinaryOp::Add
    } else if op == OpCode::ATAN2 {
        BinaryOp::Atan2
    } else if op == OpCode::DIV {
        BinaryOp::Div
    } else if op == OpCode::MAXIMUM {
        BinaryOp::Max
    } else if op == OpCode::MINIMUM {
        BinaryOp::Min
    } else if op == OpCode::MUL {
        BinaryOp::Mul
    } else if op == OpCode::POW {
        BinaryOp::Pow
    } else if op == OpCode::REMAINDER {
        BinaryOp::Rem
    } else if op == OpCode::SUB {
        BinaryOp::Sub
    } else {
        return None;
    })
}

fn ternary_op(op: OpCode) -> Option<TernaryOp> {
    if op == OpCode::CLAMP {
        Some(TernaryOp::Clamp)
    } else if op == OpCode::FMA {
        Some(TernaryOp::Fma)
    } else {
        None
    }
}

fn compare_op(op: OpCode) -> Option<CompareOp> {
    if op == OpCode::EQUAL {
        Some(CompareOp::Equal)
    } else if op == OpCode::GREATER {
        Some(CompareOp::Greater)
    } else if op == OpCode::GREATER_EQUAL {
        Some(CompareOp::GreaterEqual)
    } else {
        None
    }
}

fn logical_binary_op(op: OpCode) -> Option<LogicalOp> {
    if op == OpCode::LOGICAL_AND {
        Some(LogicalOp::And)
    } else if op == OpCode::LOGICAL_OR {
        Some(LogicalOp::Or)
    } else if op == OpCode::LOGICAL_XOR {
        Some(LogicalOp::Xor)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use alloc::alloc::Global;
    use axnn::lowering::SemanticOp;

    use super::*;
    use crate::{DType, GraphBuilder, NodeFlags, TensorFlags};

    #[test]
    fn rope_node_admits_to_the_neox_request() {
        let tensors = [
            crate::TensorSpec { dtype: DType::F32, flags: TensorFlags::INPUT,
                dimensions: &[1, 2, 128], constant: &[] },
            crate::TensorSpec { dtype: DType::F32, flags: TensorFlags::INPUT,
                dimensions: &[1, 64], constant: &[] },
            crate::TensorSpec { dtype: DType::F32, flags: TensorFlags::INPUT,
                dimensions: &[1, 64], constant: &[] },
            crate::TensorSpec { dtype: DType::F32, flags: TensorFlags::default(),
                dimensions: &[1, 2, 128], constant: &[] },
        ];
        let attrs = crate::tensor_ops::encode_rope_attributes(axnn::RopeMode::NeoX);
        let node = crate::NodeSpec {
            op: OpCode::ROPE, flags: NodeFlags::default(),
            inputs: &[TensorId(0), TensorId(1), TensorId(2)],
            outputs: &[TensorId(3)], attributes: &attrs,
        };
        let bytes = crate::GraphBuf::encode_in(&tensors, &[node], Global).unwrap();
        assert_eq!(u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()), 21);
        let graph = bytes.parse().unwrap();
        let request = node_kernel_request(&graph, graph.node(0).unwrap()).unwrap();
        assert_eq!(request.semantic, SemanticOp::Rope(axnn::RopeMode::NeoX));

        // Unknown tags and wrong lengths are refused by the record decoder.
        assert!(crate::tensor_ops::decode_rope_attributes(&[2]).is_none());
        assert!(crate::tensor_ops::decode_rope_attributes(&[]).is_none());
        assert!(crate::tensor_ops::decode_rope_attributes(&[0, 0]).is_none());
        assert_eq!(
            crate::tensor_ops::decode_rope_attributes(&attrs),
            Some((axnn::RopeMode::NeoX, None))
        );
    }

    /// A 1.22 partial record rotates only the named leading span, needs
    /// phase rows of half that span, and lifts the graph to minor 22.
    #[test]
    fn a_partial_rope_names_its_span_and_needs_minor_twenty_two() {
        let tensor = |flags, dimensions| crate::TensorSpec {
            dtype: DType::F32,
            flags,
            dimensions,
            constant: &[],
        };
        let encode = |phase: &'static [u32], attrs: &[u8]| {
            let tensors = [
                tensor(TensorFlags::INPUT, &[1, 4, 256]),
                tensor(TensorFlags::INPUT, phase),
                tensor(TensorFlags::INPUT, phase),
                tensor(TensorFlags::default(), &[1, 4, 256]),
            ];
            let node = crate::NodeSpec {
                op: OpCode::ROPE,
                flags: NodeFlags::default(),
                inputs: &[TensorId(0), TensorId(1), TensorId(2)],
                outputs: &[TensorId(3)],
                attributes: attrs,
            };
            crate::GraphBuf::encode_in(&tensors, &[node], Global).unwrap()
        };
        let attrs =
            crate::tensor_ops::encode_rope_partial_attributes(axnn::RopeMode::NeoX, 64).unwrap();
        let bytes = encode(&[1, 32], &attrs);
        assert_eq!(
            u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()),
            22
        );
        let graph = bytes.parse().unwrap();
        let request = node_kernel_request(&graph, graph.node(0).unwrap()).unwrap();
        assert_eq!(
            request.params(),
            axnn::lowering::KernelParams::Rope {
                mode: axnn::RopeMode::NeoX,
                rotary_dims: 64
            }
        );
        // Phase rows for the whole head do not fit the named span.
        let graph = encode(&[1, 128], &attrs);
        let graph = graph.parse().unwrap();
        assert!(node_kernel_request(&graph, graph.node(0).unwrap()).is_none());
        // Odd and empty spans are not records.
        assert!(
            crate::tensor_ops::encode_rope_partial_attributes(axnn::RopeMode::NeoX, 63).is_none()
        );
        assert!(crate::tensor_ops::decode_rope_attributes(&[1, 0, 0, 0, 0]).is_none());
        assert!(crate::tensor_ops::decode_rope_attributes(&[1, 3, 0, 0, 0]).is_none());
        assert!(crate::tensor_ops::decode_rope_attributes(&attrs[..4]).is_none());
    }

    #[test]
    fn i8_per_channel_projection_has_checked_geometry_and_version() {
        let tensors = [
            crate::TensorSpec { dtype: DType::F32, flags: TensorFlags::INPUT,
                dimensions: &[8, 128], constant: &[] },
            crate::TensorSpec { dtype: DType::I8, flags: TensorFlags::INPUT,
                dimensions: &[96, 128], constant: &[] },
            crate::TensorSpec { dtype: DType::F32, flags: TensorFlags::INPUT,
                dimensions: &[1, 96], constant: &[] },
            crate::TensorSpec { dtype: DType::F32, flags: TensorFlags::OUTPUT,
                dimensions: &[8, 96], constant: &[] },
        ];
        let node = crate::NodeSpec {
            op: OpCode::MATMUL_I8_PER_CHANNEL, flags: NodeFlags::default(),
            inputs: &[TensorId(0), TensorId(1), TensorId(2)],
            outputs: &[TensorId(3)], attributes: &[],
        };
        let bytes = crate::GraphBuf::encode_in(&tensors, &[node], Global).unwrap();
        assert_eq!(u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()), 10);
        let graph = bytes.parse().unwrap();
        let request = node_kernel_request(&graph, graph.node(0).unwrap()).unwrap();
        assert_eq!(request.semantic, SemanticOp::MatMulI8PerChannel);
        assert_eq!(request.shape.dims(), &[8, 96]);
        for (tensor, bad_shape) in [(1, &[96, 127][..]), (2, &[96][..]), (3, &[8, 95][..])] {
            let mut bad = tensors;
            bad[tensor].dimensions = bad_shape;
            let bytes = crate::GraphBuf::encode_in(&bad, &[node], Global).unwrap();
            let graph = bytes.parse().unwrap();
            assert!(node_kernel_request(&graph, graph.node(0).unwrap()).is_none());
        }
        let mut invalid = tensors;
        invalid[1].dtype = DType::U8;
        let bytes = crate::GraphBuf::encode_in(&invalid, &[node], Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert!(node_kernel_request(&graph, graph.node(0).unwrap()).is_none());
    }

    fn request_for(op: OpCode, semantic_dtype: crate::DType, arity: usize) -> KernelRequest {
        let mut builder = GraphBuilder::new_in(Global);
        let input = builder
            .tensor(semantic_dtype, TensorFlags::INPUT, &[4], &[])
            .unwrap();
        let output = builder
            .tensor(semantic_dtype, TensorFlags::OUTPUT, &[4], &[])
            .unwrap();
        let inputs = [input; 3];
        builder
            .node(op, NodeFlags::default(), &inputs[..arity], &[output], &[])
            .unwrap();
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        node_kernel_request(&graph, graph.node(0).unwrap()).unwrap()
    }

    #[test]
    fn typed_bridge_validates_broadcast_result_and_semantics() {
        let mut builder = GraphBuilder::new_in(Global);
        let y = builder
            .tensor(crate::DType::F32, TensorFlags::INPUT, &[3, 1], &[])
            .unwrap();
        let x = builder
            .tensor(crate::DType::F32, TensorFlags::INPUT, &[1, 4], &[])
            .unwrap();
        let output = builder
            .tensor(crate::DType::F32, TensorFlags::OUTPUT, &[3, 4], &[])
            .unwrap();
        builder
            .node(OpCode::ATAN2, NodeFlags::default(), &[y, x], &[output], &[])
            .unwrap();
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        let request = node_kernel_request(&graph, graph.node(0).unwrap()).unwrap();
        assert_eq!(request.semantic, SemanticOp::Binary(BinaryOp::Atan2));
        assert_eq!(request.shape.dims(), &[3, 4]);
        assert_eq!(request.input_shapes[0].unwrap().dims(), &[3, 1]);
        assert_eq!(request.input_shapes[1].unwrap().dims(), &[1, 4]);
    }

    #[test]
    fn bridge_rejects_an_authored_output_shape_that_disagrees() {
        let mut builder = GraphBuilder::new_in(Global);
        let input = builder
            .tensor(crate::DType::F32, TensorFlags::INPUT, &[8], &[])
            .unwrap();
        let output = builder
            .tensor(crate::DType::F32, TensorFlags::OUTPUT, &[7], &[])
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
        assert!(node_kernel_request(&graph, graph.node(0).unwrap()).is_none());
    }

    #[test]
    fn bridge_preserves_typed_matmul_geometry() {
        let mut builder = GraphBuilder::new_in(Global);
        let left = builder
            .tensor(crate::DType::F32, TensorFlags::INPUT, &[2, 3], &[])
            .unwrap();
        let right = builder
            .tensor(crate::DType::F32, TensorFlags::INPUT, &[3, 4], &[])
            .unwrap();
        let output = builder
            .tensor(crate::DType::F32, TensorFlags::OUTPUT, &[2, 4], &[])
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
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();

        let request = node_kernel_request(&graph, graph.node(0).unwrap()).unwrap();

        assert_eq!(request.semantic, SemanticOp::MatMul);
        assert_eq!(request.input(0).unwrap().shape.dims(), &[2, 3]);
        assert_eq!(request.input(1).unwrap().shape.dims(), &[3, 4]);
        assert_eq!(request.shape.dims(), &[2, 4]);
    }

    #[test]
    fn full_reductions_reach_the_shared_scalar_contract() {
        for (op, reduction) in [
            (OpCode::REDUCE_SUM, ReduceOp::Sum),
            (OpCode::REDUCE_MAX, ReduceOp::Max),
        ] {
            let mut builder = GraphBuilder::new_in(Global);
            let input = builder
                .tensor(crate::DType::F32, TensorFlags::INPUT, &[2, 4], &[])
                .unwrap();
            let output = builder
                .tensor(crate::DType::F32, TensorFlags::OUTPUT, &[], &[])
                .unwrap();
            builder
                .node(op, NodeFlags::default(), &[input], &[output], &[])
                .unwrap();
            let bytes = builder.finish().unwrap();
            let graph = bytes.parse().unwrap();
            let request = node_kernel_request(&graph, graph.node(0).unwrap()).unwrap();

            assert_eq!(request.semantic, SemanticOp::Reduce(reduction));
            assert_eq!(request.input(0).unwrap().shape.dims(), &[2, 4]);
            assert_eq!(request.shape, Shape::SCALAR);
            assert_eq!(
                request.result(),
                axnn::lowering::KernelResult::Scalar(axnn::lowering::KernelScalar::F32)
            );
        }
    }

    #[test]
    fn bridge_rejects_concrete_dimensions_whose_product_overflows() {
        let mut builder = GraphBuilder::new_in(Global);
        let huge = DYNAMIC_DIM - 1;
        let dimensions = [huge, huge, 2];
        let input = builder
            .tensor(crate::DType::F32, TensorFlags::INPUT, &dimensions, &[])
            .unwrap();
        let output = builder
            .tensor(crate::DType::F32, TensorFlags::OUTPUT, &dimensions, &[])
            .unwrap();
        builder
            .node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[input],
                &[output],
                &[],
            )
            .unwrap();
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        assert!(node_kernel_request(&graph, graph.node(0).unwrap()).is_none());
    }

    #[test]
    fn every_appended_numeric_opcode_reaches_the_shared_semantic_catalog() {
        let cases = [
            (OpCode::SQRT, SemanticOp::Unary(UnaryOp::Sqrt), 1),
            (OpCode::FLOOR, SemanticOp::Unary(UnaryOp::Floor), 1),
            (OpCode::CEIL, SemanticOp::Unary(UnaryOp::Ceil), 1),
            (OpCode::ROUND, SemanticOp::Unary(UnaryOp::Round), 1),
            (OpCode::TRUNC, SemanticOp::Unary(UnaryOp::Trunc), 1),
            (OpCode::REMAINDER, SemanticOp::Binary(BinaryOp::Rem), 2),
            (OpCode::ATAN, SemanticOp::Unary(UnaryOp::Atan), 1),
            (OpCode::ATAN2, SemanticOp::Binary(BinaryOp::Atan2), 2),
            (OpCode::ACOS, SemanticOp::Unary(UnaryOp::Acos), 1),
            (OpCode::ASIN, SemanticOp::Unary(UnaryOp::Asin), 1),
            (OpCode::TAN, SemanticOp::Unary(UnaryOp::Tan), 1),
            (OpCode::EXP2, SemanticOp::Unary(UnaryOp::Exp2), 1),
            (OpCode::LOG2, SemanticOp::Unary(UnaryOp::Log2), 1),
            (OpCode::SIGN, SemanticOp::Unary(UnaryOp::Sign), 1),
            (OpCode::CLAMP, SemanticOp::Ternary(TernaryOp::Clamp), 3),
            (OpCode::FMA, SemanticOp::Ternary(TernaryOp::Fma), 3),
        ];

        for (op, expected, arity) in cases {
            assert_eq!(request_for(op, crate::DType::F32, arity).semantic, expected);
        }
    }

    #[test]
    fn identity_copy_and_boolean_algebra_use_typed_catalog_requests() {
        for op in [OpCode::IDENTITY, OpCode::COPY] {
            let request = request_for(op, crate::DType::F32, 1);
            assert_eq!(request.semantic, SemanticOp::Identity);
            assert_eq!(request.output_dtype, crate::DType::F32);
        }
        for (op, expected, arity) in [
            (OpCode::LOGICAL_NOT, LogicalOp::Not, 1),
            (OpCode::LOGICAL_AND, LogicalOp::And, 2),
            (OpCode::LOGICAL_OR, LogicalOp::Or, 2),
            (OpCode::LOGICAL_XOR, LogicalOp::Xor, 2),
        ] {
            let request = request_for(op, crate::DType::U8, arity);
            assert_eq!(request.semantic, SemanticOp::Logical(expected));
            assert_eq!(request.output_dtype, crate::DType::U8);
        }
    }

    #[test]
    fn interleave_builder_lowers_explicit_uniform_and_varying_planes() {
        for shapes in [[[1, 1], [1, 1], [1, 1]], [[2, 1], [1, 3], [2, 3]]] {
            let mut builder = GraphBuilder::new_in(Global);
            let mut components = [TensorId(0); 3];
            for (id, shape) in components.iter_mut().zip(shapes) {
                *id = builder
                    .tensor(crate::DType::F32, TensorFlags::INPUT, &shape, &[])
                    .unwrap();
            }
            builder.interleave3_f32(components, [2, 3]).unwrap();
            let bytes = builder.finish().unwrap();
            let graph = bytes.parse().unwrap();
            let request = node_kernel_request(&graph, graph.node(0).unwrap()).unwrap();
            assert_eq!(request.semantic, SemanticOp::Interleave3F32);
            assert_eq!(request.shape.dims(), &[2, 3, 3]);
            assert_eq!(request.result_count(), 1);
        }
    }

    #[test]
    fn query_bridge_checks_both_results_and_every_input_before_admission() {
        use crate::voxel::{ResourceKey, VoxelBrickTraceParams, encode_brick_trace_attributes};
        use crate::{DType, GraphBuf, NodeSpec, TensorSpec};
        use axnn::lowering::{KernelEffects, KernelParams};

        let params = VoxelBrickTraceParams::new(
            ResourceKey {
                group: 2,
                binding: 5,
            },
            [32, 16, 8],
        )
        .unwrap();
        let attributes = encode_brick_trace_attributes(params).unwrap();
        let tensors = [
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[2, 3, 3],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[2, 3, 3],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::INPUT,
                dimensions: &[1, 1],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::F32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[2, 3],
                constant: &[],
            },
            TensorSpec {
                dtype: DType::I32,
                flags: TensorFlags::OUTPUT,
                dimensions: &[2, 3],
                constant: &[],
            },
        ];
        let request_for = |tensors: &[TensorSpec<'_>], outputs: &[TensorId]| {
            let node = NodeSpec {
                op: OpCode::VOXEL_BRICK_TRACE,
                flags: NodeFlags::READS_RESOURCE,
                inputs: &[TensorId(0), TensorId(1), TensorId(2)],
                outputs,
                attributes: &attributes,
            };
            let bytes = GraphBuf::encode_in(tensors, &[node], Global).unwrap();
            let graph = bytes.parse().unwrap();
            node_kernel_request(&graph, graph.node(0).unwrap())
        };
        let outputs = [TensorId(3), TensorId(4)];
        let request = request_for(&tensors, &outputs).unwrap();
        assert_eq!(request.semantic, SemanticOp::VoxelBrickTrace);
        assert_eq!(request.params(), KernelParams::VoxelBrickTrace(params));
        assert_eq!(request.effects(), KernelEffects::READS_RESOURCE);
        assert_eq!(request.result_count(), 2);
        assert_eq!(request.result_tensor(0).unwrap().dtype, DType::F32);
        assert_eq!(request.result_tensor(1).unwrap().dtype, DType::I32);
        assert_eq!(request.result_tensor(1).unwrap().shape.dims(), &[2, 3]);

        for index in 0..tensors.len() {
            let mut invalid = tensors;
            invalid[index].dtype = if index == 4 { DType::F32 } else { DType::F16 };
            assert!(
                request_for(&invalid, &outputs).is_none(),
                "dtype at tensor {index}"
            );
        }
        for (index, shape) in [
            (0, &[2, 3, DYNAMIC_DIM][..]),
            (1, &[3, 2, 3][..]),
            (2, &[2][..]),
            (3, &[3, 2][..]),
            (4, &[3, 2][..]),
        ] {
            let mut invalid = tensors;
            invalid[index].dimensions = shape;
            assert!(
                request_for(&invalid, &outputs).is_none(),
                "shape at tensor {index}"
            );
        }
        assert!(request_for(&tensors, &[outputs[1], outputs[0]]).is_none());
        assert!(request_for(&tensors, &[outputs[0], outputs[0]]).is_none());

        // Even a one-element limit matching the distance shape cannot become
        // an in-place result of this read-only query.
        let mut single = tensors;
        single[0].dimensions = &[1, 1, 3];
        single[1].dimensions = &[1, 1, 3];
        single[3].dimensions = &[1, 1];
        single[4].dimensions = &[1, 1];
        assert!(request_for(&single, &outputs).is_some());
        assert!(request_for(&single, &[TensorId(2), TensorId(4)]).is_none());
    }

    #[test]
    fn volume_builder_lowers_explicit_uniform_and_broadcast_coordinates() {
        use crate::sampling::{ResourceKey, VolumeRgbParams, VolumeRgbStorage};

        for storage in [
            VolumeRgbStorage::F32,
            VolumeRgbStorage::Uf16E5M11,
            VolumeRgbStorage::Uf8E5M3,
        ] {
            for shapes in [[[1, 1]; 3], [[2, 1], [1, 3], [2, 3]]] {
                let mut builder = GraphBuilder::new_in(Global);
                let mut coordinates = [TensorId(0); 3];
                for (id, shape) in coordinates.iter_mut().zip(shapes) {
                    *id = builder
                        .tensor(crate::DType::F32, TensorFlags::INPUT, &shape, &[])
                        .unwrap();
                }
                let params = VolumeRgbParams::new(
                    ResourceKey {
                        group: 0,
                        binding: 5,
                    },
                    [1, 12, 24],
                    storage,
                )
                .unwrap();
                let sample = builder
                    .volume_trilinear_rgb(coordinates, params, [2, 3])
                    .unwrap();
                let bytes = builder.finish().unwrap();
                let graph = bytes.parse().unwrap();
                let node = graph.node(0).unwrap();
                assert!(node.outputs().eq([sample.r, sample.g, sample.b]));
                let request = node_kernel_request(&graph, node).unwrap();
                assert_eq!(request.semantic, SemanticOp::VolumeTrilinearRgb);
                assert_eq!(request.result_count(), 3);
                for index in 0..3 {
                    let desc = request.result_tensor(index).unwrap();
                    assert_eq!(desc.dtype, crate::DType::F32);
                    assert_eq!(desc.shape.dims(), &[2, 3]);
                }
            }
        }
    }

    #[test]
    fn volume_bridge_checks_all_results_inputs_and_distinct_output_ids() {
        use crate::sampling::{
            ResourceKey, VolumeRgbParams, VolumeRgbStorage, encode_volume_rgb_attributes,
        };
        use crate::{DType, GraphBuf, NodeSpec, TensorSpec};
        use axnn::lowering::{KernelEffects, KernelParams};

        let params = VolumeRgbParams::new(
            ResourceKey {
                group: 2,
                binding: 5,
            },
            [24, 12, 24],
            VolumeRgbStorage::Uf16E5M11,
        )
        .unwrap();
        let attributes = encode_volume_rgb_attributes(params).unwrap();
        let mut tensors = [TensorSpec {
            dtype: DType::F32,
            flags: TensorFlags::INPUT,
            dimensions: &[1, 1],
            constant: &[],
        }; 6];
        for tensor in &mut tensors[3..] {
            tensor.flags = TensorFlags::OUTPUT;
            tensor.dimensions = &[2, 3];
        }
        let request_for = |tensors: &[TensorSpec<'_>], outputs: &[TensorId]| {
            let node = NodeSpec {
                op: OpCode::VOLUME_TRILINEAR_RGB,
                flags: NodeFlags::READS_RESOURCE,
                inputs: &[TensorId(0), TensorId(1), TensorId(2)],
                outputs,
                attributes: &attributes,
            };
            let bytes = GraphBuf::encode_in(tensors, &[node], Global).unwrap();
            let graph = bytes.parse().unwrap();
            node_kernel_request(&graph, graph.node(0).unwrap())
        };
        let outputs = [TensorId(3), TensorId(4), TensorId(5)];
        let request = request_for(&tensors, &outputs).unwrap();
        assert_eq!(request.params(), KernelParams::VolumeTrilinearRgb(params));
        assert_eq!(request.effects(), KernelEffects::READS_RESOURCE);
        assert_eq!(request.result_count(), 3);
        for index in 0..tensors.len() {
            for dtype in [DType::I32, DType::F16, DType::B16] {
                let mut invalid = tensors;
                invalid[index].dtype = dtype;
                assert!(
                    request_for(&invalid, &outputs).is_none(),
                    "dtype at tensor {index}"
                );
            }
            for shape in [
                &[2, 2][..],
                &[1, 2, 3][..],
                &[DYNAMIC_DIM, 3][..],
                &[DYNAMIC_DIM - 1; 2][..],
            ] {
                let mut invalid = tensors;
                invalid[index].dimensions = shape;
                assert!(
                    request_for(&invalid, &outputs).is_none(),
                    "shape at tensor {index}: {shape:?}"
                );
            }
        }
        for index in 0..3 {
            let mut aliases = outputs;
            aliases[index] = outputs[(index + 1) % 3];
            assert!(request_for(&tensors, &aliases).is_none());
        }
        // Scalar inputs and one-pixel results have identical geometry, but
        // the resource read still may not overwrite any coordinate operand.
        for tensor in &mut tensors[3..] {
            tensor.dimensions = &[1, 1];
        }
        assert!(request_for(&tensors, &outputs).is_some());
        for output_index in 0..3 {
            for input_index in 0..3 {
                let mut aliases = outputs;
                aliases[output_index] = TensorId(input_index);
                assert!(request_for(&tensors, &aliases).is_none());
            }
        }
    }

    #[test]
    fn resource_schema_does_not_broaden_pure_operator_attributes() {
        let mut builder = GraphBuilder::new_in(Global);
        let input = builder
            .tensor(crate::DType::F32, TensorFlags::INPUT, &[2], &[])
            .unwrap();
        let output = builder
            .tensor(crate::DType::F32, TensorFlags::OUTPUT, &[2], &[])
            .unwrap();
        builder
            .node(
                OpCode::IDENTITY,
                NodeFlags::default(),
                &[input],
                &[output],
                &[1, 1, 0, 0],
            )
            .unwrap();
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        assert!(node_kernel_request(&graph, graph.node(0).unwrap()).is_none());
    }
}

// Explicit little-endian attributes: softmax axis i32; normalization epsilon f32.
// Normalization currently follows the existing full-tensor axnn contract.
fn neural_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if node.flags != NodeFlags::default() || node.output_count() != 1 {
        return None;
    }
    let request = match node.op {
        OpCode::RMS_NORM => {
            let [input, weight] = exact_inputs::<2>(node)?;
            let (epsilon, axis) = crate::tensor_ops::decode_rms_norm_attributes(node.attributes())?;
            KernelRequest::rms_norm_over(
                &tensor_desc(graph, input)?,
                &tensor_desc(graph, weight)?,
                epsilon,
                axis as isize,
            )?
        }
        _ => {
            let attribute: [u8; 4] = node.attributes().try_into().ok()?;
            neural_4_byte_request(graph, node, attribute)?
        }
    };
    (request.result_tensor(0) == Some(&tensor_desc(graph, only(node.outputs())?)?))
        .then_some(request)
}

/// Softmax and layer norm, whose records are one 4-byte field.
fn neural_4_byte_request(
    graph: &Graph<'_>,
    node: NodeView<'_>,
    attribute: [u8; 4],
) -> Option<KernelRequest> {
    Some(match node.op {
        OpCode::SOFTMAX => KernelRequest::softmax(
            &tensor_desc(graph, only(node.inputs())?)?,
            i32::from_le_bytes(attribute) as isize,
        )?,
        OpCode::LAYER_NORM => {
            let [input, weight, bias] = exact_inputs::<3>(node)?;
            let epsilon = f32::from_le_bytes(attribute);
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return None;
            }
            KernelRequest::layer_norm(
                &tensor_desc(graph, input)?,
                &tensor_desc(graph, weight)?,
                &tensor_desc(graph, bias)?,
                epsilon,
            )?
        }
        _ => return None,
    })
}

/// `VIEW` and `TOP_K`: one input, attribute-checked, exact results.
fn selection_request(graph: &Graph<'_>, node: NodeView<'_>) -> Option<KernelRequest> {
    if node.flags != NodeFlags::default() {
        return None;
    }
    let input = only(node.inputs())?;
    if node.outputs().any(|output| output == input) {
        return None;
    }
    let desc = tensor_desc(graph, input)?;
    let request = if node.op == OpCode::VIEW {
        let output = tensor_desc(graph, only(node.outputs())?)?;
        KernelRequest::view(
            &desc,
            &output,
            crate::tensor_ops::decode_view_attributes(node.attributes())?,
        )?
    } else {
        KernelRequest::top_k(
            &desc,
            crate::tensor_ops::decode_top_k_attributes(node.attributes())?,
        )?
    };
    if node.output_count() != request.result_count() {
        return None;
    }
    for (index, id) in node.outputs().enumerate() {
        if request.result_tensor(index) != Some(&tensor_desc(graph, id)?) {
            return None;
        }
    }
    Some(request)
}
