//! Exact typed shape of one fixed DSpark-style sink grouped-query block.
//!
//! The operation is device-neutral: a query head attends its own block rows
//! causally and the committed context held in a physical-order circular ring.
//! Its sink contributes one learned logit to the denominator with zero value.
//! This first profile fixes the published 8x32x128, 2-KV, 1024-window shape;
//! widening geometry requires a versioned attribute and new validation.

use core::alloc::Allocator;

use crate::{
    AllocationError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags, NodeView, OpCode,
    TensorFlags, TensorId,
};

pub const ROWS: usize = 8;
pub const HEADS: usize = 32;
pub const KV_HEADS: usize = 2;
pub const DIM: usize = 128;
pub const WINDOW: usize = 1024;

/// Author the exact standalone five-input block graph. No checkpoint bytes or
/// runtime buffer identity enter the canonical graph.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    let mut builder = GraphBuilder::new_in(allocator);
    let input = TensorFlags::INPUT;
    let query = builder.tensor(DType::F32, input, &[8, 32, 128], &[])?;
    let ring = builder.tensor(DType::F32, input, &[2, 1024, 2, 128], &[])?;
    let block = builder.tensor(DType::F32, input, &[2, 8, 2, 128], &[])?;
    let sinks = builder.tensor(DType::F32, input, &[32], &[])?;
    let controls = builder.tensor(DType::I32, input, &[2], &[])?;
    let output = builder.tensor(DType::F32, TensorFlags::OUTPUT, &[8, 32, 128], &[])?;
    builder.node(
        OpCode::SINK_GQA_BLOCK,
        NodeFlags::default(),
        &[query, ring, block, sinks, controls],
        &[output],
        &[],
    )?;
    builder.finish()
}

/// The only native-port signature for `SINK_GQA_BLOCK` version 1.
///
/// `controls = [base_slot, held]`: the caller proves `base_slot < WINDOW`
/// and `held <= WINDOW` before submitting. Every row sees at most
/// `min(held, WINDOW - 1 - row)` context keys, and block keys `0..=row`.
/// The ring remains in physical slot order, `[K/V, slot, KV head, dim]`.
pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if node.op != OpCode::SINK_GQA_BLOCK
        || node.flags != NodeFlags::default()
        || !node.attributes().is_empty()
    {
        return false;
    }
    let mut inputs = node.inputs();
    let (Some(query), Some(ring), Some(block), Some(sinks), Some(controls), None) =
        (inputs.next(), inputs.next(), inputs.next(), inputs.next(), inputs.next(), inputs.next())
    else {
        return false;
    };
    let mut outputs = node.outputs();
    let (Some(output), None) = (outputs.next(), outputs.next()) else {
        return false;
    };
    let valid = |id: TensorId, dtype: DType, shape: &[usize]| {
        graph.tensor(id).is_ok_and(|tensor| {
            tensor.dtype == dtype && tensor.dimensions().map(|d| d as usize).eq(shape.iter().copied())
        })
    };
    valid(query, DType::F32, &[ROWS, HEADS, DIM])
        && valid(ring, DType::F32, &[2, WINDOW, KV_HEADS, DIM])
        && valid(block, DType::F32, &[2, ROWS, KV_HEADS, DIM])
        && valid(sinks, DType::F32, &[HEADS])
        && valid(controls, DType::I32, &[2])
        && valid(output, DType::F32, &[ROWS, HEADS, DIM])
        && ![query, ring, block, sinks, controls].contains(&output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    #[test]
    fn canonical_graph_has_exact_six_ports_and_version() {
        let bytes = graph_in(Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.tensor_count(), 6);
        assert_eq!(graph.node_count(), 1);
        assert!(validate_node(&graph, graph.node(0).unwrap()));
        assert_eq!(u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()), 10);
    }

    #[test]
    fn a_float_control_record_is_not_the_same_operation() {
        let mut builder = GraphBuilder::new_in(Global);
        let input = TensorFlags::INPUT;
        let q = builder.tensor(DType::F32, input, &[8, 32, 128], &[]).unwrap();
        let ring = builder.tensor(DType::F32, input, &[2, 1024, 2, 128], &[]).unwrap();
        let block = builder.tensor(DType::F32, input, &[2, 8, 2, 128], &[]).unwrap();
        let sinks = builder.tensor(DType::F32, input, &[32], &[]).unwrap();
        let controls = builder.tensor(DType::F32, input, &[2], &[]).unwrap();
        let output = builder.tensor(DType::F32, TensorFlags::OUTPUT, &[8, 32, 128], &[]).unwrap();
        builder.node(OpCode::SINK_GQA_BLOCK, NodeFlags::default(),
                     &[q, ring, block, sinks, controls], &[output], &[]).unwrap();
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        assert!(!validate_node(&graph, graph.node(0).unwrap()));
    }
}
