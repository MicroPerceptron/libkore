//! The target's shared LM head over eight final-normed drafter rows.
//!
//! The target checkpoint stores this matrix as NVFP4. The I8 matrix and row
//! scales are a separately verified, one-time model-local preparation. Both
//! remain dynamic read-only bindings; no target weights enter the `.ax` graph.

use core::alloc::Allocator;

use crate::{AllocationError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags,
            NodeView, OpCode, TensorFlags, TensorId};

pub const ROWS: u32 = 8;
pub const HIDDEN: u32 = 2688;
pub const VOCAB: u32 = 131072;

/// Hidden `[8,2688]`, I8 LM rows `[131072,2688]`, row scales
/// `[1,131072]` -> base logits `[8,131072]`.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    let mut graph = GraphBuilder::new_in(allocator);
    let hidden = graph.tensor(DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN], &[])?;
    let weights = graph.tensor(DType::I8, TensorFlags::INPUT, &[VOCAB, HIDDEN], &[])?;
    let scales = graph.tensor(DType::F32, TensorFlags::INPUT, &[1, VOCAB], &[])?;
    let logits = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, VOCAB], &[])?;
    graph.node(OpCode::DSPARK_LM_HEAD_I8, NodeFlags::default(),
               &[hidden, weights, scales], &[logits], &[])?;
    graph.finish()
}

pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    let expected = [
        (DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN][..]),
        (DType::I8, TensorFlags::INPUT, &[VOCAB, HIDDEN]),
        (DType::F32, TensorFlags::INPUT, &[1, VOCAB]),
        (DType::F32, TensorFlags::OUTPUT, &[ROWS, VOCAB]),
    ];
    node.op == OpCode::DSPARK_LM_HEAD_I8
        && node.flags == NodeFlags::default()
        && node.attributes().is_empty()
        && node.inputs().eq((0..3).map(TensorId))
        && node.outputs().eq([TensorId(3)])
        && graph.tensor_count() == expected.len()
        && expected.iter().enumerate().all(|(id, (dtype, flags, shape))| {
            graph.tensor(TensorId(id as u32)).is_ok_and(|tensor| {
                tensor.dtype == *dtype && tensor.flags == *flags
                    && tensor.dimensions().eq(shape.iter().copied())
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    #[test]
    fn exact_weight_free_head_contract() {
        let bytes = graph_in(Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert!(validate_node(&graph, graph.node(0).unwrap()));
        assert_eq!((graph.tensor_count(), graph.node_count()), (4, 1));
        assert_eq!(u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()), 17);
    }
}
