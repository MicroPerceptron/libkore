//! Seven serial Markov choices after the target LM head has produced logits.
//!
//! Row zero is the bonus anchor and is never sampled. For rows one through
//! seven, `d_i = argmax_min_id(base_i + W2 * W1[d_(i-1)])`, with `d_0` equal
//! to the anchor. The previous token is carried **inside** one device graph.
//! A tie chooses the lowest token id, matching the CPU drafter; it cannot be replaced with an
//! unspecified TopK tie rule. W1 is the original BF16 checkpoint table; W2
//! and scales are a one-time model-local I8 conversion. All are borrowed
//! dynamic bindings, retained across the native event by the caller.

use core::alloc::Allocator;

use crate::{AllocationError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags,
            NodeView, OpCode, TensorFlags, TensorId};

pub const ROWS: u32 = 8;
pub const PROPOSALS: u32 = ROWS - 1;
pub const VOCAB: u32 = 131072;
pub const RANK: u32 = 512;

/// Eight base-logit rows, W1 BF16, W2 I8, scales, anchor -> seven proposals.
/// The five inputs and one output exactly fill the native six-port ABI.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    let mut graph = GraphBuilder::new_in(allocator);
    let logits = graph.tensor(DType::F32, TensorFlags::INPUT, &[ROWS, VOCAB], &[])?;
    let w1 = graph.tensor(DType::B16, TensorFlags::INPUT, &[VOCAB, RANK], &[])?;
    let w2 = graph.tensor(DType::I8, TensorFlags::INPUT, &[VOCAB, RANK], &[])?;
    let scales = graph.tensor(DType::F32, TensorFlags::INPUT, &[1, VOCAB], &[])?;
    let anchor = graph.tensor(DType::I32, TensorFlags::INPUT, &[1], &[])?;
    let ids = graph.tensor(DType::I32, TensorFlags::OUTPUT, &[PROPOSALS], &[])?;
    graph.node(OpCode::DSPARK_MARKOV_FEEDBACK, NodeFlags::default(),
               &[logits, w1, w2, scales, anchor], &[ids], &[])?;
    graph.finish()
}

pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    let expected = [
        (DType::F32, TensorFlags::INPUT, &[ROWS, VOCAB][..]),
        (DType::B16, TensorFlags::INPUT, &[VOCAB, RANK]),
        (DType::I8, TensorFlags::INPUT, &[VOCAB, RANK]),
        (DType::F32, TensorFlags::INPUT, &[1, VOCAB]),
        (DType::I32, TensorFlags::INPUT, &[1]),
        (DType::I32, TensorFlags::OUTPUT, &[PROPOSALS]),
    ];
    node.op == OpCode::DSPARK_MARKOV_FEEDBACK
        && node.flags == NodeFlags::default()
        && node.attributes().is_empty()
        && node.inputs().eq((0..5).map(TensorId))
        && node.outputs().eq([TensorId(5)])
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
    fn exact_six_port_feedback_contract() {
        let bytes = graph_in(Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert!(validate_node(&graph, graph.node(0).unwrap()));
        assert_eq!((graph.tensor_count(), graph.node_count()), (6, 1));
        assert_eq!(
            graph
                .tensor(TensorId(5))
                .unwrap()
                .dimensions()
                .collect::<alloc::vec::Vec<_>>(),
            [7]
        );
        assert_eq!(u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()), 18);
    }
}
