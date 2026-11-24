//! Device-neutral eight-row gated MLP with dynamic signed-I8 weights.
//!
//! The input is already post-attention RMS-normalized. For each output
//! channel, a signed-I8 row is multiplied by its F32 scale. The graph computes
//! `SiLU(x @ gate.T * gate_scale) * (x @ up.T * up_scale)` and leaves both
//! matrices as inputs, so a provider can bind resident buffers once.

use core::alloc::Allocator;

use crate::{
    AllocationError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags, NodeView, OpCode,
    TensorFlags, TensorId,
};

pub const ROWS: u32 = 8;
pub const HIDDEN: u32 = 2688;
pub const INTERMEDIATE: u32 = 6144;

/// Port order: normalized hidden, gate I8 rows, gate scales, up I8 rows,
/// up scales, then the gated product. Scales are `[1, INTERMEDIATE]`.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    let mut graph = GraphBuilder::new_in(allocator);
    let input = TensorFlags::INPUT;
    let x = graph.tensor(DType::F32, input, &[ROWS, HIDDEN], &[])?;
    let gate = graph.tensor(DType::I8, input, &[INTERMEDIATE, HIDDEN], &[])?;
    let gate_scales = graph.tensor(DType::F32, input, &[1, INTERMEDIATE], &[])?;
    let up = graph.tensor(DType::I8, input, &[INTERMEDIATE, HIDDEN], &[])?;
    let up_scales = graph.tensor(DType::F32, input, &[1, INTERMEDIATE], &[])?;
    let result = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, INTERMEDIATE], &[])?;
    graph.node(
        OpCode::MLP_GATE_UP_I8_BLOCK,
        NodeFlags::default(),
        &[x, gate, gate_scales, up, up_scales],
        &[result],
        &[],
    )?;
    graph.finish()
}

pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if node.op != OpCode::MLP_GATE_UP_I8_BLOCK
        || node.flags != NodeFlags::default()
        || !node.attributes().is_empty()
        || !node.inputs().eq((0..5).map(TensorId))
        || !node.outputs().eq([TensorId(5)])
    {
        return false;
    }
    let expected = [
        (DType::F32, &[ROWS, HIDDEN][..]),
        (DType::I8, &[INTERMEDIATE, HIDDEN]),
        (DType::F32, &[1, INTERMEDIATE]),
        (DType::I8, &[INTERMEDIATE, HIDDEN]),
        (DType::F32, &[1, INTERMEDIATE]),
        (DType::F32, &[ROWS, INTERMEDIATE]),
    ];
    expected.iter().enumerate().all(|(id, (dtype, shape))| {
        graph.tensor(TensorId(id as u32)).is_ok_and(|tensor| {
            tensor.dtype == *dtype && tensor.dimensions().eq(shape.iter().copied())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    #[test]
    fn exact_six_port_versioned_graph() {
        let bytes = graph_in(Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.tensor_count(), 6);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(
            u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()),
            12
        );
        assert!(validate_node(&graph, graph.node(0).unwrap()));
        for id in 0..6 {
            assert!(
                !graph
                    .tensor(TensorId(id))
                    .unwrap()
                    .flags
                    .contains(TensorFlags::CONSTANT)
            );
        }
    }

    #[test]
    fn wrong_scale_view_cannot_claim_the_fused_operation() {
        let mut graph = GraphBuilder::new_in(Global);
        let x = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN], &[])
            .unwrap();
        let gate = graph
            .tensor(DType::I8, TensorFlags::INPUT, &[INTERMEDIATE, HIDDEN], &[])
            .unwrap();
        let gate_scales = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[1, INTERMEDIATE], &[])
            .unwrap();
        let up = graph
            .tensor(DType::I8, TensorFlags::INPUT, &[INTERMEDIATE, HIDDEN], &[])
            .unwrap();
        let up_scales = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[INTERMEDIATE], &[])
            .unwrap();
        let output = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, INTERMEDIATE], &[])
            .unwrap();
        graph
            .node(
                OpCode::MLP_GATE_UP_I8_BLOCK,
                NodeFlags::default(),
                &[x, gate, gate_scales, up, up_scales],
                &[output],
                &[],
            )
            .unwrap();
        let graph = graph.finish().unwrap();
        let parsed = graph.parse().unwrap();
        assert!(!validate_node(&parsed, parsed.node(0).unwrap()));
    }
}
