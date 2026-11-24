//! Device-neutral eight-row I8 down projection with a device-side residual.
//!
//! The gated intermediate and the residual are separate dynamic inputs. The
//! signed-I8 rows and per-output scales remain dynamic as well, so one native
//! program can bind retained weight leases for every draft without embedding
//! checkpoint bytes or reading its intermediate back to the host.

use core::alloc::Allocator;

use crate::mlp_gate_up_i8::{HIDDEN, INTERMEDIATE, ROWS};
use crate::{
    AllocationError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags, NodeView, OpCode,
    TensorFlags, TensorId,
};

/// Port order: gated intermediate, down I8 rows, row scales, residual, output.
/// The output is `residual + x @ (I8_rows * scales).T`.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    let mut graph = GraphBuilder::new_in(allocator);
    let input = TensorFlags::INPUT;
    let x = graph.tensor(DType::F32, input, &[ROWS, INTERMEDIATE], &[])?;
    let weights = graph.tensor(DType::I8, input, &[HIDDEN, INTERMEDIATE], &[])?;
    let scales = graph.tensor(DType::F32, input, &[1, HIDDEN], &[])?;
    let residual = graph.tensor(DType::F32, input, &[ROWS, HIDDEN], &[])?;
    let output = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, HIDDEN], &[])?;
    graph.node(
        OpCode::MLP_DOWN_RESIDUAL_I8_BLOCK,
        NodeFlags::default(),
        &[x, weights, scales, residual],
        &[output],
        &[],
    )?;
    graph.finish()
}

pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if node.op != OpCode::MLP_DOWN_RESIDUAL_I8_BLOCK
        || node.flags != NodeFlags::default()
        || !node.attributes().is_empty()
        || !node.inputs().eq((0..4).map(TensorId))
        || !node.outputs().eq([TensorId(4)])
    {
        return false;
    }
    let expected = [
        (DType::F32, &[ROWS, INTERMEDIATE][..]),
        (DType::I8, &[HIDDEN, INTERMEDIATE]),
        (DType::F32, &[1, HIDDEN]),
        (DType::F32, &[ROWS, HIDDEN]),
        (DType::F32, &[ROWS, HIDDEN]),
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
    fn exact_five_port_versioned_graph() {
        let bytes = graph_in(Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.tensor_count(), 5);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(
            u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()),
            14
        );
        assert!(validate_node(&graph, graph.node(0).unwrap()));
        for id in 0..5 {
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
    fn residual_shape_mismatch_is_not_the_fused_operation() {
        let mut graph = GraphBuilder::new_in(Global);
        let x = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[ROWS, INTERMEDIATE], &[])
            .unwrap();
        let weights = graph
            .tensor(DType::I8, TensorFlags::INPUT, &[HIDDEN, INTERMEDIATE], &[])
            .unwrap();
        let scales = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[1, HIDDEN], &[])
            .unwrap();
        let residual = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[1, HIDDEN], &[])
            .unwrap();
        let output = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, HIDDEN], &[])
            .unwrap();
        graph
            .node(
                OpCode::MLP_DOWN_RESIDUAL_I8_BLOCK,
                NodeFlags::default(),
                &[x, weights, scales, residual],
                &[output],
                &[],
            )
            .unwrap();
        let graph = graph.finish().unwrap();
        let parsed = graph.parse().unwrap();
        assert!(!validate_node(&parsed, parsed.node(0).unwrap()));
    }
}
