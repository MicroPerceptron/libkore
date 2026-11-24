//! One of two DSpark context FC partitions over three separate target captures.
//!
//! The dynamic I8 rows are the left or right half of the converted full FC
//! matrix. Input order fixes the three consecutive target capture positions;
//! the companion partition consumes the other three. Their outputs are added
//! and normalized by a later device program. Neither partition copies or
//! concatenates captured rows on the host.

use core::alloc::Allocator;

use crate::{
    AllocationError, AxError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags, NodeView, OpCode,
    TensorFlags, TensorId,
};

pub const ROWS: u32 = 8;
pub const HIDDEN: u32 = 2688;
pub const CAPTURES: u32 = 3;
pub const WEIGHT_WIDTH: u32 = HIDDEN * CAPTURES;

/// The original eight-row graph remains the default and keeps its encoding.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    graph_rows_in(ROWS, allocator)
}

/// Five dynamic inputs plus one output consume all six binding slots. Each
/// admitted graph has one static row count; capture slices need no padding.
pub fn graph_rows_in<A: Allocator + Clone>(
    rows: u32,
    allocator: A,
) -> Result<GraphBuf<A>, AllocationError> {
    if !(1..=ROWS).contains(&rows) {
        return Err(AxError::InvalidTensorRange.into());
    }
    let mut graph = GraphBuilder::new_in(allocator);
    let input = TensorFlags::INPUT;
    let first = graph.tensor(DType::F32, input, &[rows, HIDDEN], &[])?;
    let second = graph.tensor(DType::F32, input, &[rows, HIDDEN], &[])?;
    let third = graph.tensor(DType::F32, input, &[rows, HIDDEN], &[])?;
    let weights = graph.tensor(DType::I8, input, &[HIDDEN, WEIGHT_WIDTH], &[])?;
    let scales = graph.tensor(DType::F32, input, &[1, HIDDEN], &[])?;
    let output = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[rows, HIDDEN], &[])?;
    graph.node(
        OpCode::CONTEXT_PARTIAL_I8_BLOCK,
        NodeFlags::default(),
        &[first, second, third, weights, scales],
        &[output],
        &[],
    )?;
    graph.finish()
}

pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if node.op != OpCode::CONTEXT_PARTIAL_I8_BLOCK
        || node.flags != NodeFlags::default()
        || !node.attributes().is_empty()
        || !node.inputs().eq((0..5).map(TensorId))
        || !node.outputs().eq([TensorId(5)])
    {
        return false;
    }
    let rows = graph
        .tensor(TensorId(0))
        .ok()
        .and_then(|tensor| tensor.dimensions().next());
    let Some(rows @ 1..=ROWS) = rows else {
        return false;
    };
    let expected = [
        (DType::F32, &[rows, HIDDEN][..]),
        (DType::F32, &[rows, HIDDEN]),
        (DType::F32, &[rows, HIDDEN]),
        (DType::I8, &[HIDDEN, WEIGHT_WIDTH]),
        (DType::F32, &[1, HIDDEN]),
        (DType::F32, &[rows, HIDDEN]),
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
            13
        );
        assert!(validate_node(&graph, graph.node(0).unwrap()));
    }

    #[test]
    fn each_static_row_count_has_matching_capture_and_output_shape() {
        let legacy = graph_in(Global).unwrap();
        for rows in 1..=ROWS {
            let bytes = graph_rows_in(rows, Global).unwrap();
            let graph = bytes.parse().unwrap();
            assert!(validate_node(&graph, graph.node(0).unwrap()), "rows={rows}");
            for id in [0, 1, 2, 5] {
                assert_eq!(
                    graph
                        .tensor(TensorId(id))
                        .unwrap()
                        .dimensions()
                        .collect::<alloc::vec::Vec<_>>(),
                    [rows, HIDDEN],
                    "rows={rows} tensor={id}"
                );
            }
            if rows == ROWS {
                assert_eq!(bytes.as_bytes(), legacy.as_bytes());
            }
        }
        for rows in [0, ROWS + 1, u32::MAX] {
            assert!(matches!(
                graph_rows_in(rows, Global),
                Err(AllocationError::Format(AxError::InvalidTensorRange))
            ));
        }
    }

    #[test]
    fn mixed_capture_rows_are_refused() {
        let mut graph = GraphBuilder::new_in(Global);
        let input = TensorFlags::INPUT;
        let captures = [
            graph.tensor(DType::F32, input, &[3, HIDDEN], &[]).unwrap(),
            graph.tensor(DType::F32, input, &[4, HIDDEN], &[]).unwrap(),
            graph.tensor(DType::F32, input, &[3, HIDDEN], &[]).unwrap(),
        ];
        let weights = graph
            .tensor(DType::I8, input, &[HIDDEN, WEIGHT_WIDTH], &[])
            .unwrap();
        let scales = graph.tensor(DType::F32, input, &[1, HIDDEN], &[]).unwrap();
        let output = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[3, HIDDEN], &[])
            .unwrap();
        graph
            .node(
                OpCode::CONTEXT_PARTIAL_I8_BLOCK,
                NodeFlags::default(),
                &[captures[0], captures[1], captures[2], weights, scales],
                &[output],
                &[],
            )
            .unwrap();
        let bytes = graph.finish().unwrap();
        let parsed = bytes.parse().unwrap();
        assert!(!validate_node(&parsed, parsed.node(0).unwrap()));
    }

    #[test]
    fn wrong_weight_extent_is_refused() {
        let mut graph = GraphBuilder::new_in(Global);
        let inputs = [
            graph
                .tensor(DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN], &[])
                .unwrap(),
            graph
                .tensor(DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN], &[])
                .unwrap(),
            graph
                .tensor(DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN], &[])
                .unwrap(),
            graph
                .tensor(
                    DType::I8,
                    TensorFlags::INPUT,
                    &[HIDDEN, WEIGHT_WIDTH - 1],
                    &[],
                )
                .unwrap(),
            graph
                .tensor(DType::F32, TensorFlags::INPUT, &[1, HIDDEN], &[])
                .unwrap(),
        ];
        let output = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, HIDDEN], &[])
            .unwrap();
        graph
            .node(
                OpCode::CONTEXT_PARTIAL_I8_BLOCK,
                NodeFlags::default(),
                &inputs,
                &[output],
                &[],
            )
            .unwrap();
        let graph = graph.finish().unwrap();
        let parsed = graph.parse().unwrap();
        assert!(!validate_node(&parsed, parsed.node(0).unwrap()));
    }
}
