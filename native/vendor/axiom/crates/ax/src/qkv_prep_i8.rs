//! I8 variant of the fixed eight-row DSpark Q/K/V preparation block.
//!
//! The I8 matrix is output-major so it can be borrowed directly from the
//! prepared resident archive. Its per-output scales share one F32 parameter
//! port with the input, Q, and K RMSNorm vectors. All four inputs remain
//! dynamic DMA bindings; no model weights are serialized in the graph.

use core::alloc::Allocator;

use crate::{
    AllocationError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags, NodeView, OpCode,
    TensorFlags, TensorId,
};

pub const ROWS: u32 = 8;
pub const HIDDEN: u32 = 2688;
pub const HEADS: u32 = 32;
pub const KV_HEADS: u32 = 2;
pub const DIM: u32 = 128;
pub const PROJECTED: u32 = (HEADS + 2 * KV_HEADS) * DIM;
pub const PARAMS: u32 = HIDDEN + 2 * DIM + PROJECTED;

/// Hidden `[8,2688]`, output-major I8 QKV `[4608,2688]`, F32 parameters
/// `[7552]`, rotary phase `[8,128]` -> query `[8,32,128]` and packed block
/// K/V `[2,8,2,128]` compatible with the sink attention profile.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    let mut graph = GraphBuilder::new_in(allocator);
    let hidden = graph.tensor(DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN], &[])?;
    let weights = graph.tensor(DType::I8, TensorFlags::INPUT, &[PROJECTED, HIDDEN], &[])?;
    let params = graph.tensor(DType::F32, TensorFlags::INPUT, &[PARAMS], &[])?;
    let phase = graph.tensor(DType::F32, TensorFlags::INPUT, &[ROWS, DIM], &[])?;
    let query = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, HEADS, DIM], &[])?;
    let block = graph.tensor(
        DType::F32,
        TensorFlags::OUTPUT,
        &[2, ROWS, KV_HEADS, DIM],
        &[],
    )?;
    graph.node(
        OpCode::QKV_PREP_I8_BLOCK,
        NodeFlags::default(),
        &[hidden, weights, params, phase],
        &[query, block],
        &[],
    )?;
    graph.finish()
}

pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    let expected = [
        (DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN][..]),
        (DType::I8, TensorFlags::INPUT, &[PROJECTED, HIDDEN]),
        (DType::F32, TensorFlags::INPUT, &[PARAMS]),
        (DType::F32, TensorFlags::INPUT, &[ROWS, DIM]),
        (DType::F32, TensorFlags::OUTPUT, &[ROWS, HEADS, DIM]),
        (DType::F32, TensorFlags::OUTPUT, &[2, ROWS, KV_HEADS, DIM]),
    ];
    node.op == OpCode::QKV_PREP_I8_BLOCK
        && node.flags == NodeFlags::default()
        && node.attributes().is_empty()
        && node.inputs().eq((0..4).map(TensorId))
        && node.outputs().eq([TensorId(4), TensorId(5)])
        && graph.tensor_count() == expected.len()
        && expected
            .iter()
            .enumerate()
            .all(|(id, (dtype, flags, shape))| {
                graph.tensor(TensorId(id as u32)).is_ok_and(|tensor| {
                    tensor.dtype == *dtype
                        && tensor.flags == *flags
                        && tensor.dimensions().eq(shape.iter().copied())
                })
            })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    #[test]
    fn exact_archive_binding_and_sink_output() {
        let bytes = graph_in(Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert!(validate_node(&graph, graph.node(0).unwrap()));
        assert_eq!(
            u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()),
            19
        );
        let sink = crate::sink_gqa::graph_in(Global).unwrap();
        let sink = sink.parse().unwrap();
        for (produced, consumed) in [(4, 0), (5, 2)] {
            let source = graph.tensor(TensorId(produced)).unwrap();
            let target = sink.tensor(TensorId(consumed)).unwrap();
            assert_eq!(source.dtype, target.dtype);
            assert!(source.dimensions().eq(target.dimensions()));
        }
    }
}
