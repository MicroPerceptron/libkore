//! Exact typed preparation of one fixed eight-row grouped-query block.
//!
//! This device-neutral operation applies the input RMS norm, the dynamic
//! Q/K/V matrix, the Q and K per-head RMS norms, and NeoX rotary position
//! encoding. Its two outputs are laid out for `SINK_GQA_BLOCK` without a host
//! repack. All checkpoint tensors are inputs; the graph embeds no weights.

use core::alloc::Allocator;

use crate::{
    AllocationError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags, NodeView, OpCode,
    TensorFlags, TensorId,
};

pub const ROWS: usize = 8;
pub const HIDDEN: usize = 2688;
pub const HEADS: usize = 32;
pub const KV_HEADS: usize = 2;
pub const DIM: usize = 128;
pub const QUERY: usize = HEADS * DIM;
pub const KEY_VALUE: usize = KV_HEADS * DIM;
pub const PROJECTED: usize = QUERY + 2 * KEY_VALUE;
pub const NORM: usize = HIDDEN + 2 * DIM;
pub const PHASE: usize = DIM;
/// The published DSpark configuration and Axiom's scalar RMS norm use 1e-6.
/// A different epsilon requires a versioned operation, not an implicit change.
pub const EPSILON: f32 = 1.0e-6;

/// External port order: hidden, transposed/concatenated QKV weights, packed
/// norms, phase; then query and block K/V outputs.
///
/// The weights are `[HIDDEN, PROJECTED]`, each row laid out as `Q | K | V`.
/// Packed norms are `[input_norm[HIDDEN], q_norm[DIM], k_norm[DIM]]`.
/// Phase is `[ROWS, DIM]`: for row `r`, the first DIM/2 elements are
/// `cos((base+r)*inv_freq[d])`, followed by DIM/2 sine elements. Each Q/K head
/// rotates the `(d, d+DIM/2)` pair, exactly as Axiom's DSpark scalar path.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    let mut builder = GraphBuilder::new_in(allocator);
    let input = TensorFlags::INPUT;
    let hidden = builder.tensor(DType::F32, input, &[ROWS as u32, HIDDEN as u32], &[])?;
    let weights = builder.tensor(DType::F32, input, &[HIDDEN as u32, PROJECTED as u32], &[])?;
    let norms = builder.tensor(DType::F32, input, &[NORM as u32], &[])?;
    let phase = builder.tensor(DType::F32, input, &[ROWS as u32, PHASE as u32], &[])?;
    let query = builder.tensor(
        DType::F32,
        TensorFlags::OUTPUT,
        &[ROWS as u32, HEADS as u32, DIM as u32],
        &[],
    )?;
    let block = builder.tensor(
        DType::F32,
        TensorFlags::OUTPUT,
        &[2, ROWS as u32, KV_HEADS as u32, DIM as u32],
        &[],
    )?;
    builder.node(
        OpCode::QKV_PREP_BLOCK,
        NodeFlags::default(),
        &[hidden, weights, norms, phase],
        &[query, block],
        &[],
    )?;
    builder.finish()
}

/// Only this complete six-port signature may identify `QKV_PREP_BLOCK` v1.
pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if node.op != OpCode::QKV_PREP_BLOCK
        || node.flags != NodeFlags::default()
        || !node.attributes().is_empty()
    {
        return false;
    }
    let mut inputs = node.inputs();
    let (Some(hidden), Some(weights), Some(norms), Some(phase), None) = (
        inputs.next(),
        inputs.next(),
        inputs.next(),
        inputs.next(),
        inputs.next(),
    ) else {
        return false;
    };
    let mut outputs = node.outputs();
    let (Some(query), Some(block), None) = (outputs.next(), outputs.next(), outputs.next()) else {
        return false;
    };
    let valid = |id: TensorId, shape: &[usize]| {
        graph.tensor(id).is_ok_and(|tensor| {
            tensor.dtype == DType::F32
                && tensor
                    .dimensions()
                    .map(|d| d as usize)
                    .eq(shape.iter().copied())
        })
    };
    valid(hidden, &[ROWS, HIDDEN])
        && valid(weights, &[HIDDEN, PROJECTED])
        && valid(norms, &[NORM])
        && valid(phase, &[ROWS, PHASE])
        && valid(query, &[ROWS, HEADS, DIM])
        && valid(block, &[2, ROWS, KV_HEADS, DIM])
        && ![hidden, weights, norms, phase].contains(&query)
        && ![hidden, weights, norms, phase, query].contains(&block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    #[test]
    fn graph_has_six_exact_ports_and_new_minor_version() {
        let bytes = graph_in(Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.tensor_count(), 6);
        assert_eq!(graph.node_count(), 1);
        assert!(validate_node(&graph, graph.node(0).unwrap()));
        assert_eq!(
            u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()),
            11
        );
        for id in 0..graph.tensor_count() {
            assert!(
                !graph
                    .tensor(TensorId(id as u32))
                    .unwrap()
                    .flags
                    .contains(TensorFlags::CONSTANT)
            );
        }
        let sink_bytes = crate::sink_gqa::graph_in(Global).unwrap();
        let sink = sink_bytes.parse().unwrap();
        for (produced, consumed) in [(4, 0), (5, 2)] {
            let produced = graph.tensor(TensorId(produced)).unwrap();
            let consumed = sink.tensor(TensorId(consumed)).unwrap();
            assert_eq!(produced.dtype, consumed.dtype);
            assert!(produced.dimensions().eq(consumed.dimensions()));
        }
    }

    #[test]
    fn wrong_attention_view_or_port_type_is_rejected() {
        let mut builder = GraphBuilder::new_in(Global);
        let h = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[8, 2688], &[])
            .unwrap();
        let w = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[2688, 4608], &[])
            .unwrap();
        let n = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[2944], &[])
            .unwrap();
        let p = builder
            .tensor(DType::I32, TensorFlags::INPUT, &[8, 128], &[])
            .unwrap();
        let q = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[8, 32, 128], &[])
            .unwrap();
        let kv = builder
            .tensor(DType::F32, TensorFlags::OUTPUT, &[8, 2, 2, 128], &[])
            .unwrap();
        builder
            .node(
                OpCode::QKV_PREP_BLOCK,
                NodeFlags::default(),
                &[h, w, n, p],
                &[q, kv],
                &[],
            )
            .unwrap();
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        assert!(!validate_node(&graph, graph.node(0).unwrap()));
    }
}
