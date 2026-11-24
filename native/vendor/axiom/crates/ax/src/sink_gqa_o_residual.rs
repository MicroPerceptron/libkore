//! One fixed DSpark sink-attention block, output projection, and residual add.
//!
//! Query and block K/V already include the Q/K norms and NeoX rotation from
//! `QKV_PREP_BLOCK`. The ring is kept in its physical K plane then V plane
//! order; the block K/V output occupies the tail of the same DMA allocation.
//! This packed view saves a binding and requires a producer/consumer exact-cover
//! proof at admission, not a copy of either tensor. Sinks, two small per-draft
//! controls, and per-row O scales share another binding. The O matrix is
//! prepared once as I8 `[output, input]` rows from checkpoint BF16. Its
//! quantization accuracy is a separate admission gate.

use core::alloc::Allocator;

use crate::{
    AllocationError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags, NodeView, OpCode,
    TensorFlags, TensorId,
};

pub const ROWS: u32 = 8;
pub const HEADS: u32 = 32;
pub const KV_HEADS: u32 = 2;
pub const DIM: u32 = 128;
pub const HIDDEN: u32 = 2688;
pub const WINDOW: u32 = 1024;
/// The resident ring has eight spare slots so staged block writes cannot
/// overwrite the 1024-row committed history before publication.
pub const RING_SLOTS: u32 = WINDOW + ROWS;
pub const QUERY_WIDTH: u32 = HEADS * DIM;
pub const CONTROL_WIDTH: u32 = HEADS + 2;
pub const PARAM_WIDTH: u32 = CONTROL_WIDTH + HIDDEN;
/// F32 elements before the block producer's `[2, ROWS, KV_HEADS, DIM]`
/// output, which may bind directly at this offset in the same DMA buffer.
pub const BLOCK_OFFSET: u64 = 2 * RING_SLOTS as u64 * KV_HEADS as u64 * DIM as u64;

/// External port order: Q, packed ring+block K/V, parameters, prepared O
/// weights, residual, then output. All six are dynamic bindings.
///
/// The packed K/V input's flat layout is `[ring K (1032 slots), ring V (1032
/// slots), block K (8 rows), block V (8 rows)]`; each slot/row has two heads
/// of 128 F32 values. The block suffix begins at `BLOCK_OFFSET * 4` bytes.
/// `params = [sink logits (32), head, held, O row scales (2688)]` as F32.
/// Both controls are exact integers: `head < 1032`, `held <= 1024`. `head` is
/// the physical next-write slot of the committed ring generation.
/// The caller checks those ranges before submission and keeps the parameter
/// view leased until the event completes. For block row `r`, only the most recent
/// `min(held, 1023-r)` ring slots preceding `head` and block rows `0..=r`
/// are visible. Ring addresses wrap modulo 1032; the logical window is 1024.
/// The sink contributes to the softmax denominator with zero value. Finally
/// `output[r,o] = residual[r,o] + scale[o] * Σ_i attended[r,i] * O_i8[o,i]`.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    let mut graph = GraphBuilder::new_in(allocator);
    let input = TensorFlags::INPUT;
    let query = graph.tensor(DType::F32, input, &[ROWS, HEADS, DIM], &[])?;
    let ring_block = graph.tensor(
        DType::F32,
        input,
        &[2 * (RING_SLOTS + ROWS), KV_HEADS, DIM],
        &[],
    )?;
    let params = graph.tensor(DType::F32, input, &[PARAM_WIDTH], &[])?;
    let o = graph.tensor(DType::I8, input, &[HIDDEN, QUERY_WIDTH], &[])?;
    let residual = graph.tensor(DType::F32, input, &[ROWS, HIDDEN], &[])?;
    let output = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, HIDDEN], &[])?;
    graph.node(
        OpCode::SINK_GQA_O_RESIDUAL_BLOCK,
        NodeFlags::default(),
        &[query, ring_block, params, o, residual],
        &[output],
        &[],
    )?;
    graph.finish()
}

/// The only version 1 native signature. An alias proof for the packed input
/// belongs to sequence admission; this validates the individual artifact.
pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if node.op != OpCode::SINK_GQA_O_RESIDUAL_BLOCK
        || node.flags != NodeFlags::default()
        || !node.attributes().is_empty()
        || !node.inputs().eq((0..5).map(TensorId))
        || !node.outputs().eq([TensorId(5)])
    {
        return false;
    }
    let expected = [
        (DType::F32, TensorFlags::INPUT, &[ROWS, HEADS, DIM][..]),
        (
            DType::F32,
            TensorFlags::INPUT,
            &[2 * (RING_SLOTS + ROWS), KV_HEADS, DIM],
        ),
        (DType::F32, TensorFlags::INPUT, &[PARAM_WIDTH]),
        (DType::I8, TensorFlags::INPUT, &[HIDDEN, QUERY_WIDTH]),
        (DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN]),
        (DType::F32, TensorFlags::OUTPUT, &[ROWS, HIDDEN]),
    ];
    graph.tensor_count() == expected.len()
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

/// Select the physical ring range and causal block prefix for one query row.
/// Returns `(first_ring_slot, ring_rows, block_rows)` after validating the
/// caller's integer controls. A wrapped ring range continues at slot zero.
pub fn attention_window(head: u32, held: u32, row: u32) -> Option<(u32, u32, u32)> {
    if head >= RING_SLOTS || held > WINDOW || row >= ROWS {
        return None;
    }
    let ring_rows = held.min(WINDOW - 1 - row);
    let first = (head + RING_SLOTS - ring_rows) % RING_SLOTS;
    Some((first, ring_rows, row + 1))
}

/// Check the two F32 control elements without accepting fractional, NaN, or
/// out-of-range values. The device compiler must enforce the same predicate
/// before using them as physical ring addresses.
pub fn checked_controls(head: f32, held: f32) -> Option<(u32, u32)> {
    if !(head >= 0.0 && head < RING_SLOTS as f32 && held >= 0.0 && held <= WINDOW as f32) {
        return None;
    }
    let (head_slot, held_rows) = (head as u32, held as u32);
    (head_slot as f32 == head && held_rows as f32 == held).then_some((head_slot, held_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    #[test]
    fn exact_six_port_graph_and_qkv_suffix() {
        let bytes = graph_in(Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert_eq!((graph.tensor_count(), graph.node_count()), (6, 1));
        assert_eq!(
            u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()),
            16
        );
        assert!(validate_node(&graph, graph.node(0).unwrap()));
        let producer = crate::qkv_prep::graph_in(Global).unwrap();
        let producer = producer.parse().unwrap();
        let produced_query = producer.tensor(TensorId(4)).unwrap();
        let consumed_query = graph.tensor(TensorId(0)).unwrap();
        assert_eq!(produced_query.dtype, consumed_query.dtype);
        assert!(produced_query.dimensions().eq(consumed_query.dimensions()));
        let produced = producer.tensor(TensorId(5)).unwrap();
        let suffix_elements = 2 * ROWS as u64 * KV_HEADS as u64 * DIM as u64;
        let packed_elements = graph
            .tensor(TensorId(1))
            .unwrap()
            .dimensions()
            .map(u64::from)
            .product::<u64>();
        let produced_elements = produced.dimensions().map(u64::from).product::<u64>();
        assert_eq!(produced_elements, suffix_elements);
        assert_eq!(BLOCK_OFFSET + suffix_elements, packed_elements);
        assert_eq!(produced.dtype, DType::F32);
    }

    #[test]
    fn wrong_weight_or_metadata_shape_is_rejected() {
        let mut graph = GraphBuilder::new_in(Global);
        let q = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[ROWS, HEADS, DIM], &[])
            .unwrap();
        let kv = graph
            .tensor(
                DType::F32,
                TensorFlags::INPUT,
                &[2 * (RING_SLOTS + ROWS), KV_HEADS, DIM],
                &[],
            )
            .unwrap();
        let params = graph
            .tensor(DType::I32, TensorFlags::INPUT, &[PARAM_WIDTH], &[])
            .unwrap();
        let o = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[HIDDEN, QUERY_WIDTH], &[])
            .unwrap();
        let residual = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN], &[])
            .unwrap();
        let output = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, HIDDEN], &[])
            .unwrap();
        graph
            .node(
                OpCode::SINK_GQA_O_RESIDUAL_BLOCK,
                NodeFlags::default(),
                &[q, kv, params, o, residual],
                &[output],
                &[],
            )
            .unwrap();
        let bytes = graph.finish().unwrap();
        let graph = bytes.parse().unwrap();
        assert!(!validate_node(&graph, graph.node(0).unwrap()));
    }

    #[test]
    fn wrapped_window_sink_and_residual_numerics() {
        // For zero query/key logits the sink's exp(log 2) has weight two.
        // Two wrapped ring values and one block value each have weight one.
        let (first, ring_rows, block_rows) = attention_window(0, 2, 0).unwrap();
        assert_eq!((first, ring_rows, block_rows), (RING_SLOTS - 2, 2, 1));
        let ring = [(RING_SLOTS - 2, 3.0_f64), (RING_SLOTS - 1, 5.0)];
        let block = [7.0_f64];
        let mut numerator = 0.0;
        let mut denominator = (2.0_f64.ln()).exp();
        for offset in 0..ring_rows {
            let slot = (first + offset) % RING_SLOTS;
            numerator += ring.iter().find(|(id, _)| *id == slot).unwrap().1;
            denominator += 1.0;
        }
        for value in block.iter().take(block_rows as usize) {
            numerator += value;
            denominator += 1.0;
        }
        let quantized_o_weight = 127_i8;
        let o_row_scale = 2.0_f64 / 127.0;
        let projected = (numerator / denominator) * f64::from(quantized_o_weight) * o_row_scale;
        let residual = 11.0;
        assert!((residual + projected - 17.0_f64).abs() < 1e-12);
        assert_eq!(attention_window(0, WINDOW, ROWS - 1), Some((16, 1016, 8)));
        assert_eq!(attention_window(RING_SLOTS, 0, 0), None);
        assert_eq!(attention_window(0, WINDOW + 1, 0), None);
        assert_eq!(attention_window(0, 0, ROWS), None);
        assert_eq!(checked_controls(1031.0, 1024.0), Some((1031, 1024)));
        assert_eq!(checked_controls(1.5, 2.0), None);
        assert_eq!(checked_controls(f32::NAN, 2.0), None);
        assert_eq!(checked_controls(0.0, f32::INFINITY), None);
    }
}
