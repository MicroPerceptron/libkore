//! Typed DSpark context-partial consumer and one layer's K/V producer.
//!
//! Two partial FC outputs occupy disjoint halves of one borrowed DMA view.
//! This graph adds them, applies the checkpoint hidden RMSNorm, projects one
//! layer's combined K/V BF16-derived matrix, applies key RMSNorm and NeoX
//! rotation, then emits distinct contiguous key and value rows. The outputs
//! can bind distinct existing-buffer ranges; ring publication is outside this
//! pure operation and requires its own lifetime proof.

use core::alloc::Allocator;

use crate::{
    AllocationError, AxError, DType, Graph, GraphBuf, GraphBuilder, NodeFlags, NodeView, OpCode,
    TensorFlags, TensorId,
};

pub const ROWS: u32 = 8;
pub const HIDDEN: u32 = 2688;
pub const KV_HEADS: u32 = 2;
pub const DIM: u32 = 128;
pub const KV_WIDTH: u32 = KV_HEADS * DIM;
pub const NORM_WIDTH: u32 = HIDDEN + DIM;
/// Hidden norm, key norm, and one scale per K/V output row.
pub const I8_PARAM_WIDTH: u32 = NORM_WIDTH + 2 * KV_WIDTH;
pub const EPSILON: f32 = 1.0e-6;

/// The DSpark attention window, with one full graph block of unpublished slots.
/// A writer of at most `ROWS` rows can therefore never overwrite a committed
/// row, even when the logical window is full.
pub const WINDOW: usize = 1024;
pub const SHADOW_CAPACITY: usize = WINDOW + ROWS as usize;
/// One F32 key row or value row, `[KV_HEADS, DIM]`.
pub const ROW_BYTES: usize = KV_WIDTH as usize * size_of::<f32>();
pub const LAYER_BYTES: usize = 2 * SHADOW_CAPACITY * ROW_BYTES;

/// Refusals from the allocation-free, per-sequence context ring planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingError {
    Rows,
    Position,
    Pending,
    Stale,
    Accepted,
    Overflow,
}

/// Only published rows are visible to attention. `head` is the physical slot
/// after the newest committed row, not `next % WINDOW`. A reader must use
/// `SHADOW_CAPACITY` for physical wrap and `held` for its logical window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingSnapshot {
    pub next: u64,
    pub head: usize,
    pub held: usize,
    pub generation: u64,
}

/// One contiguous device-output range within a K/V plane. All offsets are
/// relative to one layer's `[K,V][SHADOW_CAPACITY][KV_HEADS,DIM]` DMA buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingSegment {
    pub first_row: usize,
    pub rows: usize,
    pub key_offset: usize,
    pub value_offset: usize,
}

/// Affine description of a single unpublished multi-layer write. The same
/// segments apply to every layer, and must all retire before publication.
#[derive(Debug)]
pub struct StagedRows {
    ticket: u64,
    first: u64,
    rows: usize,
    segments: [RingSegment; 2],
    segment_count: usize,
}

impl StagedRows {
    pub fn first(&self) -> u64 {
        self.first
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Compile the corresponding static-row op84 and op86 graph for each
    /// segment; bind op86 outputs directly at these K/V offsets. A physical
    /// wrap needs two graph executions, never a host copy of K/V rows.
    pub fn segments(&self) -> &[RingSegment] {
        &self.segments[..self.segment_count]
    }
}

/// Tracks a *sequence-wide* K/V generation shared by all context layers.
///
/// `stage` reserves only spare physical slots. Call `retire_complete` after
/// every layer's writes are known retired successfully; it publishes just the
/// accepted prefix. Call `retire_failed` only after every possibly submitted
/// write is known stopped. If completion is unknown, leave the stage pending:
/// dropping `StagedRows` intentionally does not release its reservation.
/// A caller must not expose an unpublished slot through an attention binding.
///
/// Reader integration is still required: `sink_gqa` version 1 admits a
/// `[2,1024,2,128]` ring and wraps physical slots modulo 1024. A versioned
/// reader profile must instead accept `[2,1032,2,128]`, take `snapshot.head`
/// as its physical base slot, retain the 1024-token *logical* causal window,
/// and locate its `seen` context rows starting at
/// `(head + SHADOW_CAPACITY - seen) % SHADOW_CAPACITY`. That compiler artifact
/// and its numerical behavior require separate VPU validation.
#[derive(Debug)]
pub struct ContextRing {
    snapshot: RingSnapshot,
    next_ticket: u64,
    pending: Option<u64>,
}

impl ContextRing {
    pub const fn new(next: u64) -> Self {
        Self {
            snapshot: RingSnapshot {
                next,
                head: 0,
                held: 0,
                generation: 0,
            },
            next_ticket: 0,
            pending: None,
        }
    }

    pub const fn snapshot(&self) -> RingSnapshot {
        self.snapshot
    }

    /// Discard the old logical sequence after all device writes have retired.
    pub fn reset_to(&mut self, next: u64) -> Result<(), RingError> {
        if self.pending.is_some() {
            return Err(RingError::Pending);
        }
        let generation = self
            .snapshot
            .generation
            .checked_add(1)
            .ok_or(RingError::Overflow)?;
        self.snapshot = RingSnapshot {
            next,
            head: 0,
            held: 0,
            generation,
        };
        Ok(())
    }

    pub fn stage(&mut self, first: u64, rows: usize) -> Result<StagedRows, RingError> {
        if self.pending.is_some() {
            return Err(RingError::Pending);
        }
        if !(1..=ROWS as usize).contains(&rows) {
            return Err(RingError::Rows);
        }
        if first != self.snapshot.next {
            return Err(RingError::Position);
        }
        first.checked_add(rows as u64).ok_or(RingError::Overflow)?;
        self.snapshot
            .generation
            .checked_add(1)
            .ok_or(RingError::Overflow)?;
        let ticket = self.next_ticket.checked_add(1).ok_or(RingError::Overflow)?;
        let head = self.snapshot.head;
        let leading = rows.min(SHADOW_CAPACITY - head);
        let segment = |first_row: usize, rows: usize, slot: usize| RingSegment {
            first_row,
            rows,
            key_offset: slot * ROW_BYTES,
            value_offset: (SHADOW_CAPACITY + slot) * ROW_BYTES,
        };
        self.next_ticket = ticket;
        self.pending = Some(ticket);
        Ok(StagedRows {
            ticket,
            first,
            rows,
            segments: [
                segment(0, leading, head),
                segment(leading, rows - leading, 0),
            ],
            segment_count: if leading == rows { 1 } else { 2 },
        })
    }

    /// Publish the accepted prefix only after all six layers' writes have
    /// retired complete in one ordered transaction. Rejected suffix rows stay
    /// unpublished and are eligible for overwrite by the next stage.
    pub fn retire_complete(
        &mut self,
        stage: StagedRows,
        accepted: usize,
    ) -> Result<RingSnapshot, RingError> {
        if self.pending != Some(stage.ticket) || self.snapshot.next != stage.first {
            return Err(RingError::Stale);
        }
        if accepted > stage.rows {
            return Err(RingError::Accepted);
        }
        let next = self
            .snapshot
            .next
            .checked_add(accepted as u64)
            .ok_or(RingError::Overflow)?;
        let generation = self
            .snapshot
            .generation
            .checked_add(1)
            .ok_or(RingError::Overflow)?;
        self.snapshot.next = next;
        self.snapshot.head = (self.snapshot.head + accepted) % SHADOW_CAPACITY;
        self.snapshot.held = (self.snapshot.held + accepted).min(WINDOW);
        self.snapshot.generation = generation;
        self.pending = None;
        Ok(self.snapshot)
    }

    /// A failed execution may have dirtied spare slots, but cannot have
    /// changed the committed generation. Failure must be terminal, not merely
    /// a timeout or an unknown completion state.
    pub fn retire_failed(&mut self, stage: StagedRows) -> Result<RingSnapshot, RingError> {
        if self.pending != Some(stage.ticket) || self.snapshot.next != stage.first {
            return Err(RingError::Stale);
        }
        self.pending = None;
        Ok(self.snapshot)
    }
}

/// The original eight-row graph remains the default and keeps its encoding.
pub fn graph_in<A: Allocator + Clone>(allocator: A) -> Result<GraphBuf<A>, AllocationError> {
    graph_rows_in(ROWS, allocator)
}

/// Port order: packed partials, transposed K|V weights, hidden/key norms,
/// rotary phases, then separate key and value outputs. All are dynamic ports,
/// while each admitted graph has one static row count.
pub fn graph_rows_in<A: Allocator + Clone>(
    rows: u32,
    allocator: A,
) -> Result<GraphBuf<A>, AllocationError> {
    if !(1..=ROWS).contains(&rows) {
        return Err(AxError::InvalidTensorRange.into());
    }
    let mut graph = GraphBuilder::new_in(allocator);
    let input = TensorFlags::INPUT;
    let partials = graph.tensor(DType::F32, input, &[2, rows, HIDDEN], &[])?;
    let weights = graph.tensor(DType::F32, input, &[HIDDEN, 2 * KV_WIDTH], &[])?;
    let norms = graph.tensor(DType::F32, input, &[NORM_WIDTH], &[])?;
    let phase = graph.tensor(DType::F32, input, &[rows, DIM], &[])?;
    let key = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[rows, KV_HEADS, DIM], &[])?;
    let value = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[rows, KV_HEADS, DIM], &[])?;
    graph.node(
        OpCode::CONTEXT_KV_BLOCK,
        NodeFlags::default(),
        &[partials, weights, norms, phase],
        &[key, value],
        &[],
    )?;
    graph.finish()
}

/// Zero-copy resident K/V variant: the canonical `[K rows, V rows]` signed-I8
/// archive and packed F32 params are borrowed as separate dynamic DMA views.
/// The output geometry and ring publication contract match opcode 86.
pub fn graph_i8_rows_in<A: Allocator + Clone>(
    rows: u32,
    allocator: A,
) -> Result<GraphBuf<A>, AllocationError> {
    if !(1..=ROWS).contains(&rows) {
        return Err(AxError::InvalidTensorRange.into());
    }
    let mut graph = GraphBuilder::new_in(allocator);
    let input = TensorFlags::INPUT;
    let partials = graph.tensor(DType::F32, input, &[2, rows, HIDDEN], &[])?;
    let weights = graph.tensor(DType::I8, input, &[2 * KV_WIDTH, HIDDEN], &[])?;
    let params = graph.tensor(DType::F32, input, &[I8_PARAM_WIDTH], &[])?;
    let phase = graph.tensor(DType::F32, input, &[rows, DIM], &[])?;
    let key = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[rows, KV_HEADS, DIM], &[])?;
    let value = graph.tensor(DType::F32, TensorFlags::OUTPUT, &[rows, KV_HEADS, DIM], &[])?;
    graph.node(
        OpCode::CONTEXT_KV_I8_BLOCK,
        NodeFlags::default(),
        &[partials, weights, params, phase],
        &[key, value],
        &[],
    )?;
    graph.finish()
}

pub fn validate_i8_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if node.op != OpCode::CONTEXT_KV_I8_BLOCK
        || node.flags != NodeFlags::default()
        || !node.attributes().is_empty()
        || !node.inputs().eq((0..4).map(TensorId))
        || !node.outputs().eq([TensorId(4), TensorId(5)])
    {
        return false;
    }
    let rows = graph.tensor(TensorId(0)).ok().and_then(|tensor| {
        let mut dimensions = tensor.dimensions();
        (dimensions.next() == Some(2))
            .then(|| dimensions.next())
            .flatten()
    });
    let Some(rows @ 1..=ROWS) = rows else {
        return false;
    };
    let expected = [
        (DType::F32, &[2, rows, HIDDEN][..]),
        (DType::I8, &[2 * KV_WIDTH, HIDDEN][..]),
        (DType::F32, &[I8_PARAM_WIDTH][..]),
        (DType::F32, &[rows, DIM][..]),
        (DType::F32, &[rows, KV_HEADS, DIM][..]),
        (DType::F32, &[rows, KV_HEADS, DIM][..]),
    ];
    expected.iter().enumerate().all(|(id, (dtype, shape))| {
        graph.tensor(TensorId(id as u32)).is_ok_and(|tensor| {
            tensor.dtype == *dtype && tensor.dimensions().eq(shape.iter().copied())
        })
    })
}

pub fn validate_node(graph: &Graph<'_>, node: NodeView<'_>) -> bool {
    if node.op != OpCode::CONTEXT_KV_BLOCK
        || node.flags != NodeFlags::default()
        || !node.attributes().is_empty()
        || !node.inputs().eq((0..4).map(TensorId))
        || !node.outputs().eq([TensorId(4), TensorId(5)])
    {
        return false;
    }
    let rows = graph.tensor(TensorId(0)).ok().and_then(|tensor| {
        let mut dimensions = tensor.dimensions();
        (dimensions.next() == Some(2))
            .then(|| dimensions.next())
            .flatten()
    });
    let Some(rows @ 1..=ROWS) = rows else {
        return false;
    };
    let expected = [
        (DType::F32, &[2, rows, HIDDEN][..]),
        (DType::F32, &[HIDDEN, 2 * KV_WIDTH]),
        (DType::F32, &[NORM_WIDTH]),
        (DType::F32, &[rows, DIM]),
        (DType::F32, &[rows, KV_HEADS, DIM]),
        (DType::F32, &[rows, KV_HEADS, DIM]),
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

    use alloc::vec;

    #[test]
    fn shadow_outputs_never_overlap_the_committed_window() {
        let mut ring = ContextRing::new(0);
        // Fill the logical window, then move the physical cursor close enough
        // to the end of the shadow allocation to require two output ranges.
        for first in (0..WINDOW).step_by(ROWS as usize) {
            let stage = ring.stage(first as u64, ROWS as usize).unwrap();
            ring.retire_complete(stage, ROWS as usize).unwrap();
        }
        let stage = ring.stage(WINDOW as u64, 4).unwrap();
        ring.retire_complete(stage, 4).unwrap();
        let before = ring.snapshot();
        assert_eq!((before.head, before.held), (WINDOW + 4, WINDOW));
        let stage = ring.stage(before.next, ROWS as usize).unwrap();
        assert_eq!(
            stage.segments(),
            &[
                RingSegment {
                    first_row: 0,
                    rows: 4,
                    key_offset: (WINDOW + 4) * ROW_BYTES,
                    value_offset: (SHADOW_CAPACITY + WINDOW + 4) * ROW_BYTES,
                },
                RingSegment {
                    first_row: 4,
                    rows: 4,
                    key_offset: 0,
                    value_offset: SHADOW_CAPACITY * ROW_BYTES,
                },
            ]
        );
        let mut occupied = vec![false; SHADOW_CAPACITY];
        for age in 1..=before.held {
            occupied[(before.head + SHADOW_CAPACITY - age) % SHADOW_CAPACITY] = true;
        }
        for segment in stage.segments() {
            assert!(segment.key_offset + segment.rows * ROW_BYTES <= SHADOW_CAPACITY * ROW_BYTES);
            assert!(segment.value_offset + segment.rows * ROW_BYTES <= LAYER_BYTES);
            for slot in
                segment.key_offset / ROW_BYTES..segment.key_offset / ROW_BYTES + segment.rows
            {
                assert!(!occupied[slot], "slot {slot} was committed");
            }
        }
        assert_eq!(ring.snapshot(), before, "staging cannot publish");
        assert!(matches!(
            ring.stage(before.next, 1),
            Err(RingError::Pending)
        ));

        // Only the accepted prefix becomes visible. The rejected suffix may
        // contain device writes, but remains spare and can be overwritten.
        let after = ring.retire_complete(stage, 2).unwrap();
        assert_eq!(
            (after.next, after.head, after.held),
            (before.next + 2, WINDOW + 6, WINDOW)
        );
        assert_eq!(after.generation, before.generation + 1);
        let retry = ring.stage(after.next, 3).unwrap();
        assert_eq!(retry.segments()[0].key_offset, (WINDOW + 6) * ROW_BYTES);
        assert_eq!(retry.segments()[1].rows, 1);
        ring.retire_failed(retry).unwrap();
        assert_eq!(ring.snapshot(), after, "failed device write cannot publish");
    }

    #[test]
    fn reservation_and_retirement_refuse_invalid_transitions() {
        let mut ring = ContextRing::new(41);
        for rows in [0, ROWS as usize + 1] {
            assert!(matches!(ring.stage(41, rows), Err(RingError::Rows)));
        }
        assert!(matches!(ring.stage(42, 1), Err(RingError::Position)));
        let stage = ring.stage(41, 1).unwrap();
        assert_eq!(ring.reset_to(100), Err(RingError::Pending));
        assert_eq!(ring.retire_complete(stage, 2), Err(RingError::Accepted));
        assert_eq!(ring.snapshot(), ContextRing::new(41).snapshot());
        assert!(matches!(ring.stage(41, 1), Err(RingError::Pending)));
        // A dropped or malformed completion is deliberately quarantined.
        // The owner must tear down the context rather than recycle its DMA.

        let mut ring = ContextRing::new(u64::MAX);
        assert!(matches!(ring.stage(u64::MAX, 1), Err(RingError::Overflow)));
        let mut ring = ContextRing::new(0);
        let stage = ring.stage(0, 8).unwrap();
        let original = ring.snapshot();
        assert_eq!(ring.retire_failed(stage).unwrap(), original);
        ring.reset_to(100).unwrap();
        assert_eq!(ring.snapshot().next, 100);
        assert_eq!(ring.snapshot().held, 0);
    }

    #[test]
    fn every_static_row_width_splits_at_the_shadow_boundary() {
        for rows in 1..=ROWS as usize {
            let mut ring = ContextRing::new(0);
            for first in (0..WINDOW).step_by(ROWS as usize) {
                let stage = ring.stage(first as u64, ROWS as usize).unwrap();
                ring.retire_complete(stage, ROWS as usize).unwrap();
            }
            let stage = ring.stage(WINDOW as u64, 7).unwrap();
            ring.retire_complete(stage, 7).unwrap();
            assert_eq!(ring.snapshot().head, SHADOW_CAPACITY - 1);
            let stage = ring.stage((WINDOW + 7) as u64, rows).unwrap();
            assert_eq!(stage.segments()[0].rows, 1);
            assert_eq!(stage.segments().len(), if rows == 1 { 1 } else { 2 });
            if rows > 1 {
                assert_eq!(stage.segments()[1].rows, rows - 1);
                assert_eq!(stage.segments()[1].key_offset, 0);
                assert_eq!(
                    stage.segments()[1].value_offset,
                    SHADOW_CAPACITY * ROW_BYTES
                );
            }
            let previous = ring.snapshot();
            let after = ring.retire_complete(stage, 0).unwrap();
            assert_eq!(
                (after.next, after.head, after.held),
                (previous.next, previous.head, previous.held)
            );
            let retry = ring.stage(after.next, rows).unwrap();
            ring.retire_failed(retry).unwrap();
        }
    }

    #[test]
    fn exact_six_port_versioned_graph() {
        let bytes = graph_in(Global).unwrap();
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.tensor_count(), 6);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(
            u16::from_le_bytes(bytes.as_bytes()[6..8].try_into().unwrap()),
            15
        );
        assert!(validate_node(&graph, graph.node(0).unwrap()));
    }

    #[test]
    fn each_static_row_count_has_matching_partials_phase_and_outputs() {
        let legacy = graph_in(Global).unwrap();
        for rows in 1..=ROWS {
            let bytes = graph_rows_in(rows, Global).unwrap();
            let graph = bytes.parse().unwrap();
            assert!(validate_node(&graph, graph.node(0).unwrap()), "rows={rows}");
            let shape = |id| {
                graph
                    .tensor(TensorId(id))
                    .unwrap()
                    .dimensions()
                    .collect::<alloc::vec::Vec<_>>()
            };
            assert_eq!(shape(0), [2, rows, HIDDEN]);
            assert_eq!(shape(3), [rows, DIM]);
            for id in [4, 5] {
                assert_eq!(shape(id), [rows, KV_HEADS, DIM]);
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
    fn i8_archive_variant_has_exact_six_ports_at_each_static_width() {
        for rows in 1..=ROWS {
            let graph = graph_i8_rows_in(rows, Global).unwrap();
            assert_eq!(
                u16::from_le_bytes(graph.as_bytes()[6..8].try_into().unwrap()),
                20
            );
            let parsed = graph.parse().unwrap();
            assert!(validate_i8_node(&parsed, parsed.node(0).unwrap()));
            let expected = [
                (DType::F32, alloc::vec![2, rows, HIDDEN]),
                (DType::I8, alloc::vec![512, HIDDEN]),
                (DType::F32, alloc::vec![I8_PARAM_WIDTH]),
                (DType::F32, alloc::vec![rows, DIM]),
                (DType::F32, alloc::vec![rows, KV_HEADS, DIM]),
                (DType::F32, alloc::vec![rows, KV_HEADS, DIM]),
            ];
            for (id, (dtype, shape)) in expected.into_iter().enumerate() {
                let tensor = parsed.tensor(TensorId(id as u32)).unwrap();
                assert_eq!(tensor.dtype, dtype);
                assert!(tensor.dimensions().eq(shape));
            }
        }
        for rows in [0, ROWS + 1, u32::MAX] {
            assert!(matches!(
                graph_i8_rows_in(rows, Global),
                Err(AllocationError::Format(AxError::InvalidTensorRange))
            ));
        }
    }

    #[test]
    fn mismatched_phase_rows_are_refused() {
        let mut graph = GraphBuilder::new_in(Global);
        let input = TensorFlags::INPUT;
        let partials = graph
            .tensor(DType::F32, input, &[2, 3, HIDDEN], &[])
            .unwrap();
        let weights = graph
            .tensor(DType::F32, input, &[HIDDEN, 2 * KV_WIDTH], &[])
            .unwrap();
        let norms = graph.tensor(DType::F32, input, &[NORM_WIDTH], &[]).unwrap();
        let phase = graph.tensor(DType::F32, input, &[4, DIM], &[]).unwrap();
        let key = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[3, KV_HEADS, DIM], &[])
            .unwrap();
        let value = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[3, KV_HEADS, DIM], &[])
            .unwrap();
        graph
            .node(
                OpCode::CONTEXT_KV_BLOCK,
                NodeFlags::default(),
                &[partials, weights, norms, phase],
                &[key, value],
                &[],
            )
            .unwrap();
        let bytes = graph.finish().unwrap();
        let parsed = bytes.parse().unwrap();
        assert!(!validate_node(&parsed, parsed.node(0).unwrap()));
    }

    #[test]
    fn separate_partials_are_not_the_same_operation() {
        let mut graph = GraphBuilder::new_in(Global);
        let first = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[ROWS, HIDDEN], &[])
            .unwrap();
        let weights = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[HIDDEN, 2 * KV_WIDTH], &[])
            .unwrap();
        let norms = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[NORM_WIDTH], &[])
            .unwrap();
        let phase = graph
            .tensor(DType::F32, TensorFlags::INPUT, &[ROWS, DIM], &[])
            .unwrap();
        let key = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, KV_HEADS, DIM], &[])
            .unwrap();
        let value = graph
            .tensor(DType::F32, TensorFlags::OUTPUT, &[ROWS, KV_HEADS, DIM], &[])
            .unwrap();
        graph
            .node(
                OpCode::CONTEXT_KV_BLOCK,
                NodeFlags::default(),
                &[first, weights, norms, phase],
                &[key, value],
                &[],
            )
            .unwrap();
        let graph = graph.finish().unwrap();
        let parsed = graph.parse().unwrap();
        assert!(!validate_node(&parsed, parsed.node(0).unwrap()));
    }
}
