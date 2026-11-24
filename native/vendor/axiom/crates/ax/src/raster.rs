//! Raster nodes: the operator family a compositor's retained scene lowers
//! to, one node per run of same-kind scene nodes in paint order.
//!
//! A raster node paints a bound packed-XRGB target from its rows of the
//! frame program's parameter table (an I32 `[rows, RASTER_ROW_WORDS]`
//! input, see [`RasterRow`](axnn::RasterRow)), in order, and the band's
//! dirty tile columns (a U8 input); its tensor result is a one-word token.
//! The rows are what changes per frame — rect, radii, colour, opacity,
//! source slot, tile mask — so one program per screen, prepared once,
//! serves every frame; the node's own attributes are static: which rows it
//! paints, which target, and where its surfaces, atlas and glyph table are
//! bound.  A run of consecutive same-kind rows is one node, so a scene of
//! hundreds of slots is a program of tens of nodes and a band's run pays
//! the executor's per-node walk tens of times, not hundreds.  Nodes carry `ORDERED | READS_RESOURCE | WRITES_RESOURCE`: the
//! scheduler keeps their order (painter's algorithm) and never memoizes
//! them.  A portable executor runs them on its raster kernels; an
//! accelerator lowers each to its native blit, fill or blend.

use crate::{
    AllocationError, AxError, DType, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId,
};
pub use axnn::lowering::{RasterNodeParams, RasterOp, ResourceKey};
use core::alloc::Allocator;

pub const RASTER_ATTRIBUTE_BYTES: usize = 44;

/// The node flags every raster node carries.
pub const RASTER_NODE_FLAGS: NodeFlags =
    NodeFlags(NodeFlags::ORDERED.0 | NodeFlags::READS_RESOURCE.0 | NodeFlags::WRITES_RESOURCE.0);

pub const fn raster_opcode(op: RasterOp) -> OpCode {
    match op {
        RasterOp::Fill => OpCode::RASTER_FILL,
        RasterOp::Ring => OpCode::RASTER_RING,
        RasterOp::Shadow => OpCode::RASTER_SHADOW,
        RasterOp::SurfaceBlit => OpCode::SURFACE_BLIT,
        RasterOp::SurfaceSample => OpCode::SURFACE_SAMPLE,
        RasterOp::Glyphs => OpCode::RASTER_GLYPHS,
    }
}

pub const fn raster_op(op: OpCode) -> Option<RasterOp> {
    match op.0 {
        60 => Some(RasterOp::Fill),
        61 => Some(RasterOp::Ring),
        62 => Some(RasterOp::Shadow),
        63 => Some(RasterOp::SurfaceBlit),
        64 => Some(RasterOp::SurfaceSample),
        65 => Some(RasterOp::Glyphs),
        _ => None,
    }
}

pub fn encode_raster_attributes(params: RasterNodeParams) -> Option<[u8; RASTER_ATTRIBUTE_BYTES]> {
    if !params.is_valid() {
        return None;
    }
    let mut bytes = [0; RASTER_ATTRIBUTE_BYTES];
    bytes[..4].copy_from_slice(&[1, 4, 0, 0]);
    for (slot, value) in bytes[4..].chunks_exact_mut(4).zip([
        params.row,
        params.rows,
        params.target.group,
        params.target.binding,
        params.surfaces,
        params.atlas.group,
        params.atlas.binding,
        params.glyphs.group,
        params.glyphs.binding,
        params.tile_width,
    ]) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    Some(bytes)
}

pub fn decode_raster_attributes(bytes: &[u8]) -> Option<RasterNodeParams> {
    if bytes.len() != RASTER_ATTRIBUTE_BYTES || bytes[..4] != [1, 4, 0, 0] {
        return None;
    }
    let read = |start| {
        Some(u32::from_le_bytes(
            bytes.get(start..start + 4)?.try_into().ok()?,
        ))
    };
    RasterNodeParams::new(
        read(4)?,
        read(8)?,
        ResourceKey {
            group: read(12)?,
            binding: read(16)?,
        },
        read(20)?,
        ResourceKey {
            group: read(24)?,
            binding: read(28)?,
        },
        ResourceKey {
            group: read(32)?,
            binding: read(36)?,
        },
        read(40)?,
    )
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// Add one raster node painting rows `node.row .. node.row + node.rows`
    /// of `params`, in order, over the band's `dirty` columns onto
    /// `node.target`; returns its token tensor,
    /// created with `token_flags` (mark the program's last token `OUTPUT`).
    pub fn raster(
        &mut self,
        op: RasterOp,
        params: TensorId,
        dirty: TensorId,
        node: RasterNodeParams,
        token_flags: TensorFlags,
    ) -> Result<TensorId, AllocationError> {
        let attributes = encode_raster_attributes(node)
            .ok_or(AxError::InvalidNodeAttributes(raster_opcode(op)))?;
        let token = self.tensor(DType::I32, token_flags, &[1], &[])?;
        self.node(
            raster_opcode(op),
            RASTER_NODE_FLAGS,
            &[params, dirty],
            &[token],
            &attributes,
        )?;
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;
    use axnn::RASTER_ROW_WORDS;

    fn params() -> RasterNodeParams {
        RasterNodeParams::new(
            7,
            2,
            ResourceKey {
                group: 2,
                binding: 0,
            },
            3,
            ResourceKey {
                group: 4,
                binding: 0,
            },
            ResourceKey {
                group: 4,
                binding: 1,
            },
            64,
        )
        .unwrap()
    }

    #[test]
    fn raster_schema_is_canonical_and_bounded() {
        let bytes = encode_raster_attributes(params()).unwrap();
        assert_eq!(&bytes[..12], &[1, 4, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(decode_raster_attributes(&bytes), Some(params()));
        assert_eq!(decode_raster_attributes(&bytes[..40]), None);
        let mut wrong = bytes;
        wrong[1] = 3;
        assert_eq!(decode_raster_attributes(&wrong), None);
        let (target, atlas, glyphs) = (params().target, params().atlas, params().glyphs);
        let new =
            |row, rows, tile| RasterNodeParams::new(row, rows, target, 3, atlas, glyphs, tile);
        assert!(new(RasterNodeParams::MAX_ROWS, 1, 64).is_none());
        assert!(
            new(RasterNodeParams::MAX_ROWS - 1, 2, 64).is_none(),
            "the range is bounded"
        );
        assert!(new(RasterNodeParams::MAX_ROWS - 1, 1, 64).is_some());
        assert!(new(0, 0, 64).is_none(), "at least one row");
        assert!(new(0, 1, 0).is_none());
        for op in [
            RasterOp::Fill,
            RasterOp::Ring,
            RasterOp::Shadow,
            RasterOp::SurfaceBlit,
            RasterOp::SurfaceSample,
            RasterOp::Glyphs,
        ] {
            assert_eq!(raster_op(raster_opcode(op)), Some(op));
        }
        assert_eq!(raster_op(OpCode::ADD), None);
    }

    #[test]
    #[cfg(feature = "kernel-lowering")]
    fn raster_nodes_serialize_ordered_effects_and_lower_to_typed_requests() {
        let mut graph = GraphBuilder::new_in(Global);
        // Ten rows: `params()` paints rows 7 and 8, the blit row 1.
        let rows = graph
            .tensor(
                DType::I32,
                TensorFlags::INPUT,
                &[10, RASTER_ROW_WORDS as u32],
                &[],
            )
            .unwrap();
        let dirty = graph
            .tensor(DType::U8, TensorFlags::INPUT, &[30], &[])
            .unwrap();
        let first = graph
            .raster(
                RasterOp::Fill,
                rows,
                dirty,
                params(),
                TensorFlags::default(),
            )
            .unwrap();
        let last = graph
            .raster(
                RasterOp::SurfaceBlit,
                rows,
                dirty,
                RasterNodeParams { row: 1, ..params() },
                TensorFlags::OUTPUT,
            )
            .unwrap();
        assert_ne!(first, last);
        let bytes = graph.finish().unwrap();
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.node_count(), 2);
        for index in 0..2 {
            let node = graph.node(index).unwrap();
            assert_eq!(node.flags, RASTER_NODE_FLAGS);
            assert!(!node.cacheable());
            let request = crate::node_kernel_request(&graph, node).unwrap();
            assert!(matches!(
                request.semantic,
                axnn::lowering::SemanticOp::Raster(_)
            ));
            assert!(
                request
                    .effects()
                    .contains(axnn::lowering::KernelEffects::WRITES_RESOURCE)
            );
            assert_eq!(request.result_count(), 1);
        }
    }
}
