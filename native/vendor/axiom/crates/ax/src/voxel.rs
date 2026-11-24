//! Typed voxel query construction and its portable resource attribute schema.
//!
//! Graphs store a binding key and fixed dimensions only. Brick occupancy is
//! borrowed by the execution adapter; no pointer, tensor, or constant payload
//! stands in for the resource contents.

use core::alloc::Allocator;

pub use axnn::lowering::{ResourceKey, VoxelBrickTraceParams};

use crate::{
    AllocationError, AxError, DType, DYNAMIC_DIM, GraphBuilder, NodeFlags, OpCode, TensorFlags,
    TensorId,
};

/// Version 1 attribute record for a `Brick4x4x4U64` resource query.
pub const VOXEL_BRICK_TRACE_ATTRIBUTE_BYTES: usize = 24;

/// Encode a checked resource identity and geometry as canonical LE bytes.
pub fn encode_brick_trace_attributes(
    params: VoxelBrickTraceParams,
) -> Option<[u8; VOXEL_BRICK_TRACE_ATTRIBUTE_BYTES]> {
    if !params.is_valid() {
        return None;
    }
    let mut bytes = [0_u8; VOXEL_BRICK_TRACE_ATTRIBUTE_BYTES];
    bytes[0] = 1; // schema version
    bytes[1] = 1; // Brick4x4x4U64; bytes 2..4 are reserved zero
    for (slot, value) in bytes[4..].chunks_exact_mut(4).zip([
        params.resource.group,
        params.resource.binding,
        params.dims[0],
        params.dims[1],
        params.dims[2],
    ]) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    Some(bytes)
}

/// Decode only the exact version 1 `Brick4x4x4U64` record.
///
/// Unknown versions/kinds, nonzero reserved bytes, invalid geometry, and
/// truncated or extended records are rejected without allocation.
pub fn decode_brick_trace_attributes(bytes: &[u8]) -> Option<VoxelBrickTraceParams> {
    if bytes.len() != VOXEL_BRICK_TRACE_ATTRIBUTE_BYTES || bytes[..4] != [1, 1, 0, 0] {
        return None;
    }
    let read = |start| {
        Some(u32::from_le_bytes(
            bytes.get(start..start + 4)?.try_into().ok()?,
        ))
    };
    VoxelBrickTraceParams::new(
        ResourceKey {
            group: read(4)?,
            binding: read(8)?,
        },
        [read(12)?, read(16)?, read(20)?],
    )
}

/// Separate graph results in query ABI order. Misses are `(+infinity, -1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelHit {
    /// F32 `[H, W]` ray distance.
    pub t: TensorId,
    /// Exact I32 `[H, W]` x-major cell ID, with -1 for a miss.
    pub cell_id: TensorId,
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// Pack three broadcast F32 component planes into `[H, W, 3]`.
    ///
    /// `dimensions` is `[H, W]`, even when every component is uniform. Input
    /// types and broadcast geometry are checked by the semantic bridge.
    pub fn interleave3_f32(
        &mut self,
        components: [TensorId; 3],
        dimensions: [u32; 2],
    ) -> Result<TensorId, AllocationError> {
        validate_dimensions(dimensions)?;
        let output = self.tensor(
            DType::F32,
            TensorFlags::default(),
            &[dimensions[0], dimensions[1], 3],
            &[],
        )?;
        self.node(
            OpCode::INTERLEAVE3_F32,
            NodeFlags::default(),
            &components,
            &[output],
            &[],
        )?;
        Ok(output)
    }

    /// Query a caller-bound brick resource with packed origins/directions and
    /// a one-element F32 limit. Result dimensions are `[H, W]`.
    ///
    /// This emits a read-resource node, never occupancy tensor uploads. Input
    /// descriptors and both authored results are validated by the semantic
    /// bridge before a catalog can admit an implementation.
    pub fn voxel_brick_trace(
        &mut self,
        origins: TensorId,
        directions: TensorId,
        limit: TensorId,
        params: VoxelBrickTraceParams,
        dimensions: [u32; 2],
    ) -> Result<VoxelHit, AllocationError> {
        validate_dimensions(dimensions)?;
        let attributes = encode_brick_trace_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::VOXEL_BRICK_TRACE))?;
        let t = self.tensor(DType::F32, TensorFlags::default(), &dimensions, &[])?;
        let cell_id = self.tensor(DType::I32, TensorFlags::default(), &dimensions, &[])?;
        self.node(
            OpCode::VOXEL_BRICK_TRACE,
            NodeFlags::READS_RESOURCE,
            &[origins, directions, limit],
            &[t, cell_id],
            &attributes,
        )?;
        Ok(VoxelHit { t, cell_id })
    }
}

fn validate_dimensions(dimensions: [u32; 2]) -> Result<(), AxError> {
    if dimensions.contains(&DYNAMIC_DIM) {
        return Err(AxError::InvalidTensorRange);
    }
    let height = usize::try_from(dimensions[0]).map_err(|_| AxError::CountOverflow)?;
    let width = usize::try_from(dimensions[1]).map_err(|_| AxError::CountOverflow)?;
    height
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(3 * core::mem::size_of::<f32>()))
        .ok_or(AxError::CountOverflow)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::alloc::Global;

    use super::*;

    fn params() -> VoxelBrickTraceParams {
        VoxelBrickTraceParams::new(
            ResourceKey {
                group: 0x1234_5678,
                binding: 0x90ab_cdef,
            },
            [32, 16, 8],
        )
        .unwrap()
    }

    #[test]
    fn resource_record_roundtrips_exact_little_endian_fields() {
        let bytes = encode_brick_trace_attributes(params()).unwrap();
        assert_eq!(
            bytes,
            [
                1, 1, 0, 0, 0x78, 0x56, 0x34, 0x12, 0xef, 0xcd, 0xab, 0x90, 32, 0, 0, 0, 16, 0, 0,
                0, 8, 0, 0, 0,
            ]
        );
        assert_eq!(decode_brick_trace_attributes(&bytes), Some(params()));
    }

    #[test]
    fn resource_record_rejects_unknown_or_noncanonical_schemas() {
        let valid = encode_brick_trace_attributes(params()).unwrap();
        for len in 0..valid.len() {
            assert!(decode_brick_trace_attributes(&valid[..len]).is_none());
        }
        let mut extended = [0_u8; VOXEL_BRICK_TRACE_ATTRIBUTE_BYTES + 1];
        extended[..valid.len()].copy_from_slice(&valid);
        assert!(decode_brick_trace_attributes(&extended).is_none());
        for (offset, value) in [
            (0, 0),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 1),
            (3, 1),
            (12, 0),
            (16, 3),
        ] {
            let mut invalid = valid;
            invalid[offset] = value;
            assert!(
                decode_brick_trace_attributes(&invalid).is_none(),
                "offset={offset} value={value}"
            );
        }
        let invalid = VoxelBrickTraceParams {
            dims: [134_217_728, 4, 4],
            ..params()
        };
        assert!(encode_brick_trace_attributes(invalid).is_none());
        let mut bytes = valid;
        for (slot, value) in bytes[12..].chunks_exact_mut(4).zip(invalid.dims) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        assert!(decode_brick_trace_attributes(&bytes).is_none());
    }

    #[test]
    fn builders_emit_typed_results_and_only_a_binding_record() {
        let mut builder = GraphBuilder::new_in(Global);
        let uniform = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[1, 1], &[])
            .unwrap();
        let packed = builder.interleave3_f32([uniform; 3], [2, 3]).unwrap();
        let hit = builder
            .voxel_brick_trace(packed, packed, uniform, params(), [2, 3])
            .unwrap();
        assert_ne!(hit.t, hit.cell_id);
        let bytes = builder.finish().unwrap();
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.tensor_count(), 4);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(
            graph
                .tensor(packed)
                .unwrap()
                .dimensions()
                .collect::<alloc::vec::Vec<_>>(),
            [2, 3, 3]
        );
        assert_eq!(graph.tensor(hit.t).unwrap().dtype, DType::F32);
        assert_eq!(graph.tensor(hit.cell_id).unwrap().dtype, DType::I32);
        let query = graph.node(1).unwrap();
        assert_eq!(query.op, OpCode::VOXEL_BRICK_TRACE);
        assert_eq!(query.flags, NodeFlags::READS_RESOURCE);
        assert!(query.inputs().eq([packed, packed, uniform]));
        assert!(query.outputs().eq([hit.t, hit.cell_id]));
        assert_eq!(
            decode_brick_trace_attributes(query.attributes()),
            Some(params())
        );
        assert!(!query.cacheable());
        for index in 0..graph.tensor_count() {
            assert!(
                graph
                    .tensor(TensorId(index as u32))
                    .unwrap()
                    .constant()
                    .is_empty()
            );
        }
    }

    #[test]
    fn builders_reject_dynamic_or_overflowing_result_dimensions() {
        for dimensions in [[DYNAMIC_DIM, 1], [1, DYNAMIC_DIM], [DYNAMIC_DIM - 1; 2]] {
            let mut builder = GraphBuilder::new_in(Global);
            assert!(
                builder
                    .interleave3_f32([TensorId(0); 3], dimensions)
                    .is_err()
            );
            assert!(
                builder
                    .voxel_brick_trace(TensorId(0), TensorId(0), TensorId(0), params(), dimensions,)
                    .is_err()
            );
            assert_eq!(builder.finish().unwrap().parse().unwrap().tensor_count(), 0);
        }
    }
}
