//! Typed RGB volume sampling and its portable resource attribute schema.
//!
//! Graphs contain only a binding key, dimensions, and storage encoding. The
//! execution adapter borrows validated immutable storage for each invocation;
//! no resource pointer or volume payload is serialized in the graph.

use core::alloc::Allocator;

pub use axnn::lowering::{ResourceKey, VolumeRgbParams, VolumeRgbStorage};

use crate::{
    AllocationError, AxError, DType, DYNAMIC_DIM, GraphBuilder, NodeFlags, OpCode, TensorFlags,
    TensorId,
};

/// Version 1 attribute record for a tightly packed, x-major RGB volume.
pub const VOLUME_TRILINEAR_RGB_ATTRIBUTE_BYTES: usize = 24;

/// Encode a checked resource identity, geometry, and storage encoding.
pub fn encode_volume_rgb_attributes(
    params: VolumeRgbParams,
) -> Option<[u8; VOLUME_TRILINEAR_RGB_ATTRIBUTE_BYTES]> {
    if !params.is_valid() {
        return None;
    }
    let mut bytes = [0_u8; VOLUME_TRILINEAR_RGB_ATTRIBUTE_BYTES];
    bytes[0] = 1; // schema version
    bytes[1] = 2; // x-major interleaved RGB volume
    bytes[2] = match params.storage {
        VolumeRgbStorage::F32 => 0,
        VolumeRgbStorage::Uf16E5M11 => 1,
        VolumeRgbStorage::Uf8E5M3 => 2,
    }; // byte 3 is reserved zero
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

/// Decode only the exact version 1 RGB volume record.
///
/// Unknown versions, kinds, storage tags, nonzero reserved bytes, invalid
/// lengths or geometry, and truncated/extended records are rejected.
pub fn decode_volume_rgb_attributes(bytes: &[u8]) -> Option<VolumeRgbParams> {
    if bytes.len() != VOLUME_TRILINEAR_RGB_ATTRIBUTE_BYTES
        || bytes[0] != 1
        || bytes[1] != 2
        || bytes[3] != 0
    {
        return None;
    }
    let storage = match bytes[2] {
        0 => VolumeRgbStorage::F32,
        1 => VolumeRgbStorage::Uf16E5M11,
        2 => VolumeRgbStorage::Uf8E5M3,
        _ => return None,
    };
    let read = |start| {
        Some(u32::from_le_bytes(
            bytes.get(start..start + 4)?.try_into().ok()?,
        ))
    };
    VolumeRgbParams::new(
        ResourceKey {
            group: read(4)?,
            binding: read(8)?,
        },
        [read(12)?, read(16)?, read(20)?],
        storage,
    )
}

/// Three distinct F32 `[H, W]` graph results in red/green/blue ABI order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RgbSample {
    pub r: TensorId,
    pub g: TensorId,
    pub b: TensorId,
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// Sample a bound RGB volume using three broadcast F32 index coordinates.
    ///
    /// `dimensions` is the explicit `[H, W]` output field, including when all
    /// coordinates are scalar. Coordinates clamp to the volume edges; eight
    /// corners are promoted and accumulated in F32 in z/y/x order. See
    /// `KernelRequest::volume_trilinear_rgb` for the precise numeric contract.
    /// The semantic bridge validates all input and output descriptors.
    pub fn volume_trilinear_rgb(
        &mut self,
        coordinates: [TensorId; 3],
        params: VolumeRgbParams,
        dimensions: [u32; 2],
    ) -> Result<RgbSample, AllocationError> {
        validate_dimensions(dimensions)?;
        let attributes = encode_volume_rgb_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::VOLUME_TRILINEAR_RGB))?;
        let r = self.tensor(DType::F32, TensorFlags::default(), &dimensions, &[])?;
        let g = self.tensor(DType::F32, TensorFlags::default(), &dimensions, &[])?;
        let b = self.tensor(DType::F32, TensorFlags::default(), &dimensions, &[])?;
        self.node(
            OpCode::VOLUME_TRILINEAR_RGB,
            NodeFlags::READS_RESOURCE,
            &coordinates,
            &[r, g, b],
            &attributes,
        )?;
        Ok(RgbSample { r, g, b })
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
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or(AxError::CountOverflow)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::alloc::Global;

    use super::*;

    fn params(storage: VolumeRgbStorage) -> VolumeRgbParams {
        VolumeRgbParams::new(
            ResourceKey {
                group: 0x1234_5678,
                binding: 0x90ab_cdef,
            },
            [24, 12, 24],
            storage,
        )
        .unwrap()
    }

    #[test]
    fn volume_attributes_have_exact_little_endian_storage_and_binding_layout() {
        for (storage, tag) in [
            (VolumeRgbStorage::F32, 0),
            (VolumeRgbStorage::Uf16E5M11, 1),
            (VolumeRgbStorage::Uf8E5M3, 2),
        ] {
            let expected = [
                1, 2, tag, 0, 0x78, 0x56, 0x34, 0x12, 0xef, 0xcd, 0xab, 0x90, 24, 0, 0, 0, 12, 0,
                0, 0, 24, 0, 0, 0,
            ];
            assert_eq!(
                encode_volume_rgb_attributes(params(storage)),
                Some(expected)
            );
            assert_eq!(
                decode_volume_rgb_attributes(&expected),
                Some(params(storage))
            );
        }
    }

    #[test]
    fn volume_attributes_reject_other_schemas_and_unrepresentable_lengths() {
        let valid = encode_volume_rgb_attributes(params(VolumeRgbStorage::Uf16E5M11)).unwrap();
        for length in 0..valid.len() {
            assert!(decode_volume_rgb_attributes(&valid[..length]).is_none());
        }
        let mut extended = [0_u8; VOLUME_TRILINEAR_RGB_ATTRIBUTE_BYTES + 1];
        extended[..valid.len()].copy_from_slice(&valid);
        assert!(decode_volume_rgb_attributes(&extended).is_none());
        for (offset, value) in [(0, 0), (0, 2), (1, 1), (1, 3), (2, 3), (3, 1)] {
            let mut invalid = valid;
            invalid[offset] = value;
            assert!(decode_volume_rgb_attributes(&invalid).is_none());
        }
        for dims in [
            [0, 1, 1],
            [1, VolumeRgbParams::MAX_DIMENSION + 1, 1],
            [VolumeRgbParams::MAX_DIMENSION; 3],
            [u32::MAX; 3],
        ] {
            let invalid = VolumeRgbParams {
                dims,
                ..params(VolumeRgbStorage::F32)
            };
            assert!(encode_volume_rgb_attributes(invalid).is_none());
            let mut bytes = valid;
            for (slot, dim) in bytes[12..].chunks_exact_mut(4).zip(dims) {
                slot.copy_from_slice(&dim.to_le_bytes());
            }
            assert!(decode_volume_rgb_attributes(&bytes).is_none());
        }
        let brick = crate::voxel::encode_brick_trace_attributes(
            crate::voxel::VoxelBrickTraceParams::new(
                params(VolumeRgbStorage::F32).resource,
                [4; 3],
            )
            .unwrap(),
        )
        .unwrap();
        assert!(decode_volume_rgb_attributes(&brick).is_none());
        assert!(crate::voxel::decode_brick_trace_attributes(&valid).is_none());
    }

    #[test]
    fn volume_builder_keeps_resource_contents_out_of_graph_tensor_storage() {
        let mut builder = GraphBuilder::new_in(Global);
        let coordinate = builder
            .tensor(DType::F32, TensorFlags::INPUT, &[], &[])
            .unwrap();
        let parameters = params(VolumeRgbStorage::Uf16E5M11);
        let sample = builder
            .volume_trilinear_rgb([coordinate; 3], parameters, [2, 3])
            .unwrap();
        let bytes = builder.finish().unwrap();
        assert_eq!(&bytes.as_bytes()[4..8], &[1, 0, 1, 0]);
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.tensor_count(), 4);
        assert_eq!(graph.node_count(), 1);
        let node = graph.node(0).unwrap();
        assert_eq!(node.op, OpCode::VOLUME_TRILINEAR_RGB);
        assert_eq!(node.flags, NodeFlags::READS_RESOURCE);
        assert!(node.inputs().eq([coordinate; 3]));
        assert!(node.outputs().eq([sample.r, sample.g, sample.b]));
        assert_eq!(
            decode_volume_rgb_attributes(node.attributes()),
            Some(parameters)
        );
        assert!(!node.cacheable());
        for id in [sample.r, sample.g, sample.b] {
            let tensor = graph.tensor(id).unwrap();
            assert_eq!(tensor.dtype, DType::F32);
            assert!(tensor.dimensions().eq([2, 3]));
            assert!(tensor.constant().is_empty());
        }
    }

    #[test]
    fn volume_builder_rejects_invalid_geometry_before_adding_results() {
        for dimensions in [[DYNAMIC_DIM, 1], [1, DYNAMIC_DIM], [DYNAMIC_DIM - 1; 2]] {
            let mut builder = GraphBuilder::new_in(Global);
            assert!(
                builder
                    .volume_trilinear_rgb(
                        [TensorId(0); 3],
                        params(VolumeRgbStorage::F32),
                        dimensions
                    )
                    .is_err()
            );
            assert_eq!(builder.finish().unwrap().parse().unwrap().tensor_count(), 0);
        }
        let mut builder = GraphBuilder::new_in(Global);
        let invalid = VolumeRgbParams {
            dims: [0; 3],
            ..params(VolumeRgbStorage::F32)
        };
        assert!(
            builder
                .volume_trilinear_rgb([TensorId(0); 3], invalid, [2, 3])
                .is_err()
        );
        assert_eq!(builder.finish().unwrap().parse().unwrap().tensor_count(), 0);
    }
}
