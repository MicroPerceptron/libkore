//! Bounded, immutable byte-resource loads. Only identity and length are
//! serialized; each invocation supplies a fresh borrowed payload.

use crate::{
    AllocationError, AxError, DType, DYNAMIC_DIM, GraphBuilder, NodeFlags, OpCode, TensorFlags,
    TensorId,
};
pub use axnn::lowering::{ResourceKey, U8BufferParams};
use core::alloc::Allocator;

pub const BUFFER_LOAD_U8_ATTRIBUTE_BYTES: usize = 16;

pub fn encode_u8_buffer_attributes(
    params: U8BufferParams,
) -> Option<[u8; BUFFER_LOAD_U8_ATTRIBUTE_BYTES]> {
    if !params.is_valid() {
        return None;
    }
    let mut bytes = [0; BUFFER_LOAD_U8_ATTRIBUTE_BYTES];
    bytes[..4].copy_from_slice(&[1, 3, 0, 0]);
    for (slot, value) in bytes[4..].chunks_exact_mut(4).zip([
        params.resource.group,
        params.resource.binding,
        params.elements,
    ]) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    Some(bytes)
}

pub fn decode_u8_buffer_attributes(bytes: &[u8]) -> Option<U8BufferParams> {
    if bytes.len() != BUFFER_LOAD_U8_ATTRIBUTE_BYTES || bytes[..4] != [1, 3, 0, 0] {
        return None;
    }
    let read = |start| {
        Some(u32::from_le_bytes(
            bytes.get(start..start + 4)?.try_into().ok()?,
        ))
    };
    U8BufferParams::new(
        ResourceKey {
            group: read(4)?,
            binding: read(8)?,
        },
        read(12)?,
    )
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// Load a byte and zero-extend to I32, returning zero for negative or
    /// out-of-range indices. Indices broadcast into the explicit H×W field.
    pub fn buffer_load_u8(
        &mut self,
        index: TensorId,
        params: U8BufferParams,
        dimensions: [u32; 2],
    ) -> Result<TensorId, AllocationError> {
        if dimensions.contains(&DYNAMIC_DIM) {
            return Err(AxError::InvalidTensorRange.into());
        }
        (dimensions[0] as usize)
            .checked_mul(dimensions[1] as usize)
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(AxError::CountOverflow)?;
        let attributes = encode_u8_buffer_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::BUFFER_LOAD_U8))?;
        let output = self.tensor(DType::I32, TensorFlags::default(), &dimensions, &[])?;
        self.node(
            OpCode::BUFFER_LOAD_U8,
            NodeFlags::READS_RESOURCE,
            &[index],
            &[output],
            &attributes,
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    #[test]
    fn byte_buffer_schema_is_canonical_and_bounded() {
        let params = U8BufferParams::new(
            ResourceKey {
                group: 0x12345678,
                binding: 9,
            },
            257,
        )
        .unwrap();
        let bytes = encode_u8_buffer_attributes(params).unwrap();
        assert_eq!(
            bytes,
            [1, 3, 0, 0, 0x78, 0x56, 0x34, 0x12, 9, 0, 0, 0, 1, 1, 0, 0]
        );
        assert_eq!(decode_u8_buffer_attributes(&bytes), Some(params));
        for length in 0..16 {
            assert!(decode_u8_buffer_attributes(&bytes[..length]).is_none());
        }
        for offset in 0..4 {
            let mut invalid = bytes;
            invalid[offset] ^= 1;
            assert!(decode_u8_buffer_attributes(&invalid).is_none());
        }
        let mut extended = bytes.to_vec();
        extended.push(0);
        assert!(decode_u8_buffer_attributes(&extended).is_none());
        assert!(U8BufferParams::new(params.resource, u32::MAX).is_none());
        assert!(U8BufferParams::new(params.resource, 0).is_some());
    }

    #[test]
    fn byte_buffer_builder_preserves_typed_binding_without_payload() {
        let mut builder = GraphBuilder::new_in(Global);
        let index = builder
            .tensor(DType::I32, TensorFlags::INPUT, &[], &[])
            .unwrap();
        let params = U8BufferParams::new(
            ResourceKey {
                group: 1,
                binding: 2,
            },
            19,
        )
        .unwrap();
        let output = builder.buffer_load_u8(index, params, [2, 3]).unwrap();
        let bytes = builder.finish().unwrap();
        assert_eq!(&bytes.as_bytes()[4..8], &[1, 0, 1, 0]);
        let graph = bytes.parse().unwrap();
        assert_eq!(graph.tensor_count(), 2);
        let node = graph.node(0).unwrap();
        assert_eq!(decode_u8_buffer_attributes(node.attributes()), Some(params));
        assert_eq!(graph.tensor(output).unwrap().dtype, DType::I32);
        assert!(graph.tensor(output).unwrap().dimensions().eq([2, 3]));
        #[cfg(feature = "kernel-lowering")]
        assert!(crate::node_kernel_request(&graph, node).is_some());
    }
}
