//! The mixture-of-experts node and its attribute schema.
//!
//! Expert weights are a bound, read-only store: the graph records the
//! store's key and geometry, never its bytes, so a store can be memory-
//! mapped checkpoint tensors or whichever experts a residency cache holds.

use core::alloc::Allocator;

pub use axnn::lowering::{ExpertActivation, F32Bits, MAX_MOE_TOP_K, MoeParams, ResourceKey};

use crate::{
    AllocationError, AxError, DType, DYNAMIC_DIM, GraphBuilder, NodeFlags, OpCode, TensorFlags,
    TensorId,
};

/// Version 1 attribute record of a `MOE_EXPERTS` node.
pub const MOE_EXPERTS_ATTRIBUTE_BYTES: usize = 36;

const MOE_KIND: u8 = 9;
const NORMALIZE: u32 = 1;

const fn weights_tag(dtype: DType) -> Option<u8> {
    match dtype {
        DType::B8 => Some(0),
        DType::B16 => Some(1),
        DType::U8 => Some(2),
        _ => None,
    }
}

const fn weights_from_tag(tag: u8) -> Option<DType> {
    match tag {
        0 => Some(DType::B8),
        1 => Some(DType::B16),
        2 => Some(DType::U8),
        _ => None,
    }
}

fn read_u32(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(start..start + 4)?.try_into().ok()?,
    ))
}

/// Layout: `[1, 9, weights, activation]` (weights `0` = FP8 E4M3, `1` =
/// BF16, `2` = checkpoint NVFP4; activation `0` = squared ReLU), then LE u32 group, binding,
/// experts, top_k, hidden, intermediate, flags (bit 0: normalize) and the
/// F32 bits of `routed_scale`.
pub fn encode_moe_attributes(params: MoeParams) -> Option<[u8; MOE_EXPERTS_ATTRIBUTE_BYTES]> {
    if !params.is_valid() {
        return None;
    }
    let mut bytes = [0_u8; MOE_EXPERTS_ATTRIBUTE_BYTES];
    bytes[..4].copy_from_slice(&[
        1,
        MOE_KIND,
        weights_tag(params.weights)?,
        match params.activation {
            ExpertActivation::SquaredRelu => 0,
        },
    ]);
    for (slot, value) in bytes[4..].chunks_exact_mut(4).zip([
        params.resource.group,
        params.resource.binding,
        params.experts,
        params.top_k,
        params.hidden,
        params.intermediate,
        if params.normalize { NORMALIZE } else { 0 },
        params.routed_scale.bits(),
    ]) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    Some(bytes)
}

/// Decode only the exact version 1 record: known tags, no unknown flag
/// bits, and valid geometry and scale.
pub fn decode_moe_attributes(bytes: &[u8]) -> Option<MoeParams> {
    if bytes.len() != MOE_EXPERTS_ATTRIBUTE_BYTES || bytes[0] != 1 || bytes[1] != MOE_KIND {
        return None;
    }
    let activation = match bytes[3] {
        0 => ExpertActivation::SquaredRelu,
        _ => return None,
    };
    let flags = read_u32(bytes, 28)?;
    if flags & !NORMALIZE != 0 {
        return None;
    }
    MoeParams::new(
        ResourceKey {
            group: read_u32(bytes, 4)?,
            binding: read_u32(bytes, 8)?,
        },
        read_u32(bytes, 12)?,
        read_u32(bytes, 16)?,
        read_u32(bytes, 20)?,
        read_u32(bytes, 24)?,
        weights_from_tag(bytes[2])?,
        activation,
        flags == NORMALIZE,
        F32Bits::from_bits(read_u32(bytes, 32)?).to_f32(),
    )
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// Route F32 `hidden` `[tokens, hidden]` by `scores` `[tokens, experts]`
    /// and `correction_bias` `[experts]` to the bound store's experts.
    /// Returns the F32 `[tokens, hidden]` weighted sum.
    pub fn moe_experts(
        &mut self,
        hidden: TensorId,
        scores: TensorId,
        correction_bias: TensorId,
        tokens: u32,
        params: MoeParams,
        output_flags: TensorFlags,
    ) -> Result<TensorId, AllocationError> {
        let attributes = encode_moe_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::MOE_EXPERTS))?;
        if tokens == 0 || tokens == DYNAMIC_DIM {
            return Err(AxError::InvalidTensorRange.into());
        }
        let output = self.tensor(DType::F32, output_flags, &[tokens, params.hidden], &[])?;
        self.node(
            OpCode::MOE_EXPERTS,
            NodeFlags::READS_RESOURCE,
            &[hidden, scores, correction_bias],
            &[output],
            &attributes,
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_record_is_canonical_and_rejects_malformed_bytes() {
        let params = MoeParams::new(
            ResourceKey {
                group: 6,
                binding: 23,
            },
            128,
            6,
            2688,
            1856,
            DType::B8,
            ExpertActivation::SquaredRelu,
            true,
            2.5,
        )
        .unwrap();
        let bytes = encode_moe_attributes(params).unwrap();
        assert_eq!(bytes[..4], [1, 9, 0, 0]);
        assert_eq!(bytes[28..32], [1, 0, 0, 0]);
        assert_eq!(decode_moe_attributes(&bytes), Some(params));
        for (offset, value) in [(0, 2), (1, 8), (2, 3), (3, 1), (29, 1), (16, 0), (16, 17)] {
            let mut invalid = bytes;
            invalid[offset] = value;
            assert!(decode_moe_attributes(&invalid).is_none(), "byte {offset}");
        }
        assert!(decode_moe_attributes(&bytes[..35]).is_none());
        let bf16 = MoeParams {
            weights: DType::B16,
            normalize: false,
            ..params
        };
        let other = encode_moe_attributes(bf16).unwrap();
        assert_eq!((other[2], other[28]), (1, 0));
        assert_eq!(decode_moe_attributes(&other), Some(bf16));
        let nvfp4 = MoeParams {
            weights: DType::U8,
            ..params
        };
        let packed = encode_moe_attributes(nvfp4).unwrap();
        assert_eq!((packed[2], packed[28]), (2, 1));
        assert_eq!(decode_moe_attributes(&packed), Some(nvfp4));
    }
}
