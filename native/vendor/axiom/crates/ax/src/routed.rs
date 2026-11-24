//! Mixture-of-experts over an expert table of graph weights (1.23): each
//! token's route from its router logits, then the routed NVFP4 SwiGLU and
//! down projection.
//!
//! Unlike `MOE_EXPERTS` (`crate::moe`), whose experts are a bound store a
//! residency cache may page, these nodes are pure: the experts are ordinary
//! weight tensors, one table a projection (`[experts, record bytes]` U8,
//! one [`ExpertRecord`] an expert, the shared experts after the routed
//! ones), and the route a node of its own between them. A device holding
//! every expert resident then routes and computes without the host:
//!
//! ```text
//! routes = MOE_ROUTE(router(x))            // I32 [tokens, slots, 2]
//! h      = MOE_GATE_UP_NVFP4(x, gate_up, routes)
//! y      = MOE_DOWN_NVFP4(h, down, routes)
//! ```

use core::alloc::Allocator;

pub use axnn::lowering::{ExpertRecord, MAX_MOE_SHARED, MoeRouteParams, RoutedExpertsParams};

use crate::{AllocationError, AxError, DType, DYNAMIC_DIM, GraphBuilder, NodeFlags, OpCode, TensorFlags, TensorId};

/// Version 1 attribute record of a `MOE_ROUTE` node.
pub const MOE_ROUTE_ATTRIBUTE_BYTES: usize = 16;
/// Version 1 attribute record of a `MOE_GATE_UP_NVFP4` or `MOE_DOWN_NVFP4`
/// node.
pub const ROUTED_EXPERTS_ATTRIBUTE_BYTES: usize = 20;

const ROUTE_KIND: u8 = 10;
const ROUTED_KIND: u8 = 11;
/// Byte 2 of a routed record: the expert's gate activation. 0 is SiLU, the
/// only one defined.
const SILU: u8 = 0;

fn read_u32(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(start..start + 4)?.try_into().ok()?,
    ))
}

/// Header `[1, kind, tag, 0]`, then `fields` as LE u32.
fn encode<const N: usize>(kind: u8, tag: u8, fields: &[u32]) -> [u8; N] {
    let mut bytes = [0_u8; N];
    bytes[..4].copy_from_slice(&[1, kind, tag, 0]);
    for (slot, value) in bytes[4..].chunks_exact_mut(4).zip(fields) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Layout: `[1, 10, 0, 0]`, then LE u32 experts, top_k and shared.
pub fn encode_route_attributes(params: MoeRouteParams) -> Option<[u8; MOE_ROUTE_ATTRIBUTE_BYTES]> {
    params.is_valid().then(|| {
        encode(
            ROUTE_KIND,
            0,
            &[params.experts, params.top_k, params.shared],
        )
    })
}

/// Decode only the exact version 1 record with valid routing.
pub fn decode_route_attributes(bytes: &[u8]) -> Option<MoeRouteParams> {
    if bytes.len() != MOE_ROUTE_ATTRIBUTE_BYTES || bytes[..4] != [1, ROUTE_KIND, 0, 0] {
        return None;
    }
    MoeRouteParams::new(read_u32(bytes, 4)?, read_u32(bytes, 8)?, read_u32(bytes, 12)?)
}

/// Layout: `[1, 11, activation, 0]` (activation 0: SiLU), then LE u32
/// experts, slots, hidden and intermediate.
pub fn encode_routed_attributes(
    params: RoutedExpertsParams,
) -> Option<[u8; ROUTED_EXPERTS_ATTRIBUTE_BYTES]> {
    params.is_valid().then(|| {
        encode(
            ROUTED_KIND,
            SILU,
            &[
                params.experts,
                params.slots,
                params.hidden,
                params.intermediate,
            ],
        )
    })
}

/// Decode only the exact version 1 record with a valid table geometry.
pub fn decode_routed_attributes(bytes: &[u8]) -> Option<RoutedExpertsParams> {
    if bytes.len() != ROUTED_EXPERTS_ATTRIBUTE_BYTES || bytes[..4] != [1, ROUTED_KIND, SILU, 0] {
        return None;
    }
    RoutedExpertsParams::new(
        read_u32(bytes, 4)?,
        read_u32(bytes, 8)?,
        read_u32(bytes, 12)?,
        read_u32(bytes, 16)?,
    )
}

fn concrete(tokens: u32) -> Result<u32, AllocationError> {
    if tokens == 0 || tokens == DYNAMIC_DIM {
        return Err(AxError::InvalidTensorRange.into());
    }
    Ok(tokens)
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// Route `tokens` rows of F32 router `logits` `[tokens, experts +
    /// shared]`: I32 `[tokens, top_k + shared, 2]` `(expert, weight bits)`
    /// slots ([`MoeRouteParams`]).
    pub fn moe_route(
        &mut self,
        logits: TensorId,
        tokens: u32,
        params: MoeRouteParams,
        output_flags: TensorFlags,
    ) -> Result<TensorId, AllocationError> {
        let attributes = encode_route_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::MOE_ROUTE))?;
        let output = self.tensor(
            DType::I32,
            output_flags,
            &[concrete(tokens)?, params.slots(), 2],
            &[],
        )?;
        self.node(
            OpCode::MOE_ROUTE,
            NodeFlags::default(),
            &[logits],
            &[output],
            &attributes,
        )?;
        Ok(output)
    }

    /// The routed SwiGLU of F32 `x` `[tokens, hidden]` through the gate/up
    /// table `experts` (U8 `[experts, gate_up_record bytes]`) its `routes`
    /// pick: F32 `[tokens, slots, intermediate]`.
    pub fn moe_gate_up_nvfp4(
        &mut self,
        x: TensorId,
        experts: TensorId,
        routes: TensorId,
        tokens: u32,
        params: RoutedExpertsParams,
        output_flags: TensorFlags,
    ) -> Result<TensorId, AllocationError> {
        let attributes = encode_routed_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::MOE_GATE_UP_NVFP4))?;
        let output = self.tensor(
            DType::F32,
            output_flags,
            &[concrete(tokens)?, params.slots, params.intermediate],
            &[],
        )?;
        self.node(
            OpCode::MOE_GATE_UP_NVFP4,
            NodeFlags::default(),
            &[x, experts, routes],
            &[output],
            &attributes,
        )?;
        Ok(output)
    }

    /// The routed down projection of F32 `h` `[tokens, slots,
    /// intermediate]` through the down table `experts` (U8 `[experts,
    /// down_record bytes]`), each slot weighted by its route and summed in
    /// slot order: F32 `[tokens, hidden]`.
    pub fn moe_down_nvfp4(
        &mut self,
        h: TensorId,
        experts: TensorId,
        routes: TensorId,
        tokens: u32,
        params: RoutedExpertsParams,
        output_flags: TensorFlags,
    ) -> Result<TensorId, AllocationError> {
        let attributes = encode_routed_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::MOE_DOWN_NVFP4))?;
        let output = self.tensor(
            DType::F32,
            output_flags,
            &[concrete(tokens)?, params.hidden],
            &[],
        )?;
        self.node(
            OpCode::MOE_DOWN_NVFP4,
            NodeFlags::default(),
            &[h, experts, routes],
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
    fn route_records_round_trip_and_refuse_malformed_bytes() {
        let params = MoeRouteParams::new(256, 8, 1).unwrap();
        assert_eq!((params.logits(), params.slots()), (257, 9));
        let bytes = encode_route_attributes(params).unwrap();
        assert_eq!(decode_route_attributes(&bytes), Some(params));
        let mut tagged = bytes;
        tagged[2] = 1;
        assert!(decode_route_attributes(&tagged).is_none());
        assert!(decode_route_attributes(&bytes[..12]).is_none());
        // More picks than experts, or than a route may hold, never encode.
        assert!(MoeRouteParams::new(4, 5, 0).is_none());
        assert!(MoeRouteParams::new(256, 17, 0).is_none());
        assert!(MoeRouteParams::new(256, 8, MAX_MOE_SHARED + 1).is_none());
        let mut picks = bytes;
        picks[8..12].copy_from_slice(&0_u32.to_le_bytes());
        assert!(decode_route_attributes(&picks).is_none());
    }

    #[test]
    fn routed_records_round_trip_and_size_their_tables() {
        // Qwen3.6-35B-A3B: 256 routed experts and one shared, 512 wide.
        let params = RoutedExpertsParams::new(257, 9, 2048, 512).unwrap();
        let bytes = encode_routed_attributes(params).unwrap();
        assert_eq!(decode_routed_attributes(&bytes), Some(params));
        let gate_up = params.gate_up_record();
        // 1024 rows of 1024 code bytes, 128 scale bytes each, after a
        // sixteen-byte header, padded to 64.
        assert_eq!(gate_up.block_scale_offset(), Some(16 + 1024 * 1024));
        assert_eq!(gate_up.bytes(), Some((16 + 1024 * 1024 + 1024 * 128_usize).next_multiple_of(64)));
        assert_eq!(params.down_record().bytes(), Some((16 + 2048 * 256 + 2048 * 32_usize).next_multiple_of(64)));
        let mut activation = bytes;
        activation[2] = 1;
        assert!(decode_routed_attributes(&activation).is_none());
        // Inputs too narrow for whole sixty-four-value chunks never encode.
        assert!(RoutedExpertsParams::new(257, 9, 2048, 48).is_none());
        assert!(RoutedExpertsParams::new(0, 9, 2048, 512).is_none());
        assert!(RoutedExpertsParams::new(257, 0, 2048, 512).is_none());
    }
}
