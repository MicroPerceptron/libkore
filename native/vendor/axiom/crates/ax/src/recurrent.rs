//! Recurrent-state nodes and their attribute schema: a depthwise causal
//! convolution window, the Mamba-2 selective state-space step and the gated
//! delta rule (Gated DeltaNet).
//!
//! Each node reads its carried state from a source resource and writes the
//! successor to a distinct target resource. The graph keeps both keys; the
//! caller alternates the buffers bound to them after each completed
//! invocation, so a failed or repeated step recomputes from an unchanged
//! source.

use core::alloc::Allocator;

pub use axnn::lowering::{
    CausalConv1dParams, EpilogueActivation, F32Bits, GatedDeltaRuleParams, ResourceKey,
    SsmScanParams, StateBinding,
};

use crate::{
    AllocationError, AxError, DType, DYNAMIC_DIM, GraphBuilder, NodeFlags, OpCode, TensorFlags,
    TensorId,
};

/// Version 1 attribute record of a `CAUSAL_CONV1D` node.
pub const CAUSAL_CONV1D_ATTRIBUTE_BYTES: usize = 28;
/// A `CAUSAL_CONV1D` record carrying a fused activation, appended as a
/// further LE u32 (1.8, #107). The shorter record keeps its exact 1.6 bytes
/// and its no-activation meaning.
pub const CAUSAL_CONV1D_ACTIVATED_ATTRIBUTE_BYTES: usize = 32;
/// Version 1 attribute record of an `SSM_SCAN` node.
pub const SSM_SCAN_ATTRIBUTE_BYTES: usize = 44;
/// Version 1 attribute record of a `GATED_DELTA_RULE` node (1.22).
pub const GATED_DELTA_RULE_ATTRIBUTE_BYTES: usize = 40;

/// The exact flags of every recurrent node: an ordered read of the source
/// state and write of the target.
pub const RECURRENT_NODE_FLAGS: NodeFlags =
    NodeFlags(NodeFlags::ORDERED.0 | NodeFlags::READS_RESOURCE.0 | NodeFlags::WRITES_RESOURCE.0);

const CAUSAL_CONV1D_KIND: u8 = 7;
const SSM_SCAN_KIND: u8 = 8;
const GATED_DELTA_RULE_KIND: u8 = 9;
/// Bit 0 of a gated delta rule's flag word: L2-normalize the query and key.
const QK_L2NORM: u32 = 1;

fn read_u32(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(start..start + 4)?.try_into().ok()?,
    ))
}

/// Header `[1, kind, 0, 0]`, then the source and target keys, then `fields`.
fn encode<const N: usize>(kind: u8, state: StateBinding, fields: &[u32]) -> [u8; N] {
    let mut bytes = [0_u8; N];
    bytes[0] = 1;
    bytes[1] = kind;
    for (slot, value) in bytes[4..].chunks_exact_mut(4).zip(
        [
            state.source.group,
            state.source.binding,
            state.target.group,
            state.target.binding,
        ]
        .iter()
        .chain(fields),
    ) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_state(bytes: &[u8], kind: u8, len: usize) -> Option<StateBinding> {
    if bytes.len() != len || bytes[..4] != [1, kind, 0, 0] {
        return None;
    }
    Some(StateBinding {
        source: ResourceKey {
            group: read_u32(bytes, 4)?,
            binding: read_u32(bytes, 8)?,
        },
        target: ResourceKey {
            group: read_u32(bytes, 12)?,
            binding: read_u32(bytes, 16)?,
        },
    })
}

/// Layout: `[1, 7, 0, 0]`, then LE u32 source group, source binding, target
/// group, target binding, channels and kernel. Params carrying a fused
/// activation need [`encode_causal_conv1d_activated_attributes`].
pub fn encode_causal_conv1d_attributes(
    params: CausalConv1dParams,
) -> Option<[u8; CAUSAL_CONV1D_ATTRIBUTE_BYTES]> {
    (params.is_valid() && params.activation == EpilogueActivation::None).then(|| {
        encode(
            CAUSAL_CONV1D_KIND,
            params.state,
            &[params.channels, params.kernel],
        )
    })
}

/// The same record with the fused activation code appended (1.8, #107).
pub fn encode_causal_conv1d_activated_attributes(
    params: CausalConv1dParams,
) -> Option<[u8; CAUSAL_CONV1D_ACTIVATED_ATTRIBUTE_BYTES]> {
    params.is_valid().then(|| {
        encode(
            CAUSAL_CONV1D_KIND,
            params.state,
            &[params.channels, params.kernel, params.activation.code()],
        )
    })
}

/// Decode either `CAUSAL_CONV1D` record, with valid geometry and distinct
/// source and target keys. The 28-byte form applies no activation; the
/// 32-byte form carries one, and an unknown code is rejected.
pub fn decode_causal_conv1d_attributes(bytes: &[u8]) -> Option<CausalConv1dParams> {
    let activation = match bytes.len() {
        CAUSAL_CONV1D_ATTRIBUTE_BYTES => EpilogueActivation::None,
        CAUSAL_CONV1D_ACTIVATED_ATTRIBUTE_BYTES => {
            EpilogueActivation::from_code(read_u32(bytes, 28)?)?
        }
        _ => return None,
    };
    let state = decode_state(bytes, CAUSAL_CONV1D_KIND, bytes.len())?;
    CausalConv1dParams::new_activated(
        state,
        read_u32(bytes, 20)?,
        read_u32(bytes, 24)?,
        activation,
    )
}

/// Layout: `[1, 8, 0, 0]`, then LE u32 source group, source binding, target
/// group, target binding, heads, head_dim, groups, state_size, and the F32
/// bits of `dt_min` and `dt_max`.
pub fn encode_ssm_scan_attributes(params: SsmScanParams) -> Option<[u8; SSM_SCAN_ATTRIBUTE_BYTES]> {
    params.is_valid().then(|| {
        encode(
            SSM_SCAN_KIND,
            params.state,
            &[
                params.heads,
                params.head_dim,
                params.groups,
                params.state_size,
                params.dt_min.bits(),
                params.dt_max.bits(),
            ],
        )
    })
}

pub fn decode_ssm_scan_attributes(bytes: &[u8]) -> Option<SsmScanParams> {
    let state = decode_state(bytes, SSM_SCAN_KIND, SSM_SCAN_ATTRIBUTE_BYTES)?;
    SsmScanParams::new(
        state,
        read_u32(bytes, 20)?,
        read_u32(bytes, 24)?,
        read_u32(bytes, 28)?,
        read_u32(bytes, 32)?,
        F32Bits::from_bits(read_u32(bytes, 36)?).to_f32(),
        F32Bits::from_bits(read_u32(bytes, 40)?).to_f32(),
    )
}

/// Layout: `[1, 9, 0, 0]`, then LE u32 source group, source binding, target
/// group, target binding, key heads, key width, value heads, value width
/// and a flag word (bit 0: L2-normalize the query and key; the others zero).
pub fn encode_gated_delta_rule_attributes(
    params: GatedDeltaRuleParams,
) -> Option<[u8; GATED_DELTA_RULE_ATTRIBUTE_BYTES]> {
    params.is_valid().then(|| {
        encode(
            GATED_DELTA_RULE_KIND,
            params.state,
            &[
                params.key_heads,
                params.key_dim,
                params.value_heads,
                params.value_dim,
                if params.qk_l2norm { QK_L2NORM } else { 0 },
            ],
        )
    })
}

/// Decode a `GATED_DELTA_RULE` record with valid geometry, distinct source
/// and target keys, and no unknown flag bits.
pub fn decode_gated_delta_rule_attributes(bytes: &[u8]) -> Option<GatedDeltaRuleParams> {
    let state = decode_state(
        bytes,
        GATED_DELTA_RULE_KIND,
        GATED_DELTA_RULE_ATTRIBUTE_BYTES,
    )?;
    let flags = read_u32(bytes, 36)?;
    if flags & !QK_L2NORM != 0 {
        return None;
    }
    GatedDeltaRuleParams::new(
        state,
        (read_u32(bytes, 20)?, read_u32(bytes, 24)?),
        (read_u32(bytes, 28)?, read_u32(bytes, 32)?),
        flags & QK_L2NORM != 0,
    )
}

fn rows(tokens: u32, width: usize) -> Result<[u32; 2], AxError> {
    let width = u32::try_from(width).map_err(|_| AxError::CountOverflow)?;
    if tokens == 0 || tokens == DYNAMIC_DIM || width == DYNAMIC_DIM {
        return Err(AxError::InvalidTensorRange);
    }
    Ok([tokens, width])
}

impl<A: Allocator + Clone> GraphBuilder<A> {
    /// `tokens` steps of a depthwise causal convolution: F32 `input`
    /// `[tokens, channels]`, `weight` `[channels, kernel]` and `bias`
    /// `[channels]`. Returns the F32 `[tokens, channels]` result.
    pub fn causal_conv1d(
        &mut self,
        input: TensorId,
        weight: TensorId,
        bias: TensorId,
        tokens: u32,
        params: CausalConv1dParams,
        output_flags: TensorFlags,
    ) -> Result<TensorId, AllocationError> {
        // An activated convolution (#107) needs the longer 1.8 record; the
        // plain form keeps its exact 1.6 bytes.
        let plain;
        let activated;
        let attributes: &[u8] = if params.activation == EpilogueActivation::None {
            plain = encode_causal_conv1d_attributes(params)
                .ok_or(AxError::InvalidNodeAttributes(OpCode::CAUSAL_CONV1D))?;
            &plain
        } else {
            activated = encode_causal_conv1d_activated_attributes(params)
                .ok_or(AxError::InvalidNodeAttributes(OpCode::CAUSAL_CONV1D))?;
            &activated
        };
        let dimensions = rows(tokens, params.channels as usize)?;
        let output = self.tensor(DType::F32, output_flags, &dimensions, &[])?;
        self.node(
            OpCode::CAUSAL_CONV1D,
            RECURRENT_NODE_FLAGS,
            &[input, weight, bias],
            &[output],
            attributes,
        )?;
        Ok(output)
    }

    /// `tokens` Mamba-2 steps: F32 `xbc` `[tokens, conv_dim]`, `dt`
    /// `[tokens, heads]` and `head_params` `[3, heads]` (`A_log`, `dt_bias`,
    /// `D`). Returns the F32 `[tokens, heads * head_dim]` result.
    pub fn ssm_scan(
        &mut self,
        xbc: TensorId,
        dt: TensorId,
        head_params: TensorId,
        tokens: u32,
        params: SsmScanParams,
        output_flags: TensorFlags,
    ) -> Result<TensorId, AllocationError> {
        let attributes = encode_ssm_scan_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::SSM_SCAN))?;
        let width = (params.heads as usize)
            .checked_mul(params.head_dim as usize)
            .ok_or(AxError::CountOverflow)?;
        let dimensions = rows(tokens, width)?;
        let output = self.tensor(DType::F32, output_flags, &dimensions, &[])?;
        self.node(
            OpCode::SSM_SCAN,
            RECURRENT_NODE_FLAGS,
            &[xbc, dt, head_params],
            &[output],
            &attributes,
        )?;
        Ok(output)
    }

    /// `tokens` gated delta rule steps: F32 `qkv` `[tokens, qkv_dim]` (the
    /// query, key and value heads), log-space `decay` and write strength
    /// `beta`, each `[tokens, value_heads]`. Returns the F32 `[tokens,
    /// value_heads * value_dim]` result.
    pub fn gated_delta_rule(
        &mut self,
        qkv: TensorId,
        decay: TensorId,
        beta: TensorId,
        tokens: u32,
        params: GatedDeltaRuleParams,
        output_flags: TensorFlags,
    ) -> Result<TensorId, AllocationError> {
        let attributes = encode_gated_delta_rule_attributes(params)
            .ok_or(AxError::InvalidNodeAttributes(OpCode::GATED_DELTA_RULE))?;
        let width = params.value_width().ok_or(AxError::CountOverflow)?;
        let dimensions = rows(tokens, width)?;
        let output = self.tensor(DType::F32, output_flags, &dimensions, &[])?;
        self.node(
            OpCode::GATED_DELTA_RULE,
            RECURRENT_NODE_FLAGS,
            &[qkv, decay, beta],
            &[output],
            &attributes,
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATE: StateBinding = StateBinding {
        source: ResourceKey {
            group: 4,
            binding: 0,
        },
        target: ResourceKey {
            group: 4,
            binding: 1,
        },
    };

    /// The 1.8 activated convolution record appends the activation code and
    /// round-trips; the 1.6 record still decodes with no activation, and the
    /// two encoders each reject the other's params.
    #[test]
    fn activated_convolution_records_round_trip_beside_the_plain_form() {
        let plain = CausalConv1dParams::new(STATE, 7424, 4).unwrap();
        assert_eq!(plain.activation, EpilogueActivation::None);
        let activated =
            CausalConv1dParams::new_activated(STATE, 7424, 4, EpilogueActivation::Silu).unwrap();

        let bytes = encode_causal_conv1d_activated_attributes(activated).unwrap();
        assert_eq!(bytes.len(), CAUSAL_CONV1D_ACTIVATED_ATTRIBUTE_BYTES);
        assert_eq!(bytes[..4], [1, 7, 0, 0]);
        assert_eq!(bytes[28..], EpilogueActivation::Silu.code().to_le_bytes());
        assert_eq!(decode_causal_conv1d_attributes(&bytes), Some(activated));

        // The shorter record keeps its exact 1.6 bytes and meaning.
        let short = encode_causal_conv1d_attributes(plain).unwrap();
        assert_eq!(short.len(), CAUSAL_CONV1D_ATTRIBUTE_BYTES);
        assert_eq!(decode_causal_conv1d_attributes(&short), Some(plain));

        // An activated params never encodes as the short record.
        assert!(encode_causal_conv1d_attributes(activated).is_none());
        // A zero activation code decodes as the plain params.
        let widened = encode_causal_conv1d_activated_attributes(plain).unwrap();
        assert_eq!(decode_causal_conv1d_attributes(&widened), Some(plain));
        // An unknown activation code is rejected.
        let mut unknown = bytes;
        unknown[28] = 7;
        assert!(decode_causal_conv1d_attributes(&unknown).is_none());
        // Lengths between the two records are not records.
        assert!(decode_causal_conv1d_attributes(&bytes[..30]).is_none());
    }

    #[test]
    fn gated_delta_rule_records_round_trip_and_reject_malformed_bytes() {
        let rule = GatedDeltaRuleParams::new(STATE, (16, 128), (48, 128), true).unwrap();
        let bytes = encode_gated_delta_rule_attributes(rule).unwrap();
        assert_eq!(bytes.len(), GATED_DELTA_RULE_ATTRIBUTE_BYTES);
        assert_eq!(bytes[..8], [1, 9, 0, 0, 4, 0, 0, 0]);
        assert_eq!(bytes[36..], 1_u32.to_le_bytes());
        assert_eq!(decode_gated_delta_rule_attributes(&bytes), Some(rule));
        let plain = GatedDeltaRuleParams {
            qk_l2norm: false,
            ..rule
        };
        let record = encode_gated_delta_rule_attributes(plain).unwrap();
        assert_eq!(record[36..], [0; 4]);
        assert_eq!(decode_gated_delta_rule_attributes(&record), Some(plain));
        // Unknown flag bits, other kinds and other lengths are not records.
        let mut unknown = bytes;
        unknown[36] = 3;
        assert!(decode_gated_delta_rule_attributes(&unknown).is_none());
        let mut kind = bytes;
        kind[1] = 8;
        assert!(decode_gated_delta_rule_attributes(&kind).is_none());
        assert!(decode_gated_delta_rule_attributes(&bytes[..39]).is_none());
        // Value heads must be a multiple of key heads; the keys distinct.
        let mut ragged = bytes;
        ragged[28] = 47;
        assert!(decode_gated_delta_rule_attributes(&ragged).is_none());
        let mut aliased = bytes;
        aliased[16] = 0;
        assert!(decode_gated_delta_rule_attributes(&aliased).is_none());
        assert_eq!(rule.qkv_dim(), Some(2 * 16 * 128 + 48 * 128));
        assert_eq!(rule.state_elements(), Some(48 * 128 * 128));
    }

    #[test]
    fn recurrent_records_are_canonical_and_reject_malformed_bytes() {
        let conv = CausalConv1dParams::new(STATE, 7424, 4).unwrap();
        let bytes = encode_causal_conv1d_attributes(conv).unwrap();
        assert_eq!(bytes[..8], [1, 7, 0, 0, 4, 0, 0, 0]);
        assert_eq!(decode_causal_conv1d_attributes(&bytes), Some(conv));
        let scan = SsmScanParams::new(STATE, 64, 64, 8, 128, 0.0, f32::INFINITY).unwrap();
        let record = encode_ssm_scan_attributes(scan).unwrap();
        assert_eq!(record[..4], [1, 8, 0, 0]);
        assert_eq!(record[40..], f32::INFINITY.to_bits().to_le_bytes());
        assert_eq!(decode_ssm_scan_attributes(&record), Some(scan));

        assert!(decode_causal_conv1d_attributes(&record).is_none());
        assert!(decode_ssm_scan_attributes(&bytes).is_none());
        for offset in 0..4 {
            let mut invalid = bytes;
            invalid[offset] ^= 1;
            assert!(decode_causal_conv1d_attributes(&invalid).is_none());
            let mut invalid = record;
            invalid[offset] ^= 1;
            assert!(decode_ssm_scan_attributes(&invalid).is_none());
        }
        // The target key equals the source key.
        let mut aliased = bytes;
        aliased[16] = 0;
        assert!(decode_causal_conv1d_attributes(&aliased).is_none());
        // Groups must divide heads; the step range must be nonempty.
        let mut invalid = record;
        invalid[28] = 7;
        assert!(decode_ssm_scan_attributes(&invalid).is_none());
        let mut invalid = record;
        invalid[40..].copy_from_slice(&0.0_f32.to_bits().to_le_bytes());
        assert!(decode_ssm_scan_attributes(&invalid).is_none());
        assert!(decode_ssm_scan_attributes(&record[..43]).is_none());
    }
}
