//! Explicit AMD XDNA probe artifacts and host-side Kerr boundary conversion.
//!
//! This module does not claim that XDNA executes Axiom's FP32 Kerr step graph.
//! It authors one admitted BF16 identity probe over a coarse ray plane and an
//! FP32 negative-control graph which the provider must reject.

use virtio_accel_tosa::DType;
use virtio_accel_tosa_build::{OperatorKind, OwnedGraph, OwnedOperator, OwnedTensor};

use crate::artifact::ArtifactError;

/// XDNA's BF16 identity DMA line, in elements.
pub const BF16_IDENTITY_LINE_ELEMENTS: usize = 1_024;

fn identity(dtype: DType, elements: usize) -> Result<Vec<u8>, ArtifactError> {
    let elements = i32::try_from(elements).map_err(|_| ArtifactError::DimensionTooLarge)?;
    if elements == 0 {
        return Err(ArtifactError::ZeroDimension);
    }
    let shape = vec![1, 1, elements];
    let mut graph = OwnedGraph::new("main");
    graph
        .push_tensor(OwnedTensor::new("x", shape.clone(), dtype))
        .push_tensor(OwnedTensor::new("y", shape, dtype))
        .push_operator(OwnedOperator::new(
            OperatorKind::Identity,
            vec!["x".into()],
            vec!["y".into()],
        ))
        .push_input("x")
        .push_output("y");
    graph
        .build(virtio_accel_xdna::XDNA_TOSA_TARGET)
        .map_err(|_| ArtifactError::AuthoringFailed)
}

/// Author the exact BF16 identity specialization admitted by XDNA.
pub fn bf16_identity(elements: usize) -> Result<Vec<u8>, ArtifactError> {
    if !elements.is_multiple_of(BF16_IDENTITY_LINE_ELEMENTS) {
        return Err(ArtifactError::InvalidLayerShape);
    }
    identity(DType::BF16, elements)
}

/// Author a valid TOSA FP32 identity as a fail-loud XDNA negative control.
pub fn unsupported_fp32_identity(elements: usize) -> Result<Vec<u8>, ArtifactError> {
    identity(DType::FP32, elements)
}

/// Narrow packed little-endian FP32 values to IEEE BF16 with round-to-nearest,
/// ties-to-even. NaNs remain NaNs and are quieted.
pub fn f32_bytes_to_bf16_rne(input: &[u8]) -> Result<Vec<u8>, ArtifactError> {
    if !input.len().is_multiple_of(4) {
        return Err(ArtifactError::InvalidLayerShape);
    }
    let mut output = Vec::with_capacity(input.len() / 2);
    for bytes in input.as_chunks::<4>().0 {
        let bits = u32::from_le_bytes(*bytes);
        let magnitude = bits & 0x7fff_ffff;
        let bf16 = if magnitude > 0x7f80_0000 {
            ((bits >> 16) as u16) | 0x0040
        } else {
            let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
            (rounded >> 16) as u16
        };
        output.extend_from_slice(&bf16.to_le_bytes());
    }
    Ok(output)
}

/// Widen packed little-endian BF16 values exactly to FP32.
pub fn bf16_bytes_to_f32(input: &[u8]) -> Result<Vec<f32>, ArtifactError> {
    if !input.len().is_multiple_of(2) {
        return Err(ArtifactError::InvalidLayerShape);
    }
    Ok(input
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f32::from_bits((u16::from_le_bytes(*bytes) as u32) << 16))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion_rounds_ties_to_even_and_quiets_nan() {
        let values = [
            f32::from_bits(0x3f80_7fff),
            f32::from_bits(0x3f80_8000),
            f32::from_bits(0x3f81_8000),
            f32::from_bits(0x7f80_0001),
        ];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let got = f32_bytes_to_bf16_rne(&bytes).unwrap();
        let bits: Vec<u16> = got
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect();
        assert_eq!(bits, [0x3f80, 0x3f80, 0x3f82, 0x7fc0]);
    }

    #[test]
    fn identity_shape_must_match_xdna_dma_line() {
        assert!(bf16_identity(BF16_IDENTITY_LINE_ELEMENTS).is_ok());
        assert_eq!(
            bf16_identity(BF16_IDENTITY_LINE_ELEMENTS - 1),
            Err(ArtifactError::InvalidLayerShape)
        );
    }
}
