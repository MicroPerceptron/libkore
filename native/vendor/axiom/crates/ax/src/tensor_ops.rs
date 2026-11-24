//! Attribute records of the 1.5 decoder operations: views, top-k selection
//! and axis-aware RMS normalization.
//!
//! Like the other neural records these are headerless little-endian fields;
//! each operation admits exactly one record length (`RMS_NORM` two).
//!
//! `MATMUL_TRANSPOSED` gained an optional 1.8 epilogue record (#107) that
//! folds its per-tensor scale, an elementwise activation and a layer's
//! closing residual add into the product's write-out.

pub use axnn::lowering::EpilogueActivation;
pub use axnn::RopeMode;

/// `VIEW`: LE u64 element offset into the input.
pub const VIEW_ATTRIBUTE_BYTES: usize = 8;
/// `TOP_K`: LE u32 `k`.
pub const TOP_K_ATTRIBUTE_BYTES: usize = 4;
/// Whole-tensor `RMS_NORM`: LE f32 epsilon (1.3).
pub const RMS_NORM_ATTRIBUTE_BYTES: usize = 4;
/// Axis-aware `RMS_NORM`: LE f32 epsilon, then LE i32 first normalized axis
/// (1.5).
pub const RMS_NORM_AXIS_ATTRIBUTE_BYTES: usize = 8;
/// Fused epilogue of a `MATMUL_TRANSPOSED`: LE f32 scale, then LE u32
/// activation code (1.8). A node without this record keeps its plain
/// unscaled product.
///
/// The residual accumulate the kernel also supports is deliberately absent:
/// it aliases its result onto an input, and every `.ax` tensor has exactly
/// one producer, so expressing it needs alias semantics this format does not
/// have yet. It stays reachable only through the kernel API.
pub const MATMUL_EPILOGUE_ATTRIBUTE_BYTES: usize = 8;
/// `ROPE`: LE u8 [`RopeMode`] tag (0 interleaved, 1 NeoX).
pub const ROPE_ATTRIBUTE_BYTES: usize = 1;
/// A partially rotated `ROPE`: the tag, then the LE u32 count of leading
/// values of each head that rotate (1.22). The others pass through.
pub const ROPE_PARTIAL_ATTRIBUTE_BYTES: usize = 5;

fn read_u32(bytes: &[u8], start: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(start..start + 4)?.try_into().ok()?,
    ))
}

/// A view's element offset into its input; the view's shape is its output
/// tensor's.
pub fn encode_view_attributes(offset: u64) -> [u8; VIEW_ATTRIBUTE_BYTES] {
    offset.to_le_bytes()
}

const fn rope_mode_tag(mode: RopeMode) -> u8 {
    match mode {
        RopeMode::Interleaved => 0,
        RopeMode::NeoX => 1,
    }
}

/// A `ROPE` node's one-tag attribute record: each head rotates whole.
pub fn encode_rope_attributes(mode: RopeMode) -> [u8; ROPE_ATTRIBUTE_BYTES] {
    [rope_mode_tag(mode)]
}

/// The 1.22 record of a `ROPE` whose heads rotate only their leading
/// `rotary_dims` values: a nonzero even count.
pub fn encode_rope_partial_attributes(
    mode: RopeMode,
    rotary_dims: u32,
) -> Option<[u8; ROPE_PARTIAL_ATTRIBUTE_BYTES]> {
    if rotary_dims == 0 || !rotary_dims.is_multiple_of(2) {
        return None;
    }
    let mut bytes = [0; ROPE_PARTIAL_ATTRIBUTE_BYTES];
    bytes[0] = rope_mode_tag(mode);
    bytes[1..].copy_from_slice(&rotary_dims.to_le_bytes());
    Some(bytes)
}

/// Decode a `ROPE` record: the layout, and the rotated span when the record
/// names one (`None` rotates each head whole). Other lengths, unknown tags
/// and an odd or empty span are rejected.
pub fn decode_rope_attributes(bytes: &[u8]) -> Option<(RopeMode, Option<u32>)> {
    let mode = match bytes.first()? {
        0 => RopeMode::Interleaved,
        1 => RopeMode::NeoX,
        _ => return None,
    };
    match bytes.len() {
        ROPE_ATTRIBUTE_BYTES => Some((mode, None)),
        ROPE_PARTIAL_ATTRIBUTE_BYTES => {
            let rotary = read_u32(bytes, 1)?;
            (rotary != 0 && rotary.is_multiple_of(2)).then_some((mode, Some(rotary)))
        }
        _ => None,
    }
}

pub fn decode_view_attributes(bytes: &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

pub fn encode_top_k_attributes(k: u32) -> Option<[u8; TOP_K_ATTRIBUTE_BYTES]> {
    (k != 0).then(|| k.to_le_bytes())
}

pub fn decode_top_k_attributes(bytes: &[u8]) -> Option<u32> {
    let k = u32::from_le_bytes(bytes.try_into().ok()?);
    (k != 0).then_some(k)
}

/// Encode a fused matmul epilogue (#107): `out = activation(product *
/// scale)`. A finite scale is required.
pub fn encode_matmul_epilogue_attributes(
    scale: f32,
    activation: EpilogueActivation,
) -> Option<[u8; MATMUL_EPILOGUE_ATTRIBUTE_BYTES]> {
    if !scale.is_finite() {
        return None;
    }
    let mut bytes = [0; MATMUL_EPILOGUE_ATTRIBUTE_BYTES];
    bytes[..4].copy_from_slice(&scale.to_le_bytes());
    bytes[4..].copy_from_slice(&activation.code().to_le_bytes());
    Some(bytes)
}

/// Decode the epilogue record as `(scale, activation)`. An unknown
/// activation code is rejected, so a reader never silently ignores an
/// epilogue it does not implement.
pub fn decode_matmul_epilogue_attributes(bytes: &[u8]) -> Option<(f32, EpilogueActivation)> {
    if bytes.len() != MATMUL_EPILOGUE_ATTRIBUTE_BYTES {
        return None;
    }
    let scale = f32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
    let activation = EpilogueActivation::from_code(read_u32(bytes, 4)?)?;
    scale.is_finite().then_some((scale, activation))
}

/// Encode an RMS norm over the trailing axes from `axis`. A positive, finite
/// epsilon is required.
pub fn encode_rms_norm_attributes(
    epsilon: f32,
    axis: i32,
) -> Option<[u8; RMS_NORM_AXIS_ATTRIBUTE_BYTES]> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    let mut bytes = [0; RMS_NORM_AXIS_ATTRIBUTE_BYTES];
    bytes[..4].copy_from_slice(&epsilon.to_le_bytes());
    bytes[4..].copy_from_slice(&axis.to_le_bytes());
    Some(bytes)
}

/// Decode either `RMS_NORM` record as `(epsilon, first normalized axis)`.
/// The 4-byte form normalizes the whole tensor (axis 0).
pub fn decode_rms_norm_attributes(bytes: &[u8]) -> Option<(f32, i32)> {
    let epsilon = f32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
    let axis = match bytes.len() {
        RMS_NORM_ATTRIBUTE_BYTES => 0,
        RMS_NORM_AXIS_ATTRIBUTE_BYTES => read_u32(bytes, 4)? as i32,
        _ => return None,
    };
    (epsilon.is_finite() && epsilon > 0.0).then_some((epsilon, axis))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1.8 epilogue record round-trips exactly, and rejects a
    /// non-finite scale, an unknown activation code and a wrong length.
    #[test]
    fn matmul_epilogue_records_round_trip_and_reject_unknown_codes() {
        for activation in [
            EpilogueActivation::None,
            EpilogueActivation::Silu,
            EpilogueActivation::Sigmoid,
        ] {
            let bytes = encode_matmul_epilogue_attributes(0.125, activation).unwrap();
            assert_eq!(bytes.len(), MATMUL_EPILOGUE_ATTRIBUTE_BYTES);
            assert_eq!(
                decode_matmul_epilogue_attributes(&bytes),
                Some((0.125, activation))
            );
            // The scale occupies the first four bytes, the code the rest.
            assert_eq!(bytes[..4], 0.125_f32.to_le_bytes());
            assert_eq!(bytes[4..], activation.code().to_le_bytes());
        }
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(encode_matmul_epilogue_attributes(scale, EpilogueActivation::None).is_none());
        }
        // A negative scale is legitimate.
        assert_eq!(
            decode_matmul_epilogue_attributes(
                &encode_matmul_epilogue_attributes(-2.0, EpilogueActivation::Silu).unwrap()
            ),
            Some((-2.0, EpilogueActivation::Silu))
        );
        let good = encode_matmul_epilogue_attributes(1.0, EpilogueActivation::None).unwrap();
        assert!(decode_matmul_epilogue_attributes(&good[..7]).is_none());
        assert!(decode_matmul_epilogue_attributes(&[0; 12]).is_none());
        let mut unknown = good;
        unknown[4] = 3;
        assert!(decode_matmul_epilogue_attributes(&unknown).is_none());
        let mut nan = good;
        nan[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_matmul_epilogue_attributes(&nan).is_none());
    }

    #[test]
    fn records_are_canonical_and_reject_malformed_bytes() {
        let view = encode_view_attributes(5376);
        assert_eq!(view, 5376_u64.to_le_bytes());
        assert_eq!(decode_view_attributes(&view), Some(5376));
        assert!(decode_view_attributes(&view[..7]).is_none());
        assert!(decode_view_attributes(&[0; 9]).is_none());

        assert_eq!(
            decode_top_k_attributes(&encode_top_k_attributes(6).unwrap()),
            Some(6)
        );
        assert!(encode_top_k_attributes(0).is_none());
        assert!(decode_top_k_attributes(&[0; 4]).is_none());
        assert!(decode_top_k_attributes(&[6, 0, 0]).is_none());

        let norm = encode_rms_norm_attributes(1.0e-5, 1).unwrap();
        assert_eq!(decode_rms_norm_attributes(&norm), Some((1.0e-5, 1)));
        assert_eq!(decode_rms_norm_attributes(&norm[..4]), Some((1.0e-5, 0)));
        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(encode_rms_norm_attributes(epsilon, 0).is_none());
            assert!(decode_rms_norm_attributes(&epsilon.to_le_bytes()).is_none());
        }
        assert!(decode_rms_norm_attributes(&norm[..6]).is_none());
    }
}
