//! Device-neutral numerical conformance: the shared virtio-accel
//! corpus executed by the axtosa CPU backend.
//!
//! The same artifacts and oracles the CoreML / OpenVINO / Hexagon host
//! backends consume — a provider cannot substitute its own graphs.

use virtio_accel_conformance::numerics::{
    self as n, Float16Tensor, Float32Tensor, PackedTensor, TosaFloat16Case, TosaFloat32Case,
    TosaInt8MatmulCase, TosaPackedCase, TosaRawCase,
};
use virtio_accel_tosa::{DType, NanPropagationMode, Op, OpAttributes, parse};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f16_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn run(artifact: &[u8], inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    match axtosa::execute(artifact, &refs) {
        Ok(outputs) => outputs,
        Err(e) => panic!("execute failed: {e:?}"),
    }
}

fn artifact_nan_mode(artifact: &[u8], target: Op) -> Option<NanPropagationMode> {
    let model = parse(artifact).ok()?;
    model
        .regions()
        .flat_map(|region| region.blocks())
        .flat_map(|block| block.operators())
        .find_map(|operator| {
            if operator.op() != target {
                return None;
            }
            match operator.attributes() {
                OpAttributes::Maximum { nan_mode } | OpAttributes::Minimum { nan_mode } => {
                    Some(nan_mode)
                }
                _ => None,
            }
        })
}

/// The shared conformance fixtures use PROPAGATE. Patch only the verified
/// scalar attribute field to exercise the other legal mode (and UNKNOWN)
/// without maintaining a parallel hand-written TOSA builder in this crate.
fn with_nan_mode(artifact: &[u8], target: Op, mode: NanPropagationMode) -> Vec<u8> {
    assert_eq!(
        artifact_nan_mode(artifact, target),
        Some(NanPropagationMode::PROPAGATE)
    );
    let encoded = NanPropagationMode::PROPAGATE.get().to_le_bytes();
    for offset in 0..=artifact.len() - encoded.len() {
        if artifact[offset..offset + encoded.len()] != encoded {
            continue;
        }
        let mut patched = artifact.to_vec();
        patched[offset..offset + encoded.len()].copy_from_slice(&mode.get().to_le_bytes());
        if artifact_nan_mode(&patched, target) == Some(mode) {
            return patched;
        }
    }
    panic!("fixture did not contain the expected NaN-mode field")
}

fn check_extrema_nan_mode(
    artifact: &[u8],
    target: Op,
    mode: NanPropagationMode,
    expected: &[Option<u16>],
) {
    let artifact = with_nan_mode(artifact, target, mode);
    let inputs = [
        f16_bytes(&[0x7e00, 0x4000]),
        f16_bytes(&[0x3c00, 0x7e00, 0x4200]),
    ];
    let outputs = run(&artifact, &inputs);
    let actual = outputs[0]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes));
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.zip(expected).enumerate() {
        match expected {
            Some(expected) => assert_eq!(actual, *expected, "element {index}"),
            None => assert!(
                actual & 0x7c00 == 0x7c00 && actual & 0x03ff != 0,
                "element {index}: expected NaN, got {actual:#06x}"
            ),
        }
    }
}

fn check_f32_case(case: &TosaFloat32Case) {
    let inputs: Vec<Vec<u8>> = case
        .inputs
        .iter()
        .map(|t: &Float32Tensor| f32_bytes(t.values))
        .collect();
    let outputs = run(case.artifact, &inputs);
    assert_eq!(
        outputs.len(),
        case.outputs.len(),
        "{}: output arity",
        case.name
    );
    for (slot, oracle) in case.outputs.iter().enumerate() {
        let actual: Vec<f32> = outputs[slot]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(
            actual.len(),
            oracle.values.len(),
            "{}: output {slot} length",
            case.name
        );
        for (i, (&want, &got)) in oracle.values.iter().zip(&actual).enumerate() {
            if want.is_nan() {
                assert!(
                    got.is_nan(),
                    "{}: [{slot}][{i}] expected NaN, got {got}",
                    case.name
                );
                continue;
            }
            let err = (got - want).abs();
            let ok = err <= case.absolute_tolerance
                || err <= case.relative_tolerance * want.abs()
                || got == want;
            assert!(
                ok,
                "{}: [{slot}][{i}] {got} vs {want} (err {err})",
                case.name
            );
        }
    }
}

fn check_f16_case(case: &TosaFloat16Case) {
    let inputs: Vec<Vec<u8>> = case
        .inputs
        .iter()
        .map(|t: &Float16Tensor| f16_bytes(t.bits))
        .collect();
    let outputs = run(case.artifact, &inputs);
    assert_eq!(
        outputs.len(),
        case.outputs.len(),
        "{}: output arity",
        case.name
    );
    for (slot, output) in outputs.iter().enumerate() {
        let actual: Vec<u16> = output
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect();
        assert!(
            case.output_matches(slot, &actual),
            "{}: output {slot} mismatch\n  got  {actual:04x?}\n  want {:04x?}",
            case.name,
            case.outputs[slot].bits
        );
    }
}

fn check_packed_case(case: &TosaPackedCase) {
    let inputs: Vec<Vec<u8>> = case
        .inputs
        .iter()
        .map(|t: &PackedTensor| t.bytes.to_vec())
        .collect();
    let outputs = run(case.artifact, &inputs);
    assert_eq!(
        outputs.len(),
        case.outputs.len(),
        "{}: output arity",
        case.name
    );
    assert!(
        case.output_matches(0, &outputs[0]),
        "{}: packed output mismatch\n  got  {:02x?}\n  want {:02x?}",
        case.name,
        outputs[0],
        case.outputs[0].bytes
    );
}

fn raw_case(name: &str) -> &'static TosaRawCase {
    n::HEXAGON_UNARY_FP16_CASES
        .iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("missing conformance case {name}"))
}

fn check_raw_case(case: &TosaRawCase) {
    let inputs: Vec<Vec<u8>> = case.inputs.iter().map(|tensor| tensor.bytes()).collect();
    let outputs = run(case.artifact, &inputs);
    assert_eq!(outputs.len(), 1, "{}: output arity", case.name);
    assert!(
        case.output_matches(&outputs[0]),
        "{}: output mismatch\n  got {:02x?}",
        case.name,
        outputs[0]
    );
}

fn check_f16_edges(artifact: &[u8], input: &[u16; 4], expected: &[u16; 4]) {
    let outputs = run(artifact, &[f16_bytes(input)]);
    let actual = outputs[0]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes));
    for (index, (actual, expected)) in actual.zip(expected).enumerate() {
        let expected_nan = expected & 0x7c00 == 0x7c00 && expected & 0x03ff != 0;
        if expected_nan {
            assert!(
                actual & 0x7c00 == 0x7c00 && actual & 0x03ff != 0,
                "element {index}: expected NaN, got {actual:#06x}"
            );
        } else {
            assert_eq!(actual, *expected, "element {index}");
        }
    }
}

fn check_f32_edges(artifact: &[u8], input: &[f32; 4], expected: &[f32; 4]) {
    let outputs = run(artifact, &[f32_bytes(input)]);
    let actual = outputs[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes));
    for (index, (actual, expected)) in actual.zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "element {index}: expected NaN, got {actual}"
            );
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "element {index}: {actual:?} vs {expected:?}"
            );
        }
    }
}

/// Turn the shared two-tensor unary fixture into its FP32 equivalent without
/// maintaining a second generated FlatBuffer. Candidate edits are accepted
/// only when the public parser sees exactly the original tensors, all with the
/// requested dtype, and the complete TOSA semantic pass still succeeds.
fn fp32_variant(artifact: &[u8]) -> Vec<u8> {
    let source = parse(artifact).expect("source conformance fixture parses");
    let source_types: Vec<DType> = source
        .regions()
        .flat_map(|region| region.blocks())
        .flat_map(|block| block.tensors())
        .map(|tensor| tensor.dtype())
        .collect();
    assert!(!source_types.is_empty());
    assert!(source_types.iter().all(|dtype| *dtype == DType::FP16));

    let needle = DType::FP16.get().to_le_bytes();
    let replacement = DType::FP32.get().to_le_bytes();
    let candidates: Vec<usize> = artifact
        .windows(needle.len())
        .enumerate()
        .filter_map(|(offset, bytes)| (bytes == needle).then_some(offset))
        .collect();

    // The fixture has one input tensor and one output tensor. Searching the
    // handful of matching u32 fields keeps this test independent of private
    // generated FlatBuffer accessors while still validating the final bytes.
    for (first_index, &first) in candidates.iter().enumerate() {
        for &second in &candidates[first_index + 1..] {
            let mut patched = artifact.to_vec();
            patched[first..first + replacement.len()].copy_from_slice(&replacement);
            patched[second..second + replacement.len()].copy_from_slice(&replacement);
            let Ok(model) = parse(&patched) else {
                continue;
            };
            let types: Vec<DType> = model
                .regions()
                .flat_map(|region| region.blocks())
                .flat_map(|block| block.tensors())
                .map(|tensor| tensor.dtype())
                .collect();
            if types.len() == source_types.len()
                && types.iter().all(|dtype| *dtype == DType::FP32)
                && model.validate_for(axtosa::target()).is_ok()
            {
                return patched;
            }
        }
    }
    panic!("could not derive a validated FP32 unary fixture")
}

// ── FP32 ────────────────────────────────────────────────────────────

#[test]
fn matmul_fp32() {
    check_f32_case(&n::MATMUL_FP32);
}

#[test]
fn max_pool2d_fp32() {
    check_f32_case(&n::MAX_POOL2D_FP32);
}

#[test]
fn identity_edges_fp32() {
    check_f32_case(&n::IDENTITY_EDGES_FP32);
}

#[test]
fn ceil_and_floor_fp32_edges() {
    let ceil = fp32_variant(raw_case("ceil-fp16").artifact);
    check_f32_edges(
        &ceil,
        &[
            f32::NEG_INFINITY,
            -1.5,
            f32::from_bits(0x8000_0001),
            f32::from_bits(0x0000_0001),
        ],
        &[f32::NEG_INFINITY, -1.0, -0.0, 1.0],
    );
    check_f32_edges(
        &ceil,
        &[f32::NAN, -f32::NAN, -0.0, 0.0],
        &[f32::NAN, f32::NAN, -0.0, 0.0],
    );

    let floor = fp32_variant(raw_case("floor-fp16").artifact);
    check_f32_edges(
        &floor,
        &[
            f32::INFINITY,
            1.5,
            f32::from_bits(0x0000_0001),
            f32::from_bits(0x8000_0001),
        ],
        &[f32::INFINITY, 1.0, 0.0, -1.0],
    );
    check_f32_edges(
        &floor,
        &[f32::NAN, -f32::NAN, -0.0, 0.0],
        &[f32::NAN, f32::NAN, -0.0, 0.0],
    );
}

// ── FP16 (bit-exact oracles) ────────────────────────────────────────

#[test]
fn matmul_fp16() {
    check_f16_case(&n::MATMUL_FP16);
}

#[test]
fn elementwise_fp16() {
    for case in [
        &n::ADD_FP16,
        &n::SUB_FP16,
        &n::MUL_FP16,
        &n::POW_FP16,
        &n::MAXIMUM_FP16,
        &n::MINIMUM_FP16,
    ] {
        check_f16_case(case);
    }
}

#[test]
fn maximum_and_minimum_honor_both_nan_modes() {
    check_extrema_nan_mode(
        n::MAXIMUM_FP16.artifact,
        Op::MAXIMUM,
        NanPropagationMode::PROPAGATE,
        &[None, None, None, Some(0x4000), None, Some(0x4200)],
    );
    check_extrema_nan_mode(
        n::MAXIMUM_FP16.artifact,
        Op::MAXIMUM,
        NanPropagationMode::IGNORE,
        &[
            Some(0x3c00),
            None,
            Some(0x4200),
            Some(0x4000),
            Some(0x4000),
            Some(0x4200),
        ],
    );
    check_extrema_nan_mode(
        n::MINIMUM_FP16.artifact,
        Op::MINIMUM,
        NanPropagationMode::PROPAGATE,
        &[None, None, None, Some(0x3c00), None, Some(0x4000)],
    );
    check_extrema_nan_mode(
        n::MINIMUM_FP16.artifact,
        Op::MINIMUM,
        NanPropagationMode::IGNORE,
        &[
            Some(0x3c00),
            None,
            Some(0x4200),
            Some(0x3c00),
            Some(0x4000),
            Some(0x4000),
        ],
    );
}

#[test]
fn unknown_default_nan_mode_is_rejected_during_compile() {
    let artifact = with_nan_mode(
        n::MAXIMUM_FP16.artifact,
        Op::MAXIMUM,
        NanPropagationMode::UNKNOWN,
    );
    assert!(matches!(
        axtosa::Program::compile(&artifact),
        Err(axtosa::ExecError::Analysis(_))
    ));
}

#[test]
fn max_pool2d_fp16() {
    check_f16_case(&n::MAX_POOL2D_FP16);
}

#[test]
fn identity_edges_fp16() {
    check_f16_case(&n::IDENTITY_EDGES_FP16);
}

#[test]
fn ceil_and_floor_fp16_conformance_and_edges() {
    let ceil = raw_case("ceil-fp16");
    let floor = raw_case("floor-fp16");
    check_raw_case(ceil);
    check_raw_case(floor);

    check_f16_edges(
        ceil.artifact,
        &[0xfc00, 0xbe00, 0x8001, 0x0001],
        &[0xfc00, 0xbc00, 0x8000, 0x3c00],
    );
    check_f16_edges(
        ceil.artifact,
        &[0x7e01, 0xfe01, 0x8000, 0x0000],
        &[0x7e00, 0x7e00, 0x8000, 0x0000],
    );
    check_f16_edges(
        floor.artifact,
        &[0x7c00, 0x3e00, 0x0001, 0x8001],
        &[0x7c00, 0x3c00, 0x0000, 0xbc00],
    );
    check_f16_edges(
        floor.artifact,
        &[0x7e01, 0xfe01, 0x8000, 0x0000],
        &[0x7e00, 0x7e00, 0x8000, 0x0000],
    );
}

#[test]
fn mock_linear_classifier_fp16() {
    check_f16_case(&n::MOCK_LINEAR_CLASSIFIER_FP16);
}

// ── FP8: the tier every vendor backend currently rejects ────────────

#[test]
fn identity_fp8e4m3() {
    check_packed_case(&n::IDENTITY_FP8E4M3);
}

#[test]
fn identity_fp8e5m2() {
    check_packed_case(&n::IDENTITY_FP8E5M2);
}

// ── Integer tier ────────────────────────────────────────────────────

#[test]
fn identity_int8() {
    check_packed_case(&n::IDENTITY_INT8);
}

#[test]
fn identity_int4() {
    check_packed_case(&n::IDENTITY_INT4);
}

#[test]
fn matmul_int8() {
    let case: &TosaInt8MatmulCase = &n::MATMUL_INT8;
    let inputs: Vec<Vec<u8>> = case.inputs.iter().map(|t| t.bytes.to_vec()).collect();
    let outputs = run(case.artifact, &inputs);
    assert_eq!(
        outputs.len(),
        case.outputs.len(),
        "{}: output arity",
        case.name
    );
    for (slot, output) in outputs.iter().enumerate() {
        let actual: Vec<i32> = output
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| i32::from_le_bytes(*bytes))
            .collect();
        assert!(
            case.output_matches(slot, &actual),
            "{}: output {slot} mismatch: {actual:?}",
            case.name
        );
    }
}
