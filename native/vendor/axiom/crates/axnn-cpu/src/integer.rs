//! Exact word arithmetic and explicit F32/I32 conversions for field graphs.

use crate::planner::{
    map_binary_broadcast, map_ternary_broadcast, typed_input_matches, typed_output_matches,
};
use crate::{CpuError, CpuImplementation, CpuKernelPlan};
use axnn::lowering::{
    CandidateVisitor, CapabilityMask, DTypeSet, IntegerBinaryOp, KernelAccuracy, KernelCandidate,
    KernelConstraints, KernelCost, KernelEffects, KernelParams, KernelRequest, KernelShapePolicy,
    SemanticOp,
};
use axnn::{BinaryOp, DType, TernaryOp, UnaryOp};

fn integer_math(op: SemanticOp) -> bool {
    matches!(
        op,
        SemanticOp::Unary(UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Sign)
            | SemanticOp::Binary(
                BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem
                    | BinaryOp::Min
                    | BinaryOp::Max
            )
            | SemanticOp::Ternary(TernaryOp::Clamp)
            | SemanticOp::IntegerBinary(_)
    )
}

pub(crate) fn visit_candidates<V: CandidateVisitor<CpuImplementation>>(
    request: &KernelRequest,
    visitor: &mut V,
) -> bool {
    let constraints = match request.semantic {
        SemanticOp::Cast => {
            if !matches!(
                (request.primary_input_dtype(), request.output_dtype),
                (DType::F32, DType::I32) | (DType::I32, DType::F32)
            ) {
                return true;
            }
            KernelConstraints {
                input_dtypes: [DTypeSet::one(request.primary_input_dtype()); 3],
                output_dtypes: DTypeSet::one(request.output_dtype),
                shape_policy: KernelShapePolicy::Exact,
                ..KernelConstraints::elementwise(request.output_dtype)
            }
        }
        SemanticOp::BufferLoadU8 => KernelConstraints {
            max_rank: 2,
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(DType::I32)
        },
        SemanticOp::Select if request.output_dtype == DType::I32 => KernelConstraints {
            shape_policy: KernelShapePolicy::Broadcast,
            ..KernelConstraints::select(DType::I32)
        },
        semantic if request.primary_input_dtype() == DType::I32 && integer_math(semantic) => {
            KernelConstraints {
                shape_policy: KernelShapePolicy::Broadcast,
                ..KernelConstraints::elementwise(DType::I32)
            }
        }
        _ => return false,
    };
    visitor.visit(KernelCandidate {
        semantic: request.semantic,
        implementation: CpuImplementation::Scalar,
        required_capabilities: CapabilityMask::NONE,
        constraints,
        accuracy: KernelAccuracy::Strict,
        cost: KernelCost::new(0, 12),
    });
    true
}

fn validate(
    plan: &CpuKernelPlan,
    inputs: &[(DType, usize)],
    output: (DType, usize),
) -> Result<(), CpuError> {
    let valid = plan.implementation == CpuImplementation::Scalar
        && plan.params() == KernelParams::None
        && plan.effects() == KernelEffects::PURE
        && plan.temporary_bytes == 0
        && plan.result_count() == 1
        && plan.output_dtype == output.0
        && plan
            .result_tensor(0)
            .is_some_and(|d| d.layout.offset() == 0)
        && plan.semantic.arity() as usize == inputs.len()
        && typed_output_matches(plan, 0, output.0, plan.shape, output.1)
        && inputs.iter().enumerate().all(|(i, &(dtype, n))| {
            typed_input_matches(plan, i, dtype, n)
                && plan.input(i).is_some_and(|d| d.layout.offset() == 0)
        })
        && (inputs.len()..3).all(|i| plan.input(i).is_none());
    if !valid {
        return Err(CpuError::ShapeMismatch {
            op: "planned integer field",
        });
    }
    let mut shape = plan.input_shapes[0].unwrap();
    for input in plan.input_shapes[1..inputs.len()].iter().flatten() {
        shape = shape.broadcast_with(input).ok_or(CpuError::ShapeMismatch {
            op: "integer broadcast",
        })?;
    }
    if shape != plan.shape {
        return Err(CpuError::ShapeMismatch {
            op: "integer broadcast result",
        });
    }
    Ok(())
}

/// Word arithmetic has no signed-overflow traps: +, -, *, abs and negate
/// wrap modulo 2^32. Division truncates toward zero, returning the numerator
/// on division by zero or MIN/-1; remainder returns zero for both exceptions.
pub fn invoke_integer_i32(
    plan: &CpuKernelPlan,
    first: &[i32],
    second: Option<&[i32]>,
    third: Option<&[i32]>,
    dst: &mut [i32],
) -> Result<(), CpuError> {
    if !integer_math(plan.semantic) {
        return Err(CpuError::UnsupportedOp {
            op: "planned integer math",
            dtype: plan.output_dtype,
        });
    }
    let arity = plan.semantic.arity() as usize;
    if second.is_some() != (arity >= 2) || third.is_some() != (arity == 3) {
        return Err(CpuError::ShapeMismatch {
            op: "integer arity",
        });
    }
    let inputs = [
        (DType::I32, first.len()),
        (DType::I32, second.map_or(0, |v| v.len())),
        (DType::I32, third.map_or(0, |v| v.len())),
    ];
    validate(plan, &inputs[..arity], (DType::I32, dst.len()))?;
    let lhs_shape = plan.input_shapes[0].unwrap();
    macro_rules! binary {
        ($f:expr) => {
            map_binary_broadcast(
                first,
                lhs_shape,
                second.unwrap(),
                plan.input_shapes[1].unwrap(),
                plan.shape,
                dst,
                $f,
            )
        };
    }
    match plan.semantic {
        SemanticOp::Unary(op) => {
            for (out, &value) in dst.iter_mut().zip(first) {
                *out = match op {
                    UnaryOp::Neg => value.wrapping_neg(),
                    UnaryOp::Abs => value.wrapping_abs(),
                    UnaryOp::Sign => value.signum(),
                    _ => unreachable!(),
                };
            }
        }
        SemanticOp::Binary(BinaryOp::Add) => binary!(i32::wrapping_add),
        SemanticOp::Binary(BinaryOp::Sub) => binary!(i32::wrapping_sub),
        SemanticOp::Binary(BinaryOp::Mul) => binary!(i32::wrapping_mul),
        SemanticOp::Binary(BinaryOp::Div) => {
            binary!(|a: i32, b: i32| a.checked_div(b).unwrap_or(a))
        }
        SemanticOp::Binary(BinaryOp::Rem) => {
            binary!(|a: i32, b: i32| a.checked_rem(b).unwrap_or(0))
        }
        SemanticOp::Binary(BinaryOp::Min) => binary!(i32::min),
        SemanticOp::Binary(BinaryOp::Max) => binary!(i32::max),
        SemanticOp::IntegerBinary(IntegerBinaryOp::And) => binary!(|a, b| a & b),
        SemanticOp::IntegerBinary(IntegerBinaryOp::Or) => binary!(|a, b| a | b),
        SemanticOp::IntegerBinary(IntegerBinaryOp::Xor) => binary!(|a, b| a ^ b),
        SemanticOp::IntegerBinary(IntegerBinaryOp::ShiftLeft) => {
            binary!(|a: i32, b: i32| a.wrapping_shl(b as u32))
        }
        SemanticOp::IntegerBinary(IntegerBinaryOp::ShiftRight) => {
            binary!(|a: i32, b: i32| a.wrapping_shr(b as u32))
        }
        SemanticOp::Ternary(TernaryOp::Clamp) => map_ternary_broadcast(
            (first, lhs_shape),
            (second.unwrap(), plan.input_shapes[1].unwrap()),
            (third.unwrap(), plan.input_shapes[2].unwrap()),
            plan.shape,
            dst,
            |v: i32, lo: i32, hi: i32| v.max(lo).min(hi),
        ),
        _ => unreachable!("integer family checked before output writes"),
    }
    Ok(())
}

pub fn invoke_select_i32(
    plan: &CpuKernelPlan,
    mask: &[u8],
    on_true: &[i32],
    on_false: &[i32],
    dst: &mut [i32],
) -> Result<(), CpuError> {
    if plan.semantic != SemanticOp::Select {
        return Err(CpuError::ShapeMismatch {
            op: "integer select",
        });
    }
    validate(
        plan,
        &[
            (DType::U8, mask.len()),
            (DType::I32, on_true.len()),
            (DType::I32, on_false.len()),
        ],
        (DType::I32, dst.len()),
    )?;
    map_ternary_broadcast(
        (mask, plan.input_shapes[0].unwrap()),
        (on_true, plan.input_shapes[1].unwrap()),
        (on_false, plan.input_shapes[2].unwrap()),
        plan.shape,
        dst,
        |m, yes, no| if m != 0 { yes } else { no },
    );
    Ok(())
}

pub fn invoke_cast_f32_i32(
    plan: &CpuKernelPlan,
    input: &[f32],
    dst: &mut [i32],
) -> Result<(), CpuError> {
    if plan.semantic != SemanticOp::Cast {
        return Err(CpuError::ShapeMismatch { op: "F32 to I32" });
    }
    validate(plan, &[(DType::F32, input.len())], (DType::I32, dst.len()))?;
    for (out, &value) in dst.iter_mut().zip(input) {
        *out = value as i32;
    }
    Ok(())
}

pub fn invoke_cast_i32_f32(
    plan: &CpuKernelPlan,
    input: &[i32],
    dst: &mut [f32],
) -> Result<(), CpuError> {
    if plan.semantic != SemanticOp::Cast {
        return Err(CpuError::ShapeMismatch { op: "I32 to F32" });
    }
    validate(plan, &[(DType::I32, input.len())], (DType::F32, dst.len()))?;
    for (out, &value) in dst.iter_mut().zip(input) {
        *out = value as f32;
    }
    Ok(())
}

/// Cast three dense F32 planes to I32, shift each word, then combine with OR.
///
/// This is a storage sink for an already-checked CAST/SHIFT_LEFT/BIT_OR region,
/// not a color conversion: no clamping to channel widths, scaling, bias or
/// alpha bits are added. Casts follow Rust's saturating `as i32` semantics
/// (NaN becomes zero); shift counts are reduced modulo 32. A zero shift models
/// a directly ORed cast. The scalar loop respects a scalar-only CPU policy and
/// does not enter any architecture-dispatched helper or allocate storage.
/// Invalid lengths leave the destination unchanged.
pub fn cast_shift_or3_f32_i32(
    planes: [&[f32]; 3],
    shifts: [u32; 3],
    dst: &mut [i32],
) -> Result<(), CpuError> {
    if planes.iter().any(|plane| plane.len() != dst.len()) {
        return Err(CpuError::ShapeMismatch {
            op: "cast/shift/or3 field",
        });
    }
    for (index, out) in dst.iter_mut().enumerate() {
        *out = (planes[0][index] as i32).wrapping_shl(shifts[0])
            | (planes[1][index] as i32).wrapping_shl(shifts[1])
            | (planes[2][index] as i32).wrapping_shl(shifts[2]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuCapabilities, plan_cpu_kernel};
    use axnn::{Shape, TensorDesc, shape};

    #[test]
    fn cast_shift_or3_matches_exact_cast_words_for_short_and_tail_lengths() {
        let samples = [
            0.0,
            -0.0,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffc0_5678),
            f32::INFINITY,
            f32::NEG_INFINITY,
            2_147_483_520.0,
            2_147_483_648.0,
            -2_147_483_648.0,
            -2_147_483_904.0,
            16_777_216.0,
            255.75,
            -255.75,
            f32::from_bits(1),
            -f32::from_bits(1),
            1.999_999_9,
        ];
        // Explicit words make the oracle independent of a repeated float cast.
        let words = [
            0,
            0,
            0,
            0,
            i32::MAX,
            i32::MIN,
            2_147_483_520,
            i32::MAX,
            i32::MIN,
            i32::MIN,
            16_777_216,
            255,
            -255,
            0,
            0,
            1,
        ];
        let planes: [[f32; 35]; 3] = core::array::from_fn(|axis| {
            core::array::from_fn(|lane| samples[(lane + axis * 5) % samples.len()])
        });
        for len in 0..=35 {
            for shifts in [
                [16, 8, 0],
                [32, 33, u32::MAX],
                [0, 31, 64],
                [u32::MAX, 0, 96],
            ] {
                let mut output = [0x1234_5678; 37];
                cast_shift_or3_f32_i32(
                    planes.each_ref().map(|plane| &plane[..len]),
                    shifts,
                    &mut output[1..=len],
                )
                .unwrap();
                assert_eq!(output[0], 0x1234_5678);
                assert!(output[len + 1..].iter().all(|&word| word == 0x1234_5678));
                for lane in 0..len {
                    let expected = (0..3).fold(0_u32, |word, axis| {
                        word | ((words[(lane + axis * 5) % words.len()] as u32)
                            << (shifts[axis] & 31))
                    });
                    assert_eq!(output[lane + 1] as u32, expected, "len {len}, lane {lane}");
                }
            }
        }
    }

    #[test]
    fn cast_shift_or3_rejects_each_short_plane_without_output_writes() {
        let plane = [1.0, 2.0, 3.0];
        for short in 0..3 {
            let mut inputs = [&plane[..]; 3];
            inputs[short] = &plane[..2];
            let mut output = [0x1234_5678; 3];
            assert!(cast_shift_or3_f32_i32(inputs, [16, 8, 0], &mut output).is_err());
            assert_eq!(output, [0x1234_5678; 3]);
        }
    }

    #[test]
    fn integer_math_broadcasts_preserves_words_and_defines_division_edges() {
        let lhs = [i32::MIN, i32::MAX, 16_777_217];
        let rhs = [-1, 0, 2, 31];
        let l = TensorDesc::new(shape![3, 1], DType::I32);
        let r = TensorDesc::new(shape![1, 4], DType::I32);
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Rem,
            BinaryOp::Min,
            BinaryOp::Max,
        ] {
            let request = KernelRequest::binary(op, &l, &r).unwrap();
            let plan = plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap();
            assert_eq!(plan.accuracy, KernelAccuracy::Strict);
            let mut actual = [0; 12];
            invoke_integer_i32(&plan, &lhs, Some(&rhs), None, &mut actual).unwrap();
            for (row, &a) in lhs.iter().enumerate() {
                for (col, &b) in rhs.iter().enumerate() {
                    let expected = match op {
                        BinaryOp::Add => (i64::from(a) + i64::from(b)) as i32,
                        BinaryOp::Sub => (i64::from(a) - i64::from(b)) as i32,
                        BinaryOp::Mul => (i64::from(a) * i64::from(b)) as i32,
                        BinaryOp::Div => {
                            if b == 0 {
                                a
                            } else {
                                (i64::from(a) / i64::from(b)) as i32
                            }
                        }
                        BinaryOp::Rem => {
                            if b == 0 {
                                0
                            } else {
                                (i64::from(a) % i64::from(b)) as i32
                            }
                        }
                        BinaryOp::Min => {
                            if a < b {
                                a
                            } else {
                                b
                            }
                        }
                        BinaryOp::Max => {
                            if a > b {
                                a
                            } else {
                                b
                            }
                        }
                        _ => unreachable!(),
                    };
                    assert_eq!(actual[row * 4 + col], expected, "{op:?} {a} {b}");
                }
            }
            let before = actual;
            assert!(invoke_integer_i32(&plan, &lhs, Some(&rhs[..3]), None, &mut actual).is_err());
            assert_eq!(actual, before);
        }
    }

    #[test]
    fn bit_operations_and_shift_counts_keep_all_32_bits() {
        let desc = TensorDesc::new(shape![1, 5], DType::I32);
        let lhs = [i32::MIN, -1, 16_777_217, 0x1234_5678, 1];
        let rhs = [0, 1, 32, -1, 63];
        for (op, expected) in [
            (
                IntegerBinaryOp::ShiftLeft,
                [i32::MIN, -2, 16_777_217, 0, i32::MIN],
            ),
            (
                IntegerBinaryOp::ShiftRight,
                [i32::MIN, -1, 16_777_217, 0, 0],
            ),
        ] {
            let request = KernelRequest::integer_binary(op, &desc, &desc).unwrap();
            let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
            let mut out = [0; 5];
            invoke_integer_i32(&plan, &lhs, Some(&rhs), None, &mut out).unwrap();
            assert_eq!(out, expected);
        }
        for op in [
            IntegerBinaryOp::And,
            IntegerBinaryOp::Or,
            IntegerBinaryOp::Xor,
        ] {
            let request = KernelRequest::integer_binary(op, &desc, &desc).unwrap();
            let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
            let mut out = [0; 5];
            invoke_integer_i32(&plan, &lhs, Some(&rhs), None, &mut out).unwrap();
            for i in 0..5 {
                let mut bits = 0_u32;
                for bit in 0..32 {
                    let a = (lhs[i] as u32 >> bit) & 1 != 0;
                    let b = (rhs[i] as u32 >> bit) & 1 != 0;
                    let value = match op {
                        IntegerBinaryOp::And => a && b,
                        IntegerBinaryOp::Or => a || b,
                        _ => a != b,
                    };
                    bits |= u32::from(value) << bit;
                }
                assert_eq!(out[i] as u32, bits);
            }
        }
    }

    #[test]
    fn explicit_casts_cover_saturation_rounding_and_invalid_descriptors() {
        let input = [
            f32::NEG_INFINITY,
            -2_147_483_648.0,
            -3.9,
            -0.0,
            f32::NAN,
            3.9,
            2_147_483_520.0,
            f32::INFINITY,
        ];
        let shape = shape![2, 4];
        let request = KernelRequest::cast(&TensorDesc::new(shape, DType::F32), DType::I32).unwrap();
        let mut plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
        let mut out = [-9; 8];
        invoke_cast_f32_i32(&plan, &input, &mut out).unwrap();
        assert_eq!(
            out,
            [i32::MIN, i32::MIN, -3, 0, 0, 3, 2_147_483_520, i32::MAX]
        );
        let before = out;
        plan.output_dtype = DType::F32;
        assert!(invoke_cast_f32_i32(&plan, &input, &mut out).is_err());
        assert_eq!(out, before);
        let input = [
            16_777_215,
            16_777_216,
            16_777_217,
            16_777_219,
            i32::MIN,
            i32::MAX,
        ];
        let request =
            KernelRequest::cast(&TensorDesc::new(shape![2, 3], DType::I32), DType::F32).unwrap();
        let plan = plan_cpu_kernel(&request, CpuCapabilities::SCALAR).unwrap();
        let mut out = [0.0; 6];
        invoke_cast_i32_f32(&plan, &input, &mut out).unwrap();
        assert_eq!(
            out,
            [
                16_777_215.0,
                16_777_216.0,
                16_777_216.0,
                16_777_220.0,
                -2_147_483_648.0,
                2_147_483_648.0
            ]
        );
        assert!(
            KernelRequest::cast(&TensorDesc::new(Shape::SCALAR, DType::U8), DType::F32).is_none()
        );
    }
}
