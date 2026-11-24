//! Data-movement and selection kernels a decoder step needs between its
//! matrix products: embedding gathers and top-k selection. Views need no
//! kernel: the executor aliases their input.
//! All are exact, allocation-free, and validate before any output write.

use crate::quant::dequantize_q8;
use crate::{CpuError, CpuImplementation, CpuKernelPlan};
use axnn::lowering::{
    CandidateVisitor, CapabilityMask, DTypeSet, KernelAccuracy, KernelCandidate, KernelConstraints,
    KernelCost, KernelEffects, KernelParams, KernelRequest, KernelShapePolicy, SemanticOp,
};
use axnn::{DType, TensorDesc};

/// Offer the exact scalar kernels for gather and top-k requests, and admit
/// views.
pub(crate) fn visit_candidates<V: CandidateVisitor<CpuImplementation>>(
    request: &KernelRequest,
    visitor: &mut V,
) -> bool {
    let (inputs, output) = match request.semantic {
        SemanticOp::Gather => (
            [
                DTypeSet::one(DType::F32)
                    .union(DTypeSet::one(DType::Q8))
                    .union(DTypeSet::one(DType::B16)),
                DTypeSet::one(DType::I32),
                DTypeSet::NONE,
            ],
            DTypeSet::one(DType::F32),
        ),
        // The executor aliases a view's input; no kernel runs. The
        // candidate only admits views to native placement.
        SemanticOp::View => {
            let stored = [
                DType::F32,
                DType::I32,
                DType::U8,
                DType::Q4,
                DType::Q6,
                DType::Q8,
                DType::B8,
                DType::B16,
            ]
            .into_iter()
            .fold(DTypeSet::NONE, |set, dtype| set.union(DTypeSet::one(dtype)));
            visitor.visit(KernelCandidate {
                semantic: request.semantic,
                implementation: CpuImplementation::Scalar,
                required_capabilities: CapabilityMask::NONE,
                constraints: KernelConstraints {
                    input_dtypes: [stored, DTypeSet::NONE, DTypeSet::NONE],
                    output_dtypes: stored,
                    max_rank: axnn::MAX_RANK as u8,
                    shape_policy: KernelShapePolicy::Typed,
                    ..KernelConstraints::elementwise(DType::F32)
                },
                accuracy: KernelAccuracy::Strict,
                cost: KernelCost::new(0, 0),
            });
            return true;
        }
        SemanticOp::TopK => (
            [DTypeSet::one(DType::F32), DTypeSet::NONE, DTypeSet::NONE],
            DTypeSet::one(DType::F32).union(DTypeSet::one(DType::I32)),
        ),
        _ => return false,
    };
    visitor.visit(KernelCandidate {
        semantic: request.semantic,
        implementation: CpuImplementation::Scalar,
        required_capabilities: CapabilityMask::NONE,
        constraints: KernelConstraints {
            input_dtypes: inputs,
            output_dtypes: output,
            max_rank: axnn::MAX_RANK as u8,
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(DType::F32)
        },
        accuracy: KernelAccuracy::Strict,
        cost: KernelCost::new(0, 1),
    });
    true
}

fn invalid(op: &'static str) -> CpuError {
    CpuError::ShapeMismatch { op }
}

/// The plan's request, rebuilt from its own descriptors, must be the plan:
/// the scalar implementation, pure, contiguous and zero-offset.
fn checked(
    plan: &CpuKernelPlan,
    rebuilt: Option<KernelRequest>,
    op: &'static str,
) -> Result<(), CpuError> {
    let rebuilt = rebuilt.ok_or_else(|| invalid(op))?;
    let flat = |d: &TensorDesc| d.is_contiguous() && d.layout.offset() == 0;
    if rebuilt.result() != plan.result()
        || rebuilt.inputs() != plan.inputs()
        || rebuilt.params() != plan.params()
        || plan.effects() != KernelEffects::PURE
        || plan.implementation != CpuImplementation::Scalar
        || !plan.inputs().iter().flatten().all(flat)
        || !(0..plan.result_count()).all(|i| plan.result_tensor(i).is_some_and(flat))
    {
        return Err(invalid(op));
    }
    Ok(())
}

/// A gather table: F32 rows, packed Q8 blocks with whole blocks per row, or
/// little-endian BF16 rows. Every encoding decodes exactly.
#[derive(Clone, Copy, Debug)]
pub enum GatherTable<'a> {
    F32(&'a [f32]),
    Q8(&'a [u8]),
    B16(&'a [u8]),
}

/// Copy (or dequantize) the rows of `table` named by `indices` into `output`.
///
/// Every index is checked against the row count before any row is written.
pub fn invoke_gather(
    plan: &CpuKernelPlan,
    table: GatherTable<'_>,
    indices: &[i32],
    output: &mut [f32],
) -> Result<(), CpuError> {
    const OP: &str = "planned gather";
    let (Some(desc), Some(index_desc)) = (plan.input(0), plan.input(1)) else {
        return Err(invalid(OP));
    };
    checked(plan, KernelRequest::gather(desc, index_desc), OP)?;
    let (rows, width) = (desc.shape.dim(0), desc.shape.dim(1));
    let row_bytes = desc.dtype.storage_bytes(width);
    let table_ok = match (table, desc.dtype) {
        (GatherTable::F32(values), DType::F32) => Some(values.len()) == rows.checked_mul(width),
        (GatherTable::Q8(bytes), DType::Q8) | (GatherTable::B16(bytes), DType::B16) => {
            Some(bytes.len()) == rows.checked_mul(row_bytes)
        }
        _ => false,
    };
    if !table_ok
        || Some(indices.len()) != index_desc.shape.checked_num_elements()
        || Some(output.len()) != indices.len().checked_mul(width)
    {
        return Err(invalid(OP));
    }
    if indices
        .iter()
        .any(|&index| usize::try_from(index).map_or(true, |index| index >= rows))
    {
        return Err(invalid("gather index out of range"));
    }
    if width == 0 {
        return Ok(());
    }
    for (&index, row) in indices.iter().zip(output.chunks_exact_mut(width)) {
        let index = index as usize;
        match table {
            GatherTable::F32(values) => {
                row.copy_from_slice(&values[index * width..(index + 1) * width]);
            }
            GatherTable::Q8(bytes) => {
                dequantize_q8(&bytes[index * row_bytes..(index + 1) * row_bytes], row);
            }
            GatherTable::B16(bytes) => {
                let source = &bytes[index * row_bytes..(index + 1) * row_bytes];
                for (value, pair) in row.iter_mut().zip(source.as_chunks::<2>().0) {
                    *value = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16);
                }
            }
        }
    }
    Ok(())
}

/// The `k` largest values of each last-axis row, in descending
/// `f32::total_cmp` order with ties to the lower index, and their indices.
///
/// Selection keeps a sorted `k`-element prefix per row, so rows cost
/// `O(width * k)`: meant for router widths and small `k`.
pub fn invoke_top_k(
    plan: &CpuKernelPlan,
    input: &[f32],
    values: &mut [f32],
    indices: &mut [i32],
) -> Result<(), CpuError> {
    const OP: &str = "planned top-k";
    let (Some(desc), KernelParams::TopK { k }) = (plan.input(0), plan.params()) else {
        return Err(invalid(OP));
    };
    checked(plan, KernelRequest::top_k(desc, k), OP)?;
    let (width, k) = (desc.shape.dim(-1), k as usize);
    let rows = desc
        .shape
        .checked_num_elements()
        .ok_or_else(|| invalid(OP))?
        / width;
    if input.len() != rows * width || values.len() != rows * k || indices.len() != rows * k {
        return Err(invalid(OP));
    }
    for ((row, top_values), top_indices) in input
        .chunks_exact(width)
        .zip(values.chunks_exact_mut(k))
        .zip(indices.chunks_exact_mut(k))
    {
        select_top_k(row.iter().copied(), top_values, top_indices);
    }
    Ok(())
}

/// Keep the `values.len()` largest items of `row` in descending
/// `f32::total_cmp` order, ties to the lower index, with their indices.
///
/// A sorted prefix makes this `O(width * k)`: meant for small `k`. Fewer
/// items than `k` leave the tail untouched.
pub(crate) fn select_top_k(
    row: impl Iterator<Item = f32>,
    values: &mut [f32],
    indices: &mut [i32],
) {
    let k = values.len();
    let mut filled = 0;
    for (index, value) in row.enumerate() {
        // Strictly greater: an equal value never displaces a lower index.
        let slot = values[..filled]
            .iter()
            .position(|&kept| value.total_cmp(&kept).is_gt())
            .unwrap_or(filled);
        if slot == k {
            continue;
        }
        let end = (filled + 1).min(k);
        values.copy_within(slot..end - 1, slot + 1);
        indices.copy_within(slot..end - 1, slot + 1);
        values[slot] = value;
        indices[slot] = index as i32;
        filled = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuCapabilities, plan_cpu_kernel};
    use alloc::vec;
    use alloc::vec::Vec;
    use axnn::Shape;

    fn desc(dims: &[usize], dtype: DType) -> TensorDesc {
        TensorDesc::new(Shape::try_from(dims).unwrap(), dtype)
    }

    fn plan(request: KernelRequest) -> CpuKernelPlan {
        plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap()
    }

    #[test]
    fn gathers_f32_and_q8_rows_and_rejects_bad_indices_before_writing() {
        let table: Vec<f32> = (0..4 * 32).map(|i| i as f32 * 0.25 - 7.0).collect();
        let f32_plan = plan(
            KernelRequest::gather(&desc(&[4, 32], DType::F32), &desc(&[2, 1], DType::I32)).unwrap(),
        );
        assert_eq!(f32_plan.shape.dims(), &[2, 1, 32]);
        let mut out = vec![0.0; 64];
        invoke_gather(&f32_plan, GatherTable::F32(&table), &[3, 0], &mut out).unwrap();
        assert_eq!(out[..32], table[96..]);
        assert_eq!(out[32..], table[..32]);

        // Q8 rows decode exactly as a whole-tensor dequantization would.
        let mut packed = vec![0_u8; DType::Q8.storage_bytes(4 * 32)];
        crate::quant::quantize_q8(&table, &mut packed);
        let mut dense = vec![0.0; 4 * 32];
        dequantize_q8(&packed, &mut dense);
        let q8_plan = plan(
            KernelRequest::gather(&desc(&[4, 32], DType::Q8), &desc(&[3], DType::I32)).unwrap(),
        );
        let mut out = vec![0.0; 96];
        invoke_gather(&q8_plan, GatherTable::Q8(&packed), &[1, 1, 2], &mut out).unwrap();
        assert_eq!(out[..32], dense[32..64]);
        assert_eq!(out[32..64], dense[32..64]);
        assert_eq!(out[64..], dense[64..96]);

        for bad in [[4, 0, 0], [0, -1, 0]] {
            let mut out = vec![9.0; 96];
            assert!(invoke_gather(&q8_plan, GatherTable::Q8(&packed), &bad, &mut out).is_err());
            assert!(out.iter().all(|&v| v == 9.0));
        }
        assert!(invoke_gather(&q8_plan, GatherTable::F32(&table), &[0, 0, 0], &mut out).is_err());
        // Q8 rows must hold whole 32-element blocks.
        assert!(
            KernelRequest::gather(&desc(&[4, 48], DType::Q8), &desc(&[1], DType::I32)).is_none()
        );
    }

    #[test]
    fn views_keep_their_bytes_and_whole_blocks() {
        let view = KernelRequest::view(&desc(&[2, 64], DType::Q8), &desc(&[1, 32], DType::Q8), 64)
            .unwrap();
        assert_eq!(view.semantic, SemanticOp::View);
        assert!(plan_cpu_kernel(&view, CpuCapabilities::compiled()).is_some());
        let f32_input = desc(&[1, 12], DType::F32);
        assert!(KernelRequest::view(&f32_input, &desc(&[3, 4], DType::F32), 0).is_some());
        assert!(KernelRequest::view(&f32_input, &desc(&[1, 4], DType::F32), 8).is_some());
        for (output, offset) in [
            (desc(&[1, 4], DType::F32), 9),
            (desc(&[3, 4], DType::I32), 0),
        ] {
            assert!(KernelRequest::view(&f32_input, &output, offset).is_none());
        }
        let packed = desc(&[2, 64], DType::Q8);
        assert!(KernelRequest::view(&packed, &desc(&[1, 32], DType::Q8), 16).is_none());
        assert!(KernelRequest::view(&packed, &desc(&[2, 16], DType::Q8), 0).is_none());
    }

    #[test]
    fn top_k_orders_by_total_order_with_ties_to_the_lower_index() {
        let row = [0.5_f32, 2.0, -1.0, 2.0, f32::NEG_INFINITY, 0.5, -0.0, 0.0];
        let top = plan(KernelRequest::top_k(&desc(&[2, 8], DType::F32), 4).unwrap());
        let input: Vec<f32> = row.iter().chain(row.iter().rev()).copied().collect();
        let (mut values, mut indices) = (vec![0.0; 8], vec![0; 8]);
        invoke_top_k(&top, &input, &mut values, &mut indices).unwrap();
        assert_eq!(values[..4], [2.0, 2.0, 0.5, 0.5]);
        assert_eq!(indices[..4], [1, 3, 0, 5]);
        // Reversed row: index 7 - i.
        assert_eq!(indices[4..], [4, 6, 2, 7]);

        // k = width sorts the whole row; +0 ranks above -0 and NaN above all.
        let all = plan(KernelRequest::top_k(&desc(&[4], DType::F32), 4).unwrap());
        let (mut values, mut indices) = ([0.0; 4], [0; 4]);
        invoke_top_k(
            &all,
            &[-0.0, f32::NAN, 0.0, -3.0],
            &mut values,
            &mut indices,
        )
        .unwrap();
        assert_eq!(indices, [1, 2, 0, 3]);
        assert!(values[0].is_nan());
        assert!(values[2].is_sign_negative());

        assert!(KernelRequest::top_k(&desc(&[4], DType::F32), 0).is_none());
        assert!(KernelRequest::top_k(&desc(&[4], DType::F32), 5).is_none());
        assert!(KernelRequest::top_k(&desc(&[4], DType::I32), 1).is_none());
    }

    #[test]
    fn grouped_rms_norm_normalizes_each_row_with_its_weight_slice() {
        let input: Vec<f32> = (0..12).map(|i| (i as f32 - 5.5) * 0.5).collect();
        let weight: Vec<f32> = (0..12).map(|i| 1.0 + i as f32 * 0.1).collect();
        let grouped = plan(
            KernelRequest::rms_norm_over(
                &desc(&[3, 4], DType::F32),
                &desc(&[3, 4], DType::F32),
                1.0e-5,
                1,
            )
            .unwrap(),
        );
        let mut out = vec![0.0; 12];
        crate::invoke_neural_f32(&grouped, &input, Some(&weight), None, &mut out).unwrap();
        for group in 0..3 {
            let row = &input[group * 4..group * 4 + 4];
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1.0e-5).sqrt();
            for j in 0..4 {
                let expected = row[j] / rms * weight[group * 4 + j];
                let actual = out[group * 4 + j];
                assert!((actual - expected).abs() <= 1.0e-5 * expected.abs().max(1.0));
            }
        }
        // A trailing-axis weight broadcasts across leading rows.
        let shared = plan(
            KernelRequest::rms_norm_over(
                &desc(&[3, 4], DType::F32),
                &desc(&[4], DType::F32),
                1.0e-5,
                -1,
            )
            .unwrap(),
        );
        let mut broadcast = vec![0.0; 12];
        crate::invoke_neural_f32(&shared, &input, Some(&weight[..4]), None, &mut broadcast)
            .unwrap();
        let mut per_row = vec![0.0; 12];
        let tiled: Vec<f32> = weight[..4].iter().cycle().take(12).copied().collect();
        crate::invoke_neural_f32(&grouped, &input, Some(&tiled), None, &mut per_row).unwrap();
        assert_eq!(broadcast, per_row);
        // The weight must cover at least the normalized axes.
        assert!(
            KernelRequest::rms_norm_over(
                &desc(&[3, 4], DType::F32),
                &desc(&[4], DType::F32),
                1.0e-5,
                0,
            )
            .is_none()
        );
    }
}
