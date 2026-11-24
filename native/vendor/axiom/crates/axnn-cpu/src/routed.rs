//! Mixture-of-experts over an expert table of graph weights: each token's
//! route from its router logits, then the routed NVFP4 SwiGLU and down
//! projection (`axnn::lowering::{MoeRouteParams, RoutedExpertsParams}`).
//!
//! An expert's projection is one [`ExpertRecord`] of its table: tensor
//! scales, E2M1 code rows, E4M3 block scales. Its products are the plain
//! NVFP4 matmul's row dots (`arch::f4::nvfp4_f32`), bit for bit, each
//! scaled by its run's tensor scale.

use crate::fork_join::{ForkJoin, for_rows};
use crate::tensor_ops::select_top_k;
use crate::{CpuError, CpuImplementation, CpuKernelPlan, arch};
use axnn::lowering::{
    CandidateVisitor, CapabilityMask, DTypeSet, ExpertRecord, KernelAccuracy, KernelCandidate,
    KernelConstraints, KernelCost, KernelEffects, KernelParams, KernelRequest, KernelShapePolicy,
    MAX_MOE_TOP_K, MoeRouteParams, RoutedExpertsParams, SemanticOp,
};
use axnn::{DType, TensorDesc};

/// Rows of one expert projection computed per stack-held chunk.
const CHUNK: usize = 64;

/// Offer the scalar routing and routed-projection kernels.
pub(crate) fn visit_candidates<V: CandidateVisitor<CpuImplementation>>(
    request: &KernelRequest,
    visitor: &mut V,
) -> bool {
    let (inputs, output) = match request.semantic {
        SemanticOp::MoeRoute => (
            [DTypeSet::one(DType::F32), DTypeSet::NONE, DTypeSet::NONE],
            DTypeSet::one(DType::I32),
        ),
        SemanticOp::MoeGateUpNvfp4 | SemanticOp::MoeDownNvfp4 => (
            [
                DTypeSet::one(DType::F32),
                DTypeSet::one(DType::U8),
                DTypeSet::one(DType::I32),
            ],
            DTypeSet::one(DType::F32),
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
            max_rank: 3,
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(DType::F32)
        },
        // Exponentials and sums in a fixed order, but transcendental.
        accuracy: KernelAccuracy::Faithful,
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

/// Each token's route ([`MoeRouteParams`]): its `top_k` largest routed
/// logits in descending `f32::total_cmp` order (ties to the lower index),
/// weighted by a softmax over those logits, then each shared expert with
/// the sigmoid of its gate logit. Writes `(expert, weight bits)` slots.
pub fn invoke_moe_route(
    plan: &CpuKernelPlan,
    logits: &[f32],
    routes: &mut [i32],
) -> Result<(), CpuError> {
    const OP: &str = "planned MoE route";
    let (Some(desc), KernelParams::MoeRoute(params)) = (plan.input(0), plan.params()) else {
        return Err(invalid(OP));
    };
    checked(plan, KernelRequest::moe_route(desc, params), OP)?;
    let (width, slots) = (params.logits() as usize, params.slots() as usize);
    let tokens = desc.shape.dim(0);
    if logits.len() != tokens * width || routes.len() != tokens * slots * 2 {
        return Err(invalid(OP));
    }
    for (row, route) in logits
        .chunks_exact(width)
        .zip(routes.chunks_exact_mut(slots * 2))
    {
        route_token(params, row, route);
    }
    Ok(())
}

/// One token's route into `route` (`slots * 2` words).
fn route_token(params: MoeRouteParams, logits: &[f32], route: &mut [i32]) {
    let (experts, k) = (params.experts as usize, params.top_k as usize);
    let mut values = [0.0_f32; MAX_MOE_TOP_K as usize];
    let mut indices = [0_i32; MAX_MOE_TOP_K as usize];
    select_top_k(
        logits[..experts].iter().copied(),
        &mut values[..k],
        &mut indices[..k],
    );
    // Softmax over the picks, from the largest: the renormalized softmax
    // probabilities of every routed expert.
    let largest = values[0];
    let mut sum = 0.0_f32;
    for value in &mut values[..k] {
        *value = axmath::exp_f32(*value - largest);
        sum += *value;
    }
    for (slot, (&index, &value)) in indices[..k].iter().zip(&values[..k]).enumerate() {
        route[slot * 2] = index;
        route[slot * 2 + 1] = (value / sum).to_bits() as i32;
    }
    for shared in 0..params.shared as usize {
        let slot = k + shared;
        route[slot * 2] = (experts + shared) as i32;
        route[slot * 2 + 1] = axmath::sigmoid_f32(logits[experts + shared]).to_bits() as i32;
    }
}

/// An expert's record within `table`, checked against the table.
fn record<'t>(
    table: &'t [u8],
    stride: usize,
    experts: usize,
    expert: i32,
) -> Option<&'t [u8]> {
    let expert = usize::try_from(expert).ok().filter(|&e| e < experts)?;
    table.get(expert * stride..(expert + 1) * stride)
}

/// The tensor scale of run `index` of a record.
fn tensor_scale(record: &[u8], index: usize) -> f32 {
    f32::from_le_bytes(record[index * 4..index * 4 + 4].try_into().unwrap())
}

/// Rows `first..first + out.len()` of a record's projection against `x`.
fn record_rows(record: &[u8], layout: ExpertRecord, first: usize, x: &[f32], out: &mut [f32]) {
    let cols = layout.cols as usize;
    let codes = &record[ExpertRecord::HEADER + first * cols / 2..];
    // Checked when the plan was validated: the record holds every row.
    let scales_at = ExpertRecord::HEADER + layout.rows as usize * cols / 2;
    let scales = &record[scales_at + first * cols / 16..];
    arch::f4::nvfp4_f32(codes, scales, x, out);
}

/// The routes' `(expert, weight)` slots of one token, refused when an
/// expert lies outside the table.
fn slots(route: &[i32], experts: usize) -> Option<impl Iterator<Item = (i32, f32)> + '_> {
    route
        .chunks_exact(2)
        .all(|slot| usize::try_from(slot[0]).is_ok_and(|e| e < experts))
        .then(|| {
            route
                .chunks_exact(2)
                .map(|slot| (slot[0], f32::from_bits(slot[1] as u32)))
        })
}

fn routed_params(plan: &CpuKernelPlan, op: &'static str) -> Result<RoutedExpertsParams, CpuError> {
    match plan.params() {
        KernelParams::RoutedExperts(params) => Ok(params),
        _ => Err(invalid(op)),
    }
}

/// The routed SwiGLU ([`RoutedExpertsParams`]): for each token and slot,
/// `silu(gate(x)) * up(x)` of the slot's expert, each product scaled by its
/// run's tensor scale. `routes` must name experts inside the table.
pub fn invoke_moe_gate_up_nvfp4(
    plan: &CpuKernelPlan,
    x: &[f32],
    experts: &[u8],
    routes: &[i32],
    output: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    const OP: &str = "planned MoE gate/up";
    let params = routed_params(plan, OP)?;
    let (Some(xd), Some(ed), Some(rd)) = (plan.input(0), plan.input(1), plan.input(2)) else {
        return Err(invalid(OP));
    };
    checked(plan, KernelRequest::moe_gate_up_nvfp4(xd, ed, rd, params), OP)?;
    let layout = params.gate_up_record();
    let stride = layout.bytes().ok_or_else(|| invalid(OP))?;
    let (hidden, inner, slot_count) = (
        params.hidden as usize,
        params.intermediate as usize,
        params.slots as usize,
    );
    let (count, tokens) = (params.experts as usize, xd.shape.dim(0));
    if x.len() != tokens * hidden
        || experts.len() != count * stride
        || routes.len() != tokens * slot_count * 2
        || output.len() != tokens * slot_count * inner
    {
        return Err(invalid(OP));
    }
    for route in routes.chunks_exact(slot_count * 2) {
        slots(route, count).ok_or_else(|| invalid("MoE route outside the expert table"))?;
    }
    // One output row a (token, slot) pair.
    for_rows(parallel, output, inner, 2 * inner * hidden, |first, rows| {
        for (offset, row) in rows.chunks_exact_mut(inner).enumerate() {
            let pair = first + offset;
            let token = pair / slot_count;
            let record = record(experts, stride, count, routes[pair * 2]).expect("checked above");
            let (gate_scale, up_scale) = (tensor_scale(record, 0), tensor_scale(record, 1));
            let input = &x[token * hidden..(token + 1) * hidden];
            let mut up = [0.0_f32; CHUNK];
            for (chunk, out) in row.chunks_mut(CHUNK).enumerate() {
                let first_row = chunk * CHUNK;
                let up = &mut up[..out.len()];
                record_rows(record, layout, first_row, input, out);
                record_rows(record, layout, inner + first_row, input, up);
                for (value, &up) in out.iter_mut().zip(up.iter()) {
                    *value = axmath::silu_f32(*value * gate_scale) * (up * up_scale);
                }
            }
        }
    });
    Ok(())
}

/// The routed down projection ([`RoutedExpertsParams`]): for each token,
/// each slot's expert's product of its `h` row, scaled by the expert's
/// tensor scale and the slot's weight, summed in slot order.
pub fn invoke_moe_down_nvfp4(
    plan: &CpuKernelPlan,
    h: &[f32],
    experts: &[u8],
    routes: &[i32],
    output: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    const OP: &str = "planned MoE down";
    let params = routed_params(plan, OP)?;
    let (Some(hd), Some(ed), Some(rd)) = (plan.input(0), plan.input(1), plan.input(2)) else {
        return Err(invalid(OP));
    };
    checked(plan, KernelRequest::moe_down_nvfp4(hd, ed, rd, params), OP)?;
    let layout = params.down_record();
    let stride = layout.bytes().ok_or_else(|| invalid(OP))?;
    let (hidden, inner, slot_count) = (
        params.hidden as usize,
        params.intermediate as usize,
        params.slots as usize,
    );
    let (count, tokens) = (params.experts as usize, hd.shape.dim(0));
    if h.len() != tokens * slot_count * inner
        || experts.len() != count * stride
        || routes.len() != tokens * slot_count * 2
        || output.len() != tokens * hidden
    {
        return Err(invalid(OP));
    }
    for route in routes.chunks_exact(slot_count * 2) {
        slots(route, count).ok_or_else(|| invalid("MoE route outside the expert table"))?;
    }
    // One output row a token.
    for_rows(
        parallel,
        output,
        hidden,
        slot_count * hidden * inner,
        |first, rows| {
            for (offset, row) in rows.chunks_exact_mut(hidden).enumerate() {
                let token = first + offset;
                let route = &routes[token * slot_count * 2..(token + 1) * slot_count * 2];
                row.fill(0.0);
                let mut product = [0.0_f32; CHUNK];
                for (slot, (expert, weight)) in slots(route, count).expect("checked above").enumerate() {
                    let record = record(experts, stride, count, expert).expect("checked above");
                    let scale = tensor_scale(record, 0) * weight;
                    let input = &h[(token * slot_count + slot) * inner..][..inner];
                    for (chunk, out) in row.chunks_mut(CHUNK).enumerate() {
                        let product = &mut product[..out.len()];
                        record_rows(record, layout, chunk * CHUNK, input, product);
                        for (value, &product) in out.iter_mut().zip(product.iter()) {
                            *value += product * scale;
                        }
                    }
                }
            }
        },
    );
    Ok(())
}
