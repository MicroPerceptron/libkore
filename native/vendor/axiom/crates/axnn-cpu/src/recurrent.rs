//! Recurrences over carried state: a depthwise causal convolution window,
//! the Mamba-2 selective state-space step and the gated delta rule.
//!
//! Every step reads its state from a source buffer and writes the successor
//! to a distinct target buffer, never the source: the caller alternates the
//! two between steps, so a failed step leaves the previous state intact.
//! Several tokens in one invocation run the same per-token recurrence in
//! order, so a chunk equals the same tokens stepped one at a time.

use crate::fork_join::{ForkJoin, for_row_pairs, for_rows};
use crate::planner::{typed_input_matches, typed_output_matches};
use crate::{CpuError, CpuImplementation, CpuKernelPlan};
use axnn::DType;
use axnn::lowering::{
    CandidateVisitor, CapabilityMask, DTypeSet, EpilogueActivation, GatedDeltaRuleParams,
    KernelAccuracy, KernelCandidate, KernelConstraints, KernelCost, KernelEffects, KernelParams,
    KernelRequest, KernelShapePolicy, SemanticOp, TokenTree,
};
use core::marker::PhantomData;

/// The state before a step, borrowed read-only.
#[derive(Clone, Copy, Debug)]
pub struct StateSourceRef<'a> {
    values: &'a [f32],
    tree: Option<TokenTree<'a>>,
}

impl<'a> StateSourceRef<'a> {
    pub const fn new(values: &'a [f32]) -> Self {
        Self { values, tree: None }
    }

    /// Interpret token rows as this validated tree, retaining a read-only source.
    pub const fn with_tree(mut self, tree: TokenTree<'a>) -> Self {
        self.tree = Some(tree);
        self
    }

    pub const fn tree(self) -> Option<TokenTree<'a>> {
        self.tree
    }

    pub const fn len(self) -> usize {
        self.values.len()
    }

    pub const fn is_empty(self) -> bool {
        self.values.is_empty()
    }

    /// The state before the step.
    pub const fn values(self) -> &'a [f32] {
        self.values
    }
}

/// The buffer a step writes its successor state into: a copyable view of an
/// exclusive borrow, handed to one node at a time like the raster target.
#[derive(Clone, Copy, Debug)]
pub struct StateTargetRef<'a> {
    base: *mut f32,
    len: usize,
    _borrow: PhantomData<&'a mut [f32]>,
}

impl<'a> StateTargetRef<'a> {
    pub fn new(values: &'a mut [f32]) -> Self {
        Self {
            base: values.as_mut_ptr(),
            len: values.len(),
            _borrow: PhantomData,
        }
    }

    pub const fn len(self) -> usize {
        self.len
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Address metadata without creating a CPU reference to the contents.
    /// This grants no authority to retain the pointer or access the buffer:
    /// an asynchronous device executor needs independently retained storage
    /// and must exclude CPU access until its writes have retired.
    pub const fn as_mut_ptr(self) -> *mut f32 {
        self.base
    }

    /// # Safety
    /// No other view of this buffer is live while the result is. The
    /// executor runs one node at a time, and kernels (or a delegate offered
    /// the state) drop the view before returning.
    pub unsafe fn values(self) -> &'a mut [f32] {
        unsafe { core::slice::from_raw_parts_mut(self.base, self.len) }
    }
}

/// Offer the scalar recurrence kernels.
pub(crate) fn visit_candidates<V: CandidateVisitor<CpuImplementation>>(
    request: &KernelRequest,
    visitor: &mut V,
) -> bool {
    let accuracy = match request.semantic {
        // A fixed tap order. A fused activation (#107) adds transcendental
        // rounding on top, so an activated convolution is only Faithful.
        SemanticOp::CausalConv1d => match request.params() {
            KernelParams::CausalConv1d(params) if params.activation != EpilogueActivation::None => {
                KernelAccuracy::Faithful
            }
            _ => KernelAccuracy::Strict,
        },
        // The readout sums over the state in an unspecified order.
        SemanticOp::SsmScan | SemanticOp::GatedDeltaRule => KernelAccuracy::Faithful,
        _ => return false,
    };
    visitor.visit(KernelCandidate {
        semantic: request.semantic,
        implementation: CpuImplementation::Scalar,
        required_capabilities: CapabilityMask::NONE,
        constraints: KernelConstraints {
            input_dtypes: [DTypeSet::one(DType::F32); 3],
            output_dtypes: DTypeSet::one(DType::F32),
            max_rank: 2,
            shape_policy: KernelShapePolicy::Typed,
            ..KernelConstraints::elementwise(DType::F32)
        },
        accuracy,
        cost: KernelCost::new(0, 2),
    });
    true
}

fn invalid(op: &'static str) -> CpuError {
    CpuError::ShapeMismatch { op }
}

/// Descriptors shared by every recurrence: scalar, updating a resource,
/// scratch-free, and every operand dense F32 at offset zero.
fn check_plan(
    plan: &CpuKernelPlan,
    semantic: SemanticOp,
    operands: [usize; 3],
    output: usize,
    op: &'static str,
) -> Result<(), CpuError> {
    if plan.semantic != semantic
        || plan.implementation != CpuImplementation::Scalar
        || plan.effects() != KernelEffects::UPDATES_RESOURCE
        || plan.temporary_bytes != 0
        || plan.result_count() != 1
        || (0..3).any(|i| !typed_input_matches(plan, i, DType::F32, operands[i]))
        || (0..3).any(|i| plan.input(i).is_none_or(|d| d.layout.offset() != 0))
        || !typed_output_matches(plan, 0, DType::F32, plan.shape, output)
        || plan.result_tensor(0).is_none_or(|d| d.layout.offset() != 0)
    {
        return Err(invalid(op));
    }
    Ok(())
}

/// Depthwise causal convolution continuing the source window.
///
/// For each token `t` and channel `c`, with `w = weight[c]` (oldest tap
/// first) and window `prev` of the previous `kernel - 1` inputs:
/// `y = (sum_i w[i] * prev[i]) + w[kernel - 1] * x[t, c] + bias[c]`,
/// summed in that order. The target receives the window after the last
/// token.
///
/// `params.activation` is applied to each element after its bias (#107),
/// folding a following `silu` into this kernel. Taps read the input rows
/// and the source window, never the output, so the activation cannot
/// affect a later token's sum.
#[allow(clippy::too_many_arguments)]
pub fn invoke_causal_conv1d(
    plan: &CpuKernelPlan,
    source: StateSourceRef<'_>,
    target: StateTargetRef<'_>,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    const OP: &str = "planned causal convolution";
    let KernelParams::CausalConv1d(params) = plan.params() else {
        return Err(invalid(OP));
    };
    let (channels, kernel) = (params.channels as usize, params.kernel as usize);
    let state = params.state_elements().ok_or_else(|| invalid(OP))?;
    check_plan(
        plan,
        SemanticOp::CausalConv1d,
        [input.len(), weight.len(), bias.len()],
        output.len(),
        OP,
    )?;
    let taps = kernel - 1;
    let tokens = input.len() / channels;
    // One target slot for the window after the last token, or one per token.
    let slots = if state == 0 { 1 } else { target.len() / state };
    if source.len() != state
        || (state == 0 && target.len() != 0)
        || (state > 0 && !target.len().is_multiple_of(state))
        || !(slots == 1 || slots == tokens)
        || weight.len() != channels * kernel
    {
        return Err(invalid("causal convolution state length"));
    }
    if let Some(tree) = source.tree {
        if tree.len() != tokens || slots != tokens {
            return Err(invalid("causal convolution tree shape"));
        }
        // The parent precedes this node, so its whole target slot is final.
        // Split before borrowing it and the disjoint current slot together.
        let windows = unsafe { target.values() };
        for token in 0..tokens {
            let (done, rest) = windows.split_at_mut(token * state);
            let parent = tree.parents()[token];
            let prior = if parent == TokenTree::ROOT {
                source.values
            } else {
                &done[parent as usize * state..][..state]
            };
            let current = &mut rest[..state];
            let x = &input[token * channels..][..channels];
            let y = &mut output[token * channels..][..channels];
            for_row_pairs(
                parallel,
                (current, taps),
                (y, 1),
                kernel,
                |first, current, y| {
                    for (channel, (window, y)) in
                        (first..).zip(current.chunks_exact_mut(taps).zip(y))
                    {
                        let previous = &prior[channel * taps..][..taps];
                        let w = &weight[channel * kernel..][..kernel];
                        let mut sum = 0.0_f32;
                        for (&w, &old) in w[..taps].iter().zip(previous) {
                            sum += w * old;
                        }
                        sum += w[taps] * x[channel];
                        *y = crate::neural::apply_epilogue(params.activation, sum + bias[channel]);
                        window[..taps - 1].copy_from_slice(&previous[1..]);
                        window[taps - 1] = x[channel];
                    }
                },
            );
        }
        return Ok(());
    }
    // Channel `c` sees the sequence `source[c]` (`taps` values, oldest
    // first) followed by `input[.., c]`. Token `t`'s tap `i` reads element
    // `t + i` of it, so nothing rotates: a tap reads either the source
    // window or a whole earlier input row.
    for (t, (x, y)) in input
        .chunks_exact(channels)
        .zip(output.chunks_exact_mut(channels))
        .enumerate()
    {
        for_rows(parallel, y, 1, kernel, |first, y| {
            let weights = weight[first * kernel..].chunks_exact(kernel);
            for (c, (y, w)) in (first..).zip(y.iter_mut().zip(weights)) {
                // Each channel sums in the documented order, from 0.0.
                let mut sum = 0.0_f32;
                for (tap, &w) in w[..taps].iter().enumerate() {
                    sum += w * match (t + tap).checked_sub(taps) {
                        Some(row) => input[row * channels + c],
                        None => source.values[c * taps + t + tap],
                    };
                }
                sum += w[taps] * x[c];
                *y = crate::neural::apply_epilogue(params.activation, sum + bias[c]);
            }
        });
    }
    // SAFETY: the target is this node's exclusive resource for the call and
    // a distinct buffer from the source.
    let windows = unsafe { target.values() };
    // A window after `seen` tokens is the sequence's last `taps` elements
    // then: after the last token with one slot, after each with a slot per
    // token.
    for (slot, window) in windows.chunks_exact_mut(state.max(1)).enumerate() {
        let seen = if slots == 1 { tokens } else { slot + 1 };
        for (channel, window) in window.chunks_exact_mut(taps.max(1)).enumerate() {
            for (k, value) in window.iter_mut().enumerate() {
                *value = match (seen + k).checked_sub(taps) {
                    Some(row) => input[row * channels + channel],
                    None => source.values[channel * taps + seen + k],
                };
            }
        }
    }
    Ok(())
}

/// Clamp that keeps NaN.
#[inline]
fn clamp_keep_nan(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

/// Mamba-2 recurrence continuing the source state.
///
/// Per token and head `h` (group `g = h / (heads / groups)`), with
/// `dt = clamp(softplus(dt_raw + dt_bias[h]))` and `A = -exp(A_log[h])`:
/// `state[h, p, n] = state[h, p, n] * exp(dt * A) + (dt * B[g, n]) * x[h, p]`
/// and `y[h, p] = sum_n state[h, p, n] * C[g, n] + D[h] * x[h, p]`. The
/// target receives the state after the last token.
///
/// Every `(h, p)` row of the state evolves on its own, so each token's
/// rows are split between `parallel`'s tasks, each row computed as the
/// serial scan computes it.
#[allow(clippy::too_many_arguments)]
pub fn invoke_ssm_scan(
    plan: &CpuKernelPlan,
    source: StateSourceRef<'_>,
    target: StateTargetRef<'_>,
    xbc: &[f32],
    dt: &[f32],
    head_params: &[f32],
    output: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    const OP: &str = "planned state-space scan";
    let KernelParams::SsmScan(params) = plan.params() else {
        return Err(invalid(OP));
    };
    let (heads, head_dim) = (params.heads as usize, params.head_dim as usize);
    let (groups, size) = (params.groups as usize, params.state_size as usize);
    let state = params.state_elements().ok_or_else(|| invalid(OP))?;
    let conv_dim = params.conv_dim().ok_or_else(|| invalid(OP))?;
    check_plan(
        plan,
        SemanticOp::SsmScan,
        [xbc.len(), dt.len(), head_params.len()],
        output.len(),
        OP,
    )?;
    let tokens = xbc.len() / conv_dim;
    let slots = target.len() / state;
    let per_token = slots > 1;
    if source.len() != state
        || !target.len().is_multiple_of(state)
        || !(slots == 1 || slots == tokens)
    {
        return Err(invalid("state-space scan state length"));
    }
    if source
        .tree
        .is_some_and(|tree| tree.len() != tokens || slots != tokens)
    {
        return Err(invalid("state-space scan tree shape"));
    }
    let (dt_min, dt_max) = (params.dt_min.to_f32(), params.dt_max.to_f32());
    let (a_log, rest) = head_params.split_at(heads);
    let (dt_bias, skip) = rest.split_at(heads);
    let heads_per_group = heads / groups;
    // SAFETY: the target is this node's exclusive resource for the call and
    // a distinct buffer from the source.
    let states = unsafe { target.values() };
    if xbc.is_empty() {
        states.copy_from_slice(source.values);
    }
    // With one target slot, the first token reads the source and writes the
    // target, and later tokens update the target in place. With a slot per
    // token, token `t` reads slot `t - 1` (the source for the first) and
    // writes slot `t`: the same arithmetic, every intermediate state kept.
    // Every element is written, so the source is never copied.
    for (token, ((row, dt_row), out)) in xbc
        .chunks_exact(conv_dim)
        .zip(dt.chunks_exact(heads))
        .zip(output.chunks_exact_mut(heads * head_dim))
        .enumerate()
    {
        let (x, bc) = row.split_at(heads * head_dim);
        let (b, c) = bc.split_at(groups * size);
        let (prior, current): (Option<&[f32]>, &mut [f32]) = if per_token {
            let (done, rest) = states.split_at_mut(token * state);
            let parent = source.tree.map_or_else(
                || token.checked_sub(1),
                |tree| {
                    (tree.parents()[token] != TokenTree::ROOT)
                        .then_some(tree.parents()[token] as usize)
                },
            );
            let prior = match parent {
                None => source.values,
                Some(parent) => &done[parent * state..][..state],
            };
            (Some(prior), &mut rest[..state])
        } else {
            ((token == 0).then_some(source.values), &mut states[..])
        };
        for_row_pairs(
            parallel,
            (current, size),
            (out, 1),
            2 * size,
            |first, states, out| {
                // A head's step and decay, recomputed where a task's rows
                // enter it.
                let mut current = (usize::MAX, 0.0, 0.0);
                for (r, (state, y)) in (first..).zip(states.chunks_exact_mut(size).zip(out)) {
                    let head = r / head_dim;
                    if current.0 != head {
                        let step = clamp_keep_nan(
                            axmath::softplus_f32(dt_row[head] + dt_bias[head]),
                            dt_min,
                            dt_max,
                        );
                        let decay = axmath::exp_f32(step * -axmath::exp_f32(a_log[head]));
                        current = (head, step, decay);
                    }
                    let (_, step, decay) = current;
                    let group = head / heads_per_group;
                    let (b, c) = (
                        &b[group * size..(group + 1) * size],
                        &c[group * size..(group + 1) * size],
                    );
                    let xp = x[r];
                    let mut sum = 0.0_f32;
                    if let Some(prior) = prior {
                        let prior = &prior[r * size..][..size];
                        for (((s, &old), &bn), &cn) in state.iter_mut().zip(prior).zip(b).zip(c) {
                            *s = old * decay + (step * bn) * xp;
                            sum += *s * cn;
                        }
                    } else {
                        for ((s, &bn), &cn) in state.iter_mut().zip(b).zip(c) {
                            *s = *s * decay + (step * bn) * xp;
                            sum += *s * cn;
                        }
                    }
                    *y = sum + skip[head] * xp;
                }
            },
        );
    }
    Ok(())
}

/// `1 / sqrt(sum(x * x) + 1e-6)`: the factor that L2-normalizes `x`, as FLA
/// and Transformers compute it.
fn inverse_l2_norm(x: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &value in x {
        sum += value * value;
    }
    1.0 / axmath::sqrt_f32(sum + GatedDeltaRuleParams::L2_EPSILON)
}

/// One value head's gated delta step over its `[key_dim, value_dim]` state
/// `s`, reading the state before the step from `prior` (or from `s` itself
/// when it is updated in place). Every column `j` evolves on its own: its
/// key read, delta, update and query read-out, each sum in ascending `i`
/// from 0.0, with the decayed state rounded before it is read or updated.
#[allow(clippy::too_many_arguments)]
fn gated_delta_head(
    prior: Option<&[f32]>,
    s: &mut [f32],
    (q, k, v): (&[f32], &[f32], &[f32]),
    (q_scale, k_scale): (f32, f32),
    (gate, beta): (f32, f32),
    y: &mut [f32],
) {
    let width = v.len();
    for (j, (&v, y)) in v.iter().zip(y).enumerate() {
        let mut read = 0.0_f32;
        for (i, &k) in k.iter().enumerate() {
            let at = i * width + j;
            let old = prior.map_or(s[at], |prior| prior[at]);
            read += (old * gate) * (k * k_scale);
        }
        let delta = (v - read) * beta;
        let mut out = 0.0_f32;
        for (i, (&k, &q)) in k.iter().zip(q).enumerate() {
            let at = i * width + j;
            let old = prior.map_or(s[at], |prior| prior[at]);
            let next = old * gate + (k * k_scale) * delta;
            s[at] = next;
            out += next * (q * q_scale);
        }
        *y = out;
    }
}

/// Gated delta rule continuing the source state.
///
/// Per token and value head `h`, reading key head `h / (value_heads /
/// key_heads)`: `q` and `k` that key head's rows of `qkv`, each scaled by
/// its inverse L2 norm when `params.qk_l2norm`, the query then by
/// `1 / sqrt(key_dim)`; `v` the head's value row; and `S` its `[key_dim,
/// value_dim]` state:
///
/// ```text
/// S        = S * exp(g[h])
/// delta[j] = (v[j] - sum_i S[i][j] * k[i]) * beta[h]
/// S[i][j] += k[i] * delta[j]
/// y[j]     = sum_i S[i][j] * q[i]
/// ```
///
/// The target receives the state after the last token, or after each token
/// with a slot per token. Heads evolve on their own, so each token's heads
/// are split between `parallel`'s tasks, each computed as the serial loop
/// computes it.
#[allow(clippy::too_many_arguments)]
pub fn invoke_gated_delta_rule(
    plan: &CpuKernelPlan,
    source: StateSourceRef<'_>,
    target: StateTargetRef<'_>,
    qkv: &[f32],
    decay: &[f32],
    beta: &[f32],
    output: &mut [f32],
    parallel: &dyn ForkJoin,
) -> Result<(), CpuError> {
    const OP: &str = "planned gated delta rule";
    let KernelParams::GatedDeltaRule(params) = plan.params() else {
        return Err(invalid(OP));
    };
    let (key_heads, key_dim) = (params.key_heads as usize, params.key_dim as usize);
    let (heads, value_dim) = (params.value_heads as usize, params.value_dim as usize);
    let state = params.state_elements().ok_or_else(|| invalid(OP))?;
    let qkv_dim = params.qkv_dim().ok_or_else(|| invalid(OP))?;
    check_plan(
        plan,
        SemanticOp::GatedDeltaRule,
        [qkv.len(), decay.len(), beta.len()],
        output.len(),
        OP,
    )?;
    let tokens = qkv.len() / qkv_dim;
    let slots = target.len() / state;
    let per_token = slots > 1;
    if source.len() != state
        || !target.len().is_multiple_of(state)
        || !(slots == 1 || slots == tokens)
    {
        return Err(invalid("gated delta rule state length"));
    }
    if source
        .tree
        .is_some_and(|tree| tree.len() != tokens || slots != tokens)
    {
        return Err(invalid("gated delta rule tree shape"));
    }
    let head_state = key_dim * value_dim;
    let shared = heads / key_heads;
    let keys = key_heads * key_dim;
    let q_scale = 1.0 / axmath::sqrt_f32(key_dim as f32);
    // SAFETY: the target is this node's exclusive resource for the call and
    // a distinct buffer from the source.
    let states = unsafe { target.values() };
    // With one target slot, the first token reads the source and writes the
    // target, and later tokens update the target in place. With a slot per
    // token, token `t` reads its parent's slot (the previous token's, or
    // the source for the first) and writes slot `t`.
    for (token, ((row, (gates, betas)), out)) in qkv
        .chunks_exact(qkv_dim)
        .zip(decay.chunks_exact(heads).zip(beta.chunks_exact(heads)))
        .zip(output.chunks_exact_mut(heads * value_dim))
        .enumerate()
    {
        let (queries, rest) = row.split_at(keys);
        let (key_rows, values) = rest.split_at(keys);
        let (prior, current): (Option<&[f32]>, &mut [f32]) = if per_token {
            let (done, rest) = states.split_at_mut(token * state);
            let parent = source.tree.map_or_else(
                || token.checked_sub(1),
                |tree| {
                    (tree.parents()[token] != TokenTree::ROOT)
                        .then_some(tree.parents()[token] as usize)
                },
            );
            let prior = match parent {
                None => source.values,
                Some(parent) => &done[parent * state..][..state],
            };
            (Some(prior), &mut rest[..state])
        } else {
            ((token == 0).then_some(source.values), &mut states[..])
        };
        for_row_pairs(
            parallel,
            (current, head_state),
            (out, value_dim),
            3 * head_state,
            |first, states, out| {
                for (h, (s, y)) in (first..).zip(
                    states
                        .chunks_exact_mut(head_state)
                        .zip(out.chunks_exact_mut(value_dim)),
                ) {
                    let key_at = h / shared * key_dim;
                    let (q, k) = (
                        &queries[key_at..][..key_dim],
                        &key_rows[key_at..][..key_dim],
                    );
                    let (q_norm, k_norm) = if params.qk_l2norm {
                        (inverse_l2_norm(q), inverse_l2_norm(k))
                    } else {
                        (1.0, 1.0)
                    };
                    gated_delta_head(
                        prior.map(|prior| &prior[h * head_state..][..head_state]),
                        s,
                        (q, k, &values[h * value_dim..][..value_dim]),
                        (q_norm * q_scale, k_norm),
                        (axmath::exp_f32(gates[h]), betas[h]),
                        y,
                    );
                }
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuCapabilities, plan_cpu_kernel};
    use alloc::vec;
    use alloc::vec::Vec;
    use axnn::lowering::{CausalConv1dParams, ResourceKey, SsmScanParams, StateBinding};
    use axnn::{Shape, TensorDesc};

    const STATE: StateBinding = StateBinding {
        source: ResourceKey {
            group: 2,
            binding: 0,
        },
        target: ResourceKey {
            group: 2,
            binding: 1,
        },
    };

    fn desc(dims: &[usize]) -> TensorDesc {
        TensorDesc::new(Shape::try_from(dims).unwrap(), DType::F32)
    }

    fn values(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 8) as f32 / (1 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn conv_plan(params: CausalConv1dParams, tokens: usize) -> CpuKernelPlan {
        let (c, k) = (params.channels as usize, params.kernel as usize);
        let request =
            KernelRequest::causal_conv1d(&desc(&[tokens, c]), &desc(&[c, k]), &desc(&[c]), params)
                .unwrap();
        plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap()
    }

    fn scan_plan(params: SsmScanParams, tokens: usize) -> CpuKernelPlan {
        let heads = params.heads as usize;
        let request = KernelRequest::ssm_scan(
            &desc(&[tokens, params.conv_dim().unwrap()]),
            &desc(&[tokens, heads]),
            &desc(&[3, heads]),
            params,
        )
        .unwrap();
        plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap()
    }

    #[test]
    fn tree_recurrences_match_each_independent_serial_path() {
        let nodes = TokenTree::MAX_NODES;
        let parents: Vec<u32> = (0..nodes)
            .map(|i| {
                if i == 0 {
                    TokenTree::ROOT
                } else {
                    ((i - 1) / 2) as u32
                }
            })
            .collect();
        let tree = TokenTree::new(&parents).unwrap();
        let conv =
            CausalConv1dParams::new_activated(STATE, 3, 4, EpilogueActivation::Silu).unwrap();
        let initial_conv = values(9, 81);
        let input = values(nodes * 3, 82);
        let weight = values(12, 83);
        let bias = values(3, 84);
        let mut conv_states = vec![0.0; nodes * 9];
        let mut conv_output = vec![0.0; nodes * 3];
        invoke_causal_conv1d(
            &conv_plan(conv, nodes),
            StateSourceRef::new(&initial_conv).with_tree(tree),
            StateTargetRef::new(&mut conv_states),
            &input,
            &weight,
            &bias,
            &mut conv_output,
            &crate::Serial,
        )
        .unwrap();
        let scan = SsmScanParams::new(STATE, 4, 3, 2, 5, 0.0, f32::INFINITY).unwrap();
        let size = scan.state_elements().unwrap();
        let dim = scan.conv_dim().unwrap();
        let initial_scan = values(size, 85);
        let xbc = values(nodes * dim, 86);
        let dt = values(nodes * 4, 87);
        let head = values(12, 88);
        let mut scan_states = vec![0.0; nodes * size];
        let mut scan_output = vec![0.0; nodes * 12];
        invoke_ssm_scan(
            &scan_plan(scan, nodes),
            StateSourceRef::new(&initial_scan).with_tree(tree),
            StateTargetRef::new(&mut scan_states),
            &xbc,
            &dt,
            &head,
            &mut scan_output,
            &crate::Serial,
        )
        .unwrap();
        for node in 0..nodes {
            let mut path = [0; TokenTree::MAX_NODES];
            let count = tree.path(node, &mut path).unwrap();
            let mut prior_conv = initial_conv.clone();
            let mut next_conv = vec![0.0; 9];
            let mut y_conv = vec![0.0; 3];
            let mut prior_scan = initial_scan.clone();
            let mut next_scan = vec![0.0; size];
            let mut y_scan = vec![0.0; 12];
            for &id in &path[..count] {
                let id = id as usize;
                invoke_causal_conv1d(
                    &conv_plan(conv, 1),
                    StateSourceRef::new(&prior_conv),
                    StateTargetRef::new(&mut next_conv),
                    &input[id * 3..][..3],
                    &weight,
                    &bias,
                    &mut y_conv,
                    &crate::Serial,
                )
                .unwrap();
                core::mem::swap(&mut prior_conv, &mut next_conv);
                invoke_ssm_scan(
                    &scan_plan(scan, 1),
                    StateSourceRef::new(&prior_scan),
                    StateTargetRef::new(&mut next_scan),
                    &xbc[id * dim..][..dim],
                    &dt[id * 4..][..4],
                    &head,
                    &mut y_scan,
                    &crate::Serial,
                )
                .unwrap();
                core::mem::swap(&mut prior_scan, &mut next_scan);
            }
            assert_eq!(&conv_states[node * 9..][..9], prior_conv.as_slice());
            assert_eq!(&conv_output[node * 3..][..3], y_conv.as_slice());
            assert_eq!(&scan_states[node * size..][..size], prior_scan.as_slice());
            assert_eq!(&scan_output[node * 12..][..12], y_scan.as_slice());
        }
        let before = scan_states.clone();
        assert!(
            invoke_ssm_scan(
                &scan_plan(scan, nodes),
                StateSourceRef::new(&initial_scan)
                    .with_tree(TokenTree::new(&parents[..8]).unwrap()),
                StateTargetRef::new(&mut scan_states),
                &xbc,
                &dt,
                &head,
                &mut scan_output,
                &crate::Serial
            )
            .is_err()
        );
        assert_eq!(scan_states, before);
    }

    /// A fused activation (#107) equals applying it to the plain
    /// convolution's output, and leaves the successor window untouched —
    /// taps read the input rows, never the activated output.
    #[test]
    fn activated_convolution_equals_the_plain_one_with_the_activation_applied() {
        let plain = CausalConv1dParams::new(STATE, 3, 4).unwrap();
        let weight = values(12, 1);
        let bias = values(3, 2);
        let initial = values(9, 3);
        let input = values(15, 4);

        let mut plain_window = vec![0.0; 9];
        let mut plain_out = vec![0.0; 15];
        invoke_causal_conv1d(
            &conv_plan(plain, 5),
            StateSourceRef::new(&initial),
            StateTargetRef::new(&mut plain_window),
            &input,
            &weight,
            &bias,
            &mut plain_out,
            &crate::Serial,
        )
        .unwrap();

        for activation in [EpilogueActivation::Silu, EpilogueActivation::Sigmoid] {
            let params = CausalConv1dParams::new_activated(STATE, 3, 4, activation).unwrap();
            let mut window = vec![0.0; 9];
            let mut out = vec![0.0; 15];
            invoke_causal_conv1d(
                &conv_plan(params, 5),
                StateSourceRef::new(&initial),
                StateTargetRef::new(&mut window),
                &input,
                &weight,
                &bias,
                &mut out,
                &crate::Serial,
            )
            .unwrap();
            for (fused, unfused) in out.iter().zip(&plain_out) {
                assert_eq!(
                    *fused,
                    crate::neural::apply_epilogue(activation, *unfused),
                    "{activation:?}"
                );
            }
            // The carried window is the input history, so it is unchanged.
            assert_eq!(window, plain_window, "{activation:?}");
        }
    }

    #[test]
    fn convolution_continues_its_window_and_a_chunk_equals_single_steps() {
        let params = CausalConv1dParams::new(STATE, 3, 4).unwrap();
        let weight = values(12, 1);
        let bias = values(3, 2);
        let initial = values(9, 3);
        let input = values(15, 4);

        // Whole chunk of five tokens.
        let mut after_chunk = vec![0.0; 9];
        let mut chunk = vec![0.0; 15];
        invoke_causal_conv1d(
            &conv_plan(params, 5),
            StateSourceRef::new(&initial),
            StateTargetRef::new(&mut after_chunk),
            &input,
            &weight,
            &bias,
            &mut chunk,
            &crate::Serial,
        )
        .unwrap();

        // Reference over the concatenated history.
        for t in 0..5 {
            for c in 0..3 {
                let history: Vec<f32> = (0..3)
                    .map(|i| initial[c * 3 + i])
                    .chain((0..=t).map(|u| input[u * 3 + c]))
                    .collect();
                let window = &history[history.len() - 4..];
                let mut expected = 0.0_f32;
                for i in 0..4 {
                    expected += weight[c * 4 + i] * window[i];
                }
                assert_eq!(
                    chunk[t * 3 + c],
                    expected + bias[c],
                    "token {t} channel {c}"
                );
            }
        }
        for c in 0..3 {
            assert_eq!(
                after_chunk[c * 3..c * 3 + 3],
                [input[6 + c], input[9 + c], input[12 + c]]
            );
        }

        // The same tokens one at a time, alternating buffers.
        let (mut a, mut b) = (initial.clone(), vec![0.0; 9]);
        let mut stepped = Vec::new();
        for t in 0..5 {
            let mut y = [0.0; 3];
            invoke_causal_conv1d(
                &conv_plan(params, 1),
                StateSourceRef::new(&a),
                StateTargetRef::new(&mut b),
                &input[t * 3..t * 3 + 3],
                &weight,
                &bias,
                &mut y,
                &crate::Serial,
            )
            .unwrap();
            stepped.extend_from_slice(&y);
            core::mem::swap(&mut a, &mut b);
        }
        assert_eq!(stepped, chunk);
        assert_eq!(a, after_chunk);
    }

    /// F64 reference of the documented recurrence.
    fn scan_reference(
        params: SsmScanParams,
        state: &mut [f64],
        xbc: &[f32],
        dt: &[f32],
        head_params: &[f32],
    ) -> Vec<f64> {
        let (heads, hd, groups, n) = (
            params.heads as usize,
            params.head_dim as usize,
            params.groups as usize,
            params.state_size as usize,
        );
        let conv_dim = params.conv_dim().unwrap();
        let mut out = Vec::new();
        for (row, dt_row) in xbc.chunks_exact(conv_dim).zip(dt.chunks_exact(heads)) {
            for h in 0..heads {
                let g = h / (heads / groups);
                let raw = f64::from(dt_row[h]) + f64::from(head_params[heads + h]);
                let step = if raw > 20.0 { raw } else { raw.exp().ln_1p() }.clamp(
                    f64::from(params.dt_min.to_f32()),
                    f64::from(params.dt_max.to_f32()),
                );
                let decay = (step * -f64::from(head_params[h]).exp()).exp();
                for p in 0..hd {
                    let x = f64::from(row[h * hd + p]);
                    let mut y = 0.0;
                    for k in 0..n {
                        let b = f64::from(row[heads * hd + g * n + k]);
                        let c = f64::from(row[heads * hd + groups * n + g * n + k]);
                        let s = &mut state[(h * hd + p) * n + k];
                        *s = *s * decay + step * b * x;
                        y += *s * c;
                    }
                    out.push(y + f64::from(head_params[2 * heads + h]) * x);
                }
            }
        }
        out
    }

    fn rule_plan(params: GatedDeltaRuleParams, tokens: usize) -> CpuKernelPlan {
        let heads = params.value_heads as usize;
        let request = KernelRequest::gated_delta_rule(
            &desc(&[tokens, params.qkv_dim().unwrap()]),
            &desc(&[tokens, heads]),
            &desc(&[tokens, heads]),
            params,
        )
        .unwrap();
        plan_cpu_kernel(&request, CpuCapabilities::compiled()).unwrap()
    }

    /// The gated delta rule in F64, straight from its definition (decay the
    /// whole state, then one delta write per value column, then read out),
    /// advancing `state` and returning every token's output.
    fn rule_reference(
        params: GatedDeltaRuleParams,
        state: &mut [f64],
        qkv: &[f32],
        decay: &[f32],
        beta: &[f32],
    ) -> Vec<f64> {
        let (key_heads, dk) = (params.key_heads as usize, params.key_dim as usize);
        let (heads, dv) = (params.value_heads as usize, params.value_dim as usize);
        let width = params.qkv_dim().unwrap();
        let mut out = Vec::new();
        for (t, row) in qkv.chunks_exact(width).enumerate() {
            let row: Vec<f64> = row.iter().map(|&v| v.into()).collect();
            for h in 0..heads {
                let key = h / (heads / key_heads);
                let mut q = row[key * dk..][..dk].to_vec();
                let mut k = row[key_heads * dk + key * dk..][..dk].to_vec();
                let v = &row[2 * key_heads * dk + h * dv..][..dv];
                if params.qk_l2norm {
                    for x in [&mut q, &mut k] {
                        let norm = (x.iter().map(|a| a * a).sum::<f64>() + 1e-6).sqrt();
                        x.iter_mut().for_each(|a| *a /= norm);
                    }
                }
                q.iter_mut().for_each(|a| *a /= (dk as f64).sqrt());
                let s = &mut state[h * dk * dv..][..dk * dv];
                let gate = f64::from(decay[t * heads + h]).exp();
                s.iter_mut().for_each(|x| *x *= gate);
                for j in 0..dv {
                    let read: f64 = (0..dk).map(|i| s[i * dv + j] * k[i]).sum();
                    let delta = (v[j] - read) * f64::from(beta[t * heads + h]);
                    for i in 0..dk {
                        s[i * dv + j] += k[i] * delta;
                    }
                }
                for j in 0..dv {
                    out.push((0..dk).map(|i| s[i * dv + j] * q[i]).sum());
                }
            }
        }
        out
    }

    /// Gates as a model feeds them: a log-space decay `<= 0` and a write
    /// strength in (0, 1).
    fn rule_gates(count: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let decay = values(count, seed).iter().map(|v| -4.0 * v.abs()).collect();
        let beta = values(count, seed + 1)
            .iter()
            .map(|&v| axmath::sigmoid_f32(4.0 * v))
            .collect();
        (decay, beta)
    }

    /// Three value heads per key head (the published 48:16 ratio), with and
    /// without the L2-normalized query and key: the chunk matches the F64
    /// recurrence, and equals the same tokens stepped one at a time, bit for
    /// bit, outputs and state alike.
    #[test]
    fn gated_delta_rule_matches_its_definition_and_chunks_equal_steps() {
        for l2 in [true, false] {
            let params = GatedDeltaRuleParams::new(STATE, (2, 4), (6, 3), l2).unwrap();
            let width = params.qkv_dim().unwrap();
            assert_eq!(width, 2 * 2 * 4 + 6 * 3);
            let state = params.state_elements().unwrap();
            let tokens = 4;
            let qkv = values(tokens * width, 51);
            let (decay, beta) = rule_gates(tokens * 6, 52);
            let initial = values(state, 54);

            let mut after = vec![0.0; state];
            let mut chunk = vec![0.0; tokens * 18];
            invoke_gated_delta_rule(
                &rule_plan(params, tokens),
                StateSourceRef::new(&initial),
                StateTargetRef::new(&mut after),
                &qkv,
                &decay,
                &beta,
                &mut chunk,
                &crate::Serial,
            )
            .unwrap();
            let mut reference: Vec<f64> = initial.iter().map(|&v| f64::from(v)).collect();
            let expected = rule_reference(params, &mut reference, &qkv, &decay, &beta);
            for (actual, expected) in chunk.iter().zip(&expected) {
                assert!(
                    (f64::from(*actual) - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                    "l2 {l2}: output {actual} vs {expected}"
                );
            }
            for (actual, expected) in after.iter().zip(&reference) {
                assert!(
                    (f64::from(*actual) - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                    "l2 {l2}: state {actual} vs {expected}"
                );
            }

            let (mut a, mut b) = (initial.clone(), vec![0.0; state]);
            let mut stepped = Vec::new();
            for t in 0..tokens {
                let mut y = [0.0; 18];
                invoke_gated_delta_rule(
                    &rule_plan(params, 1),
                    StateSourceRef::new(&a),
                    StateTargetRef::new(&mut b),
                    &qkv[t * width..][..width],
                    &decay[t * 6..][..6],
                    &beta[t * 6..][..6],
                    &mut y,
                    &crate::Serial,
                )
                .unwrap();
                stepped.extend_from_slice(&y);
                core::mem::swap(&mut a, &mut b);
            }
            assert_eq!(stepped, chunk, "l2 {l2}");
            assert_eq!(a, after, "l2 {l2}");
        }
    }

    /// A slot per token keeps every intermediate state; a token tree runs
    /// each node from its parent's state, as its root-to-node path stepped
    /// serially; any other slot count is refused before anything is written.
    #[test]
    fn gated_delta_rule_slots_and_trees_keep_every_state() {
        let params = GatedDeltaRuleParams::new(STATE, (1, 4), (2, 3), true).unwrap();
        let width = params.qkv_dim().unwrap();
        let state = params.state_elements().unwrap();
        let nodes = 7;
        let parents = [TokenTree::ROOT, 0, 0, 1, 1, 2, 2];
        let tree = TokenTree::new(&parents).unwrap();
        let qkv = values(nodes * width, 61);
        let (decay, beta) = rule_gates(nodes * 2, 62);
        let initial = values(state, 64);
        let step = |prior: &[f32], id: usize| {
            let mut next = vec![0.0; state];
            let mut y = vec![0.0; 6];
            invoke_gated_delta_rule(
                &rule_plan(params, 1),
                StateSourceRef::new(prior),
                StateTargetRef::new(&mut next),
                &qkv[id * width..][..width],
                &decay[id * 2..][..2],
                &beta[id * 2..][..2],
                &mut y,
                &crate::Serial,
            )
            .unwrap();
            (next, y)
        };

        // Linear: slot t is the state after t + 1 serial steps.
        let mut slots = vec![0.0; nodes * state];
        let mut out = vec![0.0; nodes * 6];
        invoke_gated_delta_rule(
            &rule_plan(params, nodes),
            StateSourceRef::new(&initial),
            StateTargetRef::new(&mut slots),
            &qkv,
            &decay,
            &beta,
            &mut out,
            &crate::Serial,
        )
        .unwrap();
        let mut prior = initial.clone();
        for t in 0..nodes {
            let (next, y) = step(&prior, t);
            assert_eq!(&slots[t * state..][..state], next.as_slice(), "slot {t}");
            assert_eq!(&out[t * 6..][..6], y.as_slice(), "output {t}");
            prior = next;
        }

        // Tree: node n continues from its parent's slot.
        invoke_gated_delta_rule(
            &rule_plan(params, nodes),
            StateSourceRef::new(&initial).with_tree(tree),
            StateTargetRef::new(&mut slots),
            &qkv,
            &decay,
            &beta,
            &mut out,
            &crate::Serial,
        )
        .unwrap();
        for node in 0..nodes {
            let mut path = [0; TokenTree::MAX_NODES];
            let count = tree.path(node, &mut path).unwrap();
            let (mut prior, mut y) = (initial.clone(), vec![0.0; 6]);
            for &id in &path[..count] {
                (prior, y) = step(&prior, id as usize);
            }
            assert_eq!(
                &slots[node * state..][..state],
                prior.as_slice(),
                "node {node}"
            );
            assert_eq!(&out[node * 6..][..6], y.as_slice(), "node {node}");
        }

        let before = slots.clone();
        let mut two = vec![0.0; 2 * state];
        for (target, tree) in [
            (&mut two[..], None),
            (&mut slots[..], Some(TokenTree::new(&parents[..3]).unwrap())),
        ] {
            let source = StateSourceRef::new(&initial);
            assert!(
                invoke_gated_delta_rule(
                    &rule_plan(params, nodes),
                    tree.map_or(source, |tree| source.with_tree(tree)),
                    StateTargetRef::new(target),
                    &qkv,
                    &decay,
                    &beta,
                    &mut out,
                    &crate::Serial,
                )
                .is_err()
            );
        }
        assert_eq!(slots, before);
    }

    #[test]
    fn a_split_gated_delta_rule_matches_the_serial_one_bit_for_bit() {
        // Enough heads to split: 16 value heads over 4 key heads of 32.
        let params = GatedDeltaRuleParams::new(STATE, (4, 32), (16, 32), true).unwrap();
        let width = params.qkv_dim().unwrap();
        let state = params.state_elements().unwrap();
        let tokens = 2;
        let qkv = values(tokens * width, 71);
        let (decay, beta) = rule_gates(tokens * 16, 72);
        let initial = values(state, 74);
        let run = |parallel: &dyn crate::ForkJoin| {
            let mut after = vec![0.0; state];
            let mut y = vec![0.0; tokens * 16 * 32];
            invoke_gated_delta_rule(
                &rule_plan(params, tokens),
                StateSourceRef::new(&initial),
                StateTargetRef::new(&mut after),
                &qkv,
                &decay,
                &beta,
                &mut y,
                parallel,
            )
            .unwrap();
            (y, after)
        };
        let serial = run(&crate::Serial);
        for width in [2, 3, 8] {
            assert_eq!(
                run(&crate::fork_join::Reversed(width)),
                serial,
                "width {width}"
            );
        }
    }

    #[test]
    fn scan_matches_the_documented_recurrence_and_chunks_equal_steps() {
        // Four heads in two groups, so heads 0-1 share group 0.
        let params = SsmScanParams::new(STATE, 4, 3, 2, 5, 0.0, f32::INFINITY).unwrap();
        let conv_dim = params.conv_dim().unwrap();
        assert_eq!(conv_dim, 4 * 3 + 2 * 2 * 5);
        let tokens = 3;
        let xbc = values(tokens * conv_dim, 7);
        let mut dt = values(tokens * 4, 8);
        dt[1] = 30.0; // exercises the linear softplus branch
        let head_params = values(12, 9);
        let initial = values(4 * 3 * 5, 10);

        let mut after = vec![0.0; 60];
        let mut chunk = vec![0.0; tokens * 12];
        invoke_ssm_scan(
            &scan_plan(params, tokens),
            StateSourceRef::new(&initial),
            StateTargetRef::new(&mut after),
            &xbc,
            &dt,
            &head_params,
            &mut chunk,
            &crate::Serial,
        )
        .unwrap();
        let mut reference_state: Vec<f64> = initial.iter().map(|&v| f64::from(v)).collect();
        let expected = scan_reference(params, &mut reference_state, &xbc, &dt, &head_params);
        for (actual, expected) in chunk.iter().zip(&expected) {
            assert!((f64::from(*actual) - expected).abs() <= 1e-5 * (1.0 + expected.abs()));
        }
        for (actual, expected) in after.iter().zip(&reference_state) {
            assert!((f64::from(*actual) - expected).abs() <= 1e-5 * (1.0 + expected.abs()));
        }

        let (mut a, mut b) = (initial.clone(), vec![0.0; 60]);
        let mut stepped = Vec::new();
        for t in 0..tokens {
            let mut y = [0.0; 12];
            invoke_ssm_scan(
                &scan_plan(params, 1),
                StateSourceRef::new(&a),
                StateTargetRef::new(&mut b),
                &xbc[t * conv_dim..(t + 1) * conv_dim],
                &dt[t * 4..(t + 1) * 4],
                &head_params,
                &mut y,
                &crate::Serial,
            )
            .unwrap();
            stepped.extend_from_slice(&y);
            core::mem::swap(&mut a, &mut b);
        }
        assert_eq!(stepped, chunk);
        assert_eq!(a, after);
    }

    /// A target of one slot per token keeps every intermediate state: slot
    /// `t` is what `t + 1` single-token steps leave, bit for bit, for the
    /// scan and the convolution alike, and the outputs do not change.
    #[test]
    fn per_token_state_slots_hold_every_intermediate_state() {
        let tokens = 3;
        let scan_params = SsmScanParams::new(STATE, 4, 3, 2, 5, 0.0, f32::INFINITY).unwrap();
        let conv_dim = scan_params.conv_dim().unwrap();
        let scan_state = scan_params.state_elements().unwrap();
        let xbc = values(tokens * conv_dim, 41);
        let dt = values(tokens * 4, 42);
        let head_params = values(12, 43);
        let initial = values(scan_state, 44);
        let mut slots = vec![0.0; tokens * scan_state];
        let mut y = vec![0.0; tokens * 12];
        invoke_ssm_scan(
            &scan_plan(scan_params, tokens),
            StateSourceRef::new(&initial),
            StateTargetRef::new(&mut slots),
            &xbc,
            &dt,
            &head_params,
            &mut y,
            &crate::Serial,
        )
        .unwrap();
        let (mut state, mut next) = (initial.clone(), vec![0.0; scan_state]);
        for t in 0..tokens {
            let mut step_y = vec![0.0; 12];
            invoke_ssm_scan(
                &scan_plan(scan_params, 1),
                StateSourceRef::new(&state),
                StateTargetRef::new(&mut next),
                &xbc[t * conv_dim..(t + 1) * conv_dim],
                &dt[t * 4..(t + 1) * 4],
                &head_params,
                &mut step_y,
                &crate::Serial,
            )
            .unwrap();
            core::mem::swap(&mut state, &mut next);
            assert_eq!(step_y, y[t * 12..(t + 1) * 12], "scan output {t}");
            assert_eq!(
                state,
                slots[t * scan_state..(t + 1) * scan_state],
                "scan slot {t}"
            );
        }

        let conv = CausalConv1dParams::new(STATE, 3, 4).unwrap();
        let weight = values(12, 45);
        let bias = values(3, 46);
        let window = values(9, 47);
        let input = values(tokens * 3, 48);
        let mut windows = vec![0.0; tokens * 9];
        let mut out = vec![0.0; tokens * 3];
        invoke_causal_conv1d(
            &conv_plan(conv, tokens),
            StateSourceRef::new(&window),
            StateTargetRef::new(&mut windows),
            &input,
            &weight,
            &bias,
            &mut out,
            &crate::Serial,
        )
        .unwrap();
        let (mut state, mut next) = (window.clone(), vec![0.0; 9]);
        for t in 0..tokens {
            let mut step_out = vec![0.0; 3];
            invoke_causal_conv1d(
                &conv_plan(conv, 1),
                StateSourceRef::new(&state),
                StateTargetRef::new(&mut next),
                &input[t * 3..(t + 1) * 3],
                &weight,
                &bias,
                &mut step_out,
                &crate::Serial,
            )
            .unwrap();
            core::mem::swap(&mut state, &mut next);
            assert_eq!(step_out, out[t * 3..(t + 1) * 3], "conv output {t}");
            assert_eq!(state, windows[t * 9..(t + 1) * 9], "conv slot {t}");
        }

        // Any other slot count is refused.
        let mut two = vec![0.0; 2 * scan_state];
        assert!(
            invoke_ssm_scan(
                &scan_plan(scan_params, tokens),
                StateSourceRef::new(&initial),
                StateTargetRef::new(&mut two),
                &xbc,
                &dt,
                &head_params,
                &mut y,
                &crate::Serial,
            )
            .is_err()
        );
    }

    #[test]
    fn a_split_convolution_matches_the_serial_one_bit_for_bit() {
        // Enough channels to split.
        let (channels, kernel, tokens) = (20_000, 4, 2);
        let params = CausalConv1dParams::new(STATE, channels as u32, kernel as u32).unwrap();
        let weight = values(channels * kernel, 31);
        let bias = values(channels, 32);
        let initial = values(channels * (kernel - 1), 33);
        let input = values(tokens * channels, 34);
        let run = |parallel: &dyn crate::ForkJoin| {
            let mut after = vec![0.0; initial.len()];
            let mut y = vec![0.0; input.len()];
            invoke_causal_conv1d(
                &conv_plan(params, tokens),
                StateSourceRef::new(&initial),
                StateTargetRef::new(&mut after),
                &input,
                &weight,
                &bias,
                &mut y,
                parallel,
            )
            .unwrap();
            (y, after)
        };
        let serial = run(&crate::Serial);
        for width in [2, 8] {
            assert_eq!(
                run(&crate::fork_join::Reversed(width)),
                serial,
                "width {width}"
            );
        }
    }

    #[test]
    fn a_split_scan_matches_the_serial_one_bit_for_bit() {
        // Enough rows to split: 16 heads of 32, state 128.
        let params = SsmScanParams::new(STATE, 16, 32, 2, 128, 0.0, f32::INFINITY).unwrap();
        let conv_dim = params.conv_dim().unwrap();
        let state = params.state_elements().unwrap();
        let tokens = 2;
        let xbc = values(tokens * conv_dim, 11);
        let dt = values(tokens * 16, 12);
        let head_params = values(48, 13);
        let initial = values(state, 14);
        let run = |parallel: &dyn crate::ForkJoin| {
            let mut after = vec![0.0; state];
            let mut y = vec![0.0; tokens * 16 * 32];
            invoke_ssm_scan(
                &scan_plan(params, tokens),
                StateSourceRef::new(&initial),
                StateTargetRef::new(&mut after),
                &xbc,
                &dt,
                &head_params,
                &mut y,
                parallel,
            )
            .unwrap();
            (y, after)
        };
        let serial = run(&crate::Serial);
        for width in [2, 3, 8] {
            assert_eq!(
                run(&crate::fork_join::Reversed(width)),
                serial,
                "width {width}"
            );
        }
    }

    #[test]
    fn scan_clamps_the_step_and_propagates_nan() {
        let params = SsmScanParams::new(STATE, 1, 1, 1, 1, 0.5, 2.0).unwrap();
        let run = |dt: f32, x: f32| {
            let mut after = [0.0_f32];
            let mut y = [0.0_f32];
            // A_log = 0 (A = -1), dt_bias = 0, D = 0; B = C = 1; state 0.
            invoke_ssm_scan(
                &scan_plan(params, 1),
                StateSourceRef::new(&[0.0]),
                StateTargetRef::new(&mut after),
                &[x, 1.0, 1.0],
                &[dt],
                &[0.0, 0.0, 0.0],
                &mut y,
                &crate::Serial,
            )
            .unwrap();
            (y[0], after[0])
        };
        // A very negative dt clamps up to 0.5; a large one down to 2.0.
        assert_eq!(run(-50.0, 1.0).1, 0.5);
        assert_eq!(run(50.0, 1.0).1, 2.0);
        let (y, state) = run(f32::NAN, 1.0);
        assert!(y.is_nan() && state.is_nan());
        assert!(
            SsmScanParams::new(STATE, 3, 1, 2, 1, 0.0, 1.0).is_none(),
            "groups divide heads"
        );
        assert!(
            SsmScanParams::new(STATE, 1, 1, 1, 1, 1.0, 1.0).is_none(),
            "empty step range"
        );
        let aliased = StateBinding {
            target: STATE.source,
            ..STATE
        };
        assert!(SsmScanParams::new(aliased, 1, 1, 1, 1, 0.0, 1.0).is_none());
        assert!(CausalConv1dParams::new(aliased, 1, 4).is_none());
        assert!(
            CausalConv1dParams::new(STATE, 1, 1).is_none(),
            "a window needs two taps"
        );
    }
}
